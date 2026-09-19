"""Low-latency semantic generation for camera-backed gesture imitation."""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import aclosing
from typing import Any, Iterable

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.serve.realtime.action.routing import (
    visual_deictic_category_scope,
    visual_deictic_scope_candidates,
)
from sglang_omni.serve.realtime.protocol.common import (
    IMAGE_ROLE_USER_CAMERA,
    MAX_ACTION_VISUAL_SCOPE_USER_CAMERA_IMAGES,
)
from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
)
from sglang_omni.serve.realtime.visual_observation import (
    VISUAL_OBSERVATION_TOP_LOGPROBS,
    VisualObservationConfidence,
    choose_visual_equivalent,
    summarize_visual_observation_confidence,
    visual_candidate_definition,
    visual_equivalence_key,
)


_UNSUPPORTED_OUTPUT = "UNSUPPORTED"
_NUMERIC_GESTURE_LABEL_RE = re.compile(
    r"\A(数字(?:零|一|二|三|四|五|六|七|八|九|十))手势\Z"
)
_NUMERIC_GESTURE_OUTPUT_RE = re.compile(
    r"\A(数字(?:零|一|二|三|四|五|六|七|八|九|十))(?:手势)?\Z"
)
_HARMLESS_TERMINAL_PUNCTUATION_RE = re.compile(r"[。.!！？?]+\Z")
VISUAL_GESTURE_COPY_ROUTES = frozenset({"COPY_ACTION", "COPY_HAND"})


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


def visual_gesture_output_label(candidate: SessionActionCandidate) -> str:
    """Return the shortest unambiguous semantic label exposed to the model."""

    source_label = candidate.source_label.strip()
    numeric_match = _NUMERIC_GESTURE_LABEL_RE.fullmatch(source_label)
    return numeric_match.group(1) if numeric_match is not None else source_label


def visual_gesture_candidates(
    categories: Iterable[SessionActionCategory],
    candidates: Iterable[SessionActionCandidate],
) -> tuple[SessionActionCandidate, ...]:
    """Return unambiguous catalog-owned gesture labels eligible this session."""

    scope = visual_deictic_category_scope(tuple(categories), "gesture")
    if scope is None:
        return ()
    allowed_ids = {candidate.candidate_id for candidate in candidates}
    scoped = [
        candidate
        for candidate in visual_deictic_scope_candidates(scope)
        if candidate.candidate_id in allowed_ids
    ]
    by_label: dict[str, list[SessionActionCandidate]] = {}
    for candidate in scoped:
        by_label.setdefault(visual_gesture_output_label(candidate), []).append(
            candidate
        )
    # A semantic label must resolve to exactly one executable catalog entry.
    unambiguous = tuple(
        matches[0]
        for label, matches in by_label.items()
        if label and len(matches) == 1
    )
    by_visual_class: dict[str, list[SessionActionCandidate]] = {}
    for candidate in unambiguous:
        by_visual_class.setdefault(
            visual_equivalence_key(candidate.source_label), []
        ).append(candidate)
    # Keep explicit catalog aliases for text routing, but expose only one
    # executor for visually equivalent hand shapes such as digit two and V.
    return tuple(
        choose_visual_equivalent(matches)
        for matches in by_visual_class.values()
    )


def build_visual_gesture_system_prompt(
    candidates: Iterable[SessionActionCandidate],
) -> str:
    """Build the cacheable semantic-label contract from the active catalog."""

    catalog_lines = [
        f"- {visual_gesture_output_label(candidate)}: "
        f"{visual_candidate_definition(candidate.source_label, candidate.short_definition)}"
        for candidate in candidates
    ]
    return (
        "你执行当前摄像头画面的封闭集视觉分类，不进行自然语言回答。\n"
        "【输出协议（最高优先级）】\n"
        "输出必须逐字等于下方目录中某一行冒号左侧的完整标签，或 UNSUPPORTED。"
        "第一个字符直接是标签；标签结束后立即停止。不得添加‘手势’等后缀，"
        "不得输出项目符号、引号、标点、解释、前缀、代码块或多个标签。\n"
        "【视觉判定规则】\n"
        "只判断用户正在清晰、刻意展示的动作，不要理解语音、不要回答问题、不要计算，"
        "也不要根据历史猜测。静态手势依据最清晰的稳定画面；挥手、招手、飞吻等动态动作"
        "依据多张画面的连续变化。忽略空白、遮挡、手已放下和无关过渡画面；"
        "多张画面属于同一次展示，只能产生一个标签。\n"
        "【候选目录】\n"
        + "\n".join(catalog_lines)
        + f"\n- {_UNSUPPORTED_OUTPUT}: 没有清晰手势，或手势不在目录中。\n"
        "数字手势必须按目录中每项的伸指形态和区分要点严格判断。"
        "视觉形态等价的数字二与单手比耶只保留一个目录标签。\n"
        "【提交前检查】全部输出必须只是一个目录标签；无法得到唯一匹配时只输出 "
        "UNSUPPORTED。不要复述说明，不要添加任何其他字符。"
    )


def parse_visual_gesture_output(
    text: str,
    candidates: Iterable[SessionActionCandidate],
) -> SessionActionCandidate | None:
    """Resolve one semantic label after safe formatting normalization."""

    candidate_list = tuple(candidates)
    label = normalize_visual_gesture_output(text, candidate_list)
    if label == _UNSUPPORTED_OUTPUT:
        return None
    by_label = {
        visual_gesture_output_label(candidate): candidate
        for candidate in candidate_list
    }
    return by_label.get(label)


def normalize_visual_gesture_output(
    text: str,
    candidates: Iterable[SessionActionCandidate],
) -> str | None:
    """Return a catalog label for harmless variants; reject semantic extras."""

    candidate_list = tuple(candidates)
    label = _HARMLESS_TERMINAL_PUNCTUATION_RE.sub("", text.strip()).strip()
    if label == _UNSUPPORTED_OUTPUT:
        return label
    allowed_labels = {
        visual_gesture_output_label(candidate) for candidate in candidate_list
    }
    if label in allowed_labels:
        return label
    numeric_match = _NUMERIC_GESTURE_OUTPUT_RE.fullmatch(label)
    if numeric_match is not None and numeric_match.group(1) in allowed_labels:
        return numeric_match.group(1)
    # The copy gesture catalog intentionally exposes one executor for each
    # visual hand shape.  Accept a model-emitted catalog alias and resolve it
    # to that executor (for example, 单手比耶 -> 数字二).
    visual_key = visual_equivalence_key(label)
    equivalent_labels = {
        visual_gesture_output_label(candidate)
        for candidate in candidate_list
        if visual_equivalence_key(candidate.source_label) == visual_key
    }
    if visual_key != label and len(equivalent_labels) == 1:
        return next(iter(equivalent_labels))
    return None


class VisualGestureGenerationComponent:
    def _visual_gesture_generation_candidates(
        self, turn: TurnBuffer | None = None
    ) -> tuple[SessionActionCandidate, ...]:
        candidates: Iterable[SessionActionCandidate] = self.candidates
        if turn is not None:
            candidates = self._filter_turn_action_candidates(turn, list(candidates))
        return visual_gesture_candidates(self.categories, candidates)

    def _build_visual_gesture_request(
        self,
        turn: TurnBuffer,
        images: list[Any],
        image_roles: list[str],
        candidates: tuple[SessionActionCandidate, ...],
    ) -> tuple[GenerateRequest, list[str]]:
        selected = [
            (image, role)
            for image, role in zip(images, image_roles, strict=True)
            if role == IMAGE_ROLE_USER_CAMERA
        ][-MAX_ACTION_VISUAL_SCOPE_USER_CAMERA_IMAGES:]
        selected_images = [image for image, _ in selected]
        selected_roles = [role for _, role in selected]
        request = GenerateRequest(
            model=self.model_name,
            messages=[
                Message(
                    role="system",
                    content=build_visual_gesture_system_prompt(candidates),
                ),
                Message(
                    role="user",
                    content=[
                        *({"type": "image"} for _ in selected_images),
                        {
                            "type": "text",
                            "text": "识别这些当前画面中用户稳定展示的一个手势。",
                        },
                    ],
                ),
            ],
            sampling=SamplingParams(
                temperature=0,
                top_p=1.0,
                max_new_tokens=24,
                stop=["\n", "。", ".", "！", "!", "？", "?"],
            ),
            stream=True,
            extra_params={
                "return_logprob": True,
                "top_logprobs_num": VISUAL_OBSERVATION_TOP_LOGPROBS,
            },
            output_modalities=["text"],
            metadata={
                "audios": [],
                "images": selected_images,
                "image_roles": selected_roles,
                "session_id": self.session_id,
                "session_instance_id": self.session_instance_id,
                "turn_id": turn.turn_id,
                "logical_request_id": turn.request_base,
                "task": "session_visual_gesture_probe",
                "private_output": True,
            },
        )
        return request, selected_roles

    @staticmethod
    def _visual_gesture_action_result(
        candidate: SessionActionCandidate | None,
        *,
        compute_ms: float,
        output_valid: bool,
        output_chars: int,
        ttft_ms: float | None,
        candidate_count: int,
        confidence: VisualObservationConfidence | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        if candidate is None:
            action = {
                "candidate_id": UNSUPPORTED_DECISION_ID,
                "action_id": UNSUPPORTED_DECISION_ID,
                "execution_binding": {},
                "execute": False,
                "support_status": "unsupported",
                "fallback_applied": False,
                "reason_code": (
                    confidence.rejection_reason
                    if confidence is not None and not confidence.accepted
                    else (
                        "visual_gesture_unsupported"
                        if output_valid
                        else "visual_gesture_invalid_output"
                    )
                ),
            }
        else:
            action = {
                "candidate_id": candidate.candidate_id,
                "action_id": candidate.action_id,
                **(
                    {"category_id": candidate.category_id}
                    if candidate.category_id
                    else {}
                ),
                "execution_binding": dict(candidate.execution_binding),
                "execute": True,
                "support_status": "supported",
                "fallback_applied": False,
                "allow_adjacent_repeat": True,
            }
        context = {
            "selection_stages": 1,
            "selection_mode": "visual_gesture_generation",
            "selection_basis": "visual_gesture_semantic_label",
            "selection_definition_source": "short_definition_visual",
            "candidate_count": candidate_count,
            "output_valid": output_valid,
            "output_chars": output_chars,
            "compute_ms": round(compute_ms, 3),
            "generic_action_scoring_bypassed": True,
            "visual_observation_confidence": (
                confidence.as_dict() if confidence is not None else None
            ),
            "action_timing_breakdown": {
                "selection_mode": "visual_gesture_generation",
                "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                "total_ms": round(compute_ms, 3),
            },
        }
        return action, [], round(compute_ms, 3), context

    async def _run_visual_gesture_probe_after_route(
        self,
        turn: TurnBuffer,
        images: list[Any],
        image_roles: list[str],
        visual_scope_future: asyncio.Future[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]] | None:
        route_code = await visual_scope_future
        if route_code not in VISUAL_GESTURE_COPY_ROUTES:
            return None
        return await self._run_visual_gesture_probe(turn, images, image_roles)

    async def _run_visual_gesture_probe(
        self,
        turn: TurnBuffer,
        images: list[Any],
        image_roles: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        """Generate one semantic label and map it to the catalog without PPL."""

        self._ensure_turn_processing(turn)
        candidates = self._visual_gesture_generation_candidates(turn)
        request_id = f"{turn.request_base}-visual-gesture"
        started = time.perf_counter()
        first_token_ms: float | None = None
        text_parts: list[str] = []
        output_token_logprobs: list[Any] = []
        output_top_logprobs: list[Any] = []
        if not candidates or IMAGE_ROLE_USER_CAMERA not in image_roles:
            return self._visual_gesture_action_result(
                None,
                compute_ms=0.0,
                output_valid=True,
                output_chars=0,
                ttft_ms=None,
                candidate_count=len(candidates),
            )
        request, forwarded_roles = self._build_visual_gesture_request(
            turn, images, image_roles, candidates
        )
        self._register_turn_request(turn, request_id)
        emit_structured_log(
            "action",
            "visual_gesture_probe_submitted",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            candidate_count=len(candidates),
            # Catalog labels are static, non-user data.  Persist the exact
            # closed set locally so an invalid classifier output can be
            # diagnosed against the choices that were actually available.
            candidate_labels=[
                visual_gesture_output_label(candidate)
                for candidate in candidates
            ],
            forwarded_image_count=len(forwarded_roles),
            after_commit_ms=self._after_commit_ms(turn),
        )
        try:
            completion_stream = getattr(self.client, "completion_stream", None)
            if callable(completion_stream):
                stream = completion_stream(request, request_id=request_id)
                async with aclosing(stream):
                    async for chunk in stream:
                        self._ensure_turn_processing(turn)
                        if chunk.modality == "text" and chunk.text:
                            if first_token_ms is None:
                                first_token_ms = (
                                    time.perf_counter() - started
                                ) * 1000.0
                            text_parts.append(chunk.text)
                        if chunk.output_token_logprobs is not None:
                            output_token_logprobs.extend(
                                chunk.output_token_logprobs
                            )
                        if chunk.output_top_logprobs is not None:
                            output_top_logprobs.extend(chunk.output_top_logprobs)
            else:
                result = await self.client.completion(
                    request, request_id=request_id
                )
                if result.text:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                    text_parts.append(result.text)
                if result.output_token_logprobs is not None:
                    output_token_logprobs.extend(result.output_token_logprobs)
                if result.output_top_logprobs is not None:
                    output_top_logprobs.extend(result.output_top_logprobs)
        except asyncio.CancelledError:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                await asyncio.gather(abort(request_id), return_exceptions=True)
            raise
        except Exception as exc:
            emit_structured_log(
                "error",
                "visual_gesture_probe_failed",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._visual_gesture_action_result(
                None,
                compute_ms=elapsed_ms,
                output_valid=False,
                output_chars=0,
                ttft_ms=first_token_ms,
                candidate_count=len(candidates),
            )
        finally:
            self._unregister_turn_request(turn, request_id)

        text = "".join(text_parts)
        normalized_output = normalize_visual_gesture_output(text, candidates)
        selected = parse_visual_gesture_output(text, candidates)
        output_valid = normalized_output is not None
        confidence = summarize_visual_observation_confidence(
            output_token_logprobs,
            output_top_logprobs,
        )
        if selected is not None and not confidence.accepted:
            selected = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        emit_structured_log(
            "action",
            "visual_gesture_probe_completed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            candidate_count=len(candidates),
            selected_candidate_id=(selected.candidate_id if selected else None),
            selected_candidate_label=(
                visual_gesture_output_label(selected) if selected else None
            ),
            # This probe is capped at 24 generated tokens and is instructed to
            # emit only a catalog label.  Keeping its bounded output in the
            # local action log makes format and out-of-scope failures
            # observable without logging user audio, transcripts, or images.
            model_output=text,
            normalized_output=normalized_output,
            output_valid=output_valid,
            output_chars=len(text),
            visual_observation_confidence=confidence.as_dict(),
            ttft_ms=round(first_token_ms or elapsed_ms, 3),
            total_ms=round(elapsed_ms, 3),
        )
        return self._visual_gesture_action_result(
            selected,
            compute_ms=elapsed_ms,
            output_valid=output_valid,
            output_chars=len(text),
            ttft_ms=first_token_ms,
            candidate_count=len(candidates),
            confidence=confidence,
        )
