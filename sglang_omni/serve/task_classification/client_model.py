"""Prompted turn-router adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
)

from .contracts import ClassificationMediaRef, TaskClassificationRequest
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .route_candidates import (
    ROUTE_CANDIDATE_BY_ID,
    ROUTE_CANDIDATES,
    SCORING_CANDIDATE_CATALOG,
    RouteCandidate,
)


logger = logging.getLogger(__name__)


_CUE_SYSTEM_PROMPT = """你是请求承接计划提取器。只处理已经确定为 delegate 的当前请求。
只输出单行 JSON 对象，字段必须恰好是 verb、object、language。
verb 表示正在进行的行为；object 必须是规范化名词性任务对象，不得复制完整问句、
不得重复 verb，也不得声称工具成功或结果已找到。language 只能是 zh-CN 或 en-US。"""

_SCORING_SYSTEM_PROMPT = f"""{SYSTEM_PROMPT}

[FIXED_CANDIDATE_SCORING]
本阶段不要生成 JSON。根据上述规则，只选择下列一个候选标签；候选 suffix 就是标签本身。
delegate 的 request_cue 将在下一阶段生成，cue_required 仅表示必须生成。
{SCORING_CANDIDATE_CATALOG}
"""

_SCORING_OUTPUT_PROMPT = "[OUTPUT]\n只输出一个候选标签（例如 R001）："


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...

    async def score_action_suffixes(
        self,
        request: ActionSuffixScoreRequest,
    ) -> ActionSuffixScoreResult: ...


class SglangClientTaskClassificationModel:
    """Route a Turn with the shared base model; a trained model can replace it."""

    def __init__(
        self,
        client: CompletionClient,
        *,
        model_id: str,
        model_version: str,
    ) -> None:
        if not model_id.strip() or not model_version.strip():
            raise ValueError("turn-router model_id and model_version are required")
        self._client = client
        self.model_id = model_id
        self.model_version = model_version

    async def classify(self, request: TaskClassificationRequest) -> str:
        score = getattr(self._client, "score_action_suffixes", None)
        if callable(score):
            try:
                return await self._classify_by_candidate_score(request)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "turn-router candidate scoring failed; using JSON generation "
                    "request_id=%s turn_id=%s",
                    request.request_id,
                    request.turn_id,
                    exc_info=True,
                )
        return await self._classify_by_generation(request)

    async def prewarm(self) -> None:
        """Populate the immutable public router prefix without decoding JSON."""

        await self._score_route_candidates(
            TaskClassificationRequest(
                request_id="turn-router-prewarm",
                session_id="turn-router-prewarm",
                turn_id="turn-router-prewarm",
                identity_epoch=0,
                input_revision=0,
                text="你好",
                media=(),
            )
        )

    async def _classify_by_candidate_score(
        self,
        request: TaskClassificationRequest,
    ) -> str:
        result = await self._score_route_candidates(request)
        candidate = _select_candidate(result)
        cue = (
            await self._generate_request_cue(request, candidate)
            if candidate.requires_cue
            else None
        )
        return json.dumps(
            {
                "route_token": candidate.route_token,
                "output_directive": candidate.output_directive,
                "task_directive": candidate.task_directive,
                "media_directive": candidate.media_directive,
                "response_locale": candidate.response_locale,
                "request_cue": cue,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def _score_route_candidates(
        self,
        request: TaskClassificationRequest,
    ) -> ActionSuffixScoreResult:
        audio = request.media[0] if request.media else None
        prompt_parts = build_user_prompt(
            text=request.text,
            has_audio=audio is not None,
            history=request.router_history,
            brain1_capabilities=request.brain1_capabilities,
            brain2_capabilities=request.brain2_capabilities,
            has_active_agent=request.has_active_agent,
            pending_confirmation=request.pending_confirmation,
            follow_up_required=request.follow_up_required,
        )
        prefix = str(prompt_parts[0]["text"])
        return await self._client.score_action_suffixes(
            ActionSuffixScoreRequest(
                request_id=f"{request.request_id}:route-score",
                model=self.model_id,
                prefix=prefix,
                system_prompt=_SCORING_SYSTEM_PROMPT,
                language="zh",
                candidates=[item.scoring_candidate() for item in ROUTE_CANDIDATES],
                audios=[_audio_data_url(audio)] if audio is not None else [],
                images=[],
                sample_rate=16_000,
                current_text=request.text if request.text is not None else "",
                output_prompt=_SCORING_OUTPUT_PROMPT,
                micro_batch_size=32,
                session_id=request.session_id,
                session_instance_id=request.session_id,
                logical_request_id=request.turn_id,
                prefix_cache_namespace="turn-router:v5",
                cache_static_system_only=True,
                stage="router",
                admission_priority=0,
            )
        )

    async def _generate_request_cue(
        self,
        request: TaskClassificationRequest,
        candidate: RouteCandidate,
    ) -> dict[str, str]:
        audio = request.media[0] if request.media else None
        content = build_user_prompt(
            text=request.text,
            has_audio=audio is not None,
            history=request.router_history,
            brain1_capabilities=request.brain1_capabilities,
            brain2_capabilities=request.brain2_capabilities,
            has_active_agent=request.has_active_agent,
            pending_confirmation=request.pending_confirmation,
            follow_up_required=request.follow_up_required,
        )
        content.append(
            {
                "type": "text",
                "text": (
                    "[SELECTED_ROUTE]\n"
                    f"delegate|{candidate.response_locale}\n"
                    "[OUTPUT]\n只输出 request cue JSON。"
                ),
            }
        )
        metadata = self._metadata(request)
        if audio is not None:
            metadata["audios"] = [_audio_data_url(audio)]
        result = await self._client.completion(
            GenerateRequest(
                model=self.model_id,
                messages=[
                    Message(role="system", content=_CUE_SYSTEM_PROMPT),
                    Message(role="user", content=content),
                ],
                sampling=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=64,
                    stop=["\n"],
                ),
                stream=False,
                max_tokens=64,
                output_modalities=["text"],
                metadata=metadata,
            ),
            request_id=f"{request.request_id}:request-cue",
        )
        raw = json.loads(str(result.text))
        if not isinstance(raw, dict) or set(raw) != {"verb", "object", "language"}:
            raise ValueError("request cue generation returned an invalid object")
        if raw.get("language") != candidate.response_locale:
            raise ValueError("request cue language does not match the selected locale")
        if not all(isinstance(raw.get(key), str) and raw[key].strip() for key in raw):
            raise ValueError("request cue generation returned blank fields")
        return {key: raw[key].strip() for key in ("verb", "object", "language")}

    async def _classify_by_generation(
        self,
        request: TaskClassificationRequest,
    ) -> str:
        audio = request.media[0] if request.media else None
        metadata = self._metadata(request)
        if audio is not None:
            metadata["audios"] = [_audio_data_url(audio)]
        result = await self._client.completion(
            GenerateRequest(
                model=self.model_id,
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(
                        role="user",
                        content=build_user_prompt(
                            text=request.text,
                            has_audio=audio is not None,
                            history=request.router_history,
                            brain1_capabilities=request.brain1_capabilities,
                            brain2_capabilities=request.brain2_capabilities,
                            has_active_agent=request.has_active_agent,
                            pending_confirmation=request.pending_confirmation,
                            follow_up_required=request.follow_up_required,
                        ),
                    ),
                ],
                sampling=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=128,
                    stop=["\n"],
                ),
                stream=False,
                max_tokens=128,
                output_modalities=["text"],
                metadata=metadata,
            ),
            request_id=request.request_id,
        )
        return str(result.text)

    @staticmethod
    def _metadata(request: TaskClassificationRequest) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "task": "turn_router",
            "task_role": "turn_router",
            "contract_version": request.contract_version,
            "logical_request_id": request.turn_id,
            # Keep the immutable router prompt in one private cache namespace
            # for the lifetime of the D conversation session.
            "session_instance_id": request.session_id,
            "session_id": request.session_id,
        }
        return metadata


def _select_candidate(result: ActionSuffixScoreResult) -> RouteCandidate:
    if not result.scores:
        raise ValueError("turn-router candidate scoring returned no scores")
    unknown = [
        score.candidate_id
        for score in result.scores
        if score.candidate_id not in ROUTE_CANDIDATE_BY_ID
    ]
    if unknown:
        raise ValueError(
            f"turn-router candidate scoring returned unknown IDs: {unknown}"
        )
    top = max(result.scores, key=lambda item: item.mean_logprob)
    return ROUTE_CANDIDATE_BY_ID[top.candidate_id]


def _audio_data_url(media: ClassificationMediaRef) -> str:
    mime = _audio_mime_type(media.encoding)
    payload = media.payload
    if media.encoding.strip().lower() in {"pcm16", "pcm_s16le"}:
        if len(payload) % 2:
            raise ValueError("PCM16 turn-router audio has an incomplete sample")
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(payload)
        payload = output.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _audio_mime_type(encoding: str) -> str:
    normalized = encoding.strip().lower()
    if "/" in normalized:
        return normalized
    aliases = {
        "wav": "audio/wav",
        "pcm16": "audio/wav",
        "pcm_s16le": "audio/wav",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported audio encoding: {encoding}") from exc
