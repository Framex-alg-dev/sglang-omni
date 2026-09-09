# SPDX-License-Identifier: Apache-2.0
"""Client wrapper for coordinator-based pipelines."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import aclosing, asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator, Callable

import numpy as np

from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    FORMAT_MIME_TYPES,
    audio_to_base64,
    encode_audio,
    to_numpy,
)
from sglang_omni.client.types import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    AbortLevel,
    AbortResult,
    ClientError,
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
    SpeechResult,
    UsageInfo,
)
from sglang_omni.models.qwen3_omni.action_scoring import (
    CandidateScore,
    TokenScore,
    validate_action_suffix_request,
    validate_score_result,
)
from sglang_omni.models.qwen3_omni.prompt_localization import (
    localized_prompt,
    normalize_prompt_language,
)
from sglang_omni.preprocessing.image import is_prepared_image_wire
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage
from sglang_omni.utils.async_jsonl import enqueue_jsonl
from sglang_omni.utils.structured_logs import emit_structured_log


logger = logging.getLogger(__name__)

_ACTION_SCORE_TIMEOUT_S = float(os.environ.get("SGLANG_OMNI_ACTION_SCORE_TIMEOUT_S", "120"))
_ACTION_DEBUG_LOG_FILE = os.environ.get("SGLANG_OMNI_ACTION_DEBUG_LOG_FILE")


def _bounded_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("invalid %s=%r; using %d", name, raw_value, default)
        return default
    return max(minimum, min(maximum, value))


_ACTION_SCORE_MAX_INFLIGHT = _bounded_int_env(
    "SGLANG_OMNI_ACTION_SCORE_MAX_INFLIGHT",
    2,
    minimum=1,
    maximum=4,
)


class _PriorityAdmissionGate:
    """Small FIFO-within-priority gate for action-scoring submissions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._active = 0
        self._sequence = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._condition = asyncio.Condition()

    def _wake_locked(self) -> None:
        while self._active < self.capacity and self._waiters:
            _, _, waiter = heapq.heappop(self._waiters)
            if waiter.cancelled():
                continue
            self._active += 1
            waiter.set_result(None)

    async def acquire(self, priority: int) -> None:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        async with self._condition:
            self._sequence += 1
            heapq.heappush(
                self._waiters,
                (priority, self._sequence, waiter),
            )
            self._wake_locked()
        try:
            await waiter
        except BaseException:
            admitted = waiter.done() and not waiter.cancelled()
            if not admitted:
                waiter.cancel()
            else:
                await self.release()
            raise

    async def release(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError("action-scoring admission gate released too often")
            self._active -= 1
            self._wake_locked()

    @asynccontextmanager
    async def slot(self, priority: int):
        await self.acquire(priority)
        try:
            yield
        finally:
            await self.release()


def _summarize_debug_media(values: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for index, value in enumerate(values if isinstance(values, list) else []):
        if is_prepared_image_wire(value):
            summary.append(
                {
                    "index": index,
                    "type": "prepared_image_rgb",
                    "width": value["width"],
                    "height": value["height"],
                    "pixel_bytes": len(value["pixel_bytes"]),
                    "source_sha256": value["source_sha256"],
                    "pixel_sha256": value["pixel_sha256"],
                }
            )
            continue
        if not isinstance(value, str):
            summary.append(
                {"index": index, "type": type(value).__name__, "repr": repr(value)}
            )
            continue
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if value.startswith("data:") and "," in value:
            header, encoded = value.split(",", 1)
            summary.append(
                {
                    "index": index,
                    "type": "data_uri",
                    "header": header,
                    "encoded_chars": len(encoded),
                    "sha256": digest,
                }
            )
        else:
            summary.append(
                {
                    "index": index,
                    "type": "reference",
                    "value": value,
                    "chars": len(value),
                    "sha256": digest,
                }
            )
    return summary


def _action_request_debug_payload(
    request: ActionSuffixScoreRequest,
    omni_request: OmniRequest,
) -> dict[str, Any]:
    inputs = getattr(omni_request, "inputs", {}) or {}
    return {
        "request_id": request.request_id,
        "session_id": request.session_id,
        "stage": request.stage,
        "admission_priority": request.admission_priority,
        "logical_request_id": request.logical_request_id,
        "prefix": request.prefix,
        "session_instruction": request.session_instruction,
        "current_text": request.current_text,
        "output_prompt": request.output_prompt,
        "system_prompt": request.system_prompt,
        "messages": inputs.get("messages", []),
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "suffix": item.suffix,
                "action_id": item.action_id,
                "execution_binding": dict(item.execution_binding),
            }
            for item in request.candidates
        ],
        "audios": _summarize_debug_media(request.audios),
        "images": _summarize_debug_media(request.images),
        "history_audios": _summarize_debug_media(request.history_audios),
        "history_images": _summarize_debug_media(request.history_images),
        "avatar_state": dict(request.avatar_state),
        "params": getattr(omni_request, "params", {}),
        "metadata": getattr(omni_request, "metadata", {}),
    }


def _write_action_debug_record(record: dict[str, Any]) -> None:
    """Queue one replayable action-scoring diagnostic record for JSONL output."""
    event = str(record.get("event") or "action_diagnostic")
    log_type = (
        "error"
        if event.endswith(("_failed", "_timeout", "_cancelled"))
        else "diagnostic"
    )
    emit_structured_log(
        log_type,
        event,
        component="client",
        **{key: value for key, value in record.items() if key != "event"},
    )
    # Compatibility sink for deployments that explicitly request the legacy
    # single-file action log.  New deployments should use the partitioned
    # SGLANG_OMNI_REALTIME_LOG_DIR sink instead.
    if _ACTION_DEBUG_LOG_FILE:
        enqueue_jsonl(_ACTION_DEBUG_LOG_FILE, record)


class Client:
    """Internal client used by API adapters."""

    def __init__(
        self,
        coordinator: Coordinator,
        result_builder: Callable[[str, Any], GenerateChunk] | None = None,
        stream_builder: Callable[[str, StreamMessage], GenerateChunk] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._result_builder = result_builder or self._default_result_builder
        self._stream_builder = stream_builder or self._default_stream_builder
        self._action_scoring_gate = _PriorityAdmissionGate(
            _ACTION_SCORE_MAX_INFLIGHT
        )
        self._action_scoring_capacity = _ACTION_SCORE_MAX_INFLIGHT
        self._action_scoring_waiting = 0
        self._action_scoring_inflight = 0
        self._action_scoring_submitted_total = 0
        self._action_scoring_finished_total = 0
        self._action_scoring_cancelled_before_admission_total = 0

    def action_scoring_load(self) -> dict[str, int]:
        """Return API-process action admission load for resource telemetry."""

        return {
            "capacity": self._action_scoring_capacity,
            "waiting": self._action_scoring_waiting,
            "inflight": self._action_scoring_inflight,
            "submitted_total": self._action_scoring_submitted_total,
            "finished_total": self._action_scoring_finished_total,
            "cancelled_before_admission_total": (
                self._action_scoring_cancelled_before_admission_total
            ),
        }

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult:
        """Score all suffixes as one logical multimodal pipeline request."""
        score_started = time.perf_counter()
        validate_action_suffix_request(request)
        candidates = [
            {
                "candidate_id": item.candidate_id,
                "suffix": item.suffix,
                "action_id": item.action_id,
                "execution_binding": dict(item.execution_binding),
            }
            for item in request.candidates
        ]
        build_started = time.perf_counter()
        omni_request = self._build_action_scoring_request(request, candidates)
        client_build_ms = (time.perf_counter() - build_started) * 1000.0
        action_params = omni_request.params.get("action_scoring")
        if isinstance(action_params, dict):
            action_params["client_build_ms"] = round(client_build_ms, 3)
        phase: dict[str, Any] = {
            "client_build_ms": round(client_build_ms, 3),
            "name": "waiting_for_action_score_slot",
            "slot_wait_ms": None,
            "pipeline_ms": None,
            "admission_priority": request.admission_priority,
            "admission_capacity": self._action_scoring_capacity,
        }
        request_started = time.perf_counter()
        _write_action_debug_record({
            "event": "action_scoring_started",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "full_logical_input": _action_request_debug_payload(request, omni_request),
        })

        async def _submit() -> Any:
            slot_started = time.perf_counter()
            admitted = False
            self._action_scoring_submitted_total += 1
            self._action_scoring_waiting += 1
            try:
                async with self._action_scoring_gate.slot(
                    request.admission_priority
                ):
                    admitted = True
                    self._action_scoring_waiting -= 1
                    self._action_scoring_inflight += 1
                    phase["slot_wait_ms"] = round(
                        (time.perf_counter() - slot_started) * 1000.0, 3
                    )
                    phase["name"] = "coordinator_pipeline"
                    pipeline_started = time.perf_counter()
                    try:
                        return await self._coordinator.submit(
                            request.request_id, omni_request
                        )
                    finally:
                        phase["pipeline_ms"] = round(
                            (time.perf_counter() - pipeline_started) * 1000.0,
                            3,
                        )
            finally:
                if admitted:
                    self._action_scoring_inflight -= 1
                    self._action_scoring_finished_total += 1
                else:
                    self._action_scoring_waiting -= 1
                    self._action_scoring_cancelled_before_admission_total += 1

        task = asyncio.create_task(
            _submit(), name=f"action-score-{request.request_id}"
        )
        try:
            raw_result = await asyncio.wait_for(task, timeout=_ACTION_SCORE_TIMEOUT_S)
        except asyncio.TimeoutError:
            diagnostic = {
                "event": "action_scoring_timeout",
                "timestamp_unix_ms": round(time.time() * 1000.0),
                "timeout_s": _ACTION_SCORE_TIMEOUT_S,
                "phase": dict(phase),
                "full_logical_input": _action_request_debug_payload(request, omni_request),
            }
            logger.error(
                "action scoring timed out; full logical input=%s",
                json.dumps(diagnostic, ensure_ascii=False, default=str),
                exc_info=True,
            )
            _write_action_debug_record(diagnostic)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self._coordinator.abort(request.request_id)
            raise
        except asyncio.CancelledError:
            diagnostic = {
                "event": "action_scoring_cancelled",
                "timestamp_unix_ms": round(time.time() * 1000.0),
                "phase": dict(phase),
                "full_logical_input": _action_request_debug_payload(request, omni_request),
            }
            logger.warning(
                "action scoring cancelled; full logical input=%s",
                json.dumps(diagnostic, ensure_ascii=False, default=str),
            )
            _write_action_debug_record(diagnostic)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self._coordinator.abort(request.request_id)
            raise
        except Exception as exc:
            diagnostic = {
                "event": "action_scoring_failed",
                "request_id": request.request_id,
                "session_id": request.session_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp_unix_ms": round(time.time() * 1000.0),
                "phase": dict(phase),
                "full_logical_input": _action_request_debug_payload(request, omni_request),
            }
            logger.error(
                "action scoring request failed; full logical input=%s",
                json.dumps(diagnostic, ensure_ascii=False, default=str),
                exc_info=True,
            )
            _write_action_debug_record(diagnostic)
            raise
        result_processing_started = time.perf_counter()
        result = _coerce_action_score_result(raw_result, request)
        try:
            validate_score_result(request, result)
        except RuntimeError as exc:
            raise ClientError(str(exc)) from exc
        result_processing_ms = (
            time.perf_counter() - result_processing_started
        ) * 1000.0
        result.stats.update(
            {
                "action_slot_wait_ms": float(phase["slot_wait_ms"] or 0.0),
                "action_admission_priority": request.admission_priority,
                "action_admission_capacity": self._action_scoring_capacity,
                "coordinator_pipeline_ms": float(phase["pipeline_ms"] or 0.0),
                "client_result_processing_ms": round(result_processing_ms, 3),
                "client_total_ms": round(
                    (time.perf_counter() - score_started) * 1000.0, 3
                ),
            }
        )
        _write_action_debug_record({
            "event": "action_scoring_completed",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "request_id": request.request_id,
            "session_id": request.session_id,
            "elapsed_ms": round((time.perf_counter() - request_started) * 1000.0, 3),
            "phase": dict(phase),
            "prefix_cached": result.prefix_cached,
            "stats": result.stats,
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob,
                    "mean_nll": score.mean_nll,
                    "ppl": score.ppl,
                    "token_scores": [
                        {"token_id": token.token_id, "logprob": token.logprob}
                        for token in score.token_scores
                    ],
                }
                for score in result.scores
            ],
        })
        return result


    @staticmethod
    def _build_action_warmup_request(
        *,
        request_id: str,
        model: str,
        stage: str,
        candidate_prefix: str,
        candidate_count: int,
        audio_path: str | None = None,
        language: str = "zh",
    ) -> ActionSuffixScoreRequest:
        candidates = [
            ActionScoreCandidate(
                candidate_id=f"{candidate_prefix}{index:03d}",
                suffix=f"{candidate_prefix}{index:03d}",
                action_id=f"{candidate_prefix}{index:03d}",
            )
            for index in range(candidate_count)
        ]
        definitions = "; ".join(
            f"{item.candidate_id}=warmup candidate" for item in candidates
        )
        if stage == "category":
            system_prompt = localized_prompt(
                language,
                zh=(
                    "你是数字人动作类别识别器。只能输出一个 category_id。"
                    f"固定类别集合：{definitions}"
                ),
                en=(
                    "You are a digital-character action category classifier. Output "
                    f"exactly one category_id. Fixed category set: {definitions}"
                ),
            )
            prefix = localized_prompt(
                language,
                zh="请根据当前输入选择一个动作类别。下一步 category_id 是：",
                en="Select an action category for the current input. Next category_id:",
            )
        elif stage == "child":
            system_prompt = localized_prompt(
                language,
                zh=(
                    "你是数字人动作识别器。只能输出一个 action_id。"
                    f"固定子动作集合：{definitions}"
                ),
                en=(
                    "You are a digital-character action classifier. Output exactly "
                    f"one action_id. Fixed child action set: {definitions}"
                ),
            )
            prefix = localized_prompt(
                language,
                zh="请根据当前输入选择一个子动作。下一步 action_id 是：",
                en="Select a child action for the current input. Next action_id:",
            )
        elif stage == "single":
            system_prompt = localized_prompt(
                language,
                zh=(
                    "你是数字人动作识别器。只能输出一个 action_id。"
                    f"固定具体动作集合：{definitions}"
                ),
                en=(
                    "You are a digital-character action classifier. Output exactly "
                    f"one action_id. Fixed concrete action set: {definitions}"
                ),
            )
            prefix = localized_prompt(
                language,
                zh="请根据当前输入选择一个具体动作。下一步 action_id 是：",
                en="Select a concrete action for the current input. Next action_id:",
            )
        else:
            raise ValueError(f"unsupported action warmup stage: {stage!r}")
        return ActionSuffixScoreRequest(
            request_id=request_id,
            model=model,
            prefix=prefix,
            language=language,
            candidates=candidates,
            audios=[audio_path] if audio_path else [],
            images=[],
            sample_rate=16000,
            system_prompt=system_prompt,
            micro_batch_size=64,
            stage=stage,
            logical_request_id=f"warmup-process-{language}",
            suffix_tokenization_mode="short_id",
        )

    async def warmup_action_score(
        self,
        *,
        model: str,
        category_count: int = 60,
        child_count: int = 8,
        selection_mode: str = "hierarchical",
        timeout_s: float = 30.0,
        audio_path: str | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        # Warm the action-score request path before accepting user turns. The
        # first stage may carry one internal audio asset; requests still have
        # no session/history and all results are discarded.
        if category_count <= 0 or child_count <= 0:
            raise ValueError("warmup candidate counts must be positive")
        if selection_mode not in {"hierarchical", "flat_children"}:
            raise ValueError(f"unsupported action selection mode: {selection_mode!r}")
        started = time.perf_counter()
        logger.info(
            "[ACTION_WARMUP] started model=%s language=%s mode=%s category_candidates=%d child_candidates=%d audio=%s",
            model,
            language,
            selection_mode,
            category_count,
            child_count,
            audio_path or "disabled",
        )
        _write_action_debug_record({
            "event": "action_score_warmup_started",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "model": model,
            "language": language,
            "selection_mode": selection_mode,
            "category_candidates": category_count,
            "child_candidates": child_count,
            "audio_path": audio_path,
        })

        async def run_warmup() -> tuple[float | None, float | None, float | None]:
            if selection_mode == "flat_children":
                single_started = time.perf_counter()
                await self.score_action_suffixes(
                    self._build_action_warmup_request(
                        request_id=f"warmup-process-{language}-single",
                        model=model,
                        stage="single",
                        candidate_prefix="D",
                        candidate_count=child_count,
                        audio_path=audio_path,
                        language=language,
                    )
                )
                return None, None, (time.perf_counter() - single_started) * 1000.0

            category_started = time.perf_counter()
            await self.score_action_suffixes(
                self._build_action_warmup_request(
                    request_id=f"warmup-process-{language}-category",
                    model=model,
                    stage="category",
                    candidate_prefix="C",
                    candidate_count=category_count,
                    audio_path=audio_path,
                    language=language,
                )
            )
            category_ms = (time.perf_counter() - category_started) * 1000.0

            child_started = time.perf_counter()
            await self.score_action_suffixes(
                self._build_action_warmup_request(
                    request_id=f"warmup-process-{language}-child",
                    model=model,
                    stage="child",
                    candidate_prefix="D",
                    candidate_count=child_count,
                    language=language,
                )
            )
            child_ms = (time.perf_counter() - child_started) * 1000.0
            return category_ms, child_ms, None

        try:
            category_ms, child_ms, single_ms = await asyncio.wait_for(
                run_warmup(), timeout=timeout_s
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            diagnostic = {
                "event": "action_score_warmup_failed",
                "timestamp_unix_ms": round(time.time() * 1000.0),
                "model": model,
                "language": language,
                "selection_mode": selection_mode,
                "audio_path": audio_path,
                "elapsed_ms": round(elapsed_ms, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            logger.warning(
                "[ACTION_WARMUP] failed elapsed_ms=%.3f error=%s",
                elapsed_ms,
                exc,
                exc_info=True,
            )
            _write_action_debug_record(diagnostic)
            return {
                "ready": False,
                "language": language,
                "selection_mode": selection_mode,
                "audio_warmup_enabled": audio_path is not None,
                "elapsed_ms": round(elapsed_ms, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "ready": True,
            "language": language,
            "selection_mode": selection_mode,
            "audio_warmup_enabled": audio_path is not None,
            "elapsed_ms": round(elapsed_ms, 3),
            "category_ms": round(category_ms, 3) if category_ms is not None else None,
            "child_ms": round(child_ms, 3) if child_ms is not None else None,
            "single_ms": round(single_ms, 3) if single_ms is not None else None,
        }
        logger.info(
            "[ACTION_WARMUP] completed elapsed_ms=%.3f mode=%s category_ms=%s child_ms=%s single_ms=%s",
            elapsed_ms,
            selection_mode,
            category_ms,
            child_ms,
            single_ms,
        )
        _write_action_debug_record({
            "event": "action_score_warmup_completed",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "model": model,
            **result,
        })
        return result

    async def prefill_action_catalog(
        self,
        *,
        model: str,
        request_id: str | None = None,
        system_prompt: str,
        candidates: list[ActionScoreCandidate],
        prefix_cache_namespace: str,
        stage: str,
        language: str = "zh",
        session_instruction: str = "",
        admission_priority: int = 3,
    ) -> bool:
        """Prefill one immutable action catalog prefix."""
        if not candidates:
            return False
        probe = candidates[0]
        request = ActionSuffixScoreRequest(
            request_id=request_id or f"catalog-prefill-{stage}-{uuid.uuid4().hex}",
            model=model,
            prefix=localized_prompt(
                language,
                zh="请根据当前输入选择动作。下一步 action_id 是：",
                en="Select an action for the current input. Next action_id:",
            ),
            language=language,
            candidates=[probe],
            audios=[],
            images=[],
            sample_rate=16000,
            session_instruction=session_instruction,
            # Match the realtime Turn's list-form message layout so the
            # rendered tokens through the Session boundary are identical.
            current_text="",
            system_prompt=system_prompt,
            stage=stage,
            admission_priority=admission_priority,
            logical_request_id=f"catalog-prefill-{prefix_cache_namespace}",
            prefix_cache_namespace=prefix_cache_namespace,
            cache_static_system_only=not bool(session_instruction),
            suffix_tokenization_mode="short_id",
        )
        try:
            await self.score_action_suffixes(request)
        except Exception:
            logger.warning(
                "[ACTION_CATALOG_PREFILL] failed namespace=%s stage=%s",
                prefix_cache_namespace,
                stage,
                exc_info=True,
            )
            return False
        logger.info(
            "[ACTION_CATALOG_PREFILL] ready namespace=%s stage=%s",
            prefix_cache_namespace,
            stage,
        )
        return True


    @staticmethod
    def _compact_image_indices(indices: list[int]) -> str:
        if not indices:
            return ""
        ranges: list[str] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = index
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ",".join(ranges)

    @staticmethod
    def _current_image_content_parts(
        image_roles: list[str],
        language: str = "zh",
    ) -> list[dict[str, Any]]:
        """Put one compact role map before all current image placeholders."""
        user_indices = [
            i for i, role in enumerate(image_roles, 1) if role == "user_camera"
        ]
        avatar_indices = [
            i for i, role in enumerate(image_roles, 1) if role == "avatar_state"
        ]
        descriptions: list[str] = []
        if user_indices:
            index_text = Client._compact_image_indices(user_indices)
            descriptions.append(
                localized_prompt(
                    language,
                    zh=f"用户摄像头画面={index_text}（只描述用户及其环境）",
                    en=f"User camera view={index_text} (describe only the user and their environment)",
                )
            )
        if avatar_indices:
            latest = avatar_indices[-1]
            index_text = Client._compact_image_indices(avatar_indices)
            descriptions.append(
                localized_prompt(
                    language,
                    zh=(
                        f"数字人当前状态画面={index_text}；"
                        f"画面{latest}是本轮时间最新的数字人照片，"
                        "当前可视姿态和行为以它为准"
                    ),
                    en=(
                        f"Current digital character state view={index_text}; image "
                        f"{latest} is the latest character image in this interaction "
                        "and is authoritative for the currently visible pose and behavior"
                    ),
                )
            )
        elif user_indices:
            descriptions.append(
                localized_prompt(
                    language,
                    zh=(
                        "本轮没有数字人当前状态画面，数字人姿态只能使用结构化的"
                        "数字人当前状态信息，不得从用户摄像头画面或历史图片推断"
                    ),
                    en=(
                        "No current digital character state view is provided in this "
                        "interaction. Use only structured current character state "
                        "information for the character pose; do not infer it from the "
                        "user camera view or historical images"
                    ),
                )
            )
        if not descriptions:
            descriptions.append(
                localized_prompt(
                    language,
                    zh="当前图片用途未提供",
                    en="Current image purposes are not provided",
                )
            )
        separator = localized_prompt(language, zh="；", en="; ")
        return [
            {
                "type": "text",
                "text": localized_prompt(
                    language,
                    zh="[当前图片用途] " + separator.join(descriptions) + "。",
                    en="[Current image purposes] "
                    + separator.join(descriptions)
                    + ".",
                ),
            },
            *({"type": "image"} for _ in image_roles),
        ]

    @staticmethod
    def _action_instruction_content(
        content: Any,
        instruction: str,
        *,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        language: str = "zh",
        current_text: str | None = None,
        output_prompt: str | None = None,
        text_role: str = "user_input",
    ) -> Any:
        """Append the action instruction to a user message.

        Media placeholders must be kept in the message list when the
        corresponding top-level media payload is present. Otherwise the
        multimodal preprocessor cannot distinguish historical media from the
        current turn.
        """
        ordered_current_turn = current_text is not None or output_prompt is not None
        if ordered_current_turn:
            parts: list[Any] = [{"type": "text", "text": instruction}]
            if images:
                parts.extend(
                    Client._current_image_content_parts(image_roles, language)
                )
            parts.extend({"type": "audio"} for _ in audios)
            if isinstance(current_text, str) and current_text.strip():
                label = localized_prompt(
                    language,
                    zh=(
                        "[当前用户文本]"
                        if text_role == "user_input"
                        else "[数字人本轮将说出的文本]"
                    ),
                    en=(
                        "[Current user text]"
                        if text_role == "user_input"
                        else "[Text the digital character will say this interaction]"
                    ),
                )
                parts.append(
                    {"type": "text", "text": f"{label}\n{current_text.strip()}"}
                )
            if isinstance(output_prompt, str) and output_prompt.strip():
                parts.append({"type": "text", "text": output_prompt.strip()})
            return parts

        if isinstance(content, list):
            parts = [dict(part) if isinstance(part, dict) else part for part in content]
            parts.extend({"type": "audio"} for _ in audios)
            if images:
                parts.extend(
                    Client._current_image_content_parts(image_roles, language)
                )
            parts.append({"type": "text", "text": instruction})
            return parts

        if not audios and not images and isinstance(content, str):
            return f"{content}\n{instruction}"

        parts = []
        if content is not None:
            parts.append({"type": "text", "text": str(content)})
        parts.extend({"type": "audio"} for _ in audios)
        if images:
            parts.extend(Client._current_image_content_parts(image_roles, language))
        parts.append({"type": "text", "text": instruction})
        return parts

    @staticmethod
    def _build_action_context_messages(
        history: list[dict[str, Any]],
        instruction: str,
        *,
        session_instruction: str = "",
        avatar_state: dict[str, Any] | None,
        system_prompt: str | None,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        language: str = "zh",
        current_text: str | None = None,
        output_prompt: str | None = None,
        text_role: str = "user_input",
    ) -> list[dict[str, Any]]:
        """Build action context with static catalog first and current turn last.

        The caller controls whether a just-generated assistant response is
        included in ``history``. If history already ends in a user message,
        the instruction is merged into that message; if it ends in an
        assistant message, a new user message is appended.
        """
        messages: list[dict[str, Any]] = []
        # Keep the catalog at the beginning so its KV can be prefetched at
        # session.start. State is turn-local and must not precede the cache.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Keep immutable Session guidance first and turn-local state directly
        # after it.  This preserves the intended authority order and gives the
        # preprocessor an exact, explicit Session KV-cache boundary.
        current_instruction = session_instruction
        if current_instruction and not current_instruction.endswith("\n"):
            current_instruction += "\n"
        has_avatar_image = "avatar_state" in image_roles
        # Current structured state complements the current avatar image. A
        # previous-action ID is deliberately never model-visible: animation
        # transition and reset are responsibilities of the animation engine.
        state_for_prompt = {
            key: value
            for key, value in dict(avatar_state or {}).items()
            if key != "current_action_id"
        }
        if state_for_prompt:
            state_labels = (
                {
                    "state_description": "本轮主动场景约束",
                }
                if normalize_prompt_language(language) == "zh"
                else {
                    "state_description": "proactive-scene constraints for this interaction",
                }
            )
            state_for_prompt = {
                state_labels.get(key, key): value
                for key, value in state_for_prompt.items()
            }
            state_text = json.dumps(
                state_for_prompt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            state_label = localized_prompt(
                language,
                zh=(
                    "本轮动作选择补充信息"
                    if has_avatar_image
                    else "数字人当前状态信息"
                ),
                en=(
                    "Additional action-selection information for this interaction"
                    if has_avatar_image
                    else "Current digital character state information"
                ),
            )
            delimiter = localized_prompt(language, zh="：", en=": ")
            current_instruction += f"{state_label}{delimiter}{state_text}\n"
        current_instruction += instruction

        messages.extend(dict(message) for message in history)

        # ``audios``/``images`` contain the complete media payload for the
        # action-scoring request, including media from earlier turns. A
        # history message may already carry placeholders for some of those
        # files. Add only missing placeholders to the latest user turn;
        # otherwise the multimodal preprocessor rejects the request because
        # placeholder counts do not match the top-level media arrays.
        existing_audio_placeholders = sum(
            1
            for message in messages
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
            if isinstance(part, dict) and part.get("type") == "audio"
        )
        existing_image_placeholders = sum(
            1
            for message in messages
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
            if isinstance(part, dict) and part.get("type") == "image"
        )
        missing_audios = [""] * max(len(audios) - existing_audio_placeholders, 0)
        missing_images = [""] * max(len(images) - existing_image_placeholders, 0)
        missing_image_roles = (
            image_roles[-len(missing_images) :]
            if missing_images and image_roles
            else ["unknown"] * len(missing_images)
        )
        ordered_current_turn = current_text is not None or output_prompt is not None
        if ordered_current_turn:
            current_content = Client._action_instruction_content(
                None,
                current_instruction,
                audios=missing_audios,
                images=missing_images,
                image_roles=missing_image_roles,
                language=language,
                current_text=current_text,
                output_prompt=output_prompt,
                text_role=text_role,
            )
            messages.append({"role": "user", "content": current_content})
        elif messages and messages[-1].get("role") == "user":
            latest_user = messages.pop()
            latest_user["content"] = Client._action_instruction_content(
                latest_user.get("content"),
                current_instruction,
                audios=missing_audios,
                images=missing_images,
                image_roles=missing_image_roles,
                language=language,
            )
            messages.append(latest_user)
        else:
            current_content: Any
            if missing_audios or missing_images:
                current_content = [
                    *({"type": "audio"} for _ in missing_audios),
                    *(
                        Client._current_image_content_parts(
                            missing_image_roles, language
                        )
                        if missing_images
                        else []
                    ),
                    {"type": "text", "text": current_instruction},
                ]
            else:
                current_content = current_instruction
            messages.append({"role": "user", "content": current_content})
        return messages

    @staticmethod
    def _build_action_scoring_request(
        request: ActionSuffixScoreRequest,
        candidates: list[dict[str, Any]] | None = None,
    ) -> OmniRequest:
        """Build one multimodal request from the complete session turn."""
        messages = Client._build_action_context_messages(
            request.history,
            request.prefix,
            session_instruction=request.session_instruction,
            avatar_state=request.avatar_state,
            system_prompt=request.system_prompt,
            audios=[*request.history_audios, *request.audios],
            images=[*request.history_images, *request.images],
            image_roles=request.image_roles,
            language=request.language,
            current_text=request.current_text,
            output_prompt=request.output_prompt,
            text_role=request.text_role,
        )
        if candidates is None:
            candidates = [
                {
                    "candidate_id": item.candidate_id,
                    "suffix": item.suffix,
                    "action_id": item.action_id,
                    "execution_binding": dict(item.execution_binding),
                }
                for item in request.candidates
            ]
        # If the supplied history ends with a user message, the builder
        # merges the current instruction/media into that message.  Exclude
        # that message from the reusable boundary; otherwise the current turn
        # would be marked as cacheable history.
        history_message_count = len(request.history)
        history_audio_count = len(request.history_audios)
        history_image_count = len(request.history_images)
        if request.history and request.history[-1].get("role") == "user":
            last_content = request.history[-1].get("content")
            last_parts = last_content if isinstance(last_content, list) else []
            history_message_count -= 1
            history_audio_count -= sum(
                1
                for part in last_parts
                if isinstance(part, dict) and part.get("type") == "audio"
            )
            history_image_count -= sum(
                1
                for part in last_parts
                if isinstance(part, dict) and part.get("type") == "image"
            )
        history_audio_count = max(history_audio_count, 0)
        history_image_count = max(history_image_count, 0)
        metadata: dict[str, Any] = {
            "task": "action_suffix_scoring",
            "model": request.model,
            "language": request.language,
            "turn_origin": request.turn_origin,
            "text_role": request.text_role,
        }
        if request.trigger is not None:
            metadata["trigger"] = request.trigger
        if request.session_id is not None:
            metadata["session_id"] = request.session_id
        if request.avatar_state:
            metadata["avatar_state"] = dict(request.avatar_state)
        metadata["action_stage"] = request.stage
        if request.logical_request_id is not None:
            metadata["logical_request_id"] = request.logical_request_id
        return OmniRequest(
            inputs={
                "messages": messages,
                "audios": [*request.history_audios, *request.audios],
                "images": [*request.history_images, *request.images],
                "audio_target_sr": request.sample_rate,
            },
            params={
                "max_new_tokens": 0,
                "action_scoring": {
                    "prefix": request.prefix,
                    "session_instruction": request.session_instruction,
                    "language": request.language,
                    "candidates": candidates,
                    "micro_batch_size": request.micro_batch_size,
                    "sample_rate": request.sample_rate,
                    "client_started_at": time.perf_counter(),
                    "static_system_prompt": request.system_prompt,
                    "prefix_cache_namespace": request.prefix_cache_namespace,
                    "cache_static_system_only": request.cache_static_system_only,
                    "admission_priority": request.admission_priority,
                    "action_context_cache_key": request.action_context_cache_key,
                    "turn_origin": request.turn_origin,
                    "text_role": request.text_role,
                    "trigger": request.trigger,
                    "history_message_count": history_message_count,
                    "history_audio_count": history_audio_count,
                    "history_image_count": history_image_count,
                    "suffix_tokenization_mode": request.suffix_tokenization_mode,
                },
            },
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Low-level generate (backward compatible)
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        req_id = request_id or str(uuid.uuid4())
        omni_request = self._build_omni_request(request)
        if request.stream:
            coordinator_stream = self._coordinator.stream(req_id, omni_request)
            async with aclosing(coordinator_stream):
                async for msg in coordinator_stream:
                    if isinstance(msg, StreamMessage):
                        yield self._stream_builder(req_id, msg)
                    else:
                        yield self._result_builder(req_id, msg.result)
            return

        result = await self._coordinator.submit(req_id, omni_request)
        yield self._result_builder(req_id, result)

    # ------------------------------------------------------------------
    # High-level: non-streaming completion
    # ------------------------------------------------------------------

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        """Run a non-streaming completion and return an aggregated result.

        Iterates ``generate()``, accumulates text, concatenates audio chunks,
        and encodes audio to base64.

        Raises:
            ClientError: If the pipeline produces no response at all.
        """
        text_parts: list[str] = []
        audio_chunks: list[Any] = []
        sample_rate: int | None = None
        last_chunk: GenerateChunk | None = None
        finish_reason: str | None = None
        logprobs_parts: list[Any] = []
        saw_output_token_logprobs = False
        omni_rollout: dict[str, Any] | None = None
        weight_version: str | None = None

        async for chunk in self.generate(request, request_id=request_id):
            last_chunk = chunk
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if chunk.output_token_logprobs is not None:
                saw_output_token_logprobs = True
                logprobs_parts.extend(chunk.output_token_logprobs)
            if chunk.omni_rollout is not None:
                omni_rollout = chunk.omni_rollout
            if chunk.weight_version is not None:
                weight_version = chunk.weight_version

        if last_chunk is None:
            raise ClientError("No response from pipeline")

        full_text = "".join(text_parts)

        audio: CompletionAudio | None = None
        if audio_chunks:
            if len(audio_chunks) == 1:
                combined = audio_chunks[0]
            else:
                arrays = [to_numpy(c) for c in audio_chunks]
                axis = -1 if arrays[0].ndim > 1 else 0
                combined = np.concatenate(arrays, axis=axis)
            audio_b64 = audio_to_base64(
                combined,
                sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
                output_format=audio_format,
            )
            audio = CompletionAudio(
                id=f"audio-{request_id}",
                data=audio_b64,
                transcript=full_text if full_text else None,
            )

        return CompletionResult(
            request_id=request_id,
            text=full_text,
            audio=audio,
            finish_reason=finish_reason or "stop",
            usage=last_chunk.usage,
            output_token_logprobs=(
                logprobs_parts if saw_output_token_logprobs else None
            ),
            omni_rollout=omni_rollout,
            weight_version=weight_version,
        )

    # ------------------------------------------------------------------
    # High-level: streaming completion
    # ------------------------------------------------------------------

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        """Iterate ``generate()`` and yield high-level stream chunks.

        Audio data is base64-encoded before yielding so that callers never
        need to touch numpy / raw bytes.
        """
        streamed_text = ""
        generate_stream = self.generate(request, request_id=request_id)
        async with aclosing(generate_stream):
            async for chunk in generate_stream:
                audio_b64: str | None = None
                if chunk.modality == "audio" and chunk.audio_data is not None:
                    audio_b64 = audio_to_base64(
                        chunk.audio_data,
                        sample_rate=chunk.sample_rate or DEFAULT_SAMPLE_RATE,
                        output_format=audio_format,
                    )

                text = chunk.text
                if chunk.modality == "text" and text:
                    if chunk.finish_reason is None:
                        streamed_text += text
                    elif streamed_text and text.startswith(streamed_text):
                        text = text[len(streamed_text) :] or None

                yield CompletionStreamChunk(
                    request_id=request_id,
                    text=text,
                    modality=chunk.modality,
                    audio_b64=audio_b64,
                    finish_reason=chunk.finish_reason,
                    usage=chunk.usage,
                    stage_name=chunk.stage_name,
                )

    # ------------------------------------------------------------------
    # High-level: text-to-speech
    # ------------------------------------------------------------------

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        """Run a TTS request and return encoded audio bytes.

        Raises:
            ClientError: If the pipeline produces no audio output.
        """
        audio_chunks: list[Any] = []
        sample_rate: int | None = None
        last_chunk: GenerateChunk | None = None
        extra_params = dict(request.extra_params)
        extra_params.pop("stream", None)
        request = replace(request, stream=False, extra_params=extra_params)

        async for chunk in self.generate(request, request_id=request_id):
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate
            last_chunk = chunk

        if not audio_chunks:
            raise ClientError("No audio output generated from the pipeline.")

        if len(audio_chunks) == 1:
            audio_data = audio_chunks[0]
        else:
            arrays = [to_numpy(c) for c in audio_chunks]
            axis = -1 if arrays[0].ndim > 1 else 0
            audio_data = np.concatenate(arrays, axis=axis)

        encode_kwargs: dict[str, Any] = {
            "response_format": response_format,
            "speed": speed,
            "allow_format_fallback": allow_format_fallback,
        }
        if sample_rate is not None:
            encode_kwargs["sample_rate"] = sample_rate

        audio_bytes, mime_type = await asyncio.to_thread(
            encode_audio, audio_data, **encode_kwargs
        )

        # Derive actual format from MIME type (encode_audio may fall back
        # to WAV if the requested codec is unavailable).
        actual_format = response_format
        for ext, mt in FORMAT_MIME_TYPES.items():
            if mt == mime_type:
                actual_format = ext
                break

        return SpeechResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            format=actual_format,
            sample_rate=sample_rate,
            usage=last_chunk.usage if last_chunk else None,
        )

    # ------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        success = await self._coordinator.abort(request_id)
        return AbortResult(success=success, level_applied=level)

    async def get_status(self, request_id: str) -> RequestState | None:
        info = self._coordinator.get_request_info(request_id)
        if info is None:
            return None
        return info.state

    def health(self) -> dict[str, Any]:
        return self._coordinator.health()

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self._coordinator.admin(
            action,
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def model_info(
        self,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self._coordinator.model_info(
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self._coordinator.pause_generation(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self._coordinator.continue_generation(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self._coordinator.update_weights_from_disk(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self._coordinator.init_weights_update_group(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self._coordinator.destroy_weights_update_group(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self._coordinator.update_weights_from_distributed(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self._coordinator.weights_checker(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _set_audio_data(chunk: GenerateChunk, data: dict[str, Any]) -> None:
        audio_data = data.get("audio_data") or data.get("audio")
        if audio_data is None and data.get("audio_waveform") is not None:
            raw = data.get("audio_waveform")
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
            arr = np.frombuffer(raw, dtype=dtype)
            shape = data.get("audio_waveform_shape")
            if shape:
                arr = arr.reshape(shape)
            audio_data = arr.copy()
        if audio_data is not None:
            chunk.audio_data = audio_data
            chunk.modality = "audio"
        sample_rate = data.get("sample_rate")
        if sample_rate is not None:
            chunk.sample_rate = sample_rate

    @staticmethod
    def _build_usage_info(data: dict[str, Any]) -> UsageInfo | None:
        usage = dict(data.get("usage") or {})
        if "prompt_tokens" not in usage and data.get("prompt_tokens") is not None:
            usage["prompt_tokens"] = data.get("prompt_tokens")
        if (
            "completion_tokens" not in usage
            and data.get("completion_tokens") is not None
        ):
            usage["completion_tokens"] = data.get("completion_tokens")
        if "total_tokens" not in usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None or completion_tokens is not None:
                usage["total_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)
        if "engine_time_s" not in usage and data.get("engine_time_s") is not None:
            usage["engine_time_s"] = data.get("engine_time_s")
        return UsageInfo.from_dict(usage)

    @staticmethod
    def _build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = _extract_inputs(request)
        params = _build_params(request)
        metadata = dict(request.metadata)
        if request.model:
            metadata.setdefault("model", request.model)
        if request.output_modalities:
            metadata["output_modalities"] = request.output_modalities
        return OmniRequest(inputs=inputs, params=params, metadata=metadata)

    @staticmethod
    def _default_result_builder(request_id: str, result: Any) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id, finish_reason="stop")
        if isinstance(result, GenerateChunk):
            result.request_id = request_id
            return result
        if isinstance(result, dict):
            # Multi-terminal merged result, e.g. decode + code2wav/talker/
            # talker_stream.
            audio_result = None
            if "decode" in result:
                for audio_stage in ("code2wav", "talker", "talker_stream"):
                    if audio_stage in result:
                        audio_result = result[audio_stage] or {}
                        break
            if audio_result is not None:
                decode_result = result["decode"] or {}
                text = decode_result.get("text")
                if isinstance(text, str):
                    chunk.text = text
                finish_reason = decode_result.get("finish_reason")
                if finish_reason is not None:
                    chunk.finish_reason = finish_reason
                output_token_logprobs = decode_result.get("output_token_logprobs")
                if output_token_logprobs is not None:
                    chunk.output_token_logprobs = output_token_logprobs
                omni_rollout = decode_result.get("omni_rollout")
                if omni_rollout is not None:
                    chunk.omni_rollout = omni_rollout
                weight_version = decode_result.get("weight_version")
                if weight_version is not None:
                    chunk.weight_version = weight_version
                Client._set_audio_data(chunk, audio_result)
                chunk.usage = Client._build_usage_info(
                    decode_result
                ) or Client._build_usage_info(audio_result)
                return chunk
            text = result.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = result.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = result.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            output_token_logprobs = result.get("output_token_logprobs")
            if output_token_logprobs is not None:
                chunk.output_token_logprobs = output_token_logprobs
            omni_rollout = result.get("omni_rollout")
            if omni_rollout is not None:
                chunk.omni_rollout = omni_rollout
            weight_version = result.get("weight_version")
            if weight_version is not None:
                chunk.weight_version = weight_version
            finish_reason = result.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            chunk.stage_id = result.get("stage_id")
            chunk.stage_name = result.get("stage_name")
            modality = result.get("modality")
            if modality is not None:
                chunk.modality = modality
            Client._set_audio_data(chunk, result)
            chunk.usage = Client._build_usage_info(result)
            return chunk
        if isinstance(result, str):
            chunk.text = result
            return chunk
        chunk.text = str(result)
        return chunk

    @staticmethod
    def _default_stream_builder(request_id: str, msg: StreamMessage) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id)
        chunk.stage_name = msg.stage_name or msg.from_stage
        chunk.stage_id = msg.stage_id
        if msg.modality:
            chunk.modality = msg.modality

        data = msg.chunk
        if isinstance(data, GenerateChunk):
            data.request_id = request_id
            if data.stage_name is None:
                data.stage_name = chunk.stage_name
            if data.stage_id is None:
                data.stage_id = chunk.stage_id
            if not data.modality and chunk.modality:
                data.modality = chunk.modality
            return data
        if isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = data.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = data.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            output_token_logprobs = data.get("output_token_logprobs")
            if output_token_logprobs is not None:
                chunk.output_token_logprobs = output_token_logprobs
            omni_rollout = data.get("omni_rollout")
            if omni_rollout is not None:
                chunk.omni_rollout = omni_rollout
            weight_version = data.get("weight_version")
            if weight_version is not None:
                chunk.weight_version = weight_version
            finish_reason = data.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            chunk.usage = Client._build_usage_info(data)
            stage_name = data.get("stage_name")
            if stage_name is not None:
                chunk.stage_name = stage_name
            stage_id = data.get("stage_id")
            if stage_id is not None:
                chunk.stage_id = stage_id
            modality = data.get("modality")
            if modality is not None:
                chunk.modality = modality
            Client._set_audio_data(chunk, data)
            return chunk
        if isinstance(data, str):
            chunk.text = data
            return chunk
        if isinstance(data, int):
            chunk.token_ids = [data]
            return chunk
        chunk.text = str(data)
        return chunk


def _coerce_action_score_result(raw_result: Any, request: ActionSuffixScoreRequest) -> ActionSuffixScoreResult:
    if isinstance(raw_result, ActionSuffixScoreResult):
        return raw_result
    if not isinstance(raw_result, dict):
        raise ClientError("action scoring returned an invalid result")
    raw_scores = raw_result.get("scores")
    if not isinstance(raw_scores, list):
        raise ClientError("action scoring result is missing scores")
    scores = []
    for item in raw_scores:
        if not isinstance(item, dict):
            raise ClientError("action scoring returned an invalid candidate score")
        token_scores = [
            TokenScore(int(token["token_id"]), float(token["logprob"]))
            for token in item.get("token_scores", [])
        ]
        scores.append(
            CandidateScore(
                candidate_id=str(item["candidate_id"]),
                token_count=int(item["token_count"]),
                mean_logprob=float(item["mean_logprob"]),
                mean_nll=float(item["mean_nll"]),
                ppl=float(item["ppl"]),
                token_scores=token_scores,
            )
        )
    return ActionSuffixScoreResult(
        request_id=str(raw_result.get("request_id", request.request_id)),
        model=str(raw_result.get("model", request.model)),
        prefix_cached=bool(raw_result.get("prefix_cached", False)),
        scores=scores,
        stats=dict(raw_result.get("stats") or {}),
    )


def _extract_inputs(request: GenerateRequest) -> Any:
    choices = [
        request.prompt is not None,
        request.prompt_token_ids is not None,
        request.messages is not None,
    ]
    if sum(choices) != 1:
        raise ValueError(
            "GenerateRequest requires exactly one input: "
            "prompt, prompt_token_ids, or messages."
        )
    if request.multimodal_train_inputs is not None:
        if request.prompt_token_ids is None:
            raise ValueError(
                "multimodal_train_inputs requires prompt_token_ids "
                "(the processor-expanded input_ids)"
            )
        return {
            "input_ids": list(request.prompt_token_ids),
            "multimodal_train_inputs": request.multimodal_train_inputs,
        }
    if request.prompt is not None:
        return request.prompt
    if request.prompt_token_ids is not None:
        return list(request.prompt_token_ids)

    # Build messages list
    messages = [msg.to_dict() for msg in request.messages or []]

    # Check if we have audios, images, or videos in metadata
    audios = request.metadata.get("audios")
    images = request.metadata.get("images")
    videos = request.metadata.get("videos")

    # If we have any media, return a dict with messages and media
    # Otherwise, return just the messages list (for backward compatibility)
    if audios or images or videos:
        result = {"messages": messages}
        if images:
            result["images"] = images
        if audios:
            result["audios"] = audios
        if videos:
            result["videos"] = videos
        for key in (
            "video_fps",
            "video_max_frames",
            "video_min_pixels",
            "video_max_pixels",
            "video_total_pixels",
        ):
            value = request.metadata.get(key)
            if value is not None:
                result[key] = value
        return result
    return messages


def _build_params(request: GenerateRequest) -> dict[str, Any]:
    params = request.sampling.to_dict()
    max_new_tokens = request.sampling.max_new_tokens
    if request.max_tokens is not None:
        max_new_tokens = request.max_tokens
    if max_new_tokens is None:
        params.pop("max_new_tokens", None)
    else:
        params["max_new_tokens"] = max_new_tokens
    params["stream"] = request.stream
    if request.stage_sampling:
        params["stage_sampling"] = {
            key: value.to_dict() for key, value in request.stage_sampling.items()
        }
    if request.stage_params:
        params["stage_params"] = request.stage_params
    if request.extra_params:
        params.update(request.extra_params)
    return params
