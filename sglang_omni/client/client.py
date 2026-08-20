# SPDX-License-Identifier: Apache-2.0
"""Client wrapper for coordinator-based pipelines."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
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
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage


logger = logging.getLogger(__name__)

_ACTION_SCORE_TIMEOUT_S = float(os.environ.get("SGLANG_OMNI_ACTION_SCORE_TIMEOUT_S", "120"))
_ACTION_DEBUG_LOG_FILE = os.environ.get("SGLANG_OMNI_ACTION_DEBUG_LOG_FILE", "/tmp/sglang-omni-action-debug.jsonl")


def _summarize_debug_media(values: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for index, value in enumerate(values if isinstance(values, list) else []):
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
        "logical_request_id": request.logical_request_id,
        "prefix": request.prefix,
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
    """Append one replayable action-scoring diagnostic record to JSONL."""
    try:
        path = Path(_ACTION_DEBUG_LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("failed to write action debug log file=%s", _ACTION_DEBUG_LOG_FILE)


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
        self._action_scoring_semaphore = asyncio.Semaphore(1)

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult:
        """Score all suffixes as one logical multimodal pipeline request."""
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
        }
        request_started = time.perf_counter()
        _write_action_debug_record({
            "event": "action_scoring_started",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "full_logical_input": _action_request_debug_payload(request, omni_request),
        })

        async def _submit() -> Any:
            slot_started = time.perf_counter()
            async with self._action_scoring_semaphore:
                phase["slot_wait_ms"] = round((time.perf_counter() - slot_started) * 1000.0, 3)
                phase["name"] = "coordinator_pipeline"
                pipeline_started = time.perf_counter()
                try:
                    return await self._coordinator.submit(request.request_id, omni_request)
                finally:
                    phase["pipeline_ms"] = round((time.perf_counter() - pipeline_started) * 1000.0, 3)

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
        result = _coerce_action_score_result(raw_result, request)
        try:
            validate_score_result(request, result)
        except RuntimeError as exc:
            raise ClientError(str(exc)) from exc
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
            system_prompt = (
                "你是数字人动作类别识别器。只能输出一个 category_id。"
                f"固定类别集合：{definitions}"
            )
            prefix = "请根据当前输入选择一个动作类别。下一步 category_id 是："
        elif stage == "child":
            system_prompt = (
                "你是数字人动作识别器。只能输出一个 action_id。"
                f"固定子动作集合：{definitions}"
            )
            prefix = "请根据当前输入选择一个子动作。下一步 action_id 是："
        elif stage == "single":
            system_prompt = (
                "你是数字人动作识别器。只能输出一个 action_id。"
                f"固定具体动作集合：{definitions}"
            )
            prefix = "请根据当前输入选择一个具体动作。下一步 action_id 是："
        else:
            raise ValueError(f"unsupported action warmup stage: {stage!r}")
        return ActionSuffixScoreRequest(
            request_id=request_id,
            model=model,
            prefix=prefix,
            language="zh",
            candidates=candidates,
            audios=[],
            images=[],
            sample_rate=16000,
            system_prompt=system_prompt,
            micro_batch_size=64,
            stage=stage,
            logical_request_id="warmup-process",
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
    ) -> dict[str, Any]:
        # Warm the action-score request path before accepting user turns.
        # The requests have no session, media, or history; results are discarded.
        if category_count <= 0 or child_count <= 0:
            raise ValueError("warmup candidate counts must be positive")
        if selection_mode not in {"hierarchical", "flat_children"}:
            raise ValueError(f"unsupported action selection mode: {selection_mode!r}")
        started = time.perf_counter()
        logger.info(
            "[ACTION_WARMUP] started model=%s mode=%s category_candidates=%d child_candidates=%d",
            model,
            selection_mode,
            category_count,
            child_count,
        )
        _write_action_debug_record({
            "event": "action_score_warmup_started",
            "timestamp_unix_ms": round(time.time() * 1000.0),
            "model": model,
            "selection_mode": selection_mode,
            "category_candidates": category_count,
            "child_candidates": child_count,
        })

        async def run_warmup() -> tuple[float | None, float | None, float | None]:
            if selection_mode == "flat_children":
                single_started = time.perf_counter()
                await self.score_action_suffixes(
                    self._build_action_warmup_request(
                        request_id="warmup-process-single",
                        model=model,
                        stage="single",
                        candidate_prefix="D",
                        candidate_count=child_count,
                    )
                )
                return None, None, (time.perf_counter() - single_started) * 1000.0

            category_started = time.perf_counter()
            await self.score_action_suffixes(
                self._build_action_warmup_request(
                    request_id="warmup-process-category",
                    model=model,
                    stage="category",
                    candidate_prefix="C",
                    candidate_count=category_count,
                )
            )
            category_ms = (time.perf_counter() - category_started) * 1000.0

            child_started = time.perf_counter()
            await self.score_action_suffixes(
                self._build_action_warmup_request(
                    request_id="warmup-process-child",
                    model=model,
                    stage="child",
                    candidate_prefix="D",
                    candidate_count=child_count,
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
                "selection_mode": selection_mode,
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
                "selection_mode": selection_mode,
                "elapsed_ms": round(elapsed_ms, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "ready": True,
            "selection_mode": selection_mode,
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
    ) -> bool:
        """Prefill one immutable action catalog prefix."""
        if not candidates:
            return False
        probe = candidates[0]
        request = ActionSuffixScoreRequest(
            request_id=request_id or f"catalog-prefill-{stage}-{uuid.uuid4().hex}",
            model=model,
            prefix="请根据当前输入选择动作。下一步 action_id 是：",
            language="zh",
            candidates=[probe],
            audios=[],
            images=[],
            sample_rate=16000,
            system_prompt=system_prompt,
            stage=stage,
            logical_request_id=f"catalog-prefill-{prefix_cache_namespace}",
            prefix_cache_namespace=prefix_cache_namespace,
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
    def _action_instruction_content(
        content: Any,
        instruction: str,
        *,
        audios: list[str],
        images: list[str],
    ) -> Any:
        """Append the action instruction to a user message.

        Media placeholders must be kept in the message list when the
        corresponding top-level media payload is present. Otherwise the
        multimodal preprocessor cannot distinguish historical media from the
        current turn.
        """
        if isinstance(content, list):
            parts = [dict(part) if isinstance(part, dict) else part for part in content]
            parts.extend({"type": "audio"} for _ in audios)
            parts.extend({"type": "image"} for _ in images)
            parts.append({"type": "text", "text": instruction})
            return parts

        if not audios and not images and isinstance(content, str):
            return f"{content}\n{instruction}"

        parts = []
        if content is not None:
            parts.append({"type": "text", "text": str(content)})
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        parts.append({"type": "text", "text": instruction})
        return parts

    @staticmethod
    def _build_action_context_messages(
        history: list[dict[str, Any]],
        instruction: str,
        *,
        avatar_state: dict[str, Any] | None,
        system_prompt: str | None,
        audios: list[str],
        images: list[str],
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

        current_instruction = instruction
        if avatar_state:
            state_text = json.dumps(
                avatar_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            current_instruction = f"当前数字人状态：{state_text}\n{instruction}"

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
        if messages and messages[-1].get("role") == "user":
            latest_user = messages.pop()
            latest_user["content"] = Client._action_instruction_content(
                latest_user.get("content"),
                current_instruction,
                audios=missing_audios,
                images=missing_images,
            )
            messages.append(latest_user)
        else:
            current_content: Any
            if missing_audios or missing_images:
                current_content = [
                    *({"type": "audio"} for _ in missing_audios),
                    *({"type": "image"} for _ in missing_images),
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
            avatar_state=request.avatar_state,
            system_prompt=request.system_prompt,
            audios=[*request.history_audios, *request.audios],
            images=[*request.history_images, *request.images],
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
                    "language": request.language,
                    "candidates": candidates,
                    "micro_batch_size": request.micro_batch_size,
                    "sample_rate": request.sample_rate,
                    "client_started_at": time.perf_counter(),
                    "static_system_prompt": request.system_prompt,
                    "prefix_cache_namespace": request.prefix_cache_namespace,
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
