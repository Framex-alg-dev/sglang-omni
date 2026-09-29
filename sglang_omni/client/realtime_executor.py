# SPDX-License-Identifier: Apache-2.0
"""Remote and routed model clients for the multimodal realtime gateway.

The gateway remains the sole owner of session state.  Only fully materialized
``GenerateRequest`` objects are sent to a secondary model executor, so moving
reply generation to another GPU cannot split history, turn, TTS, or knowledge
ownership across processes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import struct
from contextlib import aclosing
from enum import Enum
from typing import Any, AsyncIterator

import httpx
import msgpack
import numpy as np

from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    ClientError,
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateRequest,
    Message,
    SamplingParams,
    UsageInfo,
)
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
from sglang_omni.utils.structured_logs import emit_structured_log

logger = logging.getLogger(__name__)

INTERNAL_MODEL_API_ENV = "SGLANG_OMNI_INTERNAL_MODEL_API"
INTERNAL_MODEL_TOKEN_ENV = "SGLANG_OMNI_INTERNAL_MODEL_TOKEN"
REPLY_EXECUTOR_URL_ENV = "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_URL"
REPLY_EXECUTOR_TOKEN_ENV = "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_TOKEN"
REPLY_EXECUTOR_TASKS_ENV = "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_TASKS"
REPLY_EXECUTOR_SCORE_STAGES_ENV = (
    "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_SCORE_STAGES"
)
REPLY_EXECUTOR_CONNECT_TIMEOUT_ENV = (
    "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_CONNECT_TIMEOUT_S"
)
REPLY_EXECUTOR_CONTROL_TIMEOUT_ENV = (
    "SGLANG_OMNI_REALTIME_REPLY_EXECUTOR_CONTROL_TIMEOUT_S"
)

DEFAULT_REPLY_EXECUTOR_TASKS = frozenset(
    {
        "session_reply",
        "session_pure_action_reply",
        "session_action_rejection",
        "session_turn_intent",
        "session_turn_intent_prewarm",
        "session_memory_extract",
        "session_visual_arithmetic_probe",
        "session_visual_arithmetic_prewarm",
    }
)
DEFAULT_REPLY_EXECUTOR_SCORE_STAGES = frozenset(
    {"reply_history_route", "reply_speech_mode", "pure_action_reply_validation"}
)

_FRAME_HEADER = struct.Struct("!I")
_WIRE_MEDIA_TYPE = "application/vnd.sglang-omni.msgpack"


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def internal_model_api_enabled() -> bool:
    return _env_flag(INTERNAL_MODEL_API_ENV)


def internal_model_api_token() -> str | None:
    value = os.environ.get(INTERNAL_MODEL_TOKEN_ENV, "").strip()
    return value or None


def _msgpack_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return list(value)
    raise TypeError(f"unsupported internal model wire type: {type(value).__name__}")


def _msgpack_object_hook(value: dict[str, Any]) -> Any:
    if value.get("__ndarray__") is True:
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"]))
    return value


def pack_wire(value: Any) -> bytes:
    return msgpack.packb(value, use_bin_type=True, default=_msgpack_default)


def unpack_wire(value: bytes) -> Any:
    return msgpack.unpackb(
        value,
        raw=False,
        strict_map_key=False,
        object_hook=_msgpack_object_hook,
    )


def frame_wire(value: Any) -> bytes:
    payload = pack_wire(value)
    return _FRAME_HEADER.pack(len(payload)) + payload


def request_to_wire(request: GenerateRequest) -> dict[str, Any]:
    return request.to_dict()


def request_from_wire(payload: dict[str, Any]) -> GenerateRequest:
    sampling = SamplingParams(**(payload.get("sampling") or {}))
    stage_sampling_payload = payload.get("stage_sampling")
    stage_sampling = (
        {
            str(name): SamplingParams(**(params or {}))
            for name, params in stage_sampling_payload.items()
        }
        if stage_sampling_payload
        else None
    )
    messages_payload = payload.get("messages")
    messages = (
        [
            Message(role=item["role"], content=item.get("content"))
            for item in messages_payload
        ]
        if messages_payload
        else None
    )
    return GenerateRequest(
        model=payload.get("model"),
        prompt=payload.get("prompt"),
        prompt_token_ids=payload.get("prompt_token_ids"),
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=payload.get("stage_params"),
        extra_params=dict(payload.get("extra_params") or {}),
        stream=bool(payload.get("stream", True)),
        max_tokens=payload.get("max_tokens"),
        output_modalities=payload.get("output_modalities"),
        multimodal_train_inputs=payload.get("multimodal_train_inputs"),
        metadata=dict(payload.get("metadata") or {}),
    )


def action_score_request_from_wire(payload: dict[str, Any]) -> ActionSuffixScoreRequest:
    return ActionSuffixScoreRequest(
        **{
            **payload,
            "candidates": [
                ActionScoreCandidate(**candidate)
                for candidate in payload.get("candidates", [])
            ],
        }
    )


def action_score_result_from_wire(payload: dict[str, Any]) -> ActionSuffixScoreResult:
    return ActionSuffixScoreResult(
        request_id=str(payload["request_id"]),
        model=str(payload["model"]),
        prefix_cached=bool(payload["prefix_cached"]),
        scores=[
            CandidateScore(
                **{
                    **score,
                    "token_scores": [
                        TokenScore(**token) for token in score.get("token_scores", [])
                    ],
                }
            )
            for score in payload.get("scores", [])
        ],
        stats=dict(payload.get("stats") or {}),
    )


def _usage_from_wire(payload: dict[str, Any] | None) -> UsageInfo | None:
    return UsageInfo.from_dict(payload)


def stream_chunk_to_wire(chunk: CompletionStreamChunk) -> dict[str, Any]:
    return {
        "request_id": chunk.request_id,
        "text": chunk.text,
        "modality": chunk.modality,
        "audio_b64": chunk.audio_b64,
        "finish_reason": chunk.finish_reason,
        "usage": chunk.usage.to_dict() if chunk.usage else None,
        "stage_name": chunk.stage_name,
        "output_token_logprobs": chunk.output_token_logprobs,
        "output_top_logprobs": chunk.output_top_logprobs,
    }


def stream_chunk_from_wire(payload: dict[str, Any]) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id=str(payload["request_id"]),
        text=payload.get("text") or "",
        modality=payload.get("modality") or "text",
        audio_b64=payload.get("audio_b64"),
        finish_reason=payload.get("finish_reason"),
        usage=_usage_from_wire(payload.get("usage")),
        stage_name=payload.get("stage_name"),
        output_token_logprobs=payload.get("output_token_logprobs"),
        output_top_logprobs=payload.get("output_top_logprobs"),
    )


def completion_to_wire(result: CompletionResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "text": result.text,
        "audio": dataclasses.asdict(result.audio) if result.audio else None,
        "finish_reason": result.finish_reason,
        "usage": result.usage.to_dict() if result.usage else None,
        "output_token_logprobs": result.output_token_logprobs,
        "output_top_logprobs": result.output_top_logprobs,
        "omni_rollout": result.omni_rollout,
        "weight_version": result.weight_version,
    }


def completion_from_wire(payload: dict[str, Any]) -> CompletionResult:
    audio_payload = payload.get("audio")
    return CompletionResult(
        request_id=str(payload["request_id"]),
        text=payload.get("text") or "",
        audio=CompletionAudio(**audio_payload) if audio_payload else None,
        finish_reason=payload.get("finish_reason") or "stop",
        usage=_usage_from_wire(payload.get("usage")),
        output_token_logprobs=payload.get("output_token_logprobs"),
        output_top_logprobs=payload.get("output_top_logprobs"),
        omni_rollout=payload.get("omni_rollout"),
        weight_version=payload.get("weight_version"),
    )


class RemoteRealtimeModelClient:
    """Client-compatible adapter for a loopback secondary model executor."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        connect_timeout_s: float = 1.0,
        control_timeout_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("reply executor base URL must not be empty")
        if not token:
            raise ValueError("reply executor token must not be empty")
        self.base_url = base_url.rstrip("/")
        self.control_timeout_s = max(0.1, control_timeout_s)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": _WIRE_MEDIA_TYPE,
        }
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=connect_timeout_s),
            trust_env=False,
        )

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        response = await self._client.post(
            f"{self.base_url}/internal/realtime-model/completion",
            headers=self._headers,
            content=pack_wire(
                {
                    "request_id": request_id,
                    "audio_format": audio_format,
                    "request": request_to_wire(request),
                }
            ),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ClientError(
                f"secondary executor completion failed: HTTP {response.status_code}"
            ) from exc
        payload = unpack_wire(response.content)
        if payload.get("kind") == "error":
            raise ClientError(str(payload.get("message") or "secondary executor failed"))
        return completion_from_wire(payload["result"])

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult:
        response = await self._client.post(
            f"{self.base_url}/internal/realtime-model/action-score",
            content=pack_wire(dataclasses.asdict(request)),
            headers=self._headers,
            timeout=self.control_timeout_s,
        )
        response.raise_for_status()
        payload = unpack_wire(response.content)
        if payload.get("kind") == "error":
            raise ClientError(
                f"remote action scoring failed: {payload.get('error_type')}: "
                f"{payload.get('message')}"
            )
        return action_score_result_from_wire(payload["result"])

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        body = pack_wire(
            {
                "request_id": request_id,
                "audio_format": audio_format,
                "request": request_to_wire(request),
            }
        )
        async with self._client.stream(
            "POST",
            f"{self.base_url}/internal/realtime-model/completion-stream",
            headers=self._headers,
            content=body,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ClientError(
                    f"secondary executor stream failed: HTTP {response.status_code}"
                ) from exc
            buffer = bytearray()
            expected: int | None = None
            async for data in response.aiter_bytes():
                buffer.extend(data)
                while True:
                    if expected is None:
                        if len(buffer) < _FRAME_HEADER.size:
                            break
                        expected = _FRAME_HEADER.unpack(
                            bytes(buffer[: _FRAME_HEADER.size])
                        )[0]
                        del buffer[: _FRAME_HEADER.size]
                    if len(buffer) < expected:
                        break
                    payload = unpack_wire(bytes(buffer[:expected]))
                    del buffer[:expected]
                    expected = None
                    if payload.get("kind") == "error":
                        raise ClientError(
                            str(payload.get("message") or "secondary executor failed")
                        )
                    yield stream_chunk_from_wire(payload["chunk"])
            if expected is not None or buffer:
                raise ClientError("secondary executor returned a truncated stream")

    async def prefill_completion_prefix(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
    ) -> bool:
        await self.completion(request, request_id=request_id)
        return True

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        response = await asyncio.wait_for(
            self._client.post(
                f"{self.base_url}/internal/realtime-model/abort",
                headers=self._headers,
                content=pack_wire(
                    {"request_id": request_id, "level": level.value}
                ),
            ),
            timeout=self.control_timeout_s,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ClientError(
                f"secondary executor abort failed: HTTP {response.status_code}"
            ) from exc
        payload = unpack_wire(response.content)
        return AbortResult(
            success=bool(payload.get("success")),
            level_applied=AbortLevel(payload.get("level_applied", level.value)),
        )

    async def health(self) -> dict[str, Any]:
        response = await self._client.get(
            f"{self.base_url}/internal/realtime-model/health",
            headers=self._headers,
        )
        response.raise_for_status()
        return dict(unpack_wire(response.content))

    async def release_session_cache(self, session_instance_id: str) -> Any:
        response = await asyncio.wait_for(
            self._client.post(
                f"{self.base_url}/internal/realtime-model/release-session-cache",
                headers=self._headers,
                content=pack_wire({"session_instance_id": session_instance_id}),
            ),
            timeout=self.control_timeout_s,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ClientError(
                "secondary executor session-cache release failed: "
                f"HTTP {response.status_code}"
            ) from exc
        return unpack_wire(response.content)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RoutedRealtimeModelClient:
    """Route only reply-generation tasks to a secondary executor.

    All unknown tasks stay on the control executor.  This is deliberately an
    allowlist so newly added control or safety model calls cannot silently move
    to GPU1.  Fallback is automatic before any streamed output is observed;
    retrying after output would duplicate user-visible text or TTS input.
    """

    def __init__(
        self,
        control_client: Any,
        reply_client: RemoteRealtimeModelClient,
        *,
        reply_tasks: set[str] | frozenset[str] = DEFAULT_REPLY_EXECUTOR_TASKS,
        reply_score_stages: set[str] | frozenset[str] = (
            DEFAULT_REPLY_EXECUTOR_SCORE_STAGES
        ),
    ) -> None:
        self.control_client = control_client
        self.reply_client = reply_client
        self.reply_tasks = frozenset(reply_tasks)
        self.reply_score_stages = frozenset(reply_score_stages)
        self._request_backends: dict[str, Any] = {}
        self._intentionally_aborted: set[str] = set()
        self._lock = asyncio.Lock()

    def _select(self, request: GenerateRequest) -> tuple[Any, str]:
        task = str((request.metadata or {}).get("task") or "")
        if task in self.reply_tasks:
            return self.reply_client, "reply"
        return self.control_client, "control"

    async def _track(self, request_id: str, backend: Any) -> None:
        async with self._lock:
            self._request_backends[request_id] = backend

    async def _untrack(self, request_id: str, backend: Any) -> None:
        async with self._lock:
            if self._request_backends.get(request_id) is backend:
                self._request_backends.pop(request_id, None)
            self._intentionally_aborted.discard(request_id)

    async def _was_intentionally_aborted(self, request_id: str) -> bool:
        async with self._lock:
            return request_id in self._intentionally_aborted

    async def mark_intentional_abort(self, request_id: str) -> None:
        """Fence fallback before local stream cancellation starts.

        Cancelling a task that is currently advancing an async HTTP stream can
        make its close path raise a regular ``RuntimeError`` instead of
        ``CancelledError``.  Callers that are deliberately discarding a reply
        mark the request first so that teardown errors cannot be mistaken for
        a secondary-executor failure and retried on the control GPU.
        """

        async with self._lock:
            if request_id in self._request_backends:
                self._intentionally_aborted.add(request_id)

    @staticmethod
    def _log_route(
        request: GenerateRequest,
        request_id: str,
        executor: str,
        *,
        fallback_reason: str | None = None,
    ) -> None:
        emit_structured_log(
            "performance",
            "realtime_model_request_routed",
            request_id=request_id,
            session_id=request.metadata.get("session_id"),
            session_instance_id=request.metadata.get("session_instance_id"),
            turn_id=request.metadata.get("turn_id"),
            task=request.metadata.get("task"),
            executor=executor,
            fallback_reason=fallback_reason,
        )

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        backend, executor = self._select(request)
        await self._track(request_id, backend)
        self._log_route(request, request_id, executor)
        try:
            try:
                return await backend.completion(
                    request, request_id=request_id, audio_format=audio_format
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await self._was_intentionally_aborted(request_id):
                    raise asyncio.CancelledError from exc
                if backend is self.control_client:
                    raise
                await self._abort_secondary_best_effort(request_id)
                await self._track(request_id, self.control_client)
                self._log_route(
                    request,
                    request_id,
                    "control_fallback",
                    fallback_reason=type(exc).__name__,
                )
                return await self.control_client.completion(
                    request, request_id=request_id, audio_format=audio_format
                )
        finally:
            async with self._lock:
                self._request_backends.pop(request_id, None)
                self._intentionally_aborted.discard(request_id)

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        backend, executor = self._select(request)
        await self._track(request_id, backend)
        self._log_route(request, request_id, executor)
        emitted = False
        try:
            try:
                stream = backend.completion_stream(
                    request, request_id=request_id, audio_format=audio_format
                )
                async with aclosing(stream):
                    async for chunk in stream:
                        emitted = True
                        yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await self._was_intentionally_aborted(request_id):
                    raise asyncio.CancelledError from exc
                if backend is self.control_client or emitted:
                    raise
                await self._abort_secondary_best_effort(request_id)
                backend = self.control_client
                await self._track(request_id, backend)
                self._log_route(
                    request,
                    request_id,
                    "control_fallback",
                    fallback_reason=type(exc).__name__,
                )
                stream = backend.completion_stream(
                    request, request_id=request_id, audio_format=audio_format
                )
                async with aclosing(stream):
                    async for chunk in stream:
                        yield chunk
        finally:
            await self._untrack(request_id, backend)

    async def prefill_completion_prefix(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
    ) -> bool:
        backend, executor = self._select(request)
        await self._track(request_id, backend)
        self._log_route(request, request_id, f"{executor}_prefill")
        try:
            try:
                return bool(
                    await backend.prefill_completion_prefix(
                        request, request_id=request_id
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await self._was_intentionally_aborted(request_id):
                    raise asyncio.CancelledError from exc
                if backend is self.control_client:
                    raise
                await self._abort_secondary_best_effort(request_id)
                backend = self.control_client
                await self._track(request_id, backend)
                self._log_route(
                    request,
                    request_id,
                    "control_prefill_fallback",
                    fallback_reason=type(exc).__name__,
                )
                return bool(
                    await backend.prefill_completion_prefix(
                        request, request_id=request_id
                    )
                )
        finally:
            await self._untrack(request_id, backend)

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult:
        backend = (
            self.reply_client
            if request.stage in self.reply_score_stages
            else self.control_client
        )
        executor = "reply" if backend is self.reply_client else "control"
        await self._track(request.request_id, backend)
        emit_structured_log(
            "performance",
            "realtime_model_request_routed",
            request_id=request.request_id,
            session_id=request.session_id,
            session_instance_id=request.session_instance_id,
            task=f"action_score:{request.stage}",
            executor=executor,
        )
        try:
            try:
                return await backend.score_action_suffixes(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await self._was_intentionally_aborted(request.request_id):
                    raise asyncio.CancelledError from exc
                if backend is self.control_client:
                    raise
                await self._abort_secondary_best_effort(request.request_id)
                await self._track(request.request_id, self.control_client)
                emit_structured_log(
                    "performance",
                    "realtime_model_request_routed",
                    request_id=request.request_id,
                    session_id=request.session_id,
                    session_instance_id=request.session_instance_id,
                    task=f"action_score:{request.stage}",
                    executor="control_fallback",
                    fallback_reason=type(exc).__name__,
                )
                return await self.control_client.score_action_suffixes(request)
        finally:
            async with self._lock:
                self._request_backends.pop(request.request_id, None)
                self._intentionally_aborted.discard(request.request_id)

    async def _abort_secondary_best_effort(self, request_id: str) -> None:
        try:
            await asyncio.wait_for(self.reply_client.abort(request_id), timeout=1.0)
        except Exception:
            logger.warning(
                "failed to abort secondary request before control fallback request_id=%s",
                request_id,
                exc_info=True,
            )

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        await self.mark_intentional_abort(request_id)
        async with self._lock:
            backend = self._request_backends.get(request_id)
        if backend is not None:
            return await backend.abort(request_id, level=level)

        # A cancel can race request registration or cleanup.  Abort both
        # executors so no generation survives a turn cancellation.
        results = await asyncio.gather(
            self.control_client.abort(request_id, level=level),
            self.reply_client.abort(request_id, level=level),
            return_exceptions=True,
        )
        successes = [r for r in results if isinstance(r, AbortResult) and r.success]
        errors = [r for r in results if isinstance(r, Exception)]
        if not successes and len(errors) == len(results):
            raise ClientError(
                "abort failed on both realtime model executors: "
                + "; ".join(type(error).__name__ for error in errors)
            )
        return AbortResult(success=bool(successes), level_applied=level)

    async def release_session_cache(self, session_instance_id: str) -> Any:
        control_result, reply_result = await asyncio.gather(
            self.control_client.release_session_cache(session_instance_id),
            self.reply_client.release_session_cache(session_instance_id),
            return_exceptions=True,
        )
        if isinstance(reply_result, Exception):
            logger.warning(
                "secondary session-cache release failed session_instance_id=%s",
                session_instance_id,
                exc_info=(
                    type(reply_result),
                    reply_result,
                    reply_result.__traceback__,
                ),
            )
        if isinstance(control_result, Exception):
            raise control_result
        return control_result

    def __getattr__(self, name: str) -> Any:
        # Unclassified model operations, admin, and status remain authoritative
        # on GPU0 without duplicating a broad wrapper surface.
        return getattr(self.control_client, name)

    async def aclose(self) -> None:
        await self.reply_client.aclose()


def build_realtime_model_client(control_client: Any) -> Any:
    """Build the capability-safe routed client from deployment environment."""

    base_url = os.environ.get(REPLY_EXECUTOR_URL_ENV, "").strip()
    if not base_url:
        return control_client
    token = (
        os.environ.get(REPLY_EXECUTOR_TOKEN_ENV, "").strip()
        or os.environ.get(INTERNAL_MODEL_TOKEN_ENV, "").strip()
    )
    if not token:
        raise ValueError(
            f"{REPLY_EXECUTOR_TOKEN_ENV} or {INTERNAL_MODEL_TOKEN_ENV} is required "
            f"when {REPLY_EXECUTOR_URL_ENV} is set"
        )
    raw_tasks = os.environ.get(REPLY_EXECUTOR_TASKS_ENV, "")
    tasks = (
        {part.strip() for part in raw_tasks.split(",") if part.strip()}
        if raw_tasks.strip()
        else set(DEFAULT_REPLY_EXECUTOR_TASKS)
    )
    if not tasks:
        raise ValueError("reply executor task allowlist must not be empty")
    raw_score_stages = os.environ.get(REPLY_EXECUTOR_SCORE_STAGES_ENV, "")
    score_stages = (
        {part.strip() for part in raw_score_stages.split(",") if part.strip()}
        if raw_score_stages.strip()
        else set(DEFAULT_REPLY_EXECUTOR_SCORE_STAGES)
    )
    if not score_stages:
        raise ValueError("reply executor score-stage allowlist must not be empty")
    connect_timeout_s = float(
        os.environ.get(REPLY_EXECUTOR_CONNECT_TIMEOUT_ENV, "1.0")
    )
    control_timeout_s = float(
        os.environ.get(REPLY_EXECUTOR_CONTROL_TIMEOUT_ENV, "2.0")
    )
    remote = RemoteRealtimeModelClient(
        base_url,
        token=token,
        connect_timeout_s=connect_timeout_s,
        control_timeout_s=control_timeout_s,
    )
    logger.info(
        "Configured realtime reply executor url=%s tasks=%s score_stages=%s",
        base_url,
        sorted(tasks),
        sorted(score_stages),
    )
    return RoutedRealtimeModelClient(
        control_client,
        remote,
        reply_tasks=tasks,
        reply_score_stages=score_stages,
    )
