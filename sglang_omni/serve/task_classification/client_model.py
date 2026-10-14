"""Prompted Router v0.9 adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import asyncio
import base64
import io
import time
import wave
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.profiler.decision_timing import emit_decision_timing

from .contracts import ClassificationMediaRef, TaskClassificationRequest
from .prompt_v09 import (
    DEFAULT_MCP_SCOPE,
    DEFAULT_MULTIMODAL_SCOPE,
    SINGLE_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
    normalize_history,
)


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...


class SglangClientTaskClassificationModel:
    """Classify one text or raw-audio turn with Router v0.9."""

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
        audio = request.media[0] if request.media else None
        history = normalize_history(request.router_history)
        query_mode = "audio" if audio is not None else "text"
        content = build_user_prompt(
            query_mode=query_mode,
            text=request.text,
            history=history,
            mcp_scope=DEFAULT_MCP_SCOPE,
            multimodal_scope=DEFAULT_MULTIMODAL_SCOPE,
            has_active_agent=request.has_active_agent,
            pending_confirmation=request.pending_confirmation,
            follow_up_required=request.follow_up_required,
        )
        metadata = self._metadata(request)
        if audio is not None:
            media_started = time.perf_counter()
            emit_decision_timing(
                "decision_media_preprocess_started",
                request_id=request.request_id,
                metadata=metadata,
                component="turn_router",
                media_kind="audio",
                media_bytes=len(audio.payload),
                media_encoding=audio.encoding,
                media_duration_ms=audio.end_ms - audio.start_ms,
            )
            try:
                metadata["audios"] = [_audio_data_url(audio)]
            except Exception as exc:
                emit_decision_timing(
                    "decision_media_preprocess_completed",
                    request_id=request.request_id,
                    metadata=metadata,
                    component="turn_router",
                    status="error",
                    error_type=type(exc).__name__,
                    media_preprocess_ms=round(
                        (time.perf_counter() - media_started) * 1000.0,
                        3,
                    ),
                )
                raise
            emit_decision_timing(
                "decision_media_preprocess_completed",
                request_id=request.request_id,
                metadata=metadata,
                component="turn_router",
                status="success",
                media_preprocess_ms=round(
                    (time.perf_counter() - media_started) * 1000.0,
                    3,
                ),
            )
        generate_request = GenerateRequest(
            model=self.model_id,
            messages=[
                Message(
                    role="system",
                    content=(SYSTEM_PROMPT if history else SINGLE_TURN_SYSTEM_PROMPT),
                ),
                Message(role="user", content=content),
            ],
            sampling=SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=1,
                stop=["\n"],
            ),
            stream=False,
            max_tokens=1,
            output_modalities=["text"],
            metadata=metadata,
        )
        model_started = time.perf_counter()
        emit_decision_timing(
            "decision_model_request_started",
            request_id=request.request_id,
            metadata=metadata,
            component="turn_router",
        )
        try:
            result = await self._client.completion(
                generate_request,
                request_id=request.request_id,
            )
        except asyncio.CancelledError:
            emit_decision_timing(
                "decision_model_request_completed",
                request_id=request.request_id,
                metadata=metadata,
                component="turn_router",
                status="cancelled",
                model_request_ms=round(
                    (time.perf_counter() - model_started) * 1000.0,
                    3,
                ),
            )
            raise
        except Exception as exc:
            emit_decision_timing(
                "decision_model_request_completed",
                request_id=request.request_id,
                metadata=metadata,
                component="turn_router",
                status="error",
                error_type=type(exc).__name__,
                model_request_ms=round(
                    (time.perf_counter() - model_started) * 1000.0,
                    3,
                ),
            )
            raise
        emit_decision_timing(
            "decision_model_request_completed",
            request_id=request.request_id,
            metadata=metadata,
            component="turn_router",
            status="success",
            model_request_ms=round(
                (time.perf_counter() - model_started) * 1000.0,
                3,
            ),
        )
        return str(result.text)

    async def prewarm(self) -> None:
        """Populate both Router prefixes and verify their route invariants."""

        direct = await self.classify(
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
        if direct != "0":
            raise RuntimeError(
                "turn-router prewarm expected ordinary conversation to route 0, "
                f"got {direct!r}"
            )

        delegate = await self.classify(
            TaskClassificationRequest(
                request_id="turn-router-multi-turn-prewarm",
                session_id="turn-router-prewarm",
                turn_id="turn-router-multi-turn-prewarm",
                identity_epoch=0,
                input_revision=0,
                text="继续",
                media=(),
                router_history=({
                    "role": "assistant",
                    "text": "需要我查询当前天气吗？",
                    "playback_status": "completed",
                },),
                has_active_agent=True,
                follow_up_required=True,
            )
        )
        if delegate != "1":
            raise RuntimeError(
                "turn-router prewarm expected an inherited MCP request to route 1, "
                f"got {delegate!r}"
            )

    def _metadata(
        self,
        request: TaskClassificationRequest,
    ) -> dict[str, Any]:
        return {
            "task": "turn_router",
            "task_role": "turn_router",
            "trace_id": request.request_id,
            "operation_id": request.request_id,
            "classification_id": request.request_id,
            "contract_version": request.contract_version,
            "logical_request_id": request.turn_id,
            "turn_id": request.turn_id,
            "session_instance_id": request.session_id,
            "session_id": request.session_id,
            "identity_epoch": request.identity_epoch,
            "input_revision": request.input_revision,
            "model": self.model_id,
            "model_version": self.model_version,
            "prefix_cache_namespace": "turn-router:v09",
            "cache_static_system_only": True,
            "stage": "router",
            "admission_priority": 0,
        }


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
    if normalized in {"wav", "wave", "pcm16", "pcm_s16le"}:
        return "audio/wav"
    if normalized in {"mp3", "mpeg"}:
        return "audio/mpeg"
    if normalized in {"ogg", "opus"}:
        return "audio/ogg"
    if normalized in {"m4a", "mp4", "aac"}:
        return "audio/mp4"
    raise ValueError(f"unsupported turn-router audio encoding: {encoding}")
