"""Prompted turn-router adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.profiler.decision_timing import emit_decision_timing
from sglang_omni.serve.model_io import get_model_io_recorder
from sglang_omni.utils.framex_service_logs import current_service_log_context
from sglang_omni.utils.model_trace_logs import emit_model_trace

from . import prompt_v10
from .contracts import (
    ClassificationMediaRef,
    TaskClassificationRequest,
)

logger = logging.getLogger(__name__)


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...


class SglangClientTaskClassificationModel:
    """Classify one text or raw-audio turn with the selected Router profile."""

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
        input_started = time.perf_counter()
        supplied = request.prompt_snapshot
        assert supplied is not None
        if (
            supplied.binding.model_id != self.model_id
            or supplied.binding.model_version != self.model_version
        ):
            logger.warning(
                "Router prompt serving model drift",
                extra={
                    "request_id": request.request_id,
                    "root_session_id": request.session_id,
                    "turn_id": request.turn_id,
                    "release_id": supplied.release_id,
                    "bundle_sha256": supplied.bundle_sha256,
                    "bound_model_id": supplied.binding.model_id,
                    "bound_model_version": supplied.binding.model_version,
                    "serving_model_id": self.model_id,
                    "serving_model_version": self.model_version,
                },
            )
        snapshot = supplied.rules
        audio = request.media[0] if request.media else None
        history = prompt_v10.normalize_history(request.router_history)
        query_mode = "audio" if audio is not None else "text"
        prompt_kwargs = dict(
            query_mode=query_mode,
            text=request.text,
            history=history,
            mcp_scope=snapshot["mcp_scope"],
            multimodal_scope=snapshot["multimodal_scope"],
            has_active_agent=request.has_active_agent,
            pending_confirmation=request.pending_confirmation,
            follow_up_required=request.follow_up_required,
        )
        content = prompt_v10.build_user_prompt(**prompt_kwargs)
        metadata = self._metadata(request)
        log_context = current_service_log_context()
        trace_request_id = log_context.request_id or request.request_id
        root_session_id = log_context.root_session_id or request.session_id
        agent_session_id = log_context.agent_session_id or request.session_id
        trace_turn_id = log_context.turn_id or request.turn_id
        if audio is not None:
            media_started = time.perf_counter()
            emit_decision_timing(
                "decision_media_preprocess_started",
                request_id=trace_request_id,
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
                    request_id=trace_request_id,
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
                request_id=trace_request_id,
                metadata=metadata,
                component="turn_router",
                status="success",
                media_preprocess_ms=round(
                    (time.perf_counter() - media_started) * 1000.0,
                    3,
                ),
            )
        system_prompt = snapshot["system_prompt"]
        generate_request = GenerateRequest(
            model=self.model_id,
            messages=[
                Message(role="system", content=system_prompt),
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
        input_prepare_ms = (time.perf_counter() - input_started) * 1000.0
        emit_model_trace(
            "reply",
            "model.input.prepared",
            component="turn_router",
            request_id=trace_request_id,
            sequence=2,
            root_session_id=root_session_id,
            agent_session_id=agent_session_id,
            session_id=request.session_id,
            turn_id=trace_turn_id,
            model_id=self.model_id,
            model_version=self.model_version_for(request),
            model_input={
                "system_prompt": system_prompt,
                "user_prompt": content,
                "sampling": generate_request.sampling,
                "media": request.media,
            },
            input_prepare_ms=round(input_prepare_ms, 3),
        )
        model_started = time.perf_counter()
        archive_started = time.time()
        archive_context = {
            "request_id": trace_request_id,
            "provider_request_id": request.request_id,
            "root_session_id": root_session_id,
            "agent_session_id": agent_session_id,
            "turn_id": trace_turn_id,
            "model_id": self.model_id,
            "model_version": self.model_version_for(request),
            "release_id": supplied.release_id if supplied is not None else None,
            "prompt_snapshot_sha256": (
                supplied.bundle_sha256 if supplied is not None else None
            ),
        }
        emit_decision_timing(
            "decision_model_request_started",
            request_id=trace_request_id,
            metadata=metadata,
            component="turn_router",
        )
        emit_model_trace(
            "reply",
            "model.request.started",
            component="turn_router",
            request_id=trace_request_id,
            sequence=3,
            root_session_id=root_session_id,
            agent_session_id=agent_session_id,
            session_id=request.session_id,
            turn_id=trace_turn_id,
            model_id=self.model_id,
            model_version=self.model_version_for(request),
            input_prepare_ms=round(input_prepare_ms, 3),
        )
        try:
            result = await self._client.completion(
                generate_request,
                request_id=request.request_id,
            )
        except asyncio.CancelledError:
            get_model_io_recorder("router").record(
                stage="router",
                input_value=generate_request,
                output_value={"error_type": "CancelledError"},
                context=archive_context,
                status="cancelled",
                started_at=archive_started,
            )
            emit_decision_timing(
                "decision_model_request_completed",
                request_id=trace_request_id,
                metadata=metadata,
                component="turn_router",
                status="cancelled",
                model_request_ms=round(
                    (time.perf_counter() - model_started) * 1000.0,
                    3,
                ),
            )
            emit_model_trace(
                "reply",
                "model.response.received",
                component="turn_router",
                request_id=trace_request_id,
                sequence=4,
                root_session_id=root_session_id,
                agent_session_id=agent_session_id,
                turn_id=trace_turn_id,
                status="cancelled",
                model_inference_ms=round(
                    (time.perf_counter() - model_started) * 1000.0, 3
                ),
                model_output={"error_type": "CancelledError"},
            )
            raise
        except Exception as exc:
            get_model_io_recorder("router").record(
                stage="router",
                input_value=generate_request,
                output_value={"error_type": type(exc).__name__, "detail": str(exc)},
                context=archive_context,
                status="error",
                started_at=archive_started,
            )
            emit_decision_timing(
                "decision_model_request_completed",
                request_id=trace_request_id,
                metadata=metadata,
                component="turn_router",
                status="error",
                error_type=type(exc).__name__,
                model_request_ms=round(
                    (time.perf_counter() - model_started) * 1000.0,
                    3,
                ),
            )
            emit_model_trace(
                "reply",
                "model.response.received",
                component="turn_router",
                request_id=trace_request_id,
                sequence=4,
                root_session_id=root_session_id,
                agent_session_id=agent_session_id,
                turn_id=trace_turn_id,
                status="error",
                model_inference_ms=round(
                    (time.perf_counter() - model_started) * 1000.0, 3
                ),
                model_output={
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            raise
        model_inference_ms = (time.perf_counter() - model_started) * 1000.0
        emit_decision_timing(
            "decision_model_request_completed",
            request_id=trace_request_id,
            metadata=metadata,
            component="turn_router",
            status="success",
            model_request_ms=round(
                (time.perf_counter() - model_started) * 1000.0,
                3,
            ),
        )
        get_model_io_recorder("router").record(
            stage="router",
            input_value=generate_request,
            output_value=result,
            context=archive_context,
            status="success",
            started_at=archive_started,
        )
        model_output = str(result.text)
        emit_model_trace(
            "reply",
            "model.response.received",
            component="turn_router",
            request_id=trace_request_id,
            sequence=4,
            root_session_id=root_session_id,
            agent_session_id=agent_session_id,
            status="success",
            session_id=request.session_id,
            turn_id=trace_turn_id,
            model_output=model_output,
            usage=(
                result.usage.to_dict()
                if getattr(result, "usage", None) is not None
                else None
            ),
            weight_version=getattr(result, "weight_version", None),
            model_inference_ms=round(model_inference_ms, 3),
        )
        return model_output

    def model_version_for(self, request: TaskClassificationRequest) -> str:
        return self.model_version

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
            "model_version": self.model_version_for(request),
            "router_profile": request.router_profile,
            "prefix_cache_namespace": _prefix_namespace(request.router_profile),
            "cache_static_system_only": True,
            "stage": "router",
            "admission_priority": 0,
        }


def _prefix_namespace(router_profile: str) -> str:
    return f"turn-router:{router_profile.replace('.', '')}"


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
