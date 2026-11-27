"""Prompted turn-router adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from types import MappingProxyType
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.profiler.decision_timing import emit_decision_timing
from sglang_omni.serve.model_io import get_model_io_recorder
from sglang_omni.utils.model_trace_logs import emit_model_trace

from . import prompt_v09, prompt_v10, prompt_v11, prompt_v12
from .contracts import (
    DEFAULT_ROUTER_PROFILE,
    LEGACY_CONTRACT_VERSION,
    SUPPORTED_ROUTER_PROFILES,
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
        prompt_snapshot: dict[str, str] | None = None,
    ) -> None:
        if not model_id.strip() or not model_version.strip():
            raise ValueError("turn-router model_id and model_version are required")
        if prompt_snapshot is not None and set(prompt_snapshot) != {
            "system_prompt",
            "mcp_scope",
            "multimodal_scope",
        }:
            raise ValueError("complete Router prompt snapshot required")
        self._prompt_snapshot = (
            MappingProxyType(dict(prompt_snapshot))
            if prompt_snapshot is not None
            else None
        )
        self._client = client
        self.model_id = model_id
        self.model_version = model_version

    async def classify(self, request: TaskClassificationRequest) -> str:
        input_started = time.perf_counter()
        if self._prompt_snapshot is not None and request.router_profile != "v1.0":
            raise ValueError("debug snapshots require Router v1.0")
        supplied = request.prompt_snapshot
        if supplied is not None:
            if (
                supplied.binding.model_id != self.model_id
                or supplied.binding.model_version != self.model_version_for(request)
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
                        "serving_model_version": self.model_version_for(request),
                    },
                )
            snapshot = supplied.rules
        else:
            snapshot = self._prompt_snapshot
        audio = request.media[0] if request.media else None
        prompt = _prompt_contract(request.router_profile)
        history = prompt.normalize_history(request.router_history)
        query_mode = "audio" if audio is not None else "text"
        prompt_kwargs = dict(
            query_mode=query_mode,
            text=request.text,
            history=history,
            mcp_scope=(
                snapshot["mcp_scope"]
                if snapshot is not None
                else prompt.DEFAULT_MCP_SCOPE
            ),
            multimodal_scope=(
                snapshot["multimodal_scope"]
                if snapshot is not None
                else prompt.DEFAULT_MULTIMODAL_SCOPE
            ),
            has_active_agent=request.has_active_agent,
            pending_confirmation=request.pending_confirmation,
            follow_up_required=request.follow_up_required,
        )
        if request.router_profile in {"v1.1", "v1.2"}:
            prompt_kwargs.update(
                active_task_goal=request.active_task_goal,
                active_task_status=request.active_task_status,
            )
        if request.router_profile == "v1.2":
            prompt_kwargs.update(
                active_task_family=request.active_task_family,
                active_task_phase=request.active_task_phase,
                active_task_awaiting_input=request.active_task_awaiting_input,
            )
        content = prompt.build_user_prompt(**prompt_kwargs)
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
        system_prompt = (
            snapshot["system_prompt"]
            if snapshot is not None
            else (
                prompt.SYSTEM_PROMPT
                if history
                else prompt.SINGLE_TURN_SYSTEM_PROMPT
            )
        )
        generate_request = GenerateRequest(
            model=self.model_id,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=content),
            ],
            sampling=(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=1,
                    stop=["\n"],
                    seed=0,
                    regex=("[0-5]" if request.has_active_agent else "[0-3]"),
                )
                if request.router_profile == "v1.2"
                else SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=1,
                    stop=["\n"],
                )
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
            request_id=request.request_id,
            sequence=2,
            root_session_id=request.session_id,
            agent_session_id=request.session_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
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
            "request_id": request.request_id,
            "root_session_id": request.session_id,
            "turn_id": request.turn_id,
            "model_id": self.model_id,
            "model_version": self.model_version_for(request),
            "release_id": supplied.release_id if supplied is not None else None,
            "prompt_snapshot_sha256": (
                supplied.bundle_sha256 if supplied is not None else None
            ),
        }
        emit_decision_timing(
            "decision_model_request_started",
            request_id=request.request_id,
            metadata=metadata,
            component="turn_router",
        )
        emit_model_trace(
            "reply",
            "model.request.started",
            component="turn_router",
            request_id=request.request_id,
            sequence=3,
            root_session_id=request.session_id,
            agent_session_id=request.session_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
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
                request_id=request.request_id,
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
                request_id=request.request_id,
                sequence=4,
                root_session_id=request.session_id,
                agent_session_id=request.session_id,
                turn_id=request.turn_id,
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
            emit_model_trace(
                "reply",
                "model.response.received",
                component="turn_router",
                request_id=request.request_id,
                sequence=4,
                root_session_id=request.session_id,
                agent_session_id=request.session_id,
                turn_id=request.turn_id,
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
            request_id=request.request_id,
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
            request_id=request.request_id,
            sequence=4,
            root_session_id=request.session_id,
            agent_session_id=request.session_id,
            status="success",
            session_id=request.session_id,
            turn_id=request.turn_id,
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

    async def prewarm(self) -> None:
        """Populate every supported prompt prefix and verify route invariants."""

        for router_profile in SUPPORTED_ROUTER_PROFILES:
            direct = await self.classify(
                TaskClassificationRequest(
                    request_id=f"turn-router-{router_profile}-prewarm",
                    session_id="turn-router-prewarm",
                    turn_id=f"turn-router-{router_profile}-prewarm",
                    identity_epoch=0,
                    input_revision=0,
                    text="你好",
                    media=(),
                    router_profile=router_profile,
                    contract_version=LEGACY_CONTRACT_VERSION,
                )
            )
            if direct != "0":
                raise RuntimeError(
                    "turn-router prewarm expected ordinary conversation to route 0, "
                    f"got {direct!r} for {router_profile}"
                )

            delegate = await self.classify(
                TaskClassificationRequest(
                    request_id=f"turn-router-{router_profile}-multi-turn-prewarm",
                    session_id="turn-router-prewarm",
                    turn_id=f"turn-router-{router_profile}-multi-turn-prewarm",
                    identity_epoch=0,
                    input_revision=0,
                    text="继续",
                    media=(),
                    router_history=(
                        {
                            "role": "assistant",
                            "text": "需要我查询当前天气吗？",
                            "playback_status": "completed",
                        },
                    ),
                    has_active_agent=True,
                    follow_up_required=True,
                    active_task_goal="查询当前天气",
                    active_task_status="waiting_user",
                    active_task_family="weather",
                    active_task_phase="waiting_user",
                    active_task_awaiting_input="details",
                    router_profile=router_profile,
                    contract_version=LEGACY_CONTRACT_VERSION,
                )
            )
            expected_delegate = "4" if router_profile in {"v1.1", "v1.2"} else "1"
            if delegate != expected_delegate:
                raise RuntimeError(
                    "turn-router prewarm expected an inherited MCP request to route "
                    f"{expected_delegate}, "
                    f"got {delegate!r} for {router_profile}"
                )

    def model_version_for(self, request: TaskClassificationRequest) -> str:
        if request.router_profile == DEFAULT_ROUTER_PROFILE:
            return self.model_version
        suffix = f"-{DEFAULT_ROUTER_PROFILE}"
        if self.model_version.endswith(suffix):
            prefix = self.model_version[: -len(DEFAULT_ROUTER_PROFILE)]
            return prefix + request.router_profile
        return f"{self.model_version}+prompt-{request.router_profile}"

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


def _prompt_contract(router_profile: str) -> Any:
    if router_profile == "v1.2":
        return prompt_v12
    if router_profile == "v1.1":
        return prompt_v11
    if router_profile == "v1.0":
        return prompt_v10
    if router_profile == "v0.9":
        return prompt_v09
    raise ValueError("unsupported turn-router profile")


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
