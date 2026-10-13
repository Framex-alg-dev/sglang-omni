"""Prompted Router v0.7 adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import base64
import io
import wave
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .contracts import ClassificationMediaRef, TaskClassificationRequest
from .prompt import (
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
    """Classify one text or raw-audio turn with Router v0.7."""

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
            metadata["audios"] = [_audio_data_url(audio)]
        result = await self._client.completion(
            GenerateRequest(
                model=self.model_id,
                messages=[
                    Message(
                        role="system",
                        content=(
                            SYSTEM_PROMPT
                            if history
                            else SINGLE_TURN_SYSTEM_PROMPT
                        ),
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
            ),
            request_id=request.request_id,
        )
        return str(result.text)

    async def prewarm(self) -> None:
        """Populate both immutable Router system prefixes."""

        await self.classify(
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
        await self.classify(
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
                    "text": "请确认是否继续",
                    "playback_status": "completed",
                },),
                has_active_agent=True,
                follow_up_required=True,
            )
        )

    @staticmethod
    def _metadata(request: TaskClassificationRequest) -> dict[str, Any]:
        return {
            "task": "turn_router",
            "task_role": "turn_router",
            "contract_version": request.contract_version,
            "logical_request_id": request.turn_id,
            "session_instance_id": request.session_id,
            "session_id": request.session_id,
            "prefix_cache_namespace": "turn-router:v08",
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
