"""Prompted turn-router adapter backed by the local SGLang Omni client."""

from __future__ import annotations

import base64
import io
import wave
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .contracts import ClassificationMediaRef, TaskClassificationRequest
from .prompt import SYSTEM_PROMPT, build_user_prompt


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...


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
        audio = request.media[0] if request.media else None
        metadata: dict[str, Any] = {
            "task": "turn_router",
            "task_role": "turn_router",
            "contract_version": request.contract_version,
            "logical_request_id": request.turn_id,
        }
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
                    max_new_tokens=32,
                    stop=["\n"],
                ),
                stream=False,
                max_tokens=32,
                output_modalities=["text"],
                metadata=metadata,
            ),
            request_id=request.request_id,
        )
        return str(result.text)


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
