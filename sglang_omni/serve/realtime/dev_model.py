# SPDX-License-Identifier: Apache-2.0
"""Deterministic text-only model substitute for local Realtime development."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    CompletionStreamChunk,
    GenerateRequest,
)

_PREFIX = "SGLANG_OMNI_DEV_FAKE_MODEL_"
_DEFAULT_RESPONSE_TEXT = "这是本地开发模型返回的固定回复。"
_DEFAULT_TRANSCRIPT_TEXT = "这是本地开发模型生成的固定转写。"
_DEFAULT_CHUNK_SIZE = 4
_DEFAULT_CHUNK_INTERVAL_MS = 0
_AUDIO_DATA_PREFIX = "data:audio/"
_OCTET_STREAM_DATA_PREFIX = "data:application/octet-stream;base64,"
_SUPPORTED_HTTP_ROUTES = frozenset({"/health", "/v1/models"})

if TYPE_CHECKING:
    from fastapi import FastAPI


class RealtimeModelClient(Protocol):
    """The narrow client contract consumed by ``RealtimeSession``."""

    def completion_stream(
        self, request: GenerateRequest, *, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]: ...

    async def abort(self, request_id: str) -> AbortResult: ...


class DevRealtimeModelRequestError(ValueError):
    """Raised when Realtime builds a request outside the development contract."""


class DevRealtimeModelUnsupportedError(RuntimeError):
    """Raised when an endpoint outside the development contract is requested."""


@dataclass(frozen=True)
class DevRealtimeModelConfig:
    enabled: bool = False
    response_text: str = _DEFAULT_RESPONSE_TEXT
    transcript_text: str = _DEFAULT_TRANSCRIPT_TEXT
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    chunk_interval_ms: int = _DEFAULT_CHUNK_INTERVAL_MS

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DevRealtimeModelConfig":
        values = os.environ if environ is None else environ
        enabled = _parse_bool(values.get(f"{_PREFIX}ENABLED", "false"))
        response_text = values.get(f"{_PREFIX}RESPONSE_TEXT", _DEFAULT_RESPONSE_TEXT)
        transcript_text = values.get(
            f"{_PREFIX}TRANSCRIPT_TEXT", _DEFAULT_TRANSCRIPT_TEXT
        )
        chunk_size = _parse_int(
            values.get(f"{_PREFIX}CHUNK_SIZE", str(_DEFAULT_CHUNK_SIZE)),
            name=f"{_PREFIX}CHUNK_SIZE",
            minimum=1,
        )
        chunk_interval_ms = _parse_int(
            values.get(
                f"{_PREFIX}CHUNK_INTERVAL_MS",
                str(_DEFAULT_CHUNK_INTERVAL_MS),
            ),
            name=f"{_PREFIX}CHUNK_INTERVAL_MS",
            minimum=0,
        )
        if not response_text:
            raise ValueError(f"{_PREFIX}RESPONSE_TEXT must not be empty")
        if not transcript_text:
            raise ValueError(f"{_PREFIX}TRANSCRIPT_TEXT must not be empty")
        return cls(
            enabled=enabled,
            response_text=response_text,
            transcript_text=transcript_text,
            chunk_size=chunk_size,
            chunk_interval_ms=chunk_interval_ms,
        )

    def log_summary(self) -> dict[str, int | bool]:
        """Return a safe summary without exposing configured text."""
        return {
            "enabled": self.enabled,
            "response_text_length": len(self.response_text),
            "transcript_text_length": len(self.transcript_text),
            "chunk_size": self.chunk_size,
            "chunk_interval_ms": self.chunk_interval_ms,
        }


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{_PREFIX}ENABLED must be 'true' or 'false'")


def _parse_int(value: str, *, name: str, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


class DevRealtimeModelClient:
    """Validate Realtime requests and stream deterministic fixed text."""

    def __init__(self, config: DevRealtimeModelConfig) -> None:
        if not config.enabled:
            raise ValueError("development model client requires enabled=true")
        self.config = config
        self._active_request_ids: set[str] = set()
        self._aborted_request_ids: set[str] = set()

    def health(self) -> dict[str, object]:
        return {"running": True, "mode": "dev-fake-realtime-model"}

    def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        del audio_format
        output_text = self._validate_and_select_text(request, request_id)
        return self._stream_text(output_text, request_id)

    async def _stream_text(
        self, output_text: str, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        self._active_request_ids.add(request_id)
        try:
            for index in range(0, len(output_text), self.config.chunk_size):
                if request_id in self._aborted_request_ids:
                    return
                if index and self.config.chunk_interval_ms:
                    await asyncio.sleep(self.config.chunk_interval_ms / 1000)
                yield CompletionStreamChunk(
                    request_id=request_id,
                    text=output_text[index : index + self.config.chunk_size],
                    modality="text",
                )
            if request_id not in self._aborted_request_ids:
                yield CompletionStreamChunk(
                    request_id=request_id,
                    modality="text",
                    finish_reason="stop",
                )
        finally:
            self._active_request_ids.discard(request_id)
            self._aborted_request_ids.discard(request_id)

    def _validate_and_select_text(
        self, request: GenerateRequest, request_id: str
    ) -> str:
        if not request_id.strip():
            raise DevRealtimeModelRequestError("request_id must not be empty")
        if request.stream is not True:
            raise DevRealtimeModelRequestError("stream must be true")
        if request.output_modalities != ["text"]:
            raise DevRealtimeModelRequestError(
                "output_modalities must be exactly ['text']"
            )
        if not request.messages:
            raise DevRealtimeModelRequestError("messages must not be empty")

        roles = {message.role for message in request.messages}
        if not {"system", "user"}.issubset(roles):
            raise DevRealtimeModelRequestError(
                "messages must contain system and user roles"
            )
        audios = request.metadata.get("audios")
        if not isinstance(audios, list) or not audios:
            raise DevRealtimeModelRequestError(
                "metadata.audios must be a non-empty list"
            )
        if any(not _is_supported_audio_data_uri(audio) for audio in audios):
            raise DevRealtimeModelRequestError(
                "metadata.audios entries must be supported audio data URIs"
            )

        prompt = "\n".join(str(message.content).lower() for message in request.messages)
        if "transcribe the spoken audio" in prompt:
            return self.config.transcript_text
        if "listen to the spoken audio above and respond to it" in prompt:
            return self.config.response_text
        raise DevRealtimeModelRequestError(
            "unable to classify request as response or transcription pass"
        )

    async def abort(self, request_id: str) -> AbortResult:
        active = request_id in self._active_request_ids
        if active:
            self._aborted_request_ids.add(request_id)
        return AbortResult(success=active, level_applied=AbortLevel.SOFT)

    def __getattr__(self, name: str):
        raise DevRealtimeModelUnsupportedError(
            f"{name} is unsupported while the Realtime development model is enabled"
        )


def install_dev_model_error_handler(app: "FastAPI") -> None:
    """Restrict the development app to its documented transport contract."""

    if getattr(app.state, "dev_fake_model_errors_installed", False):
        return
    app.state.dev_fake_model_errors_installed = True

    @app.middleware("http")
    async def reject_unsupported_http_routes(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path not in _SUPPORTED_HTTP_ROUTES:
            return _unsupported_response(request.url.path)
        return await call_next(request)

    @app.exception_handler(DevRealtimeModelUnsupportedError)
    async def handle_unsupported_operation(
        request: Request, exc: DevRealtimeModelUnsupportedError
    ) -> JSONResponse:
        del request
        return _unsupported_response(str(exc))


def _unsupported_response(operation: str) -> JSONResponse:
    detail = operation
    if not operation.endswith("Realtime development model is enabled"):
        detail = (
            f"{operation} is unsupported while the Realtime development "
            "model is enabled"
        )
    return JSONResponse(
        status_code=501,
        content={
            "detail": detail,
            "type": "dev_fake_model_unsupported",
        },
    )


def _is_supported_audio_data_uri(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    if lowered.startswith(_AUDIO_DATA_PREFIX):
        header, separator, payload = value.partition(",")
        if not separator or not header.lower().endswith(";base64"):
            return False
    elif lowered.startswith(_OCTET_STREAM_DATA_PREFIX):
        payload = value[len(_OCTET_STREAM_DATA_PREFIX) :]
    else:
        return False
    if not payload:
        return False
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True
