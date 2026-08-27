# SPDX-License-Identifier: Apache-2.0
"""Deterministic text-only model substitute for local Realtime development."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.routing import WebSocketRoute

from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    CompletionStreamChunk,
    GenerateRequest,
)
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)

_PREFIX = "SGLANG_OMNI_DEV_FAKE_MODEL_"
_DEFAULT_RESPONSE_TEXT = "这是本地开发模型返回的固定回复。"
_DEFAULT_CHUNK_SIZE = 4
_DEFAULT_CHUNK_INTERVAL_MS = 0
_ACTION_CANDIDATE_ENV = "SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID"
_RESERVED_ACTION_IDS = frozenset({"A000", "B000", "UNSUPPORTED"})
_SUPPORTED_HTTP_ROUTES = frozenset({"/health", "/v1/models"})
_SUPPORTED_WEBSOCKET_ROUTES = frozenset({"/v1/session/realtime"})

if TYPE_CHECKING:
    from fastapi import FastAPI


class RealtimeModelClient(Protocol):
    """The narrow client contract consumed by ``MultimodalSession``."""

    def completion_stream(
        self, request: GenerateRequest, *, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]: ...

    async def abort(self, request_id: str) -> AbortResult: ...

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult: ...


class DevRealtimeModelRequestError(ValueError):
    """Raised when Realtime builds a request outside the development contract."""


class DevRealtimeModelUnsupportedError(RuntimeError):
    """Raised when an endpoint outside the development contract is requested."""


@dataclass(frozen=True)
class DevRealtimeModelConfig:
    enabled: bool = False
    response_text: str = _DEFAULT_RESPONSE_TEXT
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    chunk_interval_ms: int = _DEFAULT_CHUNK_INTERVAL_MS
    action_candidate_id: str = ""

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DevRealtimeModelConfig":
        values = os.environ if environ is None else environ
        enabled = _parse_bool(values.get(f"{_PREFIX}ENABLED", "false"))
        response_text = values.get(f"{_PREFIX}RESPONSE_TEXT", _DEFAULT_RESPONSE_TEXT)
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
        action_candidate_id = values.get(_ACTION_CANDIDATE_ENV, "").strip()
        if (
            action_candidate_id in _RESERVED_ACTION_IDS
            or action_candidate_id.startswith("DEV_NONE_")
        ):
            raise ValueError(
                f"{_ACTION_CANDIDATE_ENV} uses a reserved action identifier: "
                f"{action_candidate_id}"
            )
        return cls(
            enabled=enabled,
            response_text=response_text,
            chunk_size=chunk_size,
            chunk_interval_ms=chunk_interval_ms,
            action_candidate_id=action_candidate_id,
        )

    def log_summary(self) -> dict[str, int | bool]:
        """Return a safe summary without exposing configured text."""
        return {
            "enabled": self.enabled,
            "response_text_length": len(self.response_text),
            "chunk_size": self.chunk_size,
            "chunk_interval_ms": self.chunk_interval_ms,
            "action_candidate_configured": bool(self.action_candidate_id),
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
        return self.config.response_text

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult:
        if not request.candidates:
            raise DevRealtimeModelRequestError(
                "action scoring candidates must not be empty"
            )
        candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        configured = self.config.action_candidate_id
        selected = configured if configured in candidate_ids else candidate_ids[0]
        if (
            configured
            and configured not in candidate_ids
            and request.stage != "category"
        ):
            raise DevRealtimeModelRequestError(
                "configured development action candidate is not in the "
                f"Session whitelist: {configured}"
            )
        scores = []
        for index, candidate_id in enumerate(candidate_ids):
            logprob = -0.01 if candidate_id == selected else -1.0 - index
            scores.append(
                CandidateScore(
                    candidate_id=candidate_id,
                    token_count=1,
                    mean_logprob=logprob,
                    mean_nll=-logprob,
                    ppl=1.0 if candidate_id == selected else 3.0 + index,
                    token_scores=[TokenScore(token_id=index + 1, logprob=logprob)],
                )
            )
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=scores,
            stats={"mode": "dev-fake-action"},
        )

    async def abort(self, request_id: str) -> AbortResult:
        active = request_id in self._active_request_ids
        if active:
            self._aborted_request_ids.add(request_id)
        return AbortResult(success=active, level_applied=AbortLevel.SOFT)

    def __getattr__(self, name: str):
        # Preserve normal Python feature detection. Shared app infrastructure
        # uses ``hasattr`` for optional action/resource-monitor capabilities.
        # Unsupported external routes are rejected by the transport boundary
        # installed below, before they can call the narrow client.
        raise AttributeError(name)


def install_dev_model_error_handler(app: "FastAPI") -> None:
    """Restrict the development app to its documented transport contract."""

    if getattr(app.state, "dev_fake_model_errors_installed", False):
        return
    app.state.dev_fake_model_errors_installed = True

    # ``create_app`` may gain new WebSocket APIs whose client contracts are
    # broader than RealtimeSession's completion_stream/abort pair. Development
    # mode exposes only its explicit WebSocket contract, so remove other WS
    # routes instead of allowing them to fail after accepting a connection.
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not isinstance(route, WebSocketRoute)
        or route.path in _SUPPORTED_WEBSOCKET_ROUTES
    ]

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
