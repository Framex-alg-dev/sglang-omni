# SPDX-License-Identifier: Apache-2.0
"""ASGI HTTP lifecycle logging for Omni services."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send

from sglang_omni.utils.framex_service_logs import (
    SERVICES,
    adopt_request_id,
    bind_service_log_context,
    emit_service_event,
    valid_correlation_id,
)

logger = logging.getLogger(__name__)


def install_framex_http_logging(
    app: FastAPI,
    *,
    service: str = "omni-reply-worker",
) -> None:
    """Install one HTTP lifecycle logger and traceable error responses."""

    installed = getattr(app.state, "framex_http_log_service", None)
    if installed is not None:
        if installed != service:
            raise ValueError(
                "FrameX HTTP logging service cannot change after installation"
            )
        return
    app.state.framex_http_log_service = service
    app.add_middleware(FramexHTTPRequestLogMiddleware, service=service)

    async def http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        )

    async def validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            detail=jsonable_encoder(exc.errors()),
        )

    async def internal_error(
        request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=500,
            detail="Internal Server Error",
        )

    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, internal_error)


class FramexHTTPRequestLogMiddleware:
    """Log the complete ASGI HTTP response lifecycle without buffering it."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        service: str = "omni-reply-worker",
    ) -> None:
        if service not in SERVICES:
            raise ValueError(f"unsupported FrameX service: {service!r}")
        self.app = app
        self.service = service

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        if scope.get("framex_http_log_active"):
            # Mounted FastAPI applications can install the same middleware.
            # The outermost instance owns the single lifecycle record.
            await self.app(scope, receive, send)
            return
        scope["framex_http_log_active"] = True

        request_id, rejected_request_id = adopt_request_id(
            _asgi_header(scope, b"x-request-id")
        )
        root_session_id, rejected_root_session_id = _optional_log_header(
            _asgi_header(scope, b"x-root-session-id")
        )
        turn_id, rejected_turn_id = _optional_log_header(
            _asgi_header(scope, b"x-turn-id")
        )
        if isinstance(state, dict):
            state["framex_request_id"] = request_id
            state["framex_root_session_id"] = root_session_id
            state["framex_turn_id"] = turn_id
        method = str(scope.get("method", ""))
        started = time.perf_counter()
        response_status: int | None = None
        response_started = False
        finalized = False
        client_disconnected = False
        transport_send_failed = False

        def finalize(
            outcome: str,
            *,
            level: int,
            status: int,
            error_type: str | None = None,
        ) -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            try:
                emit_service_event(
                    self.service,
                    "http.request.completed",
                    level=level,
                    logger_name=__name__,
                    method=method,
                    route=_route_template(scope),
                    status=status,
                    outcome=outcome,
                    response_started=response_started,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                    inbound_request_id_invalid=rejected_request_id,
                    inbound_root_session_id_invalid=rejected_root_session_id,
                    inbound_turn_id_invalid=rejected_turn_id,
                    error_type=error_type,
                )
            except Exception:
                # Observability must never change the HTTP response outcome.
                logger.exception("Failed to emit HTTP request lifecycle log")

        async def logging_receive() -> dict[str, Any]:
            nonlocal client_disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                client_disconnected = True
            return message

        async def logging_send(message: dict[str, Any]) -> None:
            nonlocal response_started, response_status, transport_send_failed
            message_type = message["type"]
            if message_type == "http.response.start":
                pending_status = int(message["status"])
                message = _with_asgi_response_header(
                    message,
                    b"x-request-id",
                    request_id.encode("ascii"),
                )
            try:
                await send(message)
            except BaseException:
                transport_send_failed = True
                raise
            if message_type == "http.response.start":
                response_started = True
                response_status = pending_status
            if (
                message_type == "http.response.body"
                and not message.get("more_body", False)
            ):
                status = response_status or 200
                finalize(
                    "completed",
                    level=logging.INFO if status < 500 else logging.ERROR,
                    status=status,
                )

        with bind_service_log_context(
            request_id=request_id,
            root_session_id=root_session_id,
            turn_id=turn_id,
        ):
            try:
                await self.app(scope, logging_receive, logging_send)
            except asyncio.CancelledError:
                finalize(
                    "cancelled",
                    level=logging.WARNING,
                    status=response_status or 499,
                    error_type="CancelledError",
                )
                raise
            except Exception as exc:
                disconnected = client_disconnected or transport_send_failed
                finalize(
                    "cancelled" if disconnected else "failed",
                    level=logging.WARNING if disconnected else logging.ERROR,
                    status=response_status or (499 if disconnected else 500),
                    error_type=type(exc).__name__,
                )
                raise
            finally:
                if not finalized:
                    disconnected = client_disconnected or transport_send_failed
                    finalize(
                        "cancelled" if disconnected else "failed",
                        level=(
                            logging.WARNING
                            if disconnected
                            else logging.ERROR
                        ),
                        status=(
                            response_status
                            or (499 if disconnected else 500)
                        ),
                        error_type=(
                            None if disconnected else "IncompleteResponse"
                        ),
                    )


def _optional_log_header(value: str | None) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if valid_correlation_id(value):
        return value, False
    return None, True


def _asgi_header(scope: Scope, name: bytes) -> str | None:
    for header_name, value in scope.get("headers", []):
        if header_name.lower() == name:
            return value.decode("latin-1")
    return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    for attribute in ("path_format", "path"):
        value = getattr(route, attribute, None)
        if isinstance(value, str) and value:
            return value
    # Do not put an unmatched raw path into CLS: it can contain IDs or content.
    return "<unmatched>"


def _with_asgi_response_header(
    message: dict[str, Any],
    name: bytes,
    value: bytes,
) -> dict[str, Any]:
    headers = [
        (header_name, header_value)
        for header_name, header_value in message.get("headers", [])
        if header_name.lower() != name
    ]
    headers.append((name, value))
    return {**message, "headers": headers}


def _error_response(
    request: Request,
    *,
    status_code: int,
    detail: Any,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "framex_request_id", None)
    if not valid_correlation_id(request_id):
        request_id = adopt_request_id(None)[0]
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        {
            "detail": detail,
            "error": {"request_id": request_id},
        },
        status_code=status_code,
        headers=response_headers,
    )


__all__ = [
    "FramexHTTPRequestLogMiddleware",
    "install_framex_http_logging",
]
