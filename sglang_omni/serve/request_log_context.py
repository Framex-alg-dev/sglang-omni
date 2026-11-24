# SPDX-License-Identifier: Apache-2.0
"""Shared correlation and lifecycle logging for Omni model requests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from starlette.websockets import WebSocketDisconnect

from sglang_omni.utils.framex_service_logs import (
    adopt_request_id,
    bind_service_log_context,
    correlation_id_or_digest,
    current_service_log_context,
    emit_service_event,
    valid_correlation_id,
)


@dataclass(frozen=True)
class RequestLogContext:
    """The bounded correlation fields accepted from an internal caller."""

    request_id: str
    root_session_id: str | None = None
    turn_id: str | None = None
    agent_session_id: str | None = None


def payload_request_log_context(
    payload: Mapping[str, Any],
    *,
    fallback_request_id: str,
    fallback_root_session_id: str | None = None,
    fallback_turn_id: str | None = None,
    fallback_agent_session_id: str | None = None,
) -> RequestLogContext:
    """Read ``payload.log_context`` without leaking model input into logs."""

    raw = payload.get("log_context")
    if raw is None:
        context: Mapping[str, Any] = {}
    elif isinstance(raw, Mapping):
        context = raw
    else:
        raise ValueError("log_context must be an object")

    bound = current_service_log_context()
    explicit_request_id = _optional_id(context, "request_id")
    return RequestLogContext(
        request_id=(
            bound.request_id
            or explicit_request_id
            or adopt_request_id(fallback_request_id)[0]
        ),
        root_session_id=(
            bound.root_session_id
            or _optional_id(context, "root_session_id")
            or _fallback_id(fallback_root_session_id, prefix="legacy_root")
        ),
        turn_id=(
            bound.turn_id
            or _optional_id(context, "turn_id")
            or _fallback_id(fallback_turn_id, prefix="legacy_turn")
        ),
        agent_session_id=(
            _optional_id(context, "agent_session_id")
            or _fallback_id(fallback_agent_session_id, prefix="legacy_agent")
        ),
    )


@asynccontextmanager
async def model_request_log(
    service: str,
    operation: str,
    context: RequestLogContext,
    *,
    operation_id: str,
    logger_name: str,
) -> AsyncIterator[None]:
    """Emit exactly one terminal lifecycle event for a model request."""

    started = time.perf_counter()
    logged_operation_id = correlation_id_or_digest(
        operation_id, prefix="legacy_operation"
    )
    with bind_service_log_context(
        request_id=context.request_id,
        root_session_id=context.root_session_id,
        turn_id=context.turn_id,
    ):
        emit_service_event(
            service,
            "inference.started",
            logger_name=logger_name,
            operation=operation,
            operation_id=logged_operation_id,
        )
        try:
            yield
        except asyncio.CancelledError:
            emit_service_event(
                service,
                "inference.cancelled",
                level=logging.WARNING,
                logger_name=logger_name,
                operation=operation,
                operation_id=logged_operation_id,
                status=499,
                duration_ms=_duration_ms(started),
                error_type="CancelledError",
            )
            raise
        except Exception as exc:
            emit_service_event(
                service,
                "inference.failed",
                level=logging.ERROR,
                logger_name=logger_name,
                operation=operation,
                operation_id=logged_operation_id,
                status=500,
                duration_ms=_duration_ms(started),
                error_type=type(exc).__name__,
            )
            raise
        else:
            emit_service_event(
                service,
                "inference.completed",
                logger_name=logger_name,
                operation=operation,
                operation_id=logged_operation_id,
                status=200,
                duration_ms=_duration_ms(started),
            )


@asynccontextmanager
async def websocket_request_log(
    service: str,
    connection_type: str,
    context: RequestLogContext,
    *,
    operation_id: str,
    logger_name: str,
) -> AsyncIterator[None]:
    """Log a one-request WebSocket connection and its terminal reason."""

    logged_operation_id = correlation_id_or_digest(
        operation_id, prefix="legacy_operation"
    )
    emit_service_event(
        service,
        "ws.connected",
        logger_name=logger_name,
        request_id=context.request_id,
        root_session_id=context.root_session_id,
        turn_id=context.turn_id,
        connection_type=connection_type,
        operation_id=logged_operation_id,
        agent_session_id=context.agent_session_id,
    )
    reason = "completed"
    try:
        yield
    except WebSocketDisconnect:
        reason = "peer_disconnected"
        raise
    except asyncio.CancelledError:
        reason = "server_cancelled"
        raise
    except Exception:
        reason = "failed"
        raise
    finally:
        emit_service_event(
            service,
            "ws.closed",
            logger_name=logger_name,
            request_id=context.request_id,
            root_session_id=context.root_session_id,
            turn_id=context.turn_id,
            connection_type=connection_type,
            operation_id=logged_operation_id,
            reason=reason,
        )


def _optional_id(raw: Mapping[str, Any], name: str) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not valid_correlation_id(value):
        raise ValueError(f"log_context.{name} is not a valid correlation ID")
    return value


def _fallback_id(value: str | None, *, prefix: str) -> str | None:
    if value is None:
        return None
    return correlation_id_or_digest(value, prefix=prefix)


def _duration_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = [
    "RequestLogContext",
    "model_request_log",
    "payload_request_log_context",
    "websocket_request_log",
]
