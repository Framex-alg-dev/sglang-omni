"""Private WebSocket transport for continuous timeline detection."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from framex_event_policy.windows import DetectionWindow, RAW_WINDOW_MODE, LEGACY_OBSERVATION_MODE

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.http_request_logs import install_framex_http_logging
from .contracts import (
    CONTRACT_VERSION,
    MediaKind,
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from .session import TimelineDetectionModel, TimelineDetectionSession
from .diagnostics import TimelineIngressDiagnostics
from sglang_omni.utils.framex_service_logs import (
    correlation_id_or_digest,
    adopt_request_id,
    bind_service_log_context,
    emit_service_event,
    new_request_id,
)
from sglang_omni.utils.model_trace_logs import emit_model_trace


TimelineModelFactory = Callable[[TimelineSessionStart], TimelineDetectionModel]


@dataclass(frozen=True)
class _MediaWork:
    chunk: TimelineMediaChunk


@dataclass(frozen=True)
class _DiscontinuityWork:
    event: TimelineDiscontinuity
    applied: asyncio.Future[None]


_Work = _MediaWork | _DiscontinuityWork


def create_timeline_detection_app(
    model_factory: TimelineModelFactory,
    *,
    token: str,
    max_chunk_bytes: int = 8 * 1024 * 1024,
    max_pending_chunks: int = 64,
    max_pending_bytes: int = 128 * 1024 * 1024,
) -> FastAPI:
    if not token:
        raise ValueError("timeline-detection service token is required")
    if min(max_chunk_bytes, max_pending_chunks, max_pending_bytes) <= 0:
        raise ValueError("timeline service queue limits must be positive")
    app = FastAPI(title="sglang-omni-timeline-detection", version="1")
    install_framex_http_logging(app, service="omni-event-worker")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {token}"):
            raise HTTPException(status_code=404, detail="not found")
        return JSONResponse({"ok": True, "contract_version": CONTRACT_VERSION})

    @app.websocket("/v1/timeline")
    async def timeline(websocket: WebSocket) -> None:
        supplied = websocket.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {token}"):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        connection_request_id = new_request_id()
        session: TimelineDetectionSession | None = None
        worker: asyncio.Task[None] | None = None
        observation_worker: asyncio.Task[None] | None = None
        diagnostic_worker: asyncio.Task[None] | None = None
        diagnostics: TimelineIngressDiagnostics | None = None
        close_reason = "closed"
        work_queue: asyncio.Queue[_Work] = asyncio.Queue(max_pending_chunks)
        send_lock = asyncio.Lock()
        pending_bytes = 0

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def report_model_failure(exc: BaseException) -> None:
            while not work_queue.empty():
                pending = work_queue.get_nowait()
                if (
                    isinstance(pending, _DiscontinuityWork)
                    and not pending.applied.done()
                ):
                    pending.applied.set_exception(exc)
                work_queue.task_done()
            try:
                await send_json(
                    {
                        "type": "error",
                        "code": "model_failure",
                        "detail": "timeline model inference failed",
                    }
                )
                await websocket.close(code=1011)
            except (RuntimeError, WebSocketDisconnect):
                pass

        async def run_model() -> None:
            nonlocal pending_bytes
            assert session is not None
            try:
                while True:
                    work = await work_queue.get()
                    try:
                        if isinstance(work, _MediaWork):
                            await session.append(work.chunk)
                            if diagnostics is not None:
                                diagnostics.processed(work.chunk)
                        else:
                            await session.discontinuity(work.event)
                            if diagnostics is not None:
                                diagnostics.reset(work.event.new_stream_epoch)
                            if not work.applied.done():
                                work.applied.set_result(None)
                    except Exception as exc:
                        if (
                            isinstance(work, _DiscontinuityWork)
                            and not work.applied.done()
                        ):
                            work.applied.set_exception(exc)
                        raise
                    finally:
                        if isinstance(work, _MediaWork):
                            pending_bytes -= len(work.chunk.payload)
                        work_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await report_model_failure(exc)
                raise

        async def send_observations() -> None:
            assert session is not None
            try:
                while True:
                    started = time.perf_counter()
                    events = await session.next_observations()
                    context = session.output_log_context
                    with bind_service_log_context(
                        request_id=context.request_id if context else connection_request_id,
                        root_session_id=start.root_session_id or start.session_id,
                        turn_id=context.turn_id if context else None,
                    ):
                        if isinstance(events, DetectionWindow):
                            payload = events.to_wire()
                            if context is not None:
                                payload["log_context"] = {
                                    "request_id": context.request_id,
                                    "root_session_id": context.root_session_id,
                                    "turn_id": context.turn_id,
                                }
                            send_started = time.perf_counter()
                            await send_json(payload)
                            emit_model_trace(
                                "diagnostic",
                                "agent.response.sent",
                                component="timeline_detection",
                                request_id=(
                                    context.request_id
                                    if context is not None
                                    else connection_request_id
                                ),
                                invocation_id=(
                                    context.request_id
                                    if context is not None
                                    else connection_request_id
                                ),
                                sequence=6,
                                root_session_id=start.root_session_id
                                or start.session_id,
                                agent_session_id=start.agent_session_id
                                or start.session_id,
                                turn_id=context.turn_id if context else None,
                                transport="websocket",
                                agent_response=payload,
                                response_send_ms=round(
                                    (time.perf_counter() - send_started) * 1000.0,
                                    3,
                                ),
                            )
                            emit_service_event(
                                "omni-event-worker",
                                "inference.completed",
                                logger_name=__name__,
                                root_session_id=(
                                    start.root_session_id or start.session_id
                                ),
                                operation="timeline_window",
                                operation_id=context.turn_id if context else None,
                                window_id=events.evidence_end_ms,
                                status=200,
                                duration_ms=round(
                                    (time.perf_counter() - started) * 1000, 3
                                ),
                                event_count=len(events.events),
                                model_id=start.model_id,
                            )
                        else:
                            payloads = [_event_to_json(event) for event in events]
                            send_started = time.perf_counter()
                            for payload in payloads:
                                await send_json(payload)
                            emit_model_trace(
                                "diagnostic",
                                "agent.response.sent",
                                component="timeline_detection",
                                request_id=(
                                    context.request_id
                                    if context is not None
                                    else connection_request_id
                                ),
                                invocation_id=(
                                    context.request_id
                                    if context is not None
                                    else connection_request_id
                                ),
                                sequence=6,
                                root_session_id=start.root_session_id
                                or start.session_id,
                                agent_session_id=start.agent_session_id
                                or start.session_id,
                                turn_id=context.turn_id if context else None,
                                transport="websocket",
                                agent_response=payloads,
                                response_send_ms=round(
                                    (time.perf_counter() - send_started) * 1000.0,
                                    3,
                                ),
                            )
                            emit_service_event(
                                "omni-event-worker",
                                "inference.completed",
                                logger_name=__name__,
                                root_session_id=(
                                    start.root_session_id or start.session_id
                                ),
                                operation="timeline_observation",
                                operation_id=context.turn_id if context else None,
                                status=200,
                                duration_ms=round(
                                    (time.perf_counter() - started) * 1000, 3
                                ),
                                event_count=len(events),
                                model_id=start.model_id,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await report_model_failure(exc)
                raise

        try:
            first = await websocket.receive_json()
            start = _start_from_json(_object(first, "session.start"))
            connection_request_id = adopt_request_id(first.get("request_id"))[0]
            emit_service_event(
                "omni-event-worker",
                "ws.connected",
                logger_name=__name__,
                request_id=connection_request_id,
                root_session_id=start.root_session_id or start.session_id,
                agent_session_id=start.agent_session_id or start.session_id,
                connection_type="timeline",
            )
            session = TimelineDetectionSession(start, model_factory(start))
            await session.start()
            diagnostics = TimelineIngressDiagnostics(start)
            diagnostic_worker = asyncio.create_task(
                diagnostics.run(lambda: (work_queue.qsize(), pending_bytes)),
                name=f"timeline-ingress-diagnostics-{start.session_id}",
            )
            worker = asyncio.create_task(
                run_model(),
                name=f"timeline-media-{start.session_id}",
            )
            observation_worker = asyncio.create_task(
                send_observations(),
                name=f"timeline-observations-{start.session_id}",
            )
            await send_json(
                {
                    "type": "session.ready",
                    "session_id": start.session_id,
                    "identity_epoch": start.identity_epoch,
                    "observer_epoch": start.observer_epoch,
                    "stream_epoch": start.stream_epoch,
                    "next_sequence": start.next_sequence,
                    "contract_version": start.contract_version,
                    "root_session_id": start.root_session_id or start.session_id,
                    "agent_session_id": start.agent_session_id or start.session_id,
                    **({"result_mode": RAW_WINDOW_MODE} if start.result_mode == RAW_WINDOW_MODE else {}),
                    **(
                        {
                            "prompt_release_id": start.prompt_snapshot.release_id,
                            "prompt_bundle_sha256": start.prompt_snapshot.bundle_sha256,
                        }
                        if start.prompt_snapshot is not None else {}
                    ),
                }
            )
            while True:
                message = _object(await websocket.receive_json(), "timeline message")
                message_type = message.get("type")
                if message_type == "media":
                    declared_bytes = _integer(message, "payload_bytes")
                    if declared_bytes <= 0 or declared_bytes > max_chunk_bytes:
                        raise ValueError("payload_bytes is outside the service limit")
                    payload = await websocket.receive_bytes()
                    if len(payload) != declared_bytes:
                        raise ValueError("binary payload length does not match header")
                    if pending_bytes + declared_bytes > max_pending_bytes:
                        raise ValueError("timeline pending media exceeds byte limit")
                    chunk = _chunk_from_json(message, payload)
                    try:
                        work_queue.put_nowait(_MediaWork(chunk))
                    except asyncio.QueueFull as exc:
                        raise ValueError(
                            "timeline pending media queue is full"
                        ) from exc
                    pending_bytes += declared_bytes
                    diagnostics.received(chunk)
                    await send_json(
                        {
                            "type": "media.ack",
                            "observer_epoch": start.observer_epoch,
                            "sequence": _integer(message, "sequence"),
                        }
                    )
                elif message_type == "discontinuity":
                    event = _discontinuity_from_json(message)
                    applied = asyncio.get_running_loop().create_future()
                    await work_queue.put(_DiscontinuityWork(event, applied))
                    await applied
                    await send_json(
                        {
                            "type": "discontinuity.ack",
                            "observer_epoch": start.observer_epoch,
                            "stream_epoch": event.new_stream_epoch,
                        }
                    )
                elif message_type == "session.close":
                    # A media ACK means the server accepted ownership. Drain
                    # every accepted item before acknowledging a graceful close.
                    await work_queue.join()
                    if worker is not None and worker.done():
                        await worker
                    await send_json(
                        {
                            "type": "session.closed",
                            "observer_epoch": start.observer_epoch,
                        }
                    )
                    return
                else:
                    raise ValueError(f"unsupported timeline message type: {message_type!r}")
        except WebSocketDisconnect:
            close_reason = "peer_disconnected"
            return
        except asyncio.CancelledError:
            close_reason = "server_cancelled"
            raise
        except ValueError as exc:
            close_reason = "invalid_request"
            emit_service_event(
                "omni-event-worker",
                "ws.request.failed",
                level=logging.WARNING,
                logger_name=__name__,
                request_id=connection_request_id,
                root_session_id=(
                    start.root_session_id or start.session_id
                    if session is not None
                    else None
                ),
                error_type=type(exc).__name__,
            )
            try:
                await send_json(
                    {"type": "error", "code": "invalid_request", "detail": str(exc)}
                )
                await websocket.close(code=4400)
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception as exc:
            close_reason = "internal_error"
            emit_service_event(
                "omni-event-worker",
                "ws.request.failed",
                level=logging.ERROR,
                logger_name=__name__,
                request_id=connection_request_id,
                root_session_id=(
                    start.root_session_id or start.session_id
                    if session is not None
                    else None
                ),
                error_type=type(exc).__name__,
            )
            try:
                await websocket.close(code=1011)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if diagnostic_worker is not None:
                diagnostic_worker.cancel()
                await asyncio.gather(diagnostic_worker, return_exceptions=True)
            if diagnostics is not None:
                diagnostics.report(
                    reason="closed", pending_chunks=work_queue.qsize(),
                    pending_bytes=pending_bytes,
                )
            if observation_worker is not None:
                observation_worker.cancel()
                await asyncio.gather(observation_worker, return_exceptions=True)
            if worker is not None:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            if session is not None:
                await session.close()
            if session is not None:
                emit_service_event(
                    "omni-event-worker",
                    "ws.closed",
                    logger_name=__name__,
                    request_id=connection_request_id,
                    root_session_id=start.root_session_id or start.session_id,
                    reason=close_reason,
                )

    return app


def _start_from_json(raw: dict[str, Any]) -> TimelineSessionStart:
    if raw.get("type") != "session.start":
        raise ValueError("first message must be session.start")
    from sglang_omni.serve.agent_prompt_snapshot import decode_timeline_prompt_snapshot
    model_id = _string(raw, "model_id")
    session_id = _string(raw, "session_id")
    root_session_id = _optional_string(raw, "root_session_id")
    agent_session_id = _optional_string(raw, "agent_session_id")
    if raw.get("prompt_snapshot") is None:
        raise ValueError("timeline session requires Agent prompt_snapshot")
    prompt_snapshot = decode_timeline_prompt_snapshot(raw["prompt_snapshot"])
    from .event1115 import event1115_names_from_agent_snapshot
    event1115_names_from_agent_snapshot(
        prompt_snapshot.system_prompt,
        prompt_snapshot.event_registry_json,
        prompt_snapshot.event_codebook_csv,
    )
    return TimelineSessionStart(
        session_id=session_id,
        identity_epoch=_integer(raw, "identity_epoch"),
        stream_epoch=_integer(raw, "stream_epoch"),
        audio_format=_string(raw, "audio_format"),
        video_format=_string(raw, "video_format"),
        model_id=model_id,
        observer_epoch=_integer(raw, "observer_epoch"),
        next_sequence=_integer(raw, "next_sequence"),
        contract_version=_integer(raw, "contract_version"),
        result_mode=raw.get("result_mode", LEGACY_OBSERVATION_MODE),
        prompt_snapshot=prompt_snapshot,
        root_session_id=root_session_id or correlation_id_or_digest(
            session_id, prefix="legacy_root"
        ),
        agent_session_id=agent_session_id or correlation_id_or_digest(
            session_id, prefix="legacy_agent"
        ),
    )


def _chunk_from_json(raw: dict[str, Any], payload: bytes) -> TimelineMediaChunk:
    return TimelineMediaChunk(
        session_id=_string(raw, "session_id"),
        identity_epoch=_integer(raw, "identity_epoch"),
        stream_epoch=_integer(raw, "stream_epoch"),
        sequence=_integer(raw, "sequence"),
        kind=MediaKind(_string(raw, "kind")),
        start_ms=_integer(raw, "start_ms"),
        end_ms=_integer(raw, "end_ms"),
        encoding=_string(raw, "encoding"),
        payload=payload,
        observer_epoch=_integer(raw, "observer_epoch"),
    )


def _discontinuity_from_json(raw: dict[str, Any]) -> TimelineDiscontinuity:
    return TimelineDiscontinuity(
        session_id=_string(raw, "session_id"),
        identity_epoch=_integer(raw, "identity_epoch"),
        old_stream_epoch=_integer(raw, "old_stream_epoch"),
        new_stream_epoch=_integer(raw, "new_stream_epoch"),
        reason=_string(raw, "reason"),
        observer_epoch=_integer(raw, "observer_epoch"),
    )


def _event_to_json(event: ObservationEvent) -> dict[str, Any]:
    return {
        "type": "observation",
        "contract_version": event.contract_version,
        "observation_id": event.observation_id,
        "session_id": event.session_id,
        "identity_epoch": event.identity_epoch,
        "observer_epoch": event.observer_epoch,
        "stream_epoch": event.stream_epoch,
        "event_type": event.event_type,
        "summary": event.summary,
        "evidence_start_ms": event.evidence_start_ms,
        "evidence_end_ms": event.evidence_end_ms,
        "model_id": event.model_id,
        "model_version": event.model_version,
        "evidence_mode": event.evidence_mode,
        "audio_status": event.audio_status,
    }


def _object(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    return raw


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value
