"""Private WebSocket transport for continuous timeline detection."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from .contracts import (
    CONTRACT_VERSION,
    MediaKind,
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from .session import TimelineDetectionModel, TimelineDetectionSession


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
        session: TimelineDetectionSession | None = None
        worker: asyncio.Task[None] | None = None
        observation_worker: asyncio.Task[None] | None = None
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
                        else:
                            await session.discontinuity(work.event)
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
                    events = await session.next_observations()
                    for event in events:
                        await send_json(_event_to_json(event))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await report_model_failure(exc)
                raise

        try:
            first = await websocket.receive_json()
            start = _start_from_json(_object(first, "session.start"))
            session = TimelineDetectionSession(start, model_factory(start))
            await session.start()
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
            return
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            try:
                await send_json(
                    {"type": "error", "code": "invalid_request", "detail": str(exc)}
                )
                await websocket.close(code=4400)
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception:
            try:
                await websocket.close(code=1011)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if observation_worker is not None:
                observation_worker.cancel()
                await asyncio.gather(observation_worker, return_exceptions=True)
            if worker is not None:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            if session is not None:
                await session.close()

    return app


def _start_from_json(raw: dict[str, Any]) -> TimelineSessionStart:
    if raw.get("type") != "session.start":
        raise ValueError("first message must be session.start")
    return TimelineSessionStart(
        session_id=_string(raw, "session_id"),
        identity_epoch=_integer(raw, "identity_epoch"),
        stream_epoch=_integer(raw, "stream_epoch"),
        audio_format=_string(raw, "audio_format"),
        video_format=_string(raw, "video_format"),
        model_id=_string(raw, "model_id"),
        observer_epoch=_integer(raw, "observer_epoch"),
        next_sequence=_integer(raw, "next_sequence"),
        contract_version=_integer(raw, "contract_version"),
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
