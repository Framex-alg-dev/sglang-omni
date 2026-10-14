"""Private authenticated HTTP service for the model-1 turn router."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import msgpack
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.websockets import WebSocketDisconnect

from sglang_omni.profiler.decision_timing import emit_decision_timing
from sglang_omni.serve.streaming_request import (
    receive_streamed_request,
    run_until_websocket_disconnect,
)

from .contracts import (
    CONTRACT_VERSION,
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from .pipeline import InvalidRouteOutput, TaskClassificationPipeline


logger = logging.getLogger(__name__)


def create_task_classification_app(
    pipeline: TaskClassificationPipeline,
    *,
    token: str,
    max_body_bytes: int = 128 * 1024 * 1024,
) -> FastAPI:
    if not token:
        raise ValueError("task-classification service token is required")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            await pipeline.prewarm()
        except Exception:
            # Warmup is an optimization. Keep the service available and let the
            # first live request populate the cache if the model was not ready.
            logger.warning("turn-router prefix prewarm failed", exc_info=True)
        yield

    app = FastAPI(
        title="sglang-omni-turn-router",
        version="6",
        lifespan=lifespan,
    )

    def authenticate(request: Request) -> None:
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {token}"
        ):
            raise HTTPException(status_code=404, detail="not found")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        authenticate(request)
        return JSONResponse(
            {
                "ok": True,
                "contract": f"turn-router.v{CONTRACT_VERSION}",
                "contract_version": CONTRACT_VERSION,
            }
        )

    @app.post("/v1/task-classification")
    async def classify(request: Request) -> JSONResponse:
        authenticate(request)
        body = await _read_envelope(request, max_body_bytes=max_body_bytes)
        try:
            payload = await _classify_payload(
                pipeline,
                body,
                transport="http",
            )
            return JSONResponse(
                payload,
                background=BackgroundTask(
                    _emit_response_sent,
                    payload,
                    "http",
                ),
            )
        except InvalidRouteOutput as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.websocket("/v1/task-classification/realtime")
    async def classify_realtime(websocket: WebSocket) -> None:
        try:
            streamed = await receive_streamed_request(
                websocket,
                expected_authorization=f"Bearer {token}",
                max_media_items=16,
                max_total_bytes=max_body_bytes,
            )
            if streamed is None:
                return
            body = dict(streamed.payload)
            body["media"] = [
                {
                    "media_id": item.media_id,
                    "kind": item.kind,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "encoding": item.encoding,
                    "checksum": item.checksum,
                    "payload": item.payload,
                    "evidence_role": item.evidence_role,
                }
                for item in streamed.media
            ]
            result = await run_until_websocket_disconnect(
                websocket,
                _classify_payload(pipeline, body, transport="websocket"),
            )
            await websocket.send_json(
                {
                    "type": "response.completed",
                    "request_id": streamed.request_id,
                    "response": result,
                }
            )
            _emit_response_sent(result, "websocket")
        except WebSocketDisconnect:
            return
        except (InvalidRouteOutput, ValueError) as exc:
            await websocket.send_json(
                {"type": "error", "code": "invalid_request", "detail": str(exc)}
            )
            await websocket.close(code=4400)

    return app


async def _classify_payload(
    pipeline: TaskClassificationPipeline,
    body: dict[str, Any],
    *,
    transport: str = "internal",
) -> dict[str, Any]:
    classification_request = _request_from_json(body)
    metadata = _timing_metadata(classification_request)
    started = time.perf_counter()
    emit_decision_timing(
        "decision_request_received",
        request_id=classification_request.request_id,
        metadata=metadata,
        component="turn_router",
        transport=transport,
        input_mode="audio" if classification_request.media else "text",
    )
    try:
        result = await pipeline.classify(classification_request)
    except asyncio.CancelledError:
        emit_decision_timing(
            "decision_response_completed",
            request_id=classification_request.request_id,
            metadata=metadata,
            component="turn_router",
            status="cancelled",
            transport=transport,
            server_total_ms=round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
        )
        raise
    except Exception as exc:
        emit_decision_timing(
            "decision_response_completed",
            request_id=classification_request.request_id,
            metadata=metadata,
            component="turn_router",
            status="error",
            error_type=type(exc).__name__,
            transport=transport,
            server_total_ms=round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
        )
        raise
    response = {
        "contract_version": result.contract_version,
        "request_id": result.request_id,
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "identity_epoch": result.identity_epoch,
        "input_revision": result.input_revision,
        "route_token": result.route_token.value,
        "route": result.route.value,
        "route_code": result.route_code,
        "needs_mcp": result.needs_mcp,
        "needs_multimodal": result.needs_multimodal,
        "output_directive": result.output_directive.value,
        "task_directive": result.task_directive.value,
        "media_directive": result.media_directive.value,
        "response_locale": result.response_locale,
        "request_cue": (
            {
                "verb": result.request_cue.verb,
                "object": result.request_cue.object,
                "language": result.request_cue.language,
            }
            if result.request_cue is not None
            else None
        ),
        "model_id": result.model_id,
        "model_version": result.model_version,
    }
    completed_metadata = {
        **metadata,
        "model": result.model_id,
        "model_version": result.model_version,
    }
    emit_decision_timing(
        "decision_response_completed",
        request_id=classification_request.request_id,
        metadata=completed_metadata,
        component="turn_router",
        status="success",
        transport=transport,
        route_code=result.route_code,
        server_total_ms=round(
            (time.perf_counter() - started) * 1000.0,
            3,
        ),
    )
    return response


def _timing_metadata(request: TaskClassificationRequest) -> dict[str, Any]:
    return {
        "task": "turn_router",
        "trace_id": request.request_id,
        "operation_id": request.request_id,
        "classification_id": request.request_id,
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "logical_request_id": request.turn_id,
        "identity_epoch": request.identity_epoch,
        "input_revision": request.input_revision,
        "audio_count": len(request.media),
        "media_count": len(request.media),
        "stage": "router",
    }


def _emit_response_sent(response: dict[str, Any], transport: str) -> None:
    request_id = response.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return
    metadata = {
        "task": "turn_router",
        "trace_id": request_id,
        "operation_id": request_id,
        "classification_id": request_id,
        "session_id": response.get("session_id"),
        "turn_id": response.get("turn_id"),
        "logical_request_id": response.get("turn_id"),
        "identity_epoch": response.get("identity_epoch"),
        "input_revision": response.get("input_revision"),
        "model": response.get("model_id"),
        "model_version": response.get("model_version"),
        "stage": "router",
    }
    emit_decision_timing(
        "decision_response_sent",
        request_id=request_id,
        metadata=metadata,
        component="turn_router",
        transport=transport,
        response_chunk_count=1,
    )


def create_decision_services_app(
    *,
    classifier_app: FastAPI,
    brain_app: FastAPI,
) -> FastAPI:
    """Expose both task contracts while classifier and Brain share one runtime."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Starlette does not run mounted sub-application lifespans. Enter the
        # classifier lifespan explicitly so its immutable route prefix is warm
        # before the combined decision service accepts traffic.
        async with classifier_app.router.lifespan_context(classifier_app):
            yield

    app = FastAPI(
        title="sglang-omni-decision-services",
        version="1",
        lifespan=lifespan,
    )
    app.mount("/classifier", classifier_app)
    app.mount("/brain", brain_app)
    return app


async def _read_envelope(
    request: Request, *, max_body_bytes: int
) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_body_bytes:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
    body = await request.body()
    if len(body) > max_body_bytes:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = msgpack.unpackb(body, raw=False, strict_map_key=False)
    except (ValueError, msgpack.ExtraData) as exc:
        raise HTTPException(status_code=400, detail="invalid msgpack body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    return payload


def _request_from_json(raw: dict[str, Any]) -> TaskClassificationRequest:
    required_fields = {
        "contract_version",
        "request_id",
        "session_id",
        "turn_id",
        "identity_epoch",
        "input_revision",
        "text",
        "media",
        "router_history",
        "brain1_capabilities",
        "brain2_capabilities",
        "has_active_agent",
        "pending_confirmation",
        "follow_up_required",
    }
    if set(raw) != required_fields:
        raise ValueError("turn-router request fields are invalid")
    media_raw = raw["media"]
    if not isinstance(media_raw, list):
        raise ValueError("media must be an array")
    history = raw["router_history"]
    if not isinstance(history, list) or not all(
        isinstance(item, dict) for item in history
    ):
        raise ValueError("router_history must be an object array")
    return TaskClassificationRequest(
        request_id=_string(raw, "request_id"),
        session_id=_string(raw, "session_id"),
        turn_id=_string(raw, "turn_id"),
        identity_epoch=_integer(raw, "identity_epoch"),
        input_revision=_integer(raw, "input_revision"),
        text=_optional_string(raw, "text"),
        media=tuple(_media_ref(item) for item in media_raw),
        router_history=tuple(dict(item) for item in history),
        brain1_capabilities=_string(raw, "brain1_capabilities"),
        brain2_capabilities=_string(raw, "brain2_capabilities"),
        has_active_agent=_boolean(raw, "has_active_agent"),
        pending_confirmation=_boolean(raw, "pending_confirmation"),
        follow_up_required=_boolean(raw, "follow_up_required"),
        contract_version=_integer(raw, "contract_version"),
    )


def _media_ref(raw: Any) -> ClassificationMediaRef:
    if not isinstance(raw, dict):
        raise ValueError("media item must be an object")
    return ClassificationMediaRef(
        media_id=_string(raw, "media_id"),
        kind=_string(raw, "kind"),
        start_ms=_integer(raw, "start_ms"),
        end_ms=_integer(raw, "end_ms"),
        encoding=_string(raw, "encoding"),
        checksum=_string(raw, "checksum"),
        payload=_bytes(raw, "payload"),
    )


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _integer(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _bytes(raw: dict[str, Any], key: str) -> bytes:
    value = raw.get(key)
    if not isinstance(value, bytes):
        raise ValueError(f"{key} must be bytes")
    return value
