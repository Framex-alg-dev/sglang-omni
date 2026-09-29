"""Private authenticated HTTP service for the model-1 turn router."""

from __future__ import annotations

import hmac
from typing import Any

import msgpack
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.streaming_request import (
    receive_streamed_request,
    run_until_websocket_disconnect,
)

from .contracts import (
    DEFAULT_BRAIN1_CAPABILITIES,
    DEFAULT_BRAIN2_CAPABILITIES,
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from .pipeline import InvalidRouteOutput, TaskClassificationPipeline


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
    app = FastAPI(title="sglang-omni-turn-router", version="1")

    def authenticate(request: Request) -> None:
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {token}"
        ):
            raise HTTPException(status_code=404, detail="not found")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        authenticate(request)
        return JSONResponse(
            {"ok": True, "contract": "turn-router.v1", "contract_version": 1}
        )

    @app.post("/v1/task-classification")
    async def classify(request: Request) -> JSONResponse:
        authenticate(request)
        body = await _read_envelope(request, max_body_bytes=max_body_bytes)
        try:
            return JSONResponse(await _classify_payload(pipeline, body))
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
                }
                for item in streamed.media
            ]
            result = await run_until_websocket_disconnect(
                websocket,
                _classify_payload(pipeline, body),
            )
            await websocket.send_json(
                {
                    "type": "response.completed",
                    "request_id": streamed.request_id,
                    "response": result,
                }
            )
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
) -> dict[str, Any]:
    classification_request = _request_from_json(body)
    result = await pipeline.classify(classification_request)
    return {
        "contract_version": result.contract_version,
        "request_id": result.request_id,
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "identity_epoch": result.identity_epoch,
        "input_revision": result.input_revision,
        "route_token": result.route_token.value,
        "route": result.route.value,
        "model_id": result.model_id,
        "model_version": result.model_version,
    }


def create_decision_services_app(
    *,
    classifier_app: FastAPI,
    brain_app: FastAPI,
) -> FastAPI:
    """Expose both task contracts while classifier and Brain share one runtime."""

    app = FastAPI(title="sglang-omni-decision-services", version="1")
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
    media_raw = raw.get("media", [])
    if not isinstance(media_raw, list):
        raise ValueError("media must be an array")
    history = raw.get("router_history", [])
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
        brain1_capabilities=_optional_string(
            raw, "brain1_capabilities"
        ) or DEFAULT_BRAIN1_CAPABILITIES,
        brain2_capabilities=_optional_string(
            raw, "brain2_capabilities"
        ) or DEFAULT_BRAIN2_CAPABILITIES,
        has_active_agent=_boolean(raw, "has_active_agent", default=False),
        pending_confirmation=_boolean(raw, "pending_confirmation", default=False),
        follow_up_required=_boolean(raw, "follow_up_required", default=False),
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


def _boolean(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _bytes(raw: dict[str, Any], key: str) -> bytes:
    value = raw.get(key)
    if not isinstance(value, bytes):
        raise ValueError(f"{key} must be bytes")
    return value
