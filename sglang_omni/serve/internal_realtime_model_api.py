# SPDX-License-Identifier: Apache-2.0
"""Authenticated loopback transport for the secondary realtime executor."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import aclosing
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from sglang_omni.client.realtime_executor import (
    completion_to_wire,
    frame_wire,
    internal_model_api_enabled,
    internal_model_api_token,
    pack_wire,
    request_from_wire,
    stream_chunk_to_wire,
    unpack_wire,
)
from sglang_omni.client.types import AbortLevel

logger = logging.getLogger(__name__)

_WIRE_MEDIA_TYPE = "application/vnd.sglang-omni.msgpack"
_DEFAULT_MAX_BODY_BYTES = 128 * 1024 * 1024


def register_internal_realtime_model_api(app: FastAPI) -> None:
    """Register private executor routes only when explicitly enabled."""

    if not internal_model_api_enabled():
        return
    token = internal_model_api_token()
    if not token:
        raise ValueError(
            "SGLANG_OMNI_INTERNAL_MODEL_TOKEN is required when "
            "SGLANG_OMNI_INTERNAL_MODEL_API is enabled"
        )
    max_body_bytes = int(
        os.environ.get(
            "SGLANG_OMNI_INTERNAL_MODEL_MAX_BODY_BYTES",
            str(_DEFAULT_MAX_BODY_BYTES),
        )
    )
    if max_body_bytes <= 0:
        raise ValueError(
            "SGLANG_OMNI_INTERNAL_MODEL_MAX_BODY_BYTES must be positive"
        )

    def authenticate(request: Request) -> None:
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {token}"
        if not hmac.compare_digest(supplied, expected):
            # Use 404 to avoid advertising private routes on a misconfigured
            # interface while still requiring a strong deployment token.
            raise HTTPException(status_code=404, detail="not found")

    async def read_envelope(request: Request) -> dict[str, Any]:
        authenticate(request)
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_body_bytes:
                    raise HTTPException(
                        status_code=413, detail="request body too large"
                    )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="invalid content-length"
                ) from exc
        body = await request.body()
        if len(body) > max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        try:
            payload = unpack_wire(body)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="invalid msgpack body"
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a map")
        return payload

    @app.post("/internal/realtime-model/completion")
    async def internal_completion(request: Request) -> Response:
        envelope = await read_envelope(request)
        request_id = str(envelope.get("request_id") or "")
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")
        try:
            model_request = request_from_wire(envelope["request"])
            result = await app.state.client.completion(
                model_request,
                request_id=request_id,
                audio_format=str(envelope.get("audio_format") or "wav"),
            )
            payload = {"kind": "result", "result": completion_to_wire(result)}
        except asyncio.CancelledError:
            await app.state.client.abort(request_id)
            raise
        except Exception as exc:
            logger.exception(
                "Internal realtime completion failed request_id=%s", request_id
            )
            payload = {
                "kind": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        return Response(content=pack_wire(payload), media_type=_WIRE_MEDIA_TYPE)

    @app.post("/internal/realtime-model/completion-stream")
    async def internal_completion_stream(request: Request) -> StreamingResponse:
        envelope = await read_envelope(request)
        request_id = str(envelope.get("request_id") or "")
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")
        try:
            model_request = request_from_wire(envelope["request"])
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid model request") from exc
        audio_format = str(envelope.get("audio_format") or "wav")

        async def body() -> AsyncIterator[bytes]:
            completed = False
            try:
                stream = app.state.client.completion_stream(
                    model_request,
                    request_id=request_id,
                    audio_format=audio_format,
                )
                async with aclosing(stream):
                    async for chunk in stream:
                        yield frame_wire(
                            {"kind": "chunk", "chunk": stream_chunk_to_wire(chunk)}
                        )
                completed = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Internal realtime completion stream failed request_id=%s",
                    request_id,
                )
                yield frame_wire(
                    {
                        "kind": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            finally:
                if not completed:
                    try:
                        await app.state.client.abort(request_id)
                    except Exception:
                        logger.warning(
                            "Internal realtime stream abort failed request_id=%s",
                            request_id,
                            exc_info=True,
                        )

        return StreamingResponse(body(), media_type=_WIRE_MEDIA_TYPE)

    @app.post("/internal/realtime-model/abort")
    async def internal_abort(request: Request) -> Response:
        envelope = await read_envelope(request)
        request_id = str(envelope.get("request_id") or "")
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")
        level = AbortLevel(str(envelope.get("level") or AbortLevel.SOFT.value))
        result = await app.state.client.abort(request_id, level=level)
        return Response(
            content=pack_wire(
                {
                    "success": result.success,
                    "level_applied": result.level_applied.value,
                }
            ),
            media_type=_WIRE_MEDIA_TYPE,
        )

    @app.get("/internal/realtime-model/health")
    async def internal_health(request: Request) -> Response:
        authenticate(request)
        health = app.state.client.health()
        return Response(
            content=pack_wire({"ok": True, "model": health}),
            media_type=_WIRE_MEDIA_TYPE,
        )

    @app.post("/internal/realtime-model/release-session-cache")
    async def internal_release_session_cache(request: Request) -> Response:
        envelope = await read_envelope(request)
        session_instance_id = str(envelope.get("session_instance_id") or "")
        if not session_instance_id:
            raise HTTPException(
                status_code=400, detail="session_instance_id is required"
            )
        result = await app.state.client.release_session_cache(session_instance_id)
        return Response(content=pack_wire(result), media_type=_WIRE_MEDIA_TYPE)
