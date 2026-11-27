"""Bounded binary-media WebSocket request transport for model services."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import re
import wave
import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect


T = TypeVar("T")


@dataclass(frozen=True)
class StreamedMedia:
    media_id: str
    kind: str
    start_ms: int
    end_ms: int
    encoding: str
    checksum: str
    payload: bytes
    evidence_role: str

    def header(self) -> dict[str, object]:
        return {
            "media_id": self.media_id,
            "kind": self.kind,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "encoding": self.encoding,
            "checksum": self.checksum,
            "payload_bytes": len(self.payload),
            "evidence_role": self.evidence_role,
        }


@dataclass(frozen=True)
class StreamedRequest:
    request_id: str
    payload: dict[str, Any]
    media: tuple[StreamedMedia, ...]


async def run_until_websocket_disconnect(
    websocket: WebSocket,
    operation: Awaitable[T],
    *,
    abort: Callable[[], Awaitable[Any]] | None = None,
) -> T:
    """Own one model task and abort it when the committed client disconnects."""

    operation_task = asyncio.create_task(operation)
    disconnect_task = asyncio.create_task(websocket.receive())
    try:
        done, _ = await asyncio.wait(
            {operation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            return await operation_task
        operation_task.cancel()
        if abort is not None:
            await abort()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise WebSocketDisconnect
    except asyncio.CancelledError:
        operation_task.cancel()
        disconnect_task.cancel()
        if abort is not None:
            await abort()
        await asyncio.gather(operation_task, disconnect_task, return_exceptions=True)
        raise
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)


async def receive_streamed_request(
    websocket: WebSocket,
    *,
    expected_authorization: str = "",
    max_media_items: int = 64,
    max_item_bytes: int = 32 * 1024 * 1024,
    max_total_bytes: int = 128 * 1024 * 1024,
    allowed_audio_roles: frozenset[str] = frozenset({"user_audio", "assistant_spoken"}),
) -> StreamedRequest | None:
    """Accept one request, incrementally own its binary media, then commit it."""

    if not await _accept_stream(websocket, expected_authorization):
        return None
    request_id, payload = _parse_stream_start(await websocket.receive_json())
    await websocket.send_json(
        {"type": "request.ready", "request_id": request_id, "contract_version": 2}
    )
    media: list[StreamedMedia] = []
    media_ids: set[str] = set()
    total_bytes = 0
    while True:
        message = _object(await websocket.receive_json(), "stream request message")
        message_type = message.get("type")
        if message_type == "input.media":
            if _string(message, "request_id") != request_id:
                raise ValueError("input.media request_id mismatch")
            payload_bytes = _check_media_size(
                message, len(media), total_bytes, max_media_items, max_item_bytes,
                max_total_bytes,
            )
            raw = await websocket.receive_bytes()
            item = _parse_stream_media(message, raw, payload_bytes, allowed_audio_roles)
            if item.media_id in media_ids:
                raise ValueError("stream media_id must be unique")
            total_bytes += payload_bytes
            media.append(item)
            media_ids.add(item.media_id)
            await websocket.send_json(
                {
                    "type": "input.media.ack",
                    "request_id": request_id,
                    **item.header(),
                }
            )
            continue
        if message_type == "request.commit":
            if _string(message, "request_id") != request_id:
                raise ValueError("request.commit request_id mismatch")
            return StreamedRequest(request_id, payload, tuple(media))
        raise ValueError(f"unsupported stream request message: {message_type!r}")


async def _accept_stream(websocket: WebSocket, expected_authorization: str) -> bool:
    supplied = websocket.headers.get("authorization", "")
    if expected_authorization and not hmac.compare_digest(supplied, expected_authorization):
        await websocket.close(code=4404)
        return False
    await websocket.accept()
    return True


def _parse_stream_start(value: Any) -> tuple[str, dict[str, Any]]:
    message = _object(value, "request.start")
    if message.get("type") != "request.start":
        raise ValueError("first message must be request.start")
    if _integer(message, "contract_version") != 2:
        raise ValueError("unsupported stream request contract_version")
    payload = dict(_object(message.get("payload"), "request payload"))
    envelope_context = message.get("log_context")
    if envelope_context is not None:
        if not isinstance(envelope_context, dict):
            raise ValueError("log_context must be an object")
        payload_context = payload.get("log_context")
        if payload_context is not None and not isinstance(payload_context, dict):
            raise ValueError("payload.log_context must be an object")
        payload["log_context"] = {
            **(payload_context or {}),
            **envelope_context,
        }
    return _string(message, "request_id"), payload


def _check_media_size(
    message: dict[str, Any], count: int, total_bytes: int, max_media_items: int,
    max_item_bytes: int, max_total_bytes: int,
) -> int:
    if count >= max_media_items:
        raise ValueError("stream request has too many media items")
    payload_bytes = _integer(message, "payload_bytes")
    if payload_bytes <= 0 or payload_bytes > max_item_bytes:
        raise ValueError("stream media payload_bytes is outside the limit")
    if total_bytes + payload_bytes > max_total_bytes:
        raise ValueError("stream request media exceeds the total byte limit")
    return payload_bytes


def _parse_stream_media(
    message: dict[str, Any], raw: bytes, payload_bytes: int,
    allowed_audio_roles: frozenset[str],
) -> StreamedMedia:
    if len(raw) != payload_bytes:
        raise ValueError("stream media binary length does not match header")
    item = StreamedMedia(
        media_id=_string(message, "media_id"),
        kind=_one_of(message, "kind", {"audio", "image", "video"}),
        start_ms=_integer(message, "start_ms"),
        end_ms=_integer(message, "end_ms"),
        encoding=_string(message, "encoding"),
        checksum=_string(message, "checksum"),
        payload=raw,
        evidence_role=_one_of(
            message, "evidence_role", {"user_camera", "avatar_state"} | set(allowed_audio_roles),
        ),
    )
    if item.start_ms < 0 or item.end_ms <= item.start_ms:
        raise ValueError("stream media range is invalid")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", item.checksum) is None:
        raise ValueError("stream media checksum must be sha256")
    actual_checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_checksum, item.checksum):
        raise ValueError("stream media checksum does not match payload")
    _validate_evidence_role(item, allowed_audio_roles=allowed_audio_roles)
    return item


@dataclass
class _StreamUpload:
    payload: dict[str, Any]
    media: dict[str, StreamedMedia] = field(default_factory=dict)
    total_bytes: int = 0


async def serve_streamed_requests(
    websocket: WebSocket,
    handle: Callable[[StreamedRequest], Awaitable[dict[str, Any]]],
    *,
    expected_authorization: str,
    max_requests: int,
    max_media_items: int,
    max_item_bytes: int,
    max_total_bytes: int,
    allowed_audio_roles: frozenset[str],
    on_response_sent: Callable[
        [StreamedRequest, dict[str, Any], float], None
    ] | None = None,
) -> None:
    """Receive several requests on one socket and dispatch commits concurrently."""
    if not await _accept_stream(websocket, expected_authorization):
        return
    uploads: dict[str, _StreamUpload] = {}
    running: dict[str, asyncio.Task[None]] = {}
    send_lock = asyncio.Lock()

    async def send(event: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(event)

    async def complete(request: StreamedRequest) -> None:
        request_id = request.request_id
        started = asyncio.get_running_loop().time()
        try:
            try:
                response = await handle(request)
                event = {"type": "response.completed", "response": response}
            except (TypeError, ValueError) as exc:
                event = {"type": "error", "code": "invalid_request", "detail": str(exc)}
            except Exception:
                event = {"type": "error", "code": "model_error", "detail": "recall failed"}
            wire_event = {**event, "request_id": request_id}
            await send(wire_event)
            if on_response_sent is not None:
                on_response_sent(
                    request,
                    wire_event,
                    (asyncio.get_running_loop().time() - started) * 1000.0,
                )
        except WebSocketDisconnect:
            pass
        finally:
            running.pop(request_id, None)

    try:
        while True:
            request_id = ""
            kind = None
            try:
                message = _object(await websocket.receive_json(), "stream request message")
                kind = message.get("type")
                supplied_id = message.get("request_id")
                if isinstance(supplied_id, str):
                    request_id = supplied_id
                # Consume the byte frame even when its header is rejected, so the
                # next logical request still starts on a JSON frame.
                raw = await websocket.receive_bytes() if kind == "input.media" else None
                request_id = _string(message, "request_id")
                if kind == "request.start":
                    _, payload = _parse_stream_start(message)
                    if request_id in uploads or request_id in running:
                        raise ValueError("request_id is already active on this connection")
                    if len(uploads) + len(running) >= max_requests:
                        raise ValueError("too many active requests on this connection")
                    uploads[request_id] = _StreamUpload(payload)
                    await send({"type": "request.ready", "request_id": request_id, "contract_version": 2})
                    continue
                upload = uploads.get(request_id)
                if upload is None:
                    raise ValueError("request_id has no pending request.start")
                if kind == "input.media":
                    assert raw is not None
                    size = _check_media_size(
                        message, len(upload.media), upload.total_bytes,
                        max_media_items, max_item_bytes, max_total_bytes,
                    )
                    item = _parse_stream_media(message, raw, size, allowed_audio_roles)
                    if item.media_id in upload.media:
                        raise ValueError("stream media_id must be unique")
                    upload.media[item.media_id] = item
                    upload.total_bytes += size
                    await send({"type": "input.media.ack", "request_id": request_id, **item.header()})
                elif kind == "request.commit":
                    del uploads[request_id]
                    request = StreamedRequest(request_id, upload.payload, tuple(upload.media.values()))
                    running[request_id] = asyncio.create_task(complete(request))
                else:
                    raise ValueError("unsupported stream request message")
            except (KeyError, ValueError) as exc:
                if kind != "request.start":
                    uploads.pop(request_id, None)
                await send({
                    "type": "error", "request_id": request_id,
                    "code": "invalid_request", "detail": str(exc),
                })
    except WebSocketDisconnect:
        pass
    finally:
        tasks = tuple(running.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def inject_openai_media(
    payload: dict[str, Any],
    media: tuple[StreamedMedia, ...],
) -> dict[str, Any]:
    """Attach streamed media to the final user message as OpenAI content parts."""

    if not media:
        return payload
    result = dict(payload)
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("streamed chat request requires messages")
    copied = [dict(item) if isinstance(item, dict) else item for item in messages]
    user_index = next(
        (
            index
            for index in range(len(copied) - 1, -1, -1)
            if isinstance(copied[index], dict)
            and copied[index].get("role") == "user"
        ),
        None,
    )
    if user_index is None:
        raise ValueError("streamed chat request requires a user message")
    user = dict(copied[user_index])
    content = user.get("content", "")
    parts = (
        list(content)
        if isinstance(content, list)
        else [{"type": "text", "text": str(content)}]
    )
    parts.extend(_openai_media_part(item) for item in media)
    user["content"] = parts
    copied[user_index] = user
    result["messages"] = copied
    audio_urls = [media_data_uri(item) for item in media if item.kind == "audio"]
    image_urls = [
        media_data_uri(item) for item in media if item.kind in {"image", "video"}
    ]
    if audio_urls:
        result["audios"] = [*(result.get("audios") or []), *audio_urls]
    if image_urls:
        result["images"] = [*(result.get("images") or []), *image_urls]
    return result


def _validate_evidence_role(
    item: StreamedMedia, *, allowed_audio_roles: frozenset[str] = frozenset({"user_audio", "assistant_spoken"}),
) -> None:
    allowed = {
        "audio": allowed_audio_roles,
        "video": {"user_camera"},
        "image": {"user_camera", "avatar_state"},
    }
    if item.evidence_role not in allowed[item.kind]:
        raise ValueError("stream media kind and evidence_role are inconsistent")
    if item.evidence_role == "avatar_state" and item.encoding.strip().lower() not in {
        "jpeg",
        "jpg",
        "image/jpeg",
    }:
        raise ValueError("avatar_state stream media must be a JPEG image")


def media_data_uri(item: StreamedMedia) -> str:
    return f"data:{_media_mime(item)};base64," + base64.b64encode(
        _media_payload(item)
    ).decode("ascii")


def _openai_media_part(item: StreamedMedia) -> dict[str, object]:
    return {"type": "audio" if item.kind == "audio" else "image"}


def _media_mime(item: StreamedMedia) -> str:
    normalized = item.encoding.strip().lower()
    if item.kind == "audio" and normalized in {
        "pcm_s16le",
        "pcm16",
        "audio/l16",
    }:
        return "audio/wav"
    if "/" in normalized:
        return normalized
    aliases = {
        ("audio", "wav"): "audio/wav",
        ("image", "jpeg"): "image/jpeg",
        ("video", "jpeg"): "image/jpeg",
        ("image", "png"): "image/png",
        ("video", "png"): "image/png",
    }
    try:
        return aliases[(item.kind, normalized)]
    except KeyError as exc:
        raise ValueError(f"unsupported stream media encoding: {item.encoding}") from exc


def _media_payload(item: StreamedMedia) -> bytes:
    normalized = item.encoding.strip().lower()
    if item.kind != "audio" or normalized not in {
        "pcm_s16le",
        "pcm16",
        "audio/l16",
    }:
        return item.payload
    if len(item.payload) % 2:
        raise ValueError("PCM16 stream media must contain complete samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(item.payload)
    return output.getvalue()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise ValueError(f"{key} must be an integer")
    return item


def _one_of(value: dict[str, Any], key: str, allowed: set[str]) -> str:
    item = _string(value, key)
    if item not in allowed:
        raise ValueError(f"{key} is unsupported")
    return item
