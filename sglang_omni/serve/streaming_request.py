"""Bounded binary-media WebSocket request transport for model services."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import re
import wave
import asyncio
from dataclasses import dataclass
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
) -> StreamedRequest | None:
    """Accept one request, incrementally own its binary media, then commit it."""

    supplied = websocket.headers.get("authorization", "")
    if expected_authorization and not hmac.compare_digest(
        supplied, expected_authorization
    ):
        await websocket.close(code=4404)
        return None
    await websocket.accept()
    first = _object(await websocket.receive_json(), "request.start")
    if first.get("type") != "request.start":
        raise ValueError("first message must be request.start")
    if _integer(first, "contract_version") != 1:
        raise ValueError("unsupported stream request contract_version")
    request_id = _string(first, "request_id")
    payload = _object(first.get("payload"), "request payload")
    await websocket.send_json(
        {"type": "request.ready", "request_id": request_id, "contract_version": 1}
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
            if len(media) >= max_media_items:
                raise ValueError("stream request has too many media items")
            payload_bytes = _integer(message, "payload_bytes")
            if payload_bytes <= 0 or payload_bytes > max_item_bytes:
                raise ValueError("stream media payload_bytes is outside the limit")
            total_bytes += payload_bytes
            if total_bytes > max_total_bytes:
                raise ValueError("stream request media exceeds the total byte limit")
            raw = await websocket.receive_bytes()
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
            )
            if item.media_id in media_ids:
                raise ValueError("stream media_id must be unique")
            if item.start_ms < 0 or item.end_ms <= item.start_ms:
                raise ValueError("stream media range is invalid")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", item.checksum) is None:
                raise ValueError("stream media checksum must be sha256")
            actual_checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
            if not hmac.compare_digest(actual_checksum, item.checksum):
                raise ValueError("stream media checksum does not match payload")
            media.append(item)
            media_ids.add(item.media_id)
            await websocket.send_json(
                {
                    "type": "input.media.ack",
                    "request_id": request_id,
                    "media_id": item.media_id,
                }
            )
            continue
        if message_type == "request.commit":
            if _string(message, "request_id") != request_id:
                raise ValueError("request.commit request_id mismatch")
            return StreamedRequest(request_id, payload, tuple(media))
        raise ValueError(f"unsupported stream request message: {message_type!r}")


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
