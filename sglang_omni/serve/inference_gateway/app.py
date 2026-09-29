"""Multiplex model stages over one session WebSocket and one media registry."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import websockets
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.inference_gateway.speech_synthesis import (
    GatewaySpeechConfig,
    GatewaySpeechError,
    GatewaySpeechSynthesizer,
)


logger = logging.getLogger(__name__)
_STAGES = frozenset({"classifier", "brain", "reply", "body", "expression"})


@dataclass(frozen=True)
class UpstreamStage:
    url: str
    authorization: str = ""

    def __post_init__(self) -> None:
        if not self.url.startswith(("ws://", "wss://")):
            raise ValueError("gateway upstream URL must be ws:// or wss://")


@dataclass(frozen=True)
class InferenceGatewayConfig:
    stages: dict[str, UpstreamStage]
    reply_speech: GatewaySpeechConfig | None = None
    request_timeout_seconds: float = 60.0
    receive_idle_timeout_seconds: float = 30.0
    max_media_items: int = 64
    max_media_bytes: int = 128 * 1024 * 1024
    max_item_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if set(self.stages) != _STAGES:
            raise ValueError("gateway requires exactly the five model stages")
        if self.request_timeout_seconds <= 0 or self.receive_idle_timeout_seconds <= 0:
            raise ValueError("gateway timeouts must be positive")
        if min(self.max_media_items, self.max_media_bytes, self.max_item_bytes) <= 0:
            raise ValueError("gateway media limits must be positive")


@dataclass(frozen=True)
class _Media:
    media_id: str
    kind: str
    start_ms: int
    end_ms: int
    encoding: str
    checksum: str
    payload: bytes


class _Session:
    def __init__(self, websocket: WebSocket, config: InferenceGatewayConfig) -> None:
        self.websocket = websocket
        self.config = config
        self.session_id = ""
        self.media: dict[str, _Media] = {}
        self.media_refcounts: dict[str, int] = {}
        self.media_bytes = 0
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.reply_speech: GatewaySpeechSynthesizer | None = None

    async def run(self) -> None:
        await self.websocket.accept()
        first = await self._receive_json()
        if first.get("type") != "session.open" or first.get("contract_version") != 2:
            raise ValueError("first gateway message must be session.open v2")
        self.session_id = _string(first, "session_id")
        if self.config.reply_speech is not None:
            self.reply_speech = GatewaySpeechSynthesizer(
                self.config.reply_speech,
            )
        await self.send(
            {
                "type": "session.ready",
                "contract_version": 2,
                "session_id": self.session_id,
            }
        )
        while True:
            message = await self._receive_json()
            message_type = message.get("type")
            if message_type == "media.put":
                await self._put_media(message)
            elif message_type == "stage.request":
                await self._start_stage(message)
            elif message_type == "speech.request":
                await self._start_speech(message)
            elif message_type in {"stage.cancel", "request.cancel"}:
                await self._cancel_stage(_string(message, "request_id"))
            elif message_type == "session.close":
                return
            else:
                raise ValueError(f"unsupported gateway message: {message_type!r}")

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        tasks = tuple(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.media.clear()
        self.media_refcounts.clear()
        self.media_bytes = 0
        if self.reply_speech is not None:
            await self.reply_speech.close()

    async def send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def _receive_json(self) -> dict[str, Any]:
        value = await asyncio.wait_for(
            self.websocket.receive_json(),
            timeout=self.config.receive_idle_timeout_seconds,
        )
        if not isinstance(value, dict):
            raise ValueError("gateway message must be an object")
        return value

    async def _put_media(self, message: dict[str, Any]) -> None:
        payload_bytes = _integer(message, "payload_bytes")
        if payload_bytes <= 0 or payload_bytes > self.config.max_item_bytes:
            raise ValueError("gateway media item exceeds its byte limit")
        raw = await asyncio.wait_for(
            self.websocket.receive_bytes(),
            timeout=self.config.receive_idle_timeout_seconds,
        )
        if len(raw) != payload_bytes:
            raise ValueError("gateway media length does not match its header")
        media_id = _string(message, "media_id")
        checksum = _string(message, "checksum")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", checksum) is None:
            raise ValueError("gateway media checksum must be sha256")
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual, checksum):
            raise ValueError("gateway media checksum mismatch")
        existing = self.media.get(media_id)
        if existing is not None:
            if existing.checksum != checksum:
                raise ValueError("gateway media_id was reused with different content")
            await self.send({"type": "media.ack", "media_id": media_id})
            return
        await self._evict_for(payload_bytes)
        item = _Media(
            media_id=media_id,
            kind=_one_of(message, "kind", {"audio", "image", "video"}),
            start_ms=_integer(message, "start_ms"),
            end_ms=_integer(message, "end_ms"),
            encoding=_string(message, "encoding"),
            checksum=checksum,
            payload=raw,
        )
        if item.start_ms < 0 or item.end_ms <= item.start_ms:
            raise ValueError("gateway media range is invalid")
        self.media[media_id] = item
        self.media_refcounts[media_id] = 0
        self.media_bytes += payload_bytes
        await self.send({"type": "media.ack", "media_id": media_id})

    async def _start_stage(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        if request_id in self.tasks:
            raise ValueError("gateway request_id is already active")
        stage = _one_of(message, "stage", _STAGES)
        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("gateway stage payload must be an object")
        payload = dict(payload)
        if stage in {"body", "expression"}:
            declared_channel = payload.get("channel")
            if declared_channel not in (None, stage):
                raise ValueError("action stage and payload channel do not match")
            payload["channel"] = stage
        media_refs = message.get("media_refs", [])
        if not isinstance(media_refs, list) or not all(
            isinstance(item, str) and item for item in media_refs
        ):
            raise ValueError("gateway media_refs must be a string array")
        missing = [item for item in media_refs if item not in self.media]
        if missing:
            await self.send(
                {
                    "type": "stage.error",
                    "request_id": request_id,
                    "code": "unknown_media_ref",
                    "detail": ",".join(missing),
                    "retryable": True,
                }
            )
            return
        for media_id in media_refs:
            self.media_refcounts[media_id] += 1
        task = asyncio.create_task(
            self._run_stage(request_id, stage, payload, tuple(media_refs)),
            name=f"inference-gateway:{self.session_id}:{stage}:{request_id}",
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))
        await self.send(
            {"type": "stage.accepted", "request_id": request_id, "stage": stage}
        )

    async def _start_speech(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        if request_id in self.tasks:
            raise ValueError("gateway request_id is already active")
        text = _string(message, "text")
        voice = _string(message, "voice")
        instruction = _string(message, "instruction")
        if len(text) > 16_384 or len(text.encode("utf-8")) > 65_536:
            raise ValueError("speech text exceeds its size limit")
        if len(voice) > 256:
            raise ValueError("speech voice exceeds its size limit")
        if len(instruction) > 1_000:
            raise ValueError("speech instruction exceeds its size limit")
        if self.reply_speech is None:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "tts_unavailable",
                    "detail": "reply TTS is not configured",
                    "retryable": False,
                }
            )
            return
        task = asyncio.create_task(
            self._run_speech(request_id, text, voice, instruction),
            name=f"inference-gateway:{self.session_id}:speech:{request_id}",
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))
        await self.send({"type": "speech.accepted", "request_id": request_id})

    async def _run_speech(
        self,
        request_id: str,
        text: str,
        voice: str,
        instruction: str,
    ) -> None:
        assert self.reply_speech is not None
        next_seq = 0

        async def audio_sink(chunk: bytes) -> None:
            nonlocal next_seq
            if not chunk:
                return
            await self.send(
                {
                    "type": "speech.audio.delta",
                    "request_id": request_id,
                    "seq": next_seq,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
            next_seq += 1

        try:
            result = await self.reply_speech.synthesize(
                text=text,
                voice=voice,
                instruction=instruction,
                audio_sink=audio_sink,
            )
            await self.send(
                {
                    "type": "speech.audio.done",
                    "request_id": request_id,
                    "seq": next_seq - 1,
                    "audio_bytes": result.audio_bytes,
                    "chunk_count": result.chunk_count,
                    "provider_response_id": result.provider_response_id,
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
        except asyncio.CancelledError:
            await self.send({"type": "speech.cancelled", "request_id": request_id})
            raise
        except GatewaySpeechError as exc:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "tts_failed",
                    "detail": f"speech synthesis failed during {exc.phase}",
                    "retryable": exc.retryable,
                }
            )
        except Exception:
            logger.exception("inference gateway speech rendering failed")
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "tts_failed",
                    "detail": "speech synthesis failed",
                    "retryable": False,
                }
            )

    def _task_done(self, request_id: str, task: asyncio.Task[None]) -> None:
        self.tasks.pop(request_id, None)
        if not task.cancelled():
            task.exception()

    async def _cancel_stage(self, request_id: str) -> None:
        task = self.tasks.get(request_id)
        if task is None:
            await self.send({"type": "request.cancelled", "request_id": request_id})
            return
        task.cancel()

    async def _run_stage(
        self,
        request_id: str,
        stage: str,
        payload: dict[str, Any],
        media_refs: tuple[str, ...],
    ) -> None:
        try:
            await asyncio.wait_for(
                self._forward_stage(request_id, stage, payload, media_refs),
                timeout=self.config.request_timeout_seconds,
            )
        except asyncio.CancelledError:
            await self.send({"type": "stage.cancelled", "request_id": request_id})
            raise
        except asyncio.TimeoutError:
            await self.send(
                {
                    "type": "stage.error",
                    "request_id": request_id,
                    "code": "timeout",
                    "detail": "model stage exceeded its deadline",
                    "retryable": True,
                }
            )
        except Exception as exc:
            logger.exception("inference gateway stage failed", extra={"stage": stage})
            await self.send(
                {
                    "type": "stage.error",
                    "request_id": request_id,
                    "code": "upstream_unavailable",
                    "detail": str(exc),
                    "retryable": True,
                }
            )
        finally:
            for media_id in media_refs:
                if media_id in self.media_refcounts:
                    self.media_refcounts[media_id] = max(
                        0, self.media_refcounts[media_id] - 1
                    )

    async def _evict_for(self, incoming_bytes: int) -> None:
        while (
            len(self.media) >= self.config.max_media_items
            or self.media_bytes + incoming_bytes > self.config.max_media_bytes
        ):
            victim = next(
                (
                    media_id
                    for media_id in self.media
                    if self.media_refcounts.get(media_id, 0) == 0
                ),
                None,
            )
            if victim is None:
                raise ValueError("gateway session media exceeds its active limit")
            removed = self.media.pop(victim)
            self.media_refcounts.pop(victim, None)
            self.media_bytes -= len(removed.payload)
            await self.send({"type": "media.evicted", "media_id": victim})
    async def _forward_stage(
        self,
        request_id: str,
        stage: str,
        payload: dict[str, Any],
        media_refs: tuple[str, ...],
    ) -> None:
        upstream = self.config.stages[stage]
        headers = (
            {"Authorization": upstream.authorization}
            if upstream.authorization
            else None
        )
        async with websockets.connect(
            upstream.url,
            additional_headers=headers,
            max_size=16 * 1024 * 1024,
            proxy=None,
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "request.start",
                        "contract_version": 1,
                        "request_id": request_id,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            ready = _json_object(await socket.recv())
            if ready.get("type") != "request.ready":
                raise RuntimeError("model stage rejected request.start")
            for media_id in media_refs:
                item = self.media[media_id]
                await socket.send(
                    json.dumps(
                        {
                            "type": "input.media",
                            "request_id": request_id,
                            "media_id": item.media_id,
                            "kind": item.kind,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                            "encoding": item.encoding,
                            "checksum": item.checksum,
                            "payload_bytes": len(item.payload),
                        },
                        separators=(",", ":"),
                    )
                )
                await socket.send(item.payload)
                ack = _json_object(await socket.recv())
                if ack.get("type") != "input.media.ack" or ack.get("media_id") != media_id:
                    raise RuntimeError("model stage returned invalid media acknowledgement")
            await socket.send(
                json.dumps(
                    {"type": "request.commit", "request_id": request_id},
                    separators=(",", ":"),
                )
            )
            while True:
                event = _json_object(await socket.recv())
                event_type = event.get("type")
                if event_type == "response.delta":
                    await self.send(
                        {
                            "type": "stage.delta",
                            "request_id": request_id,
                            "response": event.get("response", {}),
                        }
                    )
                    continue
                if event_type == "response.completed":
                    await self.send(
                        {
                            "type": "stage.completed",
                            "request_id": request_id,
                            "response": event.get("response", {}),
                        }
                    )
                    return
                if event_type == "error":
                    await self.send(
                        {
                            "type": "stage.error",
                            "request_id": request_id,
                            "code": event.get("code", "upstream_error"),
                            "detail": event.get("detail", "model stage failed"),
                            "retryable": event.get("code") in {"overloaded", "timeout"},
                        }
                    )
                    return
                raise RuntimeError("model stage returned an unsupported event")


def create_inference_gateway_app(config: InferenceGatewayConfig) -> FastAPI:
    app = FastAPI(title="sglang-omni-inference-gateway", version="1")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "contract": "inference-session-v2",
            "reply_speech": config.reply_speech is not None,
        }

    @app.websocket("/v1/inference-session")
    async def inference_session(websocket: WebSocket) -> None:
        session = _Session(websocket, config)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            await _close_with_error(websocket, "idle_timeout", "gateway receive timeout", 4408)
        except ValueError as exc:
            await _close_with_error(websocket, "invalid_request", str(exc), 4400)
        except Exception:
            logger.exception("inference gateway session failed")
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass
        finally:
            await session.close()

    return app


async def _close_with_error(
    websocket: WebSocket, code: str, detail: str, close_code: int
) -> None:
    try:
        await websocket.send_json({"type": "error", "code": code, "detail": detail})
        await websocket.close(code=close_code)
    except RuntimeError:
        pass


def _json_object(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("model stage event must be an object")
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


def _one_of(value: dict[str, Any], key: str, allowed: set[str] | frozenset[str]) -> str:
    item = _string(value, key)
    if item not in allowed:
        raise ValueError(f"{key} is unsupported")
    return item
