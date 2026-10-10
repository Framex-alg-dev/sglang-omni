"""Multiplex model stages over one session WebSocket and one media registry."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import websockets
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.inference_gateway.speech_synthesis import (
    GatewaySpeechConfig,
    GatewaySpeechError,
    GatewaySpeechSynthesizer,
)
from sglang_omni.serve.realtime.embedded_tts import (
    EmbeddedTTSConfig,
    EmbeddedTTSConnection,
    EmbeddedTTSError,
)


logger = logging.getLogger(__name__)
_STAGES = frozenset(
    {"classifier", "brain", "reply", "body", "expression", "performance"}
)
_EVIDENCE_ROLES = frozenset({"user_audio", "user_camera", "avatar_state"})


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
    reply_streaming_speech: EmbeddedTTSConfig | None = None
    request_timeout_seconds: float = 60.0
    receive_idle_timeout_seconds: float = 30.0
    max_media_items: int = 64
    max_media_bytes: int = 128 * 1024 * 1024
    max_item_bytes: int = 32 * 1024 * 1024
    max_session_speculative_audio_bytes: int = 64 * 1024 * 1024
    adaptive_plain_reply_speech: bool = False
    plain_reply_segment_max_chars: int = 120
    plain_reply_segment_max_delay_ms: float = 160.0

    def __post_init__(self) -> None:
        if set(self.stages) != _STAGES:
            raise ValueError("gateway requires exactly the six model stages")
        if self.request_timeout_seconds <= 0 or self.receive_idle_timeout_seconds <= 0:
            raise ValueError("gateway timeouts must be positive")
        if min(
            self.max_media_items,
            self.max_media_bytes,
            self.max_item_bytes,
            self.max_session_speculative_audio_bytes,
            self.plain_reply_segment_max_chars,
        ) <= 0:
            raise ValueError("gateway media limits must be positive")
        if type(self.adaptive_plain_reply_speech) is not bool:
            raise TypeError("adaptive_plain_reply_speech must be boolean")
        if not 120.0 <= self.plain_reply_segment_max_delay_ms <= 200.0:
            raise ValueError(
                "plain_reply_segment_max_delay_ms must be between 120 and 200"
            )


@dataclass(frozen=True)
class _Media:
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


@dataclass
class _SpeechSpeculation:
    request_id: str
    source_request_id: str
    voice: str
    instruction: str
    generation_id: str
    output_epoch: int
    text_mode: str
    synthesis_mode: str = "whole_text"
    text: str = ""
    text_hash: str = ""
    text_parts: list[str] = field(default_factory=list)
    text_queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    instruction_event: asyncio.Event = field(default_factory=asyncio.Event)
    text_completed: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: list[bytes] = field(default_factory=list)
    buffered_bytes: int = 0
    next_seq: int = 0
    committed: bool = False
    commit_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _validate_evidence_role(item: _Media) -> None:
    allowed = {
        "audio": {"user_audio"},
        "video": {"user_camera"},
        "image": {"user_camera", "avatar_state"},
    }
    if item.evidence_role not in allowed[item.kind]:
        raise ValueError("gateway media kind and evidence_role are inconsistent")
    if item.evidence_role == "avatar_state" and item.encoding.strip().lower() not in {
        "jpeg",
        "jpg",
        "image/jpeg",
    }:
        raise ValueError("gateway avatar_state evidence must be a JPEG image")


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
        self.streaming_reply_speech: EmbeddedTTSConnection | None = None
        self.speech_speculations: dict[str, _SpeechSpeculation] = {}
        self.reserved_request_ids: set[str] = set()
        self.speculative_buffered_bytes = 0
        self.segment_flush_tasks: dict[str, asyncio.Task[None]] = {}
        self.stage_available = {stage: True for stage in _STAGES}
        self.availability_epoch = 0
        self.availability_probes: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        await self.websocket.accept()
        first = await self._receive_json()
        contract_version = first.get("contract_version")
        if first.get("type") != "session.open" or contract_version != 4:
            raise ValueError("first gateway message must be session.open v4")
        self.session_id = _string(first, "session_id")
        if self.config.reply_speech is not None:
            self.reply_speech = GatewaySpeechSynthesizer(
                self.config.reply_speech,
            )
        if self.config.reply_streaming_speech is not None:
            self.streaming_reply_speech = EmbeddedTTSConnection(
                self.config.reply_streaming_speech,
                session_id=self.session_id,
            )
        ready = {
            "type": "session.ready",
            "contract_version": contract_version,
            "session_id": self.session_id,
        }
        ready.update(
            availability_epoch=self.availability_epoch,
            stages=dict(sorted(self.stage_available.items())),
        )
        await self.send(ready)
        while True:
            message = await self._receive_json()
            message_type = message.get("type")
            if message_type == "media.put":
                await self._put_media(message)
            elif message_type == "stage.request":
                await self._start_stage(message)
            elif message_type == "speech.request":
                await self._start_speech(message)
            elif message_type == "speech.commit":
                await self._commit_speech(message)
            elif message_type == "speech.configure":
                await self._configure_speech(message)
            elif message_type == "speech.cancel":
                await self._cancel_stage(_string(message, "request_id"))
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
        flush_tasks = tuple(self.segment_flush_tasks.values())
        for task in flush_tasks:
            task.cancel()
        if flush_tasks:
            await asyncio.gather(*flush_tasks, return_exceptions=True)
        self.segment_flush_tasks.clear()
        probes = tuple(self.availability_probes.values())
        for task in probes:
            task.cancel()
        if probes:
            await asyncio.gather(*probes, return_exceptions=True)
        self.availability_probes.clear()
        self.speech_speculations.clear()
        self.reserved_request_ids.clear()
        self.media.clear()
        self.media_refcounts.clear()
        self.media_bytes = 0
        if self.reply_speech is not None:
            await self.reply_speech.close()
        if self.streaming_reply_speech is not None:
            await self.streaming_reply_speech.close()

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
            candidate_header = {
                "media_id": media_id,
                "kind": _one_of(message, "kind", {"audio", "image", "video"}),
                "start_ms": _integer(message, "start_ms"),
                "end_ms": _integer(message, "end_ms"),
                "encoding": _string(message, "encoding"),
                "checksum": checksum,
                "payload_bytes": payload_bytes,
                "evidence_role": _one_of(
                    message, "evidence_role", _EVIDENCE_ROLES
                ),
            }
            if existing.header() != candidate_header or existing.payload != raw:
                raise ValueError(
                    "gateway media_id was reused with different media identity"
                )
            await self.send(
                {"type": "media.ack", **existing.header(), "duplicate": True}
            )
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
            evidence_role=_one_of(message, "evidence_role", _EVIDENCE_ROLES),
        )
        if item.start_ms < 0 or item.end_ms <= item.start_ms:
            raise ValueError("gateway media range is invalid")
        _validate_evidence_role(item)
        self.media[media_id] = item
        self.media_refcounts[media_id] = 0
        self.media_bytes += payload_bytes
        await self.send(
            {"type": "media.ack", **item.header(), "duplicate": False}
        )

    async def _start_stage(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        if request_id in self.tasks or request_id in self.reserved_request_ids:
            raise ValueError("gateway request_id is already active")
        stage = _one_of(message, "stage", _STAGES)
        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("gateway stage payload must be an object")
        payload = dict(payload)
        speech_speculation = self._speech_speculation_request(message, stage)
        if speech_speculation is not None and speech_speculation[0] == request_id:
            raise ValueError("speech speculation request_id must differ from reply request_id")
        if stage in {"body", "expression"}:
            declared_channel = payload.get("channel")
            if declared_channel not in (None, stage):
                raise ValueError("action stage and payload channel do not match")
            payload["channel"] = stage
        elif stage == "reply":
            # The OpenAI-compatible reply worker uses this stable, gateway-owned
            # namespace for its private radix cache.  Always overwrite a caller
            # value so two browser sessions cannot opt into the same cache owner.
            payload["session_instance_id"] = self.session_id
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
        if speech_speculation is not None:
            self.reserved_request_ids.add(speech_speculation[0])
            if (
                self.config.adaptive_plain_reply_speech
                and speech_speculation[5]
                in {"plain", "plain_with_user_turn_v1"}
                and self.reply_speech is not None
            ):
                self._open_adaptive_speculation(
                    source_request_id=request_id,
                    metadata=speech_speculation,
                )
            elif self.streaming_reply_speech is not None:
                self._open_streaming_speculation(
                    source_request_id=request_id,
                    metadata=speech_speculation,
                )
        task = asyncio.create_task(
            self._run_stage(
                request_id,
                stage,
                payload,
                tuple(media_refs),
                speech_speculation,
            ),
            name=f"inference-gateway:{self.session_id}:{stage}:{request_id}",
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))
        await self.send(
            {"type": "stage.accepted", "request_id": request_id, "stage": stage}
        )

    def _speech_speculation_request(
        self,
        message: dict[str, Any],
        stage: str,
    ) -> tuple[str, str, str, str, int, str] | None:
        raw = message.get("speech_speculation")
        if raw is None:
            return None
        if stage != "reply" or not isinstance(raw, dict):
            raise ValueError("speech speculation is only valid for reply stages")
        request_id = _string(raw, "request_id")
        voice = _string(raw, "voice")
        instruction = _string(raw, "instruction")
        generation_id = _string(raw, "generation_id")
        output_epoch = int(_string(raw, "output_epoch"))
        if output_epoch < 0:
            raise ValueError("speech speculation output_epoch is invalid")
        text_mode = str(raw.get("text_mode") or "plain")
        if text_mode not in {
            "plain",
            "reply_envelope_v1",
            "plain_with_user_turn_v1",
        }:
            raise ValueError("speech speculation text_mode is unsupported")
        if (
            request_id in self.tasks
            or request_id in self.speech_speculations
            or request_id in self.reserved_request_ids
        ):
            raise ValueError("speech speculation request_id is already active")
        if len(voice) > 256 or len(instruction) > 1_000:
            raise ValueError("speech speculation metadata exceeds its size limit")
        return request_id, voice, instruction, generation_id, output_epoch, text_mode

    def _open_streaming_speculation(
        self,
        *,
        source_request_id: str,
        metadata: tuple[str, str, str, str, int, str],
    ) -> None:
        request_id, voice, instruction, generation_id, output_epoch, text_mode = metadata
        self.reserved_request_ids.discard(request_id)
        state = _SpeechSpeculation(
            request_id=request_id,
            source_request_id=source_request_id,
            voice=voice,
            instruction=instruction,
            generation_id=generation_id,
            output_epoch=output_epoch,
            text_mode=text_mode,
            synthesis_mode="incremental",
        )
        self.speech_speculations[request_id] = state
        task = asyncio.create_task(
            self._run_streaming_speculative_speech(state),
            name=(
                f"inference-gateway:{self.session_id}:"
                f"streaming-speech-speculation:{request_id}"
            ),
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))

    def _open_adaptive_speculation(
        self,
        *,
        source_request_id: str,
        metadata: tuple[str, str, str, str, int, str],
    ) -> None:
        request_id, voice, instruction, generation_id, output_epoch, text_mode = metadata
        self.reserved_request_ids.discard(request_id)
        state = _SpeechSpeculation(
            request_id=request_id,
            source_request_id=source_request_id,
            voice=voice,
            instruction=instruction,
            generation_id=generation_id,
            output_epoch=output_epoch,
            text_mode=text_mode,
            synthesis_mode="adaptive_whole_or_sentence",
        )
        self.speech_speculations[request_id] = state
        task = asyncio.create_task(
            self._run_adaptive_speculative_speech(state),
            name=(
                f"inference-gateway:{self.session_id}:"
                f"adaptive-speech-speculation:{request_id}"
            ),
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))

    def _disable_speech_speculation(self, request_id: str, *, reason: str) -> None:
        """Abandon optional speech work without failing the owning reply stage."""
        self.reserved_request_ids.discard(request_id)
        task = self.tasks.get(request_id)
        if task is not None:
            task.cancel()
        logger.warning(
            "inference gateway disabled reply speech speculation",
            extra={"request_id": request_id, "reason": reason},
        )

    async def _start_speech(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        if request_id in self.tasks or request_id in self.reserved_request_ids:
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
        sent_bytes = 0

        async def audio_sink(chunk: bytes) -> None:
            nonlocal next_seq, sent_bytes
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
            sent_bytes += len(chunk)
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
                    "phase": exc.phase,
                    "audio_bytes": sent_bytes,
                    "chunk_count": next_seq,
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
                    "phase": "provider_callback",
                    "audio_bytes": sent_bytes,
                    "chunk_count": next_seq,
                }
            )

    async def _start_speculative_speech(
        self,
        *,
        source_request_id: str,
        request_id: str,
        text: str,
        voice: str,
        instruction: str,
        generation_id: str,
        output_epoch: int,
        text_mode: str,
    ) -> None:
        text = text.strip()
        if self.reply_speech is None or not text:
            return
        if request_id not in self.reserved_request_ids:
            return
        self.reserved_request_ids.remove(request_id)
        text_hash = _speculative_speech_hash(
            text=text,
            voice=voice,
            instruction=instruction,
            generation_id=generation_id,
            output_epoch=output_epoch,
        )
        state = _SpeechSpeculation(
            request_id=request_id,
            source_request_id=source_request_id,
            voice=voice,
            instruction=instruction,
            generation_id=generation_id,
            output_epoch=output_epoch,
            text_mode=text_mode,
            text=text,
            text_hash=text_hash,
        )
        state.instruction_event.set()
        state.text_completed.set()
        self.speech_speculations[request_id] = state
        task = asyncio.create_task(
            self._run_speculative_speech(state),
            name=f"inference-gateway:{self.session_id}:speech-speculation:{request_id}",
        )
        self.tasks[request_id] = task
        task.add_done_callback(lambda done, key=request_id: self._task_done(key, done))

    async def _run_speculative_speech(self, state: _SpeechSpeculation) -> None:
        assert self.reply_speech is not None

        async def send_chunk(chunk: bytes) -> None:
            await self.send(
                {
                    "type": "speech.audio.delta",
                    "request_id": state.request_id,
                    "seq": state.next_seq,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
            state.next_seq += 1

        async def audio_sink(chunk: bytes) -> None:
            if not chunk:
                return
            async with state.lock:
                if state.committed:
                    await send_chunk(chunk)
                    return
                state.buffered_bytes += len(chunk)
                self.speculative_buffered_bytes += len(chunk)
                if state.buffered_bytes > self.config.reply_speech.max_audio_bytes:
                    raise GatewaySpeechError(
                        "speculative speech exceeded its audio budget",
                        phase="protocol",
                        retryable=False,
                    )
                if (
                    self.speculative_buffered_bytes
                    > self.config.max_session_speculative_audio_bytes
                ):
                    raise GatewaySpeechError(
                        "session speculative speech exceeded its audio budget",
                        phase="protocol",
                        retryable=False,
                    )
                state.chunks.append(chunk)

        try:
            result = await self.reply_speech.synthesize(
                text=state.text,
                voice=state.voice,
                instruction=state.instruction,
                audio_sink=audio_sink,
            )
            await asyncio.wait_for(
                state.commit_event.wait(),
                timeout=self.config.request_timeout_seconds,
            )
            await self.send(
                {
                    "type": "speech.audio.done",
                    "request_id": state.request_id,
                    "seq": state.next_seq - 1,
                    "audio_bytes": result.audio_bytes,
                    "chunk_count": result.chunk_count,
                    "provider_response_id": result.provider_response_id,
                    "speculative": True,
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
        except asyncio.CancelledError:
            await self.send(
                {"type": "speech.cancelled", "request_id": state.request_id}
            )
            raise
        except Exception as exc:
            logger.exception("inference gateway speculative speech failed")
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": state.request_id,
                    "code": "tts_failed",
                    "detail": "speculative speech synthesis failed",
                    "retryable": isinstance(exc, (asyncio.TimeoutError, GatewaySpeechError))
                    and getattr(exc, "retryable", True),
                }
            )
        finally:
            self.speculative_buffered_bytes = max(
                0,
                self.speculative_buffered_bytes - state.buffered_bytes,
            )
            self.speech_speculations.pop(state.request_id, None)

    async def _run_streaming_speculative_speech(
        self,
        state: _SpeechSpeculation,
    ) -> None:
        assert self.streaming_reply_speech is not None

        async def text_chunks():
            while True:
                chunk = await state.text_queue.get()
                if chunk is None:
                    return
                if chunk:
                    yield chunk

        async def instruction() -> str:
            await state.instruction_event.wait()
            return state.instruction

        async def send_chunk(chunk: bytes) -> None:
            await self.send(
                {
                    "type": "speech.audio.delta",
                    "request_id": state.request_id,
                    "seq": state.next_seq,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
            state.next_seq += 1

        async def audio_sink(chunk: bytes) -> None:
            if not chunk:
                return
            async with state.lock:
                if state.committed:
                    await send_chunk(chunk)
                    return
                state.buffered_bytes += len(chunk)
                self.speculative_buffered_bytes += len(chunk)
                if (
                    state.buffered_bytes
                    > self.config.reply_streaming_speech.provisional_audio_max_bytes
                    or self.speculative_buffered_bytes
                    > self.config.max_session_speculative_audio_bytes
                ):
                    raise EmbeddedTTSError(
                        "speculative audio budget exceeded",
                        phase="protocol",
                    )
                state.chunks.append(chunk)

        try:
            result = await self.streaming_reply_speech.synthesize_streaming(
                turn_id=state.request_id,
                text_chunks=text_chunks(),
                audio_sink=audio_sink,
                voice=state.voice,
                instruct=instruction(),
            )
            await asyncio.wait_for(
                state.commit_event.wait(),
                timeout=self.config.request_timeout_seconds,
            )
            await self.send(
                {
                    "type": "speech.audio.done",
                    "request_id": state.request_id,
                    "seq": state.next_seq - 1,
                    "audio_bytes": result.audio_bytes,
                    "chunk_count": result.chunk_count,
                    "provider_response_id": result.provider_response_id,
                    "speculative": True,
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
        except asyncio.CancelledError:
            await self.send(
                {"type": "speech.cancelled", "request_id": state.request_id}
            )
            raise
        except Exception as exc:
            logger.exception("inference gateway streaming speculation failed")
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": state.request_id,
                    "code": "tts_failed",
                    "detail": "streaming speculative speech synthesis failed",
                    "retryable": isinstance(exc, (asyncio.TimeoutError, EmbeddedTTSError)),
                }
            )
        finally:
            self.speculative_buffered_bytes = max(
                0,
                self.speculative_buffered_bytes - state.buffered_bytes,
            )
            self.speech_speculations.pop(state.request_id, None)

    async def _run_adaptive_speculative_speech(
        self,
        state: _SpeechSpeculation,
    ) -> None:
        assert self.reply_speech is not None

        async def send_chunk(chunk: bytes) -> None:
            await self.send(
                {
                    "type": "speech.audio.delta",
                    "request_id": state.request_id,
                    "seq": state.next_seq,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
            state.next_seq += 1

        async def audio_sink(chunk: bytes) -> None:
            if not chunk:
                return
            async with state.lock:
                if state.committed:
                    await send_chunk(chunk)
                    return
                state.buffered_bytes += len(chunk)
                self.speculative_buffered_bytes += len(chunk)
                if state.buffered_bytes > self.config.reply_speech.max_audio_bytes:
                    raise GatewaySpeechError(
                        "speculative speech exceeded its audio budget",
                        phase="protocol",
                        retryable=False,
                    )
                if (
                    self.speculative_buffered_bytes
                    > self.config.max_session_speculative_audio_bytes
                ):
                    raise GatewaySpeechError(
                        "session speculative speech exceeded its audio budget",
                        phase="protocol",
                        retryable=False,
                    )
                state.chunks.append(chunk)

        audio_bytes = 0
        chunk_count = 0
        provider_response_id = ""
        try:
            while True:
                segment = await state.text_queue.get()
                if segment is None:
                    break
                if not segment.strip():
                    continue
                state.instruction_event.set()
                result = await self.reply_speech.synthesize(
                    text=segment,
                    voice=state.voice,
                    instruction=state.instruction,
                    audio_sink=audio_sink,
                )
                audio_bytes += result.audio_bytes
                chunk_count += result.chunk_count
                provider_response_id = result.provider_response_id
            await asyncio.wait_for(
                state.commit_event.wait(),
                timeout=self.config.request_timeout_seconds,
            )
            await self.send(
                {
                    "type": "speech.audio.done",
                    "request_id": state.request_id,
                    "seq": state.next_seq - 1,
                    "audio_bytes": audio_bytes,
                    "chunk_count": chunk_count,
                    "provider_response_id": provider_response_id,
                    "speculative": True,
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
        except asyncio.CancelledError:
            await self.send(
                {"type": "speech.cancelled", "request_id": state.request_id}
            )
            raise
        except Exception as exc:
            logger.exception("inference gateway adaptive speech speculation failed")
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": state.request_id,
                    "code": "tts_failed",
                    "detail": "adaptive speculative speech synthesis failed",
                    "retryable": isinstance(
                        exc,
                        (asyncio.TimeoutError, GatewaySpeechError),
                    )
                    and getattr(exc, "retryable", True),
                }
            )
        finally:
            self.speculative_buffered_bytes = max(
                0,
                self.speculative_buffered_bytes - state.buffered_bytes,
            )
            self.speech_speculations.pop(state.request_id, None)

    async def _configure_speech(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        state = self.speech_speculations.get(request_id)
        if state is None:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_unavailable",
                    "detail": "speculative speech is not available",
                    "retryable": True,
                }
            )
            return
        generation_id = _string(message, "generation_id")
        output_epoch = _integer(message, "output_epoch")
        instruction = _string(message, "instruction")
        if generation_id != state.generation_id or output_epoch != state.output_epoch:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_mismatch",
                    "detail": "speculative speech generation fence does not match",
                    "retryable": True,
                }
            )
            return
        if state.instruction_event.is_set() and instruction != state.instruction:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_mismatch",
                    "detail": "speculative speech instruction changed",
                    "retryable": True,
                }
            )
            return
        state.instruction = instruction
        state.instruction_event.set()
        if state.text_completed.is_set():
            state.text_hash = _speculative_speech_hash(
                text=state.text,
                voice=state.voice,
                instruction=state.instruction,
                generation_id=state.generation_id,
                output_epoch=state.output_epoch,
            )
        await self.send(
            {
                "type": "speech.configured",
                "request_id": request_id,
                "generation_id": generation_id,
                "output_epoch": output_epoch,
            }
        )

    async def _commit_speech(self, message: dict[str, Any]) -> None:
        request_id = _string(message, "request_id")
        state = self.speech_speculations.get(request_id)
        if state is None:
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_unavailable",
                    "detail": "speculative speech is not available",
                    "retryable": True,
                }
            )
            return
        text_hash = _string(message, "text_hash")
        voice = _string(message, "voice")
        instruction = _string(message, "instruction")
        generation_id = _string(message, "generation_id")
        output_epoch = _integer(message, "output_epoch")
        if not state.text_completed.is_set():
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_unavailable",
                    "detail": "speculative reply text is not complete",
                    "retryable": True,
                }
            )
            return
        if (
            not hmac.compare_digest(text_hash, state.text_hash)
            or voice != state.voice
            or instruction != state.instruction
            or generation_id != state.generation_id
            or output_epoch != state.output_epoch
        ):
            await self.send(
                {
                    "type": "speech.error",
                    "request_id": request_id,
                    "code": "speculation_mismatch",
                    "detail": "speculative speech metadata does not match",
                    "retryable": True,
                }
            )
            task = self.tasks.get(request_id)
            if task is not None:
                task.cancel()
            return
        await self.send(
            {
                "type": "speech.accepted",
                "request_id": request_id,
                "speculative": True,
            }
        )
        async with state.lock:
            state.committed = True
            chunks = tuple(state.chunks)
            state.chunks.clear()
            self.speculative_buffered_bytes = max(
                0,
                self.speculative_buffered_bytes - state.buffered_bytes,
            )
            state.buffered_bytes = 0
            for chunk in chunks:
                await self.send(
                    {
                        "type": "speech.audio.delta",
                        "request_id": request_id,
                        "seq": state.next_seq,
                        "delta": base64.b64encode(chunk).decode("ascii"),
                        "audio": {
                            "format": "pcm16le",
                            "sample_rate_hz": 24_000,
                            "channels": 1,
                        },
                    }
                )
                state.next_seq += 1
        state.commit_event.set()

    def _task_done(self, request_id: str, task: asyncio.Task[None]) -> None:
        self.tasks.pop(request_id, None)
        state = self.speech_speculations.pop(request_id, None)
        if state is not None:
            self.speculative_buffered_bytes = max(
                0,
                self.speculative_buffered_bytes - state.buffered_bytes,
            )
        if not task.cancelled():
            task.exception()

    async def _cancel_stage(self, request_id: str) -> None:
        task = self.tasks.get(request_id)
        if task is None:
            self.reserved_request_ids.discard(request_id)
            await self.send({"type": "request.cancelled", "request_id": request_id})
            return
        task.cancel()

    async def _run_stage(
        self,
        request_id: str,
        stage: str,
        payload: dict[str, Any],
        media_refs: tuple[str, ...],
        speech_speculation: tuple[str, str, str, str, int, str] | None,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._forward_stage(
                    request_id,
                    stage,
                    payload,
                    media_refs,
                    speech_speculation,
                ),
                timeout=self.config.request_timeout_seconds,
            )
            await self._set_stage_availability(stage, True, "request_succeeded")
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
            code, retryable = _classify_stage_failure(exc)
            if code == "provider_unavailable":
                await self._set_stage_availability(stage, False, type(exc).__name__)
            await self.send(
                {
                    "type": "stage.error",
                    "request_id": request_id,
                    "code": code,
                    "detail": str(exc),
                    "retryable": retryable,
                    "availability_epoch": self.availability_epoch,
                }
            )
        finally:
            flush_task = self.segment_flush_tasks.pop(request_id, None)
            if flush_task is not None:
                flush_task.cancel()
                await asyncio.gather(flush_task, return_exceptions=True)
            if speech_speculation is not None:
                self.reserved_request_ids.discard(speech_speculation[0])
                state = self.speech_speculations.get(speech_speculation[0])
                if state is not None and not state.text_completed.is_set():
                    speculation_task = self.tasks.get(state.request_id)
                    if speculation_task is not None:
                        speculation_task.cancel()
            for media_id in media_refs:
                if media_id in self.media_refcounts:
                    self.media_refcounts[media_id] = max(
                        0, self.media_refcounts[media_id] - 1
                    )

    async def _set_stage_availability(
        self,
        stage: str,
        available: bool,
        reason: str,
    ) -> None:
        if self.stage_available.get(stage) is available:
            return
        self.stage_available[stage] = available
        self.availability_epoch += 1
        await self.send(
            {
                "type": "stage.availability",
                "stage": stage,
                "available": available,
                "availability_epoch": self.availability_epoch,
                "reason": reason,
            }
        )
        if not available and stage not in self.availability_probes:
            task = asyncio.create_task(
                self._probe_stage_until_available(stage),
                name=f"inference-gateway-stage-probe:{self.session_id}:{stage}",
            )
            self.availability_probes[stage] = task
            task.add_done_callback(
                lambda completed, stage=stage: self.availability_probes.pop(
                    stage, None
                )
            )

    async def _probe_stage_until_available(self, stage: str) -> None:
        upstream = self.config.stages[stage]
        headers = (
            {"Authorization": upstream.authorization}
            if upstream.authorization
            else None
        )
        while not self.closed and not self.stage_available.get(stage, False):
            try:
                async with asyncio.timeout(2.0):
                    async with websockets.connect(
                        upstream.url,
                        additional_headers=headers,
                        max_size=16 * 1024 * 1024,
                        proxy=None,
                    ):
                        pass
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(1.0)
                continue
            await self._set_stage_availability(stage, True, "probe_succeeded")

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
        speech_speculation: tuple[str, str, str, str, int, str] | None,
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
                        "contract_version": 2,
                        "request_id": request_id,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            ready = _json_object(await socket.recv())
            if (
                ready.get("type") != "request.ready"
                or ready.get("request_id") != request_id
                or ready.get("contract_version") != 2
            ):
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
                            "evidence_role": item.evidence_role,
                        },
                        separators=(",", ":"),
                    )
                )
                await socket.send(item.payload)
                ack = _json_object(await socket.recv())
                expected_ack = {
                    "type": "input.media.ack",
                    "request_id": request_id,
                    **item.header(),
                }
                if ack != expected_ack:
                    raise RuntimeError("model stage returned invalid media acknowledgement")
            await socket.send(
                json.dumps(
                    {"type": "request.commit", "request_id": request_id},
                    separators=(",", ":"),
                )
            )
            reply_text_parts: list[str] = []
            speech_text = (
                _ReplySpeechTextExtractor(speech_speculation[5])
                if speech_speculation is not None
                else None
            )
            speech_segmenter = (
                _ReplySpeechSegmenter(
                    max_chars=self.config.plain_reply_segment_max_chars
                )
                if speech_speculation is not None
                and self.config.adaptive_plain_reply_speech
                and speech_speculation[5] in {"plain", "plain_with_user_turn_v1"}
                and self.reply_speech is not None
                else None
            )

            async def queue_spoken_delta(speculation_id: str, delta: str) -> None:
                state = self.speech_speculations.get(speculation_id)
                if state is None:
                    return
                state.text_parts.append(delta)
                segments = (
                    speech_segmenter.feed(delta)
                    if speech_segmenter is not None
                    else ((delta,) if delta else ())
                )
                for segment in segments:
                    await state.text_queue.put(segment)
                flush_task = self.segment_flush_tasks.get(request_id)
                if speech_segmenter is None or not speech_segmenter.has_pending:
                    if flush_task is not None:
                        flush_task.cancel()
                        self.segment_flush_tasks.pop(request_id, None)
                        await asyncio.gather(flush_task, return_exceptions=True)
                    return
                if flush_task is None or flush_task.done():
                    async def flush_after_deadline() -> None:
                        await asyncio.sleep(
                            self.config.plain_reply_segment_max_delay_ms / 1000.0
                        )
                        segment = speech_segmenter.flush()
                        current = self.speech_speculations.get(speculation_id)
                        if segment and current is not None:
                            await current.text_queue.put(segment)

                    self.segment_flush_tasks[request_id] = asyncio.create_task(
                        flush_after_deadline(),
                        name=f"reply-speech-segment-flush:{request_id}",
                    )

            while True:
                event = _json_object(await socket.recv())
                event_type = event.get("type")
                if event_type == "response.delta":
                    if speech_speculation is not None:
                        delta = _reply_delta_text(event.get("response"))
                        if delta and speech_text is not None:
                            try:
                                spoken_deltas = speech_text.feed(delta)
                            except ValueError:
                                self._disable_speech_speculation(
                                    speech_speculation[0],
                                    reason="reply_text_parse_failed",
                                )
                                speech_text = None
                                speech_speculation = None
                            else:
                                for spoken_delta in spoken_deltas:
                                    reply_text_parts.append(spoken_delta)
                                    await queue_spoken_delta(
                                        speech_speculation[0],
                                        spoken_delta,
                                    )
                    await self.send(
                        {
                            "type": "stage.delta",
                            "request_id": request_id,
                            "response": event.get("response", {}),
                        }
                    )
                    continue
                if event_type == "response.completed":
                    if speech_speculation is not None:
                        (
                            speculation_id,
                            voice,
                            instruction,
                            generation_id,
                            output_epoch,
                            text_mode,
                        ) = speech_speculation
                        if speech_text is not None:
                            try:
                                spoken_deltas = speech_text.finish()
                            except ValueError:
                                self._disable_speech_speculation(
                                    speculation_id,
                                    reason="reply_text_parse_failed",
                                )
                                speech_text = None
                                speech_speculation = None
                            else:
                                for spoken_delta in spoken_deltas:
                                    reply_text_parts.append(spoken_delta)
                                    await queue_spoken_delta(
                                        speculation_id,
                                        spoken_delta,
                                    )
                        if speech_speculation is None:
                            await self.send(
                                {
                                    "type": "stage.completed",
                                    "request_id": request_id,
                                    "response": event.get("response", {}),
                                }
                            )
                            return
                        state = self.speech_speculations.get(speculation_id)
                        if state is not None:
                            if speech_segmenter is not None:
                                flush_task = self.segment_flush_tasks.pop(
                                    request_id, None
                                )
                                if flush_task is not None:
                                    flush_task.cancel()
                                    await asyncio.gather(
                                        flush_task, return_exceptions=True
                                    )
                                for segment in speech_segmenter.finish():
                                    await state.text_queue.put(segment)
                            state.text = "".join(state.text_parts).strip()
                            state.text_hash = _speculative_speech_hash(
                                text=state.text,
                                voice=state.voice,
                                instruction=state.instruction,
                                generation_id=state.generation_id,
                                output_epoch=state.output_epoch,
                            )
                            await state.text_queue.put(None)
                            state.text_completed.set()
                        else:
                            await self._start_speculative_speech(
                                source_request_id=request_id,
                                request_id=speculation_id,
                                text="".join(reply_text_parts),
                                voice=voice,
                                instruction=instruction,
                                generation_id=generation_id,
                                output_epoch=output_epoch,
                                text_mode=text_mode,
                            )
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
    app = FastAPI(title="sglang-omni-inference-gateway", version="4")

    @app.get("/live")
    async def live() -> dict[str, object]:
        return {"ok": True, "contract": "inference-session-v4"}

    @app.get("/ready")
    async def ready() -> dict[str, object]:
        return {
            "ok": True,
            "contract": "inference-session-v4",
            "stages": {stage: True for stage in sorted(config.stages)},
            "reply_speech": config.reply_speech is not None,
            "reply_streaming_speech": config.reply_streaming_speech is not None,
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        return await ready()

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


def _classify_stage_failure(exc: Exception) -> tuple[str, bool]:
    if isinstance(
        exc,
        (
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
            websockets.exceptions.ConnectionClosed,
        ),
    ):
        return "provider_unavailable", True
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "upstream_protocol_error", False
    return "stage_execution_failed", False


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


def _reply_delta_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


class _ReplySpeechTextExtractor:
    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._buffer = ""
        self._started = mode in {"plain", "plain_with_user_turn_v1"}
        self._stopped = False

    def feed(self, delta: str) -> tuple[str, ...]:
        if self._mode == "plain_with_user_turn_v1":
            return self._feed_visible_before_metadata(delta)
        if self._started:
            return (delta,) if delta else ()
        self._buffer += delta
        marker = "<<TEXT>>"
        marker_index = self._buffer.find(marker)
        if marker_index >= 0:
            self._started = True
            text = self._buffer[marker_index + len(marker):].lstrip("\r\n")
            self._buffer = ""
            return (text,) if text else ()
        if "\n" not in self._buffer:
            return ()
        _plan, remainder = self._buffer.split("\n", 1)
        normalized = remainder.lstrip("\r\n")
        if len(normalized) < len(marker):
            return ()
        if not normalized.startswith(marker):
            raise ValueError("reply speech envelope marker is invalid")
        self._started = True
        self._buffer = ""
        text = normalized[len(marker):].lstrip("\r\n")
        return (text,) if text else ()

    def finish(self) -> tuple[str, ...]:
        if self._mode == "reply_envelope_v1" and not self._started:
            raise ValueError("reply speech envelope did not contain a text marker")
        if self._mode == "plain_with_user_turn_v1" and self._stopped:
            return ()
        if self._started and self._buffer:
            value = self._buffer
            self._buffer = ""
            return (value,)
        return ()

    def _feed_visible_before_metadata(self, delta: str) -> tuple[str, ...]:
        if self._stopped or not delta:
            return ()
        marker = "<<USER_TURN_TEXT>>"
        self._buffer += delta
        marker_index = self._buffer.find(marker)
        if marker_index >= 0:
            visible = self._buffer[:marker_index].rstrip("\r\n")
            self._buffer = ""
            self._stopped = True
            return (visible,) if visible else ()
        retained = _marker_prefix_suffix_length(self._buffer, marker)
        emit_length = len(self._buffer) - retained
        while emit_length > 0 and self._buffer[emit_length - 1].isspace():
            emit_length -= 1
        if emit_length <= 0:
            return ()
        visible = self._buffer[:emit_length]
        self._buffer = self._buffer[emit_length:]
        return (visible,)


def _marker_prefix_suffix_length(value: str, marker: str) -> int:
    maximum = min(len(value), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if value.endswith(marker[:length]):
            return length
    return 0


class _ReplySpeechSegmenter:
    """Release natural sentences while bounding punctuation-free buffering."""

    _BOUNDARY = re.compile(r".*?(?:[。！？!?；;]+|\n+)", re.DOTALL)

    def __init__(self, *, max_chars: int) -> None:
        self._max_chars = max_chars
        self._buffer = ""

    def feed(self, delta: str) -> tuple[str, ...]:
        self._buffer += delta
        segments: list[str] = []
        while match := self._BOUNDARY.match(self._buffer):
            segment = match.group(0)
            self._buffer = self._buffer[len(segment):]
            if segment.strip():
                segments.append(segment)
        if len(self._buffer) > self._max_chars:
            split_at = max(
                self._buffer.rfind(mark, 0, self._max_chars + 1)
                for mark in ("，", ",", "：", ":", " ")
            )
            split_at = split_at + 1 if split_at >= 0 else self._max_chars
            segment = self._buffer[:split_at]
            self._buffer = self._buffer[split_at:]
            if segment.strip():
                segments.append(segment)
        return tuple(segments)

    @property
    def has_pending(self) -> bool:
        return bool(self._buffer.strip())

    def flush(self) -> str | None:
        if not self._buffer.strip():
            self._buffer = ""
            return None
        segment = self._buffer
        self._buffer = ""
        return segment

    def finish(self) -> tuple[str, ...]:
        if not self._buffer:
            return ()
        segment = self._buffer
        self._buffer = ""
        return (segment,) if segment.strip() else ()


def _speculative_speech_hash(
    *,
    text: str,
    voice: str,
    instruction: str,
    generation_id: str,
    output_epoch: int,
) -> str:
    text = text.strip()
    text_hash = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "text_hash": text_hash,
            "voice": voice,
            "instruction": instruction,
            "generation_id": generation_id,
            "output_epoch": output_epoch,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
