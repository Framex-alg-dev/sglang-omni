"""Portable self-hosted realtime text-in/audio-out TTS WebSocket client.

One instance owns one reusable, serialized conversation connection. Text chunks
arrive in the order in which the caller's async iterator yields them.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect

from .contracts import SpeechAudioSink, SpeechSynthesisRequest, SpeechSynthesisResult


TraceSink = Callable[["SynthesisTrace"], Awaitable[None] | None]


@dataclass(frozen=True)
class RealtimeWsTtsConfig:
    url: str
    timeout_seconds: float = 30.0
    model: str = "cosyvoice3-realtime"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("TTS URL must be an absolute ws:// or wss:// URL")
        if self.timeout_seconds <= 0:
            raise ValueError("TTS timeout_seconds must be positive")


@dataclass
class SynthesisTrace:
    turn_id: str
    provider: str = "self_hosted_realtime_ws"
    request_sequence: int = 0
    adapter_started: float = field(default_factory=time.monotonic)
    connection_ready: float | None = None
    request_started: float | None = None
    input_committed: float | None = None
    first_chunk_received: float | None = None
    completed: float | None = None
    input_text_chunk_sizes_chars: list[int] = field(default_factory=list)
    input_text_chunk_sent_at: list[float] = field(default_factory=list)
    audio_chunk_sizes_bytes: list[int] = field(default_factory=list)
    audio_chunk_arrivals: list[float] = field(default_factory=list)
    failure_phase: str | None = None
    failure_detail: str | None = None

    def elapsed_ms(self, start: str, end: str) -> float | None:
        start_value = getattr(self, start)
        end_value = getattr(self, end)
        if start_value is None or end_value is None:
            return None
        return round((end_value - start_value) * 1000.0, 3)


class RealtimeWsTtsError(RuntimeError):
    def __init__(self, message: str, *, phase: str, error_detail: str,
                 provider_request_id: str | None = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.error_detail = error_detail[:1000]
        self.provider_request_id = provider_request_id


def _session_url(url: str, *, voice: str, session_id: str) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("voice", voice)
    query.setdefault("session_id", session_id)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


class RealtimeWsSpeechSynthesizer:
    """Reuse one WebSocket and serialize all TTS turns on that connection."""

    def __init__(self, config: RealtimeWsTtsConfig, *,
                 connector: Callable[..., Any] = connect,
                 trace_sink: TraceSink | None = None) -> None:
        self._config = config
        self._connector = connector
        self._trace_sink = trace_sink
        self._connection_lock = asyncio.Lock()
        self._connection_context: Any | None = None
        self._websocket: Any | None = None
        self._connection_voice: str | None = None
        self._request_sequence = 0
        self._closed = False

    async def close(self) -> None:
        async with self._connection_lock:
            self._closed = True
            await self._discard_connection()

    async def synthesize(self, request: SpeechSynthesisRequest, *,
                         audio_sink: SpeechAudioSink) -> SpeechSynthesisResult:
        if not request.text.strip():
            raise RealtimeWsTtsError("realtime TTS requires non-empty text",
                                     phase="request", error_detail="empty text")

        async def one_chunk() -> AsyncIterator[str]:
            yield request.text

        return await self.synthesize_streaming(request, text_chunks=one_chunk(), audio_sink=audio_sink)

    async def synthesize_streaming(self, request: SpeechSynthesisRequest, *,
                                   text_chunks: AsyncIterator[str],
                                   audio_sink: SpeechAudioSink) -> SpeechSynthesisResult:
        trace = SynthesisTrace(turn_id=request.turn_id)
        async with self._connection_lock:
            if self._closed:
                raise RealtimeWsTtsError("realtime TTS session is closed",
                                         phase="connection_borrow",
                                         error_detail="synthesizer was closed")
            return await self._synthesize_serialized(request, text_chunks=text_chunks,
                                                     audio_sink=audio_sink, trace=trace)

    async def _synthesize_serialized(self, request: SpeechSynthesisRequest, *,
                                     text_chunks: AsyncIterator[str],
                                     audio_sink: SpeechAudioSink,
                                     trace: SynthesisTrace) -> SpeechSynthesisResult:
        provider_request_id: str | None = None
        phase = "connect"
        sender: asyncio.Task[None] | None = None
        websocket: Any | None = None
        audio_bytes = chunk_count = 0
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                phase = "connection_borrow"
                websocket, session_id = await self._borrow_connection(request)
                provider_request_id = session_id
                self._request_sequence += 1
                trace.request_sequence = self._request_sequence
                trace.connection_ready = time.monotonic()
                sender = asyncio.create_task(self._send_text(websocket, text_chunks, trace),
                                             name=f"realtime-ws-tts-sender:{request.turn_id}")
                phase = "receive_audio"
                while True:
                    event = await self._receive_event(websocket)
                    event_type = event.get("type")
                    if event_type == "response.created" and event.get("response_id"):
                        provider_request_id = str(event["response_id"])
                    elif event_type == "response.audio.delta":
                        chunk = self._decode_audio(event)
                        if not chunk:
                            continue
                        arrived_at = time.monotonic()
                        trace.first_chunk_received = trace.first_chunk_received or arrived_at
                        trace.audio_chunk_arrivals.append(arrived_at)
                        trace.audio_chunk_sizes_bytes.append(len(chunk))
                        await audio_sink(chunk)
                        audio_bytes += len(chunk)
                        chunk_count += 1
                    elif event_type == "error":
                        raise RealtimeWsTtsError("realtime TTS server returned an error",
                            phase=phase, error_detail=json.dumps(event, ensure_ascii=False),
                            provider_request_id=provider_request_id)
                    elif event_type == "response.done":
                        break
                    # response.audio.done and session.updated are informational.
                await sender
                trace.completed = time.monotonic()
            if audio_bytes <= 0:
                raise RealtimeWsTtsError("realtime TTS completed without audio",
                    phase="response_done", error_detail="no response.audio.delta payloads",
                    provider_request_id=provider_request_id)
            return SpeechSynthesisResult(audio_bytes, chunk_count, provider_request_id)
        except asyncio.CancelledError:
            if websocket is not None:
                try:
                    await websocket.send(json.dumps({"type": "response.cancel"}))
                except Exception:
                    pass
            await self._discard_connection()
            raise
        except Exception as exc:
            trace.failure_phase = getattr(exc, "phase", phase)
            trace.failure_detail = getattr(exc, "error_detail", str(exc))[:1000]
            await self._discard_connection()
            if isinstance(exc, RealtimeWsTtsError):
                raise
            raise RealtimeWsTtsError("realtime TTS call failed", phase=phase,
                                     error_detail=str(exc),
                                     provider_request_id=provider_request_id) from exc
        finally:
            if sender is not None and not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            await self._publish_trace(trace)

    async def _borrow_connection(self, request: SpeechSynthesisRequest) -> tuple[Any, str | None]:
        requested_voice = self._effective_voice(request.voice)
        if self._websocket is not None and self._connection_voice == requested_voice:
            return self._websocket, None
        await self._discard_connection()
        context = self._connector(_session_url(self._config.url, voice=request.voice,
                                               session_id=request.turn_id), max_size=None)
        try:
            websocket = await context.__aenter__()
            created = await self._receive_event(websocket)
            if created.get("type") != "session.created":
                raise RealtimeWsTtsError("realtime TTS expected session.created",
                    phase="session_created", error_detail=f"unexpected event: {created.get('type')}")
        except BaseException:
            await context.__aexit__(None, None, None)
            raise
        self._connection_context, self._websocket = context, websocket
        self._connection_voice = requested_voice
        session = created.get("session")
        session_id = str(session["id"]) if isinstance(session, dict) and session.get("id") else None
        return websocket, session_id

    def _effective_voice(self, request_voice: str) -> str:
        configured = dict(parse_qsl(urlsplit(self._config.url).query,
                                    keep_blank_values=True)).get("voice")
        return configured or request_voice

    async def _discard_connection(self) -> None:
        context = self._connection_context
        self._connection_context = self._websocket = self._connection_voice = None
        self._request_sequence = 0
        if context is not None:
            try:
                await context.__aexit__(None, None, None)
            except Exception:
                pass

    @staticmethod
    async def _send_text(websocket: Any, text_chunks: AsyncIterator[str],
                         trace: SynthesisTrace) -> None:
        saw_text = False
        async for chunk in text_chunks:
            if not chunk:
                continue
            trace.request_started = trace.request_started or time.monotonic()
            await websocket.send(json.dumps(
                {"type": "input_text_buffer.append", "text": chunk}, ensure_ascii=False))
            trace.input_text_chunk_sizes_chars.append(len(chunk))
            trace.input_text_chunk_sent_at.append(time.monotonic())
            saw_text = True
        if not saw_text:
            raise RealtimeWsTtsError("realtime TTS stream received no text",
                                     phase="send_text", error_detail="no non-empty text chunks")
        await websocket.send(json.dumps({"type": "input_text_buffer.commit"}))
        trace.input_committed = time.monotonic()

    @staticmethod
    async def _receive_event(websocket: Any) -> dict[str, Any]:
        message = await websocket.recv()
        if not isinstance(message, str):
            raise RealtimeWsTtsError("realtime TTS returned a non-JSON frame",
                phase="receive_event", error_detail=f"frame type: {type(message).__name__}")
        try:
            event = json.loads(message)
        except json.JSONDecodeError as exc:
            raise RealtimeWsTtsError("realtime TTS returned invalid JSON",
                                     phase="receive_event", error_detail=str(exc)) from exc
        if not isinstance(event, dict):
            raise RealtimeWsTtsError("realtime TTS returned a non-object event",
                                     phase="receive_event", error_detail=type(event).__name__)
        return event

    @staticmethod
    def _decode_audio(event: dict[str, Any]) -> bytes:
        encoded = event.get("delta")
        if not isinstance(encoded, str):
            raise RealtimeWsTtsError("response.audio.delta is missing delta",
                                     phase="decode_audio", error_detail="delta must be base64")
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RealtimeWsTtsError("response.audio.delta contains invalid base64",
                                     phase="decode_audio", error_detail=str(exc)) from exc

    async def _publish_trace(self, trace: SynthesisTrace) -> None:
        if self._trace_sink is None:
            return
        try:
            result = self._trace_sink(trace)
            if result is not None:
                await result
        except Exception:
            pass  # Telemetry must not change synthesis semantics.
