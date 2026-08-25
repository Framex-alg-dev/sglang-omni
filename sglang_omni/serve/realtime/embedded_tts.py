# SPDX-License-Identifier: Apache-2.0
"""Session-owned reusable WebSocket connection for embedded realtime TTS."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

AudioSink = Callable[[bytes], Awaitable[None]]
logger = logging.getLogger(__name__)


class EmbeddedTTSError(RuntimeError):
    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase


async def _wait_for_phase(
    awaitable: Awaitable[Any], *, timeout: float, phase: str
) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise EmbeddedTTSError(f"TTS {phase} timed out", phase=phase) from exc


@dataclass(frozen=True)
class EmbeddedTTSConfig:
    url: str
    voice: str
    connect_timeout_seconds: float = 10.0
    ready_timeout_seconds: float = 10.0
    send_timeout_seconds: float = 10.0
    first_audio_timeout_seconds: float = 10.0
    turn_timeout_seconds: float = 30.0
    text_queue_max_chunks: int = 64
    max_audio_chunk_bytes: int = 1024 * 1024
    max_turn_audio_bytes: int = 32 * 1024 * 1024
    provisional_audio_max_bytes: int = 8 * 1024 * 1024
    provisional_audio_max_milliseconds: int = 10000

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("TTS URL must be an absolute ws:// or wss:// URL")
        if not self.voice.strip():
            raise ValueError("TTS voice must be non-empty")
        for name in (
            "connect_timeout_seconds",
            "ready_timeout_seconds",
            "send_timeout_seconds",
            "first_audio_timeout_seconds",
            "turn_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"TTS {name} must be positive")
        for name in (
            "text_queue_max_chunks",
            "max_audio_chunk_bytes",
            "max_turn_audio_bytes",
            "provisional_audio_max_bytes",
            "provisional_audio_max_milliseconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TTS {name} must be a positive integer")


@dataclass(frozen=True)
class EmbeddedTTSResult:
    audio_bytes: int
    chunk_count: int
    provider_response_id: str | None


def _session_url(url: str, *, voice: str, session_id: str) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["voice"] = voice
    query["session_id"] = session_id
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _exception_chain(exception: BaseException) -> list[BaseException]:
    chain = [exception]
    seen = {id(exception)}
    while True:
        current = chain[-1]
        nested = current.__cause__ or current.__context__
        if nested is None or id(nested) in seen:
            return chain
        chain.append(nested)
        seen.add(id(nested))


def _safe_close_reason(reason: str) -> str:
    # A provider-controlled close reason may contain a URL, credentials,
    # headers, request text, or encoded media. Its content cannot be made safe
    # with pattern-based redaction, so retain only bounded structural metadata.
    return f"<redacted:{min(len(reason), 9999)} chars>"


def _failure_diagnostics(exception: Exception) -> tuple[str, int | None, str | None]:
    chain = _exception_chain(exception)
    underlying_type = type(chain[-1]).__name__
    for nested in chain:
        if isinstance(nested, ConnectionClosed):
            close = nested.rcvd or nested.sent
            if close is None:
                return underlying_type, None, None
            return underlying_type, int(close.code), _safe_close_reason(close.reason)
    return underlying_type, None, None


class EmbeddedTTSConnection:
    """Own one lazy provider connection for exactly one external Session."""

    def __init__(
        self,
        config: EmbeddedTTSConfig,
        *,
        session_id: str,
        connector: Callable[..., Any] = connect,
    ) -> None:
        if not session_id.strip():
            raise ValueError("external session_id must be non-empty")
        self._config = config
        self._session_id = session_id
        self._connector = connector
        self._turn_lock = asyncio.Lock()
        self._connection_context: Any | None = None
        self._websocket: Any | None = None
        self._connection_voice: str | None = None
        self._active_owner: asyncio.Task[Any] | None = None
        self._broken = False
        self._closed = False

    @property
    def connected(self) -> bool:
        if self._websocket is None or self._broken:
            return False
        return not bool(getattr(self._websocket, "closed", False))

    async def synthesize_streaming(
        self,
        *,
        turn_id: str,
        text_chunks: AsyncIterator[str],
        audio_sink: AudioSink,
        voice: str | None = None,
    ) -> EmbeddedTTSResult:
        if not turn_id.strip():
            raise ValueError("TTS turn_id must be non-empty")
        selected_voice = (voice or self._config.voice).strip()
        if not selected_voice:
            raise ValueError("TTS voice must be non-empty")
        async with self._turn_lock:
            if self._closed:
                raise EmbeddedTTSError("TTS Session is closed", phase="borrow")
            self._active_owner = asyncio.current_task()
            websocket: Any | None = None
            producer: asyncio.Task[None] | None = None
            sender: asyncio.Task[None] | None = None
            receiver: asyncio.Task[EmbeddedTTSResult] | None = None
            try:
                websocket = await self._borrow_connection(selected_voice)
                logger.debug(
                    "Embedded TTS turn started session_id=%s turn_id=%s lifecycle=streaming",
                    self._session_id,
                    turn_id,
                )
                queue: asyncio.Queue[str | None] = asyncio.Queue(
                    maxsize=self._config.text_queue_max_chunks
                )
                first_text_sent = asyncio.Event()
                producer = asyncio.create_task(
                    self._produce_text(text_chunks, queue),
                    name=f"embedded-tts-producer:{turn_id}",
                )
                sender = asyncio.create_task(
                    self._send_text(websocket, queue, first_text_sent),
                    name=f"embedded-tts-sender:{turn_id}",
                )
                receiver = asyncio.create_task(
                    self._receive_turn(websocket, audio_sink, first_text_sent),
                    name=f"embedded-tts-receiver:{turn_id}",
                )
                result, _, _ = await _wait_for_phase(
                    asyncio.gather(receiver, producer, sender),
                    timeout=self._config.turn_timeout_seconds,
                    phase="turn_timeout",
                )
                logger.debug(
                    "Embedded TTS turn completed session_id=%s turn_id=%s "
                    "audio_chunks=%d audio_bytes=%d lifecycle=ready",
                    self._session_id,
                    turn_id,
                    result.chunk_count,
                    result.audio_bytes,
                )
                return result
            except asyncio.CancelledError:
                await self._cancel_and_discard(websocket)
                raise
            except Exception as exc:
                phase = exc.phase if isinstance(exc, EmbeddedTTSError) else "transport"
                exception_type, close_code, close_reason = _failure_diagnostics(exc)
                logger.error(
                    "Embedded TTS turn failed session_id=%s turn_id=%s phase=%s "
                    "exception_type=%s close_code=%s close_reason=%r",
                    self._session_id,
                    turn_id,
                    phase,
                    exception_type,
                    close_code,
                    close_reason,
                )
                await self._cancel_and_discard(websocket)
                if isinstance(exc, EmbeddedTTSError):
                    raise
                # Provider transport exceptions may include the full URL and
                # query credentials. Do not retain them as a chained cause:
                # the Session owner logs this stable public-safe exception.
                raise EmbeddedTTSError(
                    "embedded TTS turn failed", phase="turn"
                ) from None
            finally:
                for task in (producer, sender, receiver):
                    if task is not None and not task.done():
                        task.cancel()
                pending = [
                    task for task in (producer, sender, receiver) if task is not None
                ]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                self._active_owner = None

    async def cancel_active_turn(self) -> None:
        owner = self._active_owner
        websocket = self._websocket
        if owner is None:
            return
        await self._cancel_and_discard(websocket)
        if owner is not asyncio.current_task() and not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.cancel_active_turn()
        async with self._turn_lock:
            await self._discard_connection()

    async def _borrow_connection(self, voice: str) -> Any:
        if self.connected and self._connection_voice == voice:
            logger.debug(
                "Embedded TTS connection reused session_id=%s lifecycle=ready",
                self._session_id,
            )
            return self._websocket
        await self._discard_connection()
        self._broken = False
        context = self._connector(
            _session_url(self._config.url, voice=voice, session_id=self._session_id),
            max_size=self._config.max_audio_chunk_bytes * 2,
        )
        try:
            websocket = await _wait_for_phase(
                context.__aenter__(),
                timeout=self._config.connect_timeout_seconds,
                phase="connect_timeout",
            )
            event = await _wait_for_phase(
                self._receive_event(websocket),
                timeout=self._config.ready_timeout_seconds,
                phase="ready_timeout",
            )
            if event.get("type") != "session.created":
                raise EmbeddedTTSError(
                    "TTS provider expected session.created", phase="handshake"
                )
        except BaseException:
            await context.__aexit__(None, None, None)
            raise
        self._connection_context = context
        self._websocket = websocket
        self._connection_voice = voice
        logger.debug(
            "Embedded TTS connection ready session_id=%s lifecycle=ready",
            self._session_id,
        )
        return websocket

    @staticmethod
    async def _produce_text(
        text_chunks: AsyncIterator[str], queue: asyncio.Queue[str | None]
    ) -> None:
        async for chunk in text_chunks:
            if not isinstance(chunk, str):
                raise EmbeddedTTSError("TTS text chunk must be a string", phase="send")
            if chunk:
                await queue.put(chunk)
        await queue.put(None)

    async def _send_text(
        self,
        websocket: Any,
        queue: asyncio.Queue[str | None],
        first_text_sent: asyncio.Event,
    ) -> None:
        saw_text = False
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            saw_text = True
            await _wait_for_phase(
                websocket.send(
                    json.dumps(
                        {"type": "input_text_buffer.append", "text": chunk},
                        ensure_ascii=False,
                    )
                ),
                timeout=self._config.send_timeout_seconds,
                phase="send_timeout",
            )
            first_text_sent.set()
        if not saw_text:
            raise EmbeddedTTSError("TTS stream contains no text", phase="send")
        await _wait_for_phase(
            websocket.send(json.dumps({"type": "input_text_buffer.commit"})),
            timeout=self._config.send_timeout_seconds,
            phase="send_timeout",
        )

    async def _receive_turn(
        self,
        websocket: Any,
        audio_sink: AudioSink,
        first_text_sent: asyncio.Event,
    ) -> EmbeddedTTSResult:
        response_created = False
        audio_done = False
        audio_bytes = 0
        chunk_count = 0
        response_id: str | None = None
        await first_text_sent.wait()
        first_audio_deadline = (
            time.monotonic() + self._config.first_audio_timeout_seconds
        )
        while True:
            if chunk_count == 0:
                remaining = first_audio_deadline - time.monotonic()
                if remaining <= 0:
                    raise EmbeddedTTSError(
                        "TTS first audio timed out", phase="first_audio_timeout"
                    )
                try:
                    event = await asyncio.wait_for(
                        self._receive_event(websocket), timeout=remaining
                    )
                except asyncio.TimeoutError as exc:
                    raise EmbeddedTTSError(
                        "TTS first audio timed out", phase="first_audio_timeout"
                    ) from exc
            else:
                event = await self._receive_event(websocket)
            event_type = event.get("type")
            if event_type == "response.created":
                if response_created:
                    raise EmbeddedTTSError(
                        "duplicate response.created", phase="protocol"
                    )
                response_created = True
                raw_response_id = event.get("response_id")
                response_id = str(raw_response_id) if raw_response_id else None
            elif event_type == "response.audio.delta":
                if not response_created or audio_done:
                    raise EmbeddedTTSError(
                        "response.audio.delta is out of order", phase="protocol"
                    )
                chunk = self._decode_audio(event)
                if len(chunk) > self._config.max_audio_chunk_bytes:
                    raise EmbeddedTTSError(
                        "TTS audio chunk exceeds configured limit", phase="protocol"
                    )
                audio_bytes += len(chunk)
                if audio_bytes > self._config.max_turn_audio_bytes:
                    raise EmbeddedTTSError(
                        "TTS turn audio exceeds configured limit", phase="protocol"
                    )
                if chunk:
                    await audio_sink(chunk)
                    chunk_count += 1
            elif event_type == "response.audio.done":
                if not response_created or audio_done:
                    raise EmbeddedTTSError(
                        "response.audio.done is out of order", phase="protocol"
                    )
                audio_done = True
            elif event_type == "response.done":
                if not response_created or not audio_done:
                    raise EmbeddedTTSError(
                        "response.done is out of order", phase="protocol"
                    )
                if audio_bytes == 0:
                    raise EmbeddedTTSError(
                        "TTS completed without audio", phase="protocol"
                    )
                return EmbeddedTTSResult(audio_bytes, chunk_count, response_id)
            elif event_type == "error":
                raise EmbeddedTTSError(
                    "TTS provider returned an error", phase="provider"
                )
            elif event_type not in {"session.updated"}:
                raise EmbeddedTTSError(
                    f"unsupported TTS provider event: {event_type!r}", phase="protocol"
                )

    @staticmethod
    async def _receive_event(websocket: Any) -> dict[str, Any]:
        message = await websocket.recv()
        if not isinstance(message, str):
            raise EmbeddedTTSError("TTS provider sent a binary frame", phase="protocol")
        try:
            event = json.loads(message)
        except json.JSONDecodeError as exc:
            raise EmbeddedTTSError(
                "TTS provider sent invalid JSON", phase="protocol"
            ) from exc
        if not isinstance(event, dict):
            raise EmbeddedTTSError(
                "TTS provider event must be an object", phase="protocol"
            )
        return event

    @staticmethod
    def _decode_audio(event: dict[str, Any]) -> bytes:
        encoded = event.get("delta")
        if not isinstance(encoded, str):
            raise EmbeddedTTSError("TTS audio delta must be base64", phase="protocol")
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise EmbeddedTTSError(
                "TTS audio delta is invalid base64", phase="protocol"
            ) from exc

    async def _cancel_and_discard(self, websocket: Any | None) -> None:
        self._broken = True
        if websocket is not None:
            try:
                await asyncio.wait_for(
                    websocket.send(json.dumps({"type": "response.cancel"})),
                    timeout=self._config.send_timeout_seconds,
                )
            except Exception:
                pass
        await self._discard_connection()

    async def _discard_connection(self) -> None:
        context = self._connection_context
        self._connection_context = None
        self._websocket = None
        self._connection_voice = None
        if context is not None:
            try:
                await asyncio.wait_for(
                    context.__aexit__(None, None, None),
                    timeout=self._config.connect_timeout_seconds,
                )
            except Exception:
                pass
