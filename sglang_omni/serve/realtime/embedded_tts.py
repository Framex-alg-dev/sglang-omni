# SPDX-License-Identifier: Apache-2.0
"""Session-owned reusable WebSocket connection for embedded realtime TTS."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from sglang_omni.serve.realtime.tts_buffer import buffered_appends
from sglang_omni.serve.realtime.tts_text import (
    StreamingTTSWhitespace,
    TTSTextAppend,
    split_tts_append,
)
from sglang_omni.utils.structured_logs import emit_structured_log

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
    audio_idle_timeout_seconds: float = 10.0
    completion_timeout_seconds: float = 5.0
    turn_timeout_seconds: float = 300.0
    connection_count: int = 1
    text_queue_max_chunks: int = 64
    normalize_text_whitespace: bool = True
    text_buffer_enabled: bool = True
    text_first_buffer_seconds: float = 0.06
    text_later_buffer_seconds: float = 0.10
    text_coalesce: bool = False
    text_append_target_chars: int = 0
    max_turn_text_chars: int = 16384
    max_turn_text_bytes: int = 65536
    log_text_payloads: bool = False
    max_audio_chunk_bytes: int = 1024 * 1024
    max_turn_audio_bytes: int = 32 * 1024 * 1024
    provisional_audio_max_bytes: int = 8 * 1024 * 1024
    provisional_audio_max_milliseconds: int = 10000

    @staticmethod
    def text_options_from_env() -> dict[str, Any]:
        """Shared production/dev entry settings; direct constructors stay explicit."""
        options: dict[str, Any] = {}
        for field, suffix, default in (
            ("normalize_text_whitespace", "NORMALIZE_WHITESPACE", "1"),
            ("text_coalesce", "COALESCE", "0"),
            ("text_buffer_enabled", "BUFFER_ENABLED", "1"),
            ("log_text_payloads", "LOG_PAYLOADS", "0"),
        ):
            value = os.environ.get("SGLANG_OMNI_TTS_TEXT_" + suffix, default).strip().lower()
            if value not in {"1", "true", "0", "false"}:
                raise ValueError(f"SGLANG_OMNI_TTS_TEXT_{suffix} must be 0/1 or false/true")
            options[field] = value in {"1", "true"}
        for field, suffix, default in (
            ("text_append_target_chars", "APPEND_TARGET_CHARS", "0"),
            ("max_turn_text_chars", "MAX_TURN_CHARS", "16384"),
            ("max_turn_text_bytes", "MAX_TURN_BYTES", "65536"),
        ):
            options[field] = int(os.environ.get("SGLANG_OMNI_TTS_TEXT_" + suffix, default))
        return options

    def __post_init__(self) -> None:
        for name in ("normalize_text_whitespace", "text_coalesce", "log_text_payloads", "text_buffer_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"TTS {name} must be a bool")
        if type(self.connection_count) is not int or self.connection_count not in (1, 2):
            raise ValueError("TTS connection_count must be 1 or 2")
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
            "audio_idle_timeout_seconds",
            "completion_timeout_seconds",
            "text_first_buffer_seconds",
            "text_later_buffer_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"TTS {name} must be positive")
        for name in (
            "text_queue_max_chunks",
            "max_turn_text_chars",
            "max_turn_text_bytes",
            "max_audio_chunk_bytes",
            "max_turn_audio_bytes",
            "provisional_audio_max_bytes",
            "provisional_audio_max_milliseconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TTS {name} must be a positive integer")
        if (type(self.text_append_target_chars) is not int
                or not 0 <= self.text_append_target_chars <= self.max_turn_text_chars):
            raise ValueError("TTS text_append_target_chars must be between zero and max_turn_text_chars")


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
        session_instance_id: str | None = None,
        connector: Callable[..., Any] = connect,
    ) -> None:
        if not session_id.strip():
            raise ValueError("external session_id must be non-empty")
        self._config = config
        self._session_id = session_id
        self._provider_session_id = f"{session_id}:{session_instance_id}" if session_instance_id else session_id
        self._connector = connector
        self._session_instance_id = session_instance_id
        self._connection_epoch = 0
        self._turn_lock = asyncio.Lock()
        self._connection_context: Any | None = None
        self._websocket: Any | None = None
        self._connection_voice: str | None = None
        self._active_owner: asyncio.Task[Any] | None = None
        self._active_turn_id: str | None = None
        self._broken = False
        self._closed = False
        self._standby = (EmbeddedTTSConnection(replace(config, connection_count=1),
                         session_id=session_id + ":standby", session_instance_id=session_instance_id, connector=connector)
                         if config.connection_count == 2 else None)
        self._standby_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        if self._websocket is None or self._broken:
            return False
        state = getattr(self._websocket, "state", None)
        if state is not None:
            return state == State.OPEN
        return not bool(getattr(self._websocket, "closed", False))

    async def synthesize_streaming(
        self,
        *,
        turn_id: str,
        text_chunks: AsyncIterator[str],
        audio_sink: AudioSink,
        voice: str | None = None,
        instruct: str | Awaitable[str] | None = None,
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
            self._active_turn_id = turn_id
            websocket: Any | None = None
            producer: asyncio.Task[None] | None = None
            sender: asyncio.Task[None] | None = None
            receiver: asyncio.Task[EmbeddedTTSResult] | None = None
            deadline = time.monotonic() + self._config.turn_timeout_seconds
            try:
                emit_structured_log(
                    "performance",
                    "tts_turn_started",
                    session_id=self._session_id,
                    turn_id=turn_id,
                    first_audio_timeout_seconds=self._config.first_audio_timeout_seconds,
                    audio_idle_timeout_seconds=self._config.audio_idle_timeout_seconds,
                    completion_timeout_seconds=self._config.completion_timeout_seconds,
                    turn_timeout_seconds=self._config.turn_timeout_seconds,
                )
                if (not self.connected and self._standby is not None
                        and (self._standby_task is None or self._standby_task.done())
                        and self._standby.connected and self._standby._connection_voice == selected_voice):
                    await self._discard_connection()
                    for name in ("_connection_context", "_websocket", "_connection_voice", "_broken"):
                        current = getattr(self, name)
                        setattr(self, name, getattr(self._standby, name))
                        setattr(self._standby, name, current)
                websocket = await self._borrow_with_retry(selected_voice, turn_id, deadline)
                self._warm_standby(selected_voice)
                logger.debug(
                    "Embedded TTS turn started session_id=%s turn_id=%s lifecycle=streaming",
                    self._session_id,
                    turn_id,
                )
                queue: asyncio.Queue[TTSTextAppend | None] = asyncio.Queue(
                    maxsize=self._config.text_queue_max_chunks
                )
                first_text_sent = asyncio.Event()
                producer = asyncio.create_task(
                    self._produce_text(text_chunks, queue, turn_id),
                    name=f"embedded-tts-producer:{turn_id}",
                )
                sender = asyncio.create_task(
                    self._send_text(
                        websocket,
                        queue,
                        first_text_sent,
                        turn_id,
                        instruct=instruct,
                    ),
                    name=f"embedded-tts-sender:{turn_id}",
                )
                receiver = asyncio.create_task(
                    self._receive_turn(websocket, audio_sink, first_text_sent, turn_id),
                    name=f"embedded-tts-receiver:{turn_id}",
                )
                result, _, _ = await _wait_for_phase(
                    asyncio.gather(receiver, producer, sender),
                    timeout=max(0.0, deadline - time.monotonic()),
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
                emit_structured_log(
                    "performance",
                    "tts_stream_completed",
                    session_id=self._session_id,
                    turn_id=turn_id,
                    audio_chunks=result.chunk_count,
                    audio_bytes=result.audio_bytes,
                    audio_ms=round(result.audio_bytes * 1000 / 48000, 3),
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
                emit_structured_log(
                    "error",
                    "tts_turn_failed",
                    level="error",
                    session_id=self._session_id,
                    turn_id=turn_id,
                    phase=phase,
                    exception_type=exception_type,
                    close_code=close_code,
                    close_reason=close_reason,
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
                self._active_turn_id = None

    async def cancel_active_turn(self) -> None:
        owner = self._active_owner
        websocket = self._websocket
        if owner is None:
            return
        await self._cancel_and_discard(websocket)
        if owner is not asyncio.current_task() and not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    def _warm_standby(self, voice: str) -> None:
        if self._standby is None or self._closed or (self._standby_task is not None and not self._standby_task.done()):
            return
        async def warm():
            try:
                await self._standby._borrow_with_retry(voice, "standby-warm", time.monotonic() + self._config.connect_timeout_seconds + self._config.ready_timeout_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.info("TTS standby warm failed; primary remains usable")
        self._standby_task = asyncio.create_task(warm())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.cancel_active_turn()
        if self._standby_task is not None:
            self._standby_task.cancel()
            await asyncio.gather(self._standby_task, return_exceptions=True)
        if self._standby is not None:
            await self._standby.close()
        async with self._turn_lock:
            await self._discard_connection()

    async def _borrow_with_retry(self, voice: str, turn_id: str, deadline: float) -> Any:
        """Retry transport failure once, before consuming or sending any text.

        A send failure is deliberately outside this boundary: lack of audio
        does not prove that the provider failed to accept the text.
        """
        for attempt in range(2):
            if self._closed:
                raise EmbeddedTTSError("TTS Session is closed", phase="borrow")
            try:
                websocket = await _wait_for_phase(
                    self._borrow_connection(voice, turn_id),
                    timeout=max(0.0, deadline - time.monotonic()),
                    phase="turn_timeout",
                )
                if not self.connected:
                    raise OSError("TTS connection closed during handshake")
                return websocket
            except (ConnectionClosed, OSError):
                await self._discard_connection()
                if attempt or time.monotonic() >= deadline:
                    raise
                emit_structured_log(
                    "performance", "tts_connection_retry",
                    session_id=self._session_id, turn_id=turn_id,
                    attempt=attempt + 1, text_send_attempted=False,
                )
        raise AssertionError("unreachable TTS retry state")

    async def _borrow_connection(self, voice: str, turn_id: str) -> Any:
        if self.connected and self._connection_voice == voice:
            logger.debug(
                "Embedded TTS connection reused session_id=%s lifecycle=ready",
                self._session_id,
            )
            emit_structured_log(
                "performance",
                "tts_connection_reused",
                session_id=self._session_id,
                turn_id=turn_id,
            )
            return self._websocket
        await self._discard_connection()
        self._broken = False
        connect_started = time.monotonic()
        emit_structured_log(
            "performance",
            "tts_connect_begin",
            session_id=self._session_id,
            turn_id=turn_id,
            provider="configured_tts",
        )
        self._connection_epoch += 1
        provider_identity = (f"{self._provider_session_id}:{self._connection_epoch}"
                             if self._session_instance_id else self._provider_session_id)
        context = self._connector(
            _session_url(self._config.url, voice=voice, session_id=provider_identity),
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
        emit_structured_log(
            "performance",
            "tts_session_ready",
            session_id=self._session_id,
            turn_id=turn_id,
            reused=False,
            elapsed_ms=round((time.monotonic() - connect_started) * 1000, 3),
        )
        return websocket

    async def _produce_text(
        self, text_chunks: AsyncIterator[str], queue: asyncio.Queue[TTSTextAppend | None],
        turn_id: str,
    ) -> None:
        normalizer = StreamingTTSWhitespace()
        input_chars = input_bytes = output_chars = source_seq = 0
        pending_source: int | None = None
        pending_at: float | None = None
        async for chunk in text_chunks:
            if not isinstance(chunk, str):
                raise EmbeddedTTSError("TTS text chunk must be a string", phase="send")
            if not chunk:
                continue
            arrived = time.monotonic()
            source_seq += 1
            input_chars += len(chunk)
            input_bytes += len(chunk.encode("utf-8"))
            if (input_chars > self._config.max_turn_text_chars
                    or input_bytes > self._config.max_turn_text_bytes):
                raise EmbeddedTTSError("TTS turn text budget exceeded", phase="text_budget")
            if self._config.log_text_payloads:
                emit_structured_log(
                    "diagnostic", "tts_text_delta_received", session_id=self._session_id,
                    turn_id=turn_id, source_seq=source_seq, text=chunk,
                )
            first_source = pending_source or source_seq
            first_at = pending_at if pending_at is not None else arrived
            normalized = normalizer.feed(chunk) if self._config.normalize_text_whitespace else chunk
            output_chars += len(normalized)
            if normalizer.pending:
                if normalized or pending_source is None:
                    pending_source, pending_at = source_seq, arrived
            else:
                pending_source = pending_at = None
            try:
                pieces = split_tts_append(
                    normalized, self._config.text_append_target_chars,
                    self._config.max_turn_text_chars,
                )
            except ValueError as exc:
                raise EmbeddedTTSError(str(exc), phase="text_budget") from None
            for piece in pieces:
                await queue.put(TTSTextAppend(
                    piece, first_source, source_seq, first_at,
                    "size_split" if len(pieces) > 1 else "delta",
                ))
        normalizer.finish()
        emit_structured_log(
            "performance", "tts_text_normalization_completed", session_id=self._session_id,
            turn_id=turn_id, input_chars=input_chars, input_bytes=input_bytes,
            output_chars=output_chars, removed_chars=input_chars - output_chars,
            source_delta_count=source_seq,
            normalization_enabled=self._config.normalize_text_whitespace,
            leading_whitespace_chars=normalizer.leading_whitespace_chars,
            trailing_whitespace_chars=normalizer.trailing_whitespace_chars,
            collapsed_whitespace_chars=normalizer.collapsed_whitespace_chars,
            cr_chars=normalizer.cr_chars,
        )
        await queue.put(None)

    async def _text_batches(
        self, queue: asyncio.Queue[TTSTextAppend | None]
    ) -> AsyncIterator[TTSTextAppend]:
        """One batching owner: timed buffering, or the legacy zero-wait path."""
        if self._config.text_buffer_enabled:
            async for item in buffered_appends(
                queue, first_wait=self._config.text_first_buffer_seconds,
                later_wait=self._config.text_later_buffer_seconds,
            ):
                yield item
            return
        carry = None
        while True:
            item = carry if carry is not None else await queue.get()
            carry = None
            if item is None:
                return
            eof = False
            target = self._config.text_append_target_chars or 512
            if self._config.text_coalesce:
                while len(item.text) < target:
                    # Send likely gateway boundaries promptly. The gateway still
                    # decides whether punctuation is internal to a word/number.
                    if any(c in item.text for c in "\n。！？；.!?;,:，："):
                        break
                    try:
                        following = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if following is None:
                        eof = True
                        break
                    if len(item.text) + len(following.text) > target:
                        carry = following
                        break
                    item = TTSTextAppend(
                        item.text + following.text, item.source_first,
                        following.source_last, item.received_at, "queued_coalesce",
                    )
            yield item
            if eof:
                return

    async def _send_text(
        self,
        websocket: Any,
        queue: asyncio.Queue[TTSTextAppend | None],
        first_text_sent: asyncio.Event,
        turn_id: str,
        *,
        instruct: str | Awaitable[str] | None,
    ) -> None:
        instruct_wait_started = time.monotonic()
        resolved_instruct = await self._resolve_instruct(instruct)
        instruct_wait_ms = round(
            (time.monotonic() - instruct_wait_started) * 1000,
            3,
        )
        saw_text = False
        text_chunks = 0
        text_chars = 0
        first_append_started: float | None = None
        async for item in self._text_batches(queue):
            chunk = item.text
            saw_text = True
            if first_append_started is None:
                first_append_started = time.monotonic()
            append_event = {
                "type": "input_text_buffer.append",
                "text": chunk,
            }
            if resolved_instruct is not None:
                append_event["instruct"] = resolved_instruct
            send_started = time.monotonic()
            await _wait_for_phase(
                websocket.send(json.dumps(append_event, ensure_ascii=False)),
                timeout=self._config.send_timeout_seconds,
                phase="send_timeout",
            )
            text_chunks += 1
            text_chars += len(chunk)
            sent_at = time.monotonic()
            encoded = chunk.encode("utf-8")
            emit_structured_log(
                "performance", "tts_text_append_sent", session_id=self._session_id,
                turn_id=turn_id, seq=text_chunks, chars=len(chunk), utf8_bytes=len(encoded),
                source_first=item.source_first, source_last=item.source_last,
                reason=item.reason, text_sha256=hashlib.sha256(encoded).hexdigest(),
                buffer_wait_ms=round((send_started - item.received_at) * 1000, 3),
                send_ms=round((sent_at - send_started) * 1000, 3),
                queue_depth=queue.qsize(),
                **({"text": chunk} if self._config.log_text_payloads else {}),
            )
            if text_chunks == 1:
                emit_structured_log(
                    "performance",
                    "tts_first_append_sent",
                    session_id=self._session_id,
                    turn_id=turn_id,
                    chars=len(chunk),
                    instruct_applied=resolved_instruct is not None,
                    instruct_chars=(
                        len(resolved_instruct) if resolved_instruct is not None else 0
                    ),
                    instruct_wait_ms=instruct_wait_ms,
                    elapsed_ms=round(
                        (time.monotonic() - first_append_started) * 1000, 3
                    ),
                )
            first_text_sent.set()
        if not saw_text:
            raise EmbeddedTTSError("TTS stream contains no text", phase="send")
        await _wait_for_phase(
            websocket.send(json.dumps({"type": "input_text_buffer.commit"})),
            timeout=self._config.send_timeout_seconds,
            phase="send_timeout",
        )
        emit_structured_log(
            "performance",
            "tts_commit_sent",
            session_id=self._session_id,
            turn_id=turn_id,
            text_chunks=text_chunks,
            text_chars=text_chars,
        )

    @staticmethod
    async def _resolve_instruct(
        instruct: str | Awaitable[str] | None,
    ) -> str | None:
        if instruct is None:
            return None
        value = (
            instruct
            if isinstance(instruct, str)
            else await asyncio.shield(instruct)
        )
        if not isinstance(value, str):
            raise EmbeddedTTSError(
                "TTS instruct must resolve to a string",
                phase="send",
            )
        value = value.strip()
        if not value:
            raise EmbeddedTTSError("TTS instruct must be non-empty", phase="send")
        if len(value) > 1000:
            raise EmbeddedTTSError(
                "TTS instruct exceeds 1000 characters",
                phase="send",
            )
        return value

    async def _receive_turn(
        self,
        websocket: Any,
        audio_sink: AudioSink,
        first_text_sent: asyncio.Event,
        turn_id: str,
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
        progress_deadline = first_audio_deadline
        while True:
            phase = (
                "completion_timeout" if audio_done else
                "first_audio_timeout" if chunk_count == 0 else "audio_idle_timeout"
            )
            remaining = progress_deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddedTTSError(f"TTS {phase} timed out", phase=phase)
            event = await _wait_for_phase(
                self._receive_event(websocket), timeout=remaining, phase=phase,
            )
            event_type = event.get("type")
            if event_type == "response.created":
                if response_created:
                    raise EmbeddedTTSError(
                        "duplicate response.created", phase="protocol"
                    )
                response_created = True
                raw_response_id = event.get("response_id")
                response_id = str(raw_response_id) if raw_response_id else None
                emit_structured_log(
                    "performance",
                    "tts_response_created",
                    session_id=self._session_id,
                    turn_id=turn_id,
                    provider_response_id=response_id,
                )
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
                    # Bound downstream backpressure separately from provider stalls.
                    await _wait_for_phase(
                        audio_sink(chunk), timeout=self._config.send_timeout_seconds,
                        phase="audio_delivery_timeout",
                    )
                    progress_deadline = time.monotonic() + self._config.audio_idle_timeout_seconds
                    chunk_count += 1
                    if chunk_count == 1:
                        emit_structured_log(
                            "performance",
                            "tts_first_audio_received",
                            session_id=self._session_id,
                            turn_id=turn_id,
                            audio_bytes=len(chunk),
                        )
                    if audio_bytes >= 12000 and audio_bytes - len(chunk) < 12000:
                        emit_structured_log(
                            "performance",
                            "tts_playable_250ms_ready",
                            session_id=self._session_id,
                            turn_id=turn_id,
                            audio_bytes=audio_bytes,
                            audio_chunks=chunk_count,
                        )
            elif event_type == "response.audio.done":
                if not response_created or audio_done:
                    raise EmbeddedTTSError(
                        "response.audio.done is out of order", phase="protocol"
                    )
                audio_done = True
                progress_deadline = time.monotonic() + self._config.completion_timeout_seconds
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
        cancel_started = time.monotonic()
        turn_id = self._active_turn_id
        if turn_id is not None:
            emit_structured_log(
                "performance",
                "tts_cancel_begin",
                session_id=self._session_id,
                turn_id=turn_id,
            )
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
        if turn_id is not None:
            emit_structured_log(
                "performance",
                "tts_cancel_completed",
                session_id=self._session_id,
                turn_id=turn_id,
                elapsed_ms=round((time.monotonic() - cancel_started) * 1000, 3),
            )

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
