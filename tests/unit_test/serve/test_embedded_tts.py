# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from collections import deque
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from sglang_omni.serve.realtime.embedded_tts import (
    EmbeddedTTSConfig,
    EmbeddedTTSConnection,
    EmbeddedTTSError,
)


def _event(type_: str, **values: Any) -> str:
    return json.dumps({"type": type_, **values})


def _successful_events(payload: bytes = b"\x00\x01") -> list[str]:
    return [
        _event("session.created", session={"id": "provider-session"}),
        _event("response.created", response_id="provider-response"),
        _event("response.audio.delta", delta=base64.b64encode(payload).decode()),
        _event("response.audio.done"),
        _event("response.done"),
    ]


class FakeWebSocket:
    def __init__(self, events: list[str]) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return await self.events.get()


class FakeConnectionContext:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.closed = False

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *args: object) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self, scripts: list[list[str]]) -> None:
        self.scripts = deque(scripts)
        self.urls: list[str] = []
        self.contexts: list[FakeConnectionContext] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeConnectionContext:
        del kwargs
        self.urls.append(url)
        context = FakeConnectionContext(FakeWebSocket(self.scripts.popleft()))
        self.contexts.append(context)
        return context


async def _chunks(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value


def _manager(
    connector: FakeConnector, session_id: str = "external-session"
) -> EmbeddedTTSConnection:
    return EmbeddedTTSConnection(
        EmbeddedTTSConfig(url="ws://tts.local/realtime", voice="voice-a"),
        session_id=session_id,
        connector=connector,
    )


@pytest.mark.asyncio
async def test_streaming_protocol_and_connection_reuse() -> None:
    connector = FakeConnector([_successful_events() + _successful_events()[1:]])
    manager = _manager(connector)
    audio: list[bytes] = []

    async def collect(chunk: bytes) -> None:
        audio.append(chunk)

    first = await manager.synthesize_streaming(
        turn_id="turn-1", text_chunks=_chunks("你", "好"), audio_sink=collect
    )
    second = await manager.synthesize_streaming(
        turn_id="turn-2", text_chunks=_chunks("再见"), audio_sink=collect
    )

    assert first.audio_bytes == second.audio_bytes == 2
    assert audio == [b"\x00\x01", b"\x00\x01"]
    assert len(connector.urls) == 1
    assert parse_qs(urlsplit(connector.urls[0]).query) == {
        "voice": ["voice-a"],
        "session_id": ["external-session"],
    }
    assert [item["type"] for item in connector.contexts[0].websocket.sent] == [
        "input_text_buffer.append",
        "input_text_buffer.append",
        "input_text_buffer.commit",
        "input_text_buffer.append",
        "input_text_buffer.commit",
    ]
    await manager.close()
    assert connector.contexts[0].closed is True
    await manager.close()


@pytest.mark.asyncio
async def test_voice_change_and_protocol_failure_force_reconnect() -> None:
    connector = FakeConnector(
        [
            _successful_events(),
            [
                _event("session.created"),
                _event("response.created"),
                _event("response.audio.delta", delta="not-base64"),
            ],
            _successful_events(),
        ]
    )
    manager = _manager(connector)

    await manager.synthesize_streaming(
        turn_id="turn-1",
        text_chunks=_chunks("a"),
        audio_sink=lambda _: asyncio.sleep(0),
    )
    with pytest.raises(EmbeddedTTSError, match="invalid base64"):
        await manager.synthesize_streaming(
            turn_id="turn-2",
            text_chunks=_chunks("b"),
            audio_sink=lambda _: asyncio.sleep(0),
            voice="voice-b",
        )
    assert connector.contexts[0].closed is True
    assert connector.contexts[1].closed is True
    await manager.synthesize_streaming(
        turn_id="turn-3",
        text_chunks=_chunks("c"),
        audio_sink=lambda _: asyncio.sleep(0),
        voice="voice-b",
    )
    assert len(connector.urls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        ["not-json"],
        [_event("response.created")],
        [_event("session.created"), _event("response.audio.done")],
        [
            _event("session.created"),
            _event("response.created"),
            _event("response.done"),
        ],
    ],
)
async def test_invalid_json_and_event_order_are_protocol_errors(
    events: list[str],
) -> None:
    connector = FakeConnector([events])
    manager = _manager(connector)

    with pytest.raises(EmbeddedTTSError):
        await manager.synthesize_streaming(
            turn_id="turn",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )
    assert connector.contexts[0].closed is True


@pytest.mark.asyncio
async def test_cancel_closes_connection_and_next_turn_reconnects() -> None:
    connector = FakeConnector(
        [
            [_event("session.created")],
            _successful_events(),
        ]
    )
    manager = _manager(connector)
    active = asyncio.create_task(
        manager.synthesize_streaming(
            turn_id="turn-1",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await manager.cancel_active_turn()

    with pytest.raises(asyncio.CancelledError):
        await active
    assert connector.contexts[0].closed is True
    assert {item["type"] for item in connector.contexts[0].websocket.sent} >= {
        "response.cancel"
    }
    await manager.synthesize_streaming(
        turn_id="turn-2",
        text_chunks=_chunks("again"),
        audio_sink=lambda _: asyncio.sleep(0),
    )
    assert len(connector.urls) == 2


@pytest.mark.asyncio
async def test_timeout_sends_cancel_and_discards_connection() -> None:
    connector = FakeConnector([[_event("session.created")]])
    manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="voice-a",
            turn_timeout_seconds=0.01,
        ),
        session_id="external-session",
        connector=connector,
    )

    with pytest.raises(EmbeddedTTSError) as exc_info:
        await manager.synthesize_streaming(
            turn_id="turn-timeout",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )

    assert exc_info.value.phase == "turn_timeout"
    assert connector.contexts[0].closed is True
    assert "response.cancel" in {
        item["type"] for item in connector.contexts[0].websocket.sent
    }


@pytest.mark.asyncio
async def test_turn_timeout_covers_text_producer_after_provider_done() -> None:
    connector = FakeConnector([_successful_events()])
    manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="voice-a",
            turn_timeout_seconds=0.01,
        ),
        session_id="external-session",
        connector=connector,
    )

    async def stalled_chunks() -> AsyncIterator[str]:
        yield "text"
        await asyncio.Event().wait()

    with pytest.raises(EmbeddedTTSError) as exc_info:
        await manager.synthesize_streaming(
            turn_id="turn-stalled-producer",
            text_chunks=stalled_chunks(),
            audio_sink=lambda _: asyncio.sleep(0),
        )

    assert exc_info.value.phase == "turn_timeout"
    assert connector.contexts[0].closed is True
    assert "response.cancel" in {
        item["type"] for item in connector.contexts[0].websocket.sent
    }


@pytest.mark.asyncio
async def test_ready_and_first_audio_timeouts_are_distinct() -> None:
    ready_connector = FakeConnector([[]])
    ready_manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="voice-a",
            ready_timeout_seconds=0.01,
            turn_timeout_seconds=0.1,
        ),
        session_id="external-session",
        connector=ready_connector,
    )
    with pytest.raises(EmbeddedTTSError) as ready_error:
        await ready_manager.synthesize_streaming(
            turn_id="turn-ready",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )
    assert ready_error.value.phase == "ready_timeout"

    audio_connector = FakeConnector(
        [[_event("session.created"), _event("response.created")]]
    )
    audio_manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="voice-a",
            first_audio_timeout_seconds=0.01,
            turn_timeout_seconds=0.1,
        ),
        session_id="external-session",
        connector=audio_connector,
    )
    with pytest.raises(EmbeddedTTSError) as audio_error:
        await audio_manager.synthesize_streaming(
            turn_id="turn-first-audio",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )
    assert audio_error.value.phase == "first_audio_timeout"


@pytest.mark.asyncio
async def test_first_audio_timeout_starts_after_first_text_is_sent() -> None:
    connector = FakeConnector([_successful_events()])
    manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="voice-a",
            first_audio_timeout_seconds=0.01,
            turn_timeout_seconds=0.2,
        ),
        session_id="external-session",
        connector=connector,
    )

    async def slow_first_chunk() -> AsyncIterator[str]:
        await asyncio.sleep(0.03)
        yield "text"

    result = await manager.synthesize_streaming(
        turn_id="turn-slow-first-text",
        text_chunks=slow_first_chunk(),
        audio_sink=lambda _: asyncio.sleep(0),
    )

    assert result.audio_bytes == 2


@pytest.mark.asyncio
async def test_text_producer_failure_cancels_blocked_receiver_immediately() -> None:
    connector = FakeConnector([[_event("session.created")]])
    manager = _manager(connector)

    async def invalid_chunks() -> AsyncIterator[Any]:
        yield 1

    with pytest.raises(EmbeddedTTSError, match="must be a string"):
        await asyncio.wait_for(
            manager.synthesize_streaming(
                turn_id="turn-invalid-text",
                text_chunks=invalid_chunks(),
                audio_sink=lambda _: asyncio.sleep(0),
            ),
            timeout=0.1,
        )

    assert connector.contexts[0].closed is True
    assert not any(
        task.get_name() == "embedded-tts-receiver:turn-invalid-text"
        for task in asyncio.all_tasks()
        if not task.done()
    )


@pytest.mark.asyncio
async def test_transport_failure_does_not_retain_sensitive_url_cause() -> None:
    secret_url = "wss://tts.local/realtime?token=top-secret"

    def failing_connector(url: str, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError(f"cannot connect to {url}")

    manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(url=secret_url, voice="voice-a"),
        session_id="external-session",
        connector=failing_connector,
    )

    with pytest.raises(EmbeddedTTSError) as exc_info:
        await manager.synthesize_streaming(
            turn_id="turn-transport-error",
            text_chunks=_chunks("text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )

    assert str(exc_info.value) == "embedded TTS turn failed"
    assert exc_info.value.__cause__ is None
    assert "top-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_transport_failure_log_is_correlated_and_redacted(caplog) -> None:
    secret_url = "wss://tts.local/realtime?token=top-secret"

    def failing_connector(url: str, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError(f"cannot connect to {url}; body=private-text")

    manager = EmbeddedTTSConnection(
        EmbeddedTTSConfig(url=secret_url, voice="secret-voice"),
        session_id="external-session",
        connector=failing_connector,
    )

    with (
        caplog.at_level("ERROR"),
        pytest.raises(EmbeddedTTSError, match="embedded TTS turn failed"),
    ):
        await manager.synthesize_streaming(
            turn_id="turn-transport-log",
            text_chunks=_chunks("private-text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "session_id=external-session" in message
    assert "turn_id=turn-transport-log" in message
    assert "phase=transport" in message
    assert "exception_type=RuntimeError" in message
    assert "top-secret" not in message
    assert "secret-voice" not in message
    assert "private-text" not in message
    assert "wss://" not in message


@pytest.mark.asyncio
async def test_connection_closed_log_sanitizes_bounded_reason(caplog) -> None:
    reason = "provider\nclosed\ttoken=" + ("x" * 200)
    closed = ConnectionClosedError(Close(1011, reason), None, None)

    class ClosingWebSocket(FakeWebSocket):
        async def recv(self) -> str:
            if not self.events.empty():
                return await super().recv()
            raise closed

    class ClosingConnector(FakeConnector):
        def __call__(self, url: str, **kwargs: Any) -> FakeConnectionContext:
            del kwargs
            self.urls.append(url)
            context = FakeConnectionContext(
                ClosingWebSocket([_event("session.created")])
            )
            self.contexts.append(context)
            return context

    manager = _manager(ClosingConnector([[]]), session_id="close-session")
    with (
        caplog.at_level("ERROR"),
        pytest.raises(EmbeddedTTSError, match="embedded TTS turn failed"),
    ):
        await manager.synthesize_streaming(
            turn_id="turn-close",
            text_chunks=_chunks("private-text"),
            audio_sink=lambda _: asyncio.sleep(0),
        )

    message = caplog.records[0].getMessage()
    assert "session_id=close-session" in message
    assert "turn_id=turn-close" in message
    assert "phase=transport" in message
    assert "exception_type=ConnectionClosedError" in message
    assert "close_code=1011" in message
    assert "close_reason='<redacted:222 chars>'" in message
    assert "provider" not in message
    assert "token=" not in message
    assert "x" * 20 not in message
    assert "\n" not in message
    assert "\t" not in message
    assert len(message) < 350


@pytest.mark.asyncio
async def test_different_sessions_never_share_connections() -> None:
    first_connector = FakeConnector([_successful_events()])
    second_connector = FakeConnector([_successful_events()])
    first = _manager(first_connector, "session-1")
    second = _manager(second_connector, "session-2")

    await first.synthesize_streaming(
        turn_id="turn-1",
        text_chunks=_chunks("a"),
        audio_sink=lambda _: asyncio.sleep(0),
    )
    await second.synthesize_streaming(
        turn_id="turn-2",
        text_chunks=_chunks("b"),
        audio_sink=lambda _: asyncio.sleep(0),
    )

    assert "session_id=session-1" in first_connector.urls[0]
    assert "session_id=session-2" in second_connector.urls[0]
    assert (
        first_connector.contexts[0].websocket
        is not second_connector.contexts[0].websocket
    )
