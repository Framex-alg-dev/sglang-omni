# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.dev_model import (
    DevRealtimeModelClient,
    DevRealtimeModelConfig,
)
from sglang_omni.serve.realtime.dev_server import create_dev_app
from sglang_omni.serve.realtime.embedded_tts import EmbeddedTTSConfig


class ProgrammableTTSWebSocket:
    def __init__(
        self, *, invalid_audio: bool = False, audio_payload: bytes = b"\x01\x02"
    ) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.events.put_nowait(
            json.dumps({"type": "session.created", "session": {"id": "provider"}})
        )
        self.invalid_audio = invalid_audio
        self.audio_payload = audio_payload
        self.active = False
        self.sent: list[dict[str, Any]] = []
        self.committed_texts: list[str] = []
        self._parts: list[str] = []

    async def send(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "input_text_buffer.append":
            self._parts.append(event["text"])
            if not self.active:
                self.active = True
                self.events.put_nowait(
                    json.dumps(
                        {"type": "response.created", "response_id": "provider-resp"}
                    )
                )
                encoded = (
                    "invalid-base64"
                    if self.invalid_audio
                    else base64.b64encode(self.audio_payload).decode("ascii")
                )
                self.events.put_nowait(
                    json.dumps({"type": "response.audio.delta", "delta": encoded})
                )
        elif event["type"] == "input_text_buffer.commit":
            self.committed_texts.append("".join(self._parts))
            self._parts.clear()
            self.active = False
            self.events.put_nowait(json.dumps({"type": "response.audio.done"}))
            self.events.put_nowait(json.dumps({"type": "response.done"}))

    async def recv(self) -> str:
        return await self.events.get()


class ProgrammableTTSContext:
    def __init__(self, websocket: ProgrammableTTSWebSocket) -> None:
        self.websocket = websocket
        self.closed = False

    async def __aenter__(self) -> ProgrammableTTSWebSocket:
        return self.websocket

    async def __aexit__(self, *args: object) -> None:
        self.closed = True


class ProgrammableTTSConnector:
    def __init__(
        self, *, invalid_audio: bool = False, audio_payload: bytes = b"\x01\x02"
    ) -> None:
        self.invalid_audio = invalid_audio
        self.audio_payload = audio_payload
        self.contexts: list[ProgrammableTTSContext] = []

    def __call__(self, url: str, **kwargs: object) -> ProgrammableTTSContext:
        del url, kwargs
        context = ProgrammableTTSContext(
            ProgrammableTTSWebSocket(
                invalid_audio=self.invalid_audio,
                audio_payload=self.audio_payload,
            )
        )
        self.contexts.append(context)
        return context


def _app(connector: ProgrammableTTSConnector, *, interval_ms: int = 0):
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="固定流式回复",
            chunk_size=2,
            chunk_interval_ms=interval_ms,
        )
    )
    return create_dev_app(
        client,
        model_name="dev-model",
        embedded_tts_config=EmbeddedTTSConfig(
            url="ws://tts.local/realtime", voice="test-voice"
        ),
        embedded_tts_connector=connector,
    )


def _fusion_app(
    connector: ProgrammableTTSConnector,
    *,
    provisional_audio_max_bytes: int = 8 * 1024 * 1024,
    provisional_audio_max_milliseconds: int = 10000,
):
    class DelayedActionClient(DevRealtimeModelClient):
        async def score_action_suffixes(self, request: Any):
            await asyncio.sleep(0.05)
            return await super().score_action_suffixes(request)

    client = DelayedActionClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="固定流式回复",
            chunk_size=2,
        )
    )
    return create_dev_app(
        client,
        model_name="dev-model",
        embedded_tts_config=EmbeddedTTSConfig(
            url="ws://tts.local/realtime",
            voice="test-voice",
            provisional_audio_max_bytes=provisional_audio_max_bytes,
            provisional_audio_max_milliseconds=(provisional_audio_max_milliseconds),
        ),
        embedded_tts_connector=connector,
    )


def _start_session(ws: Any) -> None:
    ws.send_json(
        {
            "type": "session.start",
            "protocol_version": 1,
            "session_id": "tts-session",
            "outputs": ["text", "audio"],
            "locale": "zh-CN",
            "reply": {"instructions": "简短回复"},
        }
    )
    started = ws.receive_json()
    assert started["type"] == "session.started"
    assert started["outputs"] == ["text", "audio"]


def _start_fusion_session(ws: Any) -> None:
    ws.send_json(
        {
            "type": "session.start",
            "protocol_version": 1,
            "session_id": "tts-fusion-session",
            "outputs": ["text", "audio", "action"],
            "locale": "zh-CN",
            "reply": {
                "instructions": "简短回复",
                "unsupported_action_text": "不支持该动作",
            },
            "action": {
                "fallback_category_ids": ["BDEV"],
                "allowed_candidates": [{"candidate_id": "ADEV"}],
            },
        }
    )
    started = ws.receive_json()
    assert started["type"] == "session.started"
    assert started["outputs"] == ["text", "audio", "action"]


def _run_turn(ws: Any, turn_id: str) -> list[dict[str, Any]]:
    ws.send_json({"type": "turn.start", "turn_id": turn_id, "origin": "user"})
    assert ws.receive_json()["type"] == "turn.started"
    ws.send_json({"type": "input.text.set", "turn_id": turn_id, "text": "你好"})
    assert ws.receive_json()["type"] == "input.text.ack"
    ws.send_json({"type": "turn.commit", "turn_id": turn_id})
    events: list[dict[str, Any]] = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in {"turn.result", "error", "turn.cancelled"}:
            return events


def test_text_audio_streams_pcm_and_reuses_connection_across_turns() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(_app(connector)).websocket_connect("/v1/session/realtime") as ws:
        _start_session(ws)
        first = _run_turn(ws, "turn-1")
        second = _run_turn(ws, "turn-2")

    assert len(connector.contexts) == 1
    assert connector.contexts[0].websocket.committed_texts == [
        "固定流式回复",
        "固定流式回复",
    ]
    for events in (first, second):
        types = [event["type"] for event in events]
        assert types.index("response.created") < types.index("response.text.done")
        assert types.index("response.text.done") < types.index("response.audio.done")
        assert types.index("response.audio.done") < types.index("response.done")
        assert types.index("response.done") < types.index("turn.result")
        assert (
            "".join(
                event["delta"]
                for event in events
                if event["type"] == "response.text.delta"
            )
            == "固定流式回复"
        )
        audio_events = [
            event for event in events if event["type"] == "response.audio.delta"
        ]
        assert [event["seq"] for event in audio_events] == [1]
        assert base64.b64decode(audio_events[0]["delta"]) == b"\x01\x02"
        assert audio_events[0]["audio"] == {
            "format": "pcm16le",
            "sample_rate_hz": 24000,
            "channels": 1,
        }
        assert events[-1]["outputs"] == {"text": "completed", "audio": "completed"}


def test_provider_protocol_failure_fails_turn_without_success_terminal() -> None:
    connector = ProgrammableTTSConnector(invalid_audio=True)
    with TestClient(_app(connector)).websocket_connect("/v1/session/realtime") as ws:
        _start_session(ws)
        events = _run_turn(ws, "turn-failed")

    types = [event["type"] for event in events]
    assert types[-1] == "error"
    assert "response.audio.done" not in types
    assert "response.done" not in types
    assert "turn.result" not in types
    assert connector.contexts[0].closed is True
    assert "response.cancel" in {
        event["type"] for event in connector.contexts[0].websocket.sent
    }


def test_text_only_session_has_zero_tts_side_effects() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(_app(connector)).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "protocol_version": 1,
                "session_id": "text-session",
                "outputs": ["text"],
                "locale": "zh-CN",
                "reply": {"instructions": "简短回复"},
            }
        )
        assert ws.receive_json()["type"] == "session.started"
        events = _run_turn(ws, "turn-text")

    assert events[-1]["type"] == "turn.result"
    assert connector.contexts == []


def test_cancel_closes_provider_and_next_turn_reconnects() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(_app(connector, interval_ms=100)).websocket_connect(
        "/v1/session/realtime"
    ) as ws:
        _start_session(ws)
        ws.send_json({"type": "turn.start", "turn_id": "turn-cancel", "origin": "user"})
        assert ws.receive_json()["type"] == "turn.started"
        ws.send_json(
            {"type": "input.text.set", "turn_id": "turn-cancel", "text": "你好"}
        )
        assert ws.receive_json()["type"] == "input.text.ack"
        ws.send_json({"type": "turn.commit", "turn_id": "turn-cancel"})
        while True:
            event = ws.receive_json()
            if event["type"] == "response.audio.delta":
                break
        ws.send_json({"type": "turn.cancel", "turn_id": "turn-cancel"})
        cancelled_events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            cancelled_events.append(event)
            if event["type"] == "turn.cancelled":
                break
        next_turn = _run_turn(ws, "turn-after-cancel")

    assert connector.contexts[0].closed is True
    assert "response.cancel" in {
        event["type"] for event in connector.contexts[0].websocket.sent
    }
    assert len(connector.contexts) == 2
    assert next_turn[-1]["type"] == "turn.result"
    forbidden = {"response.audio.done", "response.done", "turn.result"}
    assert forbidden.isdisjoint(
        event["type"]
        for event in cancelled_events
        if event.get("turn_id") == "turn-cancel"
    )


def test_fusion_buffers_audio_until_promotion_and_synthesizes_text_once() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(_fusion_app(connector)).websocket_connect(
        "/v1/session/realtime"
    ) as ws:
        _start_fusion_session(ws)
        events = _run_turn(ws, "turn-fusion")

    types = [event["type"] for event in events]
    promoted_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.provisional.resolved"
        and event["status"] == "promoted"
    )
    official_created_index = types.index("response.created")
    audio_indexes = [
        index
        for index, event in enumerate(events)
        if event["type"] == "response.audio.delta"
    ]
    assert audio_indexes
    assert min(audio_indexes) > promoted_index
    assert min(audio_indexes) > official_created_index
    assert types.index("response.audio.done") < types.index("response.done")
    assert types.index("response.done") < types.index("turn.result")
    assert events[-1]["outputs"] == {
        "text": "completed",
        "audio": "completed",
        "action": "completed",
    }
    provider = connector.contexts[0].websocket
    assert provider.committed_texts == ["固定流式回复"]
    assert (
        "".join(
            event["text"]
            for event in provider.sent
            if event["type"] == "input_text_buffer.append"
        )
        == "固定流式回复"
    )


def test_fusion_provisional_audio_overflow_fails_without_audio_leak() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(
        _fusion_app(connector, provisional_audio_max_bytes=1)
    ).websocket_connect("/v1/session/realtime") as ws:
        _start_fusion_session(ws)
        events = _run_turn(ws, "turn-overflow")

    types = [event["type"] for event in events]
    assert types[-1] == "error"
    assert "response.audio.delta" not in types
    assert "response.audio.done" not in types
    assert "response.done" not in types
    assert "turn.result" not in types
    assert connector.contexts[0].closed is True


def test_fusion_provisional_audio_duration_limit_uses_pcm_duration() -> None:
    connector = ProgrammableTTSConnector(audio_payload=b"\x00" * 96)
    with TestClient(
        _fusion_app(connector, provisional_audio_max_milliseconds=1)
    ).websocket_connect("/v1/session/realtime") as ws:
        _start_fusion_session(ws)
        events = _run_turn(ws, "turn-duration-overflow")

    types = [event["type"] for event in events]
    assert types[-1] == "error"
    assert "response.audio.delta" not in types
    assert "response.audio.done" not in types
    assert "response.done" not in types
    assert "turn.result" not in types
    assert connector.contexts[0].closed is True


def test_tts_failure_aborts_model_while_model_stream_is_stalled() -> None:
    class StalledModelClient(DevRealtimeModelClient):
        def __init__(self) -> None:
            super().__init__(DevRealtimeModelConfig(enabled=True))
            self.abort_calls: list[str] = []

        def completion_stream(self, request: Any, *, request_id: str, **kwargs: Any):
            del request, kwargs

            async def stream():
                self._active_request_ids.add(request_id)
                try:
                    yield CompletionStreamChunk(
                        request_id=request_id, modality="text", text="首段"
                    )
                    await asyncio.Event().wait()
                finally:
                    self._active_request_ids.discard(request_id)

            return stream()

        async def abort(self, request_id: str):
            self.abort_calls.append(request_id)
            return await super().abort(request_id)

    connector = ProgrammableTTSConnector(invalid_audio=True)
    client = StalledModelClient()
    app = create_dev_app(
        client,
        model_name="dev-model",
        embedded_tts_config=EmbeddedTTSConfig(
            url="ws://tts.local/realtime", voice="test-voice"
        ),
        embedded_tts_connector=connector,
    )
    with TestClient(app).websocket_connect("/v1/session/realtime") as ws:
        _start_session(ws)
        events = _run_turn(ws, "turn-stalled-model")

    assert events[-1]["type"] == "error"
    assert len(client.abort_calls) == 1
    assert client.abort_calls[0].endswith("-reply")
    assert connector.contexts[0].closed is True


def test_fusion_cancel_discards_buffered_audio_without_leak() -> None:
    connector = ProgrammableTTSConnector()
    with TestClient(_fusion_app(connector)).websocket_connect(
        "/v1/session/realtime"
    ) as ws:
        _start_fusion_session(ws)
        ws.send_json(
            {"type": "turn.start", "turn_id": "turn-cancel-fusion", "origin": "user"}
        )
        assert ws.receive_json()["type"] == "turn.started"
        ws.send_json(
            {
                "type": "input.text.set",
                "turn_id": "turn-cancel-fusion",
                "text": "你好",
            }
        )
        assert ws.receive_json()["type"] == "input.text.ack"
        ws.send_json({"type": "turn.commit", "turn_id": "turn-cancel-fusion"})
        before_cancel: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            before_cancel.append(event)
            if event["type"] == "response.provisional.text.delta":
                break
        ws.send_json({"type": "turn.cancel", "turn_id": "turn-cancel-fusion"})
        after_cancel: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            after_cancel.append(event)
            if event["type"] == "turn.cancelled":
                break

    all_types = [event["type"] for event in before_cancel + after_cancel]
    assert "response.audio.delta" not in all_types
    assert "response.audio.done" not in all_types
    assert "response.done" not in all_types
    assert "turn.result" not in all_types
    assert connector.contexts[0].closed is True


def test_fusion_unsupported_action_discards_online_tts_audio() -> None:
    connector = ProgrammableTTSConnector()
    app = _fusion_app(connector)
    manager = app.state.multimodal_realtime_manager
    original_create = manager.create

    def create_unsupported_session(websocket: Any):
        session = original_create(websocket)

        async def score_unsupported(*args: Any, **kwargs: Any):
            del args
            callback = kwargs.get("on_category_selected")
            if callback is not None:
                callback(None, "unsupported")
            return (
                {
                    "candidate_id": "ADEV",
                    "action_id": "ADEV",
                    "category_id": "BDEV",
                    "execution_binding": {},
                    "execute": False,
                    "support_status": "unsupported",
                },
                [],
                0.0,
                {"category_decision_id": "UNSUPPORTED"},
            )

        session._score_action = score_unsupported
        return session

    manager.create = create_unsupported_session
    with TestClient(app).websocket_connect("/v1/session/realtime") as ws:
        _start_fusion_session(ws)
        events = _run_turn(ws, "turn-unsupported")

    types = [event["type"] for event in events]
    assert "response.audio.delta" not in types
    assert "response.audio.done" not in types
    assert "response.done" not in types
    assert events[-1]["type"] == "turn.result"
    assert events[-1]["reply"]["source"] == "client_prerecorded_audio"
    assert connector.contexts[0].closed is True
