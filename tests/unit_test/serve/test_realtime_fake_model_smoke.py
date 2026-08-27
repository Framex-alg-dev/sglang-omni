# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import pytest

from scripts import realtime_fake_model_smoke as smoke


class FakeWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self.events = iter(events)
        self.sent: list[dict] = []

    async def recv(self) -> str:
        try:
            event = next(self.events)
        except StopIteration:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        return json.dumps(event)

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class FakeConnection:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *args) -> None:
        return None


@pytest.mark.asyncio
async def test_text_smoke_uses_session_protocol_and_cancel() -> None:
    websocket = FakeWebSocket(
        [
            {"type": "session.started"},
            {"type": "turn.started"},
            {"type": "input.text.ack"},
            {"type": "turn.committed"},
            {"type": "response.created"},
            {"type": "response.text.delta", "delta": "固定回复"},
            {"type": "response.text.done", "text": "固定回复"},
            {"type": "response.done"},
            {"type": "turn.result", "reply": {"text": "固定回复"}},
            {"type": "turn.started"},
            {"type": "input.text.ack"},
            {"type": "turn.committed"},
            {"type": "turn.cancelled"},
        ]
    )

    def connect(url: str, *, open_timeout: float) -> FakeConnection:
        assert url.endswith("/v1/session/realtime")
        assert open_timeout == 3
        return FakeConnection(websocket)

    events = await smoke.run_smoke(
        url="ws://test/v1/session/realtime",
        mode="text",
        response_text="固定回复",
        text="你好",
        pcm=None,
        timeout=3,
        connect=connect,
    )
    assert events[-1]["type"] == "turn.cancelled"
    sent_types = [event["type"] for event in websocket.sent]
    assert sent_types[:4] == [
        "session.start",
        "turn.start",
        "input.text.set",
        "turn.commit",
    ]
    assert sent_types[-1] == "turn.cancel"


def test_parser_defaults_to_session_realtime() -> None:
    args = smoke.build_parser().parse_args([])
    assert args.url.endswith("/v1/session/realtime")
    assert args.mode == "text"


def _text_audio_events(*, seq: int = 1) -> list[dict]:
    return [
        {
            "type": "session.started",
            "session_id": "dev-smoke-text-audio",
            "outputs": ["text", "audio"],
        },
        {"type": "turn.committed", "turn_id": "turn-1"},
        {
            "type": "response.created",
            "session_id": "dev-smoke-text-audio",
            "turn_id": "turn-1",
            "response": {"id": "response-1"},
        },
        {"type": "response.text.delta", "delta": "reply"},
        {
            "type": "response.audio.delta",
            "session_id": "dev-smoke-text-audio",
            "turn_id": "turn-1",
            "response_id": "response-1",
            "seq": seq,
            "delta": "AQI=",
            "audio": {
                "format": "pcm16le",
                "sample_rate_hz": 24000,
                "channels": 1,
            },
        },
        {"type": "response.text.done", "text": "reply"},
        {
            "type": "response.audio.done",
            "session_id": "dev-smoke-text-audio",
            "turn_id": "turn-1",
            "response_id": "response-1",
            "seq": 1,
        },
        {
            "type": "response.done",
            "session_id": "dev-smoke-text-audio",
            "turn_id": "turn-1",
            "response": {"id": "response-1", "status": "completed"},
        },
        {
            "type": "turn.result",
            "session_id": "dev-smoke-text-audio",
            "turn_id": "turn-1",
            "status": "completed",
            "outputs": {"text": "completed", "audio": "completed"},
            "reply": {"text": "reply"},
        },
    ]


def test_text_audio_validation_checks_pcm_sequence_and_correlation() -> None:
    smoke._validate_turn_events(
        _text_audio_events(), mode="text-audio", response_text="reply"
    )


def test_text_audio_validation_rejects_wrong_audio_sequence() -> None:
    with pytest.raises(AssertionError, match="audio seq"):
        smoke._validate_turn_events(
            _text_audio_events(seq=2), mode="text-audio", response_text="reply"
        )


def test_text_audio_validation_rejects_incomplete_terminal_status() -> None:
    events = _text_audio_events()
    events[-1]["outputs"]["audio"] = "failed"
    with pytest.raises(AssertionError, match="turn result terminal"):
        smoke._validate_turn_events(events, mode="text-audio", response_text="reply")


def test_parser_rejects_non_positive_timeout() -> None:
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(["--timeout", "0"])


def test_turn_validation_rejects_out_of_order_events() -> None:
    events = [
        {"type": "session.started"},
        {"type": "response.created"},
        {"type": "response.text.done", "text": "回复"},
        {"type": "turn.committed"},
        {"type": "response.text.delta", "delta": "回复"},
        {"type": "response.done"},
        {"type": "turn.result", "reply": {"text": "回复"}},
    ]
    with pytest.raises(AssertionError, match="out of order"):
        smoke._validate_turn_events(events, mode="text", response_text="回复")


def test_fusion_allows_action_ready_during_text_stream() -> None:
    events = [
        {"type": "session.started"},
        {"type": "turn.committed"},
        {"type": "response.created"},
        {"type": "response.text.delta", "delta": "固"},
        {"type": "turn.action.ready", "action": {"candidate_id": "ADEV"}},
        {"type": "response.text.delta", "delta": "定回复"},
        {"type": "response.text.done", "text": "固定回复"},
        {"type": "response.done"},
        {
            "type": "turn.result",
            "reply": {"text": "固定回复"},
            "action": {"candidate_id": "ADEV"},
        },
    ]

    smoke._validate_turn_events(events, mode="fusion", response_text="固定回复")


def test_fusion_rejects_text_delta_after_text_done() -> None:
    events = [
        {"type": "session.started"},
        {"type": "turn.committed"},
        {"type": "response.created"},
        {"type": "response.text.done", "text": "固定回复"},
        {"type": "turn.action.ready", "action": {"candidate_id": "ADEV"}},
        {"type": "response.text.delta", "delta": "固定回复"},
        {"type": "response.done"},
        {
            "type": "turn.result",
            "reply": {"text": "固定回复"},
            "action": {"candidate_id": "ADEV"},
        },
    ]

    with pytest.raises(AssertionError, match="out of order"):
        smoke._validate_turn_events(events, mode="fusion", response_text="固定回复")


def test_audio_loader_rejects_empty_wav(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"")
    with pytest.raises(ValueError, match="at least one PCM frame"):
        smoke.load_pcm16_16k_mono(path)
