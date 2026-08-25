# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from scripts import realtime_fake_model_smoke as smoke


def _events(response_text: str, transcript_text: str) -> list[dict]:
    return [
        {"type": "session.created"},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created"},
        {"type": "response.text.delta", "delta": response_text},
        {"type": "response.text.done", "text": response_text},
        {
            "type": "response.done",
            "response": {
                "output": [{"content": [{"type": "text", "text": response_text}]}]
            },
        },
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": transcript_text,
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": transcript_text,
        },
    ]


class FakeWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self.events = iter(events)
        self.sent: list[dict] = []

    async def recv(self) -> str:
        return json.dumps(next(self.events))

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
async def test_smoke_sends_audio_and_validates_expected_events() -> None:
    websocket = FakeWebSocket(_events("固定回复", "固定转写"))
    connect_calls: list[tuple[str, float]] = []

    def connect(url: str, *, open_timeout: float) -> FakeConnection:
        connect_calls.append((url, open_timeout))
        return FakeConnection(websocket)

    events = await smoke.run_smoke(
        url="ws://127.0.0.1:8123/v1/realtime",
        pcm=b"\x01\x00" * 3200,
        response_text="固定回复",
        transcript_text="固定转写",
        timeout=3,
        connect=connect,
    )
    assert connect_calls == [("ws://127.0.0.1:8123/v1/realtime", 3)]
    assert events[-1]["type"].endswith("transcription.completed")
    assert websocket.sent
    assert all(event["type"] == "input_audio_buffer.append" for event in websocket.sent)


@pytest.mark.asyncio
async def test_smoke_reports_text_mismatch() -> None:
    websocket = FakeWebSocket(_events("实际回复", "固定转写"))

    def connect(url: str, *, open_timeout: float) -> FakeConnection:
        del url, open_timeout
        return FakeConnection(websocket)

    with pytest.raises(AssertionError, match="response mismatch"):
        await smoke.run_smoke(
            url="ws://test/v1/realtime",
            pcm=b"\x01\x00",
            response_text="预期回复",
            transcript_text="固定转写",
            timeout=1,
            connect=connect,
        )


def test_validate_events_reports_missing_contract_event() -> None:
    with pytest.raises(AssertionError, match="missing events"):
        smoke.validate_events(
            [{"type": "session.created"}],
            response_text="回复",
            transcript_text="转写",
        )


def test_validate_events_rejects_out_of_order_and_partial_deltas() -> None:
    out_of_order = _events("回复", "转写")
    out_of_order[4], out_of_order[5] = out_of_order[5], out_of_order[4]
    with pytest.raises(AssertionError, match="out of order"):
        smoke.validate_events(
            out_of_order, response_text="回复", transcript_text="转写"
        )

    partial = _events("回复", "转写")
    partial[5]["delta"] = "回"
    with pytest.raises(AssertionError, match="response stream mismatch"):
        smoke.validate_events(partial, response_text="回复", transcript_text="转写")


def test_audio_loader_requires_pcm16_16k_mono_and_appends_silence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 10)
    pcm = smoke.load_pcm16_16k_mono(path)
    assert pcm[:20] == b"\x01\x00" * 10
    assert pcm[20:] == b"\x00\x00" * 16000


def test_audio_loader_rejects_empty_wav(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"")
    with pytest.raises(ValueError, match="at least one PCM frame"):
        smoke.load_pcm16_16k_mono(path)


def test_main_returns_nonzero_with_invalid_audio(tmp_path: Path, capsys) -> None:
    result = smoke.main(
        [
            "--audio",
            str(tmp_path / "missing.wav"),
            "--response-text",
            "回复",
            "--transcript-text",
            "转写",
        ]
    )
    assert result == 1
    assert "FAIL:" in capsys.readouterr().err


def test_main_reports_unexpected_connection_failure(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00")

    async def fail_run_smoke(**kwargs):
        del kwargs
        raise RuntimeError("connection closed unexpectedly")

    monkeypatch.setattr(smoke, "run_smoke", fail_run_smoke)
    result = smoke.main(
        [
            "--audio",
            str(path),
            "--response-text",
            "回复",
            "--transcript-text",
            "转写",
        ]
    )
    assert result == 1
    assert "FAIL: connection closed unexpectedly" in capsys.readouterr().err


def test_parser_rejects_non_positive_timeout() -> None:
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(
            ["--response-text", "回复", "--transcript-text", "转写", "--timeout", "0"]
        )
