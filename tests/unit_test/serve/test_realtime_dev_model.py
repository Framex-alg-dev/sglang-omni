# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.dev_model import (
    DevRealtimeModelClient,
    DevRealtimeModelConfig,
    DevRealtimeModelRequestError,
    install_dev_model_error_handler,
)
from sglang_omni.serve.realtime.vad import VADEvent


def _request(*, transcription: bool = False) -> GenerateRequest:
    instruction = (
        "Transcribe the spoken audio."
        if transcription
        else "Listen to the spoken audio above and respond to it."
    )
    return GenerateRequest(
        messages=[
            Message(role="system", content="development test"),
            Message(role="user", content=instruction),
        ],
        stream=True,
        output_modalities=["text"],
        metadata={"audios": ["data:audio/wav;base64,UklGRg=="]},
    )


def test_config_defaults_to_disabled_and_parses_strict_values() -> None:
    assert DevRealtimeModelConfig.from_env({}).enabled is False
    config = DevRealtimeModelConfig.from_env(
        {
            "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED": "true",
            "SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT": "回复",
            "SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT": "转写",
            "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE": "1",
            "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS": "2",
        }
    )
    assert config == DevRealtimeModelConfig(
        enabled=True,
        response_text="回复",
        transcript_text="转写",
        chunk_size=1,
        chunk_interval_ms=2,
    )
    assert "回复" not in str(config.log_summary())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ENABLED", "1", "must be 'true' or 'false'"),
        ("CHUNK_SIZE", "0", "must be >= 1"),
        ("CHUNK_INTERVAL_MS", "-1", "must be >= 0"),
        ("CHUNK_SIZE", "one", "must be an integer"),
        ("RESPONSE_TEXT", "", "must not be empty"),
    ],
)
def test_config_rejects_invalid_values(name: str, value: str, message: str) -> None:
    env = {
        "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED": "true",
        f"SGLANG_OMNI_DEV_FAKE_MODEL_{name}": value,
    }
    with pytest.raises(ValueError, match=message):
        DevRealtimeModelConfig.from_env(env)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcription", "expected"),
    [(False, "固定回复文本"), (True, "固定转写文本")],
)
async def test_client_streams_selected_text_and_terminal_chunk(
    transcription: bool, expected: str
) -> None:
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="固定回复文本",
            transcript_text="固定转写文本",
            chunk_size=2,
        )
    )
    chunks = [
        chunk
        async for chunk in client.completion_stream(
            _request(transcription=transcription), request_id="request-1"
        )
    ]
    assert "".join(chunk.text for chunk in chunks) == expected
    assert all(chunk.request_id == "request-1" for chunk in chunks)
    assert all(chunk.modality == "text" for chunk in chunks)
    assert all(chunk.finish_reason is None for chunk in chunks[:-1])
    assert chunks[-1].text == ""
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_abort_stops_an_active_stream() -> None:
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(enabled=True, response_text="abcdef", chunk_size=1)
    )
    stream = client.completion_stream(_request(), request_id="request-1")
    first = await anext(stream)
    result = await client.abort("request-1")
    remaining = [chunk async for chunk in stream]
    unknown = await client.abort("unknown")
    assert first.text == "a"
    assert result.success is True
    assert remaining == []
    assert unknown.success is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda req: setattr(req, "stream", False), "stream must be true"),
        (
            lambda req: setattr(req, "output_modalities", ["audio"]),
            "output_modalities",
        ),
        (lambda req: setattr(req, "messages", []), "messages must not be empty"),
        (lambda req: req.metadata.clear(), "metadata.audios"),
        (
            lambda req: req.metadata.update({"audios": ["not-a-data-uri"]}),
            "audio data URIs",
        ),
        (
            lambda req: req.metadata.update({"audios": ["data:audio/wav"]}),
            "audio data URIs",
        ),
        (
            lambda req: req.metadata.update(
                {"audios": ["data:audio/wav;base64,not base64"]}
            ),
            "audio data URIs",
        ),
        (
            lambda req: setattr(
                req,
                "messages",
                [Message(role="system", content="unknown")],
            ),
            "system and user",
        ),
    ],
)
async def test_client_rejects_invalid_realtime_contract(mutate, message: str) -> None:
    client = DevRealtimeModelClient(DevRealtimeModelConfig(enabled=True))
    request = _request()
    mutate(request)
    with pytest.raises(DevRealtimeModelRequestError, match=message):
        await anext(client.completion_stream(request, request_id="request-1"))


@pytest.mark.asyncio
async def test_client_rejects_empty_id_and_unknown_pass() -> None:
    client = DevRealtimeModelClient(DevRealtimeModelConfig(enabled=True))
    with pytest.raises(DevRealtimeModelRequestError, match="request_id"):
        await anext(client.completion_stream(_request(), request_id=" "))

    request = _request()
    request.messages[-1].content = "unknown prompt"
    with pytest.raises(DevRealtimeModelRequestError, match="unable to classify"):
        await anext(client.completion_stream(request, request_id="request-1"))


def test_client_rejects_unsupported_non_realtime_operations() -> None:
    client = DevRealtimeModelClient(DevRealtimeModelConfig(enabled=True))
    with pytest.raises(RuntimeError, match="unsupported"):
        client.generate


def test_interval_is_applied_only_between_text_chunks(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="abcd",
            chunk_size=2,
            chunk_interval_ms=25,
        )
    )

    async def consume() -> None:
        _ = [
            chunk
            async for chunk in client.completion_stream(
                _request(), request_id="request-1"
            )
        ]

    asyncio.run(consume())
    assert sleeps == [0.025]


@pytest.fixture
def dev_app_client() -> TestClient:
    model = DevRealtimeModelClient(DevRealtimeModelConfig(enabled=True))
    app = create_app(model, model_name="dev-model", enable_realtime=True)
    install_dev_model_error_handler(app)
    return TestClient(app, raise_server_exceptions=False)


def test_supported_http_endpoints_remain_available(dev_app_client: TestClient) -> None:
    assert dev_app_client.get("/health").status_code == 200
    assert dev_app_client.get("/v1/models").status_code == 200


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/model_info", None),
        (
            "/v1/chat/completions",
            {"model": "dev-model", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/generate", {"prompt": "hi"}),
        ("/v1/audio/speech", {"model": "dev-model", "input": "hi"}),
        ("/v1/audio/transcriptions", {}),
    ],
)
def test_unsupported_http_endpoints_return_explicit_development_error(
    dev_app_client: TestClient, path: str, payload: dict[str, object] | None
) -> None:
    response = (
        dev_app_client.get(path)
        if payload is None
        else dev_app_client.post(path, json=payload)
    )

    assert response.status_code == 501
    assert response.json() == {
        "detail": f"{path} is unsupported while the Realtime development model is enabled",
        "type": "dev_fake_model_unsupported",
    }


def test_realtime_websocket_audio_runs_response_and_transcription(monkeypatch) -> None:
    from sglang_omni.serve.realtime import session as session_module

    @dataclass
    class Emit:
        event_type: str
        sample_offset: int

    class DeterministicVAD:
        """Exercise the VAD boundary without loading Silero weights."""

        def __init__(self, config) -> None:
            del config

        def process(self, pcm_bytes: bytes) -> list[Emit]:
            assert pcm_bytes
            samples = len(pcm_bytes) // 2
            return [
                Emit(VADEvent.SPEECH_STARTED, 0),
                Emit(VADEvent.SPEECH_STOPPED, samples),
            ]

        def reset(self) -> None:
            pass

    monkeypatch.setattr(session_module, "StreamingVAD", DeterministicVAD)
    model = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="固定回复",
            transcript_text="固定转写",
            chunk_size=2,
        )
    )
    app = create_app(model, model_name="dev-model", enable_realtime=True)
    install_dev_model_error_handler(app)
    pcm = b"\x00\x01" * 512

    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] == (
                "conversation.item.input_audio_transcription.completed"
            ):
                break

    event_types = [event["type"] for event in events]
    assert event_types[:3] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
    ]
    assert "response.done" in event_types
    assert "conversation.item.input_audio_transcription.delta" in event_types
    response_done = next(event for event in events if event["type"] == "response.done")
    transcript_done = events[-1]
    assert response_done["response"]["output"][0]["content"] == [
        {"type": "text", "text": "固定回复"}
    ]
    assert transcript_done["transcript"] == "固定转写"
