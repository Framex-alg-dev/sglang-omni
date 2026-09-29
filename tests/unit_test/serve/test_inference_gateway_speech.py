from __future__ import annotations

import json

import httpx
import pytest

from sglang_omni.serve.inference_gateway.speech_synthesis import (
    GatewaySpeechConfig,
    GatewaySpeechError,
    GatewaySpeechSynthesizer,
)


class _AudioStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_mixed_language_final_reply_is_one_upstream_request() -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={
                "x-audio-sample-rate": "24000",
                "x-request-id": "provider-1",
            },
            stream=_AudioStream(b"a" * 5_000, b"b" * 9_000, b"c" * 10_000),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    synthesizer = GatewaySpeechSynthesizer(
        GatewaySpeechConfig(endpoint="http://tts/v1/tts/stream"),
        client=client,
    )
    frames: list[bytes] = []

    async def collect(frame: bytes) -> None:
        frames.append(frame)

    text = (
        "可以。我这里有：With or Without You、"
        "Two Hearts Beat As One、EEヨ。你想选哪一个？"
    )
    try:
        result = await synthesizer.synthesize(
            text=text,
            voice="spk-1",
            instruction="自然地说",
            audio_sink=collect,
        )
    finally:
        await client.aclose()

    assert requests == [
        {
            "text": text,
            "speaker_id": "spk-1",
            "seed": 0,
            "instruct": "自然地说",
        }
    ]
    assert [len(frame) for frame in frames] == [12_000, 12_000]
    assert b"".join(frames) == b"a" * 5_000 + b"b" * 9_000 + b"c" * 10_000
    assert result.audio_bytes == 24_000
    assert result.chunk_count == 2
    assert result.provider_response_id == "provider-1"


@pytest.mark.asyncio
async def test_provider_failure_is_safe_and_retryable() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    )
    synthesizer = GatewaySpeechSynthesizer(
        GatewaySpeechConfig(endpoint="http://tts/v1/tts/stream"),
        client=client,
    )
    try:
        with pytest.raises(GatewaySpeechError) as error:
            await synthesizer.synthesize(
                text="你好",
                voice="spk-1",
                instruction="自然地说",
                audio_sink=lambda _frame: None,
            )
    finally:
        await client.aclose()

    assert error.value.phase == "provider"
    assert error.value.retryable


@pytest.mark.asyncio
async def test_audio_budget_stops_unbounded_provider_stream() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                stream=_AudioStream(b"a" * 20),
            )
        )
    )
    synthesizer = GatewaySpeechSynthesizer(
        GatewaySpeechConfig(
            endpoint="http://tts/v1/tts/stream",
            frame_bytes=8,
            max_audio_bytes=16,
        ),
        client=client,
    )
    try:
        with pytest.raises(GatewaySpeechError) as error:
            await synthesizer.synthesize(
                text="你好",
                voice="spk-1",
                instruction="",
                audio_sink=lambda _frame: None,
            )
    finally:
        await client.aclose()

    assert error.value.phase == "protocol"
    assert not error.value.retryable
