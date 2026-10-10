from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize('target,first_size', [(None, 12000), (250, 12000), (334, 16032), (1000, 48000)])
@pytest.mark.parametrize('total_size', [100, 50000])
async def test_first_packet_target_is_lossless_and_later_frames_stay_250ms(target, first_size, total_size):
    source = bytes(range(256)) * (total_size // 256) + bytes(range(total_size % 256))
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, headers={
            'x-tts-first-chunk-ms': str(target),
        }, stream=_AudioStream(source[:10001], source[10001:16002], source[16002:]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        synthesizer = GatewaySpeechSynthesizer(GatewaySpeechConfig(endpoint='http://tts/stream'), client=client)
        frames = []
        async def collect(frame):
            frames.append(frame)
        result = await synthesizer.synthesize(text='你好', voice='voice', instruction='', audio_sink=collect, first_chunk_ms=target)
    assert b''.join(frames) == source
    assert len(frames[0]) == min(total_size, first_size)
    assert all(len(frame) == 12000 for frame in frames[1:-1])
    assert result.audio_bytes == total_size
    assert requests[0].get('first_chunk_ms') == target
    assert ('first_chunk_ms' in requests[0]) == (target is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [True, False, 0, 39, 1001, 334.0, '334'])
async def test_first_packet_rejects_invalid_target_before_http(target):
    def reject(_):
        raise AssertionError('invalid target reached provider')
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        synthesizer = GatewaySpeechSynthesizer(GatewaySpeechConfig(endpoint='http://tts/stream'), client=client)
        with pytest.raises(ValueError, match='first_chunk_ms'):
            await synthesizer.synthesize(text='你好', voice='voice', instruction='', audio_sink=lambda _: None, first_chunk_ms=target)


@pytest.mark.asyncio
@pytest.mark.parametrize('ack', [None, '250'])
async def test_first_packet_rejects_provider_that_ignores_configuration(ack):
    headers = {} if ack is None else {'x-tts-first-chunk-ms': ack}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, content=b'\0' * 16032))) as client:
        synthesizer = GatewaySpeechSynthesizer(GatewaySpeechConfig(endpoint='http://tts/stream'), client=client)
        with pytest.raises(GatewaySpeechError, match='acknowledge'):
            await synthesizer.synthesize(text='你好', voice='voice', instruction='', audio_sink=lambda _: None, first_chunk_ms=334)


@pytest.mark.asyncio
async def test_334ms_packet_is_emitted_from_native_360ms_without_next_provider_chunk():
    first_packet = asyncio.Event()
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'a' * 17280  # One native 360 ms model chunk.
            await asyncio.wait_for(first_packet.wait(), timeout=1)
            yield b'b' * 12000
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={'x-tts-first-chunk-ms': '334'}, stream=Stream()))) as client:
        synthesizer = GatewaySpeechSynthesizer(GatewaySpeechConfig(endpoint='http://tts/stream'), client=client)
        frames = []
        async def collect(frame):
            frames.append(frame)
            first_packet.set()
        await synthesizer.synthesize(text='hello', voice='voice', instruction='', audio_sink=collect, first_chunk_ms=334)
    assert list(map(len, frames)) == [16032, 12000, 1248]
    assert b''.join(frames) == b'a' * 17280 + b'b' * 12000
