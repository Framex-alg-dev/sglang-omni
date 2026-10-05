from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.serve.timeline_detection.client_model import (
    SglangClientTimelineDetectionModel,
)
from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from sglang_omni.serve.timeline_detection.prompt import PROMPT_VERSION, SYSTEM_PROMPT


class Client:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = []

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id))
        return SimpleNamespace(text=self.outputs.pop(0))


def start() -> TimelineSessionStart:
    return TimelineSessionStart(
        session_id="session-1",
        identity_epoch=2,
        stream_epoch=0,
        audio_format="pcm_s16le/16000/1",
        video_format="image/jpeg",
        model_id="timeline-model",
    )


def audio(sequence: int = 1) -> TimelineMediaChunk:
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=2,
        stream_epoch=0,
        sequence=sequence,
        kind=MediaKind.AUDIO,
        start_ms=0,
        end_ms=3_000,
        encoding="pcm_s16le",
        payload=b"\x00\x00" * 48_000,
    )


def frame(at_ms: int, sequence: int) -> TimelineMediaChunk:
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=2,
        stream_epoch=0,
        sequence=sequence,
        kind=MediaKind.VIDEO,
        start_ms=at_ms,
        end_ms=at_ms + 100,
        encoding="image/jpeg",
        payload=f"jpeg-{at_ms}".encode(),
    )


async def feed_first_window(model: SglangClientTimelineDetectionModel) -> None:
    await model.append(audio())
    for sequence, at_ms in enumerate(range(0, 3_001, 500), start=2):
        await model.append(frame(at_ms, sequence))


async def feed_first_video_only_window(
    model: SglangClientTimelineDetectionModel,
) -> None:
    for sequence, at_ms in enumerate(range(0, 3_501, 500), start=1):
        await model.append(frame(at_ms, sequence))


@pytest.mark.asyncio
async def test_builds_event_v120_audio_video_payload_and_observations() -> None:
    client = Client(['{"e":["O2","X1"]}'])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_window(model)

    events = await model.next_observations()

    assert [item.event_type for item in events] == ["O2", "X1"]
    assert all(item.evidence_start_ms == 0 for item in events)
    assert all(item.evidence_end_ms == 3_000 for item in events)
    assert all(item.evidence_mode == "audio_video" for item in events)
    assert all(item.audio_status == "complete" for item in events)
    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.metadata["prompt_version"] == PROMPT_VERSION
    assert len(request.metadata["images"]) == 7
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")
    assert request.sampling.seed == 0
    assert request.max_tokens == 64
    assert [part["type"] for part in request.messages[1].content].count("image") == 7
    await model.close()


@pytest.mark.asyncio
async def test_builds_video_only_request_without_audio_placeholder() -> None:
    client = Client(['{"e":["P2"]}'])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_video_only_window(model)

    events = await model.next_observations()

    assert [item.event_type for item in events] == ["P2"]
    assert events[0].evidence_mode == "video_only"
    assert events[0].audio_status == "missing"
    request = client.calls[0][0]
    assert request.metadata["evidence_mode"] == "video_only"
    assert request.metadata["audio_status"] == "missing"
    assert "audios" not in request.metadata
    assert "音频状态=missing，本次为纯视觉判断" in request.messages[1].content[-1]["text"]
    assert not any(
        part["type"] == "audio" for part in request.messages[1].content
    )
    await model.close()


@pytest.mark.asyncio
async def test_retries_invalid_output_with_retry_prompt() -> None:
    client = Client(["invalid", '{"e":[]}'])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_window(model)

    assert await model.next_observations() == ()
    assert len(client.calls) == 2
    assert "格式错误" in client.calls[1][0].messages[1].content[-1]["text"]
    await model.close()


def test_accepts_one_second_sliding_cadence() -> None:
    model = SglangClientTimelineDetectionModel(
        Client([]),
        start(),
        model_version=PROMPT_VERSION,
        inference_interval_ms=1_000,
    )
    assert model.prompt_version == PROMPT_VERSION


def test_rejects_non_event_window_timing() -> None:
    with pytest.raises(ValueError, match="1s or 3s cadence"):
        SglangClientTimelineDetectionModel(
            Client([]),
            start(),
            model_version=PROMPT_VERSION,
            inference_interval_ms=500,
        )
