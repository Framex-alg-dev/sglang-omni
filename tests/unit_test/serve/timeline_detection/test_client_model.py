from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang_omni.serve.timeline_detection import client_model as client_model_module
from sglang_omni.serve.timeline_detection.client_model import SglangClientTimelineDetectionModel
from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile


class Client:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = []

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id))
        return SimpleNamespace(text=self.outputs.pop(0))


class PrefillClient(Client):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs)
        self.prefill_calls = []

    async def prefill_completion_prefix(self, request, *, request_id):
        self.prefill_calls.append((request, request_id))
        return True


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


@lru_cache(maxsize=1)
def jpeg_frame() -> bytes:
    output = BytesIO()
    Image.new("RGB", (640, 480), (48, 96, 144)).save(
        output,
        format="JPEG",
        quality=80,
    )
    return output.getvalue()


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
        payload=jpeg_frame(),
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
async def test_direct_e50_prewarms_static_prefix_before_inference(
    monkeypatch,
) -> None:
    profile = DirectE50Profile(
        system_prompt="strict training system prompt",
        user_prompt="strict training user prompt",
        prompt_version="event-e50-direct-checkpoint1500-v1",
    )
    monkeypatch.setattr(
        client_model_module.DirectE50Profile,
        "from_environment",
        classmethod(lambda cls: profile),
    )
    client = PrefillClient([])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version="e50-direct-checkpoint-1500-fp8-sidecar-fixed",
        inference_interval_ms=1_000,
    )

    await model.start(start())

    assert len(client.prefill_calls) == 1
    request, request_id = client.prefill_calls[0]
    assert request_id == "timeline:session-1:0:prefix-prefill"
    assert request.messages[0].content == profile.system_prompt
    assert request.metadata["videos"] == []
    assert request.metadata["audios"] == []
    await model.close()


@pytest.mark.asyncio
async def test_direct_model_emits_raw_event_without_server_frequency(monkeypatch) -> None:
    profile = DirectE50Profile(
        system_prompt="strict training system prompt",
        user_prompt="strict training user prompt",
        prompt_version="event-e50-direct-checkpoint1500-v1",
    )
    monkeypatch.setattr(
        client_model_module.DirectE50Profile,
        "from_environment",
        classmethod(lambda cls: profile),
    )
    model = SglangClientTimelineDetectionModel(
        Client(["E20"]),
        start(),
        model_version="e50-direct-checkpoint-1500-fp8-sidecar-fixed",
        inference_interval_ms=1_000,
    )
    await model.start(start())
    await feed_first_window(model)

    window = await model.next_observations()

    assert [event.event_type for event in window.events] == ["E20"]
    assert window.to_wire()["contract_version"] == 3
    await model.close()
