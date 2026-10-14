from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang_omni.serve.timeline_detection import client_model as client_model_module
from sglang_omni.serve.timeline_detection.client_model import (
    _EventEdgeLatch,
    _V25R12Compatibility,
    SglangClientTimelineDetectionModel,
)
from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile
from sglang_omni.serve.timeline_detection.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from sglang_omni.serve.timeline_detection.parser import ModelOutputError


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
async def test_builds_v25_audio_video_payload_and_e_observations() -> None:
    client = Client(["g,U"])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version=PROMPT_VERSION,
        inference_interval_ms=1_000,
    )
    await model.start(start())
    await feed_first_window(model)

    events = await model.next_observations()

    assert [item.event_type for item in events] == ["E19", "E31"]
    assert all(item.evidence_start_ms == 0 for item in events)
    assert all(item.evidence_end_ms == 3_000 for item in events)
    assert all(item.evidence_mode == "audio_video" for item in events)
    assert all(item.audio_status == "complete" for item in events)
    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.metadata["prompt_version"] == PROMPT_VERSION
    assert request.metadata["attribution_duration_ms"] == 3_000
    assert request.metadata["input_layout"] == "ordered_hc_images"
    assert request.metadata["history_frame_count"] == 0
    assert request.metadata["current_frame_count"] == 6
    assert request.metadata["image_width"] == 448
    assert request.metadata["image_height"] == 252
    assert len(request.metadata["images"]) == 6
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")
    assert request.sampling.seed == 0
    assert request.max_tokens == 48
    assert [part["type"] for part in request.messages[1].content].count("video") == 0
    assert [part["type"] for part in request.messages[1].content].count("image") == 6
    boundary_text = "\n".join(
        part["text"]
        for part in request.messages[1].content
        if part["type"] == "text"
    )
    assert "CURRENT C开始｜3秒" in boundary_text
    assert "完整C=(T-3s,T]均为本轮判定范围" in boundary_text
    assert "ATTRIBUTION A" not in boundary_text
    for encoded_image in request.metadata["images"]:
        assert encoded_image.startswith("data:image/jpeg;base64,")
    await model.close()


@pytest.mark.asyncio
async def test_builds_video_only_request_without_audio_placeholder() -> None:
    client = Client(["i"])
    model = SglangClientTimelineDetectionModel(
        client,
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_video_only_window(model)

    events = await model.next_observations()

    assert [item.event_type for item in events] == ["E33"]
    assert events[0].evidence_mode == "video_only"
    assert events[0].audio_status == "missing"
    request = client.calls[0][0]
    assert request.metadata["evidence_mode"] == "video_only"
    assert request.metadata["audio_status"] == "missing"
    assert "audios" not in request.metadata
    boundary_text = "\n".join(
        part["text"]
        for part in request.messages[1].content
        if part["type"] == "text"
    )
    assert "没有完整同步音频，状态=missing" in boundary_text
    assert "音频状态=missing，本次为纯视觉判断" in boundary_text
    assert not any(
        part["type"] == "audio" for part in request.messages[1].content
    )
    await model.close()


@pytest.mark.asyncio
async def test_retries_invalid_output_with_retry_prompt() -> None:
    client = Client(["invalid", "N"])
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


@pytest.mark.asyncio
async def test_artifacts_are_not_written_for_no_event(
    monkeypatch, tmp_path
) -> None:
    artifact_root = tmp_path / "timeline-artifacts"
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT", str(artifact_root)
    )
    monkeypatch.setattr(
        client_model_module.DirectE50Profile,
        "from_environment",
        classmethod(lambda cls: None),
    )
    model = SglangClientTimelineDetectionModel(
        Client(["N"]),
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_window(model)

    assert await model.next_observations() == ()
    await model.close()

    assert not artifact_root.exists()


@pytest.mark.asyncio
async def test_complete_call_is_written_after_event_output(
    monkeypatch, tmp_path
) -> None:
    artifact_root = tmp_path / "timeline-artifacts"
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT", str(artifact_root)
    )
    monkeypatch.setattr(
        client_model_module.DirectE50Profile,
        "from_environment",
        classmethod(lambda cls: None),
    )
    model = SglangClientTimelineDetectionModel(
        Client(["g"]),
        start(),
        model_version=PROMPT_VERSION,
    )
    await model.start(start())
    await feed_first_window(model)

    events = await model.next_observations()
    assert [item.event_type for item in events] == ["E31"]
    await model.close()

    session_dirs = list(artifact_root.iterdir())
    assert len(session_dirs) == 1
    call_dirs = list(session_dirs[0].glob("stream-*"))
    assert len(call_dirs) == 1
    assert (call_dirs[0] / "request.json").is_file()
    assert (call_dirs[0] / "result.json").is_file()
    assert len(list((call_dirs[0] / "input" / "frames").glob("*.jpg"))) == 6


@pytest.mark.asyncio
async def test_direct_e50_artifact_records_frequency_decisions(
    monkeypatch, tmp_path
) -> None:
    artifact_root = tmp_path / "timeline-artifacts"
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT", str(artifact_root)
    )
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
        PrefillClient(["E20"]),
        start(),
        model_version="e50-direct-checkpoint-1500-fp8-sidecar-fixed",
        inference_interval_ms=1_000,
    )
    await model.start(start())
    await feed_first_window(model)

    events = await model.next_observations()
    assert events == ()
    await model.close()

    result_paths = list(artifact_root.glob("*/stream-*/result.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert result["accepted_event_ids"] == []
    assert result["frequency_gate"]["event_decisions"] == [
        {
            "cooldown_remaining_ms": 0,
            "disposition": "suppressed",
            "event_class": "control",
            "event_id": "E20",
            "reason": "awaiting_confirmation",
        }
    ]


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


def test_reuses_normalized_frame_inside_sliding_session(monkeypatch) -> None:
    calls = []

    def normalize(payload: bytes) -> bytes:
        calls.append(payload)
        return b"normalized"

    monkeypatch.setattr(client_model_module, "_normalize_frame_jpeg", normalize)
    model = SglangClientTimelineDetectionModel(
        Client([]),
        start(),
        model_version=PROMPT_VERSION,
        inference_interval_ms=1_000,
    )
    selected = frame(500, 7)

    assert model._normalized_frame_jpeg(selected) == b"normalized"
    assert model._normalized_frame_jpeg(selected) == b"normalized"
    assert calls == [selected.payload]


def test_event_edge_latch_suppresses_repeated_overlapping_windows() -> None:
    latch = _EventEdgeLatch(rearm_absent_windows=3)

    assert latch.accept(["E5"]) == ("E5",)
    assert latch.accept(["E5"]) == ()
    assert latch.accept(["E5", "E12"]) == ("E12",)
    assert latch.accept([]) == ()
    assert latch.accept([]) == ()
    assert latch.accept([]) == ()
    assert latch.accept(["E5"]) == ("E5",)


def test_event_edge_latch_resets_on_discontinuity_boundary() -> None:
    latch = _EventEdgeLatch(rearm_absent_windows=3)

    assert latch.accept(["E5"]) == ("E5",)
    assert latch.accept(["E5"]) == ()
    latch.clear()
    assert latch.accept(["E5"]) == ("E5",)


def test_event_edge_latch_tracks_new_occurrences_in_overlapping_windows() -> None:
    latch = _EventEdgeLatch(rearm_absent_windows=3)

    assert latch.accept(["E8"]) == ("E8",)
    assert latch.accept(["E8"]) == ()
    assert latch.accept(["E8", "E8"]) == ("E8",)
    assert latch.accept(["E8", "E8"]) == ()


def test_v25_compatibility_derives_r12_e49_and_e50() -> None:
    compatibility = _V25R12Compatibility()

    assert compatibility.translate(("E8",), end_ms=3_000) == ("E8",)
    assert compatibility.translate(("E8",), end_ms=5_000) == ("E50",)
    assert compatibility.translate(("E49",), end_ms=6_000) == ()
    assert compatibility.translate(("E49",), end_ms=8_000) == ("E49",)
    assert compatibility.translate(("E8",), end_ms=16_000) == ("E8",)


def test_v25_compatibility_rejects_events_outside_current_catalog() -> None:
    compatibility = _V25R12Compatibility()

    with pytest.raises(ModelOutputError, match="outside the 50-event catalog"):
        compatibility.translate(("E51",), end_ms=3_000)
