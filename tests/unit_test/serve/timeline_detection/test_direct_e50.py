from __future__ import annotations

import base64
import io

import av
import pytest
from PIL import Image

from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    TimelineMediaChunk,
)
from sglang_omni.serve.timeline_detection.direct_e50 import (
    DirectE50FrequencyGate,
    DirectE50Profile,
    build_direct_e50_prefill_request,
    build_direct_e50_request,
    encode_hc_video_mp4,
    parse_direct_e50_output,
)
from sglang_omni.serve.timeline_detection.parser import ModelOutputError
from sglang_omni.serve.timeline_detection.window import (
    AudioEvidenceStatus,
    EventDetectionWindow,
)


SYSTEM_PROMPT = "strict training system prompt"
USER_PROMPT = "strict training user prompt"


def _jpeg(sequence: int, at_ms: int) -> TimelineMediaChunk:
    output = io.BytesIO()
    Image.new("RGB", (320, 240), (sequence * 10 % 255, 80, 120)).save(
        output, format="JPEG"
    )
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=1,
        stream_epoch=2,
        sequence=sequence,
        kind=MediaKind.VIDEO,
        start_ms=at_ms,
        end_ms=at_ms + 100,
        encoding="image/jpeg",
        payload=output.getvalue(),
    )


def _wav() -> bytes:
    # Request construction only transports the already assembled WAV payload.
    return b"RIFF" + b"\x00" * 40


def _window(*, with_audio: bool = True) -> EventDetectionWindow:
    frames = tuple(_jpeg(index + 1, (index + 1) * 500) for index in range(14))
    return EventDetectionWindow(
        start_ms=0,
        current_start_ms=4_000,
        end_ms=7_000,
        history_frames=frames[:8],
        current_frames=frames[8:],
        audio_wav=_wav() if with_audio else None,
        audio_status=(
            AudioEvidenceStatus.COMPLETE
            if with_audio
            else AudioEvidenceStatus.MISSING
        ),
        audio_start_ms=0 if with_audio else None,
    )


def _profile() -> DirectE50Profile:
    return DirectE50Profile(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        prompt_version="event-e50-direct-checkpoint1500-v1",
    )


def _request(window: EventDetectionWindow, video: bytes = b"mp4"):
    return build_direct_e50_request(
        profile=_profile(),
        model_id="qwen3-omni-gpu4-timeline",
        session_id="session-1",
        window=window,
        video_mp4=video,
        retry=False,
        logical_request_id="request-1",
        contract_version=2,
        observer_epoch=3,
        stream_epoch=2,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("N", []),
        ("E13", ["E13"]),
        ("E0,E28", ["E0", "E28"]),
        ("E38,E38", ["E38", "E38"]),
    ],
)
def test_parse_direct_e50_output(raw: str, expected: list[str]) -> None:
    assert parse_direct_e50_output(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "E13, E14", "P13", "E27", "E8,E50", "E13,E13"],
)
def test_parse_direct_e50_output_rejects_non_contract_output(raw: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_direct_e50_output(raw)


def test_frequency_gate_globally_requires_three_n_windows_and_applies_cooldown() -> None:
    gate = DirectE50FrequencyGate(cooldown_ms=30_000, rearm_absent_windows=3)

    assert gate.accept(["E20"], end_ms=3_000) == ("E20",)
    assert gate.accept(["E20"], end_ms=4_000) == ()
    assert gate.accept([], end_ms=5_000) == ()
    assert gate.accept([], end_ms=6_000) == ()
    # A different positive event also remains blocked and resets the N streak.
    assert gate.accept(["E12"], end_ms=7_000) == ()
    assert gate.accept([], end_ms=8_000) == ()
    assert gate.accept([], end_ms=9_000) == ()
    assert gate.accept([], end_ms=10_000) == ()
    assert gate.accept(["E12"], end_ms=11_000) == ("E12",)
    assert gate.accept([], end_ms=12_000) == ()
    assert gate.accept([], end_ms=13_000) == ()
    assert gate.accept([], end_ms=14_000) == ()
    # Rearming the global gate does not bypass E20's 30-second cooldown.
    assert gate.accept(["E20"], end_ms=15_000) == ()
    assert gate.accept([], end_ms=16_000) == ()
    assert gate.accept([], end_ms=17_000) == ()
    assert gate.accept([], end_ms=18_000) == ()
    assert gate.accept(["E20"], end_ms=33_000) == ("E20",)


def test_frequency_gate_emits_multiple_codes_from_first_positive_window() -> None:
    gate = DirectE50FrequencyGate(cooldown_ms=30_000, rearm_absent_windows=3)

    assert gate.accept(["E0", "E28"], end_ms=3_000) == ("E0", "E28")
    assert gate.accept(["E12"], end_ms=4_000) == ()


def test_build_request_matches_training_wrapper_and_media_cardinality() -> None:
    request = _request(_window())
    parts = request.messages[1].content

    assert request.messages[0].content == SYSTEM_PROMPT
    assert [part["type"] for part in parts].count("video") == 1
    assert [part["type"] for part in parts].count("audio") == 1
    assert len(request.metadata["videos"]) == 1
    assert len(request.metadata["audios"]) == 1
    rendered = "".join(
        "<video>" if part["type"] == "video"
        else "<audio>" if part["type"] == "audio"
        else part["text"]
        for part in parts
    )
    assert rendered == (
        "【连续H+C视频开始｜窗口截止T=7秒｜2fps右端对齐帧】\n"
        "<video>\n"
        "【连续H+C视频结束｜前4秒/8帧=HISTORY H，只作上下文；"
        "后3秒/6帧=CURRENT C；完整C=(T-3s,T]均为本轮判定范围】\n"
        "【同步音频｜覆盖完整H+C】\n"
        "<audio>\n"
        + USER_PROMPT
    )
    assert request.metadata["input_layout"] == "single_continuous_hc_native_video"
    assert request.metadata["video_fps"] == 2.0
    assert request.metadata["video_max_frames"] == 20
    assert request.metadata["video_max_pixels"] == 130_000
    assert request.max_tokens == 8
    assert request.sampling.temperature == 0.0
    assert request.sampling.top_p == 1.0
    assert request.sampling.seed == 0


def test_build_prefill_request_only_contains_static_training_prefix() -> None:
    request = build_direct_e50_prefill_request(
        profile=_profile(),
        model_id="qwen3-omni-gpu4-timeline",
        session_id="session-1",
        logical_request_id="prefill-1",
    )

    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content == [{"type": "text", "text": " "}]
    assert request.metadata["videos"] == []
    assert request.metadata["audios"] == []
    assert request.metadata["prefill_only"] is True
    assert request.metadata["input_layout"] == "static_system_prefix_only"
    assert request.max_tokens == 1


def test_video_only_wrapper_has_no_untrained_blank_line() -> None:
    request = _request(_window(with_audio=False))
    parts = request.messages[1].content

    assert [part["type"] for part in parts].count("video") == 1
    assert [part["type"] for part in parts].count("audio") == 0
    assert "audios" not in request.metadata
    rendered = "".join(
        "<video>" if part["type"] == "video" else part["text"]
        for part in parts
    )
    assert "【同步音频缺失】\n" + USER_PROMPT in rendered
    assert "【同步音频缺失】\n\n" not in rendered


def test_encoded_hc_video_is_h264_and_preserves_frame_count() -> None:
    frames = tuple(_jpeg(index + 1, (index + 1) * 500) for index in range(14))
    payload = encode_hc_video_mp4(frames, fps=2.0)

    with av.open(io.BytesIO(payload), mode="r") as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
        assert stream.codec_context.name == "h264"
        assert len(decoded) == 14
        assert decoded[0].width == 320
        assert decoded[0].height == 240

    request = _request(_window(), video=payload)
    encoded = request.metadata["videos"][0].split(",", 1)[1]
    assert base64.b64decode(encoded) == payload
