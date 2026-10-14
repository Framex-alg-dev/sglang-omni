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
    DIRECT_E50_SILENT_EVENT_IDS,
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


def test_profile_reads_all_frequency_policy_layers(monkeypatch, tmp_path) -> None:
    system_prompt = tmp_path / "system.txt"
    user_prompt = tmp_path / "user.txt"
    system_prompt.write_text(SYSTEM_PROMPT, encoding="utf-8")
    user_prompt.write_text(USER_PROMPT, encoding="utf-8")
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL", "direct-e50")
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH", str(system_prompt)
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_USER_PROMPT_PATH", str(user_prompt)
    )
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_PROMPT_VERSION", "test-v1")
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_EVENT_COOLDOWN_MS", "15000")
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_CONTROL_EVENT_COOLDOWN_MS", "3000"
    )
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_CONFIRMATION_WINDOWS", "2")
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_SILENT_EVENT_IDS", "E1,E3")

    profile = DirectE50Profile.from_environment()

    assert profile is not None
    assert profile.cooldown_ms == 15_000
    assert profile.control_cooldown_ms == 3_000
    assert profile.confirmation_windows == 2
    assert profile.silent_event_ids == frozenset({"E1", "E3"})


def test_profile_rejects_unknown_silent_event(monkeypatch, tmp_path) -> None:
    system_prompt = tmp_path / "system.txt"
    user_prompt = tmp_path / "user.txt"
    system_prompt.write_text(SYSTEM_PROMPT, encoding="utf-8")
    user_prompt.write_text(USER_PROMPT, encoding="utf-8")
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL", "direct-e50")
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH", str(system_prompt)
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_USER_PROMPT_PATH", str(user_prompt)
    )
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_PROMPT_VERSION", "test-v1")
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_SILENT_EVENT_IDS", "E51")

    with pytest.raises(ValueError, match="contains unknown event ids"):
        DirectE50Profile.from_environment()


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


def _frequency_gate(*, confirmation_windows: int = 1) -> DirectE50FrequencyGate:
    return DirectE50FrequencyGate(
        cooldown_ms=15_000,
        control_cooldown_ms=3_000,
        confirmation_windows=confirmation_windows,
        rearm_absent_windows=3,
        silent_event_ids=DIRECT_E50_SILENT_EVENT_IDS,
    )


def test_frequency_gate_requires_two_consecutive_windows() -> None:
    gate = _frequency_gate(confirmation_windows=2)

    first = gate.evaluate(["E37"], end_ms=3_000)
    assert first.accepted_event_ids == ()
    assert first.event_decisions[0].reason == "awaiting_confirmation"

    # An empty or different-event window breaks the confirmation streak.
    assert gate.accept([], end_ms=4_000) == ()
    assert gate.accept(["E37"], end_ms=5_000) == ()
    assert gate.accept(["E38"], end_ms=6_000) == ()

    second = gate.evaluate(["E38"], end_ms=7_000)
    assert second.accepted_event_ids == ("E38",)
    assert second.event_decisions[0].reason == "accepted"


def test_frequency_gate_confirms_all_event_classes_on_second_window() -> None:
    gate = _frequency_gate(confirmation_windows=2)

    assert gate.accept(["E0", "E1", "E20"], end_ms=3_000) == ()
    assert gate.accept(["E0", "E1", "E20"], end_ms=4_000) == (
        "E0",
        "E1",
        "E20",
    )


def test_frequency_gate_protects_ordinary_transitions_and_uses_15s_cooldown() -> None:
    gate = _frequency_gate()

    assert gate.accept(["E5"], end_ms=3_000) == ("E5",)
    # A different actionable event cannot consume the tail of E5.
    decision = gate.evaluate(["E6"], end_ms=4_000)
    assert decision.accepted_event_ids == ()
    assert decision.event_decisions[0].reason == "ordinary_window_latched"

    # Silent-only windows count as ordinary absence instead of extending the
    # global latch.
    assert gate.accept(["E1"], end_ms=5_000) == ("E1",)
    assert gate.accept(["E1"], end_ms=6_000) == ()
    assert gate.accept(["E3"], end_ms=7_000) == ("E3",)
    assert gate.accept(["E6"], end_ms=8_000) == ("E6",)

    assert gate.accept([], end_ms=9_000) == ()
    assert gate.accept([], end_ms=10_000) == ()
    assert gate.accept([], end_ms=11_000) == ()
    decision = gate.evaluate(["E5"], end_ms=12_000)
    assert decision.accepted_event_ids == ()
    assert decision.event_decisions[0].reason == "event_cooldown"
    assert decision.event_decisions[0].cooldown_remaining_ms == 6_000

    assert gate.accept([], end_ms=13_000) == ()
    assert gate.accept([], end_ms=14_000) == ()
    assert gate.accept([], end_ms=15_000) == ()
    assert gate.accept(["E5"], end_ms=18_000) == ("E5",)


def test_control_events_bypass_ordinary_latch_without_suppressing_each_other() -> None:
    gate = _frequency_gate()

    assert gate.accept(["E5"], end_ms=3_000) == ("E5",)
    assert gate.accept(["E20"], end_ms=4_000) == ("E20",)
    decision = gate.evaluate(["E20", "E21", "E6"], end_ms=5_000)

    assert decision.accepted_event_ids == ("E21",)
    assert [item.reason for item in decision.event_decisions] == [
        "continuous_event",
        "accepted",
        "ordinary_window_latched",
    ]

    assert gate.accept([], end_ms=6_000) == ()
    assert gate.accept([], end_ms=7_000) == ()
    assert gate.accept([], end_ms=8_000) == ()
    assert gate.accept(["E20"], end_ms=9_000) == ("E20",)


def test_silent_events_neither_occupy_nor_extend_the_ordinary_window() -> None:
    gate = _frequency_gate()

    assert gate.accept(["E1"], end_ms=3_000) == ("E1",)
    assert gate.accept(["E5"], end_ms=4_000) == ("E5",)
    assert gate.accept(["E1"], end_ms=5_000) == ()
    assert gate.accept(["E3"], end_ms=6_000) == ("E3",)
    assert gate.accept(["E4"], end_ms=7_000) == ("E4",)
    assert gate.accept(["E6"], end_ms=8_000) == ("E6",)


def test_frequency_gate_emits_all_classes_from_first_mixed_window() -> None:
    gate = _frequency_gate()

    decision = gate.evaluate(["E0", "E1", "E20", "E28"], end_ms=3_000)

    assert decision.accepted_event_ids == ("E0", "E1", "E20", "E28")
    assert [item.event_class for item in decision.event_decisions] == [
        "ordinary",
        "silent",
        "control",
        "ordinary",
    ]


def test_frequency_gate_reports_duplicate_codes_and_control_cooldown() -> None:
    gate = _frequency_gate()

    decision = gate.evaluate(["E20", "E20"], end_ms=3_000)
    assert decision.accepted_event_ids == ("E20",)
    assert [item.reason for item in decision.event_decisions] == [
        "accepted",
        "duplicate_in_window",
    ]

    assert gate.accept([], end_ms=3_100) == ()
    assert gate.accept([], end_ms=3_200) == ()
    assert gate.accept([], end_ms=3_300) == ()
    decision = gate.evaluate(["E20"], end_ms=3_400)
    assert decision.accepted_event_ids == ()
    assert decision.event_decisions[0].reason == "event_cooldown"
    assert decision.event_decisions[0].cooldown_remaining_ms == 2_600


def test_frequency_gate_clear_resets_every_policy_layer() -> None:
    gate = _frequency_gate()

    assert gate.accept(["E5", "E20"], end_ms=3_000) == ("E5", "E20")
    gate.clear()
    assert gate.accept(["E5", "E20"], end_ms=3_001) == ("E5", "E20")


def test_frequency_gate_rejects_invalid_policy_ids() -> None:
    with pytest.raises(ValueError, match="outside the catalog"):
        DirectE50FrequencyGate(
            cooldown_ms=15_000,
            control_cooldown_ms=3_000,
            confirmation_windows=2,
            rearm_absent_windows=3,
            silent_event_ids=frozenset({"E51"}),
        )

    with pytest.raises(ValueError, match="both control and silent"):
        DirectE50FrequencyGate(
            cooldown_ms=15_000,
            control_cooldown_ms=3_000,
            confirmation_windows=2,
            rearm_absent_windows=3,
            silent_event_ids=frozenset({"E20"}),
        )

    with pytest.raises(ValueError, match="frequency limits must be positive"):
        DirectE50FrequencyGate(
            cooldown_ms=15_000,
            control_cooldown_ms=3_000,
            confirmation_windows=0,
            rearm_absent_windows=3,
        )


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
