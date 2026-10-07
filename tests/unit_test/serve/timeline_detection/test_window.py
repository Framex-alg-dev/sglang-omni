from __future__ import annotations

import logging
import wave
from io import BytesIO

from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    TimelineMediaChunk,
)
from sglang_omni.serve.timeline_detection.window import (
    AudioEvidenceStatus,
    EventWindowAssembler,
)


def _audio(start_ms: int, end_ms: int, sequence: int) -> TimelineMediaChunk:
    samples = (end_ms - start_ms) * 16
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=1,
        stream_epoch=0,
        sequence=sequence,
        kind=MediaKind.AUDIO,
        start_ms=start_ms,
        end_ms=end_ms,
        encoding="pcm_s16le",
        payload=b"\x00\x00" * samples,
    )


def _frame(at_ms: int, sequence: int) -> TimelineMediaChunk:
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=1,
        stream_epoch=0,
        sequence=sequence,
        kind=MediaKind.VIDEO,
        start_ms=at_ms,
        end_ms=at_ms + 100,
        encoding="image/jpeg",
        payload=f"jpeg-{at_ms}".encode(),
    )


def test_builds_cold_start_with_zero_history_and_seven_current_frames() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assert assembler.append(_audio(0, 3_000, 1)) == ()
    windows = ()
    for index, at_ms in enumerate(range(0, 3_001, 500), start=2):
        windows = assembler.append(_frame(at_ms, index))

    assert len(windows) == 1
    window = windows[0]
    assert window.start_ms == window.current_start_ms == 0
    assert window.end_ms == 3_000
    assert window.history_frames == ()
    assert [item.start_ms for item in window.current_frames] == list(
        range(0, 3_001, 500)
    )
    assert window.audio_status is AudioEvidenceStatus.COMPLETE
    assert window.evidence_mode == "audio_video"
    assert window.audio_wav is not None
    with wave.open(BytesIO(window.audio_wav), "rb") as source:
        assert source.getframerate() == 16_000
        assert source.getnchannels() == 1
        assert source.getnframes() == 48_000


def test_accepts_contract_channel_name_alias() -> None:
    assembler = EventWindowAssembler(audio_format="pcm16/16000/mono")
    assert assembler.append(_audio(0, 1_000, 1)) == ()


def test_builds_steady_window_with_seven_history_frames() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    windows = []
    sequence = 1
    for second in range(12):
        windows.extend(assembler.append(_audio(second * 1_000, (second + 1) * 1_000, sequence)))
        sequence += 1
        for at_ms in (second * 1_000, second * 1_000 + 500):
            windows.extend(assembler.append(_frame(at_ms, sequence)))
            sequence += 1
    windows.extend(assembler.append(_frame(12_000, sequence)))

    steady = next(item for item in windows if item.end_ms == 12_000)
    assert steady.start_ms == 2_000
    assert steady.current_start_ms == 9_000
    assert len(steady.history_frames) == 7
    assert len(steady.current_frames) == 7


def test_does_not_fabricate_missing_current_frames() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assembler.append(_audio(0, 3_000, 1))
    windows = ()
    for index, at_ms in enumerate((0, 1_000, 2_000, 3_000), start=2):
        windows = assembler.append(_frame(at_ms, index))
    assert windows == ()


def test_builds_video_only_window_after_bounded_audio_grace() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    windows = []

    for sequence, at_ms in enumerate(range(0, 3_501, 500), start=1):
        windows.extend(assembler.append(_frame(at_ms, sequence)))

    assert len(windows) == 1
    assert windows[0].end_ms == 3_000
    assert windows[0].audio_wav is None
    assert windows[0].audio_status is AudioEvidenceStatus.MISSING
    assert windows[0].evidence_mode == "video_only"


def test_gapped_audio_falls_back_to_video_only() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assembler.append(_audio(0, 1_000, 1))
    assembler.append(_audio(2_000, 3_000, 2))
    windows = []

    for sequence, at_ms in enumerate(range(0, 3_501, 500), start=3):
        windows.extend(assembler.append(_frame(at_ms, sequence)))

    assert len(windows) == 1
    assert windows[0].audio_wav is None
    assert windows[0].audio_status is AudioEvidenceStatus.GAPPED


def test_history_gap_keeps_complete_current_audio() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    windows = []
    sequence = 1
    for start_ms, end_ms in ((0, 4_000), (5_000, 12_000)):
        windows.extend(assembler.append(_audio(start_ms, end_ms, sequence)))
        sequence += 1
    for at_ms in range(0, 12_501, 500):
        windows.extend(assembler.append(_frame(at_ms, sequence)))
        sequence += 1

    recovered = next(item for item in windows if item.end_ms == 12_000)
    assert recovered.start_ms == 2_000
    assert recovered.current_start_ms == 9_000
    assert recovered.audio_status is AudioEvidenceStatus.COMPLETE
    assert recovered.audio_start_ms == 9_000
    assert recovered.audio_wav is not None
    with wave.open(BytesIO(recovered.audio_wav), "rb") as source:
        assert source.getnframes() == 48_000


def test_late_audio_falls_back_after_video_grace() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assembler.append(_audio(0, 2_500, 1))
    windows = []

    for sequence, at_ms in enumerate(range(0, 3_501, 500), start=2):
        windows.extend(assembler.append(_frame(at_ms, sequence)))

    assert len(windows) == 1
    assert windows[0].audio_wav is None
    assert windows[0].audio_status is AudioEvidenceStatus.BEHIND


def test_reports_candidate_count_when_current_window_is_discarded(caplog) -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assembler.append(_audio(0, 3_000, 1))

    with caplog.at_level(
        logging.INFO,
        logger="sglang_omni.serve.timeline_detection.window",
    ):
        for index, at_ms in enumerate((0, 1_000, 2_000, 3_000), start=2):
            assembler.append(_frame(at_ms, index))

    assert "reason=current_frames_unavailable" in caplog.text
    assert "current_candidates=4" in caplog.text
    assert "current_required=7" in caplog.text
    assert "discarded_total=1" in caplog.text


def test_reports_when_audio_advances_without_video(caplog) -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")

    with caplog.at_level(
        logging.INFO,
        logger="sglang_omni.serve.timeline_detection.window",
    ):
        assembler.append(_audio(0, 3_000, 1))

    assert "timeline window waiting" in caplog.text
    assert "reason=no_video" in caplog.text
    assert "audio_chunks=1" in caplog.text
    assert "video_frames=0" in caplog.text


def test_window_cadence_is_anchored_to_first_real_video_frame() -> None:
    assembler = EventWindowAssembler(audio_format="pcm_s16le/16000/1")
    assembler.append(_audio(0, 4_000, 1))
    windows = ()
    for index, at_ms in enumerate(range(137, 3_138, 500), start=2):
        windows = assembler.append(_frame(at_ms, index))

    assert len(windows) == 1
    assert windows[0].current_start_ms == 137
    assert windows[0].end_ms == 3_137


def test_one_second_stride_keeps_three_second_current_window() -> None:
    assembler = EventWindowAssembler(
        audio_format="pcm_s16le/16000/1",
        window_stride_ms=1_000,
    )
    assembler.append(_audio(0, 6_000, 1))
    windows = []
    for index, at_ms in enumerate(range(0, 5_001, 500), start=2):
        windows.extend(assembler.append(_frame(at_ms, index)))

    assert [item.end_ms for item in windows] == [3_000, 4_000, 5_000]
    assert [item.current_start_ms for item in windows] == [0, 1_000, 2_000]
    assert all(item.end_ms - item.current_start_ms == 3_000 for item in windows)
    assert windows[-1].start_ms == 0
    assert windows[-1].history_duration_ms == 2_000


def test_rejects_an_oversized_retained_window() -> None:
    assembler = EventWindowAssembler(
        audio_format="pcm_s16le/16000/1",
        max_buffer_bytes=10,
    )

    try:
        assembler.append(_audio(0, 1_000, 1))
    except ValueError as exc:
        assert "byte limit" in str(exc)
    else:
        raise AssertionError("oversized event window was accepted")
