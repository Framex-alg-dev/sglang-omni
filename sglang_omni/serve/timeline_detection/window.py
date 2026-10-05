"""Build aligned event-v1.20 H/C windows from continuous media chunks."""

from __future__ import annotations

import io
import logging
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .contracts import MediaKind, TimelineMediaChunk


logger = logging.getLogger(__name__)


class AudioEvidenceStatus(str, Enum):
    COMPLETE = "complete"
    MISSING = "missing"
    GAPPED = "gapped"
    BEHIND = "behind"


@dataclass(frozen=True)
class EventDetectionWindow:
    start_ms: int
    current_start_ms: int
    end_ms: int
    history_frames: tuple[TimelineMediaChunk, ...]
    current_frames: tuple[TimelineMediaChunk, ...]
    audio_wav: bytes | None
    audio_status: AudioEvidenceStatus

    @property
    def evidence_mode(self) -> str:
        return "audio_video" if self.audio_wav is not None else "video_only"

    @property
    def history_duration_ms(self) -> int:
        return self.current_start_ms - self.start_ms


class EventWindowAssembler:
    """Own sliding 3-second decisions over a bounded 10-second media history."""

    def __init__(
        self,
        *,
        audio_format: str,
        current_duration_ms: int = 3_000,
        history_duration_ms: int = 7_000,
        window_stride_ms: int | None = None,
        current_frame_count: int = 7,
        audio_grace_ms: int = 350,
        max_buffer_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        stride_ms = current_duration_ms if window_stride_ms is None else window_stride_ms
        if min(
            current_duration_ms,
            history_duration_ms,
            stride_ms,
            current_frame_count,
            max_buffer_bytes,
        ) <= 0:
            raise ValueError("event window limits must be positive")
        if stride_ms > current_duration_ms:
            raise ValueError("event window stride cannot exceed current duration")
        if audio_grace_ms < 0:
            raise ValueError("event audio grace must be non-negative")
        self._sample_rate, self._channels = _parse_audio_format(audio_format)
        self._current_duration_ms = current_duration_ms
        self._history_duration_ms = history_duration_ms
        self._window_stride_ms = stride_ms
        self._current_frame_count = current_frame_count
        self._audio_grace_ms = audio_grace_ms
        self._max_buffer_bytes = max_buffer_bytes
        self._buffered_bytes = 0
        self._audio: deque[TimelineMediaChunk] = deque()
        self._video: deque[TimelineMediaChunk] = deque()
        self._origin_ms: int | None = None
        self._next_end_ms: int | None = None
        self._session_id: str | None = None
        self._window_attempts = 0
        self._window_built = 0
        self._window_discarded = 0
        self._last_diagnostic_progress_ms = 0

    def append(self, chunk: TimelineMediaChunk) -> tuple[EventDetectionWindow, ...]:
        if self._session_id is None:
            self._session_id = chunk.session_id
        if chunk.kind is MediaKind.AUDIO:
            _require_encoding(chunk.encoding, {"pcm16", "pcm_s16le", "audio/l16"})
            self._append_ordered(self._audio, chunk)
        else:
            _require_encoding(chunk.encoding, {"jpeg", "jpg", "image/jpeg"})
            self._append_ordered(self._video, chunk)
        self._buffered_bytes += len(chunk.payload)
        if self._origin_ms is None and chunk.kind is MediaKind.VIDEO:
            self._origin_ms = chunk.start_ms
            self._next_end_ms = self._origin_ms + self._current_duration_ms

        ready: list[EventDetectionWindow] = []
        while self._can_decide_next_window():
            assert self._next_end_ms is not None
            self._window_attempts += 1
            window = self._build_window(self._next_end_ms)
            self._next_end_ms += self._window_stride_ms
            self._last_diagnostic_progress_ms = self._media_progress_ms()
            if window is not None:
                ready.append(window)
        self._log_waiting_if_due()
        self._prune()
        if self._buffered_bytes > self._max_buffer_bytes:
            raise ValueError("timeline event window exceeds its byte limit")
        return tuple(ready)

    def clear(self) -> None:
        self._audio.clear()
        self._video.clear()
        self._buffered_bytes = 0
        self._origin_ms = None
        self._next_end_ms = None
        self._last_diagnostic_progress_ms = 0

    @staticmethod
    def _append_ordered(
        target: deque[TimelineMediaChunk],
        chunk: TimelineMediaChunk,
    ) -> None:
        if target and chunk.start_ms < target[-1].start_ms:
            raise ValueError("timeline media timestamps must be ordered per modality")
        target.append(chunk)

    def _can_decide_next_window(self) -> bool:
        if (
            self._origin_ms is None
            or self._next_end_ms is None
            or not self._video
        ):
            return False
        if self._video[-1].start_ms < self._next_end_ms:
            return False
        current_start_ms = self._next_end_ms - self._current_duration_ms
        start_ms = max(
            self._origin_ms,
            current_start_ms - self._history_duration_ms,
        )
        if self._audio_status(start_ms, self._next_end_ms) is AudioEvidenceStatus.COMPLETE:
            return True
        return self._video[-1].start_ms >= self._next_end_ms + self._audio_grace_ms

    def _build_window(self, end_ms: int) -> EventDetectionWindow | None:
        assert self._origin_ms is not None
        current_start_ms = end_ms - self._current_duration_ms
        start_ms = max(
            self._origin_ms,
            current_start_ms - self._history_duration_ms,
        )
        current_candidates = tuple(
            item
            for item in self._video
            if current_start_ms <= item.start_ms <= end_ms
        )
        current_frames = _select_nearest_frames(
            current_candidates,
            start_ms=current_start_ms,
            end_ms=end_ms,
            count=self._current_frame_count,
            maximum_distance_ms=400,
        )
        if current_frames is None:
            self._log_discarded_window(
                end_ms=end_ms,
                reason="current_frames_unavailable",
                current_candidates=len(current_candidates),
                history_candidates=0,
            )
            return None

        history_duration_ms = current_start_ms - start_ms
        history_count = min(7, history_duration_ms // 1_000)
        history_candidates = tuple(
            item
            for item in self._video
            if start_ms <= item.start_ms < current_start_ms
        )
        history_frames = _select_nearest_frames(
            history_candidates,
            start_ms=start_ms,
            end_ms=current_start_ms,
            count=history_count,
            maximum_distance_ms=750,
            include_end=False,
        )
        if history_frames is None:
            self._log_discarded_window(
                end_ms=end_ms,
                reason="history_frames_unavailable",
                current_candidates=len(current_candidates),
                history_candidates=len(history_candidates),
            )
            return None
        audio_status = self._audio_status(start_ms, end_ms)
        audio_wav = (
            self._wav_between(start_ms, end_ms)
            if audio_status is AudioEvidenceStatus.COMPLETE
            else None
        )
        self._window_built += 1
        logger.info(
            "timeline window ready session_id=%s end_ms=%s current_candidates=%s "
            "current_selected=%s history_candidates=%s history_selected=%s "
            "evidence_mode=%s audio_status=%s built_total=%s attempted_total=%s",
            self._session_id,
            end_ms,
            len(current_candidates),
            len(current_frames),
            len(history_candidates),
            len(history_frames),
            "audio_video" if audio_wav is not None else "video_only",
            audio_status.value,
            self._window_built,
            self._window_attempts,
        )
        return EventDetectionWindow(
            start_ms=start_ms,
            current_start_ms=current_start_ms,
            end_ms=end_ms,
            history_frames=history_frames,
            current_frames=current_frames,
            audio_wav=audio_wav,
            audio_status=audio_status,
        )

    def _log_discarded_window(
        self,
        *,
        end_ms: int,
        reason: str,
        current_candidates: int,
        history_candidates: int,
    ) -> None:
        self._window_discarded += 1
        logger.info(
            "timeline window discarded session_id=%s end_ms=%s reason=%s "
            "current_candidates=%s current_required=%s history_candidates=%s "
            "discarded_total=%s attempted_total=%s",
            self._session_id,
            end_ms,
            reason,
            current_candidates,
            self._current_frame_count,
            history_candidates,
            self._window_discarded,
            self._window_attempts,
        )

    def _media_progress_ms(self) -> int:
        audio_end_ms = self._audio[-1].end_ms if self._audio else 0
        video_end_ms = self._video[-1].start_ms if self._video else 0
        return max(audio_end_ms, video_end_ms)

    def _log_waiting_if_due(self) -> None:
        progress_ms = self._media_progress_ms()
        if progress_ms - self._last_diagnostic_progress_ms < self._window_stride_ms:
            return
        if self._origin_ms is None or not self._video:
            reason = "no_video"
        else:
            assert self._next_end_ms is not None
            if self._video[-1].start_ms < self._next_end_ms:
                reason = "video_watermark_behind"
            else:
                current_start_ms = self._next_end_ms - self._current_duration_ms
                start_ms = max(
                    self._origin_ms,
                    current_start_ms - self._history_duration_ms,
                )
                audio_status = self._audio_status(start_ms, self._next_end_ms)
                if audio_status is AudioEvidenceStatus.MISSING:
                    reason = "audio_missing_grace"
                elif audio_status is AudioEvidenceStatus.BEHIND:
                    reason = "audio_watermark_behind"
                elif audio_status is AudioEvidenceStatus.GAPPED:
                    reason = "audio_gapped_grace"
                else:
                    return
        self._last_diagnostic_progress_ms = progress_ms
        logger.info(
            "timeline window waiting session_id=%s reason=%s next_end_ms=%s "
            "audio_chunks=%s video_frames=%s audio_end_ms=%s video_end_ms=%s",
            self._session_id,
            reason,
            self._next_end_ms,
            len(self._audio),
            len(self._video),
            self._audio[-1].end_ms if self._audio else None,
            self._video[-1].start_ms if self._video else None,
        )

    def _wav_between(self, start_ms: int, end_ms: int) -> bytes | None:
        frame_bytes = self._channels * 2
        bytes_per_second = self._sample_rate * frame_bytes
        pieces: list[bytes] = []
        covered_until = start_ms
        for chunk in self._audio:
            if chunk.end_ms <= start_ms:
                continue
            if chunk.start_ms >= end_ms:
                break
            if chunk.start_ms > covered_until + 20:
                return None
            clip_start = max(start_ms, chunk.start_ms)
            clip_end = min(end_ms, chunk.end_ms)
            if clip_end <= clip_start:
                continue
            first = (clip_start - chunk.start_ms) * bytes_per_second // 1_000
            last = (clip_end - chunk.start_ms) * bytes_per_second // 1_000
            first -= first % frame_bytes
            last -= last % frame_bytes
            pieces.append(chunk.payload[first:last])
            covered_until = max(covered_until, clip_end)
        if covered_until < end_ms - 20 or not pieces:
            return None
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(self._channels)
            writer.setsampwidth(2)
            writer.setframerate(self._sample_rate)
            writer.writeframes(b"".join(pieces))
        return output.getvalue()

    def _audio_status(
        self,
        start_ms: int,
        end_ms: int,
    ) -> AudioEvidenceStatus:
        if not self._audio:
            return AudioEvidenceStatus.MISSING
        covered_until = start_ms
        found = False
        for chunk in self._audio:
            if chunk.end_ms <= start_ms:
                continue
            if chunk.start_ms >= end_ms:
                break
            found = True
            if chunk.start_ms > covered_until + 20:
                return AudioEvidenceStatus.GAPPED
            covered_until = max(covered_until, min(end_ms, chunk.end_ms))
        if not found:
            return AudioEvidenceStatus.MISSING
        if covered_until < end_ms - 20:
            if self._audio[-1].end_ms < end_ms:
                return AudioEvidenceStatus.BEHIND
            return AudioEvidenceStatus.GAPPED
        return AudioEvidenceStatus.COMPLETE

    def _prune(self) -> None:
        if self._next_end_ms is None:
            return
        keep_from = max(
            0,
            self._next_end_ms
            - self._current_duration_ms
            - self._history_duration_ms,
        )
        while self._audio and self._audio[0].end_ms <= keep_from:
            self._buffered_bytes -= len(self._audio.popleft().payload)
        while self._video and self._video[0].start_ms < keep_from:
            self._buffered_bytes -= len(self._video.popleft().payload)


def _select_nearest_frames(
    candidates: tuple[TimelineMediaChunk, ...],
    *,
    start_ms: int,
    end_ms: int,
    count: int,
    maximum_distance_ms: int,
    include_end: bool = True,
) -> tuple[TimelineMediaChunk, ...] | None:
    if count == 0:
        return ()
    if len(candidates) < count:
        return None
    if count == 1:
        targets = ((start_ms + end_ms) // 2,)
    else:
        span = end_ms - start_ms
        divisor = count if not include_end else count - 1
        targets = tuple(
            start_ms + (span * index // divisor)
            for index in range(count)
        )
    selected: list[TimelineMediaChunk] = []
    remaining = list(candidates)
    for target in targets:
        nearest = min(remaining, key=lambda item: abs(item.start_ms - target))
        if abs(nearest.start_ms - target) > maximum_distance_ms:
            return None
        selected.append(nearest)
        remaining.remove(nearest)
    selected.sort(key=lambda item: item.start_ms)
    return tuple(selected)


def _parse_audio_format(value: str) -> tuple[int, int]:
    normalized = value.strip().lower()
    if normalized in {"pcm16", "pcm_s16le"}:
        return 16_000, 1
    parts = normalized.split("/")
    if len(parts) != 3 or parts[0] not in {"pcm16", "pcm_s16le"}:
        raise ValueError("timeline event detector requires pcm_s16le/rate/channels")
    channel_aliases = {"mono": 1, "stereo": 2}
    try:
        sample_rate = int(parts[1])
        channels = (
            channel_aliases[parts[2]]
            if parts[2] in channel_aliases
            else int(parts[2])
        )
    except ValueError as exc:
        raise ValueError(
            "timeline event detector requires pcm_s16le/rate/channels"
        ) from exc
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("timeline audio format values must be positive")
    return sample_rate, channels


def _require_encoding(value: str, supported: set[str]) -> None:
    if value.strip().lower() not in supported:
        raise ValueError(f"unsupported event detector media encoding: {value}")
