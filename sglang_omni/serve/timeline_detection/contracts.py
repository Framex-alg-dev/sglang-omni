"""Contracts for continuous full-media timeline observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .prompt import EVENT_IDS


CONTRACT_VERSION = 2


def _required(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


class MediaKind(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class TimelineSessionStart:
    session_id: str
    identity_epoch: int
    stream_epoch: int
    audio_format: str
    video_format: str
    model_id: str
    observer_epoch: int = 0
    next_sequence: int = 1
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _required("session_id", self.session_id)
        _required("audio_format", self.audio_format)
        _required("video_format", self.video_format)
        _required("model_id", self.model_id)
        if min(self.identity_epoch, self.observer_epoch, self.stream_epoch) < 0:
            raise ValueError("timeline epochs must be non-negative")
        if self.next_sequence <= 0:
            raise ValueError("timeline next_sequence must be positive")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported timeline contract version")


@dataclass(frozen=True)
class TimelineMediaChunk:
    session_id: str
    identity_epoch: int
    stream_epoch: int
    sequence: int
    kind: MediaKind
    start_ms: int
    end_ms: int
    encoding: str
    payload: bytes
    observer_epoch: int = 0

    def __post_init__(self) -> None:
        _required("session_id", self.session_id)
        _required("encoding", self.encoding)
        if self.sequence <= 0:
            raise ValueError("timeline sequence must be positive")
        if self.observer_epoch < 0:
            raise ValueError("timeline observer_epoch must be non-negative")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("timeline media range is invalid")
        if not self.payload:
            raise ValueError("timeline media payload must not be empty")


@dataclass(frozen=True)
class TimelineDiscontinuity:
    session_id: str
    identity_epoch: int
    old_stream_epoch: int
    new_stream_epoch: int
    reason: str
    observer_epoch: int = 0

    def __post_init__(self) -> None:
        _required("session_id", self.session_id)
        _required("discontinuity reason", self.reason)
        if self.new_stream_epoch <= self.old_stream_epoch:
            raise ValueError("new stream epoch must advance")
        if self.observer_epoch < 0:
            raise ValueError("timeline observer_epoch must be non-negative")


@dataclass(frozen=True)
class ObservationEvent:
    observation_id: str
    session_id: str
    identity_epoch: int
    stream_epoch: int
    event_type: str
    summary: str
    evidence_start_ms: int
    evidence_end_ms: int
    model_id: str
    model_version: str
    observer_epoch: int = 0
    evidence_mode: str = "audio_video"
    audio_status: str = "complete"
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("session_id", self.session_id),
            ("event_type", self.event_type),
            ("summary", self.summary),
            ("model_id", self.model_id),
            ("model_version", self.model_version),
        ):
            _required(name, value)
        if self.evidence_start_ms < 0 or self.evidence_end_ms <= self.evidence_start_ms:
            raise ValueError("observation evidence range is invalid")
        if self.observer_epoch < 0:
            raise ValueError("timeline observer_epoch must be non-negative")
        if self.event_type not in EVENT_IDS:
            raise ValueError("observation event_type is outside the timeline catalog")
        if self.evidence_mode not in {"audio_video", "video_only"}:
            raise ValueError("observation evidence_mode is invalid")
        if self.audio_status not in {"complete", "missing", "gapped", "behind"}:
            raise ValueError("observation audio_status is invalid")
        if (self.evidence_mode == "audio_video") != (
            self.audio_status == "complete"
        ):
            raise ValueError("observation evidence mode and audio status disagree")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported timeline contract version")
