"""Contracts for continuous full-media timeline observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from framex_event_policy.windows import TIMELINE_CONTRACT_VERSION
from sglang_omni.serve.agent_prompt_snapshot import TimelinePromptSnapshot
from sglang_omni.utils.framex_service_logs import valid_correlation_id

CONTRACT_VERSION = TIMELINE_CONTRACT_VERSION


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
    prompt_snapshot: TimelinePromptSnapshot | None = None
    # Log-correlation fields are explicit in the v3 wire contract.
    root_session_id: str | None = None
    agent_session_id: str | None = None

    def __post_init__(self) -> None:
        _required("session_id", self.session_id)
        _required("audio_format", self.audio_format)
        _required("video_format", self.video_format)
        _required("model_id", self.model_id)
        if self.root_session_id is not None:
            _required("root_session_id", self.root_session_id)
            if not valid_correlation_id(self.root_session_id):
                raise ValueError("root_session_id is not a valid correlation ID")
        if self.agent_session_id is not None:
            _required("agent_session_id", self.agent_session_id)
            if not valid_correlation_id(self.agent_session_id):
                raise ValueError("agent_session_id is not a valid correlation ID")
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
