"""Versioned raw-window contract shared by producers and consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .rules import EVENT_IDS, validate_event_ids

TIMELINE_CONTRACT_VERSION = 3
WINDOW_FIELDS = frozenset({
    "session_id", "identity_epoch", "observer_epoch", "stream_epoch", "window_sequence",
    "evidence_start_ms", "evidence_end_ms", "model_id", "model_version",
    "evidence_mode", "audio_status", "events",
})


@dataclass(frozen=True)
class WindowEvent:
    event_type: str
    summary: str

    def __post_init__(self) -> None:
        validate_event_ids((self.event_type,))
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("window event summary is required")


@dataclass(frozen=True)
class DetectionWindow:
    session_id: str
    identity_epoch: int
    observer_epoch: int
    stream_epoch: int
    window_sequence: int
    evidence_start_ms: int
    evidence_end_ms: int
    model_id: str
    model_version: str
    evidence_mode: str
    audio_status: str
    events: tuple[WindowEvent, ...]

    def __post_init__(self) -> None:
        for name in ("session_id", "model_id", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        for name in (
            "identity_epoch", "observer_epoch", "stream_epoch", "window_sequence",
            "evidence_start_ms", "evidence_end_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.window_sequence == 0 or self.evidence_end_ms <= self.evidence_start_ms:
            raise ValueError("invalid detection window sequence/range")
        if self.evidence_mode not in {"audio_video", "video_only"}:
            raise ValueError("invalid window evidence mode")
        if self.audio_status not in {"complete", "missing", "gapped", "behind"}:
            raise ValueError("invalid window audio status")
        if (self.evidence_mode == "audio_video") != (self.audio_status == "complete"):
            raise ValueError("window evidence mode and audio status disagree")
        if not isinstance(self.events, tuple) or not all(isinstance(event, WindowEvent) for event in self.events):
            raise ValueError("window events must be WindowEvent values")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "window.result",
            "contract_version": TIMELINE_CONTRACT_VERSION,
            "status": "complete",
            **asdict(self),
            "events": [asdict(event) for event in self.events],
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> DetectionWindow:
        if (
            raw.get("type") != "window.result"
            or type(raw.get("contract_version")) is not int
            or raw["contract_version"] != TIMELINE_CONTRACT_VERSION
            or raw.get("status") != "complete"
            or set(raw) != WINDOW_FIELDS | {"type", "contract_version", "status"}
        ):
            raise ValueError("unsupported raw detection window contract")
        events = raw["events"]
        if not isinstance(events, list) or len(events) > len(EVENT_IDS) + 3:
            raise ValueError("window events must be a bounded array")
        parsed = []
        for event in events:
            if not isinstance(event, dict) or set(event) != {"event_type", "summary"}:
                raise ValueError("invalid window event")
            parsed.append(WindowEvent(**event))
        return cls(**{field: raw[field] for field in WINDOW_FIELDS - {"events"}}, events=tuple(parsed))
