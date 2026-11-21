"""One session's whitelist and trigger-frequency authority."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .frequency import EventFrequencyGate, EventFrequencyDecision, FrequencyDecision
from .rules import (
    CONTROL_EVENT_IDS, SILENT_EVENT_IDS, configured_trigger_ids, validate_event_ids,
    DEFAULT_EMISSION_COOLDOWN_MS, DEFAULT_CONTROL_COOLDOWN_MS,
    DEFAULT_CONFIRMATION_WINDOWS, DEFAULT_REARM_ABSENT_WINDOWS,
)
from .windows import DetectionWindow, WindowEvent


@dataclass(frozen=True)
class EventPolicySettings:
    """A frequency profile frozen for a session; presets can share this interface."""

    emission_cooldown_ms: int = DEFAULT_EMISSION_COOLDOWN_MS
    control_cooldown_ms: int = DEFAULT_CONTROL_COOLDOWN_MS
    confirmation_windows: int = DEFAULT_CONFIRMATION_WINDOWS
    rearm_absent_windows: int = DEFAULT_REARM_ABSENT_WINDOWS

    def __post_init__(self) -> None:
        for value in (
            self.emission_cooldown_ms, self.control_cooldown_ms,
            self.confirmation_windows, self.rearm_absent_windows,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("event frequency settings must be positive integers")


class EventPolicy:
    """Keep one frequency state machine per logical consumer, never per model.

    A raw detector never runs this policy. A confirmed-observation facade runs it
    instead of its consumer. Gaps are unknown, not negative windows. Hosts retain
    output leases, user-activity fences and expiry, not a second reply cooldown.
    """

    def __init__(
        self, *, allowed_triggers: Iterable[str] | None = None,
        settings: EventPolicySettings = EventPolicySettings(),
    ) -> None:
        self._allowed_triggers = frozenset(configured_trigger_ids(allowed_triggers))
        self._settings = settings
        self._frequency = EventFrequencyGate(
            cooldown_ms=settings.emission_cooldown_ms,
            control_cooldown_ms=settings.control_cooldown_ms,
            confirmation_windows=settings.confirmation_windows,
            rearm_absent_windows=settings.rearm_absent_windows,
        )
        self._fence: tuple[str, int, int, int] | None = None
        self._last_sequence = 0
        self._last_end_ms = -1
        self._last_decision: FrequencyDecision | None = None

    @property
    def allowed_triggers(self) -> frozenset[str]:
        return self._allowed_triggers

    @property
    def settings(self) -> EventPolicySettings:
        return self._settings

    @property
    def last_decision(self) -> FrequencyDecision | None:
        return self._last_decision

    def event_suppression_reason(self, event_type: str) -> str | None:
        if event_type in CONTROL_EVENT_IDS:
            return None
        if event_type in SILENT_EVENT_IDS:
            return "event_silent"
        if event_type not in self.allowed_triggers:
            return "trigger_not_configured"
        return None

    def evaluate(self, event_ids: list[str], *, end_ms: int) -> FrequencyDecision:
        """Evaluate completed detections; adapters do not classify product events."""
        validate_event_ids(event_ids)
        return self._evaluate(event_ids, end_ms=end_ms)

    def _evaluate(self, event_ids: list[str], *, end_ms: int) -> FrequencyDecision:
        visible = []
        filtered = []
        for event_id in event_ids:
            reason = self.event_suppression_reason(event_id)
            if reason is None:
                visible.append(event_id)
            else:
                filtered.append(EventFrequencyDecision(
                    event_id=event_id,
                    event_class="silent" if event_id in SILENT_EVENT_IDS else "ordinary",
                    disposition="suppressed", reason=reason,
                ))
        decision = self._frequency.evaluate(visible, end_ms=end_ms)
        self._last_decision = replace(decision, event_decisions=decision.event_decisions + tuple(filtered))
        return self._last_decision

    def accept_window(self, window: DetectionWindow) -> tuple[WindowEvent, ...]:
        fence = (window.session_id, window.identity_epoch, window.stream_epoch, window.observer_epoch)
        if self._fence is not None and self._fence[:2] != fence[:2]:
            raise ValueError("event policy belongs to another logical session")
        if self._fence is not None and (
            window.stream_epoch < self._fence[2] or window.observer_epoch < self._fence[3]
        ):
            raise ValueError("raw detection window epoch regressed")
        if self._fence != fence:
            if self._fence is None or self._fence[:3] != fence[:3]:
                self._frequency.clear()
            else:
                self._frequency.break_continuity()
            self._fence = fence
            self._last_sequence = 0
            self._last_end_ms = -1
        if window.window_sequence <= self._last_sequence or window.evidence_end_ms <= self._last_end_ms:
            raise ValueError("raw detection windows must advance in order")
        if window.window_sequence != self._last_sequence + 1:
            self._frequency.break_continuity()
        decision = self._evaluate([event.event_type for event in window.events], end_ms=window.evidence_end_ms)
        self._last_sequence = window.window_sequence
        self._last_end_ms = window.evidence_end_ms
        by_type = {event.event_type: event for event in window.events}
        return tuple(by_type[event_id] for event_id in decision.accepted_event_ids)

    def clear(self) -> None:
        """Reset detection state after a real media discontinuity."""
        self._frequency.clear()
        self._fence = None
        self._last_sequence = 0
        self._last_end_ms = -1
        self._last_decision = None
