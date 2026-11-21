"""Deterministic event confirmation, latching and emission cooldowns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rules import (
    EVENT_IDS, CONTROL_EVENT_IDS, SILENT_EVENT_IDS, DEFAULT_EMISSION_COOLDOWN_MS,
    DEFAULT_CONTROL_COOLDOWN_MS, DEFAULT_CONFIRMATION_WINDOWS, DEFAULT_REARM_ABSENT_WINDOWS,
)


@dataclass(frozen=True)
class EventFrequencyDecision:
    event_id: str
    event_class: str
    disposition: str
    reason: str
    cooldown_remaining_ms: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_class": self.event_class,
            "disposition": self.disposition,
            "reason": self.reason,
            "cooldown_remaining_ms": self.cooldown_remaining_ms,
        }


@dataclass(frozen=True)
class FrequencyDecision:
    accepted_event_ids: tuple[str, ...]
    event_decisions: tuple[EventFrequencyDecision, ...]
    ordinary_latched: bool
    ordinary_clear_streak: int

    def to_record(self) -> dict[str, Any]:
        return {
            "accepted_event_ids": list(self.accepted_event_ids),
            "event_decisions": [
                decision.to_record() for decision in self.event_decisions
            ],
            "ordinary_latched": self.ordinary_latched,
            "ordinary_clear_streak": self.ordinary_clear_streak,
        }


class EventFrequencyGate:
    """Apply control, silent, and ordinary event frequency semantics.

    Every event must first appear in the configured number of consecutive
    inference windows.  A missing window resets that event's confirmation
    streak, so separate one-off guesses never combine into a trigger.

    Ordinary events share a global latch so the tail of one event cannot be
    published as a different event without an explicit absence run.  Control
    events bypass that latch so E20/E21 can always stop active output.  Silent
    events retain independent edge/cooldown suppression but never occupy or
    extend the ordinary global latch.
    """

    def __init__(
        self,
        *,
        cooldown_ms: int = DEFAULT_EMISSION_COOLDOWN_MS,
        confirmation_windows: int = DEFAULT_CONFIRMATION_WINDOWS,
        rearm_absent_windows: int = DEFAULT_REARM_ABSENT_WINDOWS,
        control_cooldown_ms: int = DEFAULT_CONTROL_COOLDOWN_MS,
        silent_event_ids: frozenset[str] = SILENT_EVENT_IDS,
    ) -> None:
        if (
            cooldown_ms <= 0
            or control_cooldown_ms <= 0
            or confirmation_windows <= 0
            or rearm_absent_windows <= 0
        ):
            raise ValueError("event frequency limits must be positive")
        unknown_silent_ids = silent_event_ids - EVENT_IDS
        if unknown_silent_ids:
            raise ValueError(
                "event silent event ids are outside the catalog: "
                f"{sorted(unknown_silent_ids)}"
            )
        overlap = silent_event_ids & CONTROL_EVENT_IDS
        if overlap:
            raise ValueError(
                "event events cannot be both control and silent: "
                f"{sorted(overlap)}"
            )
        self._cooldown_ms = cooldown_ms
        self._control_cooldown_ms = control_cooldown_ms
        self._confirmation_windows = confirmation_windows
        self._rearm_absent_windows = rearm_absent_windows
        self._silent_event_ids = silent_event_ids
        self._independent_latched: set[str] = set()
        self._independent_absent_streaks: dict[str, int] = {}
        self._ordinary_latched = False
        self._ordinary_clear_streak = 0
        self._cooldown_until: dict[str, int] = {}
        self._confirmation_streaks: dict[str, int] = {}

    def accept(self, event_ids: list[str], *, end_ms: int) -> tuple[str, ...]:
        return self.evaluate(event_ids, end_ms=end_ms).accepted_event_ids

    def evaluate(
        self, event_ids: list[str], *, end_ms: int
    ) -> FrequencyDecision:
        unknown_ids = set(event_ids) - EVENT_IDS
        if unknown_ids:
            raise ValueError(
                "event frequency input is outside the catalog: "
                f"{sorted(unknown_ids)}"
            )
        if end_ms < 0:
            raise ValueError("event frequency timestamp must be non-negative")

        present = set(event_ids)
        for event_id in tuple(self._confirmation_streaks):
            if event_id not in present:
                self._confirmation_streaks.pop(event_id, None)
        independent_ids = (
            CONTROL_EVENT_IDS | self._silent_event_ids
        )
        for event_id in tuple(self._independent_latched):
            if event_id in present:
                self._independent_absent_streaks[event_id] = 0
                continue
            streak = self._independent_absent_streaks.get(event_id, 0) + 1
            if streak >= self._rearm_absent_windows:
                self._independent_latched.remove(event_id)
                self._independent_absent_streaks.pop(event_id, None)
            else:
                self._independent_absent_streaks[event_id] = streak

        ordinary_present = bool(present - independent_ids)
        ordinary_was_latched = self._ordinary_latched
        if ordinary_present:
            self._ordinary_clear_streak = 0
        elif self._ordinary_latched:
            self._ordinary_clear_streak += 1
            if self._ordinary_clear_streak >= self._rearm_absent_windows:
                self._ordinary_latched = False
                self._ordinary_clear_streak = 0

        accepted: list[str] = []
        accepted_independent: set[str] = set()
        ordinary_should_latch = False
        decisions: list[EventFrequencyDecision] = []
        seen: set[str] = set()
        for event_id in event_ids:
            event_class = self._event_class(event_id)
            if event_id in seen:
                decisions.append(
                    EventFrequencyDecision(
                        event_id=event_id,
                        event_class=event_class,
                        disposition="suppressed",
                        reason="duplicate_in_window",
                    )
                )
                continue
            seen.add(event_id)

            cooldown_until = self._cooldown_until.get(event_id, 0)
            cooldown_remaining_ms = max(0, cooldown_until - end_ms)
            if event_class != "ordinary" and event_id in self._independent_latched:
                disposition = "suppressed"
                reason = "continuous_event"
            elif event_class == "ordinary" and ordinary_was_latched:
                disposition = "suppressed"
                reason = "ordinary_window_latched"
            elif cooldown_remaining_ms:
                disposition = "suppressed"
                reason = "event_cooldown"
                if event_class == "ordinary":
                    ordinary_should_latch = True
            else:
                confirmation_streak = (
                    self._confirmation_streaks.get(event_id, 0) + 1
                )
                if confirmation_streak < self._confirmation_windows:
                    self._confirmation_streaks[event_id] = confirmation_streak
                    disposition = "suppressed"
                    reason = "awaiting_confirmation"
                else:
                    self._confirmation_streaks.pop(event_id, None)
                    disposition = "accepted"
                    reason = "accepted"
                    accepted.append(event_id)
                    if event_class == "ordinary":
                        ordinary_should_latch = True
                    else:
                        accepted_independent.add(event_id)
                    cooldown_ms = (
                        self._control_cooldown_ms
                        if event_class == "control"
                        else self._cooldown_ms
                    )
                    self._cooldown_until[event_id] = end_ms + cooldown_ms
            if reason not in {"awaiting_confirmation", "accepted"}:
                self._confirmation_streaks.pop(event_id, None)
            decisions.append(
                EventFrequencyDecision(
                    event_id=event_id,
                    event_class=event_class,
                    disposition=disposition,
                    reason=reason,
                    cooldown_remaining_ms=(
                        cooldown_remaining_ms
                        if reason == "event_cooldown"
                        else 0
                    ),
                )
            )

        for event_id in accepted_independent:
            self._independent_latched.add(event_id)
            self._independent_absent_streaks[event_id] = 0
        if ordinary_should_latch:
            self._ordinary_latched = True

        return FrequencyDecision(
            accepted_event_ids=tuple(accepted),
            event_decisions=tuple(decisions),
            ordinary_latched=self._ordinary_latched,
            ordinary_clear_streak=self._ordinary_clear_streak,
        )

    def _event_class(self, event_id: str) -> str:
        if event_id in CONTROL_EVENT_IDS:
            return "control"
        if event_id in self._silent_event_ids:
            return "silent"
        return "ordinary"

    def break_continuity(self) -> None:
        """Unknown/skipped windows neither confirm nor prove absence."""
        self._confirmation_streaks.clear()
        self._ordinary_clear_streak = 0
        self._independent_absent_streaks.clear()

    def clear(self) -> None:
        self._independent_latched.clear()
        self._independent_absent_streaks.clear()
        self._ordinary_latched = False
        self._ordinary_clear_streak = 0
        self._cooldown_until.clear()
        self._confirmation_streaks.clear()
