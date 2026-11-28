"""Canonical event IDs and product classifications, independent of model labels."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_EMISSION_COOLDOWN_MS = 15_000
DEFAULT_CONTROL_COOLDOWN_MS = 3_000
DEFAULT_CONFIRMATION_WINDOWS = 2
DEFAULT_REARM_ABSENT_WINDOWS = 3

EVENT_IDS = frozenset(f"E{event_id}" for event_id in range(96))
CONTROL_EVENT_IDS = frozenset({"E20", "E21"})
SILENT_EVENT_IDS: frozenset[str] = frozenset()
REPLY_EVENT_IDS = EVENT_IDS - SILENT_EVENT_IDS - CONTROL_EVENT_IDS
TRIGGER_EVENT_IDS = REPLY_EVENT_IDS | CONTROL_EVENT_IDS
OUTPUT_INTERRUPT_EVENT_IDS = frozenset({
    "E2", "E3", "E5", "E8", "E17", "E20", "E21", "E28", "E29", "E30",
    "E31", "E32", "E33", "E34", "E36", "E37", "E38", "E39", "E40", "E41",
    "E42", "E43", "E44", "E49", "E50", "E60", "E65", "E75", "E76", "E82",
    "E83", "E85", "E91", "E92", "E93", "E94", "E95",
})
MULTIMODAL_GESTURE_EVENT_IDS = frozenset({
    "E11", "E12", "E13", "E14", "E18", "E20", "E21", "E22", "E23",
})


def validate_event_ids(event_ids: Iterable[str]) -> frozenset[str]:
    """Reject obsolete/unknown IDs instead of silently changing their meaning."""
    items = tuple(event_ids)
    if any(not isinstance(event_id, str) for event_id in items):
        raise ValueError("event IDs must be strings")
    values = frozenset(items)
    unknown = values - EVENT_IDS
    if unknown:
        raise ValueError(f"unknown event IDs (use canonical E0–E95): {sorted(unknown)}")
    return values


def configured_trigger_ids(override: Iterable[str] | None = None) -> tuple[str, ...]:
    """Omitted override uses product defaults; an explicit empty override disables replies."""
    values = TRIGGER_EVENT_IDS if override is None else validate_event_ids(override)
    return tuple(sorted(values, key=lambda event_id: int(event_id[1:])))
