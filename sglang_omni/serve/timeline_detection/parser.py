"""Canonicalize the event detector's short JSON output."""

from __future__ import annotations

import json
from typing import Any

from .prompt import EVENT_IDS


class ModelOutputError(ValueError):
    """Raised when no usable event JSON can be recovered."""


_EVENT_ORDER = {event_id: index for index, event_id in enumerate(EVENT_IDS)}


def normalize_event_ids(events: set[str]) -> list[str]:
    """Apply the deterministic ontology conflicts from event-v1.20."""

    normalized = set(events)
    if normalized & {"P1", "P2"}:
        normalized.difference_update({"D1", "D2"})
    if "B3" in normalized:
        normalized.difference_update({"D1", "D2"})
    if "P2" in normalized:
        normalized.discard("B3")
    return sorted(normalized, key=_EVENT_ORDER.__getitem__)


def _decode_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if not text:
        raise ModelOutputError("model returned empty text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ModelOutputError("model output does not contain a JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ModelOutputError("model output contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise ModelOutputError("model output must be a JSON object")
    return value


def parse_event_output(raw_output: str) -> list[str]:
    """Return whitelisted, de-duplicated events in canonical order."""

    value = _decode_json_object(raw_output)
    events = value.get("e")
    if not isinstance(events, list):
        raise ModelOutputError("model output must contain an e array")
    if any(not isinstance(event_id, str) for event_id in events):
        raise ModelOutputError("every e item must be a string")
    canonical = {event_id for event_id in events if event_id in _EVENT_ORDER}
    return normalize_event_ids(canonical)
