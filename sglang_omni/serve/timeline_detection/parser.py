"""Parse V25 aliases/P-codes and expose the stable E-code protocol."""

from __future__ import annotations

from .catalog import EVENT_IDS, OCCURRENCE_EVENT_IDS, P_TO_EVENT
from .prompt import ALIAS_TO_P


class ModelOutputError(ValueError):
    """Raised when no usable compact event output can be recovered."""


_EVENT_ORDER = {event_id: index for index, event_id in enumerate(EVENT_IDS)}


def _strip_accidental_wrapper(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _to_event_id(token: str) -> str | None:
    model_code = ALIAS_TO_P.get(token, token if token.startswith("P") else None)
    if model_code is not None:
        if model_code == "P27":
            return None
        event_id = P_TO_EVENT.get(model_code)
        if event_id is None:
            raise ModelOutputError(f"invalid model event token: {token}")
        return event_id
    if token.startswith("E"):
        if token == "E27":
            return None
        if token in _EVENT_ORDER:
            return token
    raise ModelOutputError(f"invalid model event token: {token}")


def parse_event_output(raw_output: str) -> list[str]:
    """Return canonical E-codes while retaining independent-instance counts."""

    text = _strip_accidental_wrapper(raw_output)
    if not text:
        raise ModelOutputError("model returned empty text")
    if text == "N":
        return []
    if any(char.isspace() for char in text):
        raise ModelOutputError("model output must not contain whitespace")

    tokens = text.split(",")
    if len(tokens) > 16:
        raise ModelOutputError("model returned too many event tokens")

    events: list[str] = []
    seen: set[str] = set()
    occurrence_counts: dict[str, int] = {}
    for token in tokens:
        event_id = _to_event_id(token)
        if event_id is None:
            continue
        if event_id in OCCURRENCE_EVENT_IDS:
            count = occurrence_counts.get(event_id, 0) + 1
            if count > 8:
                raise ModelOutputError("model repeated one occurrence event too many times")
            occurrence_counts[event_id] = count
            events.append(event_id)
        elif event_id not in seen:
            events.append(event_id)
        seen.add(event_id)
    return normalize_event_ids(events)


def normalize_event_ids(events: list[str]) -> list[str]:
    """Apply deterministic ontology conflicts without collapsing occurrences."""

    present = set(events)
    suppressed: set[str] = set()
    if present & {"E33", "E34"}:
        suppressed.update({"E24", "E25", "E26", "E30"})
    if "E28" in present:
        suppressed.add("E31")
    if "E48" in present:
        suppressed.add("E9")
    if "E17" in present:
        suppressed.update({"E24", "E36"})

    normalized = [event_id for event_id in events if event_id not in suppressed]
    return sorted(normalized, key=_EVENT_ORDER.__getitem__)
