"""Versioned, training/serving-shared semantics for persona-aware actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PERSONA_WINDOW_PROFILE = "persona-window-v1"
PERSONA_AUDIO_PROFILE = "persona-audio-window-v2"
PERSONA_PROFILES = frozenset({PERSONA_WINDOW_PROFILE, PERSONA_AUDIO_PROFILE})
DECISION_MODES = frozenset({"user_request", "speech_window"})


@dataclass(frozen=True)
class SpeechWindow:
    window_id: str
    start_ms: int
    end_ms: int
    upcoming_text: str
    previous_text: str = ""
    history_basis: str = "synthesized"
    current_action_id: str | None = None
    state_at_ms: int = 0
    current_action_remaining_ms: int = 0
    user_action_status: str = "not_requested"
    recent_action_ids: tuple[str, ...] = ()
    fulfilled_action_ids: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: Any, *, require_text: bool = True) -> "SpeechWindow":
        if not isinstance(raw, Mapping):
            raise ValueError("speech_window must be an object")
        allowed = set(cls.__dataclass_fields__)
        if set(raw) - allowed:
            raise ValueError("unknown speech_window fields")
        for key in ("start_ms", "end_ms"):
            if type(raw.get(key)) is not int or raw[key] < 0:
                raise ValueError(f"speech_window.{key} must be a nonnegative integer")
        if raw["end_ms"] <= raw["start_ms"]:
            raise ValueError("speech_window must have positive duration")
        if raw["end_ms"] - raw["start_ms"] > 60_000:
            raise ValueError("speech_window duration exceeds 60 seconds")
        values = dict(raw)
        for key in (("window_id", "upcoming_text") if require_text else ("window_id",)):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"speech_window.{key} is required")
        for key, limit in (("window_id", 256), ("upcoming_text", 4000), ("previous_text", 4000)):
            value = values.get(key, "")
            if not isinstance(value, str) or len(value) > limit or "\x00" in value:
                raise ValueError(f"speech_window.{key} exceeds its text budget")
        if values.get("history_basis", "synthesized") not in {"synthesized", "submitted", "played"}:
            raise ValueError("speech_window.history_basis is invalid")
        action = values.get("current_action_id")
        if action is not None and (not isinstance(action, str) or not action or len(action) > 256):
            raise ValueError("speech_window.current_action_id is invalid")
        for key in ("state_at_ms", "current_action_remaining_ms"):
            number = values.get(key, 0)
            if type(number) is not int or number < 0:
                raise ValueError(f"speech_window.{key} must be a nonnegative integer")
        if values.get("state_at_ms", 0) > values["start_ms"]:
            raise ValueError("speech_window state must not come from after its target start")
        if values.get("current_action_remaining_ms", 0) and action is None:
            raise ValueError("remaining motion duration requires a current action")
        if values.get("user_action_status", "not_requested") not in {
            "not_requested", "pending", "running", "completed", "failed",
        }:
            raise ValueError("speech_window.user_action_status is invalid")
        for key in ("recent_action_ids", "fulfilled_action_ids"):
            items = values.get(key, ())
            if not isinstance(items, (list, tuple)) or len(items) > 32 or any(
                not isinstance(item, str) or not item or len(item) > 256 for item in items
            ):
                raise ValueError(f"speech_window.{key} must contain bounded action IDs")
            values[key] = tuple(items)
        if not require_text:
            values.setdefault("upcoming_text", "")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["recent_action_ids"] = list(self.recent_action_ids)
        value["fulfilled_action_ids"] = list(self.fulfilled_action_ids)
        return value

    def model_context(self) -> dict[str, Any]:
        value = self.to_dict()
        # Opaque orchestration IDs are for idempotency, not action semantics.
        value.pop("window_id")
        return value

