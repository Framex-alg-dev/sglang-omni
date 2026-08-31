"""Typed wire and pipeline state for multimodal realtime sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer

# Kept local to make protocol-state parsing independent from the session
# orchestrator. The façade still exports the established constants.
MAX_ACTION_CHILDREN_PER_CATEGORY = 128
MAX_ACTION_PROFILE_CHARS = 8 * 1024
MAX_ACTION_PROFILE_FIELD_CHARS = 2 * 1024
MAX_CHARACTER_PROFILE_ROLE_CHARS = 5_000
ACTION_PERSONA_FIELDS = (
    "gender_expression",
    "visual_style",
    "role",
    "personality",
)
TURN_PHASE_COLLECTING = "collecting"
REPLY_MODE_LANGUAGE_REQUIRED = "LANGUAGE_REQUIRED"

@dataclass(frozen=True, slots=True)
class SessionActionCandidate:
    candidate_id: str
    action_id: str
    source_label: str
    short_definition: str
    execution_binding: dict[str, str]
    category_id: str | None = None

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCandidate":
        if not isinstance(value, dict):
            raise ValueError("action candidate must be an object")
        candidate_id = value.get("candidate_id")
        action_id = value.get("action_id")
        source_label = value.get("source_label") or action_id
        short_definition = value.get("short_definition") or source_label
        binding = value.get("execution_binding") or {}
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"action_id must be non-empty: {candidate_id!r}")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(f"source_label must be non-empty: {candidate_id!r}")
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(f"short_definition must be non-empty: {candidate_id!r}")
        if not isinstance(binding, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in binding.items()
        ):
            raise ValueError(
                f"execution_binding must be a string dictionary: {candidate_id!r}"
            )
        return cls(
            candidate_id=candidate_id.strip(),
            action_id=action_id.strip(),
            source_label=source_label.strip(),
            short_definition=short_definition.strip(),
            execution_binding=dict(binding),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
            "source_label": self.source_label,
            "short_definition": self.short_definition,
            "execution_binding": dict(self.execution_binding),
            **({"category_id": self.category_id} if self.category_id else {}),
        }


@dataclass(frozen=True, slots=True)
class SessionActionProfile:
    persona: tuple[tuple[str, str], ...] = ()
    visual_behavior_preferences: str = ""
    category_preferences: str = ""
    action_preferences: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionProfile":
        if not isinstance(value, dict):
            raise ValueError("action_profile must be an object")
        allowed_fields = {
            "persona",
            "visual_behavior_preferences",
            "category_preferences",
            "action_preferences",
        }
        unknown_fields = sorted(set(value) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "action_profile contains unsupported fields: "
                + ", ".join(unknown_fields)
            )

        raw_persona = value.get("persona", {})
        if not isinstance(raw_persona, dict):
            raise ValueError("action_profile.persona must be an object")
        unknown_persona_fields = sorted(set(raw_persona) - set(ACTION_PERSONA_FIELDS))
        if unknown_persona_fields:
            raise ValueError(
                "action_profile.persona contains unsupported fields: "
                + ", ".join(unknown_persona_fields)
            )
        persona: list[tuple[str, str]] = []
        for field_name in ACTION_PERSONA_FIELDS:
            if field_name not in raw_persona:
                continue
            field_value = raw_persona[field_name]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"action_profile.persona.{field_name} must be a non-empty string"
                )
            normalized = field_value.strip()
            max_chars = (
                MAX_CHARACTER_PROFILE_ROLE_CHARS
                if field_name == "role"
                else MAX_ACTION_PROFILE_FIELD_CHARS
            )
            if len(normalized) > max_chars:
                raise ValueError(
                    f"action_profile.persona.{field_name} must contain at most "
                    f"{max_chars} characters"
                )
            persona.append((field_name, normalized))

        preferences: dict[str, str] = {}
        for field_name in (
            "visual_behavior_preferences",
            "category_preferences",
            "action_preferences",
        ):
            field_value = value.get(field_name, "")
            if field_value is None:
                field_value = ""
            if not isinstance(field_value, str):
                raise ValueError(f"action_profile.{field_name} must be a string")
            normalized = field_value.strip()
            if len(normalized) > MAX_ACTION_PROFILE_FIELD_CHARS:
                raise ValueError(
                    f"action_profile.{field_name} must contain at most "
                    f"{MAX_ACTION_PROFILE_FIELD_CHARS} characters"
                )
            preferences[field_name] = normalized

        profile = cls(
            persona=tuple(persona),
            visual_behavior_preferences=preferences[
                "visual_behavior_preferences"
            ],
            category_preferences=preferences["category_preferences"],
            action_preferences=preferences["action_preferences"],
        )
        if not profile.persona and not (
            profile.visual_behavior_preferences
            or profile.category_preferences
            or profile.action_preferences
        ):
            raise ValueError("action_profile must contain at least one non-empty field")
        encoded = json.dumps(
            profile.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded) > MAX_ACTION_PROFILE_CHARS:
            raise ValueError(
                f"action_profile must contain at most {MAX_ACTION_PROFILE_CHARS} "
                "serialized characters"
            )
        return profile

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.persona:
            payload["persona"] = dict(self.persona)
        if self.visual_behavior_preferences:
            payload["visual_behavior_preferences"] = (
                self.visual_behavior_preferences
            )
        if self.category_preferences:
            payload["category_preferences"] = self.category_preferences
        if self.action_preferences:
            payload["action_preferences"] = self.action_preferences
        return payload


@dataclass(frozen=True, slots=True)
class SessionActionCategory:
    category_id: str
    source_label: str
    short_definition: str
    category_path: tuple[str, ...]
    children: tuple[SessionActionCandidate, ...]

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCategory":
        if not isinstance(value, dict):
            raise ValueError("action category must be an object")
        category_id = value.get("category_id")
        source_label = value.get("source_label") or category_id
        short_definition = value.get("short_definition") or source_label
        category_path_value = value.get("category_path") or []
        if not isinstance(category_path_value, list) or not all(
            isinstance(item, str) and item.strip() for item in category_path_value
        ):
            raise ValueError(f"category_path must be a string list: {category_id!r}")
        children = value.get("children")
        if not isinstance(category_id, str) or not category_id.strip():
            raise ValueError("category_id must be a non-empty string")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(
                f"category source_label must be non-empty: {category_id!r}"
            )
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(
                f"category short_definition must be non-empty: {category_id!r}"
            )
        if not isinstance(children, list) or not children:
            raise ValueError(
                f"category children must be a non-empty list: {category_id!r}"
            )
        if len(children) > MAX_ACTION_CHILDREN_PER_CATEGORY:
            raise ValueError(
                f"category children must contain at most {MAX_ACTION_CHILDREN_PER_CATEGORY} items"
            )
        parsed = []
        for child in children:
            item = SessionActionCandidate.from_payload(child)
            parsed.append(
                SessionActionCandidate(
                    candidate_id=item.candidate_id,
                    action_id=item.action_id,
                    source_label=item.source_label,
                    short_definition=item.short_definition,
                    execution_binding=dict(item.execution_binding),
                    category_id=category_id.strip(),
                )
            )
        return cls(
            category_id=category_id.strip(),
            source_label=source_label.strip(),
            short_definition=short_definition.strip(),
            category_path=tuple(item.strip() for item in category_path_value),
            children=tuple(parsed),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "source_label": self.source_label,
            "short_definition": self.short_definition,
            "category_path": list(self.category_path),
            "children": [item.as_dict() for item in self.children],
        }


@dataclass(slots=True)
class ImageFrame:
    seq: int
    timestamp_ms: int
    data_uri: str
    image_role: Literal["user_camera", "avatar_state"]
    preprocess_task: asyncio.Task[dict[str, Any]] | None = None


@dataclass(slots=True)
class ActionHistoryTurn:
    turn_id: str
    turn_origin: Literal["user", "proactive"]
    text_role: Literal["user_input", "character_reply"]
    messages: list[dict[str, Any]]
    audios: list[str]
    images: list[str]


@dataclass(slots=True)
class ReplyHistoryTurn:
    turn_id: str
    messages: list[dict[str, Any]]
    audios: list[str]
    images: list[str]
    image_roles: list[str]
    model_visible: bool = True
    history_kind: Literal["reply", "unsupported_action_notice"] = "reply"


@dataclass(frozen=True, slots=True)
class ExecutedActionRecord:
    """An action result treated as executed for subsequent turns.

    There is currently no client playback acknowledgement.  A successful
    action inference is therefore the session's authoritative execution fact.
    Failed action branches never create one of these records.
    """

    turn_id: str
    turn_origin: Literal["user", "proactive"]
    candidate_id: str
    action_id: str
    category_id: str | None
    source_label: str
    short_definition: str
    execute: bool


@dataclass(slots=True)
class ProvisionalReplyState:
    """A reply generated before the two-stage action decision is final.

    The same response is either promoted to the authoritative text response or
    discarded. Its text must never enter session history while still pending.
    """

    response_id: str
    source: Literal["generated", "provided"]
    started_at: float
    created_after_commit_ms: float | None
    request_id: str | None = None
    status: Literal["pending", "promoted", "discarded"] = "pending"
    resolution_reason: str | None = None
    text_parts: list[str] = field(default_factory=list)
    first_nonempty_at: float | None = None
    failed: bool = False
    cancelled: bool = False
    content_available: asyncio.Event = field(default_factory=asyncio.Event)
    sentence_ready: asyncio.Event = field(default_factory=asyncio.Event)
    delta_count: int = 0
    first_token_ms: float | None = None
    first_delta_after_commit_ms: float | None = None
    provisional_done_after_commit_ms: float | None = None
    official_text_done_after_commit_ms: float | None = None
    official_response_done_after_commit_ms: float | None = None
    finish_reason: str = "stop"
    usage: dict[str, Any] | None = None
    completed: bool = False
    official_created: bool = False
    official_done: bool = False
    task: asyncio.Task[tuple[str, dict[str, Any]]] | None = None
    finalization_task: asyncio.Task[Any] | None = None
    cleanup_task: asyncio.Task[Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tts_state: ReplyTTSState | None = None


@dataclass(slots=True)
class TurnBuffer:
    turn_id: str
    started_at: float
    audio: RealtimeAudioBuffer
    images: list[ImageFrame]
    audio_seqs: set[int]
    image_seqs: set[int]
    turn_origin: Literal["user", "proactive"]
    text_role: Literal["user_input", "character_reply"]
    audio_chunk_hashes: dict[int, str] = field(default_factory=dict)
    image_frame_signatures: dict[int, str] = field(default_factory=dict)
    trigger: str | None = None
    text: str | None = None
    reply_provided: bool = False
    reply_context: str | None = None
    avatar_state: dict[str, Any] | None = None
    audio_chunk_count: int = 0
    duplicate_audio_chunks: int = 0
    duplicate_image_frames: int = 0
    phase: Literal["collecting", "processing", "cancelling", "completed"] = (
        TURN_PHASE_COLLECTING
    )
    request_base: str | None = None
    current_request_id: str | None = None
    active_request_ids: set[str] = field(default_factory=set)
    branch_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    inference_task: asyncio.Task[None] | None = None
    trace_id: str = ""
    commit_started_at: float | None = None
    prepared_image_bytes: int = 0
    image_preprocess_stats: dict[str, Any] | None = None
    provisional_reply: ProvisionalReplyState | None = None
    first_text_received_at: float | None = None
    first_audio_received_at: float | None = None
    first_image_received_at: float | None = None
    session_turn_seq: int = 0


@dataclass(slots=True)
class ReplyTTSState:
    turn_id: str
    trace_id: str
    response_id: str
    text_queue: asyncio.Queue[str | None]
    allow_commit: asyncio.Event
    task: asyncio.Task[Any]
    next_audio_seq: int = 1
    input_finished: bool = False
    provisional: ProvisionalReplyState | None = None
    buffered_audio: list[bytes] = field(default_factory=list)
    buffered_audio_bytes: int = 0
    first_text_queued: bool = False
    first_audio_sent: bool = False


@dataclass(slots=True)
class ReplyHistoryRouteResult:
    decision: Literal["CURRENT_ONLY", "HISTORY_REQUIRED"]
    reply_mode: Literal["LANGUAGE_REQUIRED", "PURE_ACTION"] = (
        REPLY_MODE_LANGUAGE_REQUIRED
    )
    elapsed_ms: float = 0.0
    confidence_margin: float | None = None
    pure_action_ambiguous: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplySpeechModeResult:
    reply_mode: Literal["LANGUAGE_REQUIRED", "PURE_ACTION"]
    elapsed_ms: float = 0.0
    confidence_margin: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

