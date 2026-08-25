from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.action_scoring import (
    MAX_MICRO_BATCH_SIZE,
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    ACTION_HISTORY_INSTRUCTION,
    ACTION_HISTORY_INSTRUCTION_EN,
    CATEGORY_CONTEXT_POLICY,
    CATEGORY_CONTEXT_POLICY_EN,
    DEFAULT_ACTION_PROMPT_LOCALE,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
    GlobalActionCatalog,
    GlobalActionCatalogPrewarmStatus,
)
from sglang_omni.models.qwen3_omni.prompt_localization import (
    PROMPT_LANGUAGE_BY_LOCALE,
    localized_prompt,
)
from sglang_omni.preprocessing.image import prepare_image_bytes_for_wire
from sglang_omni.serve.realtime.audio_buffer import BufferOverflow, RealtimeAudioBuffer
from sglang_omni.utils.structured_logs import (
    emit_structured_log,
    get_structured_log_writer,
    new_trace_id,
)

if TYPE_CHECKING:
    from sglang_omni.client.client import Client

logger = logging.getLogger(__name__)


MAX_ACTION_CANDIDATES = 512
MAX_ACTION_CATEGORIES = 128
MAX_ACTION_CHILDREN_PER_CATEGORY = 128
MAX_PREWARM_CHILD_CATEGORIES = 16  # Internal legacy handler limit; not wire-visible.
REALTIME_PROTOCOL_VERSION = 1
SUPPORTED_MODALITIES = frozenset({"text", "action"})
DEFAULT_MODALITIES = ("text", "action")
MAX_SESSION_ID_CHARS = 128
MAX_TURN_ID_CHARS = 128
MAX_TRIGGER_TYPE_CHARS = 256
MAX_TURN_TEXT_CHARS = 64 * 1024
MAX_AVATAR_STATE_CHARS = 16 * 1024
MAX_EXECUTION_BINDING_CHARS = 4 * 1024
MAX_INSTRUCTIONS_CHARS = 32 * 1024
MAX_UNSUPPORTED_ACTION_TEXT_CHARS = 2 * 1024
MAX_REPLY_CONTEXT_CHARS = 8 * 1024
MAX_ACTION_PROFILE_CHARS = 8 * 1024
MAX_ACTION_PROFILE_FIELD_CHARS = 2 * 1024
ACTION_PERSONA_FIELDS = (
    "gender_expression",
    "visual_style",
    "role",
    "personality",
)
DEFAULT_REPLY_MAX_NEW_TOKENS = 512
DEFAULT_REPLY_TEMPERATURE = 0.4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_TURN = 64
MAX_AUDIO_CHUNKS_PER_TURN = 4096
MAX_IMAGE_PREPROCESS_TASKS_PER_TURN = 8
MAX_PREPARED_IMAGE_BYTES_PER_FRAME = 32 * 1024 * 1024
MAX_PREPARED_IMAGE_BYTES_PER_TURN = 64 * 1024 * 1024
# Action scoring runs on the thinker stage. Keep the action context bounded
# while retaining recent session history, including action state records.
MAX_ACTION_HISTORY_TURNS = 4
MAX_ACTION_HISTORY_AUDIOS = 4
MAX_ACTION_HISTORY_IMAGES = 8
MAX_ACTION_CURRENT_IMAGES = 8
# Keep lightweight, text-free action facts separately from the multimodal
# action-scoring history. They are used by action selection and diagnostics,
# but are never injected automatically into reply generation.
MAX_EXECUTED_ACTION_HISTORY_TURNS = 64
FULL_INSTRUCTIONS_LOG_ENV = "SGLANG_OMNI_REALTIME_LOG_FULL_INSTRUCTIONS"
ACTION_SELECTION_MODE_ENV = "SGLANG_OMNI_ACTION_SELECTION_MODE"
ACTION_SELECTION_MODE_HIERARCHICAL = "hierarchical"
ACTION_SELECTION_MODE_FLAT_CHILDREN = "flat_children"
ACTION_MICRO_BATCH_SIZE_ENV = "SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE"
DEFAULT_ACTION_MICRO_BATCH_SIZE = 64
ACTION_CATEGORY_TOP_K_ENV = "SGLANG_OMNI_ACTION_CATEGORY_TOP_K"
DEFAULT_ACTION_CATEGORY_TOP_K = 1
MAX_ACTION_CATEGORY_TOP_K = 3
TURN_ORIGIN_USER = "user"
TURN_ORIGIN_PROACTIVE = "proactive"
IMAGE_ROLE_USER_CAMERA = "user_camera"
IMAGE_ROLE_AVATAR_STATE = "avatar_state"
IMAGE_SOURCE_AVATAR_CURRENT = "avatar_current"
IMAGE_ROLES = {IMAGE_ROLE_USER_CAMERA, IMAGE_ROLE_AVATAR_STATE}
DEFAULT_IMAGE_ROLE_BY_ORIGIN = {
    TURN_ORIGIN_USER: IMAGE_ROLE_USER_CAMERA,
    TURN_ORIGIN_PROACTIVE: IMAGE_ROLE_AVATAR_STATE,
}
TEXT_ROLE_USER_INPUT = "user_input"
TEXT_ROLE_CHARACTER_REPLY = "character_reply"
TURN_PHASE_COLLECTING = "collecting"
TURN_PHASE_PROCESSING = "processing"
TURN_PHASE_CANCELLING = "cancelling"
TURN_PHASE_COMPLETED = "completed"
TURN_TEXT_ROLE_BY_ORIGIN = {
    TURN_ORIGIN_USER: TEXT_ROLE_USER_INPUT,
    TURN_ORIGIN_PROACTIVE: TEXT_ROLE_CHARACTER_REPLY,
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _text_audit_fields(prefix: str, value: str | None) -> dict[str, Any]:
    text = value if isinstance(value, str) else None
    return {
        f"{prefix}_present": bool(text and text.strip()),
        f"{prefix}_chars": len(text) if text is not None else None,
        f"{prefix}_sha256": (
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text is not None
            else None
        ),
    }


def _json_audit_fields(prefix: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_present": False,
            f"{prefix}_chars": None,
            f"{prefix}_sha256": None,
        }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        f"{prefix}_present": bool(value),
        f"{prefix}_chars": len(encoded),
        f"{prefix}_sha256": (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        ),
    }


def normalize_action_micro_batch_size(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get(ACTION_MICRO_BATCH_SIZE_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_ACTION_MICRO_BATCH_SIZE
    try:
        batch_size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ACTION_MICRO_BATCH_SIZE_ENV} must be an integer between 1 and "
            f"{MAX_MICRO_BATCH_SIZE}; got {raw!r}"
        ) from exc
    if not 1 <= batch_size <= MAX_MICRO_BATCH_SIZE:
        raise ValueError(
            f"{ACTION_MICRO_BATCH_SIZE_ENV} must be between 1 and "
            f"{MAX_MICRO_BATCH_SIZE}; got {batch_size}"
        )
    return batch_size


def normalize_action_category_top_k(value: int | str | None = None) -> int:
    """Normalize optional hierarchical category fallback width.

    The default remains Top-1, preserving the existing hierarchical prompt and
    latency.  Values greater than one deliberately score the children of the
    best categories together, which is an accuracy/latency trade-off for
    ambiguous category boundaries.
    """
    raw = value if value is not None else os.environ.get(ACTION_CATEGORY_TOP_K_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_ACTION_CATEGORY_TOP_K
    try:
        top_k = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ACTION_CATEGORY_TOP_K_ENV} must be an integer between 1 and "
            f"{MAX_ACTION_CATEGORY_TOP_K}; got {raw!r}"
        ) from exc
    if not 1 <= top_k <= MAX_ACTION_CATEGORY_TOP_K:
        raise ValueError(
            f"{ACTION_CATEGORY_TOP_K_ENV} must be between 1 and "
            f"{MAX_ACTION_CATEGORY_TOP_K}; got {top_k}"
        )
    return top_k


def _action_timing_breakdown(stats: dict[str, Any]) -> dict[str, Any]:
    """Expose stable action latency buckets without the diagnostic GPU payload."""
    suffix_batch_ms = [float(value) for value in stats.get("suffix_batch_ms", [])]
    suffix_queue_ms = [
        float(value) for value in stats.get("suffix_batch_queue_wait_ms", [])
    ]
    return {
        "client": {
            "request_build_ms": float(stats.get("client_request_build_ms", 0.0)),
            "slot_wait_ms": float(stats.get("action_slot_wait_ms", 0.0)),
            "result_processing_ms": float(
                stats.get("client_result_processing_ms", 0.0)
            ),
            "total_ms": float(stats.get("client_total_ms", 0.0)),
        },
        "pipeline": {
            "coordinator_ms": float(stats.get("coordinator_pipeline_ms", 0.0)),
            "preprocessing_ms": float(stats.get("preprocessing_ms", 0.0)),
            "image_encoder_ms": float(stats.get("image_encoder_ms", 0.0)),
            "audio_encoder_ms": float(stats.get("audio_encoder_ms", 0.0)),
            "mm_aggregate_ms": float(stats.get("mm_aggregate_ms", 0.0)),
            "stages": dict(stats.get("pipeline_stage_timing", {})),
        },
        "scheduler": {
            "request_build_ms": float(stats.get("server_request_build_ms", 0.0)),
            "admission_ms": float(stats.get("scheduler_admission_ms", 0.0)),
            "wait_ms": float(stats.get("scheduler_wait_ms", 0.0)),
            "prefix_prefill_ms": float(stats.get("prefix_prefill_ms", 0.0)),
        },
        "suffix": {
            "batch_count": int(stats.get("suffix_batch_count", len(suffix_batch_ms))),
            "batch_sizes": list(stats.get("suffix_batch_sizes", [])),
            "batch_ms": suffix_batch_ms,
            "batch_total_ms": round(sum(suffix_batch_ms), 3),
            "queue_wait_ms": suffix_queue_ms,
            "queue_wait_total_ms": round(sum(suffix_queue_ms), 3),
            "aggregation_ms": float(stats.get("aggregation_ms", 0.0)),
        },
        "server_total_ms": float(stats.get("total_ms", 0.0)),
    }


def normalize_action_selection_mode(value: str | None) -> str:
    mode = (
        (
            value
            or os.environ.get(ACTION_SELECTION_MODE_ENV)
            or ACTION_SELECTION_MODE_HIERARCHICAL
        )
        .strip()
        .lower()
    )
    if mode not in {
        ACTION_SELECTION_MODE_HIERARCHICAL,
        ACTION_SELECTION_MODE_FLAT_CHILDREN,
    }:
        raise ValueError(
            f"{ACTION_SELECTION_MODE_ENV} must be one of "
            f"{ACTION_SELECTION_MODE_HIERARCHICAL!r}, "
            f"{ACTION_SELECTION_MODE_FLAT_CHILDREN!r}; got {mode!r}"
        )
    return mode


def try_normalize_action_selection_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    if mode in {
        ACTION_SELECTION_MODE_HIERARCHICAL,
        ACTION_SELECTION_MODE_FLAT_CHILDREN,
    }:
        return mode
    return None


def _summarize_media(values: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "chars": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "data_uri_header": (
                value.split(",", 1)[0]
                if value.startswith("data:") and "," in value
                else None
            ),
        }
        for index, value in enumerate(values)
    ]


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
    category_preferences: str = ""
    action_preferences: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionProfile":
        if not isinstance(value, dict):
            raise ValueError("action_profile must be an object")
        allowed_fields = {
            "persona",
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
            if len(normalized) > MAX_ACTION_PROFILE_FIELD_CHARS:
                raise ValueError(
                    f"action_profile.persona.{field_name} must contain at most "
                    f"{MAX_ACTION_PROFILE_FIELD_CHARS} characters"
                )
            persona.append((field_name, normalized))

        preferences: dict[str, str] = {}
        for field_name in ("category_preferences", "action_preferences"):
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
            category_preferences=preferences["category_preferences"],
            action_preferences=preferences["action_preferences"],
        )
        if not profile.persona and not (
            profile.category_preferences or profile.action_preferences
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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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


class MultimodalSession:
    """Manual-turn, multimodal session for audio chunks and image frames.

    This session owns the ``/v1/session/realtime`` protocol, including
    explicit ``turn.start``/``turn.commit`` boundaries and the fixed
    scheme-B action catalog.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        action_selection_mode: str | None = None,
        action_micro_batch_size: int | None = None,
        action_category_top_k: int | None = None,
        global_action_catalog: GlobalActionCatalog | None = None,
        global_action_prewarm: GlobalActionCatalogPrewarmStatus | None = None,
        allow_unregistered_protocol_actions: bool = False,
        claim_session: Callable[[str, "MultimodalSession"], None],
        release_session: Callable[[str, "MultimodalSession"], None],
        request_resource_sample: Callable[..., bool] | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(
            action_selection_mode
        )
        self.action_micro_batch_size = normalize_action_micro_batch_size(
            action_micro_batch_size
        )
        self.action_category_top_k = normalize_action_category_top_k(
            action_category_top_k
        )
        self.global_action_catalog = global_action_catalog
        self.allow_unregistered_protocol_actions = allow_unregistered_protocol_actions
        self.global_action_prewarm = (
            global_action_prewarm or GlobalActionCatalogPrewarmStatus.not_run()
        )
        self.claim_session = claim_session
        self.release_session = release_session
        self.request_resource_sample = request_resource_sample
        self.log_full_instructions = _env_flag(FULL_INSTRUCTIONS_LOG_ENV)

        self.session_id: str | None = None
        self.protocol_version: int | None = None
        self.locale = DEFAULT_ACTION_PROMPT_LOCALE
        self.language = "en"
        self.instructions = ""
        self.unsupported_action_text = ""
        self.action_profile: SessionActionProfile | None = None
        self.modalities: tuple[str, ...] = DEFAULT_MODALITIES
        self.closed = False
        self.started = False
        self.active_turn: TurnBuffer | None = None
        self.used_turn_ids: set[str] = set()
        self.cancelled_turn_ids: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.history_audios: list[str] = []
        self.history_images: list[str] = []
        self.history_image_roles: list[str] = []
        self._image_preprocess_semaphore = asyncio.Semaphore(2)
        self.history_turns: list[ActionHistoryTurn] = []
        self.reply_history_turns: list[ReplyHistoryTurn] = []
        self.executed_action_history: list[ExecutedActionRecord] = []
        self.last_executed_action: ExecutedActionRecord | None = None
        # A proactive action can become the current physical state, but must
        # not replace the target of a later user reference such as "do that
        # last action again".
        self.last_user_executed_action: ExecutedActionRecord | None = None
        self.candidates: list[SessionActionCandidate] = []
        self.categories: list[SessionActionCategory] = []
        self.fallback_category_ids: tuple[str, ...] = ()
        self.candidate_by_id: dict[str, SessionActionCandidate] = {}
        self.action_system_prompt = ""
        self.action_catalog_hash = ""
        self.global_action_catalog_hash = (
            global_action_catalog.catalog_hash if global_action_catalog else ""
        )
        self.action_prefix_cache_namespace = ""
        self.action_prefix_prefilled = False
        self._prefilled_action_prefix_namespaces: set[str] = set()
        self.prewarm_child_category_ids: tuple[str, ...] = ()
        self.prewarmed_child_category_ids: list[str] = []
        self.last_avatar_state: dict[str, Any] = {}
        self._send_lock = asyncio.Lock()
        # Detailed candidate scores are useful for diagnostics, but are not
        # needed by the action executor. Keep the production response small
        # unless the caller opts in at session.start.
        self.include_scores = False

    async def run(self) -> None:
        try:
            while not self.closed:
                try:
                    message = await self.websocket.receive()
                except WebSocketDisconnect:
                    logger.info(
                        "[SESSION_ACTION_REALTIME] client disconnected session_id=%s",
                        self.session_id,
                    )
                    break
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] != "websocket.receive":
                    continue
                if message.get("text") is None:
                    await self.send_error(
                        "invalid_request",
                        "binary_frames_not_supported",
                        "Use JSON events with base64 media payloads.",
                    )
                    continue
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError as exc:
                    await self.send_error("invalid_request", "invalid_json", str(exc))
                    continue
                if not isinstance(payload, dict):
                    await self.send_error(
                        "invalid_request",
                        "invalid_event",
                        "Top-level event must be a JSON object.",
                    )
                    continue
                try:
                    await self.dispatch(payload)
                except WebSocketDisconnect:
                    logger.info(
                        "[SESSION_ACTION_REALTIME] client disconnected during event "
                        "session_id=%s event=%s",
                        self.session_id,
                        payload.get("type"),
                    )
                    break
                except (BufferOverflow, ValueError, KeyError) as exc:
                    session_id = (
                        self._event_context_id(payload, "session_id") or self.session_id
                    )
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id
                        if self.active_turn is not None
                        else None
                    )
                    await self.send_error(
                        "invalid_request",
                        self._classify_error(payload, exc),
                        str(exc),
                        session_id=session_id,
                        turn_id=turn_id,
                    )
                except Exception as exc:
                    session_id = (
                        self._event_context_id(payload, "session_id") or self.session_id
                    )
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id
                        if self.active_turn is not None
                        else None
                    )
                    await self.send_error(
                        "server_error",
                        "server_error",
                        str(exc),
                        session_id=session_id,
                        turn_id=turn_id,
                    )
        finally:
            self.closed = True
            await self._cancel_active_turn(send_event=False)
            if self.session_id is not None:
                self.release_session(self.session_id, self)
            await self._close_websocket()

    async def _close_websocket(self) -> None:
        """Close only when both sides still allow a close frame.

        Starlette tracks peer state and application state separately. A failed
        server send can make application_state DISCONNECTED while client_state
        is still CONNECTED, so checking only client_state can attempt to send a
        second close frame and raise RuntimeError.
        """
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        try:
            await self.websocket.close()
        except (OSError, WebSocketDisconnect, RuntimeError):
            logger.debug(
                "[SESSION_ACTION_REALTIME] websocket already closed session_id=%s",
                self.session_id,
                exc_info=True,
            )

    @staticmethod
    def _strict_object(
        value: Any,
        name: str,
        *,
        allowed: set[str],
        required: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"{name} contains unsupported fields: {', '.join(unknown)}"
            )
        missing = sorted((required or set()) - set(value))
        if missing:
            raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
        return value

    @staticmethod
    def _bounded_optional_text(
        value: Any,
        name: str,
        *,
        max_chars: int,
        allow_empty: bool = True,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
        if not allow_empty and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
        if len(value) > max_chars:
            raise ValueError(f"{name} must contain at most {max_chars} characters")
        return value

    @staticmethod
    def _serialized_chars(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _wire_turn_id(self, event: dict[str, Any]) -> str:
        turn_id = self._bounded_optional_text(
            event.get("turn_id"),
            "turn_id",
            max_chars=MAX_TURN_ID_CHARS,
            allow_empty=False,
        )
        assert turn_id is not None
        return turn_id.strip()

    def _normalize_character_profile(self, value: Any) -> dict[str, str]:
        profile = self._strict_object(
            value,
            "character_profile",
            allowed=set(ACTION_PERSONA_FIELDS),
        )
        if not profile:
            raise ValueError("character_profile must contain at least one field")
        normalized: dict[str, str] = {}
        for field_name in ACTION_PERSONA_FIELDS:
            if field_name not in profile:
                continue
            field_value = self._bounded_optional_text(
                profile[field_name],
                f"character_profile.{field_name}",
                max_chars=MAX_ACTION_PROFILE_FIELD_CHARS,
                allow_empty=False,
            )
            assert field_value is not None
            normalized[field_name] = field_value.strip()
        return normalized

    def _compact_action_catalog(
        self, action_config: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
        if (
            self.global_action_catalog is None
            and not self.allow_unregistered_protocol_actions
        ):
            raise ValueError("action output requires the server global action catalog")

        raw_fallback_ids = action_config.get("fallback_category_ids")
        if not isinstance(raw_fallback_ids, list) or not raw_fallback_ids:
            raise ValueError("action.fallback_category_ids must be a non-empty list")
        if len(raw_fallback_ids) > MAX_ACTION_CATEGORIES:
            raise ValueError(
                "action.fallback_category_ids must contain at most "
                f"{MAX_ACTION_CATEGORIES} items"
            )
        fallback_category_ids: list[str] = []
        reserved_action_ids = {
            UNSUPPORTED_CATEGORY_SCORE_ID,
            UNSUPPORTED_CHILD_SCORE_ID,
            UNSUPPORTED_DECISION_ID,
        }
        for raw_category_id in raw_fallback_ids:
            if not isinstance(raw_category_id, str) or not raw_category_id.strip():
                raise ValueError(
                    "action.fallback_category_ids items must be non-empty strings"
                )
            category_id = raw_category_id.strip()
            if self.global_action_catalog is None and (
                category_id in reserved_action_ids or category_id == "DEV_ACTIONS"
            ):
                raise ValueError(
                    "action.fallback_category_ids contains a reserved "
                    f"development category_id: {category_id}"
                )
            if category_id in fallback_category_ids:
                raise ValueError(
                    "action.fallback_category_ids must not contain duplicates"
                )
            if (
                self.global_action_catalog is not None
                and category_id not in self.global_action_catalog.category_by_id
            ):
                raise ValueError(
                    "action.fallback_category_ids contains unknown global "
                    f"category_id: {category_id}"
                )
            fallback_category_ids.append(category_id)

        raw_allowed = action_config.get("allowed_candidates")
        if raw_allowed is None:
            raw_allowed = []
        if not isinstance(raw_allowed, list):
            raise ValueError("action.allowed_candidates must be a list when provided")
        if len(raw_allowed) > MAX_ACTION_CANDIDATES:
            raise ValueError(
                "action.allowed_candidates must contain at most "
                f"{MAX_ACTION_CANDIDATES} items"
            )
        bindings: dict[str, dict[str, str]] = {}
        for index, raw_candidate in enumerate(raw_allowed):
            item = self._strict_object(
                raw_candidate,
                f"action.allowed_candidates[{index}]",
                allowed={"candidate_id", "execution_binding"},
                required={"candidate_id"},
            )
            candidate_id = self._bounded_optional_text(
                item.get("candidate_id"),
                f"action.allowed_candidates[{index}].candidate_id",
                max_chars=MAX_TURN_ID_CHARS,
                allow_empty=False,
            )
            assert candidate_id is not None
            candidate_id = candidate_id.strip()
            if self.global_action_catalog is None and (
                candidate_id in reserved_action_ids
                or candidate_id.startswith("DEV_NONE_")
            ):
                raise ValueError(
                    "action.allowed_candidates contains a reserved "
                    f"development candidate_id: {candidate_id}"
                )
            if candidate_id in bindings:
                raise ValueError(f"duplicate action candidate_id: {candidate_id}")
            if (
                self.global_action_catalog is not None
                and candidate_id not in self.global_action_catalog.candidate_by_id
            ):
                raise ValueError(f"unknown global action candidate_id: {candidate_id}")
            binding = item.get("execution_binding") or {}
            if not isinstance(binding, dict) or not all(
                isinstance(key, str) and key.strip() and isinstance(binding_value, str)
                for key, binding_value in binding.items()
            ):
                raise ValueError(
                    "execution_binding must be a dictionary with non-empty "
                    "string keys and string values"
                )
            if self._serialized_chars(binding) > MAX_EXECUTION_BINDING_CHARS:
                raise ValueError(
                    "execution_binding must contain at most "
                    f"{MAX_EXECUTION_BINDING_CHARS} serialized characters"
                )
            if self.global_action_catalog is None and any(
                not binding_value.strip() for binding_value in binding.values()
            ):
                raise ValueError(
                    "execution_binding values must be non-empty strings in "
                    "development mode"
                )
            bindings[candidate_id] = dict(binding)

        if not raw_allowed and self.global_action_catalog is None:
            raise ValueError(
                "action.allowed_candidates must be non-empty in development mode"
            )
        if self.global_action_catalog is None:
            collisions = set(fallback_category_ids) & set(bindings)
            if collisions:
                raise ValueError(
                    "development category_id and candidate_id values must be "
                    "disjoint: " + ", ".join(sorted(collisions))
                )
        if not raw_allowed:
            for category_id in fallback_category_ids:
                for candidate in self.global_action_catalog.category_by_id[
                    category_id
                ].children:
                    bindings[candidate.candidate_id] = {}
        if len(bindings) > MAX_ACTION_CANDIDATES:
            raise ValueError(
                "expanded fallback action candidates must contain at most "
                f"{MAX_ACTION_CANDIDATES} items"
            )

        if self.global_action_catalog is None:
            categories = [
                {
                    "category_id": "DEV_ACTIONS",
                    "source_label": "Development actions",
                    "short_definition": "Session-scoped development actions",
                    "category_path": [],
                    "children": [
                        {
                            "candidate_id": candidate_id,
                            "action_id": candidate_id,
                            "source_label": candidate_id,
                            "short_definition": candidate_id,
                            "execution_binding": binding,
                        }
                        for candidate_id, binding in bindings.items()
                    ],
                }
            ]
            categories.extend(
                {
                    "category_id": category_id,
                    "source_label": category_id,
                    "short_definition": "Development fallback action",
                    "category_path": [],
                    "children": [
                        {
                            "candidate_id": f"DEV_NONE_{index}",
                            "action_id": "no_action",
                            "source_label": "No action",
                            "short_definition": "Keep current avatar state",
                            "execution_binding": {},
                        }
                    ],
                }
                for index, category_id in enumerate(fallback_category_ids)
            )
        else:
            categories = []
        for category in (
            self.global_action_catalog.categories
            if self.global_action_catalog is not None
            else ()
        ):
            children = []
            for candidate in category.children:
                if candidate.candidate_id not in bindings:
                    continue
                children.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "action_id": candidate.action_id,
                        "source_label": candidate.source_label,
                        "short_definition": candidate.source_short_definition,
                        "execution_binding": bindings[candidate.candidate_id],
                    }
                )
            if children:
                categories.append(
                    {
                        "category_id": category.category_id,
                        "source_label": category.source_label,
                        "short_definition": category.short_definition,
                        "category_path": list(category.category_path),
                        "children": children,
                    }
                )

        allowed_category_ids = {item["category_id"] for item in categories}
        for category_id in fallback_category_ids:
            if category_id not in allowed_category_ids:
                raise ValueError(
                    "action.fallback_category_ids category must have at least "
                    "one allowed candidate in this Session: "
                    f"{category_id}"
                )

        profile_payload: dict[str, Any] = {}
        category_guidance = self._bounded_optional_text(
            action_config.get("category_guidance"),
            "action.category_guidance",
            max_chars=MAX_ACTION_PROFILE_FIELD_CHARS,
        )
        candidate_guidance = self._bounded_optional_text(
            action_config.get("candidate_guidance"),
            "action.candidate_guidance",
            max_chars=MAX_ACTION_PROFILE_FIELD_CHARS,
        )
        if category_guidance and category_guidance.strip():
            profile_payload["category_preferences"] = category_guidance.strip()
        if candidate_guidance and candidate_guidance.strip():
            profile_payload["action_preferences"] = candidate_guidance.strip()
        return categories, profile_payload or None, fallback_category_ids

    def _normalize_session_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._strict_object(
            payload,
            "session.start",
            allowed={
                "type",
                "protocol_version",
                "session_id",
                "outputs",
                "locale",
                "character_profile",
                "reply",
                "action",
                "input_audio",
                "diagnostics",
            },
            required={"type", "protocol_version", "session_id"},
        )
        version = event.get("protocol_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("protocol_version must be an integer")
        if version != REALTIME_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version: {version}; supported: "
                f"{REALTIME_PROTOCOL_VERSION}"
            )
        session_id = self._bounded_optional_text(
            event.get("session_id"),
            "session_id",
            max_chars=MAX_SESSION_ID_CHARS,
            allow_empty=False,
        )
        assert session_id is not None
        outputs = self._normalize_outputs(event.get("outputs"))
        locale = event.get("locale", DEFAULT_ACTION_PROMPT_LOCALE)
        locale_to_language = PROMPT_LANGUAGE_BY_LOCALE
        if locale not in locale_to_language:
            raise ValueError("locale must be 'zh-CN' or 'en-US'")

        character_profile = (
            self._normalize_character_profile(event["character_profile"])
            if "character_profile" in event
            else {}
        )
        reply_config = event.get("reply", {})
        reply_config = self._strict_object(
            reply_config,
            "reply",
            allowed={"instructions", "unsupported_action_text"},
        )
        if reply_config and "text" not in outputs:
            raise ValueError("reply requires the text output")
        instructions = self._bounded_optional_text(
            reply_config.get("instructions"),
            "reply.instructions",
            max_chars=MAX_INSTRUCTIONS_CHARS,
        )
        unsupported_action_text = self._bounded_optional_text(
            reply_config.get("unsupported_action_text"),
            "reply.unsupported_action_text",
            max_chars=MAX_UNSUPPORTED_ACTION_TEXT_CHARS,
            allow_empty=False,
        )
        fusion_outputs = "text" in outputs and "action" in outputs
        if fusion_outputs and unsupported_action_text is None:
            raise ValueError(
                "reply.unsupported_action_text is required when outputs contain "
                "both text and action"
            )
        if unsupported_action_text is not None and not fusion_outputs:
            raise ValueError(
                "reply.unsupported_action_text requires both text and action outputs"
            )
        # The client owns the complete reply System Prompt, including any
        # persona, defaults, or fallback behavior. Character profile remains
        # action-only context and must never be injected into the reply prompt.
        effective_instructions = instructions or ""

        action_config = event.get("action")
        action_candidates: list[dict[str, Any]] | None = None
        action_profile: dict[str, Any] | None = None
        fallback_category_ids: list[str] = []
        if "action" in outputs:
            action_config = self._strict_object(
                action_config,
                "action",
                allowed={
                    "category_guidance",
                    "candidate_guidance",
                    "allowed_candidates",
                    "fallback_category_ids",
                },
                required={"fallback_category_ids"},
            )
            (
                action_candidates,
                action_profile,
                fallback_category_ids,
            ) = self._compact_action_catalog(action_config)
            if character_profile:
                action_profile = dict(action_profile or {})
                action_profile["persona"] = character_profile
        elif action_config is not None:
            raise ValueError("action requires the action output")

        input_audio = self._strict_object(
            event.get("input_audio", {}),
            "input_audio",
            allowed={"format", "sample_rate_hz", "channels"},
        )
        audio_format = input_audio.get("format", "pcm16le")
        if audio_format != "pcm16le":
            raise ValueError("input_audio.format must be 'pcm16le'")
        sample_rate = input_audio.get("sample_rate_hz", 16000)
        channels = input_audio.get("channels", 1)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise ValueError("input_audio.sample_rate_hz must be an integer")
        if isinstance(channels, bool) or not isinstance(channels, int):
            raise ValueError("input_audio.channels must be an integer")

        diagnostics = self._strict_object(
            event.get("diagnostics", {}),
            "diagnostics",
            allowed={"include_action_scores"},
        )
        include_scores = diagnostics.get("include_action_scores", False)
        if not isinstance(include_scores, bool):
            raise ValueError("diagnostics.include_action_scores must be a boolean")
        if include_scores and "action" not in outputs:
            raise ValueError(
                "diagnostics.include_action_scores requires the action output"
            )

        normalized: dict[str, Any] = {
            "type": "session.start",
            "session_id": session_id.strip(),
            "modalities": list(outputs),
            "language": locale_to_language[locale],
            "instructions": effective_instructions,
            "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
            "input_audio_format": "pcm16",
            "sample_rate": sample_rate,
            "channels": channels,
            "include_scores": include_scores,
            "_protocol_version": version,
            "_locale": locale,
            "_reply_instructions_provided": "instructions" in reply_config,
        }
        if unsupported_action_text is not None:
            normalized["_unsupported_action_text"] = unsupported_action_text.strip()
        if action_candidates is not None:
            normalized["action_candidates"] = action_candidates
        if action_profile is not None:
            normalized["action_profile"] = action_profile
        if "action" in outputs:
            normalized["_fallback_category_ids"] = fallback_category_ids
        return normalized

    def _normalize_wire_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = payload.get("type")
        if event_type == "session.start":
            return self._normalize_session_start(payload)
        if event_type == "turn.start":
            event = self._strict_object(
                payload,
                "turn.start",
                allowed={"type", "turn_id", "origin", "trigger_type"},
                required={"type", "turn_id", "origin"},
            )
            turn_id = self._wire_turn_id(event)
            origin = event.get("origin")
            if origin not in TURN_TEXT_ROLE_BY_ORIGIN:
                raise ValueError("origin must be 'user' or 'proactive'")
            trigger = self._bounded_optional_text(
                event.get("trigger_type"),
                "trigger_type",
                max_chars=MAX_TRIGGER_TYPE_CHARS,
                allow_empty=False,
            )
            if origin == TURN_ORIGIN_USER and trigger is not None:
                raise ValueError("trigger_type is only supported for proactive turns")
            return {
                "type": "turn.start",
                "turn_id": turn_id,
                "turn_origin": origin,
                "text_role": TURN_TEXT_ROLE_BY_ORIGIN[origin],
                "trigger": trigger,
            }
        if event_type == "input.text.set":
            event = self._strict_object(
                payload,
                "input.text.set",
                allowed={"type", "turn_id", "text"},
                required={"type", "turn_id", "text"},
            )
            text = self._bounded_optional_text(
                event.get("text"),
                "text",
                max_chars=MAX_TURN_TEXT_CHARS,
            )
            turn = self._require_collecting_turn(event)
            if turn.turn_origin != TURN_ORIGIN_USER:
                raise ValueError(
                    "input.text.set is only supported for user-origin turns; "
                    "use turn.commit.reply.provided_text for proactive replies"
                )
            return {
                "type": "turn.text.update",
                "turn_id": self._wire_turn_id(event),
                "text": text,
            }
        if event_type == "input.audio.append":
            event = self._strict_object(
                payload,
                "input.audio.append",
                allowed={"type", "turn_id", "seq", "data"},
                required={"type", "turn_id", "seq", "data"},
            )
            data = event.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("input.audio.append.data must be non-empty base64")
            if data.startswith("data:"):
                raise ValueError(
                    "input.audio.append.data must be raw base64 without a data URI header"
                )
            return {
                "type": "input_audio.append",
                "turn_id": self._wire_turn_id(event),
                "seq": event.get("seq"),
                "audio": data,
            }
        if event_type == "input.image.append":
            event = self._strict_object(
                payload,
                "input.image.append",
                allowed={
                    "type",
                    "turn_id",
                    "seq",
                    "capture_timestamp_ms",
                    "media_type",
                    "image_source",
                    "data",
                },
                required={"type", "turn_id", "seq", "data"},
            )
            source = event.get("image_source")
            if source is None:
                active_origin = (
                    self.active_turn.turn_origin
                    if self.active_turn is not None
                    else None
                )
                source = (
                    "user_camera"
                    if active_origin == TURN_ORIGIN_USER
                    else IMAGE_SOURCE_AVATAR_CURRENT
                )
            source_map = {
                "user_camera": IMAGE_ROLE_USER_CAMERA,
                IMAGE_SOURCE_AVATAR_CURRENT: IMAGE_ROLE_AVATAR_STATE,
            }
            if source not in source_map:
                raise ValueError(
                    "image_source must be 'user_camera' or 'avatar_current'"
                )
            data = event.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("input.image.append.data must be non-empty base64")
            if data.startswith("data:"):
                raise ValueError(
                    "input.image.append.data must be raw base64 without a data URI header"
                )
            return {
                "type": "input_image.append",
                "turn_id": self._wire_turn_id(event),
                "seq": event.get("seq"),
                "timestamp_ms": event.get("capture_timestamp_ms", 0),
                "mime_type": event.get("media_type", "image/jpeg"),
                "image_role": source_map[source],
                "image": data,
            }
        if event_type == "turn.commit":
            event = self._strict_object(
                payload,
                "turn.commit",
                allowed={"type", "turn_id", "reply", "action", "avatar_state"},
                required={"type", "turn_id"},
            )
            turn = self._require_collecting_turn(event)
            reply = self._strict_object(
                event.get("reply", {}),
                "turn.commit.reply",
                allowed={"context", "provided_text"},
            )
            context = self._bounded_optional_text(
                reply.get("context"),
                "turn.commit.reply.context",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            provided_text = self._bounded_optional_text(
                reply.get("provided_text"),
                "turn.commit.reply.provided_text",
                max_chars=MAX_TURN_TEXT_CHARS,
            )
            reply_provided = "provided_text" in reply and provided_text is not None
            if reply and "text" not in self.modalities:
                raise ValueError("turn.commit.reply requires the text output")
            if reply_provided and turn.turn_origin != TURN_ORIGIN_PROACTIVE:
                raise ValueError(
                    "reply.provided_text is only supported for proactive turns"
                )
            if reply_provided and context is not None:
                raise ValueError(
                    "reply.provided_text and reply.context are mutually exclusive"
                )

            action = self._strict_object(
                event.get("action", {}),
                "turn.commit.action",
                allowed={"last_executed_action_id", "guidance"},
            )
            if action and "action" not in self.modalities:
                raise ValueError("turn.commit.action requires the action output")
            last_action_id = self._bounded_optional_text(
                action.get("last_executed_action_id"),
                "turn.commit.action.last_executed_action_id",
                max_chars=MAX_TURN_ID_CHARS,
                allow_empty=False,
            )
            if last_action_id is not None and self.global_action_catalog is not None:
                known_action_ids = {
                    candidate.action_id
                    for candidate in self.global_action_catalog.candidate_by_id.values()
                }
                if last_action_id.strip() not in known_action_ids:
                    raise ValueError(
                        "turn.commit.action.last_executed_action_id is not in "
                        f"the global action catalog: {last_action_id.strip()}"
                    )
            guidance = self._bounded_optional_text(
                action.get("guidance"),
                "turn.commit.action.guidance",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            if guidance is not None and turn.turn_origin != TURN_ORIGIN_PROACTIVE:
                raise ValueError(
                    "turn.commit.action.guidance is only supported for proactive turns"
                )

            avatar_state = event.get("avatar_state", {})
            if not isinstance(avatar_state, dict):
                raise ValueError("turn.commit.avatar_state must be an object")
            if {"current_action_id", "state_description"} & set(avatar_state):
                raise ValueError(
                    "turn.commit.avatar_state must not contain legacy action fields"
                )
            if self._serialized_chars(avatar_state) > MAX_AVATAR_STATE_CHARS:
                raise ValueError(
                    "turn.commit.avatar_state must contain at most "
                    f"{MAX_AVATAR_STATE_CHARS} serialized characters"
                )
            internal_state = dict(avatar_state)
            if last_action_id is not None:
                internal_state["current_action_id"] = last_action_id.strip()
            if guidance is not None:
                internal_state["state_description"] = guidance
            normalized = {
                "type": "turn.commit",
                "turn_id": self._wire_turn_id(event),
                "turn_origin": turn.turn_origin,
                "text_role": turn.text_role,
                "trigger": turn.trigger,
                "reply_context": context,
                "avatar_state": internal_state,
                "_reply_provided": reply_provided,
            }
            if reply_provided:
                normalized["text"] = provided_text
            return normalized
        if event_type == "turn.cancel":
            event = self._strict_object(
                payload,
                "turn.cancel",
                allowed={"type", "turn_id"},
                required={"type", "turn_id"},
            )
            return {
                "type": "turn.cancel",
                "turn_id": self._wire_turn_id(event),
            }
        if event_type == "session.close":
            self._require_started()
            event = self._strict_object(
                payload,
                "session.close",
                allowed={"type", "reason"},
                required={"type"},
            )
            self._bounded_optional_text(
                event.get("reason"),
                "session.close.reason",
                max_chars=MAX_TRIGGER_TYPE_CHARS,
                allow_empty=False,
            )
            return event
        raise ValueError(f"unsupported event type: {event_type!r}")

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        turn_id = payload.get("turn_id")
        trace_id = None
        if self.active_turn is not None and turn_id == self.active_turn.turn_id:
            trace_id = self.active_turn.trace_id
        emit_structured_log(
            "protocol",
            "ws_event_received",
            session_id=payload.get("session_id") or self.session_id,
            turn_id=turn_id,
            trace_id=trace_id,
            ws_event_type=event_type,
            payload_keys=sorted(str(key) for key in payload),
        )
        normalized = self._normalize_wire_event(payload)
        normalized_type = normalized["type"]
        handlers = {
            "session.start": self.handle_session_start,
            "turn.start": self.handle_turn_start,
            "input_audio.append": self.handle_audio_append,
            "input_image.append": self.handle_image_append,
            "turn.text.update": self.handle_text_update,
            "turn.commit": self.handle_turn_commit,
            "turn.cancel": self.handle_turn_cancel,
            "session.close": self.handle_session_close,
        }
        handler = handlers.get(normalized_type)
        if handler is None:
            raise ValueError(f"unsupported event type: {event_type!r}")
        if normalized_type == "turn.commit":
            await self._dispatch_turn_commit(normalized)
            return
        await handler(normalized)

    async def _dispatch_turn_commit(self, event: dict[str, Any]) -> None:
        """Start committed-turn inference without blocking WebSocket input."""
        turn = self._require_collecting_turn(event)
        task = asyncio.create_task(
            self.handle_turn_commit(event),
            name=f"session-action-{self.session_id}-{turn.turn_id}",
        )
        turn.inference_task = task
        await asyncio.sleep(0)
        if task.done():
            await task

    @staticmethod
    def _validate_catalog_semantic_field(
        payload: dict[str, Any],
        field: str,
        expected: Any,
        *,
        entity_id: str,
    ) -> None:
        if field not in payload:
            return
        actual = payload[field]
        if field == "category_path":
            if not isinstance(actual, list) or not all(
                isinstance(item, str) for item in actual
            ):
                raise ValueError(f"category_path must be a string list: {entity_id!r}")
            actual = tuple(item.strip() for item in actual)
        elif isinstance(actual, str):
            actual = actual.strip()
        if actual != expected:
            raise ValueError(
                f"{field} does not match the global catalog: {entity_id!r}"
            )

    def _canonicalize_global_hierarchical_catalog(
        self,
        raw_categories: list[Any],
    ) -> tuple[list[SessionActionCategory], list[SessionActionCandidate]]:
        catalog = self.global_action_catalog
        if catalog is None:
            raise RuntimeError("global action catalog is not configured")
        categories: list[SessionActionCategory] = []
        candidates: list[SessionActionCandidate] = []
        for raw_category in raw_categories:
            if not isinstance(raw_category, dict):
                raise ValueError("action category must be an object")
            category_id = raw_category.get("category_id")
            if not isinstance(category_id, str) or not category_id.strip():
                raise ValueError("category_id must be a non-empty string")
            category_id = category_id.strip()
            global_category = catalog.category_by_id.get(category_id)
            if global_category is None:
                raise ValueError(
                    f"unknown category_id in global action catalog: {category_id!r}"
                )
            self._validate_catalog_semantic_field(
                raw_category,
                "source_label",
                global_category.source_label,
                entity_id=category_id,
            )
            self._validate_catalog_semantic_field(
                raw_category,
                "short_definition",
                global_category.short_definition,
                entity_id=category_id,
            )
            self._validate_catalog_semantic_field(
                raw_category,
                "category_path",
                global_category.category_path,
                entity_id=category_id,
            )
            raw_children = raw_category.get("children")
            if not isinstance(raw_children, list) or not raw_children:
                raise ValueError(
                    f"category children must be a non-empty list: {category_id!r}"
                )
            session_children: list[SessionActionCandidate] = []
            for raw_child in raw_children:
                parsed = SessionActionCandidate.from_payload(raw_child)
                global_child = catalog.candidate_by_id.get(parsed.candidate_id)
                if global_child is None:
                    raise ValueError(
                        "unknown candidate_id in global action catalog: "
                        f"{parsed.candidate_id!r}"
                    )
                if global_child.category_id != category_id:
                    raise ValueError(
                        f"candidate_id {parsed.candidate_id!r} belongs to category "
                        f"{global_child.category_id!r}, not {category_id!r}"
                    )
                if parsed.action_id != global_child.action_id:
                    raise ValueError(
                        "action_id does not match the global catalog: "
                        f"{parsed.candidate_id!r}"
                    )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "source_label",
                    global_child.source_label,
                    entity_id=parsed.candidate_id,
                )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "short_definition",
                    global_child.source_short_definition,
                    entity_id=parsed.candidate_id,
                )
                child = SessionActionCandidate(
                    candidate_id=global_child.candidate_id,
                    action_id=global_child.action_id,
                    source_label=global_child.source_label,
                    short_definition=global_child.short_definition,
                    execution_binding=dict(parsed.execution_binding),
                    category_id=category_id,
                )
                session_children.append(child)
                candidates.append(child)
            categories.append(
                SessionActionCategory(
                    category_id=global_category.category_id,
                    source_label=global_category.source_label,
                    short_definition=global_category.short_definition,
                    category_path=global_category.category_path,
                    children=tuple(session_children),
                )
            )
        return categories, candidates

    def _canonicalize_global_flat_catalog(
        self,
        raw_candidates: list[Any],
    ) -> list[SessionActionCandidate]:
        catalog = self.global_action_catalog
        if catalog is None:
            raise RuntimeError("global action catalog is not configured")
        candidates: list[SessionActionCandidate] = []
        for raw_child in raw_candidates:
            parsed = SessionActionCandidate.from_payload(raw_child)
            global_child = catalog.candidate_by_id.get(parsed.candidate_id)
            if global_child is None:
                raise ValueError(
                    "unknown candidate_id in global action catalog: "
                    f"{parsed.candidate_id!r}"
                )
            if parsed.action_id != global_child.action_id:
                raise ValueError(
                    "action_id does not match the global catalog: "
                    f"{parsed.candidate_id!r}"
                )
            self._validate_catalog_semantic_field(
                raw_child,
                "source_label",
                global_child.source_label,
                entity_id=parsed.candidate_id,
            )
            self._validate_catalog_semantic_field(
                raw_child,
                "short_definition",
                global_child.source_short_definition,
                entity_id=parsed.candidate_id,
            )
            candidates.append(
                SessionActionCandidate(
                    candidate_id=global_child.candidate_id,
                    action_id=global_child.action_id,
                    source_label=global_child.source_label,
                    short_definition=global_child.short_definition,
                    execution_binding=dict(parsed.execution_binding),
                    category_id=global_child.category_id,
                )
            )
        return candidates

    async def handle_session_start(self, event: dict[str, Any]) -> None:
        if self.started:
            raise ValueError("session.start can only be sent once")
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        modalities = self._normalize_modalities(event.get("modalities"))
        raw_instructions = event.get("instructions")
        raw_unsupported_action_text = event.get(
            "_unsupported_action_text", event.get("unsupported_action_text")
        )
        raw_action_profile = event.get("action_profile")
        emit_structured_log(
            "lifecycle",
            "session_start_received",
            session_id=session_id.strip(),
            protocol_version=event.get("_protocol_version"),
            modalities=list(modalities),
            requested_selection_mode=event.get("selection_mode"),
            instructions_provided=event.get(
                "_reply_instructions_provided", "instructions" in event
            ),
            **_text_audit_fields("instructions", raw_instructions),
            **_text_audit_fields(
                "unsupported_action_text", raw_unsupported_action_text
            ),
            action_profile_provided="action_profile" in event,
            **_json_audit_fields("action_profile", raw_action_profile),
            fallback_category_ids=event.get(
                "_fallback_category_ids", event.get("fallback_category_ids")
            ),
        )

        action_profile = (
            SessionActionProfile.from_payload(raw_action_profile)
            if raw_action_profile is not None
            else None
        )
        if action_profile is not None and "action" not in modalities:
            raise ValueError("action_profile requires the action modality")
        action_profile_payload = (
            action_profile.as_dict() if action_profile is not None else None
        )
        action_profile_audit = _json_audit_fields(
            "action_profile", action_profile_payload
        )

        raw_candidates = event.get("action_candidates")
        raw_fallback_category_ids = event.get(
            "_fallback_category_ids", event.get("fallback_category_ids")
        )
        categories: list[SessionActionCategory] = []
        candidates: list[SessionActionCandidate] = []
        fallback_category_ids: list[str] = []
        if "action" in modalities:
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(
                    "action_candidates must be a non-empty list when action modality is enabled"
                )
            if len(raw_candidates) > MAX_ACTION_CANDIDATES:
                raise ValueError(
                    f"action_candidates must contain at most {MAX_ACTION_CANDIDATES} items"
                )

            nested = all(
                isinstance(x, dict) and "children" in x for x in raw_candidates
            )
            categories = (
                [SessionActionCategory.from_payload(x) for x in raw_candidates]
                if nested
                else []
            )
            if categories:
                if len(categories) > MAX_ACTION_CATEGORIES:
                    raise ValueError(
                        f"action categories must contain at most {MAX_ACTION_CATEGORIES} items"
                    )
                candidates = [
                    child for category in categories for child in category.children
                ]
                all_ids = [category.category_id for category in categories] + [
                    x.candidate_id for x in candidates
                ]
                if len(set(all_ids)) != len(all_ids):
                    raise ValueError(
                        "action category and candidate IDs must be globally unique"
                    )
            else:
                candidates = [
                    SessionActionCandidate.from_payload(x) for x in raw_candidates
                ]
                candidate_ids = [x.candidate_id for x in candidates]
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError("action candidate IDs must be unique")
            if self.global_action_catalog is not None:
                if categories:
                    categories, candidates = (
                        self._canonicalize_global_hierarchical_catalog(raw_candidates)
                    )
                else:
                    candidates = self._canonicalize_global_flat_catalog(raw_candidates)
            if len(candidates) > MAX_ACTION_CANDIDATES:
                raise ValueError(
                    f"action candidates must contain at most {MAX_ACTION_CANDIDATES} children"
                )
            if categories:
                if (
                    not isinstance(raw_fallback_category_ids, list)
                    or not raw_fallback_category_ids
                ):
                    # The public protocol is already rejected by
                    # _normalize_session_start. Keep the legacy/internal test
                    # path compatible by recovering its historical no_action
                    # category instead of weakening the wire contract.
                    legacy_fallback = next(
                        (
                            category.category_id
                            for category in categories
                            if any(
                                child.action_id == "no_action"
                                for child in category.children
                            )
                        ),
                        None,
                    )
                    if event.get("_protocol_version") is None and legacy_fallback:
                        raw_fallback_category_ids = [legacy_fallback]
                    else:
                        raise ValueError(
                            "fallback_category_ids must be a non-empty list when "
                            "hierarchical action selection is enabled"
                        )
                known_category_ids = {item.category_id for item in categories}
                for raw_category_id in raw_fallback_category_ids:
                    if (
                        not isinstance(raw_category_id, str)
                        or not raw_category_id.strip()
                    ):
                        raise ValueError(
                            "fallback_category_ids items must be non-empty strings"
                        )
                    category_id = raw_category_id.strip()
                    if category_id in fallback_category_ids:
                        raise ValueError(
                            "fallback_category_ids must not contain duplicates"
                        )
                    if category_id not in known_category_ids:
                        raise ValueError(
                            "fallback_category_ids contains a category without "
                            "an executable candidate in this Session: "
                            f"{category_id}"
                        )
                    fallback_category_ids.append(category_id)
            elif raw_fallback_category_ids is not None:
                raise ValueError(
                    "fallback_category_ids requires hierarchical action_candidates"
                )
        elif raw_candidates is not None and not isinstance(raw_candidates, list):
            raise ValueError("action_candidates must be a list when provided")
        elif raw_fallback_category_ids is not None:
            raise ValueError("fallback_category_ids requires the action modality")

        raw_prewarm_category_ids = event.get("prewarm_child_category_ids", [])
        if not isinstance(raw_prewarm_category_ids, list):
            raise ValueError("prewarm_child_category_ids must be a list")
        if len(raw_prewarm_category_ids) > MAX_PREWARM_CHILD_CATEGORIES:
            raise ValueError(
                "prewarm_child_category_ids must contain at most "
                f"{MAX_PREWARM_CHILD_CATEGORIES} items"
            )
        prewarm_child_category_ids: list[str] = []
        for raw_category_id in raw_prewarm_category_ids:
            if not isinstance(raw_category_id, str) or not raw_category_id.strip():
                raise ValueError(
                    "prewarm_child_category_ids items must be non-empty strings"
                )
            category_id = raw_category_id.strip()
            if category_id in prewarm_child_category_ids:
                raise ValueError(
                    "prewarm_child_category_ids must not contain duplicates"
                )
            prewarm_child_category_ids.append(category_id)
        if prewarm_child_category_ids:
            if not categories:
                raise ValueError(
                    "prewarm_child_category_ids requires hierarchical action_candidates"
                )
            known_category_ids = {item.category_id for item in categories}
            unknown_category_ids = [
                item
                for item in prewarm_child_category_ids
                if item not in known_category_ids
            ]
            if unknown_category_ids:
                raise ValueError(
                    "prewarm_child_category_ids contains unknown category IDs: "
                    + ", ".join(unknown_category_ids)
                )

        language = event.get("language", "en")
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        instructions = event.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        if isinstance(instructions, str) and len(instructions) > MAX_INSTRUCTIONS_CHARS:
            raise ValueError(
                f"instructions must contain at most {MAX_INSTRUCTIONS_CHARS} characters"
            )
        if raw_unsupported_action_text is not None:
            if not isinstance(raw_unsupported_action_text, str):
                raise ValueError("unsupported_action_text must be a string")
            if not raw_unsupported_action_text.strip():
                raise ValueError("unsupported_action_text must be non-empty")
            if len(raw_unsupported_action_text) > MAX_UNSUPPORTED_ACTION_TEXT_CHARS:
                raise ValueError(
                    "unsupported_action_text must contain at most "
                    f"{MAX_UNSUPPORTED_ACTION_TEXT_CHARS} characters"
                )
        include_scores = event.get("include_scores", False)
        if not isinstance(include_scores, bool):
            raise ValueError("include_scores must be a boolean")
        if event.get("input_audio_format", "pcm16") != "pcm16":
            raise ValueError("only pcm16 audio is supported")
        try:
            sample_rate = int(event.get("sample_rate", 16000))
            channels = int(event.get("channels", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_rate and channels must be integers") from exc
        if sample_rate != 16000:
            raise ValueError("only 16000 Hz audio is supported")
        if channels != 1:
            raise ValueError("only mono audio is supported")

        if "selection_mode" in event:
            requested_mode = event.get("selection_mode")
            selected_mode = try_normalize_action_selection_mode(requested_mode)
            if selected_mode is None:
                logger.warning(
                    "[SESSION_ACTION_REALTIME] invalid session.start selection_mode=%r "
                    "session_id=%s; using %s",
                    requested_mode,
                    session_id,
                    self.action_selection_mode,
                )
            else:
                self.action_selection_mode = selected_mode

        if "text" in modalities and "action" in modalities:
            if self.action_selection_mode != ACTION_SELECTION_MODE_HIERARCHICAL:
                raise ValueError(
                    "text and action fusion currently requires selection_mode=hierarchical"
                )
            if not categories:
                raise ValueError(
                    "text and action fusion requires hierarchical action_candidates"
                )
            if self.action_category_top_k != 1:
                raise ValueError(
                    "text and action fusion currently requires action category top-k=1"
                )
        if (
            self.global_action_catalog is not None
            and categories
            and self.action_category_top_k != 1
        ):
            raise ValueError(
                "global hierarchical action catalog currently requires "
                "action category top-k=1"
            )

        self.claim_session(session_id, self)
        self.session_id = session_id
        self.protocol_version = event.get("_protocol_version")
        self.locale = event.get("_locale", "zh-CN" if language == "zh" else "en-US")
        self.language = language
        self.modalities = modalities
        if instructions is not None:
            self.instructions = instructions
        if raw_unsupported_action_text is not None:
            self.unsupported_action_text = raw_unsupported_action_text.strip()
        self.action_profile = action_profile
        if self.log_full_instructions:
            emit_structured_log(
                "diagnostic",
                "session_instructions_received",
                session_id=self.session_id,
                instructions=self.instructions,
                **_text_audit_fields("instructions", self.instructions),
            )
            if self.action_profile is not None:
                emit_structured_log(
                    "diagnostic",
                    "session_action_profile_received",
                    session_id=self.session_id,
                    action_profile=self.action_profile.as_dict(),
                    **action_profile_audit,
                )
        self.include_scores = include_scores
        self.candidates = candidates
        self.categories = categories
        self.fallback_category_ids = tuple(fallback_category_ids)
        self.prewarm_child_category_ids = tuple(prewarm_child_category_ids)
        self.candidate_by_id = {x.candidate_id: x for x in candidates}
        if (
            categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
        ):
            self.action_system_prompt = self._build_category_system_prompt()
        elif candidates:
            self.action_system_prompt = self._build_action_system_prompt()
        else:
            self.action_system_prompt = ""
        canonical = json.dumps(
            (
                [category.as_dict() for category in categories]
                if categories
                else [x.as_dict() for x in candidates]
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.action_catalog_hash = (
            "sha256:" + hashlib.sha256(canonical).hexdigest() if candidates else ""
        )
        mode_namespace = (
            "hierarchical"
            if categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
            else "flat_children"
        )
        self.action_prefix_cache_namespace = (
            self.global_action_catalog.category_cache_namespace(self.locale)
            if self.global_action_catalog is not None and categories
            else f"{mode_namespace}:{self.locale}:{self.action_catalog_hash}"
        )
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if self.global_action_catalog is not None and categories:
            locale_prewarm = self.global_action_prewarm.for_locale(self.locale)
            self.action_prefix_prefilled = locale_prewarm.category_ready
            self.prewarmed_child_category_ids = sorted(
                {item.category_id for item in categories}
                & set(locale_prewarm.ready_child_category_ids)
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    self.action_prefix_cache_namespace
                )
            self._prefilled_action_prefix_namespaces.update(
                self.global_action_catalog.child_cache_namespace(
                    category_id, self.locale
                )
                for category_id in self.prewarmed_child_category_ids
            )
            if prewarm_child_category_ids:
                logger.info(
                    "[SESSION_ACTION_REALTIME] prewarm_child_category_ids is "
                    "deprecated because all global Child prefixes are warmed at "
                    "startup session_id=%s requested=%s",
                    session_id,
                    prewarm_child_category_ids,
                )
        elif candidates and callable(prefill):
            if (
                categories
                and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
            ):
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.category_id,
                        suffix=item.category_id,
                        action_id=item.category_id,
                    )
                    for item in categories
                ]
                prefill_stage = "category"
            else:
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.candidate_id,
                        suffix=item.candidate_id,
                        action_id=item.action_id,
                        execution_binding=dict(item.execution_binding),
                    )
                    for item in candidates
                ]
                prefill_stage = "single"
            self.action_prefix_prefilled = await prefill(
                model=self.model_name,
                system_prompt=self.action_system_prompt,
                candidates=prefill_candidates,
                prefix_cache_namespace=self.action_prefix_cache_namespace,
                stage=prefill_stage,
                language=self.language,
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    self.action_prefix_cache_namespace
                )
            if categories:
                category_by_id = {item.category_id: item for item in categories}
                for category_id in self.prewarm_child_category_ids:
                    category = category_by_id[category_id]
                    child_candidates = list(category.children)
                    child_namespace = (
                        f"{self.action_prefix_cache_namespace}:child:{category_id}"
                    )
                    child_prewarm_started = time.perf_counter()
                    prewarmed = await prefill(
                        request_id=(
                            f"session-{session_id}-child-prewarm-{category_id}"
                        ),
                        model=self.model_name,
                        system_prompt=self._build_child_system_prompt(
                            category, child_candidates
                        ),
                        candidates=[
                            ActionScoreCandidate(
                                candidate_id=item.candidate_id,
                                suffix=item.candidate_id,
                                action_id=item.action_id,
                                execution_binding=dict(item.execution_binding),
                            )
                            for item in child_candidates
                        ],
                        prefix_cache_namespace=child_namespace,
                        stage="child",
                        language=self.language,
                    )
                    elapsed_ms = round(
                        (time.perf_counter() - child_prewarm_started) * 1000.0,
                        3,
                    )
                    if prewarmed:
                        self._prefilled_action_prefix_namespaces.add(child_namespace)
                        self.prewarmed_child_category_ids.append(category_id)
                    emit_structured_log(
                        "performance",
                        "child_prefix_prewarm_completed",
                        session_id=session_id,
                        category_id=category_id,
                        child_candidate_count=len(child_candidates),
                        prefix_cache_namespace=child_namespace,
                        prewarmed=prewarmed,
                        elapsed_ms=elapsed_ms,
                    )
        self.started = True

        started_payload: dict[str, Any] = {
            "type": "session.started",
            "session_id": self.session_id,
            "model": self.model_name,
            "action_catalog_hash": self.action_catalog_hash,
            "action_candidate_count": len(candidates),
            "action_category_count": len(categories),
            "action_selection_mode": self.action_selection_mode,
            "action_selection_stages": (
                2
                if categories
                and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
                else 1
            ),
            "action_prefix_prefilled": self.action_prefix_prefilled,
            "action_profile_applied": self.action_profile is not None,
            "action_profile_sha256": action_profile_audit["action_profile_sha256"],
            "prewarmed_child_category_ids": list(self.prewarmed_child_category_ids),
            "fallback_category_ids": list(self.fallback_category_ids),
            "unsupported_action_text_configured": bool(self.unsupported_action_text),
            "unsupported_action_text_sha256": _text_audit_fields(
                "unsupported_action_text", self.unsupported_action_text or None
            )["unsupported_action_text_sha256"],
        }
        if self.protocol_version is not None:
            started_payload.update(
                {
                    "protocol_version": self.protocol_version,
                    "outputs": list(self.modalities),
                    "locale": self.locale,
                }
            )
        else:
            started_payload["modalities"] = list(self.modalities)
        if self.global_action_catalog is not None:
            started_payload.update(
                {
                    "session_action_catalog_hash": self.action_catalog_hash,
                    "global_action_catalog_hash": self.global_action_catalog_hash,
                    "global_action_catalog_version": (
                        self.global_action_catalog.catalog_version
                    ),
                }
            )
        await self.send(started_payload)
        emit_structured_log(
            "lifecycle",
            "session_started",
            session_id=self.session_id,
            protocol_version=self.protocol_version,
            locale=self.locale,
            modalities=list(self.modalities),
            action_selection_mode=self.action_selection_mode,
            action_catalog_hash=self.action_catalog_hash,
            session_action_catalog_hash=self.action_catalog_hash,
            global_action_catalog_hash=self.global_action_catalog_hash,
            global_action_catalog_version=(
                self.global_action_catalog.catalog_version
                if self.global_action_catalog is not None
                else None
            ),
            action_candidate_count=len(candidates),
            action_category_count=len(categories),
            action_prefix_prefilled=self.action_prefix_prefilled,
            **_text_audit_fields(
                "unsupported_action_text", self.unsupported_action_text or None
            ),
            **action_profile_audit,
            requested_prewarm_child_category_ids=list(self.prewarm_child_category_ids),
            prewarmed_child_category_ids=list(self.prewarmed_child_category_ids),
        )

    async def handle_turn_start(self, event: dict[str, Any]) -> None:
        self._require_started()
        if self.active_turn is not None:
            raise ValueError("another turn is already active")
        turn_id = event.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError(
                "turn_id must be generated by the caller and be a non-empty string"
            )
        turn_id = turn_id.strip()
        turn_origin, text_role, trigger = self._parse_turn_semantics(event)
        if turn_id in self.used_turn_ids:
            raise ValueError(f"turn_id has already been used: {turn_id}")
        self.used_turn_ids.add(turn_id)
        self.active_turn = TurnBuffer(
            turn_id=turn_id,
            started_at=time.perf_counter(),
            audio=RealtimeAudioBuffer(source_sr=16000, target_sr=16000),
            images=[],
            audio_seqs=set(),
            image_seqs=set(),
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            trace_id=new_trace_id(self.session_id, turn_id),
        )
        emit_structured_log(
            "lifecycle",
            "turn_started",
            session_id=self.session_id,
            turn_id=turn_id,
            trace_id=self.active_turn.trace_id,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            modalities=list(self.modalities),
        )
        await self.send(
            {
                "type": "turn.started",
                "session_id": self.session_id,
                "turn_id": turn_id,
            }
        )

    async def handle_audio_append(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        seq = self._positive_int(event.get("seq"), "seq")
        audio = event.get("audio")
        if not isinstance(audio, str) or not audio:
            raise ValueError("audio must be a non-empty base64 string")
        try:
            decoded = base64.b64decode(audio, validate=True)
        except Exception as exc:
            raise ValueError("audio is not valid base64") from exc
        if len(decoded) % 2:
            raise ValueError("PCM16 audio must contain an even number of bytes")
        chunk_hash = hashlib.sha256(decoded).hexdigest()
        if seq in turn.audio_seqs:
            if turn.audio_chunk_hashes.get(seq) != chunk_hash:
                raise ValueError(
                    f"audio seq {seq} conflicts with the previously accepted chunk"
                )
            turn.duplicate_audio_chunks += 1
            await self._ack_media(turn, "audio", seq, duplicate=True)
            return
        expected_seq = turn.audio_chunk_count + 1
        if seq != expected_seq:
            raise ValueError(
                f"audio seq must be monotonic starting at 1; expected {expected_seq}, got {seq}"
            )
        if turn.audio_chunk_count >= MAX_AUDIO_CHUNKS_PER_TURN:
            raise ValueError(f"audio chunk count exceeds {MAX_AUDIO_CHUNKS_PER_TURN}")
        turn.audio.append_b64(audio)
        turn.audio_seqs.add(seq)
        turn.audio_chunk_hashes[seq] = chunk_hash
        turn.audio_chunk_count += 1
        await self._ack_media(turn, "audio", seq)

    async def _prepare_image_frame(
        self,
        turn: TurnBuffer,
        *,
        seq: int,
        decoded: bytes,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            async with self._image_preprocess_semaphore:
                if turn.phase not in {
                    TURN_PHASE_COLLECTING,
                    TURN_PHASE_PROCESSING,
                }:
                    return {
                        "status": "stale",
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "prepared_bytes": 0,
                    }
                payload = await asyncio.to_thread(
                    prepare_image_bytes_for_wire,
                    decoded,
                )
            prepared_bytes = len(payload["pixel_bytes"])
            if prepared_bytes > MAX_PREPARED_IMAGE_BYTES_PER_FRAME:
                status = "frame_too_large"
                payload = None
                prepared_bytes = 0
            elif (
                turn.prepared_image_bytes + prepared_bytes
                > MAX_PREPARED_IMAGE_BYTES_PER_TURN
            ):
                status = "turn_budget_exceeded"
                payload = None
                prepared_bytes = 0
            else:
                status = "prepared"
                turn.prepared_image_bytes += prepared_bytes
            return {
                "status": status,
                "payload": payload,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "prepared_bytes": prepared_bytes,
            }
        except Exception as exc:
            logger.warning(
                "[SESSION_ACTION_REALTIME] image preprocessing fell back "
                "session_id=%s turn_id=%s seq=%s error=%s",
                self.session_id,
                turn.turn_id,
                seq,
                exc,
            )
            return {
                "status": "failed",
                "error": str(exc),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "prepared_bytes": 0,
            }

    @staticmethod
    def _discard_image_preprocess(turn: TurnBuffer, frame: ImageFrame) -> None:
        task = frame.preprocess_task
        if task is None:
            return
        if task.done() and not task.cancelled():
            try:
                result = task.result()
            except Exception:
                result = {}
            turn.prepared_image_bytes = max(
                turn.prepared_image_bytes - int(result.get("prepared_bytes", 0)),
                0,
            )
        elif not task.done():
            task.cancel()
        frame.preprocess_task = None

    async def _resolve_prepared_images(
        self,
        turn: TurnBuffer,
        frames: list[ImageFrame],
    ) -> tuple[list[Any], dict[str, Any]]:
        wait_started = time.perf_counter()
        tasks = [frame.preprocess_task for frame in frames if frame.preprocess_task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        wait_ms = (time.perf_counter() - wait_started) * 1000.0
        images: list[Any] = []
        results: list[dict[str, Any]] = []
        for frame in frames:
            task = frame.preprocess_task
            result: dict[str, Any] = {}
            if task is not None and task.done() and not task.cancelled():
                try:
                    value = task.result()
                except Exception:
                    value = None
                if isinstance(value, dict):
                    result = value
            payload = result.get("payload")
            images.append(payload if isinstance(payload, dict) else frame.data_uri)
            results.append(result)
        statuses = [str(result.get("status", "not_scheduled")) for result in results]
        stats = {
            "scheduled_count": len(tasks),
            "prepared_count": statuses.count("prepared"),
            "fallback_count": sum(
                status not in {"prepared", "not_scheduled"} for status in statuses
            ),
            "not_scheduled_count": statuses.count("not_scheduled"),
            "worker_total_ms": round(
                sum(float(result.get("elapsed_ms", 0.0)) for result in results), 3
            ),
            "commit_wait_ms": round(wait_ms, 3),
            "prepared_bytes": sum(
                int(result.get("prepared_bytes", 0)) for result in results
            ),
            "statuses": statuses,
        }
        turn.image_preprocess_stats = stats
        return images, stats

    async def handle_image_append(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        seq = self._positive_int(event.get("seq"), "seq")
        image_role = event.get(
            "image_role", DEFAULT_IMAGE_ROLE_BY_ORIGIN[turn.turn_origin]
        )
        if not isinstance(image_role, str) or image_role not in IMAGE_ROLES:
            raise ValueError("image_role must be 'user_camera' or 'avatar_state'")
        image = event.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError("image must be a non-empty base64 string or data URI")
        mime_type = event.get("mime_type", "image/jpeg")
        if mime_type not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("mime_type must be image/jpeg, image/png, or image/webp")
        if image.startswith("data:"):
            if ";base64," not in image:
                raise ValueError("image data URI must use base64 encoding")
            data_uri = image
            encoded = image.split(",", 1)[1]
        else:
            data_uri = f"data:{mime_type};base64,{image}"
            encoded = image
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("image is not valid base64") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
        timestamp_ms = self._nonnegative_int(
            event.get("timestamp_ms", 0), "timestamp_ms"
        )
        signature_payload = {
            "bytes_sha256": hashlib.sha256(decoded).hexdigest(),
            "data_uri_header": data_uri.split(",", 1)[0],
            "image_role": image_role,
            "timestamp_ms": timestamp_ms,
        }
        frame_signature = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if seq in turn.image_seqs:
            if turn.image_frame_signatures.get(seq) != frame_signature:
                raise ValueError(
                    f"image seq {seq} conflicts with the previously accepted frame"
                )
            turn.duplicate_image_frames += 1
            await self._ack_media(
                turn, "image", seq, duplicate=True, image_role=image_role
            )
            return
        if len(turn.images) >= MAX_IMAGES_PER_TURN:
            raise ValueError(f"image frame count exceeds {MAX_IMAGES_PER_TURN}")
        frame = ImageFrame(
            seq=seq,
            timestamp_ms=timestamp_ms,
            data_uri=data_uri,
            image_role=image_role,
        )
        frame.preprocess_task = asyncio.create_task(
            self._prepare_image_frame(turn, seq=seq, decoded=decoded),
            name=f"image-preprocess-{turn.turn_id}-{seq}",
        )
        turn.images.append(frame)
        scheduled_frames = [
            item for item in turn.images if item.preprocess_task is not None
        ]
        while len(scheduled_frames) > MAX_IMAGE_PREPROCESS_TASKS_PER_TURN:
            self._discard_image_preprocess(turn, scheduled_frames.pop(0))
        turn.image_seqs.add(seq)
        turn.image_frame_signatures[seq] = frame_signature
        await self._ack_media(turn, "image", seq, image_role=image_role)

    async def handle_text_update(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        text = event.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError("text must be a string or null")
        turn.text = text
        await self.send(
            {
                "type": (
                    "input.text.ack"
                    if self.protocol_version is not None
                    else "turn.text.updated"
                ),
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "text_present": bool(text),
            }
        )

    async def handle_turn_cancel(self, event: dict[str, Any]) -> None:
        turn_id = event.get("turn_id")
        if turn_id in self.cancelled_turn_ids:
            return
        if (
            self.active_turn is None or self.active_turn.turn_id != turn_id
        ) and turn_id in self.used_turn_ids:
            # Completion and cancellation may cross on the wire. Preserve the
            # first terminal state without emitting an error or second terminal.
            return
        turn = self._require_turn(event)
        await self._cancel_active_turn(send_event=True, expected_turn=turn)
        self.cancelled_turn_ids.add(turn.turn_id)

    async def handle_session_close(self, event: dict[str, Any]) -> None:
        reason = event.get("reason")
        self.closed = True
        await self._cancel_active_turn(send_event=False)
        await self.send({"type": "session.closed", "session_id": self.session_id})
        emit_structured_log(
            "lifecycle",
            "session_closed",
            session_id=self.session_id,
            reason=reason,
            modalities=list(self.modalities),
            completed_reply_turn_count=len(self.reply_history_turns),
            completed_action_turn_count=len(self.history_turns),
            assumed_executed_action_count=len(self.executed_action_history),
            last_executed_action=self._executed_action_log_fields(
                self.last_executed_action
            ),
            last_user_executed_action=self._executed_action_log_fields(
                self.last_user_executed_action
            ),
        )

    async def _cancel_active_turn(
        self,
        *,
        send_event: bool,
        expected_turn: TurnBuffer | None = None,
    ) -> None:
        turn = self.active_turn
        if turn is None:
            return
        if expected_turn is not None and turn is not expected_turn:
            return

        cancel_started = time.perf_counter()
        if turn.phase == TURN_PHASE_PROCESSING:
            request_ids = list(turn.active_request_ids)
            turn.phase = TURN_PHASE_CANCELLING
            if (
                turn.provisional_reply is not None
                and turn.provisional_reply.status == "pending"
            ):
                await self._discard_provisional_reply(
                    turn,
                    turn.provisional_reply,
                    reason="turn_cancelled",
                    send_event=send_event,
                    abort_request=False,
                )
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                abort_results = await asyncio.gather(
                    *(abort(request_id) for request_id in request_ids),
                    return_exceptions=True,
                )
                for request_id, result in zip(request_ids, abort_results):
                    if isinstance(result, Exception):
                        logger.error(
                            "[SESSION_ACTION_REALTIME] direct abort failed "
                            "session_id=%s turn_id=%s request_id=%s error=%s",
                            self.session_id,
                            turn.turn_id,
                            request_id,
                            result,
                        )
            branch_tasks = [task for task in turn.branch_tasks if not task.done()]
            for branch_task in branch_tasks:
                branch_task.cancel()
            if branch_tasks:
                await asyncio.gather(*branch_tasks, return_exceptions=True)
            task = turn.inference_task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        image_tasks = [
            frame.preprocess_task
            for frame in turn.images
            if frame.preprocess_task is not None
        ]
        for image_task in image_tasks:
            if not image_task.done():
                image_task.cancel()
        if image_tasks:
            await asyncio.gather(*image_tasks, return_exceptions=True)

        if self.active_turn is turn:
            self.active_turn = None
        turn.current_request_id = None
        turn.active_request_ids.clear()
        turn.branch_tasks.clear()
        turn.images.clear()
        turn.audio.clear()
        logger.info(
            "[SESSION_ACTION_REALTIME] turn cancelled session_id=%s turn_id=%s "
            "phase=%s elapsed_ms=%.3f",
            self.session_id,
            turn.turn_id,
            turn.phase,
            (time.perf_counter() - cancel_started) * 1000.0,
        )
        if send_event:
            await self.send(
                {
                    "type": "turn.cancelled",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                }
            )
        emit_structured_log(
            "lifecycle",
            "turn_cancelled",
            level="warning",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            elapsed_ms=round((time.perf_counter() - cancel_started) * 1000.0, 3),
        )

    async def handle_turn_commit(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        turn_origin, text_role, trigger = self._parse_turn_semantics(event)
        if (turn_origin, text_role, trigger) != (
            turn.turn_origin,
            turn.text_role,
            turn.trigger,
        ):
            raise ValueError(
                "turn_origin, text_role, and trigger must match turn.start"
            )
        if turn_origin == TURN_ORIGIN_PROACTIVE and event.get("user_input") is not None:
            raise ValueError("user_input must be null or omitted for proactive turns")
        if "text" in event:
            text = event.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be a string or null")
            turn.text = text
        if "_reply_provided" in event:
            turn.reply_provided = bool(event["_reply_provided"])
        elif (
            turn_origin == TURN_ORIGIN_PROACTIVE
            and isinstance(turn.text, str)
            and bool(turn.text.strip())
        ):
            # Internal legacy unit-test path. The public protocol always sends
            # the explicit marker produced by _normalize_wire_event.
            turn.reply_provided = True
        if "reply_context" in event:
            reply_context = event.get("reply_context")
            if reply_context is not None and not isinstance(reply_context, str):
                raise ValueError("reply_context must be a string or null")
            if (
                isinstance(reply_context, str)
                and len(reply_context) > MAX_REPLY_CONTEXT_CHARS
            ):
                raise ValueError(
                    f"reply_context must contain at most {MAX_REPLY_CONTEXT_CHARS} characters"
                )
            turn.reply_context = reply_context
        if (
            turn.reply_provided
            and isinstance(turn.reply_context, str)
            and turn.reply_context is not None
        ):
            raise ValueError("provided reply and reply context are mutually exclusive")
        if "avatar_state" in event:
            state = event.get("avatar_state")
            if not isinstance(state, dict):
                raise ValueError("avatar_state must be an object")
            current_action_id = state.get("current_action_id")
            if current_action_id is not None and (
                not isinstance(current_action_id, str) or not current_action_id.strip()
            ):
                raise ValueError(
                    "avatar_state.current_action_id must be a non-empty string or null"
                )
            state_description = state.get("state_description")
            if state_description is not None and not isinstance(state_description, str):
                raise ValueError("avatar_state.state_description must be a string")
            turn.avatar_state = dict(state)
        current_audio = (
            turn.audio.to_full_wav_data_uri() if not turn.audio.is_empty() else None
        )
        current_image_frames = sorted(
            turn.images, key=lambda x: (x.timestamp_ms, x.seq)
        )
        current_images = [frame.data_uri for frame in current_image_frames]
        current_image_roles = [frame.image_role for frame in current_image_frames]
        current_audio_list = [current_audio] if current_audio else []
        ingest_ms = (time.perf_counter() - turn.started_at) * 1000.0
        turn.phase = TURN_PHASE_PROCESSING
        turn.request_base = (
            f"session-{self.session_id}-turn-{turn.turn_id}-{uuid.uuid4().hex}"
        )

        turn_id = turn.turn_id
        commit_started = time.perf_counter()
        turn.commit_started_at = commit_started
        emit_structured_log(
            "lifecycle",
            "turn_commit_received",
            session_id=self.session_id,
            turn_id=turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            modalities=list(self.modalities),
            audio_chunk_count=turn.audio_chunk_count,
            image_frame_count=len(current_images),
            text_present=bool(turn.text),
            reply_context_present=bool(turn.reply_context),
        )
        turn_outcome = "failed"
        self._request_turn_resource_sample(
            "turn_before_inference",
            turn=turn,
            audio_chunk_count=turn.audio_chunk_count,
            image_frame_count=len(current_images),
        )

        try:
            committed_payload: dict[str, Any] = {
                "type": "turn.committed",
                "session_id": self.session_id,
                "turn_id": turn_id,
                "audio_chunk_count": turn.audio_chunk_count,
                "image_frame_count": len(current_images),
            }
            if self.protocol_version is not None:
                committed_payload["image_sources"] = [
                    (
                        IMAGE_SOURCE_AVATAR_CURRENT
                        if role == IMAGE_ROLE_AVATAR_STATE
                        else role
                    )
                    for role in current_image_roles
                ]
            else:
                committed_payload["image_roles"] = current_image_roles
            await self.send(committed_payload)
            if self.closed:
                turn_outcome = "cancelled"
                await self._cancel_active_turn(send_event=False, expected_turn=turn)
                return
            prepared_current_images, image_preprocess_stats = (
                await self._resolve_prepared_images(turn, current_image_frames)
            )
            logger.info(
                "[SESSION_ACTION_REALTIME] turn.commit input session_id=%s turn_id=%s payload=%s",
                self.session_id,
                turn_id,
                json.dumps(
                    {
                        "text": turn.text,
                        "avatar_state": turn.avatar_state or self.last_avatar_state,
                        "turn_origin": turn.turn_origin,
                        "text_role": turn.text_role,
                        "trigger": turn.trigger,
                        "audio_chunk_count": turn.audio_chunk_count,
                        "image_frame_count": len(current_images),
                        "image_roles": current_image_roles,
                        "audio": _summarize_media(current_audio_list),
                        "images": _summarize_media(current_images),
                        "history_turn_count": len(self.history_turns),
                        "candidate_count": len(self.candidates),
                        "action_catalog_hash": self.action_catalog_hash,
                        "global_action_catalog_hash": self.global_action_catalog_hash,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
            action: dict[str, Any] | None = None
            scores: list[dict[str, Any]] = []
            action_timing = 0.0
            action_context: dict[str, Any] = {}
            action_error: Exception | None = None
            reply_text: str | None = None
            reply_timing: dict[str, Any] | None = None
            selected_category: SessionActionCategory | None = None
            category_decision_received = False
            reply_task: asyncio.Task[tuple[str, dict[str, Any]]] | None = None
            provisional_state: ProvisionalReplyState | None = None
            provisional_discard_task: asyncio.Task[Any] | None = None

            def track_branch(coroutine: Any, *, name: str) -> asyncio.Task[Any]:
                task = asyncio.create_task(coroutine, name=name)
                turn.branch_tasks.add(task)
                task.add_done_callback(turn.branch_tasks.discard)
                return task

            def on_category_selected(
                category: SessionActionCategory | None,
                support_status: str,
            ) -> None:
                nonlocal selected_category, category_decision_received
                nonlocal provisional_discard_task
                selected_category = category
                category_decision_received = True
                emit_structured_log(
                    "action",
                    "category_selected",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                    category_id=(
                        category.category_id
                        if category is not None
                        else UNSUPPORTED_DECISION_ID
                    ),
                    category_label=(
                        category.source_label if category is not None else None
                    ),
                    support_status=support_status,
                )
                if (
                    support_status == "unsupported"
                    and provisional_state is not None
                    and provisional_discard_task is None
                ):
                    provisional_discard_task = track_branch(
                        self._discard_provisional_reply(
                            turn,
                            provisional_state,
                            reason="category_unsupported",
                        ),
                        name=(
                            f"session-provisional-discard-{self.session_id}-"
                            f"{turn.turn_id}"
                        ),
                    )

            provided_reply = turn.reply_provided
            fusion_reply = "text" in self.modalities and "action" in self.modalities
            if fusion_reply:
                provisional_state = await self._create_provisional_reply(
                    turn,
                    source="provided" if provided_reply else "generated",
                )
                if provided_reply:
                    reply_task = track_branch(
                        self._run_provided_reply(
                            turn,
                            turn.text or "",
                            provisional=provisional_state,
                        ),
                        name=(
                            f"session-provisional-provided-reply-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                else:
                    reply_task = track_branch(
                        self._run_generated_reply(
                            turn,
                            current_audio_list,
                            prepared_current_images,
                            current_image_roles,
                            None,
                            provisional=provisional_state,
                        ),
                        name=(
                            f"session-provisional-reply-{self.session_id}-"
                            f"{turn.turn_id}"
                        ),
                    )
                provisional_state.task = reply_task
            elif "text" in self.modalities and provided_reply:
                reply_task = track_branch(
                    self._run_provided_reply(turn, turn.text or ""),
                    name=f"session-provided-reply-{self.session_id}-{turn.turn_id}",
                )
            elif "text" in self.modalities and "action" not in self.modalities:
                reply_task = track_branch(
                    self._run_generated_reply(
                        turn,
                        current_audio_list,
                        prepared_current_images,
                        current_image_roles,
                        None,
                    ),
                    name=f"session-reply-{self.session_id}-{turn.turn_id}",
                )

            action_task: asyncio.Task[Any] | None = None
            if "action" in self.modalities:
                action_started = time.perf_counter()
                action_task = track_branch(
                    self._score_action(
                        current_audio_list,
                        prepared_current_images,
                        current_image_roles,
                        turn.text,
                        turn.avatar_state,
                        turn_origin=turn.turn_origin,
                        text_role=turn.text_role,
                        trigger=turn.trigger,
                        turn_id=turn_id,
                        turn=turn,
                        request_base=turn.request_base,
                        on_category_selected=(
                            on_category_selected if "text" in self.modalities else None
                        ),
                    ),
                    name=f"session-action-{self.session_id}-{turn.turn_id}",
                )
                try:
                    action, scores, action_timing, action_context = await action_task
                except Exception as exc:
                    if not category_decision_received or "text" not in self.modalities:
                        raise
                    action_error = exc
                    fallback = (
                        self._default_fallback_candidate()
                        if self.global_action_catalog is not None
                        else self._no_action_candidate()
                    )
                    action = {
                        "candidate_id": fallback.candidate_id,
                        "action_id": fallback.action_id,
                        "category_id": fallback.category_id,
                        "execution_binding": dict(fallback.execution_binding),
                        "execute": fallback.action_id != "no_action",
                        "support_status": "unknown",
                        "fallback_applied": True,
                    }
                    emit_structured_log(
                        "error",
                        "child_action_failed",
                        level="error",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        logical_request_id=turn.request_base,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        fallback_action_id=fallback.action_id,
                        fallback_category_id=fallback.category_id,
                    )
                action_unsupported = (
                    action is not None and action.get("support_status") == "unsupported"
                )
                if provisional_state is not None:
                    if action_unsupported:
                        unsupported_reason = (
                            "category_unsupported"
                            if action_context.get("category_decision_id")
                            == UNSUPPORTED_DECISION_ID
                            else "child_unsupported"
                        )
                        await self._discard_provisional_reply(
                            turn,
                            provisional_state,
                            reason=unsupported_reason,
                        )
                        if provisional_discard_task is not None:
                            await asyncio.gather(
                                provisional_discard_task,
                                return_exceptions=True,
                            )
                    else:
                        reply_failed = (
                            reply_task is not None
                            and reply_task.done()
                            and not reply_task.cancelled()
                            and reply_task.exception() is not None
                        )
                        if reply_failed:
                            await self._discard_provisional_reply(
                                turn,
                                provisional_state,
                                reason="reply_failed",
                            )
                        else:
                            await self._promote_provisional_reply(
                                turn, provisional_state
                            )
                logger.info(
                    "[SESSION_ACTION_REALTIME] action completed session_id=%s "
                    "turn_id=%s elapsed_ms=%.3f top_action=%s",
                    self.session_id,
                    turn_id,
                    (time.perf_counter() - action_started) * 1000.0,
                    action.get("action_id") if action else None,
                )
                if action is not None:
                    self._ensure_turn_processing(turn)
                    action_ready_payload: dict[str, Any] = {
                        "type": "turn.action.ready",
                        "session_id": self.session_id,
                        "turn_id": turn_id,
                        "action": self._compact_action(action),
                    }
                    if self.global_action_catalog is not None:
                        action_ready_payload.update(
                            {
                                "session_action_catalog_hash": self.action_catalog_hash,
                                "global_action_catalog_hash": self.global_action_catalog_hash,
                            }
                        )
                    await self.send(action_ready_payload)
                    if action_error is None:
                        self._record_action_as_executed(
                            turn=turn,
                            action=action,
                        )

            action_unsupported = (
                action is not None and action.get("support_status") == "unsupported"
            )
            if reply_task is not None and not action_unsupported:
                reply_text, reply_timing = await reply_task
                if provisional_state is not None:
                    reply_timing = self._provisional_reply_timing(provisional_state)
            elif provisional_state is not None:
                reply_timing = self._provisional_reply_timing(provisional_state)

            action_finished = time.perf_counter()
            if self.active_turn is not turn or turn.phase != TURN_PHASE_PROCESSING:
                turn_outcome = "cancelled"
                return
            turn.phase = TURN_PHASE_COMPLETED
            if "action" in self.modalities:
                self._persist_avatar_state(
                    turn.avatar_state,
                    has_avatar_image=(IMAGE_ROLE_AVATAR_STATE in current_image_roles),
                )
            history_reply_text = (
                self.unsupported_action_text if action_unsupported else reply_text
            )
            self._append_reply_history(
                turn,
                current_audio_list,
                current_images,
                current_image_roles,
                history_reply_text,
                model_visible=not action_unsupported,
                history_kind=(
                    "unsupported_action_notice" if action_unsupported else "reply"
                ),
            )
            if action is not None:
                self._append_action_history(
                    current_audio_list,
                    current_images,
                    current_image_roles,
                    turn.text,
                    turn_id=turn_id,
                    turn_origin=turn.turn_origin,
                    text_role=turn.text_role,
                    action=action,
                    # The prerecorded unsupported notice is retained for
                    # diagnostics, but is not a model reply example. Feeding
                    # it back as an ordinary assistant message causes later
                    # supported turns to imitate the notice.
                    reply_text=(None if action_unsupported else history_reply_text),
                )
            self.active_turn = None
            turn_status = "partial" if action_error is not None else "completed"
            turn_outcome = turn_status
            result = {
                "type": "turn.result",
                "session_id": self.session_id,
                "turn_id": turn_id,
                "timing": {
                    "server_turn_ingest_ms": round(ingest_ms, 3),
                    "image_preprocessing": image_preprocess_stats,
                },
            }
            if self.protocol_version is not None or self.modalities != ("action",):
                result["status"] = turn_status
                result[
                    "outputs" if self.protocol_version is not None else "modalities"
                ] = {
                    modality: (
                        "failed"
                        if modality == "action" and action_error is not None
                        else (
                            "suppressed"
                            if modality == "text" and action_unsupported
                            else "completed"
                        )
                    )
                    for modality in self.modalities
                }
            if action_unsupported and "text" in self.modalities:
                result["reply"] = {
                    "source": "client_prerecorded_audio",
                    "reason": "unsupported_action",
                    "recorded_in_history": bool(self.unsupported_action_text),
                }
                result["timing"]["reply"] = reply_timing or {}
            elif reply_text is not None:
                result["reply"] = {
                    "text": reply_text,
                    "source": (
                        reply_timing.get("source", "generated")
                        if reply_timing
                        else "generated"
                    ),
                }
                result["timing"]["reply"] = reply_timing or {}
            if action_error is not None:
                result["errors"] = {
                    "action": {"message": str(action_error)},
                }
            if action is not None:
                result["action_catalog_hash"] = self.action_catalog_hash
                if self.global_action_catalog is not None:
                    result["session_action_catalog_hash"] = self.action_catalog_hash
                    result["global_action_catalog_hash"] = (
                        self.global_action_catalog_hash
                    )
                result["action"] = self._compact_action(action)
                result["timing"].update(
                    {
                        "server_action_compute_ms": action_timing,
                        "action_breakdown": action_context.get(
                            "action_timing_breakdown", {}
                        ),
                    }
                )
            if self.include_scores and action is not None:
                result["scores"] = scores
                result["media_summary"] = {
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                    "user_camera_image_count": current_image_roles.count(
                        IMAGE_ROLE_USER_CAMERA
                    ),
                    "avatar_state_image_count": current_image_roles.count(
                        IMAGE_ROLE_AVATAR_STATE
                    ),
                    "received_image_count": len(turn.images),
                    "scored_image_count": action_context.get(
                        "scored_current_image_count", 0
                    ),
                    "text_present": bool(turn.text),
                    "duplicate_audio_chunks": turn.duplicate_audio_chunks,
                    "duplicate_image_frames": turn.duplicate_image_frames,
                    "action_context": action_context,
                }
            result_finalize_ms = (time.perf_counter() - action_finished) * 1000.0
            total_after_commit_ms = (time.perf_counter() - commit_started) * 1000.0
            result["timing"].update(
                {
                    "server_result_finalize_ms": round(result_finalize_ms, 3),
                    "server_total_after_commit_ms": round(total_after_commit_ms, 3),
                }
            )
            send_started = time.perf_counter()
            await self.send(result)
            emit_structured_log(
                "performance",
                "turn_timing",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                modalities=list(self.modalities),
                turn_origin=turn.turn_origin,
                status=turn_status,
                ingest_ms=round(ingest_ms, 3),
                category_compute_ms=action_context.get("category_compute_ms"),
                child_compute_ms=action_context.get("child_compute_ms"),
                action_support_status=(
                    action.get("support_status") if action is not None else None
                ),
                action_fallback_applied=(
                    action.get("fallback_applied") if action is not None else None
                ),
                category_decision_id=action_context.get("category_decision_id"),
                child_decision_id=action_context.get("child_decision_id"),
                reply_ttft_ms=(reply_timing or {}).get("ttft_ms"),
                reply_total_ms=(reply_timing or {}).get("total_ms"),
                reply_created_after_commit_ms=(reply_timing or {}).get(
                    "created_after_commit_ms"
                ),
                reply_first_delta_after_commit_ms=(reply_timing or {}).get(
                    "first_delta_after_commit_ms"
                ),
                reply_text_done_after_commit_ms=(reply_timing or {}).get(
                    "text_done_after_commit_ms"
                ),
                reply_response_done_after_commit_ms=(reply_timing or {}).get(
                    "response_done_after_commit_ms"
                ),
                reply_stream_duration_ms=(reply_timing or {}).get("stream_duration_ms"),
                reply_delta_count=(reply_timing or {}).get("delta_count"),
                reply_completion_tokens=(reply_timing or {}).get("completion_tokens"),
                reply_provisional=(reply_timing or {}).get("provisional"),
                reply_provisional_status=(reply_timing or {}).get("provisional_status"),
                reply_provisional_done_after_commit_ms=(reply_timing or {}).get(
                    "provisional_done_after_commit_ms"
                ),
                reply_resolution_reason=(reply_timing or {}).get("resolution_reason"),
                reply_discarded_chars=(
                    (reply_timing or {}).get("chars") if action_unsupported else 0
                ),
                total_after_commit_ms=round(total_after_commit_ms, 3),
                logger_health=get_structured_log_writer().health(),
            )
            emit_structured_log(
                "lifecycle",
                "turn_completed",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                status=turn_status,
                modalities=list(self.modalities),
                total_after_commit_ms=round(total_after_commit_ms, 3),
            )
            logger.info(
                "[SESSION_ACTION_REALTIME] turn.result sent session_id=%s "
                "turn_id=%s serialize_and_send_ms=%.3f total_after_commit_ms=%.3f",
                self.session_id,
                turn_id,
                (time.perf_counter() - send_started) * 1000.0,
                (time.perf_counter() - commit_started) * 1000.0,
            )
        except asyncio.CancelledError:
            turn_outcome = "cancelled"
            logger.info(
                "[SESSION_ACTION_REALTIME] turn inference cancelled "
                "session_id=%s turn_id=%s request_id=%s",
                self.session_id,
                turn_id,
                turn.current_request_id,
            )
            raise
        except Exception as exc:
            if turn.phase == TURN_PHASE_CANCELLING:
                turn_outcome = "cancelled"
                logger.warning(
                    "[SESSION_ACTION_REALTIME] cancelled turn cleanup failed "
                    "session_id=%s turn_id=%s",
                    self.session_id,
                    turn_id,
                    exc_info=True,
                )
                return
            turn_outcome = "failed"
            logger.exception(
                "[SESSION_ACTION_REALTIME] turn failed session_id=%s turn_id=%s",
                self.session_id,
                turn_id,
            )
            turn.phase = TURN_PHASE_COMPLETED
            if self.active_turn is turn:
                self.active_turn = None
            message = str(exc)
            if (
                "action" in self.modalities
                and "prefix selected-token logprobs are missing" in message
            ):
                code = "action_score_logprob_unavailable"
                error_type = "action_score_error"
            elif self.modalities == ("action",):
                code = "action_score_failed"
                error_type = "action_score_error"
            else:
                code = "turn_processing_failed"
                error_type = "turn_processing_error"
            emit_structured_log(
                "error",
                "turn_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                error_type=type(exc).__name__,
                error_message=message,
                error_code=code,
            )
            await self.send_error(
                error_type,
                code,
                message,
                session_id=self.session_id,
                turn_id=turn_id,
            )
        finally:
            self._request_turn_resource_sample(
                "turn_after_terminal",
                turn=turn,
                turn_outcome=turn_outcome,
                elapsed_after_commit_ms=round(
                    max(0.0, time.perf_counter() - commit_started) * 1000.0,
                    3,
                ),
            )

    def _request_turn_resource_sample(
        self,
        sample_trigger: str,
        *,
        turn: TurnBuffer,
        **fields: Any,
    ) -> bool:
        requester = self.request_resource_sample
        if requester is None:
            return False
        try:
            return bool(
                requester(
                    sample_trigger,
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    turn_origin=turn.turn_origin,
                    modalities=list(self.modalities),
                    **fields,
                )
            )
        except Exception as exc:
            emit_structured_log(
                "resource",
                "resource_sample_request_failed",
                level="warning",
                sample_trigger=sample_trigger,
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False

    def _build_bounded_action_context(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        *,
        include_history: bool = True,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[str],
        list[Any],
        list[str],
        dict[str, Any],
    ]:
        """Bound action-scoring media without changing normal chat history.

        History messages contain media placeholders, while the media arrays are
        flattened in the same order. This helper walks both together and drops
        whole old turns plus excess media placeholders atomically, so the
        preprocessor never sees a placeholder/media count mismatch.
        """
        if len(images) != len(image_roles):
            raise ValueError("images and image_roles must have the same length")
        if len(self.history_images) != len(self.history_image_roles):
            raise ValueError(
                "history_images and history_image_roles must have the same length"
            )
        # Proactive turns start with an assistant message, so role changes
        # cannot reliably identify turn boundaries.
        selected_turns = (
            self.history_turns[-MAX_ACTION_HISTORY_TURNS:] if include_history else []
        )
        selected_message_ids = {
            id(message) for turn in selected_turns for message in turn.messages
        }
        bounded_history: list[dict[str, Any]] = []
        bounded_history_audios: list[str] = []
        bounded_history_images: list[str] = []
        ignored_history_avatar_image_count = 0
        audio_index = 0
        image_index = 0

        for message in self.history if include_history else ():
            selected = id(message) in selected_message_ids
            content = message.get("content")
            if not selected:
                if isinstance(content, list):
                    audio_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "audio"
                    )
                    image_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "image"
                    )
                continue

            if not isinstance(content, list):
                bounded_history.append(dict(message))
                continue

            bounded_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    bounded_parts.append(part)
                    continue
                part_type = part.get("type")
                if part_type == "audio":
                    if audio_index < len(self.history_audios):
                        media = self.history_audios[audio_index]
                        if len(bounded_history_audios) < MAX_ACTION_HISTORY_AUDIOS:
                            bounded_history_audios.append(media)
                            bounded_parts.append({"type": "audio"})
                    audio_index += 1
                elif part_type == "image":
                    if image_index < len(self.history_images):
                        media = self.history_images[image_index]
                        image_role = self.history_image_roles[image_index]
                        if image_role == IMAGE_ROLE_AVATAR_STATE:
                            # A previous turn's avatar frame may be visually
                            # unrelated to the current rendered pose. Never
                            # expose it as evidence of current avatar state.
                            ignored_history_avatar_image_count += 1
                        elif len(bounded_history_images) < MAX_ACTION_HISTORY_IMAGES:
                            bounded_history_images.append(media)
                            bounded_parts.append(self._image_role_text_part(image_role))
                            bounded_parts.append({"type": "image"})
                    image_index += 1
                else:
                    bounded_parts.append(dict(part))
            if not bounded_parts:
                bounded_parts = [{"type": "text", "text": "（历史多媒体内容已裁剪）"}]
            bounded_history.append({**message, "content": bounded_parts})

        latest_avatar_index = next(
            (
                index
                for index in range(len(image_roles) - 1, -1, -1)
                if image_roles[index] == IMAGE_ROLE_AVATAR_STATE
            ),
            None,
        )
        eligible_indices = [
            index
            for index, role in enumerate(image_roles)
            if role != IMAGE_ROLE_AVATAR_STATE or index == latest_avatar_index
        ]
        selected_indices = eligible_indices[-MAX_ACTION_CURRENT_IMAGES:]
        if (
            latest_avatar_index is not None
            and latest_avatar_index not in selected_indices
        ):
            selected_indices = sorted(
                [
                    latest_avatar_index,
                    *selected_indices[-(MAX_ACTION_CURRENT_IMAGES - 1) :],
                ]
            )
        bounded_images = [images[index] for index in selected_indices]
        bounded_image_roles = [image_roles[index] for index in selected_indices]
        bounded_audios = list(audios)
        truncated = (
            len(self.history_turns) > len(selected_turns)
            or len(self.history_audios) != len(bounded_history_audios)
            or len(self.history_images) != len(bounded_history_images)
            or len(images) != len(bounded_images)
        )
        context_summary = {
            "history_turn_count": len(selected_turns),
            "history_audio_count": len(bounded_history_audios),
            "history_image_count": len(bounded_history_images),
            "ignored_history_avatar_image_count": (ignored_history_avatar_image_count),
            "received_current_image_count": len(images),
            "scored_current_image_count": len(bounded_images),
            "truncated": truncated,
        }
        return (
            bounded_history,
            bounded_history_audios,
            bounded_history_images,
            bounded_images,
            bounded_image_roles,
            context_summary,
        )

    def _prompt(self, *, zh: str, en: str) -> str:
        return localized_prompt(self.language, zh=zh, en=en)

    def _model_action_history_record(
        self,
        *,
        candidate_id: str,
        action_id: str,
        category_id: str | None,
        source_label: str,
        short_definition: str,
        execute: bool,
        record_kind: Literal[
            "history",
            "current_physical",
            "last_user",
            "current_physical_and_last_user",
        ] = "history",
    ) -> str:
        execution_result = self._prompt(
            zh="已按执行处理" if execute else "未执行新动作（保持当前姿态）",
            en=(
                "treated as executed"
                if execute
                else "no new action executed (current pose retained)"
            ),
        )
        zh_labels = {
            "history": "历史动作记录",
            "current_physical": "当前实际动作状态",
            "last_user": "最近一次用户触发动作",
            "current_physical_and_last_user": "当前实际动作状态；同时是最近一次用户触发动作",
        }
        en_labels = {
            "history": "Historical action record",
            "current_physical": "Current physical action state",
            "last_user": "Most recent user-triggered action",
            "current_physical_and_last_user": (
                "Current physical action state; also the most recent user-triggered action"
            ),
        }
        zh_category = f"category_id={category_id}｜" if category_id else ""
        en_category = f"category_id={category_id} | " if category_id else ""
        return self._prompt(
            zh=(
                f"[{zh_labels[record_kind]}] "
                f"处理结果={execution_result}｜{zh_category}candidate_id={candidate_id}｜"
                f"action_id={action_id}｜动作={source_label}｜"
                f"说明={short_definition}。"
            ),
            en=(
                f"[{en_labels[record_kind]}] "
                f"result={execution_result} | {en_category}candidate_id={candidate_id} | "
                f"action_id={action_id} | action={source_label} | "
                f"description={short_definition}."
            ),
        )

    def _build_compact_action_history(self) -> list[dict[str, Any]]:
        """Return the small cross-turn context needed by action scoring.

        Category and Child deliberately share this exact history so the
        same-turn prepared-media cache remains valid.  Reply generation keeps
        its independent bounded conversation history.
        """

        latest_reply: str | None = None
        for history_turn in reversed(self.reply_history_turns):
            if not history_turn.model_visible:
                continue
            for message in reversed(history_turn.messages):
                content = message.get("content")
                if (
                    message.get("role") == "assistant"
                    and isinstance(content, str)
                    and content.strip()
                ):
                    latest_reply = content.strip()
                    break
            if latest_reply is not None:
                break

        parts: list[str] = []
        if latest_reply is not None:
            parts.append(
                self._prompt(
                    zh=f"[数字人最近一次回复] {latest_reply}",
                    en=f"[Digital character's most recent reply] {latest_reply}",
                )
            )

        physical_record = self.last_executed_action
        user_record = self.last_user_executed_action
        if physical_record is not None:
            record_kind: Literal[
                "current_physical", "current_physical_and_last_user"
            ] = (
                "current_physical_and_last_user"
                if user_record is not None
                and user_record.turn_id == physical_record.turn_id
                else "current_physical"
            )
            parts.append(
                self._model_action_history_record(
                    candidate_id=physical_record.candidate_id,
                    action_id=physical_record.action_id,
                    category_id=physical_record.category_id,
                    source_label=physical_record.source_label,
                    short_definition=physical_record.short_definition,
                    execute=physical_record.execute,
                    record_kind=record_kind,
                )
            )
        if user_record is not None and (
            physical_record is None or user_record.turn_id != physical_record.turn_id
        ):
            parts.append(
                self._model_action_history_record(
                    candidate_id=user_record.candidate_id,
                    action_id=user_record.action_id,
                    category_id=user_record.category_id,
                    source_label=user_record.source_label,
                    short_definition=user_record.short_definition,
                    execute=user_record.execute,
                    record_kind="last_user",
                )
            )

        if not parts:
            return []
        return [{"role": "assistant", "content": "\n".join(parts)}]

    def _image_role_label(self, image_role: str) -> str:
        if image_role == IMAGE_ROLE_USER_CAMERA:
            return self._prompt(
                zh="用户摄像头画面（用于观察用户及其环境）",
                en="User camera view (for observing the user and their environment)",
            )
        return self._prompt(
            zh="数字人当前状态画面（用于观察数字人自身当前可视姿态和行为）",
            en=(
                "Current digital character state view (for observing the "
                "character's visible pose and behavior)"
            ),
        )

    def _image_role_text_part(self, image_role: str) -> dict[str, str]:
        return {
            "type": "text",
            "text": self._prompt(
                zh=f"[图片用途] {self._image_role_label(image_role)}：",
                en=f"[Image purpose] {self._image_role_label(image_role)}:",
            ),
        }

    @staticmethod
    def _has_structured_avatar_state(avatar_state: dict[str, Any]) -> bool:
        """Return whether reusable structured avatar state is actually present."""
        for key, value in avatar_state.items():
            if key in {"current_action_id", "state_description"} or value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, dict, set)) and not value:
                continue
            return True
        return False

    @classmethod
    def _avatar_state_source(
        cls,
        avatar_state: dict[str, Any],
        image_roles: list[str],
    ) -> Literal["image", "structured", "unknown"]:
        if IMAGE_ROLE_AVATAR_STATE in image_roles:
            return "image"
        if cls._has_structured_avatar_state(avatar_state):
            return "structured"
        return "unknown"

    def _build_avatar_state_instruction(
        self,
        source: Literal["image", "structured", "unknown"],
    ) -> str:
        """Build stage-neutral state guidance outside the static KV prefix."""
        if source == "image":
            return self._prompt(
                zh=(
                    "判断候选动作可执行性时，以本轮最新数字人照片中的当前可视姿态和行为"
                    "为准。\n"
                ),
                en=(
                    "When deciding whether a candidate action is executable, use the "
                    "visible pose and behavior in the latest current digital character "
                    "image from this interaction.\n"
                ),
            )
        if source == "structured":
            return self._prompt(
                zh=(
                    "本轮未提供数字人状态照片；判断候选动作可执行性时，以结构化 "
                    "数字人当前状态信息为准；不得从历史图片或用户摄像头画面"
                    "推断数字人状态。\n"
                ),
                en=(
                    "No current digital character state image is provided in this "
                    "interaction. Use the structured current character state when deciding "
                    "whether a candidate is executable. Do not infer the character state "
                    "from historical images or the user camera view.\n"
                ),
            )
        return self._prompt(
            zh=(
                "本轮没有可用的数字人当前状态信息；不得从历史图片或用户摄像头画面"
                "推断数字人状态。不依赖特定起始姿态的候选，不应仅因状态未知而被排除。\n"
            ),
            en=(
                "No current digital character state information is available in this "
                "interaction. Do not infer it from historical images or the user camera "
                "view. Do not exclude candidates that do not require a specific starting "
                "pose solely because the state is unknown.\n"
            ),
        )

    def _effective_avatar_state(
        self,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: str,
        has_avatar_image: bool,
    ) -> dict[str, Any]:
        """Build model-visible state without mixing visual and turn-local fields."""
        explicit = dict(avatar_state or {})
        effective = {} if has_avatar_image else dict(self.last_avatar_state)
        for key, value in explicit.items():
            if key == "state_description":
                if turn_origin == TURN_ORIGIN_PROACTIVE:
                    effective[key] = value
            elif key == "current_action_id" or not has_avatar_image:
                effective[key] = value
        return effective

    def _persist_avatar_state(
        self,
        avatar_state: dict[str, Any] | None,
        *,
        has_avatar_image: bool,
    ) -> None:
        """Persist only reusable structured visual state across turns."""
        if has_avatar_image:
            self.last_avatar_state = {}
        elif avatar_state is not None:
            self.last_avatar_state = {
                key: value
                for key, value in avatar_state.items()
                if key not in {"current_action_id", "state_description"}
            }

    @staticmethod
    def _with_current_proactive_text(
        history: list[dict[str, Any]], text: str | None, turn_origin: str
    ) -> list[dict[str, Any]]:
        if (
            turn_origin != TURN_ORIGIN_PROACTIVE
            or not isinstance(text, str)
            or not text.strip()
        ):
            return history
        return [*history, {"role": "assistant", "content": text.strip()}]

    def _build_turn_action_instruction(
        self,
        text: str | None,
        *,
        turn_origin: str,
        trigger: str | None,
        has_audio: bool = False,
        image_roles: list[str] | None = None,
        has_current_action_id: bool = False,
        has_state_description: bool = False,
        avatar_state_source: Literal["image", "structured", "unknown"] | None = None,
    ) -> str:
        if self.language == "en":
            return self._build_turn_action_instruction_en(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                has_audio=has_audio,
                image_roles=image_roles,
                has_current_action_id=has_current_action_id,
                has_state_description=has_state_description,
                avatar_state_source=avatar_state_source,
            )
        resolved_image_roles = image_roles or []
        if avatar_state_source is None:
            avatar_state_source = (
                "image"
                if IMAGE_ROLE_AVATAR_STATE in resolved_image_roles
                else "unknown"
            )
        state_instruction = self._build_avatar_state_instruction(avatar_state_source)
        scene_constraint = (
            "候选必须满足本轮主动场景约束中给出的目标、指引、要求和禁止项。"
            if has_state_description
            else ""
        )
        transition_constraint = (
            "候选必须与本轮提供的“当前实际动作 ID”所表示的动作自然衔接，"
            "并避免无意义重复。"
            if has_current_action_id
            else "结合历史动作判断衔接关系，并避免无意义重复。"
        )
        if turn_origin == TURN_ORIGIN_PROACTIVE:
            trigger_text = f"主动触发原因：{trigger}。\n" if trigger is not None else ""
            if not isinstance(text, str) or not text.strip():
                return (
                    state_instruction
                    + "本轮由数字人主动发起，未提供数字人本轮将要说出的文本；"
                    "不要把历史中的数字人回复当成本轮将要说出的文本。\n"
                    + trigger_text
                    + "根据主动触发原因、本次主动场景说明（如有）、当前媒体和历史动作"
                    "选择自然衔接的动作。"
                    + scene_constraint
                    + transition_constraint
                    + "选择与表达目标和状态约束最匹配的候选项；不要生成回复。"
                )
            return (
                state_instruction
                + "本轮由数字人主动发起，且已提供数字人本轮将要说出的文本。"
                "该文本位于紧邻本指令之前、本轮新增的数字人消息中，"
                "不是用户输入或用户动作请求。\n"
                + trigger_text
                + "数字人本轮将要说出的文本，其语义、语气和表达目标是本轮核心约束。"
                + "候选动作必须与该文本的语义、语气和表达目标直接相关。"
                + scene_constraint
                + transition_constraint
                + "选择与表达目标和状态约束最匹配的候选项；不要生成回复。"
            )
        modalities: list[str] = []
        if isinstance(text, str) and text.strip():
            modalities.append("用户文本")
        if has_audio:
            modalities.append("用户音频")
        if image_roles:
            modalities.append("当前图片")
        input_summary = (
            "、".join(modalities) if modalities else "未提供文本、音频或图片"
        )
        text_prefix = (
            f"当前用户文本：{text.strip()}\n"
            if isinstance(text, str) and text.strip()
            else ""
        )
        return (
            state_instruction
            + f"本轮由用户输入触发；有效输入：{input_summary}。\n"
            + text_prefix
            + "根据用户的语言、语音语义或可观察行为，选择与输入和状态约束最匹配的候选项。"
        )

    def _build_turn_action_instruction_en(
        self,
        text: str | None,
        *,
        turn_origin: str,
        trigger: str | None,
        has_audio: bool,
        image_roles: list[str] | None,
        has_current_action_id: bool,
        has_state_description: bool,
        avatar_state_source: Literal["image", "structured", "unknown"] | None,
    ) -> str:
        resolved_image_roles = image_roles or []
        if avatar_state_source is None:
            avatar_state_source = (
                "image"
                if IMAGE_ROLE_AVATAR_STATE in resolved_image_roles
                else "unknown"
            )
        state_instruction = self._build_avatar_state_instruction(avatar_state_source)
        scene_constraint = (
            "The candidate must satisfy every goal, instruction, requirement, and "
            "prohibition in the proactive-scene constraints for this interaction."
            if has_state_description
            else ""
        )
        transition_constraint = (
            "The candidate must transition naturally from the action represented by "
            "the 'current physical action ID' supplied in this interaction and "
            "must avoid meaningless repetition."
            if has_current_action_id
            else "Use the action history to judge a natural transition and avoid "
            "meaningless repetition."
        )
        if turn_origin == TURN_ORIGIN_PROACTIVE:
            trigger_text = (
                f"Proactive trigger reason: {trigger}.\n" if trigger is not None else ""
            )
            if not isinstance(text, str) or not text.strip():
                return (
                    state_instruction
                    + "This interaction is initiated by the digital character, and no "
                    "text for the character to say in this interaction is supplied. Do "
                    "not treat a historical character reply as the text for this "
                    "interaction.\n"
                    + trigger_text
                    + "Select an action that transitions naturally based on the proactive "
                    "trigger, proactive-scene description if supplied, current media, and "
                    "action history."
                    + scene_constraint
                    + transition_constraint
                    + "Select the candidate that best matches the expression goal and state "
                    "constraints. Do not generate a reply."
                )
            return (
                state_instruction
                + "This interaction is initiated by the digital character, and the text "
                "that the character will say is supplied. It is the newly added character "
                "message immediately before this instruction, not user input or a user "
                "action request.\n"
                + trigger_text
                + "The semantics, tone, and expression goal of that text are the primary "
                "constraints. The candidate action must be directly relevant to them."
                + scene_constraint
                + transition_constraint
                + "Select the candidate that best matches the expression goal and state "
                "constraints. Do not generate a reply."
            )
        modalities: list[str] = []
        if isinstance(text, str) and text.strip():
            modalities.append("user text")
        if has_audio:
            modalities.append("user audio")
        if resolved_image_roles:
            modalities.append("current images")
        input_summary = (
            ", ".join(modalities) if modalities else "no text, audio, or image"
        )
        text_prefix = (
            f"Current user text: {text.strip()}\n"
            if isinstance(text, str) and text.strip()
            else ""
        )
        return (
            state_instruction
            + f"This interaction is triggered by user input. Available input: {input_summary}.\n"
            + text_prefix
            + "Use the user's language, speech semantics, or observable behavior to "
            "select the candidate that best matches the input and state constraints."
        )

    def _state_description_priority_instruction(
        self,
        stage: Literal["category", "child", "single"],
        *,
        enabled: bool,
    ) -> str:
        """Keep turn-local proactive guidance above later dynamic constraints."""
        if not enabled:
            return ""
        if stage == "category":
            return self._prompt(
                zh=(
                    "[本轮主动场景约束优先级]\n"
                    "本轮已提供“本轮主动场景约束”，它是本轮动作类别选择的最高优先级依据，"
                    "高于数字人人设与动作偏好、主动触发原因、将要说出的文本、历史动作、"
                    "默认动作类别和其他通用选择规则。必须先满足其中明确的动作目标、要求和"
                    "禁止项；明确禁止的动作语义不得选择。默认动作类别仅在不与该约束冲突，"
                    "且该约束没有给出更具体动作目标时使用。若该约束给出了明确动作目标，"
                    "应选择能够完成该目标的类别，不得仅因其他类别是默认动作类别而改选它。\n"
                ),
                en=(
                    "[Priority of proactive-scene constraints for this interaction]\n"
                    "The proactive-scene constraints supplied for this interaction are "
                    "the highest-priority basis for category selection. They override the "
                    "character persona and action preferences, proactive trigger, text the "
                    "character will say, action history, default action categories, and other "
                    "general selection rules. First satisfy every explicit action goal, "
                    "requirement, and prohibition; do not select prohibited action semantics. "
                    "Use a default action category only when it does not conflict with these "
                    "constraints and no more specific action goal is given. When an explicit "
                    "action goal is given, select a category that can accomplish it rather "
                    "than preferring another category merely because it is a default.\n"
                ),
            )
        if stage == "child":
            return self._prompt(
                zh=(
                    "[本轮主动场景约束优先级]\n"
                    "本轮已提供“本轮主动场景约束”，它是本轮具体动作选择的最高优先级依据，"
                    "高于数字人人设与动作偏好、主动触发原因、将要说出的文本、历史动作、"
                    "默认动作规则和其他通用选择规则。必须先满足其中明确的动作目标、要求和"
                    "禁止项；任何违反明确禁止项的 candidate_id 都不得选择，不能为了选择真实"
                    "动作、避免重复或使用默认动作而忽略该约束。\n"
                ),
                en=(
                    "[Priority of proactive-scene constraints for this interaction]\n"
                    "The proactive-scene constraints supplied for this interaction are "
                    "the highest-priority basis for concrete-action selection. They override "
                    "the character persona and action preferences, proactive trigger, text "
                    "the character will say, action history, default-action rules, and other "
                    "general selection rules. First satisfy every explicit action goal, "
                    "requirement, and prohibition. Never select a candidate_id that violates "
                    "an explicit prohibition merely to select a real action, avoid repetition, "
                    "or use a default action.\n"
                ),
            )
        return self._prompt(
            zh=(
                "[本轮主动场景约束优先级]\n"
                "本轮已提供“本轮主动场景约束”，它是本轮动作选择的最高优先级依据。"
                "必须先满足其中明确的动作目标、要求和禁止项；任何违反明确禁止项的动作"
                "都不得选择，会话级偏好、历史动作和通用规则不能覆盖该约束。\n"
            ),
            en=(
                "[Priority of proactive-scene constraints for this interaction]\n"
                "The proactive-scene constraints supplied for this interaction are the "
                "highest-priority basis for action selection. First satisfy every explicit "
                "action goal, requirement, and prohibition. Never select an action that "
                "violates an explicit prohibition; conversation-level preferences, action "
                "history, and general rules cannot override these constraints.\n"
            ),
        )

    @staticmethod
    def _explicit_prohibition_text(state_description: Any) -> str:
        """Return only clauses that explicitly prohibit an action semantic.

        ``state_description`` remains free-form client text.  We therefore do
        not attempt to turn every preference into a hard rule.  Only text
        following an explicit negative marker is eligible for deterministic
        candidate filtering; the complete description is still sent to the
        model for the broader semantic decision.
        """
        if not isinstance(state_description, str) or not state_description.strip():
            return ""
        markers = (
            "严禁",
            "禁止",
            "不得",
            "不要",
            "不能",
            "避免",
            "must not",
            "do not",
            "never",
            "avoid",
            "prohibit",
        )
        prohibited_parts: list[str] = []
        for sentence in re.split(r"[。.!?\n]+", state_description):
            lowered = sentence.lower()
            marker_positions = [
                lowered.find(marker) for marker in markers if lowered.find(marker) >= 0
            ]
            if marker_positions:
                prohibited_parts.append(lowered[min(marker_positions) :])
        return "\n".join(prohibited_parts)

    @staticmethod
    def _action_semantic_terms(*values: str) -> set[str]:
        stop_terms = {
            "动作",
            "姿态",
            "身体",
            "状态",
            "场景",
            "交互",
            "用户",
            "要求",
            "说明",
            "表达",
            "当前",
            "自然",
            "action",
            "motion",
            "pose",
            "body",
            "state",
            "current",
            "natural",
        }
        terms: set[str] = set()
        for value in values:
            for term in re.split(r"[\s,，、;；/|:：()（）]+", value.lower()):
                normalized = term.strip("-_.。.!?")
                if len(normalized) >= 2 and normalized not in stop_terms:
                    terms.add(normalized)
                # Chinese catalog labels commonly combine the requested verb
                # with execution detail (for example, “自然呼吸起伏”).  Keep
                # short CJK semantic fragments so an explicit prohibition of
                # “呼吸” or “喝水” can remove the concrete candidate as well.
                if re.search(r"[\u4e00-\u9fff]", normalized):
                    for width in range(2, min(4, len(normalized)) + 1):
                        for start in range(0, len(normalized) - width + 1):
                            fragment = normalized[start : start + width]
                            if fragment not in stop_terms:
                                terms.add(fragment)
        return terms

    @classmethod
    def _semantic_is_explicitly_prohibited(
        cls,
        prohibition_text: str,
        *semantic_values: str,
    ) -> bool:
        if not prohibition_text:
            return False
        compact_prohibition = re.sub(r"\s+", "", prohibition_text.lower())
        return any(
            re.sub(r"\s+", "", term) in compact_prohibition
            for term in cls._action_semantic_terms(*semantic_values)
        )

    def _state_description_excluded_category_ids(
        self,
        state_description: Any,
    ) -> tuple[str, ...]:
        prohibition_text = self._explicit_prohibition_text(state_description)
        return tuple(
            category.category_id
            for category in self.categories
            if self._semantic_is_explicitly_prohibited(
                prohibition_text,
                category.source_label,
                category.short_definition,
            )
        )

    def _state_description_excluded_candidate_ids(
        self,
        state_description: Any,
        candidates: list[SessionActionCandidate],
    ) -> tuple[str, ...]:
        prohibition_text = self._explicit_prohibition_text(state_description)
        return tuple(
            candidate.candidate_id
            for candidate in candidates
            if self._semantic_is_explicitly_prohibited(
                prohibition_text,
                candidate.source_label,
                candidate.short_definition,
            )
        )

    def _state_description_exclusion_instruction(
        self,
        *,
        category_ids: tuple[str, ...] = (),
        candidate_ids: tuple[str, ...] = (),
    ) -> str:
        if not category_ids and not candidate_ids:
            return ""
        separator = self._prompt(zh="、", en=", ")
        if category_ids:
            excluded = separator.join(category_ids)
            return self._prompt(
                zh=(
                    "[本轮明确禁止项的确定性过滤]\n"
                    f"以下 category_id 与本轮主动场景约束的明确禁止项直接冲突，"
                    f"已从本轮可选集合移除：{excluded}。不得输出这些 ID。\n"
                ),
                en=(
                    "[Deterministic filtering for explicit prohibitions]\n"
                    "The following category_id values directly conflict with an "
                    f"explicit prohibition and have been removed: {excluded}. Do not "
                    "output these IDs.\n"
                ),
            )
        excluded = separator.join(candidate_ids)
        return self._prompt(
            zh=(
                "[本轮明确禁止项的确定性过滤]\n"
                f"以下 candidate_id 与本轮主动场景约束的明确禁止项直接冲突，"
                f"已从本轮可选集合移除：{excluded}。不得输出这些 ID。\n"
            ),
            en=(
                "[Deterministic filtering for explicit prohibitions]\n"
                "The following candidate_id values directly conflict with an explicit "
                f"prohibition and have been removed: {excluded}. Do not output these "
                "IDs.\n"
            ),
        )

    def _build_session_action_profile_instruction(
        self,
        stage: Literal["category", "child", "single"],
    ) -> str:
        profile = self.action_profile
        if profile is None:
            return ""
        if self.language == "en":
            persona_labels = {
                "gender_expression": "gender expression",
                "visual_style": "visual style",
                "role": "occupation or role",
                "personality": "personality",
            }
            lines = [
                "[Digital character persona and action preferences for this conversation]"
            ]
            if profile.persona:
                lines.append(
                    "Digital character persona: "
                    + "; ".join(
                        f"{persona_labels[field_name]}={field_value}"
                        for field_name, field_value in profile.persona
                    )
                )
            if stage == "category":
                if profile.category_preferences:
                    lines.append(
                        "Category preferences (primary constraints for category "
                        "selection): " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append(
                        "Action preferences (feasibility constraints for category "
                        "selection): exclude categories that cannot satisfy these "
                        "preferences at all, but do not select a concrete action in "
                        "this stage; " + profile.action_preferences
                    )
            elif stage == "child":
                if profile.category_preferences:
                    lines.append(
                        "Category preferences (background constraints for concrete "
                        "action selection): do not rewrite or expand the selected "
                        "category; " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append(
                        "Action preferences (primary constraints for concrete action "
                        "selection): " + profile.action_preferences
                    )
            else:
                if profile.category_preferences:
                    lines.append(
                        "Category preferences: " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append("Action preferences: " + profile.action_preferences)
            lines.append(
                "The content above constrains action selection only for this "
                "conversation. It must not expand or rewrite the allowed category or "
                "action set. If it conflicts with proactive-scene constraints for this "
                "interaction, the proactive-scene constraints take precedence."
            )
            return "\n".join(lines) + "\n"
        persona_labels = {
            "gender_expression": "性别表达",
            "visual_style": "画风",
            "role": "职业或角色定位",
            "personality": "性格基调",
        }
        lines = ["[本次会话数字人人设与动作偏好]"]
        if profile.persona:
            lines.append(
                "数字人人设："
                + "；".join(
                    f"{persona_labels[field_name]}={field_value}"
                    for field_name, field_value in profile.persona
                )
            )
        if stage == "category":
            if profile.category_preferences:
                lines.append(
                    "类别偏好（动作类别选择的主要约束）："
                    + profile.category_preferences
                )
            if profile.action_preferences:
                lines.append(
                    "动作偏好（动作类别选择的可行性约束）："
                    "排除整体上无法满足该偏好的类别，但不要在本阶段选择具体动作；"
                    + profile.action_preferences
                )
        elif stage == "child":
            if profile.category_preferences:
                lines.append(
                    "类别偏好（具体动作选择的背景约束）：已选类别不得被改写或扩展；"
                    + profile.category_preferences
                )
            if profile.action_preferences:
                lines.append(
                    "动作偏好（具体动作选择的主要约束）：" + profile.action_preferences
                )
        else:
            if profile.category_preferences:
                lines.append("类目偏好：" + profile.category_preferences)
            if profile.action_preferences:
                lines.append("动作偏好：" + profile.action_preferences)
        lines.append(
            "以上内容仅约束本次会话的动作选择，不得扩展或改写允许选择的类别和动作范围；"
            "若与本轮主动场景约束冲突，以本轮主动场景约束为准。"
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _ensure_turn_processing(turn: TurnBuffer) -> None:
        if turn.phase != TURN_PHASE_PROCESSING:
            raise asyncio.CancelledError

    @staticmethod
    def _after_commit_ms(
        turn: TurnBuffer, *, observed_at: float | None = None
    ) -> float | None:
        if turn.commit_started_at is None:
            return None
        observed_at = observed_at if observed_at is not None else time.perf_counter()
        return round(max(0.0, observed_at - turn.commit_started_at) * 1000.0, 3)

    def _build_reply_request(
        self,
        turn: TurnBuffer,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        category: SessionActionCategory | None,
        *,
        support_status: str = "supported",
    ) -> tuple[GenerateRequest, list[str]]:
        reply_images, reply_image_roles = self._select_reply_user_camera_images(
            images, image_roles
        )
        messages: list[Message] = []
        if self.instructions.strip():
            messages.append(Message(role="system", content=self.instructions.strip()))
        history_audios: list[str] = []
        history_images: list[str] = []
        visible_history_turns = [
            history_turn
            for history_turn in self.reply_history_turns
            if history_turn.model_visible
        ][-MAX_ACTION_HISTORY_TURNS:]
        for history_turn in visible_history_turns:
            if len(history_turn.images) != len(history_turn.image_roles):
                raise ValueError(
                    "reply history images and image_roles must have the same length"
                )
            messages.extend(
                Message(role=item["role"], content=item["content"])
                for item in history_turn.messages
            )
            history_audios.extend(history_turn.audios)
            history_images.extend(history_turn.images)

        parts: list[dict[str, Any]] = []
        if reply_image_roles:
            # Put visual evidence before the user's speech/text so the actual
            # request remains closest to the assistant generation. Only one
            # latest camera frame is forwarded, but keep this grouped in case
            # that policy changes later.
            parts.append(self._reply_user_camera_context_part())
            parts.extend({"type": "image"} for _ in reply_image_roles)
        parts.extend({"type": "audio"} for _ in audios)
        if turn.turn_origin == TURN_ORIGIN_USER:
            if isinstance(turn.text, str) and turn.text.strip():
                parts.append({"type": "text", "text": turn.text.strip()})
        if isinstance(turn.reply_context, str) and turn.reply_context.strip():
            parts.append({"type": "text", "text": turn.reply_context.strip()})
        if not reply_image_roles:
            parts.append(
                {
                    "type": "text",
                    "text": self._prompt(
                        zh=(
                            "本轮未提供用户摄像头画面，不能声称看见用户或"
                            "根据用户外观作出判断。"
                        ),
                        en=(
                            "No user-camera image is provided in this interaction. "
                            "Do not claim to see the user or make judgments based on "
                            "the user's appearance."
                        ),
                    ),
                }
            )
        if parts:
            messages.append(Message(role="user", content=parts))
        request = GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=SamplingParams(
                temperature=DEFAULT_REPLY_TEMPERATURE,
                top_p=1.0,
                max_new_tokens=DEFAULT_REPLY_MAX_NEW_TOKENS,
            ),
            stream=True,
            output_modalities=["text"],
            metadata={
                "audios": [*history_audios, *audios],
                "images": [*history_images, *reply_images],
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "logical_request_id": turn.request_base,
                "task": "session_reply",
            },
        )
        return request, reply_image_roles

    @staticmethod
    def _select_reply_user_camera_images(
        images: list[Any], image_roles: list[str]
    ) -> tuple[list[Any], list[str]]:
        if len(images) != len(image_roles):
            raise ValueError("reply images and image_roles must have the same length")
        selected = [
            (image, role)
            for image, role in zip(images, image_roles, strict=True)
            if role == IMAGE_ROLE_USER_CAMERA
        ]
        # Camera frames describe transient current state. Multiple samples
        # from one turn are usually near-duplicates and can overpower the
        # spoken request, so replies use only the latest frame.
        selected = selected[-1:]
        return (
            [image for image, _ in selected],
            [role for _, role in selected],
        )

    def _reply_user_camera_context_part(self) -> dict[str, str]:
        return {
            "type": "text",
            "text": self._prompt(
                zh=(
                    "[用户摄像头画面，仅作为回答当前问题时的视觉依据；"
                    "不要主动描述正在观看用户]"
                ),
                en=(
                    "[User camera view: use only as visual evidence for the "
                    "current question; do not proactively describe watching "
                    "the user]"
                ),
            ),
        }

    async def _create_provisional_reply(
        self,
        turn: TurnBuffer,
        *,
        source: Literal["generated", "provided"],
    ) -> ProvisionalReplyState:
        state = ProvisionalReplyState(
            response_id=f"resp-{uuid.uuid4().hex}",
            source=source,
            started_at=time.perf_counter(),
            created_after_commit_ms=None,
        )
        turn.provisional_reply = state
        await self.send(
            {
                "type": "response.provisional.created",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "provisional_id": state.response_id,
                "response": {
                    "id": state.response_id,
                    "status": "in_progress",
                    "source": source,
                    "provisional": True,
                },
            }
        )
        state.created_after_commit_ms = self._after_commit_ms(turn)
        emit_structured_log(
            "reply",
            "provisional_reply_created",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            response_id=state.response_id,
            reply_source=source,
            created_after_commit_ms=state.created_after_commit_ms,
        )
        return state

    async def _send_provisional_reply_delta(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        delta: str,
    ) -> None:
        if not delta:
            return
        async with state.lock:
            if state.status == "discarded":
                return
            is_first_delta = state.first_token_ms is None
            if is_first_delta:
                state.first_token_ms = (time.perf_counter() - state.started_at) * 1000.0
            state.text_parts.append(delta)
            state.delta_count += 1
            event_type = (
                "response.provisional.text.delta"
                if state.status == "pending"
                else "response.text.delta"
            )
            payload: dict[str, Any] = {
                "type": event_type,
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response_id": state.response_id,
                "provisional_id": state.response_id,
                "seq": state.delta_count,
                "delta": delta,
            }
            await self.send(payload)
            if is_first_delta:
                state.first_delta_after_commit_ms = self._after_commit_ms(turn)
                emit_structured_log(
                    "reply",
                    "provisional_reply_first_token",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                    response_id=state.response_id,
                    ttft_ms=round(state.first_token_ms, 3),
                    first_delta_after_commit_ms=(state.first_delta_after_commit_ms),
                )

    async def _finish_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        finish_reason: str,
        usage: dict[str, Any] | None,
    ) -> dict[str, float | None]:
        async with state.lock:
            state.completed = True
            state.finish_reason = finish_reason
            state.usage = usage
            text = "".join(state.text_parts)
            if state.status == "discarded":
                return {
                    "text_done_after_commit_ms": None,
                    "response_done_after_commit_ms": None,
                }
            if state.status == "pending":
                await self.send(
                    {
                        "type": "response.provisional.text.done",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "response_id": state.response_id,
                        "provisional_id": state.response_id,
                        "text": text,
                    }
                )
                state.provisional_done_after_commit_ms = self._after_commit_ms(turn)
                return {
                    "text_done_after_commit_ms": (
                        state.provisional_done_after_commit_ms
                    ),
                    "response_done_after_commit_ms": None,
                }
            done_timing = await self._send_reply_done(
                turn,
                response_id=state.response_id,
                text=text,
                source=state.source,
                finish_reason=finish_reason,
                usage=usage,
                provisional_id=state.response_id,
            )
            state.official_done = True
            state.official_text_done_after_commit_ms = done_timing[
                "text_done_after_commit_ms"
            ]
            state.official_response_done_after_commit_ms = done_timing[
                "response_done_after_commit_ms"
            ]
            return done_timing

    def _provisional_reply_timing(
        self,
        state: ProvisionalReplyState,
        *,
        total_ms: float | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = (
            total_ms
            if total_ms is not None
            else (time.perf_counter() - state.started_at) * 1000.0
        )
        text_done_after_commit_ms = (
            state.official_text_done_after_commit_ms
            if state.status == "promoted"
            else state.provisional_done_after_commit_ms
        )
        stream_duration_ms = (
            round(
                text_done_after_commit_ms - state.first_delta_after_commit_ms,
                3,
            )
            if text_done_after_commit_ms is not None
            and state.first_delta_after_commit_ms is not None
            else None
        )
        return {
            "source": state.source,
            "ttft_ms": round(state.first_token_ms or elapsed_ms, 3),
            "total_ms": round(elapsed_ms, 3),
            "chars": len("".join(state.text_parts)),
            "created_after_commit_ms": state.created_after_commit_ms,
            "first_delta_after_commit_ms": state.first_delta_after_commit_ms,
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": (
                state.official_response_done_after_commit_ms
            ),
            "stream_duration_ms": stream_duration_ms,
            "delta_count": state.delta_count,
            "completion_tokens": (
                state.usage.get("completion_tokens")
                if state.usage is not None
                else None
            ),
            "provisional": True,
            "provisional_status": state.status,
            "provisional_done_after_commit_ms": (
                state.provisional_done_after_commit_ms
            ),
            "resolution_reason": state.resolution_reason,
        }

    async def _promote_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
    ) -> None:
        async with state.lock:
            if state.status != "pending":
                return
            state.status = "promoted"
            state.resolution_reason = "action_supported"
            buffered_text = "".join(state.text_parts)
            await self.send(
                {
                    "type": "response.provisional.resolved",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "provisional_id": state.response_id,
                    "status": "promoted",
                    "reason": state.resolution_reason,
                    "promoted_prefix_chars": len(buffered_text),
                    "promoted_prefix_delta_count": state.delta_count,
                }
            )
            await self.send(
                {
                    "type": "response.created",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "response": {
                        "id": state.response_id,
                        "status": "in_progress",
                        "source": state.source,
                        "provisional_id": state.response_id,
                    },
                }
            )
            state.official_created = True
            if buffered_text:
                await self.send(
                    {
                        "type": "response.text.delta",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "response_id": state.response_id,
                        "provisional_id": state.response_id,
                        "delta": buffered_text,
                        "replayed_from_provisional": True,
                    }
                )
            if state.completed:
                done_timing = await self._send_reply_done(
                    turn,
                    response_id=state.response_id,
                    text=buffered_text,
                    source=state.source,
                    finish_reason=state.finish_reason,
                    usage=state.usage,
                    provisional_id=state.response_id,
                )
                state.official_done = True
                state.official_text_done_after_commit_ms = done_timing[
                    "text_done_after_commit_ms"
                ]
                state.official_response_done_after_commit_ms = done_timing[
                    "response_done_after_commit_ms"
                ]
            emit_structured_log(
                "reply",
                "provisional_reply_resolved",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=state.response_id,
                status=state.status,
                reason=state.resolution_reason,
                chars=len(buffered_text),
                delta_count=state.delta_count,
                resolved_after_commit_ms=self._after_commit_ms(turn),
            )

    async def _discard_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        reason: str,
        send_event: bool = True,
        abort_request: bool = True,
    ) -> None:
        async with state.lock:
            if state.status != "pending":
                return
            state.status = "discarded"
            state.resolution_reason = reason
            if send_event:
                await self.send(
                    {
                        "type": "response.provisional.resolved",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "provisional_id": state.response_id,
                        "status": "discarded",
                        "reason": reason,
                        "discarded_chars": len("".join(state.text_parts)),
                        "discarded_delta_count": state.delta_count,
                    }
                )
            emit_structured_log(
                "reply",
                "provisional_reply_resolved",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=state.response_id,
                status=state.status,
                reason=reason,
                chars=len("".join(state.text_parts)),
                delta_count=state.delta_count,
                resolved_after_commit_ms=self._after_commit_ms(turn),
            )
        if abort_request and state.request_id is not None:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                try:
                    await abort(state.request_id)
                except Exception:
                    logger.warning(
                        "[SESSION_ACTION_REALTIME] provisional reply abort failed "
                        "session_id=%s turn_id=%s request_id=%s",
                        self.session_id,
                        turn.turn_id,
                        state.request_id,
                        exc_info=True,
                    )
        if state.task is not None and state.task is not asyncio.current_task():
            if not state.task.done():
                state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)

    async def _run_generated_reply(
        self,
        turn: TurnBuffer,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        category: SessionActionCategory | None,
        *,
        support_status: str = "supported",
        provisional: ProvisionalReplyState | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._ensure_turn_processing(turn)
        request_id = f"{turn.request_base}-reply"
        response_id = (
            provisional.response_id
            if provisional is not None
            else f"resp-{uuid.uuid4().hex}"
        )
        if provisional is not None:
            provisional.request_id = request_id
        request, reply_forwarded_image_roles = self._build_reply_request(
            turn,
            audios,
            images,
            image_roles,
            category,
            support_status=support_status,
        )
        effective_system_prompt = next(
            (
                message.content
                for message in request.messages or []
                if message.role == "system" and isinstance(message.content, str)
            ),
            None,
        )
        system_prompt_audit = _text_audit_fields(
            "system_prompt", effective_system_prompt
        )
        diagnostic_messages = [message.to_dict() for message in request.messages or []]
        if not self.log_full_instructions:
            for message in diagnostic_messages:
                if message.get("role") == "system":
                    message["content"] = "<redacted; see system_prompt_sha256>"
        started = (
            provisional.started_at if provisional is not None else time.perf_counter()
        )
        first_token_ms: float | None = None
        first_delta_after_commit_ms: float | None = None
        delta_count = 0
        text_parts: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        emit_structured_log(
            "diagnostic",
            "reply_logical_input",
            component="api",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            response_id=response_id,
            reply_source="generated",
            provisional=provisional is not None,
            turn_origin=turn.turn_origin,
            instructions_applied=bool(effective_system_prompt),
            effective_system_prompt=(
                effective_system_prompt if self.log_full_instructions else None
            ),
            **system_prompt_audit,
            messages=diagnostic_messages,
            selected_category=(
                {
                    "category_id": category.category_id,
                    "source_label": category.source_label,
                    "short_definition": category.short_definition,
                }
                if category is not None
                else None
            ),
            support_status=support_status,
            sampling=request.sampling.to_dict(),
            output_modalities=list(request.output_modalities or []),
            current_audio=_summarize_media(audios),
            current_images=_summarize_media([frame.data_uri for frame in turn.images]),
            current_image_roles=list(image_roles),
            received_image_roles=list(image_roles),
            reply_forwarded_image_roles=list(reply_forwarded_image_roles),
            reply_filtered_avatar_image_count=image_roles.count(
                IMAGE_ROLE_AVATAR_STATE
            ),
            reply_filtered_stale_user_camera_image_count=max(
                0, image_roles.count(IMAGE_ROLE_USER_CAMERA) - 1
            ),
            user_camera_present=bool(reply_forwarded_image_roles),
            reply_history_turn_count=len(self.reply_history_turns),
            last_executed_action=self._executed_action_log_fields(
                self.last_executed_action
            ),
            last_user_executed_action=self._executed_action_log_fields(
                self.last_user_executed_action
            ),
        )
        if provisional is None:
            await self.send(
                {
                    "type": "response.created",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "response": {
                        "id": response_id,
                        "status": "in_progress",
                        "source": "generated",
                    },
                }
            )
            created_after_commit_ms = self._after_commit_ms(turn)
        else:
            created_after_commit_ms = provisional.created_after_commit_ms
        self._register_turn_request(turn, request_id)
        emit_structured_log(
            "reply",
            "reply_submitted",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            response_id=response_id,
            reply_source="generated",
            provisional=provisional is not None,
            instructions_applied=bool(effective_system_prompt),
            **system_prompt_audit,
            created_after_commit_ms=created_after_commit_ms,
        )
        try:
            completion_stream = getattr(self.client, "completion_stream", None)
            if callable(completion_stream):
                stream = completion_stream(request, request_id=request_id)
                async with aclosing(stream):
                    async for chunk in stream:
                        self._ensure_turn_processing(turn)
                        if chunk.modality == "text" and chunk.text:
                            is_first_delta = first_token_ms is None
                            if is_first_delta:
                                first_token_ms = (
                                    time.perf_counter() - started
                                ) * 1000.0
                            text_parts.append(chunk.text)
                            if provisional is None:
                                await self.send(
                                    {
                                        "type": "response.text.delta",
                                        "session_id": self.session_id,
                                        "turn_id": turn.turn_id,
                                        "response_id": response_id,
                                        "delta": chunk.text,
                                    }
                                )
                            else:
                                await self._send_provisional_reply_delta(
                                    turn, provisional, chunk.text
                                )
                            delta_count += 1
                            if is_first_delta:
                                first_delta_after_commit_ms = self._after_commit_ms(
                                    turn
                                )
                                emit_structured_log(
                                    "reply",
                                    "reply_first_token",
                                    session_id=self.session_id,
                                    turn_id=turn.turn_id,
                                    trace_id=turn.trace_id,
                                    request_id=request_id,
                                    ttft_ms=round(first_token_ms, 3),
                                    first_delta_after_commit_ms=(
                                        first_delta_after_commit_ms
                                    ),
                                )
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                            if chunk.usage is not None:
                                usage = chunk.usage.to_dict()
            else:
                result = await self.client.completion(request, request_id=request_id)
                if result.text:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                    text_parts.append(result.text)
                    if provisional is None:
                        await self.send(
                            {
                                "type": "response.text.delta",
                                "session_id": self.session_id,
                                "turn_id": turn.turn_id,
                                "response_id": response_id,
                                "delta": result.text,
                            }
                        )
                    else:
                        await self._send_provisional_reply_delta(
                            turn, provisional, result.text
                        )
                    delta_count = 1
                    first_delta_after_commit_ms = self._after_commit_ms(turn)
                    emit_structured_log(
                        "reply",
                        "reply_first_token",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        request_id=request_id,
                        ttft_ms=round(first_token_ms, 3),
                        first_delta_after_commit_ms=first_delta_after_commit_ms,
                    )
                finish_reason = result.finish_reason
                if result.usage is not None:
                    usage = result.usage.to_dict()
            reply_text = "".join(text_parts)
            total_ms = (time.perf_counter() - started) * 1000.0
            self._ensure_turn_processing(turn)
            if provisional is None:
                done_timing = await self._send_reply_done(
                    turn,
                    response_id=response_id,
                    text=reply_text,
                    source="generated",
                    finish_reason=finish_reason,
                    usage=usage,
                )
            else:
                done_timing = await self._finish_provisional_reply(
                    turn,
                    provisional,
                    finish_reason=finish_reason,
                    usage=usage,
                )
            text_done_after_commit_ms = done_timing["text_done_after_commit_ms"]
            stream_duration_ms = (
                round(
                    text_done_after_commit_ms - first_delta_after_commit_ms,
                    3,
                )
                if text_done_after_commit_ms is not None
                and first_delta_after_commit_ms is not None
                else None
            )
            completion_tokens = (
                usage.get("completion_tokens") if usage is not None else None
            )
            if provisional is not None:
                timing = self._provisional_reply_timing(provisional, total_ms=total_ms)
            else:
                timing = {
                    "source": "generated",
                    "ttft_ms": round(first_token_ms or total_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "chars": len(reply_text),
                    "created_after_commit_ms": created_after_commit_ms,
                    "first_delta_after_commit_ms": first_delta_after_commit_ms,
                    "text_done_after_commit_ms": text_done_after_commit_ms,
                    "response_done_after_commit_ms": done_timing[
                        "response_done_after_commit_ms"
                    ],
                    "stream_duration_ms": stream_duration_ms,
                    "delta_count": delta_count,
                    "completion_tokens": completion_tokens,
                    "provisional": False,
                    "provisional_status": None,
                    "provisional_done_after_commit_ms": None,
                    "resolution_reason": None,
                }
            emit_structured_log(
                "reply",
                "reply_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
                response_id=response_id,
                logical_request_id=turn.request_base,
                output_text=reply_text,
                finish_reason=finish_reason,
                usage=usage,
                **timing,
            )
            return reply_text, timing
        except asyncio.CancelledError:
            emit_structured_log(
                "reply",
                "reply_cancelled",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
            )
            raise
        except Exception as exc:
            emit_structured_log(
                "error",
                "reply_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                response_id=response_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            self._unregister_turn_request(turn, request_id)

    async def _run_provided_reply(
        self,
        turn: TurnBuffer,
        text: str,
        *,
        provisional: ProvisionalReplyState | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._ensure_turn_processing(turn)
        response_id = (
            provisional.response_id
            if provisional is not None
            else f"resp-{uuid.uuid4().hex}"
        )
        started = (
            provisional.started_at if provisional is not None else time.perf_counter()
        )
        if provisional is not None:
            if text:
                await self._send_provisional_reply_delta(turn, provisional, text)
            await self._finish_provisional_reply(
                turn,
                provisional,
                finish_reason="provided",
                usage=None,
            )
            total_ms = (time.perf_counter() - started) * 1000.0
            timing = self._provisional_reply_timing(provisional, total_ms=total_ms)
            emit_structured_log(
                "reply",
                "provided_reply_used",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=response_id,
                output_text=text,
                finish_reason="provided",
                usage=None,
                instructions_applied=False,
                **_text_audit_fields("instructions", self.instructions),
                **timing,
            )
            return text, timing
        await self.send(
            {
                "type": "response.created",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response": {
                    "id": response_id,
                    "status": "in_progress",
                    "source": "provided",
                },
            }
        )
        created_after_commit_ms = self._after_commit_ms(turn)
        await self.send(
            {
                "type": "response.text.delta",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response_id": response_id,
                "delta": text,
            }
        )
        first_delta_after_commit_ms = self._after_commit_ms(turn)
        done_timing = await self._send_reply_done(
            turn,
            response_id=response_id,
            text=text,
            source="provided",
            finish_reason="provided",
            usage=None,
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        text_done_after_commit_ms = done_timing["text_done_after_commit_ms"]
        timing = {
            "source": "provided",
            "ttft_ms": 0.0,
            "total_ms": round(total_ms, 3),
            "chars": len(text),
            "created_after_commit_ms": created_after_commit_ms,
            "first_delta_after_commit_ms": first_delta_after_commit_ms,
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": done_timing[
                "response_done_after_commit_ms"
            ],
            "stream_duration_ms": (
                round(
                    text_done_after_commit_ms - first_delta_after_commit_ms,
                    3,
                )
                if text_done_after_commit_ms is not None
                and first_delta_after_commit_ms is not None
                else None
            ),
            "delta_count": 1,
            "completion_tokens": None,
        }
        emit_structured_log(
            "reply",
            "provided_reply_used",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            response_id=response_id,
            output_text=text,
            finish_reason="provided",
            usage=None,
            instructions_applied=False,
            **_text_audit_fields("instructions", self.instructions),
            **timing,
        )
        return text, timing

    async def _send_reply_done(
        self,
        turn: TurnBuffer,
        *,
        response_id: str,
        text: str,
        source: Literal["generated", "provided"],
        finish_reason: str,
        usage: dict[str, Any] | None,
        provisional_id: str | None = None,
    ) -> dict[str, float | None]:
        text_done_payload: dict[str, Any] = {
            "type": "response.text.done",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "response_id": response_id,
            "text": text,
        }
        if provisional_id is not None:
            text_done_payload["provisional_id"] = provisional_id
        await self.send(text_done_payload)
        text_done_after_commit_ms = self._after_commit_ms(turn)
        response: dict[str, Any] = {
            "id": response_id,
            "status": "completed",
            "status_details": {"reason": finish_reason},
            "source": source,
            "output": [{"type": "text", "text": text}],
        }
        if usage is not None:
            response["usage"] = usage
        if provisional_id is not None:
            response["provisional_id"] = provisional_id
        await self.send(
            {
                "type": "response.done",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response": response,
            }
        )
        return {
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": self._after_commit_ms(turn),
        }

    def _append_reply_history(
        self,
        turn: TurnBuffer,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        reply_text: str | None,
        *,
        model_visible: bool = True,
        history_kind: Literal["reply", "unsupported_action_notice"] = "reply",
    ) -> None:
        if not reply_text:
            return
        if len(images) != len(image_roles):
            raise ValueError("reply images and image_roles must have the same length")
        if turn.turn_origin == TURN_ORIGIN_USER:
            messages = [
                {
                    "role": "user",
                    "content": self._reply_history_user_content(
                        audios,
                        [],
                        [],
                        turn.text,
                    ),
                },
                {"role": "assistant", "content": reply_text},
            ]
        else:
            messages = [{"role": "assistant", "content": reply_text}]
        self.reply_history_turns.append(
            ReplyHistoryTurn(
                turn_id=turn.turn_id,
                messages=messages,
                audios=list(audios) if turn.turn_origin == TURN_ORIGIN_USER else [],
                # Both camera roles represent transient visual state. Keep
                # only the spoken/text exchange in reply history so an old
                # frame cannot become evidence for a later question.
                images=[],
                image_roles=[],
                model_visible=model_visible,
                history_kind=history_kind,
            )
        )

    def _reply_history_user_content(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
    ) -> list[dict[str, Any]]:
        if len(images) != len(image_roles):
            raise ValueError(
                "reply history images and image_roles must have the same length"
            )
        parts: list[dict[str, Any]] = []
        if image_roles:
            parts.append(self._reply_user_camera_context_part())
            parts.extend({"type": "image"} for _ in image_roles)
        parts.extend({"type": "audio"} for _ in audios)
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        return parts

    def _record_action_as_executed(
        self,
        *,
        turn: TurnBuffer,
        action: dict[str, Any],
    ) -> None:
        """Persist a successful inference as an execution fact for later turns."""
        candidate_id = str(action["candidate_id"])
        candidate = self.candidate_by_id[candidate_id]
        record = ExecutedActionRecord(
            turn_id=turn.turn_id,
            turn_origin=turn.turn_origin,
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            category_id=candidate.category_id,
            source_label=candidate.source_label,
            short_definition=candidate.short_definition,
            execute=bool(action.get("execute", True)),
        )
        self.last_executed_action = record
        if turn.turn_origin == TURN_ORIGIN_USER:
            self.last_user_executed_action = record
        self.executed_action_history.append(record)
        if len(self.executed_action_history) > MAX_EXECUTED_ACTION_HISTORY_TURNS:
            del self.executed_action_history[:-MAX_EXECUTED_ACTION_HISTORY_TURNS]
        log_fields = self._executed_action_log_fields(record)
        assert log_fields is not None
        emit_structured_log(
            "action",
            "action_execution_assumed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            **log_fields,
            retained_execution_history_count=len(self.executed_action_history),
        )

    @staticmethod
    def _executed_action_log_fields(
        record: ExecutedActionRecord | None,
    ) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "source_turn_id": record.turn_id,
            "source_turn_origin": record.turn_origin,
            "candidate_id": record.candidate_id,
            "action_id": record.action_id,
            "category_id": record.category_id,
            "source_label": record.source_label,
            "execute": record.execute,
        }

    @staticmethod
    def _register_turn_request(turn: TurnBuffer, request_id: str) -> None:
        turn.active_request_ids.add(request_id)
        turn.current_request_id = request_id

    @staticmethod
    def _unregister_turn_request(turn: TurnBuffer, request_id: str) -> None:
        turn.active_request_ids.discard(request_id)
        if turn.current_request_id == request_id:
            turn.current_request_id = next(iter(turn.active_request_ids), None)

    async def _score_action_request(
        self,
        turn: TurnBuffer,
        request: ActionSuffixScoreRequest,
    ) -> Any:
        self._ensure_turn_processing(turn)
        self._register_turn_request(turn, request.request_id)
        started = time.perf_counter()
        emit_structured_log(
            "action",
            "action_scoring_started",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=request.logical_request_id,
            request_id=request.request_id,
            stage=request.stage,
            locale=self.locale,
            language=request.language,
            candidate_count=len(request.candidates),
            prefix_cache_namespace=request.prefix_cache_namespace,
            **_text_audit_fields("system_prompt", request.system_prompt),
        )
        try:
            result = await self.client.score_action_suffixes(request)
            emit_structured_log(
                "action",
                "action_scoring_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                locale=self.locale,
                language=request.language,
                candidate_count=len(request.candidates),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                prefix_cached=result.prefix_cached,
                stats=result.stats,
            )
            if not result.prefix_cached:
                emit_structured_log(
                    "action",
                    "action_prefix_cache_miss",
                    level="warning",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    request_id=request.request_id,
                    stage=request.stage,
                    locale=self.locale,
                    language=request.language,
                    prefix_cache_namespace=request.prefix_cache_namespace,
                    **_text_audit_fields("system_prompt", request.system_prompt),
                )
        except asyncio.CancelledError:
            emit_structured_log(
                "action",
                "action_scoring_cancelled",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                locale=self.locale,
                language=request.language,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            raise
        except Exception as exc:
            emit_structured_log(
                "error",
                "action_scoring_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                locale=self.locale,
                language=request.language,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            self._unregister_turn_request(turn, request.request_id)
        self._ensure_turn_processing(turn)
        return result

    async def _score_action(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
        on_category_selected: (
            Callable[[SessionActionCategory | None, str], None] | None
        ) = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        if (
            self.categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
        ):
            return await self._score_action_hierarchical(
                audios,
                images,
                image_roles,
                text,
                avatar_state,
                turn_origin=turn_origin,
                text_role=text_role,
                trigger=trigger,
                turn_id=turn_id,
                turn=turn,
                request_base=request_base,
                on_category_selected=on_category_selected,
            )
        return await self._score_action_flat(
            audios,
            images,
            image_roles,
            text,
            avatar_state,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            turn_id=turn_id,
            turn=turn,
            request_base=request_base,
        )

    async def _score_action_hierarchical(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
        on_category_selected: (
            Callable[[SessionActionCategory | None, str], None] | None
        ) = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_image_roles,
            action_context,
        ) = self._build_bounded_action_context(
            audios, images, image_roles, include_history=False
        )
        action_history = self._build_compact_action_history()
        action_history_audios = []
        action_history_images = []
        action_context.update(
            {
                "history_policy": ("latest_reply_physical_and_user_action_facts"),
                "source_history_turn_count": len(self.history_turns),
                "history_turn_count": 1 if action_history else 0,
                "history_audio_count": 0,
                "history_image_count": 0,
                "truncated": bool(self.history_turns),
            }
        )
        action_history = self._with_current_proactive_text(
            action_history, text, turn_origin
        )
        effective_avatar_state = self._effective_avatar_state(
            avatar_state,
            turn_origin=turn_origin,
            has_avatar_image=IMAGE_ROLE_AVATAR_STATE in action_image_roles,
        )
        excluded_category_ids = (
            self._state_description_excluded_category_ids(
                effective_avatar_state.get("state_description")
            )
            if self.global_action_catalog is not None
            else ()
        )
        excluded_category_id_set = set(excluded_category_ids)
        if excluded_category_ids:
            emit_structured_log(
                "action",
                "state_description_candidates_filtered",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                stage="category",
                excluded_category_ids=list(excluded_category_ids),
                **_text_audit_fields(
                    "state_description",
                    effective_avatar_state.get("state_description"),
                ),
            )
        base = self._build_turn_action_instruction(
            text,
            turn_origin=turn_origin,
            trigger=trigger,
            has_audio=bool(audios),
            image_roles=action_image_roles,
            has_current_action_id=(
                effective_avatar_state.get("current_action_id") is not None
            ),
            has_state_description=("state_description" in effective_avatar_state),
            avatar_state_source=self._avatar_state_source(
                effective_avatar_state, action_image_roles
            ),
        )
        common = dict(
            model=self.model_name,
            language=self.language,
            audios=audios,
            images=action_images,
            sample_rate=16000,
            image_roles=action_image_roles,
            session_id=self.session_id,
            history=action_history,
            stage="category",
            logical_request_id=request_base,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            action_context_cache_key=request_base,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            cache_static_system_only=self.global_action_catalog is not None,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=effective_avatar_state,
        )
        category_candidates = [
            ActionScoreCandidate(
                candidate_id=item.category_id,
                suffix=item.category_id,
                action_id=item.category_id,
            )
            for item in self.categories
            if item.category_id not in excluded_category_id_set
        ]
        if self.global_action_catalog is not None:
            category_candidates.append(
                ActionScoreCandidate(
                    candidate_id=UNSUPPORTED_CATEGORY_SCORE_ID,
                    suffix=UNSUPPORTED_CATEGORY_SCORE_ID,
                    action_id=UNSUPPORTED_DECISION_ID,
                )
            )
        category_request = ActionSuffixScoreRequest(
            request_id=request_base + "-category",
            prefix=(
                self._build_session_action_profile_instruction("category")
                + base
                + self._category_whitelist_instruction()
                + self._state_description_exclusion_instruction(
                    category_ids=excluded_category_ids
                )
                + self._state_description_priority_instruction(
                    "category",
                    enabled=("state_description" in effective_avatar_state),
                )
                + self._prompt(
                    zh="最合适的 category_id：",
                    en="Best matching category_id:",
                )
            ),
            system_prompt=self._build_category_system_prompt(),
            candidates=category_candidates,
            suffix_tokenization_mode="short_id",
            micro_batch_size=self.action_micro_batch_size,
            **common,
        )
        started = time.perf_counter()
        category_started = time.perf_counter()
        category_result = await self._score_action_request(turn, category_request)
        category_ms = round((time.perf_counter() - category_started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=category candidates=%d "
            "elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(category_request.candidates),
            category_ms,
            category_result.prefix_cached,
            json.dumps(category_result.stats, ensure_ascii=False, default=str),
        )
        category_by_id = {item.category_id: item for item in self.categories}
        category_ranked = sorted(
            category_result.scores, key=lambda item: item.mean_logprob, reverse=True
        )
        if not category_ranked:
            raise ValueError("category action score did not return a decision")
        category_unsupported = (
            category_ranked[0].candidate_id == UNSUPPORTED_CATEGORY_SCORE_ID
        )
        if category_unsupported:
            selected_category = None
            selected_categories = [self._primary_fallback_category()]
        else:
            if category_ranked[0].candidate_id not in category_by_id:
                raise ValueError(
                    "category action score did not return a valid category"
                )
            selected_categories = [
                category_by_id[item.candidate_id]
                for item in category_ranked[: self.action_category_top_k]
                if item.candidate_id in category_by_id
            ]
            if not selected_categories:
                raise ValueError(
                    "category action score did not select a valid category"
                )
            selected_category = selected_categories[0]
        execution_category = selected_categories[0]
        selected_category_ids = [item.category_id for item in selected_categories]
        if on_category_selected is not None:
            on_category_selected(
                selected_category,
                "unsupported" if category_unsupported else "supported",
            )

        def compact_stage_score(score: Any) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id,
                "token_count": score.token_count,
                "mean_logprob": score.mean_logprob,
                "mean_nll": score.mean_nll,
                "ppl": score.ppl,
                "token_scores": [
                    {"token_id": item.token_id, "logprob": item.logprob}
                    for item in score.token_scores
                ],
            }

        child_candidates = self._child_candidates_for_categories(selected_categories)
        excluded_candidate_ids = (
            self._state_description_excluded_candidate_ids(
                effective_avatar_state.get("state_description"), child_candidates
            )
            if self.global_action_catalog is not None
            else ()
        )
        excluded_candidate_id_set = set(excluded_candidate_ids)
        if excluded_candidate_ids:
            emit_structured_log(
                "action",
                "state_description_candidates_filtered",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                stage="child",
                selected_category_ids=selected_category_ids,
                excluded_candidate_ids=list(excluded_candidate_ids),
                **_text_audit_fields(
                    "state_description",
                    effective_avatar_state.get("state_description"),
                ),
            )
        child_candidates = [
            candidate
            for candidate in child_candidates
            if candidate.candidate_id not in excluded_candidate_id_set
        ]
        if (
            len(selected_categories) == 1
            and len(child_candidates) == 1
            and not self.include_scores
            and self.global_action_catalog is None
        ):
            only_child = child_candidates[0]
            total_ms = round((time.perf_counter() - started) * 1000.0, 3)
            action = {
                "candidate_id": only_child.candidate_id,
                "action_id": only_child.action_id,
                "category_id": only_child.category_id,
                "execution_binding": dict(only_child.execution_binding),
                "execute": only_child.action_id != "no_action",
            }
            if self.global_action_catalog is not None:
                action.update(
                    {
                        "support_status": (
                            "unsupported" if category_unsupported else "supported"
                        ),
                        "fallback_applied": category_unsupported,
                    }
                )
            action_context.update(
                {
                    "selection_stages": 1,
                    "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                    "logical_request_id": request_base,
                    "selected_category_id": execution_category.category_id,
                    "category_decision_id": (
                        UNSUPPORTED_DECISION_ID
                        if category_unsupported
                        else execution_category.category_id
                    ),
                    "category_scoring_candidate_id": category_ranked[0].candidate_id,
                    "support_status": (
                        "unsupported" if category_unsupported else "supported"
                    ),
                    "fallback_applied": category_unsupported,
                    "selected_category_ids": selected_category_ids,
                    "category_top_k": self.action_category_top_k,
                    "state_description_excluded_category_ids": list(
                        excluded_category_ids
                    ),
                    "state_description_excluded_candidate_ids": list(
                        excluded_candidate_ids
                    ),
                    "category_compute_ms": category_ms,
                    "child_compute_ms": 0.0,
                    "child_prefix_prefilled": (
                        execution_category.category_id
                        in self.global_action_prewarm.for_locale(
                            self.locale
                        ).ready_child_category_ids
                        if self.global_action_catalog is not None
                        else False
                    ),
                    "child_scoring_skipped": True,
                    "child_scoring_skip_reason": "single_child",
                    "action_timing_breakdown": {
                        "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                        "category": _action_timing_breakdown(category_result.stats),
                        "child": {
                            "skipped": True,
                            "reason": "single_child",
                        },
                        "child_catalog_prefill_ms": 0.0,
                        "total_ms": total_ms,
                    },
                }
            )
            return action, [], total_ms, action_context

        child_namespace = ",".join(selected_category_ids)
        if len(selected_categories) == 1:
            child_system_prompt = self._build_child_system_prompt(
                execution_category, child_candidates
            )
        else:
            child_system_prompt = self._build_child_system_prompt(
                selected_categories, child_candidates
            )
        child_namespace = (
            self.global_action_catalog.child_cache_namespace(
                execution_category.category_id, self.locale
            )
            if self.global_action_catalog is not None
            else f"{self.action_prefix_cache_namespace}:child:{child_namespace}"
        )

        # Global Child catalogs are prewarmed before the server starts. The
        # legacy/session-local path retains its lazy first-use prefill.
        child_prefix_prefilled = (
            execution_category.category_id
            in self.global_action_prewarm.for_locale(
                self.locale
            ).ready_child_category_ids
            if self.global_action_catalog is not None
            else False
        )
        child_catalog_prefill_ms = 0.0
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if (
            callable(prefill)
            and self.global_action_catalog is None
            and child_namespace not in self._prefilled_action_prefix_namespaces
        ):
            child_prefill_started = time.perf_counter()
            prefill_request_id = request_base + "-child-prefill"
            self._ensure_turn_processing(turn)
            self._register_turn_request(turn, prefill_request_id)
            try:
                child_prefix_prefilled = await prefill(
                    request_id=prefill_request_id,
                    model=self.model_name,
                    system_prompt=child_system_prompt,
                    candidates=[
                        ActionScoreCandidate(
                            candidate_id=item.candidate_id,
                            suffix=item.candidate_id,
                            action_id=item.action_id,
                            execution_binding=dict(item.execution_binding),
                        )
                        for item in child_candidates
                    ],
                    prefix_cache_namespace=child_namespace,
                    stage="child",
                    language=self.language,
                )
            finally:
                self._unregister_turn_request(turn, prefill_request_id)
            self._ensure_turn_processing(turn)
            if child_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(child_namespace)
            child_catalog_prefill_ms = round(
                (time.perf_counter() - child_prefill_started) * 1000.0, 3
            )

        action_common = {
            **common,
            "stage": "child",
            "micro_batch_size": self.action_micro_batch_size,
            "prefix_cache_namespace": child_namespace,
        }
        action_request = ActionSuffixScoreRequest(
            request_id=request_base + "-child",
            prefix=(
                self._build_session_action_profile_instruction("child")
                + base
                + self._child_whitelist_instruction(
                    execution_category, child_candidates
                )
                + self._state_description_exclusion_instruction(
                    candidate_ids=excluded_candidate_ids
                )
                + self._state_description_priority_instruction(
                    "child",
                    enabled=("state_description" in effective_avatar_state),
                )
                + self._prompt(
                    zh="最合适的 candidate_id：",
                    en="Best matching candidate_id:",
                )
            ),
            system_prompt=child_system_prompt,
            candidates=[
                ActionScoreCandidate(
                    candidate_id=item.candidate_id,
                    suffix=item.candidate_id,
                    action_id=item.action_id,
                    execution_binding=dict(item.execution_binding),
                )
                for item in child_candidates
            ]
            + (
                []
                if self.global_action_catalog is None
                else [
                    ActionScoreCandidate(
                        candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                        suffix=UNSUPPORTED_CHILD_SCORE_ID,
                        action_id=UNSUPPORTED_DECISION_ID,
                    )
                ]
            ),
            suffix_tokenization_mode="short_id",
            **action_common,
        )
        child_started = time.perf_counter()
        child_result = await self._score_action_request(turn, action_request)
        child_ms = round((time.perf_counter() - child_started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=child candidates=%d "
            "selected_category_ids=%s elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(action_request.candidates),
            ",".join(selected_category_ids),
            child_ms,
            child_result.prefix_cached,
            json.dumps(child_result.stats, ensure_ascii=False, default=str),
        )
        child_by_id = {item.candidate_id: item for item in child_candidates}
        ranked = sorted(
            child_result.scores, key=lambda item: item.mean_logprob, reverse=True
        )
        if not ranked:
            raise ValueError("child action score did not return a decision")
        child_unsupported = ranked[0].candidate_id == UNSUPPORTED_CHILD_SCORE_ID
        if not child_unsupported and ranked[0].candidate_id not in child_by_id:
            raise ValueError("child action score did not return a valid candidate")

        def score_dict(score: Any, candidate: SessionActionCandidate) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id,
                "action_id": candidate.action_id,
                "category_id": candidate.category_id,
                "source_label": candidate.source_label,
                "short_definition": candidate.short_definition,
                "execution_binding": dict(candidate.execution_binding),
                "token_count": score.token_count,
                "mean_logprob": score.mean_logprob,
                "mean_nll": score.mean_nll,
                "ppl": score.ppl,
                "token_scores": [
                    {"token_id": item.token_id, "logprob": item.logprob}
                    for item in score.token_scores
                ],
            }

        scores = [
            (
                compact_stage_score(score) | {"decision": "unsupported"}
                if score.candidate_id == UNSUPPORTED_CHILD_SCORE_ID
                else score_dict(score, child_by_id[score.candidate_id])
            )
            for score in ranked
        ]
        if child_unsupported:
            fallback = self._default_fallback_candidate()
            action = {
                "candidate_id": fallback.candidate_id,
                "action_id": fallback.action_id,
                "category_id": fallback.category_id,
                "execution_binding": dict(fallback.execution_binding),
                "execute": True,
                "support_status": "unsupported",
                "fallback_applied": True,
            }
        else:
            top = scores[0]
            action = {
                "candidate_id": top["candidate_id"],
                "action_id": top["action_id"],
                "category_id": top["category_id"],
                "execution_binding": dict(top.get("execution_binding") or {}),
                "execute": top["action_id"] != "no_action",
                "mean_logprob": top["mean_logprob"],
                "ppl": top["ppl"],
                "token_count": top["token_count"],
            }
            if self.global_action_catalog is not None:
                action.update(
                    {
                        "support_status": (
                            "unsupported" if category_unsupported else "supported"
                        ),
                        "fallback_applied": category_unsupported,
                    }
                )
        action_context.update(
            {
                "selection_stages": 2,
                "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                "logical_request_id": request_base,
                "selected_category_id": execution_category.category_id,
                "category_decision_id": (
                    UNSUPPORTED_DECISION_ID
                    if category_unsupported
                    else execution_category.category_id
                ),
                "category_scoring_candidate_id": category_ranked[0].candidate_id,
                "child_decision_id": (
                    UNSUPPORTED_DECISION_ID
                    if child_unsupported
                    else ranked[0].candidate_id
                ),
                "child_scoring_candidate_id": ranked[0].candidate_id,
                "support_status": action.get("support_status"),
                "fallback_applied": action.get("fallback_applied"),
                "selected_category_ids": selected_category_ids,
                "category_top_k": self.action_category_top_k,
                "state_description_excluded_category_ids": list(excluded_category_ids),
                "state_description_excluded_candidate_ids": list(
                    excluded_candidate_ids
                ),
                "category_scores": [
                    compact_stage_score(score) for score in category_ranked
                ],
                "category_compute_ms": category_ms,
                "child_compute_ms": child_ms,
                "child_prefix_prefilled": child_prefix_prefilled,
                "child_prefix_cache_namespace": child_namespace,
                "child_catalog_prefill_ms": child_catalog_prefill_ms,
                "action_timing_breakdown": {
                    "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                    "category": _action_timing_breakdown(category_result.stats),
                    "child": _action_timing_breakdown(child_result.stats),
                    "child_catalog_prefill_ms": child_catalog_prefill_ms,
                    "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
                },
            }
        )
        return (
            action,
            scores,
            round((time.perf_counter() - started) * 1000.0, 3),
            action_context,
        )

    async def _score_action_flat(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_image_roles,
            action_context,
        ) = self._build_bounded_action_context(audios, images, image_roles)
        action_history = self._with_current_proactive_text(
            action_history, text, turn_origin
        )
        effective_avatar_state = self._effective_avatar_state(
            avatar_state,
            turn_origin=turn_origin,
            has_avatar_image=IMAGE_ROLE_AVATAR_STATE in action_image_roles,
        )
        prefix = (
            self._build_session_action_profile_instruction("single")
            + self._build_turn_action_instruction(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                has_audio=bool(audios),
                image_roles=action_image_roles,
                has_current_action_id=(
                    effective_avatar_state.get("current_action_id") is not None
                ),
                has_state_description=("state_description" in effective_avatar_state),
                avatar_state_source=self._avatar_state_source(
                    effective_avatar_state, action_image_roles
                ),
            )
            + self._state_description_priority_instruction(
                "single",
                enabled=("state_description" in effective_avatar_state),
            )
            + self._prompt(
                zh="最合适的 candidate_id：",
                en="Best matching candidate_id:",
            )
        )
        candidates = [
            ActionScoreCandidate(
                candidate_id=item.candidate_id,
                suffix=item.candidate_id,
                action_id=item.action_id,
                execution_binding=dict(item.execution_binding),
            )
            for item in self.candidates
        ]
        request = ActionSuffixScoreRequest(
            request_id=request_base + "-single",
            model=self.model_name,
            prefix=prefix,
            system_prompt=self.action_system_prompt,
            language=self.language,
            candidates=candidates,
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=action_images,
            image_roles=action_image_roles,
            sample_rate=16000,
            micro_batch_size=self.action_micro_batch_size,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            cache_static_system_only=self.global_action_catalog is not None,
            session_id=self.session_id,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            history=action_history,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=effective_avatar_state,
        )
        started = time.perf_counter()
        result = await self._score_action_request(turn, request)
        compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=flat candidates=%d "
            "elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(request.candidates),
            compute_ms,
            result.prefix_cached,
            json.dumps(result.stats, ensure_ascii=False, default=str),
        )
        ranked = sorted(result.scores, key=lambda x: x.mean_logprob, reverse=True)
        scores: list[dict[str, Any]] = []
        for score in ranked:
            candidate = self.candidate_by_id[score.candidate_id]
            scores.append(
                {
                    "candidate_id": score.candidate_id,
                    "action_id": candidate.action_id,
                    **(
                        {"category_id": candidate.category_id}
                        if candidate.category_id
                        else {}
                    ),
                    "source_label": candidate.source_label,
                    "short_definition": candidate.short_definition,
                    "execution_binding": dict(candidate.execution_binding),
                    "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob,
                    "mean_nll": score.mean_nll,
                    "ppl": score.ppl,
                    "token_scores": [
                        {"token_id": item.token_id, "logprob": item.logprob}
                        for item in score.token_scores
                    ],
                }
            )
        top = scores[0]
        action = {
            "candidate_id": top["candidate_id"],
            "action_id": top["action_id"],
            **({"category_id": top["category_id"]} if top.get("category_id") else {}),
            "execution_binding": dict(top.get("execution_binding") or {}),
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"],
            "ppl": top["ppl"],
            "token_count": top["token_count"],
        }
        action_context.update(
            {
                "selection_stages": 1,
                "selection_mode": (
                    self.action_selection_mode if self.categories else "flat"
                ),
                "flattened_child_count": (
                    len(self.candidates) if self.categories else None
                ),
                "compute_ms": compute_ms,
                "action_timing_breakdown": {
                    "selection_mode": self.action_selection_mode,
                    "single": _action_timing_breakdown(result.stats),
                    "total_ms": compute_ms,
                },
            }
        )
        return action, scores, compute_ms, action_context

    @staticmethod
    def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
        """Return only fields required by the external action executor."""
        compact = {
            "action_id": action["action_id"],
            "candidate_id": action["candidate_id"],
        }
        if action.get("category_id"):
            compact["category_id"] = action["category_id"]
        compact["execute"] = action["execute"]
        if action.get("support_status"):
            compact["support_status"] = action["support_status"]
        if "fallback_applied" in action:
            compact["fallback_applied"] = bool(action["fallback_applied"])
        execution_binding = action.get("execution_binding")
        if execution_binding:
            compact["execution_binding"] = dict(execution_binding)
        return compact

    def _append_action_history(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        *,
        turn_id: str,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        action: dict[str, Any],
        reply_text: str | None = None,
    ) -> None:
        if len(images) != len(image_roles):
            raise ValueError("images and image_roles must have the same length")
        retained_media = [
            (image, role)
            for image, role in zip(images, image_roles, strict=True)
            if role != IMAGE_ROLE_AVATAR_STATE
        ]
        retained_images = [image for image, _ in retained_media]
        retained_image_roles = [role for _, role in retained_media]
        candidate_id = str(action["candidate_id"])
        candidate = self.candidate_by_id[candidate_id]
        action_id = str(action["action_id"])
        action_state = self._model_action_history_record(
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            category_id=candidate.category_id,
            source_label=candidate.source_label,
            short_definition=candidate.short_definition,
            execute=action_id != "no_action",
        )

        if turn_origin == TURN_ORIGIN_USER:
            assistant_content = (
                f"{reply_text}\n{action_state}" if reply_text else action_state
            )
            messages = [
                {
                    "role": "user",
                    "content": self._current_user_content(
                        audios, retained_images, text
                    ),
                },
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": self._current_character_content(
                        audios,
                        retained_images,
                        reply_text or text,
                        action_state,
                    ),
                }
            ]

        history_turn = ActionHistoryTurn(
            turn_id=turn_id,
            turn_origin=turn_origin,
            text_role=text_role,
            messages=messages,
            audios=list(audios),
            images=list(retained_images),
        )
        self.history_turns.append(history_turn)
        self.history.extend(messages)
        self.history_audios.extend(audios)
        self.history_images.extend(retained_images)
        self.history_image_roles.extend(retained_image_roles)

    def _current_user_content(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
    ) -> Any:
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        if not parts:
            parts.append(
                {
                    "type": "text",
                    "text": self._prompt(
                        zh="本轮没有文本输入，请根据当前会话内容选择动作。",
                        en=(
                            "No text input is provided in this interaction. Select "
                            "an action from the current conversation context."
                        ),
                    ),
                }
            )
        return parts

    @staticmethod
    def _current_character_content(
        audios: list[str],
        images: list[str],
        text: str | None,
        action_state: str,
    ) -> Any:
        normalized_text = (
            text.strip() if isinstance(text, str) and text.strip() else None
        )
        if not audios and not images:
            return (
                f"{normalized_text}\n{action_state}"
                if normalized_text is not None
                else action_state
            )
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if normalized_text is not None:
            parts.append({"type": "text", "text": normalized_text})
        parts.append({"type": "text", "text": action_state})
        return parts

    def _no_action_candidate(self) -> SessionActionCandidate:
        for candidate in self.candidates:
            if candidate.action_id == "no_action":
                return candidate
        raise ValueError("session has no no_action candidate")

    def _fallback_categories(self) -> list[SessionActionCategory]:
        category_by_id = {
            category.category_id: category for category in self.categories
        }
        categories = [
            category_by_id[category_id]
            for category_id in self.fallback_category_ids
            if category_id in category_by_id
        ]
        if len(categories) != len(self.fallback_category_ids) or not categories:
            raise ValueError(
                "session has no valid fallback categories with executable actions"
            )
        return categories

    def _primary_fallback_category(self) -> SessionActionCategory:
        """Return the highest-priority fallback category configured by the client."""
        return self._fallback_categories()[0]

    def _default_fallback_candidate(self) -> SessionActionCandidate:
        """Return the stable executable action from the primary fallback category."""
        return self._primary_fallback_category().children[0]

    def _no_action_candidate_id(self) -> str:
        return self._no_action_candidate().candidate_id

    def _format_candidate_for_prompt(self, candidate: SessionActionCandidate) -> str:
        return self._prompt(
            zh=(
                f"candidate_id={candidate.candidate_id}｜动作={candidate.source_label}｜"
                f"说明={candidate.short_definition}"
            ),
            en=(
                f"candidate_id={candidate.candidate_id} | action={candidate.source_label} | "
                f"description={candidate.short_definition}"
            ),
        )

    def _child_candidates_for_categories(
        self, categories: list[SessionActionCategory]
    ) -> list[SessionActionCandidate]:
        return [child for category in categories for child in category.children]

    def _build_category_system_prompt(self) -> str:
        if self.global_action_catalog is not None:
            return self.global_action_catalog.category_system_prompt_for(self.locale)
        if self.language == "en":
            lines = [
                "You are a digital-character action category classifier. Select one category_id from the fixed category set.",
                ACTION_HISTORY_INSTRUCTION_EN,
                CATEGORY_CONTEXT_POLICY_EN,
            ]
            no_action_categories = [
                item.category_id
                for item in self.categories
                if any(child.action_id == "no_action" for child in item.children)
            ]
            if no_action_categories:
                lines.append(
                    "If no candidate action satisfies the input and state constraints, "
                    "use one of these default category_id values: "
                    + ", ".join(no_action_categories)
                    + "."
                )
            lines.append("Fixed category set:")
            lines.extend(
                f"category_id={item.category_id} | category={item.source_label} | description={item.short_definition}"
                for item in self.categories
            )
            lines.append(
                "Select the category_id that best matches the current input. Output "
                "exactly one category_id and stop immediately. Do not explain."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作类别识别器。请从固定类别集合中选择一个 category_id。",
            ACTION_HISTORY_INSTRUCTION,
            CATEGORY_CONTEXT_POLICY,
        ]
        no_action_categories = [
            item.category_id
            for item in self.categories
            if any(child.action_id == "no_action" for child in item.children)
        ]
        if no_action_categories:
            lines.append(
                "没有候选动作满足输入与状态约束时，可使用兜底 category_id="
                + ",".join(no_action_categories)
                + "。"
            )
        lines.append("固定类别集合如下：")
        # Category descriptions are opaque external metadata. Do not compress,
        # deduplicate, or rewrite them here; callers may optimize their wording
        # before session.start and the exact rendered catalog participates in
        # the catalog hash/prefix-cache identity.
        for item in self.categories:
            lines.append(
                f"category_id={item.category_id}｜类别={item.source_label}｜"
                f"说明={item.short_definition}"
            )
        lines.append(
            "请根据当前输入选择最匹配的 category_id；只输出一个 category_id，"
            "输出后立即结束，不要解释。"
        )
        return "\n".join(lines)

    def _build_child_system_prompt(
        self,
        category: SessionActionCategory | list[SessionActionCategory],
        candidates: list[SessionActionCandidate],
    ) -> str:
        categories = category if isinstance(category, list) else [category]
        if self.global_action_catalog is not None:
            if len(categories) != 1:
                raise ValueError(
                    "global Child prefixes require exactly one selected category"
                )
            return self.global_action_catalog.child_system_prompt_for(
                self.locale, categories[0].category_id
            )
        if self.language == "en":
            lines = [
                "You are a digital-character action classifier. Select one candidate_id from the following set.",
                ACTION_HISTORY_INSTRUCTION_EN,
            ]
            lines.extend(
                f"Selected category: category_id={selected.category_id} | category={selected.source_label} | description={selected.short_definition}"
                for selected in categories
            )
            lines.extend(self._format_candidate_for_prompt(item) for item in candidates)
            lines.append(
                "Select the candidate_id that best matches from the candidates in the "
                "selected category above. Do not introduce another category or an "
                "extra default action."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作识别器。请从以下集合中选择一个 candidate_id。",
            ACTION_HISTORY_INSTRUCTION,
        ]
        for selected in categories:
            lines.append(
                f"已选类别：category_id={selected.category_id}｜类别={selected.source_label}｜"
                f"说明={selected.short_definition}"
            )
        lines.extend(self._format_candidate_for_prompt(item) for item in candidates)
        lines.append(
            "只能从以上已选类别的候选动作中选择最匹配的 candidate_id；"
            "不得引入其他类别或系统兜底动作。"
        )
        return "\n".join(lines)

    def _category_whitelist_instruction(self) -> str:
        if self.global_action_catalog is None:
            return ""
        separator = self._prompt(zh="、", en=", ")
        allowed_ids = separator.join(
            category.category_id for category in self.categories
        )
        fallback_items = separator.join(
            self._prompt(
                zh=f"{category.category_id}（{category.source_label}）",
                en=f"{category.category_id} ({category.source_label})",
            )
            for category in self._fallback_categories()
        )
        if self.language == "en":
            return (
                "[Action categories allowed in this conversation]\n"
                f"Allowed real category_id values: {allowed_ids}\n"
                f"Select only one of these category_id values, or select {UNSUPPORTED_CATEGORY_SCORE_ID} "
                "under its defined conditions. Other categories in the fixed set are "
                "not available in this conversation.\n"
                "[Default action categories for this conversation]\n"
                f"In descending priority: {fallback_items}\n"
                "Use these categories only when the current input does not explicitly "
                "request a concrete action, such as ordinary dialogue, silent observation, "
                "low-disturbance situations, or natural idle behavior. When the current "
                "input explicitly requests an action whose semantic category is not allowed "
                f"in this conversation, return {UNSUPPORTED_CATEGORY_SCORE_ID}; do not use a "
                "default category as a supported substitute.\n"
            )
        return (
            "[本次会话允许选择的动作类别]\n"
            f"可用的真实 category_id：{allowed_ids}\n"
            f"只能选择以上 category_id，或按既定条件选择 {UNSUPPORTED_CATEGORY_SCORE_ID}；"
            "固定类别集合中的其他类别在本次会话中不可用。\n"
            "[本次会话默认动作类别]\n"
            f"按优先级从高到低为：{fallback_items}\n"
            "这些类别只用于当前输入没有明确要求具体动作的情况，例如普通对话、静默观察、"
            "低打扰或自然待机。当前输入明确要求动作，但该动作的语义类别不在本次会话允许"
            f"范围内时，必须返回 {UNSUPPORTED_CATEGORY_SCORE_ID}，不得把默认动作类别作为"
            "已支持该请求的替代类别。\n"
        )

    def _child_whitelist_instruction(
        self,
        category: SessionActionCategory,
        candidates: list[SessionActionCandidate],
    ) -> str:
        if self.global_action_catalog is None:
            return ""
        allowed_ids = self._prompt(zh="、", en=", ").join(
            item.candidate_id for item in candidates
        )
        if self.language == "en":
            return (
                "[Concrete actions allowed in this conversation]\n"
                f"Selected category_id={category.category_id}. Only these candidate_id "
                f"values may be selected: {allowed_ids}\n"
                + f"You may also return {UNSUPPORTED_CHILD_SCORE_ID}; it indicates that "
                "the concrete action is unsupported and is not executable. Even when the "
                "selected category is a default action category, an explicit action request "
                f"that none of the real candidates can fulfill must return {UNSUPPORTED_CHILD_SCORE_ID}. "
                "When there is no explicit action request, select an appropriate real "
                "candidate_id instead.\n"
                + "Other actions in this category that are not listed above cannot be "
                "selected in this conversation.\n"
            )
        return (
            "[本次会话允许选择的具体动作]\n"
            f"已选 category_id={category.category_id}。"
            f"只允许从以下 candidate_id 中选择：{allowed_ids}\n"
            + f"此外可以返回 {UNSUPPORTED_CHILD_SCORE_ID}；它只表示具体动作不支持，"
            "不是可执行动作。即使当前类别是默认动作类别，只要当前输入明确要求动作，且"
            f"真实候选都无法完成该请求，也必须返回 {UNSUPPORTED_CHILD_SCORE_ID}。当前输入"
            "没有明确动作请求时，应选择合适的真实 candidate_id。\n"
            + "该类别中未列出的其他动作在本次会话中不可选择。\n"
        )

    def _build_action_system_prompt(self) -> str:
        if self.language == "en":
            lines = [
                "You are a digital-character action classifier. Select one candidate_id from the fixed set for this conversation.",
                ACTION_HISTORY_INSTRUCTION_EN,
            ]
            lines.extend(
                self._format_candidate_for_prompt(item) for item in self.candidates
            )
            lines.append(
                "If no candidate satisfies the input and state constraints, or a "
                "conflict or meaningless repetition must be avoided, select the "
                f"default candidate_id={self._no_action_candidate_id()}."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作识别器。请从本次会话的固定集合中选择一个 candidate_id。",
            ACTION_HISTORY_INSTRUCTION,
        ]
        lines.extend(
            self._format_candidate_for_prompt(item) for item in self.candidates
        )
        lines.append(
            "没有候选动作满足输入与状态约束，或需要避免冲突、重复时，选择兜底 "
            f"candidate_id={self._no_action_candidate_id()}。"
        )
        return "\n".join(lines)

    async def _ack_media(
        self,
        turn: TurnBuffer,
        media_type: str,
        seq: int,
        duplicate: bool = False,
        image_role: str | None = None,
    ) -> None:
        payload = {
            "type": "input.ack",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "media_type": media_type,
            "seq": seq,
            "duplicate": duplicate,
        }
        if image_role is not None:
            if self.protocol_version is not None:
                payload["image_source"] = (
                    IMAGE_SOURCE_AVATAR_CURRENT
                    if image_role == IMAGE_ROLE_AVATAR_STATE
                    else image_role
                )
            else:
                payload["image_role"] = image_role
        await self.send(payload)

    def _require_started(self) -> None:
        if not self.started or self.session_id is None:
            raise ValueError("session.start must be sent first")

    @staticmethod
    def _parse_turn_semantics(
        event: dict[str, Any],
    ) -> tuple[
        Literal["user", "proactive"],
        Literal["user_input", "character_reply"],
        str | None,
    ]:
        turn_origin = event.get("turn_origin")
        text_role = event.get("text_role")
        expected_text_role = TURN_TEXT_ROLE_BY_ORIGIN.get(turn_origin)
        if expected_text_role is None:
            raise ValueError("turn_origin must be 'user' or 'proactive'")
        if text_role != expected_text_role:
            raise ValueError(
                f"text_role must be {expected_text_role!r} when "
                f"turn_origin is {turn_origin!r}"
            )
        trigger = event.get("trigger")
        if trigger is not None:
            if not isinstance(trigger, str) or not trigger.strip():
                raise ValueError("trigger must be a non-empty string or null")
            trigger = trigger.strip()
        if turn_origin == TURN_ORIGIN_USER and trigger is not None:
            raise ValueError("trigger is only supported for proactive turns")
        return turn_origin, text_role, trigger

    def _require_turn(self, event: dict[str, Any]) -> TurnBuffer:
        self._require_started()
        if self.active_turn is None:
            raise ValueError(
                "turn already committed or no active turn exists; send turn.start first"
            )
        turn_id = event.get("turn_id")
        if turn_id != self.active_turn.turn_id:
            raise ValueError("turn_id does not match the active turn")
        return self.active_turn

    def _require_collecting_turn(self, event: dict[str, Any]) -> TurnBuffer:
        turn = self._require_turn(event)
        if turn.phase != TURN_PHASE_COLLECTING:
            raise ValueError("turn already committed")
        return turn

    @staticmethod
    def _event_context_id(payload: dict[str, Any], field: str) -> str | None:
        value = payload.get(field)
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _classify_error(payload: dict[str, Any], exc: Exception) -> str:
        message = str(exc).lower()
        event_type = payload.get("type")
        if "protocol_version" in message:
            return "unsupported_protocol_version"
        if event_type == "session.start" and (
            "unsupported outputs" in message
            or "unsupported output modalities" in message
        ):
            return "unsupported_output"
        if "unsupported fields" in message or "missing required fields" in message:
            return "invalid_event_field"
        if (
            "turn_origin" in message
            or "text_role" in message
            or "proactive turn" in message
            or "user_input" in message
        ):
            return "invalid_turn_semantics"
        if "session_id is already active" in message:
            return "duplicate_session_id"
        if "another turn is already active" in message:
            return "duplicate_active_turn"
        if "turn already committed" in message:
            return "turn_already_committed"
        if "turn_id" in message:
            return "invalid_turn_id"
        if "seq" in message or "sequence" in message:
            return "invalid_sequence"
        if "audio" in message or "pcm16" in message:
            return "invalid_audio"
        if "image" in message or "mime_type" in message:
            return "invalid_image"
        if "session.start can only" in message or event_type == "session.start":
            return "session_candidate_invalid"
        if "action_candidates" in message or "candidate" in message:
            return "session_candidate_invalid"
        if (
            event_type
            in {
                "turn.commit",
                "input.audio.append",
                "input.image.append",
                "input.text.set",
                "turn.cancel",
            }
            and "turn.start" in message
        ):
            return "turn_already_committed"
        if "session.start" in message:
            return "session_not_started"
        return "invalid_event"

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _normalize_modalities(value: Any) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_MODALITIES
        if not isinstance(value, list) or not value:
            raise ValueError("modalities must be a non-empty list")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("modalities entries must be strings")
        if len(set(value)) != len(value):
            raise ValueError("modalities must not contain duplicates")
        unsupported = sorted(set(value) - SUPPORTED_MODALITIES)
        if unsupported:
            raise ValueError("unsupported output modalities: " + ", ".join(unsupported))
        return tuple(item for item in DEFAULT_MODALITIES if item in value)

    @staticmethod
    def _normalize_outputs(value: Any) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_MODALITIES
        if not isinstance(value, list) or not value:
            raise ValueError("outputs must be a non-empty list")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("outputs entries must be strings")
        if len(set(value)) != len(value):
            raise ValueError("outputs must not contain duplicates")
        unsupported = sorted(set(value) - SUPPORTED_MODALITIES)
        if unsupported:
            raise ValueError("unsupported outputs: " + ", ".join(unsupported))
        return tuple(item for item in DEFAULT_MODALITIES if item in value)

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._send_unlocked(payload)

    async def _send_unlocked(self, payload: dict[str, Any]) -> None:
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
            await self.websocket.send_text(encoded)
            turn_id = payload.get("turn_id")
            trace_id = None
            if self.active_turn is not None and turn_id == self.active_turn.turn_id:
                trace_id = self.active_turn.trace_id
            emit_structured_log(
                "protocol",
                "ws_event_sent",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=trace_id,
                ws_event_type=payload.get("type"),
                payload_bytes=len(encoded.encode("utf-8")),
            )
        except (OSError, WebSocketDisconnect):
            self.closed = True
            logger.info(
                "[SESSION_ACTION_REALTIME] client disconnected during send "
                "session_id=%s event=%s",
                self.session_id,
                payload.get("type"),
            )
        except RuntimeError as exc:
            if "close message has been sent" not in str(exc):
                raise
            self.closed = True
            logger.info(
                "[SESSION_ACTION_REALTIME] websocket already closed during send "
                "session_id=%s event=%s",
                self.session_id,
                payload.get("type"),
            )

    async def send_error(
        self,
        type_: str,
        code: str,
        message: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "error",
            "error": {"type": type_, "code": code, "message": message},
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if turn_id is not None:
            payload["turn_id"] = turn_id
        emit_structured_log(
            "error",
            "websocket_error_sent",
            level="error",
            session_id=session_id or self.session_id,
            turn_id=turn_id,
            error_type=type_,
            error_code=code,
            error_message=message,
        )
        await self.send(payload)


class MultimodalSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        action_selection_mode: str | None = None,
        global_action_catalog: GlobalActionCatalog | None = None,
        global_action_prewarm: GlobalActionCatalogPrewarmStatus | None = None,
        allow_unregistered_protocol_actions: bool = False,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(
            action_selection_mode
        )
        self.action_micro_batch_size = normalize_action_micro_batch_size()
        self.action_category_top_k = normalize_action_category_top_k()
        self.global_action_catalog = global_action_catalog
        self.allow_unregistered_protocol_actions = allow_unregistered_protocol_actions
        self.global_action_prewarm = (
            global_action_prewarm or GlobalActionCatalogPrewarmStatus.not_run()
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] action_selection_mode=%s",
            self.action_selection_mode,
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] action_micro_batch_size=%s",
            self.action_micro_batch_size,
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] action_category_top_k=%s",
            self.action_category_top_k,
        )
        self.sessions: dict[str, MultimodalSession] = {}
        self.resource_sample_requester: Callable[..., bool] | None = None

    def set_resource_sample_requester(
        self,
        requester: Callable[..., bool] | None,
    ) -> None:
        self.resource_sample_requester = requester

    def create(self, websocket: WebSocket) -> MultimodalSession:
        return MultimodalSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
            action_selection_mode=self.action_selection_mode,
            action_micro_batch_size=self.action_micro_batch_size,
            action_category_top_k=self.action_category_top_k,
            global_action_catalog=self.global_action_catalog,
            global_action_prewarm=self.global_action_prewarm,
            allow_unregistered_protocol_actions=(
                self.allow_unregistered_protocol_actions
            ),
            claim_session=self.claim,
            release_session=self.release,
            request_resource_sample=self.resource_sample_requester,
        )

    def claim(self, session_id: str, session: MultimodalSession) -> None:
        if session_id in self.sessions:
            raise ValueError(f"session_id is already active: {session_id}")
        self.sessions[session_id] = session

    def release(self, session_id: str, session: MultimodalSession) -> None:
        if self.sessions.get(session_id) is session:
            del self.sessions[session_id]

    def active_sessions(self) -> list[str]:
        return list(self.sessions)

    def load_snapshot(self) -> dict[str, Any]:
        """Return aggregate session load without exposing session identifiers."""

        turn_phase_counts: dict[str, int] = {}
        modality_counts: dict[str, int] = {}
        active_turn_count = 0
        started_session_count = 0
        action_history_turn_count = 0
        reply_history_turn_count = 0
        for session in self.sessions.values():
            if session.started:
                started_session_count += 1
            modality_key = "+".join(session.modalities) or "not_started"
            modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1
            action_history_turn_count += len(session.history_turns)
            reply_history_turn_count += len(session.reply_history_turns)
            turn = session.active_turn
            if turn is not None:
                active_turn_count += 1
                turn_phase_counts[turn.phase] = turn_phase_counts.get(turn.phase, 0) + 1
        return {
            "active_session_count": len(self.sessions),
            "started_session_count": started_session_count,
            "active_turn_count": active_turn_count,
            "turn_phase_counts": turn_phase_counts,
            "modality_counts": modality_counts,
            "retained_action_history_turn_count": action_history_turn_count,
            "retained_reply_history_turn_count": reply_history_turn_count,
            "global_action_catalog": {
                "configured": self.global_action_catalog is not None,
                "catalog_hash": (
                    self.global_action_catalog.catalog_hash
                    if self.global_action_catalog is not None
                    else None
                ),
                "category_count": (
                    len(self.global_action_catalog.categories)
                    if self.global_action_catalog is not None
                    else 0
                ),
                "candidate_count": (
                    self.global_action_catalog.candidate_count
                    if self.global_action_catalog is not None
                    else 0
                ),
                "category_prefix_ready": self.global_action_prewarm.category_ready,
                "child_prefix_ready_count": len(
                    self.global_action_prewarm.ready_child_category_ids
                ),
                "child_prefix_failed_count": len(
                    self.global_action_prewarm.failed_child_category_ids
                ),
                "locale_prefix_statuses": {
                    locale: {
                        "category_prefix_ready": status.category_ready,
                        "child_prefix_ready_count": len(
                            status.ready_child_category_ids
                        ),
                        "child_prefix_failed_count": len(
                            status.failed_child_category_ids
                        ),
                    }
                    for locale, status in self.global_action_prewarm.by_locale.items()
                },
            },
        }
