"""Shared protocol constants and pure normalization helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from sglang_omni.models.qwen3_omni.action_scoring import MAX_MICRO_BATCH_SIZE
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    DEFAULT_ACTION_PROMPT_LOCALE,
)

MAX_ACTION_CANDIDATES = 700
MAX_ACTION_CATEGORIES = 128
MAX_ACTION_CHILDREN_PER_CATEGORY = 128
SYSTEM_REPLY_PREFIX_MAX_CHARS = 256
SYSTEM_REPLY_SENTENCE_WAIT_S = 0.1
SYSTEM_REPLY_SENTENCE_END_RE = re.compile(r"[。！？!?；;\.\n]")
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
MAX_CHARACTER_PROFILE_ROLE_CHARS = 5_000
ACTION_PERSONA_FIELDS = (
    "gender_expression",
    "visual_style",
    "role",
    "personality",
)
CHARACTER_PROFILE_FIELDS = ACTION_PERSONA_FIELDS + (
    "visual_behavior_preferences",
)
DEFAULT_REPLY_MAX_NEW_TOKENS = 512
DEFAULT_REPLY_TEMPERATURE = 0.4
PURE_ACTION_REPLY_MAX_NEW_TOKENS = 48
PURE_ACTION_REPLY_MAX_CHARS = 18
PURE_ACTION_ROUTE_AMBIGUITY_MARGIN = 0.5
PURE_ACTION_REPLY_VALIDATION_MIN_MARGIN = 0.1
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_TURN = 64
MAX_AUDIO_CHUNKS_PER_TURN = 4096
MAX_IMAGE_PREPROCESS_TASKS_PER_TURN = 8
MAX_PREPARED_IMAGE_BYTES_PER_FRAME = 32 * 1024 * 1024
MAX_PREPARED_IMAGE_BYTES_PER_TURN = 64 * 1024 * 1024
# Reply generation retains only a narrow recent window. The current user turn
# remains authoritative; older turns are context for explicit references, not
# examples whose topic or wording should be continued by default. Action
# scoring omits general history; its one reference-only action anchor does not
# use this limit.
MAX_REPLY_HISTORY_TURNS = 2
REPLY_HISTORY_CURRENT_ONLY = "CURRENT_ONLY"
REPLY_HISTORY_REQUIRED = "HISTORY_REQUIRED"
REPLY_MODE_LANGUAGE_REQUIRED = "LANGUAGE_REQUIRED"
REPLY_MODE_PURE_ACTION = "PURE_ACTION"
REPLY_HISTORY_ROUTE_STAGE = "reply_history_route"
REPLY_SPEECH_MODE_STAGE = "reply_speech_mode"
PURE_ACTION_REPLY_VALIDATION_STAGE = "pure_action_reply_validation"
REPLY_HISTORY_ROUTE_TIMEOUT_ENV = "SGLANG_OMNI_REPLY_HISTORY_ROUTE_TIMEOUT_S"
DEFAULT_REPLY_HISTORY_ROUTE_TIMEOUT_S = 0.5
PURE_ACTION_REPLY_VALIDATION_TIMEOUT_ENV = (
    "SGLANG_OMNI_PURE_ACTION_REPLY_VALIDATION_TIMEOUT_S"
)
DEFAULT_PURE_ACTION_REPLY_VALIDATION_TIMEOUT_S = 0.5
MAX_ACTION_CURRENT_IMAGES = 8
# Legacy diagnostic-only bounds used when explicitly constructing an action
# context with history. Production action scoring keeps include_history=False.
MAX_ACTION_HISTORY_TURNS = 2
MAX_ACTION_HISTORY_AUDIOS = 2
MAX_ACTION_HISTORY_IMAGES = 8
# Keep lightweight action facts outside multimodal history. Reply generation
# never receives them; action scoring may receive only the latest user-triggered
# action as a reference-only anchor for explicit cross-turn references.
MAX_EXECUTED_ACTION_HISTORY_TURNS = 64
FULL_INSTRUCTIONS_LOG_ENV = "SGLANG_OMNI_REALTIME_LOG_FULL_INSTRUCTIONS"
ACTION_READY_TTS_DECOUPLED_ENV = (
    "SGLANG_OMNI_REALTIME_ACTION_READY_TTS_DECOUPLED"
)
ROUTE_ACTION_PARALLEL_ENV = "SGLANG_OMNI_REALTIME_ROUTE_ACTION_PARALLEL"
ACTION_SELECTION_MODE_ENV = "SGLANG_OMNI_ACTION_SELECTION_MODE"
ACTION_SELECTION_MODE_HIERARCHICAL = "hierarchical"
ACTION_SELECTION_MODE_FLAT_CHILDREN = "flat_children"
ACTION_MICRO_BATCH_SIZE_ENV = "SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE"
DEFAULT_ACTION_MICRO_BATCH_SIZE = 64
ACTION_CATEGORY_TOP_K_ENV = "SGLANG_OMNI_ACTION_CATEGORY_TOP_K"
DEFAULT_ACTION_CATEGORY_TOP_K = 2
MAX_ACTION_CATEGORY_TOP_K = 3
TURN_ORIGIN_USER = "user"
TURN_ORIGIN_PROACTIVE = "proactive"
ACTION_FINISHED_TRIGGER = "action_finished"
PODCAST_REPLY_CONTEXT_MARKER = "INTERNAL PODCAST CONTEXT"
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

    The default is Top-2 so that a close or semantically overlapping runner-up
    category can still contribute the correct concrete action. Values greater
    than one deliberately score the children of the best categories together,
    which is an accuracy/latency trade-off for ambiguous category boundaries.
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
