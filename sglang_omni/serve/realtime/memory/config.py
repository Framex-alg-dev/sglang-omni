"""Environment-backed configuration for session-scoped memory."""

from __future__ import annotations

import os
from dataclasses import dataclass

SESSION_MEMORY_ENABLED_ENV = "SGLANG_OMNI_SESSION_MEMORY_ENABLED"
SESSION_MEMORY_WRITE_ENABLED_ENV = "SGLANG_OMNI_SESSION_MEMORY_WRITE_ENABLED"
SESSION_MEMORY_READ_ENABLED_ENV = "SGLANG_OMNI_SESSION_MEMORY_READ_ENABLED"
SESSION_MEMORY_TIMEOUT_ENV = "SGLANG_OMNI_SESSION_MEMORY_TIMEOUT_S"
SESSION_MEMORY_CATCHUP_TIMEOUT_ENV = "SGLANG_OMNI_SESSION_MEMORY_CATCHUP_TIMEOUT_S"
SESSION_MEMORY_MAX_PENDING_TURNS_ENV = "SGLANG_OMNI_SESSION_MEMORY_MAX_PENDING_TURNS"
SESSION_MEMORY_MAX_QUEUED_SESSIONS_ENV = "SGLANG_OMNI_SESSION_MEMORY_MAX_QUEUED_SESSIONS"
SESSION_MEMORY_MAX_CONCURRENT_EXTRACTIONS_ENV = (
    "SGLANG_OMNI_SESSION_MEMORY_MAX_CONCURRENT_EXTRACTIONS"
)
SESSION_MEMORY_EXTRACTION_TASK = "session_memory_extract"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


@dataclass(frozen=True, slots=True)
class SessionMemoryConfig:
    enabled: bool = True
    write_enabled: bool = True
    read_enabled: bool = True
    extraction_timeout_s: float = 8.0
    catchup_timeout_s: float = 0.05
    batch_turns: int = 3
    max_pending_turns: int = 12
    max_operations_per_turn: int = 3
    max_active_claims: int = 16
    max_inactive_claims: int = 32
    max_injected_claims: int = 8
    max_episodes: int = 64
    max_injected_episodes: int = 4
    max_artifacts: int = 8
    max_injected_artifacts: int = 2
    max_context_chars: int = 4096
    max_input_text_chars: int = 4096
    max_claim_content_chars: int = 256
    max_summary_chars: int = 320
    max_artifact_content_chars: int = 4096
    max_new_tokens: int = 512
    max_retries: int = 1
    max_queued_sessions: int = 256
    max_concurrent_extractions: int = 1

    @classmethod
    def from_env(cls) -> "SessionMemoryConfig":
        enabled = _env_bool(SESSION_MEMORY_ENABLED_ENV, True)
        return cls(
            enabled=enabled,
            write_enabled=enabled and _env_bool(SESSION_MEMORY_WRITE_ENABLED_ENV, True),
            read_enabled=enabled and _env_bool(SESSION_MEMORY_READ_ENABLED_ENV, True),
            extraction_timeout_s=_env_float(SESSION_MEMORY_TIMEOUT_ENV, 8.0, minimum=0.1),
            catchup_timeout_s=_env_float(
                SESSION_MEMORY_CATCHUP_TIMEOUT_ENV, 0.05, minimum=0.0
            ),
            max_pending_turns=_env_int(
                SESSION_MEMORY_MAX_PENDING_TURNS_ENV, 12, minimum=3, maximum=128
            ),
            max_queued_sessions=_env_int(
                SESSION_MEMORY_MAX_QUEUED_SESSIONS_ENV, 256, minimum=1, maximum=4096
            ),
            max_concurrent_extractions=_env_int(
                SESSION_MEMORY_MAX_CONCURRENT_EXTRACTIONS_ENV,
                1,
                minimum=1,
                maximum=4,
            ),
        )

