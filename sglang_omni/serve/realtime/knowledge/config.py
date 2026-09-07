from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class RealtimeKnowledgeConfig:
    enabled: bool = False
    url: str = ""
    service_token: str | None = None
    default_tenant_id: str | None = None
    connect_timeout_seconds: float = 1.0
    turn_timeout_ms: int = 1_200
    max_concurrency: int = 128
    max_context_chars: int = 6_000
    max_evidence: int = 4
    speculative_enabled: bool = False
    commit_reserve_ms: int = 200
    commit_recovery_timeout_ms: int = 500

    @classmethod
    def from_env(cls) -> "RealtimeKnowledgeConfig":
        url = os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_URL", "").rstrip("/")
        return cls(
            enabled=_env_bool(
                "SGLANG_OMNI_REALTIME_KNOWLEDGE_ENABLED", bool(url)
            ),
            url=url,
            service_token=os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_TOKEN"),
            default_tenant_id=os.getenv(
                "SGLANG_OMNI_REALTIME_KNOWLEDGE_DEFAULT_TENANT"
            ),
            connect_timeout_seconds=float(
                os.getenv(
                    "SGLANG_OMNI_REALTIME_KNOWLEDGE_CONNECT_TIMEOUT_SECONDS", "1.0"
                )
            ),
            turn_timeout_ms=int(
                os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_TURN_TIMEOUT_MS", "1200")
            ),
            max_concurrency=int(
                os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_CONCURRENCY", "128")
            ),
            max_context_chars=int(
                os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_CONTEXT_CHARS", "6000")
            ),
            max_evidence=int(
                os.getenv("SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_EVIDENCE", "4")
            ),
            speculative_enabled=_env_bool(
                "SGLANG_OMNI_REALTIME_KNOWLEDGE_SPECULATIVE_ENABLED", False
            ),
            commit_reserve_ms=int(
                os.getenv(
                    "SGLANG_OMNI_REALTIME_KNOWLEDGE_COMMIT_RESERVE_MS", "200"
                )
            ),
            commit_recovery_timeout_ms=int(
                os.getenv(
                    "SGLANG_OMNI_REALTIME_KNOWLEDGE_COMMIT_RECOVERY_TIMEOUT_MS",
                    "500",
                )
            ),
        )

    def validate(self) -> None:
        if self.enabled and not self.url:
            raise ValueError("realtime knowledge is enabled but URL is empty")
        if self.turn_timeout_ms < 10:
            raise ValueError("realtime knowledge turn timeout must be at least 10ms")
        if self.max_concurrency < 1:
            raise ValueError("realtime knowledge max concurrency must be positive")
        if self.max_context_chars < 256:
            raise ValueError("realtime knowledge context limit must be at least 256")
        if self.max_evidence < 1:
            raise ValueError("realtime knowledge evidence limit must be positive")
        if self.speculative_enabled and not (
            10 <= self.commit_reserve_ms < self.turn_timeout_ms
        ):
            raise ValueError(
                "knowledge commit reserve must be at least 10ms and smaller than "
                "the turn timeout"
            )
        if self.commit_recovery_timeout_ms < 10:
            raise ValueError("knowledge commit recovery timeout must be at least 10ms")
