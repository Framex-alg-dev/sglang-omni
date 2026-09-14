"""Typed outputs for the per-turn performance-control branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PerformanceRequestScope = Literal[
    "none",
    "expression_only",
    "body_only",
    "both",
]


@dataclass(frozen=True, slots=True)
class PerformanceDecision:
    request_scope: PerformanceRequestScope
    expression: dict[str, Any] | None
    expression_unsupported: bool
    tts_instruction: str
    elapsed_ms: float

