"""Deterministic facial-expression and body-action support fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.serve.realtime.performance.models import PerformanceDecision


@dataclass(frozen=True, slots=True)
class PerformanceFusionResult:
    action: dict[str, Any] | None
    action_error: Exception | None
    expression: dict[str, Any] | None


def fuse_performance_decision(
    *,
    action: dict[str, Any] | None,
    action_error: Exception | None,
    performance: PerformanceDecision,
    expression_enabled: bool,
) -> PerformanceFusionResult:
    """Apply channel ownership and atomicity after parallel inference finishes."""
    action = dict(action) if action is not None else None
    # Audio-only callers may still run this branch to derive the internal TTS
    # instruction. Without expression output, expression scope must not alter
    # the independently inferred body-action support result.
    if not expression_enabled:
        return PerformanceFusionResult(
            action=action,
            action_error=action_error,
            expression=None,
        )
    body_unavailable = bool(
        action is None
        or action_error is not None
        or action.get("support_status") == "unsupported"
    )
    if performance.request_scope == "expression_only":
        # A body inference failure is irrelevant when no body action was asked for.
        action_error = None
        if (
            performance.expression is not None
            and not performance.expression_unsupported
        ):
            action = {
                "candidate_id": "expression_only",
                "action_id": "no_action",
                "execute": False,
                "support_status": "not_required",
                "fallback_applied": False,
                "reason_code": "expression_only",
            }
        else:
            action = {
                "candidate_id": "expression_unsupported",
                "action_id": "no_action",
                "execute": False,
                "support_status": "unsupported",
                "fallback_applied": False,
                "reason_code": "expression_unsupported",
            }
    elif performance.request_scope == "both" and (
        body_unavailable
        or performance.expression is None
        or performance.expression_unsupported
    ):
        if action is None:
            action = {
                "candidate_id": "combined_request_unsupported",
                "action_id": "no_action",
                "fallback_applied": False,
            }
        action["execute"] = False
        action["support_status"] = "unsupported"
        action["reason_code"] = "combined_request_atomic_failure"

    expression_allowed = bool(
        expression_enabled
        and performance.expression is not None
        and not performance.expression_unsupported
        and (
            performance.request_scope in {"none", "expression_only"}
            or (
                performance.request_scope in {"body_only", "both"}
                and action is not None
                and action.get("support_status") != "unsupported"
            )
        )
    )
    expression = (
        dict(performance.expression)
        if expression_allowed and performance.expression is not None
        else None
    )
    return PerformanceFusionResult(
        action=action,
        action_error=action_error,
        expression=expression,
    )
