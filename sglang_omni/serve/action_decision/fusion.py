"""Deterministic fusion after two independent channel decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ActionDecision, DecisionChannel


@dataclass(frozen=True)
class FusedActionDecision:
    body: ActionDecision
    expression: ActionDecision
    expression_publication: str
    reason_code: str

    def to_dict(self) -> dict[str, object]:
        return {
            "body": self.body.to_dict(),
            "expression": self.expression.to_dict(),
            "expression_publication": self.expression_publication,
            "reason_code": self.reason_code,
        }


def fuse_action_decisions(
    body: ActionDecision,
    expression: ActionDecision,
    *,
    expression_priority: str = "normal",
) -> FusedActionDecision:
    if body.channel is not DecisionChannel.BODY:
        raise ValueError("body decision has the wrong channel")
    if expression.channel is not DecisionChannel.EXPRESSION:
        raise ValueError("expression decision has the wrong channel")
    expression_applies = expression.outcome == "apply"
    if "face" in body.occupies_channels and expression_applies:
        if expression_priority == "explicit":
            return FusedActionDecision(
                body, expression, "delay", "body_occupies_expression_channel"
            )
        return FusedActionDecision(
            body, expression, "suppress", "body_occupies_expression_channel"
        )
    return FusedActionDecision(body, expression, "publish", "channels_independent")
