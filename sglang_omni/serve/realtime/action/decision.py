"""Grouped, fixed-label intent decisions scored with the action suffix batch.

The labels in this module are deliberately not executable action candidates.
They share the action request's multimodal prefix and physical suffix batch, but
are normalized only against labels in the same decision group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    CandidateScore,
)


BODY_LABELS = {
    "IB0": "none",
    "IB1": "perform",
    "IB2": "prohibit",
    "IB3": "capability_query",
}
FACE_LABELS = {"IF0": "none", "IF1": "perform"}
REACTION_LABELS = {
    "IR0": "none",
    "IR1": "greeting",
    "IR2": "farewell",
    "IR3": "thanks",
    "IR4": "congratulation",
    "IR5": "affection",
}
VISUAL_LABELS = {f"IV{index:02d}": f"V{index:02d}" for index in range(12)}

ACTION_DECISION_LABELS = frozenset(
    (*BODY_LABELS, *FACE_LABELS, *REACTION_LABELS, *VISUAL_LABELS)
)


def _candidate_blueprints(groups: tuple[dict[str, str], ...]) -> tuple[ActionScoreCandidate, ...]:
    return tuple(
        ActionScoreCandidate(candidate_id=label, suffix=label, action_id=value)
        for group in groups
        for label, value in group.items()
    )


_CORE_CANDIDATE_BLUEPRINTS = _candidate_blueprints(
    (BODY_LABELS, FACE_LABELS, REACTION_LABELS)
)
_VISUAL_CANDIDATE_BLUEPRINTS = _candidate_blueprints((VISUAL_LABELS,))


def action_decision_candidates(*, include_visual: bool = False) -> list[ActionScoreCandidate]:
    """Return real suffix requests; no padding candidates are ever emitted."""

    return list(
        _CORE_CANDIDATE_BLUEPRINTS
        + (_VISUAL_CANDIDATE_BLUEPRINTS if include_visual else ())
    )


def action_decision_prompt(*, include_visual: bool = False, english: bool = False) -> str:
    """Describe independent decision groups embedded in the suffix batch."""

    if english:
        prompt = """
[Independent action-safety decision labels]
The IB/IF/IR labels below are independent classification groups. Scores are
compared only within the same prefix. They are not executable candidate_id
values and must not compete with concrete actions.
IB0=no body request; IB1=perform a body action now; IB2=explicitly prohibit a
body action; IB3=ask whether the character can perform an action.
IF0=no requested facial expression; IF1=perform a requested facial expression.
IR0=no direct social reaction; IR1=greeting; IR2=farewell; IR3=thanks;
IR4=congratulation; IR5=affection. Explicit body requests, prohibitions,
capability questions, quotations, and ordinary questions use IR0.
""".strip()
    else:
        prompt = """
[独立的动作安全决策标签]
以下 IB/IF/IR 是互相独立的分类组；只在同前缀标签内比较分数。它们不是可执行
candidate_id，也不得与具体动作竞争。
IB0=没有身体动作请求；IB1=要求现在执行身体动作；IB2=明确禁止身体动作；
IB3=询问数字人是否能够执行动作。
IF0=没有要求脸部表情；IF1=要求执行脸部表情。
IR0=没有直接社交反应；IR1=问候；IR2=道别；IR3=感谢；IR4=祝贺；IR5=亲昵。
明确动作、禁止、能力询问、引用朗读和普通问答均选择 IR0。
""".strip()
    if not include_visual:
        return prompt
    visual = """
IV00=非视觉模仿；IV01=模仿手势；IV02=模仿表情；IV03=模仿头部或视线；
IV04=模仿手臂；IV05=模仿肩膀或躯干；IV06=模仿腿脚；IV07=模仿全身；
IV08=模仿姿势；IV09=模仿物品交互；IV10=模仿屏幕交互；
IV11=观察、计算或推理画面后用手势表达答案。
普通问句、识别描述、能力询问、禁止、明确同时说话、无需看图的具体动作均选 IV00。
""".strip()
    if english:
        visual = """
IV00=no visual imitation; IV01=imitate a hand gesture; IV02=imitate a facial
expression; IV03=head/gaze; IV04=arm; IV05=shoulder/torso; IV06=leg/foot;
IV07=full body; IV08=pose; IV09=object interaction; IV10=screen interaction;
IV11=reason over the image and express the derived answer with a gesture.
Questions, descriptions, capability queries, prohibitions, explicit concurrent
speech, and named actions not requiring the image use IV00.
""".strip()
    return prompt + "\n" + visual


@dataclass(frozen=True, slots=True)
class DecisionGroupResult:
    winner: str
    value: str
    margin: float | None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionDecision:
    body_mode: str
    face_mode: str
    reaction_type: str
    visual_scope: str = ""
    body_confident: bool = False
    face_confident: bool = False
    reaction_confident: bool = False
    visual_confident: bool = True
    body_gate_confident: bool = False
    body_gate_margin: float | None = None
    # Retained as an all-groups diagnostic for compatibility. Body delivery
    # must use body_gate_confident/body_gate_margin instead.
    confident: bool = False
    min_margin: float | None = None
    groups: dict[str, DecisionGroupResult] = field(default_factory=dict)

    @property
    def allows_body(self) -> bool:
        return (
            self.body_gate_confident
            and self.visual_scope != "V11"
            and self.body_mode not in {"prohibit", "capability_query"}
            and (self.body_mode == "perform" or self.reaction_type != "none")
        )


def _rank_group(
    score_by_id: dict[str, float], labels: dict[str, str]
) -> DecisionGroupResult:
    present = [(label, score_by_id[label]) for label in labels if label in score_by_id]
    if len(present) != len(labels):
        missing = sorted(set(labels) - set(score_by_id))
        raise ValueError(f"missing action-decision labels: {missing}")
    ranked = sorted(present, key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    return DecisionGroupResult(
        winner=ranked[0][0],
        value=labels[ranked[0][0]],
        margin=margin,
        scores=dict(present),
    )


def aggregate_action_decision(
    scores: Iterable[CandidateScore],
    *,
    include_visual: bool = False,
    min_margin: float = 0.0,
) -> ActionDecision:
    score_by_id = {
        score.candidate_id: float(score.mean_logprob)
        for score in scores
        if score.candidate_id in ACTION_DECISION_LABELS
    }
    groups = {
        "body": _rank_group(score_by_id, BODY_LABELS),
        "face": _rank_group(score_by_id, FACE_LABELS),
        "reaction": _rank_group(score_by_id, REACTION_LABELS),
    }
    if include_visual:
        groups["visual"] = _rank_group(score_by_id, VISUAL_LABELS)

    def margin_is_confident(result: DecisionGroupResult) -> bool:
        return result.margin is not None and result.margin >= min_margin

    body_confident = margin_is_confident(groups["body"])
    face_confident = margin_is_confident(groups["face"])
    reaction_confident = margin_is_confident(groups["reaction"])
    visual = groups.get("visual")
    visual_confident = visual is None or margin_is_confident(visual)

    # Body delivery is gated only by groups that participate in the selected
    # delivery path. In particular, an ambiguous face decision must not block
    # an otherwise clear body action.
    body_gate_margins: list[float | None] = []
    if groups["body"].value == "perform":
        body_gate_margins.append(groups["body"].margin)
    elif groups["reaction"].value != "none":
        # A reaction may synthesize a body action, so require both a confident
        # reaction and a confident body safety classification.
        body_gate_margins.extend(
            (groups["body"].margin, groups["reaction"].margin)
        )
    if body_gate_margins and visual is not None:
        body_gate_margins.append(visual.margin)
    body_gate_margin = (
        min(margin for margin in body_gate_margins if margin is not None)
        if body_gate_margins and all(
            margin is not None for margin in body_gate_margins
        )
        else None
    )
    body_gate_confident = bool(
        body_gate_margin is not None and body_gate_margin >= min_margin
    )

    margins = [
        result.margin for result in groups.values() if result.margin is not None
    ]
    smallest = min(margins) if margins else None
    return ActionDecision(
        body_mode=groups["body"].value,
        face_mode=groups["face"].value,
        reaction_type=groups["reaction"].value,
        visual_scope=groups.get(
            "visual", DecisionGroupResult("", "", None)
        ).value,
        body_confident=body_confident,
        face_confident=face_confident,
        reaction_confident=reaction_confident,
        visual_confident=visual_confident,
        body_gate_confident=body_gate_confident,
        body_gate_margin=body_gate_margin,
        confident=bool(smallest is not None and smallest >= min_margin),
        min_margin=smallest,
        groups=groups,
    )


def decision_as_dict(decision: ActionDecision) -> dict[str, Any]:
    return {
        "body_mode": decision.body_mode,
        "face_mode": decision.face_mode,
        "reaction_type": decision.reaction_type,
        "visual_scope": decision.visual_scope,
        "body_confident": decision.body_confident,
        "face_confident": decision.face_confident,
        "reaction_confident": decision.reaction_confident,
        "visual_confident": decision.visual_confident,
        "body_gate_confident": decision.body_gate_confident,
        "body_gate_margin": decision.body_gate_margin,
        "confident": decision.confident,
        "min_margin": decision.min_margin,
        "groups": {
            name: {
                "winner": result.winner,
                "value": result.value,
                "margin": result.margin,
                "scores": dict(result.scores),
            }
            for name, result in decision.groups.items()
        },
    }
