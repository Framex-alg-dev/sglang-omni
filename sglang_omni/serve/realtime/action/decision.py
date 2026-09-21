"""Grouped, fixed-label intent decisions scored with the action suffix batch.

The labels in this module are deliberately not executable action candidates.
They share the action request's multimodal prefix and physical suffix batch, but
are normalized only against labels in the same decision group.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

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
BODY_SUFFIXES_ZH = {
    "IB0": "IB0=没有身体动作请求",
    "IB1": "IB1=要求执行身体动作",
    "IB2": "IB2=明确禁止身体动作",
    "IB3": "IB3=询问身体动作能力",
}
BODY_SUFFIXES_EN = {
    "IB0": "IB0=no body action request",
    "IB1": "IB1=perform action now",
    "IB2": "IB2=action explicitly prohibited",
    "IB3": "IB3=ask action capability",
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
# The grouped PPL decision is only an action-publication safety gate.  Exact
# visual scope (hand, face, head, body, and so on) is owned by the unified turn
# intent JSON, so scoring one suffix per body part here duplicates semantics and
# creates avoidable low-margin ties for generic requests such as "do this".
VISUAL_LABELS = {
    "IV00": "",
    "IV01": "COPY_ACTION",
    "IV11": "VISUAL_ANSWER",
}

ACTION_DECISION_LABELS = frozenset(
    (*BODY_LABELS, *FACE_LABELS, *REACTION_LABELS, *VISUAL_LABELS)
)
CATEGORY_GATE_LABEL_PREFIX = "IC"
CATEGORY_GATE_UNSUPPORTED_LABEL = "IC00"
CATEGORY_GATE_NONE_LABEL = "ICN0"
ACTION_SUPPORT_LABELS = {
    "IS0": "supported",
    "IS1": "unsupported",
}
ACTION_SUPPORT_SUFFIXES_ZH = {
    "IS0": "IS0=存在能实际完成请求的具体动作",
    "IS1": "IS1=不存在能实际完成请求的具体动作",
}
ACTION_SUPPORT_SUFFIXES_EN = {
    "IS0": "IS0=a concrete action can fulfill the request",
    "IS1": "IS1=no concrete action can fulfill the request",
}


def category_gate_label(category_id: str) -> str:
    """Return the non-executable score label for one catalog category."""

    normalized = str(category_id).strip()
    if not normalized or normalized == "00":
        raise ValueError(f"invalid concrete category_id: {category_id!r}")
    return CATEGORY_GATE_LABEL_PREFIX + normalized


def category_gate_score_ids(categories: Iterable[Any]) -> frozenset[str]:
    return frozenset(
        [
            *(category_gate_label(category.category_id) for category in categories),
            CATEGORY_GATE_UNSUPPORTED_LABEL,
            CATEGORY_GATE_NONE_LABEL,
        ]
    )


def category_gate_candidates(
    categories: Iterable[Any], *, english: bool = False
) -> list[ActionScoreCandidate]:
    """Build category labels scored in the existing physical suffix batch."""

    values = list(categories)
    candidates = [
        ActionScoreCandidate(
            candidate_id=category_gate_label(category.category_id),
            # Keep every category result in the same compact structure. The
            # semantic legend lives in the system prompt; repeating variable-
            # length category names in the scored suffix introduces a strong
            # mean-logprob length/token bias between categories.
            suffix=category_gate_label(category.category_id),
            action_id=category.category_id,
        )
        for category in values
    ]
    candidates.append(
        ActionScoreCandidate(
            candidate_id=CATEGORY_GATE_UNSUPPORTED_LABEL,
            suffix=CATEGORY_GATE_UNSUPPORTED_LABEL,
            action_id="UNSUPPORTED",
        )
    )
    candidates.append(
        ActionScoreCandidate(
            candidate_id=CATEGORY_GATE_NONE_LABEL,
            suffix=CATEGORY_GATE_NONE_LABEL,
            action_id="NO_ACTION_REQUEST",
        )
    )
    return candidates


def action_support_candidates(
    *, english: bool = False
) -> list[ActionScoreCandidate]:
    suffixes = (
        ACTION_SUPPORT_SUFFIXES_EN if english else ACTION_SUPPORT_SUFFIXES_ZH
    )
    return [
        ActionScoreCandidate(
            candidate_id=label,
            suffix=suffixes[label],
            action_id=value,
        )
        for label, value in ACTION_SUPPORT_LABELS.items()
    ]


def category_gate_prompt(
    categories: Iterable[Any], *, english: bool = False
) -> str:
    """Describe a category group that is independent of concrete actions."""

    values = list(categories)
    if english:
        lines = [
            "[Independent action-category gate labels]",
            "IC labels form one classification group scored in the same physical batch as concrete actions. Compare IC labels only with other IC labels. They are not executable candidate_id values and must not compete with concrete actions or IB/IF/IR labels.",
            "Select the semantic category that can actually fulfill the user's requested observable action. The explicit action goal overrides idle behavior, persona, style, and framing preferences. Select IC00 only when the user explicitly requests an action but none of the allowed categories contains that semantic action. Do not replace an unsupported request with an idle or merely similar category. Select ICN0 when no observable action or natural social reaction is requested; use the idle category itself only when idle behavior or natural breathing is explicitly requested.",
            "Choose a concrete IC category or IC00 only for a present execution request, not merely because an action is mentioned. Third-person or past-tense statements, capability-only questions, quotations, and explicit prohibitions do not request execution and therefore use ICN0. A polite question that pragmatically asks the avatar to act now is an execution request.",
        ]
        lines.extend(
            f"{category_gate_label(category.category_id)}=category_id {category.category_id} | category={getattr(category, 'prompt_label', '') or category.source_label} | description={getattr(category, 'prompt_definition', '') or category.short_definition}"
            for category in values
        )
        lines.append(
            f"{CATEGORY_GATE_UNSUPPORTED_LABEL}=unsupported action category | the requested action's semantic category is absent from the allowed category list"
        )
        lines.append(
            f"{CATEGORY_GATE_NONE_LABEL}=no observable action request or social reaction"
        )
        lines.append(
            "IS0=a concrete allowed action can actually fulfill the requested execution; IS1=no concrete allowed action can actually fulfill it. Compare IS0 only with IS1. A merely similar action is unsupported. Explicit hand count, side, body part, posture transition, movement, direction, and interaction object must match. In particular, leaning the torso cannot fulfill standing up or sitting down."
        )
        return "\n".join(lines)
    lines = [
        "[独立的动作类别门控标签]",
        "IC 标签构成一个独立分类组，与具体动作在同一个物理 batch 中评分；IC 只和其他 IC 标签比较。IC 不是可执行 candidate_id，不得与具体动作或 IB/IF/IR 标签竞争。",
        "选择能够实际完成用户所要求外部可观察动作的语义类别。明确动作目标高于待机、人设、风格和取景偏好。只有用户明确要求动作、但允许类别中完全不存在该动作的语义类别时，才选择 IC00；不得用待机类别或仅含义相近的类别替代不支持请求。没有要求外部可观察动作、也没有需要动作回应的直接社交事件时选择 ICN0；只有明确要求待机或自然呼吸时才选择待机类别本身。",
        "只有要求当前执行时才选择具体 IC 类别或 IC00，不能因为文本或音频提到某个动作就选择动作类别。第三人称或过去动作陈述、只询问能力、引用动作说法以及明确禁止执行都不要求当前执行，应选 ICN0；采用疑问句形式但交际目的确实是礼貌要求角色现在行动时，仍属于执行请求。",
    ]
    lines.extend(
        f"{category_gate_label(category.category_id)}=category_id {category.category_id}｜类别={getattr(category, 'prompt_label', '') or category.source_label}｜说明={getattr(category, 'prompt_definition', '') or category.short_definition}"
        for category in values
    )
    lines.append(
        f"{CATEGORY_GATE_UNSUPPORTED_LABEL}=不支持的动作类别｜用户要求动作，但允许类别列表中不存在该动作的语义类别"
    )
    lines.append(
        f"{CATEGORY_GATE_NONE_LABEL}=没有外部可观察动作请求，也没有需要动作回应的直接社交事件"
    )
    lines.append(
        "IS0=存在允许的具体动作能够实际完成所要求的执行方式；IS1=不存在允许的具体动作能够实际完成。IS0 只和 IS1 比较。仅含义相近但执行不同仍属于不支持；用户限定的手数、左右、身体部位、姿态转换、位移、方向和交互物体必须匹配。尤其是躯干前倾不能完成站起或坐下。"
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CategoryGateDecision:
    winner: str
    category_id: str | None
    unsupported: bool
    no_action_request: bool
    margin: float | None
    action_request_margin: float | None
    support_margin: float | None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionSupportDecision:
    winner: str
    supported: bool
    margin: float | None
    scores: dict[str, float] = field(default_factory=dict)


def aggregate_category_gate(
    scores: Iterable[CandidateScore], categories: Iterable[Any]
) -> CategoryGateDecision:
    """Rank only IC labels and return their winning semantic category."""

    values = list(categories)
    category_by_label = {
        category_gate_label(category.category_id): category.category_id
        for category in values
    }
    expected = frozenset(
        [
            *category_by_label,
            CATEGORY_GATE_UNSUPPORTED_LABEL,
            CATEGORY_GATE_NONE_LABEL,
        ]
    )
    score_by_id = {
        score.candidate_id: float(score.mean_logprob)
        for score in scores
        if score.candidate_id in expected
    }
    if set(score_by_id) != set(expected):
        missing = sorted(expected - set(score_by_id))
        raise ValueError(f"missing category-gate labels: {missing}")
    ranked = sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)
    winner = ranked[0][0]
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    best_request_score = max(
        value
        for label, value in score_by_id.items()
        if label != CATEGORY_GATE_NONE_LABEL
    )
    best_category_score = max(
        score_by_id[label] for label in category_by_label
    )
    action_request_margin = (
        best_request_score - score_by_id[CATEGORY_GATE_NONE_LABEL]
    )
    # Positive means a concrete category is better than unsupported; negative
    # means unsupported is better.  This isolates the support boundary from
    # unrelated category-to-category ties and from ICN0.
    support_margin = (
        best_category_score - score_by_id[CATEGORY_GATE_UNSUPPORTED_LABEL]
    )
    unsupported = winner == CATEGORY_GATE_UNSUPPORTED_LABEL
    no_action_request = winner == CATEGORY_GATE_NONE_LABEL
    return CategoryGateDecision(
        winner=winner,
        category_id=(
            None
            if unsupported or no_action_request
            else category_by_label[winner]
        ),
        unsupported=unsupported,
        no_action_request=no_action_request,
        margin=margin,
        action_request_margin=action_request_margin,
        support_margin=support_margin,
        scores=dict(score_by_id),
    )


def category_gate_as_dict(
    decision: CategoryGateDecision,
) -> dict[str, Any]:
    return {
        "winner": decision.winner,
        "category_id": decision.category_id,
        "unsupported": decision.unsupported,
        "no_action_request": decision.no_action_request,
        "margin": decision.margin,
        "action_request_margin": decision.action_request_margin,
        "support_margin": decision.support_margin,
        "scores": dict(decision.scores),
    }


def aggregate_action_support(
    scores: Iterable[CandidateScore],
) -> ActionSupportDecision:
    score_by_id = {
        score.candidate_id: float(score.mean_logprob)
        for score in scores
        if score.candidate_id in ACTION_SUPPORT_LABELS
    }
    if set(score_by_id) != set(ACTION_SUPPORT_LABELS):
        missing = sorted(set(ACTION_SUPPORT_LABELS) - set(score_by_id))
        raise ValueError(f"missing action-support labels: {missing}")
    ranked = sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)
    return ActionSupportDecision(
        winner=ranked[0][0],
        supported=ranked[0][0] == "IS0",
        margin=ranked[0][1] - ranked[1][1],
        scores=dict(score_by_id),
    )


def action_support_as_dict(
    decision: ActionSupportDecision,
) -> dict[str, Any]:
    return {
        "winner": decision.winner,
        "supported": decision.supported,
        "margin": decision.margin,
        "scores": dict(decision.scores),
    }


def _candidate_blueprints(
    groups: tuple[dict[str, str], ...],
    *,
    suffixes: Mapping[str, str] | None = None,
) -> tuple[ActionScoreCandidate, ...]:
    return tuple(
        ActionScoreCandidate(
            candidate_id=label,
            suffix=(suffixes or {}).get(label, label),
            action_id=value,
        )
        for group in groups
        for label, value in group.items()
    )


_VISUAL_CANDIDATE_BLUEPRINTS = _candidate_blueprints((VISUAL_LABELS,))


def action_decision_candidates(
    *, include_visual: bool = False, english: bool = False
) -> list[ActionScoreCandidate]:
    """Return real suffix requests; no padding candidates are ever emitted."""

    core = _candidate_blueprints(
        (BODY_LABELS, FACE_LABELS, REACTION_LABELS),
        suffixes=(BODY_SUFFIXES_EN if english else BODY_SUFFIXES_ZH),
    )
    return list(
        core
        + (_VISUAL_CANDIDATE_BLUEPRINTS if include_visual else ())
    )


def action_decision_prompt(
    *,
    include_visual: bool = False,
    english: bool = False,
    selection_tokens: Mapping[str, str] | None = None,
    output_selection_token: bool = False,
) -> str:
    """Describe independent decision groups embedded in the suffix batch."""

    if english:
        prompt = """
[Independent action-safety decision labels]
The IB/IF/IR labels below are independent classification groups. Scores are
compared only within the same prefix. They are not executable candidate_id
values and must not compete with concrete actions.
Use the complete semantic result identifiers below when the IB group is scored:
IB0=no body action request; IB1=perform action now;
IB2=action explicitly prohibited; IB3=ask action capability.
IB1 covers any request to visibly execute an action now, including polite or
question-shaped commands, whether or not the catalog can perform that action.
Use IB0 only when no execution is requested. An unsupported action request is
still IB1, not IB0. Use IB2 only for an explicit prohibition, and use IB3 only
when the user asks for capability information rather than present execution.
A third-person or past-tense statement and a quotation are IB0 unless the user
also explicitly prohibits execution, which is IB2. Distinguish a polite request
such as "could you do it now?" (IB1) from an ability-only question such as "do
you know how to do it?" (IB3) by communicative intent, not punctuation alone.
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
评分 IB 组时使用下列完整语义结果标识：
IB0=没有身体动作请求；IB1=要求执行身体动作；
IB2=明确禁止身体动作；IB3=询问身体动作能力。
凡是要求现在可观察地执行动作，都选 IB1；礼貌说法、疑问句形式以及目录中没有
对应动作的请求也不例外。只有完全没有要求执行动作时才选 IB0；不支持的动作请求
仍是 IB1，不得选 IB0。只有明确禁止执行才选 IB2；只有询问能力信息而非要求现在
执行时才选 IB3。
第三人称、过去发生的动作陈述以及只引用动作词均选 IB0；如果引用的同时明确要求
不要做动作，则选 IB2。根据交际目的区分礼貌的即时请求（如“你能现在做一下吗”，
选 IB1）与只了解能力（如“你会不会做这个动作”，选 IB3），不得只按问号判断。
IF0=没有要求脸部表情；IF1=要求执行脸部表情。
IR0=没有直接社交反应；IR1=问候；IR2=道别；IR3=感谢；IR4=祝贺；IR5=亲昵。
明确动作、禁止、能力询问、引用朗读和普通问答均选择 IR0。
""".strip()
    if not include_visual:
        visual = ""
    else:
        visual = """
IV00=不依赖当前画面执行动作；IV01=照抄、模仿或重复当前画面里的动作；
IV11=观察、计算或推理当前画面后，用手势表达新答案。
普通问答、识别或描述画面、能力询问、禁止动作和无需看图的明确动作均选 IV00。
模仿画面动作时，即使同时要求说话，仍选 IV01。具体身体范围由统一意图解析，
本组只判断是否依赖当前画面，不区分手、脸、头部或全身。
""".strip()
        if english:
            visual = """
IV00=no action execution dependency on the current image; IV01=copy, imitate,
or repeat an action visible in the current image; IV11=reason over the current
image and express the newly derived answer with a gesture. Questions, image
recognition or description, capability queries, prohibitions, and named actions
not requiring the image use IV00. Visual imitation remains IV01 when speech is
also requested. Exact body scope is owned by the unified intent parser; this
group only decides whether current-image-dependent execution is required.
""".strip()
    if visual:
        prompt += "\n" + visual
    if selection_tokens:
        labels = [
            *BODY_LABELS,
            *FACE_LABELS,
            *REACTION_LABELS,
            *(VISUAL_LABELS if include_visual else {}),
        ]
        legend = "; ".join(
            f"{selection_tokens[label]}={label}" for label in labels
        )
        prompt += "\n" + (
            (
                "Output the selection_token on the left, not the decision label: "
                if output_selection_token else
                "Equivalent selection_token=decision-label pairs for shadow scoring: "
            ) if english else (
                "输出等号左侧的 selection_token，不要输出决策标签："
                if output_selection_token else
                "用于影子评分的等价 selection_token=决策标签映射："
            )
        ) + legend
    return prompt


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
    body_evidence_source: str = "body_group"
    # Retained as an all-groups diagnostic for compatibility. Body delivery
    # must use body_gate_confident/body_gate_margin instead.
    confident: bool = False
    min_margin: float | None = None
    groups: dict[str, DecisionGroupResult] = field(default_factory=dict)

    @property
    def allows_body(self) -> bool:
        return (
            self.body_gate_confident
            and self.visual_scope != "VISUAL_ANSWER"
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


def fuse_category_gate_into_action_decision(
    decision: ActionDecision,
    category_decision: CategoryGateDecision,
    *,
    min_margin: float,
) -> ActionDecision:
    """Use independent same-batch category evidence to resolve an IB tie."""

    if (
        decision.body_mode in {"prohibit", "capability_query"}
        or (decision.body_mode == "perform" and decision.body_gate_confident)
        or (
            decision.body_mode == "none"
            and decision.groups["body"].margin is not None
            and decision.groups["body"].margin >= min_margin * 3.5
        )
        or category_decision.no_action_request
        or category_decision.action_request_margin is None
        or category_decision.action_request_margin < min_margin
    ):
        return decision
    return replace(
        decision,
        body_mode="perform",
        body_confident=True,
        body_gate_confident=True,
        body_gate_margin=category_decision.action_request_margin,
        body_evidence_source="category_gate",
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
        "body_evidence_source": decision.body_evidence_source,
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
