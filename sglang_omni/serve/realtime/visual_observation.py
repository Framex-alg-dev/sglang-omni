"""Shared visual-hand semantics and same-forward confidence accounting."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable


VISUAL_OBSERVATION_TOP_LOGPROBS = 2
VISUAL_OBSERVATION_MIN_MEAN_LOGPROB_ENV = (
    "SGLANG_OMNI_VISUAL_OBSERVATION_MIN_MEAN_LOGPROB"
)
VISUAL_OBSERVATION_MIN_TOKEN_MARGIN_ENV = (
    "SGLANG_OMNI_VISUAL_OBSERVATION_MIN_TOKEN_MARGIN"
)


_VISUAL_EQUIVALENCE_KEYS = {
    "数字二": "V_SIGN",
    "数字二手势": "V_SIGN",
    "单手比耶": "V_SIGN",
}


_VISUAL_DISAMBIGUATION_ZH = {
    "数字一手势": "必须是食指伸直，其余手指收拢；仅拇指伸直是点赞。",
    "数字二手势": "必须是食指和中指伸直成 V，其余手指收拢；与单手比耶视为同一手型。",
    "单手比耶": "必须是食指和中指伸直成 V，其余手指收拢；与数字二视为同一手型。",
    "数字三手势": "必须恰好是食指、中指、无名指三根伸直。",
    "数字四手势": "必须是除拇指外四指伸直且拇指内扣；拇指也展开是数字五。",
    "数字五手势": "必须五指全部伸直张开；拇指和食指形成 L 且其余手指收拢是数字八。",
    "数字六手势": "必须只有拇指和小指伸直；拇指和食指伸直不是数字六；若同一手型贴近耳侧模拟听筒，则是打电话手势。",
    "数字七手势": "必须是拇指、食指和中指指尖靠拢，形成三指捏合手型。",
    "数字八手势": "必须只有拇指和食指伸直形成 L；拇指和小指伸直是数字六。",
    "数字九手势": "必须是食指弯曲成钩，其余手指收拢；食指伸直是数字一。",
    "点赞": "必须是拇指单独竖起、其余手指握拢；不是数字一。",
    "双手比心": "必须由双手共同围出完整心形；单手指尖捏合不属于双手比心。",
    "打电话手势": "必须是拇指和小指伸直并靠近耳侧或脸侧模拟听筒；仅在身前展示相同手型是数字六。",
    "单手指心": "必须是拇指和食指交错形成清晰 X；两指仅相对靠拢而不交错是两指捏合。",
    "两指捏合": "必须是单手拇指与食指相对靠拢且不交错；食指和中指伸直成 V 不是捏合，交错成 X 是单手指心。",
}


def visual_equivalence_key(source_label: str) -> str:
    """Return the visual class used to collapse catalog aliases."""

    return _VISUAL_EQUIVALENCE_KEYS.get(source_label.strip(), source_label.strip())


def choose_visual_equivalent(candidates: Iterable[Any]) -> Any:
    """Choose a stable executor when multiple actions share one hand shape."""

    items = list(candidates)
    if not items:
        raise ValueError("visual-equivalence group must not be empty")
    return next(
        (item for item in items if item.source_label == "数字二手势"),
        items[0],
    )


def visual_candidate_definition(source_label: str, short_definition: str) -> str:
    """Render one catalog definition with centrally owned disambiguation."""

    distinction = _VISUAL_DISAMBIGUATION_ZH.get(source_label.strip())
    definition = short_definition.strip()
    if not distinction:
        return definition
    return f"{definition} 区分要点：{distinction}"


def numeric_visual_observation_rules_zh() -> str:
    """Rules shared by gesture imitation and visual arithmetic."""

    labels = (
        "数字一手势",
        "数字二手势",
        "数字三手势",
        "数字四手势",
        "数字五手势",
        "数字六手势",
        "数字七手势",
        "数字八手势",
        "数字九手势",
    )
    return "\n".join(
        f"- {label.removesuffix('手势')}：{_VISUAL_DISAMBIGUATION_ZH[label]}"
        for label in labels
    )


def _optional_finite_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class VisualObservationConfidence:
    """Confidence derived only from the already executed decode forward."""

    available: bool
    accepted: bool
    token_count: int
    mean_logprob: float | None
    min_token_margin: float | None
    rejection_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_visual_observation_confidence(
    output_token_logprobs: list[Any] | None,
    output_top_logprobs: list[Any] | None,
) -> VisualObservationConfidence:
    """Summarize selected-token confidence without another model request.

    Each selected-token entry is ``[logprob, token_id]``.  Each top-token
    entry is a list of the same pairs for that decode position.  Thresholds
    are optional so deployments can collect calibrated telemetry before they
    enforce fail-closed behavior.
    """

    min_mean = _optional_finite_env(VISUAL_OBSERVATION_MIN_MEAN_LOGPROB_ENV)
    min_margin = _optional_finite_env(VISUAL_OBSERVATION_MIN_TOKEN_MARGIN_ENV)
    thresholds_enabled = min_mean is not None or min_margin is not None

    selected: list[tuple[float, int]] = []
    for raw in output_token_logprobs or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            logprob = float(raw[0])
            token_id = int(raw[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(logprob):
            selected.append((logprob, token_id))

    if not selected:
        return VisualObservationConfidence(
            available=False,
            accepted=not thresholds_enabled,
            token_count=0,
            mean_logprob=None,
            min_token_margin=None,
            rejection_reason=(
                "confidence_unavailable" if thresholds_enabled else None
            ),
        )

    margins: list[float] = []
    top_rows = output_top_logprobs or []
    for index, (chosen_logprob, chosen_token_id) in enumerate(selected):
        if index >= len(top_rows) or not isinstance(top_rows[index], list):
            continue
        alternatives: list[float] = []
        for raw in top_rows[index]:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            try:
                logprob = float(raw[0])
                token_id = int(raw[1])
            except (TypeError, ValueError):
                continue
            if token_id != chosen_token_id and math.isfinite(logprob):
                alternatives.append(logprob)
        if alternatives:
            margins.append(chosen_logprob - max(alternatives))

    mean_logprob = math.fsum(item[0] for item in selected) / len(selected)
    observed_min_margin = min(margins) if margins else None
    rejection_reason = None
    if min_mean is not None and mean_logprob < min_mean:
        rejection_reason = "mean_logprob_below_threshold"
    elif min_margin is not None and (
        observed_min_margin is None or observed_min_margin < min_margin
    ):
        rejection_reason = "token_margin_below_threshold"
    return VisualObservationConfidence(
        available=True,
        accepted=rejection_reason is None,
        token_count=len(selected),
        mean_logprob=round(mean_logprob, 6),
        min_token_margin=(
            round(observed_min_margin, 6)
            if observed_min_margin is not None
            else None
        ),
        rejection_reason=rejection_reason,
    )


__all__ = [
    "VISUAL_OBSERVATION_TOP_LOGPROBS",
    "VisualObservationConfidence",
    "choose_visual_equivalent",
    "numeric_visual_observation_rules_zh",
    "summarize_visual_observation_confidence",
    "visual_candidate_definition",
    "visual_equivalence_key",
]
