"""Deterministic, catalog-bounded action routing policies."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping


class IntentShortcutBackoff:
    """Session-owned, bounded negative cache; only skips speculative recall.

    Never caches an action result or changes the ordinary constrained matcher.
    """

    def __init__(self, ttl_seconds: float = 30.0, capacity: int = 32):
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self.failures: OrderedDict[tuple[str, str], float] = OrderedDict()

    def blocked(self, key: tuple[str, str], now: float) -> bool:
        expiry = self.failures.get(key)
        if expiry is None:
            return False
        if now >= expiry:
            self.failures.pop(key, None)
            return False
        return True

    def reject(self, key: tuple[str, str], now: float) -> None:
        self.failures[key] = now + self.ttl_seconds
        self.failures.move_to_end(key)
        while len(self.failures) > self.capacity:
            self.failures.popitem(last=False)

from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
)


_TRAILING_QUALIFIER_RE = re.compile(r"\s*[（(][^（）()]*[）)]\s*$")
_VISUAL_CATEGORY_SCOPES = (
    (
        "gesture",
        ("手势", "手型", "gesture", "handsign"),
        ("手部与手势", "handandgesture", "handgesture"),
    ),
    (
        "head_gaze",
        ("头部", "视线", "眼神", "head", "gaze"),
        ("头部与视线", "headandgaze", "headgaze"),
    ),
    (
        "upper_limb",
        ("手臂", "胳膊", "上肢", "arm", "upperlimb"),
        ("上肢大臂与前臂", "upperarmandforearm", "upperlimb"),
    ),
    (
        "torso",
        ("肩膀", "肩部", "躯干", "上身", "shoulder", "torso"),
        ("颈肩与躯干", "neckshoulderandtorso", "torso"),
    ),
    (
        "lower_limb",
        ("腿部", "脚步", "下肢", "leg", "footwork", "lowerlimb"),
        ("下肢与脚步", "lowerlimbandfootwork", "lowerlimb"),
    ),
    (
        "full_body",
        ("全身", "wholebody", "fullbody"),
        ("全身动作与复合动作", "wholebodyandcompound", "fullbody"),
    ),
    (
        "pose",
        ("姿势", "姿态", "体态", "pose", "posture"),
        (
            "基础姿态与动作转场",
            "全身动作与复合动作",
            "baseposeandtransition",
            "wholebodyandcompound",
        ),
    ),
    (
        "object_interaction",
        ("物品", "物体", "道具", "object", "prop"),
        ("现实物品交互", "physicalobjectinteraction", "objectinteraction"),
    ),
    (
        "screen_interaction",
        ("屏幕", "虚拟空间", "screen", "virtualspace"),
        ("屏幕与虚拟空间交互", "screenandvirtualspace"),
    ),
)
_VISUAL_EXPRESSION_SCOPE_MARKERS = (
    "表情",
    "神情",
    "脸部",
    "expression",
    "facialexpression",
)


@dataclass(frozen=True, slots=True)
class ExplicitActionRoute:
    """One unambiguous candidate resolved from catalog-owned labels."""

    category: SessionActionCategory
    candidate: SessionActionCandidate
    matched_alias: str


@dataclass(frozen=True, slots=True)
class CategoryWidthDecision:
    """Effective Category recall width and its auditable confidence facts."""

    effective_top_k: int
    adaptive_top1: bool
    reason: str
    top_ppl: float | None
    confidence_margin: float | None


@dataclass(frozen=True, slots=True)
class VisualDeicticCategoryScope:
    """Catalog category family explicitly named by a camera-backed request."""

    name: str
    categories: tuple[SessionActionCategory, ...]


@dataclass(frozen=True, slots=True)
class NumericGestureCandidate:
    """A catalog candidate whose label exactly names one integer gesture."""

    value: int
    candidate: SessionActionCandidate


@dataclass(frozen=True, slots=True)
class NumericReplyActionRoute:
    """Eligibility result for complete-reply numeric action scoring."""

    enabled: bool
    reason: str
    candidates: tuple[NumericGestureCandidate, ...] = ()


_NUMERIC_GESTURE_LABELS = (
    "数字零手势",
    "数字一手势",
    "数字二手势",
    "数字三手势",
    "数字四手势",
    "数字五手势",
    "数字六手势",
    "数字七手势",
    "数字八手势",
    "数字九手势",
    "数字十手势",
)
_NUMERIC_REPLY_ARABIC_RE = re.compile(
    r"(?<![\d.])(?:10|[0-9])(?![\d.])"
)
_NUMERIC_REPLY_CHINESE_RE = re.compile(r"[零〇一二两三四五六七八九十]")
_NUMERIC_REPLY_ENGLISH_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def _normalized_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
    )


def numeric_gesture_candidates(
    candidates: Iterable[SessionActionCandidate],
) -> tuple[NumericGestureCandidate, ...]:
    """Return unique 0-10 candidates derived only from exact catalog labels.

    Candidate IDs are deliberately not part of this policy. Ambiguous duplicate
    labels are omitted instead of choosing one by ordering.
    """

    labels_by_value = {
        _normalized_label(label): value
        for value, label in enumerate(_NUMERIC_GESTURE_LABELS)
    }
    matches: dict[int, dict[str, SessionActionCandidate]] = {}
    for candidate in candidates:
        value = labels_by_value.get(_normalized_label(candidate.source_label))
        if value is not None:
            matches.setdefault(value, {}).setdefault(
                candidate.candidate_id, candidate
            )
    return tuple(
        NumericGestureCandidate(value, next(iter(matches[value].values())))
        for value in range(11)
        if len(matches.get(value, ())) == 1
    )


def complete_reply_may_contain_numeric_answer(text: str) -> bool:
    """Cheap gate before the bounded semantic scorer is admitted."""

    return bool(
        text.strip()
        and (
            _NUMERIC_REPLY_ARABIC_RE.search(text)
            or _NUMERIC_REPLY_CHINESE_RE.search(text)
            or _NUMERIC_REPLY_ENGLISH_RE.search(text)
        )
    )


def route_numeric_reply_action(
    *,
    turn_origin: str,
    reply_provided: bool,
    speech_kind: str | None,
    body_mode: str | None,
    has_user_camera: bool,
    has_text_output: bool,
    has_action_output: bool,
    candidates: Iterable[SessionActionCandidate],
) -> NumericReplyActionRoute:
    """Enable the second action pass only for generated body-neutral replies."""

    if turn_origin != "user":
        return NumericReplyActionRoute(False, "non_user_turn")
    if reply_provided:
        return NumericReplyActionRoute(False, "provided_reply")
    if not has_text_output or not has_action_output:
        return NumericReplyActionRoute(False, "required_modality_missing")
    if speech_kind != "generated" and not (
        speech_kind == "none" and has_user_camera
    ):
        return NumericReplyActionRoute(False, "not_generated_reply")
    if body_mode != "none":
        return NumericReplyActionRoute(False, "explicit_body_directive")
    numeric_candidates = numeric_gesture_candidates(candidates)
    if not numeric_candidates:
        return NumericReplyActionRoute(False, "numeric_candidates_missing")
    return NumericReplyActionRoute(True, "eligible", numeric_candidates)


def _catalog_aliases(candidate: SessionActionCandidate) -> tuple[str, ...]:
    # TODO(action-routing): Evaluate a versioned, collision-free wire contract
    # before accepting D_video_call reviewed/generated aliases. Until then,
    # keep this route bounded to fields already owned by the Session catalog;
    # do not read cross-repository alias files or add them to session.start.
    labels = [
        candidate.candidate_id,
        candidate.action_id,
        candidate.source_label,
    ]
    unqualified = _TRAILING_QUALIFIER_RE.sub("", candidate.source_label).strip()
    if unqualified and unqualified != candidate.source_label:
        labels.append(unqualified)
    return tuple(dict.fromkeys(label for label in labels if label.strip()))


def resolve_unique_explicit_action(
    body_task: str,
    candidates: Iterable[tuple[SessionActionCategory, SessionActionCandidate]],
    *,
    aliases_by_candidate_id: Mapping[str, Iterable[str]] | None = None,
) -> ExplicitActionRoute | None:
    """Return a route only when a parsed body task names one concrete action.

    Matching is equality-only after Unicode, whitespace, and punctuation
    normalization.  The aliases are derived from the current Session catalog;
    raw-speech substring rules and hand-maintained phrase maps are deliberately
    excluded.  Multiple categories for the same concrete candidate are safe;
    multiple candidate IDs remain ambiguous and therefore fall back to model
    Category scoring.
    """

    normalized_task = _normalized_label(body_task)
    if not normalized_task:
        return None
    matches: dict[str, ExplicitActionRoute] = {}
    for category, candidate in candidates:
        aliases = list(_catalog_aliases(candidate))
        if aliases_by_candidate_id is not None:
            aliases.extend(aliases_by_candidate_id.get(candidate.candidate_id, ()))
        matched_alias = next(
            (
                alias
                for alias in aliases
                if _normalized_label(alias) == normalized_task
            ),
            None,
        )
        if matched_alias is None:
            continue
        matches.setdefault(
            candidate.candidate_id,
            ExplicitActionRoute(category, candidate, matched_alias),
        )
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def scope_visual_deictic_categories(
    categories: Iterable[SessionActionCategory],
    *,
    body_task: str,
    body_mode: str,
    has_user_camera: bool,
) -> VisualDeicticCategoryScope | None:
    """Resolve a camera-backed execution request only when language names its range.

    Generic phrases such as ``这个动作`` intentionally return ``None``. They do
    not justify scanning the complete catalog. Scope terms such as ``这个手势``
    or ``做出手势`` select catalog-owned families; the current user-camera image
    supplies the explicit or implicit visual reference, and child scoring still
    identifies the concrete visible action without a candidate-specific rule.
    """

    normalized_task = _normalized_label(body_task)
    if (
        body_mode != "perform"
        or not has_user_camera
    ):
        return None
    scoped_categories = tuple(categories)
    for scope_name, task_markers, path_markers in _VISUAL_CATEGORY_SCOPES:
        if not any(marker in normalized_task for marker in task_markers):
            continue
        matches = tuple(
            category
            for category in scoped_categories
            if any(
                marker in _normalized_label(" ".join(category.category_path))
                for marker in path_markers
            )
        )
        if matches:
            return VisualDeicticCategoryScope(scope_name, matches)
    return None


def is_visual_deictic_expression_request(
    *,
    face_task: str,
    has_user_camera: bool,
) -> bool:
    """Return whether a camera-backed request names the expression range."""

    normalized_task = _normalized_label(face_task)
    return bool(
        has_user_camera
        and any(
            marker in normalized_task
            for marker in _VISUAL_EXPRESSION_SCOPE_MARKERS
        )
    )


def choose_category_width(
    *,
    configured_top_k: int,
    ranked_real_scores: list[tuple[str, float, float]],
    overall_top_candidate_id: str,
    adaptive_enabled: bool,
    top_category_prewarmed: bool,
    min_margin: float,
    max_ppl: float,
) -> CategoryWidthDecision:
    """Choose Top-1 only for a strong, globally prewarmed Category result."""

    if not ranked_real_scores:
        return CategoryWidthDecision(0, False, "no_real_category", None, None)
    top_id, top_logprob, top_ppl = ranked_real_scores[0]
    margin = (
        top_logprob - ranked_real_scores[1][1]
        if len(ranked_real_scores) > 1
        else None
    )
    bounded_top_k = min(configured_top_k, len(ranked_real_scores))
    if bounded_top_k <= 1:
        return CategoryWidthDecision(
            bounded_top_k, False, "configured_top1", top_ppl, margin
        )
    if not adaptive_enabled:
        return CategoryWidthDecision(
            bounded_top_k, False, "adaptive_disabled", top_ppl, margin
        )
    if overall_top_candidate_id != top_id:
        return CategoryWidthDecision(
            bounded_top_k, False, "unsupported_ranked_first", top_ppl, margin
        )
    if not top_category_prewarmed:
        return CategoryWidthDecision(
            bounded_top_k, False, "top_child_prefix_not_prewarmed", top_ppl, margin
        )
    if not math.isfinite(top_ppl) or top_ppl > max_ppl:
        return CategoryWidthDecision(
            bounded_top_k, False, "top_ppl_too_high", top_ppl, margin
        )
    if margin is None or not math.isfinite(margin) or margin < min_margin:
        return CategoryWidthDecision(
            bounded_top_k, False, "category_margin_too_small", top_ppl, margin
        )
    return CategoryWidthDecision(1, True, "high_confidence_prewarmed_top1", top_ppl, margin)
