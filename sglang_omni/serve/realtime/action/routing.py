"""Deterministic, catalog-bounded action routing policies."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
)


_TRAILING_QUALIFIER_RE = re.compile(r"\s*[（(][^（）()]*[）)]\s*$")


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


def _normalized_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
    )


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
        matched_alias = next(
            (
                alias
                for alias in _catalog_aliases(candidate)
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
