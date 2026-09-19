"""Supported-action background calibration used for shadow comparison only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ACTION_BACKGROUND_CALIBRATION_PATH_ENV = (
    "SGLANG_OMNI_ACTION_BACKGROUND_CALIBRATION_PATH"
)
DEFAULT_ACTION_BACKGROUND_CALIBRATION_RESOURCE = (
    "assets/action_score_background_calibration.json"
)
WATCHED_CANDIDATE_IDS = ("130", "135")


@dataclass(frozen=True, slots=True)
class ActionBackgroundCalibration:
    calibration_version: str
    calibration_hash: str
    catalog_hash: str
    alpha: float
    method: str
    baseline_case_count: int
    candidate_centered_bias: Mapping[str, float]


def _calibration_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv(ACTION_BACKGROUND_CALIBRATION_PATH_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(
        str(
            files("sglang_omni").joinpath(
                DEFAULT_ACTION_BACKGROUND_CALIBRATION_RESOURCE
            )
        )
    )


@lru_cache(maxsize=8)
def load_action_background_calibration(
    path: str | Path | None = None,
) -> ActionBackgroundCalibration:
    calibration_path = _calibration_path(path)
    raw = calibration_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError("action background calibration schema_version must be 1")
    calibration_version = payload.get("calibration_version")
    catalog_hash = payload.get("catalog_hash")
    method = payload.get("method")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (calibration_version, catalog_hash, method)
    ):
        raise ValueError("action background calibration metadata is incomplete")
    alpha = payload.get("alpha")
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise ValueError("action background calibration alpha must be finite")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("action background calibration alpha must be between 0 and 1")
    baseline_case_count = payload.get("baseline_case_count")
    if not isinstance(baseline_case_count, int) or baseline_case_count <= 0:
        raise ValueError(
            "action background calibration baseline_case_count must be positive"
        )
    raw_bias = payload.get("candidate_centered_bias")
    if not isinstance(raw_bias, dict) or not raw_bias:
        raise ValueError("action background calibration candidate bias is empty")
    bias: dict[str, float] = {}
    for candidate_id, value in raw_bias.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("action background calibration candidate_id is invalid")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"action background calibration bias is invalid: {candidate_id!r}"
            )
        bias[candidate_id] = float(value)
    if payload.get("candidate_count") != len(bias):
        raise ValueError("action background calibration candidate_count mismatch")
    return ActionBackgroundCalibration(
        calibration_version=calibration_version.strip(),
        calibration_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
        catalog_hash=catalog_hash.strip(),
        alpha=float(alpha),
        method=method.strip(),
        baseline_case_count=baseline_case_count,
        candidate_centered_bias=MappingProxyType(bias),
    )


def compare_supported_action_scores(
    scores: Iterable[Any],
    calibration: ActionBackgroundCalibration,
    *,
    catalog_hash: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float | int]]]:
    """Return a shadow summary and per-candidate diagnostics without enforcement."""

    score_by_id = {
        str(score.candidate_id): float(score.mean_logprob) for score in scores
    }
    base = {
        "mode": "shadow",
        "calibration_version": calibration.calibration_version,
        "calibration_hash": calibration.calibration_hash,
        "calibration_catalog_hash": calibration.catalog_hash,
        "runtime_catalog_hash": catalog_hash,
        "alpha": calibration.alpha,
        "method": calibration.method,
    }
    if calibration.catalog_hash != catalog_hash:
        return base | {"status": "skipped", "reason": "catalog_hash_mismatch"}, {}

    matched = {
        candidate_id: raw_score
        for candidate_id, raw_score in score_by_id.items()
        if candidate_id in calibration.candidate_centered_bias
    }
    if not matched:
        return base | {"status": "skipped", "reason": "no_supported_scores"}, {}

    raw_ranking = sorted(matched, key=matched.__getitem__, reverse=True)
    calibrated = {
        candidate_id: raw_score
        - calibration.alpha
        * calibration.candidate_centered_bias[candidate_id]
        for candidate_id, raw_score in matched.items()
    }
    calibrated_ranking = sorted(
        calibrated, key=calibrated.__getitem__, reverse=True
    )
    raw_rank = {
        candidate_id: index for index, candidate_id in enumerate(raw_ranking, 1)
    }
    calibrated_rank = {
        candidate_id: index
        for index, candidate_id in enumerate(calibrated_ranking, 1)
    }
    diagnostics = {
        candidate_id: {
            "raw_score": matched[candidate_id],
            "centered_bias": calibration.candidate_centered_bias[candidate_id],
            "calibrated_score": calibrated[candidate_id],
            "raw_rank": raw_rank[candidate_id],
            "calibrated_rank": calibrated_rank[candidate_id],
        }
        for candidate_id in matched
    }
    authoritative_id = max(score_by_id, key=score_by_id.__getitem__)
    raw_supported_id = raw_ranking[0]
    shadow_supported_id = calibrated_ranking[0]
    watched = {
        candidate_id: diagnostics[candidate_id]
        for candidate_id in WATCHED_CANDIDATE_IDS
        if candidate_id in diagnostics
    }
    summary = base | {
        "status": "completed",
        "scored_candidate_count": len(score_by_id),
        "matched_supported_candidate_count": len(matched),
        "unmatched_candidate_ids": sorted(set(score_by_id) - set(matched)),
        "authoritative_candidate_id": authoritative_id,
        "authoritative_is_outside_supported_calibration": (
            authoritative_id not in matched
        ),
        "raw_supported_candidate_id": raw_supported_id,
        "raw_supported_score": matched[raw_supported_id],
        "shadow_supported_candidate_id": shadow_supported_id,
        "shadow_supported_raw_score": matched[shadow_supported_id],
        "shadow_supported_centered_bias": calibration.candidate_centered_bias[
            shadow_supported_id
        ],
        "shadow_supported_calibrated_score": calibrated[shadow_supported_id],
        "selection_changed": raw_supported_id != shadow_supported_id,
        "watched_candidates": watched,
    }
    return summary, diagnostics


__all__ = [
    "ACTION_BACKGROUND_CALIBRATION_PATH_ENV",
    "ActionBackgroundCalibration",
    "compare_supported_action_scores",
    "load_action_background_calibration",
]
