from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.qwen3_omni.global_action_catalog import (
    load_runtime_action_catalog,
)
from sglang_omni.serve.realtime.action.background_calibration import (
    compare_supported_action_scores,
    load_action_background_calibration,
)


def test_background_calibration_covers_runtime_limited_candidate_ids() -> None:
    catalog = load_runtime_action_catalog()
    calibration = load_action_background_calibration()

    assert set(calibration.candidate_centered_bias) == set(catalog.candidate_by_id)


def test_background_calibration_flips_supported_shadow_only() -> None:
    calibration = load_action_background_calibration()
    summary, diagnostics = compare_supported_action_scores(
        [
            SimpleNamespace(candidate_id="000", mean_logprob=-0.9),
            SimpleNamespace(candidate_id="130", mean_logprob=-1.0),
            SimpleNamespace(candidate_id="285", mean_logprob=-1.1),
        ],
        calibration,
        catalog_hash=calibration.catalog_hash,
    )

    assert summary["status"] == "completed"
    assert summary["authoritative_candidate_id"] == "000"
    assert summary["raw_supported_candidate_id"] == "130"
    assert summary["shadow_supported_candidate_id"] == "285"
    assert summary["selection_changed"] is True
    assert summary["unmatched_candidate_ids"] == ["000"]
    assert diagnostics["130"]["raw_rank"] == 1
    assert diagnostics["130"]["calibrated_rank"] == 2
    assert diagnostics["285"]["raw_rank"] == 2
    assert diagnostics["285"]["calibrated_rank"] == 1


def test_background_calibration_skips_catalog_mismatch() -> None:
    calibration = load_action_background_calibration()
    summary, diagnostics = compare_supported_action_scores(
        [SimpleNamespace(candidate_id="130", mean_logprob=-1.0)],
        calibration,
        catalog_hash="sha256:not-the-calibrated-catalog",
    )

    assert summary["status"] == "skipped"
    assert summary["reason"] == "catalog_hash_mismatch"
    assert diagnostics == {}
