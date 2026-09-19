from __future__ import annotations

from scripts.evaluate_text_action_recall import (
    judge_shadow,
    normalize_ranking,
    single_token_shadow_result,
    summarize_single_token_shadow,
)


def test_shadow_ranking_and_summary_use_ground_truth() -> None:
    ranking = normalize_ranking(
        [
            {"candidate_id": "270", "mean_logprob": -0.1, "ppl": 1.1},
            {"candidate_id": "000", "mean_logprob": -1.0, "ppl": 2.7},
            {"candidate_id": "IB1", "mean_logprob": -0.2, "ppl": 1.2},
        ],
        {"270"},
    )
    action, checks = judge_shadow(
        {
            "expected": {
                "candidate_ids": ["270"],
                "support_status": "supported",
            }
        },
        ranking,
    )
    row = {
        "id": "like",
        "group": "natural_request",
        "expected": {"candidate_ids": ["270"]},
        "commit_to_result_ms": 200.0,
        "single_token_shadow_action": action,
        "single_token_shadow_checks": checks,
        "single_token_shadow_passed": True,
        "single_token_shadow_metadata": {
            "agreement": {"action": False, "body": True}
        },
    }

    summary = summarize_single_token_shadow([row])

    assert [item["candidate_id"] for item in ranking] == ["270", "UNSUPPORTED"]
    assert summary is not None
    assert summary["checks"]["candidate_top1"]["rate"] == 1.0
    assert summary["legacy_agreement"]["action"]["rate"] == 0.0
    assert summary["legacy_agreement"]["body"]["rate"] == 1.0


def test_shadow_result_is_diagnostic_only() -> None:
    shadow = {"ranking": [{"candidate_id": "270"}]}
    result = {
        "media_summary": {
            "action_context": {"single_token_shadow": shadow}
        }
    }

    assert single_token_shadow_result(result) is shadow
    assert single_token_shadow_result({}) is None
