import json
from pathlib import Path

from scripts.evaluate_action_intent_boundary import build_session_start, summarize


ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "tests/unit_test/fixtures/realtime_action_intent_boundary_cases.json"
CATALOG = ROOT / "sglang_omni/assets/character_limited_action_global_catalog.json"
SESSION = ROOT / "reports/soo_multimodal_506/session_start.json"


def test_intent_boundary_cases_are_strict_text_audio_pairs() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    pairs = {case["pair"] for case in cases}

    assert len(cases) == 16
    assert len(pairs) == 8
    for pair in pairs:
        paired = [case for case in cases if case["pair"] == pair]
        assert {case["modality"] for case in paired} == {"text", "audio"}
        assert len({case["text"] for case in paired}) in {1, 2}
        assert {case["expected"]["body_mode"] for case in paired} == {
            paired[0]["expected"]["body_mode"]
        }


def test_evaluation_uses_character_profile_with_limited_catalog_ids() -> None:
    template = json.loads(SESSION.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    start = build_session_start(template, catalog, session_id="eval-session")

    assert start["session_id"] == "eval-session"
    assert start["locale"] == template["locale"]
    assert start["reply"] == template["reply"]
    assert start["character_profile"] == template["character_profile"]
    assert start["action"]["category_guidance"] == template["action"][
        "category_guidance"
    ]
    assert start["action"]["candidate_guidance"] == template["action"][
        "candidate_guidance"
    ]
    expected_ids = {
        child["candidate_id"]
        for category in catalog["categories"]
        for child in category["children"]
    }
    actual_ids = {
        child["candidate_id"]
        for child in start["action"]["allowed_candidates"]
    }
    assert actual_ids == expected_ids
    assert all(not candidate_id.startswith("A") for candidate_id in actual_ids)
    assert start["outputs"] == ["text", "action"]
    assert "output_audio" not in start
    assert start["diagnostics"] == {"include_action_scores": True}


def test_summary_reports_action_accuracy_separately_from_body_mode() -> None:
    rows = [
        {
            "passed": False,
            "checks": {
                "body_mode": False,
                "execute": True,
                "support_status": True,
            },
        },
        {
            "passed": False,
            "checks": {
                "body_mode": True,
                "execute": True,
                "candidate_id": False,
            },
        },
    ]

    summary = summarize(rows)

    assert summary["action_accuracy"] == {
        "evaluated": 2,
        "passed": 1,
        "rate": 0.5,
    }
