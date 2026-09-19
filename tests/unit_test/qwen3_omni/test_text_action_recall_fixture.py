from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.generate_text_action_recall_cases import build_cases


ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = (
    ROOT / "sglang_omni/assets/character_limited_action_global_catalog.json"
)
FIXTURE_PATH = (
    ROOT / "tests/unit_test/fixtures/realtime_text_action_recall_cases.json"
)


def test_text_action_recall_fixture_is_complete_and_valid() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    candidate_ids = {
        action["candidate_id"]
        for category in catalog["categories"]
        for action in category["children"]
    }

    assert fixture["schema_version"] == 1
    assert fixture["catalog_candidate_count"] == 117
    assert fixture["case_count"] == len(cases) == 238
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["text"].strip() for case in cases)
    assert Counter(case["group"] for case in cases) == {
        "canonical_label": 117,
        "natural_request": 99,
        "intent_boundary": 12,
        "unsupported_request": 10,
    }
    assert {
        case["expected"]["candidate_ids"][0]
        for case in cases
        if case["group"] == "canonical_label"
    } == candidate_ids
    assert all(
        set(case["expected"].get("candidate_ids", ()))
        <= candidate_ids | {"UNSUPPORTED"}
        for case in cases
    )


def test_text_action_recall_fixture_matches_generator() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    metadata, cases = build_cases(CATALOG_PATH)

    assert fixture == {**metadata, "case_count": len(cases), "cases": cases}
