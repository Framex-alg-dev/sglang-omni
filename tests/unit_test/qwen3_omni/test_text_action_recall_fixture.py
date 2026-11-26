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
    contract = json.loads((ROOT / "tests/unit_test/fixtures/limited_action_catalog_contract.json").read_text())
    candidate_ids = {
        action["candidate_id"]
        for category in catalog["categories"]
        for action in category["children"]
    }

    assert fixture["schema_version"] == 1
    assert fixture["catalog_candidate_count"] == len(contract["candidate_ids"])
    assert fixture["catalog_hash"] == contract["catalog_hash"]
    assert candidate_ids == set(contract["candidate_ids"])
    assert fixture["case_count"] == len(cases) == sum(contract["recall_fixture_groups"].values())
    assert [row["case_id"] for row in fixture["excluded_natural_cases"]] == contract["omitted_natural_case_ids"]
    assert all(row["absent_candidate_id"] not in candidate_ids for row in fixture["excluded_natural_cases"])
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["text"].strip() for case in cases)
    assert Counter(case["group"] for case in cases) == contract["recall_fixture_groups"]
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
