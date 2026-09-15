from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.generate_realtime_action_recall_fixture import build_cases


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    ROOT / "tests/unit_test/fixtures/realtime_action_recall_cases.json"
)
CATALOG_PATH = ROOT / "sglang_omni/assets/character_action_global_catalog.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_realtime_action_recall_fixture_has_broad_reviewable_coverage() -> None:
    fixture = _load(FIXTURE_PATH)
    catalog = _load(CATALOG_PATH)
    cases = fixture["cases"]
    changed_category_ids = set(fixture["changed_category_ids"])
    catalog_category_ids = {
        category["category_id"] for category in catalog["categories"]
    }

    assert fixture["schema_version"] == 1
    assert len(cases) == 245
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["input"]["text"] for case in cases}) == len(cases)
    assert {case["locale"] for case in cases} == {"zh-CN"}
    assert {case["coverage_kind"] for case in cases} == {
        "normal",
        "boundary",
        "adversarial",
    }
    assert Counter(case["group"] for case in cases) == {
        "catalog_definition_seed": 150,
        "changed_category_boundary": 75,
        "system_route": 10,
        "restricted_category_unsupported": 10,
    }
    assert len(changed_category_ids) == 25
    assert changed_category_ids < catalog_category_ids

    expected_category_counts = Counter(
        category_id
        for case in cases
        for category_id in case["expected"]["acceptable_category_ids"]
        if category_id != "00"
    )
    assert set(expected_category_counts) == catalog_category_ids
    for category_id in catalog_category_ids - {"01", "02"}:
        minimum = 6 if category_id in changed_category_ids else 3
        assert expected_category_counts[category_id] >= minimum
    assert expected_category_counts["01"] == 5
    assert expected_category_counts["02"] == 5


def test_realtime_action_recall_fixture_references_valid_catalog_memberships() -> None:
    fixture = _load(FIXTURE_PATH)
    catalog = _load(CATALOG_PATH)
    categories = {
        category["category_id"]: category for category in catalog["categories"]
    }
    memberships = {
        (category["category_id"], child["candidate_id"])
        for category in catalog["categories"]
        for child in category["children"]
    }

    for case in fixture["cases"]:
        expected = case["expected"]
        expected_categories = expected["acceptable_category_ids"]
        expected_candidates = expected["acceptable_candidate_ids"]
        allowed_categories = case["allowed_category_ids"]

        assert case["input"]["text"].strip()
        assert expected_categories
        assert len(expected_categories) == len(set(expected_categories))
        assert len(expected_candidates) == len(set(expected_candidates))
        assert len(allowed_categories) == len(set(allowed_categories))
        assert set(allowed_categories) <= set(categories)

        if expected_categories == ["00"]:
            assert expected["support_status"] == "unsupported"
            assert not expected_candidates
            assert allowed_categories
            continue

        assert expected["support_status"] == "supported"
        assert set(expected_categories) <= set(categories)
        for candidate_id in expected_candidates:
            assert any(
                (category_id, candidate_id) in memberships
                for category_id in expected_categories
            )


def test_changed_category_boundary_cases_are_literal_human_authored_oracles() -> None:
    fixture = _load(FIXTURE_PATH)
    boundary_cases = [
        case
        for case in fixture["cases"]
        if case["group"] == "changed_category_boundary"
    ]
    per_category = Counter(
        case["expected"]["acceptable_category_ids"][0]
        for case in boundary_cases
    )

    assert set(per_category) == set(fixture["changed_category_ids"])
    assert set(per_category.values()) == {3}
    assert all(case["source"] == "reviewed_boundary" for case in boundary_cases)
    assert all(case["expected"]["acceptable_candidate_ids"] for case in boundary_cases)
    assert all(case["changed_category"] is True for case in boundary_cases)


def test_realtime_action_recall_fixture_matches_the_reviewed_generator_output() -> None:
    fixture = _load(FIXTURE_PATH)
    catalog = _load(CATALOG_PATH)

    assert fixture["cases"] == build_cases(catalog)
