from __future__ import annotations

import json
from pathlib import Path

from scripts.qwen3_omni_action_scheme_comparison import compact_candidates
from scripts.qwen3_omni_atomic_action_smoke import (
    build_catalog_candidates,
    build_common_action_scenarios,
    load_action_catalog,
)


CATALOG = Path(__file__).parents[3] / "tests/data/actions/character_action_catalog.json"


def test_catalog_backed_smoke_candidates_use_canonical_actions_and_bindings():
    catalog = load_action_catalog(str(CATALOG))
    selection = json.loads((CATALOG.parent / "common_actions_20/selection.json").read_text())
    labels = [item["label"] for item in selection["actions"]]
    candidates = build_catalog_candidates(catalog, labels)
    by_id = {item["candidate_id"]: item for item in candidates}

    assert len(catalog) == 386
    assert len(labels) == 20
    assert [item["candidate_id"] for item in candidates[:-1]] == labels
    assert len(candidates) == len(labels) + 1
    assert by_id["大笑"]["source_label"] == "大笑"
    assert by_id["双手比心"]["source_label"] == "双手比心"
    assert by_id["no_action"]["action_id"] == "no_action"


def test_common_action_scenarios_have_text_and_audio_inputs():
    catalog = load_action_catalog(str(CATALOG))
    root = CATALOG.parent / "common_actions_20"
    scenarios = build_common_action_scenarios(
        str(root),
        "/opt/sglang-omni/tests/data/actions/common_actions_20",
        catalog,
        variant_index=1,
    )
    candidates = build_catalog_candidates(
        catalog, [scenario["source_label"] for scenario in scenarios]
    )

    assert len(scenarios) == 20
    assert len(candidates) == 21
    assert all(
        isinstance(scenario["turns"][1]["content"], str)
        and len(scenario["turns"][1]["audios"]) == 1
        and scenario["turns"][1]["audios"][0].endswith(".wav")
        for scenario in scenarios
    )
    assert all(
        scenario["expected_catalog_action_id"]
        == catalog[scenario["source_label"]]["action_id"]
        for scenario in scenarios
    )


def test_short_id_candidates_preserve_selection_order_and_mapping():
    catalog = load_action_catalog(str(CATALOG))
    selection = json.loads((CATALOG.parent / "common_actions_20/selection.json").read_text())
    labels = [item["label"] for item in selection["actions"]]
    candidates = compact_candidates(catalog, labels)

    assert len(candidates) == 21
    assert [item["candidate_id"] for item in candidates[:-1]] == [
        f"a{index:02d}" for index in range(1, 21)
    ]
    assert [item["source_label"] for item in candidates[:-1]] == labels
    assert [item["suffix"] for item in candidates] == [
        *[f"a{index:02d}" for index in range(1, 21)],
        "none",
    ]
    assert candidates[-1]["action_id"] == "no_action"
