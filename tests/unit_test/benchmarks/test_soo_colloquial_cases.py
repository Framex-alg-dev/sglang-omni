import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASE_PATH = ROOT / "benchmarks/eval/soo_colloquial_20_turns.json"


def test_soo_suite_has_twenty_ordered_distinct_spoken_turns():
    suite = json.loads(CASE_PATH.read_text())
    turns = suite["turns"]
    assert suite["character_id"] == "character_4ef8a49978444c9aba77fc7293fb5984"
    assert len(turns) == suite["run_contract"]["turns_per_session"] == 20
    assert [t["id"] for t in turns] == [f"{i:02d}" for i in range(1, 21)]
    assert len({t["prompt"] for t in turns}) == 20
    assert all(t["scene"] and t["manual_assertions"] and t["coverage_targets"] for t in turns)
    assert suite["run_contract"]["never_send_prompt_as_transcript_with_audio"]


def test_soo_suite_candidates_exist_in_catalog():
    suite = json.loads(CASE_PATH.read_text())
    catalog = json.loads((ROOT / "sglang_omni/assets/character_action_global_catalog.json").read_text())
    ids = {a["candidate_id"] for c in catalog["categories"] for a in c["children"]}
    categories = {c["category_id"]: {a["candidate_id"] for a in c["children"]}
                  for c in catalog["categories"]}
    for t in suite["turns"]:
        for key in ("expected_action_candidates", "expected_expression_candidates"):
            assert set(t.get(key, [])) <= ids
        if t.get("expected_action_categories"):
            valid = set().union(*(categories[c] for c in t["expected_action_categories"]))
            assert set(t["expected_action_candidates"]) <= valid


def test_soo_suite_keeps_visual_history_and_disabled_numeric_regressions():
    turns = json.loads(CASE_PATH.read_text())["turns"]
    assert turns[15]["expected_action_candidates"] == ["274"]
    assert turns[16]["image_scenes"] == ["number_two_gesture", "number_one_gesture"]
    assert "numeric_override_disabled" in turns[16]["coverage_targets"]
    assert "expected_action_candidates" not in turns[16]
    assert "long_history_read" in turns[19]["coverage_targets"]
