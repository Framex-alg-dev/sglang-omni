from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_number_one_gesture_case_has_visual_action_expectations() -> None:
    payload = json.loads(
        (
            ROOT
            / "benchmarks"
            / "eval"
            / "realtime_multimodal_user_journey_cases.json"
        ).read_text()
    )
    case = next(turn for turn in payload["turns"] if turn["id"] == "25")

    assert case["scene"] == "number_one_gesture"
    assert case["prompt"] == "请做出这个手势"
    assert case["expected_action_candidates"] == ["258"]
    assert case["expected_action_categories"] == ["31"]

    implicit_case = next(turn for turn in payload["turns"] if turn["id"] == "26")
    assert implicit_case["prompt"] == "请做出手势"
    assert implicit_case["scene"] == "number_one_gesture"
    assert implicit_case["expected_action_candidates"] == ["258"]
    assert implicit_case["expected_action_categories"] == ["31"]

    number_two_case = next(turn for turn in payload["turns"] if turn["id"] == "27")
    assert number_two_case["prompt"] == "请做出这个手势"
    assert number_two_case["scene"] == "number_two_gesture"
    assert number_two_case["expected_action_candidates"] == ["274"]
    assert number_two_case["expected_action_categories"] == ["31"]

    question_case = next(turn for turn in payload["turns"] if turn["id"] == "28")
    assert question_case["prompt"] == "这个是什么手势"
    assert question_case["scene"] == "number_two_gesture"
    assert "expected_action_candidates" not in question_case

    sum_case = next(turn for turn in payload["turns"] if turn["id"] == "29")
    assert sum_case["prompt"] == "这两个加起来等于多少"
    assert sum_case["image_scenes"] == [
        "number_two_gesture",
        "number_one_gesture",
    ]
    assert sum_case["expected_reply_groups"] == [["三", "3", "three"]]
    assert "expected_action_candidates" not in sum_case
    assert sum_case["expected_action_categories"] == ["01"]


def test_number_one_gesture_scene_is_renderable() -> None:
    prepare = _load_script("prepare_realtime_multimodal_user_journey")

    image = prepare._render_scene("number_one_gesture")

    assert image.size == (1024, 768)
    assert image.mode == "RGB"


def test_number_two_gesture_scene_is_renderable() -> None:
    prepare = _load_script("prepare_realtime_multimodal_user_journey")

    image = prepare._render_scene("number_two_gesture")

    assert image.size == (1024, 768)
    assert image.mode == "RGB"


def test_journey_can_select_the_number_one_turn() -> None:
    prepare = _load_script("prepare_realtime_multimodal_user_journey")
    turns = prepare._load_turns(
        ROOT / "benchmarks" / "eval" / "realtime_multimodal_user_journey_cases.json"
    )

    selected = prepare._select_turns(turns, ["25"])

    assert [turn["id"] for turn in selected] == ["25"]


def test_numeric_sum_turn_prepares_and_evaluates_two_camera_images(
    tmp_path: Path,
) -> None:
    prepare = _load_script("prepare_realtime_multimodal_user_journey")
    evaluate = _load_script("evaluate_realtime_multimodal_user_journey")
    turns = prepare._load_turns(
        ROOT / "benchmarks" / "eval" / "realtime_multimodal_user_journey_cases.json"
    )
    case = next(turn for turn in turns if turn["id"] == "29")

    assert prepare._turn_image_scenes(case) == (
        "number_two_gesture",
        "number_one_gesture",
    )
    assert evaluate._case_image_scenes(case) == (
        "number_two_gesture",
        "number_one_gesture",
    )
    prepare._prepare_images([case], tmp_path)
    assert (tmp_path / "number_two_gesture.jpg").is_file()
    assert (tmp_path / "number_one_gesture.jpg").is_file()


def test_action_evaluation_checks_expected_category() -> None:
    evaluate = _load_script("evaluate_realtime_multimodal_user_journey")
    record = evaluate.TurnRecord(
        case_id="25",
        request_id="request-25",
        prompt="请做出这个手势",
        scene="number_one_gesture",
        image_file="number_one_gesture.jpg",
        audio_file="turn_25.wav",
        image_bytes=1,
        audio_bytes=1,
        audio_chunks=1,
        started_at=0.0,
        transcript="请做出这个手势",
        result={"status": "completed"},
        action={"candidate_id": "258", "category_id": "30"},
        completed_at=1.0,
    )

    evaluate._evaluate_turn(
        record,
        {
            "expected_action_candidates": ["258"],
            "expected_action_categories": ["31"],
        },
    )

    assert not record.motion_pass
    assert record.motion_failures == ["category 30 not in ['31']"]
