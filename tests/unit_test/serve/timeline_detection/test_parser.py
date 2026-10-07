from __future__ import annotations

import pytest

from sglang_omni.serve.timeline_detection.parser import (
    ModelOutputError,
    parse_event_output,
)
from sglang_omni.serve.timeline_detection.prompt import EVENT_IDS


def test_catalog_contains_the_43_reviewed_product_events() -> None:
    assert len(EVENT_IDS) == 43
    assert EVENT_IDS[:3] == ("E01", "E02", "E03")
    assert "E20" not in EVENT_IDS
    assert "E21" not in EVENT_IDS
    assert EVENT_IDS[-1] == "E45"


def test_canonicalizes_order_duplicates_unknowns_and_conflicts() -> None:
    assert parse_event_output(
        '{"e":["E44","E11","E05","E05","NOPE","E01"]}'
    ) == ["E01", "E05", "E44"]
    assert parse_event_output('{"e":["E09","E10"]}') == ["E10"]
    assert parse_event_output(
        '{"e":["E08","E24","E29","E33"]}'
    ) == ["E33"]


def test_recovers_one_json_object_from_model_wrapping() -> None:
    assert parse_event_output('```json\n{"e":["E06"]}\n```') == ["E06"]


@pytest.mark.parametrize(
    "raw_output",
    ["", "not json", "[]", '{"event":["E01"]}', '{"e":"E01"}', '{"e":[1]}'],
)
def test_rejects_unusable_output(raw_output: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_event_output(raw_output)
