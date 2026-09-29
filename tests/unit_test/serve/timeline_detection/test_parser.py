from __future__ import annotations

import pytest

from sglang_omni.serve.timeline_detection.parser import (
    ModelOutputError,
    parse_event_output,
)


def test_canonicalizes_order_duplicates_unknowns_and_conflicts() -> None:
    assert parse_event_output(
        '{"e":["X1","D1","O2","O2","NOPE","P1"]}'
    ) == ["P1", "O2", "X1"]
    assert parse_event_output('{"e":["D1","B3","M3"]}') == ["B3", "M3"]
    assert parse_event_output('{"e":["P2","B3","M3"]}') == ["P2", "M3"]


def test_recovers_one_json_object_from_model_wrapping() -> None:
    assert parse_event_output('```json\n{"e":["B1"]}\n```') == ["B1"]


@pytest.mark.parametrize(
    "raw_output",
    ["", "not json", "[]", '{"event":["P1"]}', '{"e":"P1"}', '{"e":[1]}'],
)
def test_rejects_unusable_output(raw_output: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_event_output(raw_output)
