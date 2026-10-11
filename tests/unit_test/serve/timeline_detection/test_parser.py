from __future__ import annotations

import hashlib

import pytest

from sglang_omni.serve.timeline_detection.parser import (
    ModelOutputError,
    parse_event_output,
)
from sglang_omni.serve.timeline_detection.prompt import (
    ALIAS_TO_P,
    EVENT_IDS,
    SYSTEM_PROMPT,
)


def test_catalog_contains_50_e_events_and_keeps_e27_retired() -> None:
    assert len(EVENT_IDS) == 50
    assert EVENT_IDS[:3] == ("E0", "E1", "E2")
    assert "E27" not in EVENT_IDS
    assert EVENT_IDS[-1] == "E50"
    assert "转身并背对" in SYSTEM_PROMPT


def test_maps_v25_single_token_aliases_to_unique_e_codes() -> None:
    assert ALIAS_TO_P["A"] == "P0"
    assert ALIAS_TO_P["d"] == "P28"
    assert parse_event_output("A,d,z") == ["E0", "E28", "E49"]


def test_v25_thumbs_up_and_ok_are_both_mapped_to_current_ok_event() -> None:
    assert parse_event_output("O") == ["E14"]
    assert parse_event_output("P") == ["E14"]
    assert parse_event_output("P13,P14") == ["E14"]


def test_model_prompt_is_byte_exact_v25_frozen_profile() -> None:
    assert len(SYSTEM_PROMPT) == 8544
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == (
        "9b14e51306c60b4d15d8be71da2f22f8d23af60457edee86909573e06f704745"
    )


def test_accepts_transitional_p_and_future_e_outputs() -> None:
    assert parse_event_output("P0,P28") == ["E0", "E28"]
    assert parse_event_output("E0,E28") == ["E0", "E28"]
    assert parse_event_output("P50") == ["E50"]
    assert parse_event_output("E50") == ["E50"]


def test_preserves_occurrence_counts_but_deduplicates_state_events() -> None:
    assert parse_event_output("I,I,A,A") == ["E0", "E8", "E8"]
    assert parse_event_output("o,o") == ["E38", "E38"]


def test_filters_retired_27_and_applies_conflicts() -> None:
    assert parse_event_output("c") == []
    assert parse_event_output("P27") == []
    assert parse_event_output("E27") == []
    assert parse_event_output("J,y") == ["E48"]
    assert parse_event_output("Z,l,S") == ["E17"]


def test_recovers_one_compact_value_from_code_fence() -> None:
    assert parse_event_output("```text\nF\n```") == ["E5"]


@pytest.mark.parametrize(
    "raw_output",
    ["", "not-a-code", "[]", "A,?", "E51", "P51", "A, d"],
)
def test_rejects_unusable_output(raw_output: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_event_output(raw_output)


def test_rejects_pathological_occurrence_repetition() -> None:
    with pytest.raises(ModelOutputError, match="repeated"):
        parse_event_output(",".join("I" for _ in range(9)))
