from __future__ import annotations

import csv
import hashlib
import json

import pytest

from framex_event_policy.windows import (
    DetectionWindow,
    RAW_WINDOW_MODE,
    WindowEvent,
)
from sglang_omni.serve.timeline_detection.event1115 import (
    event1115_allowed_ids,
    event1115_names,
    parse_event1115_output,
)
from sglang_omni.serve.timeline_detection.parser import ModelOutputError
from sglang_omni.serve.timeline_detection.service import _start_from_json


def _install_frozen_codebook(tmp_path, monkeypatch) -> None:
    codes = [f"E{index}" for index in range(96)]
    names = {code: f"event-{code[1:]}" for code in codes}
    system_prompt = "\n".join(
        f"{code} {names[code]}" for code in codes
    ) + "\n"
    files = {
        "system_prompt.txt": system_prompt,
        "registry_1115.json": json.dumps(
            {"codes": codes, "names": names},
            ensure_ascii=False,
        ),
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    with (tmp_path / "codebook.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("new_code", "event_name"))
        writer.writeheader()
        writer.writerows(
            {"new_code": code, "event_name": names[code]} for code in codes
        )
    frozen_names = ("system_prompt.txt", "registry_1115.json", "codebook.csv")
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}\n"
            for name in frozen_names
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH",
        str(tmp_path / "system_prompt.txt"),
    )
    event1115_names.cache_clear()
    event1115_allowed_ids.cache_clear()


def test_raw_window_session_start_matches_agent_contract() -> None:
    start = _start_from_json(
        {
            "type": "session.start",
            "contract_version": 2,
            "session_id": "session-41035",
            "identity_epoch": 3,
            "observer_epoch": 4,
            "stream_epoch": 5,
            "next_sequence": 6,
            "audio_format": "pcm_s16le/16000/1",
            "video_format": "jpeg",
            "model_id": "event-1115-v1-fp8",
            "result_mode": RAW_WINDOW_MODE,
        }
    )

    assert start.result_mode == RAW_WINDOW_MODE
    assert start.observer_epoch == 4
    assert start.next_sequence == 6


@pytest.mark.parametrize(
    "events",
    [(), (WindowEvent(event_type="E95", summary="event-95"),)],
)
def test_raw_window_round_trip_accepts_empty_and_e95(events) -> None:
    window = DetectionWindow(
        session_id="session-41035",
        identity_epoch=1,
        observer_epoch=2,
        stream_epoch=3,
        window_sequence=4,
        evidence_start_ms=500,
        evidence_end_ms=1500,
        model_id="event-1115-v1-fp8",
        model_version="1115_v1-checkpoint-6138-fp8",
        evidence_mode="audio_video",
        audio_status="complete",
        events=events,
    )

    assert DetectionWindow.from_wire(window.to_wire()) == window


def test_1115_output_parser_uses_frozen_96_event_registry(
    tmp_path,
    monkeypatch,
) -> None:
    _install_frozen_codebook(tmp_path, monkeypatch)

    assert parse_event1115_output("N") == []
    assert parse_event1115_output("E0,E22,E95") == ["E0", "E22", "E95"]
    with pytest.raises(ModelOutputError, match="numeric order"):
        parse_event1115_output("E22,E0")
    with pytest.raises(ModelOutputError, match="outside the frozen registry"):
        parse_event1115_output("E96")
