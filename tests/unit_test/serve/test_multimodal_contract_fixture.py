from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path

from sglang_omni.serve.task_classification.service import _request_from_json
from sglang_omni.serve.inference_gateway.app import _speculative_speech_hash
from sglang_omni.serve.timeline_detection.contracts import ObservationEvent
from sglang_omni.serve.timeline_detection.service import (
    _chunk_from_json,
    _event_to_json,
    _start_from_json,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "multimodal_orchestration_contract_v2.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _timeline_snapshot(model_id: str) -> dict:
    events = [{"code": f"E{index}", "name": f"event-{index}"} for index in range(96)]
    registry = json.dumps({"events": events}, ensure_ascii=False, separators=(",", ":"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("code", "name"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    texts = {
        "system_prompt": f"AGENT EVENT POLICY\n{registry}\nEND",
        "event_registry_json": registry,
        "event_codebook_csv": output.getvalue(),
    }
    return {
        "consumer": "d_video_call.timeline.detection",
        "release_id": "timeline-contract-test",
        "bundle_sha256": "b" * 64,
        "binding": {
            "assembly_contract": "d-timeline-detection-v1",
            "output_protocol": "d-timeline-observation-v2",
            "model_id": model_id,
            "model_version": "fixture-weights",
        },
        **texts,
        "item_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in texts.items()
        },
    }


def test_sg_accepts_classification_fixture() -> None:
    request = _fixture()["task_classification"]["request"]
    wire = dict(request)
    parsed = _request_from_json(wire)

    assert parsed.request_id == "request-contract-1"
    assert parsed.text == "帮我安排下一步"
    assert parsed.media == ()
    assert parsed.brain1_capabilities == (
        "普通聊天、简单问答、计算、下游用户视频理解；"
        "不处理应用能力、曲库或媒体播放查询"
    )
    assert parsed.brain2_capabilities == (
        "搜索、天气、票务、日历、业务服务、多步Agent；"
        "歌曲能力、曲库查询、所有立即唱歌和选歌请求，以及歌曲播放；"
        "唱歌请求不是普通对话或纯动作，媒体控制由确定性控制通道处理"
    )


def test_sg_accepts_timeline_fixture_and_emits_same_observation() -> None:
    timeline = _fixture()["timeline_detection"]
    start_wire = dict(timeline["session_start"])
    start_wire["prompt_snapshot"] = _timeline_snapshot(start_wire["model_id"])
    start = _start_from_json(start_wire)
    payload = base64.b64decode(timeline["media_payload_base64"], validate=True)
    chunk = _chunk_from_json(timeline["media_header"], payload)
    event_values = dict(timeline["observation"])
    event_values.pop("type")
    event = ObservationEvent(**event_values)

    assert start.observer_epoch == 4
    assert chunk.payload == b"jpeg-frame"
    assert _event_to_json(event) == timeline["observation"]


def test_sg_accepts_speculative_speech_generation_fence_fixture() -> None:
    session = _fixture()["inference_session"]
    speculation = session["reply_stage_request"]["speech_speculation"]
    configure = session["speech_configure"]
    commit = session["speech_commit"]

    assert speculation["generation_id"] == configure["generation_id"]
    assert int(speculation["output_epoch"]) == configure["output_epoch"]
    assert configure["instruction"] == commit["instruction"]
    assert commit["text_hash"] == _speculative_speech_hash(
        text="你好",
        voice=commit["voice"],
        instruction=commit["instruction"],
        generation_id=commit["generation_id"],
        output_epoch=commit["output_epoch"],
    )
