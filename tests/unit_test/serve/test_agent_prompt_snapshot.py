from __future__ import annotations

import csv
import hashlib
import io
import json

import pytest

from sglang_omni.serve.action_decision.catalog import ActionEntry, ResolvedCatalog
from sglang_omni.serve.action_decision.contracts import (
    ActionDecisionRequest,
    DecisionChannel,
)
from sglang_omni.serve.action_decision.prompt import build_system_prompt
from sglang_omni.serve.action_decision.service import _with_prompt_receipt
from sglang_omni.serve.agent_prompt_snapshot import (
    decode_action_prompt_snapshot,
    decode_router_prompt_snapshot,
    decode_timeline_prompt_snapshot,
)
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile
from sglang_omni.serve.timeline_detection.event1115 import (
    event1115_names_from_agent_snapshot,
    parse_event1115_output,
)
from sglang_omni.serve.timeline_detection.parser import ModelOutputError
from sglang_omni.serve.timeline_detection.service import _start_from_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _action_snapshot(prompt: str = "AGENT OWNED ACTION POLICY") -> dict:
    return {
        "consumer": "d_video_call.action.decision",
        "release_id": "action-v10",
        "bundle_sha256": "a" * 64,
        "binding": {
            "assembly_contract": "d-action-decision-v1",
            "output_protocol": "d-action-decision-response-v1",
            "model_id": "action-model",
            "model_version": "action-v10",
        },
        "system_prompt": prompt,
        "system_prompt_sha256": _sha(prompt),
    }


def _timeline_documents() -> tuple[str, str, str]:
    events = [{"code": f"E{index}", "name": f"event-{index}"} for index in range(96)]
    registry = json.dumps({"events": events}, ensure_ascii=False, separators=(",", ":"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("code", "name"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    codebook = output.getvalue()
    return f"AGENT EVENT POLICY\n{registry}\nEND", registry, codebook


def _timeline_snapshot() -> dict:
    prompt, registry, codebook = _timeline_documents()
    texts = {
        "system_prompt": prompt,
        "event_registry_json": registry,
        "event_codebook_csv": codebook,
    }
    return {
        "consumer": "d_video_call.timeline.detection",
        "release_id": "timeline-1115-v1",
        "bundle_sha256": "b" * 64,
        "binding": {
            "assembly_contract": "d-timeline-detection-v1",
            "output_protocol": "d-timeline-observation-v2",
            "model_id": "event-model",
            "model_version": "checkpoint-6138-fp8",
        },
        **texts,
        "item_sha256": {key: _sha(value) for key, value in texts.items()},
    }


def _router_snapshot() -> dict:
    texts = {
        "system_prompt": "AGENT ROUTER POLICY",
        "mcp_scope": "MCP scope",
        "multimodal_scope": "Multimodal scope",
    }
    return {
        "consumer": "sglang_omni.router.v1_0",
        "release_id": "router-v1",
        "bundle_sha256": "c" * 64,
        "binding": {
            "assembly_contract": "turn-router-v1.0",
            "output_protocol": "turn-router-code-v1.0",
            "model_id": "router-model",
            "model_version": "router-weights",
        },
        **texts,
        "item_sha256": {key: _sha(value) for key, value in texts.items()},
    }


def test_router_snapshot_is_validated_for_request_owned_prompting() -> None:
    snapshot = decode_router_prompt_snapshot(_router_snapshot())

    assert snapshot.release_id == "router-v1"
    assert snapshot.binding.model_id == "router-model"
    assert snapshot.rules["system_prompt"] == "AGENT ROUTER POLICY"

    corrupted = _router_snapshot()
    corrupted["mcp_scope"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        decode_router_prompt_snapshot(corrupted)


def test_action_snapshot_is_validated_and_drives_prompt_assembly() -> None:
    snapshot = decode_action_prompt_snapshot(
        _action_snapshot(), model_id="action-model", model_version="action-v10"
    )
    request = ActionDecisionRequest(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        decision_point_id="decision-1",
        channel=DecisionChannel.BODY,
        text="wave",
        character_prompt="friendly",
        prompt_snapshot=snapshot,
    )
    entry = ActionEntry(
        candidate_id="wave",
        action_id="wave",
        category_id="body",
        label="挥手",
        definition="挥手问候",
        code="_a_b",
        token_ids=(1, 2),
        kind="action",
    )
    catalog = ResolvedCatalog(
        entries=(entry,),
        runtime_context={},
        hard_bound=None,
        catalog_version="catalog-v1",
        catalog_hash="catalog-sha",
        mapping_version="mapping-v1",
        mapping_hash="mapping-sha",
    )

    prompt = build_system_prompt(request, catalog, prompt_snapshot=dict(snapshot.rules))

    assert prompt.startswith("AGENT OWNED ACTION POLICY\n")
    assert "_a_b | wave | 挥手 | 挥手问候" in prompt
    assert "你是数字人的动作决策模型" not in prompt
    assert _with_prompt_receipt({"ok": True}, request) == {
        "ok": True,
        "prompt_release_id": "action-v10",
        "prompt_bundle_sha256": "a" * 64,
    }


def test_action_snapshot_rejects_content_or_model_drift() -> None:
    corrupted = _action_snapshot()
    corrupted["system_prompt"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        decode_action_prompt_snapshot(
            corrupted, model_id="action-model", model_version="action-v10"
        )
    with pytest.raises(ValueError, match="binding"):
        decode_action_prompt_snapshot(
            _action_snapshot(), model_id="other-model", model_version="action-v10"
        )


def test_timeline_session_uses_agent_prompt_registry_and_receipt_identity() -> None:
    raw_snapshot = _timeline_snapshot()
    start = _start_from_json(
        {
            "type": "session.start",
            "contract_version": 2,
            "session_id": "session-1",
            "identity_epoch": 1,
            "observer_epoch": 2,
            "stream_epoch": 3,
            "next_sequence": 1,
            "audio_format": "pcm_s16le/16000/1",
            "video_format": "jpeg",
            "model_id": "event-model",
            "prompt_snapshot": raw_snapshot,
        }
    )

    assert start.prompt_snapshot is not None
    assert start.prompt_snapshot.release_id == "timeline-1115-v1"
    names = event1115_names_from_agent_snapshot(
        start.prompt_snapshot.system_prompt,
        start.prompt_snapshot.event_registry_json,
        start.prompt_snapshot.event_codebook_csv,
    )
    assert names["E0"] == "event-0"
    assert names["E95"] == "event-95"
    assert parse_event1115_output("E95", allowed_event_ids=frozenset(names)) == ["E95"]
    with pytest.raises(ModelOutputError, match="outside"):
        parse_event1115_output("E96", allowed_event_ids=frozenset(names))


def test_timeline_snapshot_rejects_item_checksum_drift() -> None:
    raw = _timeline_snapshot()
    raw["event_registry_json"] = "{}"
    with pytest.raises(ValueError, match="checksum"):
        decode_timeline_prompt_snapshot(raw, model_id="event-model")


def test_timeline_profile_uses_agent_documents_without_local_prompt_files(
    monkeypatch,
) -> None:
    snapshot = decode_timeline_prompt_snapshot(
        _timeline_snapshot(), model_id="event-model"
    )
    monkeypatch.setenv("SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL", "direct-1115-v1")
    monkeypatch.delenv("SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH", raising=False)

    profile = DirectE50Profile.from_environment(
        snapshot,
        model_id="event-model",
        model_version="checkpoint-6138-fp8",
    )

    assert profile is not None
    assert profile.system_prompt == snapshot.system_prompt
    assert profile.prompt_version == "timeline-1115-v1"
    assert profile.allowed_event_ids == frozenset(f"E{index}" for index in range(96))
