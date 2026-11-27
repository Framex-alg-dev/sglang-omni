from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from sglang_omni.serve.action_decision.recall_audit import input_snapshot
from sglang_omni.serve.action_decision.service import (
    _adopt_stream_log_request_id,
    _parse_request,
    _request_digest,
)


def _wire_request() -> dict:
    return {
        "request_id": "operation-1",
        "session_id": "legacy-agent-session",
        "turn_id": "turn-1",
        "decision_point_id": "decision-1",
        "channel": "body",
        "text": "private user content",
        "log_context": {
            "request_id": "req_0123456789abcdfg",
            "root_session_id": "ses_0123456789abcdfg",
            "agent_session_id": "agent-session-1",
        },
    }


def test_action_log_context_is_separate_from_operation_idempotency() -> None:
    config = SimpleNamespace(model_id="action-model", model_version="v1")
    request = _parse_request(
        _wire_request(), media=(), config=config, require_prompt_snapshot=False
    )
    assert request.request_id == "operation-1"
    assert request.trace_request_id == "req_0123456789abcdfg"
    assert request.root_session_id == "ses_0123456789abcdfg"
    assert request.agent_session_id == "agent-session-1"

    retried = replace(
        request,
        trace_request_id="req_z123456789abcdfg",
        root_session_id="ses_other",
        agent_session_id="agent-session-2",
    )
    assert _request_digest(retried) == _request_digest(request)


def test_action_websocket_request_id_is_log_only() -> None:
    payload = _wire_request()
    _adopt_stream_log_request_id(payload, "req_z123456789abcdfg")

    assert payload["request_id"] == "operation-1"
    assert payload["log_context"] == {
        "request_id": "req_z123456789abcdfg",
        "root_session_id": "ses_0123456789abcdfg",
        "agent_session_id": "agent-session-1",
    }


def test_action_input_snapshot_excludes_user_content_and_prompts(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_ACTION_REPLAY_ROOT", raising=False)
    request = SimpleNamespace(
        requested_action=None,
        runtime_context={"decision_mode": "user_request"},
        character_prompt="private character prompt",
        session_prompt="private session prompt",
        text="private user text",
        avatar_current_text="private avatar text",
        avatar_history_text="private history",
        media=(),
    )
    catalog = SimpleNamespace(
        catalog_version="catalog-v1",
        mapping_hash="mapping-hash",
        entries=(),
    )
    snapshot = input_snapshot(
        request,
        catalog,
        "private system prompt",
        [{"type": "text", "text": "private model input"}],
    )
    serialized = repr(snapshot)
    for private in (
        "private character prompt",
        "private session prompt",
        "private user text",
        "private avatar text",
        "private history",
        "private system prompt",
        "private model input",
    ):
        assert private not in serialized
    assert snapshot["system_sha256"]
    assert snapshot["media"] == []
