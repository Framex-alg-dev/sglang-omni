from __future__ import annotations

import hashlib

import msgpack
import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.task_classification import service as classification_service
from sglang_omni.serve.task_classification.contracts import (
    CONTRACT_VERSION,
    DEFAULT_BRAIN1_CAPABILITIES,
    DEFAULT_BRAIN2_CAPABILITIES,
)
from sglang_omni.serve.task_classification.pipeline import TaskClassificationPipeline
from sglang_omni.serve.task_classification.service import (
    _request_from_json,
    create_task_classification_app,
)


class _Model:
    model_id = "turn-router"
    model_version = "prompt-v1"

    async def classify(self, request):
        return "1" if request.text == "book a ticket" else "0"


def _prompt_snapshot() -> dict:
    rules = {
        "system_prompt": "Agent-owned Router system prompt",
        "mcp_scope": "MCP scope",
        "multimodal_scope": "Multimodal scope",
    }
    return {
        "consumer": "sglang_omni.router.v1_0",
        "release_id": "router-release-v1",
        "bundle_sha256": "a" * 64,
        "binding": {
            "assembly_contract": "turn-router-v1.0",
            "output_protocol": "turn-router-code-v1.0",
            "model_id": "turn-router",
            "model_version": "prompt-v1",
        },
        **rules,
        "item_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in rules.items()
        },
    }


_DEFAULT_PROMPT = object()


def _body(
    text: str,
    *,
    contract_version: int = CONTRACT_VERSION,
    prompt_snapshot: dict | None | object = _DEFAULT_PROMPT,
) -> bytes:
    if prompt_snapshot is _DEFAULT_PROMPT:
        prompt_snapshot = _prompt_snapshot()
    payload = {
        "contract_version": contract_version,
        "request_id": "request-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "identity_epoch": 2,
        "input_revision": 4,
        "text": text,
        "media": [],
        "router_history": [],
        "brain1_capabilities": DEFAULT_BRAIN1_CAPABILITIES,
        "brain2_capabilities": DEFAULT_BRAIN2_CAPABILITIES,
        "has_active_agent": False,
        "pending_confirmation": False,
        "follow_up_required": False,
    }
    if prompt_snapshot is not None:
        payload["prompt_snapshot"] = prompt_snapshot
    return msgpack.packb(
        payload,
        use_bin_type=True,
    )


def test_requires_the_exact_request_contract() -> None:
    raw = msgpack.unpackb(_body("你会唱歌吗"), raw=False)
    parsed = _request_from_json(raw)

    assert parsed.brain1_capabilities == DEFAULT_BRAIN1_CAPABILITIES
    assert parsed.brain2_capabilities == DEFAULT_BRAIN2_CAPABILITIES
    assert parsed.router_profile == "v1.0"
    assert parsed.contract_version == CONTRACT_VERSION
    assert parsed.prompt_snapshot is not None
    assert "曲库查询" in parsed.brain2_capabilities
    raw["router_profile"] = "unknown"
    try:
        _request_from_json(raw)
    except ValueError as exc:
        assert str(exc) == "unsupported turn-router profile"
    else:
        raise AssertionError("unknown router profiles must be rejected")
    raw["router_profile"] = "v1.0"
    raw.pop("brain2_capabilities")
    try:
        _request_from_json(raw)
    except ValueError as exc:
        assert str(exc) == "turn-router request fields are invalid"
    else:
        raise AssertionError("missing request fields must be rejected")


def test_authenticated_turn_router_contract(monkeypatch) -> None:
    timing_events = []
    monkeypatch.setattr(
        classification_service,
        "emit_decision_timing",
        lambda event, **fields: timing_events.append((event, fields)) or True,
    )
    app = create_task_classification_app(
        TaskClassificationPipeline(_Model()),
        token="secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/task-classification",
            content=_body("book a ticket"),
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/vnd.sglang-omni.msgpack",
            },
        )

    assert response.status_code == 200
    assert response.json()["route_token"] == "delegate"
    assert response.json()["route"] == "BRAIN2"
    assert response.json()["route_code"] == 1
    assert response.json()["needs_mcp"] is True
    assert response.json()["needs_multimodal"] is False
    assert response.json()["input_revision"] == 4
    assert response.json()["contract_version"] == CONTRACT_VERSION
    assert response.json()["request_cue"] is None
    assert [event for event, _ in timing_events] == [
        "decision_request_received",
        "decision_response_completed",
        "decision_response_sent",
    ]
    for _, fields in timing_events:
        assert fields["request_id"] == "request-1"
        assert fields["metadata"]["session_id"] == "session-1"
        assert fields["metadata"]["turn_id"] == "turn-1"
        assert "text" not in fields["metadata"]


def test_v7_accepts_agent_prompt_snapshot_and_echoes_contract() -> None:
    app = create_task_classification_app(
        TaskClassificationPipeline(_Model()),
        token="secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/task-classification",
            content=_body(
                "book a ticket",
                contract_version=CONTRACT_VERSION,
                prompt_snapshot=_prompt_snapshot(),
            ),
            headers={"Authorization": "Bearer secret"},
        )

    assert response.status_code == 200
    assert response.json()["contract_version"] == CONTRACT_VERSION


def test_v7_requires_snapshot_and_v6_is_retired() -> None:
    v7_without_snapshot = msgpack.unpackb(
        _body("hello", prompt_snapshot=None), raw=False
    )
    with pytest.raises(ValueError, match="v7 requires Agent prompt_snapshot"):
        _request_from_json(v7_without_snapshot)

    v6_with_snapshot = msgpack.unpackb(
        _body(
            "hello",
            contract_version=6,
            prompt_snapshot=_prompt_snapshot(),
        ),
        raw=False,
    )
    with pytest.raises(ValueError, match="unsupported turn-router contract version"):
        _request_from_json(v6_with_snapshot)


def test_websocket_streams_binary_media_before_classification(monkeypatch) -> None:
    traces = []
    monkeypatch.setattr(
        classification_service,
        "emit_model_trace",
        lambda stream, event, **fields: traces.append((stream, event, fields)),
    )
    class _MediaModel(_Model):
        async def classify(self, request):
            assert request.text is None
            assert request.media[0].payload == raw_media
            return "3"

    raw_media = b"\x01\x00" * 160
    app = create_task_classification_app(
        TaskClassificationPipeline(_MediaModel()),
        token="secret",
    )
    start_payload = msgpack.unpackb(_body("book a ticket"), raw=False)
    start_payload["text"] = None
    start_payload.pop("media")
    start_payload["log_context"] = {
        "request_id": "trace-request-1",
        "root_session_id": "root-1",
        "turn_id": "turn-1",
        "agent_session_id": "agent-1",
    }
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/task-classification/realtime",
            headers={"Authorization": "Bearer secret"},
        ) as socket:
            socket.send_json(
                {
                    "type": "request.start",
                    "contract_version": 2,
                    "request_id": "request-1",
                    "payload": start_payload,
                }
            )
            assert socket.receive_json() == {
                "type": "request.ready",
                "request_id": "request-1",
                "contract_version": 2,
            }
            socket.send_json(
                {
                    "type": "input.media",
                    "request_id": "request-1",
                    "media_id": "audio-1",
                    "kind": "audio",
                    "start_ms": 0,
                    "end_ms": 20,
                    "encoding": "pcm_s16le",
                    "checksum": "sha256:" + hashlib.sha256(raw_media).hexdigest(),
                    "payload_bytes": len(raw_media),
                    "evidence_role": "user_audio",
                }
            )
            socket.send_bytes(raw_media)
            assert socket.receive_json()["type"] == "input.media.ack"
            socket.send_json({"type": "request.commit", "request_id": "request-1"})
            completed = socket.receive_json()

    assert completed["type"] == "response.completed"
    assert completed["response"]["route"] == "BRAIN2"
    assert completed["response"]["route_code"] == 3
    assert completed["response"]["needs_multimodal"] is True
    router_traces = [
        fields
        for stream, event, fields in traces
        if stream == "reply"
        and event in {
            "agent.request.received",
            "router.resolved",
            "agent.response.sent",
        }
    ]
    assert len(router_traces) == 3
    assert all(item["request_id"] == "trace-request-1" for item in router_traces)
    assert all(item["root_session_id"] == "root-1" for item in router_traces)
    assert all(item["agent_session_id"] == "agent-1" for item in router_traces)
    assert all(item["turn_id"] == "turn-1" for item in router_traces)


def test_websocket_validation_failure_keeps_session_and_turn_context(
    monkeypatch,
) -> None:
    events = []
    monkeypatch.setattr(
        classification_service,
        "emit_service_event",
        lambda event_service, event, **fields: events.append(
            (event_service, event, fields)
        ),
    )
    app = create_task_classification_app(
        TaskClassificationPipeline(_Model()),
        token="secret",
    )
    start_payload = msgpack.unpackb(_body("hello"), raw=False)
    start_payload["brain1_capabilities"] = []
    start_payload["log_context"] = {
        "request_id": "trace-invalid-1",
        "root_session_id": "root-invalid-1",
        "turn_id": "turn-invalid-1",
        "agent_session_id": "agent-invalid-1",
    }

    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/task-classification/realtime",
            headers={"Authorization": "Bearer secret"},
        ) as socket:
            socket.send_json(
                {
                    "type": "request.start",
                    "contract_version": 2,
                    "request_id": "request-invalid-1",
                    "payload": start_payload,
                }
            )
            assert socket.receive_json()["type"] == "request.ready"
            socket.send_json(
                {"type": "request.commit", "request_id": "request-invalid-1"}
            )
            assert socket.receive_json()["type"] == "error"

    _, _, fields = next(item for item in events if item[1] == "ws.request.failed")
    assert fields["request_id"] == "request-invalid-1"
    assert fields["root_session_id"] == "root-invalid-1"
    assert fields["turn_id"] == "turn-invalid-1"


def test_invalid_model_output_is_upstream_failure() -> None:
    class _InvalidModel(_Model):
        async def classify(self, request):
            return "delegate\n"

    app = create_task_classification_app(
        TaskClassificationPipeline(_InvalidModel()),
        token="secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/task-classification",
            content=_body("book a ticket"),
            headers={"Authorization": "Bearer secret"},
        )

    assert response.status_code == 502


def test_unauthenticated_route_is_hidden() -> None:
    app = create_task_classification_app(
        TaskClassificationPipeline(_Model()),
        token="secret",
    )
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 404
