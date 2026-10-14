from __future__ import annotations

import hashlib
import msgpack
from fastapi.testclient import TestClient

from sglang_omni.serve.task_classification.contracts import (
    CONTRACT_VERSION,
    DEFAULT_BRAIN1_CAPABILITIES,
    DEFAULT_BRAIN2_CAPABILITIES,
)
from sglang_omni.serve.task_classification import service as classification_service
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


def _body(text: str) -> bytes:
    return msgpack.packb(
        {
            "contract_version": CONTRACT_VERSION,
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
        },
        use_bin_type=True,
    )


def test_requires_the_exact_request_contract() -> None:
    raw = msgpack.unpackb(_body("你会唱歌吗"), raw=False)
    parsed = _request_from_json(raw)

    assert parsed.brain1_capabilities == DEFAULT_BRAIN1_CAPABILITIES
    assert parsed.brain2_capabilities == DEFAULT_BRAIN2_CAPABILITIES
    assert "曲库查询" in parsed.brain2_capabilities
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


def test_websocket_streams_binary_media_before_classification() -> None:
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
            socket.send_json(
                {"type": "request.commit", "request_id": "request-1"}
            )
            completed = socket.receive_json()

    assert completed["type"] == "response.completed"
    assert completed["response"]["route"] == "BRAIN2"
    assert completed["response"]["route_code"] == 3
    assert completed["response"]["needs_multimodal"] is True


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
