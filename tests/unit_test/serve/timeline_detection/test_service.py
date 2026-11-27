from __future__ import annotations

import asyncio
import hashlib
import json

from fastapi.testclient import TestClient

from framex_event_policy.windows import DetectionWindow, WindowEvent
from sglang_omni.serve.timeline_detection.service import (
    create_timeline_detection_app,
)


def _agent_timeline_snapshot() -> dict:
    events = [{"code": f"E{index}", "name": f"event-{index}"} for index in range(96)]
    registry = json.dumps({"events": events}, ensure_ascii=False, separators=(",", ":"))
    codebook = "new_code,event_name,source_D97_code,legacy_alias,definition\n" + "".join(
        f"{event['code']},{event['name']},,,{event['name']}\n" for event in events
    )
    texts = {
        "system_prompt": "AGENT EVENT POLICY",
        "event_registry_json": registry,
        "event_codebook_csv": codebook,
    }
    return {
        "consumer": "d_video_call.timeline.detection",
        "release_id": "timeline-test",
        "bundle_sha256": "b" * 64,
        "binding": {
            "assembly_contract": "d-timeline-detection-v1",
            "output_protocol": "event1115-code-v1",
            "model_id": "timeline-model",
            "model_version": "sha256:456",
        },
        **texts,
        "item_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in texts.items()
        },
    }


def _session_start(*, contract_version: int = 3) -> dict:
    return {
        "type": "session.start",
        "contract_version": contract_version,
        "session_id": "session-1",
        "identity_epoch": 2,
        "observer_epoch": 4,
        "stream_epoch": 3,
        "next_sequence": 1,
        "audio_format": "pcm16/16000/mono",
        "video_format": "jpeg",
        "model_id": "timeline-model",
        "prompt_snapshot": _agent_timeline_snapshot(),
    }


class _Model:
    model_id = "timeline-model"
    model_version = "sha256:456"

    def __init__(self) -> None:
        self.closed = False
        self.observations = asyncio.Queue()
        self.contract_version = 3

    async def start(self, request):
        self.contract_version = request.contract_version

    async def append(self, chunk):
        await self.observations.put(DetectionWindow(
            session_id=chunk.session_id,
            identity_epoch=chunk.identity_epoch,
            observer_epoch=chunk.observer_epoch,
            stream_epoch=chunk.stream_epoch,
            window_sequence=chunk.sequence,
            evidence_start_ms=chunk.start_ms,
            evidence_end_ms=chunk.end_ms,
            model_id=self.model_id,
            model_version=self.model_version,
            evidence_mode="audio_video",
            audio_status="complete",
            events=(WindowEvent(event_type="E32", summary="入画"),),
        ))

    async def next_observations(self):
        return await self.observations.get()

    async def discontinuity(self, event):
        return None

    async def close(self):
        self.closed = True


class _SlowModel(_Model):
    async def next_observations(self):
        await asyncio.sleep(0.05)
        return await super().next_observations()


def test_binary_media_frame_emits_fenced_observation() -> None:
    model = _Model()
    app = create_timeline_detection_app(lambda start: model, token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(_session_start())
            ready = websocket.receive_json()
            assert ready["type"] == "session.ready"
            assert ready["contract_version"] == 3
            assert "result_mode" not in ready
            websocket.send_json(
                {
                    "type": "media",
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "sequence": 1,
                    "kind": "video",
                    "start_ms": 100,
                    "end_ms": 200,
                    "encoding": "jpeg",
                    "payload_bytes": 3,
                }
            )
            websocket.send_bytes(b"abc")
            first_result = websocket.receive_json()
            second_result = websocket.receive_json()
            results = {
                first_result["type"]: first_result,
                second_result["type"]: second_result,
            }
            websocket.send_json({"type": "session.close"})
            closed = websocket.receive_json()

    event = results["window.result"]
    ack = results["media.ack"]
    assert event["type"] == "window.result"
    assert event["evidence_end_ms"] == 200
    assert event["evidence_mode"] == "audio_video"
    assert event["audio_status"] == "complete"
    assert event["events"] == [{"event_type": "E32", "summary": "入画"}]
    assert ack == {"type": "media.ack", "observer_epoch": 4, "sequence": 1}
    assert closed["type"] == "session.closed"
    assert model.closed


def test_health_is_authenticated() -> None:
    app = create_timeline_detection_app(lambda start: _Model(), token="secret")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 404
        response = client.get(
            "/health", headers={"Authorization": "Bearer secret"}
        )
    assert response.json() == {"ok": True, "contract_version": 3}


def test_rejects_obsolete_contract_version() -> None:
    app = create_timeline_detection_app(lambda start: _Model(), token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(_session_start(contract_version=2))

            error = websocket.receive_json()

    assert error == {
        "type": "error",
        "code": "invalid_request",
        "detail": "unsupported timeline contract version",
    }


def test_media_ack_does_not_wait_for_slow_model_inference() -> None:
    app = create_timeline_detection_app(lambda start: _SlowModel(), token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(_session_start())
            assert websocket.receive_json()["type"] == "session.ready"
            websocket.send_json(
                {
                    "type": "media",
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "sequence": 1,
                    "kind": "video",
                    "start_ms": 100,
                    "end_ms": 200,
                    "encoding": "jpeg",
                    "payload_bytes": 3,
                }
            )
            websocket.send_bytes(b"abc")

            assert websocket.receive_json() == {
                "type": "media.ack",
                "observer_epoch": 4,
                "sequence": 1,
            }
            assert websocket.receive_json()["type"] == "window.result"


def test_graceful_close_drains_acknowledged_media_ingest() -> None:
    app = create_timeline_detection_app(lambda start: _SlowModel(), token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(_session_start())
            assert websocket.receive_json()["type"] == "session.ready"
            websocket.send_json(
                {
                    "type": "media",
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "sequence": 1,
                    "kind": "video",
                    "start_ms": 100,
                    "end_ms": 200,
                    "encoding": "jpeg",
                    "payload_bytes": 3,
                }
            )
            websocket.send_bytes(b"abc")
            assert websocket.receive_json() == {
                "type": "media.ack",
                "observer_epoch": 4,
                "sequence": 1,
            }
            websocket.send_json({"type": "session.close"})

            closed = websocket.receive_json()

    assert closed["type"] == "session.closed"
