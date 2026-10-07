from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from sglang_omni.serve.timeline_detection.contracts import ObservationEvent
from sglang_omni.serve.timeline_detection.service import (
    create_timeline_detection_app,
)


class _Model:
    model_id = "timeline-model"
    model_version = "sha256:456"

    def __init__(self) -> None:
        self.closed = False
        self.observations = asyncio.Queue()
        self.contract_version = 2

    async def start(self, request):
        self.contract_version = request.contract_version

    async def append(self, chunk):
        await self.observations.put((
            ObservationEvent(
                observation_id=f"observation-{chunk.sequence}",
                session_id=chunk.session_id,
                identity_epoch=chunk.identity_epoch,
                observer_epoch=chunk.observer_epoch,
                stream_epoch=chunk.stream_epoch,
                event_type="E32",
                summary="入画",
                evidence_start_ms=chunk.start_ms,
                evidence_end_ms=chunk.end_ms,
                model_id=self.model_id,
                model_version=self.model_version,
                contract_version=self.contract_version,
            ),
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
            websocket.send_json(
                {
                    "type": "session.start",
                    "contract_version": 2,
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "next_sequence": 1,
                    "audio_format": "pcm16/16000/mono",
                    "video_format": "jpeg",
                    "model_id": "timeline-model",
                }
            )
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
            first_result = websocket.receive_json()
            second_result = websocket.receive_json()
            results = {
                first_result["type"]: first_result,
                second_result["type"]: second_result,
            }
            websocket.send_json({"type": "session.close"})
            closed = websocket.receive_json()

    event = results["observation"]
    ack = results["media.ack"]
    assert event["type"] == "observation"
    assert event["evidence_end_ms"] == 200
    assert event["evidence_mode"] == "audio_video"
    assert event["audio_status"] == "complete"
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
    assert response.json() == {"ok": True, "contract_version": 2}


def test_rejects_obsolete_contract_version() -> None:
    app = create_timeline_detection_app(lambda start: _Model(), token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(
                {
                    "type": "session.start",
                    "contract_version": 1,
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "next_sequence": 1,
                    "audio_format": "pcm16/16000/mono",
                    "video_format": "jpeg",
                    "model_id": "timeline-model",
                }
            )

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
            websocket.send_json(
                {
                    "type": "session.start",
                    "contract_version": 2,
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "next_sequence": 1,
                    "audio_format": "pcm16/16000/mono",
                    "video_format": "jpeg",
                    "model_id": "timeline-model",
                }
            )
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
            assert websocket.receive_json()["type"] == "observation"


def test_graceful_close_drains_acknowledged_media_ingest() -> None:
    app = create_timeline_detection_app(lambda start: _SlowModel(), token="secret")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/timeline", headers={"Authorization": "Bearer secret"}
        ) as websocket:
            websocket.send_json(
                {
                    "type": "session.start",
                    "contract_version": 2,
                    "session_id": "session-1",
                    "identity_epoch": 2,
                    "observer_epoch": 4,
                    "stream_epoch": 3,
                    "next_sequence": 1,
                    "audio_format": "pcm16/16000/mono",
                    "video_format": "jpeg",
                    "model_id": "timeline-model",
                }
            )
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
