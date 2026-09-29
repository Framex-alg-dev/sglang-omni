from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from sglang_omni.serve.inference_gateway import app as gateway_module
from sglang_omni.serve.inference_gateway.app import (
    InferenceGatewayConfig,
    UpstreamStage,
    create_inference_gateway_app,
)
from sglang_omni.serve.inference_gateway.speech_synthesis import (
    GatewaySpeechConfig,
)


class _Upstream:
    def __init__(self) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.media_header = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def send(self, value):
        self.sent.append(value)
        if isinstance(value, bytes):
            await self.events.put(
                json.dumps(
                    {
                        "type": "input.media.ack",
                        "media_id": self.media_header["media_id"],
                    }
                )
            )
            return
        message = json.loads(value)
        if message["type"] == "request.start":
            await self.events.put(json.dumps({"type": "request.ready"}))
        elif message["type"] == "input.media":
            self.media_header = message
        elif message["type"] == "request.commit":
            await self.events.put(
                json.dumps(
                    {
                        "type": "response.completed",
                        "request_id": message["request_id"],
                        "response": {"ok": True},
                    }
                )
            )

    async def recv(self):
        return await self.events.get()


def _config(**kwargs) -> InferenceGatewayConfig:
    return InferenceGatewayConfig(
        stages={stage: UpstreamStage(f"ws://{stage}") for stage in (
            "classifier", "brain", "reply", "body", "expression"
        )},
        **kwargs,
    )


def test_session_registers_media_once_and_routes_multiple_stages(monkeypatch) -> None:
    upstreams = []

    def connect(*_args, **_kwargs):
        upstream = _Upstream()
        upstreams.append(upstream)
        return upstream

    monkeypatch.setattr(gateway_module.websockets, "connect", connect)
    client = TestClient(create_inference_gateway_app(_config()))
    raw = b"jpeg"
    checksum = "sha256:" + hashlib.sha256(raw).hexdigest()

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        assert socket.receive_json()["type"] == "session.ready"
        socket.send_json(
            {
                "type": "media.put",
                "media_id": "frame-1",
                "kind": "image",
                "start_ms": 0,
                "end_ms": 100,
                "encoding": "image/jpeg",
                "checksum": checksum,
                "payload_bytes": len(raw),
            }
        )
        socket.send_bytes(raw)
        assert socket.receive_json() == {"type": "media.ack", "media_id": "frame-1"}
        for request_id, stage in (("r1", "classifier"), ("r2", "brain")):
            socket.send_json(
                {
                    "type": "stage.request",
                    "request_id": request_id,
                    "stage": stage,
                    "payload": {},
                    "media_refs": ["frame-1"],
                }
            )
            assert socket.receive_json()["type"] == "stage.accepted"
            completed = socket.receive_json()
            assert completed["type"] == "stage.completed"
            assert completed["response"] == {"ok": True}

    assert len(upstreams) == 2
    assert all(sum(isinstance(item, bytes) for item in upstream.sent) == 1 for upstream in upstreams)


def test_action_stage_injects_its_channel(monkeypatch) -> None:
    upstreams = []

    def connect(*_args, **_kwargs):
        upstream = _Upstream()
        upstreams.append(upstream)
        return upstream

    monkeypatch.setattr(gateway_module.websockets, "connect", connect)
    client = TestClient(create_inference_gateway_app(_config()))
    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "body-1",
                "stage": "body",
                "payload": {"turn_id": "t1"},
                "media_refs": [],
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.completed"

    request_start = json.loads(upstreams[0].sent[0])
    assert request_start["payload"]["channel"] == "body"


def test_unknown_media_ref_is_rejected_without_upstream(monkeypatch) -> None:
    monkeypatch.setattr(
        gateway_module.websockets,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    client = TestClient(create_inference_gateway_app(_config()))
    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "r1",
                "stage": "brain",
                "payload": {},
                "media_refs": ["missing"],
            }
        )
        error = socket.receive_json()
    assert error["type"] == "stage.error"
    assert error["code"] == "unknown_media_ref"


def test_inactive_media_is_evicted_before_session_limit() -> None:
    client = TestClient(
        create_inference_gateway_app(_config(max_media_items=1))
    )
    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        for index in (1, 2):
            raw = f"frame-{index}".encode()
            socket.send_json(
                {
                    "type": "media.put",
                    "media_id": f"frame-{index}",
                    "kind": "image",
                    "start_ms": index * 100,
                    "end_ms": index * 100 + 50,
                    "encoding": "image/jpeg",
                    "checksum": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "payload_bytes": len(raw),
                }
            )
            socket.send_bytes(raw)
            if index == 2:
                assert socket.receive_json() == {
                    "type": "media.evicted",
                    "media_id": "frame-1",
                }
            assert socket.receive_json() == {
                "type": "media.ack",
                "media_id": f"frame-{index}",
            }


def test_speech_request_uses_one_whole_text_synthesis(monkeypatch) -> None:
    instances = []

    class FakeSpeechSynthesizer:
        def __init__(self, config):
            self.config = config
            self.calls = []
            self.closed = False
            instances.append(self)

        async def synthesize(
            self, *, text, voice, instruction, audio_sink
        ):
            self.calls.append((text, voice, instruction))
            await audio_sink(b"\x01\x00" * 2)
            await audio_sink(b"\x02\x00" * 2)
            return SimpleNamespace(
                audio_bytes=8,
                chunk_count=2,
                provider_response_id="tts-1",
            )

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        gateway_module,
        "GatewaySpeechSynthesizer",
        FakeSpeechSynthesizer,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_speech=GatewaySpeechConfig(
                    endpoint="http://tts/v1/tts/stream",
                )
            )
        )
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        ready = socket.receive_json()
        assert ready["contract_version"] == 2
        socket.send_json(
            {
                "type": "speech.request",
                "request_id": "speech-1",
                "text": "连续说完这句话。",
                "voice": "character-voice",
                "instruction": "自然地说",
            }
        )
        assert socket.receive_json()["type"] == "speech.accepted"
        first = socket.receive_json()
        second = socket.receive_json()
        done = socket.receive_json()

    assert first["type"] == "speech.audio.delta"
    assert first["seq"] == 0
    assert second["type"] == "speech.audio.delta"
    assert second["seq"] == 1
    assert done["type"] == "speech.audio.done"
    assert done["audio_bytes"] == 8
    assert done["chunk_count"] == 2
    assert instances[0].calls == [
        ("连续说完这句话。", "character-voice", "自然地说")
    ]
    assert instances[0].closed
