from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from sglang_omni.serve.inference_gateway import app as gateway_module
from sglang_omni.serve.inference_gateway import __main__ as gateway_main
from sglang_omni.serve.inference_gateway.app import (
    InferenceGatewayConfig,
    UpstreamStage,
    create_inference_gateway_app,
)
from sglang_omni.serve.inference_gateway.speech_synthesis import (
    GatewaySpeechConfig,
)
from sglang_omni.serve.realtime.embedded_tts import EmbeddedTTSConfig


class _Upstream:
    def __init__(self) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.media_header = None
        self.request_payload = None

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
            self.request_payload = message.get("payload")
            await self.events.put(json.dumps({"type": "request.ready"}))
        elif message["type"] == "input.media":
            self.media_header = message
        elif message["type"] == "request.commit":
            if isinstance(self.request_payload, dict) and self.request_payload.get("stream"):
                await self.events.put(
                    json.dumps(
                        {
                            "type": "response.delta",
                            "request_id": message["request_id"],
                            "response": {
                                "choices": [
                                    {
                                        "delta": {"content": "你好"},
                                        "finish_reason": "stop",
                                    }
                                ]
                            },
                        }
                    )
                )
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
            "classifier", "brain", "reply", "body", "expression", "performance"
        )},
        **kwargs,
    )


def test_stage_failures_are_classified_by_recovery_semantics() -> None:
    assert gateway_module._classify_stage_failure(ConnectionError("down")) == (
        "provider_unavailable",
        True,
    )
    assert gateway_module._classify_stage_failure(ValueError("bad event")) == (
        "upstream_protocol_error",
        False,
    )
    assert gateway_module._classify_stage_failure(RuntimeError("bug")) == (
        "stage_execution_failed",
        False,
    )


def test_bearer_uses_first_configured_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_ACTION_DECISION_TOKEN", "action-token")
    monkeypatch.setenv("SGLANG_OMNI_INTERNAL_MODEL_TOKEN", "internal-token")

    assert gateway_main._bearer(
        "SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN",
        fallback=(
            "SGLANG_OMNI_ACTION_DECISION_TOKEN",
            "SGLANG_OMNI_INTERNAL_MODEL_TOKEN",
        ),
    ) == "Bearer action-token"


def test_reply_speech_envelope_accepts_marker_without_newline() -> None:
    extractor = gateway_module._ReplySpeechTextExtractor("reply_envelope_v1")

    assert extractor.feed('{"speech_required":true}<<TE') == ()
    assert extractor.feed('XT>>\n你好') == ("你好",)
    assert extractor.finish() == ()


def test_reply_speech_excludes_trailing_user_turn_metadata() -> None:
    extractor = gateway_module._ReplySpeechTextExtractor(
        "plain_with_user_turn_v1"
    )

    assert extractor.feed("好的，我不说了。\n<<USER_") == ("好的，我不说了。",)
    assert extractor.feed("TURN_TEXT>>\n闭嘴") == ()
    assert extractor.finish() == ()


def test_reply_speech_segmenter_keeps_short_text_whole() -> None:
    segmenter = gateway_module._ReplySpeechSegmenter(max_chars=12)

    assert segmenter.feed("你好，") == ()
    assert segmenter.feed("很高兴认识你") == ()
    assert segmenter.finish() == ("你好，很高兴认识你",)


def test_reply_speech_segmenter_can_flush_pending_text_on_latency_budget() -> None:
    segmenter = gateway_module._ReplySpeechSegmenter(max_chars=120)

    assert segmenter.feed("正在生成一段没有标点的回复") == ()
    assert segmenter.has_pending is True
    assert segmenter.flush() == "正在生成一段没有标点的回复"
    assert segmenter.has_pending is False
    assert segmenter.finish() == ()


def test_reply_speech_segment_delay_is_bounded_to_product_budget() -> None:
    for value in (119.0, 201.0):
        try:
            _config(plain_reply_segment_max_delay_ms=value)
        except ValueError as exc:
            assert "between 120 and 200" in str(exc)
        else:
            raise AssertionError("out-of-range segment delay was accepted")


def test_reply_speech_segmenter_releases_natural_sentences_after_long_budget() -> None:
    segmenter = gateway_module._ReplySpeechSegmenter(max_chars=8)

    assert segmenter.feed("第一句话。第二句话") == ("第一句话。",)
    assert segmenter.feed("还没结束") == ()
    assert segmenter.feed("！尾巴") == ("第二句话还没结束！",)
    assert segmenter.finish() == ("尾巴",)


def test_invalid_reply_speech_envelope_does_not_fail_reply_stage(monkeypatch) -> None:
    class FakeEmbeddedTTSConnection:
        def __init__(self, _config, *, session_id):
            self.session_id = session_id

        async def synthesize_streaming(
            self, *, turn_id, text_chunks, audio_sink, voice, instruct
        ):
            await instruct
            _ = [chunk async for chunk in text_chunks]
            return SimpleNamespace(
                audio_bytes=0,
                chunk_count=0,
                provider_response_id="unused",
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        gateway_module.websockets,
        "connect",
        lambda *_args, **_kwargs: _Upstream(),
    )
    monkeypatch.setattr(
        gateway_module,
        "EmbeddedTTSConnection",
        FakeEmbeddedTTSConnection,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_streaming_speech=EmbeddedTTSConfig(
                    url="ws://tts/realtime",
                    voice="default",
                )
            )
        )
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-invalid-envelope",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-invalid-envelope",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "1",
                    "text_mode": "reply_envelope_v1",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        completed = socket.receive_json()

    assert completed["type"] == "stage.completed"


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


def test_reply_stage_overwrites_cache_owner_with_gateway_session(monkeypatch) -> None:
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
                "request_id": "reply-1",
                "stage": "reply",
                "payload": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "session_instance_id": "caller-controlled",
                },
                "media_refs": [],
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.completed"

    request_start = json.loads(upstreams[0].sent[0])
    assert request_start["payload"]["session_instance_id"] == "s1"


def test_reply_reserves_speculation_id_until_stage_finishes(monkeypatch) -> None:
    class BlockingUpstream(_Upstream):
        async def send(self, value):
            self.sent.append(value)
            if isinstance(value, bytes):
                return
            message = json.loads(value)
            if message["type"] == "request.start":
                await self.events.put(json.dumps({"type": "request.ready"}))

    monkeypatch.setattr(
        gateway_module.websockets,
        "connect",
        lambda *_args, **_kwargs: BlockingUpstream(),
    )
    client = TestClient(create_inference_gateway_app(_config()))
    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        request = {
            "type": "stage.request",
            "stage": "reply",
            "payload": {"stream": True},
            "media_refs": [],
            "speech_speculation": {
                "request_id": "shared-speculation",
                "voice": "voice-1",
                "instruction": "自然地说",
                "generation_id": "generation-1",
                "output_epoch": "1",
            },
        }
        socket.send_json({**request, "request_id": "reply-1"})
        assert socket.receive_json()["type"] == "stage.accepted"
        socket.send_json({**request, "request_id": "reply-2"})
        error = socket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "invalid_request"
    assert "already active" in error["detail"]


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
            {"type": "session.open", "contract_version": 3, "session_id": "s1"}
        )
        ready = socket.receive_json()
        assert ready["contract_version"] == 3
        assert ready["availability_epoch"] == 0
        assert ready["stages"] == {
            stage: True
            for stage in (
                "body",
                "brain",
                "classifier",
                "expression",
                "performance",
                "reply",
            )
        }
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


@pytest.mark.parametrize("target", [None, 334])
def test_reply_speculation_buffers_until_matching_commit(monkeypatch, target) -> None:
    class FakeSpeechSynthesizer:
        def __init__(self, _config):
            pass

        async def synthesize(self, *, text, voice, instruction, audio_sink, first_chunk_ms=None):
            assert first_chunk_ms == target
            assert (text, voice, instruction) == ("你好", "voice-1", "自然地说")
            await audio_sink(b"\x01\x00")
            await audio_sink(b"\x02\x00")
            return SimpleNamespace(
                audio_bytes=4,
                chunk_count=2,
                provider_response_id="speculative-1",
            )

        async def close(self):
            return None

    monkeypatch.setattr(gateway_module.websockets, "connect", lambda *_a, **_k: _Upstream())
    monkeypatch.setattr(gateway_module, "GatewaySpeechSynthesizer", FakeSpeechSynthesizer)
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_speech=GatewaySpeechConfig(
                    endpoint="http://tts/v1/tts/stream",
                )
            )
        )
    )
    text_hash = gateway_module._speculative_speech_hash(
        text="你好",
        voice="voice-1",
        instruction="自然地说",
        generation_id="generation-1",
        output_epoch=1,
        first_chunk_ms=target,
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1", **({"speech_first_chunk_ms": target} if target is not None else {})}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-1",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-spec-1",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "1",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        assert socket.receive_json()["type"] == "stage.completed"
        socket.send_json(
            {
                "type": "speech.commit",
                "request_id": "speech-spec-1",
                "text_hash": text_hash,
                "voice": "voice-1",
                "instruction": "自然地说",
                "generation_id": "generation-1",
                "output_epoch": 1,
            }
        )
        assert socket.receive_json()["type"] == "speech.accepted"
        assert socket.receive_json()["seq"] == 0
        assert socket.receive_json()["seq"] == 1
        done = socket.receive_json()

    assert done["type"] == "speech.audio.done"
    assert done["speculative"] is True


@pytest.mark.parametrize("target", [None, 334])
def test_adaptive_plain_speculation_uses_one_whole_text_request_for_short_reply(
    monkeypatch, target,
) -> None:
    speech_calls = []
    streaming_calls = []

    class FakeSpeechSynthesizer:
        def __init__(self, _config):
            pass

        async def synthesize(self, *, text, voice, instruction, audio_sink, first_chunk_ms=None):
            assert first_chunk_ms == target
            speech_calls.append((text, voice, instruction))
            await audio_sink(b"\x01\x00")
            return SimpleNamespace(
                audio_bytes=2,
                chunk_count=1,
                provider_response_id="adaptive-short",
            )

        async def close(self):
            return None

    class FakeEmbeddedTTSConnection:
        def __init__(self, _config, *, session_id):
            self.session_id = session_id

        async def synthesize_streaming(self, **kwargs):
            streaming_calls.append(kwargs)
            raise AssertionError("plain adaptive speech must not use streaming TTS")

        async def close(self):
            return None

    monkeypatch.setattr(gateway_module.websockets, "connect", lambda *_a, **_k: _Upstream())
    monkeypatch.setattr(gateway_module, "GatewaySpeechSynthesizer", FakeSpeechSynthesizer)
    monkeypatch.setattr(
        gateway_module,
        "EmbeddedTTSConnection",
        FakeEmbeddedTTSConnection,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_speech=GatewaySpeechConfig(
                    endpoint="http://tts/v1/tts/stream",
                ),
                reply_streaming_speech=EmbeddedTTSConfig(
                    url="ws://tts/realtime",
                    voice="default",
                ),
                adaptive_plain_reply_speech=True,
                plain_reply_segment_max_chars=120,
            )
        )
    )
    text_hash = gateway_module._speculative_speech_hash(
        text="你好",
        voice="voice-1",
        instruction="自然地说",
        generation_id="generation-1",
        output_epoch=1,
        first_chunk_ms=target,
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1", **({"speech_first_chunk_ms": target} if target is not None else {})}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-adaptive-short",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-adaptive-short",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "1",
                    "text_mode": "plain",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        assert socket.receive_json()["type"] == "stage.completed"
        socket.send_json(
            {
                "type": "speech.commit",
                "request_id": "speech-adaptive-short",
                "text_hash": text_hash,
                "voice": "voice-1",
                "instruction": "自然地说",
                "generation_id": "generation-1",
                "output_epoch": 1,
            }
        )
        assert socket.receive_json()["type"] == "speech.accepted"
        assert socket.receive_json()["type"] == "speech.audio.delta"
        assert socket.receive_json()["type"] == "speech.audio.done"

    assert speech_calls == [("你好", "voice-1", "自然地说")]
    assert streaming_calls == []


@pytest.mark.parametrize("target", [None, 334])
def test_streaming_reply_speculation_starts_from_delta_and_waits_for_commit(
    monkeypatch, target,
) -> None:
    instances = []

    class FakeEmbeddedTTSConnection:
        def __init__(self, config, *, session_id):
            self.config = config
            self.session_id = session_id
            self.calls = []
            instances.append(self)

        async def synthesize_streaming(
            self, *, turn_id, text_chunks, audio_sink, voice, instruct, first_chunk_ms=None
        ):
            assert first_chunk_ms == target
            instruction = await instruct
            text = "".join([chunk async for chunk in text_chunks])
            self.calls.append((turn_id, text, voice, instruction))
            await audio_sink(b"\x01\x00")
            return SimpleNamespace(
                audio_bytes=2,
                chunk_count=1,
                provider_response_id="streaming-spec-1",
            )

        async def close(self):
            return None

    monkeypatch.setattr(gateway_module.websockets, "connect", lambda *_a, **_k: _Upstream())
    monkeypatch.setattr(
        gateway_module,
        "EmbeddedTTSConnection",
        FakeEmbeddedTTSConnection,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_streaming_speech=EmbeddedTTSConfig(
                    url="ws://tts/realtime",
                    voice="default",
                )
            )
        )
    )
    text_hash = gateway_module._speculative_speech_hash(
        text="你好",
        voice="voice-1",
        instruction="温暖地说",
        generation_id="generation-1",
        output_epoch=3,
        first_chunk_ms=target,
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1", **({"speech_first_chunk_ms": target} if target is not None else {})}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-1",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-spec-1",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "3",
                    "text_mode": "plain",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        assert socket.receive_json()["type"] == "stage.completed"
        socket.send_json(
            {
                "type": "speech.configure",
                "request_id": "speech-spec-1",
                "instruction": "温暖地说",
                "generation_id": "generation-1",
                "output_epoch": 3,
            }
        )
        assert socket.receive_json()["type"] == "speech.configured"
        socket.send_json(
            {
                "type": "speech.commit",
                "request_id": "speech-spec-1",
                "text_hash": text_hash,
                "voice": "voice-1",
                "instruction": "温暖地说",
                "generation_id": "generation-1",
                "output_epoch": 3,
            }
        )
        assert socket.receive_json()["type"] == "speech.accepted"
        assert socket.receive_json()["type"] == "speech.audio.delta"
        done = socket.receive_json()

    assert done["type"] == "speech.audio.done"
    assert done["speculative"] is True
    assert instances[0].calls == [
        ("speech-spec-1", "你好", "voice-1", "温暖地说")
    ]


def test_streaming_speculation_mismatch_is_rejected_and_cancelled(monkeypatch) -> None:
    cancelled = []

    class FakeEmbeddedTTSConnection:
        def __init__(self, _config, *, session_id):
            self.session_id = session_id

        async def synthesize_streaming(
            self, *, turn_id, text_chunks, audio_sink, voice, instruct
        ):
            await instruct
            _ = [chunk async for chunk in text_chunks]
            await audio_sink(b"\x01\x00")
            await asyncio.Future()

        async def close(self):
            return None

    monkeypatch.setattr(gateway_module.websockets, "connect", lambda *_a, **_k: _Upstream())
    monkeypatch.setattr(
        gateway_module,
        "EmbeddedTTSConnection",
        FakeEmbeddedTTSConnection,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_streaming_speech=EmbeddedTTSConfig(
                    url="ws://tts/realtime",
                    voice="default",
                )
            )
        )
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-1",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-spec-1",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "3",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        assert socket.receive_json()["type"] == "stage.completed"
        socket.send_json(
            {
                "type": "speech.configure",
                "request_id": "speech-spec-1",
                "instruction": "温暖地说",
                "generation_id": "generation-1",
                "output_epoch": 3,
            }
        )
        assert socket.receive_json()["type"] == "speech.configured"
        socket.send_json(
            {
                "type": "speech.commit",
                "request_id": "speech-spec-1",
                "text_hash": "sha256:" + "0" * 64,
                "voice": "voice-1",
                "instruction": "温暖地说",
                "generation_id": "generation-1",
                "output_epoch": 3,
            }
        )
        error = socket.receive_json()
        cancelled.append(socket.receive_json())

    assert error["type"] == "speech.error"
    assert error["code"] == "speculation_mismatch"
    assert cancelled[0]["type"] == "speech.cancelled"


def test_streaming_speculation_session_budget_fails_closed(monkeypatch) -> None:
    class FakeEmbeddedTTSConnection:
        def __init__(self, _config, *, session_id):
            self.session_id = session_id

        async def synthesize_streaming(
            self, *, turn_id, text_chunks, audio_sink, voice, instruct
        ):
            await instruct
            _ = [chunk async for chunk in text_chunks]
            await audio_sink(b"\x01\x00\x02\x00")
            return SimpleNamespace(
                audio_bytes=4,
                chunk_count=1,
                provider_response_id="too-large",
            )

        async def close(self):
            return None

    monkeypatch.setattr(gateway_module.websockets, "connect", lambda *_a, **_k: _Upstream())
    monkeypatch.setattr(
        gateway_module,
        "EmbeddedTTSConnection",
        FakeEmbeddedTTSConnection,
    )
    client = TestClient(
        create_inference_gateway_app(
            _config(
                reply_streaming_speech=EmbeddedTTSConfig(
                    url="ws://tts/realtime",
                    voice="default",
                ),
                max_session_speculative_audio_bytes=2,
            )
        )
    )

    with client.websocket_connect("/v1/inference-session") as socket:
        socket.send_json(
            {"type": "session.open", "contract_version": 2, "session_id": "s1"}
        )
        socket.receive_json()
        socket.send_json(
            {
                "type": "stage.request",
                "request_id": "reply-1",
                "stage": "reply",
                "payload": {"stream": True},
                "media_refs": [],
                "speech_speculation": {
                    "request_id": "speech-spec-1",
                    "voice": "voice-1",
                    "instruction": "自然地说",
                    "generation_id": "generation-1",
                    "output_epoch": "1",
                },
            }
        )
        assert socket.receive_json()["type"] == "stage.accepted"
        assert socket.receive_json()["type"] == "stage.delta"
        assert socket.receive_json()["type"] == "stage.completed"
        socket.send_json(
            {
                "type": "speech.configure",
                "request_id": "speech-spec-1",
                "instruction": "自然地说",
                "generation_id": "generation-1",
                "output_epoch": 1,
            }
        )
        assert socket.receive_json()["type"] == "speech.configured"
        error = socket.receive_json()

    assert error["type"] == "speech.error"
    assert error["code"] == "tts_failed"
    assert error["retryable"] is True


def test_speech_first_chunk_is_session_local_and_request_override_does_not_persist(monkeypatch):
    calls = []
    class Speech:
        def __init__(self, config):
            self.config = config
        async def synthesize(self, *, text, voice, instruction, audio_sink, first_chunk_ms=None):
            calls.append((text, first_chunk_ms))
            await audio_sink(b'\x01\x00')
            return SimpleNamespace(audio_bytes=2, chunk_count=1, provider_response_id='mock')
        async def close(self):
            pass
    monkeypatch.setattr(gateway_module, 'GatewaySpeechSynthesizer', Speech)
    config = _config(reply_speech=GatewaySpeechConfig(endpoint='http://tts/stream'))
    client = TestClient(create_inference_gateway_app(config))
    with client.websocket_connect('/v1/inference-session') as configured, client.websocket_connect('/v1/inference-session') as default:
        for socket, target in ((configured, 334), (default, None)):
            socket.send_json({'type': 'session.open', 'contract_version': 3, 'session_id': str(target), **({'speech_first_chunk_ms': target} if target is not None else {})})
            ready = socket.receive_json()
            assert ready.get('speech_first_chunk_ms') == target
            assert ('speech_first_chunk_ms' in ready) == (target is not None)
        for socket, text, override in ((configured, 'configured', {}), (default, 'default', {}), (configured, 'override', {'first_chunk_ms': 250}), (configured, 'restored', {})):
            socket.send_json({'type': 'speech.request', 'request_id': text, 'text': text, 'voice': 'voice', 'instruction': 'natural', **override})
            assert socket.receive_json()['type'] == 'speech.accepted'
            assert socket.receive_json()['type'] == 'speech.audio.delta'
            assert socket.receive_json()['type'] == 'speech.audio.done'
    assert calls == [('configured', 334), ('default', None), ('override', 250), ('restored', 334)]
    assert config.reply_speech.frame_bytes == 12000


@pytest.mark.parametrize('target', [True, False, 39, 1001, 334.0, '334'])
def test_session_rejects_invalid_first_chunk(target):
    client = TestClient(create_inference_gateway_app(_config()))
    with client.websocket_connect('/v1/inference-session') as socket:
        socket.send_json({'type': 'session.open', 'contract_version': 3, 'session_id': 'bad', 'speech_first_chunk_ms': target})
        error = socket.receive_json()
        assert error['type'] == 'error'
        assert error['code'] == 'invalid_request'
        assert 'first_chunk_ms' in error['detail']


def test_speculation_hash_binds_target_but_default_digest_is_unchanged():
    fields = dict(text='你好', voice='voice', instruction='natural', generation_id='gen', output_epoch=1)
    legacy = dict(fields)
    legacy['text_hash'] = 'sha256:' + hashlib.sha256(legacy.pop('text').encode()).hexdigest()
    expected = 'sha256:' + hashlib.sha256(json.dumps(legacy, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert gateway_module._speculative_speech_hash(**fields) == expected
    assert len({gateway_module._speculative_speech_hash(**fields, first_chunk_ms=value) for value in (None, 250, 334)}) == 3


@pytest.mark.asyncio
async def test_speculation_cannot_commit_with_different_first_packet_target():
    sent = []
    class Socket:
        async def send_json(self, event):
            sent.append(event)
    session = gateway_module._Session(Socket(), _config())
    session.speech_first_chunk_ms = 334
    state = gateway_module._SpeechSpeculation(request_id='speech', source_request_id='reply', voice='voice', instruction='natural', generation_id='gen', output_epoch=1, text_mode='plain', first_chunk_ms=334, text='hello', text_hash='sha256:correct')
    state.text_completed.set()
    session.speech_speculations['speech'] = state
    await session._commit_speech({'request_id': 'speech', 'text_hash': state.text_hash, 'voice': state.voice, 'instruction': state.instruction, 'generation_id': state.generation_id, 'output_epoch': state.output_epoch, 'first_chunk_ms': 250})
    assert sent[-1]['code'] == 'speculation_mismatch'
    assert not state.committed


@pytest.mark.asyncio
@pytest.mark.parametrize('committed', [False, True])
async def test_adaptive_first_packet_accumulates_across_short_sentences(committed):
    import base64
    sent = []
    calls = []
    state = gateway_module._SpeechSpeculation(request_id='speech', source_request_id='reply', voice='voice', instruction='natural', generation_id='gen', output_epoch=1, text_mode='plain', first_chunk_ms=334)
    class Socket:
        async def send_json(self, event):
            sent.append(event)
    session = gateway_module._Session(Socket(), _config(reply_speech=GatewaySpeechConfig(endpoint='http://tts/stream')))
    session.speech_first_chunk_ms = 334
    source = [b'a' * 4800, b'b' * 17280, b'c' * 12000]
    class Speech:
        async def synthesize(self, *, text, voice, instruction, audio_sink, first_chunk_ms=None):
            index = int(text)
            calls.append(first_chunk_ms)
            await audio_sink(source[index])
            if index == 0:
                assert not state.chunks
                assert not sent
            return SimpleNamespace(audio_bytes=len(source[index]), chunk_count=1, provider_response_id='mock')
    session.reply_speech = Speech()
    session.speech_speculations[state.request_id] = state
    for text in ('0', '1', '2', None):
        await state.text_queue.put(text)
    state.committed = committed
    state.commit_event.set()
    await session._run_adaptive_speculative_speech(state)
    assert calls == [334, 334, None]
    frames = [base64.b64decode(event['delta']) for event in sent if event['type'] == 'speech.audio.delta'] if committed else state.chunks
    assert list(map(len, frames)) == [16032, 12000, 6048]
    assert b''.join(frames) == b''.join(source)
    done = sent[-1]
    assert done['type'] == 'speech.audio.done'
    assert done['audio_bytes'] == 34080
    assert done['chunk_count'] == 3
    assert session.speculative_buffered_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('budget_kind', ['request', 'session'])
async def test_adaptive_first_packet_pending_pcm_counts_toward_audio_budget(budget_kind):
    sent = []
    class Socket:
        async def send_json(self, event):
            sent.append(event)
    config = _config(
        reply_speech=GatewaySpeechConfig(endpoint='http://tts/stream', max_audio_bytes=8000 if budget_kind == 'request' else 32000),
        max_session_speculative_audio_bytes=9000 if budget_kind == 'session' else 64000,
    )
    session = gateway_module._Session(Socket(), config)
    session.speculative_buffered_bytes = 1000  # Another speculative response.
    state = gateway_module._SpeechSpeculation(request_id='speech', source_request_id='reply', voice='voice', instruction='natural', generation_id='gen', output_epoch=1, text_mode='plain', first_chunk_ms=334)
    class Speech:
        async def synthesize(self, *, audio_sink, **kwargs):
            await audio_sink(b'a' * 6000)
            return SimpleNamespace(audio_bytes=6000, chunk_count=1, provider_response_id='mock')
    session.reply_speech = Speech()
    session.speech_speculations[state.request_id] = state
    for text in ('first', 'second', None):
        await state.text_queue.put(text)
    await session._run_adaptive_speculative_speech(state)
    assert sent[-1]['type'] == 'speech.error'
    assert not state.chunks
    assert state.pending_audio_bytes == 0
    assert session.speculative_buffered_bytes == 1000


@pytest.mark.asyncio
async def test_adaptive_pending_pcm_survives_commit_and_releases_on_cancel():
    sent = []
    received = asyncio.Event()
    class Socket:
        async def send_json(self, event):
            sent.append(event)
    session = gateway_module._Session(Socket(), _config(reply_speech=GatewaySpeechConfig(endpoint='http://tts/stream')))
    session.speech_first_chunk_ms = 334
    session.speculative_buffered_bytes = 1000
    state = gateway_module._SpeechSpeculation(request_id='speech', source_request_id='reply', voice='voice', instruction='natural', generation_id='gen', output_epoch=1, text_mode='plain', first_chunk_ms=334, text_hash='sha256:correct')
    class Speech:
        async def synthesize(self, *, audio_sink, **kwargs):
            await audio_sink(b'a' * 6000)
            received.set()
            await asyncio.Event().wait()
    session.reply_speech = Speech()
    session.speech_speculations[state.request_id] = state
    await state.text_queue.put('first')
    task = asyncio.create_task(session._run_adaptive_speculative_speech(state))
    try:
        await asyncio.wait_for(received.wait(), 1)
        assert state.pending_audio_bytes == 6000
        assert session.speculative_buffered_bytes == 7000
        state.text_completed.set()
        await session._commit_speech({'request_id': state.request_id, 'text_hash': state.text_hash, 'voice': state.voice, 'instruction': state.instruction, 'generation_id': state.generation_id, 'output_epoch': state.output_epoch})
        assert state.committed
        assert session.speculative_buffered_bytes == 7000
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert state.pending_audio_bytes == 0
    assert session.speculative_buffered_bytes == 1000
    assert not any(event['type'] == 'speech.audio.delta' for event in sent)
