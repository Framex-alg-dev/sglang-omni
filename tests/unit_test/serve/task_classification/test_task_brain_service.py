from __future__ import annotations

import asyncio
import hashlib

from fastapi.testclient import TestClient

from sglang_omni.client.types import CompletionResult, UsageInfo
from sglang_omni.serve.task_brain.service import (
    TaskBrainServiceConfig,
    create_task_brain_app,
)


class _Client:
    def __init__(self) -> None:
        self.requests = []

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.requests.append((request, request_id, audio_format))
        return CompletionResult(
            request_id=request_id,
            text='{"kind":"respond"}',
            usage=UsageInfo(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )


def _payload() -> dict[str, object]:
    return {
        "model": "qwen3-omni-gpu3-task-decision",
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "plan this task"},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 128,
    }


def _app(client=None, **overrides):
    return create_task_brain_app(
        client or _Client(),
        config=TaskBrainServiceConfig(
            token="brain-secret",
            model_id="qwen3-omni-gpu3-task-decision",
            model_version="task-brain-v2",
            **overrides,
        ),
    )


def test_authenticated_openai_contract_sets_task_metadata() -> None:
    model = _Client()
    with TestClient(_app(model)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer brain-secret"},
            json=_payload(),
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        '{"kind":"respond"}'
    )
    request, _, _ = model.requests[0]
    assert request.metadata["task"] == "task_brain"
    assert request.sampling.temperature == 0
    assert request.sampling.max_new_tokens == 128
    assert "queue;dur=" in response.headers["server-timing"]


def test_strict_json_schema_is_forwarded_to_constrained_decoding() -> None:
    model = _Client()
    schema = {
        "type": "object",
        "properties": {"kind": {"const": "complete"}},
        "required": ["kind"],
        "additionalProperties": False,
    }
    payload = {
        **_payload(),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "task_decision",
                "strict": True,
                "schema": schema,
            },
        },
    }

    with TestClient(_app(model)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer brain-secret"},
            json=payload,
        )

    assert response.status_code == 200
    request, _, _ = model.requests[0]
    assert request.sampling.json_schema == (
        '{"type":"object","properties":{"kind":{"const":"complete"}},'
        '"required":["kind"],"additionalProperties":false}'
    )


def test_malformed_json_schema_is_rejected_before_inference() -> None:
    model = _Client()
    payload = {
        **_payload(),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "task_decision", "schema": "not-an-object"},
        },
    }

    with TestClient(_app(model)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer brain-secret"},
            json=payload,
        )

    assert response.status_code == 422
    assert model.requests == []


def test_realtime_carries_one_audio_and_multiple_images_to_model_metadata() -> None:
    model = _Client()
    audio = b"\x00\x00" * 160
    images = (b"jpeg-frame-1", b"jpeg-frame-2", b"jpeg-frame-3")

    with TestClient(_app(model)) as client:
        with client.websocket_connect(
            "/v1/chat/completions/realtime",
            headers={"Authorization": "Bearer brain-secret"},
        ) as socket:
            socket.send_json(
                {
                    "type": "request.start",
                    "contract_version": 2,
                    "request_id": "request-multimodal",
                    "payload": _payload(),
                }
            )
            assert socket.receive_json()["type"] == "request.ready"
            media = (("audio", "pcm_s16le", audio),) + tuple(
                ("image", "image/jpeg", image) for image in images
            )
            for index, (kind, encoding, payload) in enumerate(media):
                socket.send_json(
                    {
                        "type": "input.media",
                        "request_id": "request-multimodal",
                        "media_id": f"{kind}-{index}",
                        "kind": kind,
                        "start_ms": index * 100,
                        "end_ms": (index + 1) * 100,
                        "encoding": encoding,
                        "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        "payload_bytes": len(payload),
                        "evidence_role": (
                            "user_audio" if kind == "audio" else "user_camera"
                        ),
                    }
                )
                socket.send_bytes(payload)
                assert socket.receive_json()["type"] == "input.media.ack"
            socket.send_json(
                {"type": "request.commit", "request_id": "request-multimodal"}
            )
            completed = socket.receive_json()

    assert completed["type"] == "response.completed"
    request, _, _ = model.requests[0]
    assert len(request.metadata["audios"]) == 1
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")
    assert len(request.metadata["images"]) == 3
    assert all(
        image.startswith("data:image/jpeg;base64,")
        for image in request.metadata["images"]
    )
    assert request.messages[-1].content[1:] == [
        {"type": "audio"},
        {"type": "image"},
        {"type": "image"},
        {"type": "image"},
    ]


def test_health_and_completion_are_hidden_without_token() -> None:
    with TestClient(_app()) as client:
        assert client.get("/health").status_code == 404
        assert client.post("/v1/chat/completions", json=_payload()).status_code == 404


def test_streaming_and_non_none_reasoning_are_rejected() -> None:
    with TestClient(_app()) as client:
        streaming = {**_payload(), "stream": True}
        reasoning = {**_payload(), "reasoning_effort": "high"}
        headers = {"Authorization": "Bearer brain-secret"}
        assert client.post(
            "/v1/chat/completions", headers=headers, json=streaming
        ).status_code == 422
        assert client.post(
            "/v1/chat/completions", headers=headers, json=reasoning
        ).status_code == 422


def test_body_limit_is_enforced_before_inference() -> None:
    with TestClient(_app(max_body_bytes=64)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer brain-secret"},
            json=_payload(),
        )
    assert response.status_code == 413


def test_brain_concurrency_is_bounded() -> None:
    class _BlockingClient(_Client):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum = 0

        async def completion(self, request, *, request_id, audio_format="wav"):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return CompletionResult(request_id=request_id, text="{}")

    model = _BlockingClient()
    app = _app(model, max_concurrency=1)

    async def exercise() -> None:
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer brain-secret"},
                        json=_payload(),
                    )
                    for _ in range(3)
                )
            )

    asyncio.run(exercise())
    assert model.maximum == 1
