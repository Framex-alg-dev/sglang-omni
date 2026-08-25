# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.serve.realtime.dev_model import (
    DevRealtimeModelClient,
    DevRealtimeModelConfig,
    DevRealtimeModelRequestError,
)
from sglang_omni.serve.realtime.dev_server import create_dev_app


def _request() -> GenerateRequest:
    return GenerateRequest(
        messages=[
            Message(role="system", content="reply"),
            Message(role="user", content="hello"),
        ],
        stream=True,
        output_modalities=["text"],
    )


def _action_request(*candidate_ids: str) -> ActionSuffixScoreRequest:
    return ActionSuffixScoreRequest(
        request_id="action-1",
        model="dev-model",
        prefix="prefix",
        language="zh",
        candidates=[
            ActionScoreCandidate(candidate_id=item, suffix=item)
            for item in candidate_ids
        ],
        audios=[],
        images=[],
        sample_rate=16000,
    )


def test_config_parses_session_fake_settings() -> None:
    config = DevRealtimeModelConfig.from_env(
        {
            "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED": "true",
            "SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT": "固定回复",
            "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE": "2",
            "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS": "3",
            "SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID": "A002",
        }
    )
    assert (config.response_text, config.action_candidate_id) == ("固定回复", "A002")
    assert "固定回复" not in str(config.log_summary())


@pytest.mark.parametrize("candidate_id", ["A000", "B000", "UNSUPPORTED", "DEV_NONE_0"])
def test_config_rejects_reserved_action_candidate(candidate_id: str) -> None:
    with pytest.raises(ValueError, match="reserved action identifier"):
        DevRealtimeModelConfig.from_env(
            {
                "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED": "true",
                "SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID": candidate_id,
            }
        )


@pytest.mark.asyncio
async def test_text_stream_and_abort() -> None:
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(enabled=True, response_text="abcdef", chunk_size=2)
    )
    stream = client.completion_stream(_request(), request_id="request-1")
    assert (await anext(stream)).text == "ab"
    assert (await client.abort("request-1")).success is True
    assert [item async for item in stream] == []


@pytest.mark.asyncio
async def test_action_scoring_selects_configured_whitelisted_candidate() -> None:
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(enabled=True, action_candidate_id="A002")
    )
    result = await client.score_action_suffixes(_action_request("A001", "A002"))
    assert max(result.scores, key=lambda item: item.mean_logprob).candidate_id == "A002"
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(enabled=True, action_candidate_id="A999")
    )
    with pytest.raises(DevRealtimeModelRequestError, match="Session whitelist"):
        await client.score_action_suffixes(_action_request("A001"))


def _dev_app(*, action_candidate_id: str = ""):
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="固定回复",
            chunk_size=2,
            action_candidate_id=action_candidate_id,
        )
    )
    return create_dev_app(client, model_name="dev-model")


def _session_start(outputs: list[str]) -> dict:
    event = {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "dev-session",
        "outputs": outputs,
        "locale": "zh-CN",
    }
    if "text" in outputs:
        event["reply"] = {"instructions": "简短回复"}
    if "action" in outputs:
        event["action"] = {
            "fallback_category_ids": ["BDEV"],
            "allowed_candidates": [{"candidate_id": "ADEV"}],
        }
    if outputs == ["text", "action"]:
        event["reply"]["unsupported_action_text"] = "动作不支持"
    return event


def test_development_route_allowlist_uses_only_session_realtime() -> None:
    app = _dev_app()
    paths = {
        route.path for route in app.router.routes if isinstance(route, WebSocketRoute)
    }
    assert paths == {"/v1/session/realtime"}


def test_audio_output_is_explicitly_unsupported() -> None:
    with TestClient(_dev_app()).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(_session_start(["audio"]))
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "unsupported_output"


@pytest.mark.parametrize(
    "action",
    [
        {
            "fallback_category_ids": ["B000"],
            "allowed_candidates": [{"candidate_id": "ADEV"}],
        },
        {
            "fallback_category_ids": ["BDEV"],
            "allowed_candidates": [{"candidate_id": "A000"}],
        },
        {
            "fallback_category_ids": ["BDEV"],
            "allowed_candidates": [
                {
                    "candidate_id": "ADEV",
                    "execution_binding": {"clip": " "},
                }
            ],
        },
        {
            "fallback_category_ids": ["ADEV"],
            "allowed_candidates": [{"candidate_id": "ADEV"}],
        },
    ],
)
def test_development_action_catalog_preserves_reserved_and_binding_constraints(
    action: dict,
) -> None:
    start = _session_start(["action"])
    start["action"] = action
    with TestClient(_dev_app()).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(start)
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "session_candidate_invalid"


@pytest.mark.parametrize("outputs", [["text"], ["action"], ["text", "action"]])
def test_session_realtime_runs_deterministic_turn(outputs: list[str]) -> None:
    with TestClient(_dev_app()).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(_session_start(outputs))
        assert ws.receive_json()["type"] == "session.started"
        ws.send_json({"type": "turn.start", "turn_id": "turn-1", "origin": "user"})
        assert ws.receive_json()["type"] == "turn.started"
        ws.send_json({"type": "input.text.set", "turn_id": "turn-1", "text": "你好"})
        assert ws.receive_json()["type"] == "input.text.ack"
        ws.send_json({"type": "turn.commit", "turn_id": "turn-1"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "turn.result":
                break
        if "text" in outputs:
            done = next(item for item in events if item["type"] == "response.done")
            assert done["response"]["output"][0]["text"] == "固定回复"
        if "action" in outputs:
            ready = next(item for item in events if item["type"] == "turn.action.ready")
            assert ready["action"]["candidate_id"] == "ADEV"
        if outputs == ["text", "action"]:
            resolved = next(
                item
                for item in events
                if item["type"] == "response.provisional.resolved"
            )
            assert resolved["status"] == "promoted"


def test_optional_capability_detection_is_safe() -> None:
    client = DevRealtimeModelClient(DevRealtimeModelConfig(enabled=True))
    assert not hasattr(client, "action_scoring_load")


def test_fusion_category_ignores_unavailable_child_preference() -> None:
    with TestClient(_dev_app(action_candidate_id="A999")).websocket_connect(
        "/v1/session/realtime"
    ) as ws:
        ws.send_json(_session_start(["text", "action"]))
        assert ws.receive_json()["type"] == "session.started"
        ws.send_json(
            {"type": "turn.start", "turn_id": "turn-fallback", "origin": "user"}
        )
        assert ws.receive_json()["type"] == "turn.started"
        ws.send_json(
            {"type": "input.text.set", "turn_id": "turn-fallback", "text": "未知动作"}
        )
        assert ws.receive_json()["type"] == "input.text.ack"
        ws.send_json({"type": "turn.commit", "turn_id": "turn-fallback"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "turn.result":
                break
        result = events[-1]
        assert result["status"] == "completed"
        assert result["action"]["action_id"] == "ADEV"
        assert "fallback_applied" not in result["action"]


def test_turn_cancel_stops_stream_and_has_single_terminal_event() -> None:
    client = DevRealtimeModelClient(
        DevRealtimeModelConfig(
            enabled=True,
            response_text="取消后不应完成",
            chunk_size=1,
            chunk_interval_ms=1000,
        )
    )
    app = create_dev_app(client, model_name="dev-model")
    with TestClient(app).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(_session_start(["text"]))
        assert ws.receive_json()["type"] == "session.started"
        ws.send_json({"type": "turn.start", "turn_id": "turn-cancel", "origin": "user"})
        assert ws.receive_json()["type"] == "turn.started"


def test_late_cancel_after_result_is_idempotent_and_next_turn_can_start() -> None:
    with TestClient(_dev_app()).websocket_connect("/v1/session/realtime") as ws:
        ws.send_json(_session_start(["text"]))
        assert ws.receive_json()["type"] == "session.started"
        ws.send_json({"type": "turn.start", "turn_id": "turn-done", "origin": "user"})
        assert ws.receive_json()["type"] == "turn.started"
        ws.send_json({"type": "input.text.set", "turn_id": "turn-done", "text": "完成"})
        assert ws.receive_json()["type"] == "input.text.ack"
        ws.send_json({"type": "turn.commit", "turn_id": "turn-done"})
        while ws.receive_json()["type"] != "turn.result":
            pass

        ws.send_json({"type": "turn.cancel", "turn_id": "turn-done"})
        ws.send_json({"type": "turn.start", "turn_id": "turn-next", "origin": "user"})
        assert ws.receive_json()["type"] == "turn.started"
