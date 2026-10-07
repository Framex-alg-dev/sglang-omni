from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.types import UsageInfo
from sglang_omni.serve.action_decision.catalog import ActionCatalogRegistry
from sglang_omni.serve.action_decision.contracts import (
    ActionDecision,
    ActionDecisionRequest,
    DecisionChannel,
)
from sglang_omni.serve.action_decision.fusion import fuse_action_decisions
from sglang_omni.serve.action_decision.prompt import (
    E57A_REFERENCE_SYSTEM_PROMPT_PATH,
    build_system_prompt,
    build_user_prompt,
)
from sglang_omni.serve.action_decision.service import (
    ActionDecisionConfig,
    ActionDecisionEngine,
    create_action_decision_app,
)
from sglang_omni.serve.streaming_request import StreamedMedia
from sglang_omni.serve.launcher import _run_server


ASSETS = "/data/xingmt/model_repo/action_prediction_model/assets"


def _config() -> ActionDecisionConfig:
    return ActionDecisionConfig(
        token="secret",
        model_id="action-model",
        model_version="e43",
        mapping_path=f"{ASSETS}/action_pair_token_map_current.json",
        full_mapping_path=f"{ASSETS}/action_pair_token_map_full.json",
        product_catalog_path=f"{ASSETS}/action_catalog.json",
        agent_policy_path=f"{ASSETS}/agent_action_policy.json",
    )


def _registry() -> ActionCatalogRegistry:
    config = _config()
    return ActionCatalogRegistry(
        mapping_path=config.mapping_path,
        full_mapping_path=config.full_mapping_path,
        product_catalog_path=config.product_catalog_path,
        agent_policy_path=config.agent_policy_path,
    )


def _request(channel: DecisionChannel, **updates) -> ActionDecisionRequest:
    values = {
        "request_id": f"r-{channel.value}",
        "session_id": "s1",
        "turn_id": "t1",
        "decision_point_id": "dp1",
        "channel": channel,
        "text": "你好",
    }
    values.update(updates)
    return ActionDecisionRequest(**values)


class _FakeClient:
    def __init__(self, code: str = "_ctrl_b") -> None:
        self.code = code
        self.requests = []
        self.aborted = []

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.requests.append((request_id, request))
        return SimpleNamespace(
            text=self.code,
            usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            weight_version="weights-1",
        )

    async def abort(self, request_id):
        self.aborted.append(request_id)


def test_catalog_partitions_body_and_expression_and_controls() -> None:
    registry = _registry()
    body = registry.resolve(_request(DecisionChannel.BODY))
    expression = registry.resolve(_request(DecisionChannel.EXPRESSION))
    assert sum(entry.kind == "action" for entry in body.entries) == 74
    assert sum(entry.kind == "action" for entry in expression.entries) == 8
    assert {entry.candidate_id for entry in body.entries if entry.kind != "action"} == {
        "000",
        "IB0",
    }
    assert {
        entry.candidate_id for entry in expression.entries if entry.kind != "action"
    } == {"000", "IF0"}


def test_registered_agent_binding_injects_and_hard_binds_full_mapping_action() -> None:
    registry = _registry()
    catalog = registry.resolve(
        _request(
            DecisionChannel.BODY,
            runtime_context={"agent_event": "agent.search.started"},
        )
    )
    assert catalog.hard_bound is not None
    assert catalog.hard_bound.candidate_id == "481"
    expression = registry.resolve(
        _request(
            DecisionChannel.EXPRESSION,
            runtime_context={"agent_event": "agent.search.started"},
        )
    )
    assert expression.hard_bound is None
    assert "required_action_candidate_id" not in expression.runtime_context


def test_disabled_channel_is_deterministic_and_skips_model() -> None:
    async def run():
        client = _FakeClient()
        engine = ActionDecisionEngine(client, config=_config())
        result = await engine.decide(
            _request(DecisionChannel.EXPRESSION, channel_enabled=False)
        )
        return client, result

    client, result = asyncio.run(run())
    assert (result.outcome, result.candidate_id) == ("keep", "IF0")
    assert result.deterministic is True and result.model_invoked is False
    assert client.requests == []


def test_character_prompt_accepts_d_stage_response_budget() -> None:
    request = _request(DecisionChannel.BODY, character_prompt="角" * 32_000)
    prompt = build_system_prompt(request, _registry().resolve(request))
    assert "角" * 32_000 in prompt


def test_e57_prompt_contract_preserves_production_channel_and_catalog() -> None:
    request = _request(
        DecisionChannel.BODY,
        character_prompt="角色提示",
        session_prompt="会话提示",
        runtime_context={"turn_origin": "user"},
    )
    catalog = _registry().resolve(request)
    prompt = build_system_prompt(request, catalog)
    assert "[SELECTION_RULES]" in prompt
    assert "[ACTIVE_CHANNEL]" in prompt
    assert "本次只决定身体动作通道" in prompt
    assert "[CHARACTER_PROMPT]\n角色提示" in prompt
    assert "[SESSION_PROMPT]\n会话提示" in prompt
    assert f"candidate_count={len(catalog.entries)}" in prompt
    assert "[FINAL_TASK_CONSTRAINTS]" in prompt
    assert "结果映射：1=_x_b，2=_x_c" in prompt


def test_e57_reference_profile_uses_exact_training_prompts_without_asr() -> None:
    audio_payload = b"\0\0"
    image_payload = b"jpeg"
    media = tuple(
        StreamedMedia(
            f"i{index}",
            "image",
            index * 100,
            index * 100 + 50,
            "image/jpeg",
            "sha256:" + hashlib.sha256(image_payload).hexdigest(),
            image_payload,
        )
        for index in range(5)
    ) + (
        StreamedMedia(
            "a1",
            "audio",
            0,
            100,
            "pcm16",
            "sha256:" + hashlib.sha256(audio_payload).hexdigest(),
            audio_payload,
        ),
    )
    request = _request(
        DecisionChannel.BODY,
        text="这个 ASR 文本不得进入音频模式提示词",
        media=media,
        runtime_context={"prompt_profile": "e57a_eval"},
    )
    catalog = _registry().resolve(request)
    expected_system = E57A_REFERENCE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    assert build_system_prompt(request, catalog) == expected_system
    user_prompt = build_user_prompt(request, catalog)
    assert "capture_mode=ptt_utterance" in user_prompt
    assert "input_frame_count=5" in user_prompt
    assert "trigger=user_input" in user_prompt
    assert "ASR" not in user_prompt


def test_character_prompt_rejects_only_above_d_stage_response_budget() -> None:
    request = _request(DecisionChannel.BODY, character_prompt="角" * 32_001)
    with pytest.raises(ValueError, match="exceeds 32000 characters"):
        build_system_prompt(request, _registry().resolve(request))


def test_model_request_has_exact_constraint_and_audio_plus_multiple_images() -> None:
    audio_payload = b"\0\0"
    audio = StreamedMedia(
        "a1",
        "audio",
        0,
        100,
        "pcm16",
        "sha256:" + hashlib.sha256(audio_payload).hexdigest(),
        audio_payload,
    )
    images = tuple(
        StreamedMedia(
            f"i{index}",
            "image",
            index * 100,
            index * 100 + 50,
            "image/jpeg",
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            payload,
        )
        for index, payload in enumerate((b"jpeg-one", b"jpeg-two"), 1)
    )

    async def run():
        client = _FakeClient("_ctrl_b")
        engine = ActionDecisionEngine(client, config=_config())
        result = await engine.decide(
            _request(DecisionChannel.BODY, media=(audio, *images))
        )
        return client, result

    client, result = asyncio.run(run())
    request = client.requests[0][1]
    assert result.outcome == "no_action"
    assert (request.sampling.min_new_tokens, request.sampling.max_new_tokens) == (2, 2)
    assert request.sampling.ignore_eos is True and "_ctrl_b" in request.sampling.regex
    assert len(request.metadata["audios"]) == 1
    assert len(request.metadata["images"]) == 2
    assert request.metadata["task"] == "action_direct"
    assert request.metadata["session_instance_id"] == "s1"
    assert [part["type"] for part in request.messages[-1].content] == [
        "text",
        "audio",
        "image",
        "image",
    ]


def test_e57_reference_request_places_five_images_and_audio_before_text() -> None:
    audio_payload = b"\0\0"
    image_payload = b"jpeg"
    images = tuple(
        StreamedMedia(
            f"i{index}", "image", index, index + 1, "image/jpeg",
            "sha256:" + hashlib.sha256(image_payload).hexdigest(), image_payload,
        )
        for index in range(5)
    )
    audio = StreamedMedia(
        "a1", "audio", 0, 100, "pcm16",
        "sha256:" + hashlib.sha256(audio_payload).hexdigest(), audio_payload,
    )

    async def run():
        client = _FakeClient("_ctrl_b")
        engine = ActionDecisionEngine(client, config=_config())
        await engine.decide(
            _request(
                DecisionChannel.BODY,
                media=(*images, audio),
                runtime_context={"prompt_profile": "e57a_eval"},
            )
        )
        return client.requests[0][1]

    request = asyncio.run(run())
    assert [part["type"] for part in request.messages[-1].content] == [
        "image", "image", "image", "image", "image", "audio", "text"
    ]


def test_model_may_return_canonical_candidate_id_for_control_token() -> None:
    async def run():
        engine = ActionDecisionEngine(_FakeClient("IB0"), config=_config())
        return await engine.decide(_request(DecisionChannel.BODY))

    result = asyncio.run(run())

    assert result.outcome == "no_action"
    assert result.candidate_id == "IB0"


def test_completed_request_is_idempotent() -> None:
    async def run():
        client = _FakeClient()
        engine = ActionDecisionEngine(client, config=_config())
        request = _request(DecisionChannel.BODY)
        first = await engine.decide(request)
        second = await engine.decide(request)
        return client, first, second

    client, first, second = asyncio.run(run())
    assert first == second
    assert len(client.requests) == 1


def _decision(channel: DecisionChannel, *, candidate_id: str, outcome: str, occupies=()):
    return ActionDecision(
        1,
        f"r-{channel.value}",
        "s1",
        "t1",
        "dp1",
        channel,
        outcome,
        candidate_id,
        candidate_id,
        "_x_x",
        (1, 2),
        "test",
        tuple(occupies),
        False,
        True,
        "model_selected",
        "m",
        "v",
        "w",
        "c",
        "ch",
        "mv",
        "mh",
        1,
        1.0,
    )


def test_face_occupying_body_suppresses_or_delays_expression() -> None:
    body = _decision(
        DecisionChannel.BODY, candidate_id="287", outcome="execute", occupies=("face",)
    )
    expression = _decision(
        DecisionChannel.EXPRESSION, candidate_id="156", outcome="apply"
    )
    assert fuse_action_decisions(body, expression).expression_publication == "suppress"
    assert (
        fuse_action_decisions(body, expression, expression_priority="explicit")
        .expression_publication
        == "delay"
    )


def test_websocket_service_accepts_text_request() -> None:
    client = _FakeClient()
    app = create_action_decision_app(client, config=_config())
    with TestClient(app).websocket_connect(
        "/v1/action-decision/realtime",
        headers={"Authorization": "Bearer secret"},
    ) as socket:
        socket.send_json(
            {
                "type": "request.start",
                "contract_version": 1,
                "request_id": "r1",
                "payload": {
                    "request_id": "r1",
                    "session_id": "s1",
                    "turn_id": "t1",
                    "decision_point_id": "dp1",
                    "channel": "body",
                    "text": "你好",
                },
            }
        )
        assert socket.receive_json()["type"] == "request.ready"
        socket.send_json({"type": "request.commit", "request_id": "r1"})
        response = socket.receive_json()
    assert response["type"] == "response.completed"
    assert response["response"]["channel"] == "body"


def test_action_service_mounts_authenticated_performance_control() -> None:
    app = create_action_decision_app(
        _FakeClient(),
        config=_config(),
        performance_token="performance-secret",
    )

    assert "/v1/performance-control" in app.openapi()["paths"]
    with TestClient(app).websocket_connect(
        "/v1/performance-control/realtime",
        headers={"Authorization": "Bearer performance-secret"},
    ):
        pass


def test_sampling_params_carry_server_side_constraints() -> None:
    from sglang_omni.client.types import SamplingParams

    raw = SamplingParams(
        max_new_tokens=2,
        min_new_tokens=2,
        ignore_eos=True,
        regex="(?:_ctrl_a|_ctrl_b)",
    ).to_dict()
    assert raw["min_new_tokens"] == 2
    assert raw["ignore_eos"] is True
    assert raw["regex"] == "(?:_ctrl_a|_ctrl_b)"


@pytest.mark.asyncio
async def test_action_role_requires_private_token_before_pipeline_start(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "action-decision")
    monkeypatch.delenv("SGLANG_OMNI_ACTION_DECISION_TOKEN", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_MODEL_VERSION", "e43")
    with pytest.raises(ValueError, match="ACTION_DECISION_TOKEN"):
        await _run_server(object())


@pytest.mark.asyncio
async def test_action_role_requires_model_version_before_pipeline_start(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "action-decision")
    monkeypatch.setenv("SGLANG_OMNI_ACTION_DECISION_TOKEN", "secret")
    monkeypatch.delenv("SGLANG_OMNI_MODEL_VERSION", raising=False)
    with pytest.raises(ValueError, match="MODEL_VERSION"):
        await _run_server(object())
