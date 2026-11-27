from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
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
    build_system_prompt,
    build_user_prompt,
)
from sglang_omni.serve.action_decision.service import (
    ActionDecisionConfig,
    ActionDecisionEngine,
    create_action_decision_app,
)
from sglang_omni.serve.agent_prompt_snapshot import decode_action_prompt_snapshot
from sglang_omni.serve.streaming_request import StreamedMedia
from sglang_omni.serve.launcher import _run_server


import os
# Match deploy/launch_dual_gpu_dev.sh; callers can select an explicit artifact set.
ASSETS = os.getenv(
    "ACTION_ARTIFACT_ROOT",
    str(
        Path(__file__).resolve().parents[3]
        / "sglang_omni/assets/persona_window_v1"
    ),
)


def _config() -> ActionDecisionConfig:
    return ActionDecisionConfig(
        token="secret",
        model_id="action-model",
        model_version="e43",
        mapping_path=f"{ASSETS}/action_pair_token_map_current.json",
        full_mapping_path=f"{ASSETS}/action_pair_token_map_full.json",
        product_catalog_path=f"{ASSETS}/action_catalog.json",
    )


def _registry() -> ActionCatalogRegistry:
    config = _config()
    return ActionCatalogRegistry(
        mapping_path=config.mapping_path,
        full_mapping_path=config.full_mapping_path,
        product_catalog_path=config.product_catalog_path,
        agent_policy_path=config.agent_policy_path,
    )


def _raw_action_snapshot(prompt: str = "AGENT ACTION POLICY") -> dict:
    return {
        "consumer": "d_video_call.action.decision",
        "release_id": "action-test",
        "bundle_sha256": "a" * 64,
        "binding": {
            "assembly_contract": "d-action-decision-v1",
            "output_protocol": "d-action-decision-response-v1",
            "model_id": "action-model",
            "model_version": "e43",
        },
        "system_prompt": prompt,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def _request(channel: DecisionChannel, **updates) -> ActionDecisionRequest:
    values = {
        "request_id": f"r-{channel.value}",
        "session_id": "s1",
        "turn_id": "t1",
        "decision_point_id": "dp1",
        "channel": channel,
        "text": "你好",
        "prompt_snapshot": decode_action_prompt_snapshot(_raw_action_snapshot()),
    }
    values.update(updates)
    return ActionDecisionRequest(**values)


class _FakeClient:
    def __init__(self, code: str = "_ctrl_b") -> None:
        self.code = code
        self.requests = []
        self.prefill_requests = []
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

    async def prefill_completion_prefix(self, request, *, request_id):
        self.prefill_requests.append((request_id, request))
        return True


class _WarmupFakeClient(_FakeClient):
    async def completion(self, request, *, request_id, audio_format="wav"):
        self.requests.append((request_id, request))
        code = "_ctrl_b" if request.metadata["task_role"] == "body" else "_ctrl_f"
        return SimpleNamespace(
            text=code,
            usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            weight_version="weights-1",
        )


def test_action_warmup_exercises_both_channels_without_live_cache_or_cos(
    monkeypatch,
) -> None:
    async def run():
        recorded = []
        monkeypatch.setattr(
            "sglang_omni.serve.action_decision.service.get_model_io_recorder",
            lambda _stage: SimpleNamespace(record=lambda **value: recorded.append(value)),
        )
        client = _WarmupFakeClient()
        engine = ActionDecisionEngine(client, config=_config())
        result = await engine.warmup(timeout_s=1.0)
        return client, engine, result, recorded

    client, engine, result, recorded = asyncio.run(run())
    assert result["ready"] is True
    assert set(result["channels"]) == {"body", "expression"}
    assert [item[1].metadata["task_role"] for item in client.requests] == [
        "body",
        "expression",
    ]
    assert engine._completed == {}
    assert recorded == []


def test_action_warmup_failure_is_reported_without_raising() -> None:
    class FailingClient(_FakeClient):
        async def completion(self, request, *, request_id, audio_format="wav"):
            raise RuntimeError("warmup failed")

    result = asyncio.run(
        ActionDecisionEngine(FailingClient(), config=_config()).warmup(timeout_s=1.0)
    )
    assert result["ready"] is False
    assert result["channels"]["body"]["error_type"] == "RuntimeError"
    assert result["channels"]["expression"]["error_type"] == "RuntimeError"


def test_catalog_partitions_body_and_expression_and_controls() -> None:
    registry = _registry()
    body = registry.resolve(_request(DecisionChannel.BODY))
    expression = registry.resolve(_request(DecisionChannel.EXPRESSION))
    assert sum(entry.kind == "action" for entry in body.entries) >= 74
    assert sum(entry.kind == "action" for entry in expression.entries) >= 8
    assert {entry.candidate_id for entry in body.entries if entry.kind != "action"} == {
        "000",
        "IB0",
    }
    assert {
        entry.candidate_id for entry in expression.entries if entry.kind != "action"
    } == {"000", "IF0"}


@pytest.mark.skip(reason="V10 runtime validates candidate subsets without legacy role_pool_v1")
def test_role_pool_body_catalog_keeps_explicit_expression_actions() -> None:
    catalog = _registry().resolve(
        _request(
            DecisionChannel.BODY,
            allowed_candidate_ids=("138", "157"),
            runtime_context={"action_candidate_prompt_mode": "role_pool_v1"},
        )
    )

    assert {entry.candidate_id for entry in catalog.entries} == {
        "000",
        "138",
        "157",
        "IB0",
    }
    assert catalog.by_candidate_id["157"].occupies_channels == ("face",)

def test_turn_execution_fences_are_accepted_without_entering_model_prompt() -> None:
    request = _request(DecisionChannel.BODY, runtime_context={
        "identity_epoch": 3, "input_revision": 2,
        "expected_model_version": "e43", "expected_prompt_sha256": "a" * 64,
        "turn_origin": "user",
    })
    catalog = _registry().resolve(request)
    assert catalog.runtime_context == {"turn_origin": "user"}
    prompt = build_user_prompt(request, catalog)
    assert "expected_model_version" not in prompt
    assert "identity_epoch" not in prompt


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_turn_execution_revision_rejects_malformed_fences(value) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        _registry().resolve(_request(DecisionChannel.BODY,
                                    runtime_context={"input_revision": value}))


def test_advisory_model_binding_does_not_block_inference() -> None:
    async def run():
        client = _FakeClient()
        engine = ActionDecisionEngine(client, config=_config())
        result = await engine.decide(_request(
            DecisionChannel.BODY,
            runtime_context={"expected_model_version": "wrong-release"},
        ))
        assert result.model_invoked is True
        assert len(client.requests) == 1
    asyncio.run(run())


def test_unregistered_agent_event_is_rejected_without_policy() -> None:
    with pytest.raises(ValueError, match="unregistered agent_event"):
        _registry().resolve(
            _request(
                DecisionChannel.BODY,
                runtime_context={"agent_event": "agent.search.started"},
            )
        )


def test_agent_policy_mapping_mismatch_fails_during_startup(tmp_path) -> None:
    policy_path = tmp_path / "agent_action_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "bindings": {
                    "agent.search.started": {
                        "inject_candidate_ids": ["missing-candidate"],
                        "runtime_context": {
                            "agent_event": "agent.search.started",
                            "required_action_candidate_id": "missing-candidate",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = _config()

    with pytest.raises(
        ValueError,
        match="agent action policy candidates missing from full mapping",
    ):
        ActionCatalogRegistry(
            mapping_path=config.mapping_path,
            full_mapping_path=config.full_mapping_path,
            product_catalog_path=config.product_catalog_path,
            agent_policy_path=policy_path,
        )


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
    assert prompt.startswith("AGENT ACTION POLICY")
    assert "[TRUSTED_RUNTIME_CONTEXT]" in prompt
    assert "[CHARACTER_PERSONA]\n角色提示" in prompt
    assert "[SESSION_CONTEXT]\n会话提示" in prompt
    assert "[ACTION_CATALOG]" in prompt
    assert catalog.entries[0].candidate_id in prompt


def test_e65_reference_profile_uses_exact_training_prompts_without_asr() -> None:
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
            "user_camera",
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
            "user_audio",
        ),
    )
    request = _request(
        DecisionChannel.BODY,
        text="这个 ASR 文本不得进入音频模式提示词",
        media=media,
        runtime_context={"prompt_profile": "e65_step1512_fp8"},
    )
    catalog = _registry().resolve(request)
    assert build_system_prompt(request, catalog) == "AGENT ACTION POLICY"
    user_prompt = build_user_prompt(request, catalog)
    assert "user_video：按时间顺序提供的用户侧画面" in user_prompt
    assert "user_audio：这是本轮用户原始音频" in user_prompt
    assert "trigger=user_input" in user_prompt
    assert "ASR" not in user_prompt


def test_e66_reference_profile_uses_agent_prompt_snapshot() -> None:
    request = _request(
        DecisionChannel.BODY,
        runtime_context={"prompt_profile": "e66_v116_step935_fp8"},
    )
    catalog = _registry().resolve(request)
    assert build_system_prompt(request, catalog) == "AGENT ACTION POLICY"


def test_e67_reference_profile_uses_agent_prompt_snapshot() -> None:
    request = _request(
        DecisionChannel.BODY,
        runtime_context={"prompt_profile": "e67_v116_step695_fp8"},
    )
    catalog = _registry().resolve(request)
    assert build_system_prompt(request, catalog) == "AGENT ACTION POLICY"


@pytest.mark.skip(reason="V10 runtime validates candidate subsets without legacy role_pool_v1")
def test_e67_role_pool_prompt_filters_catalog_without_appending_a_constraint() -> None:
    request = _request(
        DecisionChannel.BODY,
        allowed_candidate_ids=("138", "157"),
        runtime_context={
            "prompt_profile": "e67_v116_step695_fp8",
            "action_candidate_prompt_mode": "role_pool_v1",
        },
    )
    catalog = _registry().resolve(request)

    prompt = build_system_prompt(request, catalog)

    assert "[GOAL_ROUTING]" in prompt
    assert "类别：视线类\n_i_a | 看镜头 |" in prompt
    assert "_i_c | 看右侧（仅眼球） |" not in prompt
    assert "类别：基础姿态" not in prompt
    assert catalog.by_candidate_id["157"].code in prompt
    assert "_ctrl_a | 不支持的具体动作 |" in prompt
    assert "本轮仅这些动作可用" not in prompt


@pytest.mark.skip(reason="V10 action-recall no longer exposes legacy prefix-prefill")
def test_role_pool_prefill_is_deduplicated_by_exact_prompt_and_catalog() -> None:
    async def run():
        client = _FakeClient()
        engine = ActionDecisionEngine(client, config=_config())
        context = {
            "prompt_profile": "e67_v116_step695_fp8",
            "action_candidate_prompt_mode": "role_pool_v1",
        }
        first = await engine.prefill(
            ActionPrefixPrefillRequest(
                request_id="prefill-1",
                session_id="session-1",
                channel=DecisionChannel.BODY,
                runtime_context=context,
                allowed_candidate_ids=("138",),
            )
        )
        second = await engine.prefill(
            ActionPrefixPrefillRequest(
                request_id="prefill-2",
                session_id="session-2",
                channel=DecisionChannel.BODY,
                runtime_context=context,
                allowed_candidate_ids=("138",),
            )
        )
        return client, first, second

    client, first, second = asyncio.run(run())

    assert len(client.prefill_requests) == 1
    assert first.cached is False
    assert second.cached is True
    assert first.prompt_hash == second.prompt_hash
    assert first.cache_key == second.cache_key
    assert first.candidate_count == 3
    prefetched = client.prefill_requests[0][1]
    assert hashlib.sha256(prefetched.messages[0].content.encode()).hexdigest() == (
        first.prompt_hash
    )


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
        "user_audio",
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
            "user_camera",
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


def test_e65_reference_request_places_five_images_and_audio_before_text() -> None:
    audio_payload = b"\0\0"
    image_payload = b"jpeg"
    images = tuple(
        StreamedMedia(
            f"i{index}", "image", index, index + 1, "image/jpeg",
            "sha256:" + hashlib.sha256(image_payload).hexdigest(), image_payload,
            "user_camera",
        )
        for index in range(5)
    )
    audio = StreamedMedia(
        "a1", "audio", 0, 100, "pcm16",
        "sha256:" + hashlib.sha256(audio_payload).hexdigest(), audio_payload,
        "user_audio",
    )

    async def run():
        client = _FakeClient("_ctrl_b")
        engine = ActionDecisionEngine(client, config=_config())
        await engine.decide(
            _request(
                DecisionChannel.BODY,
                media=(*images, audio),
                runtime_context={"prompt_profile": "e65_step1512_fp8"},
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


def test_health_reports_live_workload_queues_only_to_authenticated_clients() -> None:
    client = _FakeClient()
    load = {"capacity": 2, "active_total": 2, "waiting_body": 1,
            "waiting_route": 0, "waiting_presentation": 3}
    client.action_scoring_load = lambda: dict(load)
    client.health = lambda: {"running": True, "pending_completions": load["active_total"]}
    app = create_action_decision_app(client, config=_config())
    with TestClient(app) as http:
        assert http.get('/health').status_code == 404
        response = http.get('/health', headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200
        assert response.json()['runtime']['workload_admission'] == load
        load.update(active_total=0, waiting_body=0, waiting_presentation=0)
        refreshed = http.get('/health', headers={"Authorization": "Bearer secret"}).json()
        assert refreshed['runtime']['workload_admission']['active_total'] == 0
        assert refreshed['runtime']['pipeline']['pending_completions'] == 0
    assert client.requests == []


def test_health_rejects_traffic_until_enabled_startup_warmup_is_ready() -> None:
    app = create_action_decision_app(_FakeClient(), config=_config())
    app.state.action_decision_warmup = {"enabled": True, "ready": False}
    with TestClient(app) as http:
        response = http.get('/health', headers={"Authorization": "Bearer secret"})
        assert response.status_code == 503
        assert response.json()["ok"] is False

        app.state.action_decision_warmup = {"enabled": True, "ready": True}
        response = http.get('/health', headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200
        assert response.json()["ok"] is True


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
                "contract_version": 2,
                "request_id": "req_0123456789abcdfg",
                "payload": {
                    "request_id": "business-r1",
                    "session_id": "s1",
                    "turn_id": "t1",
                    "decision_point_id": "dp1",
                    "channel": "body",
                    "text": "你好",
                    "prompt_snapshot": _raw_action_snapshot(),
                },
            }
        )
        assert socket.receive_json()["type"] == "request.ready"
        socket.send_json(
            {
                "type": "request.commit",
                "request_id": "req_0123456789abcdfg",
            }
        )
        response = socket.receive_json()
    assert response["type"] == "response.completed"
    assert response["request_id"] == "req_0123456789abcdfg"
    assert response["response"]["channel"] == "body"
    assert client.requests[0][0] == "business-r1"


@pytest.mark.skip(reason="V10 action-recall no longer exposes legacy prefix-prefill")
def test_websocket_service_accepts_role_pool_prefill() -> None:
    client = _FakeClient()
    app = create_action_decision_app(client, config=_config())
    with TestClient(app).websocket_connect(
        "/v1/action-decision/realtime",
        headers={"Authorization": "Bearer secret"},
    ) as socket:
        socket.send_json(
            {
                "type": "request.start",
                "contract_version": 2,
                "request_id": "prefill-1",
                "payload": {
                    "operation": "prefill",
                    "request_id": "prefill-1",
                    "session_id": "s1",
                    "channel": "body",
                    "allowed_candidate_ids": ["138"],
                    "runtime_context": {
                        "prompt_profile": "e67_v116_step695_fp8",
                        "action_candidate_prompt_mode": "role_pool_v1",
                    },
                },
            }
        )
        assert socket.receive_json()["type"] == "request.ready"
        socket.send_json({"type": "request.commit", "request_id": "prefill-1"})
        response = socket.receive_json()
    assert response["type"] == "response.completed"
    assert response["response"]["cached"] is False
    assert response["response"]["candidate_count"] == 3
    assert len(client.prefill_requests) == 1


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


def test_cancel_removes_shielded_body_job_before_model_admission():
    import asyncio
    async def run():
        started, stopped = asyncio.Event(), asyncio.Event()
        class QueuedEngine(ActionDecisionEngine):
            async def _decide_once(self, request):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
        client = _FakeClient()
        engine = QueuedEngine(client, config=_config(), registry=_registry())
        req = _request(DecisionChannel.BODY)
        waiter = asyncio.create_task(engine.decide(req))
        await started.wait()
        await engine.cancel(req.request_id)
        await asyncio.gather(waiter, return_exceptions=True)
        assert stopped.is_set()
        assert client.aborted == [req.request_id]
        retry = asyncio.create_task(engine.decide(req))
        await asyncio.sleep(0)
        assert retry.cancelled()
    asyncio.run(run())


@pytest.mark.parametrize("input_fields", [
    {"text": "你好"},
    {"mode": "user_reaction", "user_input": {"text": "你好"}},
])
def test_action_recall_accepts_legacy_and_named_inputs_over_both_transports(
    monkeypatch, input_fields
) -> None:
    from sglang_omni.serve.action_decision import service

    seen = []

    async def decide(_engine, request, *, operation):
        assert operation == "action_recall"
        seen.append(request)
        return SimpleNamespace(
            channel=request.channel, candidate_id="665", code="test", label="点头",
            outcome="selected", occupies_channels=("body",),
        )

    monkeypatch.setattr(service, "_decide_with_service_log", decide)
    app = create_action_decision_app(_FakeClient(), config=_config())
    payload = {
        "channel": "body",
        "character_prompt": "自然回应",
        "prompt_snapshot": _raw_action_snapshot(),
        **input_fields,
    }
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.post("/v1/action-recall", json=payload, headers=headers).status_code == 200
        with client.websocket_connect("/v1/action-recall", headers=headers) as socket:
            operation_id = "req_0123456789abcdef"
            trace_id = "req_fedcba9876543210"
            socket.send_json({
                "type": "request.start", "contract_version": 2, "request_id": operation_id,
                "payload": payload,
                "log_context": {
                    "request_id": trace_id,
                    "root_session_id": "session-root",
                    "agent_session_id": "sess-browser",
                    "turn_id": "turn-1",
                },
            })
            assert socket.receive_json()["type"] == "request.ready"
            socket.send_json({"type": "request.commit", "request_id": operation_id})
            result = socket.receive_json()
            assert result["type"] == "response.completed", result
            assert result["request_id"] == operation_id
    assert seen[0].trace_request_id
    assert seen[1].trace_request_id == trace_id
    assert seen[1].root_session_id == "session-root"
    assert seen[1].agent_session_id == "sess-browser"
    assert seen[1].trace_turn_id == "turn-1"


def test_action_public_transports_reject_missing_agent_prompt_snapshot() -> None:
    app = create_action_decision_app(_FakeClient(), config=_config())
    headers = {"Authorization": "Bearer secret"}
    payload = {"channel": "body", "character_prompt": "自然回应", "text": "你好"}
    with TestClient(app) as client:
        response = client.post("/v1/action-recall", json=payload, headers=headers)
        assert response.status_code == 422
        assert "requires Agent prompt_snapshot" in response.text

        with client.websocket_connect("/v1/action-recall", headers=headers) as socket:
            socket.send_json(
                {
                    "type": "request.start",
                    "contract_version": 2,
                    "request_id": "req_0123456789abcdef",
                    "payload": payload,
                }
            )
            assert socket.receive_json()["type"] == "request.ready"
            socket.send_json(
                {
                    "type": "request.commit",
                    "request_id": "req_0123456789abcdef",
                }
            )
            rejected = socket.receive_json()
            assert rejected["type"] == "error"
            assert "requires Agent prompt_snapshot" in rejected["detail"]


def test_action_model_binding_drift_is_logged_without_blocking(monkeypatch) -> None:
    from sglang_omni.serve.action_decision import service

    events = []
    real_emit_service_event = service.emit_service_event

    def capture_event(component, event, **fields):
        real_emit_service_event(component, event, **fields)
        events.append((component, event, fields))

    monkeypatch.setattr(
        service,
        "emit_service_event",
        capture_event,
    )
    snapshot = _raw_action_snapshot()
    snapshot["binding"]["model_id"] = "knowledge-logical-action"
    snapshot["binding"]["model_version"] = "knowledge-release"
    payload = {
        "request_id": "business-r1",
        "session_id": "s1",
        "turn_id": "t1",
        "decision_point_id": "dp1",
        "channel": "body",
        "text": "你好",
        "prompt_snapshot": snapshot,
    }

    app = create_action_decision_app(_FakeClient(), config=_config())
    with TestClient(app) as client:
        response = client.post(
            "/v1/action-decision",
            json=payload,
            headers={"Authorization": "Bearer secret"},
        )

    assert response.status_code == 200, response.text
    drift = next(fields for _, event, fields in events if event == "prompt.binding.drift")
    assert drift["bound_model_id"] == "knowledge-logical-action"
    assert drift["serving_model_id"] == "action-model"
