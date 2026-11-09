"""Expression-only protocol checks; no semantic inference or training."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from sglang_omni.serve.action_decision.contracts import DecisionChannel
from sglang_omni.serve.action_decision.model_input import swift_user_text, user_message_content
from sglang_omni.serve.action_decision.prompt import build_system_prompt, build_user_prompt
from sglang_omni.serve.action_decision.service import ActionDecisionEngine, create_action_decision_app
from sglang_omni.serve.action_decision.usage_scenarios import load_usage_scenarios
from tests.unit_test.serve.test_persona_audio_window import Client, Socket, audio, request as body_request
from tests.unit_test.serve.test_persona_window_prompt import config, registry  # noqa: F401


def request(**changes):
    return body_request(channel=DecisionChannel.EXPRESSION,
                        media=(audio("user_audio", 0, 2000),), text="ASR_MUST_NOT_APPEAR",
                        runtime_context={
                            "prompt_profile": "persona-audio-window-v2", "decision_mode": "user_request",
                            "persona_revision": "expression-test", "last_dispatched_action": {
                                "candidate_id": "237", "status": "completed"},
                            "body_intent": "BODY_INTENT_MUST_NOT_APPEAR",
                            "body_task": "BODY_TASK_MUST_NOT_APPEAR",
                            "required_action_candidate_id": "260",
                            "required_action_semantics": "BODY_BINDING_MUST_NOT_APPEAR",
                            "avatar_state_description": "可信场景说明：正在耐心倾听",
                        }, **changes)


def rows(system):
    return {parts[1]: parts for line in system.splitlines()
            if line.startswith("_") and len(parts := line.split(" | ")) == 7}


def test_expression_catalog_preserves_all_registered_identities_and_authored_scenes(registry):
    req = request()
    catalog = registry.resolve(req)
    system = build_system_prompt(req, catalog)
    scenes = load_usage_scenarios()
    actual = rows(system)
    assert set(actual) == set(catalog.by_candidate_id)
    assert len(actual) == 14 and "000" in actual and "IF0" not in actual
    for entry in catalog.entries:
        if entry.candidate_id == "000":
            continue
        scene = scenes.entries[entry.candidate_id]
        row = actual[entry.candidate_id]
        assert entry.category_id == "19"
        assert row[:5] == [entry.code, entry.candidate_id, entry.label, entry.motion_definition, scene.meaning]
        assert scene.required_evidence in row[5]
        assert all(exclusion in row[5] for exclusion in scene.negative_scenarios)
        assert row[6] == scene.distinction
    assert "[EXPRESSION_CATALOG]" in system and "[ACTION_CATALOG]" not in system
    assert all(word not in system for word in ("tokenizer", "token", "IB0", "IF0", "_ctrl_f", "跨帧", "数字手势", "单/双手"))
    assert all(key not in system for key in ("catalog_version=", "usage_version=", "candidate_count=", "sha256="))


def test_expression_final_task_follows_media_and_context_without_body_history(registry):
    req = request()
    catalog = registry.resolve(req)
    user = build_user_prompt(req, catalog)
    content = user_message_content(req, catalog)
    assert [p["type"] for p in content] == ["text", "audio", "text", "text"]
    assert '"source":"user_audio"' in content[0]["text"]
    assert swift_user_text(req, catalog) == "".join(p["text"] if p["type"] == "text" else "<audio>" for p in content)
    assert user.index("[FINAL_TASK]") > user.index("[USER_OBSERVATION]")
    assert "EXPRESSION_CATALOG" in user.split("[FINAL_TASK]")[1]
    assert user.endswith("只输出所选项的 code。")
    assert "可信场景说明：正在耐心倾听" in user
    assert all(word not in user for word in ("BODY_", "LAST_DISPATCHED_ACTION", "ASR_MUST_NOT_APPEAR", "avatar_speech_audio"))


def test_expression_persona_is_dynamic_and_text_input_stays_original(registry):
    req = replace(request(), media=(), text="请做一个微笑表情")
    cat = registry.resolve(req)
    other = replace(req, character_prompt="活泼亲切的朋友")
    assert build_system_prompt(req, cat) != build_system_prompt(other, registry.resolve(other))
    assert build_user_prompt(req, cat) == build_user_prompt(other, registry.resolve(other))
    assert "请做一个微笑表情" in build_user_prompt(req, cat)


@pytest.mark.parametrize("code", ["_l_a", "_ctrl_a"])
def test_expression_subset_prompt_and_decoder_accept_expression_and_control_codes(code):
    async def run():
        client = Client(code)
        req = request(allowed_candidate_ids=("154", "161"))
        result = await ActionDecisionEngine(client, config=config()).decide(req)
        assert result.code == code
        call = client.calls[0]
        assert set(rows(call.messages[0].content)) == {"154", "161", "000"}
        assert all(code not in call.sampling.regex for code in ("_q_d", "_l_b", "_ctrl_f"))
    asyncio.run(run())


@pytest.mark.parametrize("code", ["_q_d", "_l_b", "_ctrl_f"])
def test_expression_subset_rejects_body_or_unlisted_expression(code):
    async def run():
        client = Client(code)
        with pytest.raises(RuntimeError, match="outside constrained"):
            await ActionDecisionEngine(client, config=config()).decide(request(allowed_candidate_ids=("154", "161")))
    asyncio.run(run())


def test_expression_empty_real_subset_is_rejected(registry):
    with pytest.raises(ValueError, match="at least one allowed real action"):
        registry.resolve(request(allowed_candidate_ids=("000", "IF0")))


def test_avatar_expression_uses_current_speech_without_body_window_state(registry):
    req = body_request(channel=DecisionChannel.EXPRESSION,
                       allowed_candidate_ids=("154", "161"), text="给我讲个笑话")
    cat = registry.resolve(req)
    system, user = build_system_prompt(req, cat), build_user_prompt(req, cat)
    content = user_message_content(req, cat)
    assert "[DECISION_TASK speech_window]" in system
    assert "[DECISION_TASK user_request]" not in system
    assert "尚未提供数字人回复语音" not in system
    assert set(rows(system)) == {"154", "161", "000"}
    assert [p["type"] for p in content].count("audio") == 2
    assert '"source":"avatar_speech_context"' in content[0]["text"]
    assert '"source":"avatar_speech_audio"' in content[3]["text"]
    assert "面部表情" in content[3]["text"] and "一个动作" not in content[3]["text"]
    assert "[USER_OBSERVATION]" not in user
    assert "[ORIGINAL_USER_CONTEXT]" in user and "给我讲个笑话" in user
    assert user.index("[FINAL_TASK]") > user.index("[ORIGINAL_USER_CONTEXT]")
    assert all(key not in user for key in ("LAST_DISPATCHED_ACTION", "current_action_id", "fulfilled_action_ids", "start_ms", "end_ms"))
    assert "不自动复制用户或故事人物的表情" in system


def test_avatar_expression_websocket_runs_separately_from_body():
    async def run():
        model = Client("_l_a")
        app = create_action_decision_app(model, config=config())
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/action-decision/realtime")
        req = body_request(channel=DecisionChannel.EXPRESSION, allowed_candidate_ids=("154", "161"))
        socket = Socket(req)
        await endpoint(socket)
        result = next(event for event in socket.sent if event["type"] == "response.completed")
        assert result["response"]["channel"] == "expression"
        assert result["response"]["candidate_id"] == "154"
        call = model.calls[0]
        assert len(call.metadata["audios"]) == 2
        assert set(rows(call.messages[0].content)) == {"154", "161", "000"}
    asyncio.run(run())
