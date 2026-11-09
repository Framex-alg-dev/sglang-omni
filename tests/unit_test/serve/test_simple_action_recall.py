"""Boundary and real model-input checks for the stateless audio HTTP adapter."""

import base64
import copy
import json

import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.action_decision.model_input import user_message_content
from sglang_omni.serve.action_decision.prompt import build_system_prompt, build_user_prompt
from sglang_omni.serve.action_decision.service import create_action_decision_app
from sglang_omni.serve.action_decision.simple_audio import parse_simple_audio_request
from tests.unit_test.serve.test_persona_audio_window import Client, audio, request as legacy_request
from tests.unit_test.serve.test_persona_window_prompt import config, registry  # noqa: F401


def payload(source="avatar_speech_audio"):
    return {"audio": {"data": base64.b64encode(audio().payload).decode(), "format": "wav"},
            "audio_source": source, "channel": "body", "character_prompt": "沉稳、耐心的老师",
            "allowed_candidate_ids": ["237", "661"]}


def test_five_fields_make_an_audio_request_without_external_timeline(registry):
    raw = payload(); req, public_id = parse_simple_audio_request(raw, registry)
    assert public_id and req.simple_audio
    assert req.media[0].payload == audio().payload
    assert req.runtime_context["decision_mode"] == "speech_window"
    system = build_system_prompt(req, registry.resolve(req))
    content = user_message_content(req, registry.resolve(req))
    visible = system + "\n" + "\n".join(p.get("text", "") for p in content)
    for hidden in ["start_ms", "end_ms", "state_at_ms", "current_action_remaining_ms",
                   "response_id", "output_epoch", "persona_revision", "AVATAR_SPEECH_WINDOW", "WINDOW_STATE_RULES"]:
        assert hidden not in visible, hidden
    assert "avatar_speech_audio" in visible and "[DECISION_TASK speech_window]" in system
    assert '"status":"unknown"' in visible
    assert "只输出所选动作的 code" in visible


def test_context_user_intent_and_actual_previous_action_preserved(registry):
    raw = payload(); raw.update(context_audio=copy.deepcopy(raw["audio"]), user_context="给我讲个笑话",
                                last_dispatched_action={"candidate_id": "253", "status": "completed"})
    req, _ = parse_simple_audio_request(raw, registry)
    cat = registry.resolve(req); content = user_message_content(req, cat)
    assert [m.evidence_role for m in req.media] == ["avatar_speech_context", "avatar_speech_audio"]
    assert [b["type"] for b in content].count("audio") == 2
    user = build_user_prompt(req, cat)
    assert "给我讲个笑话" in user and '"candidate_id":"253"' in user and '"status":"completed"' in user
    assert "253" not in cat.by_candidate_id  # history cannot expand legal choices
    assert "fulfilled_action_ids" not in user and "user_action_status" not in user


@pytest.mark.parametrize("ids", [[], ["661", "661"], ["unknown"], ["661", "unknown"], ["154"], ["000", "661"]])
def test_empty_invalid_duplicate_or_other_channel_candidates_never_widen_catalog(registry, ids):
    raw = payload(); raw["allowed_candidate_ids"] = ids
    with pytest.raises(ValueError): parse_simple_audio_request(raw, registry)


@pytest.mark.parametrize("source", ["user_audio", "avatar_speech_audio"])
@pytest.mark.parametrize("channel,real_count", [("body", 103), ("expression", 13)])
def test_omitted_candidates_use_the_full_current_channel(registry, source, channel, real_count):
    raw = payload(source); raw["channel"] = channel; raw.pop("allowed_candidate_ids")
    req, _ = parse_simple_audio_request(raw, registry)
    catalog = registry.resolve(req)
    actions = [entry for entry in catalog.entries if entry.kind == "action"]
    assert len(actions) == real_count
    assert all((entry.category_id == "19") == (channel == "expression") for entry in actions)
    assert len(catalog.entries) == real_count + 1 and "000" in catalog.by_candidate_id
    assert "IB0" not in catalog.by_candidate_id and "IF0" not in catalog.by_candidate_id
    model = Client("_q_d" if channel == "body" else "_l_a")
    with TestClient(create_action_decision_app(model, config=config())) as client:
        result = client.post("/v1/action-recall", json=raw, headers={"Authorization": "Bearer offline-test"})
    assert result.status_code == 200, result.text
    assert result.json()["channel"] == channel


def test_null_candidates_are_not_treated_as_an_omitted_filter(registry):
    raw = payload(); raw["allowed_candidate_ids"] = None
    with pytest.raises(ValueError, match="unique registered IDs"):
        parse_simple_audio_request(raw, registry)


@pytest.mark.parametrize("channel,code,ids", [
    ("body", "_q_d", ["661", "237"]),
    ("expression", "_l_a", ["154", "161"]),
])
@pytest.mark.parametrize("subset", [False, True])
def test_user_text_is_preserved_and_uses_the_same_channel_protocol(registry, channel, code, ids, subset):
    text = '  请先听我说完。\n这是用户键盘原文，不是ASR副本。  '
    raw = {"text": text, "channel": channel, "character_prompt": "耐心克制的老师"}
    if subset:
        raw["allowed_candidate_ids"] = ids
    req, _ = parse_simple_audio_request(raw, registry)
    catalog = registry.resolve(req)
    assert req.text == text and req.media == ()
    assert req.runtime_context["decision_mode"] == "user_request"
    parts = user_message_content(req, catalog)
    assert [part["type"] for part in parts] == ["text"]
    observation = parts[0]["text"].split("[USER_OBSERVATION]\n")[1].split("\n[FINAL_TASK]")[0]
    assert json.loads(observation)["typed_text"] == text
    assert "[DECISION_TASK user_request]" in build_system_prompt(req, catalog)
    assert "[DECISION_TASK speech_window]" not in build_system_prompt(req, catalog)
    assert "avatar_speech_audio" not in parts[0]["text"]
    if subset:
        assert set(catalog.by_candidate_id) == set(ids) | {"000"}
    model = Client(code)
    with TestClient(create_action_decision_app(model, config=config())) as client:
        response = client.post("/v1/action-recall", json=raw, headers={"Authorization": "Bearer offline-test"})
    assert response.status_code == 200, response.text
    assert response.json()["code"] == code and response.json()["channel"] == channel
    assert not model.calls[0].metadata.get("audios")


@pytest.mark.parametrize("text", [None, "", "   ", 123, "a" * 4001, "bad\x00text"])
def test_invalid_user_text_fails_before_inference(registry, text):
    with pytest.raises(ValueError, match="text must be"):
        parse_simple_audio_request({"text": text, "channel": "body", "character_prompt": "老师"}, registry)


@pytest.mark.parametrize("extra", [
    {"audio_source": "user_audio"}, {"audio_source": "avatar_speech_audio"},
    {"audio": {"data": "bad"}}, {"context_audio": {}}, {"user_context": "背景"},
])
def test_user_text_cannot_be_mixed_with_audio_or_avatar_context(registry, extra):
    with pytest.raises(ValueError, match="text alone"):
        parse_simple_audio_request({"text": "你好", "channel": "body", "character_prompt": "老师", **extra}, registry)


@pytest.mark.parametrize("channel,code,ids", [("body", "_q_d", ["661", "237"]), ("expression", "_l_a", ["154", "161"])])
@pytest.mark.parametrize("with_avatar_context", [False, True])
def test_optional_original_user_audio_has_separate_source_and_reply_clock(registry, channel, code, ids, with_avatar_context):
    raw = payload(); raw.update(channel=channel, allowed_candidate_ids=ids)
    original = audio("user_audio", 0, 5000)
    raw["user_context_audio"] = {"data": base64.b64encode(original.payload).decode(), "format": "wav"}
    if with_avatar_context:
        raw["context_audio"] = copy.deepcopy(raw["audio"])
    req, _ = parse_simple_audio_request(raw, registry); cat = registry.resolve(req)
    roles = ["user_context_audio"] + (["avatar_speech_context"] if with_avatar_context else []) + ["avatar_speech_audio"]
    assert [item.evidence_role for item in req.media] == roles
    assert req.media[0].payload == original.payload and req.text is None
    # User-turn duration never shifts the avatar reply timeline.
    assert req.media[-1].start_ms == (2000 if with_avatar_context else 0)
    content = user_message_content(req, cat)
    assert [json.loads(part["text"].split("\n", 1)[1])["source"] for part in content
            if part.get("text", "").startswith("[MEDIA_EVIDENCE]")] == roles
    assert "仅作为数字人本次回复的背景" in content[0]["text"]
    user = build_user_prompt(req, cat)
    assert '[ORIGINAL_USER_CONTEXT]\n{"audio":"user_context_audio"}' in user
    assert user.index("[FINAL_TASK]") > user.index("[ORIGINAL_USER_CONTEXT]")
    assert "user_context_audio" in build_system_prompt(req, cat)
    assert "start_ms" not in str(content)
    model = Client(code)
    with TestClient(create_action_decision_app(model, config=config())) as client:
        response = client.post("/v1/action-recall", json=raw, headers={"Authorization": "Bearer offline-test"})
    assert response.status_code == 200, response.text
    assert len(model.calls[0].metadata["audios"]) == len(roles)


def test_omitted_or_null_user_context_audio_preserves_existing_model_input(registry):
    raw = payload(); before, _ = parse_simple_audio_request(raw, registry)
    after, _ = parse_simple_audio_request({**raw, "user_context_audio": None}, registry)
    c1, c2 = registry.resolve(before), registry.resolve(after)
    assert build_system_prompt(before, c1) == build_system_prompt(after, c2)
    assert user_message_content(before, c1) == user_message_content(after, c2)


@pytest.mark.parametrize("source", ["user_audio", "text"])
def test_original_user_audio_context_only_belongs_to_avatar_speech(registry, source):
    raw = payload(source)
    if source == "text":
        raw = {"text": "你好", "channel": "body", "character_prompt": "老师"}
    raw["user_context_audio"] = {"data": base64.b64encode(audio().payload).decode()}
    with pytest.raises(ValueError):
        parse_simple_audio_request(raw, registry)


def test_original_user_context_cannot_mix_text_and_audio_and_counts_toward_budget(registry):
    raw = payload(); raw["user_context_audio"] = copy.deepcopy(raw["audio"])
    raw["user_context"] = "同一用户问题的文字"
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_simple_audio_request(raw, registry)
    raw.pop("user_context")
    raw["user_context_audio"] = {"data": base64.b64encode(audio(start=0, end=29_000).payload).decode()}
    with pytest.raises(ValueError, match="30 seconds"):
        parse_simple_audio_request(raw, registry)
    raw["user_context_audio"] = {"data": "invalid base64"}
    with pytest.raises(ValueError, match="base64"):
        parse_simple_audio_request(raw, registry)


def test_original_user_audio_bytes_are_part_of_idempotency(registry):
    raw = payload(); raw["user_context_audio"] = copy.deepcopy(raw["audio"])
    raw["request_id"] = "user-audio-retry"
    model = Client()
    with TestClient(create_action_decision_app(model, config=config())) as client:
        headers = {"Authorization": "Bearer offline-test"}
        first = client.post("/v1/action-recall", json=raw, headers=headers)
        assert first.status_code == 200
        assert client.post("/v1/action-recall", json=raw, headers=headers).json() == first.json()
        changed = copy.deepcopy(raw)
        changed["user_context_audio"]["data"] = base64.b64encode(audio(start=0, end=3000).payload).decode()
        assert client.post("/v1/action-recall", json=changed, headers=headers).status_code == 422
        assert len(model.calls) == 1


def test_user_text_retry_cannot_change_the_original_input(registry):
    model = Client("_q_d")
    raw = {"text": "请点头", "channel": "body", "character_prompt": "老师", "request_id": "typed-retry"}
    with TestClient(create_action_decision_app(model, config=config())) as client:
        headers = {"Authorization": "Bearer offline-test"}
        first = client.post("/v1/action-recall", json=raw, headers=headers)
        assert first.status_code == 200
        assert client.post("/v1/action-recall", json=raw, headers=headers).json() == first.json()
        changed = {**raw, "text": "请挥手"}
        assert client.post("/v1/action-recall", json=changed, headers=headers).status_code == 422
        assert len(model.calls) == 1


@pytest.mark.parametrize("change", [
    {"audio": {"data": "not-base64"}},
    {"audio": {"data": "YWJj", "format": "mp3"}},
    {"audio_source": "avatar_speech_context"},
    {"channel": "expression"},
    {"character_prompt": ""},
    {"start_ms": 123},
    {"runtime_context": {"prompt_profile": "e67_v116_step695_fp8"}},
    {"last_dispatched_action": {"candidate_id": "661", "status": "predicted"}},
])
def test_invalid_audio_sources_or_forged_control_fields_rejected(registry, change):
    raw = payload(); raw.update(change)
    with pytest.raises(ValueError): parse_simple_audio_request(raw, registry)


def test_user_expression_uses_its_own_prompt_and_cannot_consume_body_history(registry):
    raw = payload("user_audio"); raw.update(channel="expression", allowed_candidate_ids=["154", "155"])
    req, _ = parse_simple_audio_request(raw, registry); cat = registry.resolve(req)
    system = build_system_prompt(req, cat); user = build_user_prompt(req, cat)
    assert "EXPRESSION_CATALOG" in system and "ACTION_CATALOG" not in system
    assert "LAST_DISPATCHED_ACTION" not in user
    raw["last_dispatched_action"] = {"candidate_id": "661", "status": "running"}
    with pytest.raises(ValueError): parse_simple_audio_request(raw, registry)


@pytest.mark.parametrize("with_context", [False, True])
def test_avatar_expression_http_accepts_five_fields_and_optional_context(registry, with_context):
    raw = payload(); raw.update(channel="expression", allowed_candidate_ids=["154", "161"])
    if with_context:
        raw.update(context_audio=copy.deepcopy(raw["audio"]), user_context="昨天很开心，但今天很难过")
    model = Client("_l_a")
    with TestClient(create_action_decision_app(model, config=config())) as client:
        response = client.post("/v1/action-recall", json=raw, headers={"Authorization": "Bearer offline-test"})
    assert response.status_code == 200, response.text
    assert response.json()["candidate_id"] == "154" and response.json()["channel"] == "expression"
    call = model.calls[0]
    assert "[DECISION_TASK speech_window]" in call.messages[0].content
    assert len(call.metadata["audios"]) == (2 if with_context else 1)
    assert "_q_d" not in call.sampling.regex and "_ctrl_f" not in call.sampling.regex
    req, _ = parse_simple_audio_request(raw, registry)
    user = build_user_prompt(req, registry.resolve(req))
    assert all(key not in user for key in ("start_ms", "end_ms", "LAST_DISPATCHED_ACTION"))
    raw["last_dispatched_action"] = {"candidate_id": "661", "status": "running"}
    with pytest.raises(ValueError, match="only describes the body"):
        parse_simple_audio_request(raw, registry)


def test_length_limit_and_wav_format_apply_to_user_audio_too(registry):
    raw = payload("user_audio")
    raw["audio"]["data"] = base64.b64encode(audio(end=35_000, start=0).payload).decode()
    with pytest.raises(ValueError, match="30 seconds"): parse_simple_audio_request(raw, registry)
    raw = payload(); raw["context_audio"] = {"data": base64.b64encode(audio(start=0, end=29_000).payload).decode()}
    with pytest.raises(ValueError, match="30 seconds"): parse_simple_audio_request(raw, registry)


def test_legacy_timed_request_stays_timed(registry):
    req = legacy_request(); cat = registry.resolve(req)
    assert not req.simple_audio
    assert "[WINDOW_STATE_RULES]" in build_system_prompt(req, cat)
    assert "start_ms" in str(user_message_content(req, cat))
    assert "[AVATAR_SPEECH_WINDOW]" in build_user_prompt(req, cat)


def test_http_auth_inference_replay_and_conflicting_reuse(registry):
    model = Client(); app = create_action_decision_app(model, config=config())
    raw = payload(); raw["request_id"] = "agent-retry-1"
    headers = {"Authorization": "Bearer offline-test"}
    with TestClient(app) as client:
        assert client.post("/v1/action-recall", json=raw).status_code == 404
        first = client.post("/v1/action-recall", json=raw, headers=headers)
        assert first.status_code == 200, first.text
        assert first.json()["request_id"] == "agent-retry-1" and first.json()["candidate_id"] == "661"
        again = client.post("/v1/action-recall", json=raw, headers=headers)
        assert again.json() == first.json() and len(model.calls) == 1
        changed = copy.deepcopy(raw); changed["character_prompt"] = "活泼朋友"
        assert client.post("/v1/action-recall", json=changed, headers=headers).status_code == 422
        assert len(model.calls) == 1
        no_candidates = payload(); no_candidates["allowed_candidate_ids"] = []
        assert client.post("/v1/action-recall", json=no_candidates, headers=headers).status_code == 422
        assert len(model.calls) == 1
