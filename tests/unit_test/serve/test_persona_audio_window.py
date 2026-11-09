"""Audio contract tests use silence fixtures, never model-quality labels."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import wave
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sglang_omni.serve.action_decision.contracts import ActionDecisionRequest, DecisionChannel
from sglang_omni.serve.action_decision.model_input import swift_user_text, user_message_content
from sglang_omni.serve.action_decision.prompt import build_system_prompt, build_user_prompt
from sglang_omni.serve.action_decision.service import ActionDecisionEngine, create_action_decision_app
from sglang_omni.serve.action_decision.speech_audio import ACTION_AUDIO_ROLES
from sglang_omni.serve.action_decision.speech_window import PERSONA_AUDIO_PROFILE
from sglang_omni.serve.action_decision.usage_scenarios import (
    BODY_USAGE_SCENARIOS_PATH, load_body_usage_scenarios, load_usage_scenarios, parse_usage_scenarios,
)
from sglang_omni.serve.streaming_request import StreamedMedia, receive_streamed_request
from tests.unit_test.serve.test_persona_window_prompt import config, registry  # noqa: F401


def audio(role="avatar_speech_audio", start=4000, end=6000, *, encoding="wav"):
    pcm = b"\x00\x00" * ((end - start) * 16)
    payload = pcm
    if encoding == "wav":
        out = io.BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(pcm)
        payload = out.getvalue()
    return StreamedMedia(role, "audio", start, end, encoding,
                         "sha256:" + hashlib.sha256(payload).hexdigest(), payload, role)


def request(*, context_audio=True, **changes):
    values = dict(
        request_id="audio-window-1", session_id="s1", turn_id="user-turn-1",
        decision_point_id="reply-1:4000:6000", channel=DecisionChannel.BODY,
        text="请比三", character_prompt="沉稳、耐心的老师", session_prompt="正在讲解题目",
        runtime_context={
            "prompt_profile": PERSONA_AUDIO_PROFILE, "decision_mode": "speech_window",
            "persona_revision": "persona-r3", "response_id": "reply-1", "output_epoch": 2,
            "last_dispatched_action": {"candidate_id": "237", "status": "sent"},
            "speech_window": {"window_id": "window-2", "start_ms": 4000, "end_ms": 6000,
                              "state_at_ms": 3000, "user_action_status": "completed",
                              "fulfilled_action_ids": ["260"]},
        },
        media=((audio("avatar_speech_context", 2000, 4000),) if context_audio else ()) + (audio(),),
    )
    values.update(changes)
    return ActionDecisionRequest(**values)


def test_audio_only_window_and_exact_serving_training_order(registry):
    req = request()
    cat = registry.resolve(req)
    system = build_system_prompt(req, cat)
    user = build_user_prompt(req, cat)
    content = user_message_content(req, cat)
    assert [p["type"] for p in content] == ["text", "audio", "text", "text", "audio", "text", "text"]
    assert "avatar_speech_context" in content[0]["text"]
    assert "avatar_speech_audio" in content[3]["text"]
    assert "原始用户音频" not in content[0]["text"] + content[3]["text"]
    assert swift_user_text(req, cat) == "".join(
        p["text"] if p["type"] == "text" else "<audio>" for p in content
    )
    assert "[AUDIO_SOURCE_RULES]" in system and "按音色猜测" in system
    assert "讲话窗口以 upcoming_text" not in system
    assert "[ORIGINAL_USER_CONTEXT]" in user and "请比三" in user
    assert '"upcoming_text":""' in user
    assert all(value not in user for value in ("reply-1", "window-2", "persona-r3", "output_epoch"))
    assert "LAST_DISPATCHED_ACTION" in system and "立即打断" in system
    assert "tokenizer" not in system and "token" not in system
    assert "IB0" not in system and "_ctrl_b" not in system
    assert "或保持" not in system and "选择保持" not in system
    assert len(cat.entries) == 104 and "IB0" not in cat.by_candidate_id
    assert '"label":"单手摊开展示"' in user and '"status":"sent"' in user


def test_user_audio_once_and_dynamic_persona(registry):
    req = request(media=(audio("user_audio", 0, 2000),), text="ASR_MUST_NOT_BE_COPIED",
                  runtime_context={"prompt_profile": PERSONA_AUDIO_PROFILE,
                                   "decision_mode": "user_request", "persona_revision": "p1",
                                   "last_dispatched_action": None})
    cat = registry.resolve(req)
    assert "ASR_MUST_NOT_BE_COPIED" not in swift_user_text(req, cat)
    assert "avatar_speech_audio" not in build_user_prompt(req, cat)
    other = replace(req, character_prompt="活泼热情的朋友")
    assert build_system_prompt(req, cat) != build_system_prompt(other, registry.resolve(other))
    assert build_user_prompt(req, cat) == build_user_prompt(other, registry.resolve(other))


def test_body_persona_style_is_preserved_without_a_second_action_whitelist(registry):
    persona = {"人设": "阿杰", "人设描述": "热情的健身教练", "动作频率": "高，幅度较大",
               "动作白名单": "正坐、目录外动作", "允许动作": ["不存在的动作"]}
    req = request(character_prompt=json.dumps(persona, ensure_ascii=False), allowed_candidate_ids=("661",))
    cat = registry.resolve(req)
    system = build_system_prompt(req, cat)
    description = json.loads(system.split("[CHARACTER_PERSONA]\n")[1].splitlines()[0])["description"]
    assert json.loads(description) == {key: value for key, value in persona.items()
                                      if key not in {"动作白名单", "允许动作"}}
    assert "正坐" not in system and "不存在的动作" not in system
    assert "按人设强度与历史衔接选择" in system and "结合数字人人设" in build_user_prompt(req, cat)
    assert set(cat.by_candidate_id) == {"661", "000"}
    changed = replace(req, character_prompt=json.dumps({**persona, "人设描述": "克制的商务助理"}, ensure_ascii=False))
    assert build_system_prompt(changed, registry.resolve(changed)) != system


def test_body_scene_evidence_and_exclusions_reach_actual_model_input(registry):
    req = request()
    cat = registry.resolve(req)
    system = build_system_prompt(req, cat)
    revised = load_body_usage_scenarios()
    old = load_usage_scenarios()
    rows = {parts[1]: parts for line in system.splitlines()
            if line.startswith("_") and len(parts := line.split(" | ")) == 8}
    assert set(revised.entries) == set(old.entries)
    assert revised.version == "v116-body-scenes-r4"
    for entry in cat.entries:
        if entry.candidate_id == "000":
            continue
        scene = revised.entries[entry.candidate_id]
        previous = old.entries[entry.candidate_id]
        assert (scene.code, scene.label, scene.category_id, scene.token_ids) == (
            previous.code, previous.label, previous.category_id, previous.token_ids)
        assert rows[entry.candidate_id][3] == entry.motion_definition
        assert scene.required_evidence in rows[entry.candidate_id][5]
        assert scene.exclusion_summary in rows[entry.candidate_id][5]
    assert sum(bool(e.exclusion_summary) for e in revised.entries.values()) == 103
    assert [e for e in revised.entries.values() if e.category_id == "19"] == [
        e for e in old.entries.values() if e.category_id == "19"]


@pytest.mark.parametrize("version", ["v116-body-scenes-r3", "v116-body-scenes-r4"])
def test_revised_body_scene_asset_requires_exclusions(version):
    raw = json.loads(BODY_USAGE_SCENARIOS_PATH.read_text())
    raw["version"] = version
    body = next(e for e in raw["entries"] if e["category_id"] != "19")
    body.pop("exclusion_summary")
    with pytest.raises(ValueError, match="exclusion_summary"):
        parse_usage_scenarios(json.dumps(raw).encode())


@pytest.mark.parametrize("context_audio,encoding", [(False, "wav"), (False, "pcm_s16le"), (True, "wav")])
def test_focus_only_or_bounded_context(registry, context_audio, encoding):
    req = request(context_audio=context_audio)
    req = replace(req, media=(*req.media[:-1], audio(encoding=encoding)))
    assert registry.resolve(req).runtime_context["speech_window"]["end_ms"] == 6000


@pytest.mark.parametrize("mutation,match", [
    ("user_audio", "focus audio"), ("mixed", "focus audio"), ("reversed", "focus audio"),
    ("missing_focus", "focus audio"), ("overlap", "immediately precede"),
    ("gap", "immediately precede"), ("time_mismatch", "range must equal"),
    ("wrong_duration", "duration does not match"), ("bad_wav", "invalid avatar WAV"),
    ("long", "30 seconds"),
    ("no_persona", "character_prompt"), ("no_revision", "persona_revision"),
    ("no_response", "response_id"), ("bad_epoch", "output_epoch"),
    ("v1", "requires persona-audio-window-v2"), ("legacy", "requires persona-audio-window-v2"),
])
def test_invalid_audio_windows_fail_before_model(registry, mutation, match):
    req = request()
    if mutation == "user_audio": req = replace(req, media=(audio("user_audio"),))
    elif mutation == "mixed": req = replace(req, media=(audio("user_audio", 0, 2000), audio()))
    elif mutation == "reversed": req = replace(req, media=tuple(reversed(req.media)))
    elif mutation == "missing_focus": req = replace(req, media=req.media[:1])
    elif mutation in {"overlap", "gap"}:
        req = replace(req, media=(audio("avatar_speech_context", 2000, 4100 if mutation == "overlap" else 3900), audio()))
    elif mutation == "time_mismatch": req = replace(req, media=(audio(start=4100),))
    elif mutation == "wrong_duration": req = replace(req, media=(replace(audio(), end_ms=6100),))
    elif mutation == "bad_wav": req = replace(req, media=(replace(audio(), payload=b"invalid wav file"),))
    elif mutation == "long":
        req.runtime_context["speech_window"].update(start_ms=0, end_ms=31000, state_at_ms=0)
        req = replace(req, media=(audio(start=0, end=31000),))
    elif mutation == "no_persona": req = replace(req, character_prompt="")
    elif mutation == "no_revision": req.runtime_context.pop("persona_revision")
    elif mutation == "no_response": req.runtime_context.pop("response_id")
    elif mutation == "bad_epoch": req.runtime_context["output_epoch"] = True
    elif mutation == "v1": req.runtime_context["prompt_profile"] = "persona-window-v1"
    else: req = replace(req, runtime_context={})
    if mutation == "wrong_duration":
        req.runtime_context["speech_window"]["end_ms"] = 6100
    with pytest.raises(ValueError, match=match): registry.resolve(req)


class Socket:
    """Drive the real endpoint without TestClient threads or a listening server."""

    def __init__(self, req):
        payload = {k: (v.value if isinstance(v, DecisionChannel) else v)
                   for k, v in req.__dict__.items() if k != "media"}
        payload = json.loads(json.dumps(payload))
        self.headers = {"authorization": "Bearer offline-test"}
        self.events = [{"type": "request.start", "contract_version": 2,
                        "request_id": req.request_id, "payload": payload}]
        for clip in req.media:
            self.events.extend(({"type": "input.media", "request_id": req.request_id, **clip.header()}, clip.payload))
        self.events.append({"type": "request.commit", "request_id": req.request_id})
        self.sent = []
        self.closed = None

    async def accept(self): pass
    async def close(self, code): self.closed = code
    async def send_json(self, data): self.sent.append(data)
    async def receive_json(self): return self.events.pop(0)
    async def receive_bytes(self): return self.events.pop(0)
    async def receive(self): await asyncio.Event().wait()


def test_transport_requires_explicit_avatar_audio_opt_in():
    async def run():
        with pytest.raises(ValueError, match="evidence_role"):
            await receive_streamed_request(Socket(request()))
        result = await receive_streamed_request(Socket(request()), allowed_audio_roles=ACTION_AUDIO_ROLES)
        assert [m.evidence_role for m in result.media] == ["avatar_speech_context", "avatar_speech_audio"]
        assert [m.payload for m in result.media] == [m.payload for m in request().media]
    asyncio.run(run())


class Client:
    def __init__(self, code="_q_d"):
        self.calls = []
        self.code = code
    async def completion(self, req, **kwargs):
        self.calls.append(req)
        return SimpleNamespace(text=self.code, usage=None, weight_version="mock")


def test_real_websocket_handler_preserves_two_audio_inputs_and_code():
    async def run():
        client = Client()
        app = create_action_decision_app(client, config=config())
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/action-decision/realtime")
        socket = Socket(request())
        await endpoint(socket)
        assert socket.sent[-1]["type"] == "response.completed", socket.sent[-1]
        assert socket.sent[-1]["response"]["code"] == "_q_d"
        call = client.calls[0]
        assert len(call.metadata["audios"]) == 2
        assert call.sampling.max_new_tokens == call.sampling.min_new_tokens == 2
        assert [p["type"] for p in call.messages[1].content].count("audio") == 2
    asyncio.run(run())


@pytest.mark.parametrize("channel,code", [(DecisionChannel.BODY, "_q_d"), (DecisionChannel.EXPRESSION, "_l_a")])
def test_websocket_original_user_audio_has_its_own_timeline(channel, code):
    async def run():
        client = Client(code)
        app = create_action_decision_app(client, config=config())
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/action-decision/realtime")
        req = request(text=None, channel=channel)
        req = replace(req, media=(audio("user_context_audio", 0, 7000), *req.media))
        socket = Socket(req)
        await endpoint(socket)
        assert socket.sent[-1]["type"] == "response.completed", socket.sent[-1]
        assert socket.sent[-1]["response"]["code"] == code
        call = client.calls[0]
        assert len(call.metadata["audios"]) == 3
        sources = [json.loads(p["text"].split("\n", 1)[1])["source"]
                   for p in call.messages[1].content if p.get("text", "").startswith("[MEDIA_EVIDENCE]")]
        assert sources == ["user_context_audio", "avatar_speech_context", "avatar_speech_audio"]
    asyncio.run(run())


def test_removed_keep_code_is_rejected_by_model_and_grammar():
    async def run():
        client = Client("_ctrl_b")
        with pytest.raises(RuntimeError, match="outside constrained"):
            await ActionDecisionEngine(client, config=config()).decide(request())
        assert "_ctrl_b" not in client.calls[0].sampling.regex
    asyncio.run(run())


@pytest.mark.parametrize("previous", [
    {"candidate_id": "IB0", "status": "sent"},
    {"candidate_id": "237", "status": "predicted"},
    {"candidate_id": "237", "status": "sent", "label": "伪造动作名"},
])
def test_previous_action_requires_a_real_catalog_action_and_execution_status(registry, previous):
    req = request()
    req.runtime_context["last_dispatched_action"] = previous
    with pytest.raises(ValueError, match="last_dispatched_action"):
        registry.resolve(req)


def test_previous_action_can_be_outside_current_candidates_without_becoming_selectable(registry):
    req = request(allowed_candidate_ids=("661",))
    cat = registry.resolve(req)
    assert set(cat.by_candidate_id) == {"661", "000"}
    assert cat.runtime_context["last_dispatched_action"]["label"] == "单手摊开展示"
    assert "单手摊开展示" in build_user_prompt(req, cat)


def test_subset_prompt_and_decoder_share_the_same_per_request_boundary():
    async def run():
        engine_client = Client("_q_d")
        req = request(allowed_candidate_ids=("138", "661"))
        result = await ActionDecisionEngine(engine_client, config=config()).decide(req)
        assert result.code == "_q_d"
        call = engine_client.calls[0]
        prompt = call.messages[0].content
        assert {line.split(" | ")[1] for line in prompt.splitlines() if line.startswith("_")} == {"138", "661", "000"}
        assert all(key not in prompt for key in ("candidate_count=", "catalog_version=", "usage_version="))
        assert "_q_d" in call.sampling.regex and "_ctrl_a" in call.sampling.regex
        assert "_w_a" not in call.sampling.regex  # Previous dispatched action is history only.
        outside = Client("_w_a")
        with pytest.raises(RuntimeError, match="outside constrained"):
            await ActionDecisionEngine(outside, config=config()).decide(req)
    asyncio.run(run())


def test_no_previous_action_is_explicit_and_not_inferred(registry):
    req = request()
    req.runtime_context["last_dispatched_action"] = None
    assert registry.resolve(req).runtime_context["last_dispatched_action"] == {"status": "unknown"}
    req.runtime_context.pop("last_dispatched_action")
    with pytest.raises(ValueError, match="last_dispatched_action"):
        registry.resolve(req)


def test_persona_and_window_identity_are_part_of_idempotency():
    async def run():
        client = Client()
        engine = ActionDecisionEngine(client, config=config())
        req = request()
        await engine.decide(req)
        for field, value in (("persona_revision", "p2"), ("response_id", "reply-2"), ("output_epoch", 3)):
            changed = replace(req, runtime_context={**req.runtime_context, field: value})
            with pytest.raises(ValueError, match="reused with different input"):
                await engine.decide(changed)
        with pytest.raises(ValueError, match="reused with different input"):
            await engine.decide(replace(req, character_prompt="活泼的朋友"))
        await engine.decide(replace(req, request_id="window-2", decision_point_id="dp2"))
        assert len(client.calls) == 2
    asyncio.run(run())
