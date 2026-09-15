from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from sglang_omni.serve.realtime.knowledge.turn_context_gate import (
    MAX_HINT_CHARS,
    USER_KNOWLEDGE_GATE_ENV,
    build_knowledge_hint,
    filter_podcast_context,
    resolve_knowledge_gate,
)
from sglang_omni.serve.realtime.turn_intent import (
    KNOWLEDGE_SYSTEM,
    SYSTEM,
    TurnIntent,
    infer_turn_intent,
)


def intent_payload(**changes):
    data = dict(speech="generated", text="去健身吧", body="", body_mode="none",
                face="", history=False)
    data.update(changes)
    return data


def parsed_intent(**changes):
    return TurnIntent.parse(json.dumps(intent_payload(**changes)))


def podcast_text(**changes):
    data = dict(podcast_title="新闻标题", language="en-US",
                current_unit_id="unit-1", playback_precision="partial_current_unit",
                progress={"current": 2, "total": 7, "article": "NESTED_NEWS"},
                recently_completed=["PAST_NEWS"], interrupted_text="CURRENT_NEWS",
                resume_text="FUTURE_NEWS", unexpected_body="EXTRA_NEWS")
    data.update(changes)
    return ("INTERNAL PODCAST CONTEXT\nUntrusted data\n"
            + json.dumps(data, ensure_ascii=False)
            + "\nEND INTERNAL PODCAST CONTEXT\nDefault response language: English.")


def state(need=False, *, context=None, enabled=True, mode="provided_context"):
    session = SimpleNamespace(
        user_knowledge_gate_enabled=enabled,
        knowledge_binding=SimpleNamespace(mode=mode),
        provided_entity_context=object(),
        provided_entity_snapshot=SimpleNamespace(
            current_entity_text=json.dumps({"title": "AI新闻", "body": "FULL_NEWS"}),
        ),
    )
    turn = SimpleNamespace(turn_id="test-turn", turn_origin="user", reply_provided=False,
                           reply_context=context, intent=parsed_intent(needs_knowledge=need))
    return session, turn


@pytest.mark.parametrize("value", [True, False, None, "false", "true", 0, 1, [], {}])
def test_only_boolean_false_can_suppress(value):
    session, turn = state(value)
    decision = resolve_knowledge_gate(session, turn)
    assert decision.applied
    assert decision.include_context is (value is not False)
    assert turn.intent.text == "去健身吧"
    assert turn.intent.body_mode == "none"
    if type(value) is not bool:
        assert decision.reason == "invalid_field"


def test_missing_field_and_failed_intent_include():
    session, turn = state()
    turn.intent = parsed_intent()
    assert resolve_knowledge_gate(session, turn).reason == "missing_field"
    turn.intent = None
    gate = resolve_knowledge_gate(session, turn)
    assert gate.include_context and gate.reason == "intent_fallback"


@pytest.mark.parametrize("case", ["off", "retrieval", "proactive", "provided"])
def test_legacy_and_non_user_paths_are_unchanged(case):
    session, turn = state(context=podcast_text())
    if case == "off":
        session.user_knowledge_gate_enabled = False
    elif case == "retrieval":
        session.knowledge_binding.mode = "retrieval"
    elif case == "proactive":
        turn.turn_origin = "proactive"
    else:
        turn.reply_provided = True
    gate = resolve_knowledge_gate(session, turn)
    assert not gate.applied and gate.include_context
    assert filter_podcast_context(turn.reply_context, gate) == turn.reply_context


def test_no_context_is_not_fabricated_and_next_turn_rechecks():
    session, turn = state(True)
    session.provided_entity_context = None
    assert resolve_knowledge_gate(session, turn).reason == "no_context"
    session.provided_entity_context = object()
    assert resolve_knowledge_gate(session, turn).include_context
    turn.intent = parsed_intent(needs_knowledge=False)
    assert not resolve_knowledge_gate(session, turn).include_context
    turn.intent = parsed_intent(needs_knowledge=True)
    assert resolve_knowledge_gate(session, turn).include_context


def test_podcast_only_and_metadata_filter_preserve_language_not_articles():
    session, turn = state(context=podcast_text())
    session.knowledge_binding = None
    session.provided_entity_context = None
    gate = resolve_knowledge_gate(session, turn)
    assert gate.applied and not gate.include_context
    filtered = filter_podcast_context(turn.reply_context, gate)
    for token in ("PAST_NEWS", "CURRENT_NEWS", "FUTURE_NEWS", "EXTRA_NEWS", "NESTED_NEWS", "新闻标题"):
        assert token not in filtered
    assert "en-US" in filtered and "unit-1" in filtered
    assert '"current":2' in filtered and '"total":7' in filtered
    assert filtered.endswith("Default response language: English.")


@pytest.mark.parametrize("bad", [
    "INTERNAL PODCAST CONTEXT not JSON",
    "INTERNAL PODCAST CONTEXT {invalid}\nEND INTERNAL PODCAST CONTEXT",
    "INTERNAL PODCAST CONTEXT {} missing end marker",
    "unexpected prefix INTERNAL PODCAST CONTEXT {}\nEND INTERNAL PODCAST CONTEXT",
])
def test_malformed_podcast_fails_open_for_both_inputs(bad):
    session, turn = state(context=bad)
    gate = resolve_knowledge_gate(session, turn)
    assert gate.include_context and gate.reason == "context_parse_fallback"
    assert filter_podcast_context(bad, gate) == bad


def test_non_podcast_reply_context_is_untouched():
    session, turn = state(context="用户此前要求用英语回答。")
    gate = resolve_knowledge_gate(session, turn)
    assert not gate.include_context
    assert filter_podcast_context(turn.reply_context, gate) == turn.reply_context


def test_hint_is_bounded_titles_only_and_tracks_snapshot():
    session, turn = state(context=podcast_text())
    hint = build_knowledge_hint(session, turn)
    assert len(hint) <= MAX_HINT_CHARS
    assert "AI新闻" in hint and "新闻标题" in hint
    assert "FULL_NEWS" not in hint and "CURRENT_NEWS" not in hint
    session.provided_entity_snapshot.current_entity_text = json.dumps({"title": "新稿标题"})
    assert "新稿标题" in build_knowledge_hint(session, turn)
    assert "AI新闻" not in build_knowledge_hint(session, turn)
    session.provided_entity_snapshot.current_entity_text = "not-json FULL_NEWS"
    assert "FULL_NEWS" not in build_knowledge_hint(session, turn)
    session.provided_entity_snapshot.current_entity_text = json.dumps({"title": '"\\' * 1000})
    assert len(build_knowledge_hint(session, turn)) <= MAX_HINT_CHARS


def test_opt_in_prompt_examples_have_new_field_without_changing_legacy():
    assert "needs_knowledge" not in SYSTEM
    examples = [json.loads(m.group()) for m in re.finditer(r'\{[^{}\n]+\}', KNOWLEDGE_SYSTEM)]
    assert len(examples) >= 10
    assert all(type(example["needs_knowledge"]) is bool for example in examples)
    assert "共6个字段" not in KNOWLEDGE_SYSTEM
    assert "一律true" in KNOWLEDGE_SYSTEM


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["false", "bad-json", "missing"])
async def test_classification_is_one_call_with_stable_prompt_and_audio(raw):
    requests = []
    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            text = ("not json" if raw == "bad-json" else json.dumps(
                intent_payload(**({} if raw == "missing" else {"needs_knowledge": False}))))
            return SimpleNamespace(text=text)
    session, turn = state()
    session.client = Client()
    session.model_name = "fake-model"
    session.session_id = "session-test"
    session._register_turn_request = lambda *args: None
    session._unregister_turn_request = lambda *args: None
    turn.request_base = "turn-test"
    turn.text = None
    turn.intent = await infer_turn_intent(session, turn, ["audio-reference"])
    gate = resolve_knowledge_gate(session, turn)
    assert gate.include_context is (raw != "false")
    assert len(requests) == 1
    assert requests[0].messages[0].content == KNOWLEDGE_SYSTEM
    assert requests[0].messages[1].content[-1] == {"type": "audio"}
    assert requests[0].metadata["audios"] == ["audio-reference"]
    assert "FULL_NEWS" not in str(requests[0].messages)


async def started_session(monkeypatch, enabled=True):
    from tests.unit_test.qwen3_omni.test_multimodal_session import (
        FakeClient, FakeWebSocket, make_session, protocol_v1_session_start,
    )
    from sglang_omni.serve.realtime.knowledge.models import KnowledgeContext, KnowledgeEvidence
    monkeypatch.setenv(USER_KNOWLEDGE_GATE_ENV, "1" if enabled else "0")
    session = make_session(FakeWebSocket(), FakeClient())
    await session.dispatch(protocol_v1_session_start("gate-test"))
    session.knowledge_binding = SimpleNamespace(mode="provided_context")
    session.provided_entity_snapshot = SimpleNamespace(snapshot_id="s1", current_entity_text="{}")
    session.provided_entity_context = KnowledgeContext(
        decision="RETRIEVE", reason="test", result_id="stable-result", state_token="",
        snapshot_id="s1", evidence=(KnowledgeEvidence(
            evidence_id="e1", source_type="provided_entity_snapshot", source_id="unit-1",
            title="新闻标题", content="FULL_NEWS_BODY"),),
    )
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize("need", [False, True, None])
async def test_actual_reply_request_filters_before_media_and_preserves_state(monkeypatch, need):
    from tests.unit_test.qwen3_omni.test_multimodal_session import user_turn_start
    session = await started_session(monkeypatch)
    await session.handle_turn_start(user_turn_start("turn-build"))
    turn = session.active_turn
    turn.text = "去健身吧"
    turn.intent = parsed_intent(needs_knowledge=need)
    turn.reply_context = podcast_text()
    turn.knowledge_context = session.provided_entity_context
    turn.knowledge_gate = resolve_knowledge_gate(session, turn)
    snapshot = session.provided_entity_context
    request, _ = session._build_reply_request(turn, ["audio-test"], [], [], None)
    serialized = str(request.messages)
    assert ("FULL_NEWS_BODY" in serialized) is (need is not False)
    assert ("CURRENT_NEWS" in serialized) is (need is not False)
    assert "Default response language: English." in serialized
    assert "去健身吧" in serialized
    assert session.provided_entity_context is snapshot
    assert "FULL_NEWS_BODY" not in request.messages[0].content
    assert request.metadata["audios"] == ["audio-test"]
    if need is not False:
        assert request.messages[1].role == "user"
        assert "FULL_NEWS_BODY" in str(request.messages[1].content)
        assert "type='audio'" not in str(request.messages[1].content)
        assert request.messages[-1].content != request.messages[1].content
    else:
        assert request.metadata["knowledge_context_chars"] == 0
    await session.handle_session_close({"type": "session.close"})


@pytest.mark.asyncio
async def test_same_news_prefix_survives_dynamic_question_changes(monkeypatch):
    from tests.unit_test.qwen3_omni.test_multimodal_session import user_turn_start
    session = await started_session(monkeypatch)
    await session.handle_turn_start(user_turn_start("turn-prefix"))
    turn = session.active_turn
    turn.intent = parsed_intent(needs_knowledge=True)
    turn.knowledge_context = session.provided_entity_context
    turn.text = "解释一下新闻"
    first, _ = session._build_reply_request(turn, [], [], [], None)
    turn.text = "这条新闻有什么影响"
    second, _ = session._build_reply_request(
        turn, ["different-audio"], ["different-camera"], ["user_camera"], None,
    )
    assert first.messages[:2] == second.messages[:2]
    assert first.messages[-1] != second.messages[-1]
    assert second.metadata["images"] == ["different-camera"]
    # Switching the flag off restores the original request layout.
    session.user_knowledge_gate_enabled = False
    legacy, _ = session._build_reply_request(turn, [], [], [], None)
    assert len(legacy.messages) == 2
    assert "FULL_NEWS_BODY" in str(legacy.messages[-1].content)
    await session.handle_session_close({"type": "session.close"})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_enabled_gate_timeout_includes_and_cancellation_is_not_swallowed(monkeypatch, cancel):
    import asyncio
    import sglang_omni.serve.realtime.turn_intent as intent_module
    session, turn = state()
    entered = asyncio.Event()
    aborted, removed, requests = [], [], []
    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            entered.set()
            await asyncio.Event().wait()
        async def abort(self, request_id):
            aborted.append(request_id)
    session.client = Client()
    session.model_name = "fake-model"
    session.session_id = "test-session"
    session._register_turn_request = lambda *args: None
    session._unregister_turn_request = lambda t, r: removed.append(r)
    turn.text = "去健身吧"
    turn.request_base = "timeout-turn"
    monkeypatch.setattr(intent_module, "TURN_INTENT_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(infer_turn_intent(session, turn, []))
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        turn.intent = await task
        gate = resolve_knowledge_gate(session, turn)
        assert gate.include_context and gate.reason == "intent_fallback"
    assert aborted == removed == ["timeout-turn-intent"]
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("need", [False, True, None])
async def test_committed_turn_gate_precedes_model_submission(monkeypatch, need):
    import sglang_omni.serve.realtime.turn_pipeline as pipeline
    from tests.unit_test.qwen3_omni.test_multimodal_session import user_turn_start
    session = await started_session(monkeypatch)
    async def classify(*args):
        return None if need is None else parsed_intent(needs_knowledge=need)
    monkeypatch.setattr(pipeline, "infer_turn_intent", classify)
    await session.handle_turn_start(user_turn_start("turn-commit"))
    turn = session.active_turn
    await session.dispatch({"type": "input.text.set", "turn_id": turn.turn_id, "text": "去健身吧"})
    await session.dispatch({"type": "turn.commit", "turn_id": turn.turn_id,
                            "reply": {"context": podcast_text()}})
    await turn.inference_task
    requests = session.client.chat_requests
    assert requests, session.websocket.events
    submitted = requests[-1]
    assert ("FULL_NEWS_BODY" in str(submitted.messages)) is (need is not False)
    assert ("CURRENT_NEWS" in str(submitted.messages)) is (need is not False)
    assert turn.knowledge_gate.include_context is (need is not False)
    assert session.provided_entity_context.evidence[0].content == "FULL_NEWS_BODY"
    # History records the conversation, not the assembled reference prompt.
    assert "FULL_NEWS_BODY" not in str(session.reply_history_turns)
    await session.handle_session_close({"type": "session.close"})
