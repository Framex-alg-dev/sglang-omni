from dataclasses import replace

import pytest

from sglang_omni.models.qwen3_omni.global_action_catalog import (
    load_runtime_action_catalog,
    prewarm_global_action_catalog,
)
from sglang_omni.serve.realtime.multimodal import MultimodalSession
from tests.unit_test.qwen3_omni.test_global_action_catalog import _ScoreClient, _WebSocket


class Client(_ScoreClient):
    def __init__(self, selected="154"):
        super().__init__()
        self.selected = selected
        self.prefills = []

    async def completion(self, request, *, request_id):
        from sglang_omni.client.types import CompletionResult
        return CompletionResult(request_id=request_id, text="好的。")

    async def prefill_action_catalog(self, **kwargs):
        self.prefills.append(kwargs)
        return True

    async def score_action_suffixes(self, request):
        result = await super().score_action_suffixes(request)
        return replace(result, scores=[replace(score, mean_logprob=0 if score.candidate_id == self.selected else -10)
                                       for score in result.scores])


def make_session(monkeypatch, selected="154"):
    monkeypatch.delenv("SGLANG_OMNI_ACTION_CATALOG_PATH", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_ACTION_CATALOG_MODE", "limited")
    client, ws = Client(selected), _WebSocket()
    session = MultimodalSession(ws, client=client, model_name="test",
                               global_action_catalog=load_runtime_action_catalog(),
                               claim_session=lambda *_: True, release_session=lambda *_: None)
    return session, client, ws


@pytest.mark.asyncio
async def test_limited_startup_only_warms_actions(monkeypatch):
    session, client, _ = make_session(monkeypatch)
    await prewarm_global_action_catalog(client, model="test", catalog=session.global_action_catalog)
    assert len(client.prefills) == 4
    assert {call["stage"] for call in client.prefills} == {"single"}
    for call in client.prefills:
        ids = [c.candidate_id for c in call["candidates"]]
        assert len(ids) == len(set(ids)) == 169
        assert "154" in ids and "000" in ids
        assert "已选类别" not in call["system_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ["154", "000"])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("origin", ["user", "proactive"])
async def test_limited_turn_scores_all_actions_once_and_hits_session_prefix(monkeypatch, selected, fusion, origin):
    session, client, ws = make_session(monkeypatch, selected)
    payload = {"type": "session.start", "protocol_version": 1, "session_id": "limited",
               "outputs": ["action"], "locale": "zh-CN",
               "action": {"fallback_category_ids": ["02"],
                          "allowed_candidates": [{"candidate_id": "154", "execution_binding": {"asset_id": "smile"}},
                                                 {"candidate_id": "9999"}]}}
    if fusion:
        payload["outputs"] = ["text", "action"]
        payload["reply"] = {"unsupported_action_text": "暂不支持这个动作。"}
    from sglang_omni.serve.realtime.protocol.common import REALTIME_PROTOCOL_VERSION
    payload["protocol_version"] = REALTIME_PROTOCOL_VERSION
    await session.handle_session_start(session._normalize_session_start(payload))
    assert len(session.candidates) == 168
    assert session.action_selection_mode == "flat_children"
    assert len(client.prefills) == 3
    user_prefill, camera_prefill, proactive_prefill = client.prefills
    assert user_prefill["request_id"].endswith("-single-prefill-user")
    assert camera_prefill["request_id"].endswith("-single-prefill-user-camera")
    assert proactive_prefill["request_id"].endswith("-single-prefill-proactive")
    assert (
        user_prefill["prefix_cache_namespace"]
        != camera_prefill["prefix_cache_namespace"]
    )
    assert (
        "user_camera"
        not in user_prefill["session_instruction"]
    )
    assert (
        "user_camera"
        in camera_prefill["session_instruction"]
    )
    semantics = {"turn_origin": origin, "text_role": "user_input" if origin == "user" else "character_reply"}
    if origin == "proactive":
        semantics["trigger"] = "action_finished"
    await session.handle_turn_start({"type": "turn.start", "turn_id": "one", **semantics})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "one", "text": "微笑", **semantics})
    prefill = user_prefill if origin == "user" else proactive_prefill
    action_requests = [r for r in client.requests if r.stage in {"single", "category", "child"}]
    assert len(action_requests) == 1
    request = action_requests[0]
    assert request.stage == "single"
    assert len(request.candidates) == 169
    assert request.system_prompt == prefill["system_prompt"]
    assert request.prefix_cache_namespace == prefill["prefix_cache_namespace"]
    assert request.session_instance_id == prefill["session_instance_id"] == session.session_instance_id
    ready = next(event for event in ws.events if event["type"] == "turn.action.ready")
    assert ready["action"]["execute"] is (selected != "000")
    assert ready["action"]["candidate_id"] == ("UNSUPPORTED" if selected == "000" else "154")
    if selected != "000":
        assert ready["action"]["execution_binding"] == {"asset_id": "smile"}


@pytest.mark.asyncio
async def test_limited_prefill_failure_is_a_cache_miss(monkeypatch):
    session, client, _ = make_session(monkeypatch)

    async def fail_prefill(**kwargs):
        raise RuntimeError("cache unavailable")

    client.prefill_action_catalog = fail_prefill
    status = await prewarm_global_action_catalog(
        client, model="test", catalog=session.global_action_catalog,
    )
    assert len(status.action_prefix_statuses) == 4
    assert not any(status.action_prefix_statuses.values())
    from sglang_omni.serve.realtime.protocol.common import REALTIME_PROTOCOL_VERSION
    await session.handle_session_start(session._normalize_session_start({
        "type": "session.start", "protocol_version": REALTIME_PROTOCOL_VERSION,
        "session_id": "cache-miss", "outputs": ["action"], "action": {},
    }))
    assert session.started
    assert not session.action_prefix_prefilled
    semantics = {"turn_origin": "user", "text_role": "user_input"}
    await session.handle_turn_start({"type": "turn.start", "turn_id": "one", **semantics})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "one", "text": "微笑", **semantics})
    assert len(client.requests) == 1
    assert client.requests[0].stage == "single"


def test_hierarchical_mode_remains_explicitly_available(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_ACTION_CATALOG_PATH", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_ACTION_CATALOG_MODE", "hierarchical")
    catalog = load_runtime_action_catalog()
    assert not catalog.direct_action_selection
    assert len(catalog.categories) > 11


@pytest.mark.asyncio
async def test_limited_turn_constraints_intersect_legacy_client_catalog(monkeypatch):
    session, _, _ = make_session(monkeypatch)
    from sglang_omni.serve.realtime.protocol.common import REALTIME_PROTOCOL_VERSION

    await session.handle_session_start(session._normalize_session_start({
        "type": "session.start",
        "protocol_version": REALTIME_PROTOCOL_VERSION,
        "session_id": "limited-legacy-constraints",
        "outputs": ["action"],
        "action": {
            "allowed_candidates": [
                {"candidate_id": "102"},
                {"candidate_id": "154"},
            ],
        },
    }))
    assert "102" not in session.candidate_by_id
    assert "154" in session.candidate_by_id
    await session.handle_turn_start({
        "type": "turn.start",
        "turn_id": "proactive",
        "turn_origin": "proactive",
        "text_role": "character_reply",
        "trigger": "session_enter",
    })

    normalized = session._normalize_wire_event({
        "type": "turn.commit",
        "turn_id": "proactive",
        "action": {
            "allowed_candidate_ids": ["102", "154"],
        },
    })

    assert normalized["action_allowed_candidate_ids"] == ("154",)

    with pytest.raises(
        ValueError,
        match="contains no candidates in the limited session catalog",
    ):
        session._normalize_wire_event({
            "type": "turn.commit",
            "turn_id": "proactive",
            "action": {"allowed_candidate_ids": ["102"]},
        })
