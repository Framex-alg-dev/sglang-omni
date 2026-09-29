import pytest

from tests.unit_test.serve.test_limited_action_selection import make_session
from sglang_omni.serve.realtime.protocol.common import REALTIME_PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_explicit_role_subset_survives_start_and_prefill(monkeypatch):
    session, client, ws = make_session(monkeypatch, selected="130")
    payload = {
        "type": "session.start", "protocol_version": REALTIME_PROTOCOL_VERSION,
        "session_id": "limited-role", "outputs": ["action"], "locale": "zh-CN",
        "action": {"allowed_candidates": [
            {"candidate_id": "130", "execution_binding": {"asset_id": "breath"}},
            {"candidate_id": "138"}, {"candidate_id": "154"},
        ]},
    }
    await session.handle_session_start(session._normalize_session_start(payload))
    assert {c.candidate_id for c in session.candidates} == {"130", "138"}
    assert session.action_selection_mode == "flat_children"
    assert client.prefills
    for call in client.prefills:
        assert ({c.candidate_id for c in call["candidates"]}
                & set(session.global_action_catalog.candidate_by_id)) <= {"130", "138"}
    semantics = {"turn_origin": "user", "text_role": "user_input"}
    await session.handle_turn_start({"type": "turn.start", "turn_id": "one", **semantics})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "one", "text": "自然呼吸", **semantics})
    requests = [r for r in client.requests if r.stage == "single"]
    assert requests
    assert ({c.candidate_id for c in requests[-1].candidates}
            & set(session.global_action_catalog.candidate_by_id)) <= {"130", "138"}
    ready = next(e for e in ws.events if e["type"] == "turn.action.ready")
    assert ready["action"]["candidate_id"] == "130"
    assert ready["action"]["execution_binding"] == {"asset_id": "breath"}


@pytest.mark.parametrize("allowed", [[], [{"candidate_id": "154"}]])
def test_empty_role_intersection_is_rejected(monkeypatch, allowed):
    session, _, _ = make_session(monkeypatch)
    with pytest.raises(ValueError, match="no actions in the limited catalog"):
        session._compact_action_catalog({"allowed_candidates": allowed})


def test_omitted_capabilities_keep_default_catalog(monkeypatch):
    session, _, _ = make_session(monkeypatch)
    categories, _, _ = session._compact_action_catalog({})
    assert {c["candidate_id"] for category in categories for c in category["children"]} == set(session.global_action_catalog.candidate_by_id)
