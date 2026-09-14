from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate, ActionSuffixScoreRequest
from sglang_omni.serve.realtime.action.cache_observation import ChildCacheObserver


def request(**kwargs):
    return ActionSuffixScoreRequest(
        request_id=kwargs.pop("request_id", "request-1"), model="test",
        prefix="current instruction", system_prompt="fixed catalog",
        candidates=[ActionScoreCandidate(candidate_id="A001", suffix="A001")],
        prefix_cache_namespace=kwargs.pop("prefix_cache_namespace", "child:test"),
        stage="child", language="zh", audios=[], images=[], sample_rate=16000, **kwargs,
    )


def test_repeated_identity_does_not_claim_kv_hit():
    observer = ChildCacheObserver()
    req = request()
    first = observer.submitted(req, ["B002", "B001"], "zh-CN")
    assert first["selected_category_ids"] == ["B002", "B001"]
    assert not first["identity_seen_before"]
    result = observer.finished(first["prefix_identity"], req.request_id, "completed", {
        "parent_radix_cached_token_count": 0, "parent_computed_token_count": 200,
    })
    assert result["parent_computed_token_count"] == 200
    assert result["parent_cache_hit_ratio"] is None
    second = observer.submitted(request(request_id="request-2"), ["B002", "B001"], "zh-CN")
    assert second["identity_seen_before"]
    assert second["seen_count_before"] == 1
    assert second["previous_status"] == "completed"
    assert "parent_cache_hit_ratio" not in second


def test_namespace_instruction_and_candidate_order_identities():
    observer = ChildCacheObserver()
    req = request()
    first = observer.submitted(req, ["B002", "B001"], "zh-CN")
    for other in [request(prefix_cache_namespace="other"), request(session_instruction="different")]:
        assert observer.submitted(other, ["B002", "B001"], "zh-CN")["prefix_identity"] != first["prefix_identity"]
    reordered = observer.submitted(req, ["B001", "B002"], "zh-CN")
    assert reordered["prefix_identity"] == first["prefix_identity"]
    assert reordered["scoring_identity"] != first["scoring_identity"]


def test_window_eviction_and_out_of_order_completion():
    observer = ChildCacheObserver(capacity=1)
    req = request()
    first = observer.submitted(req, [], "zh-CN")
    newer = request(request_id="request-2")
    observer.submitted(newer, [], "zh-CN")
    observer.finished(first["prefix_identity"], newer.request_id, "cancelled", None)
    observer.finished(first["prefix_identity"], req.request_id, "completed", {})
    assert observer.submitted(request(request_id="request-3"), [], "zh-CN")["previous_status"] == "cancelled"
    observer.submitted(request(prefix_cache_namespace="other"), [], "zh-CN")
    assert not observer.submitted(req, [], "zh-CN")["identity_seen_before"]
    assert not ChildCacheObserver().submitted(req, [], "zh-CN")["identity_seen_before"]


def test_expired_observation_is_not_reported_as_seen():
    observer = ChildCacheObserver(ttl_seconds=0)
    observer.submitted(request(), [], "zh-CN")
    assert not observer.submitted(request(), [], "zh-CN")["identity_seen_before"]


def test_render_cache_preserves_order_and_invalidates_changed_descriptions():
    from dataclasses import replace
    from sglang_omni.serve.realtime.action.prompts import ActionPromptComponent
    from sglang_omni.serve.realtime.protocol.models import SessionActionCandidate, SessionActionCategory

    class Prompts(ActionPromptComponent):
        global_action_catalog = None
        language = "en"
        locale = "en-US"

        def _prompt(self, *, zh, en):
            return en

    body = SessionActionCandidate(
        candidate_id="A001", action_id="wave", source_label="wave",
        short_definition="Wave your hand", execution_binding={}, category_id="B001",
    )
    cat = SessionActionCategory(
        category_id="B001", source_label="greeting", short_definition="greeting",
        category_path=("greeting",), children=(body,),
    )
    other = replace(body, candidate_id="A002", short_definition="Nod your head")
    prompt = Prompts()
    initial = prompt._build_child_system_prompt(cat, [body, other])
    assert initial == prompt._render_child_system_prompt(cat, [body, other])
    assert prompt._build_child_system_prompt(cat, [body, other]) == initial
    reversed_prompt = prompt._build_child_system_prompt(cat, [other, body])
    assert reversed_prompt == prompt._render_child_system_prompt(cat, [other, body])
    assert reversed_prompt != initial
    updated = replace(body, short_definition="Wave slowly")
    assert "Wave slowly" in prompt._build_child_system_prompt(cat, [updated, other])
    assert "Wave your hand" in prompt._build_child_system_prompt(cat, [body, other])
