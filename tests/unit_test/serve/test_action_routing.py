from sglang_omni.serve.realtime.action.routing import (
    choose_category_width,
    complete_reply_may_contain_numeric_answer,
    is_visual_deictic_expression_request,
    numeric_gesture_candidates,
    resolve_unique_explicit_action,
    route_numeric_reply_action,
    scope_visual_deictic_categories,
)
from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
)


def _candidate(candidate_id: str, label: str) -> SessionActionCandidate:
    return SessionActionCandidate(
        candidate_id=candidate_id,
        action_id=candidate_id,
        source_label=label,
        short_definition=label,
        execution_binding={},
        category_id="01",
    )


def _category(*children: SessionActionCandidate) -> SessionActionCategory:
    return SessionActionCategory(
        category_id="01",
        source_label="测试类别",
        short_definition="测试类别",
        category_path=("测试",),
        children=children,
    )


def test_explicit_route_uses_catalog_alias_and_ignores_parenthetical_qualifier() -> None:
    candidate = _candidate("057", "整理衣领（轻轻拉正）")
    category = _category(candidate)

    route = resolve_unique_explicit_action("整理衣领。", [(category, candidate)])

    assert route is not None
    assert route.candidate.candidate_id == "057"
    assert route.matched_alias == "整理衣领"


def test_explicit_route_requires_equality_and_one_unique_candidate() -> None:
    first = _candidate("001", "挥手")
    second = _candidate("002", "挥手")
    category = _category(first, second)

    assert resolve_unique_explicit_action("请挥手", [(category, first)]) is None
    assert (
        resolve_unique_explicit_action(
            "挥手",
            [(category, first), (category, second)],
        )
        is None
    )


def test_category_width_uses_top1_only_for_confident_prewarmed_winner() -> None:
    decision = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=[
            ("57", -0.1, 1.1),
            ("37", -1.2, 3.3),
        ],
        overall_top_candidate_id="57",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )

    assert decision.effective_top_k == 1
    assert decision.adaptive_top1 is True
    assert decision.reason == "high_confidence_prewarmed_top1"


def test_category_width_keeps_recall_for_close_or_unsupported_first_score() -> None:
    scores = [("57", -0.1, 1.1), ("37", -0.2, 1.2)]
    close = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=scores,
        overall_top_candidate_id="57",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )
    unsupported_first = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=scores,
        overall_top_candidate_id="00",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )

    assert close.effective_top_k == 2
    assert close.reason == "category_margin_too_small"
    assert unsupported_first.effective_top_k == 2
    assert unsupported_first.reason == "unsupported_ranked_first"


def test_visual_deictic_action_requires_camera_and_named_catalog_range() -> None:
    hand = SessionActionCategory(
        category_id="31",
        source_label="符号化手势",
        short_definition="数字和约定手型",
        category_path=("手部与手势动作", "符号化手势"),
        children=(_candidate("258", "数字一手势"),),
    )
    head = SessionActionCategory(
        category_id="15",
        source_label="俯仰类",
        short_definition="头部上下朝向",
        category_path=("头部与视线动作", "俯仰类"),
        children=(_candidate("136", "抬头"),),
    )
    categories = [hand, head]

    gesture = scope_visual_deictic_categories(
        categories,
        body_task="请做出手势",
        body_mode="perform",
        has_user_camera=True,
    )
    head_action = scope_visual_deictic_categories(
        categories,
        body_task="模仿这个头部动作",
        body_mode="perform",
        has_user_camera=True,
    )

    assert gesture is not None
    assert gesture.name == "gesture"
    assert gesture.categories == (hand,)
    assert head_action is not None
    assert head_action.name == "head_gaze"
    assert head_action.categories == (head,)
    assert scope_visual_deictic_categories(
        categories,
        body_task="请做出这个动作",
        body_mode="perform",
        has_user_camera=True,
    ) is None
    assert scope_visual_deictic_categories(
        categories,
        body_task="这个手势",
        body_mode="perform",
        has_user_camera=False,
    ) is None
    assert scope_visual_deictic_categories(
        categories,
        body_task="这个动作叫什么",
        body_mode="none",
        has_user_camera=True,
    ) is None
    assert scope_visual_deictic_categories(
        categories,
        body_task="挥手",
        body_mode="perform",
        has_user_camera=True,
    ) is None


def test_visual_expression_requires_named_scope_and_current_camera() -> None:
    assert is_visual_deictic_expression_request(
        face_task="请做出表情",
        has_user_camera=True,
    )
    assert not is_visual_deictic_expression_request(
        face_task="微笑",
        has_user_camera=True,
    )
    assert not is_visual_deictic_expression_request(
        face_task="这个表情",
        has_user_camera=False,
    )


def test_numeric_gestures_derive_values_from_exact_labels_not_candidate_ids() -> None:
    candidates = [
        _candidate("arbitrary-three", "数字三手势"),
        _candidate("arbitrary-ten", "数字十手势"),
        _candidate("258", "数字一手势（旧标签）"),
        _candidate("other", "掰手指数数"),
    ]

    resolved = numeric_gesture_candidates(candidates)

    assert [(item.value, item.candidate.candidate_id) for item in resolved] == [
        (3, "arbitrary-three"),
        (10, "arbitrary-ten"),
    ]


def test_numeric_gesture_duplicate_category_membership_is_deduplicated() -> None:
    three = _candidate("260", "数字三手势")

    resolved = numeric_gesture_candidates([three, three])

    assert len(resolved) == 1
    assert resolved[0].value == 3


def test_numeric_reply_route_requires_generated_body_neutral_user_reply() -> None:
    candidates = [_candidate("260", "数字三手势")]

    enabled = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="generated",
        body_mode="none",
        has_user_camera=False,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
    )
    explicit = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="generated",
        body_mode="perform",
        has_user_camera=False,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
    )
    verbatim = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="verbatim",
        body_mode="none",
        has_user_camera=False,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
    )

    assert enabled.enabled is True
    assert enabled.candidates[0].value == 3
    assert explicit.reason == "explicit_body_directive"
    assert verbatim.reason == "not_generated_reply"

    camera_parse_fallback = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="none",
        body_mode="none",
        has_user_camera=True,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
    )
    no_camera_parse_fallback = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="none",
        body_mode="none",
        has_user_camera=False,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
    )
    assert camera_parse_fallback.enabled is True
    assert no_camera_parse_fallback.reason == "not_generated_reply"

    missing = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="generated",
        body_mode="none",
        has_user_camera=False,
        has_text_output=True,
        has_action_output=True,
        candidates=[_candidate("other", "掰手指数数")],
    )
    assert missing.reason == "numeric_candidates_missing"


def test_complete_reply_numeric_gate_is_broad_but_bounded() -> None:
    for text in ("等于三", "1+2=3", "一共有 3 个", "the answer is ten"):
        assert complete_reply_may_contain_numeric_answer(text)
    for text in ("没有明确的答案", "共有 11 个", "结果为 3.5"):
        assert not complete_reply_may_contain_numeric_answer(text)
