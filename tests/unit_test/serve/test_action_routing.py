import pytest

from sglang_omni.serve.realtime.action.routing import (
    choose_category_width,
    complete_reply_may_contain_numeric_answer,
    is_visual_deictic_expression_request,
    numeric_gesture_candidates,
    resolve_unique_explicit_action,
    resolve_unique_source_label_action,
    route_numeric_reply_action,
    scope_visual_deictic_categories,
    visual_deictic_scope_candidates,
)
from sglang_omni.serve.realtime.action.visual_generation import (
    build_visual_gesture_system_prompt,
    normalize_visual_gesture_output,
    parse_visual_gesture_output,
    visual_gesture_candidates,
    visual_gesture_number,
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


def test_explicit_route_accepts_versioned_catalog_aliases() -> None:
    candidate = _candidate("288", "单手挥手")
    category = _category(candidate)

    route = resolve_unique_explicit_action(
        "挥挥手跟我打个招呼",
        [(category, candidate)],
        aliases_by_candidate_id={
            "288": ("挥挥手跟我打个招呼",),
        },
    )

    assert route is not None
    assert route.candidate.candidate_id == "288"
    assert route.matched_alias == "挥挥手跟我打个招呼"


def test_source_label_route_excludes_ids_aliases_and_substrings() -> None:
    one = _candidate("258", "数字一手势")
    two = _candidate("259", "数字二手势")
    category = _category(one, two)

    route = resolve_unique_source_label_action(
        "数字二手势。", [(category, one), (category, two)]
    )

    assert route is not None
    assert route.candidate.candidate_id == "259"
    assert resolve_unique_source_label_action("259", [(category, two)]) is None
    assert (
        resolve_unique_source_label_action(
            "请做数字二手势", [(category, two)]
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
    generic_action = scope_visual_deictic_categories(
        categories,
        body_task="请做出这个动作",
        body_mode="perform",
        has_user_camera=True,
    )
    assert generic_action is not None
    assert generic_action.name == "gesture"
    assert generic_action.categories == (hand,)
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
    assert scope_visual_deictic_categories(
        categories,
        body_task="数字一手势",
        body_mode="perform",
        has_user_camera=True,
    ) is None


def test_copy_hand_scope_keeps_only_reviewed_hand_signs() -> None:
    symbolic = SessionActionCategory(
        category_id="31",
        source_label="符号化手势",
        short_definition="数字和约定手型",
        category_path=("手部与手势动作", "符号化手势"),
        children=(
            _candidate("258", "数字一手势"),
            _candidate("285", "双手比心"),
        ),
    )
    pointing = SessionActionCategory(
        category_id="29",
        source_label="指向类",
        short_definition="指向动作",
        category_path=("手部与手势动作", "指向类"),
        children=(_candidate("225", "指向前方"),),
    )
    greeting = SessionActionCategory(
        category_id="32",
        source_label="打招呼与告别",
        short_definition="问候动作",
        category_path=("手部与手势动作", "打招呼与告别"),
        children=(_candidate("288", "单手挥手"),),
    )

    hand_scope = scope_visual_deictic_categories(
        (symbolic, pointing, greeting),
        body_task="做这个手势",
        body_mode="perform",
        has_user_camera=True,
    )
    action_scope = scope_visual_deictic_categories(
        (symbolic, pointing, greeting),
        body_task="做这个动作",
        body_mode="perform",
        has_user_camera=True,
    )

    assert hand_scope is not None
    assert action_scope is not None
    assert {
        candidate.candidate_id
        for candidate in visual_deictic_scope_candidates(hand_scope)
    } == {"258", "285", "288"}
    assert {
        candidate.candidate_id
        for candidate in visual_deictic_scope_candidates(action_scope)
    } == {"258", "285", "288"}


def test_visual_gesture_generation_uses_catalog_semantic_labels() -> None:
    digit_four = _candidate("catalog-four", "数字四手势")
    digit_five = _candidate("catalog-five", "数字五手势")
    symbolic = SessionActionCategory(
        category_id="31",
        source_label="符号化手势",
        short_definition="数字和约定手型",
        category_path=("手部与手势动作", "符号化手势"),
        children=(digit_four, digit_five),
    )

    eligible = visual_gesture_candidates(
        (symbolic,), (digit_four, digit_five)
    )
    prompt = build_visual_gesture_system_prompt(eligible)

    assert "- 数字四:" in prompt and "- 数字五:" in prompt
    assert "四指伸直且拇指内扣" in prompt
    assert "封闭集视觉分类" in prompt
    assert "逐字等于" in prompt
    assert "标签结束后立即停止" in prompt
    assert "动态动作依据多张画面的连续变化" in prompt
    assert "不要添加任何其他字符" in prompt
    selected = parse_visual_gesture_output("数字五", eligible)
    assert selected is digit_five
    assert parse_visual_gesture_output("数字五手势", eligible) is digit_five
    assert parse_visual_gesture_output("数字五。", eligible) is digit_five
    assert parse_visual_gesture_output("数字五手势！", eligible) is digit_five
    assert parse_visual_gesture_output("目录外动作", eligible) is None
    assert parse_visual_gesture_output("UNSUPPORTED", eligible) is None
    assert normalize_visual_gesture_output("UNSUPPORTED。", eligible) == "UNSUPPORTED"
    assert normalize_visual_gesture_output("答案是数字五", eligible) is None
    assert normalize_visual_gesture_output("数字五、数字四", eligible) is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [("数字零", 0), ("数字一手势", 1), ("数字十", 10), ("双手比心", None)],
)
def test_visual_gesture_number_supports_the_full_zero_to_ten_range(
    label: str,
    expected: int | None,
) -> None:
    assert visual_gesture_number(label) == expected


def test_visual_gesture_generation_drops_ambiguous_catalog_labels() -> None:
    first = _candidate("one", "数字一手势")
    second = _candidate("another-one", "数字一")
    symbolic = SessionActionCategory(
        category_id="31",
        source_label="符号化手势",
        short_definition="数字和约定手型",
        category_path=("手部与手势动作", "符号化手势"),
        children=(first, second),
    )

    assert visual_gesture_candidates((symbolic,), (first, second)) == ()


def test_visual_gesture_generation_collapses_v_shape_but_keeps_heart() -> None:
    digit_two = _candidate("259", "数字二手势")
    victory = _candidate("274", "单手比耶")
    heart = _candidate("285", "双手比心")
    symbolic = SessionActionCategory(
        category_id="31",
        source_label="符号化手势",
        short_definition="数字和约定手型",
        category_path=("手部与手势动作", "符号化手势"),
        children=(digit_two, victory, heart),
    )

    eligible = visual_gesture_candidates(
        (symbolic,), (digit_two, victory, heart)
    )

    assert [candidate.candidate_id for candidate in eligible] == ["259", "285"]
    assert parse_visual_gesture_output("数字二", eligible) is digit_two
    assert normalize_visual_gesture_output("单手比耶", eligible) == "数字二"
    assert parse_visual_gesture_output("单手比耶", eligible) is digit_two
    assert parse_visual_gesture_output("双手比心", eligible) is heart

    victory_only = visual_gesture_candidates((symbolic,), (victory,))
    assert normalize_visual_gesture_output("数字二", victory_only) == "单手比耶"
    assert parse_visual_gesture_output("数字二", victory_only) is victory


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

    visual_answer_without_gesture = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="generated",
        body_mode="none",
        has_user_camera=True,
        has_text_output=True,
        has_action_output=True,
        candidates=[_candidate("other", "掰手指数数")],
        allow_empty_candidates=True,
    )
    assert visual_answer_without_gesture.enabled is True
    assert visual_answer_without_gesture.candidates == ()
    assert visual_answer_without_gesture.reason == "numeric_candidates_missing"

    visual_answer_with_additional_verbatim = route_numeric_reply_action(
        turn_origin="user",
        reply_provided=False,
        speech_kind="verbatim",
        body_mode="none",
        has_user_camera=True,
        has_text_output=True,
        has_action_output=True,
        candidates=candidates,
        allow_empty_candidates=True,
    )
    assert visual_answer_with_additional_verbatim.enabled is True
    assert visual_answer_with_additional_verbatim.candidates[0].value == 3


def test_complete_reply_numeric_gate_is_broad_but_bounded() -> None:
    for text in ("等于三", "1+2=3", "一共有 3 个", "the answer is ten"):
        assert complete_reply_may_contain_numeric_answer(text)
    for text in ("没有明确的答案", "共有 11 个", "结果为 3.5"):
        assert not complete_reply_may_contain_numeric_answer(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("加起来是数字三", 3),
        ("1+2=3", 3),
        ("The answer is three.", 3),
        ("3", 3),
        ("今天是 2026 年 9 月 16 日。", None),
        ("结果是 3%。", None),
        ("结果是 3.5。", None),
        ("结果是 -3。", None),
        ("答案是 11。", None),
        ("答案可能是三或者四。", None),
    ],
)
def test_explicit_primary_answer_number_is_strict(text: str, expected: int | None) -> None:
    from sglang_omni.serve.realtime.action.numeric_reply import (
        _explicit_primary_answer_number,
    )

    assert _explicit_primary_answer_number(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2,2", (2, 2)),
        (" 10, 0 ", (10, 0)),
        ("11,2", None),
        ("答案：1,2", None),
        ("1,2。", None),
        ("1,2 / 2,3", None),
        ("INVALID", None),
    ],
)
def test_explicit_visual_arithmetic_operands_is_strict(
    text: str,
    expected: tuple[int, int] | None,
) -> None:
    from sglang_omni.serve.realtime.action.numeric_reply import (
        _explicit_visual_arithmetic_operands,
    )

    assert _explicit_visual_arithmetic_operands(text) == expected


@pytest.mark.parametrize(
    ("operation", "operands", "expected"),
    [
        ("add", (7, 2), 9),
        ("subtract", (7, 2), 5),
        ("multiply", (3, 2), 6),
        ("divide", (8, 2), 4),
        ("divide", (7, 2), None),
        ("divide", (7, 0), None),
        ("", (1, 2), None),
        ("add", None, None),
    ],
)
def test_visual_arithmetic_uses_explicit_intent_operation(
    operation: str,
    operands: tuple[int, int] | None,
    expected: int | None,
) -> None:
    from sglang_omni.serve.realtime.action.numeric_reply import (
        _calculate_visual_arithmetic,
    )

    assert _calculate_visual_arithmetic(operation, operands) == expected
