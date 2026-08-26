from __future__ import annotations

import asyncio
import json

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionResult
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    GlobalActionCatalogPrewarmStatus,
    UNSUPPORTED_CATEGORY_SHORT_DEFINITION,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SHORT_DEFINITION,
    UNSUPPORTED_CHILD_SCORE_ID,
    load_global_action_catalog,
    prewarm_global_action_catalog,
)
from sglang_omni.serve.realtime.multimodal import MultimodalSession


def test_action_prompts_do_not_include_cross_turn_action_history() -> None:
    catalog = load_global_action_catalog()
    for locale in ("zh-CN", "en-US"):
        prompts = [catalog.category_system_prompt_for(locale)]
        prompts.extend(
            catalog.child_system_prompt_for(locale, category.category_id)
            for category in catalog.categories
        )
        for prompt in prompts:
            assert "[当前实际动作状态]" not in prompt
            assert "[最近一次用户触发动作]" not in prompt
            assert "[Current physical action state]" not in prompt
            assert "[Most recent user-triggered action]" not in prompt


def test_builtin_category_prompt_separates_system_accompaniment_routes() -> None:
    catalog = load_global_action_catalog()
    chinese = catalog.category_system_prompt_for("zh-CN")
    english = catalog.category_system_prompt_for("en-US")

    assert "若本轮实际产生非空回复，应选择语言表达伴随类别" in chinese
    assert "若本轮无需说话、回复为空或回复生成失败，应选择静默低扰伴随类别" in chinese
    assert "仍应根据实际回复是否为空选择语言表达伴随或静默低扰伴随类别" in chinese
    assert "actual non-empty reply text" in english
    assert "reply-accompaniment or silent low-disturbance accompaniment category" in english


def test_builtin_catalog_exposes_shy_composite_action_semantics() -> None:
    catalog = load_global_action_catalog()
    category = next(
        category for category in catalog.categories if category.source_label == "情绪动作"
    )
    action = next(
        child for child in category.children if child.source_label.startswith("娇羞（")
    )

    assert all(term in category.short_definition for term in ("害羞", "娇羞", "撒娇"))
    assert all(term in action.short_definition for term in ("害羞", "娇羞"))
    for locale in ("zh-CN", "en-US"):
        assert category.short_definition in catalog.category_system_prompt_for(locale)
        assert action.short_definition in catalog.child_system_prompt_for(
            locale, category.category_id
        )


def test_builtin_catalog_exposes_forward_approach_category_semantics() -> None:
    catalog = load_global_action_catalog()
    category = next(
        category
        for category in catalog.categories
        if category.source_label == "躯干前后动作"
    )
    action = next(
        child for child in category.children if child.source_label == "身体前倾"
    )

    assert "靠近镜头" in category.short_definition
    assert "改变身体与观察目标的距离" in category.short_definition
    assert "靠近镜头" in action.short_definition
    assert action.category_id == category.category_id
    for locale in ("zh-CN", "en-US"):
        assert category.short_definition in catalog.category_system_prompt_for(
            locale
        )


def test_builtin_catalog_exposes_self_introduction_and_greeting_semantics() -> None:
    catalog = load_global_action_catalog()
    greeting_category = next(
        category
        for category in catalog.categories
        if "greeting" in category.semantic_tags
    )
    single_hand = next(
        child for child in greeting_category.children if child.source_label == "单手挥手"
    )
    both_hands = next(
        child for child in greeting_category.children if child.source_label == "双手挥手"
    )

    assert single_hand.category_id == greeting_category.category_id
    assert both_hands.category_id == greeting_category.category_id
    assert "自我介绍" in greeting_category.short_definition

    chinese_category = catalog.category_system_prompt_for("zh-CN")
    english_category = catalog.category_system_prompt_for("en-US")
    chinese_child = catalog.child_system_prompt_for(
        "zh-CN", greeting_category.category_id
    )
    english_child = catalog.child_system_prompt_for(
        "en-US", greeting_category.category_id
    )

    assert "介绍自己、说明自身身份" in chinese_category
    assert "介绍产品、知识、地点、第三方人物" in chinese_category
    assert "asks the character to introduce itself" in english_category
    assert (
        "introducing a product, knowledge, a place, a third party"
        in english_category
    )
    assert "优先选择自然、克制、日常的单手问候候选" in chinese_child
    assert (
        "prefer a natural, restrained, everyday one-handed greeting"
        in english_child
    )
    assert single_hand.short_definition in chinese_child
    assert both_hands.short_definition in english_child
    non_greeting_category = next(
        category
        for category in catalog.categories
        if "greeting" not in category.semantic_tags
    )
    assert "单手问候候选" not in catalog.child_system_prompt_for(
        "zh-CN", non_greeting_category.category_id
    )
    assert "one-handed greeting" not in catalog.child_system_prompt_for(
        "en-US", non_greeting_category.category_id
    )


def test_builtin_catalog_exposes_strict_object_and_drinking_boundaries() -> None:
    catalog = load_global_action_catalog()
    touching = next(
        category for category in catalog.categories if category.source_label == "身体触碰"
    )
    touch_ear = next(
        child for child in touching.children if child.source_label == "摸耳朵"
    )
    drinking = next(
        category
        for category in catalog.categories
        if category.source_label == "工具与日用品使用"
    )
    cup = next(child for child in drinking.children if child.candidate_id == "A571")
    bottle = next(child for child in drinking.children if child.candidate_id == "A572")

    assert "耳廓与耳垂" in touch_ear.short_definition
    assert "喝口水、喝点水" in drinking.short_definition
    assert "未指定容器时优先选择" in cup.short_definition
    assert "明确要求瓶装水" in bottle.short_definition
    chinese_child = catalog.child_system_prompt_for("zh-CN", touching.category_id)
    english_child = catalog.child_system_prompt_for("en-US", touching.category_id)
    assert "耳环与耳朵是不同交互物体" in chinese_child
    assert "an earring" in english_child
    assert "不是关键词匹配规则" in chinese_child


def test_builtin_catalog_uses_semantic_tags_instead_of_fixed_category_ids() -> None:
    catalog = load_global_action_catalog()
    lower_body_category_ids = [
        category.category_id
        for category in catalog.categories
        if "lower_body_motion" in category.semantic_tags
    ]

    assert lower_body_category_ids == ["B039", "B040", "B041", "B042", "B043"]
    chinese = catalog.category_system_prompt_for("zh-CN")
    english = catalog.category_system_prompt_for("en-US")
    assert (
        "本目录中要求下肢、位移或全身大幅移动的类别为："
        + "、".join(lower_body_category_ids)
    ) in chinese
    assert (
        "the lower-body movement categories are: "
        + ", ".join(lower_body_category_ids)
    ) in english
    assert "B033-B037" not in chinese


def test_builtin_catalog_exposes_system_accompaniment_semantics() -> None:
    catalog = load_global_action_catalog()
    reply_category = catalog.category_with_semantic_tag(
        CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
    )
    silent_category = catalog.category_with_semantic_tag(
        CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
    )

    assert reply_category is not None
    assert silent_category is not None
    assert reply_category.category_id == "B001"
    assert silent_category.category_id == "B002"
    for locale in ("zh-CN", "en-US"):
        category_prompt = catalog.category_system_prompt_for(locale)
        reply_child_prompt = catalog.child_system_prompt_for(
            locale, reply_category.category_id
        )
        silent_child_prompt = catalog.child_system_prompt_for(
            locale, silent_category.category_id
        )
        ordinary_category = next(
            category
            for category in catalog.categories
            if category.category_id
            not in {reply_category.category_id, silent_category.category_id}
        )
        ordinary_child_prompt = catalog.child_system_prompt_for(
            locale, ordinary_category.category_id
        )

        assert reply_category.category_id in category_prompt
        assert silent_category.category_id in category_prompt
        assert "A000" not in reply_child_prompt
        assert "A000" not in silent_child_prompt
        assert "candidate_id=A000" in ordinary_child_prompt
    assert "本轮数字人实际回复开头" in catalog.child_system_prompt_for(
        "zh-CN", reply_category.category_id
    )
    assert "本轮没有需要说出的回复文本" in catalog.child_system_prompt_for(
        "zh-CN", silent_category.category_id
    )


def test_category_prompt_exposes_conversational_feedback_semantics() -> None:
    catalog = load_global_action_catalog()
    chinese = catalog.category_system_prompt_for("zh-CN")
    english = catalog.category_system_prompt_for("en-US")

    assert "评价、质疑或询问数字人此前的回答、笑话或表达效果" in chinese
    assert "表情、情绪反应或思考类别" in chinese
    assert "普通知识问答、对第三方内容的评价" in chinese
    assert "除上述直接社交事件、对话反馈和自我呈现场景外" in chinese
    assert "evaluates, challenges, or asks about the effect" in english
    assert (
        "facial-expression, emotional-response, or thinking category" in english
    )
    assert (
        "Ordinary knowledge questions, evaluations of third-party content"
        in english
    )
    assert "Except for the direct social events, conversational feedback" in english


def test_direction_reference_uses_character_body_coordinates() -> None:
    catalog = load_global_action_catalog()
    representative_pairs = (
        ("看左侧（仅眼球）", "看右侧（仅眼球）"),
        ("转头看左侧", "转头看右侧"),
        ("向左移动", "向右移动"),
        ("左转身", "右转身"),
        ("从左侧拿起物品", "从右侧拿起物品"),
    )
    direction_category_ids = set()
    for left_label, right_label in representative_pairs:
        category = next(
            category
            for category in catalog.categories
            if {left_label, right_label}
            <= {child.source_label for child in category.children}
        )
        left = next(child for child in category.children if child.source_label == left_label)
        right = next(child for child in category.children if child.source_label == right_label)
        assert left.category_id == right.category_id == category.category_id
        assert "左" in left.source_label
        assert "右" in right.source_label
        direction_category_ids.add(category.category_id)

    chinese_category = catalog.category_system_prompt_for("zh-CN")
    english_category = catalog.category_system_prompt_for("en-US")
    assert "以数字人自身的身体坐标为准" in chinese_category
    assert "数字人自身左侧通常显示在用户画面右侧" in chinese_category
    assert "屏幕左侧" in chinese_category
    assert "digital character's own body coordinates" in english_category
    assert (
        "character's own left usually appears on the right side"
        in english_category
    )
    assert "left side of the screen" in english_category

    for category_id in direction_category_ids:
        assert "以数字人自身的身体坐标为准" in (
            catalog.child_system_prompt_for("zh-CN", category_id)
        )
        assert "digital character's own body coordinates" in (
            catalog.child_system_prompt_for("en-US", category_id)
        )


def _catalog_payload() -> dict:
    return {
        "catalog_version": "test-v1",
        "categories": [
            {
                "category_id": "B008",
                "source_label": "待机动作",
                "short_definition": "自然待机",
                "category_path": ["基础姿态与动作转场", "待机动作"],
                "children": [
                    {
                        "candidate_id": "A008",
                        "action_id": "A008",
                        "source_label": "自然呼吸",
                        "short_definition": "自然待机呼吸",
                    }
                ],
            },
            {
                "category_id": "B001",
                "source_label": "问候",
                "short_definition": "问候动作",
                "category_path": ["社交", "问候"],
                "semantic_tags": ["greeting"],
                "children": [
                    {
                        "candidate_id": "A001",
                        "action_id": "A001",
                        "source_label": "单手挥手",
                        "short_definition": "单手自然挥动",
                    },
                    {
                        "candidate_id": "A002",
                        "action_id": "A002",
                        "source_label": "双手挥手",
                        "short_definition": "",
                    },
                ],
            },
            {
                "category_id": "B002",
                "source_label": "赞同",
                "short_definition": "表达赞同",
                "category_path": ["社交", "反馈"],
                "children": [
                    {
                        "candidate_id": "A003",
                        "action_id": "A003",
                        "source_label": "点赞",
                        "short_definition": "竖起拇指",
                    }
                ],
            },
        ],
    }


def _write_catalog(tmp_path, payload: dict | None = None):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(payload or _catalog_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_global_catalog_is_validated_hashed_and_immutable(tmp_path) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))

    assert catalog.catalog_version == "test-v1"
    assert len(catalog.categories) == 3
    assert catalog.candidate_count == 4
    assert catalog.candidate_by_id["A002"].source_short_definition == ""
    assert catalog.candidate_by_id["A002"].short_definition == "双手挥手"
    assert "candidate_id=A002｜动作=双手挥手｜说明=双手挥手" in (
        catalog.child_system_prompts["B001"]
    )
    assert "先综合用户摄像头画面与用户语音判断用户状态" in (
        catalog.category_system_prompt
    )
    assert "动作请求不必明确描述身体部位、运动方向或执行方式" in (
        catalog.category_system_prompt
    )
    assert "动作请求由语义目标决定，不由命令句形式决定" in (
        catalog.category_system_prompt
    )
    assert "不得仅因其使用问句或建议句形式而将其当作普通对话或单纯能力咨询" in (
        catalog.category_system_prompt
    )
    assert "属于对当前感知事实或能力的信息询问" in (
        catalog.category_system_prompt
    )
    assert "“你能看见我吗”是视觉事实询问" in catalog.category_system_prompt
    assert "“看向我”或“看向镜头”才是观察动作请求" in (
        catalog.category_system_prompt
    )
    assert "Determine an action request from its semantic goal" in (
        catalog.category_system_prompt_for("en-US")
    )
    assert "Do not treat them as ordinary conversation or a mere capability question" in (
        catalog.category_system_prompt_for("en-US")
    )
    assert "a question about a current perceptual fact or capability" in (
        catalog.category_system_prompt_for("en-US")
    )
    assert "'can you see me?' is a visual-fact question" in (
        catalog.category_system_prompt_for("en-US")
    )
    assert "以下示例仅说明语义判断方法，不是关键词匹配规则" in (
        catalog.category_system_prompt
    )
    assert "人设、风格、取景和动作偏好只能在" in (
        catalog.category_system_prompt
    )
    assert (
        "由具体动作选择阶段判断候选是否满足姿态、取景及用户明确指定的交互物体等条件"
        in catalog.category_system_prompt
    )
    assert "直接作用于数字人的社交行为" in catalog.category_system_prompt
    assert "默认选择待机或思考类别" in catalog.category_system_prompt
    assert "“给我打个招呼”要求数字人以可观察行为完成问候" in (
        catalog.category_system_prompt
    )
    assert (
        UNSUPPORTED_CATEGORY_SHORT_DEFINITION
        in catalog.category_system_prompt
    )
    assert "category_id=B000｜决策=不支持的动作类别" in (
        catalog.category_system_prompt
    )
    assert "action_id=UNSUPPORTED" not in catalog.category_system_prompt
    assert "具体动作是否支持由下一阶段判断" in (
        catalog.category_system_prompt
    )
    assert "本次会话允许的候选动作均无法完成明确动作请求时" not in (
        catalog.category_system_prompt
    )
    assert (
        UNSUPPORTED_CHILD_SHORT_DEFINITION
        in catalog.child_system_prompts["B001"]
    )
    assert "candidate_id=A000｜决策=不支持的具体动作" in (
        catalog.child_system_prompts["B001"]
    )
    assert "action_id=UNSUPPORTED" not in catalog.child_system_prompts["B001"]
    assert "执行方式不同、仅表达含义相近的动作" in (
        catalog.child_system_prompts["B001"]
    )
    assert "若姿态、取景或物体等硬性可执行条件" in (
        catalog.child_system_prompts["B001"]
    )
    assert "普通对话且实际有非空回复时" in (
        catalog.category_system_prompt
    )
    assert "本轮没有明确动作目标，且本轮主动场景约束也没有指定具体动作" in (
        catalog.category_system_prompt
    )
    assert "并在该类别内优先小幅、低打扰动作" in (
        catalog.category_system_prompt
    )
    assert "不得覆盖用户明确提出且当前会话支持的动作请求" in (
        catalog.category_system_prompt
    )
    assert "无论本轮由用户触发还是由数字人主动触发" not in (
        catalog.category_system_prompt
    )
    assert "避免选择要求下肢、位移或全身大幅移动的类别" in (
        catalog.category_system_prompt
    )
    assert "B033-B037" not in catalog.category_system_prompt
    assert "物体是否出现在“数字人当前状态画面”中，不作为" in (
        catalog.category_system_prompt
    )
    assert "选择要求与具体物体交互的 B043-B052 前" not in (
        catalog.category_system_prompt
    )
    assert "先综合用户摄像头画面与用户语音判断用户状态" not in (
        catalog.child_system_prompts["B001"]
    )
    for internal_term in (
        "Session",
        "当前 turn",
        "Child 阶段",
        "avatar_state",
        "user_camera",
        "PPL",
        "硬过滤",
    ):
        assert internal_term not in catalog.category_system_prompt
    assert "用户没有明确限定执行细节时" in (
        catalog.child_system_prompts["B001"]
    )
    assert "单手或双手、左右方向、身体部位、次数、幅度、移动方向或交互物体" in (
        catalog.child_system_prompts["B001"]
    )
    assert catalog.category_cache_namespace().endswith(catalog.category_prompt_hash)
    assert catalog.child_cache_namespace("B001").endswith(
        catalog.child_prompt_hashes["B001"]
    )
    english_category_prompt = catalog.category_system_prompt_for("en-US")
    english_child_prompt = catalog.child_system_prompt_for("en-US", "B001")
    assert "You are a digital-character action category classifier" in (
        english_category_prompt
    )
    assert "they are not keyword-matching rules" in english_category_prompt
    assert "'greet me' asks for an observable greeting" in (
        english_category_prompt
    )
    assert (
        "social act directed at the digital character"
        in english_category_prompt
    )
    assert "Before selecting B043-B052" not in english_category_prompt
    assert "is not a prerequisite for selecting an object-interaction category" in (
        english_category_prompt
    )
    assert "has no explicit action target" in english_category_prompt
    assert "do not specify a concrete action" in english_category_prompt
    assert "prefer a small low-disturbance action within that category" in (
        english_category_prompt
    )
    assert "must not override an explicit action request that is supported" in (
        english_category_prompt
    )
    assert "whether initiated by the user or the digital character" not in (
        english_category_prompt
    )
    assert "category=问候" in english_category_prompt
    assert "action=单手挥手" in english_child_prompt
    assert english_category_prompt != catalog.category_system_prompt
    assert catalog.category_cache_namespace("en-US") != (
        catalog.category_cache_namespace("zh-CN")
    )
    assert catalog.child_cache_namespace("B001", "en-US") != (
        catalog.child_cache_namespace("B001", "zh-CN")
    )
    with pytest.raises(TypeError):
        catalog.category_by_id["B999"] = catalog.categories[0]  # type: ignore[index]


def test_global_catalog_does_not_require_b008(tmp_path) -> None:
    payload = _catalog_payload()
    payload["categories"] = [
        category
        for category in payload["categories"]
        if category["category_id"] != "B008"
    ]

    catalog = load_global_action_catalog(_write_catalog(tmp_path, payload))

    assert "B008" not in catalog.category_by_id
    assert catalog.candidate_count == 3


@pytest.mark.parametrize("path_value", [None, []])
def test_global_catalog_category_path_is_optional(tmp_path, path_value) -> None:
    payload = _catalog_payload()
    if path_value is None:
        payload["categories"][0].pop("category_path")
    else:
        payload["categories"][0]["category_path"] = path_value

    catalog = load_global_action_catalog(_write_catalog(tmp_path, payload))

    assert catalog.categories[0].category_path == ()


def test_global_catalog_semantic_tags_are_optional_and_drive_prompt_rules(
    tmp_path,
) -> None:
    payload = _catalog_payload()
    payload["categories"][0].pop("semantic_tags", None)

    catalog = load_global_action_catalog(_write_catalog(tmp_path, payload))

    assert catalog.categories[0].semantic_tags == frozenset()
    greeting = catalog.category_by_id["B001"]
    assert greeting.semantic_tags == frozenset({"greeting"})
    assert "单手问候候选" in catalog.child_system_prompt_for("zh-CN", "B001")
    assert "单手问候候选" not in catalog.child_system_prompt_for("zh-CN", "B008")


def test_global_catalog_allows_one_action_in_multiple_categories(tmp_path) -> None:
    payload = _catalog_payload()
    shared = dict(payload["categories"][1]["children"][0])
    shared.update(
        source_label="系统视图中的单手挥手",
        short_definition="同一执行动作在另一类别中的场景化说明",
    )
    payload["categories"][2]["children"].append(shared)

    catalog = load_global_action_catalog(_write_catalog(tmp_path, payload))

    assert catalog.candidate_count == 4
    assert catalog.candidate_for_category("B001", "A001").source_label == (
        "单手挥手"
    )
    assert catalog.candidate_for_category("B002", "A001").source_label == (
        "系统视图中的单手挥手"
    )
    assert "candidate_id=A001｜动作=系统视图中的单手挥手" in (
        catalog.child_system_prompt_for("zh-CN", "B002")
    )


def test_global_catalog_rejects_duplicate_system_semantic_tag_owner(
    tmp_path,
) -> None:
    payload = _catalog_payload()
    payload["categories"][0]["semantic_tags"] = [
        CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
    ]
    payload["categories"][1]["semantic_tags"] = [
        CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
    ]

    with pytest.raises(
        ValueError,
        match=(
            "global category semantic tag must have at most one owner: "
            "reply_accompaniment"
        ),
    ):
        load_global_action_catalog(_write_catalog(tmp_path, payload))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda payload: payload["categories"][1].update(
                category_id="B008"
            ),
            "duplicate global category_id",
        ),
        (
            lambda payload: payload["categories"][1]["children"][0].update(
                action_id="A008"
            ),
            "duplicate global action_id",
        ),
        (
            lambda payload: payload["categories"][1]["children"][0].update(
                action_id="no_action"
            ),
            "must not contain action_id=no_action",
        ),
        (
            lambda payload: payload["categories"][1].update(category_id="B000"),
            "category_id is reserved for unsupported scoring: B000",
        ),
        (
            lambda payload: payload["categories"][1]["children"][0].update(
                candidate_id="A000"
            ),
            "candidate_id is reserved for unsupported scoring: A000",
        ),
        (
            lambda payload: payload["categories"][1]["children"][0].update(
                action_id="UNSUPPORTED"
            ),
            "action_id=UNSUPPORTED is reserved",
        ),
        (
            lambda payload: payload["categories"][1].update(
                semantic_tags=["greeting", "greeting"]
            ),
            "semantic_tags must not contain duplicates",
        ),
    ],
)
def test_global_catalog_rejects_invalid_identity_or_path(
    tmp_path, mutate, message
) -> None:
    payload = _catalog_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        load_global_action_catalog(_write_catalog(tmp_path, payload))


class _PrefillClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def prefill_action_catalog(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return not kwargs["request_id"].endswith("-B002")


@pytest.mark.asyncio
async def test_global_prewarm_covers_category_and_every_child(tmp_path) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _PrefillClient()

    status = await prewarm_global_action_catalog(
        client, model="Qwen3-Omni", catalog=catalog
    )

    assert len(client.calls) == 2 * (1 + len(catalog.categories))
    category_calls = [
        call for call in client.calls if call["stage"] == "category"
    ]
    child_calls = [call for call in client.calls if call["stage"] == "child"]
    assert {call["language"] for call in category_calls} == {"zh", "en"}
    assert {
        call["prefix_cache_namespace"] for call in category_calls
    } == {
        catalog.category_cache_namespace("zh-CN"),
        catalog.category_cache_namespace("en-US"),
    }
    assert all(
        call["candidates"][-1].candidate_id
        == UNSUPPORTED_CATEGORY_SCORE_ID
        for call in category_calls
    )
    assert {
        call["prefix_cache_namespace"] for call in child_calls
    } == {
        catalog.child_cache_namespace(category.category_id, locale)
        for locale in ("zh-CN", "en-US")
        for category in catalog.categories
    }
    assert all(
        call["candidates"][-1].candidate_id == UNSUPPORTED_CHILD_SCORE_ID
        for call in child_calls
    )
    assert status.category_ready is True
    assert status.ready_child_category_ids == frozenset({"B008", "B001"})
    assert status.failed_child_category_ids == frozenset({"B002"})
    assert set(status.by_locale) == {"zh-CN", "en-US"}
    assert all(item.category_ready for item in status.by_locale.values())


@pytest.mark.asyncio
async def test_global_prewarm_omits_a000_for_system_accompaniment_children() -> None:
    catalog = load_global_action_catalog()

    class SuccessfulPrefillClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def prefill_action_catalog(self, **kwargs) -> bool:
            self.calls.append(kwargs)
            return True

    client = SuccessfulPrefillClient()
    await prewarm_global_action_catalog(
        client, model="Qwen3-Omni", catalog=catalog
    )

    system_category_ids = {
        catalog.category_with_semantic_tag(
            CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
        ).category_id,
        catalog.category_with_semantic_tag(
            CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        ).category_id,
    }
    for call in client.calls:
        if call["stage"] != "child":
            continue
        category_id = call["request_id"].rsplit("-", 1)[-1]
        candidate_ids = {
            candidate.candidate_id for candidate in call["candidates"]
        }
        if category_id in system_category_ids:
            assert UNSUPPORTED_CHILD_SCORE_ID not in candidate_ids
        else:
            assert UNSUPPORTED_CHILD_SCORE_ID in candidate_ids


class _WebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send_text(self, value: str) -> None:
        self.events.append(json.loads(value))


class _ScoreClient:
    def __init__(self) -> None:
        self.requests = []
        self.prefill_calls = 0

    async def prefill_action_catalog(self, **kwargs) -> bool:
        self.prefill_calls += 1
        return True

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        self.requests.append(request)
        selected = "B001" if request.stage == "category" else "A001"
        scores = []
        for index, candidate in enumerate(request.candidates):
            logprob = -0.01 if candidate.candidate_id == selected else -10.0 - index
            scores.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    token_count=1,
                    mean_logprob=logprob,
                    mean_nll=-logprob,
                    ppl=1.0,
                    token_scores=[TokenScore(token_id=100 + index, logprob=logprob)],
                )
            )
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=scores,
        )


class _WhitelistScoreClient(_ScoreClient):
    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        self.requests.append(request)
        selectable = [
            item
            for item in request.candidates
            if item.candidate_id
            not in {
                UNSUPPORTED_CATEGORY_SCORE_ID,
                UNSUPPORTED_CHILD_SCORE_ID,
            }
        ]
        selected = selectable[-1].candidate_id
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    token_count=1,
                    mean_logprob=(
                        -0.01 if candidate.candidate_id == selected else -10.0
                    ),
                    mean_nll=(
                        0.01 if candidate.candidate_id == selected else 10.0
                    ),
                    ppl=1.0,
                    token_scores=[TokenScore(token_id=100 + index, logprob=-0.01)],
                )
                for index, candidate in enumerate(request.candidates)
            ],
        )


class _DecisionScoreClient(_ScoreClient):
    def __init__(self, *, category: str, child: str) -> None:
        super().__init__()
        self.category = category
        self.child = child
        self.completion_requests = []

    async def completion(self, request, *, request_id: str) -> CompletionResult:
        self.completion_requests.append(request)
        return CompletionResult(request_id=request_id, text="换个互动方式吧。")

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        self.requests.append(request)
        selected = self.category if request.stage == "category" else self.child
        scores = []
        for index, candidate in enumerate(request.candidates):
            logprob = -0.01 if candidate.candidate_id == selected else -10.0
            scores.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    token_count=1,
                    mean_logprob=logprob,
                    mean_nll=-logprob,
                    ppl=1.0,
                    token_scores=[
                        TokenScore(token_id=100 + index, logprob=logprob)
                    ],
                )
            )
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=scores,
        )


def _session_start_payload() -> dict:
    return {
        "type": "session.start",
        "session_id": "global-session",
        "modalities": ["action"],
        "selection_mode": "hierarchical",
        "language": "zh",
        "include_scores": True,
        "fallback_category_ids": ["B008"],
        "action_candidates": [
            {
                "category_id": "B008",
                "source_label": "待机动作",
                "short_definition": "自然待机",
                "category_path": ["基础姿态与动作转场", "待机动作"],
                "children": [
                    {
                        "candidate_id": "A008",
                        "action_id": "A008",
                        "source_label": "自然呼吸",
                        "short_definition": "自然待机呼吸",
                    }
                ],
            },
            {
                "category_id": "B001",
                "source_label": "问候",
                "short_definition": "问候动作",
                "category_path": ["社交", "问候"],
                "children": [
                    {
                        "candidate_id": "A001",
                        "action_id": "A001",
                        "source_label": "单手挥手",
                        "short_definition": "单手自然挥动",
                        "execution_binding": {"asset_id": "wave-1"},
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_session_uses_global_prompts_and_dynamic_whitelists(tmp_path) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _ScoreClient()
    ws = _WebSocket()
    claimed = {}
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: claimed.setdefault(
            session_id, value
        ),
        release_session=lambda session_id, value: claimed.pop(session_id, None),
    )

    await session.handle_session_start(_session_start_payload())
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-1",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-1",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "你好",
        }
    )

    assert client.prefill_calls == 0
    category_request, child_request = client.requests
    assert category_request.system_prompt == catalog.category_system_prompt
    assert "category_id=B002" in category_request.system_prompt
    assert "[本次会话允许选择的动作类别]" in category_request.prefix
    assert "可用的真实 category_id：B008、B001" in category_request.prefix
    assert "按优先级从高到低为：B008（待机动作）" in category_request.prefix
    assert "该列表只定义不支持判定后的可执行兜底顺序" in (
        category_request.prefix
    )
    assert "不得把执行兜底类别当作已支持该请求的替代类别" in (
        category_request.prefix
    )
    assert "PPL" not in category_request.prefix
    assert "硬过滤" not in category_request.prefix
    assert "B002" not in category_request.prefix
    assert [item.candidate_id for item in category_request.candidates] == [
        "B008",
        "B001",
        UNSUPPORTED_CATEGORY_SCORE_ID,
    ]
    assert category_request.candidates[-1].suffix == "B000"
    assert category_request.candidates[-1].action_id == "UNSUPPORTED"
    assert category_request.prefix_cache_namespace == (
        catalog.category_cache_namespace()
    )
    category_omni_request = Client._build_action_scoring_request(
        category_request
    )
    assert category_omni_request.params["action_scoring"][
        "cache_static_system_only"
    ] is True
    assert child_request.system_prompt == catalog.child_system_prompts["B001"]
    assert "candidate_id=A002" in child_request.system_prompt
    assert "只允许从以下 candidate_id 中选择：A001" in child_request.prefix
    assert "即使当前类别也被列为执行兜底类别" in child_request.prefix
    assert [item.candidate_id for item in child_request.candidates] == [
        "A001",
        UNSUPPORTED_CHILD_SCORE_ID,
    ]
    assert child_request.candidates[-1].suffix == "A000"
    assert child_request.candidates[-1].action_id == "UNSUPPORTED"
    assert child_request.prefix_cache_namespace == catalog.child_cache_namespace(
        "B001"
    )
    started = next(item for item in ws.events if item["type"] == "session.started")
    assert started["fallback_category_ids"] == ["B008"]
    assert started["global_action_catalog_hash"] == catalog.catalog_hash
    assert started["session_action_catalog_hash"] != catalog.catalog_hash
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["action"] == {
        "action_id": "A001",
        "candidate_id": "A001",
            "category_id": "B001",
            "execute": True,
            "support_status": "supported",
            "fallback_applied": False,
            "execution_binding": {"asset_id": "wave-1"},
    }


@pytest.mark.asyncio
async def test_english_session_uses_isolated_english_prompts_without_translating_client_text(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _ScoreClient()
    session = MultimodalSession(
        _WebSocket(),
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    payload = _session_start_payload()
    payload["language"] = "en"
    payload["action_profile"] = {
        "persona": {"role": "海洋科学家"},
        "category_preferences": "优先低打扰类别",
        "action_preferences": "避免大幅位移",
    }
    await session.handle_session_start(payload)
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-en",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-en",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "你好",
        }
    )

    category_request, child_request = client.requests
    assert session.locale == "en-US"
    assert category_request.language == "en"
    assert category_request.system_prompt == catalog.category_system_prompt_for(
        "en-US"
    )
    assert child_request.system_prompt == catalog.child_system_prompt_for(
        "en-US", "B001"
    )
    assert category_request.output_prompt == "Best matching category_id:"
    assert child_request.output_prompt == "Best matching candidate_id:"
    assert category_request.current_text == "你好"
    assert child_request.current_text == "你好"
    assert "Action categories allowed in this conversation" in category_request.prefix
    assert "海洋科学家" in category_request.prefix
    assert "优先低打扰类别" in category_request.prefix
    assert "避免大幅位移" in child_request.prefix
    assert session.action_prefix_cache_namespace == catalog.category_cache_namespace(
        "en-US"
    )


@pytest.mark.asyncio
async def test_chinese_and_english_sessions_cannot_share_localized_prefixes(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    sessions = [
        MultimodalSession(
            _WebSocket(),
            client=_ScoreClient(),  # type: ignore[arg-type]
            model_name="Qwen3-Omni",
            global_action_catalog=catalog,
            claim_session=lambda session_id, value: None,
            release_session=lambda session_id, value: None,
        )
        for _ in range(2)
    ]
    zh_payload = _session_start_payload()
    zh_payload.update({"session_id": "session-zh", "language": "zh"})
    en_payload = _session_start_payload()
    en_payload.update({"session_id": "session-en", "language": "en"})

    await asyncio.gather(
        sessions[0].handle_session_start(zh_payload),
        sessions[1].handle_session_start(en_payload),
    )

    assert sessions[0].locale == "zh-CN"
    assert sessions[1].locale == "en-US"
    assert sessions[0].action_system_prompt == catalog.category_system_prompt_for(
        "zh-CN"
    )
    assert sessions[1].action_system_prompt == catalog.category_system_prompt_for(
        "en-US"
    )
    assert (
        sessions[0].action_prefix_cache_namespace
        != sessions[1].action_prefix_cache_namespace
    )


@pytest.mark.asyncio
async def test_hierarchical_action_scoring_omits_cross_turn_history(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(category="B001", child="A001")
    session = MultimodalSession(
        _WebSocket(),
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    payload = _session_start_payload()
    payload["modalities"] = ["text", "action"]
    await session.handle_session_start(payload)

    for turn_id, text in (("turn-1", "你好"), ("turn-2", "再来一次")):
        await session.handle_turn_start(
            {
                "type": "turn.start",
                "turn_id": turn_id,
                "turn_origin": "user",
                "text_role": "user_input",
            }
        )
        await session.handle_turn_commit(
            {
                "type": "turn.commit",
                "turn_id": turn_id,
                "turn_origin": "user",
                "text_role": "user_input",
                "text": text,
            }
        )

    category_request, child_request = client.requests[2:]
    for request in (category_request, child_request):
        assert request.history_audios == []
        assert request.history_images == []
        assert request.history == []
        assert request.avatar_state == {}


@pytest.mark.asyncio
async def test_category_unsupported_routes_to_configured_primary_fallback(tmp_path) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(
        category=UNSUPPORTED_CATEGORY_SCORE_ID, child="A001"
    )
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    payload = _session_start_payload()
    payload["fallback_category_ids"] = ["B001", "B008"]
    await session.handle_session_start(payload)
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-unsupported-category",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-unsupported-category",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "做个后空翻",
        }
    )

    assert [item.stage for item in client.requests] == ["category", "child"]
    assert [
        item.candidate_id for item in client.requests[1].candidates
    ] == ["A001", UNSUPPORTED_CHILD_SCORE_ID]
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["action"]["category_id"] == "B001"
    assert result["action"]["candidate_id"] == "A001"
    assert result["action"]["execute"] is True
    assert result["action"]["support_status"] == "unsupported"
    assert result["action"]["fallback_applied"] is True


@pytest.mark.asyncio
async def test_category_unsupported_reaches_reply_before_idle_child_finishes(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(
        category=UNSUPPORTED_CATEGORY_SCORE_ID, child="A008"
    )
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    payload = _session_start_payload()
    payload["modalities"] = ["text", "action"]
    payload["instructions"] = "自然简洁地回复。"
    payload["unsupported_action_text"] = "这个动作暂时做不了。"
    await session.handle_session_start(payload)
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-unsupported-reply",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-unsupported-reply",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "做个后空翻",
        }
    )

    assert len(client.completion_requests) == 1
    reply_request = client.completion_requests[0]
    assert {"type": "text", "text": "做个后空翻"} in (
        reply_request.messages[-1].content
    )
    resolved = next(
        item
        for item in ws.events
        if item["type"] == "response.provisional.resolved"
    )
    assert resolved["status"] == "discarded"
    assert resolved["reason"] == "category_unsupported"
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert "text" not in result["reply"]
    assert result["reply"]["source"] == "client_prerecorded_audio"
    assert result["action"]["support_status"] == "unsupported"
    assert session.reply_history_turns[-1].messages[-1]["content"] == (
        "这个动作暂时做不了。"
    )
    assert session.reply_history_turns[-1].model_visible is False
    assert (
        session.reply_history_turns[-1].history_kind
        == "unsupported_action_notice"
    )


@pytest.mark.asyncio
async def test_child_unsupported_routes_to_default_idle_action(tmp_path) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(
        category="B001", child=UNSUPPORTED_CHILD_SCORE_ID
    )
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    await session.handle_session_start(_session_start_payload())
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-unsupported-child",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-unsupported-child",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "敬个军礼",
        }
    )

    child_request = client.requests[1]
    assert [item.candidate_id for item in child_request.candidates] == [
        "A001",
        UNSUPPORTED_CHILD_SCORE_ID,
    ]
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["action"]["category_id"] == "B008"
    assert result["action"]["candidate_id"] == "A008"
    assert result["action"]["execute"] is True
    assert result["action"]["support_status"] == "unsupported"
    assert result["action"]["fallback_applied"] is True


@pytest.mark.asyncio
async def test_default_category_child_can_reject_explicit_action_request(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(
        category="B008", child=UNSUPPORTED_CHILD_SCORE_ID
    )
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    await session.handle_session_start(_session_start_payload())
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-default-child-unsupported",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-default-child-unsupported",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "做个后空翻",
        }
    )

    child_request = client.requests[1]
    assert [item.candidate_id for item in child_request.candidates] == [
        "A008",
        UNSUPPORTED_CHILD_SCORE_ID,
    ]
    assert "即使当前类别也被列为执行兜底类别" in child_request.prefix
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["action"]["category_id"] == "B008"
    assert result["action"]["candidate_id"] == "A008"
    assert result["action"]["support_status"] == "unsupported"
    assert result["action"]["fallback_applied"] is True


@pytest.mark.asyncio
async def test_child_unsupported_discards_provisional_reply_and_records_fallback_text(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(
        category="B001", child=UNSUPPORTED_CHILD_SCORE_ID
    )
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    payload = _session_start_payload()
    payload["modalities"] = ["text", "action"]
    payload["unsupported_action_text"] = "这个动作暂时做不了。"
    await session.handle_session_start(payload)
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-child-unsupported-fusion",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-child-unsupported-fusion",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "敬个军礼",
        }
    )

    resolved = next(
        item
        for item in ws.events
        if item["type"] == "response.provisional.resolved"
    )
    assert resolved["status"] == "discarded"
    assert resolved["reason"] == "child_unsupported"
    assert not any(item["type"] == "response.text.delta" for item in ws.events)
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["modalities"]["text"] == "suppressed"
    assert result["reply"] == {
        "source": "client_prerecorded_audio",
        "reason": "unsupported_action",
        "recorded_in_history": True,
    }
    assert result["action"]["support_status"] == "unsupported"
    assert session.reply_history_turns[-1].messages[-1]["content"] == (
        "这个动作暂时做不了。"
    )
    assert session.reply_history_turns[-1].model_visible is False
    assert (
        session.reply_history_turns[-1].history_kind
        == "unsupported_action_notice"
    )
    assert "这个动作暂时做不了。" not in (
        session.history_turns[-1].messages[-1]["content"]
    )

    # The notice remains auditable, but must not become an ordinary assistant
    # example that a later supported reply can imitate.
    client.category = "B001"
    client.child = "A001"
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-after-unsupported",
            "turn_origin": "user",
            "text_role": "user_input",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-after-unsupported",
            "turn_origin": "user",
            "text_role": "user_input",
            "text": "向我挥手",
        }
    )
    next_reply_messages = [
        message.to_dict()
        for message in client.completion_requests[-1].messages
    ]
    assert "这个动作暂时做不了。" not in json.dumps(
        next_reply_messages, ensure_ascii=False
    )


@pytest.mark.asyncio
async def test_explicit_proactive_prohibition_filters_conflicting_category(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    client = _DecisionScoreClient(category="B008", child="A001")
    ws = _WebSocket()
    session = MultimodalSession(
        ws,
        client=client,  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        global_action_prewarm=GlobalActionCatalogPrewarmStatus(
            True,
            frozenset({"B008", "B001", "B002"}),
            frozenset(),
            1.0,
        ),
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )
    await session.handle_session_start(_session_start_payload())
    assert session._state_description_excluded_candidate_ids(
        "禁止：选择自然呼吸。",
        list(session.categories[0].children),
    ) == ("A008",)
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-proactive-no-idle",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "idle_timeout",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-proactive-no-idle",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "idle_timeout",
            "avatar_state": {
                "state_description": (
                    "选择一个清晰可见且有实际身体变化的后续动作。"
                    "禁止：选择自然待机、自然呼吸或其他微动作。"
                )
            },
        }
    )

    category_request = client.requests[0]
    assert "B008" not in {
        candidate.candidate_id for candidate in category_request.candidates
    }
    assert "已从本轮可选集合移除：B008" in category_request.prefix
    result = next(item for item in ws.events if item["type"] == "turn.result")
    assert result["action"]["category_id"] == "B001"
    assert result["action"]["candidate_id"] == "A001"


@pytest.mark.asyncio
async def test_session_rejects_semantic_mismatch_against_global_catalog(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    payload = _session_start_payload()
    payload["action_candidates"][1]["children"][0]["source_label"] = "客户端篡改名称"
    session = MultimodalSession(
        _WebSocket(),
        client=_ScoreClient(),  # type: ignore[arg-type]
        model_name="Qwen3-Omni",
        global_action_catalog=catalog,
        claim_session=lambda session_id, value: None,
        release_session=lambda session_id, value: None,
    )

    with pytest.raises(ValueError, match="does not match the global catalog"):
        await session.handle_session_start(payload)


@pytest.mark.asyncio
async def test_global_prefix_sharing_keeps_session_whitelists_and_bindings_isolated(
    tmp_path,
) -> None:
    catalog = load_global_action_catalog(_write_catalog(tmp_path))
    status = GlobalActionCatalogPrewarmStatus(
        True,
        frozenset({"B008", "B001", "B002"}),
        frozenset(),
        1.0,
    )

    async def run_session(
        *,
        session_id: str,
        category_id: str,
        candidate_id: str,
        asset_id: str,
        persona_role: str,
    ):
        client = _WhitelistScoreClient()
        session = MultimodalSession(
            _WebSocket(),
            client=client,  # type: ignore[arg-type]
            model_name="Qwen3-Omni",
            global_action_catalog=catalog,
            global_action_prewarm=status,
            claim_session=lambda claimed_id, value: None,
            release_session=lambda claimed_id, value: None,
        )
        global_category = catalog.category_by_id[category_id]
        global_candidate = catalog.candidate_by_id[candidate_id]
        payload = _session_start_payload()
        payload["session_id"] = session_id
        payload["action_profile"] = {
            "persona": {"role": persona_role},
            "category_preferences": f"{persona_role}类目偏好",
            "action_preferences": f"{persona_role}动作偏好",
        }
        payload["action_candidates"] = [
            payload["action_candidates"][0],
            {
                "category_id": category_id,
                "source_label": global_category.source_label,
                "short_definition": global_category.short_definition,
                "category_path": list(global_category.category_path),
                "children": [
                    {
                        "candidate_id": candidate_id,
                        "action_id": global_candidate.action_id,
                        "source_label": global_candidate.source_label,
                        "short_definition": global_candidate.source_short_definition,
                        "execution_binding": {"asset_id": asset_id},
                    }
                ],
            },
        ]
        await session.handle_session_start(payload)
        await session.handle_turn_start(
            {
                "type": "turn.start",
                "turn_id": "turn-1",
                "turn_origin": "user",
                "text_role": "user_input",
            }
        )
        await session.handle_turn_commit(
            {
                "type": "turn.commit",
                "turn_id": "turn-1",
                "turn_origin": "user",
                "text_role": "user_input",
                "text": "测试",
            }
        )
        return session, client

    first, first_client = await run_session(
        session_id="session-a",
        category_id="B001",
        candidate_id="A001",
        asset_id="asset-a",
        persona_role="海洋科学家",
    )
    second, second_client = await run_session(
        session_id="session-b",
        category_id="B002",
        candidate_id="A003",
        asset_id="asset-b",
        persona_role="卡通主持人",
    )

    assert first.action_prefix_cache_namespace == second.action_prefix_cache_namespace
    assert first.action_prefix_cache_namespace == catalog.category_cache_namespace()
    assert first_client.requests[0].system_prompt == second_client.requests[0].system_prompt
    assert "职业或角色定位=海洋科学家" in first_client.requests[0].prefix
    assert "卡通主持人" not in first_client.requests[0].prefix
    assert "职业或角色定位=卡通主持人" in second_client.requests[0].prefix
    assert "海洋科学家" not in second_client.requests[0].prefix
    assert [item.candidate_id for item in first_client.requests[0].candidates] == [
        "B008",
        "B001",
        UNSUPPORTED_CATEGORY_SCORE_ID,
    ]
    assert [item.candidate_id for item in second_client.requests[0].candidates] == [
        "B008",
        "B002",
        UNSUPPORTED_CATEGORY_SCORE_ID,
    ]
    assert first_client.requests[1].prefix_cache_namespace == (
        catalog.child_cache_namespace("B001")
    )
    assert "海洋科学家动作偏好" in first_client.requests[1].prefix
    assert "卡通主持人" not in first_client.requests[1].prefix
    assert second_client.requests[1].prefix_cache_namespace == (
        catalog.child_cache_namespace("B002")
    )
    assert "卡通主持人动作偏好" in second_client.requests[1].prefix
    assert "海洋科学家" not in second_client.requests[1].prefix
    assert first.candidate_by_id["A001"].execution_binding == {
        "asset_id": "asset-a"
    }
    assert second.candidate_by_id["A003"].execution_binding == {
        "asset_id": "asset-b"
    }
    assert "A003" not in first.candidate_by_id
    assert "A001" not in second.candidate_by_id
