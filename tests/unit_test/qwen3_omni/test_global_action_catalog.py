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
    ACTION_HISTORY_INSTRUCTION,
    ACTION_HISTORY_INSTRUCTION_EN,
    GlobalActionCatalogPrewarmStatus,
    UNSUPPORTED_CATEGORY_SHORT_DEFINITION,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SHORT_DEFINITION,
    UNSUPPORTED_CHILD_SCORE_ID,
    load_global_action_catalog,
    prewarm_global_action_catalog,
)
from sglang_omni.serve.realtime.multimodal import MultimodalSession


def test_action_reference_prompts_separate_physical_and_user_actions() -> None:
    assert "[当前实际动作状态]" in ACTION_HISTORY_INSTRUCTION
    assert "[最近一次用户触发动作]" in ACTION_HISTORY_INSTRUCTION
    assert "不得用后来由数字人主动触发的动作替代" in ACTION_HISTORY_INSTRUCTION
    assert "[Current physical action state]" in ACTION_HISTORY_INSTRUCTION_EN
    assert "[Most recent user-triggered action]" in ACTION_HISTORY_INSTRUCTION_EN
    assert "not a later action initiated proactively" in ACTION_HISTORY_INSTRUCTION_EN


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
    assert "以下示例仅说明语义判断方法，不是关键词匹配规则" in (
        catalog.category_system_prompt
    )
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
    assert "本次会话提供的默认动作类别" in (
        catalog.category_system_prompt
    )
    assert "避免选择要求下肢、位移或全身大幅移动的 B033-B037" in (
        catalog.category_system_prompt
    )
    assert "选择要求与具体物体交互的 B043-B052 前" in (
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
            lambda payload: payload["categories"][1].update(category_path=[]),
            "category_path",
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
    assert "这些类别只用于当前输入没有明确要求具体动作的情况" in (
        category_request.prefix
    )
    assert "不得把默认动作类别作为已支持该请求的替代类别" in (
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
    assert "即使当前类别是默认动作类别" in child_request.prefix
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
    assert category_request.prefix.endswith("Best matching category_id:")
    assert child_request.prefix.endswith("Best matching candidate_id:")
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
async def test_hierarchical_action_history_keeps_only_latest_reply_and_action_fact(
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
        assert len(request.history) == 1
        assert request.history[0]["role"] == "assistant"
        content = request.history[0]["content"]
        assert "[数字人最近一次回复] 换个互动方式吧。" in content
        assert "[当前实际动作状态；同时是最近一次用户触发动作]" in content
        assert "turn_id=" not in content
        assert "candidate_id=A001" in content
        assert "你好" not in content


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
    assert "即使当前类别是默认动作类别" in child_request.prefix
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
            "trigger": "action_finished",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-proactive-no-idle",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "action_finished",
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
