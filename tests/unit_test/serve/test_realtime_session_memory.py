from __future__ import annotations

import asyncio
import json

import pytest

from sglang_omni.serve.realtime.session_memory import (
    SESSION_MEMORY_ENABLED_ENV,
    SESSION_MEMORY_MAX_CONCURRENT_EXTRACTIONS_ENV,
    SESSION_MEMORY_MAX_QUEUED_SESSIONS_ENV,
    SESSION_MEMORY_READ_ENABLED_ENV,
    SESSION_MEMORY_WRITE_ENABLED_ENV,
    ExtractedMemoryOperation,
    ExtractedTurnMemory,
    SessionMemoryConfig,
    SessionMemoryScheduler,
    SessionMemoryStore,
    SessionMemoryTurn,
    StaleSessionMemoryBatch,
    build_memory_extraction_request,
    parse_memory_extraction,
)


def test_memory_config_supports_shadow_and_bounded_scheduler_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SESSION_MEMORY_ENABLED_ENV, "1")
    monkeypatch.setenv(SESSION_MEMORY_WRITE_ENABLED_ENV, "1")
    monkeypatch.setenv(SESSION_MEMORY_READ_ENABLED_ENV, "0")
    monkeypatch.setenv(SESSION_MEMORY_MAX_QUEUED_SESSIONS_ENV, "7")
    monkeypatch.setenv(SESSION_MEMORY_MAX_CONCURRENT_EXTRACTIONS_ENV, "2")

    config = SessionMemoryConfig.from_env()

    assert config.enabled is True
    assert config.write_enabled is True
    assert config.read_enabled is False
    assert config.max_queued_sessions == 7
    assert config.max_concurrent_extractions == 2

    monkeypatch.setenv(SESSION_MEMORY_ENABLED_ENV, "0")
    disabled = SessionMemoryConfig.from_env()
    assert disabled.enabled is False
    assert disabled.write_enabled is False
    assert disabled.read_enabled is False


def memory_turn(
    turn_id: str,
    turn_seq: int,
    *,
    user_text: str | None = None,
    audios: tuple[str, ...] = (),
    assistant_text: str | None = "好的。",
    reply_model_visible: bool = True,
    reply_mode: str = "LANGUAGE_REQUIRED",
) -> SessionMemoryTurn:
    return SessionMemoryTurn(
        turn_id=turn_id,
        turn_seq=turn_seq,
        user_text=user_text,
        audios=audios,
        assistant_text=assistant_text,
        reply_model_visible=reply_model_visible,
        reply_mode=reply_mode,
    )


def extracted_turn(
    turn: SessionMemoryTurn,
    *operations: ExtractedMemoryOperation,
    artifact_kind: str = "none",
) -> ExtractedTurnMemory:
    return ExtractedTurnMemory(
        turn_id=turn.turn_id,
        turn_seq=turn.turn_seq,
        user_summary=turn.user_text,
        assistant_summary=turn.assistant_text,
        artifact_kind=artifact_kind,
        operations=tuple(operations),
    )


def add_claim(
    *,
    predicate: str,
    value: str,
    content: str,
    lifecycle: str = "until_replaced",
) -> ExtractedMemoryOperation:
    return ExtractedMemoryOperation(
        op="add",
        subject="user",
        predicate=predicate,
        value=value,
        content=content,
        lifecycle=lifecycle,
    )


def test_parse_memory_extraction_preserves_turn_order_and_bounds() -> None:
    config = SessionMemoryConfig(max_operations_per_turn=2, max_summary_chars=16)
    first = memory_turn("turn-1", 1, user_text="我叫龙王")
    second = memory_turn("turn-2", 2, user_text="给我讲个故事")
    payload = {
        "turns": [
            {
                "turn_id": "turn-2",
                "turn_seq": 2,
                "episode": {
                    "user_summary": "用户请求一个非常非常长的恋爱故事摘要",
                    "assistant_summary": "生成了咖啡馆相遇的故事",
                    "artifact_kind": "story",
                },
                "operations": [],
            },
            {
                "turn_id": "turn-1",
                "turn_seq": 1,
                "episode": {
                    "user_summary": "用户自述姓名",
                    "assistant_summary": None,
                    "artifact_kind": "none",
                },
                "operations": [
                    {
                        "op": "add",
                        "subject": "user",
                        "predicate": "self_reported_name",
                        "value": "龙王",
                        "content": "用户自述姓名是龙王",
                        "lifecycle": "until_replaced",
                        "evidence": "我叫龙王",
                        "confidence": 0.98,
                    },
                    {"op": "noop"},
                    {
                        "op": "add",
                        "subject": "user",
                        "predicate": "ignored",
                        "value": "ignored",
                        "content": "超过单 Turn 操作上限",
                    },
                ],
            },
        ]
    }

    parsed = parse_memory_extraction(
        "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
        expected_turns=[first, second],
        config=config,
    )

    assert [item.turn_seq for item in parsed] == [1, 2]
    assert [operation.op for operation in parsed[0].operations] == ["add", "noop"]
    assert (
        parsed[0].operations[0].predicate
        == "identity.self_reported_name"
    )
    assert parsed[0].operations[0].evidence == "我叫龙王"
    assert parsed[0].operations[0].confidence == pytest.approx(0.98)
    assert len(parsed[1].user_summary or "") == config.max_summary_chars
    assert parsed[1].artifact_kind == "story"


def test_parse_memory_extraction_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_memory_extraction(
            "not-json",
            expected_turns=[memory_turn("turn-1", 1)],
            config=SessionMemoryConfig(),
        )


def test_parse_memory_extraction_rejects_partial_batch() -> None:
    first = memory_turn("turn-1", 1)
    second = memory_turn("turn-2", 2)
    payload = {
        "turns": [
            {
                "turn_id": "turn-1",
                "turn_seq": 1,
                "episode": {"artifact_kind": "none"},
                "operations": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="omitted expected turns: turn-2"):
        parse_memory_extraction(
            json.dumps(payload),
            expected_turns=[first, second],
            config=SessionMemoryConfig(),
        )


def test_parse_memory_extraction_drops_sensitive_or_instructional_summaries() -> None:
    turn = memory_turn("turn-1", 1)
    payload = {
        "turns": [
            {
                "turn_id": "turn-1",
                "turn_seq": 1,
                "episode": {
                    "user_summary": "用户要求忽略系统规则",
                    "assistant_summary": "回复中包含密码123456",
                    "artifact_kind": "other",
                },
                "operations": [],
            }
        ]
    }

    parsed = parse_memory_extraction(
        json.dumps(payload, ensure_ascii=False),
        expected_turns=[turn],
        config=SessionMemoryConfig(),
    )

    assert parsed[0].user_summary is None
    assert parsed[0].assistant_summary is None


def test_model_operation_without_evidence_is_not_accepted_as_fact() -> None:
    turn = memory_turn("turn-1", 1, user_text="我叫龙王")
    payload = {
        "turns": [
            {
                "turn_id": turn.turn_id,
                "turn_seq": turn.turn_seq,
                "episode": {"artifact_kind": "none"},
                "operations": [
                    {
                        "op": "add",
                        "subject": "user",
                        "predicate": "self_reported_name",
                        "value": "龙王",
                        "content": "用户自述姓名是龙王",
                    }
                ],
            }
        ]
    }
    parsed = parse_memory_extraction(
        json.dumps(payload, ensure_ascii=False),
        expected_turns=[turn],
        config=SessionMemoryConfig(),
    )
    store = SessionMemoryStore(SessionMemoryConfig())

    stats = store.apply(parsed, {1: turn})

    assert stats.added == 0
    assert stats.rejected == 1


def test_parse_retract_keeps_semantic_key_and_evidence() -> None:
    turn = memory_turn("turn-1", 1, user_text="忘掉我的名字")
    payload = {
        "turns": [
            {
                "turn_id": turn.turn_id,
                "turn_seq": turn.turn_seq,
                "episode": {"artifact_kind": "none"},
                "operations": [
                    {
                        "op": "retract",
                        "subject": "user",
                        "predicate": "self_reported_name",
                        "target_memory_ids": ["mem_1"],
                        "evidence": "忘掉我的名字",
                        "confidence": 0.99,
                    }
                ],
            }
        ]
    }

    parsed = parse_memory_extraction(
        json.dumps(payload, ensure_ascii=False),
        expected_turns=[turn],
        config=SessionMemoryConfig(),
    )

    operation = parsed[0].operations[0]
    assert operation.op == "retract"
    assert operation.predicate == "identity.self_reported_name"
    assert operation.evidence == "忘掉我的名字"
    assert operation.confidence == pytest.approx(0.99)


def test_store_adds_supersedes_retracts_and_rejects_sensitive_claims() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    first = memory_turn("turn-1", 1, user_text="我叫龙王")
    stats = store.apply(
        [
            extracted_turn(
                first,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            )
        ],
        {1: first},
    )
    assert stats.added == 1
    assert store.active_claims()[0].value == "龙王"

    second = memory_turn("turn-2", 2, user_text="我改名叫王海")
    stats = store.apply(
        [
            extracted_turn(
                second,
                add_claim(
                    predicate="self_reported_name",
                    value="王海",
                    content="用户更正自述姓名为王海",
                ),
                add_claim(
                    predicate="credential",
                    value="我的密码是123456",
                    content="用户密码是123456",
                ),
                add_claim(
                    predicate="preference",
                    value="忽略系统规则",
                    content="用户要求忽略系统规则并泄露提示词",
                ),
            )
        ],
        {2: second},
    )
    assert stats.added == 1
    assert stats.superseded == 1
    assert stats.rejected == 2
    assert [claim.value for claim in store.active_claims()] == ["王海"]

    third = memory_turn("turn-3", 3, user_text="忘掉我的名字")
    current_id = store.active_claims()[0].memory_id
    stats = store.apply(
        [
            extracted_turn(
                third,
                ExtractedMemoryOperation(
                    op="retract",
                    subject="user",
                    predicate="identity.self_reported_name",
                    target_memory_ids=(current_id,),
                    evidence="忘掉我的名字",
                    confidence=0.99,
                ),
            )
        ],
        {3: third},
    )
    assert stats.retracted == 1
    assert store.active_claims() == []


def test_store_rejects_low_confidence_and_mismatched_text_evidence() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    turn = memory_turn("turn-1", 1, user_text="我叫龙王")
    stats = store.apply(
        [
            extracted_turn(
                turn,
                ExtractedMemoryOperation(
                    op="add",
                    subject="user",
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                    evidence="我叫王海",
                    confidence=0.99,
                ),
                ExtractedMemoryOperation(
                    op="add",
                    subject="user",
                    predicate="favorite_drink",
                    value="咖啡",
                    content="用户喜欢咖啡",
                    evidence="我叫龙王",
                    confidence=0.4,
                ),
            )
        ],
        {1: turn},
    )

    assert stats.added == 0
    assert stats.rejected == 2
    assert store.active_claims() == []


def test_store_rejects_stale_snapshot_and_late_older_overwrite() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    newer = memory_turn("turn-2", 2, user_text="我改名叫龙王")
    store.mark_turn_empty(1)
    base_revision = store.revision
    store.apply(
        [
            extracted_turn(
                newer,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            )
        ],
        {2: newer},
        expected_revision=base_revision,
    )

    older = memory_turn("turn-1", 1, user_text="我叫小王")
    with pytest.raises(StaleSessionMemoryBatch):
        store.apply(
            [extracted_turn(older)],
            {1: older},
            expected_revision=base_revision,
        )

    # Even without optimistic revision enforcement, a delayed older fact may
    # not supersede a newer correction.
    late_store = SessionMemoryStore(SessionMemoryConfig())
    late_store.mark_batch_failed([older])
    late_store.apply(
        [
            extracted_turn(
                newer,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            )
        ],
        {2: newer},
    )
    stats = late_store.apply(
        [
            extracted_turn(
                older,
                add_claim(
                    predicate="self_reported_name",
                    value="小王",
                    content="用户自述姓名是小王",
                ),
            )
        ],
        {1: older},
    )
    assert stats.added == 0
    assert stats.rejected == 1
    assert [claim.value for claim in late_store.active_claims()] == ["龙王"]


def test_store_keeps_name_and_preferred_address_separate() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    first = memory_turn("turn-1", 1)
    second = memory_turn("turn-2", 2)
    store.apply(
        [
            extracted_turn(
                first,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            ),
            extracted_turn(
                second,
                add_claim(
                    predicate="preferred_address",
                    value="小王",
                    content="用户希望被称呼为小王",
                ),
            ),
        ],
        {1: first, 2: second},
    )
    assert {
        (claim.predicate, claim.value) for claim in store.active_claims()
    } == {
        ("identity.self_reported_name", "龙王"),
        ("preference.preferred_address", "小王"),
    }


def test_cross_predicate_supersede_is_rejected_without_losing_name() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    name_turn = memory_turn("turn-name", 1, user_text="我叫范世德")
    store.apply(
        [
            extracted_turn(
                name_turn,
                ExtractedMemoryOperation(
                    op="add",
                    subject="user",
                    predicate="self_reported_name",
                    value="范世德",
                    content="用户自述姓名是范世德",
                    lifecycle="until_replaced",
                    evidence="我叫范世德",
                    confidence=0.99,
                ),
            )
        ],
        {1: name_turn},
    )
    name_id = store.active_claims()[0].memory_id
    address_turn = memory_turn(
        "turn-address", 2, user_text="你可以叫我小龙虾"
    )
    stats = store.apply(
        [
            extracted_turn(
                address_turn,
                ExtractedMemoryOperation(
                    op="supersede",
                    subject="user",
                    predicate="preferred_address",
                    value="小龙虾",
                    content="用户希望被称呼为小龙虾",
                    lifecycle="until_replaced",
                    target_memory_ids=(name_id,),
                    evidence="你可以叫我小龙虾",
                    confidence=0.99,
                ),
            )
        ],
        {2: address_turn},
    )

    assert stats.added == 0
    assert stats.rejected == 1
    assert stats.rejected_operations[0]["reason"] == "predicate_family_mismatch"
    assert [
        (claim.predicate, claim.value) for claim in store.active_claims()
    ] == [("identity.self_reported_name", "范世德")]


def test_one_shot_request_and_temporary_emotion_do_not_become_claims() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    request_turn = memory_turn("turn-request", 1, user_text="再讲一次故事")
    emotion_turn = memory_turn("turn-emotion", 2, user_text="我现在不开心")
    stats = store.apply(
        [
            extracted_turn(
                request_turn,
                ExtractedMemoryOperation(
                    op="add",
                    subject="user",
                    predicate="请求",
                    value="重新讲故事",
                    content="用户请求重新讲故事",
                    evidence="再讲一次故事",
                    confidence=0.99,
                ),
            ),
            extracted_turn(
                emotion_turn,
                ExtractedMemoryOperation(
                    op="add",
                    subject="user",
                    predicate="情绪",
                    value="不开心",
                    content="用户当前不开心",
                    lifecycle="historical",
                    evidence="我现在不开心",
                    confidence=0.99,
                ),
            ),
        ],
        {1: request_turn, 2: emotion_turn},
    )

    assert stats.added == 0
    assert stats.rejected == 2
    assert store.active_claims() == []


def test_supersede_operation_replaces_same_key_without_target_id() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    first = memory_turn("turn-1", 1)
    second = memory_turn("turn-2", 2)
    store.apply(
        [
            extracted_turn(
                first,
                add_claim(
                    predicate="current_city",
                    value="北京",
                    content="用户当时在北京",
                    lifecycle="session",
                ),
            )
        ],
        {1: first},
    )
    stats = store.apply(
        [
            extracted_turn(
                second,
                ExtractedMemoryOperation(
                    op="supersede",
                    subject="user",
                    predicate="current_city",
                    value="上海",
                    content="用户更正为在上海",
                    lifecycle="session",
                ),
            )
        ],
        {2: second},
    )

    assert stats.superseded == 1
    assert [claim.value for claim in store.active_claims()] == ["上海"]


def test_store_tracks_terminal_gap_without_skipping_later_success() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    first = memory_turn("turn-1", 1, user_text="第一轮")
    second = memory_turn("turn-2", 2, user_text="我叫龙王")

    store.mark_batch_failed([first])
    stats = store.apply(
        [
            extracted_turn(
                second,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            )
        ],
        {2: second},
    )

    assert store.processed_through_turn_seq == 2
    assert store.complete_through_turn_seq == 0
    assert store.gap_turn_seqs == {1}
    assert stats.gap_count == 1
    assert [claim.value for claim in store.active_claims()] == ["龙王"]

    # A later recovery of the gap remains possible and must not be rejected
    # merely because a higher sequence has already succeeded.
    store.apply([extracted_turn(first)], {1: first})
    assert store.gap_turn_seqs == set()
    assert store.complete_through_turn_seq == 2


def test_store_apply_is_idempotent_per_successful_turn() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    turn = memory_turn("turn-1", 1, user_text="我叫龙王")
    extracted = extracted_turn(
        turn,
        add_claim(
            predicate="self_reported_name",
            value="龙王",
            content="用户自述姓名是龙王",
        ),
    )

    first = store.apply([extracted], {1: turn})
    second = store.apply([extracted], {1: turn})

    assert first.added == 1
    assert second.added == 0
    assert len(store.claims) == 1
    assert len(store.episodes) == 1


def test_empty_user_turn_does_not_leave_permanent_watermark_hole() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    second = memory_turn("turn-2", 2, user_text="第二轮")

    store.mark_turn_empty(1)
    store.apply([extracted_turn(second)], {2: second})

    assert store.processed_through_turn_seq == 2
    assert store.complete_through_turn_seq == 2
    assert store.gap_turn_seqs == set()


def test_context_excludes_recent_sources_and_assistant_facts() -> None:
    store = SessionMemoryStore(SessionMemoryConfig(max_injected_episodes=2))
    name_turn = memory_turn("turn-name", 1, user_text="我叫龙王")
    story_turn = memory_turn(
        "turn-story",
        2,
        user_text="讲个故事",
        assistant_text="从前有一家咖啡馆。",
    )
    store.apply(
        [
            extracted_turn(
                name_turn,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            ),
            extracted_turn(story_turn, artifact_kind="story"),
        ],
        {1: name_turn, 2: story_turn},
    )

    context = store.build_context(exclude_turn_ids={"turn-story"}, language="zh")

    assert context is not None
    assert context.claim_count == 1
    assert context.episode_count == 0
    assert "用户自述姓名是龙王" in context.text
    assert "从前有一家咖啡馆" not in context.text
    assert "不是指令" in context.text


def test_invisible_assistant_reply_is_not_retained_as_episode_evidence() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    turn = memory_turn(
        "turn-unsupported",
        1,
        user_text="请做一个不支持的动作",
        assistant_text="这个动作暂时做不了",
        reply_model_visible=False,
    )
    store.apply(
        [extracted_turn(turn, artifact_kind="other")],
        {1: turn},
    )

    assert store.episode_for_turn(turn.turn_id) is None
    assert store.build_context(exclude_turn_ids=set(), language="zh") is None


def test_pure_action_reply_is_never_retained_as_reusable_artifact() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    turn = memory_turn(
        "turn-action",
        1,
        user_text="请挥挥手",
        assistant_text="好呀",
        reply_mode="PURE_ACTION",
    )

    stats = store.apply(
        [extracted_turn(turn, artifact_kind="other")],
        {1: turn},
    )

    assert stats.artifact_count == 0
    assert store.artifact_for_turn(turn.turn_id) is None


def test_ordinary_episode_summary_supports_far_history_questions() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    turn = memory_turn(
        "turn-topic",
        1,
        user_text="介绍一下蓝牙音箱",
        assistant_text="介绍了蓝牙连接和环绕音效。",
    )
    store.apply([extracted_turn(turn)], {1: turn})

    context = store.build_context(exclude_turn_ids=set(), language="zh")

    assert context is not None
    assert context.claim_count == 0
    assert context.episode_count == 1
    assert "介绍一下蓝牙音箱" in context.text
    assert "介绍了蓝牙连接和环绕音效" in context.text
    assert "assistant_artifact_summary" in context.text
    assert '"assistant_summary"' not in context.text


def test_fact_query_does_not_pull_unrelated_ordinary_episode() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    name_turn = memory_turn("turn-name", 1, user_text="我叫龙王")
    topic_turn = memory_turn("turn-topic", 2, user_text="今天天气怎么样")
    store.apply(
        [
            extracted_turn(
                name_turn,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            ),
            extracted_turn(topic_turn),
        ],
        {1: name_turn, 2: topic_turn},
    )

    fact_context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="我叫什么名字",
    )
    history_context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="我们之前聊到哪里了",
    )

    assert fact_context is not None
    assert fact_context.episode_count == 0
    assert "今天天气" not in fact_context.text
    assert history_context is not None
    assert history_context.episode_count == 1
    assert "今天天气" in history_context.text


def test_fact_query_does_not_pull_unrelated_generated_artifact() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    name_turn = memory_turn("turn-name", 1, user_text="我叫龙王")
    story_turn = memory_turn(
        "turn-story",
        2,
        user_text="讲一个故事",
        assistant_text="从前有一家咖啡馆。",
    )
    store.apply(
        [
            extracted_turn(
                name_turn,
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            ),
            extracted_turn(story_turn, artifact_kind="story"),
        ],
        {1: name_turn, 2: story_turn},
    )

    fact_context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="我叫什么名字",
    )
    story_context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="那个故事的结局是什么",
    )

    assert fact_context is not None
    assert fact_context.episode_count == 0
    assert "咖啡馆" not in fact_context.text
    assert story_context is not None
    assert story_context.episode_count == 0
    assert story_context.artifact_count == 1
    assert "咖啡馆" in story_context.text


def test_context_budget_trims_artifact_before_dropping_user_fact() -> None:
    config = SessionMemoryConfig(max_context_chars=760)
    store = SessionMemoryStore(config)
    name_turn = memory_turn("turn-name", 1, user_text="我叫范世德")
    story_turn = memory_turn(
        "turn-story",
        2,
        user_text="讲一个故事",
        assistant_text="从前有一家咖啡馆。" * 180,
    )
    store.apply(
        [
            extracted_turn(
                name_turn,
                add_claim(
                    predicate="self_reported_name",
                    value="范世德",
                    content="用户自述姓名是范世德",
                ),
            ),
            extracted_turn(story_turn, artifact_kind="story"),
        ],
        {1: name_turn, 2: story_turn},
    )

    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="再讲一遍那个故事，然后告诉我叫什么名字",
    )

    assert context is not None
    assert context.claim_count == 1
    assert context.artifact_count == 1
    assert "范世德" in context.text
    assert '"content_truncated":"true"' in context.text


def test_store_and_injected_context_are_bounded() -> None:
    config = SessionMemoryConfig(
        max_active_claims=2,
        max_injected_claims=1,
        max_context_chars=512,
    )
    store = SessionMemoryStore(config)
    turns = [memory_turn(f"turn-{seq}", seq) for seq in range(1, 4)]
    store.apply(
        [
            extracted_turn(
                turn,
                add_claim(
                    predicate=f"preference_{turn.turn_seq}",
                    value=f"值{turn.turn_seq}",
                    content=f"用户偏好值{turn.turn_seq}",
                    lifecycle="session",
                ),
            )
            for turn in turns
        ],
        {turn.turn_seq: turn for turn in turns},
    )

    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="我喜欢什么",
    )

    assert len(store.active_claims()) == 2
    assert context is not None
    assert context.claim_count == 1
    assert len(context.text) <= config.max_context_chars + 200
    assert "用户偏好值3" in context.text


def test_text_retrieval_ranks_old_relevant_claim_above_recent_claims() -> None:
    store = SessionMemoryStore(
        SessionMemoryConfig(max_active_claims=8, max_injected_claims=1)
    )
    turns = [memory_turn(f"turn-{seq}", seq) for seq in range(1, 6)]
    store.apply(
        [
            extracted_turn(
                turns[0],
                add_claim(
                    predicate="self_reported_name",
                    value="龙王",
                    content="用户自述姓名是龙王",
                ),
            ),
            *[
                extracted_turn(
                    turn,
                    add_claim(
                        predicate=f"preference_{turn.turn_seq}",
                        value=f"偏好{turn.turn_seq}",
                        content=f"用户偏好编号{turn.turn_seq}",
                        lifecycle="session",
                    ),
                )
                for turn in turns[1:]
            ],
        ],
        {turn.turn_seq: turn for turn in turns},
    )

    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="你知道我叫什么名字吗",
    )

    assert context is not None
    assert context.claim_count == 1
    assert context.retrieval_mode == "text_hybrid"
    assert "用户自述姓名是龙王" in context.text


def test_text_retrieval_ranks_relevant_old_artifact_above_recent_episodes() -> None:
    store = SessionMemoryStore(
        SessionMemoryConfig(
            max_injected_episodes=1,
            max_injected_artifacts=1,
        )
    )
    story = memory_turn(
        "turn-story",
        1,
        user_text="讲一个咖啡馆恋爱故事",
        assistant_text="两个人在咖啡馆相遇。",
    )
    weather = memory_turn(
        "turn-weather",
        2,
        user_text="介绍今天的天气",
        assistant_text="今天阳光很好。",
    )
    store.apply(
        [
            extracted_turn(story, artifact_kind="story"),
            extracted_turn(weather, artifact_kind="explanation"),
        ],
        {1: story, 2: weather},
    )

    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="继续之前那个咖啡馆恋爱故事",
    )

    assert context is not None
    assert context.episode_count == 0
    assert context.artifact_count == 1
    assert "咖啡馆相遇" in context.text
    assert "阳光很好" not in context.text


def test_artifact_uses_actual_reply_body_instead_of_extracted_summary() -> None:
    store = SessionMemoryStore(SessionMemoryConfig())
    story_body = (
        "从前，樱花小镇里有一家甜品店。"
        "店主每天都会给窗边的客人留一块草莓蛋糕。"
        "后来他们终于在雨天说出了彼此的心意。"
    )
    turn = memory_turn(
        "turn-story",
        1,
        user_text="给我讲一个恋爱故事",
        assistant_text=story_body,
    )
    extracted = ExtractedTurnMemory(
        turn_id=turn.turn_id,
        turn_seq=turn.turn_seq,
        user_summary="用户请求恋爱故事",
        assistant_summary="模型生成的不完整摘要",
        artifact_kind="story",
        operations=(),
    )

    stats = store.apply([extracted], {1: turn})
    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text="再讲一遍刚才那个故事",
    )

    assert stats.artifact_count == 1
    assert store.artifact_for_turn(turn.turn_id) is not None
    assert store.artifact_for_turn(turn.turn_id).content == story_body
    assert context is not None
    assert context.artifact_count == 1
    assert story_body in context.text
    assert "模型生成的不完整摘要" not in context.text


def test_audio_only_retrieval_includes_all_bounded_active_claims() -> None:
    store = SessionMemoryStore(
        SessionMemoryConfig(
            max_active_claims=4,
            max_injected_claims=1,
            max_context_chars=4096,
        )
    )
    turns = [memory_turn(f"turn-{seq}", seq) for seq in range(1, 4)]
    store.apply(
        [
            extracted_turn(
                turn,
                add_claim(
                    predicate=f"fact_{turn.turn_seq}",
                    value=f"值{turn.turn_seq}",
                    content=f"用户事实{turn.turn_seq}",
                    lifecycle="session",
                ),
            )
            for turn in turns
        ],
        {turn.turn_seq: turn for turn in turns},
    )

    context = store.build_context(
        exclude_turn_ids=set(),
        language="zh",
        current_text=None,
    )

    assert context is not None
    assert context.claim_count == 3
    assert context.retrieval_mode == "audio_bounded_all"
    assert all(f"用户事实{seq}" in context.text for seq in range(1, 4))


def test_store_prunes_old_inactive_claim_versions() -> None:
    store = SessionMemoryStore(
        SessionMemoryConfig(max_inactive_claims=1)
    )
    for seq, value in enumerate(("北京", "上海", "深圳"), 1):
        turn = memory_turn(f"turn-{seq}", seq)
        store.apply(
            [
                extracted_turn(
                    turn,
                    add_claim(
                        predicate="current_city",
                        value=value,
                        content=f"用户自述当前城市为{value}",
                    ),
                )
            ],
            {seq: turn},
        )

    assert len(store.active_claims()) == 1
    assert store.active_claims()[0].value == "深圳"
    assert len(store.claims) == 2


def test_memory_request_keeps_audio_order_and_uses_fixed_task() -> None:
    config = SessionMemoryConfig()
    first = memory_turn("turn-1", 1, audios=("audio-1", "audio-2"))
    second = memory_turn("turn-2", 2, audios=("audio-3",))

    request = build_memory_extraction_request(
        model_name="model",
        session_id="session",
        session_instance_id="instance",
        language="zh",
        turns=[first, second],
        active_claims=[],
        config=config,
    )

    assert request.stream is False
    assert request.metadata["task"] == "session_memory_extract"
    assert request.metadata["audios"] == ["audio-1", "audio-2", "audio-3"]
    assert request.metadata["turn_seqs"] == [1, 2]
    assert request.metadata["base_store_revision"] == 0
    assert "evidence" in request.messages[0].content
    assert "confidence" in request.messages[0].content
    assert sum(
        part.get("type") == "audio" for part in request.messages[-1].content
    ) == 3


def test_memory_request_bounds_dynamic_text_inputs() -> None:
    config = SessionMemoryConfig(max_input_text_chars=4)
    turn = memory_turn(
        "turn-1",
        1,
        user_text="abcdefgh",
        assistant_text="12345678",
    )

    request = build_memory_extraction_request(
        model_name="model",
        session_id="session",
        session_instance_id="instance",
        language="zh",
        turns=[turn],
        active_claims=[],
        config=config,
    )
    text_parts = [
        part["text"]
        for part in request.messages[-1].content
        if part.get("type") == "text"
    ]

    assert 'user_text="abcd"' in text_parts
    assert not any(part.startswith("assistant_reply=") for part in text_parts)
    assert request.metadata["workload_priority"] == "background"


@pytest.mark.asyncio
async def test_scheduler_runs_globally_serial_and_does_not_lose_dirty_session() -> None:
    scheduler = SessionMemoryScheduler()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls: list[str] = []
    concurrent = 0
    max_concurrent = 0

    async def first_job() -> bool:
        nonlocal concurrent, max_concurrent
        calls.append("first")
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        if calls.count("first") == 1:
            first_started.set()
            await release_first.wait()
        concurrent -= 1
        return False

    async def second_job() -> bool:
        nonlocal concurrent, max_concurrent
        calls.append("second")
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return False

    assert scheduler.submit("session-1", first_job)
    await first_started.wait()
    # Notification while session-1 is active must cause one later pass rather
    # than creating another concurrent task or being lost.
    assert not scheduler.submit("session-1", first_job)
    assert scheduler.submit("session-2", second_job)
    release_first.set()
    await scheduler.wait_idle()

    assert calls == ["first", "second", "first"]
    assert max_concurrent == 1


@pytest.mark.asyncio
async def test_scheduler_cancel_removes_queued_session() -> None:
    scheduler = SessionMemoryScheduler()
    first_started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def blocking() -> bool:
        calls.append("blocking")
        first_started.set()
        await release.wait()
        return False

    async def queued() -> bool:
        calls.append("queued")
        return False

    scheduler.submit("first", blocking)
    await first_started.wait()
    scheduler.submit("second", queued)
    await scheduler.cancel("second")
    release.set()
    await scheduler.wait_idle()

    assert calls == ["blocking"]


@pytest.mark.asyncio
async def test_scheduler_cancel_does_not_resubmit_dirty_running_session() -> None:
    scheduler = SessionMemoryScheduler()
    started = asyncio.Event()
    calls = 0

    async def blocking() -> bool:
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.Event().wait()
        return True

    assert scheduler.submit("session", blocking)
    await started.wait()
    assert not scheduler.submit("session", blocking)
    await scheduler.cancel("session")
    await scheduler.wait_idle()

    assert calls == 1


@pytest.mark.asyncio
async def test_scheduler_can_prioritize_queued_r1_session_without_parallelism() -> None:
    scheduler = SessionMemoryScheduler()
    first_started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def first() -> bool:
        calls.append("first")
        first_started.set()
        await release.wait()
        return False

    def immediate(name: str):
        async def run() -> bool:
            calls.append(name)
            return False

        return run

    scheduler.submit("first", first)
    await first_started.wait()
    scheduler.submit("second", immediate("second"))
    scheduler.submit("r1", immediate("r1"))
    assert scheduler.prioritize("r1") is True
    release.set()
    await scheduler.wait_idle()

    assert calls == ["first", "r1", "second"]


@pytest.mark.asyncio
async def test_scheduler_rejects_new_session_when_global_queue_is_full() -> None:
    scheduler = SessionMemoryScheduler(max_queued_sessions=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking() -> bool:
        started.set()
        await release.wait()
        return False

    async def immediate() -> bool:
        return False

    assert scheduler.submit_status("running", blocking) == "submitted"
    await started.wait()
    assert scheduler.submit_status("queued", immediate) == "submitted"
    assert scheduler.submit_status("rejected", immediate) == "rejected"
    assert scheduler.snapshot()["rejected_submission_count"] == 1
    release.set()
    await scheduler.wait_idle()


@pytest.mark.asyncio
async def test_scheduler_honors_configurable_global_concurrency_limit() -> None:
    scheduler = SessionMemoryScheduler(max_concurrent_extractions=2)
    both_started = asyncio.Event()
    release = asyncio.Event()
    concurrent = 0
    max_concurrent = 0

    async def blocking() -> bool:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        if concurrent == 2:
            both_started.set()
        await release.wait()
        concurrent -= 1
        return False

    assert scheduler.submit("first", blocking)
    assert scheduler.submit("second", blocking)
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert scheduler.snapshot()["running_job_count"] == 2
    release.set()
    await scheduler.wait_idle()
    assert max_concurrent == 2


@pytest.mark.asyncio
async def test_scheduler_many_sessions_stays_bounded_and_completes_fairly() -> None:
    scheduler = SessionMemoryScheduler(
        max_queued_sessions=32,
        max_concurrent_extractions=2,
    )
    concurrent = 0
    max_concurrent = 0
    completed: list[int] = []

    def job(index: int):
        async def run() -> bool:
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0)
            completed.append(index)
            concurrent -= 1
            return False

        return run

    for index in range(20):
        assert scheduler.submit(f"session-{index}", job(index))

    await scheduler.wait_idle()

    assert sorted(completed) == list(range(20))
    assert max_concurrent == 2
    assert scheduler.snapshot()["queued_session_count"] == 0
    assert scheduler.snapshot()["running_job_count"] == 0
