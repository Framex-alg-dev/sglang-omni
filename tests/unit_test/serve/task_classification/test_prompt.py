from sglang_omni.serve.task_classification.prompt_v09 import (
    DEFAULT_MCP_SCOPE,
    SINGLE_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
    normalize_history,
)


def test_v09_prompt_uses_rule_only_capability_scope() -> None:
    assert "[DEFAULT_RULE]" in SINGLE_TURN_SYSTEM_PROMPT
    assert "[MCP_RULE]" in SINGLE_TURN_SYSTEM_PROMPT
    assert "[MULTIMODAL_RULE]" in SINGLE_TURN_SYSTEM_PROMPT
    assert "[MULTI_TURN_RESOLUTION]" in SYSTEM_PROMPT
    assert DEFAULT_MCP_SCOPE == "唱歌、搜索（含天气）、查交通"
    assert "CAPABILITY_POLICY" not in SINGLE_TURN_SYSTEM_PROMPT
    assert "[EXAMPLES]" not in SINGLE_TURN_SYSTEM_PROMPT
    assert "用户当前摄像头画面" in SINGLE_TURN_SYSTEM_PROMPT
    assert "当前媒体" not in SINGLE_TURN_SYSTEM_PROMPT
    assert "现场媒体" not in SINGLE_TURN_SYSTEM_PROMPT
    assert "执行、交付、重复、变更、细化或继续" in SINGLE_TURN_SYSTEM_PROMPT
    assert "完整言语行为向系统发出当前有效的邀约或期待" in SINGLE_TURN_SYSTEM_PROMPT
    assert "把系统作为白名单动作的执行者" in SINGLE_TURN_SYSTEM_PROMPT
    assert "近期或长期未获得、很久未获得或几乎忘记" in SINGLE_TURN_SYSTEM_PROMPT
    assert "过去或假设的愿望" in SINGLE_TURN_SYSTEM_PROMPT
    assert "以当前言语行为是否期待系统产生新交付为准" in SINGLE_TURN_SYSTEM_PROMPT
    assert "不表达当前未满足的交付期待" in SINGLE_TURN_SYSTEM_PROMPT
    assert "询问概念、技巧或原理" in SINGLE_TURN_SYSTEM_PROMPT
    assert "不得仅凭能力词" in SINGLE_TURN_SYSTEM_PROMPT
    assert "评价、吐槽或回应" in SINGLE_TURN_SYSTEM_PROMPT
    assert "无法独立确定操作类型或交付目标" in SINGLE_TURN_SYSTEM_PROMPT
    assert "MCP 是例外路径" in SINGLE_TURN_SYSTEM_PROMPT
    assert "证据不足时必须为 0" in SINGLE_TURN_SYSTEM_PROMPT
    assert "社交性对话、寒暄、问候" in SINGLE_TURN_SYSTEM_PROMPT
    assert "短句不能仅因语义不完整" in SINGLE_TURN_SYSTEM_PROMPT


def test_prompt_requires_exact_single_character_output() -> None:
    assert "只能输出一个字符：0、1、2 或 3" in SINGLE_TURN_SYSTEM_PROMPT
    parts = build_user_prompt(
        query_mode="text",
        text="你会唱歌吗",
        history=(),
        mcp_scope="MCP",
        multimodal_scope="MM",
    )
    assert parts[-1]["text"] == "[OUTPUT]\n只输出 0、1、2 或 3："
    assert "[CAPABILITY_POLICY]" not in str(parts)
    assert "mcp=MCP" in str(parts)
    assert "应输出 3 而不是 2" in str(parts)
    assert "明确限定在过去或假设的愿望" in str(parts)
    assert "仅评价能力或功能本身" in str(parts)
    assert "MULTI_TURN_STATE_MACHINE" not in str(parts)
    assert "TRANSITION_TABLE" not in str(parts)


def test_interrupted_history_keeps_only_confirmed_prefix_and_status() -> None:
    history = normalize_history((
        {
            "role": "assistant",
            "text": "用户实际听到的前缀",
            "playback_status": "interrupted",
        },
    ))

    assert history == [{
        "role": "assistant",
        "text": "用户实际听到的前缀",
        "playback_status": "interrupted",
    }]


def test_prompt_excludes_runtime_task_state() -> None:
    parts = build_user_prompt(
        query_mode="text",
        text="继续",
        history=(),
        mcp_scope="MCP",
        multimodal_scope="MM",
        has_active_agent=True,
        pending_confirmation=True,
        follow_up_required=True,
    )

    prompt = str(parts)
    assert "ACTIVE_TASK" not in prompt
    assert "has_active_agent" not in prompt
    assert "pending_confirmation" not in prompt
    assert "follow_up_required" not in prompt


def test_audio_prompt_uses_speech_act_semantics_without_asr() -> None:
    parts = build_user_prompt(
        query_mode="audio",
        text=None,
        history=(),
        mcp_scope="MCP",
        multimodal_scope="MM",
    )

    prompt = str(parts)
    assert "完整语义和言语行为" in prompt
    assert "陈述或评价既有结果" in prompt
    assert "评价、态度或吐槽" in prompt
    assert "不得把正面、负面或玩笑语气" in prompt
    assert "ASR 文本" in prompt
    assert "CURRENT_USER_TEXT" not in prompt
