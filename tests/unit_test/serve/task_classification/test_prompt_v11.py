from sglang_omni.serve.task_classification.prompt_v11 import (
    DEFAULT_MCP_SCOPE,
    SINGLE_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def _text_prompt(text: str, *, history=(), **state) -> str:
    parts = build_user_prompt(
        query_mode="text",
        text=text,
        history=history,
        mcp_scope=DEFAULT_MCP_SCOPE,
        multimodal_scope="用户当前摄像头画面",
        **state,
    )
    return "\n".join(part["text"] for part in parts)


def test_v11_uses_explicit_new_continue_and_side_reply_codes() -> None:
    assert SYSTEM_PROMPT == SINGLE_TURN_SYSTEM_PROMPT
    assert "[ROUTER_VERSION]\nv1.1" in SYSTEM_PROMPT
    assert "**当前用户提出与活动任务不同的完整 MCP 请求" in SYSTEM_PROMPT
    assert "0=普通回复" in SYSTEM_PROMPT
    assert "1=创建独立的新 MCP 任务" in SYSTEM_PROMPT
    assert "4=继续、补充、确认、停止或变更当前活动 MCP 任务" in SYSTEM_PROMPT
    assert "**只能输出一个字符：0、1、2、3、4 或 5。**" in SYSTEM_PROMPT


def test_v11_prompt_carries_active_goal_status_and_spoken_history() -> None:
    prompt = _text_prompt(
        "Yes, please.",
        history=(
            {
                "role": "assistant",
                "text": "Would you like me to sing another song?",
                "playback_status": "completed",
            },
        ),
        has_active_agent=True,
        pending_confirmation=True,
        follow_up_required=True,
        active_task_goal="继续唱歌",
        active_task_status="waiting_user",
    )

    assert "存在=是" in prompt
    assert "状态=waiting_user" in prompt
    assert '目标="继续唱歌"' in prompt
    assert "等待确认=是" in prompt
    assert "等待补充=是" in prompt
    assert '文本="Would you like me to sing another song?"' in prompt
    assert "[当前用户文本]\nYes, please." in prompt
    assert prompt.endswith("只输出 0、1、2、3、4 或 5：")


def test_v11_audio_uses_raw_audio_without_asr() -> None:
    parts = build_user_prompt(
        query_mode="audio",
        text=None,
        history=(),
        mcp_scope=DEFAULT_MCP_SCOPE,
        multimodal_scope="用户当前摄像头画面",
        has_active_agent=False,
    )

    assert parts[1] == {"type": "audio"}
    prompt = str(parts)
    assert "不要附加、猜测或使用 ASR 文本" in prompt
    assert "当前用户文本" not in prompt
