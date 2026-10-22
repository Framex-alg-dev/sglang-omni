from sglang_omni.serve.task_classification.prompt_v10 import (
    DEFAULT_MCP_SCOPE,
    SINGLE_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
    normalize_history,
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


def test_v10_uses_one_natural_language_prefix_with_strong_rules() -> None:
    assert SYSTEM_PROMPT == SINGLE_TURN_SYSTEM_PROMPT
    assert "[ROUTER_VERSION]\nv1.0" in SYSTEM_PROMPT
    assert "**上一条数字人明确询问或提议唱歌、搜索或查交通" in SYSTEM_PROMPT
    assert "**活动任务状态只是判断是否继续旧任务的证据" in SYSTEM_PROMPT
    assert "**当前存在活动 MCP 任务" in SYSTEM_PROMPT
    assert "**如果上一条数字人话语已经明确给出" in SYSTEM_PROMPT
    assert "**只能输出一个字符：0、1、2 或 3。**" in SYSTEM_PROMPT


def test_history_keeps_latest_assistant_utterance_and_its_ending() -> None:
    long_reply = "前文" * 500 + "Would you like me to sing another song?"
    history = normalize_history(
        (
            {"role": "assistant", "text": "旧回复"},
            {
                "role": "assistant",
                "text": long_reply,
                "playback_status": "completed",
            },
        )
    )

    assert len(history) == 1
    assert history[0]["text"].startswith("…")
    assert history[0]["text"].endswith(
        "Would you like me to sing another song?"
    )
    assert len(history[0]["text"]) == 800


def test_prompt_exposes_full_previous_question_without_active_task_gate() -> None:
    prompt = _text_prompt(
        "Yes, please.",
        history=(
            {
                "role": "assistant",
                "text": "Would you like me to sing another song?",
                "playback_status": "completed",
            },
        ),
        has_active_agent=False,
    )

    assert "文本=\"Would you like me to sing another song?\"" in prompt
    assert "存在活动 MCP 任务=否" in prompt
    assert "它不是理解上一条数字人邀请的前提" in prompt
    assert "存在活动 MCP 任务且用户明确停止、取消或变更它时" in prompt
    assert "当前指示词能在上一条数字人话语中找到明确实体时" in prompt


def test_prompt_marks_isolated_confirmation_as_having_no_history() -> None:
    prompt = _text_prompt("Yes, please.")

    assert "[上一条数字人实际说给用户的话]\n无" in prompt
    assert "[当前用户文本]\nYes, please." in prompt


def test_prompt_carries_real_active_task_flags() -> None:
    prompt = _text_prompt(
        "继续",
        has_active_agent=True,
        pending_confirmation=True,
        follow_up_required=True,
    )

    assert "存在活动 MCP 任务=是" in prompt
    assert "正在等待用户确认=是" in prompt
    assert "正在等待用户补充=是" in prompt


def test_audio_prompt_contains_raw_audio_without_asr_text() -> None:
    parts = build_user_prompt(
        query_mode="audio",
        text=None,
        history=(),
        mcp_scope=DEFAULT_MCP_SCOPE,
        multimodal_scope="用户当前摄像头画面",
    )

    assert parts[1] == {"type": "audio"}
    prompt = str(parts)
    assert "不要附加、猜测或使用 ASR 文本" in prompt
    assert "当前用户文本" not in prompt
