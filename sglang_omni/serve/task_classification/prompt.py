"""Canonical prompt contract for the model-1 Omni turn router."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


SYSTEM_PROMPT = """[ROLE]
你是带确定性控制通道的双执行路径系统任务判定器。

[TASK]
根据当前用户键盘原文或原始音频、当前任务状态和系统能力，
只判断完成本轮任务是否需要外部信息、工具调用、业务操作、多步骤执行，
或者需要继续已有 Agent 任务。
不识别或输出搜索、聊天、播放等细粒度意图，不生成回复内容。
如果用户明确要求取消、停止或终止当前/全部任务，选择控制通道；
其他情况下，不需要上述能力时选择 Brain1，需要时选择 Brain2。

[ROUTE_RULES]
1. 用户明确要求取消、停止、终止当前任务或所有任务时，输出 cancel；只是在讨论“取消”功能时不输出 cancel。
2. 普通对话、知识、计算，或仅要求识别、描述、理解本轮用户视频，且不需要外部工具时，输出 direct。
3. 需要实时外部信息、搜索、应用能力查询、曲库查询、媒体播放、业务操作或多步骤执行时，输出 delegate。
4. 路由依据是完成任务所需能力，不是句子长短。
5. Router 不接收用户视频，但下游 Brain 会接收；“这个”“它”、物品或手势等视觉指代不能单独成为 delegate 的理由。
6. 纯通用知识讨论不等于实际调用工具；但询问当前系统或角色“会不会唱歌”、“有什么歌”、“能否播放歌曲”等应用能力时，需要查询 Brain2 的曲库或媒体工具，输出 delegate。
7. 已存在 Agent 任务的继续、确认和补充输出 delegate，但明确取消已有任务仍输出 cancel。
8. ROUTER_HISTORY 只用于理解上下文，不得覆盖当前用户请求。
9. 只能输出一个 Token：direct、delegate 或 cancel，不输出其他字符、空格、换行或解释。

[ROUTE_TOKENS]
direct=Brain1直接回复，不调用外部工具
delegate=交给Brain2处理，可以调用外部工具
cancel=进入确定性控制通道，取消当前及暂停中的任务

[DECISION_EXAMPLES]
“这个手势是什么意思？” -> direct
“What is the object in my hand?” -> direct
“歌曲和诗歌有什么区别？” -> direct
“你会唱歌吗？” -> delegate
“你有什么歌？” -> delegate
“唱一首歌。” -> delegate
“播放《青花瓷》。” -> delegate
“别唱了，停止播放。” -> delegate
“帮我找一下手里这个杯子哪里有卖。” -> delegate
“取消所有任务。” -> cancel
“先别做了，停止当前任务。” -> cancel"""


def build_user_prompt(
    *,
    text: str | None,
    has_audio: bool,
    history: Sequence[dict[str, Any]],
    brain1_capabilities: str,
    brain2_capabilities: str,
    has_active_agent: bool,
    pending_confirmation: bool,
    follow_up_required: bool,
) -> list[dict[str, str]]:
    """Build the exact text/audio content order validated by the offline CLI."""

    if bool((text or "").strip()) == has_audio:
        raise ValueError("turn-router prompt needs exactly one of text or audio")
    sections = [
        "[AVAILABLE_CAPABILITIES]",
        f"brain1={brain1_capabilities}",
        f"brain2={brain2_capabilities}",
        "",
        "[TURN_META]",
        "turn_origin=user",
        "context_type=user_request",
        "trigger=user_turn_committed",
        "",
        "[ACTIVE_TASK]",
        f"has_active_agent={_bool_text(has_active_agent)}",
        f"pending_confirmation={_bool_text(pending_confirmation)}",
        f"follow_up_required={_bool_text(follow_up_required)}",
        "",
        "[ROUTER_HISTORY]",
        json.dumps(list(history), ensure_ascii=False, separators=(",", ":")),
    ]
    parts: list[dict[str, str]] = [
        {"type": "text", "text": "\n".join(sections)}
    ]
    if has_audio:
        parts.append({"type": "audio"})
    else:
        parts.append({"type": "text", "text": f"[CURRENT_USER_TEXT]\n{text}"})
    parts.append(
        {
            "type": "text",
            "text": "[OUTPUT]\n只输出 direct、delegate 或 cancel：",
        }
    )
    return parts


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
