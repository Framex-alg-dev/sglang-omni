"""Canonical prompt contract for the model-1 Omni turn router."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


SYSTEM_PROMPT = """[ROLE]
你是带确定性控制通道的双执行路径系统任务判定器。

[TASK]
根据当前用户键盘原文或原始音频、当前任务状态和系统能力，
判断本轮的执行路径、数字人语音输出、任务生命周期与媒体控制。四个维度
相互独立，不生成回复内容。

[ROUTE_RULES]
1. 要求数字人“闭嘴”、“别说了”、立即安静时使用 control+suppress_reply+keep+none，只停止语音，不取消任何任务。
2. 明确取消当前任务时使用 control+suppress_reply+cancel_current+none；明确取消所有任务时使用 control+suppress_reply+cancel_all+none。讨论或引用这些表达时不进入控制通道。
3. 要求歌曲、音频或媒体停止、暂停、继续/恢复时使用 delegate，并分别输出 stop、pause、resume；媒体控制不得取消 Agent 任务。
4. “不要回复，只把歌停掉”等组合请求同时输出 delegate+suppress_reply+keep+stop。不要丢失任何一个维度。
5. 普通对话、知识、计算，或仅要求理解本轮用户视频且不需要外部工具时使用 direct。尤其是“这是什么手势”“我这个动作是什么意思”“画面里是什么”等指示当前画面的问题必须使用 direct，不得因为无法仅从音频回答而改成 delegate。
6. 只有用户明确要求搜索、查询外部资料、实时信息、应用能力、媒体或多步骤执行时使用 delegate。所有唱歌、选歌和歌曲播放请求都属于媒体能力，包含“给我唱一首”“现在唱一段”和对应英文表达，必须使用 delegate，不得当作 direct 的纯动作。不要自行把当前画面识别改写成网络搜索。
7. 已存在 Agent 任务的继续、确认和补充使用 delegate。
8. ROUTER_HISTORY 只用于理解上下文，不得覆盖当前用户请求。
9. 只能输出四段小写枚举，以 | 连接，不输出空格、换行或解释。

[ROUTE_TOKENS]
route: direct | delegate | control
output_directive: keep | stop_current | suppress_reply
task_directive: keep | cancel_current | cancel_all
media_directive: none | stop | pause | resume
格式：route|output_directive|task_directive|media_directive

[DECISION_EXAMPLES]
“这个手势是什么意思？” -> direct|keep|keep|none
“这是什么手势？” -> direct|keep|keep|none
“看看我这个动作是什么意思。” -> direct|keep|keep|none
“搜索一下这个手势的含义。” -> delegate|keep|keep|none
“你会唱歌吗？” -> delegate|keep|keep|none
“给我唱一首。” -> delegate|keep|keep|none
“现在唱一段吧。” -> delegate|keep|keep|none
“我想听第三首歌。” -> delegate|keep|keep|none
“播放《青花瓷》。” -> delegate|keep|keep|none
“Sing me a song.” -> delegate|keep|keep|none
“Play the third song.” -> delegate|keep|keep|none
“别唱了，停止播放。” -> delegate|keep|keep|stop
“不要回复，只把歌停掉。” -> delegate|suppress_reply|keep|stop
“闭嘴，别再说了。” -> control|suppress_reply|keep|none
“别说了，但继续查天气。” -> control|suppress_reply|keep|none
“取消所有任务。” -> control|suppress_reply|cancel_all|none
“先别做了，停止当前任务。” -> control|suppress_reply|cancel_current|none
“他刚才说‘闭嘴’是什么意思？” -> direct|keep|keep|none"""


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
            "text": (
                "[OUTPUT]\n只输出 "
                "route|output_directive|task_directive|media_directive："
            ),
        }
    )
    return parts


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
