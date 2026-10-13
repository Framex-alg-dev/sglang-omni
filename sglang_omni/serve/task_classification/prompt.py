"""Compact Router v0.8 prompt contract used by the 41001 capability router."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


DEFAULT_HISTORY_LIMIT = 1
DEFAULT_HISTORY_MAX_CHARS = 50
DEFAULT_HISTORY_ENTRY_MAX_CHARS = 50

DEFAULT_MCP_SCOPE = "唱歌、搜索（含天气）、查交通"
DEFAULT_MULTIMODAL_SCOPE = "用户当前摄像头画面中的人物、物体、文字和场景理解"


_BASE_SYSTEM_PROMPT = """[ROUTER_VERSION]
v0.8

[ROLE]
你是能力路由器。不要回答用户，只输出一个路由码。

[MCP_RULE]
MCP 是严格白名单，只包含：
1. 唱歌：实际演唱、选歌、播放歌曲、查询可唱内容或唱歌能力。
2. 搜索：搜索公开信息；天气和其他需要检索的当前公开信息属于搜索。
3. 查交通：查询路况、路线、通勤、公交地铁、步行或驾车方案。

当前输入要求系统执行、交付、重复、变更、细化或继续上述白名单操作时，MCP 位为 1。完整的当前请求独立成立，不要求历史中已有对应任务。
仅提及、描述、讨论、评价、吐槽或回应某项能力及其已经完成的结果，并且没有要求新的执行或交付时，MCP 位为 0。
不得仅凭对既有结果的正面或负面态度，推断用户要求重新执行、变更或细化操作；新的执行或交付要求必须由当前输入明确表达或形成语义完整的请求。
如果普通对话回复即可完整承接当前意图，不需要产生新的查询、播放或业务动作，MCP 位为 0。
历史只可补全当前输入中被省略的操作类型或交付目标，不能凭历史把本轮的评价或闲聊变成操作请求。
提醒、日历、票务预订、支付、订单、消息、文件、账号、设备控制及其他未开放能力，MCP 位均为 0。

[MULTIMODAL_RULE]
只有完成请求必须理解用户当前摄像头画面时，多模态位才为 1。用户键盘文本或用户说话的原始音频只是请求输入，不算多模态。要求系统唱歌或播放是输出目标，也不算理解用户摄像头画面。
先检查完成请求所需的实体、属性和约束是否已由当前文本或历史完整给出。如果目标使用当前画面中的现场指代，导致必要参数必须从用户当前摄像头画面取得，多模态位为 1。
如果当前输入中的目标实体无法由文本和历史唯一解析，必须查看用户当前摄像头画面才能确定该实体，多模态位为 1。
MCP 位和多模态位必须独立判断，再组合成最终路由码。从用户当前摄像头画面取得参数之后还要执行白名单操作时，两位都为 1；目标是否需要从摄像头画面取得不能改变对 MCP 位的判断，一位为 1 不得遮蔽另一位。

[OUTPUT_CODES]
0=不需要MCP，不需要多模态
1=需要MCP，不需要多模态
2=不需要MCP，需要多模态
3=需要MCP，需要多模态

[PRIORITY]
先独立判断当前输入是否要求执行白名单操作，再独立判断完成请求是否必须理解用户当前摄像头画面，最后组合两位。只能输出一个字符：0、1、2 或 3，不输出空格、换行、JSON 或解释。"""


SINGLE_TURN_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + """

[SINGLE_TURN]
当前没有可继承的历史任务。当前输入本身已明确要求执行白名单操作时直接设置 MCP 位；只有无法独立确定操作类型或交付目标的省略表达不能继承操作。"""


SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + """

[MULTI_TURN_RESOLUTION]
ROUTER_HISTORY 只包含紧邻当前输入之前的一条数字人可听回复，仅用于还原当前输入中的省略表达。当前用户提出完整新请求时，以新请求为准。
只有上一条回复能明确补全当前省略的操作类型或交付目标，并且补全后属于唱歌、搜索或查交通，当前输入才继承 MCP 位；否则不继承。历史不能凭自身创建操作请求。"""


def normalize_history(
    value: Sequence[dict[str, Any]],
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
    max_chars: int = DEFAULT_HISTORY_MAX_CHARS,
    current_turn_id: str | None = None,
) -> list[dict[str, str]]:
    """Keep only the last prior assistant text the user could have heard."""

    if limit < 0 or max_chars <= 0:
        raise ValueError("history limits must be non-negative and positive")
    if limit == 0:
        return []

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"history entry {index} must be a JSON object")
        if item.get("role") != "assistant":
            continue
        if current_turn_id and item.get("turn_id") == current_turn_id:
            continue
        raw_text = item.get("text", item.get("content"))
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        playback_status = item.get("playback_status", "completed")
        if playback_status not in {"completed", "interrupted"}:
            continue
        normalized.append(
            {
                "role": "assistant",
                "text": raw_text.strip()[:DEFAULT_HISTORY_ENTRY_MAX_CHARS],
                "playback_status": playback_status,
            }
        )

    selected: list[dict[str, str]] = []
    selected_chars = 0
    for item in reversed(normalized):
        if len(selected) >= limit:
            break
        next_chars = selected_chars + len(item["text"])
        if selected and next_chars > max_chars:
            break
        if next_chars > max_chars:
            continue
        selected.append(item)
        selected_chars = next_chars
    selected.reverse()
    return selected


def build_user_prompt(
    *,
    query_mode: str,
    text: str | None,
    history: Sequence[dict[str, Any]],
    mcp_scope: str,
    multimodal_scope: str,
    has_active_agent: bool = False,
    pending_confirmation: bool = False,
    follow_up_required: bool = False,
) -> list[dict[str, str]]:
    if query_mode not in {"text", "audio"}:
        raise ValueError(f"unsupported query_mode: {query_mode}")
    if query_mode == "text" and not (text or "").strip():
        raise ValueError("text query requires non-empty text")
    if query_mode == "audio" and text is not None:
        raise ValueError("audio query must not include ASR or other derived text")

    compact_history = normalize_history(history)
    sections = [
        "[CAPABILITY_SCOPE]",
        f"mcp={mcp_scope}",
        f"multimodal={multimodal_scope}",
        "",
        "[ROUTER_HISTORY]",
        json.dumps(compact_history, ensure_ascii=False, separators=(",", ":")),
    ]
    parts: list[dict[str, str]] = [
        {"type": "text", "text": "\n".join(sections)}
    ]
    if query_mode == "audio":
        parts.append({"type": "audio"})
        parts.append(
            {
                "type": "text",
                "text": (
                    "[CURRENT_USER_AUDIO]\n"
                    "理解上方用户原始语音的请求语义；不要附加或假设 ASR 文本。"
                    "必须按语音的完整语义和言语行为判断，区分要求执行或交付与陈述或评价既有结果；"
                    "不得仅因语音提到白名单能力就设置 MCP 位。"
                ),
            }
        )
    else:
        parts.append({"type": "text", "text": f"[CURRENT_USER_TEXT]\n{text}"})

    history_rule = (
        "上一条数字人回复只用于补全当前省略的操作类型或交付目标；补全后明确要求执行唱歌、搜索或查交通时才继承 MCP。"
        if compact_history
        else "没有可继承的数字人回复；当前输入本身明确要求执行白名单操作时直接判断，无法独立确定操作类型或交付目标的省略表达不继承操作。"
    )
    parts.append(
        {
            "type": "text",
            "text": (
                "[FINAL_RULE]\n"
                "只有本轮创建或继续一个需要执行的操作，且该操作必须使用白名单能力时，MCP 位才为 1；历史不能自行创建操作。"
                "必要参数只能从用户当前摄像头画面取得时，多模态位为 1。两位独立判断再组合。\n"
                + history_rule
            ),
        }
    )
    parts.append({"type": "text", "text": "[OUTPUT]\n只输出 0、1、2 或 3："})
    return parts
