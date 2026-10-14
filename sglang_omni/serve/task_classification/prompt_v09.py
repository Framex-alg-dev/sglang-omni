"""Immutable Router v0.9 prompt contract used by the 41001 capability router."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .prompt import normalize_history


DEFAULT_MCP_SCOPE = "唱歌、搜索（含天气）、查交通"
DEFAULT_MULTIMODAL_SCOPE = "用户当前摄像头画面中的人物、物体、文字和场景理解"


_BASE_SYSTEM_PROMPT = """[ROUTER_VERSION]
v0.9

[ROLE]
你是能力路由器。不要回答用户，只输出一个路由码。

[DEFAULT_RULE]
MCP 是例外路径。只有能够从当前输入，或允许使用的上一条数字人回复中，识别出明确的白名单能力以及当前执行或交付意图时，MCP 位才为 1；证据不足时必须为 0。
社交性对话、寒暄、问候、告别、感谢、道歉、确认和情绪表达，如果不包含唱歌、搜索或查交通的当前执行意图，MCP 位必须为 0。短句不能仅因语义不完整而推断为 MCP 请求。

[MCP_RULE]
MCP 是严格白名单，只包含：
1. 唱歌：实际演唱、选歌、播放歌曲、查询可唱内容或唱歌能力。
2. 搜索：搜索公开信息；天气和其他需要检索的当前公开信息属于搜索。
3. 查交通：查询路况、路线、通勤、公交地铁、步行或驾车方案。

先判断当前输入是否要求系统产生一次新的白名单交付，而不是只检查是否出现能力词。按以下顺序判断：
A. 当前输入显式要求系统执行、交付、重复、变更、细化或继续白名单操作时，MCP 位为 1。完整请求独立成立，不要求历史中已有对应任务。
B. 没有显式请求词，但完整言语行为向系统发出当前有效的邀约或期待，并且把系统作为白名单动作的执行者、把用户作为新交付的接受者，自然承接必须立即产生一次白名单交付时，属于隐含请求，MCP 位为 1。命令句、问句和陈述句采用同一标准。用户面向系统陈述自己近期或长期未获得、很久未获得或几乎忘记系统可提供的白名单交付，属于表达当前期待的常规间接请求；明确限定为过去、假设或不希望执行时除外。
C. 仅提及、描述或讨论能力；询问概念、技巧或原理；评价、吐槽或回应已经完成的结果；描述用户自己或第三方的行为、过去或假设的愿望、个人经历或外部状态，MCP 位为 0。除非当前输入还有独立满足 A 或 B 的请求。
当 B 与 C 看似冲突时，以当前言语行为是否期待系统产生新交付为准：面向系统表达长期未获得其可提供的白名单服务，且没有否定当前执行，属于当前间接请求；只有已完结的过去行为或愿望，且不表达当前未满足的交付期待时，才按过去经历处理。
查询系统自身是否具备白名单能力或可提供内容，属于白名单请求。不得仅凭能力词或对既有结果的正面、负面态度猜测新的执行意图。
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
先独立判断当前输入是否要求执行白名单操作，再独立判断完成请求是否必须理解用户当前摄像头画面，最后组合两位。当前输入明确要求执行唱歌、搜索或查交通时，MCP 位必须为 1；必要参数来自用户当前摄像头画面时只额外把多模态位置为 1。只能输出一个字符：0、1、2 或 3，不输出空格、换行、JSON 或解释。"""


SINGLE_TURN_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + """

[SINGLE_TURN]
当前没有可继承的回复文本。当前输入本身已明确要求执行白名单操作时直接设置 MCP 位。
若 has_active_agent=true 且 pending_confirmation 或 follow_up_required 为 true，孤立的确认、继续或补充表达可继续当前 Agent/MCP 任务，MCP 位为 1；has_active_agent=false 时不得继承。
除此之外，无法独立确定操作类型或交付目标的省略表达不能继承操作。"""


SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + """

[MULTI_TURN_RESOLUTION]
ROUTER_HISTORY 只包含紧邻当前输入之前的一条数字人可听回复，仅用于还原当前输入中的省略表达。当前用户提出完整新请求时，以新请求为准。
has_active_agent=false 表示当前没有正在执行或等待用户确认的 Agent/MCP 任务。此时“继续、go on、continue、接着”等孤立续讲表达属于普通对话续讲，MCP 位必须为 0；不得因为故事、教学或闲聊历史偶然提到天气、地图、歌曲、搜索等词而继承 MCP。
只有 has_active_agent=true，且上一条回复能明确补全当前省略的操作类型或交付目标，并且补全后属于唱歌、搜索或查交通，当前输入才继承 MCP 位；否则不继承。历史和布尔状态都不能单独创建操作请求。"""


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
        "[ACTIVE_AGENT]",
        f"has_active_agent={'true' if has_active_agent else 'false'}",
        f"pending_confirmation={'true' if pending_confirmation else 'false'}",
        f"follow_up_required={'true' if follow_up_required else 'false'}",
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
                    "语音仅表达对已完成结果的评价、态度或吐槽，且没有当前新的交付请求时，MCP 位必须为 0；"
                    "不得把正面、负面或玩笑语气本身当成隐含请求。"
                ),
            }
        )
    else:
        parts.append({"type": "text", "text": f"[CURRENT_USER_TEXT]\n{text}"})

    if compact_history and has_active_agent:
        history_rule = (
            "存在活动 Agent：上一条数字人回复只用于补全当前省略的操作类型或交付目标；"
            "补全后明确继续唱歌、搜索或查交通时才继承 MCP。"
        )
    elif compact_history:
        history_rule = (
            "没有活动 Agent：上一条数字人回复只用于理解普通对话；孤立的继续、go on、"
            "continue、接着必须按普通续讲输出 0，不得从故事、教学或闲聊内容继承 MCP。"
        )
    elif has_active_agent and (pending_confirmation or follow_up_required):
        history_rule = (
            "没有上一条可听回复，但存在等待确认或继续的活动 Agent；当前输入若是对该任务的"
            "确认、继续或补充，继承 MCP，否则按当前完整请求判断。"
        )
    else:
        history_rule = (
            "没有可继承的数字人回复；当前输入本身明确要求执行白名单操作时直接判断，"
            "无法独立确定操作类型或交付目标的省略表达不继承操作。"
        )
    parts.append(
        {
            "type": "text",
            "text": (
                "[FINAL_RULE]\n"
                "只有本轮创建或继续一个需要执行的操作，且该操作必须使用白名单能力时，MCP 位才为 1；历史不能自行创建操作。"
                "必要参数只能从用户当前摄像头画面取得时，多模态位为 1。两位独立判断再组合。"
                "当前用户明确要求系统现在执行唱歌、搜索或查交通时，MCP 位必须为 1；若该操作的目标或参数还必须从画面取得，应输出 3 而不是 2。"
                "明确限定在过去或假设的愿望，以及仅评价能力或功能本身，且没有当前执行请求时，MCP 位必须为 0。\n"
                + history_rule
            ),
        }
    )
    parts.append({"type": "text", "text": "[OUTPUT]\n只输出 0、1、2 或 3："})
    return parts
