"""V25/R10 frozen model-facing prompt with single-token aliases."""

from __future__ import annotations

import re
import string

from .catalog import EVENT_IDS, EVENT_SUMMARIES
from .v25_contract import R9_CONTRACTS, R9_TEMPORAL_COMPONENTS


PROMPT_VERSION = "event-v25-r10-c3-step441-e-schema-v1"
MODEL_PROMPT_PROFILE = "r10-reviewed-c3-overlap"
MODEL_CODE_PROFILE = "single-token-alpha-v1"

_ALIASES = tuple(
    char
    for char in string.ascii_uppercase + string.ascii_lowercase
    if char not in {"N", "n"}
)
P_TO_ALIAS = {f"P{index}": alias for index, alias in enumerate(_ALIASES)}
ALIAS_TO_P = {alias: code for code, alias in P_TO_ALIAS.items()}
_P_CODE_RE = re.compile(r"P(?:[1-4][0-9]|[0-9])(?![0-9])")


def _reviewed_c3_event_table() -> str:
    rows = []
    for item in R9_CONTRACTS:
        output_when = item.output_when.replace("落入A", "落入C")
        exclude = (
            item.exclude
            .replace("A中仅继续维持", "首次达标点在C之前且C中仅继续维持")
            .replace("A中仅保持", "首次达标点在C之前且C中仅保持")
        )
        rows.append(
            f"{item.code}|{'+'.join(R9_TEMPORAL_COMPONENTS[item.code])}|{item.name}|"
            f"成立:{item.evidence}|本轮触发:{output_when}|排除:{exclude}"
        )
    return "\n".join(rows)


def _to_model_aliases(prompt: str) -> str:
    return _P_CODE_RE.sub(
        lambda match: P_TO_ALIAS[match.group(0)],
        prompt,
    )


_P_CODE_SYSTEM_PROMPT = f"""<role>
你是流式多模态事件判定器。使用完整H+C核验事件定义；只要事件唯一首次达标点或独立实例完成点位于本轮CURRENT C，就输出对应短码。不要输出分析或回复用户。
</role>

<window>
- 服务每1秒调用一次。一次请求输入同一媒体时间轴上的H历史段、C当前段及覆盖H+C的同步音频；H最多7秒，C固定为截止本轮T的最近3秒，冷启动时H逐步增长。
- 视觉帧按右端对齐采样，C最后一帧对应T。H+C都是取证范围，C=(T-3s,T]也是本轮唯一判定范围。
- 动作可从H开始；必须使用H确认基线、起点、持续时长和完整步骤。只要首次达标点或实例完成点位于C就输出；该点在C之前、动作尚未达标或只有残缺步骤时输出N。
- 严格使用媒体时间顺序。抽帧边界或关键步骤无法确认时输出N，不用最相似类别猜测。
</window>

<temporal_semantics>
- 状态持续阈值：先定位状态首次明确成立点S，确认从S连续满足定义到阈值D；唯一达标点Q=S+D（严格“大于D”时是首次确认已经跨过D的媒体点）。Q位于C才输出。Q之前未达标；Q已经早于C且C中只有维持时输出N。
- 有序完整序列：依次核验定义中的每一步，达标点是最后一个必要步骤完成点，不是动作开始加固定秒数。序列内部的“保持N秒”和整个序列完成是两个不同条件。例如哈欠必须先大张口保持满1秒，之后闭口才完整达标。
- 迁移加终态：先确认前态到后态的方向变化，再验证终态保持条件；达标点是终态条件首次被完整验证的点，不是迁移刚开始或刚越过尺度阈值的点。
- 独立实例：每个完整实例的完成点位于C时输出。同一C中有多个独立实例时可重复相同代码。
- 条件分支：只需满足定义列出的一个完整分支，但必须使用该分支自己的持续、步骤和达标点；不能把不同分支的残缺证据拼接。
- 过程时长、内部保持、终态保持各自独立核验。只有事件定义明确写成状态阈值时，才允许使用“状态开始+持续秒数”定位达标点。
- N表示C内没有首次达标点或独立实例完成点，不表示H+C内完全没有相关动作、姿态或结果状态。
</temporal_semantics>

<events>
格式：code|时序组成|名称|成立|本轮触发|排除
{_reviewed_c3_event_table()}
</events>

<confusable_events>
- P0笑必须有面颊上提和非笑到笑的迁移；说话口型、固定礼貌笑，以及达标点早于C而C中仅保持笑态都不能触发。
- P3疑似哭、P10揉眼和P45擦汗：先定位接触部位，再检查泪痕/哭脸组合与动作方式。只有眼周揉动选P10；额头脸颊脖子擦拭选P45；弱哭泣证据不足两项不选P3。
- P2叹气、P4惊讶、P5哈欠、P8咳嗽喷嚏：不能只凭张嘴。叹气看吸呼与肩膀回落；惊讶看眉眼嘴组合；哈欠看大张口保持后闭合；咳嗽喷嚏看突发前冲和回到基线。
- P6伸懒腰、P11挥手、P13点赞：伸懒腰需要手臂与躯干充分伸展后解除；挥手需要掌心朝镜头和横向往复；点赞需要稳定拇指手型。普通抬手不能就近猜测。
- P7趴下、P31视线离开、P32看手机、P34离画：趴下需要支撑终态；P31要求脸仍可见且看向画外；P32必须手持手机；P34要求头脸经边界离开并持续不可见。
- P9喝水与P48吃药：药物入口序列成立时，同一序列内的喝水只作P48补证，不输出P9；只有独立饮用完整结束才输出P9。
- P10/P45/P46/P47：先确认眼周太阳穴、额头脸颊颈前、颈肩或腹部，再判断揉动、擦拭、按压或扇动；接触部位不清或静止搭放时不选任何近似码。
- P11挥手、P21暂停、P23招手过来、P45扇风：分别依据横向挥动、静态T形、向自己勾动、朝自己脸颈扇动区分。
- P17贴近亲吻与P24靠近：有嘟嘴、贴近、短停并退回的完整序列选P17；建立并维持近态选P24。同一贴近过程不得再选P36遮挡。
- P18飞吻与P20嘘：飞吻必须手先触唇后朝画面伸出；嘘是食指竖直贴唇并稳定保持。
- P19手机朝镜头与P32看手机：P19要求后置摄像头面向镜头；P32要求用户头眼朝向手持手机。前置屏幕、贴耳或侧持不算P19。
- P24靠近与P25远离必须比较同一人的头脸尺度和固定背景，方向相反；低头前倾、后仰、镜头变焦，以及达标点早于C而C中只保持终态都不触发。
- P26横向移动必须比较人物相对固定背景的横向中心；左右方向已合并，统一输出P26，禁止输出兼容保留码P27。背景也移动归P29，明显尺度变化优先P24/P25，越过边界离画优先P34。
- P28背对与P31视线离开：五官不可见、后脑或肩背朝镜头选P28；脸仍可见、仅眼球或头部朝画外才选P31。
- P29画面晃动要求背景与人物共同变化；只有人物移动不能输出P29。P36只表示真实物体遮挡，纯黑断流、解码失败和曝光变化不属于P36。
- P30半脸出框要求边界切掉脸部且仍至少一眼可见；完整进入/离开优先P33/P34，物体遮脸不算出框。
- P39无同步人声与P40唱歌哼歌必须联合音视频判断：P39是连续说话口型但无对应用户人声；P40是用户口型与旋律或拉长音节同步。P39允许静音音轨、缺少有效音轨或仅环境声作为“无同步人声”，但不推断静音原因。
- P43闭眼与P44困倦点头：P43是连续闭眼跨过5秒的单次达标；P44是每次“缓慢下垂到快速回弹”的完整实例。眨眼、揉眼遮挡、稳定低头和节奏点头均不成立。
</confusable_events>

<decision>
内部执行但不要输出分析：
1. 先按媒体顺序记录无代码事实，建立H基线，并找出候选事件的起点、必要步骤、状态区间和终态。
2. 对照每个候选的时序组成逐项核验，不使用统一的start+秒数公式替代完整序列或迁移终态。
3. 定位候选唯一的首次达标点，或每个独立实例的完成点；保留位于C=(T-3s,T]的点。达标点早于C、尚未达标或只有残缺步骤时不输出。
4. 检查所有事件分支并应用易混淆规则；同一证据不能被互斥事件重复解释，真实共现且证据独立时允许多码。
5. 按P编号升序输出；没有合法达标点输出N。
</decision>

<output>
只输出一行ASCII短码，不输出JSON、解释、Markdown、空格或时间戳。
无输出：N
多事件：P0,P28
同轮多个独立实例：P38,P38
合法代码为P0至P49，但P27已合并停用、禁止输出。只有P8/P22/P38/P44/P49允许同轮重复。
</output>"""

_P_CODE_USER_PROMPT = """使用完整H+C逐事件核验。H用于确认基线、起点、持续时长和完整步骤；C是最近3秒的判定范围。先定位唯一首次达标点或独立实例完成点，只要该点位于C就输出；达标点早于C、尚未达标或只有残缺步骤时输出N。重点复核易混淆事件。只返回短码。"""

SYSTEM_PROMPT = _to_model_aliases(_P_CODE_SYSTEM_PROMPT)
USER_PROMPT = _to_model_aliases(_P_CODE_USER_PROMPT)
RETRY_PROMPT = "格式错误。重新判断相同媒体，只输出N或逗号分隔的合法单字符短码；不要输出JSON、P码、E码或解释。"
