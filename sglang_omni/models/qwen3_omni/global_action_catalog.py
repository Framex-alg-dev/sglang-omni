# SPDX-License-Identifier: Apache-2.0
"""Immutable server-wide action catalog for realtime Qwen3-Omni sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate
from sglang_omni.models.qwen3_omni.prompt_localization import (
    DEFAULT_PROMPT_LOCALE,
    PROMPT_LANGUAGE_BY_LOCALE,
    SUPPORTED_PROMPT_LOCALES,
)
from sglang_omni.utils.structured_logs import emit_structured_log


logger = logging.getLogger(__name__)


GLOBAL_ACTION_CATALOG_PATH_ENV = "SGLANG_OMNI_ACTION_CATALOG_PATH"
GLOBAL_ACTION_PREWARM_TIMEOUT_ENV = "SGLANG_OMNI_GLOBAL_ACTION_PREWARM_TIMEOUT_S"
DEFAULT_GLOBAL_ACTION_CATALOG_RESOURCE = (
    "assets/character_action_global_catalog.json"
)
UNSUPPORTED_DECISION_ID = "UNSUPPORTED"
UNSUPPORTED_CATEGORY_SCORE_ID = "B000"
UNSUPPORTED_CHILD_SCORE_ID = "A000"
UNSUPPORTED_SOURCE_LABEL = "不支持的动作"
CATEGORY_SEMANTIC_TAG_GREETING = "greeting"
CATEGORY_SEMANTIC_TAG_LOWER_BODY_MOTION = "lower_body_motion"
CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT = "reply_accompaniment"
CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT = "silent_accompaniment"
EXCLUSIVE_CATEGORY_SEMANTIC_TAGS = frozenset(
    {
        CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
        CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    }
)
SUPPORTED_ACTION_PROMPT_LOCALES = SUPPORTED_PROMPT_LOCALES
DEFAULT_ACTION_PROMPT_LOCALE = DEFAULT_PROMPT_LOCALE
ACTION_PROMPT_LANGUAGE_BY_LOCALE = PROMPT_LANGUAGE_BY_LOCALE
UNSUPPORTED_CATEGORY_SHORT_DEFINITION = (
    "用户明确要求执行动作，但该动作所属语义类别不在"
    "本次会话允许的类别中"
)
UNSUPPORTED_CHILD_SHORT_DEFINITION = (
    "用户明确要求执行动作，且动作语义类别已选定，但本次会话在该类别下允许的"
    "候选动作均无法完成该请求"
)
# Backward-compatible public alias for code that previously consumed the
# single Child-oriented definition. Prompt construction now uses the two
# stage-specific definitions above.
UNSUPPORTED_SHORT_DEFINITION = UNSUPPORTED_CHILD_SHORT_DEFINITION


def _normalize_prompt_locale(locale: str) -> str:
    if locale not in SUPPORTED_ACTION_PROMPT_LOCALES:
        raise ValueError(f"unsupported action prompt locale: {locale!r}")
    return locale


def category_unsupported_policy(locale: str = "zh-CN") -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        return (
            f"category_id={UNSUPPORTED_CATEGORY_SCORE_ID} | decision=unsupported action category | "
            "description=The user explicitly requests an action whose semantic category is not "
            "among the categories allowed in this conversation.\n"
            "If the target semantic category is allowed in this conversation, select that "
            "category even when it may lack a suitable concrete action; support for a concrete "
            "action is decided in the next stage. If the user does not explicitly request a "
            "specific action, do not select the unsupported decision. Ordinary dialogue with "
            "an actual non-empty reply uses the reply-accompaniment category; no reply, an empty "
            "reply, or reply failure uses the silent low-disturbance accompaniment category. "
            "Do not select an unrelated category merely to avoid "
            f"{UNSUPPORTED_CATEGORY_SCORE_ID}."
        )
    return (
        f"category_id={UNSUPPORTED_CATEGORY_SCORE_ID}｜决策=不支持的动作类别｜"
        f"说明={UNSUPPORTED_CATEGORY_SHORT_DEFINITION}\n"
        "如果目标语义类别在本次会话允许的类别中，应选择该类别；不得因为该类别下"
        "可能缺少具体候选动作而"
        f"选择 {UNSUPPORTED_CATEGORY_SCORE_ID}，具体动作是否支持由下一阶段判断。"
        "用户没有明确要求具体动作时，不得选择不支持判断。普通对话且实际有非空回复时，"
        "应选择语言表达伴随类别；无回复、空回复或回复失败时，应选择静默低扰伴随类别。"
        f"不得为了避免 {UNSUPPORTED_CATEGORY_SCORE_ID} 而选择与目标动作语义不相关的类别。"
    )


def child_unsupported_policy(locale: str = "zh-CN") -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        return (
            f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID} | decision=unsupported concrete action | "
            "description=The user explicitly requests an action, its semantic category has "
            "already been selected, but none of the candidates allowed in that category for "
            "this conversation can fulfill the request.\n"
            "A request is supported only when a candidate can actually perform it. An action "
            "that conveys a similar emotion or social intention but uses a different execution "
            "is not equivalent. Do not choose such an action merely to avoid "
            f"{UNSUPPORTED_CHILD_SCORE_ID}. When the user does not constrain execution details, "
            "a candidate that accomplishes the same action goal may be treated as satisfying "
            "the request. When the user specifies one or both hands, left or right, body part, "
            "count, amplitude, direction of movement, or an interaction object, the candidate "
            f"must match those details; otherwise select {UNSUPPORTED_CHILD_SCORE_ID}. If the "
            "requested object is an earring, an action that only touches the ear is not an "
            f"equivalent action; select {UNSUPPORTED_CHILD_SCORE_ID} when no earring-touch "
            "candidate is allowed. A request to touch the ear may still use a candidate that "
            "actually touches the ear. These examples define an object boundary rather than a "
            "keyword rule. If "
            "persona, style, framing, or action preferences conflict with a candidate that can "
            "actually fulfill an explicit request, those preferences must not override the "
            "matching candidate. If hard pose, framing, or object feasibility conditions make "
            "every allowed candidate unable to fulfill the request, select "
            f"{UNSUPPORTED_CHILD_SCORE_ID} instead of substituting a different action. If the "
            "user does not explicitly request a specific action, select an appropriate real "
            "action instead of the unsupported decision."
        )
    return (
        f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID}｜决策=不支持的具体动作｜"
        f"说明={UNSUPPORTED_CHILD_SHORT_DEFINITION}\n"
        "只有候选动作能够实际完成用户请求时，才视为支持。仅表达相近情绪或交际意图、"
        "但执行方式不同的动作，不属于等价动作。"
        f"不得为了避免 {UNSUPPORTED_CHILD_SCORE_ID} 而选择执行方式不同、"
        "仅表达含义相近的动作。"
        "用户没有明确限定执行细节时，能够完成同一动作目标的候选可以视为满足。"
        "用户明确限定单手或双手、左右方向、身体部位、次数、幅度、移动方向或交互物体时，"
        f"候选必须满足这些条件，否则选择 {UNSUPPORTED_CHILD_SCORE_ID}。"
        "耳环与耳朵是不同交互物体：请求摸耳环时，只有摸耳朵候选不能视为等价，"
        f"没有摸耳环候选就应选择 {UNSUPPORTED_CHILD_SCORE_ID}；请求摸耳朵时仍可选择"
        "实际触碰耳朵的候选。该例只说明物体边界，不是关键词匹配规则。"
        "人设、风格、取景或动作偏好与能够完成明确请求的候选冲突时，不得用这些偏好覆盖"
        "语义匹配候选。若姿态、取景或物体等硬性可执行条件导致全部允许候选都无法完成请求，"
        f"应选择 {UNSUPPORTED_CHILD_SCORE_ID}，不得用含义不同的动作替代。"
        "用户没有明确要求具体动作时，不得选择不支持判断，应选择合适的真实动作。"
    )


ACTION_INTENT_POLICY = (
    "先判断当前输入是否要求数字人产生外部可观察的行为。动作请求不必明确描述身体部位、"
    "运动方向或执行方式；只要用户要求数字人完成能够由动作表达的交际目标、情绪表达、"
    "姿态变化、展示或操作行为，就应选择能够直接完成该目标的动作类别。"
    "动作请求由语义目标决定，不由命令句形式决定。“你能……吗”“你可以……吗”“请……”"
    "“……吧”“帮我……”等委婉询问、建议或请求，只要要求数字人产生外部可观察的行为，"
    "都应视为明确动作目标；不得仅因其使用问句或建议句形式而将其当作普通对话或单纯能力咨询。"
    "询问数字人当前是否能够看见、听见或感知用户、环境或某项内容，属于对当前感知事实或能力的"
    "信息询问，本身不等于要求数字人执行观察动作。除非用户同时明确要求数字人看向、转向、注视"
    "或靠近某个目标，否则不应选择视线、头眼或转向动作类别；本轮产生非空回复时应选择语言表达"
    "伴随类别。例如，“你能看见我吗”是视觉事实询问，“看向我”或“看向镜头”才是观察动作请求。"
    "这些示例只说明能力询问与动作目标的边界，不是关键词匹配规则。"
    "当用户向数字人表达赠送、赞美、感谢、祝贺、亲近或其他直接作用于数字人的社交行为时，"
    "即使没有使用命令句，也应视为需要自然动作回应的社交事件。若本次会话允许的类别中存在"
    "能够自然回应这一事件的表情、情绪或社交动作，应优先选择，不得因为没有明确动作指令而"
    "默认选择待机或思考类别。只有不存在匹配的反应类别时，才选择系统伴随动作；此时不得"
    "选择不支持判断，因为用户没有明确要求具体动作。"
    "用户只是描述第三方事件、讨论某种社交行为，或只要求生成语言内容时，不表示该事件直接"
    "作用于数字人，不应据此推断具体反应动作。"
    "当用户评价、质疑或询问数字人此前的回答、笑话或表达效果时，属于直接作用于数字人的"
    "交流反馈。若当前语气具有明确的肯定、否定、疑惑、玩笑、尴尬或思考意味，且本次会话"
    "允许的类别中存在相应的表情、情绪反应或思考类别，应优先选择该类别，不得仅因用户没有"
    "明确要求身体动作而回退到系统伴随动作。只有没有明显反应语义或不存在匹配类别时，才"
    "选择系统伴随动作；此时不得选择不支持判断。普通知识问答、对第三方内容的评价以及单纯"
    "要求复述此前内容不适用本规则。"
    "用户要求数字人介绍自己、说明自身身份，或进行初次结识和建立身份关系的表达时，属于"
    "需要自然社交动作配合的自我呈现场景。若本次会话允许问候动作类别，应优先选择该类别，"
    "不得仅因用户主要请求语言内容而回退到待机类别。介绍产品、知识、地点、第三方人物或"
    "其他非数字人自身内容时，不适用本规则。"
    "除上述直接社交事件、对话反馈和自我呈现场景外，如果用户只要求说出、朗读、回答或生成"
    "语言内容，没有要求身体行为，则不应仅根据"
    "语言内容的情绪或交际含义推断具体动作。若本轮实际产生非空回复，应选择语言表达伴随类别；"
    "若本轮无需说话、回复为空或回复生成失败，应选择静默低扰伴随类别。"
    "当前输入的动作目标高于系统伴随类别和人设偏好；这些信息"
    "只能在语义匹配的类别之间辅助判断，不得把可由真实动作类别完成的请求改判为系统伴随动作。"
    "人设、风格、取景和动作偏好只能在能够完成当前动作目标的语义匹配类别之间辅助选择；"
    "不得将已存在的匹配动作压到待机类别或语义不同的相邻类别。"
    "以下示例仅说明语义判断方法，不是关键词匹配规则，也不是完整请求列表："
    "“给我打个招呼”要求数字人以可观察行为完成问候，若允许问候动作类别，应选择该类别；"
    "“说一句你好”只要求语言内容，实际产生回复时应选择语言表达伴随类别；"
    "“表示一下赞同”要求数字人表达赞同，应选择能完成该目标的动作类别；"
    "“我同意你的说法”只是用户陈述自己的态度，不等于要求数字人执行赞同动作；"
    "“抬头”只改变头部俯仰，“靠近镜头”要求上身向前靠近，二者不是相邻类别替代；"
    "“喝口水”是未限定容器的饮用动作，应进入饮用工具类别，而不是身体触碰或头发动作。"
)

ACTION_INTENT_POLICY_EN = (
    "First determine whether the current input asks the digital character to produce an "
    "externally observable behavior. An action request need not name a body part, movement "
    "direction, or execution method. If the user asks the character to accomplish a social "
    "goal, emotional expression, pose change, presentation, or operation that can be expressed "
    "through an action, select an action category that directly fulfills that goal. Determine an "
    "action request from its semantic goal, not from whether it uses an imperative sentence. "
    "Polite questions, suggestions, and requests such as 'can you ...?', 'could you ...?', "
    "'please ...', '... for me', or 'why don't you ...?' are explicit action goals whenever they "
    "ask the character to produce an externally observable behavior. Do not treat them as ordinary "
    "conversation or a mere capability question only because they use a question or suggestion "
    "form. Asking whether the character can currently see, hear, or otherwise perceive the user, "
    "the environment, or some content is a question about a current perceptual fact or capability; "
    "it does not by itself request an observation action. Unless the user also explicitly asks the "
    "character to look toward, turn toward, watch, or move closer to a target, do not select a gaze, "
    "head-and-eye, or turning action category. Use the reply-accompaniment category when the "
    "interaction produces a non-empty reply. For example, 'can you see me?' is a visual-fact "
    "question, while 'look at me' or 'look toward the camera' requests an observation action. These "
    "examples illustrate the boundary between capability questions and action goals; they are not "
    "keyword-matching rules. If the user "
    "gives something to, praises, thanks, congratulates, shows affection toward, or performs "
    "another social act directed at the digital character, treat it as a social event that calls "
    "for a natural action response even without an imperative sentence. If an allowed facial, "
    "emotional, or social-action category can naturally respond to the event, prefer it instead "
    "of defaulting to an idle or thinking category merely because no explicit action command was "
    "used. Use a system accompanying action only when no matching response category exists; do "
    "not select the unsupported decision in that case because no concrete action was requested. "
    "Merely describing a third-party event, discussing a social behavior, or requesting language "
    "content does not mean that the event is directed at the character and must not by itself "
    "trigger a concrete response action. When the user evaluates, challenges, or asks about the "
    "effect of the character's own previous answer, joke, or expression, treat it as conversational "
    "feedback directed at the character. If the current tone clearly conveys approval, disapproval, "
    "doubt, humor, embarrassment, or reflection and an allowed facial-expression, emotional-response, "
    "or thinking category matches it, prefer that category instead of defaulting to an accompanying "
    "action merely because no physical action was explicitly requested. Use a system accompanying "
    "action only when there is no clear response meaning or no matching category; do not select the "
    "unsupported decision. Ordinary knowledge questions, evaluations of third-party content, and "
    "requests that only ask to repeat earlier content do not use this rule. When the user asks "
    "the character to introduce itself, "
    "state its own identity, meet the user for the first time, or establish its relationship with "
    "the user, treat this as a self-presentation scene that calls for a natural social action. If "
    "a greeting-action category is allowed in this conversation, prefer it instead of defaulting "
    "to an idle category merely because the primary request is spoken content. This rule does not "
    "apply when introducing a product, knowledge, a place, a third party, or other content that is "
    "not about the digital character itself. Except for the direct social events, conversational "
    "feedback, and self-presentation scenes described above, if the user "
    "only asks the character to say, read, answer, or generate language and does not request "
    "physical behavior, do not infer a concrete action merely from the emotional or social "
    "meaning of the words. Use the reply-accompaniment category when this interaction produces "
    "actual non-empty reply text; use the silent low-disturbance accompaniment category when no "
    "spoken reply is needed, the reply is empty, or reply generation fails. The current input's "
    "action goal takes precedence over system accompaniment categories, "
    "and persona preferences. Those signals may only break ties among "
    "semantically matching categories and must not turn a request supported by a real action "
    "category into a default action. Persona, style, framing, and action preferences may only "
    "help choose among semantically matching categories that can fulfill the current action "
    "goal. They must not demote an existing matching action to an idle category or a "
    "semantically different neighboring category. The following examples illustrate semantic reasoning; "
    "they are not keyword-matching rules or an exhaustive request list: 'greet me' asks for an "
    "observable greeting and should use an allowed greeting-action category; 'say hello' asks "
    "only for spoken content and should use the reply-accompaniment category when a reply is "
    "actually produced; 'show "
    "agreement' requests an agreement action; 'I agree with you' only states the user's own "
    "attitude and does not request an agreement action from the character. 'Look up' changes "
    "only head pitch, while 'move closer to the camera' requires the upper body to approach; "
    "these are not interchangeable neighboring categories. 'Have some water' is a drinking "
    "request with no container specified and belongs to the tool-and-drinking category, not "
    "to body-touching or hair movement."
)


DIRECTION_REFERENCE_POLICY = (
    "方向词统一采用明确的参照系。用户没有说明参照系时，“左手”“左臂”“向左看”“向左转”"
    "“左侧拿取”等左、右方向均以数字人自身的身体坐标为准，右侧同理。数字人正面面对用户时，"
    "数字人自身左侧通常显示在用户画面右侧，这是正确表现；画面镜像不得改变 candidate_id 或"
    "action_id 的身体方向语义。用户明确说“屏幕左侧”“画面右侧”“我的左边”等参照系时，"
    "应按该明确参照系理解目标，再选择能让数字人实际朝对应目标运动的自身方向候选。"
)

DIRECTION_REFERENCE_POLICY_EN = (
    "Interpret every direction in an explicit reference frame. When the user does not name one, "
    "left and right in phrases such as 'left hand', 'left arm', 'look left', 'turn left', or "
    "'pick up from the left' always use the digital character's own body coordinates; the same "
    "applies to the right side. When the character faces the user, the character's own left usually "
    "appears on the right side of the user's view; that is correct. Mirroring the displayed image "
    "must not change the body-direction semantics of a candidate_id or action_id. If the user "
    "explicitly says 'the left side of the screen', 'the right side of the image', 'my left', or "
    "another reference frame, interpret the target in that stated frame and then select the "
    "character-relative candidate that physically moves toward it."
)


GREETING_CHILD_POLICY = (
    "用于问候、告别或建立身份关系时，如果用户没有明确指定单手、双手或热情程度，优先选择"
    "自然、克制、日常的单手问候候选。只有用户明确要求双手，或上下文清楚表达热烈欢迎、"
    "强烈兴奋等高强度语气时，才优先选择双手问候候选。"
)

GREETING_CHILD_POLICY_EN = (
    "For greetings, farewells, or establishing identity and relationship, prefer a natural, "
    "restrained, everyday one-handed greeting when the user does not specify one or both hands "
    "or an enthusiasm level. Prefer a two-handed greeting only when the user explicitly requests "
    "both hands or the context clearly calls for an enthusiastic welcome or similarly intense "
    "expression."
)

REPLY_ACCOMPANIMENT_CHILD_POLICY = (
    "本类别只用于数字人实际有非空回复文本时的语言表达伴随动作。具体动作选择必须以动态上下文中"
    "的“本轮数字人实际回复开头”为主要依据，判断其陈述、强调、列举、邀请、范围描述、过渡或"
    "指向等表达功能；用户输入只用于理解回复语境。必须从列出的真实 candidate_id 中选择。"
)

REPLY_ACCOMPANIMENT_CHILD_POLICY_EN = (
    "Use this category only when the digital character has actual non-empty reply text. Select "
    "the accompanying action primarily from the dynamically supplied 'Beginning of the digital "
    "character's actual reply', classifying its communicative function such as statement, "
    "emphasis, enumeration, invitation, range description, transition, or direction. Use the "
    "user input only as reply context. Select one of the listed real candidate_id values."
)

SILENT_ACCOMPANIMENT_CHILD_POLICY = (
    "本类别只用于本轮没有需要说出的回复文本、回复为空或回复生成失败时的静默低扰动作。"
    "根据数字人当前状态、场景和低打扰要求从列出的真实 candidate_id 中选择；不得分析回复"
    "表达功能。"
)

SILENT_ACCOMPANIMENT_CHILD_POLICY_EN = (
    "Use this category only when this interaction has no spoken reply text, the reply is empty, "
    "or reply generation failed. Select a low-disturbance real candidate_id from the character's "
    "current state and scene constraints. Do not infer a reply-expression function; select one "
    "of the listed real candidate_id values."
)

CATEGORY_CONTEXT_POLICY = (
    "选择类别前，先综合用户摄像头画面与用户语音判断用户状态："
    "情绪可归纳为开心、兴奋、惊讶、疑惑、生气、悲伤、紧张或平静；"
    "场景类型可归纳为私人空间、工作学习空间、公共空间，或驾驶、会议、医院等特殊场景；"
    "任务类型可归纳为信息获取、问题解决、情绪支持、社交闲聊、展示分享或静默观察。"
    "特殊场景下，如果本轮没有明确动作目标，且本轮主动场景约束也没有指定具体动作，"
    "仍应根据实际回复是否为空选择语言表达伴随或静默低扰伴随类别，并在该类别内"
    "优先小幅、低打扰动作。特殊场景低打扰要求不得覆盖用户明确提出且"
    "当前会话支持的动作请求。"
    "低打扰动作指幅度较小、不发生明显位移、不依赖额外物体且不会打断当前任务的动作。"
    "若 [本次会话数字人人设与动作偏好] 明确提供了数字人人设信息，包括性别、"
    "二次元/写实/卡通等画风、"
    "职业或角色定位、性格基调，选择类别时应将其纳入考虑。"
    "未提供人设信息时，可从标记为“数字人当前状态画面”的图片中观察"
    "性别表达、大致年龄段、是否有胡须、发型、是否穿裙装、是否戴眼镜等外观特征，"
    "仅依据画面可见信息判断，不得臆测未展示的设定。"
    "景别前置判断：仅根据“数字人当前状态画面”判断数字人的当前构图，"
    "不得把“用户摄像头画面”当作数字人当前状态。"
    "若数字人仅头肩或半身入镜，避免选择要求下肢、位移或全身大幅移动的类别；"
    "物体是否出现在“数字人当前状态画面”中，不作为选择物体交互类别的前置条件。"
    "上述景别条件用于没有明确动作请求时的类别偏好，以及语义匹配类别之间的辅助选择。"
    "用户明确提出动作时，如果目标语义类别在本次会话允许范围内，仍应选择该类别，由具体动作"
    "选择阶段判断候选是否满足姿态、取景及用户明确指定的交互物体等条件；不得因此改选待机类别或"
    "语义不相关的相邻类别。"
)

CATEGORY_CONTEXT_POLICY_EN = (
    "Before selecting a category, infer the user's state from the user camera view and user "
    "speech. Summarize emotion as happy, excited, surprised, confused, angry, sad, nervous, or "
    "calm; scene type as private space, work or study space, public space, or a special scene "
    "such as driving, a meeting, or a hospital; and task type as information seeking, problem "
    "solving, emotional support, social chat, presentation or sharing, or silent observation. "
    "In a special scene, if the current interaction has no explicit action target and the "
    "proactive-scene constraints for this interaction do not specify a concrete action, still "
    "select the reply-accompaniment or silent low-disturbance accompaniment category according "
    "to whether the actual reply is non-empty, then prefer a small low-disturbance action within "
    "that category. The low-disturbance requirement must not override an explicit action request "
    "that is supported in this conversation. A low-disturbance action has small "
    "amplitude, no obvious displacement, "
    "requires no extra object, and does not interrupt the current task. If [Digital character "
    "persona and action preferences for this conversation] explicitly supplies persona details "
    "such as gender expression, anime/realistic/cartoon visual style, occupation or role, and "
    "personality, incorporate them when selecting a category. If persona details are absent, "
    "visible traits such as gender expression, approximate age group, facial hair, hairstyle, "
    "skirt-like clothing, or glasses may be observed from an image labeled 'Current digital "
    "character state view'. Use only visible evidence and do not invent unseen settings. For "
    "framing decisions, infer the character's composition only from the 'Current digital "
    "character state view'; never treat a 'User camera view' as "
    "the character's current state. If only the head-and-shoulders or upper body is visible, "
    "avoid categories that require lower-body movement, displacement, or large full-body motion. "
    "Whether an object is visible in the current digital character state view is not a "
    "prerequisite for selecting an object-interaction category. These framing conditions serve "
    "as category preferences when there is no explicit action request and as tie-breakers "
    "among semantically matching categories. When the user explicitly requests an action and its "
    "target semantic category is allowed in this conversation, still select that category and let "
    "the concrete-action stage determine whether a candidate satisfies hard feasibility conditions "
    "such as pose, framing, and an interaction object explicitly specified by the user. Do not "
    "redirect the request to an idle category "
    "or a semantically unrelated neighboring category."
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class GlobalActionCandidate:
    candidate_id: str
    action_id: str
    source_label: str
    short_definition: str
    source_short_definition: str
    category_id: str


@dataclass(frozen=True, slots=True)
class GlobalActionCategory:
    category_id: str
    source_label: str
    short_definition: str
    category_path: tuple[str, ...]
    semantic_tags: frozenset[str]
    children: tuple[GlobalActionCandidate, ...]


def is_system_accompaniment_category(category: GlobalActionCategory) -> bool:
    return bool(
        category.semantic_tags
        & {
            CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
            CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
        }
    )


def category_allows_unsupported_child(category: GlobalActionCategory) -> bool:
    return not is_system_accompaniment_category(category)


@dataclass(frozen=True, slots=True)
class GlobalActionCatalog:
    catalog_version: str
    catalog_hash: str
    categories: tuple[GlobalActionCategory, ...]
    category_by_id: Mapping[str, GlobalActionCategory]
    candidate_by_id: Mapping[str, GlobalActionCandidate]
    category_system_prompt: str
    category_prompt_hash: str
    child_system_prompts: Mapping[str, str]
    child_prompt_hashes: Mapping[str, str]
    category_system_prompts_by_locale: Mapping[str, str]
    category_prompt_hashes_by_locale: Mapping[str, str]
    child_system_prompts_by_locale: Mapping[str, Mapping[str, str]]
    child_prompt_hashes_by_locale: Mapping[str, Mapping[str, str]]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_by_id)

    def candidate_for_category(
        self, category_id: str, candidate_id: str
    ) -> GlobalActionCandidate | None:
        category = self.category_by_id.get(category_id)
        if category is None:
            return None
        return next(
            (
                candidate
                for candidate in category.children
                if candidate.candidate_id == candidate_id
            ),
            None,
        )

    def category_system_prompt_for(self, locale: str) -> str:
        return self.category_system_prompts_by_locale[
            _normalize_prompt_locale(locale)
        ]

    def category_prompt_hash_for(self, locale: str) -> str:
        return self.category_prompt_hashes_by_locale[
            _normalize_prompt_locale(locale)
        ]

    def child_system_prompt_for(self, locale: str, category_id: str) -> str:
        return self.child_system_prompts_by_locale[
            _normalize_prompt_locale(locale)
        ][category_id]

    def child_prompt_hash_for(self, locale: str, category_id: str) -> str:
        return self.child_prompt_hashes_by_locale[
            _normalize_prompt_locale(locale)
        ][category_id]

    def category_with_semantic_tag(
        self, semantic_tag: str
    ) -> GlobalActionCategory | None:
        return next(
            (
                category
                for category in self.categories
                if semantic_tag in category.semantic_tags
            ),
            None,
        )

    def category_cache_namespace(self, locale: str = "zh-CN") -> str:
        normalized = _normalize_prompt_locale(locale)
        return (
            f"hierarchical:{normalized}:category:"
            f"{self.category_prompt_hash_for(normalized)}"
        )

    def child_cache_namespace(
        self, category_id: str, locale: str = "zh-CN"
    ) -> str:
        normalized = _normalize_prompt_locale(locale)
        return (
            f"hierarchical:{normalized}:child:{category_id}:"
            f"{self.child_prompt_hash_for(normalized, category_id)}"
        )


@dataclass(frozen=True, slots=True)
class GlobalActionLocalePrewarmStatus:
    category_ready: bool
    ready_child_category_ids: frozenset[str]
    failed_child_category_ids: frozenset[str]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class GlobalActionCatalogPrewarmStatus:
    category_ready: bool
    ready_child_category_ids: frozenset[str]
    failed_child_category_ids: frozenset[str]
    elapsed_ms: float
    by_locale: Mapping[str, GlobalActionLocalePrewarmStatus] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def not_run(cls) -> "GlobalActionCatalogPrewarmStatus":
        return cls(False, frozenset(), frozenset(), 0.0)

    def for_locale(self, locale: str) -> GlobalActionLocalePrewarmStatus:
        normalized = _normalize_prompt_locale(locale)
        status = self.by_locale.get(normalized)
        if status is not None:
            return status
        # Compatibility for tests and integrations that construct the legacy
        # aggregate status directly.
        return GlobalActionLocalePrewarmStatus(
            self.category_ready,
            self.ready_child_category_ids,
            self.failed_child_category_ids,
            self.elapsed_ms,
        )


async def prewarm_global_action_catalog(
    client: Any,
    *,
    model: str,
    catalog: GlobalActionCatalog,
) -> GlobalActionCatalogPrewarmStatus:
    """Best-effort prefill of both localized Category and Child prefixes."""

    started = time.perf_counter()
    prefill = getattr(client, "prefill_action_catalog", None)
    if not callable(prefill):
        logger.warning("[GLOBAL_ACTION_PREWARM] client has no catalog prefill method")
        failed_ids = frozenset(item.category_id for item in catalog.categories)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        missing = MappingProxyType(
            {
                locale: GlobalActionLocalePrewarmStatus(
                    False, frozenset(), failed_ids, elapsed_ms
                )
                for locale in SUPPORTED_ACTION_PROMPT_LOCALES
            }
        )
        return GlobalActionCatalogPrewarmStatus(
            False, frozenset(), failed_ids, elapsed_ms, missing
        )

    emit_structured_log(
        "lifecycle",
        "global_action_catalog_prewarm_started",
        catalog_version=catalog.catalog_version,
        global_catalog_hash=catalog.catalog_hash,
        locales=list(SUPPORTED_ACTION_PROMPT_LOCALES),
        category_count=len(catalog.categories),
        candidate_count=catalog.candidate_count,
    )
    logger.info(
        "[GLOBAL_ACTION_PREWARM] started catalog_version=%s locales=%s "
        "categories=%d actions=%d",
        catalog.catalog_version,
        ",".join(SUPPORTED_ACTION_PROMPT_LOCALES),
        len(catalog.categories),
        catalog.candidate_count,
    )
    try:
        timeout_s = float(os.environ.get(GLOBAL_ACTION_PREWARM_TIMEOUT_ENV, "10"))
    except ValueError as exc:
        raise ValueError(
            f"{GLOBAL_ACTION_PREWARM_TIMEOUT_ENV} must be a positive number"
        ) from exc
    if timeout_s <= 0:
        raise ValueError(
            f"{GLOBAL_ACTION_PREWARM_TIMEOUT_ENV} must be a positive number"
        )

    async def prefill_one(**kwargs: Any) -> tuple[bool, str | None]:
        try:
            ready = await asyncio.wait_for(prefill(**kwargs), timeout=timeout_s)
            return bool(ready), None
        except Exception as exc:
            logger.warning(
                "[GLOBAL_ACTION_PREWARM] prefix failed request_id=%s error=%s",
                kwargs.get("request_id"),
                exc,
                exc_info=True,
            )
            return False, f"{type(exc).__name__}: {exc}"

    async def prewarm_locale(locale: str) -> GlobalActionLocalePrewarmStatus:
        locale_started = time.perf_counter()
        prompt_language = ACTION_PROMPT_LANGUAGE_BY_LOCALE[locale]
        category_prompt = catalog.category_system_prompt_for(locale)
        category_hash = catalog.category_prompt_hash_for(locale)
        category_namespace = catalog.category_cache_namespace(locale)
        category_started = time.perf_counter()
        category_ready, category_error = await prefill_one(
            request_id=f"global-action-category-prewarm-{prompt_language}",
            model=model,
            system_prompt=category_prompt,
            candidates=[
                ActionScoreCandidate(
                    candidate_id=item.category_id,
                    suffix=item.category_id,
                    action_id=item.category_id,
                )
                for item in catalog.categories
            ]
            + [
                ActionScoreCandidate(
                    candidate_id=UNSUPPORTED_CATEGORY_SCORE_ID,
                    suffix=UNSUPPORTED_CATEGORY_SCORE_ID,
                    action_id=UNSUPPORTED_DECISION_ID,
                )
            ],
            prefix_cache_namespace=category_namespace,
            stage="category",
            language=prompt_language,
        )
        emit_structured_log(
            "performance",
            "global_action_category_prefix_prewarm_completed",
            locale=locale,
            language=prompt_language,
            global_catalog_hash=catalog.catalog_hash,
            prompt_hash=category_hash,
            prompt_chars=len(category_prompt),
            prefix_cache_namespace=category_namespace,
            candidate_count=len(catalog.categories) + 1,
            prewarmed=bool(category_ready),
            error_message=category_error,
            elapsed_ms=round((time.perf_counter() - category_started) * 1000.0, 3),
        )

        ready_children: set[str] = set()
        failed_children: set[str] = set()
        for category in catalog.categories:
            child_started = time.perf_counter()
            category_id = category.category_id
            child_prompt = catalog.child_system_prompt_for(locale, category_id)
            child_hash = catalog.child_prompt_hash_for(locale, category_id)
            namespace = catalog.child_cache_namespace(category_id, locale)
            ready, error_message = await prefill_one(
                request_id=(
                    f"global-action-child-prewarm-{prompt_language}-{category_id}"
                ),
                model=model,
                system_prompt=child_prompt,
                candidates=[
                    ActionScoreCandidate(
                        candidate_id=item.candidate_id,
                        suffix=item.candidate_id,
                        action_id=item.action_id,
                    )
                    for item in category.children
                ]
                + (
                    [
                        ActionScoreCandidate(
                            candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                            suffix=UNSUPPORTED_CHILD_SCORE_ID,
                            action_id=UNSUPPORTED_DECISION_ID,
                        )
                    ]
                    if category_allows_unsupported_child(category)
                    else []
                ),
                prefix_cache_namespace=namespace,
                stage="child",
                language=prompt_language,
            )
            (ready_children if ready else failed_children).add(category_id)
            emit_structured_log(
                "performance",
                "global_action_child_prefix_prewarm_completed",
                locale=locale,
                language=prompt_language,
                global_catalog_hash=catalog.catalog_hash,
                category_id=category_id,
                prompt_hash=child_hash,
                prompt_chars=len(child_prompt),
                prefix_cache_namespace=namespace,
                candidate_count=(
                    len(category.children)
                    + int(category_allows_unsupported_child(category))
                ),
                prewarmed=bool(ready),
                error_message=error_message,
                elapsed_ms=round(
                    (time.perf_counter() - child_started) * 1000.0, 3
                ),
            )

        locale_elapsed_ms = round(
            (time.perf_counter() - locale_started) * 1000.0, 3
        )
        locale_status = GlobalActionLocalePrewarmStatus(
            bool(category_ready),
            frozenset(ready_children),
            frozenset(failed_children),
            locale_elapsed_ms,
        )
        emit_structured_log(
            "lifecycle",
            "global_action_catalog_locale_prewarm_completed",
            level=(
                "info" if category_ready and not failed_children else "warning"
            ),
            locale=locale,
            language=prompt_language,
            global_catalog_hash=catalog.catalog_hash,
            category_ready=locale_status.category_ready,
            ready_child_count=len(locale_status.ready_child_category_ids),
            failed_child_category_ids=sorted(
                locale_status.failed_child_category_ids
            ),
            elapsed_ms=locale_elapsed_ms,
        )
        return locale_status

    by_locale = {
        locale: await prewarm_locale(locale)
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    default_status = by_locale[DEFAULT_ACTION_PROMPT_LOCALE]
    status = GlobalActionCatalogPrewarmStatus(
        default_status.category_ready,
        default_status.ready_child_category_ids,
        default_status.failed_child_category_ids,
        elapsed_ms,
        MappingProxyType(by_locale),
    )
    emit_structured_log(
        "lifecycle",
        "global_action_catalog_prewarm_completed",
        level=(
            "info"
            if all(
                item.category_ready and not item.failed_child_category_ids
                for item in by_locale.values()
            )
            else "warning"
        ),
        global_catalog_hash=catalog.catalog_hash,
        locale_statuses={
            locale: {
                "category_ready": item.category_ready,
                "ready_child_count": len(item.ready_child_category_ids),
                "failed_child_category_ids": sorted(
                    item.failed_child_category_ids
                ),
                "elapsed_ms": item.elapsed_ms,
            }
            for locale, item in by_locale.items()
        },
        elapsed_ms=elapsed_ms,
    )
    logger.info(
        "[GLOBAL_ACTION_PREWARM] completed elapsed_ms=%.3f locale_statuses=%s",
        elapsed_ms,
        {
            locale: {
                "category_ready": item.category_ready,
                "ready_children": len(item.ready_child_category_ids),
                "failed_children": len(item.failed_child_category_ids),
            }
            for locale, item in by_locale.items()
        },
    )
    return status


def build_category_system_prompt(
    categories: tuple[GlobalActionCategory, ...],
    locale: str = "zh-CN",
) -> str:
    locale = _normalize_prompt_locale(locale)
    lower_body_category_ids = tuple(
        category.category_id
        for category in categories
        if CATEGORY_SEMANTIC_TAG_LOWER_BODY_MOTION in category.semantic_tags
    )
    reply_accompaniment_ids = tuple(
        category.category_id
        for category in categories
        if CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT in category.semantic_tags
    )
    silent_accompaniment_ids = tuple(
        category.category_id
        for category in categories
        if CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT in category.semantic_tags
    )
    if locale == "en-US":
        context_policy = CATEGORY_CONTEXT_POLICY_EN
        if lower_body_category_ids:
            context_policy += (
                " In this catalog, the lower-body movement categories are: "
                + ", ".join(lower_body_category_ids)
                + "."
            )
        if reply_accompaniment_ids and silent_accompaniment_ids:
            context_policy += (
                " System accompanying-action routing uses these categories: non-empty actual "
                f"reply text -> {reply_accompaniment_ids[0]}; no spoken reply, an empty reply, "
                f"or reply generation failure -> {silent_accompaniment_ids[0]}. Apply this "
                "routing only after proactive-scene constraints, explicit user action goals, "
                "and directly matching social or emotional reactions. Those higher-priority "
                "goals must continue to use their ordinary semantic action categories."
            )
        lines = [
            "You are a digital-character action category classifier. Select one category_id from the fixed category set.",
            ACTION_INTENT_POLICY_EN,
            DIRECTION_REFERENCE_POLICY_EN,
            context_policy,
            category_unsupported_policy(locale),
            "Fixed category set:",
        ]
        lines.extend(
            f"category_id={item.category_id} | category={item.source_label} | description={item.short_definition}"
            for item in categories
        )
        lines.append(
            f"Select the category_id that best matches the current input, or {UNSUPPORTED_CATEGORY_SCORE_ID}. "
            "Output exactly one result and stop immediately. Do not explain."
        )
        return "\n".join(lines)
    context_policy = CATEGORY_CONTEXT_POLICY
    if lower_body_category_ids:
        context_policy += (
            "本目录中要求下肢、位移或全身大幅移动的类别为："
            + "、".join(lower_body_category_ids)
            + "。"
        )
    if reply_accompaniment_ids and silent_accompaniment_ids:
        context_policy += (
            "系统伴随动作按以下条件路由：数字人实际有非空回复文本时选择 "
            f"{reply_accompaniment_ids[0]}；本轮无需说话、回复为空或回复生成失败时选择 "
            f"{silent_accompaniment_ids[0]}。该路由排在本轮主动场景约束、用户明确动作目标以及"
            "能够直接匹配的社交或情绪反应之后；这些更高优先级目标仍应选择对应的普通语义类别。"
        )
    lines = [
        "你是数字人动作类别识别器。请从固定类别集合中选择一个 category_id。",
        ACTION_INTENT_POLICY,
        DIRECTION_REFERENCE_POLICY,
        context_policy,
        category_unsupported_policy(locale),
        "固定类别集合如下：",
    ]
    lines.extend(
        f"category_id={item.category_id}｜类别={item.source_label}｜说明={item.short_definition}"
        for item in categories
    )
    lines.append(
        f"请根据当前输入选择最匹配的 category_id 或 {UNSUPPORTED_CATEGORY_SCORE_ID}；"
        "只输出一个结果，输出后立即结束，不要解释。"
    )
    return "\n".join(lines)


def build_child_system_prompt(
    category: GlobalActionCategory,
    locale: str = "zh-CN",
) -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        lines = [
            "You are a digital-character action classifier. Select one candidate_id from the following set.",
            DIRECTION_REFERENCE_POLICY_EN,
            (
                f"Selected category: category_id={category.category_id} | "
                f"category={category.source_label} | description={category.short_definition}"
            ),
        ]
        if category_allows_unsupported_child(category):
            lines.append(child_unsupported_policy(locale))
        elif CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT in category.semantic_tags:
            lines.append(REPLY_ACCOMPANIMENT_CHILD_POLICY_EN)
        else:
            lines.append(SILENT_ACCOMPANIMENT_CHILD_POLICY_EN)
        if CATEGORY_SEMANTIC_TAG_GREETING in category.semantic_tags:
            lines.append(GREETING_CHILD_POLICY_EN)
        lines.extend(
            f"candidate_id={item.candidate_id} | action={item.source_label} | description={item.short_definition}"
            for item in category.children
        )
        lines.append(
            "Select the candidate_id that best matches from the candidates in the selected category above. "
            "Output exactly one result."
        )
        return "\n".join(lines)
    lines = [
        "你是数字人动作识别器。请从以下集合中选择一个 candidate_id。",
        DIRECTION_REFERENCE_POLICY,
        (
            f"已选类别：category_id={category.category_id}｜类别={category.source_label}｜"
            f"说明={category.short_definition}"
        ),
    ]
    if category_allows_unsupported_child(category):
        lines.append(child_unsupported_policy(locale))
    elif CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT in category.semantic_tags:
        lines.append(REPLY_ACCOMPANIMENT_CHILD_POLICY)
    else:
        lines.append(SILENT_ACCOMPANIMENT_CHILD_POLICY)
    if CATEGORY_SEMANTIC_TAG_GREETING in category.semantic_tags:
        lines.append(GREETING_CHILD_POLICY)
    lines.extend(
        f"candidate_id={item.candidate_id}｜动作={item.source_label}｜说明={item.short_definition}"
        for item in category.children
    )
    lines.append(
        "只能从以上已选类别的候选动作中选择最匹配的 candidate_id；"
        "只输出一个结果。"
    )
    return "\n".join(lines)


def load_global_action_catalog(path: str | Path | None = None) -> GlobalActionCatalog:
    """Load and fully validate the authoritative catalog.

    Empty action ``short_definition`` values intentionally remain untouched in
    the source file and catalog hash. At runtime their label is used as the
    prompt definition so every prompt line stays meaningful.
    """

    configured_path = path or os.environ.get(GLOBAL_ACTION_CATALOG_PATH_ENV)
    if configured_path is None:
        catalog_path = files("sglang_omni").joinpath(
            DEFAULT_GLOBAL_ACTION_CATALOG_RESOURCE
        )
        raw_text = catalog_path.read_text(encoding="utf-8")
        display_path = str(catalog_path)
    else:
        catalog_path = Path(configured_path).expanduser()
        raw_text = catalog_path.read_text(encoding="utf-8")
        display_path = str(catalog_path)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid global action catalog JSON: {display_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("global action catalog must be a JSON object")
    catalog_version = _required_string(payload.get("catalog_version"), "catalog_version")
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("global action catalog categories must be a non-empty list")

    category_ids: set[str] = set()
    candidate_ids: set[str] = set()
    candidate_action_ids: dict[str, str] = {}
    action_candidate_ids: dict[str, str] = {}
    categories: list[GlobalActionCategory] = []
    candidate_occurrences_by_id: dict[str, list[GlobalActionCandidate]] = {}
    for category_index, raw_category in enumerate(raw_categories):
        if not isinstance(raw_category, dict):
            raise ValueError(f"categories[{category_index}] must be an object")
        prefix = f"categories[{category_index}]"
        category_id = _required_string(raw_category.get("category_id"), f"{prefix}.category_id")
        if category_id in {
            UNSUPPORTED_CATEGORY_SCORE_ID,
            UNSUPPORTED_CHILD_SCORE_ID,
            UNSUPPORTED_DECISION_ID,
        }:
            raise ValueError(
                f"global category_id is reserved for unsupported scoring: {category_id}"
            )
        if category_id in category_ids:
            raise ValueError(f"duplicate global category_id: {category_id}")
        category_ids.add(category_id)
        source_label = _required_string(raw_category.get("source_label"), f"{prefix}.source_label")
        short_definition = _required_string(
            raw_category.get("short_definition"), f"{prefix}.short_definition"
        )
        raw_path = raw_category.get("category_path", [])
        if not isinstance(raw_path, list):
            raise ValueError(f"{prefix}.category_path must be a string list")
        category_path = tuple(
            _required_string(value, f"{prefix}.category_path[{index}]")
            for index, value in enumerate(raw_path)
        )
        raw_semantic_tags = raw_category.get("semantic_tags", [])
        if not isinstance(raw_semantic_tags, list):
            raise ValueError(f"{prefix}.semantic_tags must be a string list")
        semantic_tags = tuple(
            _required_string(value, f"{prefix}.semantic_tags[{index}]")
            for index, value in enumerate(raw_semantic_tags)
        )
        if len(set(semantic_tags)) != len(semantic_tags):
            raise ValueError(f"{prefix}.semantic_tags must not contain duplicates")
        raw_children = raw_category.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ValueError(f"{prefix}.children must be a non-empty list")
        children: list[GlobalActionCandidate] = []
        category_candidate_ids: set[str] = set()
        for child_index, raw_child in enumerate(raw_children):
            if not isinstance(raw_child, dict):
                raise ValueError(f"{prefix}.children[{child_index}] must be an object")
            child_prefix = f"{prefix}.children[{child_index}]"
            candidate_id = _required_string(
                raw_child.get("candidate_id"), f"{child_prefix}.candidate_id"
            )
            action_id = _required_string(raw_child.get("action_id"), f"{child_prefix}.action_id")
            if candidate_id in {
                UNSUPPORTED_CATEGORY_SCORE_ID,
                UNSUPPORTED_CHILD_SCORE_ID,
                UNSUPPORTED_DECISION_ID,
            }:
                raise ValueError(
                    "global candidate_id is reserved for unsupported scoring: "
                    f"{candidate_id}"
                )
            if action_id == UNSUPPORTED_DECISION_ID:
                raise ValueError(
                    "global action_id=UNSUPPORTED is reserved for the internal "
                    "non-executable decision"
                )
            if action_id == "no_action":
                raise ValueError(
                    "global action catalog must not contain action_id=no_action; "
                    "configure executable fallback categories in session.start"
                )
            child_label = _required_string(
                raw_child.get("source_label"), f"{child_prefix}.source_label"
            )
            source_definition = raw_child.get("short_definition")
            if not isinstance(source_definition, str):
                raise ValueError(f"{child_prefix}.short_definition must be a string")
            source_definition = source_definition.strip()
            prompt_definition = source_definition or child_label
            if candidate_id in category_candidate_ids:
                raise ValueError(
                    "duplicate global candidate_id within category "
                    f"{category_id}: {candidate_id}"
                )
            known_action_id = candidate_action_ids.get(candidate_id)
            if known_action_id is not None and known_action_id != action_id:
                raise ValueError(
                    "duplicate global candidate_id must keep the same action_id: "
                    f"{candidate_id}: {known_action_id} != {action_id}"
                )
            known_candidate_id = action_candidate_ids.get(action_id)
            if (
                known_candidate_id is not None
                and known_candidate_id != candidate_id
            ):
                raise ValueError(
                    "duplicate global action_id must keep the same candidate_id: "
                    f"{action_id}: {known_candidate_id} != {candidate_id}"
                )
            category_candidate_ids.add(candidate_id)
            candidate_ids.add(candidate_id)
            candidate_action_ids[candidate_id] = action_id
            action_candidate_ids[action_id] = candidate_id
            child = GlobalActionCandidate(
                candidate_id=candidate_id,
                action_id=action_id,
                source_label=child_label,
                short_definition=prompt_definition,
                source_short_definition=source_definition,
                category_id=category_id,
            )
            children.append(child)
            candidate_occurrences_by_id.setdefault(candidate_id, []).append(
                child
            )
        categories.append(
            GlobalActionCategory(
                category_id=category_id,
                source_label=source_label,
                short_definition=short_definition,
                category_path=category_path,
                semantic_tags=frozenset(semantic_tags),
                children=tuple(children),
            )
        )

    collisions = category_ids & candidate_ids
    if collisions:
        raise ValueError(
            "global category_id and candidate_id values must be disjoint: "
            + ", ".join(sorted(collisions))
        )
    normalized_categories = tuple(categories)
    for semantic_tag in EXCLUSIVE_CATEGORY_SEMANTIC_TAGS:
        owners = [
            category.category_id
            for category in normalized_categories
            if semantic_tag in category.semantic_tags
        ]
        if len(owners) > 1:
            raise ValueError(
                "global category semantic tag must have at most one owner: "
                f"{semantic_tag}: {', '.join(owners)}"
            )
    category_by_id = {item.category_id: item for item in normalized_categories}
    candidate_by_id = {
        candidate_id: next(
            (
                candidate
                for candidate in occurrences
                if not is_system_accompaniment_category(
                    category_by_id[candidate.category_id]
                )
            ),
            occurrences[0],
        )
        for candidate_id, occurrences in candidate_occurrences_by_id.items()
    }
    category_prompts_by_locale = {
        locale: build_category_system_prompt(normalized_categories, locale)
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    child_prompts_by_locale = {
        locale: MappingProxyType(
            {
                item.category_id: build_child_system_prompt(item, locale)
                for item in normalized_categories
            }
        )
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    # Keep the original public fields as Chinese aliases for integrations that
    # inspect the catalog directly. Runtime sessions always use locale-aware
    # accessors.
    category_prompt = category_prompts_by_locale["zh-CN"]
    child_prompts = child_prompts_by_locale["zh-CN"]
    category_hashes_by_locale = {
        locale: _sha256_text(prompt)
        for locale, prompt in category_prompts_by_locale.items()
    }
    child_hashes_by_locale = {
        locale: MappingProxyType(
            {key: _sha256_text(value) for key, value in prompts.items()}
        )
        for locale, prompts in child_prompts_by_locale.items()
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return GlobalActionCatalog(
        catalog_version=catalog_version,
        catalog_hash=_sha256_text(canonical),
        categories=normalized_categories,
        category_by_id=MappingProxyType(category_by_id),
        candidate_by_id=MappingProxyType(candidate_by_id),
        category_system_prompt=category_prompt,
        category_prompt_hash=_sha256_text(category_prompt),
        child_system_prompts=MappingProxyType(child_prompts),
        child_prompt_hashes=MappingProxyType(
            {key: _sha256_text(value) for key, value in child_prompts.items()}
        ),
        category_system_prompts_by_locale=MappingProxyType(
            category_prompts_by_locale
        ),
        category_prompt_hashes_by_locale=MappingProxyType(
            category_hashes_by_locale
        ),
        child_system_prompts_by_locale=MappingProxyType(
            child_prompts_by_locale
        ),
        child_prompt_hashes_by_locale=MappingProxyType(
            child_hashes_by_locale
        ),
    )


__all__ = [
    "GlobalActionCandidate",
    "GlobalActionCatalog",
    "GlobalActionCatalogPrewarmStatus",
    "GlobalActionLocalePrewarmStatus",
    "SUPPORTED_ACTION_PROMPT_LOCALES",
    "DEFAULT_ACTION_PROMPT_LOCALE",
    "CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT",
    "CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT",
    "UNSUPPORTED_CATEGORY_SCORE_ID",
    "UNSUPPORTED_CHILD_SCORE_ID",
    "UNSUPPORTED_CATEGORY_SHORT_DEFINITION",
    "UNSUPPORTED_CHILD_SHORT_DEFINITION",
    "UNSUPPORTED_DECISION_ID",
    "UNSUPPORTED_SHORT_DEFINITION",
    "UNSUPPORTED_SOURCE_LABEL",
    "GlobalActionCategory",
    "build_category_system_prompt",
    "build_child_system_prompt",
    "category_unsupported_policy",
    "category_allows_unsupported_child",
    "child_unsupported_policy",
    "is_system_accompaniment_category",
    "load_global_action_catalog",
    "prewarm_global_action_catalog",
]
