"""Built-in proactive scene policies.

The wire protocol names the scene and may provide a concise refinement.  The
server owns all mandatory reply/history/action behavior so older, incomplete,
or differently configured clients cannot remove the safety boundaries.
"""

from __future__ import annotations

from dataclasses import replace
from .action_policy import proactive_selection_instruction

from sglang_omni.serve.realtime.proactive.models import ProactiveScenePolicy
from sglang_omni.serve.realtime.runtime_prompt_overrides import (
    read_runtime_prompt_section,
)

SESSION_ENTER_TRIGGER = "session_enter"
IDLE_TIMEOUT_TRIGGER = "idle_timeout"
USER_RETURNED_TRIGGER = "user_returned"
CHARACTER_PROACTIVE_TRIGGER = "character_proactive"
SESSION_ENDING_TRIGGER = "session_ending"


_POLICIES = {
    SESSION_ENTER_TRIGGER: ProactiveScenePolicy(
        trigger=SESSION_ENTER_TRIGGER,
        history_policy="none",
        memory_policy="none",
        reply_policy_zh=(
            "[主动场景：首次进入会话]\n"
            "用户刚进入本次会话。结合角色人设，只生成一句简短、自然、可以直接"
            "说出口的首次欢迎语，并自然开启交流。当前没有用户曾经离开、重新返回"
            "或角色一直等待用户的事实，不得说‘欢迎回来’‘又见面了’‘终于来了’"
            "或其他暗示既往经历、等待状态的话。不要描述动作。"
        ),
        reply_policy_en=(
            "[Proactive scene: first session entry]\nThe user has just entered this "
            "session. Using the persona, produce one brief, natural spoken greeting and "
            "open the conversation. There is no evidence that the user returned, met the "
            "character before, or kept the character waiting. Do not imply any such prior "
            "event. Do not describe actions."
        ),
        default_action_guidance_zh=(
            "场景目标为首次建立交流。优先体现角色自身风格，不限定为问候手势；"
            "没有合适风格动作时才用允许的挥手、点头或致意兜底。"
        ),
        default_action_guidance_en=(
            "Open the interaction in the character's own style. A permitted wave, nod "
            "or salutation is only a fallback when no suitable style-specific action fits."
        ),
    ),
    IDLE_TIMEOUT_TRIGGER: ProactiveScenePolicy(
        trigger=IDLE_TIMEOUT_TRIGGER,
        history_policy="none",
        memory_policy="none",
        reply_policy_zh=(
            "[主动场景：低打扰提醒]\n"
            "用户暂时没有继续说话。结合角色人设，只生成一句简短、自然、低打扰的"
            "话，提醒用户仍可继续交流。语气轻柔，节奏舒缓。不得猜测用户沉默的"
            "原因、情绪或正在做什么，不得复用先前回复的话题，不要描述动作。"
        ),
        reply_policy_en=(
            "[Proactive scene: low-disturbance reminder]\nThe user has not continued "
            "speaking. Produce one brief, gentle, low-disturbance spoken reminder that the "
            "conversation may continue. Do not infer why the user is silent, their mood, "
            "or activity. Do not reuse the prior reply topic or describe actions."
        ),
        default_action_guidance_zh=(
            "场景目标为低打扰提醒。优先体现角色风格，保持轻量；没有合适风格动作"
            "时才使用允许的轻柔点头或自然关注动作兜底。"
        ),
        default_action_guidance_en=(
            "Prioritize the character's style for a low-disturbance reminder. A permitted "
            "gentle nod or subtle attentive response is only a fallback."
        ),
    ),
    USER_RETURNED_TRIGGER: ProactiveScenePolicy(
        trigger=USER_RETURNED_TRIGGER,
        history_policy="none",
        memory_policy="none",
        reply_policy_zh=(
            "[主动场景：用户重新出现]\n"
            "客户端已确认用户此前暂时离开当前互动画面，现在重新出现。结合角色"
            "人设，只生成一句简短、自然的欢迎语；可以欢迎用户回来，也可以自然"
            "询问是否继续交流。不得猜测用户离开的原因、去向或经历，不要描述动作。"
        ),
        reply_policy_en=(
            "[Proactive scene: user returned]\nThe client has confirmed that the user "
            "temporarily left the interaction view and has now reappeared. Produce one "
            "brief, natural welcome-back line and optionally ask whether to continue. Do "
            "not infer why they left, where they went, or what happened. Do not describe actions."
        ),
        default_action_guidance_zh=(
            "场景目标为回应用户重新出现。优先体现角色风格；没有合适风格动作时，"
            "才使用允许的挥手、致意或轻柔点头兜底。"
        ),
        default_action_guidance_en=(
            "Acknowledge the returning user in the character's own style. Use a permitted "
            "wave, salutation or gentle nod only as a fallback."
        ),
    ),
    CHARACTER_PROACTIVE_TRIGGER: ProactiveScenePolicy(
        trigger=CHARACTER_PROACTIVE_TRIGGER,
        history_policy="memory",
        memory_policy="proactive",
        reply_policy_zh=(
            "[主动场景：角色自然发起交流]\n"
            "当前没有新的用户问题需要回答。只选择一个自然的交流方向。服务端提供"
            "的会话信息中若存在明确、尚未结束且现在继续仍有价值的内容，可以自然"
            "承接；没有合适内容时，根据角色人设开启一个轻量话题。不得同时罗列多个"
            "历史话题，不得编造用户没有提供的经历、状态或偏好，不得把先前 assistant"
            "回复中的推测当作事实。只输出一至两句可以直接说出口的话，不要描述动作。"
        ),
        reply_policy_en=(
            "[Proactive scene: character starts a conversation]\nThere is no new user "
            "question to answer. Choose exactly one natural direction. Continue a clearly "
            "unfinished and still valuable item from server-provided session data when one "
            "exists; otherwise open one light topic consistent with the persona. Do not list "
            "multiple historical topics, invent user facts, or treat prior assistant guesses "
            "as facts. Output only one or two natural spoken sentences and do not describe actions."
        ),
        default_action_guidance_zh=(
            "动作意图为自然开启或延续交流。选择与最终回复语义一致、符合当前角色"
            "表现方式的轻量、友好动作，并保持当前姿态。"
        ),
        default_action_guidance_en=(
            "The action intent is to open or continue a conversation naturally. Choose a "
            "light, friendly action aligned with the final spoken reply and preserve "
            "the current pose."
        ),
    ),
    SESSION_ENDING_TRIGGER: ProactiveScenePolicy(
        trigger=SESSION_ENDING_TRIGGER,
        history_policy="none",
        memory_policy="none",
        reply_policy_zh=(
            "[主动场景：会话结束]\n结合角色人设，只生成一句简短、自然、可以直接"
            "说出口的告别语，不延伸新话题，不描述动作。"
        ),
        reply_policy_en=(
            "[Proactive scene: session ending]\nUsing the persona, produce one brief, "
            "natural spoken farewell. Do not open a new topic or describe actions."
        ),
        default_action_guidance_zh=(
            "场景目标为自然告别。优先体现角色风格；没有合适风格动作时才使用允许"
            "的挥手、点头或致意兜底。"
        ),
        default_action_guidance_en=(
            "Express farewell in the character's own style. Use a permitted wave, nod "
            "or salutation only as a fallback."
        ),
    ),
}


def proactive_scene_policy(trigger: str | None) -> ProactiveScenePolicy | None:
    """Return the built-in policy for a recognized language proactive scene."""
    policy = _POLICIES.get(trigger or "")
    if policy is None:
        return None
    policy = replace(
        policy,
        default_action_guidance_zh=(proactive_selection_instruction("zh") + policy.default_action_guidance_zh),
        default_action_guidance_en=(proactive_selection_instruction("en") + policy.default_action_guidance_en),
    )
    return replace(
        policy,
        reply_policy_zh=(read_runtime_prompt_section(
            "proactive_reply_rules", policy.trigger, language="zh"
        ) or policy.reply_policy_zh),
        reply_policy_en=(read_runtime_prompt_section(
            "proactive_reply_rules", policy.trigger, language="en"
        ) or policy.reply_policy_en),
        default_action_guidance_zh=(read_runtime_prompt_section(
            "proactive_action_rules", policy.trigger, language="zh"
        ) or policy.default_action_guidance_zh),
        default_action_guidance_en=(read_runtime_prompt_section(
            "proactive_action_rules", policy.trigger, language="en"
        ) or policy.default_action_guidance_en),
    )
