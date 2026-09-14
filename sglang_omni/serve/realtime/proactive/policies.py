"""Built-in proactive scene policies.

The wire protocol names the scene and may provide a concise refinement.  The
server owns all mandatory reply/history/action behavior so older, incomplete,
or differently configured clients cannot remove the safety boundaries.
"""

from __future__ import annotations

from dataclasses import replace

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
            "动作意图为首次问候。动作范围以挥手、点头、致意或其他具有明确问候"
            "含义的友好动作为主。保持当前姿态，动作清晰可见且幅度适中。"
        ),
        default_action_guidance_en=(
            "The action intent is a first greeting. Prefer a wave, nod, salutation, or "
            "another clearly friendly greeting while preserving the current pose and "
            "using a clear, moderate motion."
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
            "动作意图为低打扰提醒。优先选择轻柔点头、轻微抬手、自然关注用户或"
            "其他幅度较小的提醒动作。保持当前姿态，动作轻量且清晰。"
        ),
        default_action_guidance_en=(
            "The action intent is a low-disturbance reminder. Prefer a gentle nod, small "
            "hand motion, attentive response, or another subtle reminder that preserves "
            "the current pose."
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
            "动作意图为欢迎用户重新出现。优先选择挥手、致意、轻柔点头、微笑回应"
            "或自然欢迎手势。保持当前姿态，动作友好、清晰且不过度。"
        ),
        default_action_guidance_en=(
            "The action intent is to welcome the user back. Prefer a wave, salutation, "
            "gentle nod, smile response, or natural welcoming gesture while preserving "
            "the current pose."
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
            "动作意图为自然告别。优先选择挥手、点头、致意等清晰友好的告别动作，"
            "并保持当前姿态。"
        ),
        default_action_guidance_en=(
            "The action intent is a natural farewell. Prefer a wave, nod, salutation, or "
            "another clear friendly farewell while preserving the current pose."
        ),
    ),
}


def proactive_scene_policy(trigger: str | None) -> ProactiveScenePolicy | None:
    """Return the built-in policy for a recognized language proactive scene."""
    policy = _POLICIES.get(trigger or "")
    if policy is None:
        return None
    reply_override = read_runtime_prompt_section(
        "proactive_reply_rules", policy.trigger
    )
    action_override = read_runtime_prompt_section(
        "proactive_action_rules", policy.trigger
    )
    if reply_override is None and action_override is None:
        return policy
    return replace(
        policy,
        reply_policy_zh=reply_override or policy.reply_policy_zh,
        reply_policy_en=reply_override or policy.reply_policy_en,
        default_action_guidance_zh=(
            action_override or policy.default_action_guidance_zh
        ),
        default_action_guidance_en=(
            action_override or policy.default_action_guidance_en
        ),
    )
