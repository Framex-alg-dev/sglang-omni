"""Persona-first selection for spoken proactive scenes, not user requests."""

SPOKEN_PROACTIVE_TRIGGERS = frozenset({
    "session_enter", "idle_timeout", "user_returned", "character_proactive",
    "session_ending", "podcast_item_transition",
})

# Only consulted when category recall leaves no executable style candidate.
SCENE_FALLBACK_IDS = {
    "session_enter": ("189", "288", "289"),
    "user_returned": ("189", "288", "289"),
    "session_ending": ("189", "288"),
    "idle_timeout": ("189",),
    "character_proactive": ("189", "212"),
    "podcast_item_transition": ("189", "212"),
}


def persona_first_scene(turn_origin: str, trigger: str | None) -> bool:
    return turn_origin == "proactive" and trigger in SPOKEN_PROACTIVE_TRIGGERS


def proactive_selection_instruction(language: str) -> str:
    if language == "en":
        return (
            "[Proactive action selection]\nThis is a character-initiated scene, "
            "not a user action-support check. Preserve the allowed catalog, current "
            "pose, physical requirements and explicit prohibitions. Within those "
            "hard constraints, prioritize the character's persona and visual style. "
            "Only when no style-specific choice fits, use a permitted scene-appropriate "
            "fallback: greeting for entry/return, farewell for ending, subtle attention "
            "for idle, and natural conversational accompaniment for dialogue or podcast "
            "transitions. Select a real allowed candidate; do not output 000.\n"
        )
    return (
        "[主动场景动作选择]\n本轮是角色主动互动，不是用户动作请求的支持性判断。"
        "始终遵守允许目录、当前姿态、物体等可执行条件及明确禁止项。"
        "在这些硬性约束内，优先选择符合角色人设和视觉行为风格的动作。"
        "只有没有合适的风格动作时，才采用本场景允许的通用兜底：进入或返回时问候，"
        "结束时告别，空闲时低打扰关注，主动交流或播客串场时自然伴随。"
        "只能选择本轮允许的真实 candidate_id，不输出 000。\n"
    )
