"""Small immutable contracts for proactive scene policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProactiveScenePolicy:
    """Complete server behavior for one proactive trigger.

    Client scene text is an optional, lower-authority refinement.  These
    policies remain sufficient when the client supplies only ``trigger_type``.
    """

    trigger: str
    history_policy: Literal["none", "user_followup", "memory"]
    memory_policy: Literal["none", "proactive"]
    reply_policy_zh: str
    reply_policy_en: str
    default_action_guidance_zh: str
    default_action_guidance_en: str

    def reply_policy(self, language: str) -> str:
        return self.reply_policy_zh if language == "zh" else self.reply_policy_en

    def default_action_guidance(self, language: str) -> str:
        return (
            self.default_action_guidance_zh
            if language == "zh"
            else self.default_action_guidance_en
        )
