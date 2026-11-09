"""One multimodal message layout shared by serving and dataset export."""

from __future__ import annotations

import json

from .contracts import ActionDecisionRequest, DecisionChannel
from .catalog import ResolvedCatalog
from .prompt import build_user_prompt, uses_reference_prompt
from .speech_window import PERSONA_WINDOW_PROFILE, PERSONA_AUDIO_PROFILE


def user_message_content(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> list[dict[str, str]]:
    text = {"type": "text", "text": build_user_prompt(request, catalog)}
    media = [{"type": "audio" if item.kind == "audio" else "image"} for item in request.media]
    if request.runtime_context.get("prompt_profile") == PERSONA_AUDIO_PROFILE:
        descriptions = {
            "user_context_audio": "原用户问题音频，仅作为数字人本次回复的背景；不是当前目标片段，也不是新一轮用户输入，不为它单独召回。",
            "user_audio": "本轮用户原始音频，直接理解，不是数字人回复。",
            "user_camera": "用户侧画面，不是数字人画面。",
            "avatar_speech_context": "数字人前文音频，仅供理解，不为这段前文重新选择动作。",
            "avatar_speech_audio": "数字人本次目标语音focus，为这段待表达内容选择一个动作。",
        }
        if request.channel is DecisionChannel.EXPRESSION:
            descriptions.update({
                "avatar_speech_context": "数字人前文音频，仅供理解，不为这段前文重新选择表情。",
                "avatar_speech_audio": "数字人本次目标语音focus，为这段当前表达选择一个面部表情。",
            })
        parts = []
        for item, part in zip(request.media, media):
            evidence = {"source": item.evidence_role}
            if not request.simple_audio:
                evidence.update(start_ms=item.start_ms, end_ms=item.end_ms)
            evidence["description"] = descriptions[item.evidence_role]
            label = json.dumps(evidence,
                               ensure_ascii=False, separators=(",", ":"))
            parts.extend(({"type": "text", "text": "[MEDIA_EVIDENCE]\n" + label + "\n"},
                          part, {"type": "text", "text": "\n"}))
        return [*parts, text]
    if request.runtime_context.get("prompt_profile") == PERSONA_WINDOW_PROFILE:
        parts = []
        for item in media:
            parts.extend((item, {"type": "text", "text": "\n"}))
        return [*parts, text]
    return [*media, text] if uses_reference_prompt(request) else [text, *media]


def swift_user_text(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    """Use Swift's external-media placeholders, preserving all text bytes."""
    return "".join(
        item["text"] if item["type"] == "text" else f"<{item['type']}>"
        for item in user_message_content(request, catalog)
    )
