"""One multimodal message layout shared by serving and dataset export."""

from __future__ import annotations

import json

from .contracts import ActionDecisionRequest
from .catalog import ResolvedCatalog
from .prompt import build_user_prompt, uses_reference_prompt
from .speech_window import PERSONA_WINDOW_PROFILE, PERSONA_AUDIO_PROFILE


def user_message_content(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> list[dict[str, str]]:
    text = {"type": "text", "text": build_user_prompt(request, catalog)}
    media = [{"type": "audio" if item.kind == "audio" else "image"} for item in request.media]
    if request.runtime_context.get("prompt_profile") == PERSONA_AUDIO_PROFILE:
        labels = {
            "user_context_audio": ("USER_CONTEXT_AUDIO", "原用户问题，仅提供语境。"),
            "user_audio": ("USER_AUDIO", "本轮用户原始音频。"),
            "user_camera": ("USER_CAMERA", "按时间顺序提供的用户侧画面。"),
            "avatar_speech_context": ("AVATAR_HISTORY_AUDIO", "数字人已经说过的前文，仅提供语境。"),
            "avatar_speech_audio": ("AVATAR_CURRENT_AUDIO", "数字人当前待表达内容，是本次主要证据。"),
        }
        parts = []
        index = 0
        while index < len(request.media):
            role = request.media[index].evidence_role
            end = index + 1
            while end < len(request.media) and request.media[end].evidence_role == role:
                end += 1
            if role == "user_camera":
                # C666 learned ordered vision tasks with an explicit user-camera
                # boundary before every frame. Keep that semantic boundary while
                # omitting all old timing/window fields.
                label = json.dumps(
                    {"source": "user_camera", "description": "用户侧画面，不是数字人画面。"},
                    ensure_ascii=False, separators=(",", ":"),
                )
                for part in media[index:end]:
                    parts.extend(({"type": "text", "text": "[MEDIA_EVIDENCE]\n" + label + "\n"},
                                  part, {"type": "text", "text": "\n"}))
                index = end
                continue
            tag, description = labels[role]
            suffix = f" ordered_frames={end - index}" if role == "user_camera" else ""
            if request.requested_action is not None:
                description = "对话或画面背景；明确目标见 USER_REQUESTED_ACTION。"
            parts.append({"type": "text", "text": f"[{tag}{suffix}]\n{description}\n"})
            for part in media[index:end]:
                parts.extend((part, {"type": "text", "text": "\n"}))
            index = end
        if text["text"]:
            parts.append(text)
        return parts
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
