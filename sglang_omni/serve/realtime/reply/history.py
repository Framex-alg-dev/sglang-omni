"""Recent reply history selection and persistence."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
import uuid
from contextlib import aclosing
from typing import Any, Literal

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import (
    _summarize_media,
    _text_audit_fields,
)
from sglang_omni.serve.realtime.protocol.models import (
    ProvisionalReplyState,
    ReplyHistoryRouteResult,
    ReplyHistoryTurn,
    ReplySpeechModeResult,
    ReplyTTSState,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    """Resolve the established façade-level diagnostics hook lazily."""

    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)
class ReplyHistoryComponent:
    @staticmethod
    def _suppress_reply_history_for_turn(
        turn: TurnBuffer, audios: list[str]
    ) -> bool:
        """Keep audio-only user turns independent from prior reply examples.

        Without a current transcript, the service cannot reliably decide
        whether the new utterance refers to an earlier exchange. Forwarding
        historical user audio and assistant replies in that case can make one
        plausible response reinforce itself across otherwise unrelated turns.
        Explicit current text remains the low-cost signal that permits history.
        """
        if turn.turn_origin == TURN_ORIGIN_PROACTIVE:
            # Proactive scenes are server-planned, not conversational queries.
            # Raw recent replies otherwise cause welcome/reminder text to copy
            # itself across unrelated proactive events.  character_proactive
            # receives a separately bounded, user-backed memory block.
            return True
        return bool(audios) and not (
            isinstance(turn.text, str) and bool(turn.text.strip())
        )


    @staticmethod
    def _normalized_reply_text(text: str) -> str:
        return "".join(
            character.casefold()
            for character in text
            if character.isalnum()
        )


    @classmethod
    def _reply_history_assistant_signature(
        cls, history_turn: ReplyHistoryTurn
    ) -> str | None:
        assistant_text = "".join(
            str(message.get("content", ""))
            for message in history_turn.messages
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        )
        normalized = cls._normalized_reply_text(assistant_text)
        return normalized or None


    def _visible_reply_history_turns(self) -> list[ReplyHistoryTurn]:
        """Return the last two history turns without repeated replies.

        Keeping only the newest occurrence prevents one bad assistant sentence
        from appearing several times in the next request. The chronological
        window is bounded before de-duplication so an old, merely distinct
        reply can never be pulled back into a current request.
        """
        selected: list[ReplyHistoryTurn] = []
        seen_assistant_replies: set[str] = set()
        recent = [
            history_turn
            for history_turn in self.reply_history_turns
            if history_turn.model_visible
            and history_turn.eligible_for_user_followup
        ][-MAX_REPLY_HISTORY_TURNS:]
        for history_turn in reversed(recent):
            signature = self._reply_history_assistant_signature(history_turn)
            if signature is not None and signature in seen_assistant_replies:
                continue
            if signature is not None:
                seen_assistant_replies.add(signature)
            selected.append(history_turn)
        selected.reverse()
        return selected


    @staticmethod
    def _select_reply_user_camera_images(
        images: list[Any], image_roles: list[str]
    ) -> tuple[list[Any], list[str]]:
        if len(images) != len(image_roles):
            raise ValueError("reply images and image_roles must have the same length")
        selected = [
            (image, role)
            for image, role in zip(images, image_roles, strict=True)
            if role == IMAGE_ROLE_USER_CAMERA
        ]
        # Keep enough ordered current-turn evidence for explicit comparisons
        # without allowing a long camera stream to dominate the request.
        selected = selected[-MAX_REPLY_CURRENT_IMAGES:]  # noqa: F405
        return (
            [image for image, _ in selected],
            [role for _, role in selected],
        )


    def _reply_user_camera_context_part(self) -> dict[str, str]:
        return {
            "type": "text",
            "text": self._prompt(
                zh=(
                    "[当前用户摄像头图片]紧随其后的一张或多张图片来自本轮用户摄像头，"
                    "按采集顺序排列，只表示用户及其周围环境，不表示当前角色自身的"
                    "外观、姿势、动作或状态。"
                ),
                en=(
                    "[Current user-camera images] The one or more images immediately "
                    "following this text come from the user's camera in this turn and are "
                    "ordered by capture. They represent only the user and their surroundings, "
                    "not the current character's appearance, pose, actions, or state."
                ),
            ),
        }


    def _append_reply_history(
        self,
        turn: TurnBuffer,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        reply_text: str | None,
        *,
        model_visible: bool = True,
        history_kind: str = "reply",
    ) -> None:
        if not reply_text:
            return
        if len(images) != len(image_roles):
            raise ValueError("reply images and image_roles must have the same length")
        if turn.turn_origin == TURN_ORIGIN_USER:
            messages = [
                {
                    "role": "user",
                    "content": self._reply_history_user_content(
                        audios,
                        [],
                        [],
                        turn.text,
                    ),
                },
                {"role": "assistant", "content": reply_text},
            ]
        else:
            messages = [{"role": "assistant", "content": reply_text}]
        if turn.turn_origin == TURN_ORIGIN_PROACTIVE and history_kind == "reply":
            history_kind = {
                "session_enter": "proactive_session_enter",
                "idle_timeout": "proactive_idle_timeout",
                "user_returned": "proactive_user_returned",
                "character_proactive": "proactive_character",
                "session_ending": "proactive_session_ending",
            }.get(turn.trigger or "", "proactive_custom")
        self.reply_history_turns.append(
            ReplyHistoryTurn(
                turn_id=turn.turn_id,
                messages=messages,
                audios=list(audios) if turn.turn_origin == TURN_ORIGIN_USER else [],
                # Both camera roles represent transient visual state. Keep
                # only the spoken/text exchange in reply history so an old
                # frame cannot become evidence for a later question.
                images=[],
                image_roles=[],
                model_visible=model_visible,
                history_kind=history_kind,
                eligible_for_user_followup=model_visible,
                eligible_for_proactive_planning=(
                    model_visible and turn.turn_origin == TURN_ORIGIN_USER
                ),
            )
        )


    def _reply_history_user_content(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
    ) -> list[dict[str, Any]]:
        if len(images) != len(image_roles):
            raise ValueError(
                "reply history images and image_roles must have the same length"
            )
        parts: list[dict[str, Any]] = []
        if image_roles:
            parts.append(self._reply_user_camera_context_part())
            parts.extend({"type": "image"} for _ in image_roles)
        parts.extend({"type": "audio"} for _ in audios)
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        return parts


MultimodalReplyHistoryMixin = ReplyHistoryComponent
