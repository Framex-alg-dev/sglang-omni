"""Wire event dispatch and commit scheduling."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Any, Literal

from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    DEFAULT_ACTION_PROMPT_LOCALE,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.models.qwen3_omni.prompt_localization import PROMPT_LANGUAGE_BY_LOCALE
from sglang_omni.preprocessing.image import prepare_image_bytes_for_wire
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.embedded_tts import EmbeddedTTSConnection
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import (
    _json_audit_fields,
    _text_audit_fields,
)
from sglang_omni.serve.realtime.protocol.models import (
    ImageFrame,
    SessionActionCandidate,
    SessionActionCategory,
    SessionActionProfile,
    TurnBuffer,
)
from sglang_omni.serve.realtime.output_capabilities import SessionOutputCapabilities
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
    new_trace_id,
)

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.protocol.input import MultimodalTurnInputMixin


class ProtocolDispatchComponent:
    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        turn_id = payload.get("turn_id")
        trace_id = None
        if self.active_turn is not None and turn_id == self.active_turn.turn_id:
            trace_id = self.active_turn.trace_id
        emit_structured_log(
            "protocol",
            "ws_event_received",
            session_id=payload.get("session_id") or self.session_id,
            turn_id=turn_id,
            trace_id=trace_id,
            ws_event_type=event_type,
            payload_keys=sorted(str(key) for key in payload),
        )
        normalized = self._normalize_wire_event(payload)
        normalized_type = normalized["type"]
        handlers = {
            "session.start": self.handle_session_start,
            "turn.start": self.handle_turn_start,
            "input_audio.append": self.handle_audio_append,
            "input_image.append": self.handle_image_append,
            "turn.text.update": self.handle_text_update,
            "turn.commit": self.handle_turn_commit,
            "turn.cancel": self.handle_turn_cancel,
            "knowledge.script.event": self.handle_knowledge_script_event,
            "session.close": self.handle_session_close,
        }
        handler = handlers.get(normalized_type)
        if handler is None:
            raise ValueError(f"unsupported event type: {event_type!r}")
        if normalized_type == "turn.commit":
            await self._dispatch_turn_commit(normalized)
            return
        await handler(normalized)


    async def _dispatch_turn_commit(self, event: dict[str, Any]) -> None:
        """Start committed-turn inference without blocking WebSocket input."""
        turn = self._require_collecting_turn(event)
        task = asyncio.create_task(
            self.handle_turn_commit(event),
            name=f"session-action-{self.session_id}-{turn.turn_id}",
        )
        turn.inference_task = task
        await asyncio.sleep(0)
        if task.done():
            await task


MultimodalDispatchMixin = ProtocolDispatchComponent

