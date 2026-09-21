"""Independent current-avatar-frame analysis for Session Realtime."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import aclosing
from typing import Any

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.realtime.protocol.common import _completion_token_timing
from sglang_omni.serve.realtime.protocol.models import ImageFrame, TurnBuffer
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
)

logger = logging.getLogger(__name__)

AVATAR_STATE_FIELDS = ("pose", "gaze", "left_hand", "right_hand", "held_object")
AVATAR_STATE_ANALYSIS_TASK = "avatar_state_analysis"
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_SYSTEM_PROMPT = """你是数字人画面状态观察器。只描述图片中数字人此刻可见的状态，不推测意图。
只输出一个 JSON 对象，且必须只包含以下字符串字段：pose、gaze、left_hand、right_hand、held_object。
看不清或不存在的字段使用空字符串。不要输出 Markdown、解释或其他字段。"""


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


def _parse_avatar_state(text: str) -> dict[str, str]:
    candidate = text.strip()
    fenced = _JSON_FENCE_RE.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("avatar state analysis must return a JSON object")
    unknown = sorted(set(value) - set(AVATAR_STATE_FIELDS))
    if unknown:
        raise ValueError(
            "avatar state analysis returned unsupported fields: " + ", ".join(unknown)
        )
    normalized: dict[str, str] = {}
    for field_name in AVATAR_STATE_FIELDS:
        field_value = value.get(field_name, "")
        if not isinstance(field_value, str):
            raise ValueError(
                f"avatar state analysis field {field_name} must be a string"
            )
        normalized[field_name] = field_value.strip()
    return normalized


class AvatarStateAnalysisPipeline:
    """Run one opt-in visual state analysis as soon as the avatar frame arrives."""

    def _start_avatar_state_analysis(
        self,
        turn: TurnBuffer,
        frame: ImageFrame,
        *,
        received_at: float,
    ) -> None:
        if not self.avatar_state_analysis_enabled:
            return
        if frame.image_role != "avatar_state":
            return
        if turn.avatar_state_analysis_task is not None:
            return

        task = asyncio.create_task(
            self._run_avatar_state_analysis(turn, frame, received_at=received_at),
            name=f"avatar-state-analysis-{turn.turn_id}-{frame.seq}",
        )
        turn.avatar_state_analysis_task = task
        turn.branch_tasks.add(task)
        task.add_done_callback(turn.branch_tasks.discard)

    async def _run_avatar_state_analysis(
        self,
        turn: TurnBuffer,
        frame: ImageFrame,
        *,
        received_at: float,
    ) -> None:
        analysis_id = f"avatar-state-{uuid.uuid4().hex}"
        request_id = f"session-{self.session_id}-turn-{turn.turn_id}-{analysis_id}"
        turn.avatar_state_analysis_request_id = request_id
        request = GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=[
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "分析这张数字人当前帧，并按约定 JSON 返回。",
                        },
                    ],
                ),
            ],
            sampling=SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=160),
            output_modalities=["text"],
            metadata={
                "task": AVATAR_STATE_ANALYSIS_TASK,
                "images": [frame.data_uri],
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "image_seq": frame.seq,
            },
        )
        first_token_at: float | None = None
        text_parts: list[str] = []
        delta_count = 0
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        self._register_turn_request(turn, request_id)
        try:
            async def collect() -> None:
                nonlocal delta_count, finish_reason, first_token_at, usage
                completion_stream = getattr(self.client, "completion_stream", None)
                if callable(completion_stream):
                    stream = completion_stream(request, request_id=request_id)
                    async with aclosing(stream):
                        async for chunk in stream:
                            if self.active_turn is not turn or turn.phase not in {
                                "collecting",
                                "processing",
                            }:
                                raise asyncio.CancelledError
                            if chunk.modality == "text" and chunk.text:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                text_parts.append(chunk.text)
                                delta_count += 1
                            if chunk.finish_reason is not None:
                                finish_reason = chunk.finish_reason
                            if chunk.usage is not None:
                                usage = chunk.usage.to_dict()
                    return
                result = await self.client.completion(request, request_id=request_id)
                if result.text:
                    first_token_at = time.perf_counter()
                    text_parts.append(result.text)
                    delta_count = 1
                finish_reason = result.finish_reason
                if result.usage is not None:
                    usage = result.usage.to_dict()

            await asyncio.wait_for(
                collect(),
                timeout=self.avatar_state_analysis_timeout_s,
            )
            ready_at = time.perf_counter()
            output_text = "".join(text_parts)
            state = _parse_avatar_state(output_text)
            first_token_ms = (
                max(0.0, first_token_at - received_at) * 1000.0
                if first_token_at is not None
                else None
            )
            total_ms = max(0.0, ready_at - received_at) * 1000.0
            timing = {
                "image_received_to_first_token_ms": (
                    round(first_token_ms, 3)
                    if first_token_ms is not None
                    else None
                ),
                "image_received_to_ready_ms": round(total_ms, 3),
                **_completion_token_timing(
                    usage,
                    first_token_ms=first_token_ms,
                    total_ms=total_ms,
                ),
            }
            turn.avatar_state_analysis_result = state
            turn.avatar_state_analysis_timing = timing
            emit_structured_log(
                "performance",
                "avatar_state_analysis_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
                analysis_id=analysis_id,
                image_seq=frame.seq,
                delta_count=delta_count,
                text_chars=len(output_text),
                finish_reason=finish_reason,
                **timing,
            )
            await self.send(
                {
                    "type": "turn.avatar_state.ready",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "analysis_id": analysis_id,
                    "image_seq": frame.seq,
                    "avatar_state": state,
                    "timing": timing,
                }
            )
        except asyncio.CancelledError:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                try:
                    await abort(request_id)
                except Exception:
                    logger.warning(
                        "avatar state analysis abort failed session_id=%s turn_id=%s",
                        self.session_id,
                        turn.turn_id,
                        exc_info=True,
                    )
            raise
        except Exception as exc:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                try:
                    await abort(request_id)
                except Exception:
                    logger.warning(
                        "avatar state analysis abort after failure failed "
                        "session_id=%s turn_id=%s",
                        self.session_id,
                        turn.turn_id,
                        exc_info=True,
                    )
            elapsed_ms = round((time.perf_counter() - received_at) * 1000.0, 3)
            turn.avatar_state_analysis_error = type(exc).__name__
            logger.warning(
                "avatar state analysis failed session_id=%s turn_id=%s error_type=%s",
                self.session_id,
                turn.turn_id,
                type(exc).__name__,
                exc_info=True,
            )
            await self.send(
                {
                    "type": "turn.avatar_state.failed",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "analysis_id": analysis_id,
                    "image_seq": frame.seq,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "timing": {"image_received_to_failed_ms": elapsed_ms},
                }
            )
        finally:
            self._unregister_turn_request(turn, request_id)

    async def _wait_for_avatar_state_analysis(self, turn: TurnBuffer) -> None:
        task = turn.avatar_state_analysis_task
        if task is None or task.done():
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            return
        await asyncio.gather(task, return_exceptions=True)
