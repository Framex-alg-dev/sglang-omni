"""Generated, provided, and pure-action reply execution."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from contextlib import aclosing
from typing import Any, Literal

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
    ReplyTTSState,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


class ReplyGenerationComponent:
    """Model reply generation and terminal response events."""

    async def _run_generated_reply(
        self,
        turn: TurnBuffer,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        category: SessionActionCategory | None,
        *,
        support_status: str = "supported",
        provisional: ProvisionalReplyState | None = None,
        history_route: ReplyHistoryRouteResult | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._ensure_turn_processing(turn)
        request_id = f"{turn.request_base}-reply"
        response_id = (
            provisional.response_id
            if provisional is not None
            else f"resp-{uuid.uuid4().hex}"
        )
        if provisional is not None:
            provisional.request_id = request_id
        request_build_started = time.perf_counter()
        emit_structured_log(
            "performance",
            "model_request_build_begin",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            request_id=request_id,
            modality_count=len(self.modalities),
        )
        request, reply_forwarded_image_roles = self._build_reply_request(
            turn,
            audios,
            images,
            image_roles,
            category,
            support_status=support_status,
            history_route=history_route,
        )
        emit_structured_log(
            "performance",
            "model_request_build_end",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            request_id=request_id,
            message_count=len(request.messages or []),
            elapsed_ms=round((time.perf_counter() - request_build_started) * 1000, 3),
        )
        effective_system_prompt = next(
            (
                message.content
                for message in request.messages or []
                if message.role == "system" and isinstance(message.content, str)
            ),
            None,
        )
        system_prompt_audit = _text_audit_fields(
            "system_prompt", effective_system_prompt
        )
        diagnostic_messages = [message.to_dict() for message in request.messages or []]
        if not self.log_full_instructions:
            for message in diagnostic_messages:
                if message.get("role") == "system":
                    message["content"] = "<redacted; see system_prompt_sha256>"
        started = (
            provisional.started_at if provisional is not None else time.perf_counter()
        )
        first_token_ms: float | None = None
        first_delta_after_commit_ms: float | None = None
        delta_count = 0
        text_parts: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        tts_state: ReplyTTSState | None = None
        emit_structured_log(
            "diagnostic",
            "reply_logical_input",
            component="api",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            response_id=response_id,
            reply_source="generated",
            provisional=provisional is not None,
            turn_origin=turn.turn_origin,
            instructions_applied=bool(effective_system_prompt),
            effective_system_prompt=(
                effective_system_prompt if self.log_full_instructions else None
            ),
            **system_prompt_audit,
            messages=diagnostic_messages,
            selected_category=(
                {
                    "category_id": category.category_id,
                    "source_label": category.source_label,
                    "short_definition": category.short_definition,
                }
                if category is not None
                else None
            ),
            support_status=support_status,
            sampling=request.sampling.to_dict(),
            output_modalities=list(request.output_modalities or []),
            current_audio=_summarize_media(audios),
            current_images=_summarize_media([frame.data_uri for frame in turn.images]),
            current_image_roles=list(image_roles),
            received_image_roles=list(image_roles),
            reply_forwarded_image_roles=list(reply_forwarded_image_roles),
            reply_filtered_avatar_image_count=image_roles.count(
                IMAGE_ROLE_AVATAR_STATE
            ),
            reply_filtered_stale_user_camera_image_count=max(
                0, image_roles.count(IMAGE_ROLE_USER_CAMERA) - 1
            ),
            user_camera_present=bool(reply_forwarded_image_roles),
            reply_history_turn_count=len(self.reply_history_turns),
            reply_history_forwarded_turn_count=request.metadata.get(
                "reply_history_forwarded_turn_count", 0
            ),
            reply_history_suppressed_for_audio_only=request.metadata.get(
                "reply_history_suppressed_for_audio_only", False
            ),
            reply_history_route_decision=request.metadata.get(
                "reply_history_route_decision"
            ),
            reply_mode=request.metadata.get("reply_mode"),
            reply_history_route_ms=request.metadata.get(
                "reply_history_route_ms", 0.0
            ),
            reply_history_route_confidence_margin=request.metadata.get(
                "reply_history_route_confidence_margin"
            ),
            reply_history_route_fallback_reason=request.metadata.get(
                "reply_history_route_fallback_reason"
            ),
            last_executed_action=self._executed_action_log_fields(
                self.last_executed_action
            ),
            last_user_executed_action=self._executed_action_log_fields(
                self.last_user_executed_action
            ),
        )
        if provisional is None:
            await self.send(
                {
                    "type": "response.created",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "response": {
                        "id": response_id,
                        "status": "in_progress",
                        "source": "generated",
                    },
                }
            )
            created_after_commit_ms = self._after_commit_ms(turn)
        else:
            created_after_commit_ms = provisional.created_after_commit_ms
        tts_state = self._start_reply_tts(
            turn, response_id=response_id, provisional=provisional
        )
        self._register_turn_request(turn, request_id)
        emit_structured_log(
            "reply",
            "reply_submitted",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            response_id=response_id,
            reply_source="generated",
            provisional=provisional is not None,
            instructions_applied=bool(effective_system_prompt),
            **system_prompt_audit,
            created_after_commit_ms=created_after_commit_ms,
        )
        try:
            completion_stream = getattr(self.client, "completion_stream", None)
            if callable(completion_stream):
                emit_structured_log(
                    "performance",
                    "model_stream_submit_begin",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    request_id=request_id,
                    backend=type(self.client).__name__,
                )
                stream = completion_stream(request, request_id=request_id)
                async with aclosing(stream):
                    iterator = stream.__aiter__()
                    first_chunk_received = False
                    while True:
                        try:
                            chunk = await self._next_reply_chunk(
                                iterator,
                                tts_state=tts_state,
                                request_id=request_id,
                            )
                        except StopAsyncIteration:
                            break
                        if not first_chunk_received:
                            first_chunk_received = True
                            emit_structured_log(
                                "performance",
                                "model_stream_submit_accepted",
                                session_id=self.session_id,
                                turn_id=turn.turn_id,
                                trace_id=turn.trace_id,
                                request_id=request_id,
                            )
                            emit_structured_log(
                                "performance",
                                "model_first_chunk_received",
                                session_id=self.session_id,
                                turn_id=turn.turn_id,
                                trace_id=turn.trace_id,
                                request_id=request_id,
                                modality=chunk.modality,
                            )
                        self._ensure_turn_processing(turn)
                        if chunk.modality == "text" and chunk.text:
                            is_first_delta = first_token_ms is None
                            if is_first_delta:
                                first_token_ms = (
                                    time.perf_counter() - started
                                ) * 1000.0
                            text_parts.append(chunk.text)
                            if provisional is None:
                                await self.send(
                                    {
                                        "type": "response.text.delta",
                                        "session_id": self.session_id,
                                        "turn_id": turn.turn_id,
                                        "response_id": response_id,
                                        "delta": chunk.text,
                                    }
                                )
                                if is_first_delta:
                                    emit_structured_log(
                                        "performance",
                                        "response_first_text_delta_sent",
                                        session_id=self.session_id,
                                        turn_id=turn.turn_id,
                                        trace_id=turn.trace_id,
                                        request_id=request_id,
                                        response_id=response_id,
                                        chars=len(chunk.text),
                                        provisional=False,
                                    )
                                await self._enqueue_reply_tts_text(
                                    tts_state, chunk.text
                                )
                            else:
                                await self._send_provisional_reply_delta(
                                    turn, provisional, chunk.text
                                )
                                if is_first_delta:
                                    emit_structured_log(
                                        "performance",
                                        "response_first_text_delta_sent",
                                        session_id=self.session_id,
                                        turn_id=turn.turn_id,
                                        trace_id=turn.trace_id,
                                        request_id=request_id,
                                        response_id=response_id,
                                        chars=len(chunk.text),
                                        provisional=True,
                                    )
                                await self._enqueue_reply_tts_text(
                                    tts_state, chunk.text
                                )
                            delta_count += 1
                            if is_first_delta:
                                first_delta_after_commit_ms = self._after_commit_ms(
                                    turn
                                )
                                if (
                                    provisional is not None
                                    and provisional.first_delta_after_commit_ms is None
                                ):
                                    provisional.first_delta_after_commit_ms = (
                                        first_delta_after_commit_ms
                                    )
                                emit_structured_log(
                                    "reply",
                                    "reply_first_token",
                                    session_id=self.session_id,
                                    turn_id=turn.turn_id,
                                    trace_id=turn.trace_id,
                                    request_id=request_id,
                                    ttft_ms=round(first_token_ms, 3),
                                    first_delta_after_commit_ms=(
                                        first_delta_after_commit_ms
                                    ),
                                )
                                emit_structured_log(
                                    "performance",
                                    "model_first_text_delta_received",
                                    session_id=self.session_id,
                                    turn_id=turn.turn_id,
                                    trace_id=turn.trace_id,
                                    request_id=request_id,
                                    response_id=response_id,
                                    chars=len(chunk.text),
                                )
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                            if chunk.usage is not None:
                                usage = chunk.usage.to_dict()
            else:
                result = await self.client.completion(request, request_id=request_id)
                if result.text:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                    text_parts.append(result.text)
                    if provisional is None:
                        await self.send(
                            {
                                "type": "response.text.delta",
                                "session_id": self.session_id,
                                "turn_id": turn.turn_id,
                                "response_id": response_id,
                                "delta": result.text,
                            }
                        )
                        await self._enqueue_reply_tts_text(tts_state, result.text)
                    else:
                        await self._send_provisional_reply_delta(
                            turn, provisional, result.text
                        )
                        await self._enqueue_reply_tts_text(tts_state, result.text)
                    delta_count = 1
                    first_delta_after_commit_ms = self._after_commit_ms(turn)
                    if (
                        provisional is not None
                        and provisional.first_delta_after_commit_ms is None
                    ):
                        provisional.first_delta_after_commit_ms = (
                            first_delta_after_commit_ms
                        )
                    emit_structured_log(
                        "reply",
                        "reply_first_token",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        request_id=request_id,
                        ttft_ms=round(first_token_ms, 3),
                        first_delta_after_commit_ms=first_delta_after_commit_ms,
                    )
                finish_reason = result.finish_reason
                if result.usage is not None:
                    usage = result.usage.to_dict()
            reply_text = "".join(text_parts)
            total_ms = (time.perf_counter() - started) * 1000.0
            emit_structured_log(
                "performance",
                "model_text_stream_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
                response_id=response_id,
                delta_count=delta_count,
                text_chars=sum(len(part) for part in text_parts),
                finish_reason=finish_reason,
                elapsed_ms=round(total_ms, 3),
            )
            self._ensure_turn_processing(turn)
            if provisional is None:
                done_timing = await self._send_reply_done(
                    turn,
                    response_id=response_id,
                    text=reply_text,
                    source="generated",
                    finish_reason=finish_reason,
                    usage=usage,
                    tts_state=tts_state,
                )
            else:
                done_timing = await self._finish_provisional_reply(
                    turn,
                    provisional,
                    finish_reason=finish_reason,
                    usage=usage,
                )
            text_done_after_commit_ms = done_timing["text_done_after_commit_ms"]
            stream_duration_ms = (
                round(
                    text_done_after_commit_ms - first_delta_after_commit_ms,
                    3,
                )
                if text_done_after_commit_ms is not None
                and first_delta_after_commit_ms is not None
                else None
            )
            completion_tokens = (
                usage.get("completion_tokens") if usage is not None else None
            )
            if provisional is not None:
                timing = self._provisional_reply_timing(provisional, total_ms=total_ms)
            else:
                timing = {
                    "source": "generated",
                    "ttft_ms": round(first_token_ms or total_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "chars": len(reply_text),
                    "created_after_commit_ms": created_after_commit_ms,
                    "first_delta_after_commit_ms": first_delta_after_commit_ms,
                    "text_done_after_commit_ms": text_done_after_commit_ms,
                    "response_done_after_commit_ms": done_timing[
                        "response_done_after_commit_ms"
                    ],
                    "stream_duration_ms": stream_duration_ms,
                    "delta_count": delta_count,
                    "completion_tokens": completion_tokens,
                    "provisional": False,
                    "provisional_status": None,
                    "provisional_done_after_commit_ms": None,
                    "resolution_reason": None,
                }
            emit_structured_log(
                "reply",
                "reply_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
                response_id=response_id,
                logical_request_id=turn.request_base,
                output_text=reply_text,
                finish_reason=finish_reason,
                usage=usage,
                **timing,
            )
            return reply_text, timing
        except asyncio.CancelledError:
            if provisional is not None:
                await self._mark_provisional_reply_terminal(
                    provisional, cancelled=True
                )
            emit_structured_log(
                "reply",
                "reply_cancelled",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                request_id=request_id,
            )
            raise
        except Exception as exc:
            if provisional is not None:
                await self._mark_provisional_reply_terminal(
                    provisional, failed=True
                )
            emit_structured_log(
                "error",
                "reply_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                response_id=response_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            await self._abort_reply_tts(tts_state)
            self._unregister_turn_request(turn, request_id)
    async def _run_provided_reply(
        self,
        turn: TurnBuffer,
        text: str,
        *,
        provisional: ProvisionalReplyState | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._ensure_turn_processing(turn)
        response_id = (
            provisional.response_id
            if provisional is not None
            else f"resp-{uuid.uuid4().hex}"
        )
        started = (
            provisional.started_at if provisional is not None else time.perf_counter()
        )
        tts_state: ReplyTTSState | None = None
        if provisional is not None:
            if text:
                tts_state = self._start_reply_tts(
                    turn, response_id=response_id, provisional=provisional
                )
            if text:
                await self._send_provisional_reply_delta(turn, provisional, text)
                await self._enqueue_reply_tts_text(tts_state, text)
            try:
                await self._finish_provisional_reply(
                    turn,
                    provisional,
                    finish_reason="provided",
                    usage=None,
                )
            finally:
                await self._abort_reply_tts(tts_state)
            total_ms = (time.perf_counter() - started) * 1000.0
            timing = self._provisional_reply_timing(provisional, total_ms=total_ms)
            emit_structured_log(
                "reply",
                "provided_reply_used",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=response_id,
                output_text=text,
                finish_reason="provided",
                usage=None,
                instructions_applied=False,
                **_text_audit_fields("instructions", self.instructions),
                **timing,
            )
            return text, timing
        await self.send(
            {
                "type": "response.created",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response": {
                    "id": response_id,
                    "status": "in_progress",
                    "source": "provided",
                },
            }
        )
        created_after_commit_ms = self._after_commit_ms(turn)
        await self.send(
            {
                "type": "response.text.delta",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response_id": response_id,
                "delta": text,
            }
        )
        if text:
            tts_state = self._start_reply_tts(turn, response_id=response_id)
        await self._enqueue_reply_tts_text(tts_state, text)
        first_delta_after_commit_ms = self._after_commit_ms(turn)
        try:
            done_timing = await self._send_reply_done(
                turn,
                response_id=response_id,
                text=text,
                source="provided",
                finish_reason="provided",
                usage=None,
                tts_state=tts_state,
            )
        finally:
            await self._abort_reply_tts(tts_state)
        total_ms = (time.perf_counter() - started) * 1000.0
        text_done_after_commit_ms = done_timing["text_done_after_commit_ms"]
        timing = {
            "source": "provided",
            "ttft_ms": 0.0,
            "total_ms": round(total_ms, 3),
            "chars": len(text),
            "created_after_commit_ms": created_after_commit_ms,
            "first_delta_after_commit_ms": first_delta_after_commit_ms,
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": done_timing[
                "response_done_after_commit_ms"
            ],
            "stream_duration_ms": (
                round(
                    text_done_after_commit_ms - first_delta_after_commit_ms,
                    3,
                )
                if text_done_after_commit_ms is not None
                and first_delta_after_commit_ms is not None
                else None
            ),
            "delta_count": 1,
            "completion_tokens": None,
        }
        emit_structured_log(
            "reply",
            "provided_reply_used",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            response_id=response_id,
            output_text=text,
            finish_reason="provided",
            usage=None,
            instructions_applied=False,
            **_text_audit_fields("instructions", self.instructions),
            **timing,
        )
        return text, timing
    async def _run_empty_pure_action_reply(
        self,
        turn: TurnBuffer,
        *,
        provisional: ProvisionalReplyState,
    ) -> tuple[str, dict[str, Any]]:
        """Complete a user pure-action reply without model generation or TTS."""
        self._ensure_turn_processing(turn)
        await self._finish_provisional_reply(
            turn,
            provisional,
            finish_reason="pure_action",
            usage=None,
        )
        timing = self._provisional_reply_timing(provisional)
        emit_structured_log(
            "reply",
            "pure_action_empty_reply_used",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            response_id=provisional.response_id,
            output_text="",
            finish_reason="pure_action",
            usage=None,
            **timing,
        )
        return "", timing
    @staticmethod
    def _validate_pure_action_short_reply(text: str) -> tuple[str, str | None]:
        """Return a safe short social response or an empty fallback.

        Validation is intentionally deterministic and conservative because the
        result is spoken to the user. Invalid model output is never partially
        streamed; it degrades to the already-supported empty response.
        """
        normalized = text.strip()
        if not normalized:
            return "", None
        if len(normalized) > PURE_ACTION_REPLY_MAX_CHARS:
            return "", "too_long"
        if "\n" in normalized or "\r" in normalized:
            return "", "multiline"
        if re.search(r"[*_`#<>\[\]{}（）()【】]", normalized):
            return "", "markup_or_stage_direction"
        forbidden_phrases = (
            "动作",
            "我正",
            "我在",
            "我已经",
            "我刚刚",
            "我有点",
            "正在",
            "已经",
            "完成",
            "做出",
            "执行",
            "轻轻抬",
            "镜头",
            "姿势",
            "表情",
            "眨眼",
            "靠近镜头",
            "望向镜头",
            "飞吻",
            "比心",
            "翻跟头",
            "亲吻",
            "芭蕾",
            "挥手",
            "抬头",
            "转身",
            "手指",
            "双手",
            "单手",
            "数字人",
            "模型",
            "没有实体",
            "无法做",
            "不能做",
            "系统",
        )
        matched = next(
            (phrase for phrase in forbidden_phrases if phrase in normalized), None
        )
        if matched is not None:
            return "", f"forbidden_phrase:{matched}"
        return normalized, None
    def _pure_action_reply_validation_system_prompt(self) -> str:
        return self._prompt(
            zh=(
                "你是纯动作请求短回应的合规分类器。当前输入包含用户本轮请求，以及一条"
                "待校验的角色短回应。只判断短回应本身，选择一个结果："
                "V0=合法：一句简短、自然、面向用户的接受、配合或互动回应，不复述动作，"
                "不增加独立话题；V1=描述动作名称、身体部位、物体、方向、姿势、镜头、表情、"
                "执行过程、进行状态或完成状态；V2=编造角色当前情绪、感受、身体状态、环境或"
                "未提供的情境；V3=与当前请求无关、主动提出额外问题、延续其他话题或进行不必要"
                "展开；V4=谈论数字人、模型、实体能力、动作系统等技术身份，拒绝身体动作，或"
                "包含 Markdown、舞台说明、标签、解释、换行等非自然口语格式。"
                "人设语气本身不是违规；但人设不能成为编造当前状态或转移话题的理由。"
                "示例：‘给你呀～接住哦’→V0；‘好呀’→V0；‘我正在做飞吻呢’→V1；"
                "‘我有点害羞呢’→V2；‘你好呀，今天过得怎么样？’→V3；"
                "‘我是数字人，无法完成’→V4。只输出 V0、V1、V2、V3 或 V4，不要解释。"
            ),
            en=(
                "Classify a proposed brief spoken response to a pure-action request. The "
                "current input contains the user's request and the proposed character response. "
                "Choose exactly one result: V0=valid brief acceptance, cooperation, or "
                "user-directed interaction that does not repeat the action or add a new topic; "
                "V1=action, body part, object, direction, pose, camera, expression, execution, "
                "progress, or completion narration; V2=invented current emotion, feeling, body "
                "state, environment, or unsupported situation; V3=unrelated content, an extra "
                "question, continuation of another topic, or unnecessary expansion; V4=technical "
                "identity, embodiment or action-system discussion, refusal of a physical action, "
                "Markdown, stage directions, labels, explanation, line breaks, or other non-spoken "
                "format. Persona tone is allowed but cannot justify inventing state or changing "
                "the topic. Examples: 'Here you go—catch it' and 'Sure' -> V0; 'I am doing the "
                "kiss now' -> V1; 'I feel shy now' -> V2; 'Hi, how was your day?' -> V3; "
                "'I am a digital model and cannot do that' -> V4. Output only V0, V1, V2, V3, "
                "or V4 without explanation."
            ),
        )
    async def _validate_pure_action_short_reply_semantics(
        self,
        turn: TurnBuffer,
        audios: list[str],
        text: str,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Score a generated short response against a fixed semantic violation set."""
        if not text:
            return True, None, {}
        request_id = f"{turn.request_base}-pure-action-reply-validation"
        system_prompt = self._pure_action_reply_validation_system_prompt()
        current_text_parts: list[str] = []
        if isinstance(turn.text, str) and turn.text.strip():
            current_text_parts.append(
                self._prompt(
                    zh=f"[当前用户文本]\n{turn.text.strip()}",
                    en=f"[Current user text]\n{turn.text.strip()}",
                )
            )
        current_text_parts.append(
            self._prompt(
                zh=f"[待校验的角色短回应]\n{text}",
                en=f"[Proposed brief character response]\n{text}",
            )
        )
        request = ActionSuffixScoreRequest(
            request_id=request_id,
            model=self.model_name,
            prefix=self._prompt(zh="分类结果：", en="Classification result:"),
            system_prompt=system_prompt,
            current_text="\n".join(current_text_parts),
            output_prompt="",
            language=self.language,
            candidates=[
                ActionScoreCandidate(candidate_id=candidate_id, suffix=candidate_id)
                for candidate_id in ("V0", "V1", "V2", "V3", "V4")
            ],
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=[],
            image_roles=[],
            sample_rate=16000,
            micro_batch_size=5,
            session_id=self.session_id,
            stage=PURE_ACTION_REPLY_VALIDATION_STAGE,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            text_role=turn.text_role,
            history=[],
            history_audios=[],
            history_images=[],
            prefix_cache_namespace=(
                f"pure-action-reply-validation:v1:{self.locale}:"
                f"{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
            ),
            cache_static_system_only=True,
        )
        try:
            timeout_s = float(
                os.environ.get(
                    PURE_ACTION_REPLY_VALIDATION_TIMEOUT_ENV,
                    DEFAULT_PURE_ACTION_REPLY_VALIDATION_TIMEOUT_S,
                )
            )
        except (TypeError, ValueError):
            timeout_s = DEFAULT_PURE_ACTION_REPLY_VALIDATION_TIMEOUT_S
        timeout_s = max(0.05, timeout_s)
        started = time.perf_counter()
        self._register_turn_request(turn, request_id)
        try:
            result = await asyncio.wait_for(
                self.client.score_action_suffixes(request), timeout=timeout_s
            )
            self._ensure_turn_processing(turn)
            matched = {
                score.candidate_id: float(score.mean_logprob)
                for score in result.scores
                if score.candidate_id in {"V0", "V1", "V2", "V3", "V4"}
            }
            if not matched:
                raise ValueError("pure-action reply validation returned no V0-V4 scores")
            ranked = sorted(matched.items(), key=lambda item: item[1], reverse=True)
            winner = ranked[0][0]
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
            accepted = bool(
                winner == "V0"
                and (
                    margin is None
                    or margin >= PURE_ACTION_REPLY_VALIDATION_MIN_MARGIN
                )
            )
            reason = None if accepted else (
                f"semantic:{winner}"
                if winner != "V0"
                else "semantic:ambiguous"
            )
            metadata = {
                "winner": winner,
                "scores": matched,
                "confidence_margin": margin,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "prefix_cached": result.prefix_cached,
            }
            emit_structured_log(
                "reply",
                "pure_action_reply_semantic_validation_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                accepted=accepted,
                validation_reason=reason,
                **metadata,
            )
            return accepted, reason, metadata
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            reason = "semantic:timeout"
        except Exception as exc:
            reason = f"semantic:{type(exc).__name__}"
        finally:
            self._unregister_turn_request(turn, request_id)

        metadata = {
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "fallback_reason": reason,
        }
        emit_structured_log(
            "reply",
            "pure_action_reply_semantic_validation_fallback",
            level="warning",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            accepted=False,
            validation_reason=reason,
            **metadata,
        )
        return False, reason, metadata
    async def _run_pure_action_short_reply(
        self,
        turn: TurnBuffer,
        audios: list[str],
        *,
        provisional: ProvisionalReplyState,
        history_route: ReplyHistoryRouteResult | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate and validate a persona-aware pure-action social response.

        The model stream is buffered internally. Only a fully validated result
        is copied into the provisional response and TTS pipeline.
        """
        self._ensure_turn_processing(turn)
        request_id = f"{turn.request_base}-pure-action-reply"
        provisional.request_id = request_id
        # Action scoring owns any R3 cross-turn resolution. The spoken social
        # acknowledgement stays current-only so ordinary historical replies
        # cannot leak their topic or wording into this turn.
        pure_reply_route = ReplyHistoryRouteResult(
            decision=REPLY_HISTORY_CURRENT_ONLY,
            reply_mode=REPLY_MODE_PURE_ACTION,
            elapsed_ms=(history_route.elapsed_ms if history_route else 0.0),
            confidence_margin=(
                history_route.confidence_margin if history_route else None
            ),
            pure_action_ambiguous=(
                history_route.pure_action_ambiguous if history_route else False
            ),
            scores=(dict(history_route.scores) if history_route else {}),
            fallback_reason=(history_route.fallback_reason if history_route else None),
            stats=(dict(history_route.stats) if history_route else {}),
        )
        request, _ = self._build_reply_request(
            turn,
            audios,
            [],
            [],
            None,
            history_route=pure_reply_route,
        )
        request.metadata["task"] = "session_pure_action_reply"
        request.sampling.max_new_tokens = PURE_ACTION_REPLY_MAX_NEW_TOKENS
        started = time.perf_counter()
        text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason = "pure_action"
        validation_reason: str | None = None
        semantic_validation: dict[str, Any] = {}
        self._register_turn_request(turn, request_id)
        try:
            completion_stream = getattr(self.client, "completion_stream", None)
            if callable(completion_stream):
                stream = completion_stream(request, request_id=request_id)
                async with aclosing(stream):
                    async for chunk in stream:
                        self._ensure_turn_processing(turn)
                        if chunk.modality == "text" and chunk.text:
                            text_parts.append(chunk.text)
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                            if chunk.usage is not None:
                                usage = chunk.usage.to_dict()
            else:
                result = await self.client.completion(request, request_id=request_id)
                if result.text:
                    text_parts.append(result.text)
                finish_reason = result.finish_reason
                if result.usage is not None:
                    usage = result.usage.to_dict()
            safe_text, validation_reason = self._validate_pure_action_short_reply(
                "".join(text_parts)
            )
            if safe_text:
                (
                    semantically_valid,
                    semantic_reason,
                    semantic_validation,
                ) = await self._validate_pure_action_short_reply_semantics(
                    turn,
                    audios,
                    safe_text,
                )
                if not semantically_valid:
                    safe_text = ""
                    validation_reason = semantic_reason
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            safe_text = ""
            validation_reason = f"generation_error:{type(exc).__name__}"
            logger.warning(
                "[SESSION_ACTION_REALTIME] pure-action short reply failed "
                "session_id=%s turn_id=%s",
                self.session_id,
                turn.turn_id,
                exc_info=True,
            )
        finally:
            self._unregister_turn_request(turn, request_id)

        # The generated text has not been emitted yet. Reuse the provided-text
        # path only after validation so TTS never sees rejected content.
        text, timing = await self._run_provided_reply(
            turn,
            safe_text,
            provisional=provisional,
        )
        timing["source"] = "pure_action_generated"
        timing["validation_fallback_reason"] = validation_reason
        timing["semantic_validation"] = semantic_validation
        emit_structured_log(
            "reply",
            "pure_action_short_reply_completed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            response_id=provisional.response_id,
            output_text=text,
            raw_output_text="".join(text_parts),
            validation_fallback_reason=validation_reason,
            semantic_validation=semantic_validation,
            generation_ms=round((time.perf_counter() - started) * 1000.0, 3),
            finish_reason=finish_reason,
            usage=usage,
        )
        return text, timing
    async def _send_reply_done(
        self,
        turn: TurnBuffer,
        *,
        response_id: str,
        text: str,
        source: Literal["generated", "provided"],
        finish_reason: str,
        usage: dict[str, Any] | None,
        provisional_id: str | None = None,
        tts_state: ReplyTTSState | None = None,
    ) -> dict[str, float | None]:
        text_done_payload: dict[str, Any] = {
            "type": "response.text.done",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "response_id": response_id,
            "text": text,
        }
        if provisional_id is not None:
            text_done_payload["provisional_id"] = provisional_id
        if await self.send(text_done_payload):
            emit_structured_log(
                "performance",
                "response_text_done_sent",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                response_id=response_id,
                after_commit_ms=self._after_commit_ms(turn),
            )
        text_done_after_commit_ms = self._after_commit_ms(turn)
        if tts_state is not None:
            await self._finish_reply_tts(tts_state)
            self._ensure_turn_processing(turn)
        if self.output_capabilities.audio_enabled:
            last_audio_seq = (
                tts_state.next_audio_seq - 1 if tts_state is not None else 0
            )
            audio_done_sent = await self.send(
                {
                    "type": "response.audio.done",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "response_id": response_id,
                    "seq": last_audio_seq,
                    "audio": {
                        "format": "pcm16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                    },
                }
            )
            if audio_done_sent:
                emit_structured_log(
                    "performance",
                    "response_audio_done_sent",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    response_id=response_id,
                    last_seq=last_audio_seq,
                    after_commit_ms=self._after_commit_ms(turn),
                )
        response: dict[str, Any] = {
            "id": response_id,
            "status": "completed",
            "status_details": {"reason": finish_reason},
            "source": source,
            "output": [{"type": "text", "text": text}],
        }
        if usage is not None:
            response["usage"] = usage
        if provisional_id is not None:
            response["provisional_id"] = provisional_id
        response_done_sent = await self.send(
            {
                "type": "response.done",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response": response,
            }
        )
        if response_done_sent:
            emit_structured_log(
                "performance",
                "response_done_sent",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                response_id=response_id,
                outputs=list(self.output_capabilities.outputs),
                after_commit_ms=self._after_commit_ms(turn),
            )
        return {
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": self._after_commit_ms(turn),
        }


MultimodalReplyGenerationMixin = ReplyGenerationComponent
