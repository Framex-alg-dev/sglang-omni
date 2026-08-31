"""Provisional reply promote/discard state machine."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import Any, Literal

from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.models import (
    ProvisionalReplyState,
    ReplyTTSState,
    TurnBuffer,
)
from sglang_omni.serve.realtime.components import compose_components
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)

from sglang_omni.serve.realtime.reply.tts import ReplyTTSComponent


@compose_components(ReplyTTSComponent)
class ProvisionalReplyComponent:
    async def _create_provisional_reply(
        self,
        turn: TurnBuffer,
        *,
        source: Literal["generated", "provided"],
    ) -> ProvisionalReplyState:
        state = ProvisionalReplyState(
            response_id=f"resp-{uuid.uuid4().hex}",
            source=source,
            started_at=time.perf_counter(),
            created_after_commit_ms=None,
        )
        turn.provisional_reply = state
        await self.send(
            {
                "type": "response.provisional.created",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "provisional_id": state.response_id,
                "response": {
                    "id": state.response_id,
                    "status": "in_progress",
                    "source": source,
                    "provisional": True,
                },
            }
        )
        state.created_after_commit_ms = self._after_commit_ms(turn)
        emit_structured_log(
            "reply",
            "provisional_reply_created",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            response_id=state.response_id,
            reply_source=source,
            created_after_commit_ms=state.created_after_commit_ms,
        )
        return state

    async def _send_provisional_reply_delta(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        delta: str,
    ) -> None:
        if not delta:
            return
        async with state.lock:
            if state.status == "discarded":
                return
            is_first_delta = state.first_token_ms is None
            if is_first_delta:
                state.first_token_ms = (time.perf_counter() - state.started_at) * 1000.0
            state.text_parts.append(delta)
            state.delta_count += 1
            buffered_text = "".join(state.text_parts)
            if buffered_text.strip() and state.first_nonempty_at is None:
                state.first_nonempty_at = time.perf_counter()
                state.content_available.set()
            if (
                SYSTEM_REPLY_SENTENCE_END_RE.search(buffered_text)
                or len(buffered_text) >= SYSTEM_REPLY_PREFIX_MAX_CHARS
            ):
                state.sentence_ready.set()
            event_type = (
                "response.provisional.text.delta"
                if state.status == "pending"
                else "response.text.delta"
            )
            payload: dict[str, Any] = {
                "type": event_type,
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response_id": state.response_id,
                "provisional_id": state.response_id,
                "seq": state.delta_count,
                "delta": delta,
            }
            await self.send(payload)
            if is_first_delta:
                state.first_delta_after_commit_ms = self._after_commit_ms(turn)
                emit_structured_log(
                    "reply",
                    "provisional_reply_first_token",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                    response_id=state.response_id,
                    ttft_ms=round(state.first_token_ms, 3),
                    first_delta_after_commit_ms=(state.first_delta_after_commit_ms),
                )

    async def _finish_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        finish_reason: str,
        usage: dict[str, Any] | None,
    ) -> dict[str, float | None]:
        should_send_official_done = False
        was_pending = False
        async with state.lock:
            state.completed = True
            state.finish_reason = finish_reason
            state.usage = usage
            text = "".join(state.text_parts)
            state.content_available.set()
            state.sentence_ready.set()
            if state.status == "discarded":
                return {
                    "text_done_after_commit_ms": None,
                    "response_done_after_commit_ms": None,
                }
            state.provisional_done_after_commit_ms = self._after_commit_ms(turn)
            if state.status == "pending":
                was_pending = True
                await self.send(
                    {
                        "type": "response.provisional.text.done",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "response_id": state.response_id,
                        "provisional_id": state.response_id,
                        "text": text,
                    }
                )
            elif not state.official_done:
                state.official_done = True
                should_send_official_done = True
        if was_pending:
            if state.tts_state is not None:
                await self._finish_reply_tts(state.tts_state)
            return {
                "text_done_after_commit_ms": state.provisional_done_after_commit_ms,
                "response_done_after_commit_ms": None,
            }
        if not should_send_official_done:
            return {
                "text_done_after_commit_ms": state.official_text_done_after_commit_ms,
                "response_done_after_commit_ms": (
                    state.official_response_done_after_commit_ms
                ),
            }
        done_timing = await self._send_reply_done(
            turn,
            response_id=state.response_id,
            text=text,
            source=state.source,
            finish_reason=finish_reason,
            usage=usage,
            provisional_id=state.response_id,
            tts_state=state.tts_state,
        )
        async with state.lock:
            state.official_done = True
            state.official_text_done_after_commit_ms = done_timing[
                "text_done_after_commit_ms"
            ]
            state.official_response_done_after_commit_ms = done_timing[
                "response_done_after_commit_ms"
            ]
            return done_timing

    async def _mark_provisional_reply_terminal(
        self,
        state: ProvisionalReplyState,
        *,
        failed: bool = False,
        cancelled: bool = False,
    ) -> None:
        async with state.lock:
            state.failed = state.failed or failed
            state.cancelled = state.cancelled or cancelled
            state.content_available.set()
            state.sentence_ready.set()

    @staticmethod
    def _provisional_reply_prefix(text: str) -> tuple[str, bool]:
        stripped = text.strip()
        if not stripped:
            return "", False
        match = SYSTEM_REPLY_SENTENCE_END_RE.search(stripped)
        end = match.end() if match is not None else len(stripped)
        prefix = stripped[:end]
        return prefix[:SYSTEM_REPLY_PREFIX_MAX_CHARS], match is not None

    async def _resolve_provisional_reply_prefix(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
    ) -> tuple[str, str, float]:
        wait_started = time.perf_counter()
        await state.content_available.wait()
        self._ensure_turn_processing(turn)

        async with state.lock:
            text = "".join(state.text_parts)
            prefix, sentence_complete = self._provisional_reply_prefix(text)
            failed = state.failed or state.cancelled
            completed = state.completed
            first_nonempty_at = state.first_nonempty_at
        if failed:
            return "", "reply_failed", round(
                (time.perf_counter() - wait_started) * 1000.0, 3
            )
        if completed and not prefix:
            return "", "empty_completed", round(
                (time.perf_counter() - wait_started) * 1000.0, 3
            )
        if sentence_complete or completed:
            return prefix, "first_sentence", round(
                (time.perf_counter() - wait_started) * 1000.0, 3
            )

        elapsed_since_first = (
            time.perf_counter() - first_nonempty_at
            if first_nonempty_at is not None
            else 0.0
        )
        remaining = max(0.0, SYSTEM_REPLY_SENTENCE_WAIT_S - elapsed_since_first)
        if remaining:
            try:
                await asyncio.wait_for(state.sentence_ready.wait(), timeout=remaining)
            except TimeoutError:
                pass
        self._ensure_turn_processing(turn)
        async with state.lock:
            text = "".join(state.text_parts)
            prefix, sentence_complete = self._provisional_reply_prefix(text)
            failed = state.failed or state.cancelled
            completed = state.completed
        if failed:
            prefix = ""
            status = "reply_failed"
        elif completed and not prefix:
            status = "empty_completed"
        elif sentence_complete or completed:
            status = "first_sentence"
        else:
            status = "partial_timeout"
        return prefix, status, round(
            (time.perf_counter() - wait_started) * 1000.0, 3
        )

    def _provisional_reply_timing(
        self,
        state: ProvisionalReplyState,
        *,
        total_ms: float | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = (
            total_ms
            if total_ms is not None
            else (time.perf_counter() - state.started_at) * 1000.0
        )
        text_done_after_commit_ms = (
            state.official_text_done_after_commit_ms
            if state.status == "promoted"
            else state.provisional_done_after_commit_ms
        )
        stream_duration_ms = (
            round(
                text_done_after_commit_ms - state.first_delta_after_commit_ms,
                3,
            )
            if text_done_after_commit_ms is not None
            and state.first_delta_after_commit_ms is not None
            else None
        )
        return {
            "source": state.source,
            "ttft_ms": round(state.first_token_ms or elapsed_ms, 3),
            "total_ms": round(elapsed_ms, 3),
            "chars": len("".join(state.text_parts)),
            "created_after_commit_ms": state.created_after_commit_ms,
            "first_delta_after_commit_ms": state.first_delta_after_commit_ms,
            "text_done_after_commit_ms": text_done_after_commit_ms,
            "response_done_after_commit_ms": (
                state.official_response_done_after_commit_ms
            ),
            "stream_duration_ms": stream_duration_ms,
            "delta_count": state.delta_count,
            "completion_tokens": (
                state.usage.get("completion_tokens")
                if state.usage is not None
                else None
            ),
            "provisional": True,
            "provisional_status": state.status,
            "provisional_done_after_commit_ms": (
                state.provisional_done_after_commit_ms
            ),
            "resolution_reason": state.resolution_reason,
        }

    async def _promote_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        reason: str = "action_supported",
        wait_for_tts: bool = True,
    ) -> None:
        should_send_done = False
        buffered_text = ""
        async with state.lock:
            if state.status != "pending":
                return
            state.status = "promoted"
            state.resolution_reason = reason
            buffered_text = "".join(state.text_parts)
            await self.send(
                {
                    "type": "response.provisional.resolved",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "provisional_id": state.response_id,
                    "status": "promoted",
                    "reason": state.resolution_reason,
                    "promoted_prefix_chars": len(buffered_text),
                    "promoted_prefix_delta_count": state.delta_count,
                }
            )
            await self.send(
                {
                    "type": "response.created",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                    "response": {
                        "id": state.response_id,
                        "status": "in_progress",
                        "source": state.source,
                        "provisional_id": state.response_id,
                    },
                }
            )
            state.official_created = True
            if buffered_text:
                await self.send(
                    {
                        "type": "response.text.delta",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "response_id": state.response_id,
                        "provisional_id": state.response_id,
                        "delta": buffered_text,
                        "replayed_from_provisional": True,
                    }
                )
            if state.tts_state is not None:
                for chunk in state.tts_state.buffered_audio:
                    await self._send_reply_audio_delta(turn, state.tts_state, chunk)
                state.tts_state.buffered_audio.clear()
                state.tts_state.buffered_audio_bytes = 0
            if state.completed and not state.official_done:
                state.official_done = True
                should_send_done = True
            emit_structured_log(
                "reply",
                "provisional_reply_resolved",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=state.response_id,
                status=state.status,
                reason=state.resolution_reason,
                chars=len(buffered_text),
                delta_count=state.delta_count,
                resolved_after_commit_ms=self._after_commit_ms(turn),
            )
        if should_send_done:
            finalization = self._finalize_promoted_provisional_reply(
                turn,
                state,
                text=buffered_text,
            )
            if wait_for_tts:
                await finalization
            else:
                state.finalization_task = self._track_provisional_background_task(
                    turn,
                    finalization,
                    name=(
                        f"session-provisional-finalize-{self.session_id}-"
                        f"{turn.turn_id}"
                    ),
                )

    async def _finalize_promoted_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        text: str,
    ) -> None:
        done_timing = await self._send_reply_done(
            turn,
            response_id=state.response_id,
            text=text,
            source=state.source,
            finish_reason=state.finish_reason,
            usage=state.usage,
            provisional_id=state.response_id,
            tts_state=state.tts_state,
        )
        async with state.lock:
            state.official_text_done_after_commit_ms = done_timing[
                "text_done_after_commit_ms"
            ]
            state.official_response_done_after_commit_ms = done_timing[
                "response_done_after_commit_ms"
            ]

    @staticmethod
    def _track_provisional_background_task(
        turn: TurnBuffer,
        coroutine: Any,
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=name)
        turn.branch_tasks.add(task)
        task.add_done_callback(turn.branch_tasks.discard)
        return task

    async def _discard_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        reason: str,
        send_event: bool = True,
        abort_request: bool = True,
        wait_for_cleanup: bool = True,
    ) -> None:
        tts_state: ReplyTTSState | None = None
        async with state.lock:
            if state.status != "pending":
                return
            state.status = "discarded"
            state.resolution_reason = reason
            tts_state = state.tts_state
            if tts_state is not None:
                tts_state.buffered_audio.clear()
                tts_state.buffered_audio_bytes = 0
            if send_event:
                await self.send(
                    {
                        "type": "response.provisional.resolved",
                        "session_id": self.session_id,
                        "turn_id": turn.turn_id,
                        "provisional_id": state.response_id,
                        "status": "discarded",
                        "reason": reason,
                        "discarded_chars": len("".join(state.text_parts)),
                        "discarded_delta_count": state.delta_count,
                    }
                )
            emit_structured_log(
                "reply",
                "provisional_reply_resolved",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                response_id=state.response_id,
                status=state.status,
                reason=reason,
                chars=len("".join(state.text_parts)),
                delta_count=state.delta_count,
                resolved_after_commit_ms=self._after_commit_ms(turn),
            )
        cleanup = self._cleanup_discarded_provisional_reply(
            turn,
            state,
            tts_state=tts_state,
            abort_request=abort_request,
        )
        if wait_for_cleanup:
            await cleanup
        else:
            state.cleanup_task = self._track_provisional_background_task(
                turn,
                cleanup,
                name=(
                    f"session-provisional-cleanup-{self.session_id}-"
                    f"{turn.turn_id}"
                ),
            )

    async def _cleanup_discarded_provisional_reply(
        self,
        turn: TurnBuffer,
        state: ProvisionalReplyState,
        *,
        tts_state: ReplyTTSState | None,
        abort_request: bool,
    ) -> None:
        await self._abort_reply_tts(tts_state)
        if abort_request and state.request_id is not None:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                try:
                    await abort(state.request_id)
                except Exception:
                    logger.warning(
                        "[SESSION_ACTION_REALTIME] provisional reply abort failed "
                        "session_id=%s turn_id=%s request_id=%s",
                        self.session_id,
                        turn.turn_id,
                        state.request_id,
                        exc_info=True,
                    )
        if state.task is not None and state.task is not asyncio.current_task():
            if not state.task.done():
                state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)

    @staticmethod
    async def _wait_for_provisional_background_tasks(
        state: ProvisionalReplyState | None,
    ) -> None:
        if state is None:
            return
        tasks = [
            task
            for task in (state.finalization_task, state.cleanup_task)
            if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks)


MultimodalReplyProvisionalMixin = ProvisionalReplyComponent

