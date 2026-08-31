"""Embedded-TTS queue and audio streaming lifecycle."""

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
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


class ReplyTTSComponent:
    def _start_reply_tts(
        self,
        turn: TurnBuffer,
        *,
        response_id: str,
        provisional: ProvisionalReplyState | None = None,
    ) -> ReplyTTSState | None:
        if not self.output_capabilities.audio_enabled:
            return None
        if self.embedded_tts is None or self.embedded_tts_config is None:
            raise RuntimeError("embedded TTS is unavailable for an audio Session")

        text_queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=self.embedded_tts_config.text_queue_max_chunks
        )
        allow_commit = asyncio.Event()
        state_holder: dict[str, ReplyTTSState] = {}

        async def text_chunks():
            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    await allow_commit.wait()
                    return
                yield chunk

        async def audio_sink(chunk: bytes) -> None:
            self._ensure_turn_processing(turn)
            state = state_holder["state"]
            if state.provisional is not None:
                async with state.provisional.lock:
                    if state.provisional.status == "discarded":
                        return
                    if state.provisional.status == "pending":
                        buffered_audio_bytes = state.buffered_audio_bytes + len(chunk)
                        buffered_audio_milliseconds = (
                            buffered_audio_bytes * 1000 / (24000 * 1 * 2)
                        )
                        if (
                            buffered_audio_milliseconds
                            > self.embedded_tts_config.provisional_audio_max_milliseconds
                        ):
                            raise RuntimeError(
                                "provisional TTS audio duration exceeds configured limit"
                            )
                        if (
                            buffered_audio_bytes
                            > self.embedded_tts_config.provisional_audio_max_bytes
                        ):
                            raise RuntimeError(
                                "provisional TTS audio exceeds configured buffer limit"
                            )
                        state.buffered_audio.append(chunk)
                        state.buffered_audio_bytes = buffered_audio_bytes
                        return
                    await self._send_reply_audio_delta(turn, state, chunk)
                    return
            await self._send_reply_audio_delta(turn, state, chunk)

        task = asyncio.create_task(
            self.embedded_tts.synthesize_streaming(
                turn_id=turn.turn_id,
                text_chunks=text_chunks(),
                audio_sink=audio_sink,
            ),
            name=f"session-embedded-tts-{self.session_id}-{turn.turn_id}",
        )
        state = ReplyTTSState(
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            response_id=response_id,
            text_queue=text_queue,
            allow_commit=allow_commit,
            task=task,
            provisional=provisional,
        )
        state_holder["state"] = state
        emit_structured_log(
            "performance",
            "tts_reply_created",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            response_id=response_id,
            provisional=provisional is not None,
        )
        if provisional is not None:
            provisional.tts_state = state
        turn.branch_tasks.add(task)
        task.add_done_callback(turn.branch_tasks.discard)
        return state

    async def _send_reply_audio_delta(
        self, turn: TurnBuffer, state: ReplyTTSState, chunk: bytes
    ) -> None:
        self._ensure_turn_processing(turn)
        seq = state.next_audio_seq
        state.next_audio_seq += 1
        sent = await self.send(
            {
                "type": "response.audio.delta",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "response_id": state.response_id,
                "seq": seq,
                "delta": base64.b64encode(chunk).decode("ascii"),
                "audio": {
                    "format": "pcm16le",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                },
            }
        )
        if sent and not state.first_audio_sent:
            state.first_audio_sent = True
            emit_structured_log(
                "performance",
                "response_first_audio_delta_sent",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                response_id=state.response_id,
                seq=seq,
                audio_bytes=len(chunk),
                provisional_buffered=state.provisional is not None,
                after_commit_ms=self._after_commit_ms(turn),
            )

    async def _enqueue_reply_tts_text(
        self, state: ReplyTTSState | None, text: str
    ) -> None:
        if state is None or not text:
            return
        if not state.first_text_queued:
            state.first_text_queued = True
            emit_structured_log(
                "performance",
                "tts_first_text_queued",
                session_id=self.session_id,
                turn_id=state.turn_id,
                trace_id=state.trace_id,
                response_id=state.response_id,
                chars=len(text),
                queue_depth=state.text_queue.qsize(),
            )
        await self._put_reply_tts_queue(state, text)

    @staticmethod
    async def _put_reply_tts_queue(state: ReplyTTSState, value: str | None) -> None:
        if state.task.done():
            await state.task
        put_task = asyncio.create_task(state.text_queue.put(value))
        done, _ = await asyncio.wait(
            {put_task, state.task}, return_when=asyncio.FIRST_COMPLETED
        )
        if state.task in done:
            if not put_task.done():
                put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
            await state.task
        await put_task

    async def _abort_reply_tts(self, state: ReplyTTSState | None) -> None:
        if state is None or state.task.done():
            return
        if self.embedded_tts is not None:
            await self.embedded_tts.cancel_active_turn()
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)

    async def _finish_reply_tts(self, state: ReplyTTSState) -> None:
        if not state.input_finished:
            await self._put_reply_tts_queue(state, None)
            state.input_finished = True
            state.allow_commit.set()
        await state.task

    async def _next_reply_chunk(
        self,
        stream: Any,
        *,
        tts_state: ReplyTTSState | None,
        request_id: str,
    ) -> Any:
        next_task = asyncio.create_task(
            anext(stream), name=f"session-reply-next:{request_id}"
        )
        if tts_state is None:
            return await next_task
        done, _ = await asyncio.wait(
            {next_task, tts_state.task}, return_when=asyncio.FIRST_COMPLETED
        )
        if tts_state.task in done:
            if not next_task.done():
                next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                abort_started = time.perf_counter()
                emit_structured_log(
                    "performance",
                    "model_abort_begin",
                    session_id=self.session_id,
                    turn_id=tts_state.turn_id,
                    trace_id=tts_state.trace_id,
                    request_count=1,
                    reason="tts_completed_or_failed",
                )
                await asyncio.gather(abort(request_id), return_exceptions=True)
                emit_structured_log(
                    "performance",
                    "model_abort_completed",
                    session_id=self.session_id,
                    turn_id=tts_state.turn_id,
                    trace_id=tts_state.trace_id,
                    request_count=1,
                    elapsed_ms=round((time.perf_counter() - abort_started) * 1000, 3),
                )
            await tts_state.task
            raise RuntimeError("embedded TTS completed before model text EOF")
        return await next_task


MultimodalReplyTTSMixin = ReplyTTSComponent

