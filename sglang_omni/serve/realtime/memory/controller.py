"""Bridge between a realtime session and session-scoped memory."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.models import (
    ReplyHistoryRouteResult,
    TurnBuffer,
)
from sglang_omni.serve.realtime.memory import (
    SessionMemoryConfig,
    SessionMemoryTurn,
    build_memory_extraction_request,
    parse_memory_extraction,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


class SessionMemoryController:
    """Bounded asynchronous memory behavior for ``MultimodalSession``."""

    def _enqueue_session_memory(
        self,
        turn: TurnBuffer,
        audios: list[str],
        *,
        assistant_text: str | None,
        reply_model_visible: bool,
        reply_mode: str | None,
    ) -> None:
        config = self.session_memory_config
        store = self.session_memory_store
        scheduler = self.session_memory_scheduler
        if config is None or store is None or self.closed:
            return
        if turn.turn_origin != TURN_ORIGIN_USER:
            return
        self._bound_session_memory_reply_history(config)
        if not config.write_enabled or scheduler is None:
            self._settle_session_memory_turn_without_extraction(turn)
            return
        if reply_mode == REPLY_MODE_PURE_ACTION:
            # Pure-action history is handled by the existing bounded raw/action
            # history. It cannot create durable user claims, so avoid a second
            # audio prefill and all background model load for R2/R3.
            self._settle_session_memory_turn_without_extraction(turn)
            return
        if "text" not in self.modalities:
            # Outputs are fixed for the session. An action-only session can
            # never consume reply memory, so avoid adding background model
            # work that could contend with action scoring.
            self._settle_session_memory_turn_without_extraction(turn)
            return
        user_text = (
            turn.text.strip()
            if isinstance(turn.text, str) and turn.text.strip()
            else None
        )
        if not audios and user_text is None:
            self._settle_session_memory_turn_without_extraction(turn)
            return
        memory_turn = SessionMemoryTurn(
            turn_id=turn.turn_id,
            turn_seq=turn.session_turn_seq,
            user_text=user_text,
            audios=tuple(audios),
            assistant_text=assistant_text,
            reply_model_visible=reply_model_visible,
            reply_mode=reply_mode,
            queued_at=time.perf_counter(),
        )
        if (
            memory_turn.turn_seq in self._session_memory_attempts
            or memory_turn.turn_seq in self._session_memory_running_turn_seqs
            or store.is_settled_through(memory_turn.turn_seq)
        ):
            return
        if len(self._session_memory_pending_turns) >= config.max_pending_turns:
            dropped = self._session_memory_pending_turns.popleft()
            store.mark_batch_failed([dropped])
            self._session_memory_attempts.pop(dropped.turn_seq, None)
            self._session_memory_queue_overflow_count += 1
            emit_structured_log(
                "error",
                "session_memory_queue_overflow",
                level="warning",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                dropped_turn_id=dropped.turn_id,
                dropped_turn_seq=dropped.turn_seq,
                pending_turn_count=len(self._session_memory_pending_turns),
                overflow_count=self._session_memory_queue_overflow_count,
            )
        self._session_memory_pending_turns.append(memory_turn)
        self._session_memory_pending_turns = deque(
            sorted(
                self._session_memory_pending_turns,
                key=lambda pending_turn: pending_turn.turn_seq,
            )
        )
        self._session_memory_attempts.setdefault(memory_turn.turn_seq, 0)
        self._session_memory_updated.clear()
        submit_status = scheduler.submit_status(
            self.session_instance_id, self._run_session_memory_batch
        )
        if submit_status == "rejected":
            rejected_turns = list(self._session_memory_pending_turns)
            self._session_memory_pending_turns.clear()
            for rejected_turn in rejected_turns:
                self._session_memory_attempts.pop(
                    rejected_turn.turn_seq, None
                )
            store.mark_batch_failed(rejected_turns)
            self._session_memory_queue_overflow_count += len(rejected_turns)
            self._session_memory_updated.set()
            emit_structured_log(
                "error",
                "session_memory_queue_overflow",
                level="warning",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                dropped_turn_id=memory_turn.turn_id,
                dropped_turn_seq=memory_turn.turn_seq,
                dropped_turn_ids=[turn.turn_id for turn in rejected_turns],
                dropped_turn_count=len(rejected_turns),
                pending_turn_count=len(self._session_memory_pending_turns),
                overflow_count=self._session_memory_queue_overflow_count,
                reason="global_scheduler_admission",
            )
        emit_structured_log(
            "diagnostic",
            "session_memory_turn_queued",
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            turn_id=turn.turn_id,
            turn_seq=turn.session_turn_seq,
            pending_turn_count=len(self._session_memory_pending_turns),
            scheduler_submitted=submit_status == "submitted",
            scheduler_submit_status=submit_status,
            user_text_present=user_text is not None,
            audio_count=len(audios),
            reply_model_visible=reply_model_visible,
            reply_mode=reply_mode,
        )
    def _settle_session_memory_turn_without_extraction(
        self, turn: TurnBuffer
    ) -> None:
        store = self.session_memory_store
        turn_seq = turn.session_turn_seq
        if store is None or turn_seq <= 0:
            return
        if turn_seq in self._session_memory_attempts:
            return
        if (
            turn_seq in store.succeeded_turn_seqs
            or turn_seq in store.gap_turn_seqs
        ):
            return
        store.mark_turn_empty(turn_seq)
        self._session_memory_updated.set()
        if (
            self._session_memory_pending_turns
            and self.session_memory_scheduler is not None
            and not self.closed
        ):
            self.session_memory_scheduler.submit_status(
                self.session_instance_id, self._run_session_memory_batch
            )
    async def _run_session_memory_batch(self) -> bool:
        config = self.session_memory_config
        store = self.session_memory_store
        if config is None or store is None or self.closed:
            return False
        while (
            self._session_memory_pending_turns
            and self._session_memory_pending_turns[0].turn_seq
            <= store.processed_through_turn_seq
        ):
            settled = self._session_memory_pending_turns.popleft()
            self._session_memory_attempts.pop(settled.turn_seq, None)
        if not self._session_memory_pending_turns:
            return False
        expected_next_seq = store.processed_through_turn_seq + 1
        if self._session_memory_pending_turns[0].turn_seq != expected_next_seq:
            # A prior Turn has not completed/cancelled yet. Do not let a newer
            # result observe or overwrite state out of sequence. Settlement of
            # the missing Turn will resubmit this session.
            return False
        turns: list[SessionMemoryTurn] = []
        prior_attempt: int | None = None
        next_batch_seq = expected_next_seq
        while self._session_memory_pending_turns and len(turns) < config.batch_turns:
            next_turn = self._session_memory_pending_turns[0]
            if next_turn.turn_seq != next_batch_seq:
                break
            next_prior_attempt = self._session_memory_attempts.get(
                next_turn.turn_seq, 0
            )
            if prior_attempt is None:
                prior_attempt = next_prior_attempt
            elif next_prior_attempt != prior_attempt:
                break
            turns.append(self._session_memory_pending_turns.popleft())
            next_batch_seq += 1
        if not turns:
            return False

        first_seq = turns[0].turn_seq
        last_seq = turns[-1].turn_seq
        self._session_memory_running_turn_seqs = tuple(
            turn.turn_seq for turn in turns
        )
        for memory_turn in turns:
            self._session_memory_attempts[memory_turn.turn_seq] = (
                self._session_memory_attempts.get(memory_turn.turn_seq, 0) + 1
            )
        attempt = max(
            self._session_memory_attempts[turn.turn_seq] for turn in turns
        )
        request_id = (
            f"session-memory-{self.session_instance_id[:12]}-"
            f"{first_seq}-{last_seq}-a{attempt}"
        )
        self._session_memory_request_id = request_id
        base_store_revision = store.revision
        request = build_memory_extraction_request(
            model_name=self.model_name,
            session_id=self.session_id or "",
            session_instance_id=self.session_instance_id,
            language=self.language,
            turns=turns,
            active_claims=store.extraction_state(),
            active_threads=store.open_thread_extraction_state(),
            config=config,
            base_store_revision=base_store_revision,
        )
        started = time.perf_counter()
        oldest_queued_at = min(
            (turn.queued_at for turn in turns if turn.queued_at > 0),
            default=started,
        )
        queue_wait_ms = round(max(0.0, started - oldest_queued_at) * 1000.0, 3)
        emit_structured_log(
            "performance",
            "session_memory_extraction_started",
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            request_id=request_id,
            turn_ids=[turn.turn_id for turn in turns],
            turn_seqs=[turn.turn_seq for turn in turns],
            batch_size=len(turns),
            pending_turn_count=len(self._session_memory_pending_turns),
            active_claim_count=len(store.active_claims()),
            active_open_thread_count=len(store.active_open_threads()),
            base_store_revision=base_store_revision,
            attempt=attempt,
            queue_wait_ms=queue_wait_ms,
        )
        try:
            result = await asyncio.wait_for(
                self.client.completion(request, request_id=request_id),
                timeout=config.extraction_timeout_s,
            )
            if self.closed:
                return False
            extracted = parse_memory_extraction(
                result.text,
                expected_turns=turns,
                config=config,
            )
            apply_stats = store.apply(
                extracted,
                {turn.turn_seq: turn for turn in turns},
                expected_revision=base_store_revision,
            )
            self._compact_reply_history_media(
                tuple(episode.turn_id for episode in store.episodes)
            )
            self._bound_session_memory_reply_history(config)
            emit_structured_log(
                "diagnostic",
                "session_memory_extraction_completed",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                request_id=request_id,
                turn_ids=[turn.turn_id for turn in turns],
                batch_size=len(turns),
                added_claim_count=apply_stats.added,
                superseded_claim_count=apply_stats.superseded,
                retracted_claim_count=apply_stats.retracted,
                rejected_operation_count=apply_stats.rejected,
                rejected_operations=list(apply_stats.rejected_operations),
                episode_count=apply_stats.episode_count,
                artifact_count=apply_stats.artifact_count,
                opened_thread_count=apply_stats.opened_thread_count,
                updated_thread_count=apply_stats.updated_thread_count,
                closed_thread_count=apply_stats.closed_thread_count,
                active_claim_count=len(store.active_claims()),
                active_open_thread_count=len(store.active_open_threads()),
                retained_episode_count=len(store.episodes),
                retained_artifact_count=len(store.artifacts),
                processed_through_turn_seq=(
                    apply_stats.processed_through_turn_seq
                ),
                complete_through_turn_seq=(
                    apply_stats.complete_through_turn_seq
                ),
                gap_count=apply_stats.gap_count,
                store_revision=apply_stats.store_revision,
                attempt=attempt,
                queue_wait_ms=queue_wait_ms,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            for memory_turn in turns:
                self._session_memory_attempts.pop(memory_turn.turn_seq, None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                await asyncio.gather(
                    abort(request_id), return_exceptions=True
                )
            should_retry = (
                not self.closed
                and attempt <= config.max_retries
            )
            if should_retry:
                for memory_turn in reversed(turns):
                    self._session_memory_pending_turns.appendleft(memory_turn)
                while (
                    len(self._session_memory_pending_turns)
                    > config.max_pending_turns
                ):
                    dropped = self._session_memory_pending_turns.pop()
                    store.mark_batch_failed([dropped])
                    self._session_memory_attempts.pop(
                        dropped.turn_seq, None
                    )
                    self._session_memory_queue_overflow_count += 1
                    emit_structured_log(
                        "error",
                        "session_memory_queue_overflow",
                        level="warning",
                        session_id=self.session_id,
                        session_instance_id=self.session_instance_id,
                        dropped_turn_id=dropped.turn_id,
                        dropped_turn_seq=dropped.turn_seq,
                        pending_turn_count=len(
                            self._session_memory_pending_turns
                        ),
                        overflow_count=(
                            self._session_memory_queue_overflow_count
                        ),
                        reason="retry_requeue_capacity",
                    )
            else:
                store.mark_batch_failed(turns)
                for memory_turn in turns:
                    self._session_memory_attempts.pop(
                        memory_turn.turn_seq, None
                    )
            emit_structured_log(
                "error",
                "session_memory_extraction_failed",
                level="warning",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                request_id=request_id,
                turn_ids=[turn.turn_id for turn in turns],
                batch_size=len(turns),
                processed_through_turn_seq=store.processed_through_turn_seq,
                complete_through_turn_seq=store.complete_through_turn_seq,
                gap_count=len(store.gap_turn_seqs),
                attempt=attempt,
                will_retry=should_retry,
                queue_wait_ms=queue_wait_ms,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        finally:
            self._session_memory_request_id = None
            self._session_memory_running_turn_seqs = ()
            self._session_memory_updated.set()
        return bool(self._session_memory_pending_turns and not self.closed)
    async def _wait_for_session_memory_catchup(
        self,
        turn: TurnBuffer,
        history_route: ReplyHistoryRouteResult | None,
    ) -> None:
        config = self.session_memory_config
        store = self.session_memory_store
        if (
            config is None
            or store is None
            or not config.read_enabled
            or history_route is None
            or history_route.decision != REPLY_HISTORY_REQUIRED
            or history_route.reply_mode != REPLY_MODE_LANGUAGE_REQUIRED
            or config.catchup_timeout_s <= 0
        ):
            return
        # The two immediately preceding user turns are still forwarded in raw
        # form. Only wait when extraction is behind content that has already
        # fallen outside that window.
        required_through = max(
            0, turn.session_turn_seq - MAX_REPLY_HISTORY_TURNS - 1
        )
        if store.is_settled_through(required_through):
            return
        if (
            not self._session_memory_pending_turns
            and self._session_memory_request_id is None
        ):
            return

        scheduler = self.session_memory_scheduler
        prioritized = (
            scheduler.prioritize(self.session_instance_id)
            if scheduler is not None
            else False
        )

        started = time.perf_counter()
        deadline = started + config.catchup_timeout_s
        while (
            not store.is_settled_through(required_through)
            and not self.closed
        ):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(
                    self._session_memory_updated.wait(), timeout=remaining
                )
            except asyncio.TimeoutError:
                break
            if not store.is_settled_through(required_through):
                self._session_memory_updated.clear()
        emit_structured_log(
            "performance",
            "session_memory_catchup_checked",
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            turn_id=turn.turn_id,
            required_through_turn_seq=required_through,
            processed_through_turn_seq=store.processed_through_turn_seq,
            complete_through_turn_seq=store.complete_through_turn_seq,
            gap_count=len(store.gap_turn_seqs),
            caught_up=store.is_settled_through(required_through),
            complete_without_gaps=(
                store.complete_through_turn_seq >= required_through
            ),
            scheduler_prioritized=prioritized,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    async def _shutdown_session_memory(self) -> None:
        scheduler = self.session_memory_scheduler
        # Capture the backend request before cancelling the worker. The
        # worker's finally block clears the session field.
        request_id = self._session_memory_request_id
        if scheduler is not None:
            await scheduler.cancel(self.session_instance_id)
        abort = getattr(self.client, "abort", None)
        if request_id is not None and callable(abort):
            await asyncio.gather(abort(request_id), return_exceptions=True)
        self._session_memory_request_id = None
        self._session_memory_running_turn_seqs = ()
        self._session_memory_pending_turns.clear()
        self._session_memory_attempts.clear()
        self._session_memory_updated.set()
        if self.session_memory_store is not None:
            self.session_memory_store.clear()
    def _compact_reply_history_media(self, turn_ids: tuple[str, ...]) -> None:
        store = self.session_memory_store
        if store is None or not turn_ids:
            return
        protected_turn_ids = {
            history_turn.turn_id
            for history_turn in self.reply_history_turns[-MAX_REPLY_HISTORY_TURNS:]
        }
        compactable = set(turn_ids) - protected_turn_ids
        if not compactable:
            return
        for history_turn in self.reply_history_turns:
            if history_turn.turn_id not in compactable or not history_turn.audios:
                continue
            episode = store.episode_for_turn(history_turn.turn_id)
            if episode is None or not episode.user_summary:
                continue
            for message in history_turn.messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                retained = [
                    part
                    for part in content
                    if not (isinstance(part, dict) and part.get("type") == "audio")
                ]
                if not any(
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"].strip()
                    for part in retained
                ):
                    retained.append(
                        {"type": "text", "text": episode.user_summary}
                    )
                message["content"] = retained
            history_turn.audios.clear()
    def _bound_session_memory_reply_history(
        self, config: SessionMemoryConfig
    ) -> None:
        """Bound raw history even when a Turn intentionally skips extraction."""

        retained_reply_limit = max(
            MAX_REPLY_HISTORY_TURNS, config.max_episodes
        )
        if len(self.reply_history_turns) > retained_reply_limit:
            del self.reply_history_turns[:-retained_reply_limit]


MultimodalMemoryMixin = SessionMemoryController
