"""Turn-level preparation, concurrent inference, and finalization."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.qwen3_omni.global_action_catalog import (
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import _summarize_media
from sglang_omni.serve.realtime.protocol.models import (
    ImageFrame,
    ProvisionalReplyState,
    ReplyHistoryRouteResult,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
    get_structured_log_writer,
)

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


@dataclass(frozen=True, slots=True)
class TurnCommitInput:
    """Frozen media snapshot and timing boundary for one committed Turn."""

    turn: TurnBuffer
    current_audio_list: list[str]
    current_image_frames: list[ImageFrame]
    current_images: list[str]
    current_image_roles: list[str]
    ingest_ms: float
    commit_started: float


@dataclass(frozen=True, slots=True)
class PreparedTurnCommit:
    """Commit input after image preprocessing and client acknowledgement."""

    commit_input: TurnCommitInput
    prepared_current_images: list[str]
    image_preprocess_stats: dict[str, Any]


@dataclass(slots=True)
class TurnInferenceOutcome:
    """Outputs crossing from inference into persistence/result finalization."""

    action: dict[str, Any] | None
    scores: list[dict[str, Any]]
    action_timing: float
    action_context: dict[str, Any]
    action_error: Exception | None
    reply_text: str | None
    reply_timing: dict[str, Any] | None
    reply_history_route: ReplyHistoryRouteResult | None
    suppress_reply_for_unsupported_action: bool
    silent_action_finished: bool


class TurnPipeline:
    """Turn-level orchestration composed into ``MultimodalSession``."""

    def _prepare_turn_commit(self, event: dict[str, Any]) -> TurnCommitInput:
        """Validate commit-only fields and freeze the Turn input snapshot."""

        turn = self._require_collecting_turn(event)
        turn_origin, text_role, trigger = self._parse_turn_semantics(event)
        if (turn_origin, text_role, trigger) != (
            turn.turn_origin,
            turn.text_role,
            turn.trigger,
        ):
            raise ValueError(
                "turn_origin, text_role, and trigger must match turn.start"
            )
        if turn_origin == TURN_ORIGIN_PROACTIVE and event.get("user_input") is not None:
            raise ValueError("user_input must be null or omitted for proactive turns")
        if "text" in event:
            text = event.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be a string or null")
            turn.text = text
        if "_reply_provided" in event:
            turn.reply_provided = bool(event["_reply_provided"])
        elif (
            turn_origin == TURN_ORIGIN_PROACTIVE
            and isinstance(turn.text, str)
            and bool(turn.text.strip())
        ):
            turn.reply_provided = True
        if "reply_context" in event:
            reply_context = event.get("reply_context")
            if reply_context is not None and not isinstance(reply_context, str):
                raise ValueError("reply_context must be a string or null")
            if (
                isinstance(reply_context, str)
                and len(reply_context) > MAX_REPLY_CONTEXT_CHARS
            ):
                raise ValueError(
                    f"reply_context must contain at most {MAX_REPLY_CONTEXT_CHARS} characters"
                )
            turn.reply_context = reply_context
        if turn.reply_provided and isinstance(turn.reply_context, str):
            raise ValueError("provided reply and reply context are mutually exclusive")
        if "avatar_state" in event:
            state = event.get("avatar_state")
            if not isinstance(state, dict):
                raise ValueError("avatar_state must be an object")
            current_action_id = state.get("current_action_id")
            if current_action_id is not None and (
                not isinstance(current_action_id, str) or not current_action_id.strip()
            ):
                raise ValueError(
                    "avatar_state.current_action_id must be a non-empty string or null"
                )
            state_description = state.get("state_description")
            if state_description is not None and not isinstance(state_description, str):
                raise ValueError("avatar_state.state_description must be a string")
            turn.avatar_state = dict(state)

        current_audio = (
            turn.audio.to_full_wav_data_uri() if not turn.audio.is_empty() else None
        )
        current_image_frames = sorted(
            turn.images, key=lambda frame: (frame.timestamp_ms, frame.seq)
        )
        current_images = [frame.data_uri for frame in current_image_frames]
        current_image_roles = [frame.image_role for frame in current_image_frames]
        current_audio_list = [current_audio] if current_audio else []
        ingest_ms = (time.perf_counter() - turn.started_at) * 1000.0
        turn.phase = TURN_PHASE_PROCESSING
        turn.request_base = (
            f"session-{self.session_id}-turn-{turn.turn_id}-{uuid.uuid4().hex}"
        )
        commit_started = time.perf_counter()
        turn.commit_started_at = commit_started
        emit_structured_log(
            "lifecycle",
            "turn_commit_received",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            modalities=list(self.modalities),
            audio_chunk_count=turn.audio_chunk_count,
            image_frame_count=len(current_images),
            text_present=bool(turn.text),
            reply_context_present=bool(turn.reply_context),
        )
        emit_structured_log(
            "performance",
            "turn_input_summary_at_commit",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            text_chars=len(turn.text) if isinstance(turn.text, str) else 0,
            audio_chunks=turn.audio_chunk_count,
            image_frames=len(current_images),
            first_text_after_turn_start_ms=(
                round((turn.first_text_received_at - turn.started_at) * 1000, 3)
                if turn.first_text_received_at is not None
                else None
            ),
            first_audio_after_turn_start_ms=(
                round((turn.first_audio_received_at - turn.started_at) * 1000, 3)
                if turn.first_audio_received_at is not None
                else None
            ),
            first_image_after_turn_start_ms=(
                round((turn.first_image_received_at - turn.started_at) * 1000, 3)
                if turn.first_image_received_at is not None
                else None
            ),
        )
        self._request_turn_resource_sample(
            "turn_before_inference",
            turn=turn,
            audio_chunk_count=turn.audio_chunk_count,
            image_frame_count=len(current_images),
        )
        return TurnCommitInput(
            turn=turn,
            current_audio_list=current_audio_list,
            current_image_frames=current_image_frames,
            current_images=current_images,
            current_image_roles=current_image_roles,
            ingest_ms=ingest_ms,
            commit_started=commit_started,
        )

    async def _acknowledge_and_prepare_turn(
        self, commit_input: TurnCommitInput
    ) -> PreparedTurnCommit | None:
        """Send ``turn.committed`` and finish bounded image preprocessing."""

        turn = commit_input.turn
        committed_payload: dict[str, Any] = {
            "type": "turn.committed",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "audio_chunk_count": turn.audio_chunk_count,
            "image_frame_count": len(commit_input.current_images),
        }
        if self.protocol_version is not None:
            committed_payload["image_sources"] = [
                (
                    IMAGE_SOURCE_AVATAR_CURRENT
                    if role == IMAGE_ROLE_AVATAR_STATE
                    else role
                )
                for role in commit_input.current_image_roles
            ]
        else:
            committed_payload["image_roles"] = commit_input.current_image_roles
        if await self.send(committed_payload):
            emit_structured_log(
                "lifecycle",
                "turn_committed_sent",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                elapsed_ms=round(
                    (time.perf_counter() - commit_input.commit_started) * 1000,
                    3,
                ),
            )
        if self.closed:
            await self._cancel_active_turn(send_event=False, expected_turn=turn)
            return None
        prepared_current_images, image_preprocess_stats = (
            await self._resolve_prepared_images(
                turn, commit_input.current_image_frames
            )
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] turn.commit input session_id=%s turn_id=%s payload=%s",
            self.session_id,
            turn.turn_id,
            json.dumps(
                {
                    "text": turn.text,
                    "avatar_state": turn.avatar_state or self.last_avatar_state,
                    "turn_origin": turn.turn_origin,
                    "text_role": turn.text_role,
                    "trigger": turn.trigger,
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(commit_input.current_images),
                    "image_roles": commit_input.current_image_roles,
                    "audio": _summarize_media(commit_input.current_audio_list),
                    "images": _summarize_media(commit_input.current_images),
                    "history_turn_count": len(self.history_turns),
                    "candidate_count": len(self.candidates),
                    "action_catalog_hash": self.action_catalog_hash,
                    "global_action_catalog_hash": self.global_action_catalog_hash,
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return PreparedTurnCommit(
            commit_input=commit_input,
            prepared_current_images=prepared_current_images,
            image_preprocess_stats=image_preprocess_stats,
        )

    async def _finalize_turn_success(
        self,
        prepared_commit: PreparedTurnCommit,
        outcome: TurnInferenceOutcome,
    ) -> str | None:
        """Persist authoritative history, send result, and emit terminal logs."""

        commit_input = prepared_commit.commit_input
        turn = commit_input.turn
        turn_id = turn.turn_id
        current_audio_list = commit_input.current_audio_list
        current_images = commit_input.current_images
        current_image_roles = commit_input.current_image_roles
        ingest_ms = commit_input.ingest_ms
        commit_started = commit_input.commit_started
        image_preprocess_stats = prepared_commit.image_preprocess_stats
        action = outcome.action
        scores = outcome.scores
        action_timing = outcome.action_timing
        action_context = outcome.action_context
        action_error = outcome.action_error
        reply_text = outcome.reply_text
        reply_timing = outcome.reply_timing
        reply_history_route = outcome.reply_history_route
        suppress_reply_for_unsupported_action = (
            outcome.suppress_reply_for_unsupported_action
        )
        silent_action_finished = outcome.silent_action_finished
        action_unsupported = bool(
            action is not None and action.get("support_status") == "unsupported"
        )

        action_finished = time.perf_counter()
        if self.active_turn is not turn or turn.phase != TURN_PHASE_PROCESSING:
            return None
        turn.phase = TURN_PHASE_COMPLETED
        if "action" in self.modalities:
            self._persist_avatar_state(
                turn.avatar_state,
                has_avatar_image=(IMAGE_ROLE_AVATAR_STATE in current_image_roles),
            )
        history_reply_text = (
            self.unsupported_action_text
            if suppress_reply_for_unsupported_action
            else reply_text
        )
        self._append_reply_history(
            turn,
            current_audio_list,
            current_images,
            current_image_roles,
            history_reply_text,
            model_visible=not suppress_reply_for_unsupported_action,
            history_kind=(
                "unsupported_action_notice"
                if suppress_reply_for_unsupported_action
                else "reply"
            ),
        )
        self._enqueue_session_memory(
            turn,
            current_audio_list,
            assistant_text=(
                reply_text if not suppress_reply_for_unsupported_action else None
            ),
            reply_model_visible=not suppress_reply_for_unsupported_action,
            reply_mode=(
                reply_history_route.reply_mode
                if reply_history_route is not None
                else None
            ),
        )
        if action is not None:
            self._append_action_history(
                current_audio_list,
                current_images,
                current_image_roles,
                turn.text,
                turn_id=turn_id,
                turn_origin=turn.turn_origin,
                text_role=turn.text_role,
                action=action,
                # The prerecorded unsupported notice is retained for
                # diagnostics, but is not a model reply example. Feeding
                # it back as an ordinary assistant message causes later
                # supported turns to imitate the notice.
                reply_text=(
                    None
                    if suppress_reply_for_unsupported_action
                    else history_reply_text
                ),
            )
        self.active_turn = None
        turn_status = "partial" if action_error is not None else "completed"
        result = {
            "type": "turn.result",
            "session_id": self.session_id,
            "turn_id": turn_id,
            "timing": {
                "server_turn_ingest_ms": round(ingest_ms, 3),
                "image_preprocessing": image_preprocess_stats,
                "reply_history_route_ms": (
                    reply_history_route.elapsed_ms
                    if reply_history_route is not None
                    else 0.0
                ),
                "reply_mode": (
                    reply_history_route.reply_mode
                    if reply_history_route is not None
                    else None
                ),
            },
        }
        if self.protocol_version is not None or self.modalities != ("action",):
            result["status"] = turn_status
            result[
                "outputs" if self.protocol_version is not None else "modalities"
            ] = {
                modality: (
                    "failed"
                    if modality == "action" and action_error is not None
                    else (
                        "suppressed"
                        if modality == "text"
                        and (
                            suppress_reply_for_unsupported_action
                            or silent_action_finished
                        )
                        else "completed"
                    )
                )
                for modality in self.modalities
            }
        if suppress_reply_for_unsupported_action and "text" in self.modalities:
            result["reply"] = {
                "source": "client_prerecorded_audio",
                "reason": "unsupported_action",
                "recorded_in_history": bool(self.unsupported_action_text),
            }
            result["timing"]["reply"] = reply_timing or {}
        elif reply_text is not None:
            result["reply"] = {
                "text": reply_text,
                "source": (
                    reply_timing.get("source", "generated")
                    if reply_timing
                    else "generated"
                ),
            }
            result["timing"]["reply"] = reply_timing or {}
        if action_error is not None:
            result["errors"] = {
                "action": {"message": str(action_error)},
            }
        if action is not None:
            result["action_catalog_hash"] = self.action_catalog_hash
            if self.global_action_catalog is not None:
                result["session_action_catalog_hash"] = self.action_catalog_hash
                result["global_action_catalog_hash"] = (
                    self.global_action_catalog_hash
                )
            result["action"] = self._compact_action(action)
            result["timing"].update(
                {
                    "server_action_compute_ms": action_timing,
                    "action_breakdown": action_context.get(
                        "action_timing_breakdown", {}
                    ),
                }
            )
        if self.include_scores and action is not None:
            result["scores"] = scores
            result["media_summary"] = {
                "audio_chunk_count": turn.audio_chunk_count,
                "image_frame_count": len(current_images),
                "user_camera_image_count": current_image_roles.count(
                    IMAGE_ROLE_USER_CAMERA
                ),
                "avatar_state_image_count": current_image_roles.count(
                    IMAGE_ROLE_AVATAR_STATE
                ),
                "received_image_count": len(turn.images),
                "scored_image_count": action_context.get(
                    "scored_current_image_count", 0
                ),
                "text_present": bool(turn.text),
                "duplicate_audio_chunks": turn.duplicate_audio_chunks,
                "duplicate_image_frames": turn.duplicate_image_frames,
                "action_context": action_context,
            }
        result_finalize_ms = (time.perf_counter() - action_finished) * 1000.0
        total_after_commit_ms = (time.perf_counter() - commit_started) * 1000.0
        result["timing"].update(
            {
                "server_result_finalize_ms": round(result_finalize_ms, 3),
                "server_total_after_commit_ms": round(total_after_commit_ms, 3),
            }
        )
        send_started = time.perf_counter()
        await self.send(result)
        emit_structured_log(
            "performance",
            "turn_timing",
            session_id=self.session_id,
            turn_id=turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            modalities=list(self.modalities),
            turn_origin=turn.turn_origin,
            status=turn_status,
            ingest_ms=round(ingest_ms, 3),
            category_compute_ms=action_context.get("category_compute_ms"),
            child_compute_ms=action_context.get("child_compute_ms"),
            category_scoring_skipped=action_context.get(
                "category_scoring_skipped"
            ),
            category_scoring_skip_reason=action_context.get(
                "category_scoring_skip_reason"
            ),
            forced_semantic_tag=action_context.get(
                "forced_semantic_tag"
            ),
            resolved_category_id=action_context.get(
                "selected_category_id"
            ),
            child_candidate_count=action_context.get(
                "child_candidate_count"
            ),
            child_scoring_skipped=action_context.get(
                "child_scoring_skipped"
            ),
            child_scoring_skip_reason=action_context.get(
                "child_scoring_skip_reason"
            ),
            previous_action_finished_candidate_id=action_context.get(
                "previous_action_finished_candidate_id"
            ),
            previous_action_finished_action_id=action_context.get(
                "previous_action_finished_action_id"
            ),
            action_finished_repeat_excluded=action_context.get(
                "action_finished_repeat_excluded"
            ),
            action_finished_repeat_unavoidable=action_context.get(
                "action_finished_repeat_unavoidable"
            ),
            action_finished_random_pool_count=action_context.get(
                "action_finished_random_pool_count"
            ),
            system_route_degradation_reason=action_context.get(
                "system_route_degradation_reason"
            ),
            action_support_status=(
                action.get("support_status") if action is not None else None
            ),
            action_fallback_applied=(
                action.get("fallback_applied") if action is not None else None
            ),
            category_decision_id=action_context.get("category_decision_id"),
            child_decision_id=action_context.get("child_decision_id"),
            system_route_reconciled=action_context.get(
                "system_route_reconciled"
            ),
            reply_prefix_wait_ms=action_context.get(
                "reply_prefix_wait_ms"
            ),
            reply_prefix_status=action_context.get(
                "reply_prefix_status"
            ),
            reply_ttft_ms=(reply_timing or {}).get("ttft_ms"),
            reply_total_ms=(reply_timing or {}).get("total_ms"),
            reply_created_after_commit_ms=(reply_timing or {}).get(
                "created_after_commit_ms"
            ),
            reply_first_delta_after_commit_ms=(reply_timing or {}).get(
                "first_delta_after_commit_ms"
            ),
            reply_text_done_after_commit_ms=(reply_timing or {}).get(
                "text_done_after_commit_ms"
            ),
            reply_response_done_after_commit_ms=(reply_timing or {}).get(
                "response_done_after_commit_ms"
            ),
            reply_stream_duration_ms=(reply_timing or {}).get("stream_duration_ms"),
            reply_delta_count=(reply_timing or {}).get("delta_count"),
            reply_completion_tokens=(reply_timing or {}).get("completion_tokens"),
            reply_provisional=(reply_timing or {}).get("provisional"),
            reply_provisional_status=(reply_timing or {}).get("provisional_status"),
            reply_provisional_done_after_commit_ms=(reply_timing or {}).get(
                "provisional_done_after_commit_ms"
            ),
            reply_resolution_reason=(reply_timing or {}).get("resolution_reason"),
            reply_discarded_chars=(
                (reply_timing or {}).get("chars") if action_unsupported else 0
            ),
            total_after_commit_ms=round(total_after_commit_ms, 3),
            logger_health=get_structured_log_writer().health(),
        )
        emit_structured_log(
            "lifecycle",
            "turn_completed",
            session_id=self.session_id,
            turn_id=turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            status=turn_status,
            modalities=list(self.modalities),
            total_after_commit_ms=round(total_after_commit_ms, 3),
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] turn.result sent session_id=%s "
            "turn_id=%s serialize_and_send_ms=%.3f total_after_commit_ms=%.3f",
            self.session_id,
            turn_id,
            (time.perf_counter() - send_started) * 1000.0,
            (time.perf_counter() - commit_started) * 1000.0,
        )
        return turn_status

    async def handle_turn_commit(self, event: dict[str, Any]) -> None:
        commit_input = self._prepare_turn_commit(event)
        turn = commit_input.turn
        current_audio_list = commit_input.current_audio_list
        current_image_frames = commit_input.current_image_frames
        current_images = commit_input.current_images
        current_image_roles = commit_input.current_image_roles
        ingest_ms = commit_input.ingest_ms
        commit_started = commit_input.commit_started
        turn_id = turn.turn_id
        turn_outcome = "failed"

        try:
            prepared_commit = await self._acknowledge_and_prepare_turn(commit_input)
            if prepared_commit is None:
                turn_outcome = "cancelled"
                return
            prepared_current_images = prepared_commit.prepared_current_images
            image_preprocess_stats = prepared_commit.image_preprocess_stats
            action: dict[str, Any] | None = None
            scores: list[dict[str, Any]] = []
            action_timing = 0.0
            action_context: dict[str, Any] = {}
            action_error: Exception | None = None
            reply_text: str | None = None
            reply_timing: dict[str, Any] | None = None
            selected_category: SessionActionCategory | None = None
            category_decision_received = False
            category_support_status: str | None = None
            reply_task: asyncio.Task[tuple[str, dict[str, Any]]] | None = None
            provisional_state: ProvisionalReplyState | None = None
            provisional_discard_task: asyncio.Task[Any] | None = None
            reply_history_route: ReplyHistoryRouteResult | None = None
            preserve_language_reply_on_unsupported_action = False
            reply_route_decision_ready = False

            def track_branch(coroutine: Any, *, name: str) -> asyncio.Task[Any]:
                task = asyncio.create_task(coroutine, name=name)
                turn.branch_tasks.add(task)
                task.add_done_callback(turn.branch_tasks.discard)
                return task

            def maybe_schedule_category_discard() -> None:
                nonlocal provisional_discard_task
                if (
                    reply_route_decision_ready
                    and category_support_status == "unsupported"
                    and not preserve_language_reply_on_unsupported_action
                    and provisional_state is not None
                    and provisional_state.status == "pending"
                    and provisional_discard_task is None
                ):
                    provisional_discard_task = track_branch(
                        self._discard_provisional_reply(
                            turn,
                            provisional_state,
                            reason="category_unsupported",
                            wait_for_cleanup=(
                                not self.action_ready_tts_decoupled
                            ),
                        ),
                        name=(
                            f"session-provisional-discard-{self.session_id}-"
                            f"{turn.turn_id}"
                        ),
                    )

            def on_category_selected(
                category: SessionActionCategory | None,
                support_status: str,
            ) -> None:
                nonlocal selected_category, category_decision_received
                nonlocal category_support_status
                selected_category = category
                category_decision_received = True
                category_support_status = support_status
                emit_structured_log(
                    "action",
                    "category_selected",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                    category_id=(
                        category.category_id
                        if category is not None
                        else UNSUPPORTED_DECISION_ID
                    ),
                    category_label=(
                        category.source_label if category is not None else None
                    ),
                    support_status=support_status,
                )
                emit_structured_log(
                    "performance",
                    "action_category_ready",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    category_id=(category.category_id if category else None),
                    support_status=support_status,
                    after_commit_ms=self._after_commit_ms(turn),
                )
                maybe_schedule_category_discard()

            provided_reply = turn.reply_provided
            silent_action_finished = (
                provided_reply
                and not turn.text
                and turn.turn_origin == TURN_ORIGIN_PROACTIVE
                and turn.trigger == ACTION_FINISHED_TRIGGER
            )
            fusion_reply = (
                "text" in self.modalities
                and "action" in self.modalities
                and not silent_action_finished
            )
            if fusion_reply:
                provisional_state = await self._create_provisional_reply(
                    turn,
                    source="provided" if provided_reply else "generated",
                )

            reply_history_route_task: asyncio.Task[Any] | None = None
            if (
                "text" in self.modalities
                and not provided_reply
                and turn.turn_origin == TURN_ORIGIN_USER
                and (
                    current_audio_list
                    or (isinstance(turn.text, str) and turn.text.strip())
                )
            ):
                reply_history_route_task = track_branch(
                    self._classify_reply_history_requirement(
                        turn,
                        current_audio_list,
                        current_text=turn.text,
                    ),
                    name=(
                        f"session-reply-history-route-{self.session_id}-"
                        f"{turn.turn_id}"
                    ),
                )

            action_task: asyncio.Task[Any] | None = None
            action_started: float | None = None

            def start_action_scoring() -> None:
                nonlocal action_task, action_started
                if "action" not in self.modalities or action_task is not None:
                    return
                action_started = time.perf_counter()
                emit_structured_log(
                    "performance",
                    "action_scoring_begin",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    candidate_count=len(self.candidates),
                    category_count=len(self.categories),
                    route_action_parallel=self.route_action_parallel,
                )
                action_task = track_branch(
                    self._score_action(
                        current_audio_list,
                        prepared_current_images,
                        current_image_roles,
                        turn.text,
                        turn.avatar_state,
                        turn_origin=turn.turn_origin,
                        text_role=turn.text_role,
                        trigger=turn.trigger,
                        turn_id=turn_id,
                        turn=turn,
                        request_base=turn.request_base,
                        provisional_reply=provisional_state,
                        on_category_selected=(
                            on_category_selected
                            if "text" in self.modalities
                            else None
                        ),
                    ),
                    name=f"session-action-{self.session_id}-{turn.turn_id}",
                )

            if (
                self.route_action_parallel
                and reply_history_route_task is not None
                and "action" in self.modalities
            ):
                # Let the history-route task submit its request first, then
                # admit action scoring without waiting for the route result.
                await asyncio.sleep(0)
                start_action_scoring()

            if reply_history_route_task is not None:
                reply_history_route = await reply_history_route_task

            preserve_language_reply_on_unsupported_action = bool(
                fusion_reply
                and turn.turn_origin == TURN_ORIGIN_USER
                and reply_history_route is not None
                and reply_history_route.reply_mode == REPLY_MODE_LANGUAGE_REQUIRED
            )
            pure_action_reply = bool(
                fusion_reply
                and reply_history_route is not None
                and reply_history_route.reply_mode == REPLY_MODE_PURE_ACTION
            )
            reply_route_decision_ready = True
            maybe_schedule_category_discard()
            if provisional_discard_task is not None:
                await provisional_discard_task
            start_action_scoring()

            # Action scoring is already running, so the rare bounded memory
            # catch-up cannot postpone action admission. It only delays reply
            # construction when an R1 request depends on turns older than the
            # existing two-turn raw window.
            await self._wait_for_session_memory_catchup(
                turn, reply_history_route
            )

            if (
                fusion_reply
                and provisional_state is not None
                and provisional_state.status == "pending"
            ):
                if provided_reply:
                    reply_task = track_branch(
                        self._run_provided_reply(
                            turn,
                            turn.text or "",
                            provisional=provisional_state,
                        ),
                        name=(
                            f"session-provisional-provided-reply-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                elif pure_action_reply:
                    reply_task = track_branch(
                        self._run_pure_action_short_reply(
                            turn,
                            current_audio_list,
                            provisional=provisional_state,
                            history_route=reply_history_route,
                        ),
                        name=(
                            f"session-provisional-pure-action-reply-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                else:
                    reply_task = track_branch(
                        self._run_generated_reply(
                            turn,
                            current_audio_list,
                            prepared_current_images,
                            current_image_roles,
                            None,
                            provisional=provisional_state,
                            history_route=reply_history_route,
                        ),
                        name=(
                            f"session-provisional-reply-{self.session_id}-"
                            f"{turn.turn_id}"
                        ),
                    )
                provisional_state.task = reply_task
            elif (
                "text" in self.modalities
                and provided_reply
                and not silent_action_finished
            ):
                reply_task = track_branch(
                    self._run_provided_reply(turn, turn.text or ""),
                    name=f"session-provided-reply-{self.session_id}-{turn.turn_id}",
                )
            elif "text" in self.modalities and "action" not in self.modalities:
                reply_task = track_branch(
                    self._run_generated_reply(
                        turn,
                        current_audio_list,
                        prepared_current_images,
                        current_image_roles,
                        None,
                        history_route=reply_history_route,
                    ),
                    name=f"session-reply-{self.session_id}-{turn.turn_id}",
                )

            if "action" in self.modalities:
                assert action_task is not None
                assert action_started is not None
                try:
                    action, scores, action_timing, action_context = await action_task
                except Exception as exc:
                    if not category_decision_received or "text" not in self.modalities:
                        raise
                    action_error = exc
                    fallback = (
                        self._default_fallback_candidate()
                        if self.global_action_catalog is not None
                        else self._no_action_candidate()
                    )
                    action = {
                        "candidate_id": fallback.candidate_id,
                        "action_id": fallback.action_id,
                        "category_id": fallback.category_id,
                        "execution_binding": dict(fallback.execution_binding),
                        "execute": fallback.action_id != "no_action",
                        "support_status": "unknown",
                        "fallback_applied": True,
                    }
                    emit_structured_log(
                        "error",
                        "child_action_failed",
                        level="error",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        logical_request_id=turn.request_base,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        fallback_action_id=fallback.action_id,
                        fallback_category_id=fallback.category_id,
                    )
                emit_structured_log(
                    "performance",
                    "action_child_ready",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    candidate_id=(
                        action.get("candidate_id") if action is not None else None
                    ),
                    support_status=(
                        action.get("support_status") if action is not None else None
                    ),
                    elapsed_ms=round((time.perf_counter() - action_started) * 1000, 3),
                )
                action_unsupported = (
                    action is not None and action.get("support_status") == "unsupported"
                )
                suppress_reply_for_unsupported_action = bool(
                    action_unsupported
                    and not preserve_language_reply_on_unsupported_action
                )
                if provisional_state is not None:
                    if suppress_reply_for_unsupported_action:
                        unsupported_reason = (
                            "category_unsupported"
                            if action_context.get("category_decision_id")
                            == UNSUPPORTED_DECISION_ID
                            else "child_unsupported"
                        )
                        await self._discard_provisional_reply(
                            turn,
                            provisional_state,
                            reason=unsupported_reason,
                            wait_for_cleanup=(
                                not self.action_ready_tts_decoupled
                            ),
                        )
                        if provisional_discard_task is not None:
                            await asyncio.gather(
                                provisional_discard_task,
                                return_exceptions=True,
                            )
                    else:
                        reply_failed = (
                            reply_task is not None
                            and reply_task.done()
                            and not reply_task.cancelled()
                            and reply_task.exception() is not None
                        )
                        if reply_failed:
                            await self._discard_provisional_reply(
                                turn,
                                provisional_state,
                                reason="reply_failed",
                                wait_for_cleanup=(
                                    not self.action_ready_tts_decoupled
                                ),
                            )
                        else:
                            await self._promote_provisional_reply(
                                turn,
                                provisional_state,
                                reason=(
                                    "language_required"
                                    if action_unsupported
                                    else "action_supported"
                                ),
                                wait_for_tts=(
                                    not self.action_ready_tts_decoupled
                                ),
                            )
                logger.info(
                    "[SESSION_ACTION_REALTIME] action completed session_id=%s "
                    "turn_id=%s elapsed_ms=%.3f top_action=%s",
                    self.session_id,
                    turn_id,
                    (time.perf_counter() - action_started) * 1000.0,
                    action.get("action_id") if action else None,
                )
                if action is not None:
                    self._ensure_turn_processing(turn)
                    action_ready_payload: dict[str, Any] = {
                        "type": "turn.action.ready",
                        "session_id": self.session_id,
                        "turn_id": turn_id,
                        "action": self._compact_action(action),
                    }
                    if self.global_action_catalog is not None:
                        action_ready_payload.update(
                            {
                                "session_action_catalog_hash": self.action_catalog_hash,
                                "global_action_catalog_hash": self.global_action_catalog_hash,
                            }
                        )
                    await self.send(action_ready_payload)
                    if action_error is None:
                        self._record_action_as_executed(
                            turn=turn,
                            action=action,
                        )

            action_unsupported = (
                action is not None and action.get("support_status") == "unsupported"
            )
            suppress_reply_for_unsupported_action = bool(
                action_unsupported
                and not preserve_language_reply_on_unsupported_action
            )
            if reply_task is not None and not suppress_reply_for_unsupported_action:
                reply_text, reply_timing = await reply_task
                if provisional_state is not None:
                    reply_timing = self._provisional_reply_timing(provisional_state)
            elif provisional_state is not None:
                reply_timing = self._provisional_reply_timing(provisional_state)

            # turn.action.ready is allowed to overtake TTS finalization or
            # cancellation, but turn.result remains the terminal barrier for
            # all reply/TTS resources belonging to the turn.
            await self._wait_for_provisional_background_tasks(provisional_state)
            if provisional_state is not None and reply_timing is not None:
                reply_timing = self._provisional_reply_timing(provisional_state)

            turn_status = await self._finalize_turn_success(
                prepared_commit,
                TurnInferenceOutcome(
                    action=action,
                    scores=scores,
                    action_timing=action_timing,
                    action_context=action_context,
                    action_error=action_error,
                    reply_text=reply_text,
                    reply_timing=reply_timing,
                    reply_history_route=reply_history_route,
                    suppress_reply_for_unsupported_action=(
                        suppress_reply_for_unsupported_action
                    ),
                    silent_action_finished=silent_action_finished,
                ),
            )
            if turn_status is None:
                turn_outcome = "cancelled"
                return
            turn_outcome = turn_status
        except asyncio.CancelledError:
            turn_outcome = "cancelled"
            logger.info(
                "[SESSION_ACTION_REALTIME] turn inference cancelled "
                "session_id=%s turn_id=%s request_id=%s",
                self.session_id,
                turn_id,
                turn.current_request_id,
            )
            raise
        except Exception as exc:
            if turn.phase == TURN_PHASE_CANCELLING:
                turn_outcome = "cancelled"
                logger.warning(
                    "[SESSION_ACTION_REALTIME] cancelled turn cleanup failed "
                    "session_id=%s turn_id=%s",
                    self.session_id,
                    turn_id,
                    exc_info=True,
                )
                return
            turn_outcome = "failed"
            logger.exception(
                "[SESSION_ACTION_REALTIME] turn failed session_id=%s turn_id=%s",
                self.session_id,
                turn_id,
            )
            turn.phase = TURN_PHASE_COMPLETED
            if self.active_turn is turn:
                self.active_turn = None
            if (
                self.session_memory_store is not None
                and turn.turn_origin == TURN_ORIGIN_USER
            ):
                self._settle_session_memory_turn_without_extraction(turn)
            message = str(exc)
            if (
                "action" in self.modalities
                and "prefix selected-token logprobs are missing" in message
            ):
                code = "action_score_logprob_unavailable"
                error_type = "action_score_error"
            elif self.modalities == ("action",):
                code = "action_score_failed"
                error_type = "action_score_error"
            else:
                code = "turn_processing_failed"
                error_type = "turn_processing_error"
            emit_structured_log(
                "error",
                "turn_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                error_type=type(exc).__name__,
                error_message=message,
                error_code=code,
            )
            await self.send_error(
                error_type,
                code,
                message,
                session_id=self.session_id,
                turn_id=turn_id,
            )
        finally:
            self._request_turn_resource_sample(
                "turn_after_terminal",
                turn=turn,
                turn_outcome=turn_outcome,
                elapsed_after_commit_ms=round(
                    max(0.0, time.perf_counter() - commit_started) * 1000.0,
                    3,
                ),
            )
    def _request_turn_resource_sample(
        self,
        sample_trigger: str,
        *,
        turn: TurnBuffer,
        **fields: Any,
    ) -> bool:
        requester = self.request_resource_sample
        if requester is None:
            return False
        try:
            return bool(
                requester(
                    sample_trigger,
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    turn_origin=turn.turn_origin,
                    modalities=list(self.modalities),
                    **fields,
                )
            )
        except Exception as exc:
            emit_structured_log(
                "resource",
                "resource_sample_request_failed",
                level="warning",
                sample_trigger=sample_trigger,
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False
    @staticmethod
    def _ensure_turn_processing(turn: TurnBuffer) -> None:
        if turn.phase != TURN_PHASE_PROCESSING:
            raise asyncio.CancelledError
    @staticmethod
    def _after_commit_ms(
        turn: TurnBuffer, *, observed_at: float | None = None
    ) -> float | None:
        if turn.commit_started_at is None:
            return None
        observed_at = observed_at if observed_at is not None else time.perf_counter()
        return round(max(0.0, observed_at - turn.commit_started_at) * 1000.0, 3)
    @staticmethod
    def _register_turn_request(turn: TurnBuffer, request_id: str) -> None:
        turn.active_request_ids.add(request_id)
        turn.current_request_id = request_id
    @staticmethod
    def _unregister_turn_request(turn: TurnBuffer, request_id: str) -> None:
        turn.active_request_ids.discard(request_id)
        if turn.current_request_id == request_id:
            turn.current_request_id = next(iter(turn.active_request_ids), None)


MultimodalTurnMixin = TurnPipeline
