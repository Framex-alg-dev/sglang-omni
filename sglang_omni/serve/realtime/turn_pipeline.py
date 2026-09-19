"""Turn-level preparation, concurrent inference, and finalization."""

from __future__ import annotations

from sglang_omni.serve.realtime.turn_intent import (
    EarlyBodyIntent,
    VISUAL_GESTURE_ANSWER_GATE,
    infer_turn_intent,
)

import asyncio
import hashlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CANDIDATE_REACTION_SOURCE_LANGUAGE,
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.serve.realtime.action.category import (
    DIRECT_GREETING_CANDIDATE_ID,
    DIRECT_GREETING_REACTION,
    DIRECT_GREETING_ROUTE,
)
from sglang_omni.serve.realtime.action.decision import decision_as_dict
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
from sglang_omni.serve.realtime.proactive import proactive_scene_policy
from sglang_omni.serve.realtime.knowledge import KnowledgeEntityHint
from sglang_omni.serve.realtime.knowledge.models import PreparedKnowledgeTurn
from sglang_omni.serve.realtime.performance import (
    PerformanceDecision,
    fuse_performance_decision,
)
from sglang_omni.serve.realtime.action.routing import (
    resolve_unique_source_label_action,
    route_numeric_reply_action,
)
from sglang_omni.serve.realtime.action.visual_generation import (
    VISUAL_GESTURE_COPY_ROUTES,
)

logger = logging.getLogger(__name__)

# Action-scoring request IDs append stage suffixes to this value and the public
# scoring contract caps the final ID at 128 characters.  Provider-visible
# session/Turn IDs are intentionally unbounded, so do not embed them verbatim.
_REQUEST_ID_COMPONENT_HEX_LENGTH = 16


def _build_request_base(session_id: str, turn_id: str) -> str:
    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[
        :_REQUEST_ID_COMPONENT_HEX_LENGTH
    ]
    turn_digest = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[
        :_REQUEST_ID_COMPONENT_HEX_LENGTH
    ]
    return f"session-{session_digest}-turn-{turn_digest}-{uuid.uuid4().hex}"


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


def _is_direct_greeting_intent(turn: TurnBuffer) -> bool:
    intent = turn.intent
    return bool(
        turn.turn_origin == TURN_ORIGIN_USER
        and intent is not None
        and intent.speech == "generated"
        and intent.body_mode == "none"
        and not intent.history
        and intent.reaction_mode == "respond"
        and intent.reaction.strip() == DIRECT_GREETING_REACTION
    )


def _replace_speculative_action_with_greeting(
    result: tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]],
    greeting_candidate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
    previous_action = result[0]
    action_context = dict(result[3])
    action_context.update(
        {
            "selection_stages": 1,
            "selection_mode": "intent_reconciled",
            "selection_basis": DIRECT_GREETING_ROUTE,
            "forced_semantic_tag": DIRECT_GREETING_ROUTE,
            "speculative_candidate_id": (
                previous_action.get("candidate_id")
                if previous_action is not None
                else None
            ),
            "child_scoring_skipped": True,
            "child_scoring_skip_reason": DIRECT_GREETING_ROUTE,
            "support_status": "supported",
            "fallback_applied": False,
        }
    )
    action = {
        "candidate_id": greeting_candidate.candidate_id,
        "action_id": greeting_candidate.action_id,
        "category_id": greeting_candidate.category_id,
        "execution_binding": dict(greeting_candidate.execution_binding),
        "execute": greeting_candidate.action_id != "no_action",
        "support_status": "supported",
        "fallback_applied": False,
    }
    return action, [], result[2], action_context


def _replace_speculative_action_with_exact_intent_candidate(
    result: tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]],
    candidate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
    """Replace an ambiguous speculative winner without another model call."""

    previous_action = result[0]
    action_context = dict(result[3])
    action_context.update(
        {
            "selection_stages": 1,
            "selection_mode": "intent_reconciled",
            "selection_basis": "structured_intent_exact_source_label",
            "speculative_candidate_id": (
                previous_action.get("candidate_id")
                if previous_action is not None
                else None
            ),
            "child_scoring_skipped": True,
            "child_scoring_skip_reason": (
                "structured_intent_exact_source_label"
            ),
            "support_status": "supported",
            "fallback_applied": False,
        }
    )
    action = {
        "candidate_id": candidate.candidate_id,
        "action_id": candidate.action_id,
        "category_id": candidate.category_id,
        "execution_binding": dict(candidate.execution_binding),
        "execute": candidate.action_id != "no_action",
        "support_status": "supported",
        "fallback_applied": False,
    }
    return action, [], result[2], action_context


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
    expression: dict[str, Any] | None
    performance: PerformanceDecision | None


class TurnPipeline:
    async def _wait_for_knowledge_commit(self) -> None:
        pending = self._knowledge_commit_task
        if pending is None:
            return
        await asyncio.shield(pending)

    async def _apply_knowledge_script_event(
        self,
        *,
        turn_id: str,
        script_id: str,
        event: str,
        script_version: int | None,
        checksum: str | None,
    ) -> Any:
        await self._wait_for_knowledge_commit()
        async with self._knowledge_state_lock:
            binding = self.knowledge_binding
            if binding is None or self.knowledge_controller is None:
                raise RuntimeError("knowledge script event requires a binding")
            updated = await self.knowledge_controller.script_event(
                binding=binding,
                session_id=self.session_id,
                turn_id=turn_id,
                script_id=script_id,
                event=event,
                script_version=script_version,
                checksum=checksum,
            )
            self.knowledge_binding = updated
            return updated

    def _start_knowledge_commit(
        self, prepared: PreparedKnowledgeTurn
    ) -> asyncio.Task[Any]:
        existing = self._knowledge_commit_task
        if existing is not None and not existing.done():
            raise RuntimeError("a knowledge state commit is already in progress")

        async def commit() -> Any:
            async with self._knowledge_state_lock:
                binding = self.knowledge_binding
                if binding is None or self.knowledge_controller is None:
                    raise RuntimeError("knowledge binding disappeared before commit")
                context = await self.knowledge_controller.commit_prepared_turn(
                    binding=binding,
                    prepared=prepared,
                )
                if context.degraded_code in {
                    "COMMIT_OUTCOME_UNKNOWN",
                    "STATE_VERSION_CONFLICT",
                    "PREPARATION_EXPIRED",
                    "PREPARATION_MISMATCH",
                }:
                    self.knowledge_binding = replace(binding, status="degraded")
                else:
                    self.knowledge_binding = replace(
                        binding,
                        state_token=context.state_token,
                    )
                return context

        task = asyncio.create_task(
            commit(),
            name=f"session-knowledge-commit-{self.session_id}-{prepared.turn_id}",
        )
        self._knowledge_commit_task = task

        def clear(completed: asyncio.Task[Any]) -> None:
            if (
                self._knowledge_commit_task is completed
                and not completed.cancelled()
                and completed.exception() is None
            ):
                self._knowledge_commit_task = None

        task.add_done_callback(clear)
        return task

    async def _consume_prepared_knowledge(
        self,
        prepare_task: asyncio.Task[PreparedKnowledgeTurn],
        *,
        turn: TurnBuffer,
        route_completed_at: float,
    ) -> Any:
        consume_started = time.perf_counter()
        prepared = await prepare_task
        prepare_timing = prepared.payload.get("timing_ms", {})
        emit_structured_log(
            "performance",
            "knowledge_prepare_ready",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            preparation_id=prepared.preparation_id,
            decision=prepared.payload.get("decision"),
            prepare_ms=prepare_timing.get("total"),
            error_code=prepared.error_code,
        )
        commit_started = time.perf_counter()
        commit_task = self._start_knowledge_commit(prepared)
        emit_structured_log(
            "performance",
            "knowledge_commit_started",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            preparation_id=prepared.preparation_id,
        )
        context = await asyncio.shield(commit_task)
        emit_structured_log(
            "performance",
            "knowledge_commit_completed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            preparation_id=prepared.preparation_id,
            decision=context.decision,
            degraded_code=context.degraded_code,
            commit_ms=round((time.perf_counter() - commit_started) * 1000, 3),
            route_overlap_ms=round(
                max(0.0, route_completed_at - prepared.started_at) * 1000,
                3,
            ),
            effective_knowledge_wait_ms=round(
                (time.perf_counter() - consume_started) * 1000,
                3,
            ),
        )
        return context

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
        if "scene_context" in event:
            scene_context = event.get("scene_context")
            if scene_context is not None and not isinstance(scene_context, str):
                raise ValueError("scene_context must be a string or null")
            turn.scene_context = scene_context
        if "scene_reply_guidance" in event:
            scene_reply_guidance = event.get("scene_reply_guidance")
            if scene_reply_guidance is not None and not isinstance(
                scene_reply_guidance, str
            ):
                raise ValueError("scene_reply_guidance must be a string or null")
            turn.scene_reply_guidance = scene_reply_guidance
        if "knowledge_entity_hints" in event:
            turn.knowledge_entity_hints = tuple(
                KnowledgeEntityHint(
                    type=item["type"],
                    external_id=item["external_id"],
                    display_name=item.get("display_name"),
                )
                for item in (event.get("knowledge_entity_hints") or [])
            )
        if "knowledge_script_id" in event:
            turn.knowledge_script_id = event.get("knowledge_script_id")
            turn.knowledge_script_version = event.get("knowledge_script_version")
            turn.knowledge_script_checksum = event.get("knowledge_script_checksum")
        if "action_allowed_candidate_ids" in event:
            turn.action_allowed_candidate_ids = tuple(
                event.get("action_allowed_candidate_ids") or ()
            )
        if "action_excluded_candidate_ids" in event:
            turn.action_excluded_candidate_ids = tuple(
                event.get("action_excluded_candidate_ids") or ()
            )
        if "last_executed_action_id" in event:
            turn.client_last_executed_action_id = event.get(
                "last_executed_action_id"
            )
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
        if turn.turn_origin == TURN_ORIGIN_PROACTIVE:
            policy = proactive_scene_policy(turn.trigger)
            if turn.trigger != ACTION_FINISHED_TRIGGER:
                effective_state = dict(turn.avatar_state or {})
                client_guidance = effective_state.get("state_description")
                guidance_parts: list[str] = []
                if policy is not None:
                    guidance_parts.append(
                        policy.default_action_guidance(
                            self.action_language
                        ).strip()
                    )
                if isinstance(turn.scene_context, str) and turn.scene_context.strip():
                    guidance_parts.append(
                        self._action_prompt(
                            zh="当前主动场景补充：",
                            en="Current proactive scene refinement: ",
                        )
                        + turn.scene_context.strip()
                    )
                if isinstance(client_guidance, str) and client_guidance.strip():
                    guidance_parts.append(client_guidance.strip())
                if guidance_parts:
                    effective_state["state_description"] = "\n".join(
                        guidance_parts
                    )
                    turn.avatar_state = effective_state

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
        turn.request_base = _build_request_base(self.session_id, turn.turn_id)
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
            scene_context_present=bool(turn.scene_context),
            scene_reply_guidance_present=bool(turn.scene_reply_guidance),
            action_allowed_candidate_count=len(
                turn.action_allowed_candidate_ids
            ),
            action_excluded_candidate_count=len(
                turn.action_excluded_candidate_ids
            ),
            client_last_executed_action_id=turn.client_last_executed_action_id,
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
        expression = outcome.expression
        performance = outcome.performance
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
            reply_text or self.unsupported_action_text
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
        if (
            reply_text
            and turn.turn_origin == TURN_ORIGIN_PROACTIVE
            and turn.trigger == "character_proactive"
            and turn.proactive_memory_thread_ids
            and self.session_memory_store is not None
        ):
            self.session_memory_store.mark_proactive_threads_used(
                turn.proactive_memory_thread_ids
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
        if (
            action is not None
            and action.get("candidate_id") in self.candidate_by_id
        ):
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
                        if (
                            modality == "text"
                            and (silent_action_finished or (
                                suppress_reply_for_unsupported_action and reply_text is None
                            ))
                        ) or (
                            modality == "audio"
                            and suppress_reply_for_unsupported_action
                            and reply_text is not None
                        )
                        else (
                            "not_changed"
                            if modality == "expression" and expression is None
                            else "completed"
                        )
                    )
                )
                for modality in self.modalities
            }
        if suppress_reply_for_unsupported_action and "text" in self.modalities:
            result["reply"] = {
                "source": "client_prerecorded_audio",
                "reason": "unsupported_action",
                **({"text": history_reply_text} if reply_text is not None else {}),
                "recorded_in_history": bool(history_reply_text),
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
        if expression is not None:
            result["expression"] = dict(expression)
        if turn.intent is not None:
            result["timing"]["intent_ms"] = turn.intent.elapsed_ms
        if performance is not None:
            result["timing"]["request_scope"] = performance.request_scope
            result["timing"]["server_performance_compute_ms"] = (
                performance.elapsed_ms
            )
        elif turn.intent is not None:
            body_requested = turn.intent.body_mode == "perform"
            face_requested = bool(turn.intent.face)
            result["timing"]["request_scope"] = (
                "both"
                if body_requested and face_requested
                else "body_only"
                if body_requested
                else "expression_only"
                if face_requested
                else "none"
            )
        if turn.knowledge_context is not None:
            result["knowledge"] = {
                "decision": turn.knowledge_context.decision,
                "reason": turn.knowledge_context.reason,
                "result_id": turn.knowledge_context.result_id,
                "snapshot_id": turn.knowledge_context.snapshot_id,
                "capabilities": list(turn.knowledge_context.capabilities),
                "evidence_refs": [
                    {
                        "evidence_id": item.evidence_id,
                        "source_type": item.source_type,
                        "source_id": item.source_id,
                    }
                    for item in turn.knowledge_context.evidence
                ],
                **(
                    {"degraded_code": turn.knowledge_context.degraded_code}
                    if turn.knowledge_context.degraded_code
                    else {}
                ),
            }
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
        knowledge_script_started = False
        knowledge_script_finished = False

        try:
            prepared_commit = await self._acknowledge_and_prepare_turn(commit_input)
            if prepared_commit is None:
                turn_outcome = "cancelled"
                return
            await self._wait_for_knowledge_commit()
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
            visual_answer_public_reply_task: (
                asyncio.Task[tuple[str, dict[str, Any]]] | None
            ) = None
            provisional_state: ProvisionalReplyState | None = None
            provisional_discard_task: asyncio.Task[Any] | None = None
            rejection_task: asyncio.Task[Any] | None = None
            reply_history_route: ReplyHistoryRouteResult | None = None
            preserve_language_reply_on_unsupported_action = False
            visual_copy_without_speech = False
            reply_route_decision_ready = False
            knowledge_task: asyncio.Task[Any] | None = None
            knowledge_prepare_task: asyncio.Task[PreparedKnowledgeTurn] | None = None
            performance_task: asyncio.Task[PerformanceDecision] | None = None
            performance: PerformanceDecision | None = None
            expression: dict[str, Any] | None = None
            early_expression = False
            action_ready_sent = False
            expression_ready_sent = False
            numeric_reply_route = None
            independently_published_action = None
            intent_task: asyncio.Task[Any] | None = None
            visual_scope_future: asyncio.Future[str] | None = None
            body_intent_future: asyncio.Future[EarlyBodyIntent] | None = None
            visual_arithmetic_probe_task: (
                asyncio.Task[tuple[str, dict[str, Any]] | None] | None
            ) = None
            visual_arithmetic_image_prefetch_task: asyncio.Task[bool] | None = None
            visual_gesture_probe_task: asyncio.Task[Any] | None = None

            def track_branch(coroutine: Any, *, name: str) -> asyncio.Task[Any]:
                started_at = time.perf_counter()
                task = asyncio.create_task(coroutine, name=name)
                turn.branch_tasks.add(task)
                task.add_done_callback(turn.branch_tasks.discard)
                emit_structured_log(
                    "performance", "turn_branch_started", session_id=self.session_id,
                    turn_id=turn_id, trace_id=turn.trace_id, branch=name,
                    after_commit_ms=self._after_commit_ms(turn),
                )
                def finished(completed: asyncio.Task[Any]) -> None:
                    emit_structured_log(
                        "performance", "turn_branch_finished", session_id=self.session_id,
                        turn_id=turn_id, trace_id=turn.trace_id, branch=name,
                        status=("cancelled" if completed.cancelled() else "failed" if completed.exception() is not None else "completed"),
                        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    )
                task.add_done_callback(finished)
                return task

            async def send_expression_ready(value: dict[str, Any]) -> None:
                nonlocal expression_ready_sent
                self._ensure_turn_processing(turn)
                if expression_ready_sent:
                    return
                expression_ready_sent = True
                await self.send({
                    "type": "turn.expression.ready",
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "expression": value,
                })
                emit_structured_log(
                    "performance", "expression_published",
                    session_id=self.session_id, turn_id=turn_id,
                    trace_id=turn.trace_id,
                    after_commit_ms=self._after_commit_ms(turn),
                )

            async def send_action_ready() -> None:
                nonlocal action_ready_sent
                if action_ready_sent:
                    return
                if action is not None:
                    self._ensure_turn_processing(turn)
                    if expression is not None and turn.turn_origin != TURN_ORIGIN_USER:
                        await send_expression_ready(expression)
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
                    if (
                        provisional_state is not None
                        and provisional_state.text_completed_at is not None
                    ):
                        emit_structured_log(
                            "performance",
                            "complete_reply_to_action_ready",
                            session_id=self.session_id,
                            turn_id=turn_id,
                            trace_id=turn.trace_id,
                            selection_basis=action_context.get(
                                "selection_basis"
                            ),
                            elapsed_ms=round(
                                (
                                    time.perf_counter()
                                    - provisional_state.text_completed_at
                                )
                                * 1000.0,
                                3,
                            ),
                        )
                    if (
                        action_error is None
                        and action.get("support_status") != "unsupported"
                        and action.get("candidate_id") in self.candidate_by_id
                    ):
                        self._record_action_as_executed(
                            turn=turn,
                            action=action,
                        )

                action_ready_sent = True

            async def score_and_publish_action(
                *args: Any,
                precomputed_result_task: asyncio.Task[Any] | None = None,
                **kwargs: Any,
            ) -> Any:
                nonlocal action, independently_published_action
                try:
                    if precomputed_result_task is not None:
                        result = await precomputed_result_task
                        if result is None:
                            raise RuntimeError(
                                "visual gesture probe did not produce a result"
                            )
                    else:
                        result = await self._score_action(*args, **kwargs)
                finally:
                    # Do not let the generative reply/detail parser contend
                    # with the latency-critical suffix batch on the same GPU.
                    if intent_detail_release is not None:
                        intent_detail_release.set()
                scored_action = result[0]
                early_body_intent: EarlyBodyIntent | None = None
                if (
                    turn.turn_origin == TURN_ORIGIN_USER
                    and body_intent_future is not None
                ):
                    early_body_intent = await body_intent_future
                action_is_terminal_without_execution = bool(
                    scored_action is not None
                    and (
                        not scored_action.get("execute")
                        or scored_action.get("support_status") == "unsupported"
                    )
                )
                # Unsupported/no-op is intrinsically safe and must not wait for
                # either the legacy JSON intent parser or performance control.
                decision = turn.action_decision
                decision_mode = getattr(
                    self, "action_decision_batch_mode", "off"
                )
                use_batched_decision = bool(
                    decision_mode == "enforce"
                    and decision is not None
                )
                # In enforce mode, the bounded grouped labels are the action
                # publication authority. A non-executing result is always safe
                # to publish, and a non-social grouped result no longer waits
                # for the autoregressive detail JSON. Social reactions retain
                # the barrier because the completed intent may reconcile the
                # selected catalog action to the dedicated greeting action.
                waited_for_unified_intent = False
                category_decision = turn.action_category_decision
                ambiguous_concrete_category = bool(
                    category_decision is not None
                    and category_decision.category_id is not None
                    and category_decision.margin is not None
                    and category_decision.margin
                    < float(
                        getattr(self, "action_decision_min_margin", 0.10)
                    )
                )
                should_wait_for_unified_intent = bool(
                    turn.turn_origin == TURN_ORIGIN_USER
                    and intent_task is not None
                    and not action_is_terminal_without_execution
                    and (
                        not use_batched_decision
                        or decision.reaction_type != "none"
                        or ambiguous_concrete_category
                    )
                )
                if should_wait_for_unified_intent:
                    turn.intent = await intent_task
                    waited_for_unified_intent = True
                intent = turn.intent

                reconciled_body_task = (
                    early_body_intent.body_task
                    if early_body_intent is not None
                    and early_body_intent.body_intent == "perform"
                    else intent.body
                    if intent is not None and intent.body_mode == "perform"
                    else ""
                )
                if (
                    reconciled_body_task
                    and scored_action is not None
                ):
                    eligible_pairs = [
                        (category, candidate)
                        for category in self.categories
                        for candidate in self._filter_turn_action_candidates(
                            turn, list(category.children)
                        )
                    ]
                    exact_route = resolve_unique_source_label_action(
                        reconciled_body_task, eligible_pairs
                    )
                    if exact_route is not None:
                        action_image_roles = (
                            args[2]
                            if len(args) > 2
                            else kwargs.get("image_roles", [])
                        )
                        action_avatar_state = (
                            args[4]
                            if len(args) > 4
                            else kwargs.get("avatar_state")
                        )
                        effective_avatar_state = self._effective_avatar_state(
                            action_avatar_state,
                            turn_origin=turn.turn_origin,
                            has_avatar_image=(
                                IMAGE_ROLE_AVATAR_STATE in action_image_roles
                            ),
                        )
                        state_description = effective_avatar_state.get(
                            "state_description"
                        )
                        route_is_prohibited = bool(
                            exact_route.category.category_id
                            in self._state_description_excluded_category_ids(
                                state_description
                            )
                            or exact_route.candidate.candidate_id
                            in self._state_description_excluded_candidate_ids(
                                state_description,
                                [exact_route.candidate],
                            )
                        )
                        if (
                            not route_is_prohibited
                            and (
                                exact_route.candidate.candidate_id
                                != scored_action.get("candidate_id")
                                or not scored_action.get("execute")
                                or scored_action.get("support_status")
                                != "supported"
                            )
                        ):
                            speculative_candidate_id = scored_action.get(
                                "candidate_id"
                            )
                            result = (
                                _replace_speculative_action_with_exact_intent_candidate(
                                    result, exact_route.candidate
                                )
                            )
                            scored_action = result[0]
                            action_is_terminal_without_execution = bool(
                                not scored_action.get("execute")
                                or scored_action.get("support_status")
                                == "unsupported"
                            )
                            emit_structured_log(
                                "action",
                                "action_reconciled_from_body_intent",
                                session_id=self.session_id,
                                turn_id=turn_id,
                                trace_id=turn.trace_id,
                                speculative_candidate_id=(
                                    speculative_candidate_id
                                ),
                                selected_candidate_id=(
                                    exact_route.candidate.candidate_id
                                ),
                                category_margin=getattr(
                                    category_decision, "margin", None
                                ),
                                model_request_added=False,
                                match_mode="exact_source_label",
                                body_intent_source=(
                                    "streaming_body_channel"
                                    if early_body_intent is not None
                                    else "completed_intent"
                                ),
                            )

                # In enforce mode, action scoring is intentionally speculative:
                # it may finish before unified intent identifies a plain greeting.
                # Reconcile that one deterministic semantic route at the existing
                # publication barrier. This only swaps in a catalog candidate; it
                # does not issue another model request or re-score the action set.
                if _is_direct_greeting_intent(turn):
                    greeting_candidate = self.candidate_by_id.get(
                        DIRECT_GREETING_CANDIDATE_ID
                    )
                    catalog_candidate = (
                        self.global_action_catalog.candidate_by_id.get(
                            DIRECT_GREETING_CANDIDATE_ID
                        )
                        if self.global_action_catalog is not None
                        else None
                    )
                    greeting_category = next(
                        (
                            category
                            for category in self.categories
                            if any(
                                child.candidate_id
                                == DIRECT_GREETING_CANDIDATE_ID
                                for child in category.children
                            )
                        ),
                        None,
                    )
                    action_image_roles = (
                        args[2]
                        if len(args) > 2
                        else kwargs.get("image_roles", [])
                    )
                    action_avatar_state = (
                        args[4]
                        if len(args) > 4
                        else kwargs.get("avatar_state")
                    )
                    effective_avatar_state = self._effective_avatar_state(
                        action_avatar_state,
                        turn_origin=turn.turn_origin,
                        has_avatar_image=(
                            IMAGE_ROLE_AVATAR_STATE in action_image_roles
                        ),
                    )
                    greeting_is_prohibited = bool(
                        greeting_candidate is not None
                        and greeting_category is not None
                        and (
                            greeting_category.category_id
                            in self._state_description_excluded_category_ids(
                                effective_avatar_state.get("state_description")
                            )
                            or greeting_candidate.candidate_id
                            in self._state_description_excluded_candidate_ids(
                                effective_avatar_state.get("state_description"),
                                [greeting_candidate],
                            )
                        )
                    )
                    greeting_is_allowed = bool(
                        greeting_candidate is not None
                        and greeting_category is not None
                        and self._turn_candidate_is_allowed(
                            turn, greeting_candidate
                        )
                        and catalog_candidate is not None
                        and CANDIDATE_REACTION_SOURCE_LANGUAGE
                        in catalog_candidate.reaction_sources
                        and not greeting_is_prohibited
                    )
                    if (
                        greeting_is_allowed
                        and scored_action is not None
                        and scored_action.get("candidate_id")
                        != DIRECT_GREETING_CANDIDATE_ID
                    ):
                        speculative_candidate_id = scored_action.get(
                            "candidate_id"
                        )
                        result = _replace_speculative_action_with_greeting(
                            result, greeting_candidate
                        )
                        scored_action = result[0]
                        action_is_terminal_without_execution = False
                        emit_structured_log(
                            "action",
                            "speculative_greeting_action_reconciled",
                            session_id=self.session_id,
                            turn_id=turn_id,
                            trace_id=turn.trace_id,
                            speculative_candidate_id=(
                                speculative_candidate_id
                            ),
                            selected_candidate_id=(
                                DIRECT_GREETING_CANDIDATE_ID
                            ),
                            model_request_added=False,
                        )

                unified_intent_allows_body = bool(
                    intent is not None
                    and intent.visual_scope_gate
                    != VISUAL_GESTURE_ANSWER_GATE
                    and intent.body_mode != "prohibit"
                    and (
                        intent.body_mode == "perform"
                        or intent.reaction_mode == "respond"
                    )
                )
                if use_batched_decision:
                    # A confident grouped perform/reaction gate is the
                    # authoritative publication decision.  The generated JSON
                    # remains a detail/fallback parser and must not veto an
                    # independently confident bounded classification.  Deny
                    # and low-confidence grouped outcomes remain fail-closed.
                    grouped_authoritative_allow = decision.allows_body
                    intent_allows_body = bool(grouped_authoritative_allow)
                    if early_body_intent is not None:
                        if early_body_intent.body_intent == "perform":
                            intent_allows_body = True
                        elif not (
                            early_body_intent.body_intent == "none"
                            and decision.reaction_type != "none"
                        ):
                            intent_allows_body = False
                    emit_structured_log(
                        "action", "batched_action_decision_enforced",
                        session_id=self.session_id, turn_id=turn_id,
                        trace_id=turn.trace_id,
                        body_mode=decision.body_mode,
                        face_mode=decision.face_mode,
                        reaction_type=decision.reaction_type,
                        visual_scope=decision.visual_scope,
                        confidence_margin=decision.body_gate_margin,
                        low_confidence_fail_closed=(
                            not decision.body_gate_confident
                        ),
                        all_groups_min_margin=decision.min_margin,
                        all_groups_confident=decision.confident,
                        waited_for_unified_intent=waited_for_unified_intent,
                        grouped_authoritative_allow=(
                            grouped_authoritative_allow
                        ),
                    )
                else:
                    intent_allows_body = unified_intent_allows_body

                unsafe_decision_disagreement = False
                if decision is not None and intent is not None:
                    legacy_reaction_active = intent.reaction_mode == "respond"
                    decision_reaction_active = decision.reaction_type != "none"
                    reaction_perform_is_compatible = bool(
                        decision.body_mode == "perform"
                        and intent.body_mode == "none"
                        and decision_reaction_active
                        and legacy_reaction_active
                    )
                    disagreements = {
                        "body": (
                            decision.body_mode != intent.body_mode
                            and not reaction_perform_is_compatible
                            and not (
                                decision.body_mode == "capability_query"
                                and intent.body_mode == "none"
                            )
                        ),
                        "face": (
                            (decision.face_mode == "perform")
                            != bool(intent.face)
                        ),
                        "reaction": (
                            decision_reaction_active
                            != legacy_reaction_active
                        ),
                        "visual": bool(
                            getattr(
                                self, "action_decision_batch_visual", False
                            )
                            and (
                                (
                                    "answer"
                                    if decision.visual_scope
                                    == VISUAL_GESTURE_ANSWER_GATE
                                    else "copy"
                                    if decision.visual_scope
                                    else "general"
                                )
                                != (
                                    "answer"
                                    if intent.visual_scope_gate
                                    == VISUAL_GESTURE_ANSWER_GATE
                                    else "copy"
                                    if intent.visual_scope_gate
                                    else "general"
                                )
                            )
                        ),
                    }
                    unsafe_decision_disagreement = bool(
                        disagreements["body"] or disagreements["visual"]
                    )
                    if (
                        use_batched_decision
                        and unsafe_decision_disagreement
                        and not grouped_authoritative_allow
                    ):
                        intent_allows_body = False
                    emit_structured_log(
                        "diagnostic", "action_decision_shadow_compared",
                        session_id=self.session_id, turn_id=turn_id,
                        trace_id=turn.trace_id,
                        mode=decision_mode,
                        disagreements=disagreements,
                        unsafe_disagreement=unsafe_decision_disagreement,
                        decision_body_mode=decision.body_mode,
                        legacy_body_mode=intent.body_mode,
                        decision_face_mode=decision.face_mode,
                        legacy_has_face=bool(intent.face),
                        decision_reaction_type=decision.reaction_type,
                        legacy_reaction_mode=intent.reaction_mode,
                        reaction_perform_is_compatible=(
                            reaction_perform_is_compatible
                        ),
                        decision_visual_scope=decision.visual_scope,
                        legacy_visual_scope=intent.visual_scope_gate,
                        confidence_margin=decision.body_gate_margin,
                        all_groups_min_margin=decision.min_margin,
                    )
                if (
                    turn.turn_origin == TURN_ORIGIN_USER
                    and (intent is not None or use_batched_decision)
                    and not intent_allows_body
                    and result[0] is not None
                    and result[0].get("execute")
                ):
                    blocked = dict(result[0])
                    blocked.update(
                        execute=False,
                        support_status="unsupported",
                        intent_gate_blocked=True,
                        reason_code="intent_gate_blocked",
                    )
                    result = (blocked, *result[1:])
                    action_is_terminal_without_execution = True
                    emit_structured_log(
                        "action", "speculative_action_blocked_by_intent",
                        session_id=self.session_id, turn_id=turn_id,
                        trace_id=turn.trace_id,
                        body_mode=(
                            decision.body_mode
                            if use_batched_decision else intent.body_mode
                        ),
                        reaction_mode=(
                            decision.reaction_type
                            if use_batched_decision else intent.reaction_mode
                        ),
                        has_face=(
                            decision.face_mode == "perform"
                            if use_batched_decision else bool(intent.face)
                        ),
                        visual_scope_gate=(
                            decision.visual_scope
                            if use_batched_decision
                            else intent.visual_scope_gate
                        ),
                        candidate_id=blocked.get("candidate_id"),
                        intent_gate_source=(
                            "unified_intent+batched_labels"
                            if use_batched_decision else "unified_intent"
                        ),
                        unsafe_decision_disagreement=(
                            unsafe_decision_disagreement
                        ),
                    )
                if (
                    turn.turn_origin == TURN_ORIGIN_USER
                    and (
                        intent_allows_body
                        or action_is_terminal_without_execution
                    )
                ):
                    action = result[0]
                    independently_published_action = dict(action) if action else None
                    await send_action_ready()
                    emit_structured_log(
                        "performance",
                        (
                            "body_action_independently_published"
                            if action and action.get("execute")
                            else "action_result_independently_published"
                        ),
                        session_id=self.session_id, turn_id=turn_id,
                        trace_id=turn.trace_id,
                        after_commit_ms=self._after_commit_ms(turn),
                        execute=bool(action and action.get("execute")),
                        support_status=(
                            action.get("support_status") if action else None
                        ),
                        waited_for_performance=False, waited_for_reply=False,
                    )
                return result

            def maybe_schedule_category_discard() -> None:
                nonlocal provisional_discard_task
                if (
                    reply_route_decision_ready
                    and category_support_status == "unsupported"
                    and "expression" not in self.modalities
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
                if early_expression:
                    return
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
            if provided_reply and turn.knowledge_script_id is not None:
                if self.knowledge_binding is None:
                    raise ValueError(
                        "a bound knowledge session is required for a provided script"
                    )
                if self.knowledge_binding.mode == "retrieval":
                    if self.knowledge_controller is None:
                        raise ValueError(
                            "Knowledge Gateway integration is not configured"
                        )
                    previous_knowledge_binding = self.knowledge_binding
                    self.knowledge_binding = await self._apply_knowledge_script_event(
                        turn_id=turn.turn_id,
                        script_id=turn.knowledge_script_id,
                        event="started",
                        script_version=turn.knowledge_script_version,
                        checksum=turn.knowledge_script_checksum,
                    )
                    knowledge_script_started = (
                        self.knowledge_binding.status == "ready"
                        and self.knowledge_binding.state_token
                        != previous_knowledge_binding.state_token
                    )
                    if knowledge_script_started:
                        emit_structured_log(
                            "diagnostic", "knowledge_script_started",
                            session_id=self.session_id, turn_id=turn.turn_id,
                            trace_id=turn.trace_id, script_id=turn.knowledge_script_id,
                            snapshot_id=self.knowledge_binding.snapshot_id,
                        )
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

            action_current_images = prepared_current_images
            action_current_image_roles = current_image_roles
            visual_scope_code = ""
            visual_gesture_answer = False
            body_action_not_requested = False
            intent_supports_scope_future = False
            intent_detail_release: asyncio.Event | None = None
            if turn.turn_origin == TURN_ORIGIN_USER and not provided_reply and (turn.text or current_audio_list):
                visual_scope_future = asyncio.get_running_loop().create_future()
                intent_supports_body_future = (
                    "body_intent_future"
                    in inspect.signature(infer_turn_intent).parameters
                )
                if intent_supports_body_future:
                    body_intent_future = asyncio.get_running_loop().create_future()
                intent_supports_scope_future = (
                    "visual_scope_future"
                    in inspect.signature(infer_turn_intent).parameters
                )
                if (
                    "full_intent_start_event"
                    in inspect.signature(infer_turn_intent).parameters
                    and self.direct_action_selection
                    and "action" in self.modalities
                    and getattr(self, "action_decision_batch_mode", "off")
                    == "enforce"
                ):
                    intent_detail_release = asyncio.Event()
                if not intent_supports_scope_future:
                    visual_scope_future.set_result("")
                intent_task = track_branch(
                    infer_turn_intent(
                        self,
                        turn,
                        current_audio_list,
                        prepared_current_images,
                        current_image_roles,
                        **(
                            {
                                "visual_scope_future": visual_scope_future,
                                **(
                                    {"body_intent_future": body_intent_future}
                                    if intent_supports_body_future
                                    else {}
                                ),
                                **(
                                    {
                                        "full_intent_start_event": (
                                            intent_detail_release
                                        )
                                    }
                                    if intent_detail_release is not None
                                    else {}
                                ),
                            }
                            if intent_supports_scope_future
                            else {}
                        ),
                    ),
                    name=f"session-intent-{self.session_id}-{turn.turn_id}",
                )
                if (
                    getattr(self, "image_encoder_prefetch_enabled", False)
                    and fusion_reply
                    and provisional_state is not None
                    and "action" in self.modalities
                    and IMAGE_ROLE_USER_CAMERA in current_image_roles
                ):
                    visual_arithmetic_image_prefetch_task = track_branch(
                        self._run_visual_arithmetic_image_prefetch(
                            turn,
                            prepared_current_images,
                            current_image_roles,
                            visual_scope_future,
                        ),
                        name=(
                            f"session-visual-arithmetic-image-prefetch-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                if (
                    fusion_reply
                    and provisional_state is not None
                    and "action" in self.modalities
                    and IMAGE_ROLE_USER_CAMERA in current_image_roles
                ):
                    # Wait only for the first visual_route field from the same
                    # unified intent request. This is not a second gate: it
                    # prevents non-arithmetic camera turns from submitting a
                    # speculative GPU request while allowing operand extraction
                    # to overlap the remainder of unified intent generation.
                    visual_arithmetic_probe_task = track_branch(
                        self._run_visual_arithmetic_probe_after_route(
                            turn,
                            current_audio_list,
                            prepared_current_images,
                            current_image_roles,
                            visual_scope_future,
                            visual_arithmetic_image_prefetch_task,
                        ),
                        name=(
                            f"session-visual-arithmetic-probe-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                if (
                    getattr(self, "visual_gesture_generation_enabled", False)
                    and "action" in self.modalities
                    and IMAGE_ROLE_USER_CAMERA in current_image_roles
                ):
                    # This task waits on the first field of the same unified
                    # intent stream. Generic action imitation and explicit hand
                    # imitation both start one short semantic-label generation;
                    # every other route exits without model work.
                    visual_gesture_probe_task = track_branch(
                        self._run_visual_gesture_probe_after_route(
                            turn,
                            prepared_current_images,
                            current_image_roles,
                            visual_scope_future,
                        ),
                        name=(
                            f"session-visual-gesture-probe-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                batched_visual_gate = bool(
                    self.direct_action_selection
                    and getattr(self, "action_decision_batch_mode", "off")
                    == "enforce"
                    and getattr(self, "action_decision_batch_visual", False)
                )
                if batched_visual_gate:
                    # The IV00/IV01/IV11 safety group is scored with the
                    # concrete actions. Exact copy scope comes from the unified
                    # intent parser and is not duplicated in this PPL batch.
                    # Keep current camera pixels because the result is not yet
                    # known; publication remains behind the grouped gate.
                    emit_structured_log(
                        "performance", "visual_scope_gate_deferred_to_action_batch",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        user_camera_image_count=sum(
                            role == IMAGE_ROLE_USER_CAMERA
                            for role in current_image_roles
                        ),
                    )
                else:
                    # This resolves immediately for text-only turns and after
                    # the bounded language gate for camera turns. V00 can drop
                    # camera pixels before speculative action scoring starts.
                    visual_scope_code = await visual_scope_future
                    visual_gesture_answer = (
                        visual_scope_code == VISUAL_GESTURE_ANSWER_GATE
                    )
                    if intent_supports_scope_future and not visual_scope_code:
                        filtered = [
                            (image, role)
                            for image, role in zip(
                                prepared_current_images,
                                current_image_roles,
                                strict=True,
                            )
                            if role != IMAGE_ROLE_USER_CAMERA
                        ]
                        action_current_images = [item[0] for item in filtered]
                        action_current_image_roles = [item[1] for item in filtered]
                        emit_structured_log(
                            "performance", "action_user_camera_omitted",
                            session_id=self.session_id, turn_id=turn.turn_id,
                            trace_id=turn.trace_id,
                            omitted_count=(
                                len(prepared_current_images)
                                - len(action_current_images)
                            ),
                            visual_scope_gate=visual_scope_code,
                        )

            reply_history_route_task: asyncio.Task[Any] | None = None
            async def start_reply_history_route() -> Any:
                # Delay model-backed routing along with reply/performance work.
                await action_priority_released.wait()
                return await self._classify_reply_history_requirement(
                    turn, current_audio_list, current_text=turn.text,
                )

            action_priority_released = asyncio.Event()
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
                    start_reply_history_route(),
                    name=(
                        f"session-reply-history-route-{self.session_id}-"
                        f"{turn.turn_id}"
                    ),
                )

            action_task: asyncio.Task[Any] | None = None
            action_started: float | None = None

            def start_action_scoring() -> None:
                nonlocal action_task, action_started
                if (
                    "action" not in self.modalities
                    or action_task is not None
                    or early_expression
                    or visual_gesture_answer
                    or body_action_not_requested
                ):
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
                    score_and_publish_action(
                        current_audio_list,
                        action_current_images,
                        action_current_image_roles,
                        turn.intent.body_context(turn.text) if turn.intent and not self.direct_action_selection else turn.text,
                        turn.avatar_state,
                        turn_origin=turn.turn_origin,
                        text_role=turn.text_role,
                        trigger=turn.trigger,
                        turn_id=turn_id,
                        turn=turn,
                        request_base=turn.request_base,
                        provisional_reply=provisional_state,
                        precomputed_result_task=(
                            visual_gesture_probe_task
                            if (
                                getattr(
                                    self,
                                    "visual_gesture_generation_enabled",
                                    False,
                                )
                                and turn.intent is not None
                                and turn.intent.visual_scope_gate
                                in VISUAL_GESTURE_COPY_ROUTES
                            )
                            else None
                        ),
                        on_category_selected=(
                            on_category_selected
                            if "text" in self.modalities
                            else None
                        ),
                    ),
                    name=f"session-action-{self.session_id}-{turn.turn_id}",
                )

            # Only enforce mode may select an action from the raw turn before
            # the canonical intent is ready: its grouped labels are the active
            # safety gate. Shadow/off mode still uses the legacy intent as the
            # authority, so wait for it before scoring as well; otherwise mixed
            # commands such as "说二比一" lose the normalized body target and
            # exact-action routing cannot constrain the candidate set.
            if (
                intent_task is not None
                and intent_supports_scope_future
                and self.direct_action_selection
                and getattr(self, "action_decision_batch_mode", "off")
                == "enforce"
                and not visual_gesture_answer
                # A COPY_* gate result is already available and the intent task
                # will immediately materialize its canonical body/face target.
                # Wait for that cheap hand-off so candidate scoring can apply the
                # correct visual scope instead of racing with turn.intent=None.
                and not visual_scope_code
            ):
                start_action_scoring()
            if intent_task is not None:
                turn.intent = await intent_task
                self._ensure_turn_processing(turn)
                # The streamed first field is only a scheduling hint. If the
                # completed, validated intent repairs an early GENERAL result
                # into COPY_*, restore the camera frames before action scoring.
                # This is especially important for audio turns, where the
                # normalized imperative exists only in the completed JSON.
                if (
                    turn.intent is not None
                    and turn.intent.visual_scope_gate.startswith("COPY_")
                    and not visual_scope_code
                    and action_task is None
                ):
                    action_current_images = prepared_current_images
                    action_current_image_roles = current_image_roles
                    emit_structured_log(
                        "performance",
                        "action_user_camera_restored_after_intent",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        restored_count=sum(
                            role == IMAGE_ROLE_USER_CAMERA
                            for role in current_image_roles
                        ),
                        parsed_visual_scope_gate=(
                            turn.intent.visual_scope_gate
                        ),
                    )
                visual_gesture_answer = bool(
                    turn.intent is not None
                    and turn.intent.visual_scope_gate
                    == VISUAL_GESTURE_ANSWER_GATE
                )
                visual_copy_gesture = bool(
                    turn.intent is not None
                    and turn.intent.visual_scope_gate
                    in VISUAL_GESTURE_COPY_ROUTES
                )
                if (
                    getattr(self, "visual_gesture_generation_enabled", False)
                    and visual_copy_gesture
                    and (
                        visual_gesture_probe_task is None
                        or visual_scope_future is None
                        or not visual_scope_future.done()
                        or visual_scope_future.result()
                        not in VISUAL_GESTURE_COPY_ROUTES
                    )
                ):
                    # The completed JSON is authoritative. If its repaired
                    # route disagrees with the streamed scheduling hint, issue
                    # exactly one semantic gesture request now, never a PPL
                    # retry or re-score.
                    if (
                        visual_gesture_probe_task is not None
                        and not visual_gesture_probe_task.done()
                    ):
                        visual_gesture_probe_task.cancel()
                    if visual_gesture_probe_task is not None:
                        await asyncio.gather(
                            visual_gesture_probe_task,
                            return_exceptions=True,
                        )
                    visual_gesture_probe_task = track_branch(
                        self._run_visual_gesture_probe(
                            turn,
                            prepared_current_images,
                            current_image_roles,
                        ),
                        name=(
                            f"session-visual-gesture-probe-fallback-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                if (
                    getattr(self, "visual_gesture_generation_enabled", False)
                    and not visual_copy_gesture
                    and visual_gesture_probe_task is not None
                ):
                    if not visual_gesture_probe_task.done():
                        visual_gesture_probe_task.cancel()
                    await asyncio.gather(
                        visual_gesture_probe_task,
                        return_exceptions=True,
                    )
                    visual_gesture_probe_task = None
                visual_copy_without_speech = bool(
                    getattr(self, "visual_gesture_generation_enabled", False)
                    and visual_copy_gesture
                    and turn.intent is not None
                    and turn.intent.speech == "none"
                )
                if visual_copy_without_speech:
                    # Intent itself proves that neither conversation history
                    # nor a language reply is needed. Cancel the still-blocked
                    # history/speech route before releasing lower-priority GPU
                    # work, so pure imitation has one visual model request.
                    if reply_history_route_task is not None:
                        reply_history_route_task.cancel()
                        await asyncio.gather(
                            reply_history_route_task,
                            return_exceptions=True,
                        )
                        reply_history_route_task = None
                    reply_history_route = ReplyHistoryRouteResult(
                        decision=REPLY_HISTORY_CURRENT_ONLY,
                        reply_mode=REPLY_MODE_PURE_ACTION,
                        fallback_reason="visual_copy_without_speech",
                    )
                body_action_not_requested = bool(
                    turn.turn_origin == TURN_ORIGIN_USER
                    and not provided_reply
                    and turn.intent is not None
                    and turn.intent.body_mode == "none"
                    and turn.intent.reaction_mode == "none"
                    and not visual_gesture_answer
                    and not (
                        getattr(
                            self, "action_decision_batch_mode", "off"
                        )
                        == "enforce"
                        and turn.action_decision is not None
                        and turn.action_decision.allows_body
                    )
                )
                if (
                    visual_gesture_answer
                    and action_task is not None
                    and not batched_visual_gate
                ):
                    action_task.cancel()
                    await asyncio.gather(action_task, return_exceptions=True)
                if (
                    visual_gesture_answer
                    and visual_arithmetic_probe_task is not None
                    and visual_scope_future is not None
                    and visual_scope_future.done()
                    and visual_scope_future.result()
                    != VISUAL_GESTURE_ANSWER_GATE
                ):
                    # A malformed or duplicate early route cannot authorize
                    # publication. If the fully validated intent is visual,
                    # run the private probe once now instead of trusting the
                    # mismatched prefix or issuing any action re-score.
                    if not visual_arithmetic_probe_task.done():
                        visual_arithmetic_probe_task.cancel()
                    await asyncio.gather(
                        visual_arithmetic_probe_task,
                        return_exceptions=True,
                    )
                    visual_arithmetic_probe_task = track_branch(
                        self._run_visual_arithmetic_probe(
                            turn,
                            current_audio_list,
                            prepared_current_images,
                            current_image_roles,
                            image_encoder_prefetch_task=(
                                visual_arithmetic_image_prefetch_task
                            ),
                        ),
                        name=(
                            f"session-visual-arithmetic-probe-fallback-"
                            f"{self.session_id}-{turn.turn_id}"
                        ),
                    )
                if (
                    not visual_gesture_answer
                    and visual_arithmetic_probe_task is not None
                ):
                    if not visual_arithmetic_probe_task.done():
                        visual_arithmetic_probe_task.cancel()
                    await asyncio.gather(
                        visual_arithmetic_probe_task,
                        return_exceptions=True,
                    )
                    emit_structured_log(
                        "reply",
                        "visual_arithmetic_probe_discarded",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        reason="intent_not_visual_answer",
                        after_commit_ms=self._after_commit_ms(turn),
                    )
                    visual_arithmetic_probe_task = None

            async def infer_performance_and_release() -> PerformanceDecision:
                nonlocal performance, expression, action, early_expression
                decision = await self._infer_turn_performance(
                    turn,
                    current_audio_list,
                    current_text=(
                        turn.intent.action_context(turn.text)
                        if turn.intent
                        else turn.text
                    ),
                    images=prepared_current_images,
                    image_roles=current_image_roles,
                )
                if turn.intent is not None:
                    decision = replace(decision, tts_instruction=turn.intent.tts_instruction())
                    if turn.turn_origin == TURN_ORIGIN_USER:
                        intent = turn.intent
                        scope = (
                            "both" if intent.face and intent.body_mode != "none"
                            else "expression_only" if intent.face
                            else "body_only" if intent.body_mode != "none"
                            else "none"
                        )
                        decision = replace(decision, request_scope=scope)
                performance = decision
                emit_structured_log(
                    "performance", "expression_decided",
                    session_id=self.session_id, turn_id=turn_id,
                    trace_id=turn.trace_id, request_scope=decision.request_scope,
                    after_commit_ms=self._after_commit_ms(turn),
                    has_expression=decision.expression is not None,
                )
                if (
                    "expression" in self.modalities
                    and (decision.request_scope == "none" or turn.turn_origin == TURN_ORIGIN_USER)
                    and decision.expression is not None
                    and not decision.expression_unsupported
                ):
                    independent = fuse_performance_decision(
                        action=None, action_error=None, performance=decision,
                        expression_enabled=True,
                        independent_channels=turn.turn_origin == TURN_ORIGIN_USER,
                    ).expression
                    if independent is not None:
                        expression = independent
                        await send_expression_ready(independent)
                if (
                    turn.tts_instruction_future is not None
                    and not turn.tts_instruction_future.done()
                ):
                    turn.tts_instruction_future.set_result(decision.tts_instruction)
                if (
                    "action" in self.modalities
                    and "expression" in self.modalities
                    and not self.direct_action_selection
                    and decision.request_scope == "expression_only"
                    and decision.expression is not None
                    and not decision.expression_unsupported
                ):
                    self._ensure_turn_processing(turn)
                    early_expression = True
                    fused = fuse_performance_decision(
                        action=None, action_error=None, performance=decision,
                        expression_enabled=True,
                    )
                    action, expression = fused.action, fused.expression
                    # Cancelling the owned scoring task propagates to the client's
                    # request-specific abort. Observe cleanup at the terminal barrier.
                    if action_task is not None and not action_task.done():
                        action_task.cancel()
                    emit_structured_log(
                        "action", "expression_only_body_bypassed",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        trace_id=turn.trace_id, reason="expression_only",
                        status="not_required",
                    )
                    await send_action_ready()
                    if provisional_state is not None:
                        await self._promote_provisional_reply(
                            turn, provisional_state, reason="expression_supported",
                            wait_for_tts=False,
                        )
                    emit_structured_log(
                        "performance", "expression_only_released",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        trace_id=turn.trace_id, after_commit_ms=self._after_commit_ms(turn),
                        body_scoring_cancel_requested=action_task is not None,
                    )
                return decision

            action_priority_enabled = bool(
                turn.turn_origin == TURN_ORIGIN_USER
                and not provided_reply
                and turn.intent is not None
                and "action" in self.modalities
                and not visual_gesture_answer
                and not body_action_not_requested
                and not (turn.intent.face and turn.intent.body_mode == "none")
            )
            if action_priority_enabled:
                start_action_scoring()
                window_started = time.perf_counter()
                emit_structured_log(
                    "performance", "action_priority_window_started",
                    session_id=self.session_id, turn_id=turn_id,
                    timeout_ms=1000,
                )
                reason = "turn_cancelled"
                try:
                    # wait(), unlike wait_for(), does not cancel the action on
                    # deadline. The existing owned-task barrier handles errors.
                    done, _ = await asyncio.wait({action_task}, timeout=1.0)
                    reason = "timeout"
                    if done:
                        reason = (
                            "cancelled" if action_task.cancelled()
                            else "failed" if action_task.exception() is not None
                            else "completed"
                        )
                    self._ensure_turn_processing(turn)
                finally:
                    action_priority_released.set()
                    emit_structured_log(
                        "performance", "action_priority_window_released",
                        session_id=self.session_id, turn_id=turn_id,
                        reason=reason,
                        elapsed_ms=(time.perf_counter() - window_started) * 1000,
                    )
            else:
                action_priority_released.set()
                emit_structured_log(
                    "performance", "action_priority_window_skipped",
                    session_id=self.session_id, turn_id=turn_id,
                    reason=(
                        "non_user_turn" if turn.turn_origin != TURN_ORIGIN_USER
                        else "provided_reply" if provided_reply
                        else "intent_unavailable" if turn.intent is None
                        else "action_output_disabled" if "action" not in self.modalities
                        else "visual_gesture_answer" if visual_gesture_answer
                        else "body_action_not_requested" if body_action_not_requested
                        else "expression_only"
                    ),
                )

            if self.route_action_parallel and "action" in self.modalities:
                # History routing and Category are the latency-critical
                # branches.  Give a previously-created route task one event
                # loop turn to submit, then start Category without waiting for
                # the route result.  Performance/TTS control is created only
                # afterwards so admission priority, rather than create_task
                # timing, governs contention.
                if reply_history_route_task is not None:
                    await asyncio.sleep(0)
                start_action_scoring()

            if "expression" in self.modalities or "audio" in self.modalities:
                if "audio" in self.modalities:
                    turn.tts_instruction_future = (
                        asyncio.get_running_loop().create_future()
                    )
                    if turn.intent is not None:
                        turn.tts_instruction_future.set_result(turn.intent.tts_instruction())
                        emit_structured_log(
                            "performance", "voice_plan_released",
                            session_id=self.session_id, turn_id=turn.turn_id,
                            trace_id=turn.trace_id, after_commit_ms=self._after_commit_ms(turn),
                            voice_tone=turn.intent.voice_tone, voice_pace=turn.intent.voice_pace,
                        )
                if (
                    visual_gesture_answer
                    and turn.intent is not None
                    and bool(turn.intent.face)
                ):
                    # Unified intent has already normalized an explicitly
                    # requested face. Resolve the small fixed expression set
                    # deterministically so the visual-arithmetic fast path does
                    # not add a performance-model request or delay its action.
                    performance = self._explicit_face_performance_decision(
                        turn.intent.face
                    )
                    performance = replace(
                        performance,
                        # The numeric answer already owns the body channel;
                        # the explicit face is an independent second channel,
                        # not an expression-only turn that may suppress it.
                        request_scope="both",
                        tts_instruction=turn.intent.tts_instruction(),
                    )
                    expression = performance.expression
                    emit_structured_log(
                        "performance",
                        "visual_answer_explicit_face_resolved",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        face_task=turn.intent.face,
                        expression_candidate_id=(
                            expression.get("candidate_id")
                            if expression is not None
                            else None
                        ),
                        expression_unsupported=(
                            performance.expression_unsupported
                        ),
                        model_request_added=False,
                        after_commit_ms=self._after_commit_ms(turn),
                    )
                    if (
                        "expression" in self.modalities
                        and expression is not None
                        and not performance.expression_unsupported
                    ):
                        await send_expression_ready(expression)
                elif (
                    not visual_gesture_answer
                    and ("expression" in self.modalities or turn.intent is None)
                ):
                    performance_task = track_branch(
                        infer_performance_and_release(),
                        name=f"session-performance-{self.session_id}-{turn.turn_id}",
                    )
                else:
                    performance = replace(self._default_performance_decision(),
                                          tts_instruction=turn.intent.tts_instruction())

            knowledge_eligible = bool(
                self.knowledge_binding is not None
                and self.knowledge_controller is not None
                and self.knowledge_binding.mode == "retrieval"
                and not provided_reply
                and turn.turn_origin == TURN_ORIGIN_USER
                and isinstance(turn.text, str)
                and turn.text.strip()
            )
            if (
                self.knowledge_binding is not None
                and self.knowledge_binding.mode == "provided_context"
                and self.provided_entity_context is not None
                and not provided_reply
                and turn.turn_origin == TURN_ORIGIN_USER
            ):
                turn.knowledge_context = self.provided_entity_context
            knowledge_speculative_enabled = bool(
                getattr(
                    getattr(self.knowledge_controller, "config", None),
                    "speculative_enabled",
                    False,
                )
            )
            if (
                knowledge_eligible
                and knowledge_speculative_enabled
            ):
                recent_user_turns, recent_assistant_turns = (
                    self._knowledge_recent_text_turns()
                )
                emit_structured_log(
                    "performance",
                    "knowledge_prepare_started",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                )
                knowledge_prepare_task = track_branch(
                    self.knowledge_controller.prepare_turn(
                        binding=self.knowledge_binding,
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        text=turn.text.strip(),
                        hints=turn.knowledge_entity_hints,
                        recent_user_turns=recent_user_turns,
                        recent_assistant_turns=recent_assistant_turns,
                    ),
                    name=f"session-knowledge-prepare-{self.session_id}-{turn.turn_id}",
                )

            if reply_history_route_task is not None:
                reply_history_route = await reply_history_route_task

            preserve_language_reply_on_unsupported_action = bool(
                fusion_reply
                and turn.turn_origin == TURN_ORIGIN_USER
                and reply_history_route is not None
                and reply_history_route.reply_mode == REPLY_MODE_LANGUAGE_REQUIRED
                and turn.intent is not None
                and (
                    turn.intent.speech_independent_of_body
                    or turn.intent.body_intent == "capability"
                )
                and not visual_gesture_answer
            )
            preserve_visual_answer_speech = bool(
                visual_gesture_answer
                and turn.intent is not None
                and turn.intent.has_visual_public_speech()
            )
            pure_action_reply = bool(
                visual_gesture_answer
                or (
                    fusion_reply
                    and reply_history_route is not None
                    and reply_history_route.reply_mode == REPLY_MODE_PURE_ACTION
                )
            )
            eligible_turn_candidates = self._filter_turn_action_candidates(
                turn, self.candidates
            )
            numeric_reply_route = route_numeric_reply_action(
                turn_origin=turn.turn_origin,
                reply_provided=provided_reply,
                speech_kind=(turn.intent.speech if turn.intent else None),
                body_mode=(turn.intent.body_mode if turn.intent else None),
                has_user_camera=(
                    IMAGE_ROLE_USER_CAMERA in current_image_roles
                ),
                has_text_output="text" in self.modalities,
                has_action_output="action" in self.modalities,
                candidates=eligible_turn_candidates,
                allow_empty_candidates=visual_gesture_answer,
            )
            if not visual_gesture_answer and (
                self.direct_action_selection
                or turn.turn_origin == TURN_ORIGIN_USER
            ):
                numeric_reply_route = replace(
                    numeric_reply_route, enabled=False,
                    reason="disabled_action_first_policy",
                )
            if pure_action_reply and numeric_reply_route.enabled and not visual_gesture_answer:
                numeric_reply_route = replace(
                    numeric_reply_route,
                    enabled=False,
                    reason="pure_action_reply",
                )
            emit_structured_log(
                "action",
                "numeric_reply_action_routed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                enabled=numeric_reply_route.enabled,
                reason=numeric_reply_route.reason,
                numeric_candidate_count=len(numeric_reply_route.candidates),
            )
            reply_route_decision_ready = True
            maybe_schedule_category_discard()
            if provisional_discard_task is not None:
                await provisional_discard_task
            if not visual_gesture_answer:
                start_action_scoring()

            if knowledge_prepare_task is not None and pure_action_reply:
                if not knowledge_prepare_task.done():
                    knowledge_prepare_task.cancel()
                await asyncio.gather(knowledge_prepare_task, return_exceptions=True)
                emit_structured_log(
                    "performance",
                    "knowledge_prepare_cancelled",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    reason="pure_action",
                )
            elif knowledge_prepare_task is not None:
                knowledge_task = track_branch(
                    self._consume_prepared_knowledge(
                        knowledge_prepare_task,
                        turn=turn,
                        route_completed_at=time.perf_counter(),
                    ),
                    name=f"session-knowledge-consume-{self.session_id}-{turn.turn_id}",
                )
            elif knowledge_eligible and not pure_action_reply:
                recent_user_turns, recent_assistant_turns = (
                    self._knowledge_recent_text_turns()
                )
                knowledge_task = track_branch(
                    self.knowledge_controller.resolve_turn(
                        binding=self.knowledge_binding,
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        text=turn.text.strip(),
                        hints=turn.knowledge_entity_hints,
                        recent_user_turns=recent_user_turns,
                        recent_assistant_turns=recent_assistant_turns,
                    ),
                    name=f"session-knowledge-{self.session_id}-{turn.turn_id}",
                )

            # Action scoring is already running, so the rare bounded memory
            # catch-up cannot postpone action admission. It only delays reply
            # construction when an R1 request depends on turns older than the
            # existing two-turn raw window.
            await self._wait_for_session_memory_catchup(
                turn, reply_history_route
            )
            if knowledge_task is not None:
                turn.knowledge_context = await knowledge_task
                if not knowledge_speculative_enabled:
                    self.knowledge_binding = replace(
                        self.knowledge_binding,
                        state_token=turn.knowledge_context.state_token,
                    )

                emit_structured_log(
                    "diagnostic",
                    "knowledge_turn_resolved",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    logical_request_id=turn.request_base,
                    decision=turn.knowledge_context.decision,
                    reason=turn.knowledge_context.reason,
                    result_id=turn.knowledge_context.result_id,
                    snapshot_id=turn.knowledge_context.snapshot_id,
                    capability_count=len(turn.knowledge_context.capabilities),
                    evidence_count=len(turn.knowledge_context.evidence),
                    degraded_code=turn.knowledge_context.degraded_code,
                    elapsed_ms=round(turn.knowledge_context.elapsed_ms, 3),
                )

            if (
                fusion_reply
                and provisional_state is not None
                and provisional_state.status != "discarded"
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
                elif pure_action_reply and not visual_gesture_answer:
                    if visual_copy_without_speech:
                        reply_task = track_branch(
                            self._run_empty_pure_action_reply(
                                turn,
                                provisional=provisional_state,
                            ),
                            name=(
                                f"session-provisional-empty-visual-copy-reply-"
                                f"{self.session_id}-{turn.turn_id}"
                            ),
                        )
                    else:
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
                    generated_reply_route = reply_history_route
                    if visual_gesture_answer and generated_reply_route is not None:
                        generated_reply_route = replace(
                            generated_reply_route,
                            reply_mode=REPLY_MODE_LANGUAGE_REQUIRED,
                        )
                    if (
                        visual_gesture_answer
                        and visual_arithmetic_probe_task is not None
                    ):
                        reply_task = track_branch(
                            self._adopt_visual_arithmetic_probe(
                                turn,
                                provisional_state,
                                visual_arithmetic_probe_task,
                            ),
                            name=(
                                f"session-provisional-visual-arithmetic-"
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
                                history_route=generated_reply_route,
                            ),
                            name=(
                                f"session-provisional-reply-"
                                f"{self.session_id}-{turn.turn_id}"
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

            # Independent language must not wait for body candidate scoring.
            # The route also prevents unsupported body results from discarding
            # this reply. TTS freezes its voice plan before the first append;
            # a valid shared intent does not wait for face scoring.
            if (
                preserve_language_reply_on_unsupported_action
                and provisional_state is not None
                and reply_task is not None
                and turn.intent is not None
                and turn.intent.body_mode == "none"
            ):
                await self._promote_provisional_reply(
                    turn, provisional_state,
                    reason="language_required", wait_for_tts=False,
                )

            if performance_task is not None:
                try:
                    performance = await performance_task
                except asyncio.CancelledError:
                    if (
                        turn.tts_instruction_future is not None
                        and not turn.tts_instruction_future.done()
                    ):
                        turn.tts_instruction_future.cancel()
                    raise
                except Exception as exc:
                    if early_expression or expression_ready_sent:
                        raise
                    performance = self._default_performance_decision()
                    emit_structured_log(
                        "error",
                        "turn_performance_control_failed",
                        level="error",
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        trace_id=turn.trace_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                if (
                    turn.tts_instruction_future is not None
                    and not turn.tts_instruction_future.done()
                ):
                    turn.tts_instruction_future.set_result(
                        performance.tts_instruction
                    )

            if "action" in self.modalities:
                if action_started is None:
                    action_started = time.perf_counter()
                try:
                    if not early_expression:
                        if body_action_not_requested:
                            action = {
                                "candidate_id": "body_not_requested",
                                "action_id": "no_action",
                                "execution_binding": {},
                                "execute": False,
                                "support_status": "not_required",
                                "fallback_applied": False,
                                "reason_code": "body_action_not_requested",
                            }
                            scores = []
                            action_context = {
                                "selection_stages": 0,
                                "selection_mode": "not_requested",
                                "generic_action_scoring_bypassed": True,
                            }
                            if turn.action_decision is not None:
                                action_context["action_decision"] = (
                                    decision_as_dict(turn.action_decision)
                                )
                                action_context[
                                    "action_decision_batch_mode"
                                ] = getattr(
                                    self,
                                    "action_decision_batch_mode",
                                    "off",
                                )
                            emit_structured_log(
                                "action",
                                "generic_action_scoring_bypassed",
                                session_id=self.session_id,
                                turn_id=turn.turn_id,
                                trace_id=turn.trace_id,
                                logical_request_id=turn.request_base,
                                reason="body_action_not_requested",
                            )
                        elif visual_gesture_answer and numeric_reply_route.enabled:
                            # VISUAL_ANSWER owns the body decision. Do not spend a full
                            # catalog PPL pass on an action that would be
                            # discarded once the structured visual answer is
                            # available.
                            action = {
                                "candidate_id": "visual_answer_pending",
                                "action_id": "no_action",
                                "execution_binding": {},
                                "execute": False,
                                "support_status": "unknown",
                                "fallback_applied": False,
                                "reason_code": "visual_answer_pending",
                            }
                            scores = []
                            action_context = {
                                "selection_stages": 0,
                                "selection_mode": "visual_gesture_answer",
                                "generic_action_scoring_bypassed": True,
                            }
                            emit_structured_log(
                                "action",
                                "generic_action_scoring_bypassed",
                                session_id=self.session_id,
                                turn_id=turn.turn_id,
                                trace_id=turn.trace_id,
                                logical_request_id=turn.request_base,
                                reason="visual_gesture_answer",
                            )
                        else:
                            assert action_task is not None
                            action, scores, action_timing, action_context = (
                                await action_task
                            )
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

                if (
                    not early_expression
                    and numeric_reply_route is not None
                    and numeric_reply_route.enabled
                    and provisional_state is not None
                    and reply_task is not None
                ):
                    numeric_resolution = (
                        await self._resolve_numeric_reply_action(
                            turn,
                            route=numeric_reply_route,
                            provisional_reply=provisional_state,
                            original_question=(
                                turn.intent.text
                                if turn.intent is not None
                                else turn.text or ""
                            ),
                            avatar_state=turn.avatar_state,
                        )
                    )
                    action_context["numeric_reply_action"] = (
                        numeric_resolution.route_context
                    )
                    if numeric_resolution.action is not None:
                        action = numeric_resolution.action
                        scores = numeric_resolution.scores or []
                        action_error = None
                        action_context.update(
                            numeric_resolution.selection_context
                        )
                        if numeric_resolution.timing_breakdown is not None:
                            action_context.setdefault(
                                "action_timing_breakdown", {}
                            )["numeric_reply_action"] = (
                                numeric_resolution.timing_breakdown
                            )
                    elif visual_gesture_answer:
                        selected_number = (
                            numeric_resolution.route_context.get(
                                "selected_number"
                            )
                        )
                        fallback_reason = (
                            numeric_resolution.route_context.get(
                                "fallback_reason"
                            )
                        )
                        if isinstance(selected_number, int):
                            action = {
                                "candidate_id": UNSUPPORTED_DECISION_ID,
                                "action_id": UNSUPPORTED_DECISION_ID,
                                "execution_binding": {},
                                "execute": False,
                                "support_status": "unsupported",
                                "fallback_applied": False,
                                "reason_code": "numeric_gesture_unavailable",
                            }
                        else:
                            action = {
                                "candidate_id": "visual_answer_unresolved",
                                "action_id": "no_action",
                                "execution_binding": {},
                                "execute": False,
                                "support_status": "unknown",
                                "fallback_applied": False,
                                "reason_code": "visual_answer_unresolved",
                            }
                        action_context.update(
                            {
                                "support_status": action["support_status"],
                                "fallback_applied": False,
                                "visual_answer_failure_reason": fallback_reason,
                            }
                        )
                    if (
                        visual_gesture_answer
                        and turn.intent is not None
                        and turn.intent.has_visual_public_speech()
                    ):
                        # The generated VISUAL_ARITHMETIC contract is private
                        # model evidence, not user-facing prose. Publish a
                        # deterministic answer from the same resolved number so
                        # speech and the selected gesture cannot disagree.
                        await reply_task
                        await self._suppress_provisional_reply_content(
                            turn,
                            provisional_state,
                            reason="visual_answer_internal_evidence",
                        )
                        await self._discard_provisional_reply(
                            turn,
                            provisional_state,
                            reason="visual_answer_internal_evidence",
                            abort_request=False,
                            wait_for_cleanup=True,
                        )
                        selected_number = numeric_resolution.route_context.get(
                            "selected_number"
                        )
                        public_parts: list[str] = []
                        if turn.intent.speaks_visual_answer():
                            public_parts.append(
                                self._prompt(
                                    zh=f"答案是数字{selected_number}。",
                                    en=f"The answer is {selected_number}.",
                                )
                                if isinstance(selected_number, int)
                                else self._prompt(
                                    zh="我没能识别出答案。",
                                    en="I couldn't determine the answer.",
                                )
                            )
                        additional_speech = (
                            turn.intent.visual_additional_speech()
                        )
                        if additional_speech:
                            public_parts.append(additional_speech)
                        public_answer = " ".join(public_parts)
                        visual_answer_public_reply_task = track_branch(
                            self._run_provided_reply(
                                turn,
                                public_answer,
                                source="generated",
                            ),
                            name=(
                                f"session-visual-answer-public-reply-"
                                f"{self.session_id}-{turn.turn_id}"
                            ),
                        )

                if performance is not None:
                    fused = fuse_performance_decision(
                        action=action,
                        action_error=action_error,
                        performance=performance,
                        expression_enabled="expression" in self.modalities,
                        independent_channels=turn.turn_origin == TURN_ORIGIN_USER,
                    )
                    action = independently_published_action or fused.action
                    action_error = fused.action_error
                    expression = fused.expression
                action_timing = round(
                    (time.perf_counter() - action_started) * 1000.0,
                    3,
                )
                action_context.setdefault(
                    "action_timing_breakdown", {}
                )["total_ms"] = action_timing
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
                    and not (
                        preserve_language_reply_on_unsupported_action
                        and turn.intent is not None
                        and turn.intent.body_intent == "capability"
                    )
                    and not preserve_visual_answer_speech
                )
                if provisional_state is not None:
                    if (
                        visual_gesture_answer
                        and reply_task is not None
                        and visual_answer_public_reply_task is None
                    ):
                        await reply_task
                        await self._suppress_provisional_reply_content(
                            turn,
                            provisional_state,
                            reason="visual_gesture_answer",
                        )
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
                if (
                    suppress_reply_for_unsupported_action
                    and turn.turn_origin == TURN_ORIGIN_USER
                    and "text" in self.modalities
                ):
                    rejection_task = track_branch(
                        self._run_action_rejection_reply(
                            turn, current_audio_list, prepared_current_images,
                            current_image_roles,
                        ),
                        name=f"session-action-rejection-{self.session_id}-{turn_id}",
                    )
                logger.info(
                    "[SESSION_ACTION_REALTIME] action completed session_id=%s "
                    "turn_id=%s elapsed_ms=%.3f top_action=%s",
                    self.session_id,
                    turn_id,
                    (time.perf_counter() - action_started) * 1000.0,
                    action.get("action_id") if action else None,
                )
                await send_action_ready()

            if expression is not None:
                await send_expression_ready(expression)

            action_unsupported = (
                action is not None and action.get("support_status") == "unsupported"
            )
            suppress_reply_for_unsupported_action = bool(
                action_unsupported
                and not (
                    preserve_language_reply_on_unsupported_action
                    and turn.intent is not None
                    and turn.intent.body_intent == "capability"
                )
                and not preserve_visual_answer_speech
            )
            if (
                visual_answer_public_reply_task is not None
                and not suppress_reply_for_unsupported_action
            ):
                reply_text, reply_timing = await visual_answer_public_reply_task
            elif reply_task is not None and not suppress_reply_for_unsupported_action:
                reply_text, reply_timing = await reply_task
                if visual_gesture_answer:
                    reply_text = ""
                if provisional_state is not None:
                    reply_timing = self._provisional_reply_timing(provisional_state)
            elif provisional_state is not None:
                reply_timing = self._provisional_reply_timing(provisional_state)

            # turn.action.ready is allowed to overtake TTS finalization or
            # cancellation, but turn.result remains the terminal barrier for
            # all reply/TTS resources belonging to the turn.
            if early_expression and action_task is not None:
                await asyncio.gather(action_task, return_exceptions=True)
                emit_structured_log(
                    "performance", "expression_only_body_cleanup_completed",
                    session_id=self.session_id, turn_id=turn.turn_id,
                    trace_id=turn.trace_id, after_commit_ms=self._after_commit_ms(turn),
                    cancelled=action_task.cancelled(),
                )
            await self._wait_for_provisional_background_tasks(provisional_state)
            if (
                provisional_state is not None
                and reply_timing is not None
                and visual_answer_public_reply_task is None
            ):
                reply_timing = self._provisional_reply_timing(provisional_state)

            if rejection_task is not None:
                reply_text, reply_timing = await rejection_task
                emit_structured_log(
                    "reply", "action_rejection_reply_completed",
                    session_id=self.session_id, turn_id=turn_id,
                    trace_id=turn.trace_id, request_id=f"{turn.request_base}-action-rejection",
                    output_text=reply_text, **reply_timing,
                )

            if knowledge_script_started and not suppress_reply_for_unsupported_action:
                assert self.knowledge_binding is not None
                assert self.knowledge_controller is not None
                assert turn.knowledge_script_id is not None
                self.knowledge_binding = await self._apply_knowledge_script_event(
                    turn_id=turn.turn_id,
                    script_id=turn.knowledge_script_id,
                    event="completed",
                    script_version=turn.knowledge_script_version,
                    checksum=turn.knowledge_script_checksum,
                )
                knowledge_script_finished = True
                emit_structured_log(
                    "diagnostic", "knowledge_script_completed",
                    session_id=self.session_id, turn_id=turn.turn_id,
                    trace_id=turn.trace_id, script_id=turn.knowledge_script_id,
                    snapshot_id=self.knowledge_binding.snapshot_id,
                )

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
                    expression=expression,
                    performance=performance,
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
            if (
                knowledge_script_started
                and not knowledge_script_finished
                and turn.knowledge_script_id is not None
                and self.knowledge_binding is not None
                and self.knowledge_controller is not None
            ):
                try:
                    self.knowledge_binding = await self._apply_knowledge_script_event(
                        turn_id=turn.turn_id,
                        script_id=turn.knowledge_script_id,
                        event="interrupted",
                        script_version=turn.knowledge_script_version,
                        checksum=turn.knowledge_script_checksum,
                    )
                    knowledge_script_finished = True
                    emit_structured_log(
                        "diagnostic", "knowledge_script_interrupted",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        trace_id=turn.trace_id, script_id=turn.knowledge_script_id,
                        snapshot_id=self.knowledge_binding.snapshot_id,
                    )
                except Exception as exc:
                    emit_structured_log(
                        "error", "knowledge_script_interrupt_failed", level="warning",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        trace_id=turn.trace_id, script_id=turn.knowledge_script_id,
                        error_type=type(exc).__name__, error_message=str(exc),
                    )
            self._request_turn_resource_sample(
                "turn_after_terminal",
                turn=turn,
                turn_outcome=turn_outcome,
                elapsed_after_commit_ms=round(
                    max(0.0, time.perf_counter() - commit_started) * 1000.0,
                    3,
                ),
            )
    def _knowledge_recent_text_turns(self) -> tuple[list[str], list[str]]:
        user_turns: list[str] = []
        assistant_turns: list[str] = []
        for history_turn in self.reply_history_turns[-4:]:
            for message in history_turn.messages:
                role = message.get("role")
                text = self._knowledge_message_text(message.get("content"))
                if not text:
                    continue
                if role == "user":
                    user_turns.append(text)
                elif role == "assistant":
                    assistant_turns.append(text)
        return user_turns[-8:], assistant_turns[-4:]

    @staticmethod
    def _knowledge_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        return "\n".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "input_text"}
            and str(item.get("text", "")).strip()
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
