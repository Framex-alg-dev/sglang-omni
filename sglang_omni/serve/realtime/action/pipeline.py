"""Action scoring entry point and mode dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from typing import Any, Callable, Literal

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
    child_unsupported_policy,
)
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import (
    _action_timing_breakdown,
    _text_audit_fields,
)
from sglang_omni.serve.realtime.protocol.models import (
    ProvisionalReplyState,
    SessionActionCandidate,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log
from sglang_omni.serve.realtime.components import compose_components
from sglang_omni.serve.realtime.action.cache_observation import child_cache_observer

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.action.category import ActionCategoryComponent
from sglang_omni.serve.realtime.action.candidate import ActionCandidateComponent


@compose_components(ActionCategoryComponent, ActionCandidateComponent)
class ActionScoringPipeline:
    async def _score_action_request(
        self,
        turn: TurnBuffer,
        request: ActionSuffixScoreRequest,
        *,
        child_category_ids: list[str] | None = None,
    ) -> Any:
        self._ensure_turn_processing(turn)
        self._register_turn_request(turn, request.request_id)
        started = time.perf_counter()
        observation = None
        cache_status, cache_stats = "failed", None
        cache_error_type = None
        if request.stage == "child":
            observation = child_cache_observer.submitted(
                request, child_category_ids or [], self.action_locale,
            )
            emit_structured_log(
                "action", "child_cache_request", session_id=self.session_id,
                turn_id=turn.turn_id, trace_id=turn.trace_id,
                request_id=request.request_id, **observation,
            )
        emit_structured_log(
            "action",
            "action_scoring_started",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=request.logical_request_id,
            request_id=request.request_id,
            stage=request.stage,
            admission_priority=request.admission_priority,
            locale=self.action_locale,
            language=request.language,
            candidate_count=len(request.candidates),
            prefix_cache_namespace=request.prefix_cache_namespace,
            **_text_audit_fields("system_prompt", request.system_prompt),
        )
        try:
            result = await self.client.score_action_suffixes(request)
            cache_status, cache_stats = "completed", result.stats
            emit_structured_log(
                "action",
                "action_scoring_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                admission_priority=request.admission_priority,
                locale=self.action_locale,
                language=request.language,
                candidate_count=len(request.candidates),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                prefix_cached=result.prefix_cached,
                stats=result.stats,
            )
            if not result.prefix_cached:
                emit_structured_log(
                    "action",
                    "action_prefix_cache_miss",
                    level="warning",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    trace_id=turn.trace_id,
                    request_id=request.request_id,
                    stage=request.stage,
                    locale=self.action_locale,
                    language=request.language,
                    prefix_cache_namespace=request.prefix_cache_namespace,
                    **_text_audit_fields("system_prompt", request.system_prompt),
                )
        except asyncio.CancelledError:
            cache_status = "cancelled"
            emit_structured_log(
                "action",
                "action_scoring_cancelled",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                admission_priority=request.admission_priority,
                locale=self.action_locale,
                language=request.language,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            raise
        except Exception as exc:
            cache_error_type = type(exc).__name__
            emit_structured_log(
                "error",
                "action_scoring_failed",
                level="error",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request.logical_request_id,
                request_id=request.request_id,
                stage=request.stage,
                admission_priority=request.admission_priority,
                locale=self.action_locale,
                language=request.language,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if observation is not None:
                emit_structured_log(
                    "action", "child_cache_result", session_id=self.session_id,
                    turn_id=turn.turn_id, trace_id=turn.trace_id,
                    request_id=request.request_id,
                    prefix_identity=observation["prefix_identity"],
                    scoring_identity=observation["scoring_identity"],
                    error_type=cache_error_type,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    **child_cache_observer.finished(
                        observation["prefix_identity"], request.request_id, cache_status, cache_stats,
                    ),
                )
            self._unregister_turn_request(turn, request.request_id)
        self._ensure_turn_processing(turn)
        return result


    async def _score_action(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        provisional_reply: ProvisionalReplyState | None = None,
        turn_id: str | None = None,
        on_category_selected: (
            Callable[[SessionActionCategory | None, str], None] | None
        ) = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        self._reconcile_client_executed_action(turn=turn)
        if (
            self.categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
        ):
            return await self._score_action_hierarchical(
                audios,
                images,
                image_roles,
                text,
                avatar_state,
                turn_origin=turn_origin,
                text_role=text_role,
                trigger=trigger,
                turn_id=turn_id,
                turn=turn,
                request_base=request_base,
                provisional_reply=provisional_reply,
                on_category_selected=on_category_selected,
            )
        return await self._score_action_flat(
            audios,
            images,
            image_roles,
            text,
            avatar_state,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            turn_id=turn_id,
            turn=turn,
            request_base=request_base,
        )


MultimodalActionScoringMixin = ActionScoringPipeline
