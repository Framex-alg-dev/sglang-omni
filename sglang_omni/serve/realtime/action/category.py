"""Hierarchical category recall and child scoring."""

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

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


class ActionCategoryComponent:
    async def _score_action_hierarchical(
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
        (
            action_history, action_history_audios, action_history_images,
            action_images, action_image_roles, action_context,
        ) = self._build_bounded_action_context(audios, images, image_roles)
        action_history = []
        action_history_audios = []
        action_history_images = []
        action_context.update(
            {
                "history_policy": "current_turn_only",
                "source_history_turn_count": len(self.history_turns),
                "history_turn_count": 0,
                "history_audio_count": 0,
                "history_image_count": 0,
                "cross_turn_history_omitted": bool(self.history_turns),
            }
        )
        effective_avatar_state = self._effective_avatar_state(
            avatar_state,
            turn_origin=turn_origin,
            has_avatar_image=IMAGE_ROLE_AVATAR_STATE in action_image_roles,
        )
        forced_category, forced_semantic_tag = self._forced_trigger_category(
            turn_origin=turn_origin,
            trigger=trigger,
        )
        excluded_category_ids = (
            self._state_description_excluded_category_ids(
                effective_avatar_state.get("state_description")
            )
            if self.global_action_catalog is not None
            and forced_category is None
            and turn_origin != TURN_ORIGIN_PROACTIVE
            else ()
        )
        excluded_category_id_set = set(excluded_category_ids)
        if excluded_category_ids:
            emit_structured_log(
                "action",
                "state_description_candidates_filtered",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                stage="category",
                excluded_category_ids=list(excluded_category_ids),
                **_text_audit_fields(
                    "state_description",
                    effective_avatar_state.get("state_description"),
                ),
            )
        base = self._build_turn_action_instruction(
            text,
            turn_origin=turn_origin,
            trigger=trigger,
            has_audio=bool(audios),
            image_roles=action_image_roles,
            has_state_description=("state_description" in effective_avatar_state),
            avatar_state_source=self._avatar_state_source(
                effective_avatar_state, action_image_roles
            ),
        )
        last_user_action_reference = (
            self._last_user_action_reference_instruction(
                turn_origin=turn_origin,
            )
        )
        proactive_repeat_instruction = self._proactive_action_repeat_instruction(
            turn_origin=turn_origin,
            client_last_action_id=turn.client_last_executed_action_id,
        )
        action_context.update(
            {
                "last_user_action_reference_injected": bool(
                    last_user_action_reference
                ),
                "last_user_action_reference_turn_id": (
                    self.last_user_executed_action.turn_id
                    if last_user_action_reference
                    and self.last_user_executed_action is not None
                    else None
                ),
                "last_user_action_reference_candidate_id": (
                    self.last_user_executed_action.candidate_id
                    if last_user_action_reference
                    and self.last_user_executed_action is not None
                    else None
                ),
            }
        )
        common = dict(
            model=self.model_name,
            language=self.language,
            audios=audios,
            images=action_images,
            sample_rate=16000,
            image_roles=action_image_roles,
            session_id=self.session_id,
            history=action_history,
            stage="category",
            logical_request_id=request_base,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            action_context_cache_key=request_base,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            cache_static_system_only=self.global_action_catalog is not None,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=effective_avatar_state,
            current_text=text or "",
        )
        started = time.perf_counter()
        eligible_categories = [
            category
            for category in self.categories
            if self._filter_turn_action_candidates(turn, list(category.children))
        ]
        if not eligible_categories:
            raise ValueError(
                "per-turn action candidate constraints leave no executable action"
            )
        category_by_id = {
            item.category_id: item for item in eligible_categories
        }
        category_result = None
        category_ranked: list[Any] = []
        category_ms = 0.0
        category_scoring_skipped = forced_category is not None
        category_scoring_skip_reason = (
            "trigger_policy" if forced_category is not None else None
        )
        if forced_category is not None:
            category_unsupported = False
            selected_category = forced_category
            selected_categories = [forced_category]
            category_scoring_candidate_id = forced_category.category_id
            emit_structured_log(
                "action",
                "action_category_route_forced",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                turn_origin=turn_origin,
                trigger=trigger,
                category_scoring_skipped=True,
                category_scoring_skip_reason=category_scoring_skip_reason,
                forced_semantic_tag=forced_semantic_tag,
                resolved_category_id=forced_category.category_id,
            )
        else:
            category_candidates = [
                ActionScoreCandidate(
                    candidate_id=item.category_id,
                    suffix=item.category_id,
                    action_id=item.category_id,
                )
                for item in eligible_categories
                if item.category_id not in excluded_category_id_set
            ]
            if self.global_action_catalog is not None:
                category_candidates.append(
                    ActionScoreCandidate(
                        candidate_id=UNSUPPORTED_CATEGORY_SCORE_ID,
                        suffix=UNSUPPORTED_CATEGORY_SCORE_ID,
                        action_id=UNSUPPORTED_DECISION_ID,
                    )
                )
            category_request = ActionSuffixScoreRequest(
                request_id=request_base + "-category",
                prefix=(
                    self._build_session_action_profile_instruction("category")
                    + last_user_action_reference
                    + proactive_repeat_instruction
                    + base
                    + self._category_whitelist_instruction()
                    + self._state_description_exclusion_instruction(
                        category_ids=excluded_category_ids
                    )
                    + self._state_description_priority_instruction(
                        "category",
                        enabled=("state_description" in effective_avatar_state),
                    )
                ),
                output_prompt=self._prompt(
                    zh="最合适的 category_id：",
                    en="Best matching category_id:",
                ),
                system_prompt=self._build_category_system_prompt(),
                candidates=category_candidates,
                suffix_tokenization_mode="short_id",
                micro_batch_size=self.action_micro_batch_size,
                **common,
            )
            category_started = time.perf_counter()
            category_result = await self._score_action_request(
                turn, category_request
            )
            category_ms = round(
                (time.perf_counter() - category_started) * 1000.0, 3
            )
            logger.info(
                "[SESSION_ACTION_REALTIME] action stage completed "
                "session_id=%s turn_id=%s stage=category candidates=%d "
                "elapsed_ms=%.3f prefix_cached=%s stats=%s",
                self.session_id,
                turn_id,
                len(category_request.candidates),
                category_ms,
                category_result.prefix_cached,
                json.dumps(
                    category_result.stats, ensure_ascii=False, default=str
                ),
            )
            category_ranked = sorted(
                category_result.scores,
                key=lambda item: item.mean_logprob,
                reverse=True,
            )
            if not category_ranked:
                raise ValueError(
                    "category action score did not return a decision"
                )
            # The category stage is a recall stage. B000 remains useful as a
            # diagnostic score, but it must not prevent the best real
            # categories from reaching child scoring. Only A000 at the child
            # stage is allowed to make the final unsupported decision.
            category_unsupported = False
            ranked_real_categories = [
                item
                for item in category_ranked
                if item.candidate_id in category_by_id
            ]
            selected_categories = [
                category_by_id[item.candidate_id]
                for item in ranked_real_categories[: self.action_category_top_k]
            ]
            if not selected_categories:
                raise ValueError(
                    "category action score did not select a valid category"
                )
            selected_category = selected_categories[0]
            category_scoring_candidate_id = category_ranked[0].candidate_id
        reply_prefix = ""
        reply_prefix_status: str | None = None
        reply_prefix_wait_ms = 0.0
        system_route_reconciled = False
        system_route_original_category_id: str | None = None
        category_callback_emitted = False
        if on_category_selected is not None and (
            forced_category is not None
            or category_unsupported
            or selected_category is None
            or not self._is_system_accompaniment_category(selected_category)
        ):
            on_category_selected(
                selected_category,
                "unsupported" if category_unsupported else "supported",
            )
            category_callback_emitted = True
        if (
            forced_category is None
            and not category_unsupported
            and selected_category is not None
            and self._is_system_accompaniment_category(selected_category)
        ):
            original_category_id = selected_category.category_id
            system_route_original_category_id = original_category_id
            if provisional_reply is None:
                reply_prefix_status = "empty_completed"
            else:
                (
                    reply_prefix,
                    reply_prefix_status,
                    reply_prefix_wait_ms,
                ) = await self._resolve_provisional_reply_prefix(
                    turn, provisional_reply
                )
            desired_semantic_tag = (
                CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
                if reply_prefix
                else CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
            )
            resolved_category = self._category_with_semantic_tag(
                desired_semantic_tag
            )
            if resolved_category is None:
                raise ValueError(
                    "session is missing the resolved system accompaniment category"
                )
            if self._filter_turn_action_candidates(
                turn, list(resolved_category.children)
            ):
                selected_category = resolved_category
                selected_categories = [resolved_category] + [
                    category
                    for category in selected_categories[1:]
                    if category.category_id != resolved_category.category_id
                ]
                system_route_reconciled = (
                    resolved_category.category_id != original_category_id
                )
        execution_category = selected_categories[0]
        selected_category_ids = [item.category_id for item in selected_categories]

        def compact_stage_score(score: Any) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id,
                "token_count": score.token_count,
                "mean_logprob": score.mean_logprob,
                "mean_nll": score.mean_nll,
                "ppl": score.ppl,
                "token_scores": [
                    {"token_id": item.token_id, "logprob": item.logprob}
                    for item in score.token_scores
                ],
            }

        category_timing = (
            _action_timing_breakdown(category_result.stats)
            if category_result is not None
            else {
                "skipped": True,
                "reason": category_scoring_skip_reason,
            }
        )

        child_candidates = self._child_candidates_for_categories(selected_categories)
        child_candidates = self._filter_turn_action_candidates(
            turn, child_candidates
        )
        excluded_candidate_ids = (
            self._state_description_excluded_candidate_ids(
                effective_avatar_state.get("state_description"), child_candidates
            )
            if self.global_action_catalog is not None
            and turn_origin != TURN_ORIGIN_PROACTIVE
            else ()
        )
        excluded_candidate_id_set = set(excluded_candidate_ids)
        if excluded_candidate_ids:
            emit_structured_log(
                "action",
                "state_description_candidates_filtered",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                stage="child",
                selected_category_ids=selected_category_ids,
                excluded_candidate_ids=list(excluded_candidate_ids),
                **_text_audit_fields(
                    "state_description",
                    effective_avatar_state.get("state_description"),
                ),
            )
        child_candidates = [
            candidate
            for candidate in child_candidates
            if candidate.candidate_id not in excluded_candidate_id_set
        ]
        system_candidates_exhausted = False
        system_route_degradation_reason: str | None = None
        if (
            not child_candidates
            and self._category_has_semantic_tag(
                execution_category,
                CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
            )
        ):
            silent_category = self._category_with_semantic_tag(
                CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
            )
            if silent_category is None:
                raise ValueError(
                    "session is missing the silent accompaniment category"
                )
            execution_category = silent_category
            selected_category = silent_category
            selected_categories = [silent_category]
            selected_category_ids = [silent_category.category_id]
            system_route_reconciled = True
            system_candidates_exhausted = True
            system_route_degradation_reason = (
                "reply_candidates_exhausted_to_silent"
            )
            child_candidates = list(silent_category.children)
            child_candidates = self._filter_turn_action_candidates(
                turn, child_candidates
            )
            excluded_candidate_ids = (
                self._state_description_excluded_candidate_ids(
                    effective_avatar_state.get("state_description"),
                    child_candidates,
                )
                if self.global_action_catalog is not None
                and turn_origin != TURN_ORIGIN_PROACTIVE
                else ()
            )
            excluded_candidate_id_set = set(excluded_candidate_ids)
            child_candidates = [
                candidate
                for candidate in child_candidates
                if candidate.candidate_id not in excluded_candidate_id_set
            ]
        if (
            not child_candidates
            and self._category_has_semantic_tag(
                execution_category,
                CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
            )
        ):
            system_candidates_exhausted = True
            system_route_degradation_reason = (
                "silent_candidates_exhausted_first_real"
            )
            constrained_silent_candidates = self._filter_turn_action_candidates(
                turn, list(execution_category.children)
            )
            fallback_candidate = (
                constrained_silent_candidates[0]
                if constrained_silent_candidates
                else self._default_fallback_candidate_for_turn(turn)
            )
            child_candidates = [fallback_candidate]
            fallback_category = next(
                (
                    category
                    for category in self.categories
                    if category.category_id == fallback_candidate.category_id
                ),
                None,
            )
            if fallback_category is not None:
                execution_category = fallback_category
                selected_category = fallback_category
                selected_categories = [fallback_category]
                selected_category_ids = [fallback_category.category_id]
            emit_structured_log(
                "action",
                "system_action_candidates_exhausted",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                category_id=execution_category.category_id,
                fallback_candidate_id=child_candidates[0].candidate_id,
                excluded_candidate_ids=list(excluded_candidate_ids),
            )
        if system_route_original_category_id is not None:
            emit_structured_log(
                "action",
                "system_action_route_resolved",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                category_scoring_candidate_id=category_scoring_candidate_id,
                resolved_category_id=execution_category.category_id,
                system_route_reconciled=(
                    execution_category.category_id
                    != system_route_original_category_id
                ),
                reply_source=(
                    provisional_reply.source
                    if provisional_reply is not None
                    else None
                ),
                reply_prefix_status=reply_prefix_status,
                reply_prefix_wait_ms=reply_prefix_wait_ms,
                reply_prefix=(reply_prefix if self.log_full_instructions else None),
                system_candidates_exhausted=system_candidates_exhausted,
                **_text_audit_fields("reply_prefix", reply_prefix),
            )
        if on_category_selected is not None and not category_callback_emitted:
            on_category_selected(
                selected_category,
                "unsupported" if category_unsupported else "supported",
            )
        action_finished_random_selection = (
            turn_origin == TURN_ORIGIN_PROACTIVE
            and trigger == ACTION_FINISHED_TRIGGER
            and forced_semantic_tag
            == CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        )
        direct_child: SessionActionCandidate | None = None
        child_scoring_skip_reason: str | None = None
        previous_action_finished_candidate_id: str | None = None
        previous_action_finished_action_id: str | None = None
        action_finished_repeat_excluded = False
        action_finished_repeat_unavoidable = False
        action_finished_random_pool_count = 0
        if action_finished_random_selection:
            previous_action_finished_candidate_id = (
                self.last_action_finished_candidate_id
            )
            previous_action_finished_action_id = (
                self.last_action_finished_action_id
            )
            random_pool = list(child_candidates)
            if previous_action_finished_action_id is not None:
                non_repeating_pool = [
                    candidate
                    for candidate in random_pool
                    if candidate.action_id != previous_action_finished_action_id
                ]
                if non_repeating_pool:
                    action_finished_repeat_excluded = (
                        len(non_repeating_pool) != len(random_pool)
                    )
                    random_pool = non_repeating_pool
                elif any(
                    candidate.action_id == previous_action_finished_action_id
                    for candidate in random_pool
                ):
                    action_finished_repeat_unavoidable = True
            action_finished_random_pool_count = len(random_pool)
            direct_child = random.choice(random_pool)
            child_scoring_skip_reason = "action_finished_random"
            self.last_action_finished_candidate_id = direct_child.candidate_id
            self.last_action_finished_action_id = direct_child.action_id
            emit_structured_log(
                "action",
                "action_finished_random_selected",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                category_id=execution_category.category_id,
                selected_candidate_id=direct_child.candidate_id,
                previous_action_finished_candidate_id=(
                    previous_action_finished_candidate_id
                ),
                previous_action_finished_action_id=(
                    previous_action_finished_action_id
                ),
                eligible_candidate_count=len(child_candidates),
                random_pool_count=action_finished_random_pool_count,
                repeat_excluded=action_finished_repeat_excluded,
                repeat_unavoidable=action_finished_repeat_unavoidable,
            )
        elif (
            len(selected_categories) == 1
            and len(child_candidates) == 1
            and (
                self._is_system_accompaniment_category(execution_category)
                or (
                    self.global_action_catalog is None
                    and not self.include_scores
                )
            )
        ):
            direct_child = child_candidates[0]
            child_scoring_skip_reason = "single_child"

        if direct_child is not None:
            total_ms = round((time.perf_counter() - started) * 1000.0, 3)
            action = {
                "candidate_id": direct_child.candidate_id,
                "action_id": direct_child.action_id,
                "category_id": direct_child.category_id,
                "execution_binding": dict(direct_child.execution_binding),
                "execute": direct_child.action_id != "no_action",
            }
            if self.global_action_catalog is not None:
                action.update(
                    {
                        "support_status": "supported",
                        "fallback_applied": False,
                    }
                )
            action_context.update(
                {
                    "selection_stages": 1,
                    "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                    "logical_request_id": request_base,
                    "selected_category_id": execution_category.category_id,
                    "category_decision_id": (
                        UNSUPPORTED_DECISION_ID
                        if category_unsupported
                        else execution_category.category_id
                    ),
                    "category_scoring_candidate_id": category_ranked[
                        0
                    ].candidate_id
                    if category_ranked
                    else category_scoring_candidate_id,
                    "category_scoring_skipped": category_scoring_skipped,
                    "category_scoring_skip_reason": (
                        category_scoring_skip_reason
                    ),
                    "forced_semantic_tag": forced_semantic_tag,
                    "system_route_reconciled": system_route_reconciled,
                    "reply_prefix_wait_ms": reply_prefix_wait_ms,
                    "reply_prefix_status": reply_prefix_status,
                    "reply_prefix_chars": len(reply_prefix),
                    "system_candidates_exhausted": system_candidates_exhausted,
                    "system_route_degradation_reason": (
                        system_route_degradation_reason
                    ),
                    "child_candidate_count": len(child_candidates),
                    "support_status": (
                        "unsupported" if category_unsupported else "supported"
                    ),
                    "fallback_applied": category_unsupported,
                    "selected_category_ids": selected_category_ids,
                    "category_top_k": self.action_category_top_k,
                    "state_description_excluded_category_ids": list(
                        excluded_category_ids
                    ),
                    "state_description_excluded_candidate_ids": list(
                        excluded_candidate_ids
                    ),
                    "turn_action_allowed_candidate_ids": list(
                        turn.action_allowed_candidate_ids
                    ),
                    "turn_action_excluded_candidate_ids": list(
                        turn.action_excluded_candidate_ids
                    ),
                    "category_compute_ms": category_ms,
                    "child_compute_ms": 0.0,
                    "child_prefix_prefilled": (
                        execution_category.category_id
                        in self.global_action_prewarm.for_locale(
                            self.locale
                        ).ready_child_category_ids
                        if self.global_action_catalog is not None
                        else False
                    ),
                    "child_scoring_skipped": True,
                    "child_scoring_skip_reason": child_scoring_skip_reason,
                    "previous_action_finished_candidate_id": (
                        previous_action_finished_candidate_id
                    ),
                    "previous_action_finished_action_id": (
                        previous_action_finished_action_id
                    ),
                    "action_finished_repeat_excluded": (
                        action_finished_repeat_excluded
                    ),
                    "action_finished_repeat_unavoidable": (
                        action_finished_repeat_unavoidable
                    ),
                    "action_finished_random_pool_count": (
                        action_finished_random_pool_count
                    ),
                    "action_timing_breakdown": {
                        "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                        "category": category_timing,
                        "child": {
                            "skipped": True,
                            "reason": child_scoring_skip_reason,
                        },
                        "child_catalog_prefill_ms": 0.0,
                        "total_ms": total_ms,
                    },
                }
            )
            return action, [], total_ms, action_context

        child_namespace = ",".join(selected_category_ids)
        if len(selected_categories) == 1:
            child_system_prompt = self._build_child_system_prompt(
                execution_category, child_candidates
            )
        else:
            child_system_prompt = self._build_child_system_prompt(
                selected_categories, child_candidates
            )
        if self.global_action_catalog is not None and len(selected_categories) == 1:
            child_namespace = self.global_action_catalog.child_cache_namespace(
                execution_category.category_id, self.locale
            )
        elif self.global_action_catalog is not None:
            combined_prompt_hash = hashlib.sha256(
                child_system_prompt.encode("utf-8")
            ).hexdigest()
            child_namespace = (
                f"hierarchical:{self.locale}:child:{child_namespace}:"
                f"sha256:{combined_prompt_hash}"
            )
        else:
            child_namespace = (
                f"{self.action_prefix_cache_namespace}:child:{child_namespace}"
            )

        # Global Child catalogs are prewarmed before the server starts. The
        # legacy/session-local path retains its lazy first-use prefill.
        child_prefix_prefilled = (
            len(selected_categories) == 1
            and self.global_action_catalog is not None
            and execution_category.category_id
            in self.global_action_prewarm.for_locale(
                self.locale
            ).ready_child_category_ids
        )
        child_catalog_prefill_ms = 0.0
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if (
            callable(prefill)
            and self.global_action_catalog is None
            and child_namespace not in self._prefilled_action_prefix_namespaces
        ):
            child_prefill_started = time.perf_counter()
            prefill_request_id = request_base + "-child-prefill"
            self._ensure_turn_processing(turn)
            self._register_turn_request(turn, prefill_request_id)
            try:
                child_prefix_prefilled = await prefill(
                    request_id=prefill_request_id,
                    model=self.model_name,
                    system_prompt=child_system_prompt,
                    candidates=[
                        ActionScoreCandidate(
                            candidate_id=item.candidate_id,
                            suffix=item.candidate_id,
                            action_id=item.action_id,
                            execution_binding=dict(item.execution_binding),
                        )
                        for item in child_candidates
                    ],
                    prefix_cache_namespace=child_namespace,
                    stage="child",
                    language=self.language,
                )
            finally:
                self._unregister_turn_request(turn, prefill_request_id)
            self._ensure_turn_processing(turn)
            if child_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(child_namespace)
            child_catalog_prefill_ms = round(
                (time.perf_counter() - child_prefill_started) * 1000.0, 3
            )

        action_common = {
            **common,
            "stage": "child",
            "micro_batch_size": self.action_micro_batch_size,
            "prefix_cache_namespace": child_namespace,
        }
        action_request = ActionSuffixScoreRequest(
            request_id=request_base + "-child",
            prefix=(
                self._build_session_action_profile_instruction("child")
                + last_user_action_reference
                + proactive_repeat_instruction
                + base
                + self._system_accompaniment_child_instruction(
                    execution_category,
                    reply_prefix=reply_prefix,
                )
                + self._child_whitelist_instruction(
                    selected_categories, child_candidates
                )
                + self._state_description_exclusion_instruction(
                    candidate_ids=excluded_candidate_ids
                )
                + self._state_description_priority_instruction(
                    "child",
                    enabled=("state_description" in effective_avatar_state),
                )
            ),
            output_prompt=self._prompt(
                zh="最合适的 candidate_id：",
                en="Best matching candidate_id:",
            ),
            system_prompt=child_system_prompt,
            candidates=[
                ActionScoreCandidate(
                    candidate_id=item.candidate_id,
                    suffix=item.candidate_id,
                    action_id=item.action_id,
                    execution_binding=dict(item.execution_binding),
                )
                for item in child_candidates
            ]
            + (
                []
                if self.global_action_catalog is None
                or all(
                    self._is_system_accompaniment_category(category)
                    for category in selected_categories
                )
                else [
                    ActionScoreCandidate(
                        candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                        suffix=UNSUPPORTED_CHILD_SCORE_ID,
                        action_id=UNSUPPORTED_DECISION_ID,
                    )
                ]
            ),
            suffix_tokenization_mode="short_id",
            **action_common,
        )
        child_started = time.perf_counter()
        child_result = await self._score_action_request(turn, action_request)
        child_ms = round((time.perf_counter() - child_started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=child candidates=%d "
            "selected_category_ids=%s elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(action_request.candidates),
            ",".join(selected_category_ids),
            child_ms,
            child_result.prefix_cached,
            json.dumps(child_result.stats, ensure_ascii=False, default=str),
        )
        child_by_id = {item.candidate_id: item for item in child_candidates}
        ranked = sorted(
            child_result.scores, key=lambda item: item.mean_logprob, reverse=True
        )
        if not ranked:
            raise ValueError("child action score did not return a decision")
        child_unsupported = ranked[0].candidate_id == UNSUPPORTED_CHILD_SCORE_ID
        if not child_unsupported and ranked[0].candidate_id not in child_by_id:
            raise ValueError("child action score did not return a valid candidate")

        def score_dict(score: Any, candidate: SessionActionCandidate) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id,
                "action_id": candidate.action_id,
                "category_id": candidate.category_id,
                "source_label": candidate.source_label,
                "short_definition": candidate.short_definition,
                "execution_binding": dict(candidate.execution_binding),
                "token_count": score.token_count,
                "mean_logprob": score.mean_logprob,
                "mean_nll": score.mean_nll,
                "ppl": score.ppl,
                "token_scores": [
                    {"token_id": item.token_id, "logprob": item.logprob}
                    for item in score.token_scores
                ],
            }

        scores = [
            (
                compact_stage_score(score) | {"decision": "unsupported"}
                if score.candidate_id == UNSUPPORTED_CHILD_SCORE_ID
                else score_dict(score, child_by_id[score.candidate_id])
            )
            for score in ranked
        ]
        if child_unsupported:
            fallback = self._default_fallback_candidate_for_turn(turn)
            action = {
                "candidate_id": fallback.candidate_id,
                "action_id": fallback.action_id,
                "category_id": fallback.category_id,
                "execution_binding": dict(fallback.execution_binding),
                "execute": True,
                "support_status": "unsupported",
                "fallback_applied": True,
            }
        else:
            top = scores[0]
            action = {
                "candidate_id": top["candidate_id"],
                "action_id": top["action_id"],
                "category_id": top["category_id"],
                "execution_binding": dict(top.get("execution_binding") or {}),
                "execute": top["action_id"] != "no_action",
                "mean_logprob": top["mean_logprob"],
                "ppl": top["ppl"],
                "token_count": top["token_count"],
            }
            if self.global_action_catalog is not None:
                action.update(
                    {
                        "support_status": (
                            "unsupported" if category_unsupported else "supported"
                        ),
                        "fallback_applied": category_unsupported,
                    }
                )
        action_context.update({
            "selection_stages": 2,
            "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
            "logical_request_id": request_base,
            "selected_category_id": execution_category.category_id,
            "category_decision_id": (
                UNSUPPORTED_DECISION_ID
                if category_unsupported
                else execution_category.category_id
            ),
            "category_scoring_candidate_id": category_scoring_candidate_id,
            "category_scoring_skipped": category_scoring_skipped,
            "category_scoring_skip_reason": category_scoring_skip_reason,
            "forced_semantic_tag": forced_semantic_tag,
            "system_route_reconciled": system_route_reconciled,
            "reply_prefix_wait_ms": reply_prefix_wait_ms,
            "reply_prefix_status": reply_prefix_status,
            "reply_prefix_chars": len(reply_prefix),
            "system_candidates_exhausted": system_candidates_exhausted,
            "system_route_degradation_reason": (
                system_route_degradation_reason
            ),
            "child_candidate_count": len(child_candidates),
            "child_decision_id": (
                UNSUPPORTED_DECISION_ID
                if child_unsupported
                else ranked[0].candidate_id
            ),
            "child_scoring_candidate_id": ranked[0].candidate_id,
            "support_status": action.get("support_status"),
            "fallback_applied": action.get("fallback_applied"),
            "selected_category_ids": selected_category_ids,
            "category_top_k": self.action_category_top_k,
            "state_description_excluded_category_ids": list(
                excluded_category_ids
            ),
            "state_description_excluded_candidate_ids": list(
                excluded_candidate_ids
            ),
            "turn_action_allowed_candidate_ids": list(
                turn.action_allowed_candidate_ids
            ),
            "turn_action_excluded_candidate_ids": list(
                turn.action_excluded_candidate_ids
            ),
            "category_scores": [
                compact_stage_score(score) for score in category_ranked
            ],
            "category_compute_ms": category_ms,
            "child_compute_ms": child_ms,
            "child_prefix_prefilled": child_prefix_prefilled,
            "child_prefix_cache_namespace": child_namespace,
            "child_catalog_prefill_ms": child_catalog_prefill_ms,
            "action_timing_breakdown": {
                "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
                "category": category_timing,
                "child": _action_timing_breakdown(child_result.stats),
                "child_catalog_prefill_ms": child_catalog_prefill_ms,
                "total_ms": round(
                    (time.perf_counter() - started) * 1000.0, 3
                ),
            },
        })
        return action, scores, round((time.perf_counter() - started) * 1000.0, 3), action_context


MultimodalActionCategoryMixin = ActionCategoryComponent
