"""Hierarchical category recall and child scoring."""

from __future__ import annotations

from sglang_omni.serve.realtime.proactive.action_policy import (
    SCENE_FALLBACK_IDS, persona_first_scene, proactive_selection_instruction,
)

import asyncio
import hashlib
from sglang_omni.serve.realtime.action.cache_observation import prompt_sha256
import json
import logging
import random
import time
from typing import Any, Callable, Literal
from sglang_omni.serve.realtime.action.routing import IntentShortcutBackoff

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CANDIDATE_REACTION_SOURCE_LANGUAGE,
    CANDIDATE_REACTION_SOURCE_USER_CAMERA,
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
    child_unsupported_policy,
)
from sglang_omni.serve.realtime.action.routing import (
    choose_category_width,
    resolve_unique_explicit_action,
    scope_visual_deictic_categories,
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


DIRECT_GREETING_REACTION = "回应用户问候"
DIRECT_GREETING_CANDIDATE_ID = "288"
DIRECT_GREETING_ROUTE = "direct_greeting_reaction"


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
        allow_intent_shortcut: bool = True,
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
        persona_first = persona_first_scene(turn_origin, trigger)
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
            and (turn_origin != TURN_ORIGIN_PROACTIVE or persona_first)
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
        if persona_first:
            base += proactive_selection_instruction(self.action_language)
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
        category_session_instruction = (
            self._build_session_action_profile_instruction(
                "category",
                turn_origin=turn_origin,
                has_user_camera=IMAGE_ROLE_USER_CAMERA in action_image_roles,
            )
        )
        category_prefix_namespace = self._session_action_prefix_namespace(
            base_namespace=self.action_prefix_cache_namespace,
            stage="category",
            turn_origin=turn_origin,
            session_instruction=category_session_instruction,
        )
        common = dict(
            model=self.model_name,
            language=self.action_language,
            audios=audios,
            images=action_images,
            sample_rate=16000,
            image_roles=action_image_roles,
            session_id=self.session_id,
            history=action_history,
            session_instance_id=self.session_instance_id,
            stage="category",
            admission_priority=0,
            logical_request_id=request_base,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            action_context_cache_key=request_base,
            prefix_cache_namespace=category_prefix_namespace,
            cache_static_system_only=not bool(category_session_instruction),
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=effective_avatar_state,
            current_text=text or "",
        )
        started = time.perf_counter()
        expression_ids = self._facial_expression_candidate_ids()
        eligible_categories = [
            category
            for category in self.categories
            if category.category_id != FACIAL_EXPRESSION_CATEGORY_ID
            if self._filter_turn_action_candidates(
                turn,
                [
                    child
                    for child in category.children
                    if child.candidate_id not in expression_ids
                ],
            )
        ]
        requested_reaction_sources: set[str] = set()
        if self._body_accompaniment_only(turn):
            if turn.intent.reaction_mode == "respond":
                requested_reaction_sources.add(
                    CANDIDATE_REACTION_SOURCE_LANGUAGE
                )
            if IMAGE_ROLE_USER_CAMERA in action_image_roles:
                requested_reaction_sources.add(
                    CANDIDATE_REACTION_SOURCE_USER_CAMERA
                )
        session_candidate_ids = {
            child.candidate_id
            for category in eligible_categories
            for child in self._filter_turn_action_candidates(
                turn, list(category.children)
            )
        }
        implicit_reaction_candidate_ids = {
            candidate_id
            for candidate_id, candidate in (
                self.global_action_catalog.candidate_by_id.items()
                if self.global_action_catalog is not None
                else ()
            )
            if candidate_id in session_candidate_ids
            and candidate.reaction_sources.intersection(
                requested_reaction_sources
            )
        }
        implicit_reaction_sources = {
            source
            for candidate_id in implicit_reaction_candidate_ids
            for source in self.global_action_catalog.candidate_by_id[
                candidate_id
            ].reaction_sources
            if source in requested_reaction_sources
        }
        forced_reaction_candidate_id: str | None = None
        implicit_reaction_instruction = ""
        if implicit_reaction_sources:
            eligible_categories = [
                category
                for category in eligible_categories
                if self._is_system_accompaniment_category(category)
                or any(
                    child.candidate_id in implicit_reaction_candidate_ids
                    for child in category.children
                )
            ]
            implicit_reaction_instruction = self._action_prompt(
                zh=(
                    "\n[本轮自然动作回应边界]\n"
                    "仅可在本轮已标记的自然反应候选与系统伴随动作之间选择；"
                    f"反应来源={','.join(sorted(implicit_reaction_sources))}；"
                    "图片或语言证据不足时选择系统伴随动作，不得扩展到其他候选。"
                ),
                en=(
                    "\n[Current-turn natural action-response boundary]\n"
                    "Choose only between the configured natural-reaction candidates "
                    "and system accompaniment actions; reaction_sources="
                    f"{','.join(sorted(implicit_reaction_sources))}. Use system "
                    "accompaniment when language or image evidence is insufficient."
                ),
            )
        if (
            self._body_accompaniment_only(turn)
            and turn.intent.speech == "generated"
            and turn.intent.reaction_mode == "respond"
            and turn.intent.reaction.strip() == DIRECT_GREETING_REACTION
        ):
            greeting_route = next(
                (
                    (category, child)
                    for category in eligible_categories
                    if category.category_id not in excluded_category_id_set
                    for child in self._filter_turn_action_candidates(
                        turn, list(category.children)
                    )
                    if child.candidate_id == DIRECT_GREETING_CANDIDATE_ID
                    and child.candidate_id in implicit_reaction_candidate_ids
                ),
                None,
            )
            if greeting_route is not None:
                forced_category, greeting_candidate = greeting_route
                forced_semantic_tag = DIRECT_GREETING_ROUTE
                forced_reaction_candidate_id = greeting_candidate.candidate_id
        if not eligible_categories:
            raise ValueError(
                "per-turn action candidate constraints leave no executable action"
            )
        visual_deictic_scope = scope_visual_deictic_categories(
            eligible_categories,
            body_task=(turn.intent.body if turn.intent is not None else ""),
            body_mode=(turn.intent.body_mode if turn.intent is not None else "none"),
            has_user_camera=IMAGE_ROLE_USER_CAMERA in action_image_roles,
        )
        visual_deictic_instruction = ""
        visual_deictic_child_instruction = ""
        if visual_deictic_scope is not None:
            eligible_categories = list(visual_deictic_scope.categories)
            visual_deictic_category_ids = [
                category.category_id for category in eligible_categories
            ]
            action_context.update(
                {
                    "category_scope": (
                        "visual_deictic:" + visual_deictic_scope.name
                    ),
                    "category_scope_ids": visual_deictic_category_ids,
                }
            )
            visual_deictic_instruction = self._action_prompt(
                zh=(
                    "\n本轮用户通过范围词明确要求模仿 user_camera 中展示的动作；"
                    f"范围={visual_deictic_scope.name}。只能在该范围对应的动作目录中，"
                    "依据图片中直接可见的完整行为选择具体匹配项。"
                ),
                en=(
                    "\nThe user explicitly names the range of the action to imitate from "
                    "the user_camera image; "
                    f"range={visual_deictic_scope.name}. Match the complete directly visible "
                    "behavior only against the catalog family for that range."
                ),
            )
            visual_deictic_child_instruction = self._action_prompt(
                zh=(
                    "\n具体动作选择必须逐项比较范围部位与候选视觉定义，包括参与"
                    "部位数量、伸展或弯曲状态、相对位置、朝向以及物体关系；若关键"
                    "部位被遮挡、证据不足或没有足够匹配的候选，必须选择 000，不得"
                    "按候选常见程度猜测。"
                ),
                en=(
                    "\nCompare the in-scope body parts against each candidate's visual "
                    "definition, including the number of participating parts, extension "
                    "or flexion, relative positions, orientation, and object relations. "
                    "Select 000 when key parts are occluded, evidence is insufficient, "
                    "or no candidate matches closely enough; never guess from candidate "
                    "frequency."
                ),
            )
            emit_structured_log(
                "action",
                "visual_deictic_range_scope_applied",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                scope=visual_deictic_scope.name,
                category_ids=visual_deictic_category_ids,
                child_definition_mode="visual",
                selection_definition_source="short_definition_visual",
            )
        child_definition_mode: Literal["contextual", "visual"] = (
            "visual"
            if visual_deictic_scope is not None
            or CANDIDATE_REACTION_SOURCE_USER_CAMERA
            in implicit_reaction_sources
            else "contextual"
        )
        filtered_user_camera_image_count = 0
        if (
            visual_deictic_scope is None
            and CANDIDATE_REACTION_SOURCE_USER_CAMERA
            not in implicit_reaction_sources
            and turn_origin == TURN_ORIGIN_USER
            and self.global_action_catalog is not None
            and IMAGE_ROLE_USER_CAMERA in action_image_roles
        ):
            retained_media = [
                (image, role)
                for image, role in zip(
                    action_images,
                    action_image_roles,
                    strict=True,
                )
                if role != IMAGE_ROLE_USER_CAMERA
            ]
            filtered_user_camera_image_count = (
                len(action_image_roles) - len(retained_media)
            )
            action_images = [image for image, _ in retained_media]
            action_image_roles = [role for _, role in retained_media]
            base = self._build_turn_action_instruction(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                has_audio=bool(audios),
                image_roles=action_image_roles,
                has_state_description=(
                    "state_description" in effective_avatar_state
                ),
                avatar_state_source=self._avatar_state_source(
                    effective_avatar_state,
                    action_image_roles,
                ),
            )
            category_session_instruction = (
                self._build_session_action_profile_instruction(
                    "category",
                    turn_origin=turn_origin,
                    has_user_camera=False,
                )
            )
            category_prefix_namespace = self._session_action_prefix_namespace(
                base_namespace=self.action_prefix_cache_namespace,
                stage="category",
                turn_origin=turn_origin,
                session_instruction=category_session_instruction,
            )
            common.update(
                images=action_images,
                image_roles=action_image_roles,
                prefix_cache_namespace=category_prefix_namespace,
                cache_static_system_only=not bool(category_session_instruction),
            )
            emit_structured_log(
                "action",
                "user_camera_action_media_filtered",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                filtered_user_camera_image_count=(
                    filtered_user_camera_image_count
                ),
                reason="no_named_visual_action_scope",
            )
        action_context.update(
            {
                "visual_scope_user_camera_image_count": (
                    sum(
                        role == IMAGE_ROLE_USER_CAMERA
                        for role in action_image_roles
                    )
                    if visual_deictic_scope is not None
                    or CANDIDATE_REACTION_SOURCE_USER_CAMERA
                    in implicit_reaction_sources
                    else 0
                ),
                "filtered_user_camera_action_image_count": (
                    filtered_user_camera_image_count
                ),
            }
        )
        category_by_id = {
            item.category_id: item for item in eligible_categories
        }
        # A parsed, explicit body task may name one catalog action directly.
        # The shortcut only narrows recall; Child still validates support.
        exact_body_ids: set[str] = set()
        exact_body_matched_alias: str | None = None
        exact_catalog_alias_authoritative = False
        if forced_reaction_candidate_id is not None:
            exact_body_ids = {forced_reaction_candidate_id}
        if (
            allow_intent_shortcut
            and forced_category is None
            and turn.intent is not None
            and turn.intent.body_mode == "perform"
        ):
            shortcut_candidates = [
                (category, child)
                for category in eligible_categories
                if category.category_id not in excluded_category_id_set
                if not self._is_system_accompaniment_category(category)
                for child in self._filter_turn_action_candidates(
                    turn, list(category.children)
                )
            ]
            explicit_route = resolve_unique_explicit_action(
                turn.intent.body,
                shortcut_candidates,
                aliases_by_candidate_id={
                    candidate_id: candidate.aliases
                    for candidate_id, candidate in (
                        self.global_action_catalog.candidate_by_id.items()
                        if self.global_action_catalog is not None
                        else ()
                    )
                },
            )
            if explicit_route is not None and turn_origin == TURN_ORIGIN_USER:
                backoff = getattr(self, "_intent_shortcut_backoff", None)
                if backoff is None:
                    backoff = self._intent_shortcut_backoff = IntentShortcutBackoff()
                hint_key = (explicit_route.candidate.candidate_id, turn.intent.body.strip())
                if backoff.blocked(hint_key, time.monotonic()):
                    emit_structured_log(
                        "action", "action_intent_hint_backoff",
                        session_id=self.session_id, turn_id=turn.turn_id,
                        candidate_id=explicit_route.candidate.candidate_id,
                        reason="recent_validation_failure", ttl_seconds=30,
                    )
                    explicit_route = None
            if explicit_route is not None:
                shortcut_state_exclusions = (
                    self._state_description_excluded_candidate_ids(
                        effective_avatar_state.get("state_description"),
                        [explicit_route.candidate],
                    )
                    if self.global_action_catalog is not None
                    and turn_origin != TURN_ORIGIN_PROACTIVE
                    else ()
                )
                if not shortcut_state_exclusions:
                    forced_category = explicit_route.category
                    forced_semantic_tag = "shared_intent_unique_catalog_alias"
                    exact_body_ids = {explicit_route.candidate.candidate_id}
                    exact_body_matched_alias = explicit_route.matched_alias
                    catalog_candidate = (
                        self.global_action_catalog.candidate_by_id.get(
                            explicit_route.candidate.candidate_id
                        )
                        if self.global_action_catalog is not None
                        else None
                    )
                    exact_catalog_alias_authoritative = bool(
                        catalog_candidate is not None
                        and explicit_route.matched_alias
                        in catalog_candidate.aliases
                    )
        category_result = None
        category_ranked: list[Any] = []
        category_ms = 0.0
        category_scoring_skipped = forced_category is not None
        category_scoring_skip_reason = (
            (
                DIRECT_GREETING_ROUTE
                if forced_reaction_candidate_id is not None
                else (
                    "shared_intent_unique_catalog_alias"
                    if exact_body_ids
                    else "trigger_policy"
                )
            )
            if forced_category is not None
            else None
        )
        effective_category_top_k = 1 if forced_category is not None else 0
        category_adaptive_top1_applied = False
        category_width_reason = (
            "category_forced" if forced_category is not None else None
        )
        category_top_ppl: float | None = None
        category_confidence_margin: float | None = None
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
                matched_catalog_alias=exact_body_matched_alias,
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
                session_instruction=category_session_instruction,
                prefix=(
                    last_user_action_reference
                    + proactive_repeat_instruction
                    + base
                    + visual_deictic_instruction
                    + implicit_reaction_instruction
                    + self._category_whitelist_instruction()
                    + self._state_description_exclusion_instruction(
                        category_ids=excluded_category_ids
                    )
                    + self._state_description_priority_instruction(
                        "category",
                        enabled=("state_description" in effective_avatar_state),
                    )
                ),
                output_prompt=self._action_prompt(
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
            # The category stage is a recall stage. 00 remains useful as a
            # diagnostic score, but it must not prevent the best real
            # categories from reaching child scoring. Only 000 at the child
            # stage is allowed to make the final unsupported decision.
            category_unsupported = False
            ranked_real_categories = [
                item
                for item in category_ranked
                if item.candidate_id in category_by_id
            ]
            top_category_prewarmed = (
                turn_origin == TURN_ORIGIN_USER
                and self.global_action_catalog is not None
                and bool(ranked_real_categories)
                and ranked_real_categories[0].candidate_id
                in self.global_action_prewarm.for_locale(
                    self.action_locale
                ).ready_child_category_ids
            )
            if visual_deictic_scope is not None:
                effective_category_top_k = len(ranked_real_categories)
                category_adaptive_top1_applied = False
                category_width_reason = "visual_deictic_range_scope"
                category_top_ppl = (
                    ranked_real_categories[0].ppl
                    if ranked_real_categories
                    else None
                )
                category_confidence_margin = (
                    ranked_real_categories[0].mean_logprob
                    - ranked_real_categories[1].mean_logprob
                    if len(ranked_real_categories) > 1
                    else None
                )
            else:
                width_decision = choose_category_width(
                    configured_top_k=self.action_category_top_k,
                    ranked_real_scores=[
                        (item.candidate_id, item.mean_logprob, item.ppl)
                        for item in ranked_real_categories
                    ],
                    overall_top_candidate_id=category_ranked[0].candidate_id,
                    adaptive_enabled=self.action_category_adaptive_top1,
                    top_category_prewarmed=top_category_prewarmed,
                    min_margin=self.action_category_top1_min_margin,
                    max_ppl=self.action_category_top1_max_ppl,
                )
                effective_category_top_k = width_decision.effective_top_k
                category_adaptive_top1_applied = width_decision.adaptive_top1
                category_width_reason = width_decision.reason
                category_top_ppl = width_decision.top_ppl
                category_confidence_margin = width_decision.confidence_margin
            emit_structured_log(
                "action",
                "action_category_width_selected",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                configured_top_k=self.action_category_top_k,
                effective_top_k=effective_category_top_k,
                adaptive_top1_applied=category_adaptive_top1_applied,
                decision_reason=category_width_reason,
                top_ppl=category_top_ppl,
                confidence_margin=category_confidence_margin,
                top_category_prewarmed=top_category_prewarmed,
            )
            selected_categories = [
                category_by_id[item.candidate_id]
                for item in ranked_real_categories[:effective_category_top_k]
            ]
            if not selected_categories:
                raise ValueError(
                    "category action score did not select a valid category"
                )
            selected_category = selected_categories[0]
            category_scoring_candidate_id = category_ranked[0].candidate_id
        implicit_reaction_active = False
        if self._body_accompaniment_only(turn):
            # Keep the raw category scores/top-k untouched for audit. The parsed
            # task constrains the execution pool, including its fallback paths.
            raw_category_ids = [item.category_id for item in selected_categories]
            if (
                implicit_reaction_sources
                and selected_categories
                and not self._is_system_accompaniment_category(
                    selected_categories[0]
                )
            ):
                selected_categories = [
                    category
                    for category in selected_categories
                    if not self._is_system_accompaniment_category(category)
                    and any(
                        child.candidate_id
                        in implicit_reaction_candidate_ids
                        for child in category.children
                    )
                ]
                selected_category = selected_categories[0]
                implicit_reaction_active = True
                category_unsupported = False
            else:
                tag = (CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
                       if turn.intent.speech == "none"
                       else CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT)
                accompaniment = self._category_with_semantic_tag(tag)
                if accompaniment is None:
                    raise ValueError("session is missing the required accompaniment category")
                selected_category = accompaniment
                selected_categories = [accompaniment]
                category_unsupported = False
            emit_structured_log(
                "action", "action_execution_scope_constrained",
                session_id=self.session_id, turn_id=turn.turn_id,
                raw_selected_category_ids=raw_category_ids,
                execution_category_ids=[selected_category.category_id],
                reason=(
                    "implicit_candidate_reaction"
                    if implicit_reaction_active
                    else "shared_intent_no_body_request"
                ),
            )
        action_context.update(
            {
                "implicit_reaction_active": implicit_reaction_active,
                "implicit_reaction_sources": sorted(
                    implicit_reaction_sources
                ),
                "implicit_reaction_candidate_ids": sorted(
                    implicit_reaction_candidate_ids
                ),
            }
        )
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
            if turn_origin == TURN_ORIGIN_USER and turn.intent is not None:
                # Accompaniment follows the already parsed speech intent; never
                # wait for generated text on the action-critical user path.
                reply_prefix_status = (
                    "reply_pending" if turn.intent.speech != "none"
                    else "empty_completed"
                )
            elif provisional_reply is None:
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
                if reply_prefix or reply_prefix_status == "reply_pending"
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
        if exact_body_ids:
            child_candidates = [item for item in child_candidates if item.candidate_id in exact_body_ids]
        if implicit_reaction_active:
            child_candidates = [
                item
                for item in child_candidates
                if item.candidate_id in implicit_reaction_candidate_ids
            ]
        child_candidates = self._filter_turn_action_candidates(
            turn, child_candidates
        )
        excluded_candidate_ids = (
            self._state_description_excluded_candidate_ids(
                effective_avatar_state.get("state_description"), child_candidates
            )
            if self.global_action_catalog is not None
            and (turn_origin != TURN_ORIGIN_PROACTIVE or persona_first)
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
        if persona_first and not child_candidates:
            fallback_ids = SCENE_FALLBACK_IDS[trigger]
            pool = self._filter_turn_action_candidates(
                turn, [c for c in self.candidates if c.candidate_id in fallback_ids]
            )
            prohibited = set(self._state_description_excluded_candidate_ids(
                effective_avatar_state.get("state_description"), pool
            ))
            child_candidates = [c for c in pool if c.candidate_id not in prohibited
                                and c.category_id not in excluded_category_id_set]
            if not child_candidates:
                raise ValueError("proactive scene has no executable action candidate")
            selected_categories = [c for c in self.categories if any(
                item.category_id == c.category_id for item in child_candidates
            )]
            execution_category = selected_category = selected_categories[0]
            selected_category_ids = [c.category_id for c in selected_categories]
            emit_structured_log(
                "action", "proactive_scene_fallback_selected",
                session_id=self.session_id, turn_id=turn.turn_id, trigger=trigger,
                candidate_ids=[c.candidate_id for c in child_candidates],
            )
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
            forced_reaction_candidate_id is not None
            and len(child_candidates) == 1
        ):
            direct_child = child_candidates[0]
            child_scoring_skip_reason = DIRECT_GREETING_ROUTE
        elif (
            exact_catalog_alias_authoritative
            and len(child_candidates) == 1
        ):
            # A versioned catalog alias is an exact action contract.  The
            # allowlist and avatar-state exclusions above remain authoritative;
            # once they admit the sole target, a second semantic score must not
            # reinterpret the same request as unsupported.
            direct_child = child_candidates[0]
            child_scoring_skip_reason = "exact_catalog_alias"
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
                    "child_definition_mode": child_definition_mode,
                    "support_status": (
                        "unsupported" if category_unsupported else "supported"
                    ),
                    "fallback_applied": category_unsupported,
                    "selected_category_ids": selected_category_ids,
                    "category_top_k": self.action_category_top_k,
                    "effective_category_top_k": effective_category_top_k,
                    "category_adaptive_top1_applied": (
                        category_adaptive_top1_applied
                    ),
                    "category_width_reason": category_width_reason,
                    "category_top_ppl": category_top_ppl,
                    "category_confidence_margin": category_confidence_margin,
                    "intent_shortcut_matched_alias": exact_body_matched_alias,
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
                            self.action_locale
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
        if turn_origin == TURN_ORIGIN_USER and len(selected_categories) > 1:
            child_namespace = ",".join(sorted(selected_category_ids))
        if len(selected_categories) == 1:
            child_system_prompt = self._build_child_system_prompt(
                execution_category,
                child_candidates,
                turn_origin,
                child_definition_mode,
                persona_first=persona_first,
            )
        else:
            child_system_prompt = self._build_child_system_prompt(
                selected_categories,
                child_candidates,
                turn_origin,
                child_definition_mode,
                persona_first=persona_first,
            )
        if (
            self.global_action_catalog is not None
            and len(selected_categories) == 1
            and child_definition_mode == "contextual"
            and not persona_first
        ):
            child_namespace = self.global_action_catalog.child_cache_namespace(
                execution_category.category_id, self.action_locale, turn_origin
            )
        elif self.global_action_catalog is not None:
            combined_prompt_hash = prompt_sha256(child_system_prompt)
            child_namespace = (
                f"hierarchical:{self.action_locale}:child:{child_namespace}:"
                f"{turn_origin}:{child_definition_mode}:sha256:{combined_prompt_hash}"
            )
        else:
            child_namespace = (
                f"{self.action_prefix_cache_namespace}:child:{child_namespace}"
            )

        child_session_instruction = self._build_session_action_profile_instruction(
            "child",
            turn_origin=turn_origin,
            has_user_camera=IMAGE_ROLE_USER_CAMERA in action_image_roles,
        )
        child_session_namespace = self._session_action_prefix_namespace(
            base_namespace=child_namespace,
            stage="child",
            turn_origin=turn_origin,
            session_instruction=child_session_instruction,
        )

        # Global Child catalogs are prewarmed before the server starts. The
        # Session-specific extension is populated by the first real Child
        # score and then reused. The legacy/session-local path retains its
        # explicit lazy first-use prefill.
        child_prefix_prefilled = (
            len(selected_categories) == 1
            and self.global_action_catalog is not None
            and turn_origin == TURN_ORIGIN_USER
            and child_definition_mode == "contextual"
            and execution_category.category_id
            in self.global_action_prewarm.for_locale(
                self.action_locale
            ).ready_child_category_ids
        )
        child_catalog_prefill_ms = 0.0
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if (
            callable(prefill)
            and self.global_action_catalog is None
            and child_session_namespace
            not in self._prefilled_action_prefix_namespaces
        ):
            child_prefill_started = time.perf_counter()
            prefill_request_id = request_base + "-child-prefill"
            self._ensure_turn_processing(turn)
            self._register_turn_request(turn, prefill_request_id)
            try:
                child_prefix_prefilled = await prefill(
                    request_id=prefill_request_id,
                    session_instance_id=self.session_instance_id,
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
                    prefix_cache_namespace=child_session_namespace,
                    stage="child",
                    language=self.action_language,
                    session_instruction=child_session_instruction,
                )
            finally:
                self._unregister_turn_request(turn, prefill_request_id)
            self._ensure_turn_processing(turn)
            if child_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    child_session_namespace
                )
            child_catalog_prefill_ms = round(
                (time.perf_counter() - child_prefill_started) * 1000.0, 3
            )

        action_common = {
            **common,
            "stage": "child",
            "admission_priority": 1,
            "micro_batch_size": self.action_micro_batch_size,
            "prefix_cache_namespace": child_session_namespace,
        }
        action_common["cache_static_system_only"] = not bool(
            child_session_instruction
        )
        action_request = ActionSuffixScoreRequest(
            request_id=request_base + "-child",
            session_instruction=child_session_instruction,
            prefix=(
                last_user_action_reference
                + proactive_repeat_instruction
                + base
                + visual_deictic_instruction
                + visual_deictic_child_instruction
                + implicit_reaction_instruction
                + self._system_accompaniment_child_instruction(
                    execution_category,
                    reply_prefix=reply_prefix,
                )
                + self._child_whitelist_instruction(
                    selected_categories, child_candidates, persona_first=persona_first
                )
                + self._state_description_exclusion_instruction(
                    candidate_ids=excluded_candidate_ids
                )
                + self._state_description_priority_instruction(
                    "child",
                    enabled=("state_description" in effective_avatar_state),
                )
            ),
            output_prompt=self._action_prompt(
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
                or persona_first
                or all(
                    self._is_system_accompaniment_category(category)
                    for category in selected_categories
                )
                # Language intent has already made the bounded social-reaction
                # decision.  Keep visual-only reactions rejectable because a
                # still image may be ambiguous, but do not let Child undo an
                # explicit greeting/farewell reaction with 000.
                or (
                    implicit_reaction_active
                    and CANDIDATE_REACTION_SOURCE_LANGUAGE
                    in implicit_reaction_sources
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
        child_result = await self._score_action_request(
            turn, action_request, child_category_ids=selected_category_ids,
        )
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
        if persona_first:
            # Never accept out-of-contract sentinel scores from an adapter.
            ranked = [score for score in ranked if score.candidate_id in child_by_id]
            if not ranked:
                raise ValueError("proactive action scoring returned no allowed candidate")
        child_unsupported = ranked[0].candidate_id == UNSUPPORTED_CHILD_SCORE_ID
        if child_unsupported and exact_body_ids.intersection(child_by_id):
            if turn_origin == TURN_ORIGIN_USER and turn.intent is not None:
                backoff = getattr(self, "_intent_shortcut_backoff", None)
                if backoff is not None:
                    for candidate_id in exact_body_ids:
                        backoff.reject((candidate_id, turn.intent.body.strip()), time.monotonic())
            # A semantic hint is only a recall shortcut. If its narrowed child
            # set fails validation, run the original ranked top-k path once.
            # Do not force a hinted ID or discard state/negation constraints.
            # A target removed by the allowlist/state filter is definitive;
            # expanding recall must never work around that restriction.
            hint_ms = round((time.perf_counter() - started) * 1000.0, 3)
            emit_structured_log(
                "action", "action_intent_hint_rejected",
                session_id=self.session_id, turn_id=turn.turn_id,
                logical_request_id=request_base, reason="child_unsupported",
                elapsed_ms=hint_ms,
            )
            action, scores, _, recalled_context = await self._score_action_hierarchical(
                audios, images, image_roles, text, avatar_state,
                turn_origin=turn_origin, text_role=text_role, trigger=trigger,
                turn=turn, request_base=request_base + "-hint-recall",
                provisional_reply=provisional_reply, turn_id=turn_id,
                on_category_selected=on_category_selected,
                allow_intent_shortcut=False,
            )
            total_ms = round((time.perf_counter() - started) * 1000.0, 3)
            breakdown = recalled_context.get("action_timing_breakdown", {})
            breakdown["intent_hint_validation"] = {
                "elapsed_ms": hint_ms, "reason": "child_unsupported",
                "child": _action_timing_breakdown(child_result.stats),
            }
            breakdown["total_ms"] = total_ms
            recalled_context["action_timing_breakdown"] = breakdown
            return action, scores, total_ms, recalled_context
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
        if self._body_accompaniment_only(turn):
            candidate = self.candidate_by_id.get(action["candidate_id"])
            if candidate is None or not (
                self._is_accompaniment_candidate(candidate)
                or (
                    implicit_reaction_active
                    and candidate.candidate_id
                    in implicit_reaction_candidate_ids
                )
            ):
                raise ValueError("action exceeds parsed accompaniment scope")
        selected_candidate = self.candidate_by_id.get(
            action["candidate_id"]
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
            "selection_definition_source": (
                (
                    "short_definition_visual"
                    if child_definition_mode == "visual"
                    else selected_candidate.definition_source(turn_origin)
                )
                if selected_candidate is not None
                else None
            ),
            "selection_definition_hash": (
                "sha256:" + hashlib.sha256(
                    (
                        selected_candidate.short_definition
                        if child_definition_mode == "visual"
                        else selected_candidate.effective_definition(turn_origin)
                    ).encode("utf-8")
                ).hexdigest()
                if selected_candidate is not None
                else None
            ),
            "child_definition_mode": child_definition_mode,
            "implicit_reaction_active": implicit_reaction_active,
            "implicit_reaction_sources": sorted(
                implicit_reaction_sources
            ),
            "implicit_reaction_candidate_ids": sorted(
                implicit_reaction_candidate_ids
            ),
            "selected_category_ids": selected_category_ids,
            "category_top_k": self.action_category_top_k,
            "effective_category_top_k": effective_category_top_k,
            "category_adaptive_top1_applied": category_adaptive_top1_applied,
            "category_width_reason": category_width_reason,
            "category_top_ppl": category_top_ppl,
            "category_confidence_margin": category_confidence_margin,
            "intent_shortcut_matched_alias": exact_body_matched_alias,
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
            "child_prefix_cache_namespace": child_session_namespace,
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
