"""Flat child-candidate scoring and selection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import time
from typing import Any, Callable, Literal

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    CandidateScore,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
    child_unsupported_policy,
)
from sglang_omni.serve.realtime.action.routing import (
    scope_visual_deictic_categories,
    visual_deictic_scope_candidates,
)
from sglang_omni.serve.realtime.action.background_calibration import (
    compare_supported_action_scores,
)
from sglang_omni.serve.realtime.action.decision import (
    ACTION_DECISION_LABELS,
    ACTION_SUPPORT_LABELS,
    action_support_as_dict,
    action_support_candidates,
    action_decision_candidates,
    aggregate_action_decision,
    aggregate_action_support,
    aggregate_category_gate,
    category_gate_as_dict,
    category_gate_candidates,
    category_gate_score_ids,
    decision_as_dict,
    fuse_category_gate_into_action_decision,
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


class ActionCandidateComponent:
    async def _score_action_flat(
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
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        base_eligible_candidates = self._filter_turn_action_candidates(
            turn,
            [
                candidate
                for candidate in self.candidates
                if self.direct_action_selection
                or (
                    candidate.category_id != FACIAL_EXPRESSION_CATEGORY_ID
                    and candidate.candidate_id
                    not in self._facial_expression_candidate_ids()
                )
            ],
        )
        visual_deictic_scope = scope_visual_deictic_categories(
            self.categories,
            body_task=(turn.intent.body if turn.intent is not None else ""),
            body_mode=(
                turn.intent.body_mode if turn.intent is not None else "none"
            ),
            has_user_camera=IMAGE_ROLE_USER_CAMERA in image_roles,
        )
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_image_roles,
            action_context,
        ) = self._build_bounded_action_context(
            audios,
            images,
            image_roles,
            current_user_camera_image_limit=(
                0
                if turn.intent is not None
                and not turn.intent.visual_scope_gate
                else (
                    MAX_ACTION_VISUAL_SCOPE_USER_CAMERA_IMAGES
                    if visual_deictic_scope is not None
                    else MAX_ACTION_CURRENT_USER_CAMERA_IMAGES
                )
            ),
        )
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
        session_instruction = self._build_session_action_profile_instruction(
            "single",
            turn_origin=turn_origin,
            has_user_camera=IMAGE_ROLE_USER_CAMERA in action_image_roles,
        )
        visual_deictic_instruction = ""
        scoped_candidate_ids: set[str] | None = None
        if visual_deictic_scope is not None:
            scoped_candidates = visual_deictic_scope_candidates(
                visual_deictic_scope
            )
            scoped_candidate_ids = {
                candidate.candidate_id for candidate in scoped_candidates
            }
            session_instruction += self._visual_deictic_catalog_instruction(
                visual_deictic_scope,
                turn_origin,
            )
            visual_deictic_instruction = self._action_prompt(
                zh=(
                    "\n[本轮视觉模仿判定]\n"
                    f"使用前述范围={visual_deictic_scope.name} 的视觉候选目录，"
                    "只根据本轮 user_camera 画面选择 candidate_id；"
                    "证据不足或没有匹配项时选择 000。\n"
                ),
                en=(
                    "\n[Visual-imitation decision for this interaction]\n"
                    f"Use the visual candidate catalog for scope={visual_deictic_scope.name} "
                    "above and select the candidate_id only from the current user_camera "
                    "view. Select 000 when evidence is insufficient or no candidate matches.\n"
                ),
            )
            action_context.update(
                {
                    "category_scope": (
                        "visual_deictic:" + visual_deictic_scope.name
                    ),
                    "category_scope_ids": [
                        category.category_id
                        for category in visual_deictic_scope.categories
                    ],
                    "visual_scope_user_camera_image_count": sum(
                        role == IMAGE_ROLE_USER_CAMERA
                        for role in action_image_roles
                    ),
                    "selection_definition_source": "short_definition_visual",
                }
            )
            emit_structured_log(
                "action",
                "visual_deictic_direct_scope_applied",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                scope=visual_deictic_scope.name,
                category_ids=action_context["category_scope_ids"],
                candidate_count=len(scoped_candidate_ids),
                user_camera_image_count=action_context[
                    "visual_scope_user_camera_image_count"
                ],
                child_definition_mode="visual",
            )
        prefix = (
            self._proactive_action_repeat_instruction(
                turn_origin=turn_origin,
                client_last_action_id=turn.client_last_executed_action_id,
            )
            + self._build_turn_action_instruction(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                has_audio=bool(audios),
                image_roles=action_image_roles,
                has_state_description=(
                    "state_description" in effective_avatar_state
                ),
                avatar_state_source=self._avatar_state_source(
                    effective_avatar_state, action_image_roles
                ),
            )
            + self._state_description_priority_instruction(
                "single",
                enabled=("state_description" in effective_avatar_state),
            )
            + visual_deictic_instruction
        )
        eligible_candidates = [
            candidate
            for candidate in base_eligible_candidates
            if (
                scoped_candidate_ids is None
                or candidate.candidate_id in scoped_candidate_ids
            )
        ]
        if not eligible_candidates and not self.direct_action_selection:
            raise ValueError(
                "per-turn action candidate constraints leave no executable action"
            )
        candidate_by_id = dict(self.candidate_by_id)
        if self.direct_action_selection:
            unsupported = SessionActionCandidate(
                candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                action_id=UNSUPPORTED_DECISION_ID,
                source_label="不支持的动作",
                short_definition="没有合适的可执行动作",
                execution_binding={},
            )
            eligible_candidates.append(unsupported)
            candidate_by_id[unsupported.candidate_id] = unsupported
        candidates = [
            ActionScoreCandidate(
                candidate_id=item.candidate_id,
                suffix=item.candidate_id,
                action_id=item.action_id,
                execution_binding=dict(item.execution_binding),
            )
            for item in eligible_candidates
        ]
        decision_batch_enabled = bool(
            self.direct_action_selection
            and turn_origin == TURN_ORIGIN_USER
            and getattr(self, "action_decision_batch_mode", "off") != "off"
        )
        decision_visual_enabled = bool(
            decision_batch_enabled
            and getattr(self, "action_decision_batch_visual", False)
            and IMAGE_ROLE_USER_CAMERA in action_image_roles
        )
        eligible_category_ids = {
            candidate.category_id
            for candidate in eligible_candidates
            if candidate.category_id is not None
            and candidate.candidate_id != UNSUPPORTED_CHILD_SCORE_ID
        }
        category_gate_categories = [
            category
            for category in self.categories
            if category.category_id in eligible_category_ids
        ]
        category_gate_enabled = bool(
            self._direct_category_gate_enabled(turn_origin)
            and visual_deictic_scope is None
            and category_gate_categories
        )
        category_score_ids = (
            category_gate_score_ids(category_gate_categories)
            if category_gate_enabled
            else frozenset()
        )
        action_support_score_ids = (
            frozenset(ACTION_SUPPORT_LABELS)
            if category_gate_enabled
            else frozenset()
        )
        if decision_batch_enabled:
            candidates.extend(
                action_decision_candidates(
                    include_visual=decision_visual_enabled,
                    english=self.action_language == "en",
                )
            )
        if category_gate_enabled:
            candidates.extend(
                category_gate_candidates(
                    category_gate_categories,
                    english=self.action_language == "en",
                )
            )
            candidates.extend(
                action_support_candidates(
                    english=self.action_language == "en"
                )
            )
        if (
            decision_batch_enabled or category_gate_enabled
        ) and len(candidates) > self.action_micro_batch_size:
            raise ValueError(
                "grouped action decisions require one physical suffix batch: "
                f"candidate_count={len(candidates)} exceeds "
                f"action_micro_batch_size={self.action_micro_batch_size}"
            )
        selection_mapping = getattr(
            self, "action_selection_token_mapping", None
        )
        if selection_mapping is not None:
            for candidate in candidates:
                selection = selection_mapping.entry(candidate.candidate_id)
                candidate.selection_token = selection.text
                candidate.selection_token_id = selection.token_id
                candidate.selection_score_bias = selection.score_bias
        action_system_prompt = self._build_action_system_prompt(
            turn_origin,
            include_visual=decision_visual_enabled,
            include_category_gate=category_gate_enabled,
            category_gate_categories=category_gate_categories,
        )
        prompt_hash = hashlib.sha256(
            action_system_prompt.encode("utf-8")
        ).hexdigest()
        request = ActionSuffixScoreRequest(
            request_id=request_base + "-single",
            model=self.model_name,
            session_instruction=session_instruction,
            prefix=prefix,
            current_text=text or "",
            output_prompt=self._action_prompt(
                zh=(
                    "最合适的 selection_token："
                    if getattr(self, "action_single_token_mode", "off") == "enforce"
                    else "最合适的结果标识："
                    if selection_mapping is not None
                    else "最合适的结果标识："
                ),
                en=(
                    "Best matching selection_token:"
                    if getattr(self, "action_single_token_mode", "off") == "enforce"
                    else "Best matching result identifier:"
                    if selection_mapping is not None
                    else "Best matching result identifier:"
                ),
            ),
            system_prompt=action_system_prompt,
            language=self.action_language,
            candidates=candidates,
            suffix_tokenization_mode="short_id",
            scoring_mode={
                "off": "suffix_ppl",
                "shadow": "single_token_shadow",
                "enforce": "single_token_enforce",
            }[getattr(self, "action_single_token_mode", "off")],
            selection_mapping_version=(
                selection_mapping.mapping_version
                if selection_mapping is not None else None
            ),
            selection_mapping_hash=(
                selection_mapping.mapping_hash
                if selection_mapping is not None else None
            ),
            selection_calibration_version=(
                selection_mapping.calibration_version
                if selection_mapping is not None else None
            ),
            selection_calibration_hash=(
                selection_mapping.calibration_hash
                if selection_mapping is not None else None
            ),
            audios=audios,
            images=action_images,
            image_roles=action_image_roles,
            sample_rate=16000,
            # Capacity is an upper bound only. The request contains exactly the
            # real action and decision candidates and is never padded to it.
            micro_batch_size=min(self.action_micro_batch_size, len(candidates)),
            prefix_cache_namespace=self._direct_action_prefix_namespace(
                turn_origin,
                session_instruction,
                include_visual=decision_visual_enabled,
                include_category_gate=category_gate_enabled,
                category_gate_category_ids=tuple(
                    category.category_id
                    for category in category_gate_categories
                ),
            )
            if self.direct_action_selection else self._session_action_prefix_namespace(
                base_namespace=(
                    f"{self.action_prefix_cache_namespace}:{turn_origin}:"
                    f"sha256:{prompt_hash}"
                ),
                stage="single",
                turn_origin=turn_origin,
                session_instruction=session_instruction,
            ),
            cache_static_system_only=not bool(session_instruction),
            admission_priority=0,
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            history=action_history,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=effective_avatar_state,
        )
        started = time.perf_counter()
        result = await self._score_action_request(turn, request)
        compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=flat candidates=%d "
            "elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(request.candidates),
            compute_ms,
            result.prefix_cached,
            json.dumps(result.stats, ensure_ascii=False, default=str),
        )
        raw_action_scores = [
            score
            for score in result.scores
            if score.candidate_id not in ACTION_DECISION_LABELS
            and score.candidate_id not in category_score_ids
            and score.candidate_id not in action_support_score_ids
        ]
        if not raw_action_scores:
            raise ValueError("action scoring returned no concrete action scores")
        action_scores = raw_action_scores
        action_decision = None
        if decision_batch_enabled:
            action_decision = aggregate_action_decision(
                result.scores,
                include_visual=decision_visual_enabled,
                min_margin=float(
                    getattr(self, "action_decision_min_margin", 0.10)
                ),
            )
            turn.action_decision = action_decision
            action_context["action_decision"] = decision_as_dict(action_decision)
            action_context["action_decision_batch_mode"] = getattr(
                self, "action_decision_batch_mode", "shadow"
            )
            action_context["action_decision_visual_enabled"] = (
                decision_visual_enabled
            )
            emit_structured_log(
                "action",
                "action_decision_batch_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                physical_candidate_count=len(candidates),
                concrete_action_candidate_count=len(raw_action_scores),
                decision_candidate_count=(
                    len(candidates)
                    - len(raw_action_scores)
                    - len(category_score_ids)
                    - len(action_support_score_ids)
                ),
                category_gate_candidate_count=len(category_score_ids),
                action_support_candidate_count=len(action_support_score_ids),
                decision=action_context["action_decision"],
                mode=action_context["action_decision_batch_mode"],
            )
        if category_gate_enabled:
            category_decision = aggregate_category_gate(
                result.scores, category_gate_categories
            )
            turn.action_category_decision = category_decision
            category_decision_payload = category_gate_as_dict(
                category_decision
            )
            support_decision = aggregate_action_support(result.scores)
            support_decision_payload = action_support_as_dict(
                support_decision
            )
            category_min_margin = float(
                getattr(self, "action_decision_min_margin", 0.10)
            )
            category_support_min_margin = max(
                0.20,
                category_min_margin * 2.0,
            )
            support_unsupported_min_margin = max(
                0.50,
                category_min_margin * 5.0,
            )
            support_confidently_unsupported = bool(
                not support_decision.supported
                and support_decision.margin is not None
                and support_decision.margin
                >= support_unsupported_min_margin
            )
            support_decision_payload["unsupported_min_margin"] = (
                support_unsupported_min_margin
            )
            support_decision_payload["confidently_unsupported"] = (
                support_confidently_unsupported
            )
            action_context["category_gate"] = category_decision_payload
            action_context["action_support_decision"] = (
                support_decision_payload
            )
            action_context["category_gate_mode"] = "same_batch_enforce"
            if action_decision is not None:
                fused_decision = fuse_category_gate_into_action_decision(
                    action_decision,
                    category_decision,
                    min_margin=float(
                        getattr(self, "action_decision_min_margin", 0.10)
                    ),
                )
                if fused_decision is not action_decision:
                    action_decision = fused_decision
                    turn.action_decision = action_decision
                    action_context["action_decision"] = decision_as_dict(
                        action_decision
                    )
                    action_context["category_gate_body_fused"] = True
            if category_decision.no_action_request:
                action_scores = raw_action_scores
            elif (
                category_decision.unsupported
                or (
                    category_decision.category_id is not None
                    and category_decision.support_margin is not None
                    and category_decision.support_margin
                    < category_support_min_margin
                )
                or support_confidently_unsupported
            ):
                action_scores = [
                    score
                    for score in raw_action_scores
                    if score.candidate_id == UNSUPPORTED_CHILD_SCORE_ID
                ]
            else:
                action_scores = [
                    score
                    for score in raw_action_scores
                    if candidate_by_id[score.candidate_id].category_id
                    == category_decision.category_id
                ]
            if not action_scores:
                raise ValueError(
                    "category gate left no concrete action or unsupported score"
                )
            emit_structured_log(
                "action",
                "action_category_gate_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                physical_candidate_count=len(candidates),
                category_candidate_count=len(category_score_ids),
                action_support_candidate_count=len(action_support_score_ids),
                gated_concrete_candidate_count=len(action_scores),
                decision=category_decision_payload,
                support_decision=support_decision_payload,
                mode="same_batch_enforce",
            )
        ranked = sorted(
            action_scores, key=lambda x: x.mean_logprob, reverse=True
        )
        background_shadow: dict[str, Any] | None = None
        background_diagnostics: dict[
            str, dict[str, float | int]
        ] = {}
        calibration = getattr(self, "action_background_calibration", None)
        if calibration is not None:
            background_shadow, background_diagnostics = (
                compare_supported_action_scores(
                    raw_action_scores,
                    calibration,
                    catalog_hash=self.global_action_catalog_hash,
                )
            )
            action_context["background_calibration_shadow"] = background_shadow
            emit_structured_log(
                "diagnostic",
                "action_background_calibration_shadow_compared",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=request_base,
                turn_origin=turn_origin,
                text_role=text_role,
                intent_body_mode=(turn.intent.body_mode if turn.intent else None),
                **_text_audit_fields(
                    "intent_body", turn.intent.body if turn.intent else None
                ),
                **background_shadow,
            )
        scores: list[dict[str, Any]] = []
        for score in ranked:
            candidate = candidate_by_id[score.candidate_id]
            background = background_diagnostics.get(score.candidate_id)
            scores.append(
                {
                    "candidate_id": score.candidate_id,
                    "action_id": candidate.action_id,
                    **(
                        {"category_id": candidate.category_id}
                        if candidate.category_id
                        else {}
                    ),
                    "source_label": candidate.source_label,
                    "short_definition": candidate.short_definition,
                    "execution_binding": dict(candidate.execution_binding),
                    "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob,
                    "mean_nll": score.mean_nll,
                    "ppl": score.ppl,
                    **(
                        {
                            "background_centered_bias": background[
                                "centered_bias"
                            ],
                            "background_calibrated_score": background[
                                "calibrated_score"
                            ],
                            "background_raw_rank": background["raw_rank"],
                            "background_calibrated_rank": background[
                                "calibrated_rank"
                            ],
                        }
                        if self.include_scores and background is not None
                        else {}
                    ),
                    "token_scores": [
                        {"token_id": item.token_id, "logprob": item.logprob}
                        for item in score.token_scores
                    ],
                }
            )
        shadow_values = result.stats.get("single_token_shadow_scores")
        if self.include_scores and isinstance(shadow_values, dict):
            shadow_ranking: list[dict[str, Any]] = []
            for candidate_id, raw_value in shadow_values.items():
                if (
                    candidate_id in ACTION_DECISION_LABELS
                    or candidate_id in category_score_ids
                    or candidate_id in action_support_score_ids
                ):
                    continue
                candidate = candidate_by_id.get(candidate_id)
                if candidate is None:
                    continue
                value = float(raw_value)
                shadow_ranking.append(
                    {
                        "candidate_id": candidate_id,
                        "action_id": candidate.action_id,
                        "source_label": candidate.source_label,
                        "mean_logprob": value,
                        "mean_nll": -value,
                        "ppl": math.exp(min(-value, 700.0)),
                        "token_count": 1,
                    }
                )
            shadow_ranking.sort(
                key=lambda item: item["mean_logprob"], reverse=True
            )
            shadow_decision = None
            if decision_batch_enabled:
                decision_scores = [
                    CandidateScore(
                        candidate_id=candidate_id,
                        token_count=1,
                        mean_logprob=float(shadow_values[candidate_id]),
                        mean_nll=-float(shadow_values[candidate_id]),
                        ppl=math.exp(
                            min(-float(shadow_values[candidate_id]), 700.0)
                        ),
                        token_scores=[],
                    )
                    for candidate_id in ACTION_DECISION_LABELS
                    if candidate_id in shadow_values
                ]
                shadow_decision = decision_as_dict(
                    aggregate_action_decision(
                        decision_scores,
                        include_visual=decision_visual_enabled,
                        min_margin=float(
                            getattr(self, "action_decision_min_margin", 0.10)
                        ),
                    )
                )
            action_context["single_token_shadow"] = {
                "mapping_version": result.stats.get(
                    "selection_mapping_version"
                ),
                "mapping_hash": result.stats.get("selection_mapping_hash"),
                "calibration_version": result.stats.get(
                    "selection_calibration_version"
                ),
                "calibration_hash": result.stats.get(
                    "selection_calibration_hash"
                ),
                "ranking": shadow_ranking,
                "decision": shadow_decision,
                "winners": result.stats.get(
                    "single_token_shadow_winners", {}
                ),
                "legacy_winners": result.stats.get(
                    "suffix_ppl_winners", {}
                ),
                "agreement": result.stats.get(
                    "single_token_shadow_agreement", {}
                ),
                "legacy_winner_rank": result.stats.get(
                    "suffix_winner_single_token_rank", {}
                ),
            }
        top = scores[0]
        selected_candidate = candidate_by_id[top["candidate_id"]]
        action = {
            "candidate_id": top["candidate_id"],
            "action_id": top["action_id"],
            **({"category_id": top["category_id"]} if top.get("category_id") else {}),
            "execution_binding": dict(top.get("execution_binding") or {}),
            "execute": top["action_id"] not in {"no_action", UNSUPPORTED_DECISION_ID},
            "mean_logprob": top["mean_logprob"],
            "ppl": top["ppl"],
            "token_count": top["token_count"],
        }
        if self.direct_action_selection:
            action["support_status"] = (
                "unsupported" if top["action_id"] == UNSUPPORTED_DECISION_ID else "supported"
            )
            if action["support_status"] == "unsupported":
                action["candidate_id"] = UNSUPPORTED_DECISION_ID
        if (
            visual_deictic_scope is not None
            and visual_deictic_scope.name == "gesture"
            and action.get("support_status") != "unsupported"
        ):
            # Each explicit camera-backed imitation request is a new execution
            # command even when it resolves to the same catalog action as the
            # previous Turn.
            action["allow_adjacent_repeat"] = True
        action_context.update(
            {
                "selection_stages": 1,
                "selection_mode": (
                    self.action_selection_mode if self.categories else "flat"
                ),
                "flattened_child_count": (
                    len(eligible_candidates) if self.categories else None
                ),
                "turn_action_allowed_candidate_ids": list(
                    turn.action_allowed_candidate_ids
                ),
                "turn_action_excluded_candidate_ids": list(
                    turn.action_excluded_candidate_ids
                ),
                "compute_ms": compute_ms,
                "selection_definition_source": (
                    "short_definition_visual"
                    if visual_deictic_scope is not None
                    else selected_candidate.definition_source(turn_origin)
                ),
                "selection_definition_hash": "sha256:"
                + hashlib.sha256(
                    (
                        selected_candidate.short_definition
                        if visual_deictic_scope is not None
                        else selected_candidate.effective_definition(turn_origin)
                    ).encode("utf-8")
                ).hexdigest(),
                "action_timing_breakdown": {
                    "selection_mode": self.action_selection_mode,
                    "single": _action_timing_breakdown(result.stats),
                    "total_ms": compute_ms,
                },
            }
        )
        return action, scores, compute_ms, action_context


MultimodalActionCandidateMixin = ActionCandidateComponent
