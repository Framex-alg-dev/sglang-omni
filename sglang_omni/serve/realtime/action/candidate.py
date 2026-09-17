"""Flat child-candidate scoring and selection."""

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
from sglang_omni.serve.realtime.action.routing import (
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
                MAX_ACTION_VISUAL_SCOPE_USER_CAMERA_IMAGES
                if visual_deictic_scope is not None
                else MAX_ACTION_CURRENT_USER_CAMERA_IMAGES
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
            scoped_candidates = [
                child
                for category in visual_deictic_scope.categories
                for child in category.children
            ]
            scoped_candidate_ids = {
                candidate.candidate_id for candidate in scoped_candidates
            }
            visual_lines = "\n".join(
                self._format_candidate_for_prompt(
                    candidate,
                    turn_origin,
                    definition_mode="visual",
                )
                for candidate in scoped_candidates
            )
            visual_deictic_instruction = self._action_prompt(
                zh=(
                    "\n[视觉模仿硬约束]\n"
                    "视觉范围 gate 已确认用户要求模仿 user_camera 中展示的动作；"
                    f"范围={visual_deictic_scope.name}。avatar_state 只表示数字人当前状态，"
                    "不能作为要模仿的目标。只比较下列候选的视觉定义，逐项核对参与"
                    "部位数量、手指伸直或弯曲状态、相对位置和朝向；证据不足或没有"
                    "匹配项时选择 000，不得按候选常见程度猜测：\n"
                    f"{visual_lines}\n"
                ),
                en=(
                    "\n[Hard visual-imitation constraint]\n"
                    "The visual-scope gate has confirmed that the user asks to imitate "
                    "the action shown in user_camera; "
                    f"scope={visual_deictic_scope.name}. avatar_state describes only "
                    "the character's current state and is never the imitation target. "
                    "Compare only the visual definitions below, including participating "
                    "parts, extension or flexion, relative positions, and orientation. "
                    "Select 000 when evidence is insufficient or no candidate matches; "
                    "never guess from candidate frequency:\n"
                    f"{visual_lines}\n"
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
            self._last_user_action_reference_instruction(
                turn_origin=turn_origin,
            )
            + self._proactive_action_repeat_instruction(
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
        eligible_candidates = self._filter_turn_action_candidates(
            turn,
            [
                candidate
                for candidate in self.candidates
                if self.direct_action_selection or (
                    candidate.category_id != FACIAL_EXPRESSION_CATEGORY_ID
                    and candidate.candidate_id not in self._facial_expression_candidate_ids()
                )
                if scoped_candidate_ids is None
                or candidate.candidate_id in scoped_candidate_ids
            ],
        )
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
        action_system_prompt = self._build_action_system_prompt(turn_origin)
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
                zh="最合适的 candidate_id：",
                en="Best matching candidate_id:",
            ),
            system_prompt=action_system_prompt,
            language=self.action_language,
            candidates=candidates,
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=action_images,
            image_roles=action_image_roles,
            sample_rate=16000,
            micro_batch_size=min(self.action_micro_batch_size, len(candidates)),
            prefix_cache_namespace=self._direct_action_prefix_namespace(turn_origin, session_instruction)
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
        ranked = sorted(result.scores, key=lambda x: x.mean_logprob, reverse=True)
        scores: list[dict[str, Any]] = []
        for score in ranked:
            candidate = candidate_by_id[score.candidate_id]
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
                    "token_scores": [
                        {"token_id": item.token_id, "logprob": item.logprob}
                        for item in score.token_scores
                    ],
                }
            )
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
