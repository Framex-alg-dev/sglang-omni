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
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_image_roles,
            action_context,
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
        prefix = (
            self._build_session_action_profile_instruction("single")
            + self._last_user_action_reference_instruction(
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
        )
        eligible_candidates = self._filter_turn_action_candidates(
            turn, list(self.candidates)
        )
        if not eligible_candidates:
            raise ValueError(
                "per-turn action candidate constraints leave no executable action"
            )
        candidates = [
            ActionScoreCandidate(
                candidate_id=item.candidate_id,
                suffix=item.candidate_id,
                action_id=item.action_id,
                execution_binding=dict(item.execution_binding),
            )
            for item in eligible_candidates
        ]
        request = ActionSuffixScoreRequest(
            request_id=request_base + "-single",
            model=self.model_name,
            prefix=prefix,
            current_text=text or "",
            output_prompt=self._prompt(
                zh="最合适的 candidate_id：",
                en="Best matching candidate_id:",
            ),
            system_prompt=self.action_system_prompt,
            language=self.language,
            candidates=candidates,
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=action_images,
            image_roles=action_image_roles,
            sample_rate=16000,
            micro_batch_size=self.action_micro_batch_size,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            cache_static_system_only=self.global_action_catalog is not None,
            session_id=self.session_id,
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
            candidate = self.candidate_by_id[score.candidate_id]
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
        action = {
            "candidate_id": top["candidate_id"],
            "action_id": top["action_id"],
            **({"category_id": top["category_id"]} if top.get("category_id") else {}),
            "execution_binding": dict(top.get("execution_binding") or {}),
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"],
            "ppl": top["ppl"],
            "token_count": top["token_count"],
        }
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
                "action_timing_breakdown": {
                    "selection_mode": self.action_selection_mode,
                    "single": _action_timing_breakdown(result.stats),
                    "total_ms": compute_ms,
                },
            }
        )
        return action, scores, compute_ms, action_context


MultimodalActionCandidateMixin = ActionCandidateComponent
