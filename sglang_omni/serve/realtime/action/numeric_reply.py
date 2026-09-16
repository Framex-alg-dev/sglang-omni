"""Bounded action selection from a completed generated reply."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.serve.realtime.action.routing import (
    NumericGestureCandidate,
    NumericReplyActionRoute,
    complete_reply_may_contain_numeric_answer,
)
from sglang_omni.serve.realtime.protocol.models import (
    ProvisionalReplyState,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
)


NUMERIC_REPLY_ACTION_STAGE = "numeric_reply_action"
NUMERIC_REPLY_ACTION_TIMEOUT_SECONDS = 4.0


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


@dataclass(frozen=True, slots=True)
class NumericReplyActionDecision:
    """Result of the optional completed-reply scoring pass."""

    action: dict[str, Any] | None
    scores: list[dict[str, Any]]
    elapsed_ms: float
    candidate_count: int
    selected_number: int | None
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class NumericReplyActionResolution:
    """Optional override and audit data returned to turn orchestration."""

    action: dict[str, Any] | None
    scores: list[dict[str, Any]] | None
    route_context: dict[str, Any]
    selection_context: dict[str, Any]
    timing_breakdown: dict[str, Any] | None


class NumericReplyActionComponent:
    async def _resolve_numeric_reply_action(
        self,
        turn: TurnBuffer,
        *,
        route: NumericReplyActionRoute,
        provisional_reply: ProvisionalReplyState,
        original_question: str,
        avatar_state: dict[str, Any] | None,
    ) -> NumericReplyActionResolution:
        """Wait for immutable text, score it, and preserve fallback on failure."""

        complete_reply, reply_status, reply_wait_ms = (
            await self._resolve_complete_provisional_reply(
                turn, provisional_reply
            )
        )
        candidate_count = len(route.candidates) + 1
        fallback_reason: str | None = None
        if reply_status != "completed" or complete_reply is None:
            fallback_reason = f"reply_{reply_status}"
        elif not complete_reply_may_contain_numeric_answer(complete_reply):
            fallback_reason = "no_possible_0_10_answer"
        if fallback_reason is not None:
            route_context = {
                "routed": True,
                "candidate_count": candidate_count,
                "reply_wait_ms": reply_wait_ms,
                "fallback_reason": fallback_reason,
            }
            emit_structured_log(
                "action",
                "numeric_reply_action_skipped",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                reason=fallback_reason,
                candidate_count=candidate_count,
                reply_wait_ms=reply_wait_ms,
            )
            return NumericReplyActionResolution(
                None, None, route_context, {}, None
            )

        emit_structured_log(
            "action",
            "numeric_reply_action_started",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            candidate_count=candidate_count,
            reply_wait_ms=reply_wait_ms,
        )
        try:
            decision = await asyncio.wait_for(
                self._score_numeric_reply_action(
                    turn,
                    original_question=original_question,
                    complete_reply=complete_reply,
                    avatar_state=avatar_state,
                    candidates=route.candidates,
                    request_base=turn.request_base,
                ),
                timeout=NUMERIC_REPLY_ACTION_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fallback_reason = (
                "timeout" if isinstance(exc, TimeoutError) else "scoring_error"
            )
            route_context = {
                "routed": True,
                "candidate_count": candidate_count,
                "reply_wait_ms": reply_wait_ms,
                "fallback_reason": fallback_reason,
            }
            emit_structured_log(
                "error",
                "numeric_reply_action_failed",
                level="warning",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                candidate_count=candidate_count,
                fallback_reason=fallback_reason,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return NumericReplyActionResolution(
                None, None, route_context, {}, None
            )

        route_context = {
            "routed": True,
            "candidate_count": decision.candidate_count,
            "reply_wait_ms": reply_wait_ms,
            "scoring_ms": decision.elapsed_ms,
            "selected_number": decision.selected_number,
            "fallback_reason": decision.fallback_reason,
        }
        selection_context: dict[str, Any] = {}
        timing_breakdown = None
        if decision.action is not None:
            selected = self.candidate_by_id[decision.action["candidate_id"]]
            definition = (
                selected.proactive_expression.strip()
                or selected.short_definition
            )
            selection_context = {
                "selection_basis": "complete_reply_numeric",
                "selected_category_id": decision.action.get("category_id"),
                "child_decision_id": decision.action.get("candidate_id"),
                "support_status": "supported",
                "fallback_applied": False,
                "selection_definition_source": (
                    "proactive_expression"
                    if selected.proactive_expression.strip()
                    else "short_definition_fallback"
                ),
                "selection_definition_hash": "sha256:"
                + hashlib.sha256(definition.encode("utf-8")).hexdigest(),
            }
            timing_breakdown = {
                "candidate_count": decision.candidate_count,
                "reply_wait_ms": reply_wait_ms,
                "compute_ms": decision.elapsed_ms,
            }
        emit_structured_log(
            "action",
            "numeric_reply_action_completed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            candidate_count=decision.candidate_count,
            selected_number=decision.selected_number,
            selected_candidate_id=(
                decision.action.get("candidate_id")
                if decision.action is not None
                else None
            ),
            fallback_reason=decision.fallback_reason,
            elapsed_ms=decision.elapsed_ms,
        )
        return NumericReplyActionResolution(
            decision.action,
            decision.scores,
            route_context,
            selection_context,
            timing_breakdown,
        )

    def _numeric_reply_action_system_prompt(
        self,
        candidates: tuple[NumericGestureCandidate, ...],
    ) -> str:
        lines = [
            self._action_prompt(
                zh=(
                    "你是完整回复驱动的数字手势识别器。根据原始用户问题和数字人已生成的"
                    "完整回复，只选择一个 candidate_id。"
                ),
                en=(
                    "You are a complete-reply numeric gesture classifier. Select "
                    "exactly one candidate_id from the original user question and "
                    "the character's completed reply."
                ),
            ),
            self._action_prompt(
                zh=(
                    "只有当完整回复的主要答案明确是一个可表示的 0 到 10 整数时，才选择"
                    "对应数字手势。算式包含多个数字时选择最终结果，例如 1+2=3 选择 3。"
                    "原始用户问题中的数字或数量只是题目输入，绝不能作为目标数字；目标"
                    "必须与完整回复的主要答案一致。例如回复明确说‘数字三’时只能选择"
                    "数字三，不能因为问题提到‘两个’而选择数字二。"
                    "日期、时间、编号、百分比、列表序号等附带数字必须选择 000；小数、"
                    "负数、超过 10、答案不明确或对应候选不存在也必须选择 000。"
                ),
                en=(
                    "Select a real gesture only when the completed reply's primary "
                    "answer is one unambiguous representable integer from 0 through "
                    "10. For an equation with several numbers, select its final result; "
                    "for example, 1+2=3 selects 3. Numbers or quantities in the original "
                    "user question are problem inputs, never the target number. The target "
                    "must match the completed reply's primary answer. For example, a reply "
                    "that explicitly says 'number three' must select three even if the "
                    "question mentions 'two'. Incidental dates, times, identifiers, "
                    "percentages, and list ordinals must select 000. Decimals, negative "
                    "numbers, values above 10, ambiguous answers, and missing matching "
                    "candidates must also select 000."
                ),
            ),
            self._action_prompt(
                zh=(
                    "候选说明用于让动作伴随数字人已说出的答案，因此真实动作使用"
                    " proactive_expression，而不是把用户原句当作动作指令。"
                ),
                en=(
                    "Candidate descriptions describe accompanying the character's "
                    "answer and therefore use proactive_expression, not the user's "
                    "utterance as an action command."
                ),
            ),
        ]
        for item in candidates:
            candidate = item.candidate
            definition = (
                candidate.proactive_expression.strip()
                or candidate.short_definition
            )
            lines.append(
                self._action_prompt(
                    zh=(
                        f"candidate_id={candidate.candidate_id}｜数字={item.value}｜"
                        f"动作={candidate.source_label}｜说明={definition}"
                    ),
                    en=(
                        f"candidate_id={candidate.candidate_id} | number={item.value} | "
                        f"action={candidate.source_label} | description={definition}"
                    ),
                )
            )
        lines.append(
            self._action_prompt(
                zh=(
                    f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID}｜决策=不执行数字答案手势。"
                    "只能输出以上一个 candidate_id，输出后立即结束，不要解释。"
                ),
                en=(
                    f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID} | decision=do not "
                    "perform a numeric-answer gesture. Output exactly one listed "
                    "candidate_id and stop without explanation."
                ),
            )
        )
        return "\n".join(lines)

    async def _score_numeric_reply_action(
        self,
        turn: TurnBuffer,
        *,
        original_question: str,
        complete_reply: str,
        avatar_state: dict[str, Any] | None,
        candidates: tuple[NumericGestureCandidate, ...],
        request_base: str,
    ) -> NumericReplyActionDecision:
        """Score only catalog-derived numeric gestures plus unsupported."""

        if not candidates:
            return NumericReplyActionDecision(
                None, [], 0.0, 1, None, "numeric_candidates_missing"
            )
        candidate_by_id = {
            item.candidate.candidate_id: item for item in candidates
        }
        system_prompt = self._numeric_reply_action_system_prompt(candidates)
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        effective_avatar_state = self._effective_avatar_state(
            avatar_state,
            turn_origin="user",
            has_avatar_image=False,
        )
        session_instruction = self._build_session_action_profile_instruction(
            "child",
            turn_origin="user",
            has_user_camera=False,
        )
        request = ActionSuffixScoreRequest(
            request_id=request_base + "-numeric-reply",
            logical_request_id=request_base,
            model=self.model_name,
            session_instruction=session_instruction,
            prefix=(
                self._action_prompt(
                    zh="根据以下完整问答判断是否执行数字手势。",
                    en=(
                        "Use the complete question and answer below to decide "
                        "whether to perform a numeric gesture."
                    ),
                )
                + self._state_description_priority_instruction(
                    "child",
                    enabled="state_description" in effective_avatar_state,
                )
            ),
            current_text=self._action_prompt(
                zh=(
                    "[原始用户问题，仅用于理解语境，不得从这里选择目标数字]\n"
                    f"{original_question}\n"
                    "[数字人的完整回复，主要答案是选择目标数字的唯一依据]\n"
                    f"{complete_reply}"
                ),
                en=(
                    "[Original user question: context only; never select the target "
                    "number from this section]\n"
                    f"{original_question}\n"
                    "[Character's completed reply: its primary answer is the sole "
                    "basis for the target number]\n"
                    f"{complete_reply}"
                ),
            ),
            output_prompt=self._action_prompt(
                zh="最合适的 candidate_id：",
                en="Best matching candidate_id:",
            ),
            system_prompt=system_prompt,
            language=self.action_language,
            candidates=[
                *(
                    ActionScoreCandidate(
                        candidate_id=item.candidate.candidate_id,
                        suffix=item.candidate.candidate_id,
                        action_id=item.candidate.action_id,
                        execution_binding=dict(item.candidate.execution_binding),
                    )
                    for item in candidates
                ),
                ActionScoreCandidate(
                    candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                    suffix=UNSUPPORTED_CHILD_SCORE_ID,
                    action_id=UNSUPPORTED_DECISION_ID,
                ),
            ],
            suffix_tokenization_mode="short_id",
            audios=[],
            images=[],
            image_roles=[],
            sample_rate=16000,
            micro_batch_size=min(self.action_micro_batch_size, len(candidates) + 1),
            prefix_cache_namespace=self._session_action_prefix_namespace(
                base_namespace=(
                    f"{self.action_prefix_cache_namespace}:numeric_reply:"
                    f"sha256:{prompt_hash}"
                ),
                stage=NUMERIC_REPLY_ACTION_STAGE,
                turn_origin="user",
                session_instruction=session_instruction,
            ),
            cache_static_system_only=not bool(session_instruction),
            admission_priority=20,
            session_id=self.session_id,
            turn_origin="user",
            # The request remains a user-origin turn for protocol/audit
            # consistency. The JSON field names distinguish the completed
            # assistant reply from the original user question.
            text_role="user_input",
            stage=NUMERIC_REPLY_ACTION_STAGE,
            avatar_state=effective_avatar_state,
        )
        started = time.perf_counter()
        result = await self._score_action_request(turn, request)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        ranked = sorted(
            result.scores,
            key=lambda item: item.mean_logprob,
            reverse=True,
        )
        if not ranked:
            raise ValueError("numeric reply action score returned no decision")
        top = ranked[0]
        scores: list[dict[str, Any]] = []
        for score in ranked:
            item = candidate_by_id.get(score.candidate_id)
            scores.append(
                {
                    "candidate_id": score.candidate_id,
                    "action_id": (
                        item.candidate.action_id
                        if item is not None
                        else UNSUPPORTED_DECISION_ID
                    ),
                    **(
                        {
                            "category_id": item.candidate.category_id,
                            "source_label": item.candidate.source_label,
                            "short_definition": item.candidate.short_definition,
                            "execution_binding": dict(
                                item.candidate.execution_binding
                            ),
                        }
                        if item is not None
                        else {"decision": "unsupported"}
                    ),
                    "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob,
                    "mean_nll": score.mean_nll,
                    "ppl": score.ppl,
                    "token_scores": [
                        {
                            "token_id": token.token_id,
                            "logprob": token.logprob,
                        }
                        for token in score.token_scores
                    ],
                }
            )
        if top.candidate_id == UNSUPPORTED_CHILD_SCORE_ID:
            return NumericReplyActionDecision(
                None,
                scores,
                elapsed_ms,
                len(request.candidates),
                None,
                "scorer_unsupported",
            )
        selected = candidate_by_id.get(top.candidate_id)
        if selected is None:
            raise ValueError("numeric reply action selected an unknown candidate")
        candidate = selected.candidate
        return NumericReplyActionDecision(
            {
                "candidate_id": candidate.candidate_id,
                "action_id": candidate.action_id,
                **(
                    {"category_id": candidate.category_id}
                    if candidate.category_id
                    else {}
                ),
                "execution_binding": dict(candidate.execution_binding),
                "execute": candidate.action_id != "no_action",
                "support_status": "supported",
                "fallback_applied": False,
                "mean_logprob": top.mean_logprob,
                "ppl": top.ppl,
                "token_count": top.token_count,
            },
            scores,
            elapsed_ms,
            len(request.candidates),
            selected.value,
            None,
        )


MultimodalNumericReplyActionMixin = NumericReplyActionComponent
