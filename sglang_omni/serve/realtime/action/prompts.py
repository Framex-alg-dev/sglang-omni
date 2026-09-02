"""Action candidate helpers and localized scoring prompts.

This stateless component isolates category recall, child selection, action
prompts, and action-history policy from protocol and reply orchestration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from typing import Any, Callable, Literal

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_CONTEXT_POLICY,
    CATEGORY_CONTEXT_POLICY_EN,
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    DIRECTION_REFERENCE_POLICY,
    DIRECTION_REFERENCE_POLICY_EN,
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
    ActionHistoryTurn,
    ExecutedActionRecord,
    ProvisionalReplyState,
    SessionActionCandidate,
    SessionActionCategory,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    """Preserve the established ``multimodal.emit_structured_log`` hook.

    Existing diagnostics tests and embedders patch the façade symbol. Resolve
    it lazily to keep that compatibility while the implementation lives here.
    """

    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)




class ActionPromptComponent:
    def _no_action_candidate(self) -> SessionActionCandidate:
        for candidate in self.candidates:
            if candidate.action_id == "no_action":
                return candidate
        raise ValueError("session has no no_action candidate")


    def _fallback_categories(self) -> list[SessionActionCategory]:
        category_by_id = {
            category.category_id: category for category in self.categories
        }
        categories = [
            category_by_id[category_id]
            for category_id in self.fallback_category_ids
            if category_id in category_by_id
        ]
        if len(categories) != len(self.fallback_category_ids) or not categories:
            raise ValueError(
                "session has no valid fallback categories with executable actions"
            )
        return categories


    def _category_with_semantic_tag(
        self, semantic_tag: str
    ) -> SessionActionCategory | None:
        if self.global_action_catalog is None:
            return None
        global_category = self.global_action_catalog.category_with_semantic_tag(
            semantic_tag
        )
        if global_category is None:
            return None
        return next(
            (
                category
                for category in self.categories
                if category.category_id == global_category.category_id
            ),
            None,
        )


    def _category_has_semantic_tag(
        self, category: SessionActionCategory, semantic_tag: str
    ) -> bool:
        if self.global_action_catalog is None:
            return False
        global_category = self.global_action_catalog.category_by_id.get(
            category.category_id
        )
        return bool(
            global_category is not None
            and semantic_tag in global_category.semantic_tags
        )


    def _is_system_accompaniment_category(
        self, category: SessionActionCategory
    ) -> bool:
        return self._category_has_semantic_tag(
            category, CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
        ) or self._category_has_semantic_tag(
            category, CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        )


    def _forced_trigger_category(
        self,
        *,
        turn_origin: str,
        trigger: str | None,
    ) -> tuple[SessionActionCategory | None, str | None]:
        """Resolve protocol-owned trigger routes without model classification."""
        if (
            turn_origin != TURN_ORIGIN_PROACTIVE
            or trigger != ACTION_FINISHED_TRIGGER
        ):
            return None, None
        semantic_tag = CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        category = self._category_with_semantic_tag(semantic_tag)
        if category is None:
            # Global protocol sessions validate this invariant at session.start.
            # Legacy/internal action-only sessions may not expose semantic tags.
            return None, None
        return category, semantic_tag


    def _primary_fallback_category(self) -> SessionActionCategory:
        """Return the highest-priority fallback category configured by the client."""
        return self._fallback_categories()[0]


    def _default_fallback_candidate(self) -> SessionActionCandidate:
        """Return the stable executable action from the primary fallback category."""
        return self._primary_fallback_category().children[0]


    @staticmethod
    def _turn_candidate_is_allowed(
        turn: TurnBuffer,
        candidate: SessionActionCandidate,
    ) -> bool:
        allowed = set(turn.action_allowed_candidate_ids)
        excluded = set(turn.action_excluded_candidate_ids)
        return (
            (not allowed or candidate.candidate_id in allowed)
            and candidate.candidate_id not in excluded
        )


    def _filter_turn_action_candidates(
        self,
        turn: TurnBuffer,
        candidates: list[SessionActionCandidate],
    ) -> list[SessionActionCandidate]:
        return [
            candidate
            for candidate in candidates
            if self._turn_candidate_is_allowed(turn, candidate)
        ]


    def _default_fallback_candidate_for_turn(
        self,
        turn: TurnBuffer,
    ) -> SessionActionCandidate:
        fallback = self._default_fallback_candidate()
        if self._turn_candidate_is_allowed(turn, fallback):
            return fallback
        eligible = self._filter_turn_action_candidates(turn, list(self.candidates))
        if not eligible:
            raise ValueError(
                "per-turn action candidate constraints leave no executable action"
            )
        return eligible[0]


    def _no_action_candidate_id(self) -> str:
        return self._no_action_candidate().candidate_id


    def _format_candidate_for_prompt(self, candidate: SessionActionCandidate) -> str:
        return self._prompt(
            zh=(
                f"candidate_id={candidate.candidate_id}｜动作={candidate.source_label}｜"
                f"说明={candidate.short_definition}"
            ),
            en=(
                f"candidate_id={candidate.candidate_id} | action={candidate.source_label} | "
                f"description={candidate.short_definition}"
            ),
        )


    def _child_candidates_for_categories(
        self, categories: list[SessionActionCategory]
    ) -> list[SessionActionCandidate]:
        # A concrete action may intentionally belong to multiple semantic
        # categories. Score it only once while preserving the first occurrence,
        # which follows category rank and therefore supplies the result's
        # category_id.
        candidates_by_id: dict[str, SessionActionCandidate] = {}
        for category in categories:
            for child in category.children:
                candidates_by_id.setdefault(child.candidate_id, child)
        return list(candidates_by_id.values())


    def _build_category_system_prompt(self) -> str:
        if self.global_action_catalog is not None:
            return self.global_action_catalog.category_system_prompt_for(self.locale)
        if self.language == "en":
            lines = [
                "You are a digital-character action category classifier. Select one category_id from the fixed category set.",
                CATEGORY_CONTEXT_POLICY_EN,
            ]
            no_action_categories = [
                item.category_id
                for item in self.categories
                if any(child.action_id == "no_action" for child in item.children)
            ]
            if no_action_categories:
                lines.append(
                    "If no candidate action satisfies the input and state constraints, "
                    "use one of these default category_id values: "
                    + ", ".join(no_action_categories)
                    + "."
                )
            lines.append("Fixed category set:")
            lines.extend(
                f"category_id={item.category_id} | category={item.source_label} | description={item.short_definition}"
                for item in self.categories
            )
            lines.append(
                "Select the category_id that best matches the current input. Output "
                "exactly one category_id and stop immediately. Do not explain."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作类别识别器。请从固定类别集合中选择一个 category_id。",
            CATEGORY_CONTEXT_POLICY,
        ]
        no_action_categories = [
            item.category_id
            for item in self.categories
            if any(child.action_id == "no_action" for child in item.children)
        ]
        if no_action_categories:
            lines.append(
                "没有候选动作满足输入与状态约束时，可使用兜底 category_id="
                + ",".join(no_action_categories)
                + "。"
            )
        lines.append("固定类别集合如下：")
        # Category descriptions are opaque external metadata. Do not compress,
        # deduplicate, or rewrite them here; callers may optimize their wording
        # before session.start and the exact rendered catalog participates in
        # the catalog hash/prefix-cache identity.
        for item in self.categories:
            lines.append(
                f"category_id={item.category_id}｜类别={item.source_label}｜"
                f"说明={item.short_definition}"
            )
        lines.append(
            "请根据当前输入选择最匹配的 category_id；只输出一个 category_id，"
            "输出后立即结束，不要解释。"
        )
        return "\n".join(lines)
    def _build_child_system_prompt(
        self,
        category: SessionActionCategory | list[SessionActionCategory],
        candidates: list[SessionActionCandidate],
    ) -> str:
        categories = category if isinstance(category, list) else [category]
        if self.global_action_catalog is not None:
            if len(categories) == 1:
                return self.global_action_catalog.child_system_prompt_for(
                    self.locale, categories[0].category_id
                )
            lines = [
                self._prompt(
                    zh="你是数字人动作识别器。请从以下集合中选择一个 candidate_id。",
                    en="You are a digital-character action classifier. Select one candidate_id from the following set.",
                ),
                self._prompt(
                    zh=DIRECTION_REFERENCE_POLICY,
                    en=DIRECTION_REFERENCE_POLICY_EN,
                ),
            ]
            for selected in categories:
                lines.append(
                    self._prompt(
                        zh=(
                            f"候选类别：category_id={selected.category_id}｜"
                            f"类别={selected.source_label}｜说明={selected.short_definition}"
                        ),
                        en=(
                            f"Candidate category: category_id={selected.category_id} | "
                            f"category={selected.source_label} | description={selected.short_definition}"
                        ),
                    )
                )
            lines.append(child_unsupported_policy(self.locale))
            lines.extend(self._format_candidate_for_prompt(item) for item in candidates)
            lines.append(
                self._prompt(
                    zh=(
                        "只能从以上候选类别的动作中选择最匹配的 candidate_id；"
                        "只输出一个结果。"
                    ),
                    en=(
                        "Select the best matching candidate_id from the candidate "
                        "categories above. Output exactly one result."
                    ),
                )
            )
            return "\n".join(lines)
        if self.language == "en":
            lines = [
                "You are a digital-character action classifier. Select one candidate_id from the following set.",
            ]
            lines.extend(
                f"Selected category: category_id={selected.category_id} | category={selected.source_label} | description={selected.short_definition}"
                for selected in categories
            )
            lines.extend(self._format_candidate_for_prompt(item) for item in candidates)
            lines.append(
                "Select the candidate_id that best matches from the candidates in the "
                "selected category above. Do not introduce another category or an "
                "extra default action."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作识别器。请从以下集合中选择一个 candidate_id。",
        ]
        for selected in categories:
            lines.append(
                f"已选类别：category_id={selected.category_id}｜类别={selected.source_label}｜"
                f"说明={selected.short_definition}"
            )
        lines.extend(self._format_candidate_for_prompt(item) for item in candidates)
        lines.append(
            "只能从以上已选类别的候选动作中选择最匹配的 candidate_id；"
            "不得引入其他类别或系统兜底动作。"
        )
        return "\n".join(lines)


    def _category_whitelist_instruction(self) -> str:
        if self.global_action_catalog is None:
            return ""
        separator = self._prompt(zh="、", en=", ")
        allowed_ids = separator.join(
            category.category_id for category in self.categories
        )
        fallback_items = separator.join(
            self._prompt(
                zh=f"{category.category_id}（{category.source_label}）",
                en=f"{category.category_id} ({category.source_label})",
            )
            for category in self._fallback_categories()
        )
        reply_category = self._category_with_semantic_tag(
            CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
        )
        silent_category = self._category_with_semantic_tag(
            CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        )
        system_route = self._prompt(
            zh=(
                "[本次会话系统伴随类别]\n"
                f"有非空实际回复时：{reply_category.category_id}（{reply_category.source_label}）\n"
                f"无回复、空回复或回复失败时：{silent_category.category_id}（{silent_category.source_label}）\n"
            ),
            en=(
                "[System accompaniment categories for this conversation]\n"
                f"Non-empty actual reply: {reply_category.category_id} "
                f"({reply_category.source_label})\n"
                f"No reply, empty reply, or reply failure: {silent_category.category_id} "
                f"({silent_category.source_label})\n"
            ),
        ) if reply_category is not None and silent_category is not None else ""
        if self.language == "en":
            return (
                "[Action categories allowed in this conversation]\n"
                f"Allowed real category_id values: {allowed_ids}\n"
                f"Select only one of these category_id values, or select {UNSUPPORTED_CATEGORY_SCORE_ID} "
                "under its defined conditions. Other categories in the fixed set are "
                "not available in this conversation.\n"
                + system_route
                + "[Execution fallback categories for this conversation]\n"
                f"In descending priority: {fallback_items}\n"
                "This list defines executable fallback order after an unsupported decision; "
                "it does not route ordinary dialogue. Silent observation, low-disturbance "
                "situations, and natural idle behavior use the silent accompaniment category. "
                "When an explicit action's semantic category is unavailable, return "
                f"{UNSUPPORTED_CATEGORY_SCORE_ID}; never report a fallback category as support.\n"
            )
        return (
            "[本次会话允许选择的动作类别]\n"
            f"可用的真实 category_id：{allowed_ids}\n"
            f"只能选择以上 category_id，或按既定条件选择 {UNSUPPORTED_CATEGORY_SCORE_ID}；"
            "固定类别集合中的其他类别在本次会话中不可用。\n"
            + system_route
            + "[本次会话执行兜底类别]\n"
            f"按优先级从高到低为：{fallback_items}\n"
            "该列表只定义不支持判定后的可执行兜底顺序，不用于路由普通对话。静默观察、"
            "低打扰或自然待机应使用静默低扰伴随类别。当前输入明确要求动作，但该动作的"
            f"语义类别不在本次会话允许范围内时，必须返回 {UNSUPPORTED_CATEGORY_SCORE_ID}，"
            "不得把执行兜底类别当作已支持该请求的替代类别。\n"
        )


    def _child_whitelist_instruction(
        self,
        category: SessionActionCategory | list[SessionActionCategory],
        candidates: list[SessionActionCandidate],
    ) -> str:
        if self.global_action_catalog is None:
            return ""
        allowed_ids = self._prompt(zh="、", en=", ").join(
            item.candidate_id for item in candidates
        )
        categories = category if isinstance(category, list) else [category]
        category_ids = self._prompt(zh="、", en=", ").join(
            item.category_id for item in categories
        )
        if len(categories) == 1 and self._is_system_accompaniment_category(categories[0]):
            return self._prompt(
                zh=(
                    "[本次会话允许选择的具体动作]\n"
                    f"已选 category_id={category_ids}。"
                    f"只允许从以下真实 candidate_id 中选择：{allowed_ids}\n"
                    "不得选择该类别中未列出的其他动作。\n"
                ),
                en=(
                    "[Concrete actions allowed in this conversation]\n"
                    f"Selected category_id={category_ids}. Select only one of "
                    f"these real candidate_id values: {allowed_ids}\n"
                    "Do not select another action from this category that is not "
                    "listed above.\n"
                ),
            )
        if self.language == "en":
            return (
                "[Concrete actions allowed in this conversation]\n"
                f"Selected category_id values: {category_ids}. Only these candidate_id "
                f"values may be selected: {allowed_ids}\n"
                + f"You may also return {UNSUPPORTED_CHILD_SCORE_ID}; it indicates that "
                "the concrete action is unsupported and is not executable. Even when the "
                "selected category is also listed as an execution fallback category, an explicit action request "
                f"that none of the real candidates can fulfill must return {UNSUPPORTED_CHILD_SCORE_ID}. "
                "When there is no explicit action request, select an appropriate real "
                "candidate_id instead.\n"
                + "Other actions in this category that are not listed above cannot be "
                "selected in this conversation.\n"
            )
        return (
            "[本次会话允许选择的具体动作]\n"
            f"已选 category_id：{category_ids}。"
            f"只允许从以下 candidate_id 中选择：{allowed_ids}\n"
            + f"此外可以返回 {UNSUPPORTED_CHILD_SCORE_ID}；它只表示具体动作不支持，"
            "不是可执行动作。即使当前类别也被列为执行兜底类别，只要当前输入明确要求动作，且"
            f"真实候选都无法完成该请求，也必须返回 {UNSUPPORTED_CHILD_SCORE_ID}。当前输入"
            "没有明确动作请求时，应选择合适的真实 candidate_id。\n"
            + "该类别中未列出的其他动作在本次会话中不可选择。\n"
        )


    def _system_accompaniment_child_instruction(
        self,
        category: SessionActionCategory,
        *,
        reply_prefix: str,
    ) -> str:
        if self._category_has_semantic_tag(
            category, CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
        ):
            return self._prompt(
                zh=(
                    "[本轮数字人实际回复开头]\n"
                    f"{reply_prefix}\n"
                    "以上文本是数字人本轮实际将说出的回复开头。以其主要表达功能作为具体"
                    "动作选择依据；用户输入只用于理解回复语境。\n"
                ),
                en=(
                    "[Beginning of the digital character's actual reply]\n"
                    f"{reply_prefix}\n"
                    "This is the beginning of the character's actual reply for this "
                    "interaction. Use its primary communicative function to select the "
                    "concrete action; use the user input only as reply context.\n"
                ),
            )
        if self._category_has_semantic_tag(
            category, CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
        ):
            return self._prompt(
                zh=(
                    "[本轮回复状态]\n"
                    "本轮没有需要数字人说出的有效回复文本。根据当前状态、场景和低打扰"
                    "要求选择静默伴随动作。\n"
                ),
                en=(
                    "[Reply state for this interaction]\n"
                    "There is no effective reply text for the character to say. Select a "
                    "silent accompanying action from the current state, scene, and "
                    "low-disturbance requirements.\n"
                ),
            )
        return ""


    def _build_action_system_prompt(self) -> str:
        if self.language == "en":
            lines = [
                "You are a digital-character action classifier. Select one candidate_id from the fixed set for this conversation.",
            ]
            lines.extend(
                self._format_candidate_for_prompt(item) for item in self.candidates
            )
            lines.append(
                "If no candidate satisfies the input and state constraints, or a "
                "conflict must be avoided, select the "
                f"default candidate_id={self._no_action_candidate_id()}."
            )
            return "\n".join(lines)
        lines = [
            "你是数字人动作识别器。请从本次会话的固定集合中选择一个 candidate_id。",
        ]
        lines.extend(
            self._format_candidate_for_prompt(item) for item in self.candidates
        )
        lines.append(
            "没有候选动作满足输入与状态约束，或需要避免冲突时，选择兜底 "
            f"candidate_id={self._no_action_candidate_id()}。"
        )
        return "\n".join(lines)


MultimodalActionPromptMixin = ActionPromptComponent
