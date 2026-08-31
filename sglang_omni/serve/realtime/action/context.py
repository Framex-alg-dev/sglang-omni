"""Action context, state policy, and execution history.

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
from sglang_omni.serve.realtime.components import compose_components

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    """Preserve the established ``multimodal.emit_structured_log`` hook.

    Existing diagnostics tests and embedders patch the façade symbol. Resolve
    it lazily to keep that compatibility while the implementation lives here.
    """

    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.action.pipeline import ActionScoringPipeline


from sglang_omni.serve.realtime.action.prompts import ActionPromptComponent


@compose_components(ActionScoringPipeline, ActionPromptComponent)
class ActionPipeline:
    def _build_bounded_action_context(
        self,
        audios: list[str],
        images: list[Any],
        image_roles: list[str],
        *,
        include_history: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[str],
        list[Any],
        list[str],
        dict[str, Any],
    ]:
        """Build a bounded, current-turn-only action-scoring context."""
        if len(images) != len(image_roles):
            raise ValueError("images and image_roles must have the same length")
        if len(self.history_images) != len(self.history_image_roles):
            raise ValueError(
                "history_images and history_image_roles must have the same length"
            )
        # Proactive turns start with an assistant message, so role changes
        # cannot reliably identify turn boundaries.
        selected_turns = (
            self.history_turns[-MAX_ACTION_HISTORY_TURNS:] if include_history else []
        )
        selected_message_ids = {
            id(message) for turn in selected_turns for message in turn.messages
        }
        bounded_history: list[dict[str, Any]] = []
        bounded_history_audios: list[str] = []
        bounded_history_images: list[str] = []
        ignored_history_avatar_image_count = 0
        audio_index = 0
        image_index = 0

        for message in self.history if include_history else ():
            selected = id(message) in selected_message_ids
            content = message.get("content")
            if not selected:
                if isinstance(content, list):
                    audio_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "audio"
                    )
                    image_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "image"
                    )
                continue

            if not isinstance(content, list):
                bounded_history.append(dict(message))
                continue

            bounded_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    bounded_parts.append(part)
                    continue
                part_type = part.get("type")
                if part_type == "audio":
                    if audio_index < len(self.history_audios):
                        media = self.history_audios[audio_index]
                        if len(bounded_history_audios) < MAX_ACTION_HISTORY_AUDIOS:
                            bounded_history_audios.append(media)
                            bounded_parts.append({"type": "audio"})
                    audio_index += 1
                elif part_type == "image":
                    if image_index < len(self.history_images):
                        media = self.history_images[image_index]
                        image_role = self.history_image_roles[image_index]
                        if image_role == IMAGE_ROLE_AVATAR_STATE:
                            # A previous turn's avatar frame may be visually
                            # unrelated to the current rendered pose. Never
                            # expose it as evidence of current avatar state.
                            ignored_history_avatar_image_count += 1
                        elif len(bounded_history_images) < MAX_ACTION_HISTORY_IMAGES:
                            bounded_history_images.append(media)
                            bounded_parts.append(self._image_role_text_part(image_role))
                            bounded_parts.append({"type": "image"})
                    image_index += 1
                else:
                    bounded_parts.append(dict(part))
            if not bounded_parts:
                bounded_parts = [{"type": "text", "text": "（历史多媒体内容已裁剪）"}]
            bounded_history.append({**message, "content": bounded_parts})

        latest_avatar_index = next(
            (
                index
                for index in range(len(image_roles) - 1, -1, -1)
                if image_roles[index] == IMAGE_ROLE_AVATAR_STATE
            ),
            None,
        )
        eligible_indices = [
            index
            for index, role in enumerate(image_roles)
            if role != IMAGE_ROLE_AVATAR_STATE or index == latest_avatar_index
        ]
        selected_indices = eligible_indices[-MAX_ACTION_CURRENT_IMAGES:]
        if (
            latest_avatar_index is not None
            and latest_avatar_index not in selected_indices
        ):
            selected_indices = sorted(
                [
                    latest_avatar_index,
                    *selected_indices[-(MAX_ACTION_CURRENT_IMAGES - 1) :],
                ]
            )
        bounded_images = [images[index] for index in selected_indices]
        bounded_image_roles = [image_roles[index] for index in selected_indices]
        truncated = len(images) != len(bounded_images)
        context_summary = {
            "history_policy": "current_turn_only",
            "source_history_turn_count": len(self.history_turns),
            "history_turn_count": 0,
            "history_audio_count": 0,
            "history_image_count": 0,
            "cross_turn_history_omitted": bool(self.history_turns),
            "ignored_history_avatar_image_count": 0,
            "received_current_image_count": len(images),
            "scored_current_image_count": len(bounded_images),
            "truncated": truncated,
        }
        return (
            [],
            [],
            [],
            bounded_images,
            bounded_image_roles,
            context_summary,
        )


    def _model_action_history_record(
        self,
        *,
        candidate_id: str,
        action_id: str,
        category_id: str | None,
        source_label: str,
        short_definition: str,
        execute: bool,
        record_kind: Literal[
            "history",
            "current_physical",
            "last_user",
            "current_physical_and_last_user",
        ] = "history",
    ) -> str:
        execution_result = self._prompt(
            zh="已按执行处理" if execute else "未执行新动作（保持当前姿态）",
            en=(
                "treated as executed"
                if execute
                else "no new action executed (current pose retained)"
            ),
        )
        zh_labels = {
            "history": "历史动作记录",
            "current_physical": "当前实际动作状态",
            "last_user": "最近一次用户触发动作",
            "current_physical_and_last_user": "当前实际动作状态；同时是最近一次用户触发动作",
        }
        en_labels = {
            "history": "Historical action record",
            "current_physical": "Current physical action state",
            "last_user": "Most recent user-triggered action",
            "current_physical_and_last_user": (
                "Current physical action state; also the most recent user-triggered action"
            ),
        }
        zh_category = f"category_id={category_id}｜" if category_id else ""
        en_category = f"category_id={category_id} | " if category_id else ""
        return self._prompt(
            zh=(
                f"[{zh_labels[record_kind]}] "
                f"处理结果={execution_result}｜{zh_category}candidate_id={candidate_id}｜"
                f"action_id={action_id}｜动作={source_label}｜"
                f"说明={short_definition}。"
            ),
            en=(
                f"[{en_labels[record_kind]}] "
                f"result={execution_result} | {en_category}candidate_id={candidate_id} | "
                f"action_id={action_id} | action={source_label} | "
                f"description={short_definition}."
            ),
        )


    def _build_compact_action_history(self) -> list[dict[str, Any]]:
        """Return the small cross-turn context needed by action scoring.

        Category and Child deliberately share this exact history so the
        same-turn prepared-media cache remains valid.  Reply generation keeps
        its independent bounded conversation history.
        """

        latest_reply: str | None = None
        for history_turn in reversed(self.reply_history_turns):
            if not history_turn.model_visible:
                continue
            for message in reversed(history_turn.messages):
                content = message.get("content")
                if (
                    message.get("role") == "assistant"
                    and isinstance(content, str)
                    and content.strip()
                ):
                    latest_reply = content.strip()
                    break
            if latest_reply is not None:
                break

        parts: list[str] = []
        if latest_reply is not None:
            parts.append(
                self._prompt(
                    zh=f"[数字人最近一次回复] {latest_reply}",
                    en=f"[Digital character's most recent reply] {latest_reply}",
                )
            )

        physical_record = self.last_executed_action
        user_record = self.last_user_executed_action
        if physical_record is not None:
            record_kind: Literal[
                "current_physical", "current_physical_and_last_user"
            ] = (
                "current_physical_and_last_user"
                if user_record is not None
                and user_record.turn_id == physical_record.turn_id
                else "current_physical"
            )
            parts.append(
                self._model_action_history_record(
                    candidate_id=physical_record.candidate_id,
                    action_id=physical_record.action_id,
                    category_id=physical_record.category_id,
                    source_label=physical_record.source_label,
                    short_definition=physical_record.short_definition,
                    execute=physical_record.execute,
                    record_kind=record_kind,
                )
            )
        if user_record is not None and (
            physical_record is None or user_record.turn_id != physical_record.turn_id
        ):
            parts.append(
                self._model_action_history_record(
                    candidate_id=user_record.candidate_id,
                    action_id=user_record.action_id,
                    category_id=user_record.category_id,
                    source_label=user_record.source_label,
                    short_definition=user_record.short_definition,
                    execute=user_record.execute,
                    record_kind="last_user",
                )
            )

        if not parts:
            return []
        return [{"role": "assistant", "content": "\n".join(parts)}]


    def _last_user_action_reference_instruction(
        self,
        *,
        turn_origin: Literal["user", "proactive"],
    ) -> str:
        """Expose one user-action anchor without restoring action history."""
        record = self.last_user_executed_action
        if (
            turn_origin != TURN_ORIGIN_USER
            or record is None
            or not record.execute
        ):
            return ""
        return self._prompt(
            zh=(
                "[最近一次用户触发动作，仅用于指代解析]\n"
                f"category_id={record.category_id}｜"
                f"candidate_id={record.candidate_id}｜"
                f"action_id={record.action_id}｜动作={record.source_label}｜"
                f"说明={record.short_definition}。\n"
                "只有当本轮用户明确指代先前动作，例如要求执行“刚刚那个动作”、"
                "“上一个动作”、再次执行或重复先前动作时，才使用此记录解析目标。"
                "其他情况下必须忽略此记录，不得据此改变当前动作类别、重复动作或"
                "形成动作偏好。数字人自动触发的动作不属于此记录，也不得覆盖它。\n"
            ),
            en=(
                "[Most recent user-triggered action; for reference resolution only]\n"
                f"category_id={record.category_id} | "
                f"candidate_id={record.candidate_id} | "
                f"action_id={record.action_id} | action={record.source_label} | "
                f"description={record.short_definition}.\n"
                "Use this record only when the user explicitly refers to a prior action, "
                "such as asking for the action just performed, the previous action, or "
                "for a prior action to be performed again. Otherwise ignore it: it must "
                "not change the current action category, cause repetition, or become an "
                "action preference. Automatically triggered digital-character actions "
                "are not part of this record and must not overwrite it.\n"
            ),
        )


    def _build_turn_action_instruction(
        self,
        text: str | None,
        *,
        turn_origin: str,
        trigger: str | None,
        has_audio: bool = False,
        image_roles: list[str] | None = None,
        has_state_description: bool = False,
        avatar_state_source: Literal["image", "structured", "unknown"] | None = None,
    ) -> str:
        if self.language == "en":
            return self._build_turn_action_instruction_en(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                has_audio=has_audio,
                image_roles=image_roles,
                has_state_description=has_state_description,
                avatar_state_source=avatar_state_source,
            )
        resolved_image_roles = image_roles or []
        if avatar_state_source is None:
            avatar_state_source = (
                "image"
                if IMAGE_ROLE_AVATAR_STATE in resolved_image_roles
                else "unknown"
            )
        state_instruction = self._build_avatar_state_instruction(avatar_state_source)
        scene_constraint = (
            "候选必须满足本轮主动场景约束中给出的目标、指引、要求和禁止项。"
            if has_state_description
            else ""
        )
        if turn_origin == TURN_ORIGIN_PROACTIVE:
            trigger_text = f"主动触发原因：{trigger}。\n" if trigger is not None else ""
            if not isinstance(text, str) or not text.strip():
                return (
                    state_instruction
                    + trigger_text
                    + "根据主动触发原因、本次主动场景说明（如有）和当前媒体选择动作。"
                    + scene_constraint
                    + "选择与表达目标和状态约束最匹配的候选项。"
                )
            return (
                state_instruction
                + "本轮由数字人主动发起，且已提供数字人本轮将要说出的文本。"
                "该文本会在当前媒体之后以明确标签提供，不是用户输入或用户动作请求。\n"
                + trigger_text
                + "数字人本轮将要说出的文本，其语义、语气和表达目标是本轮核心约束。"
                + "候选动作必须与该文本的语义、语气和表达目标直接相关。"
                + scene_constraint
                + "选择与表达目标和状态约束最匹配的候选项。"
            )
        modalities: list[str] = []
        if isinstance(text, str) and text.strip():
            modalities.append("用户文本")
        if has_audio:
            modalities.append("用户音频")
        if image_roles:
            modalities.append("当前图片")
        input_summary = "、".join(modalities) if modalities else "未提供文本、音频或图片"
        return (
            state_instruction
            + f"本轮由用户输入触发；有效输入：{input_summary}。\n"
            + "根据用户的语言、语音语义或可观察行为，选择与输入和状态约束最匹配的候选项。"
        )


    def _build_turn_action_instruction_en(
        self,
        text: str | None,
        *,
        turn_origin: str,
        trigger: str | None,
        has_audio: bool,
        image_roles: list[str] | None,
        has_state_description: bool,
        avatar_state_source: Literal["image", "structured", "unknown"] | None,
    ) -> str:
        resolved_image_roles = image_roles or []
        if avatar_state_source is None:
            avatar_state_source = (
                "image"
                if IMAGE_ROLE_AVATAR_STATE in resolved_image_roles
                else "unknown"
            )
        state_instruction = self._build_avatar_state_instruction(avatar_state_source)
        scene_constraint = (
            "The candidate must satisfy every goal, instruction, requirement, and "
            "prohibition in the proactive-scene constraints for this interaction."
            if has_state_description
            else ""
        )
        if turn_origin == TURN_ORIGIN_PROACTIVE:
            trigger_text = (
                f"Proactive trigger reason: {trigger}.\n" if trigger is not None else ""
            )
            if not isinstance(text, str) or not text.strip():
                return (
                    state_instruction
                    + trigger_text
                    + "Select an action based on the proactive trigger, proactive-scene "
                    "description if supplied, and current media."
                    + scene_constraint
                    + "Select the candidate that best matches the expression goal and state "
                    "constraints."
                )
            return (
                state_instruction
                + "This interaction is initiated by the digital character, and the text "
                "that the character will say is supplied after the current media with an "
                "explicit label. It is not user input or a user action request.\n"
                + trigger_text
                + "The semantics, tone, and expression goal of that text are the primary "
                "constraints. The candidate action must be directly relevant to them."
                + scene_constraint
                + "Select the candidate that best matches the expression goal and state "
                "constraints."
            )
        modalities: list[str] = []
        if isinstance(text, str) and text.strip():
            modalities.append("user text")
        if has_audio:
            modalities.append("user audio")
        if resolved_image_roles:
            modalities.append("current images")
        input_summary = ", ".join(modalities) if modalities else "no text, audio, or image"
        return (
            state_instruction
            + f"This interaction is triggered by user input. Available input: {input_summary}.\n"
            + "Use the user's language, speech semantics, or observable behavior to "
            "select the candidate that best matches the input and state constraints."
        )


    def _state_description_priority_instruction(
        self,
        stage: Literal["category", "child", "single"],
        *,
        enabled: bool,
    ) -> str:
        """Keep turn-local proactive guidance above later dynamic constraints."""
        if not enabled:
            return ""
        if stage == "category":
            return self._prompt(
                zh=(
                    "[本轮主动场景约束优先级]\n"
                    "本轮已提供“本轮主动场景约束”，它是本轮动作类别选择的最高优先级依据，"
                    "高于数字人人设与动作偏好、主动触发原因、将要说出的文本、"
                    "系统伴随类别、执行兜底规则和其他通用选择规则。必须先满足其中明确的动作目标、要求和"
                    "禁止项；明确禁止的动作语义不得选择。系统伴随类别仅在不与该约束冲突，"
                    "且该约束没有给出更具体动作目标时使用。若该约束给出了明确动作目标，"
                    "应选择能够完成该目标的类别，不得仅因其他类别是系统伴随或执行兜底类别而改选它。\n"
                ),
                en=(
                    "[Priority of proactive-scene constraints for this interaction]\n"
                    "The proactive-scene constraints supplied for this interaction are "
                    "the highest-priority basis for category selection. They override the "
                    "character persona and action preferences, proactive trigger, text the "
                    "character will say, system accompaniment categories, execution "
                    "fallback rules, and other "
                    "general selection rules. First satisfy every explicit action goal, "
                    "requirement, and prohibition; do not select prohibited action semantics. "
                    "Use a system accompaniment category only when it does not conflict with these "
                    "constraints and no more specific action goal is given. When an explicit "
                    "action goal is given, select a category that can accomplish it rather "
                    "than preferring another category merely because it is a system "
                    "accompaniment or execution fallback category.\n"
                ),
            )
        if stage == "child":
            return self._prompt(
                zh=(
                    "[本轮主动场景约束优先级]\n"
                    "本轮已提供“本轮主动场景约束”，它是本轮具体动作选择的最高优先级依据，"
                    "高于数字人人设与动作偏好、主动触发原因、将要说出的文本、"
                    "系统伴随及执行兜底规则和其他通用选择规则。必须先满足其中明确的动作目标、要求和"
                    "禁止项；任何违反明确禁止项的 candidate_id 都不得选择，不能为了选择真实"
                    "动作或使用系统伴随、执行兜底动作而忽略该约束。\n"
                ),
                en=(
                    "[Priority of proactive-scene constraints for this interaction]\n"
                    "The proactive-scene constraints supplied for this interaction are "
                    "the highest-priority basis for concrete-action selection. They override "
                    "the character persona and action preferences, proactive trigger, text "
                    "the character will say, system-accompaniment and execution-fallback rules, and other "
                    "general selection rules. First satisfy every explicit action goal, "
                    "requirement, and prohibition. Never select a candidate_id that violates "
                    "an explicit prohibition merely to select a real action or use a system "
                    "accompaniment or execution fallback action.\n"
                ),
            )
        return self._prompt(
            zh=(
                "[本轮主动场景约束优先级]\n"
                "本轮已提供“本轮主动场景约束”，它是本轮动作选择的最高优先级依据。"
                "必须先满足其中明确的动作目标、要求和禁止项；任何违反明确禁止项的动作"
                "都不得选择，会话级偏好和通用规则不能覆盖该约束。\n"
            ),
            en=(
                "[Priority of proactive-scene constraints for this interaction]\n"
                "The proactive-scene constraints supplied for this interaction are the "
                "highest-priority basis for action selection. First satisfy every explicit "
                "action goal, requirement, and prohibition. Never select an action that "
                "violates an explicit prohibition; conversation-level preferences and general "
                "rules cannot override these constraints.\n"
            ),
        )


    @staticmethod
    def _explicit_prohibition_text(state_description: Any) -> str:
        """Return only clauses that explicitly prohibit an action semantic.

        ``state_description`` remains free-form client text.  We therefore do
        not attempt to turn every preference into a hard rule.  Only text
        following an explicit negative marker is eligible for deterministic
        candidate filtering; the complete description is still sent to the
        model for the broader semantic decision.
        """
        if not isinstance(state_description, str) or not state_description.strip():
            return ""
        markers = (
            "严禁",
            "禁止",
            "不得",
            "不要",
            "不能",
            "避免",
            "must not",
            "do not",
            "never",
            "avoid",
            "prohibit",
        )
        prohibited_parts: list[str] = []
        for sentence in re.split(r"[。.!?\n]+", state_description):
            lowered = sentence.lower()
            marker_positions = [
                lowered.find(marker) for marker in markers if lowered.find(marker) >= 0
            ]
            if marker_positions:
                prohibited_parts.append(lowered[min(marker_positions) :])
        return "\n".join(prohibited_parts)


    @staticmethod
    def _action_semantic_terms(*values: str) -> set[str]:
        stop_terms = {
            "动作",
            "姿态",
            "身体",
            "状态",
            "场景",
            "交互",
            "用户",
            "要求",
            "说明",
            "表达",
            "当前",
            "自然",
            "action",
            "motion",
            "pose",
            "body",
            "state",
            "current",
            "natural",
        }
        terms: set[str] = set()
        for value in values:
            for term in re.split(r"[\s,，、;；/|:：()（）]+", value.lower()):
                normalized = term.strip("-_.。.!?")
                if len(normalized) >= 2 and normalized not in stop_terms:
                    terms.add(normalized)
                # Chinese catalog labels commonly combine the requested verb
                # with execution detail (for example, “自然呼吸起伏”).  Keep
                # short CJK semantic fragments so an explicit prohibition of
                # “呼吸” or “喝水” can remove the concrete candidate as well.
                if re.search(r"[\u4e00-\u9fff]", normalized):
                    for width in range(2, min(4, len(normalized)) + 1):
                        for start in range(0, len(normalized) - width + 1):
                            fragment = normalized[start : start + width]
                            if fragment not in stop_terms:
                                terms.add(fragment)
        return terms


    @classmethod
    def _semantic_is_explicitly_prohibited(
        cls,
        prohibition_text: str,
        *semantic_values: str,
    ) -> bool:
        if not prohibition_text:
            return False
        compact_prohibition = re.sub(r"\s+", "", prohibition_text.lower())
        return any(
            re.sub(r"\s+", "", term) in compact_prohibition
            for term in cls._action_semantic_terms(*semantic_values)
        )


    def _state_description_excluded_category_ids(
        self,
        state_description: Any,
    ) -> tuple[str, ...]:
        prohibition_text = self._explicit_prohibition_text(state_description)
        return tuple(
            category.category_id
            for category in self.categories
            if self._semantic_is_explicitly_prohibited(
                prohibition_text,
                category.source_label,
                category.short_definition,
            )
        )


    def _state_description_excluded_candidate_ids(
        self,
        state_description: Any,
        candidates: list[SessionActionCandidate],
    ) -> tuple[str, ...]:
        prohibition_text = self._explicit_prohibition_text(state_description)
        return tuple(
            candidate.candidate_id
            for candidate in candidates
            if self._semantic_is_explicitly_prohibited(
                prohibition_text,
                candidate.source_label,
                candidate.short_definition,
            )
        )


    def _state_description_exclusion_instruction(
        self,
        *,
        category_ids: tuple[str, ...] = (),
        candidate_ids: tuple[str, ...] = (),
    ) -> str:
        if not category_ids and not candidate_ids:
            return ""
        separator = self._prompt(zh="、", en=", ")
        if category_ids:
            excluded = separator.join(category_ids)
            return self._prompt(
                zh=(
                    "[本轮明确禁止项的确定性过滤]\n"
                    f"以下 category_id 与本轮主动场景约束的明确禁止项直接冲突，"
                    f"已从本轮可选集合移除：{excluded}。不得输出这些 ID。\n"
                ),
                en=(
                    "[Deterministic filtering for explicit prohibitions]\n"
                    "The following category_id values directly conflict with an "
                    f"explicit prohibition and have been removed: {excluded}. Do not "
                    "output these IDs.\n"
                ),
            )
        excluded = separator.join(candidate_ids)
        return self._prompt(
            zh=(
                "[本轮明确禁止项的确定性过滤]\n"
                f"以下 candidate_id 与本轮主动场景约束的明确禁止项直接冲突，"
                f"已从本轮可选集合移除：{excluded}。不得输出这些 ID。\n"
            ),
            en=(
                "[Deterministic filtering for explicit prohibitions]\n"
                "The following candidate_id values directly conflict with an explicit "
                f"prohibition and have been removed: {excluded}. Do not output these "
                "IDs.\n"
            ),
        )


    def _build_session_action_profile_instruction(
        self,
        stage: Literal["category", "child", "single"],
    ) -> str:
        profile = self.action_profile
        if profile is None:
            return ""
        if self.language == "en":
            persona_labels = {
                "gender_expression": "gender expression",
                "visual_style": "visual style",
                "role": "occupation or role",
                "personality": "personality",
            }
            lines = [
                "[Digital character persona and action preferences for this conversation]"
            ]
            if profile.persona:
                lines.append(
                    "Digital character persona: "
                    + "; ".join(
                        f"{persona_labels[field_name]}={field_value}"
                        for field_name, field_value in profile.persona
                    )
                )
            if profile.visual_behavior_preferences:
                lines.append(
                    "Visual behavior preferences (action selection only; the "
                    "current user's explicit action request always takes "
                    "priority): " + profile.visual_behavior_preferences
                )
            if stage == "category":
                if profile.category_preferences:
                    lines.append(
                        "Category preferences (primary constraints for category "
                        "selection): " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append(
                        "Action preferences (feasibility constraints for category "
                        "selection): exclude categories that cannot satisfy these "
                        "preferences at all, but do not select a concrete action in "
                        "this stage; " + profile.action_preferences
                    )
            elif stage == "child":
                if profile.category_preferences:
                    lines.append(
                        "Category preferences (background constraints for concrete "
                        "action selection): do not rewrite or expand the selected "
                        "category; " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append(
                        "Action preferences (primary constraints for concrete action "
                        "selection): " + profile.action_preferences
                    )
            else:
                if profile.category_preferences:
                    lines.append(
                        "Category preferences: " + profile.category_preferences
                    )
                if profile.action_preferences:
                    lines.append("Action preferences: " + profile.action_preferences)
            lines.append(
                "The content above constrains action selection only for this "
                "conversation. It must not expand or rewrite the allowed category or "
                "action set. If it conflicts with proactive-scene constraints for this "
                "interaction, the proactive-scene constraints take precedence."
            )
            return "\n".join(lines) + "\n"
        persona_labels = {
            "gender_expression": "性别表达",
            "visual_style": "画风",
            "role": "职业或角色定位",
            "personality": "性格基调",
        }
        lines = ["[本次会话数字人人设与动作偏好]"]
        if profile.persona:
            lines.append(
                "数字人人设："
                + "；".join(
                    f"{persona_labels[field_name]}={field_value}"
                    for field_name, field_value in profile.persona
                )
            )
        if profile.visual_behavior_preferences:
            lines.append(
                "视觉行为偏好（仅用于动作选择，当前用户明确提出的动作请求"
                "始终优先）：" + profile.visual_behavior_preferences
            )
        if stage == "category":
            if profile.category_preferences:
                lines.append(
                    "类别偏好（动作类别选择的主要约束）："
                    + profile.category_preferences
                )
            if profile.action_preferences:
                lines.append(
                    "动作偏好（动作类别选择的可行性约束）："
                    "排除整体上无法满足该偏好的类别，但不要在本阶段选择具体动作；"
                    + profile.action_preferences
                )
        elif stage == "child":
            if profile.category_preferences:
                lines.append(
                    "类别偏好（具体动作选择的背景约束）：已选类别不得被改写或扩展；"
                    + profile.category_preferences
                )
            if profile.action_preferences:
                lines.append(
                    "动作偏好（具体动作选择的主要约束）：" + profile.action_preferences
                )
        else:
            if profile.category_preferences:
                lines.append("类目偏好：" + profile.category_preferences)
            if profile.action_preferences:
                lines.append("动作偏好：" + profile.action_preferences)
        lines.append(
            "以上内容仅约束本次会话的动作选择，不得扩展或改写允许选择的类别和动作范围；"
            "若与本轮主动场景约束冲突，以本轮主动场景约束为准。"
        )
        return "\n".join(lines) + "\n"


    def _session_candidate(
        self,
        candidate_id: str,
        *,
        category_id: str | None = None,
    ) -> SessionActionCandidate:
        if category_id is not None:
            for category in self.categories:
                if category.category_id != category_id:
                    continue
                candidate = next(
                    (
                        item
                        for item in category.children
                        if item.candidate_id == candidate_id
                    ),
                    None,
                )
                if candidate is not None:
                    return candidate
                break
        return self.candidate_by_id[candidate_id]


    def _record_action_as_executed(
        self,
        *,
        turn: TurnBuffer,
        action: dict[str, Any],
    ) -> None:
        """Persist a successful inference as an execution fact for later turns."""
        candidate_id = str(action["candidate_id"])
        candidate = self._session_candidate(
            candidate_id,
            category_id=str(action.get("category_id") or "") or None,
        )
        record = ExecutedActionRecord(
            turn_id=turn.turn_id,
            turn_origin=turn.turn_origin,
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            category_id=candidate.category_id,
            source_label=candidate.source_label,
            short_definition=candidate.short_definition,
            execute=bool(action.get("execute", True)),
        )
        self.last_executed_action = record
        if turn.turn_origin == TURN_ORIGIN_USER:
            self.last_user_executed_action = record
        self.executed_action_history.append(record)
        if len(self.executed_action_history) > MAX_EXECUTED_ACTION_HISTORY_TURNS:
            del self.executed_action_history[:-MAX_EXECUTED_ACTION_HISTORY_TURNS]
        log_fields = self._executed_action_log_fields(record)
        assert log_fields is not None
        emit_structured_log(
            "action",
            "action_execution_assumed",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            **log_fields,
            retained_execution_history_count=len(self.executed_action_history),
        )


    @staticmethod
    def _executed_action_log_fields(
        record: ExecutedActionRecord | None,
    ) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "source_turn_id": record.turn_id,
            "source_turn_origin": record.turn_origin,
            "candidate_id": record.candidate_id,
            "action_id": record.action_id,
            "category_id": record.category_id,
            "source_label": record.source_label,
            "execute": record.execute,
        }


    @staticmethod
    def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
        """Return only fields required by the external action executor."""
        compact = {
            "action_id": action["action_id"],
            "candidate_id": action["candidate_id"],
        }
        if action.get("category_id"):
            compact["category_id"] = action["category_id"]
        compact["execute"] = action["execute"]
        if action.get("support_status"):
            compact["support_status"] = action["support_status"]
        if "fallback_applied" in action:
            compact["fallback_applied"] = bool(action["fallback_applied"])
        execution_binding = action.get("execution_binding")
        if execution_binding:
            compact["execution_binding"] = dict(execution_binding)
        return compact


    def _append_action_history(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        *,
        turn_id: str,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        action: dict[str, Any],
        reply_text: str | None = None,
    ) -> None:
        if len(images) != len(image_roles):
            raise ValueError("images and image_roles must have the same length")
        retained_media = [
            (image, role)
            for image, role in zip(images, image_roles, strict=True)
            if role != IMAGE_ROLE_AVATAR_STATE
        ]
        retained_images = [image for image, _ in retained_media]
        retained_image_roles = [role for _, role in retained_media]
        candidate_id = str(action["candidate_id"])
        candidate = self._session_candidate(
            candidate_id,
            category_id=str(action.get("category_id") or "") or None,
        )
        action_id = str(action["action_id"])
        action_state = self._model_action_history_record(
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            category_id=candidate.category_id,
            source_label=candidate.source_label,
            short_definition=candidate.short_definition,
            execute=action_id != "no_action",
        )

        if turn_origin == TURN_ORIGIN_USER:
            assistant_content = (
                f"{reply_text}\n{action_state}" if reply_text else action_state
            )
            messages = [
                {
                    "role": "user",
                    "content": self._current_user_content(
                        audios, retained_images, text
                    ),
                },
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": self._current_character_content(
                        audios,
                        retained_images,
                        reply_text or text,
                        action_state,
                    ),
                }
            ]

        history_turn = ActionHistoryTurn(
            turn_id=turn_id,
            turn_origin=turn_origin,
            text_role=text_role,
            messages=messages,
            audios=list(audios),
            images=list(retained_images),
        )
        self.history_turns.append(history_turn)
        self.history.extend(messages)
        self.history_audios.extend(audios)
        self.history_images.extend(retained_images)
        self.history_image_roles.extend(retained_image_roles)


    def _current_user_content(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
    ) -> Any:
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        if not parts:
            parts.append(
                {
                    "type": "text",
                    "text": self._prompt(
                        zh="本轮没有文本输入，请根据当前会话内容选择动作。",
                        en=(
                            "No text input is provided in this interaction. Select "
                            "an action from the current conversation context."
                        ),
                    ),
                }
            )
        return parts
    @staticmethod
    def _current_character_content(
        audios: list[str],
        images: list[str],
        text: str | None,
        action_state: str,
    ) -> Any:
        normalized_text = (
            text.strip() if isinstance(text, str) and text.strip() else None
        )
        if not audios and not images:
            return (
                f"{normalized_text}\n{action_state}"
                if normalized_text is not None
                else action_state
            )
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if normalized_text is not None:
            parts.append({"type": "text", "text": normalized_text})
        parts.append({"type": "text", "text": action_state})
        return parts


MultimodalActionMixin = ActionPipeline

