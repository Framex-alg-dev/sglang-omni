"""Parallel facial-expression scope selection and internal TTS control."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.serve.realtime.performance.models import PerformanceDecision
from sglang_omni.serve.realtime.protocol.common import (
    FACIAL_EXPRESSION_CATEGORY_ID,
)
from sglang_omni.serve.realtime.protocol.models import TurnBuffer
from sglang_omni.utils.structured_logs import emit_structured_log

_FACE_ONLY_DESCRIPTIONS = {
    "A154": "a natural smile with gently raised mouth corners and cheerful eyes",
    "A155": "an open joyful laugh with a wide mouth and strongly smiling eyes",
    "A156": "a serious expression with tightened brows and a straight closed mouth",
    "A157": "a surprised expression with raised brows, widened eyes, and a slightly open mouth",
    "A158": "a playful funny face with exaggerated features, a crooked mouth, and the tongue out",
    "A159": "a frightened expression with wide eyes, raised brows, and lowered mouth corners",
    "A160": "an aggrieved expression with lowered mouth corners and slightly knitted brows",
    "A161": "a sad expression with drooping eyes and brows, lowered mouth corners, and a subdued gaze",
    "A162": "a puzzled expression with one raised brow and a slightly crooked mouth",
    "A163": "an angry expression with deeply knitted brows, flared nostrils, and a tense mouth",
    "A164": "a vulnerable pleading expression with a gently bitten lower lip and upward-looking eyes",
}

_TTS_BY_EXPRESSION = {
    "A154": "语气轻快温暖，音调略微上扬，语速适中，带有自然笑意",
    "A155": "语气开朗活泼，音调偏高，语速略快，笑意明显但保持吐字清楚",
    "A156": "语气克制严肃，音调平稳偏低，语速稍慢，吐字清晰",
    "A157": "语气惊讶，音调明显上扬，语速略快，保留自然停顿",
    "A158": "语气俏皮活泼，音调偏高，语速略快，节奏轻盈",
    "A159": "语气紧张不安，音调略高，语速稍快，声音力度偏弱",
    "A160": "语气轻柔委屈，音调略低，语速偏慢，声音力度较轻",
    "A161": "语气低落悲伤，音调偏低，语速缓慢，声音轻柔",
    "A162": "语气带有疑问，句尾自然上扬，语速适中，停顿清楚",
    "A163": "语气坚定不满，音调偏低，语速适中，力度稍强但不喊叫",
    "A164": "语气柔软可怜，音调略高，语速偏慢，声音轻柔",
}

_TTS_BY_EXPRESSION_EN = {
    "A154": "warm and upbeat tone, slightly rising pitch, medium pace, with a natural smile",
    "A155": "bright and lively tone, higher pitch, slightly faster pace, clear articulation",
    "A156": "restrained and serious tone, steady lower pitch, slightly slower pace, clear articulation",
    "A157": "surprised tone, clearly rising pitch, slightly faster pace, with natural pauses",
    "A158": "playful lively tone, higher pitch, slightly faster pace, light rhythm",
    "A159": "nervous uneasy tone, slightly higher pitch and faster pace, with low vocal force",
    "A160": "soft aggrieved tone, slightly lower pitch, slower pace, and gentle vocal force",
    "A161": "subdued sad tone, lower pitch, slow pace, and a soft voice",
    "A162": "questioning tone, naturally rising sentence endings, medium pace, and clear pauses",
    "A163": "firm displeased tone, lower pitch, medium pace, stronger force without shouting",
    "A164": "soft vulnerable tone, slightly higher pitch, slower pace, and gentle vocal force",
}


@dataclass(frozen=True, slots=True)
class _Choice:
    scope: str
    expression_id: str | None
    expression_unsupported: bool = False


class PerformancePipeline:
    """Runs beside reply routing and body-action scoring."""

    def _expression_candidates(self) -> list[Any]:
        category = next(
            (
                item
                for item in self.categories
                if item.category_id == FACIAL_EXPRESSION_CATEGORY_ID
            ),
            None,
        )
        if category is None:
            return []
        return [
            item
            for item in category.children
            if item.candidate_id in _FACE_ONLY_DESCRIPTIONS
        ]

    def _performance_system_prompt(self, choices: dict[str, _Choice]) -> str:
        choice_lines: list[str] = []
        labels = {
            item.candidate_id: item.source_label
            for item in self._expression_candidates()
        }
        for decision_id, choice in choices.items():
            expression = (
                "不改变脸部表情"
                if choice.expression_id is None
                else labels.get(choice.expression_id, "不支持的脸部表情")
            )
            choice_lines.append(
                f"{decision_id}: request_scope={choice.scope}; expression={expression}"
            )
        mapping = "\n".join(choice_lines)
        profile = getattr(self, "action_profile", None)
        persona_lines: list[str] = []
        if profile is not None:
            persona_lines = [
                f"{field_name}: {value}"
                for field_name, value in profile.persona
            ]
            if profile.visual_behavior_preferences:
                persona_lines.append(
                    "visual_behavior_preferences: "
                    + profile.visual_behavior_preferences
                )
        persona_body = "\n".join(persona_lines)
        persona_context_zh = (
            "\n当前角色信息：\n" + persona_body if persona_body else ""
        )
        persona_context_en = (
            "\nCurrent character information:\n" + persona_body
            if persona_body
            else ""
        )
        return self._prompt(
            zh=(
                "你只负责判断当前这条消息要求的可视执行通道，并为当前角色选择脸部表情。"
                "expression_only 表示用户只明确要求眉眼、口部或面颊构成的脸部表情；"
                "body_only 表示只明确要求头部、视线、肩部、手臂、手部、躯干、姿态或物品交互；"
                "both 表示同一请求明确要求脸部表情和身体动作；none 表示没有明确动作请求。"
                "‘笑一个’‘做个惊讶表情’属于 expression_only；‘挥挥手’‘转一圈’属于 body_only；"
                "‘笑着挥挥手’‘抱头并露出惊讶表情’属于 both。"
                "不得根据用户外观猜测用户情绪。普通语言交流可以选择与角色将要表达的语气协调的"
                "可选表情，也可以不改变表情。明确要求不在候选范围内的纯脸部表情选择不支持。"
                "角色信息只用于选择符合角色表达方式的可选表情，不得覆盖用户明确提出的表情请求。"
                "只能输出下列一个 decision_id，不要回答用户，也不要输出解释。\n"
                f"{mapping}{persona_context_zh}"
            ),
            en=(
                "Classify the execution channels explicitly requested by the current message "
                "and select one facial expression for the character. expression_only means "
                "only a facial expression made with eyes, brows, mouth, or cheeks is explicitly "
                "requested. body_only means only head, gaze, shoulders, arms, hands, torso, pose, "
                "or object interaction is requested. both means both layers are explicitly "
                "requested. none means no explicit action request. Do not infer the user's "
                "emotion from appearance. Ordinary speech may use a matching optional expression "
                "or no change. Character information may shape only optional expression style and "
                "must not override an explicit expression request. Output exactly one decision_id "
                "from this list and no explanation.\n"
                f"{mapping}{persona_context_en}"
            ),
        )

    @staticmethod
    def _choices(expression_ids: list[str]) -> dict[str, _Choice]:
        choices: dict[str, _Choice] = {
            "P000": _Choice("none", None),
            "P200": _Choice("body_only", None),
            "P199": _Choice("expression_only", None, True),
            "P399": _Choice("both", None, True),
        }
        for index, expression_id in enumerate(expression_ids, start=1):
            suffix = f"{index:02d}"
            choices[f"P0{suffix}"] = _Choice("none", expression_id)
            choices[f"P1{suffix}"] = _Choice("expression_only", expression_id)
            choices[f"P2{suffix}"] = _Choice("body_only", expression_id)
            choices[f"P3{suffix}"] = _Choice("both", expression_id)
        return choices

    def _tts_instruction(self, expression_id: str | None) -> str:
        if self.language == "en":
            turn_instruction = _TTS_BY_EXPRESSION_EN.get(
                expression_id,
                "natural calm pitch, medium pace, clear articulation, and emotion matching the reply",
            )
        else:
            turn_instruction = _TTS_BY_EXPRESSION.get(
                expression_id,
                "语调自然平和，语速适中，吐字清楚，情绪与当前回复内容一致",
            )
        return turn_instruction

    def _default_performance_decision(self) -> PerformanceDecision:
        return PerformanceDecision(
            request_scope="none",
            expression=None,
            expression_unsupported=False,
            tts_instruction=self._tts_instruction(None),
            elapsed_ms=0.0,
        )

    async def _infer_turn_performance(
        self,
        turn: TurnBuffer,
        audios: list[str],
        *,
        current_text: str | None,
    ) -> PerformanceDecision:
        started = time.perf_counter()
        expressions = self._expression_candidates()
        expression_by_id = {item.candidate_id: item for item in expressions}
        choices = self._choices(list(expression_by_id))
        system_prompt = self._performance_system_prompt(choices)
        request = ActionSuffixScoreRequest(
            request_id=f"{turn.request_base}-performance",
            model=self.model_name,
            prefix=self._prompt(zh="表现控制结果：", en="Performance control result:"),
            current_text=(current_text or "").strip(),
            output_prompt="",
            system_prompt=system_prompt,
            language=self.language,
            candidates=[
                ActionScoreCandidate(
                    candidate_id=decision_id,
                    suffix=decision_id,
                    action_id=decision_id,
                )
                for decision_id in choices
            ],
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=[],
            image_roles=[],
            sample_rate=16000,
            micro_batch_size=min(self.action_micro_batch_size, len(choices)),
            session_id=self.session_id,
            stage="performance",
            admission_priority=1,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            text_role=turn.text_role,
            history=[],
            history_audios=[],
            history_images=[],
            prefix_cache_namespace=(
                f"performance:v1:{self.locale}:"
                f"{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
            ),
            cache_static_system_only=True,
        )
        result = await self._score_action_request(turn, request)
        top = max(result.scores, key=lambda item: item.mean_logprob)
        choice = choices[top.candidate_id]
        expression = None
        if choice.expression_id is not None:
            candidate = expression_by_id[choice.expression_id]
            expression = {
                "category_id": FACIAL_EXPRESSION_CATEGORY_ID,
                "candidate_id": candidate.candidate_id,
                "expression_id": candidate.action_id,
                "label": candidate.source_label,
                "description": _FACE_ONLY_DESCRIPTIONS[candidate.candidate_id],
                "apply": True,
            }
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        decision = PerformanceDecision(
            request_scope=choice.scope,  # type: ignore[arg-type]
            expression=expression,
            expression_unsupported=choice.expression_unsupported,
            tts_instruction=self._tts_instruction(choice.expression_id),
            elapsed_ms=elapsed_ms,
        )
        emit_structured_log(
            "performance",
            "turn_performance_control_ready",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            request_scope=decision.request_scope,
            expression_candidate_id=(
                expression.get("candidate_id") if expression else None
            ),
            expression_unsupported=decision.expression_unsupported,
            tts_instruction=decision.tts_instruction,
            tts_instruction_generated=True,
            tts_instruction_delivery=(
                "embedded_tts_append"
                if "audio" in self.modalities
                else "not_requested"
            ),
            elapsed_ms=elapsed_ms,
        )
        return decision
