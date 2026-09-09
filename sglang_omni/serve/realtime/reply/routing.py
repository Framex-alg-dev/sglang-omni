"""R0-R3 history and S0-S1 speech-mode routing."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from typing import Any, Literal

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.models import (
    ReplyHistoryRouteResult,
    ReplySpeechModeResult,
    TurnBuffer,
)
from sglang_omni.utils.structured_logs import emit_structured_log as _base_emit_structured_log


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


class ReplyRoutingComponent:
    """History and pure-action classification behavior."""

    def _reply_history_route_system_prompt(self) -> str:
        return self._prompt(
            zh=(
                "你是当前用户请求的联合路由分类器。只判断两个维度：是否必须依赖"
                "历史对话，以及当前请求是否包含必须通过语言完成的独立意图。纯动作请求"
                "只要求当前角色完成可观察的身体行为、姿势变化、物体操作、对象呈现，"
                "或立即进行唱歌等声音表演，"
                "没有同时要求提供信息、识别对象、解释含义、描述内容、评价、比较、翻译、"
                "朗读或完成其他必须说出的任务。动作附带的目的、"
                "原因、语气、关心、亲近、玩笑或鼓励通常仍属于纯动作请求。若当前用户消息要求"
                "继续、重复、解释、修改或比较先前内容，或使用‘那个’‘刚才的’‘前一个’"
                "等必须依赖先前对话才能确定含义的指代，则需要历史；否则当前用户消息本身"
                "包含完整问题或指令时不需要历史。选择：R0=不需要历史且需要语言回复；"
                "R1=需要历史且需要语言回复；R2=不需要历史的纯动作请求；R3=需要历史"
                "才能确定目标的纯动作请求。\n"
                "判定边界：\n"
                "1. 是否需要历史，只判断当前请求能否独立确定含义，不判断历史是否可能有帮助。\n"
                "2. 是否为纯动作，只判断是否存在必须通过语言完成的独立意图，不判断动作是否受支持。\n"
                "3. 动作是否受支持以及选择哪个具体动作，不属于本分类任务。\n"
                "4. 动作附带的目的、原因、语气、关心、亲近、玩笑或鼓励，通常不构成独立语言意图。\n"
                "5. 同时要求执行动作和完成独立语言任务时，属于需要语言回复；动作仍由独立动作系统处理。\n"
                "6. 不得根据‘看看’‘展示’‘介绍’等单个词分类，应判断用户期望的结果是可观察行为，还是必须说出的信息。\n"
                "7. 疑问句形式本身不表示需要语言回复。用户使用‘你可以……吗’‘能不能……’"
                "‘愿意……吗’等形式，请求当前角色直接执行一个具体动作时，预期结果仍是"
                "动作执行，属于纯动作请求。“给我唱一首”“现在唱一段吧”“唱给我听”"
                "明确要求立即进行声音表演，也属于纯动作请求；“你可以唱歌吗”“你会唱歌吗”"
                "只询问能力，属于需要语言回复。\n"
                "8. 只有用户真正询问动作能力边界、支持范围、不能执行的原因、动作名称、"
                "方法或含义，或者明确要求口头说明时，才属于需要语言回复。\n"
                "9. 当前请求可以独立理解时，即使与上一轮主题相同，也不得判为需要历史。\n"
                "10. 当前消息省略对象地否定、纠正或评价上一轮回复，或者明确表示上一轮交互后"
                "某种状态仍未改变时，需要历史。即使其中的情绪或部分语义可以单独理解，也需要"
                "历史才能知道用户否定什么、什么处理没有奏效并避免重复。‘不行’‘不对’‘没用’"
                "‘不是这样’‘还是’‘仍然’‘依然’‘没有好转’‘这样也不行’等词不能单独作为"
                "关键词判断；只有它们实际承接、否定、评价或延续先前内容时才表示需要历史。\n"
                "11. “可以呀”“好”“行”“开始吧”“讲吧”“那你说吧”等肯定、许可或"
                "继续式省略表达，如果是在接受上一轮提议、授权开始上一轮尚未完成的任务，或"
                "要求继续上一轮内容，则需要历史并选择 R1。这是语义承接规则，不是关键词"
                "匹配；包含完整对象和任务的独立请求仍按当前消息判断。若仅凭当前消息无法确定"
                "被同意的上一轮任务是语言任务还是动作任务，默认选择 R1，由历史恢复任务并"
                "生成语言回复。\n"
                "12. 如果当前问题只能从本次会话中过去的用户自述、约定、偏好、先前请求、"
                "先前回复或生成内容中回答，即使句子语法完整，也需要历史并选择 R1。"
                "例如‘我叫什么名字’‘我喜欢什么’‘我们之前讲到哪里了’都需要历史；"
                "普通外部知识问题以及依据当前 system 人设回答的‘你叫什么名字’不因此需要历史。\n"
                "13. 只输出 R0、R1、R2 或 R3，不要回答用户请求，也不要输出解释。\n"
                "边界示例：\n"
                "“请介绍一下你自己”→R0；“这个动作叫什么”→R0；"
                "“一边挥手，一边介绍你自己”→R0；“介绍一下这个杯子”→R0；"
                "“看看这是什么”→R0；“比较一下这两本书”→R0；“读一下这一页”→R0；"
                "“你会撒娇吗”→R0；“你可以唱歌吗”→R0；“你能做哪些动作”→R0；"
                "“可以给我讲个故事吗”→R0；“为什么不能翻跟头”→R0；"
                "“我今天很不开心”→R0；“我不喜欢苹果”→R0；“这个方法通常没用吗”→R0。\n"
                "“继续说”“为什么”“把刚才的介绍说短一点”→R1；"
                "“不行，我还是很不开心”→R1；“这样也不行，换一种方式吧”→R1；"
                "“我仍然没有听懂”→R1；“不是这个，我说的是刚才那个”→R1；"
                "“你刚才说的方法没用”→R1；在承接上一轮提议时，“可以呀”“好，开始吧”"
                "“讲吧”“那你说吧”→R1；“我叫什么名字”“我喜欢什么”"
                "“我们之前讲到哪里了”→R1。\n"
                "“请翻个跟头”“喝口水吧，别渴着”“做个鬼脸逗我开心”→R2；"
                "“展示一下这个杯子”→R2；“拿起这本书”→R2；“翻开下一页”→R2；"
                "“你可以撒个娇吗”→R2；“能挥挥手吗”→R2；"
                "“你可以给我打个招呼吗”“给我打个招呼”“挥挥手”→R2；"
                "“可以转一圈给我看吗”→R2；“给我唱一首”“现在唱一段吧”→R2。\n"
                "“再做一次刚才那个动作”“换成上一个动作”→R3。"
            ),
            en=(
                "You jointly route the current user request along two dimensions: "
                "whether prior conversation is required and whether the request contains an "
                "independent intent that must be completed through speech. A pure-action "
                "request only asks the current character to complete observable physical "
                "behavior, a pose change, object manipulation, object presentation, or an "
                "immediate vocal performance such as singing. It "
                "does not also request information, object recognition, explanation, content "
                "description, evaluation, comparison, translation, reading, or another task "
                "that must be spoken. A purpose, reason, tone, caring "
                "remark, intimacy, joke, or encouragement attached to the action normally "
                "remains a pure-action request. Prior conversation is required only when "
                "the utterance asks to continue, repeat, explain, modify, or compare prior "
                "content, or contains a reference that cannot be resolved without it. "
                "Choose R0 for current-only language-required, R1 for history-required "
                "language-required, R2 for current-only pure-action, or R3 for a pure-action "
                "request whose target requires history.\n"
                "Decision boundaries:\n"
                "1. History requirement asks only whether the current request's meaning can "
                "be determined independently, not whether history might be helpful.\n"
                "2. Pure-action status asks only whether an independent intent must be "
                "completed through speech, not whether the requested action is supported.\n"
                "3. Action support and concrete action selection are outside this task.\n"
                "4. A purpose, reason, tone, caring remark, intimacy, joke, or encouragement "
                "attached to an action normally does not create an independent language intent.\n"
                "5. A request combining an action with an independent spoken task is "
                "language-required; the separate action system still handles the action.\n"
                "6. Do not classify from isolated words such as 'look', 'show', or 'introduce'. "
                "Decide whether the expected result is observable behavior or spoken information.\n"
                "7. Interrogative form alone does not make speech necessary. When phrases such "
                "as 'can you ...?', 'could you ...?', or 'would you ...?' pragmatically ask the "
                "current character to perform a concrete action, the expected result is still "
                "the action itself, so classify it as pure action. 'Sing me a song', 'Sing "
                "something now', or 'Sing for me' requests an immediate vocal performance and "
                "is pure action; 'Can you sing?' or 'Do you know how to sing?' asks only about "
                "capability and is language-required.\n"
                "8. Choose language-required only when the user genuinely asks about action "
                "capability boundaries, the supported action range, why an action cannot be "
                "performed, an action's name, method, or meaning, or explicitly requests a "
                "spoken explanation.\n"
                "9. When the current request is independently understandable, do not mark it "
                "history-required even if it shares a topic with the previous turn.\n"
                "10. History is required when the current message elliptically rejects, "
                "corrects, or evaluates the preceding reply, or explicitly says that a state "
                "remains unchanged after the preceding interaction. Even if an emotion or part "
                "of the meaning is understandable alone, history is needed to identify what "
                "the user rejects, what failed, and how to avoid repeating it. Words such as "
                "'no', 'wrong', 'did not work', 'not like that', 'still', 'remains', 'has not "
                "improved', or 'that also does not work' are not keyword rules. They require "
                "history only when they actually continue, reject, evaluate, or preserve a "
                "state from prior content.\n"
                "11. Elliptical affirmations, permissions, or continuation cues such as 'Sure', "
                "'Okay', 'Go ahead', 'Start', or 'Then tell me' require history and choose R1 "
                "when they accept a prior offer, authorize a previously proposed unfinished task, "
                "or continue prior content. This is a semantic continuation rule, not keyword "
                "matching; a complete standalone request with its own object and task is still "
                "classified from the current message. If the current message alone cannot reveal "
                "whether the accepted prior task was language or action, default to R1 so history "
                "can restore the task and generate a language reply.\n"
                "12. If a question can only be answered from prior user claims, agreements, "
                "preferences, earlier requests, earlier replies, or generated content in this "
                "session, history is required and the choice is R1 even when the sentence is "
                "grammatically complete. 'What is my name?', 'What do I like?', and 'Where did "
                "we leave off?' require history. An ordinary external-knowledge question or "
                "'What is your name?' answered from the current system persona does not require "
                "history for this reason.\n"
                "13. Output only R0, R1, R2, or R3. Do not answer the user or explain the choice.\n"
                "Boundary examples:\n"
                "'Introduce yourself', 'What is this action called?', and 'Wave while "
                "introducing yourself' -> R0. 'Introduce this cup', 'Look and identify this', "
                "'Compare these two books', 'Read this page', 'Do you know how to act cute?', "
                "'Can you sing?', 'What actions can you perform?', 'Can you tell me a story?', "
                "'Why can you not do a somersault?', 'I am "
                "unhappy today', 'I do not like apples', and 'Does this method usually fail?' -> R0.\n"
                "'Continue', 'Why?', 'Shorten the introduction you just gave', 'No, I am still "
                "very unhappy', 'That also did not work; try another way', 'I still do not "
                "understand', 'Not this one; I meant the previous one', and 'The method you just "
                "gave did not work' -> R1. When they accept a prior offer, 'Sure', 'Okay, start', "
                "'Go ahead', and 'Then tell me' -> R1. 'What is my name?', 'What do I like?', "
                "and 'Where did we leave off?' -> R1.\n"
                "'Do a somersault', 'Drink some water so you do not get thirsty', and "
                "'Make a funny face to cheer me up' -> R2. 'Show me this cup', 'Pick up "
                "this book', 'Turn to the next page', 'Can you act cute for me?', "
                "'Could you wave?', 'Can you greet me?', 'Give me a greeting', 'Wave to me', "
                "and 'Could you turn around so I can see?' -> R2.\n"
                "'Sing me a song' and 'Sing something now' -> R2.\n"
                "'Do that previous action again' and 'Switch to the preceding action' -> R3."
            ),
        )
    def _reply_speech_mode_system_prompt(self) -> str:
        return self._prompt(
            zh=(
                "你是当前用户请求的语言必要性分类器。只判断当前请求是否包含必须通过"
                "说话完成的独立任务。选择 S0=需要语言回复，或 S1=纯动作请求。纯动作"
                "请求只要求当前角色完成可观察的身体行为、姿势变化、物体操作、对象呈现，"
                "或立即进行唱歌等声音表演；"
                "动作附带的目的、原因、语气、关心、亲近、玩笑或鼓励通常仍属于纯动作。"
                "如果还要求提供信息、介绍或识别对象、解释、描述、评价、比较、翻译、朗读，"
                "或者完成其他明确要求说出的内容，必须选择 S0。动作是否受支持、是否能选出"
                "具体动作，不属于本分类任务，也不能用来证明没有语言任务。疑问句形式本身"
                "不表示需要语言回复。‘你可以……吗’‘能不能……’‘愿意……吗’等表达如果"
                "是在请求当前角色直接执行具体动作，应选择 S1；只有真正询问动作能力边界、"
                "支持范围、不能执行的原因、动作名称、方法或含义，或者明确要求口头说明时，"
                "才选择 S0。没有要求立即表演、只询问是否具备某项能力的“你可以唱歌吗”"
                "“你会唱歌吗”应选择 S0；“给我唱一首”“现在唱一段吧”明确要求立即表演，"
                "应选择 S1。要求交付语言内容的礼貌问句仍是语言任务，例如“可以给我讲个"
                "故事吗”应选择 S0。"
                "示例：‘挥手并介绍你自己’→S0；‘拿起杯子并说明材质’→S0；"
                "‘你会撒娇吗’→S0；‘你可以唱歌吗’→S0；‘你能做哪些动作’→S0；"
                "‘可以给我讲个故事吗’→S0；"
                "‘挥手欢迎我’→S1；‘拿起杯子让我看看’→S1；"
                "‘你可以撒个娇吗’→S1；‘能挥挥手吗’→S1；"
                "‘你可以给我打个招呼吗’‘给我打个招呼’‘挥挥手’→S1；"
                "‘给我唱一首’→S1。"
                "只输出 S0 或 S1，不要回答用户请求，也不要解释。"
            ),
            en=(
                "Classify only whether the current user request contains an independent task "
                "that must be completed through speech. Choose S0 for language required or "
                "S1 for a pure-action request. A pure-action request only asks the current "
                "character to perform observable physical behavior, a pose change, object "
                "manipulation, object presentation, or an immediate vocal performance such as "
                "singing. A purpose, reason, tone, caring remark, "
                "intimacy, joke, or encouragement attached to the action normally remains part "
                "of the pure-action request. Choose S0 if the request also asks for information, "
                "introduction or object identification, explanation, description, evaluation, "
                "comparison, translation, reading, or any other explicitly spoken content. "
                "Action support and concrete action selection are outside this task and cannot "
                "prove that no spoken task exists. Interrogative form alone does not make speech "
                "necessary. If 'can you ...?', 'could you ...?', or 'would you ...?' pragmatically "
                "asks the current character to perform a concrete action, choose S1. Choose S0 "
                "only for a genuine question about action capability boundaries, the supported "
                "action range, why an action cannot be performed, an action's name, method, or "
                "meaning, or an explicit request for spoken explanation. A bare capability "
                "question such as 'Can you sing?' or 'Do you know how to sing?' is S0 when it "
                "does not request an immediate performance; 'Sing me a song' or 'Sing something "
                "now' is S1. A polite request that asks for language content, such as 'Can you "
                "tell me a story?', is S0. Examples: 'Wave while "
                "introducing yourself', 'Pick up the cup and explain its material', 'Do you know "
                "how to act cute?', 'Can you sing?', 'Can you tell me a story?', and 'What "
                "actions can you perform?' -> S0; 'Wave to welcome "
                "me', 'Pick up the cup so I can see it', 'Can you act cute for me?', 'Could "
                "you wave?', 'Can you greet me?', 'Give me a greeting', 'Wave to me', and "
                "'Sing me a song' -> S1. Output only S0 or S1 without explanation."
            ),
        )
    async def _disambiguate_reply_speech_mode(
        self,
        turn: TurnBuffer,
        audios: list[str],
        *,
        current_text: str | None,
        fallback_reply_mode: Literal["LANGUAGE_REQUIRED", "PURE_ACTION"],
    ) -> ReplySpeechModeResult:
        """Resolve language-vs-action, preserving the initial route on failure."""
        request_id = f"{turn.request_base}-reply-speech-mode"
        system_prompt = self._reply_speech_mode_system_prompt()
        normalized_text = current_text.strip() if isinstance(current_text, str) else ""
        request = ActionSuffixScoreRequest(
            request_id=request_id,
            model=self.model_name,
            prefix=self._prompt(zh="分类结果：", en="Classification result:"),
            system_prompt=system_prompt,
            current_text=normalized_text,
            output_prompt="",
            language=self.language,
            candidates=[
                ActionScoreCandidate(
                    candidate_id="S0",
                    suffix="S0",
                    action_id=REPLY_MODE_LANGUAGE_REQUIRED,
                ),
                ActionScoreCandidate(
                    candidate_id="S1",
                    suffix="S1",
                    action_id=REPLY_MODE_PURE_ACTION,
                ),
            ],
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=[],
            image_roles=[],
            sample_rate=16000,
            micro_batch_size=2,
            session_id=self.session_id,
            stage=REPLY_SPEECH_MODE_STAGE,
            admission_priority=2,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            text_role=turn.text_role,
            history=[],
            history_audios=[],
            history_images=[],
            prefix_cache_namespace=(
                f"reply-speech-mode:v1:{self.locale}:"
                f"{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
            ),
            cache_static_system_only=True,
        )
        try:
            timeout_s = float(
                os.environ.get(
                    REPLY_HISTORY_ROUTE_TIMEOUT_ENV,
                    DEFAULT_REPLY_HISTORY_ROUTE_TIMEOUT_S,
                )
            )
        except (TypeError, ValueError):
            timeout_s = DEFAULT_REPLY_HISTORY_ROUTE_TIMEOUT_S
        timeout_s = max(0.05, timeout_s)
        started = time.perf_counter()
        self._register_turn_request(turn, request_id)
        try:
            result = await asyncio.wait_for(
                self.client.score_action_suffixes(request), timeout=timeout_s
            )
            self._ensure_turn_processing(turn)
            matched = {
                score.candidate_id: float(score.mean_logprob)
                for score in result.scores
                if score.candidate_id in {"S0", "S1"}
            }
            if not matched:
                raise ValueError("reply speech-mode route returned no S0-S1 scores")
            ranked = sorted(matched.items(), key=lambda item: item[1], reverse=True)
            winner = ranked[0][0]
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            result_value = ReplySpeechModeResult(
                reply_mode=(
                    REPLY_MODE_PURE_ACTION
                    if winner == "S1"
                    else REPLY_MODE_LANGUAGE_REQUIRED
                ),
                elapsed_ms=elapsed_ms,
                confidence_margin=margin,
                scores=matched,
                stats=dict(result.stats),
            )
            emit_structured_log(
                "reply",
                "reply_speech_mode_disambiguated",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                reply_mode=result_value.reply_mode,
                candidate_scores=matched,
                confidence_margin=margin,
                classification_ms=elapsed_ms,
                prefix_cached=result.prefix_cached,
                current_audio_count=len(audios),
                current_text_present=bool(normalized_text),
            )
            return result_value
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            fallback_reason = "timeout"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
        finally:
            self._unregister_turn_request(turn, request_id)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        emit_structured_log(
            "reply",
            "reply_speech_mode_fallback",
            level="warning",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            reply_mode=fallback_reply_mode,
            classification_ms=elapsed_ms,
            fallback_reason=fallback_reason,
        )
        return ReplySpeechModeResult(
            reply_mode=fallback_reply_mode,
            elapsed_ms=elapsed_ms,
            fallback_reason=fallback_reason,
        )
    async def _classify_reply_history_requirement(
        self,
        turn: TurnBuffer,
        audios: list[str],
        *,
        current_text: str | None = None,
    ) -> ReplyHistoryRouteResult:
        """Jointly classify history dependency and pure-action reply mode.

        The request deliberately contains only the current user text/audio.
        R0-R3 are short, symmetric score suffixes. Any operational failure
        fails closed to CURRENT_ONLY + LANGUAGE_REQUIRED so stale history
        cannot contaminate an otherwise independent turn and a legitimate
        reply is never suppressed.
        """
        normalized_text = current_text.strip() if isinstance(current_text, str) else ""
        if turn.turn_origin != TURN_ORIGIN_USER or not (audios or normalized_text):
            return ReplyHistoryRouteResult(
                decision=REPLY_HISTORY_CURRENT_ONLY,
                fallback_reason="not_user_content_turn",
            )
        request_id = f"{turn.request_base}-reply-history-route"
        system_prompt = self._reply_history_route_system_prompt()
        request = ActionSuffixScoreRequest(
            request_id=request_id,
            model=self.model_name,
            prefix=self._prompt(zh="分类结果：", en="Classification result:"),
            system_prompt=system_prompt,
            current_text=normalized_text,
            output_prompt="",
            language=self.language,
            candidates=[
                ActionScoreCandidate(
                    candidate_id="R0",
                    suffix="R0",
                    action_id=REPLY_HISTORY_CURRENT_ONLY,
                ),
                ActionScoreCandidate(
                    candidate_id="R1",
                    suffix="R1",
                    action_id=REPLY_HISTORY_REQUIRED,
                ),
                ActionScoreCandidate(
                    candidate_id="R2",
                    suffix="R2",
                    action_id=REPLY_MODE_PURE_ACTION,
                ),
                ActionScoreCandidate(
                    candidate_id="R3",
                    suffix="R3",
                    action_id=(
                        f"{REPLY_HISTORY_REQUIRED}:{REPLY_MODE_PURE_ACTION}"
                    ),
                ),
            ],
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=[],
            image_roles=[],
            sample_rate=16000,
            micro_batch_size=4,
            session_id=self.session_id,
            stage=REPLY_HISTORY_ROUTE_STAGE,
            admission_priority=0,
            logical_request_id=turn.request_base,
            turn_origin=turn.turn_origin,
            text_role=turn.text_role,
            history=[],
            history_audios=[],
            history_images=[],
            prefix_cache_namespace=(
                f"reply-history-route:v2:{self.locale}:"
                f"{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
            ),
            cache_static_system_only=True,
        )
        try:
            timeout_s = float(
                os.environ.get(
                    REPLY_HISTORY_ROUTE_TIMEOUT_ENV,
                    DEFAULT_REPLY_HISTORY_ROUTE_TIMEOUT_S,
                )
            )
        except (TypeError, ValueError):
            timeout_s = DEFAULT_REPLY_HISTORY_ROUTE_TIMEOUT_S
        timeout_s = max(0.05, timeout_s)
        started = time.perf_counter()
        self._register_turn_request(turn, request_id)
        emit_structured_log(
            "reply",
            "reply_history_route_started",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            timeout_s=timeout_s,
            history_message_count=0,
            history_audio_count=0,
            current_audio_count=len(audios),
            current_text_present=bool(normalized_text),
        )
        try:
            result = await asyncio.wait_for(
                self.client.score_action_suffixes(request), timeout=timeout_s
            )
            self._ensure_turn_processing(turn)
            matched = {
                score.candidate_id: float(score.mean_logprob)
                for score in result.scores
                if score.candidate_id in {"R0", "R1", "R2", "R3"}
            }
            if not matched:
                raise ValueError("reply route returned no R0-R3 scores")
            ranked = sorted(matched.items(), key=lambda item: item[1], reverse=True)
            winner = ranked[0][0]
            decision = (
                REPLY_HISTORY_REQUIRED
                if winner in {"R1", "R3"}
                else REPLY_HISTORY_CURRENT_ONLY
            )
            initial_reply_mode = (
                REPLY_MODE_PURE_ACTION
                if winner in {"R2", "R3"}
                else REPLY_MODE_LANGUAGE_REQUIRED
            )
            best_language_score = max(
                matched.get("R0", float("-inf")),
                matched.get("R1", float("-inf")),
            )
            best_pure_action_score = max(
                matched.get("R2", float("-inf")),
                matched.get("R3", float("-inf")),
            )
            speech_mode_pair_margin = abs(
                best_language_score - best_pure_action_score
            )
            speech_mode_disambiguation: ReplySpeechModeResult | None = None
            if speech_mode_pair_margin <= PURE_ACTION_ROUTE_AMBIGUITY_MARGIN:
                speech_mode_disambiguation = (
                    await self._disambiguate_reply_speech_mode(
                        turn,
                        audios,
                        current_text=normalized_text,
                        fallback_reply_mode=initial_reply_mode,
                    )
                )
            reply_mode = (
                speech_mode_disambiguation.reply_mode
                if speech_mode_disambiguation is not None
                else initial_reply_mode
            )
            margin = (
                ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            route_stats = dict(result.stats)
            if speech_mode_disambiguation is not None:
                route_stats["speech_mode_disambiguation"] = {
                    "reply_mode": speech_mode_disambiguation.reply_mode,
                    "elapsed_ms": speech_mode_disambiguation.elapsed_ms,
                    "confidence_margin": (
                        speech_mode_disambiguation.confidence_margin
                    ),
                    "scores": dict(speech_mode_disambiguation.scores),
                    "fallback_reason": (
                        speech_mode_disambiguation.fallback_reason
                    ),
                }
            route = ReplyHistoryRouteResult(
                decision=decision,
                reply_mode=reply_mode,
                elapsed_ms=elapsed_ms,
                confidence_margin=margin,
                pure_action_ambiguous=False,
                scores=matched,
                stats=route_stats,
            )
            audio_stage_timing = (
                result.stats.get("pipeline_stage_timing", {}).get(
                    "audio_encoder", {}
                )
            )
            audio_encoder_cache_status = (
                audio_stage_timing.get("cache_status")
                if isinstance(audio_stage_timing, dict)
                else None
            )
            emit_structured_log(
                "reply",
                "reply_history_route_completed",
                session_id=self.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
                logical_request_id=turn.request_base,
                request_id=request_id,
                decision=decision,
                reply_mode=reply_mode,
                candidate_scores=matched,
                confidence_margin=margin,
                pure_action_ambiguous=False,
                initial_reply_mode=initial_reply_mode,
                speech_mode_pair_margin=speech_mode_pair_margin,
                speech_mode_disambiguated=(
                    speech_mode_disambiguation is not None
                ),
                speech_mode_disambiguation_fallback_reason=(
                    speech_mode_disambiguation.fallback_reason
                    if speech_mode_disambiguation is not None
                    else None
                ),
                classification_ms=elapsed_ms,
                prefix_cached=result.prefix_cached,
                audio_encoder_ms=float(result.stats.get("audio_encoder_ms", 0.0)),
                audio_encoder_cache_status=audio_encoder_cache_status,
                audio_encoder_cache_hit=(
                    audio_encoder_cache_status in {"hit", "shared"}
                    if audio_encoder_cache_status is not None
                    else None
                ),
                history_available_turn_count=len(self.reply_history_turns),
                history_forwarded_turn_count=(
                    min(len(self.reply_history_turns), MAX_REPLY_HISTORY_TURNS)
                    if decision == REPLY_HISTORY_REQUIRED
                    else 0
                ),
            )
            return route
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            fallback_reason = "timeout"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
        finally:
            self._unregister_turn_request(turn, request_id)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        emit_structured_log(
            "reply",
            "reply_history_route_fallback",
            level="warning",
            session_id=self.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            logical_request_id=turn.request_base,
            request_id=request_id,
            decision=REPLY_HISTORY_CURRENT_ONLY,
            reply_mode=REPLY_MODE_LANGUAGE_REQUIRED,
            classification_ms=elapsed_ms,
            fallback_reason=fallback_reason,
        )
        return ReplyHistoryRouteResult(
            decision=REPLY_HISTORY_CURRENT_ONLY,
            reply_mode=REPLY_MODE_LANGUAGE_REQUIRED,
            elapsed_ms=elapsed_ms,
            fallback_reason=fallback_reason,
        )


MultimodalReplyRoutingMixin = ReplyRoutingComponent
