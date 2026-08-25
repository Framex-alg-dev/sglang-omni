# SPDX-License-Identifier: Apache-2.0
"""Immutable server-wide action catalog for realtime Qwen3-Omni sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate
from sglang_omni.models.qwen3_omni.prompt_localization import (
    DEFAULT_PROMPT_LOCALE,
    PROMPT_LANGUAGE_BY_LOCALE,
    SUPPORTED_PROMPT_LOCALES,
)
from sglang_omni.utils.structured_logs import emit_structured_log


logger = logging.getLogger(__name__)


GLOBAL_ACTION_CATALOG_PATH_ENV = "SGLANG_OMNI_ACTION_CATALOG_PATH"
GLOBAL_ACTION_PREWARM_TIMEOUT_ENV = "SGLANG_OMNI_GLOBAL_ACTION_PREWARM_TIMEOUT_S"
DEFAULT_GLOBAL_ACTION_CATALOG_RESOURCE = (
    "assets/character_action_global_catalog.json"
)
UNSUPPORTED_DECISION_ID = "UNSUPPORTED"
UNSUPPORTED_CATEGORY_SCORE_ID = "B000"
UNSUPPORTED_CHILD_SCORE_ID = "A000"
UNSUPPORTED_SOURCE_LABEL = "不支持的动作"
SUPPORTED_ACTION_PROMPT_LOCALES = SUPPORTED_PROMPT_LOCALES
DEFAULT_ACTION_PROMPT_LOCALE = DEFAULT_PROMPT_LOCALE
ACTION_PROMPT_LANGUAGE_BY_LOCALE = PROMPT_LANGUAGE_BY_LOCALE
UNSUPPORTED_CATEGORY_SHORT_DEFINITION = (
    "除强制选择兜底类别的规则外，用户明确要求执行动作，但该动作所属语义类别不在"
    "本次会话允许的类别中"
)
UNSUPPORTED_CHILD_SHORT_DEFINITION = (
    "用户明确要求执行动作，且动作语义类别已选定，但本次会话在该类别下允许的"
    "候选动作均无法完成该请求"
)
# Backward-compatible public alias for code that previously consumed the
# single Child-oriented definition. Prompt construction now uses the two
# stage-specific definitions above.
UNSUPPORTED_SHORT_DEFINITION = UNSUPPORTED_CHILD_SHORT_DEFINITION


def _normalize_prompt_locale(locale: str) -> str:
    if locale not in SUPPORTED_ACTION_PROMPT_LOCALES:
        raise ValueError(f"unsupported action prompt locale: {locale!r}")
    return locale


def category_unsupported_policy(locale: str = "zh-CN") -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        return (
            f"category_id={UNSUPPORTED_CATEGORY_SCORE_ID} | decision=unsupported action category | "
            "description=Except when a default category must be selected, the user explicitly "
            "requests an action whose semantic category is not among the categories allowed in "
            "this conversation.\n"
            "If the target semantic category is allowed in this conversation, select that "
            "category even when it may lack a suitable concrete action; support for a concrete "
            "action is decided in the next stage. If the user does not explicitly request a "
            "specific action, do not select the unsupported decision. For ordinary dialogue, "
            "silent observation, or natural idle behavior, select an appropriate default action "
            "category. Do not select an unrelated category merely to avoid "
            f"{UNSUPPORTED_CATEGORY_SCORE_ID}."
        )
    return (
        f"category_id={UNSUPPORTED_CATEGORY_SCORE_ID}｜决策=不支持的动作类别｜"
        f"说明={UNSUPPORTED_CATEGORY_SHORT_DEFINITION}\n"
        "如果目标语义类别在本次会话允许的类别中，应选择该类别；不得因为该类别下"
        "可能缺少具体候选动作而"
        f"选择 {UNSUPPORTED_CATEGORY_SCORE_ID}，具体动作是否支持由下一阶段判断。"
        "用户没有明确要求具体动作时，不得选择不支持判断；普通对话、静默观察或只需"
        "自然待机时，应从本次会话提供的默认动作类别中选择合适类别。"
        f"不得为了避免 {UNSUPPORTED_CATEGORY_SCORE_ID} 而选择与目标动作语义不相关的类别。"
    )


def child_unsupported_policy(locale: str = "zh-CN") -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        return (
            f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID} | decision=unsupported concrete action | "
            "description=The user explicitly requests an action, its semantic category has "
            "already been selected, but none of the candidates allowed in that category for "
            "this conversation can fulfill the request.\n"
            "A request is supported only when a candidate can actually perform it. An action "
            "that conveys a similar emotion or social intention but uses a different execution "
            "is not equivalent. Do not choose such an action merely to avoid "
            f"{UNSUPPORTED_CHILD_SCORE_ID}. When the user does not constrain execution details, "
            "a candidate that accomplishes the same action goal may be treated as satisfying "
            "the request. When the user specifies one or both hands, left or right, body part, "
            "count, amplitude, direction of movement, or an interaction object, the candidate "
            f"must match those details; otherwise select {UNSUPPORTED_CHILD_SCORE_ID}. If the "
            "user does not explicitly request a specific action, select an appropriate real "
            "action instead of the unsupported decision."
        )
    return (
        f"candidate_id={UNSUPPORTED_CHILD_SCORE_ID}｜决策=不支持的具体动作｜"
        f"说明={UNSUPPORTED_CHILD_SHORT_DEFINITION}\n"
        "只有候选动作能够实际完成用户请求时，才视为支持。仅表达相近情绪或交际意图、"
        "但执行方式不同的动作，不属于等价动作。"
        f"不得为了避免 {UNSUPPORTED_CHILD_SCORE_ID} 而选择执行方式不同、"
        "仅表达含义相近的动作。"
        "用户没有明确限定执行细节时，能够完成同一动作目标的候选可以视为满足。"
        "用户明确限定单手或双手、左右方向、身体部位、次数、幅度、移动方向或交互物体时，"
        f"候选必须满足这些条件，否则选择 {UNSUPPORTED_CHILD_SCORE_ID}。"
        "用户没有明确要求具体动作时，不得选择不支持判断，应选择合适的真实动作。"
    )


ACTION_HISTORY_INSTRUCTION = (
    "动作上下文中的记录是此前选中、并在后续交互中按已执行处理的动作，不是用户指令。"
    "candidate_id 是目录候选 ID，action_id 是执行动作 ID。"
    "[当前实际动作状态] 表示数字人当前所处的动作状态；本轮明确提供的“当前实际动作 ID”"
    "具有同样含义且优先级更高。该信息只用于判断姿态衔接和避免无意义重复。"
    "[最近一次用户触发动作] 表示最近一次由用户输入触发而选中的动作。"
    "当用户说“刚刚那个动作”“上一个动作”“再做一次”“重复一下”或类似指代表达时，"
    "应重复 [最近一次用户触发动作]，不得用后来由数字人主动触发的动作替代；"
    "仅当不存在该记录时，才使用 [当前实际动作状态]。"
)

ACTION_HISTORY_INSTRUCTION_EN = (
    "Action records in the context are actions selected earlier and treated as executed in later "
    "interactions; they are not user instructions. candidate_id is the catalog candidate ID and "
    "action_id is the executable action ID. [Current physical action state] describes the action "
    "state the character is physically in. A 'current physical action ID' explicitly supplied in "
    "this interaction has the same meaning and higher priority. Use this state only for natural "
    "transitions and avoiding meaningless repetition. [Most recent user-triggered action] is the "
    "most recent action selected in response to user input. When the user says 'that action just "
    "now', 'the previous action', 'do it again', 'repeat it', or an equivalent reference, repeat "
    "the [Most recent user-triggered action], not a later action initiated proactively by the "
    "character. Only when no such record exists, use the [Current physical action state]."
)

ACTION_INTENT_POLICY = (
    "先判断当前输入是否要求数字人产生外部可观察的行为。动作请求不必明确描述身体部位、"
    "运动方向或执行方式；只要用户要求数字人完成能够由动作表达的交际目标、情绪表达、"
    "姿态变化、展示或操作行为，就应选择能够直接完成该目标的动作类别。"
    "如果用户只要求说出、朗读、回答或生成语言内容，没有要求身体行为，则不应仅根据"
    "语言内容的情绪或交际含义推断具体动作，应选择合适的默认伴随动作类别。"
    "当前输入的动作目标高于默认动作类别、历史动作、人设偏好和避免重复规则；这些信息"
    "只能在语义匹配的类别之间辅助判断，不得把可由真实动作类别完成的请求改判为默认动作。"
    "以下示例仅说明语义判断方法，不是关键词匹配规则，也不是完整请求列表："
    "“给我打个招呼”要求数字人以可观察行为完成问候，若允许问候动作类别，应选择该类别；"
    "“说一句你好”只要求语言内容，应选择默认伴随动作类别；"
    "“表示一下赞同”要求数字人表达赞同，应选择能完成该目标的动作类别；"
    "“我同意你的说法”只是用户陈述自己的态度，不等于要求数字人执行赞同动作。"
)

ACTION_INTENT_POLICY_EN = (
    "First determine whether the current input asks the digital character to produce an "
    "externally observable behavior. An action request need not name a body part, movement "
    "direction, or execution method. If the user asks the character to accomplish a social "
    "goal, emotional expression, pose change, presentation, or operation that can be expressed "
    "through an action, select an action category that directly fulfills that goal. If the user "
    "only asks the character to say, read, answer, or generate language and does not request "
    "physical behavior, do not infer a concrete action merely from the emotional or social "
    "meaning of the words; select an appropriate default accompanying-action category. The "
    "current input's action goal takes precedence over default categories, action history, "
    "persona preferences, and repetition avoidance. Those signals may only break ties among "
    "semantically matching categories and must not turn a request supported by a real action "
    "category into a default action. The following examples illustrate semantic reasoning; "
    "they are not keyword-matching rules or an exhaustive request list: 'greet me' asks for an "
    "observable greeting and should use an allowed greeting-action category; 'say hello' asks "
    "only for spoken content and should use a default accompanying-action category; 'show "
    "agreement' requests an agreement action; 'I agree with you' only states the user's own "
    "attitude and does not request an agreement action from the character."
)

CATEGORY_CONTEXT_POLICY = (
    "选择类别前，先综合用户摄像头画面与用户语音判断用户状态："
    "情绪可归纳为开心、兴奋、惊讶、疑惑、生气、悲伤、紧张或平静；"
    "场景类型可归纳为私人空间、工作学习空间、公共空间，或驾驶、会议、医院等特殊场景；"
    "任务类型可归纳为信息获取、问题解决、情绪支持、社交闲聊、展示分享或静默观察。"
    "特殊场景下，无论本轮由用户触发还是由数字人主动触发，"
    "必须从本次会话提供的默认动作类别中，选择顺序最靠前且符合当前状态的低打扰类别。"
    "低打扰动作指幅度较小、不发生明显位移、不依赖额外物体且不会打断当前任务的动作。"
    "若 [本次会话数字人人设与动作偏好] 明确提供了数字人人设信息，包括性别、"
    "二次元/写实/卡通等画风、"
    "职业或角色定位、性格基调，选择类别时应将其纳入考虑。"
    "未提供人设信息时，可从标记为“数字人当前状态画面”的图片中观察"
    "性别表达、大致年龄段、是否有胡须、发型、是否穿裙装、是否戴眼镜等外观特征，"
    "仅依据画面可见信息判断，不得臆测未展示的设定。"
    "景别与物体前置过滤：仅根据“数字人当前状态画面”判断数字人的当前构图及其可交互物体，"
    "不得把“用户摄像头画面”当作数字人当前状态。"
    "若数字人仅头肩或半身入镜，避免选择要求下肢、位移或全身大幅移动的 B033-B037；"
    "选择要求与具体物体交互的 B043-B052 前，必须确认“数字人当前状态画面”中确实存在"
    "对应物体，否则改选不依赖物体的类别。"
)

CATEGORY_CONTEXT_POLICY_EN = (
    "Before selecting a category, infer the user's state from the user camera view and user "
    "speech. Summarize emotion as happy, excited, surprised, confused, angry, sad, nervous, or "
    "calm; scene type as private space, work or study space, public space, or a special scene "
    "such as driving, a meeting, or a hospital; and task type as information seeking, problem "
    "solving, emotional support, social chat, presentation or sharing, or silent observation. "
    "In a special scene, whether initiated by the user or the digital character, choose the "
    "first suitable low-disturbance category from the default action categories supplied for "
    "this conversation. A low-disturbance action has small amplitude, no obvious displacement, "
    "requires no extra object, and does not interrupt the current task. If [Digital character "
    "persona and action preferences for this conversation] explicitly supplies persona details "
    "such as gender expression, anime/realistic/cartoon visual style, occupation or role, and "
    "personality, incorporate them when selecting a category. If persona details are absent, "
    "visible traits such as gender expression, approximate age group, facial hair, hairstyle, "
    "skirt-like clothing, or glasses may be observed from an image labeled 'Current digital "
    "character state view'. Use only visible evidence and do not invent unseen settings. For "
    "framing and object filtering, infer the character's composition and interactable objects "
    "only from the 'Current digital character state view'; never treat a 'User camera view' as "
    "the character's current state. If only the head-and-shoulders or upper body is visible, "
    "avoid B033-B037, which require lower-body movement, displacement, or large full-body motion. "
    "Before selecting B043-B052, which require interaction with physical objects, confirm that "
    "the corresponding object is visible in the current digital character state view; otherwise "
    "select a category that does not depend on the object."
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class GlobalActionCandidate:
    candidate_id: str
    action_id: str
    source_label: str
    short_definition: str
    source_short_definition: str
    category_id: str


@dataclass(frozen=True, slots=True)
class GlobalActionCategory:
    category_id: str
    source_label: str
    short_definition: str
    category_path: tuple[str, ...]
    children: tuple[GlobalActionCandidate, ...]


@dataclass(frozen=True, slots=True)
class GlobalActionCatalog:
    catalog_version: str
    catalog_hash: str
    categories: tuple[GlobalActionCategory, ...]
    category_by_id: Mapping[str, GlobalActionCategory]
    candidate_by_id: Mapping[str, GlobalActionCandidate]
    category_system_prompt: str
    category_prompt_hash: str
    child_system_prompts: Mapping[str, str]
    child_prompt_hashes: Mapping[str, str]
    category_system_prompts_by_locale: Mapping[str, str]
    category_prompt_hashes_by_locale: Mapping[str, str]
    child_system_prompts_by_locale: Mapping[str, Mapping[str, str]]
    child_prompt_hashes_by_locale: Mapping[str, Mapping[str, str]]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_by_id)

    def category_system_prompt_for(self, locale: str) -> str:
        return self.category_system_prompts_by_locale[
            _normalize_prompt_locale(locale)
        ]

    def category_prompt_hash_for(self, locale: str) -> str:
        return self.category_prompt_hashes_by_locale[
            _normalize_prompt_locale(locale)
        ]

    def child_system_prompt_for(self, locale: str, category_id: str) -> str:
        return self.child_system_prompts_by_locale[
            _normalize_prompt_locale(locale)
        ][category_id]

    def child_prompt_hash_for(self, locale: str, category_id: str) -> str:
        return self.child_prompt_hashes_by_locale[
            _normalize_prompt_locale(locale)
        ][category_id]

    def category_cache_namespace(self, locale: str = "zh-CN") -> str:
        normalized = _normalize_prompt_locale(locale)
        return (
            f"hierarchical:{normalized}:category:"
            f"{self.category_prompt_hash_for(normalized)}"
        )

    def child_cache_namespace(
        self, category_id: str, locale: str = "zh-CN"
    ) -> str:
        normalized = _normalize_prompt_locale(locale)
        return (
            f"hierarchical:{normalized}:child:{category_id}:"
            f"{self.child_prompt_hash_for(normalized, category_id)}"
        )


@dataclass(frozen=True, slots=True)
class GlobalActionLocalePrewarmStatus:
    category_ready: bool
    ready_child_category_ids: frozenset[str]
    failed_child_category_ids: frozenset[str]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class GlobalActionCatalogPrewarmStatus:
    category_ready: bool
    ready_child_category_ids: frozenset[str]
    failed_child_category_ids: frozenset[str]
    elapsed_ms: float
    by_locale: Mapping[str, GlobalActionLocalePrewarmStatus] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def not_run(cls) -> "GlobalActionCatalogPrewarmStatus":
        return cls(False, frozenset(), frozenset(), 0.0)

    def for_locale(self, locale: str) -> GlobalActionLocalePrewarmStatus:
        normalized = _normalize_prompt_locale(locale)
        status = self.by_locale.get(normalized)
        if status is not None:
            return status
        # Compatibility for tests and integrations that construct the legacy
        # aggregate status directly.
        return GlobalActionLocalePrewarmStatus(
            self.category_ready,
            self.ready_child_category_ids,
            self.failed_child_category_ids,
            self.elapsed_ms,
        )


async def prewarm_global_action_catalog(
    client: Any,
    *,
    model: str,
    catalog: GlobalActionCatalog,
) -> GlobalActionCatalogPrewarmStatus:
    """Best-effort prefill of both localized Category and Child prefixes."""

    started = time.perf_counter()
    prefill = getattr(client, "prefill_action_catalog", None)
    if not callable(prefill):
        logger.warning("[GLOBAL_ACTION_PREWARM] client has no catalog prefill method")
        failed_ids = frozenset(item.category_id for item in catalog.categories)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        missing = MappingProxyType(
            {
                locale: GlobalActionLocalePrewarmStatus(
                    False, frozenset(), failed_ids, elapsed_ms
                )
                for locale in SUPPORTED_ACTION_PROMPT_LOCALES
            }
        )
        return GlobalActionCatalogPrewarmStatus(
            False, frozenset(), failed_ids, elapsed_ms, missing
        )

    emit_structured_log(
        "lifecycle",
        "global_action_catalog_prewarm_started",
        catalog_version=catalog.catalog_version,
        global_catalog_hash=catalog.catalog_hash,
        locales=list(SUPPORTED_ACTION_PROMPT_LOCALES),
        category_count=len(catalog.categories),
        candidate_count=catalog.candidate_count,
    )
    logger.info(
        "[GLOBAL_ACTION_PREWARM] started catalog_version=%s locales=%s "
        "categories=%d actions=%d",
        catalog.catalog_version,
        ",".join(SUPPORTED_ACTION_PROMPT_LOCALES),
        len(catalog.categories),
        catalog.candidate_count,
    )
    try:
        timeout_s = float(os.environ.get(GLOBAL_ACTION_PREWARM_TIMEOUT_ENV, "10"))
    except ValueError as exc:
        raise ValueError(
            f"{GLOBAL_ACTION_PREWARM_TIMEOUT_ENV} must be a positive number"
        ) from exc
    if timeout_s <= 0:
        raise ValueError(
            f"{GLOBAL_ACTION_PREWARM_TIMEOUT_ENV} must be a positive number"
        )

    async def prefill_one(**kwargs: Any) -> tuple[bool, str | None]:
        try:
            ready = await asyncio.wait_for(prefill(**kwargs), timeout=timeout_s)
            return bool(ready), None
        except Exception as exc:
            logger.warning(
                "[GLOBAL_ACTION_PREWARM] prefix failed request_id=%s error=%s",
                kwargs.get("request_id"),
                exc,
                exc_info=True,
            )
            return False, f"{type(exc).__name__}: {exc}"

    async def prewarm_locale(locale: str) -> GlobalActionLocalePrewarmStatus:
        locale_started = time.perf_counter()
        prompt_language = ACTION_PROMPT_LANGUAGE_BY_LOCALE[locale]
        category_prompt = catalog.category_system_prompt_for(locale)
        category_hash = catalog.category_prompt_hash_for(locale)
        category_namespace = catalog.category_cache_namespace(locale)
        category_started = time.perf_counter()
        category_ready, category_error = await prefill_one(
            request_id=f"global-action-category-prewarm-{prompt_language}",
            model=model,
            system_prompt=category_prompt,
            candidates=[
                ActionScoreCandidate(
                    candidate_id=item.category_id,
                    suffix=item.category_id,
                    action_id=item.category_id,
                )
                for item in catalog.categories
            ]
            + [
                ActionScoreCandidate(
                    candidate_id=UNSUPPORTED_CATEGORY_SCORE_ID,
                    suffix=UNSUPPORTED_CATEGORY_SCORE_ID,
                    action_id=UNSUPPORTED_DECISION_ID,
                )
            ],
            prefix_cache_namespace=category_namespace,
            stage="category",
            language=prompt_language,
        )
        emit_structured_log(
            "performance",
            "global_action_category_prefix_prewarm_completed",
            locale=locale,
            language=prompt_language,
            global_catalog_hash=catalog.catalog_hash,
            prompt_hash=category_hash,
            prompt_chars=len(category_prompt),
            prefix_cache_namespace=category_namespace,
            candidate_count=len(catalog.categories) + 1,
            prewarmed=bool(category_ready),
            error_message=category_error,
            elapsed_ms=round((time.perf_counter() - category_started) * 1000.0, 3),
        )

        ready_children: set[str] = set()
        failed_children: set[str] = set()
        for category in catalog.categories:
            child_started = time.perf_counter()
            category_id = category.category_id
            child_prompt = catalog.child_system_prompt_for(locale, category_id)
            child_hash = catalog.child_prompt_hash_for(locale, category_id)
            namespace = catalog.child_cache_namespace(category_id, locale)
            ready, error_message = await prefill_one(
                request_id=(
                    f"global-action-child-prewarm-{prompt_language}-{category_id}"
                ),
                model=model,
                system_prompt=child_prompt,
                candidates=[
                    ActionScoreCandidate(
                        candidate_id=item.candidate_id,
                        suffix=item.candidate_id,
                        action_id=item.action_id,
                    )
                    for item in category.children
                ]
                + [
                    ActionScoreCandidate(
                        candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                        suffix=UNSUPPORTED_CHILD_SCORE_ID,
                        action_id=UNSUPPORTED_DECISION_ID,
                    )
                ],
                prefix_cache_namespace=namespace,
                stage="child",
                language=prompt_language,
            )
            (ready_children if ready else failed_children).add(category_id)
            emit_structured_log(
                "performance",
                "global_action_child_prefix_prewarm_completed",
                locale=locale,
                language=prompt_language,
                global_catalog_hash=catalog.catalog_hash,
                category_id=category_id,
                prompt_hash=child_hash,
                prompt_chars=len(child_prompt),
                prefix_cache_namespace=namespace,
                candidate_count=len(category.children) + 1,
                prewarmed=bool(ready),
                error_message=error_message,
                elapsed_ms=round(
                    (time.perf_counter() - child_started) * 1000.0, 3
                ),
            )

        locale_elapsed_ms = round(
            (time.perf_counter() - locale_started) * 1000.0, 3
        )
        locale_status = GlobalActionLocalePrewarmStatus(
            bool(category_ready),
            frozenset(ready_children),
            frozenset(failed_children),
            locale_elapsed_ms,
        )
        emit_structured_log(
            "lifecycle",
            "global_action_catalog_locale_prewarm_completed",
            level=(
                "info" if category_ready and not failed_children else "warning"
            ),
            locale=locale,
            language=prompt_language,
            global_catalog_hash=catalog.catalog_hash,
            category_ready=locale_status.category_ready,
            ready_child_count=len(locale_status.ready_child_category_ids),
            failed_child_category_ids=sorted(
                locale_status.failed_child_category_ids
            ),
            elapsed_ms=locale_elapsed_ms,
        )
        return locale_status

    by_locale = {
        locale: await prewarm_locale(locale)
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    default_status = by_locale[DEFAULT_ACTION_PROMPT_LOCALE]
    status = GlobalActionCatalogPrewarmStatus(
        default_status.category_ready,
        default_status.ready_child_category_ids,
        default_status.failed_child_category_ids,
        elapsed_ms,
        MappingProxyType(by_locale),
    )
    emit_structured_log(
        "lifecycle",
        "global_action_catalog_prewarm_completed",
        level=(
            "info"
            if all(
                item.category_ready and not item.failed_child_category_ids
                for item in by_locale.values()
            )
            else "warning"
        ),
        global_catalog_hash=catalog.catalog_hash,
        locale_statuses={
            locale: {
                "category_ready": item.category_ready,
                "ready_child_count": len(item.ready_child_category_ids),
                "failed_child_category_ids": sorted(
                    item.failed_child_category_ids
                ),
                "elapsed_ms": item.elapsed_ms,
            }
            for locale, item in by_locale.items()
        },
        elapsed_ms=elapsed_ms,
    )
    logger.info(
        "[GLOBAL_ACTION_PREWARM] completed elapsed_ms=%.3f locale_statuses=%s",
        elapsed_ms,
        {
            locale: {
                "category_ready": item.category_ready,
                "ready_children": len(item.ready_child_category_ids),
                "failed_children": len(item.failed_child_category_ids),
            }
            for locale, item in by_locale.items()
        },
    )
    return status


def build_category_system_prompt(
    categories: tuple[GlobalActionCategory, ...],
    locale: str = "zh-CN",
) -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        lines = [
            "You are a digital-character action category classifier. Select one category_id from the fixed category set.",
            ACTION_HISTORY_INSTRUCTION_EN,
            ACTION_INTENT_POLICY_EN,
            CATEGORY_CONTEXT_POLICY_EN,
            category_unsupported_policy(locale),
            "Fixed category set:",
        ]
        lines.extend(
            f"category_id={item.category_id} | category={item.source_label} | description={item.short_definition}"
            for item in categories
        )
        lines.append(
            f"Select the category_id that best matches the current input, or {UNSUPPORTED_CATEGORY_SCORE_ID}. "
            "Output exactly one result and stop immediately. Do not explain."
        )
        return "\n".join(lines)
    lines = [
        "你是数字人动作类别识别器。请从固定类别集合中选择一个 category_id。",
        ACTION_HISTORY_INSTRUCTION,
        ACTION_INTENT_POLICY,
        CATEGORY_CONTEXT_POLICY,
        category_unsupported_policy(locale),
        "固定类别集合如下：",
    ]
    lines.extend(
        f"category_id={item.category_id}｜类别={item.source_label}｜说明={item.short_definition}"
        for item in categories
    )
    lines.append(
        f"请根据当前输入选择最匹配的 category_id 或 {UNSUPPORTED_CATEGORY_SCORE_ID}；"
        "只输出一个结果，输出后立即结束，不要解释。"
    )
    return "\n".join(lines)


def build_child_system_prompt(
    category: GlobalActionCategory,
    locale: str = "zh-CN",
) -> str:
    locale = _normalize_prompt_locale(locale)
    if locale == "en-US":
        lines = [
            "You are a digital-character action classifier. Select one candidate_id from the following set.",
            ACTION_HISTORY_INSTRUCTION_EN,
            (
                f"Selected category: category_id={category.category_id} | "
                f"category={category.source_label} | description={category.short_definition}"
            ),
            child_unsupported_policy(locale),
        ]
        lines.extend(
            f"candidate_id={item.candidate_id} | action={item.source_label} | description={item.short_definition}"
            for item in category.children
        )
        lines.append(
            "Select the candidate_id that best matches from the candidates in the selected category above. "
            "Output exactly one result."
        )
        return "\n".join(lines)
    lines = [
        "你是数字人动作识别器。请从以下集合中选择一个 candidate_id。",
        ACTION_HISTORY_INSTRUCTION,
        (
            f"已选类别：category_id={category.category_id}｜类别={category.source_label}｜"
            f"说明={category.short_definition}"
        ),
        child_unsupported_policy(locale),
    ]
    lines.extend(
        f"candidate_id={item.candidate_id}｜动作={item.source_label}｜说明={item.short_definition}"
        for item in category.children
    )
    lines.append(
        "只能从以上已选类别的候选动作中选择最匹配的 candidate_id；"
        "只输出一个结果。"
    )
    return "\n".join(lines)


def load_global_action_catalog(path: str | Path | None = None) -> GlobalActionCatalog:
    """Load and fully validate the authoritative catalog.

    Empty action ``short_definition`` values intentionally remain untouched in
    the source file and catalog hash. At runtime their label is used as the
    prompt definition so every prompt line stays meaningful.
    """

    configured_path = path or os.environ.get(GLOBAL_ACTION_CATALOG_PATH_ENV)
    if configured_path is None:
        catalog_path = files("sglang_omni").joinpath(
            DEFAULT_GLOBAL_ACTION_CATALOG_RESOURCE
        )
        raw_text = catalog_path.read_text(encoding="utf-8")
        display_path = str(catalog_path)
    else:
        catalog_path = Path(configured_path).expanduser()
        raw_text = catalog_path.read_text(encoding="utf-8")
        display_path = str(catalog_path)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid global action catalog JSON: {display_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("global action catalog must be a JSON object")
    catalog_version = _required_string(payload.get("catalog_version"), "catalog_version")
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("global action catalog categories must be a non-empty list")

    category_ids: set[str] = set()
    candidate_ids: set[str] = set()
    action_ids: set[str] = set()
    categories: list[GlobalActionCategory] = []
    candidate_by_id: dict[str, GlobalActionCandidate] = {}
    for category_index, raw_category in enumerate(raw_categories):
        if not isinstance(raw_category, dict):
            raise ValueError(f"categories[{category_index}] must be an object")
        prefix = f"categories[{category_index}]"
        category_id = _required_string(raw_category.get("category_id"), f"{prefix}.category_id")
        if category_id in {
            UNSUPPORTED_CATEGORY_SCORE_ID,
            UNSUPPORTED_CHILD_SCORE_ID,
            UNSUPPORTED_DECISION_ID,
        }:
            raise ValueError(
                f"global category_id is reserved for unsupported scoring: {category_id}"
            )
        if category_id in category_ids:
            raise ValueError(f"duplicate global category_id: {category_id}")
        category_ids.add(category_id)
        source_label = _required_string(raw_category.get("source_label"), f"{prefix}.source_label")
        short_definition = _required_string(
            raw_category.get("short_definition"), f"{prefix}.short_definition"
        )
        raw_path = raw_category.get("category_path")
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError(f"{prefix}.category_path must be a non-empty string list")
        category_path = tuple(
            _required_string(value, f"{prefix}.category_path[{index}]")
            for index, value in enumerate(raw_path)
        )
        raw_children = raw_category.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ValueError(f"{prefix}.children must be a non-empty list")
        children: list[GlobalActionCandidate] = []
        for child_index, raw_child in enumerate(raw_children):
            if not isinstance(raw_child, dict):
                raise ValueError(f"{prefix}.children[{child_index}] must be an object")
            child_prefix = f"{prefix}.children[{child_index}]"
            candidate_id = _required_string(
                raw_child.get("candidate_id"), f"{child_prefix}.candidate_id"
            )
            action_id = _required_string(raw_child.get("action_id"), f"{child_prefix}.action_id")
            if candidate_id in {
                UNSUPPORTED_CATEGORY_SCORE_ID,
                UNSUPPORTED_CHILD_SCORE_ID,
                UNSUPPORTED_DECISION_ID,
            }:
                raise ValueError(
                    "global candidate_id is reserved for unsupported scoring: "
                    f"{candidate_id}"
                )
            if action_id == UNSUPPORTED_DECISION_ID:
                raise ValueError(
                    "global action_id=UNSUPPORTED is reserved for the internal "
                    "non-executable decision"
                )
            if action_id == "no_action":
                raise ValueError(
                    "global action catalog must not contain action_id=no_action; "
                    "configure executable fallback categories in session.start"
                )
            child_label = _required_string(
                raw_child.get("source_label"), f"{child_prefix}.source_label"
            )
            source_definition = raw_child.get("short_definition")
            if not isinstance(source_definition, str):
                raise ValueError(f"{child_prefix}.short_definition must be a string")
            source_definition = source_definition.strip()
            prompt_definition = source_definition or child_label
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate global candidate_id: {candidate_id}")
            if action_id in action_ids:
                raise ValueError(f"duplicate global action_id: {action_id}")
            candidate_ids.add(candidate_id)
            action_ids.add(action_id)
            child = GlobalActionCandidate(
                candidate_id=candidate_id,
                action_id=action_id,
                source_label=child_label,
                short_definition=prompt_definition,
                source_short_definition=source_definition,
                category_id=category_id,
            )
            children.append(child)
            candidate_by_id[candidate_id] = child
        categories.append(
            GlobalActionCategory(
                category_id=category_id,
                source_label=source_label,
                short_definition=short_definition,
                category_path=category_path,
                children=tuple(children),
            )
        )

    collisions = category_ids & candidate_ids
    if collisions:
        raise ValueError(
            "global category_id and candidate_id values must be disjoint: "
            + ", ".join(sorted(collisions))
        )
    normalized_categories = tuple(categories)
    category_by_id = {item.category_id: item for item in normalized_categories}
    category_prompts_by_locale = {
        locale: build_category_system_prompt(normalized_categories, locale)
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    child_prompts_by_locale = {
        locale: MappingProxyType(
            {
                item.category_id: build_child_system_prompt(item, locale)
                for item in normalized_categories
            }
        )
        for locale in SUPPORTED_ACTION_PROMPT_LOCALES
    }
    # Keep the original public fields as Chinese aliases for integrations that
    # inspect the catalog directly. Runtime sessions always use locale-aware
    # accessors.
    category_prompt = category_prompts_by_locale["zh-CN"]
    child_prompts = child_prompts_by_locale["zh-CN"]
    category_hashes_by_locale = {
        locale: _sha256_text(prompt)
        for locale, prompt in category_prompts_by_locale.items()
    }
    child_hashes_by_locale = {
        locale: MappingProxyType(
            {key: _sha256_text(value) for key, value in prompts.items()}
        )
        for locale, prompts in child_prompts_by_locale.items()
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return GlobalActionCatalog(
        catalog_version=catalog_version,
        catalog_hash=_sha256_text(canonical),
        categories=normalized_categories,
        category_by_id=MappingProxyType(category_by_id),
        candidate_by_id=MappingProxyType(candidate_by_id),
        category_system_prompt=category_prompt,
        category_prompt_hash=_sha256_text(category_prompt),
        child_system_prompts=MappingProxyType(child_prompts),
        child_prompt_hashes=MappingProxyType(
            {key: _sha256_text(value) for key, value in child_prompts.items()}
        ),
        category_system_prompts_by_locale=MappingProxyType(
            category_prompts_by_locale
        ),
        category_prompt_hashes_by_locale=MappingProxyType(
            category_hashes_by_locale
        ),
        child_system_prompts_by_locale=MappingProxyType(
            child_prompts_by_locale
        ),
        child_prompt_hashes_by_locale=MappingProxyType(
            child_hashes_by_locale
        ),
    )


__all__ = [
    "ACTION_HISTORY_INSTRUCTION",
    "GlobalActionCandidate",
    "GlobalActionCatalog",
    "GlobalActionCatalogPrewarmStatus",
    "GlobalActionLocalePrewarmStatus",
    "SUPPORTED_ACTION_PROMPT_LOCALES",
    "DEFAULT_ACTION_PROMPT_LOCALE",
    "UNSUPPORTED_CATEGORY_SCORE_ID",
    "UNSUPPORTED_CHILD_SCORE_ID",
    "UNSUPPORTED_CATEGORY_SHORT_DEFINITION",
    "UNSUPPORTED_CHILD_SHORT_DEFINITION",
    "UNSUPPORTED_DECISION_ID",
    "UNSUPPORTED_SHORT_DEFINITION",
    "UNSUPPORTED_SOURCE_LABEL",
    "GlobalActionCategory",
    "build_category_system_prompt",
    "build_child_system_prompt",
    "category_unsupported_policy",
    "child_unsupported_policy",
    "load_global_action_catalog",
    "prewarm_global_action_catalog",
]
