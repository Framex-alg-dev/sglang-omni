"""Session.start initialization and action-catalog prefill."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from contextlib import suppress
from typing import Any, Literal

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    DEFAULT_ACTION_PROMPT_LOCALE,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
)
from sglang_omni.models.qwen3_omni.prompt_localization import PROMPT_LANGUAGE_BY_LOCALE
from sglang_omni.preprocessing.image import prepare_image_bytes_for_wire
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.embedded_tts import EmbeddedTTSConnection
from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import (
    _json_audit_fields,
    _text_audit_fields,
)
from sglang_omni.serve.realtime.protocol.models import (
    ImageFrame,
    SessionActionCandidate,
    SessionActionCategory,
    SessionActionProfile,
    TurnBuffer,
)
from sglang_omni.serve.realtime.output_capabilities import SessionOutputCapabilities
from sglang_omni.serve.realtime.action.routing import (
    visual_deictic_category_scope,
    visual_deictic_scope_candidates,
)
from sglang_omni.serve.realtime.action.visual_generation import (
    build_visual_gesture_system_prompt,
    visual_gesture_candidates,
)
from sglang_omni.serve.realtime.action.decision import (
    action_decision_candidates,
    action_support_candidates,
    category_gate_candidates,
)
from sglang_omni.serve.realtime.turn_intent import SYSTEM as TURN_INTENT_SYSTEM
from sglang_omni.serve.realtime.knowledge import (
    KnowledgeBinding,
    KnowledgeContext,
    KnowledgeEvidence,
    ProvidedEntitySnapshot,
)
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
    new_trace_id,
)

logger = logging.getLogger(__name__)

USER_CHILD_PREWARM_LABELS = (
    "基础表情", "躯干前后动作", "单臂抬起", "指向类", "展示类",
    "符号化手势", "打招呼与告别", "强调类", "鼓励与庆祝", "身体触碰",
    "手指精细动作",
)
# Counts complete cached prefixes (including shared public tokens), deliberately
# conservative. Two admitted sessions budget at most 170k tokens, leaving
# headroom in the deployed 222k-token pool. No promise of pinned KV residency.
USER_CHILD_PREWARM_TOKEN_BUDGET = 85_000
USER_CHILD_PREWARM_TIMEOUT_SECONDS = 90.0
TURN_INTENT_PREWARM_TIMEOUT_SECONDS = 10.0


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.protocol.input import MultimodalTurnInputMixin


class SessionStartComponent:
    async def _prewarm_visual_gesture_prefix(self, prefill: Any) -> bool:
        """Warm the semantic gesture classifier prefix for the experiment."""

        if (
            not getattr(self, "visual_gesture_generation_enabled", False)
            or not callable(prefill)
        ):
            return False
        candidates = visual_gesture_candidates(self.categories, self.candidates)
        if not candidates:
            return False
        request_id = f"session-{self.session_instance_id}-visual-gesture-prefill"
        started = time.perf_counter()
        request = GenerateRequest(
            model=self.model_name,
            messages=[
                Message(
                    role="system",
                    content=build_visual_gesture_system_prompt(candidates),
                ),
                Message(
                    role="user",
                    content=[{"type": "text", "text": " "}],
                ),
            ],
            sampling=SamplingParams(temperature=0, max_new_tokens=1),
            stream=False,
            output_modalities=["text"],
            metadata={
                "task": "session_visual_gesture_prewarm",
                "audios": [],
                "images": [],
                "image_roles": [],
                "session_id": self.session_id,
                "session_instance_id": self.session_instance_id,
                "logical_request_id": request_id,
            },
        )
        try:
            ready = bool(
                await asyncio.wait_for(
                    prefill(request, request_id=request_id),
                    timeout=TURN_INTENT_PREWARM_TIMEOUT_SECONDS,
                )
            )
        except Exception as exc:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                with suppress(Exception):
                    await abort(request_id)
            emit_structured_log(
                "error",
                "session_visual_gesture_prefill_failed",
                level="warning",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                request_id=request_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False
        emit_structured_log(
            "performance",
            "session_visual_gesture_prefill_completed",
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            request_id=request_id,
            candidate_count=len(candidates),
            prewarmed=ready,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return ready

    async def _prewarm_turn_intent_prefix(self, prefill: Any) -> bool:
        """Warm the static intent system/user-header prefix before first Turn."""

        if not callable(prefill):
            return False
        request_id = f"session-{self.session_instance_id}-intent-prefill"
        started = time.perf_counter()
        request = GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=TURN_INTENT_SYSTEM),
                # A non-empty placeholder preserves the user-role header. Real
                # Turn text/audio diverges only after the reusable static prefix.
                Message(
                    role="user",
                    content=[{"type": "text", "text": " "}],
                ),
            ],
            sampling=SamplingParams(temperature=0, max_new_tokens=1),
            stream=False,
            output_modalities=["text"],
            metadata={
                "task": "session_turn_intent_prewarm",
                "audios": [],
                "images": [],
                "image_roles": [],
                "session_id": self.session_id,
                "session_instance_id": self.session_instance_id,
                "logical_request_id": request_id,
            },
        )
        try:
            ready = bool(
                await asyncio.wait_for(
                    prefill(request, request_id=request_id),
                    timeout=TURN_INTENT_PREWARM_TIMEOUT_SECONDS,
                )
            )
        except Exception as exc:
            abort = getattr(self.client, "abort", None)
            if callable(abort):
                with suppress(Exception):
                    await abort(request_id)
            logger.warning(
                "[SESSION_ACTION_REALTIME] intent prefix prewarm failed; "
                "continuing without the cache session_id=%s",
                self.session_id,
                exc_info=True,
            )
            emit_structured_log(
                "error",
                "session_turn_intent_prefill_failed",
                level="warning",
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                request_id=request_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False
        emit_structured_log(
            "performance",
            "session_turn_intent_prefill_completed",
            session_id=self.session_id,
            session_instance_id=self.session_instance_id,
            request_id=request_id,
            prewarmed=ready,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return ready

    async def _prewarm_user_child_sessions(self, prefill: Any) -> None:
        if not callable(prefill):
            raise ValueError("session_child_prewarm_unavailable")
        by_label = {category.source_label: category for category in self.categories}
        self.session_prewarmed_child_category_ids = []
        consumed = 0
        started = time.perf_counter()
        # Ordinary contextual routing strips user-camera frames. Warm that
        # actual path; explicit visual imitation uses a separate visual prompt.
        instruction = self._build_session_action_profile_instruction(
            "child", turn_origin=TURN_ORIGIN_USER, has_user_camera=False,
        )
        optional_ids = {c.category_id for tag in (
            CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
            CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
        ) if (c := self.global_action_catalog.category_with_semantic_tag(tag)) is not None}
        optional_labels = tuple(c.source_label for c in self.categories
            if c.category_id in optional_ids and c.source_label not in USER_CHILD_PREWARM_LABELS)
        async with asyncio.timeout(USER_CHILD_PREWARM_TIMEOUT_SECONDS):
            for label in (*USER_CHILD_PREWARM_LABELS, *optional_labels):
                optional = label in optional_labels
                if optional and consumed >= USER_CHILD_PREWARM_TOKEN_BUDGET:
                    break
                category = by_label.get(label)
                if category is None or not category.children:
                    emit_structured_log(
                        "performance", "session_child_prewarm_skipped",
                        session_id=self.session_id, source_label=label,
                        reason="category_not_available_in_session",
                    )
                    continue
                children = list(category.children)
                namespace = self._session_action_prefix_namespace(
                    base_namespace=self.global_action_catalog.child_cache_namespace(
                        category.category_id, self.action_locale, TURN_ORIGIN_USER,
                    ),
                    stage="child", turn_origin=TURN_ORIGIN_USER,
                    session_instruction=instruction,
                )
                stats: dict[str, Any] = {}
                item_started = time.perf_counter()
                ready = await prefill(
                    request_id=f"session-{self.session_instance_id}-child-prewarm-{category.category_id}",
                    session_instance_id=self.session_instance_id,
                    model=self.model_name,
                    system_prompt=self._build_child_system_prompt(category, children),
                    candidates=[ActionScoreCandidate(
                        candidate_id=item.candidate_id, suffix=item.candidate_id,
                        action_id=item.action_id,
                    ) for item in children],
                    prefix_cache_namespace=namespace, stage="child",
                    language=self.action_language, session_instruction=instruction,
                    admission_priority=30, stats_out=stats,
                    max_prefix_tokens=USER_CHILD_PREWARM_TOKEN_BUDGET - consumed,
                )
                tokens = int(stats.get("prefix_token_count") or stats.get("reusable_boundary_token_count") or 0)
                consumed += tokens
                emit_structured_log(
                    "performance", "session_child_prewarm_completed",
                    session_id=self.session_id, session_instance_id=self.session_instance_id,
                    category_id=category.category_id, source_label=label,
                    candidate_count=len(children), prewarmed=bool(ready),
                    input_variant="contextual_without_user_camera",
                    prefix_cache_namespace=namespace, prefix_tokens=tokens,
                    cumulative_prefix_tokens=consumed,
                    token_budget=USER_CHILD_PREWARM_TOKEN_BUDGET,
                    elapsed_ms=round((time.perf_counter() - item_started) * 1000, 3),
                )
                if not ready:
                    if optional:
                        emit_structured_log(
                            "performance", "session_child_prewarm_skipped",
                            session_id=self.session_id, source_label=label,
                            reason="optional_prefill_unavailable_or_over_budget",
                        )
                        continue
                    raise ValueError("session_child_prewarm_failed")
                if tokens <= 0:
                    raise ValueError("session_child_prewarm_missing_token_accounting")
                if consumed > USER_CHILD_PREWARM_TOKEN_BUDGET:
                    raise ValueError("session_child_prewarm_budget_exceeded")
                self._prefilled_action_prefix_namespaces.add(namespace)
                self.session_prewarmed_child_category_ids.append(category.category_id)
        emit_structured_log(
            "performance", "session_child_prewarm_ready",
            session_id=self.session_id, category_ids=self.session_prewarmed_child_category_ids,
            prefix_tokens=consumed, elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def _prefill_action_catalog_degraded(
        self,
        prefill: Any,
        **kwargs: Any,
    ) -> bool:
        """Warm an optimization-only prefix without rejecting the Session."""

        try:
            kwargs["session_instance_id"] = self.session_instance_id
            return bool(await prefill(**kwargs))
        except Exception as exc:
            logger.warning(
                "[SESSION_ACTION_REALTIME] session prefix prefill failed; "
                "continuing without the cache session_id=%s stage=%s namespace=%s",
                self.session_id,
                kwargs.get("stage"),
                kwargs.get("prefix_cache_namespace"),
                exc_info=True,
            )
            emit_structured_log(
                "error",
                "session_action_prefix_prefill_failed",
                level="warning",
                session_id=self.session_id,
                stage=kwargs.get("stage"),
                prefix_cache_namespace=kwargs.get("prefix_cache_namespace"),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return False

    async def handle_session_start(self, event: dict[str, Any]) -> None:
        if self.started:
            raise ValueError("session.start can only be sent once")
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if event.get("_protocol_version") is not None:
            output_capabilities = SessionOutputCapabilities.parse(
                event.get("modalities")
            )
            modalities = output_capabilities.outputs
        else:
            modalities = self._normalize_modalities(event.get("modalities"))
            output_capabilities = SessionOutputCapabilities.parse(list(modalities))
        if output_capabilities.audio_enabled and self.embedded_tts_config is None:
            raise ValueError("embedded TTS provider is not configured")
        requested_output_audio_voice = event.get("_output_audio_voice")
        effective_output_audio_voice = None
        if output_capabilities.audio_enabled:
            assert self.embedded_tts_config is not None
            effective_output_audio_voice = (
                requested_output_audio_voice or self.embedded_tts_config.voice
            )
        raw_instructions = event.get("instructions")
        raw_unsupported_action_text = event.get(
            "_unsupported_action_text", event.get("unsupported_action_text")
        )
        raw_action_profile = event.get("action_profile")
        raw_knowledge = event.get("knowledge")
        emit_structured_log(
            "lifecycle",
            "session_start_received",
            session_id=session_id.strip(),
            protocol_version=event.get("_protocol_version"),
            modalities=list(modalities),
            requested_selection_mode=event.get("selection_mode"),
            instructions_provided=event.get(
                "_reply_instructions_provided", "instructions" in event
            ),
            **_text_audit_fields("instructions", raw_instructions),
            **_text_audit_fields(
                "unsupported_action_text", raw_unsupported_action_text
            ),
            action_profile_provided="action_profile" in event,
            **_json_audit_fields("action_profile", raw_action_profile),
            fallback_category_ids=event.get(
                "_fallback_category_ids", event.get("fallback_category_ids")
            ),
            knowledge_binding_id=(
                raw_knowledge.get("binding_id")
                if isinstance(raw_knowledge, dict)
                else None
            ),
        )

        action_profile = (
            SessionActionProfile.from_payload(raw_action_profile)
            if raw_action_profile is not None
            else None
        )
        if action_profile is not None and "action" not in modalities:
            raise ValueError("action_profile requires the action modality")
        action_profile_payload = (
            action_profile.as_dict() if action_profile is not None else None
        )
        action_profile_audit = _json_audit_fields(
            "action_profile", action_profile_payload
        )

        raw_candidates = event.get("action_candidates")
        raw_fallback_category_ids = event.get(
            "_fallback_category_ids", event.get("fallback_category_ids")
        )
        categories: list[SessionActionCategory] = []
        candidates: list[SessionActionCandidate] = []
        fallback_category_ids: list[str] = []
        if "action" in modalities:
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(
                    "action_candidates must be a non-empty list when action modality is enabled"
                )
            if len(raw_candidates) > MAX_ACTION_CANDIDATES:
                raise ValueError(
                    f"action_candidates must contain at most {MAX_ACTION_CANDIDATES} items"
                )

            nested = all(
                isinstance(x, dict) and "children" in x for x in raw_candidates
            )
            categories = (
                [SessionActionCategory.from_payload(x) for x in raw_candidates]
                if nested
                else []
            )
            if categories:
                if len(categories) > MAX_ACTION_CATEGORIES:
                    raise ValueError(
                        f"action categories must contain at most {MAX_ACTION_CATEGORIES} items"
                    )
                candidates = [
                    child for category in categories for child in category.children
                ]
                category_ids = [category.category_id for category in categories]
                candidate_ids = [x.candidate_id for x in candidates]
                if len(set(category_ids)) != len(category_ids):
                    raise ValueError("action category IDs must be unique")
                if set(category_ids) & set(candidate_ids):
                    raise ValueError(
                        "action category and candidate IDs must be disjoint"
                    )
            else:
                candidates = [
                    SessionActionCandidate.from_payload(x) for x in raw_candidates
                ]
                candidate_ids = [x.candidate_id for x in candidates]
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError("action candidate IDs must be unique")
            if self.global_action_catalog is not None:
                if categories:
                    categories, candidates = (
                        self._canonicalize_global_hierarchical_catalog(raw_candidates)
                    )
                else:
                    candidates = self._canonicalize_global_flat_catalog(raw_candidates)
            if len(candidates) > MAX_ACTION_CANDIDATES:
                raise ValueError(
                    f"action candidates must contain at most {MAX_ACTION_CANDIDATES} children"
                )
            if categories and not self.direct_action_selection:
                if (
                    not isinstance(raw_fallback_category_ids, list)
                    or not raw_fallback_category_ids
                ):
                    # The public protocol is already rejected by
                    # _normalize_session_start. Keep the legacy/internal test
                    # path compatible by recovering its historical no_action
                    # category instead of weakening the wire contract.
                    legacy_fallback = next(
                        (
                            category.category_id
                            for category in categories
                            if any(
                                child.action_id == "no_action"
                                for child in category.children
                            )
                        ),
                        None,
                    )
                    if event.get("_protocol_version") is None and legacy_fallback:
                        raw_fallback_category_ids = [legacy_fallback]
                    else:
                        raise ValueError(
                            "fallback_category_ids must be a non-empty list when "
                            "hierarchical action selection is enabled"
                        )
                known_category_ids = {item.category_id for item in categories}
                for raw_category_id in raw_fallback_category_ids:
                    if (
                        not isinstance(raw_category_id, str)
                        or not raw_category_id.strip()
                    ):
                        raise ValueError(
                            "fallback_category_ids items must be non-empty strings"
                        )
                    category_id = raw_category_id.strip()
                    if category_id in fallback_category_ids:
                        raise ValueError(
                            "fallback_category_ids must not contain duplicates"
                        )
                    if category_id not in known_category_ids:
                        raise ValueError(
                            "fallback_category_ids contains a category without "
                            "an executable candidate in this Session: "
                            f"{category_id}"
                        )
                    fallback_category_ids.append(category_id)
            elif raw_fallback_category_ids is not None and not self.direct_action_selection:
                raise ValueError(
                    "fallback_category_ids requires hierarchical action_candidates"
                )
        elif raw_candidates is not None and not isinstance(raw_candidates, list):
            raise ValueError("action_candidates must be a list when provided")
        elif raw_fallback_category_ids is not None:
            raise ValueError("fallback_category_ids requires the action modality")

        raw_prewarm_category_ids = [] if self.direct_action_selection else event.get("prewarm_child_category_ids", [])
        if not isinstance(raw_prewarm_category_ids, list):
            raise ValueError("prewarm_child_category_ids must be a list")
        if len(raw_prewarm_category_ids) > MAX_PREWARM_CHILD_CATEGORIES:
            raise ValueError(
                "prewarm_child_category_ids must contain at most "
                f"{MAX_PREWARM_CHILD_CATEGORIES} items"
            )
        prewarm_child_category_ids: list[str] = []
        for raw_category_id in raw_prewarm_category_ids:
            if not isinstance(raw_category_id, str) or not raw_category_id.strip():
                raise ValueError(
                    "prewarm_child_category_ids items must be non-empty strings"
                )
            category_id = raw_category_id.strip()
            if category_id in prewarm_child_category_ids:
                raise ValueError(
                    "prewarm_child_category_ids must not contain duplicates"
                )
            prewarm_child_category_ids.append(category_id)
        if prewarm_child_category_ids:
            if not categories:
                raise ValueError(
                    "prewarm_child_category_ids requires hierarchical action_candidates"
                )
            known_category_ids = {item.category_id for item in categories}
            unknown_category_ids = [
                item
                for item in prewarm_child_category_ids
                if item not in known_category_ids
            ]
            if unknown_category_ids:
                raise ValueError(
                    "prewarm_child_category_ids contains unknown category IDs: "
                    + ", ".join(unknown_category_ids)
                )

        language = event.get("language", "en")
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        instructions = event.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        if isinstance(instructions, str) and len(instructions) > MAX_INSTRUCTIONS_CHARS:
            raise ValueError(
                f"instructions must contain at most {MAX_INSTRUCTIONS_CHARS} characters"
            )
        if raw_unsupported_action_text is not None:
            if not isinstance(raw_unsupported_action_text, str):
                raise ValueError("unsupported_action_text must be a string")
            if not raw_unsupported_action_text.strip():
                raise ValueError("unsupported_action_text must be non-empty")
            if len(raw_unsupported_action_text) > MAX_UNSUPPORTED_ACTION_TEXT_CHARS:
                raise ValueError(
                    "unsupported_action_text must contain at most "
                    f"{MAX_UNSUPPORTED_ACTION_TEXT_CHARS} characters"
                )
        include_scores = event.get("include_scores", False)
        if not isinstance(include_scores, bool):
            raise ValueError("include_scores must be a boolean")
        if event.get("input_audio_format", "pcm16") != "pcm16":
            raise ValueError("only pcm16 audio is supported")
        try:
            sample_rate = int(event.get("sample_rate", 16000))
            channels = int(event.get("channels", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_rate and channels must be integers") from exc
        if sample_rate != 16000:
            raise ValueError("only 16000 Hz audio is supported")
        if channels != 1:
            raise ValueError("only mono audio is supported")

        if "selection_mode" in event:
            requested_mode = event.get("selection_mode")
            selected_mode = try_normalize_action_selection_mode(requested_mode)
            if selected_mode is None:
                logger.warning(
                    "[SESSION_ACTION_REALTIME] invalid session.start selection_mode=%r "
                    "session_id=%s; using %s",
                    requested_mode,
                    session_id,
                    self.action_selection_mode,
                )
            else:
                self.action_selection_mode = selected_mode

        if self.direct_action_selection:
            self.action_selection_mode = ACTION_SELECTION_MODE_FLAT_CHILDREN

        if "text" in modalities and "action" in modalities and not self.direct_action_selection:
            if self.action_selection_mode != ACTION_SELECTION_MODE_HIERARCHICAL:
                raise ValueError(
                    "text and action fusion currently requires selection_mode=hierarchical"
                )
            if not categories:
                raise ValueError(
                    "text and action fusion requires hierarchical action_candidates"
                )
        if self.global_action_catalog is not None and "action" in modalities:
            session_category_ids = {item.category_id for item in categories}
            if (
                output_capabilities.expression_enabled
                and FACIAL_EXPRESSION_CATEGORY_ID not in session_category_ids
            ):
                raise ValueError(
                    "expression output requires allowed candidates from facial "
                    f"expression category {FACIAL_EXPRESSION_CATEGORY_ID}"
                )
            if FACIAL_EXPRESSION_CATEGORY_ID in fallback_category_ids:
                raise ValueError(
                    "fallback_category_ids must not include the facial expression "
                    f"category {FACIAL_EXPRESSION_CATEGORY_ID}"
                )
            reply_system_category = (
                self.global_action_catalog.category_with_semantic_tag(
                    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT
                )
            )
            silent_system_category = (
                self.global_action_catalog.category_with_semantic_tag(
                    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT
                )
            )
            if silent_system_category is not None:
                if silent_system_category.category_id not in session_category_ids:
                    raise ValueError(
                        "action candidates must include the silent accompaniment "
                        f"category {silent_system_category.category_id}"
                    )
                if (
                    not fallback_category_ids
                    or fallback_category_ids[0]
                    != silent_system_category.category_id
                ):
                    raise ValueError(
                        "fallback_category_ids must start with the silent "
                        f"accompaniment category {silent_system_category.category_id}"
                    )
            if "text" in modalities and reply_system_category is not None:
                if reply_system_category.category_id not in session_category_ids:
                    raise ValueError(
                        "text and action fusion candidates must include the reply "
                        f"accompaniment category {reply_system_category.category_id}"
                    )
            if (
                reply_system_category is not None
                and reply_system_category.category_id in fallback_category_ids
            ):
                raise ValueError(
                    "fallback_category_ids must not include the reply accompaniment "
                    f"category {reply_system_category.category_id}"
                )

        emit_structured_log(
            "lifecycle",
            "session_validation_completed",
            session_id=session_id.strip(),
            outputs=list(output_capabilities.outputs),
            audio_enabled=output_capabilities.audio_enabled,
            requested_output_audio_voice=requested_output_audio_voice,
            effective_output_audio_voice=effective_output_audio_voice,
            action_candidate_count=len(candidates),
            action_category_count=len(categories),
        )
        knowledge_binding = None
        knowledge_resolve_kwargs = None
        provided_entity_snapshot = None
        provided_entity_context = None
        if raw_knowledge is not None:
            if not isinstance(raw_knowledge, dict):
                raise ValueError("knowledge must be an object")
            knowledge_mode = raw_knowledge.get("mode", "retrieval")
            if knowledge_mode == "provided_context":
                snapshot = raw_knowledge["entity_snapshot"]
                provided_entity_snapshot = ProvidedEntitySnapshot(
                    snapshot_id=snapshot["snapshot_id"],
                    revision=snapshot["revision"],
                    current_entity_id=snapshot["current_entity_id"],
                    current_entity_text=snapshot["current_entity_text"],
                    content_sha256=snapshot["content_sha256"],
                )
                knowledge_binding = KnowledgeBinding(
                    binding_id=raw_knowledge["binding_id"],
                    binding_revision=raw_knowledge.get("binding_revision"),
                    required=bool(raw_knowledge.get("required", True)),
                    tenant_id="",
                    snapshot_id=provided_entity_snapshot.snapshot_id,
                    state_token="",
                    status="ready",
                    mode="provided_context",
                )
                provided_entity_context = KnowledgeContext(
                    decision="RETRIEVE",
                    reason="client_provided_entity_snapshot",
                    result_id=provided_entity_snapshot.content_sha256,
                    state_token="",
                    snapshot_id=provided_entity_snapshot.snapshot_id,
                    evidence=(
                        KnowledgeEvidence(
                            evidence_id=provided_entity_snapshot.content_sha256,
                            source_type="provided_entity_snapshot",
                            source_id=provided_entity_snapshot.current_entity_id,
                            title="Current entity",
                            content=provided_entity_snapshot.current_entity_text,
                            authority=100,
                            metadata={"provided_context": True},
                        ),
                    ),
                )
            else:
                if self.knowledge_controller is None:
                    raise ValueError("Knowledge Gateway integration is not configured")
                tenant_id = None
                headers = getattr(self.websocket, "headers", None)
                if headers is not None:
                    tenant_id = headers.get("x-tenant-id")
                knowledge_resolve_kwargs = dict(
                    session_id=session_id.strip(),
                    tenant_id=tenant_id,
                    binding_id=raw_knowledge["binding_id"],
                    binding_revision=raw_knowledge.get("binding_revision"),
                    required=bool(raw_knowledge.get("required", True)),
                    locale=event.get(
                        "_locale", "zh-CN" if language == "zh" else "en-US"
                    ),
                )

        self.claim_session(session_id, self)
        self.session_id = session_id
        if knowledge_resolve_kwargs is not None:
            knowledge_binding = await self.knowledge_controller.resolve_session(
                **knowledge_resolve_kwargs
            )
        self.protocol_version = event.get("_protocol_version")
        self.locale = event.get("_locale", "zh-CN" if language == "zh" else "en-US")
        self.language = language
        self.action_locale = event.get("_action_locale", self.locale)
        self.action_language = PROMPT_LANGUAGE_BY_LOCALE[self.action_locale]
        self.modalities = modalities
        self.output_capabilities = output_capabilities
        self.knowledge_binding = knowledge_binding
        self.provided_entity_snapshot = provided_entity_snapshot
        self.provided_entity_context = provided_entity_context
        self.passive_action_policy_metadata = event.get("_passive_action_policy")
        if output_capabilities.audio_enabled:
            tts_kwargs: dict[str, Any] = {}
            if self.embedded_tts_connector is not None:
                tts_kwargs["connector"] = self.embedded_tts_connector
            assert self.embedded_tts_config is not None
            self.embedded_tts = EmbeddedTTSConnection(
                self.embedded_tts_config,
                session_id=session_id.strip(),
                session_instance_id=self.session_instance_id,
                **tts_kwargs,
            )
        self.output_audio_voice = effective_output_audio_voice
        if instructions is not None:
            self.instructions = instructions
        if raw_unsupported_action_text is not None:
            self.unsupported_action_text = raw_unsupported_action_text.strip()
        self.action_profile = action_profile
        if self.log_full_instructions:
            emit_structured_log(
                "diagnostic",
                "session_instructions_received",
                session_id=self.session_id,
                instructions=self.instructions,
                **_text_audit_fields("instructions", self.instructions),
            )
            if self.action_profile is not None:
                emit_structured_log(
                    "diagnostic",
                    "session_action_profile_received",
                    session_id=self.session_id,
                    action_profile=self.action_profile.as_dict(),
                    **action_profile_audit,
                )
        self.include_scores = include_scores
        self.candidates = candidates
        self.categories = categories
        self.fallback_category_ids = tuple(fallback_category_ids)
        self.prewarm_child_category_ids = tuple(prewarm_child_category_ids)
        self.candidate_by_id = {x.candidate_id: x for x in candidates}
        if (
            categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
        ):
            self.action_system_prompt = self._build_category_system_prompt()
        elif candidates:
            self.action_system_prompt = self._build_action_system_prompt()
        else:
            self.action_system_prompt = ""
        canonical = json.dumps(
            (
                [category.as_dict() for category in categories]
                if categories
                else [x.as_dict() for x in candidates]
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.action_catalog_hash = (
            "sha256:" + hashlib.sha256(canonical).hexdigest() if candidates else ""
        )
        mode_namespace = (
            "hierarchical"
            if categories
            and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
            else "flat_children"
        )
        self.action_prefix_cache_namespace = (
            self.global_action_catalog.action_cache_namespace(self.action_locale)
            if self.direct_action_selection
            else self.global_action_catalog.category_cache_namespace(self.action_locale)
            if self.global_action_catalog is not None and categories
            else f"{mode_namespace}:{self.action_locale}:{self.action_catalog_hash}"
        )
        prefill = getattr(self.client, "prefill_action_catalog", None)
        intent_prefill = getattr(self.client, "prefill_completion_prefix", None)
        intent_prefill_task = asyncio.create_task(
            self._prewarm_turn_intent_prefix(intent_prefill),
            name=f"session-intent-prefill-{self.session_instance_id}",
        )
        visual_gesture_prefill_task = asyncio.create_task(
            self._prewarm_visual_gesture_prefix(intent_prefill),
            name=f"session-visual-gesture-prefill-{self.session_instance_id}",
        )
        if self.direct_action_selection and candidates:
            # Warm every direct-action prefix that an ordinary Session can use
            # before session.started is emitted.  User camera Turns carry an
            # additional immutable policy block and therefore have a distinct
            # scoped namespace from ordinary user Turns.
            gesture_scope = visual_deictic_category_scope(
                self.categories,
                "gesture",
            )
            prefill_routes = [
                (TURN_ORIGIN_USER, False, None),
                (TURN_ORIGIN_USER, True, None),
                (TURN_ORIGIN_PROACTIVE, False, None),
            ]
            if (
                gesture_scope is not None
                and not getattr(
                    self, "visual_gesture_generation_enabled", False
                )
            ):
                prefill_routes.append(
                    (TURN_ORIGIN_USER, True, gesture_scope)
                )
            for origin, has_user_camera, visual_scope in prefill_routes:
                prefill_started = time.perf_counter()
                prefill_stats: dict[str, Any] = {}
                camera_suffix = "-camera" if has_user_camera else ""
                visual_suffix = (
                    f"-visual-{visual_scope.name}"
                    if visual_scope is not None
                    else ""
                )
                request_id = (
                    f"session-{session_id}-single-prefill-{origin}"
                    f"{camera_suffix}{visual_suffix}"
                )
                instruction = self._build_session_action_profile_instruction(
                    "single",
                    turn_origin=origin,
                    has_user_camera=has_user_camera,
                )
                prefill_candidates = list(candidates)
                if visual_scope is not None:
                    instruction += self._visual_deictic_catalog_instruction(
                        visual_scope,
                        origin,
                    )
                    scoped_candidate_ids = {
                        candidate.candidate_id
                        for candidate in visual_deictic_scope_candidates(
                            visual_scope
                        )
                    }
                    prefill_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id in scoped_candidate_ids
                    ]
                prefill_decision_visual = bool(
                    getattr(self, "action_decision_batch_visual", False)
                    and has_user_camera
                )
                prefill_category_ids = {
                    candidate.category_id
                    for candidate in prefill_candidates
                    if candidate.category_id is not None
                }
                prefill_gate_categories = [
                    category
                    for category in self.categories
                    if category.category_id in prefill_category_ids
                ]
                prefill_category_gate = bool(
                    self._direct_category_gate_enabled(origin)
                    and visual_scope is None
                    and prefill_gate_categories
                )
                namespace = self._direct_action_prefix_namespace(
                    origin,
                    instruction,
                    include_visual=prefill_decision_visual,
                    include_category_gate=prefill_category_gate,
                    category_gate_category_ids=tuple(
                        category.category_id
                        for category in prefill_gate_categories
                    ),
                )
                prefill_score_candidates = [
                    ActionScoreCandidate(
                        candidate_id=c.candidate_id,
                        suffix=c.candidate_id,
                        action_id=c.action_id,
                    )
                    for c in prefill_candidates
                ] + [ActionScoreCandidate(
                    candidate_id=UNSUPPORTED_CHILD_SCORE_ID,
                    suffix=UNSUPPORTED_CHILD_SCORE_ID,
                    action_id=UNSUPPORTED_DECISION_ID,
                )] + (
                    action_decision_candidates(
                        include_visual=prefill_decision_visual,
                        english=self.action_language == "en",
                    )
                    if getattr(self, "action_decision_batch_mode", "off")
                    != "off"
                    else []
                ) + (
                    category_gate_candidates(
                        prefill_gate_categories,
                        english=self.action_language == "en",
                    )
                    if prefill_category_gate
                    else []
                ) + (
                    action_support_candidates(
                        english=self.action_language == "en"
                    )
                    if prefill_category_gate
                    else []
                )
                ready = callable(prefill) and await self._prefill_action_catalog_degraded(
                    prefill, request_id=request_id,
                    stats_out=prefill_stats,
                    model=self.model_name,
                    session_instance_id=self.session_instance_id,
                    system_prompt=self._build_action_system_prompt(
                        origin,
                        include_visual=prefill_decision_visual,
                        include_category_gate=prefill_category_gate,
                        category_gate_categories=prefill_gate_categories,
                    ),
                    candidates=prefill_score_candidates,
                    prefix_cache_namespace=namespace, stage="single",
                    language=self.action_language, session_instruction=instruction,
                )
                if ready:
                    self._prefilled_action_prefix_namespaces.add(namespace)
                if origin == TURN_ORIGIN_USER and not has_user_camera:
                    self.action_prefix_prefilled = bool(ready)
                emit_structured_log(
                    "performance", "session_action_single_prefill_completed",
                    session_id=session_id, turn_origin=origin,
                    has_user_camera=has_user_camera,
                    prewarmed=bool(ready), stage="single",
                    selection_mode="flat_children", locale=self.action_locale,
                    session_instance_id=self.session_instance_id,
                    request_id=request_id, prefix_cache_namespace=namespace,
                    catalog_hash=self.global_action_catalog.catalog_hash,
                    action_count=len(prefill_candidates),
                    candidate_count=len(prefill_score_candidates),
                    visual_scope=(
                        visual_scope.name if visual_scope is not None else None
                    ),
                    probe_candidate_count=1,
                    elapsed_ms=round((time.perf_counter() - prefill_started) * 1000, 3),
                    stats=prefill_stats,
                )
        elif self.global_action_catalog is not None and categories:
            locale_prewarm = self.global_action_prewarm.for_locale(self.action_locale)
            self.prewarmed_child_category_ids = sorted(
                {item.category_id for item in categories}
                & set(locale_prewarm.ready_child_category_ids)
            )
            self._prefilled_action_prefix_namespaces.update(
                self.global_action_catalog.child_cache_namespace(
                    category_id, self.action_locale
                )
                for category_id in self.prewarmed_child_category_ids
            )
            if prewarm_child_category_ids:
                logger.info(
                    "[SESSION_ACTION_REALTIME] prewarm_child_category_ids is "
                    "deprecated because all global Child prefixes are warmed at "
                    "startup session_id=%s requested=%s",
                    session_id,
                    prewarm_child_category_ids,
                )
            # Server startup warms the immutable global catalog.  Before the
            # client may submit its first Turn, extend that prefix with this
            # Session's immutable persona/entity/action policy.  Failure is a
            # cache miss only: scoring remains fully functional and rebuilds
            # the prefix lazily on the first Turn.
            category_session_instruction = (
                self._build_session_action_profile_instruction(
                    "category", turn_origin=TURN_ORIGIN_USER
                )
            )
            session_category_namespace = self._session_action_prefix_namespace(
                base_namespace=self.action_prefix_cache_namespace,
                stage="category",
                turn_origin=TURN_ORIGIN_USER,
                session_instruction=category_session_instruction,
            )
            session_prefill_started = time.perf_counter()
            needs_session_category_prefill = bool(
                category_session_instruction
            ) or not locale_prewarm.category_ready
            if needs_session_category_prefill:
                self.action_prefix_prefilled = bool(
                    callable(prefill)
                    and await self._prefill_action_catalog_degraded(
                        prefill,
                        request_id=f"session-{session_id}-category-prefill",
                        model=self.model_name,
                        system_prompt=self.action_system_prompt,
                        candidates=[
                            *[
                                ActionScoreCandidate(
                                    candidate_id=item.category_id,
                                    suffix=item.category_id,
                                    action_id=item.category_id,
                                )
                                for item in categories
                            ],
                            ActionScoreCandidate(
                                candidate_id=UNSUPPORTED_CATEGORY_SCORE_ID,
                                suffix=UNSUPPORTED_CATEGORY_SCORE_ID,
                                action_id=UNSUPPORTED_DECISION_ID,
                            ),
                        ],
                        prefix_cache_namespace=session_category_namespace,
                        stage="category",
                        language=self.action_language,
                        session_instruction=category_session_instruction,
                    )
                )
            else:
                self.action_prefix_prefilled = True
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    session_category_namespace
                )
            emit_structured_log(
                "performance",
                "session_category_prefix_prefill_completed",
                session_id=session_id,
                prefix_cache_namespace=session_category_namespace,
                global_catalog_ready=locale_prewarm.category_ready,
                prefill_skipped=not needs_session_category_prefill,
                prewarmed=self.action_prefix_prefilled,
                elapsed_ms=round(
                    (time.perf_counter() - session_prefill_started) * 1000.0,
                    3,
                ),
            )
        elif candidates and callable(prefill):
            if (
                categories
                and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
            ):
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.category_id,
                        suffix=item.category_id,
                        action_id=item.category_id,
                    )
                    for item in categories
                ]
                prefill_stage = "category"
            else:
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.candidate_id,
                        suffix=item.candidate_id,
                        action_id=item.action_id,
                        execution_binding=dict(item.execution_binding),
                    )
                    for item in candidates
                ]
                prefill_stage = "single"
            session_instruction = self._build_session_action_profile_instruction(
                prefill_stage, turn_origin=TURN_ORIGIN_USER
            )
            session_prefix_namespace = self._session_action_prefix_namespace(
                base_namespace=self.action_prefix_cache_namespace,
                stage=prefill_stage,
                turn_origin=TURN_ORIGIN_USER,
                session_instruction=session_instruction,
            )
            self.action_prefix_prefilled = await self._prefill_action_catalog_degraded(
                prefill,
                model=self.model_name,
                system_prompt=self.action_system_prompt,
                candidates=prefill_candidates,
                prefix_cache_namespace=session_prefix_namespace,
                stage=prefill_stage,
                language=self.action_language,
                session_instruction=session_instruction,
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    session_prefix_namespace
                )
            if categories:
                category_by_id = {item.category_id: item for item in categories}
                for category_id in self.prewarm_child_category_ids:
                    category = category_by_id[category_id]
                    child_candidates = list(category.children)
                    child_base_namespace = (
                        f"{self.action_prefix_cache_namespace}:child:{category_id}"
                    )
                    child_session_instruction = (
                        self._build_session_action_profile_instruction(
                            "child", turn_origin=TURN_ORIGIN_USER
                        )
                    )
                    child_namespace = self._session_action_prefix_namespace(
                        base_namespace=child_base_namespace,
                        stage="child",
                        turn_origin=TURN_ORIGIN_USER,
                        session_instruction=child_session_instruction,
                    )
                    child_prewarm_started = time.perf_counter()
                    prewarmed = await self._prefill_action_catalog_degraded(
                        prefill,
                        request_id=(
                            f"session-{session_id}-child-prewarm-{category_id}"
                        ),
                        model=self.model_name,
                        system_prompt=self._build_child_system_prompt(
                            category, child_candidates
                        ),
                        candidates=[
                            ActionScoreCandidate(
                                candidate_id=item.candidate_id,
                                suffix=item.candidate_id,
                                action_id=item.action_id,
                                execution_binding=dict(item.execution_binding),
                            )
                            for item in child_candidates
                        ],
                        prefix_cache_namespace=child_namespace,
                        stage="child",
                        language=self.action_language,
                        session_instruction=child_session_instruction,
                    )
                    elapsed_ms = round(
                        (time.perf_counter() - child_prewarm_started) * 1000.0,
                        3,
                    )
                    if prewarmed:
                        self._prefilled_action_prefix_namespaces.add(child_namespace)
                        self.prewarmed_child_category_ids.append(category_id)
                    emit_structured_log(
                        "performance",
                        "child_prefix_prewarm_completed",
                        session_id=session_id,
                        category_id=category_id,
                        child_candidate_count=len(child_candidates),
                        prefix_cache_namespace=child_namespace,
                        prewarmed=prewarmed,
                        elapsed_ms=elapsed_ms,
                    )
        if not self.direct_action_selection and self.global_action_catalog is not None and categories and getattr(self, "session_child_prewarm_enabled", True):
            await self._prewarm_user_child_sessions(prefill)
        self.turn_intent_prefix_prefilled = await intent_prefill_task
        self.visual_gesture_prefix_prefilled = await visual_gesture_prefill_task
        self.started = True
        emit_structured_log(
            "lifecycle",
            "session_resources_ready",
            session_id=self.session_id,
            outputs=list(self.output_capabilities.outputs),
            tts_manager_created=self.embedded_tts is not None,
            action_prefix_prefilled=self.action_prefix_prefilled,
            turn_intent_prefix_prefilled=self.turn_intent_prefix_prefilled,
            visual_gesture_prefix_prefilled=(
                self.visual_gesture_prefix_prefilled
            ),
        )

        # ``action.allowed_candidates`` is a whitelist of unique candidate IDs.
        # One candidate may appear under multiple categories in the canonical
        # catalog (for example, a system accompaniment category and its normal
        # business category), but that does not create another executable
        # candidate from the client's perspective.
        unique_candidate_count = len(
            {candidate.candidate_id for candidate in candidates}
        )
        started_payload: dict[str, Any] = {
            "type": "session.started",
            "session_id": self.session_id,
            "model": self.model_name,
            "action_catalog_hash": self.action_catalog_hash,
            "action_candidate_count": unique_candidate_count,
            "action_category_count": len(categories),
            "action_selection_mode": self.action_selection_mode,
            "action_single_token_mode": getattr(
                self, "action_single_token_mode", "off"
            ),
            "action_selection_stages": (
                2
                if categories
                and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
                else 1
            ),
            "action_prefix_prefilled": self.action_prefix_prefilled,
            "action_profile_applied": self.action_profile is not None,
            "action_profile_sha256": action_profile_audit["action_profile_sha256"],
            "prewarmed_child_category_ids": list(self.prewarmed_child_category_ids),
            "fallback_category_ids": list(self.fallback_category_ids),
            "unsupported_action_text_configured": bool(self.unsupported_action_text),
            "unsupported_action_text_sha256": _text_audit_fields(
                "unsupported_action_text", self.unsupported_action_text or None
            )["unsupported_action_text_sha256"],
        }
        if self.protocol_version is not None:
            started_payload.update(
                {
                    "protocol_version": self.protocol_version,
                    "outputs": list(self.modalities),
                    "locale": self.locale,
                    "action_locale": self.action_locale,
                }
            )
            if self.output_audio_voice is not None:
                started_payload["output_audio"] = {
                    "voice": self.output_audio_voice,
                }
        else:
            started_payload["modalities"] = list(self.modalities)
        if self.knowledge_binding is not None:
            started_payload["knowledge"] = {
                "status": self.knowledge_binding.status,
                "snapshot_id": self.knowledge_binding.snapshot_id or None,
                "mode": self.knowledge_binding.mode,
            }
            if self.knowledge_binding.binding_revision is not None:
                started_payload["knowledge"]["binding_revision"] = (
                    self.knowledge_binding.binding_revision
                )
            if self.provided_entity_snapshot is not None:
                started_payload["knowledge"].update(
                    {
                        "revision": self.provided_entity_snapshot.revision,
                        "current_entity_id": (
                            self.provided_entity_snapshot.current_entity_id
                        ),
                        "content_sha256": (
                            self.provided_entity_snapshot.content_sha256
                        ),
                    }
                )
        if self.passive_action_policy_metadata is not None:
            started_payload["passive_action_policy"] = {
                "applied": True,
                **self.passive_action_policy_metadata,
            }
        if self.global_action_catalog is not None:
            started_payload.update(
                {
                    "session_action_catalog_hash": self.action_catalog_hash,
                    "global_action_catalog_hash": self.global_action_catalog_hash,
                    "global_action_catalog_version": (
                        self.global_action_catalog.catalog_version
                    ),
                }
            )
        selection_mapping = getattr(
            self, "action_selection_token_mapping", None
        )
        if selection_mapping is not None:
            started_payload.update(
                {
                    "action_selection_mapping_version": selection_mapping.mapping_version,
                    "action_selection_mapping_hash": selection_mapping.mapping_hash,
                    "action_selection_calibration_version": selection_mapping.calibration_version,
                    "action_selection_calibration_hash": selection_mapping.calibration_hash,
                }
            )
        if await self.send(started_payload):
            emit_structured_log(
                "lifecycle",
                "session_started_sent",
                session_id=self.session_id,
                outputs=list(self.output_capabilities.outputs),
            )
        emit_structured_log(
            "lifecycle",
            "session_started",
            session_id=self.session_id,
            protocol_version=self.protocol_version,
            locale=self.locale,
            action_locale=self.action_locale,
            modalities=list(self.modalities),
            action_selection_mode=self.action_selection_mode,
            action_single_token_mode=getattr(
                self, "action_single_token_mode", "off"
            ),
            action_selection_mapping_hash=(
                selection_mapping.mapping_hash
                if selection_mapping is not None else None
            ),
            action_selection_calibration_hash=(
                selection_mapping.calibration_hash
                if selection_mapping is not None else None
            ),
            action_ready_tts_decoupled=self.action_ready_tts_decoupled,
            route_action_parallel=self.route_action_parallel,
            action_catalog_hash=self.action_catalog_hash,
            session_action_catalog_hash=self.action_catalog_hash,
            global_action_catalog_hash=self.global_action_catalog_hash,
            global_action_catalog_version=(
                self.global_action_catalog.catalog_version
                if self.global_action_catalog is not None
                else None
            ),
            action_candidate_count=unique_candidate_count,
            action_category_count=len(categories),
            action_prefix_prefilled=self.action_prefix_prefilled,
            **_text_audit_fields(
                "unsupported_action_text", self.unsupported_action_text or None
            ),
            **action_profile_audit,
            requested_prewarm_child_category_ids=list(self.prewarm_child_category_ids),
            prewarmed_child_category_ids=list(self.prewarmed_child_category_ids),
        )


MultimodalSessionStartMixin = SessionStartComponent
