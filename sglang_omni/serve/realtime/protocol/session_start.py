"""Session.start initialization and action-catalog prefill."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Any, Literal

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
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
    new_trace_id,
)

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.protocol.input import MultimodalTurnInputMixin


class SessionStartComponent:
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
        raw_instructions = event.get("instructions")
        raw_unsupported_action_text = event.get(
            "_unsupported_action_text", event.get("unsupported_action_text")
        )
        raw_action_profile = event.get("action_profile")
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
            if categories:
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
            elif raw_fallback_category_ids is not None:
                raise ValueError(
                    "fallback_category_ids requires hierarchical action_candidates"
                )
        elif raw_candidates is not None and not isinstance(raw_candidates, list):
            raise ValueError("action_candidates must be a list when provided")
        elif raw_fallback_category_ids is not None:
            raise ValueError("fallback_category_ids requires the action modality")

        raw_prewarm_category_ids = event.get("prewarm_child_category_ids", [])
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

        if "text" in modalities and "action" in modalities:
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
            action_candidate_count=len(candidates),
            action_category_count=len(categories),
        )
        self.claim_session(session_id, self)
        self.session_id = session_id
        self.protocol_version = event.get("_protocol_version")
        self.locale = event.get("_locale", "zh-CN" if language == "zh" else "en-US")
        self.language = language
        self.modalities = modalities
        self.output_capabilities = output_capabilities
        if output_capabilities.audio_enabled:
            tts_kwargs: dict[str, Any] = {}
            if self.embedded_tts_connector is not None:
                tts_kwargs["connector"] = self.embedded_tts_connector
            assert self.embedded_tts_config is not None
            self.embedded_tts = EmbeddedTTSConnection(
                self.embedded_tts_config,
                session_id=session_id.strip(),
                **tts_kwargs,
            )
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
            self.global_action_catalog.category_cache_namespace(self.locale)
            if self.global_action_catalog is not None and categories
            else f"{mode_namespace}:{self.locale}:{self.action_catalog_hash}"
        )
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if self.global_action_catalog is not None and categories:
            locale_prewarm = self.global_action_prewarm.for_locale(self.locale)
            self.action_prefix_prefilled = locale_prewarm.category_ready
            self.prewarmed_child_category_ids = sorted(
                {item.category_id for item in categories}
                & set(locale_prewarm.ready_child_category_ids)
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    self.action_prefix_cache_namespace
                )
            self._prefilled_action_prefix_namespaces.update(
                self.global_action_catalog.child_cache_namespace(
                    category_id, self.locale
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
            self.action_prefix_prefilled = await prefill(
                model=self.model_name,
                system_prompt=self.action_system_prompt,
                candidates=prefill_candidates,
                prefix_cache_namespace=self.action_prefix_cache_namespace,
                stage=prefill_stage,
                language=self.language,
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    self.action_prefix_cache_namespace
                )
            if categories:
                category_by_id = {item.category_id: item for item in categories}
                for category_id in self.prewarm_child_category_ids:
                    category = category_by_id[category_id]
                    child_candidates = list(category.children)
                    child_namespace = (
                        f"{self.action_prefix_cache_namespace}:child:{category_id}"
                    )
                    child_prewarm_started = time.perf_counter()
                    prewarmed = await prefill(
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
                        language=self.language,
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
        self.started = True
        emit_structured_log(
            "lifecycle",
            "session_resources_ready",
            session_id=self.session_id,
            outputs=list(self.output_capabilities.outputs),
            tts_manager_created=self.embedded_tts is not None,
            action_prefix_prefilled=self.action_prefix_prefilled,
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
                }
            )
        else:
            started_payload["modalities"] = list(self.modalities)
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
            modalities=list(self.modalities),
            action_selection_mode=self.action_selection_mode,
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

