"""Wire payload validation, normalization, and catalog canonicalization."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
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
from sglang_omni.serve.realtime.components import compose_components
from sglang_omni.utils.structured_logs import (
    emit_structured_log as _base_emit_structured_log,
    new_trace_id,
)

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    from sglang_omni.serve.realtime import multimodal

    hook = getattr(multimodal, "emit_structured_log", _base_emit_structured_log)
    return hook(log_type, event, **fields)


from sglang_omni.serve.realtime.protocol.input import TurnInputComponent


class ProtocolValidationComponent:
    @staticmethod
    def _strict_object(
        value: Any,
        name: str,
        *,
        allowed: set[str],
        required: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"{name} contains unsupported fields: {', '.join(unknown)}"
            )
        missing = sorted((required or set()) - set(value))
        if missing:
            raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
        return value


    @staticmethod
    def _bounded_optional_text(
        value: Any,
        name: str,
        *,
        max_chars: int,
        allow_empty: bool = True,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
        if not allow_empty and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
        if len(value) > max_chars:
            raise ValueError(f"{name} must contain at most {max_chars} characters")
        return value


    @staticmethod
    def _serialized_chars(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


    def _bounded_candidate_id_list(
        self,
        value: Any,
        name: str,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        if len(value) > MAX_ACTION_CANDIDATES:
            raise ValueError(
                f"{name} must contain at most {MAX_ACTION_CANDIDATES} entries"
            )
        normalized: list[str] = []
        seen: set[str] = set()
        for index, raw_id in enumerate(value):
            candidate_id = self._bounded_optional_text(
                raw_id,
                f"{name}[{index}]",
                max_chars=MAX_TURN_ID_CHARS,
                allow_empty=False,
            )
            assert candidate_id is not None
            candidate_id = candidate_id.strip()
            if candidate_id in seen:
                continue
            if candidate_id not in self.candidate_by_id:
                raise ValueError(
                    f"{name}[{index}] is not available in this session: "
                    f"{candidate_id}"
                )
            seen.add(candidate_id)
            normalized.append(candidate_id)
        return tuple(normalized)


    def _wire_turn_id(self, event: dict[str, Any]) -> str:
        turn_id = self._bounded_optional_text(
            event.get("turn_id"),
            "turn_id",
            max_chars=MAX_TURN_ID_CHARS,
            allow_empty=False,
        )
        assert turn_id is not None
        return turn_id.strip()


    def _normalize_character_profile(
        self,
        value: Any,
        *,
        session_id: str | None = None,
    ) -> dict[str, str]:
        profile = self._strict_object(
            value,
            "character_profile",
            allowed=set(CHARACTER_PROFILE_FIELDS),
        )
        if not profile:
            raise ValueError("character_profile must contain at least one field")
        normalized: dict[str, str] = {}
        for field_name in CHARACTER_PROFILE_FIELDS:
            if field_name not in profile:
                continue
            max_chars = (
                MAX_CHARACTER_PROFILE_ROLE_CHARS
                if field_name == "role"
                else MAX_ACTION_PROFILE_FIELD_CHARS
            )
            field_value = self._bounded_optional_text(
                profile[field_name],
                f"character_profile.{field_name}",
                max_chars=max_chars,
                allow_empty=False,
            )
            assert field_value is not None
            normalized_value = field_value.strip()
            normalized[field_name] = normalized_value
            if (
                field_name == "role"
                and len(normalized_value) > MAX_ACTION_PROFILE_FIELD_CHARS
            ):
                logger.warning(
                    "character_profile.role exceeds recommended length "
                    "session_id=%s actual_chars=%d recommended_max_chars=%d "
                    "accepted_max_chars=%d",
                    session_id,
                    len(normalized_value),
                    MAX_ACTION_PROFILE_FIELD_CHARS,
                    MAX_CHARACTER_PROFILE_ROLE_CHARS,
                )
        return normalized


    def _compact_action_catalog(
        self, action_config: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
        if (
            self.global_action_catalog is None
            and not self.allow_unregistered_protocol_actions
        ):
            raise ValueError("action output requires the server global action catalog")

        raw_fallback_ids = action_config.get("fallback_category_ids")
        if not isinstance(raw_fallback_ids, list) or not raw_fallback_ids:
            raise ValueError("action.fallback_category_ids must be a non-empty list")
        if len(raw_fallback_ids) > MAX_ACTION_CATEGORIES:
            raise ValueError(
                "action.fallback_category_ids must contain at most "
                f"{MAX_ACTION_CATEGORIES} items"
            )
        fallback_category_ids: list[str] = []
        reserved_action_ids = {
            UNSUPPORTED_CATEGORY_SCORE_ID,
            UNSUPPORTED_CHILD_SCORE_ID,
            UNSUPPORTED_DECISION_ID,
        }
        for raw_category_id in raw_fallback_ids:
            if not isinstance(raw_category_id, str) or not raw_category_id.strip():
                raise ValueError(
                    "action.fallback_category_ids items must be non-empty strings"
                )
            category_id = raw_category_id.strip()
            if self.global_action_catalog is None and (
                category_id in reserved_action_ids or category_id == "DEV_ACTIONS"
            ):
                raise ValueError(
                    "action.fallback_category_ids contains a reserved "
                    f"development category_id: {category_id}"
                )
            if category_id in fallback_category_ids:
                raise ValueError(
                    "action.fallback_category_ids must not contain duplicates"
                )
            if (
                self.global_action_catalog is not None
                and category_id not in self.global_action_catalog.category_by_id
            ):
                raise ValueError(
                    "action.fallback_category_ids contains unknown global "
                    f"category_id: {category_id}"
                )
            fallback_category_ids.append(category_id)

        raw_allowed = action_config.get("allowed_candidates")
        if raw_allowed is None:
            raw_allowed = []
        if not isinstance(raw_allowed, list):
            raise ValueError("action.allowed_candidates must be a list when provided")
        if len(raw_allowed) > MAX_ACTION_CANDIDATES:
            raise ValueError(
                "action.allowed_candidates must contain at most "
                f"{MAX_ACTION_CANDIDATES} items"
            )
        bindings: dict[str, dict[str, str]] = {}
        for index, raw_candidate in enumerate(raw_allowed):
            item = self._strict_object(
                raw_candidate,
                f"action.allowed_candidates[{index}]",
                allowed={"candidate_id", "execution_binding"},
                required={"candidate_id"},
            )
            candidate_id = self._bounded_optional_text(
                item.get("candidate_id"),
                f"action.allowed_candidates[{index}].candidate_id",
                max_chars=MAX_TURN_ID_CHARS,
                allow_empty=False,
            )
            assert candidate_id is not None
            candidate_id = candidate_id.strip()
            if self.global_action_catalog is None and (
                candidate_id in reserved_action_ids
                or candidate_id.startswith("DEV_NONE_")
            ):
                raise ValueError(
                    "action.allowed_candidates contains a reserved "
                    f"development candidate_id: {candidate_id}"
                )
            if candidate_id in bindings:
                raise ValueError(f"duplicate action candidate_id: {candidate_id}")
            if (
                self.global_action_catalog is not None
                and candidate_id not in self.global_action_catalog.candidate_by_id
            ):
                raise ValueError(f"unknown global action candidate_id: {candidate_id}")
            binding = item.get("execution_binding") or {}
            if not isinstance(binding, dict) or not all(
                isinstance(key, str) and key.strip() and isinstance(binding_value, str)
                for key, binding_value in binding.items()
            ):
                raise ValueError(
                    "execution_binding must be a dictionary with non-empty "
                    "string keys and string values"
                )
            if self._serialized_chars(binding) > MAX_EXECUTION_BINDING_CHARS:
                raise ValueError(
                    "execution_binding must contain at most "
                    f"{MAX_EXECUTION_BINDING_CHARS} serialized characters"
                )
            if self.global_action_catalog is None and any(
                not binding_value.strip() for binding_value in binding.values()
            ):
                raise ValueError(
                    "execution_binding values must be non-empty strings in "
                    "development mode"
                )
            bindings[candidate_id] = dict(binding)

        fallback_only_category_ids: set[str] | None = None
        if not raw_allowed and self.global_action_catalog is None:
            raise ValueError(
                "action.allowed_candidates must be non-empty in development mode"
            )
        if self.global_action_catalog is None:
            collisions = set(fallback_category_ids) & set(bindings)
            if collisions:
                raise ValueError(
                    "development category_id and candidate_id values must be "
                    "disjoint: " + ", ".join(sorted(collisions))
                )
        elif not raw_allowed:
            fallback_only_category_ids = set(fallback_category_ids)
            for category_id in fallback_category_ids:
                for candidate in self.global_action_catalog.category_by_id[
                    category_id
                ].children:
                    bindings[candidate.candidate_id] = {}
        if len(bindings) > MAX_ACTION_CANDIDATES:
            raise ValueError(
                "expanded fallback action candidates must contain at most "
                f"{MAX_ACTION_CANDIDATES} items"
            )

        if self.global_action_catalog is None:
            categories = [
                {
                    "category_id": "DEV_ACTIONS",
                    "source_label": "Development actions",
                    "short_definition": "Session-scoped development actions",
                    "category_path": [],
                    "children": [
                        {
                            "candidate_id": candidate_id,
                            "action_id": candidate_id,
                            "source_label": candidate_id,
                            "short_definition": candidate_id,
                            "execution_binding": binding,
                        }
                        for candidate_id, binding in bindings.items()
                    ],
                }
            ]
            categories.extend(
                {
                    "category_id": category_id,
                    "source_label": category_id,
                    "short_definition": "Development fallback action",
                    "category_path": [],
                    "children": [
                        {
                            "candidate_id": f"DEV_NONE_{index}",
                            "action_id": "no_action",
                            "source_label": "No action",
                            "short_definition": "Keep current avatar state",
                            "execution_binding": {},
                        }
                    ],
                }
                for index, category_id in enumerate(fallback_category_ids)
            )
        else:
            categories = []
        for category in (
            self.global_action_catalog.categories
            if self.global_action_catalog is not None
            else ()
        ):
            if (
                fallback_only_category_ids is not None
                and category.category_id not in fallback_only_category_ids
            ):
                continue
            children = []
            for candidate in category.children:
                if candidate.candidate_id not in bindings:
                    continue
                children.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "action_id": candidate.action_id,
                        "source_label": candidate.source_label,
                        "short_definition": candidate.source_short_definition,
                        "proactive_expression": candidate.proactive_expression,
                        "user_reaction_expression": (
                            candidate.user_reaction_expression
                        ),
                        "execution_binding": bindings[candidate.candidate_id],
                    }
                )
            if children:
                categories.append(
                    {
                        "category_id": category.category_id,
                        "source_label": category.source_label,
                        "short_definition": category.short_definition,
                        "category_path": list(category.category_path),
                        "children": children,
                    }
                )

        allowed_category_ids = {item["category_id"] for item in categories}
        for category_id in fallback_category_ids:
            if category_id not in allowed_category_ids:
                raise ValueError(
                    "action.fallback_category_ids category must have at least "
                    "one allowed candidate in this Session: "
                    f"{category_id}"
                )

        profile_payload: dict[str, Any] = {}
        category_guidance = self._bounded_optional_text(
            action_config.get("category_guidance"),
            "action.category_guidance",
            max_chars=MAX_ACTION_PROFILE_FIELD_CHARS,
        )
        candidate_guidance = self._bounded_optional_text(
            action_config.get("candidate_guidance"),
            "action.candidate_guidance",
            max_chars=MAX_ACTION_PROFILE_FIELD_CHARS,
        )
        if category_guidance and category_guidance.strip():
            profile_payload["category_preferences"] = category_guidance.strip()
        if candidate_guidance and candidate_guidance.strip():
            profile_payload["action_preferences"] = candidate_guidance.strip()
        return categories, profile_payload or None, fallback_category_ids


    def _normalize_session_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._strict_object(
            payload,
            "session.start",
            allowed={
                "type",
                "protocol_version",
                "session_id",
                "outputs",
                "locale",
                "character_profile",
                "reply",
                "action",
                "input_audio",
                "output_audio",
                "diagnostics",
                "knowledge",
            },
            required={"type", "protocol_version", "session_id"},
        )
        version = event.get("protocol_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("protocol_version must be an integer")
        if version != REALTIME_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version: {version}; supported: "
                f"{REALTIME_PROTOCOL_VERSION}"
            )
        session_id = self._bounded_optional_text(
            event.get("session_id"),
            "session_id",
            max_chars=MAX_SESSION_ID_CHARS,
            allow_empty=False,
        )
        assert session_id is not None
        outputs = self._normalize_outputs(event.get("outputs"))
        locale = event.get("locale", DEFAULT_ACTION_PROMPT_LOCALE)
        locale_to_language = PROMPT_LANGUAGE_BY_LOCALE
        if locale not in locale_to_language:
            raise ValueError("locale must be 'zh-CN' or 'en-US'")

        character_profile = (
            self._normalize_character_profile(
                event["character_profile"], session_id=session_id.strip()
            )
            if "character_profile" in event
            else {}
        )
        reply_config = event.get("reply", {})
        reply_config = self._strict_object(
            reply_config,
            "reply",
            allowed={"instructions", "unsupported_action_text"},
        )
        if reply_config and "text" not in outputs:
            raise ValueError("reply requires the text output")
        instructions = self._bounded_optional_text(
            reply_config.get("instructions"),
            "reply.instructions",
            max_chars=MAX_INSTRUCTIONS_CHARS,
        )
        unsupported_action_text = self._bounded_optional_text(
            reply_config.get("unsupported_action_text"),
            "reply.unsupported_action_text",
            max_chars=MAX_UNSUPPORTED_ACTION_TEXT_CHARS,
            allow_empty=False,
        )
        fusion_outputs = "text" in outputs and "action" in outputs
        if fusion_outputs and unsupported_action_text is None:
            raise ValueError(
                "reply.unsupported_action_text is required when outputs contain "
                "both text and action"
            )
        if unsupported_action_text is not None and not fusion_outputs:
            raise ValueError(
                "reply.unsupported_action_text requires both text and action outputs"
            )
        # The client owns the complete reply System Prompt, including any
        # persona, defaults, or fallback behavior. Character profile remains
        # action-only context and must never be injected into the reply prompt.
        effective_instructions = instructions or ""

        action_config = event.get("action")
        action_candidates: list[dict[str, Any]] | None = None
        action_profile: dict[str, Any] | None = None
        normalized_passive_policy: dict[str, Any] | None = None
        fallback_category_ids: list[str] = []
        if "action" in outputs:
            action_config = self._strict_object(
                action_config,
                "action",
                allowed={
                    "category_guidance",
                    "candidate_guidance",
                    "passive_policy",
                    "allowed_candidates",
                    "fallback_category_ids",
                },
                required={"fallback_category_ids"},
            )
            (
                action_candidates,
                action_profile,
                fallback_category_ids,
            ) = self._compact_action_catalog(action_config)
            passive_policy = self._strict_object(
                action_config.get("passive_policy", {}),
                "action.passive_policy",
                allowed={"policy_id", "revision", "content_sha256", "guidance"},
                required={"policy_id", "revision", "content_sha256", "guidance"}
                if action_config.get("passive_policy") is not None
                else set(),
            )
            if passive_policy:
                passive_policy_id = self._bounded_optional_text(
                    passive_policy.get("policy_id"),
                    "action.passive_policy.policy_id",
                    max_chars=MAX_KNOWLEDGE_BINDING_ID_CHARS,
                    allow_empty=False,
                )
                passive_policy_revision = passive_policy.get("revision")
                if (
                    not isinstance(passive_policy_revision, int)
                    or isinstance(passive_policy_revision, bool)
                    or passive_policy_revision < 1
                ):
                    raise ValueError(
                        "action.passive_policy.revision must be a positive integer"
                    )
                passive_policy_hash = self._bounded_optional_text(
                    passive_policy.get("content_sha256"),
                    "action.passive_policy.content_sha256",
                    max_chars=71,
                    allow_empty=False,
                )
                guidance = passive_policy.get("guidance")
                if not isinstance(guidance, str) or not guidance.strip():
                    raise ValueError(
                        "action.passive_policy.guidance must be a non-empty string"
                    )
                computed_policy_hash = "sha256:" + hashlib.sha256(
                    guidance.encode("utf-8")
                ).hexdigest()
                if passive_policy_hash != computed_policy_hash:
                    raise ValueError(
                        "action.passive_policy.content_sha256 does not match guidance"
                    )
                action_profile = dict(action_profile or {})
                action_profile["passive_action_policy"] = guidance.strip()
                normalized_passive_policy = {
                    "policy_id": passive_policy_id.strip(),
                    "revision": passive_policy_revision,
                    "content_sha256": computed_policy_hash,
                }
            else:
                normalized_passive_policy = None
            if character_profile:
                action_profile = dict(action_profile or {})
                visual_behavior_preferences = character_profile.pop(
                    "visual_behavior_preferences", ""
                )
                if character_profile:
                    action_profile["persona"] = character_profile
                if visual_behavior_preferences:
                    action_profile["visual_behavior_preferences"] = (
                        visual_behavior_preferences
                    )
        elif action_config is not None:
            raise ValueError("action requires the action output")

        input_audio = self._strict_object(
            event.get("input_audio", {}),
            "input_audio",
            allowed={"format", "sample_rate_hz", "channels"},
        )
        audio_format = input_audio.get("format", "pcm16le")
        if audio_format != "pcm16le":
            raise ValueError("input_audio.format must be 'pcm16le'")
        sample_rate = input_audio.get("sample_rate_hz", 16000)
        channels = input_audio.get("channels", 1)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise ValueError("input_audio.sample_rate_hz must be an integer")
        if isinstance(channels, bool) or not isinstance(channels, int):
            raise ValueError("input_audio.channels must be an integer")

        output_audio_config = event.get("output_audio")
        output_audio_voice: str | None = None
        if "audio" in outputs:
            output_audio = self._strict_object(
                (
                    output_audio_config
                    if output_audio_config is not None
                    else {}
                ),
                "output_audio",
                allowed={"voice"},
            )
            output_audio_voice = self._bounded_optional_text(
                output_audio.get("voice"),
                "output_audio.voice",
                max_chars=MAX_OUTPUT_AUDIO_VOICE_CHARS,
                allow_empty=False,
            )
            if output_audio_voice is not None:
                output_audio_voice = output_audio_voice.strip()
                if OUTPUT_AUDIO_VOICE_RE.fullmatch(output_audio_voice) is None:
                    raise ValueError(
                        "output_audio.voice must contain only letters, numbers, "
                        "underscores, or hyphens and start with a letter or number"
                    )
        elif output_audio_config is not None:
            raise ValueError("output_audio requires the audio output")

        diagnostics = self._strict_object(
            event.get("diagnostics", {}),
            "diagnostics",
            allowed={"include_action_scores"},
        )
        include_scores = diagnostics.get("include_action_scores", False)
        if not isinstance(include_scores, bool):
            raise ValueError("diagnostics.include_action_scores must be a boolean")
        if include_scores and "action" not in outputs:
            raise ValueError(
                "diagnostics.include_action_scores requires the action output"
            )

        knowledge = self._strict_object(
            event.get("knowledge", {}),
            "knowledge",
            allowed={
                "mode",
                "binding_id",
                "binding_revision",
                "required",
                "entity_snapshot",
            },
        )
        knowledge_mode = knowledge.get("mode", "retrieval")
        if knowledge_mode not in {"retrieval", "provided_context"}:
            raise ValueError(
                "knowledge.mode must be 'retrieval' or 'provided_context'"
            )
        knowledge_binding_id = self._bounded_optional_text(
            knowledge.get("binding_id"),
            "knowledge.binding_id",
            max_chars=MAX_KNOWLEDGE_BINDING_ID_CHARS,
            allow_empty=False,
        )
        knowledge_required = knowledge.get("required", True)
        if not isinstance(knowledge_required, bool):
            raise ValueError("knowledge.required must be a boolean")
        if knowledge and knowledge_binding_id is None:
            raise ValueError("knowledge.binding_id is required")
        knowledge_binding_revision = knowledge.get("binding_revision")
        if knowledge_binding_revision is not None and (
            not isinstance(knowledge_binding_revision, int)
            or isinstance(knowledge_binding_revision, bool)
            or knowledge_binding_revision < 1
        ):
            raise ValueError("knowledge.binding_revision must be a positive integer")
        normalized_entity_snapshot = None
        raw_entity_snapshot = knowledge.get("entity_snapshot")
        if knowledge_mode == "provided_context":
            entity_snapshot = self._strict_object(
                raw_entity_snapshot,
                "knowledge.entity_snapshot",
                allowed={
                    "snapshot_id",
                    "revision",
                    "current_entity_id",
                    "current_entity_text",
                    "content_sha256",
                },
                required={
                    "snapshot_id",
                    "revision",
                    "current_entity_id",
                    "current_entity_text",
                    "content_sha256",
                },
            )
            snapshot_id = self._bounded_optional_text(
                entity_snapshot.get("snapshot_id"),
                "knowledge.entity_snapshot.snapshot_id",
                max_chars=MAX_KNOWLEDGE_BINDING_ID_CHARS,
                allow_empty=False,
            )
            entity_id = self._bounded_optional_text(
                entity_snapshot.get("current_entity_id"),
                "knowledge.entity_snapshot.current_entity_id",
                max_chars=MAX_KNOWLEDGE_BINDING_ID_CHARS,
                allow_empty=False,
            )
            snapshot_revision = entity_snapshot.get("revision")
            if (
                not isinstance(snapshot_revision, int)
                or isinstance(snapshot_revision, bool)
                or snapshot_revision < 1
            ):
                raise ValueError(
                    "knowledge.entity_snapshot.revision must be a positive integer"
                )
            entity_text = entity_snapshot.get("current_entity_text")
            # Intentionally no business length limit in the single-entity phase.
            # TODO(entity-context-size-limit): add measured character/token/request
            # limits after production entity-size and prefill-latency observation.
            if not isinstance(entity_text, str) or not entity_text.strip():
                raise ValueError(
                    "knowledge.entity_snapshot.current_entity_text must be a "
                    "non-empty string"
                )
            supplied_content_hash = self._bounded_optional_text(
                entity_snapshot.get("content_sha256"),
                "knowledge.entity_snapshot.content_sha256",
                max_chars=71,
                allow_empty=False,
            )
            computed_content_hash = "sha256:" + hashlib.sha256(
                entity_text.encode("utf-8")
            ).hexdigest()
            if supplied_content_hash != computed_content_hash:
                raise ValueError(
                    "knowledge.entity_snapshot.content_sha256 does not match "
                    "current_entity_text"
                )
            normalized_entity_snapshot = {
                "snapshot_id": snapshot_id.strip(),
                "revision": snapshot_revision,
                "current_entity_id": entity_id.strip(),
                "current_entity_text": entity_text,
                "content_sha256": computed_content_hash,
            }
        elif raw_entity_snapshot is not None:
            raise ValueError(
                "knowledge.entity_snapshot requires mode='provided_context'"
            )

        normalized: dict[str, Any] = {
            "type": "session.start",
            "session_id": session_id.strip(),
            "modalities": list(outputs),
            "language": locale_to_language[locale],
            "instructions": effective_instructions,
            "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
            "input_audio_format": "pcm16",
            "sample_rate": sample_rate,
            "channels": channels,
            "include_scores": include_scores,
            "_protocol_version": version,
            "_locale": locale,
            "_reply_instructions_provided": "instructions" in reply_config,
        }
        if unsupported_action_text is not None:
            normalized["_unsupported_action_text"] = unsupported_action_text.strip()
        if action_candidates is not None:
            normalized["action_candidates"] = action_candidates
        if action_profile is not None:
            normalized["action_profile"] = action_profile
        if normalized_passive_policy is not None:
            normalized["_passive_action_policy"] = normalized_passive_policy
        if "action" in outputs:
            normalized["_fallback_category_ids"] = fallback_category_ids
        if output_audio_voice is not None:
            normalized["_output_audio_voice"] = output_audio_voice
        if knowledge_binding_id is not None:
            normalized["knowledge"] = {
                "mode": knowledge_mode,
                "binding_id": knowledge_binding_id.strip(),
                "required": knowledge_required,
            }
            if knowledge_binding_revision is not None:
                normalized["knowledge"]["binding_revision"] = knowledge_binding_revision
            if normalized_entity_snapshot is not None:
                normalized["knowledge"]["entity_snapshot"] = normalized_entity_snapshot
        return normalized


    def _normalize_wire_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = payload.get("type")
        if event_type == "session.start":
            return self._normalize_session_start(payload)
        if event_type == "knowledge.script.event":
            event = self._strict_object(
                payload,
                "knowledge.script.event",
                allowed={"type", "request_id", "script", "event"},
                required={"type", "request_id", "script", "event"},
            )
            request_id = self._bounded_optional_text(
                event.get("request_id"),
                "request_id",
                max_chars=MAX_KNOWLEDGE_SCRIPT_EVENT_REQUEST_ID_CHARS,
                allow_empty=False,
            )
            script = self._strict_object(
                event.get("script"),
                "knowledge.script.event.script",
                allowed={"id", "version", "checksum"},
                required={"id", "version", "checksum"},
            )
            script_id = self._bounded_optional_text(
                script.get("id"),
                "knowledge.script.event.script.id",
                max_chars=MAX_KNOWLEDGE_SCRIPT_ID_CHARS,
                allow_empty=False,
            )
            script_version = script.get("version")
            if (
                not isinstance(script_version, int)
                or isinstance(script_version, bool)
                or script_version < 1
            ):
                raise ValueError(
                    "knowledge.script.event.script.version must be a positive integer"
                )
            checksum = self._bounded_optional_text(
                script.get("checksum"),
                "knowledge.script.event.script.checksum",
                max_chars=MAX_KNOWLEDGE_SCRIPT_CHECKSUM_CHARS,
                allow_empty=False,
            )
            if checksum is None or re.fullmatch(
                r"sha256:[0-9a-fA-F]{64}", checksum
            ) is None:
                raise ValueError(
                    "knowledge.script.event.script.checksum must be sha256: followed by 64 hex characters"
                )
            lifecycle_event = event.get("event")
            if lifecycle_event not in {"started", "completed", "interrupted"}:
                raise ValueError(
                    "knowledge.script.event.event must be started, completed, or interrupted"
                )
            return {
                "type": "knowledge.script.event",
                "request_id": request_id,
                "script_id": script_id,
                "script_version": script_version,
                "checksum": checksum.lower(),
                "event": lifecycle_event,
            }
        if event_type == "turn.start":
            event = self._strict_object(
                payload,
                "turn.start",
                allowed={"type", "turn_id", "origin", "trigger_type"},
                required={"type", "turn_id", "origin"},
            )
            turn_id = self._wire_turn_id(event)
            origin = event.get("origin")
            if origin not in TURN_TEXT_ROLE_BY_ORIGIN:
                raise ValueError("origin must be 'user' or 'proactive'")
            trigger = self._bounded_optional_text(
                event.get("trigger_type"),
                "trigger_type",
                max_chars=MAX_TRIGGER_TYPE_CHARS,
                allow_empty=False,
            )
            if origin == TURN_ORIGIN_USER and trigger is not None:
                raise ValueError("trigger_type is only supported for proactive turns")
            return {
                "type": "turn.start",
                "turn_id": turn_id,
                "turn_origin": origin,
                "text_role": TURN_TEXT_ROLE_BY_ORIGIN[origin],
                "trigger": trigger,
            }
        if event_type == "input.text.set":
            event = self._strict_object(
                payload,
                "input.text.set",
                allowed={"type", "turn_id", "text"},
                required={"type", "turn_id", "text"},
            )
            text = self._bounded_optional_text(
                event.get("text"),
                "text",
                max_chars=MAX_TURN_TEXT_CHARS,
            )
            turn = self._require_collecting_turn(event)
            if turn.turn_origin != TURN_ORIGIN_USER:
                raise ValueError(
                    "input.text.set is only supported for user-origin turns; "
                    "use turn.commit.reply.provided_text for proactive replies"
                )
            return {
                "type": "turn.text.update",
                "turn_id": self._wire_turn_id(event),
                "text": text,
            }
        if event_type == "input.audio.append":
            event = self._strict_object(
                payload,
                "input.audio.append",
                allowed={"type", "turn_id", "seq", "data"},
                required={"type", "turn_id", "seq", "data"},
            )
            data = event.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("input.audio.append.data must be non-empty base64")
            if data.startswith("data:"):
                raise ValueError(
                    "input.audio.append.data must be raw base64 without a data URI header"
                )
            return {
                "type": "input_audio.append",
                "turn_id": self._wire_turn_id(event),
                "seq": event.get("seq"),
                "audio": data,
            }
        if event_type == "input.image.append":
            event = self._strict_object(
                payload,
                "input.image.append",
                allowed={
                    "type",
                    "turn_id",
                    "seq",
                    "capture_timestamp_ms",
                    "media_type",
                    "image_source",
                    "data",
                },
                required={"type", "turn_id", "seq", "data"},
            )
            source = event.get("image_source")
            if source is None:
                active_origin = (
                    self.active_turn.turn_origin
                    if self.active_turn is not None
                    else None
                )
                source = (
                    "user_camera"
                    if active_origin == TURN_ORIGIN_USER
                    else IMAGE_SOURCE_AVATAR_CURRENT
                )
            source_map = {
                "user_camera": IMAGE_ROLE_USER_CAMERA,
                IMAGE_SOURCE_AVATAR_CURRENT: IMAGE_ROLE_AVATAR_STATE,
            }
            if source not in source_map:
                raise ValueError(
                    "image_source must be 'user_camera' or 'avatar_current'"
                )
            data = event.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("input.image.append.data must be non-empty base64")
            if data.startswith("data:"):
                raise ValueError(
                    "input.image.append.data must be raw base64 without a data URI header"
                )
            return {
                "type": "input_image.append",
                "turn_id": self._wire_turn_id(event),
                "seq": event.get("seq"),
                "timestamp_ms": event.get("capture_timestamp_ms", 0),
                "mime_type": event.get("media_type", "image/jpeg"),
                "image_role": source_map[source],
                "image": data,
            }
        if event_type == "turn.commit":
            event = self._strict_object(
                payload,
                "turn.commit",
                allowed={
                    "type",
                    "turn_id",
                    "reply",
                    "scene",
                    "action",
                    "avatar_state",
                    "knowledge",
                },
                required={"type", "turn_id"},
            )
            turn = self._require_collecting_turn(event)
            reply = self._strict_object(
                event.get("reply", {}),
                "turn.commit.reply",
                allowed={"context", "provided_text"},
            )
            context = self._bounded_optional_text(
                reply.get("context"),
                "turn.commit.reply.context",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            provided_text = self._bounded_optional_text(
                reply.get("provided_text"),
                "turn.commit.reply.provided_text",
                max_chars=MAX_TURN_TEXT_CHARS,
            )
            reply_provided = "provided_text" in reply and provided_text is not None
            if reply and "text" not in self.modalities:
                raise ValueError("turn.commit.reply requires the text output")
            if reply_provided and turn.turn_origin != TURN_ORIGIN_PROACTIVE:
                raise ValueError(
                    "reply.provided_text is only supported for proactive turns"
                )
            if reply_provided and context is not None:
                raise ValueError(
                    "reply.provided_text and reply.context are mutually exclusive"
                )
            if (
                turn.turn_origin == TURN_ORIGIN_PROACTIVE
                and turn.trigger == ACTION_FINISHED_TRIGGER
            ):
                if "context" in reply:
                    raise ValueError(
                        "action_finished must not include reply.context"
                    )
                if "text" in self.modalities and (
                    "provided_text" not in reply or provided_text != ""
                ):
                    raise ValueError(
                        "action_finished requires reply.provided_text to be an "
                        "empty string when text output is enabled"
                    )

            scene = self._strict_object(
                event.get("scene", {}),
                "turn.commit.scene",
                allowed={"context", "reply_guidance"},
            )
            if scene and turn.turn_origin != TURN_ORIGIN_PROACTIVE:
                raise ValueError(
                    "turn.commit.scene is only supported for proactive turns"
                )
            if scene and turn.trigger == ACTION_FINISHED_TRIGGER:
                raise ValueError("action_finished must not include turn.commit.scene")
            scene_context = self._bounded_optional_text(
                scene.get("context"),
                "turn.commit.scene.context",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            scene_reply_guidance = self._bounded_optional_text(
                scene.get("reply_guidance"),
                "turn.commit.scene.reply_guidance",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            if scene_reply_guidance is not None and "text" not in self.modalities:
                raise ValueError(
                    "turn.commit.scene.reply_guidance requires the text output"
                )
            if reply_provided and scene:
                raise ValueError(
                    "reply.provided_text and turn.commit.scene are mutually exclusive"
                )

            knowledge = self._strict_object(
                event.get("knowledge", {}),
                "turn.commit.knowledge",
                allowed={"entity_hints", "script"},
            )
            script = self._strict_object(
                knowledge.get("script", {}),
                "turn.commit.knowledge.script",
                allowed={"id", "version", "checksum"},
                required={"id"} if knowledge.get("script") is not None else set(),
            )
            script_id = self._bounded_optional_text(
                script.get("id"), "turn.commit.knowledge.script.id",
                max_chars=MAX_KNOWLEDGE_SCRIPT_ID_CHARS, allow_empty=False,
            )
            script_version = script.get("version")
            if script_version is not None and (
                not isinstance(script_version, int)
                or isinstance(script_version, bool)
                or script_version < 1
            ):
                raise ValueError("turn.commit.knowledge.script.version must be a positive integer")
            script_checksum = self._bounded_optional_text(
                script.get("checksum"), "turn.commit.knowledge.script.checksum",
                max_chars=MAX_KNOWLEDGE_SCRIPT_CHECKSUM_CHARS, allow_empty=False,
            )
            if script_checksum is not None and (
                len(script_checksum) != 71
                or not script_checksum.startswith("sha256:")
                or any(char not in "0123456789abcdefABCDEF" for char in script_checksum[7:])
            ):
                raise ValueError(
                    "turn.commit.knowledge.script.checksum must be sha256: "
                    "followed by 64 hex characters"
                )
            if script and not reply_provided:
                raise ValueError(
                    "turn.commit.knowledge.script requires reply.provided_text"
                )
            if script and not provided_text:
                raise ValueError(
                    "turn.commit.knowledge.script requires non-empty reply.provided_text"
                )
            if script and provided_text:
                computed_script_checksum = "sha256:" + hashlib.sha256(
                    provided_text.encode("utf-8")
                ).hexdigest()
                if (
                    script_checksum is not None
                    and script_checksum.lower() != computed_script_checksum
                ):
                    raise ValueError(
                        "turn.commit.knowledge.script.checksum does not match "
                        "reply.provided_text"
                    )
                script_checksum = computed_script_checksum
            raw_hints = knowledge.get("entity_hints", [])
            if not isinstance(raw_hints, list):
                raise ValueError("turn.commit.knowledge.entity_hints must be a list")
            if len(raw_hints) > MAX_KNOWLEDGE_ENTITY_HINTS:
                raise ValueError(
                    "turn.commit.knowledge.entity_hints contains too many items"
                )
            knowledge_entity_hints: list[dict[str, str]] = []
            for index, raw_hint in enumerate(raw_hints):
                hint = self._strict_object(
                    raw_hint,
                    f"turn.commit.knowledge.entity_hints[{index}]",
                    allowed={"type", "external_id", "display_name"},
                    required={"type", "external_id"},
                )
                entity_type = self._bounded_optional_text(
                    hint.get("type"),
                    f"turn.commit.knowledge.entity_hints[{index}].type",
                    max_chars=MAX_KNOWLEDGE_ENTITY_FIELD_CHARS,
                    allow_empty=False,
                )
                external_id = self._bounded_optional_text(
                    hint.get("external_id"),
                    f"turn.commit.knowledge.entity_hints[{index}].external_id",
                    max_chars=MAX_KNOWLEDGE_ENTITY_FIELD_CHARS,
                    allow_empty=False,
                )
                display_name = self._bounded_optional_text(
                    hint.get("display_name"),
                    f"turn.commit.knowledge.entity_hints[{index}].display_name",
                    max_chars=MAX_KNOWLEDGE_ENTITY_FIELD_CHARS,
                )
                assert entity_type is not None and external_id is not None
                item = {
                    "type": entity_type.strip(),
                    "external_id": external_id.strip(),
                }
                if display_name is not None:
                    item["display_name"] = display_name.strip()
                knowledge_entity_hints.append(item)

            action = self._strict_object(
                event.get("action", {}),
                "turn.commit.action",
                allowed={
                    "last_executed_action_id",
                    "guidance",
                    "allowed_candidate_ids",
                    "excluded_candidate_ids",
                },
            )
            if action and "action" not in self.modalities:
                raise ValueError("turn.commit.action requires the action output")
            last_action_id = self._bounded_optional_text(
                action.get("last_executed_action_id"),
                "turn.commit.action.last_executed_action_id",
                max_chars=MAX_TURN_ID_CHARS,
                allow_empty=False,
            )
            if last_action_id is not None and self.global_action_catalog is not None:
                known_action_ids = {
                    candidate.action_id
                    for candidate in self.global_action_catalog.candidate_by_id.values()
                }
                if last_action_id.strip() not in known_action_ids:
                    raise ValueError(
                        "turn.commit.action.last_executed_action_id is not in "
                        f"the global action catalog: {last_action_id.strip()}"
                    )
            guidance = self._bounded_optional_text(
                action.get("guidance"),
                "turn.commit.action.guidance",
                max_chars=MAX_REPLY_CONTEXT_CHARS,
            )
            if guidance is not None and turn.turn_origin != TURN_ORIGIN_PROACTIVE:
                raise ValueError(
                    "turn.commit.action.guidance is only supported for proactive turns"
                )
            allowed_candidate_ids = self._bounded_candidate_id_list(
                action.get("allowed_candidate_ids"),
                "turn.commit.action.allowed_candidate_ids",
            )
            excluded_candidate_ids = self._bounded_candidate_id_list(
                action.get("excluded_candidate_ids"),
                "turn.commit.action.excluded_candidate_ids",
            )
            if (
                (allowed_candidate_ids or excluded_candidate_ids)
                and turn.turn_origin != TURN_ORIGIN_PROACTIVE
            ):
                raise ValueError(
                    "per-turn action candidate constraints are only supported "
                    "for proactive turns"
                )
            overlap = set(allowed_candidate_ids) & set(excluded_candidate_ids)
            if overlap:
                raise ValueError(
                    "turn.commit.action candidate constraints overlap: "
                    + ", ".join(sorted(overlap))
                )
            constrained_ids = (
                set(allowed_candidate_ids)
                if allowed_candidate_ids
                else set(self.candidate_by_id)
            )
            constrained_ids.difference_update(excluded_candidate_ids)
            if (allowed_candidate_ids or excluded_candidate_ids) and not constrained_ids:
                raise ValueError(
                    "turn.commit.action candidate constraints leave no "
                    "executable action"
                )

            avatar_state = event.get("avatar_state", {})
            if not isinstance(avatar_state, dict):
                raise ValueError("turn.commit.avatar_state must be an object")
            if {"current_action_id", "state_description"} & set(avatar_state):
                raise ValueError(
                    "turn.commit.avatar_state must not contain legacy action fields"
                )
            if self._serialized_chars(avatar_state) > MAX_AVATAR_STATE_CHARS:
                raise ValueError(
                    "turn.commit.avatar_state must contain at most "
                    f"{MAX_AVATAR_STATE_CHARS} serialized characters"
                )
            internal_state = dict(avatar_state)
            if guidance is not None:
                internal_state["state_description"] = guidance
            normalized = {
                "type": "turn.commit",
                "turn_id": self._wire_turn_id(event),
                "turn_origin": turn.turn_origin,
                "text_role": turn.text_role,
                "trigger": turn.trigger,
                "reply_context": context,
                "scene_context": scene_context,
                "scene_reply_guidance": scene_reply_guidance,
                "knowledge_entity_hints": knowledge_entity_hints,
                "knowledge_script_id": script_id.strip() if script_id else None,
                "knowledge_script_version": script_version,
                "knowledge_script_checksum": (
                    script_checksum.strip() if script_checksum else None
                ),
                "last_executed_action_id": (
                    last_action_id.strip() if last_action_id is not None else None
                ),
                "action_allowed_candidate_ids": allowed_candidate_ids,
                "action_excluded_candidate_ids": excluded_candidate_ids,
                "avatar_state": internal_state,
                "_reply_provided": reply_provided,
            }
            if reply_provided:
                normalized["text"] = provided_text
            return normalized
        if event_type == "turn.cancel":
            event = self._strict_object(
                payload,
                "turn.cancel",
                allowed={"type", "turn_id"},
                required={"type", "turn_id"},
            )
            return {
                "type": "turn.cancel",
                "turn_id": self._wire_turn_id(event),
            }
        if event_type == "session.close":
            self._require_started()
            event = self._strict_object(
                payload,
                "session.close",
                allowed={"type", "reason"},
                required={"type"},
            )
            self._bounded_optional_text(
                event.get("reason"),
                "session.close.reason",
                max_chars=MAX_TRIGGER_TYPE_CHARS,
                allow_empty=False,
            )
            return event
        raise ValueError(f"unsupported event type: {event_type!r}")


    @staticmethod
    def _validate_catalog_semantic_field(
        payload: dict[str, Any],
        field: str,
        expected: Any,
        *,
        entity_id: str,
    ) -> None:
        if field not in payload:
            return
        actual = payload[field]
        if field == "category_path":
            if not isinstance(actual, list) or not all(
                isinstance(item, str) for item in actual
            ):
                raise ValueError(f"category_path must be a string list: {entity_id!r}")
            actual = tuple(item.strip() for item in actual)
        elif isinstance(actual, str):
            actual = actual.strip()
        if actual != expected:
            raise ValueError(
                f"{field} does not match the global catalog: {entity_id!r}"
            )


    def _canonicalize_global_hierarchical_catalog(
        self,
        raw_categories: list[Any],
    ) -> tuple[list[SessionActionCategory], list[SessionActionCandidate]]:
        catalog = self.global_action_catalog
        if catalog is None:
            raise RuntimeError("global action catalog is not configured")
        categories: list[SessionActionCategory] = []
        candidates: list[SessionActionCandidate] = []
        for raw_category in raw_categories:
            if not isinstance(raw_category, dict):
                raise ValueError("action category must be an object")
            category_id = raw_category.get("category_id")
            if not isinstance(category_id, str) or not category_id.strip():
                raise ValueError("category_id must be a non-empty string")
            category_id = category_id.strip()
            global_category = catalog.category_by_id.get(category_id)
            if global_category is None:
                raise ValueError(
                    f"unknown category_id in global action catalog: {category_id!r}"
                )
            self._validate_catalog_semantic_field(
                raw_category,
                "source_label",
                global_category.source_label,
                entity_id=category_id,
            )
            self._validate_catalog_semantic_field(
                raw_category,
                "short_definition",
                global_category.short_definition,
                entity_id=category_id,
            )
            self._validate_catalog_semantic_field(
                raw_category,
                "category_path",
                global_category.category_path,
                entity_id=category_id,
            )
            raw_children = raw_category.get("children")
            if not isinstance(raw_children, list) or not raw_children:
                raise ValueError(
                    f"category children must be a non-empty list: {category_id!r}"
                )
            session_children: list[SessionActionCandidate] = []
            for raw_child in raw_children:
                parsed = SessionActionCandidate.from_payload(raw_child)
                global_child = catalog.candidate_for_category(
                    category_id, parsed.candidate_id
                )
                if global_child is None:
                    raise ValueError(
                        f"candidate_id {parsed.candidate_id!r} is not a member "
                        f"of global category {category_id!r}"
                    )
                if parsed.action_id != global_child.action_id:
                    raise ValueError(
                        "action_id does not match the global catalog: "
                        f"{parsed.candidate_id!r}"
                    )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "source_label",
                    global_child.source_label,
                    entity_id=parsed.candidate_id,
                )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "proactive_expression",
                    global_child.proactive_expression,
                    entity_id=parsed.candidate_id,
                )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "user_reaction_expression",
                    global_child.user_reaction_expression,
                    entity_id=parsed.candidate_id,
                )
                self._validate_catalog_semantic_field(
                    raw_child,
                    "short_definition",
                    global_child.source_short_definition,
                    entity_id=parsed.candidate_id,
                )
                child = SessionActionCandidate(
                    candidate_id=global_child.candidate_id,
                    action_id=global_child.action_id,
                    source_label=global_child.source_label,
                    short_definition=global_child.short_definition,
                    execution_binding=dict(parsed.execution_binding),
                    category_id=category_id,
                    proactive_expression=global_child.proactive_expression,
                    user_reaction_expression=(
                        global_child.user_reaction_expression
                    ),
                )
                session_children.append(child)
                candidates.append(child)
            categories.append(
                SessionActionCategory(
                    category_id=global_category.category_id,
                    source_label=global_category.source_label,
                    short_definition=global_category.short_definition,
                    category_path=global_category.category_path,
                    children=tuple(session_children),
                )
            )
        return categories, candidates


    def _canonicalize_global_flat_catalog(
        self,
        raw_candidates: list[Any],
    ) -> list[SessionActionCandidate]:
        catalog = self.global_action_catalog
        if catalog is None:
            raise RuntimeError("global action catalog is not configured")
        candidates: list[SessionActionCandidate] = []
        for raw_child in raw_candidates:
            parsed = SessionActionCandidate.from_payload(raw_child)
            global_child = catalog.candidate_by_id.get(parsed.candidate_id)
            if global_child is None:
                raise ValueError(
                    "unknown candidate_id in global action catalog: "
                    f"{parsed.candidate_id!r}"
                )
            if parsed.action_id != global_child.action_id:
                raise ValueError(
                    "action_id does not match the global catalog: "
                    f"{parsed.candidate_id!r}"
                )
            self._validate_catalog_semantic_field(
                raw_child,
                "source_label",
                global_child.source_label,
                entity_id=parsed.candidate_id,
            )
            self._validate_catalog_semantic_field(
                raw_child,
                "proactive_expression",
                global_child.proactive_expression,
                entity_id=parsed.candidate_id,
            )
            self._validate_catalog_semantic_field(
                raw_child,
                "user_reaction_expression",
                global_child.user_reaction_expression,
                entity_id=parsed.candidate_id,
            )
            self._validate_catalog_semantic_field(
                raw_child,
                "short_definition",
                global_child.source_short_definition,
                entity_id=parsed.candidate_id,
            )
            candidates.append(
                SessionActionCandidate(
                    candidate_id=global_child.candidate_id,
                    action_id=global_child.action_id,
                    source_label=global_child.source_label,
                    short_definition=global_child.short_definition,
                    execution_binding=dict(parsed.execution_binding),
                    category_id=global_child.category_id,
                    proactive_expression=global_child.proactive_expression,
                    user_reaction_expression=(
                        global_child.user_reaction_expression
                    ),
                )
            )
        return candidates


    @staticmethod
    def _event_context_id(payload: dict[str, Any], field: str) -> str | None:
        value = payload.get(field)
        return value if isinstance(value, str) and value.strip() else None


    @staticmethod
    def _classify_error(payload: dict[str, Any], exc: Exception) -> str:
        message = str(exc).lower()
        event_type = payload.get("type")
        if "protocol_version" in message:
            return "unsupported_protocol_version"
        if event_type == "session.start" and (
            "unsupported outputs" in message
            or "unsupported output modalities" in message
            or "audio output requires" in message
        ):
            return "unsupported_output"
        if "embedded tts provider is not configured" in message:
            return "tts_provider_not_configured"
        if "unsupported fields" in message or "missing required fields" in message:
            return "invalid_event_field"
        if (
            "turn_origin" in message
            or "text_role" in message
            or "proactive turn" in message
            or "user_input" in message
        ):
            return "invalid_turn_semantics"
        if "session_id is already active" in message:
            return "duplicate_session_id"
        if "another turn is already active" in message:
            return "duplicate_active_turn"
        if "turn already committed" in message:
            return "turn_already_committed"
        if "turn_id" in message:
            return "invalid_turn_id"
        if "seq" in message or "sequence" in message:
            return "invalid_sequence"
        if "audio" in message or "pcm16" in message:
            return "invalid_audio"
        if "image" in message or "mime_type" in message:
            return "invalid_image"
        if "session.start can only" in message or event_type == "session.start":
            return "session_candidate_invalid"
        if "action_candidates" in message or "candidate" in message:
            return "session_candidate_invalid"
        if (
            event_type
            in {
                "turn.commit",
                "input.audio.append",
                "input.image.append",
                "input.text.set",
                "turn.cancel",
            }
            and "turn.start" in message
        ):
            return "turn_already_committed"
        if "session.start" in message:
            return "session_not_started"
        return "invalid_event"


    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


    @staticmethod
    def _normalize_modalities(value: Any) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_MODALITIES
        if not isinstance(value, list) or not value:
            raise ValueError("modalities must be a non-empty list")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("modalities entries must be strings")
        if len(set(value)) != len(value):
            raise ValueError("modalities must not contain duplicates")
        unsupported = sorted(set(value) - SUPPORTED_MODALITIES)
        if unsupported:
            raise ValueError("unsupported output modalities: " + ", ".join(unsupported))
        return tuple(item for item in DEFAULT_MODALITIES if item in value)


    @staticmethod
    def _normalize_outputs(value: Any) -> tuple[str, ...]:
        return SessionOutputCapabilities.parse(value).outputs


    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value



from sglang_omni.serve.realtime.protocol.dispatch import ProtocolDispatchComponent
from sglang_omni.serve.realtime.protocol.session_start import SessionStartComponent


@compose_components(
    TurnInputComponent,
    ProtocolValidationComponent,
    ProtocolDispatchComponent,
    SessionStartComponent,
)
class ProtocolComponent:
    """Complete protocol surface composed from focused components."""


MultimodalValidationMixin = ProtocolValidationComponent
MultimodalProtocolMixin = ProtocolComponent
