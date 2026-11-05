"""Stateless performance input context and bounded character capability values."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import json

FACIAL_EXPRESSION_CATEGORY_ID = "19"
IMAGE_ROLE_USER_CAMERA = "user_camera"

MAX_ACTION_CHILDREN_PER_CATEGORY = 128

MAX_ACTION_PROFILE_CHARS = 8 * 1024

MAX_ACTION_PROFILE_FIELD_CHARS = 2 * 1024

MAX_PASSIVE_ACTION_POLICY_CHARS = 64 * 1024

MAX_CHARACTER_PROFILE_ROLE_CHARS = 5_000

ACTION_PERSONA_FIELDS = (
    "gender_expression",
    "visual_style",
    "role",
    "personality",
)

@dataclass(frozen=True, slots=True)
class SessionActionCandidate:
    candidate_id: str
    action_id: str
    source_label: str
    short_definition: str
    execution_binding: dict[str, str]
    category_id: str | None = None
    proactive_expression: str = ""
    user_reaction_expression: str = ""
    # Prompt display text only; canonical fields remain unchanged for matching.
    prompt_label: str = ""
    prompt_definition: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCandidate":
        if not isinstance(value, dict):
            raise ValueError("action candidate must be an object")
        candidate_id = value.get("candidate_id")
        action_id = value.get("action_id")
        source_label = value.get("source_label") or action_id
        short_definition = value.get("short_definition") or source_label
        proactive_expression = value.get("proactive_expression", "")
        user_reaction_expression = value.get(
            "user_reaction_expression", ""
        )
        binding = value.get("execution_binding") or {}
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"action_id must be non-empty: {candidate_id!r}")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(f"source_label must be non-empty: {candidate_id!r}")
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(f"short_definition must be non-empty: {candidate_id!r}")
        if proactive_expression is None:
            proactive_expression = ""
        if user_reaction_expression is None:
            user_reaction_expression = ""
        if not isinstance(proactive_expression, str):
            raise ValueError(
                f"proactive_expression must be a string: {candidate_id!r}"
            )
        if not isinstance(user_reaction_expression, str):
            raise ValueError(
                f"user_reaction_expression must be a string: {candidate_id!r}"
            )
        if not isinstance(binding, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in binding.items()
        ):
            raise ValueError(
                f"execution_binding must be a string dictionary: {candidate_id!r}"
            )
        return cls(
            candidate_id=candidate_id.strip(),
            action_id=action_id.strip(),
            source_label=source_label.strip(),
            short_definition=short_definition.strip(),
            execution_binding=dict(binding),
            proactive_expression=proactive_expression.strip(),
            user_reaction_expression=user_reaction_expression.strip(),
        )

    def effective_definition(self, turn_origin: str) -> str:
        contextual = (
            self.proactive_expression
            if turn_origin == "proactive"
            else self.user_reaction_expression
        )
        return contextual.strip() or self.prompt_definition or self.short_definition

    def definition_source(self, turn_origin: str) -> str:
        if turn_origin == "proactive" and self.proactive_expression.strip():
            return "proactive_expression"
        if turn_origin == "user" and self.user_reaction_expression.strip():
            return "user_reaction_expression"
        return "short_definition_fallback"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
            "source_label": self.source_label,
            "short_definition": self.short_definition,
            "proactive_expression": self.proactive_expression,
            "user_reaction_expression": self.user_reaction_expression,
            "execution_binding": dict(self.execution_binding),
            **({"category_id": self.category_id} if self.category_id else {}),
        }

@dataclass(frozen=True, slots=True)
class SessionActionProfile:
    persona: tuple[tuple[str, str], ...] = ()
    visual_behavior_preferences: str = ""
    category_preferences: str = ""
    action_preferences: str = ""
    passive_action_policy: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionProfile":
        if not isinstance(value, dict):
            raise ValueError("action_profile must be an object")
        allowed_fields = {
            "persona",
            "visual_behavior_preferences",
            "category_preferences",
            "action_preferences",
            "passive_action_policy",
        }
        unknown_fields = sorted(set(value) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "action_profile contains unsupported fields: "
                + ", ".join(unknown_fields)
            )

        raw_persona = value.get("persona", {})
        if not isinstance(raw_persona, dict):
            raise ValueError("action_profile.persona must be an object")
        unknown_persona_fields = sorted(set(raw_persona) - set(ACTION_PERSONA_FIELDS))
        if unknown_persona_fields:
            raise ValueError(
                "action_profile.persona contains unsupported fields: "
                + ", ".join(unknown_persona_fields)
            )
        persona: list[tuple[str, str]] = []
        for field_name in ACTION_PERSONA_FIELDS:
            if field_name not in raw_persona:
                continue
            field_value = raw_persona[field_name]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"action_profile.persona.{field_name} must be a non-empty string"
                )
            normalized = field_value.strip()
            max_chars = (
                MAX_CHARACTER_PROFILE_ROLE_CHARS
                if field_name == "role"
                else MAX_ACTION_PROFILE_FIELD_CHARS
            )
            if len(normalized) > max_chars:
                raise ValueError(
                    f"action_profile.persona.{field_name} must contain at most "
                    f"{max_chars} characters"
                )
            persona.append((field_name, normalized))

        preferences: dict[str, str] = {}
        for field_name in (
            "visual_behavior_preferences",
            "category_preferences",
            "action_preferences",
        ):
            field_value = value.get(field_name, "")
            if field_value is None:
                field_value = ""
            if not isinstance(field_value, str):
                raise ValueError(f"action_profile.{field_name} must be a string")
            normalized = field_value.strip()
            # D supplies the complete chapters 1, 2 and 5 role context here.
            # Use the existing role budget; category/action preferences keep
            # their smaller limit and the aggregate profile remains bounded.
            max_chars = (
                MAX_CHARACTER_PROFILE_ROLE_CHARS
                if field_name == "visual_behavior_preferences"
                else MAX_ACTION_PROFILE_FIELD_CHARS
            )
            if len(normalized) > max_chars:
                raise ValueError(
                    f"action_profile.{field_name} must contain at most "
                    f"{max_chars} characters"
                )
            preferences[field_name] = normalized

        passive_action_policy = value.get("passive_action_policy", "")
        if passive_action_policy is None:
            passive_action_policy = ""
        if not isinstance(passive_action_policy, str):
            raise ValueError("action_profile.passive_action_policy must be a string")
        passive_action_policy = passive_action_policy.strip()
        if len(passive_action_policy) > MAX_PASSIVE_ACTION_POLICY_CHARS:
            raise ValueError(
                "action_profile.passive_action_policy exceeds the protocol safety limit"
            )

        profile = cls(
            persona=tuple(persona),
            visual_behavior_preferences=preferences[
                "visual_behavior_preferences"
            ],
            category_preferences=preferences["category_preferences"],
            action_preferences=preferences["action_preferences"],
            passive_action_policy=passive_action_policy,
        )
        if not profile.persona and not (
            profile.visual_behavior_preferences
            or profile.category_preferences
            or profile.action_preferences
            or profile.passive_action_policy
        ):
            raise ValueError("action_profile must contain at least one non-empty field")
        base_payload = profile.as_dict()
        base_payload.pop("passive_action_policy", None)
        encoded = json.dumps(
            base_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded) > MAX_ACTION_PROFILE_CHARS:
            raise ValueError(
                f"action_profile must contain at most {MAX_ACTION_PROFILE_CHARS} "
                "serialized characters"
            )
        return profile

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.persona:
            payload["persona"] = dict(self.persona)
        if self.visual_behavior_preferences:
            payload["visual_behavior_preferences"] = (
                self.visual_behavior_preferences
            )
        if self.category_preferences:
            payload["category_preferences"] = self.category_preferences
        if self.action_preferences:
            payload["action_preferences"] = self.action_preferences
        if self.passive_action_policy:
            payload["passive_action_policy"] = self.passive_action_policy
        return payload

@dataclass(frozen=True, slots=True)
class SessionActionCategory:
    category_id: str
    source_label: str
    short_definition: str
    category_path: tuple[str, ...]
    children: tuple[SessionActionCandidate, ...]
    # Prompt display text only; canonical fields remain unchanged for matching.
    prompt_label: str = ""
    prompt_definition: str = ""

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCategory":
        if not isinstance(value, dict):
            raise ValueError("action category must be an object")
        category_id = value.get("category_id")
        source_label = value.get("source_label") or category_id
        short_definition = value.get("short_definition") or source_label
        category_path_value = value.get("category_path") or []
        if not isinstance(category_path_value, list) or not all(
            isinstance(item, str) and item.strip() for item in category_path_value
        ):
            raise ValueError(f"category_path must be a string list: {category_id!r}")
        children = value.get("children")
        if not isinstance(category_id, str) or not category_id.strip():
            raise ValueError("category_id must be a non-empty string")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(
                f"category source_label must be non-empty: {category_id!r}"
            )
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(
                f"category short_definition must be non-empty: {category_id!r}"
            )
        if not isinstance(children, list) or not children:
            raise ValueError(
                f"category children must be a non-empty list: {category_id!r}"
            )
        if len(children) > MAX_ACTION_CHILDREN_PER_CATEGORY:
            raise ValueError(
                f"category children must contain at most {MAX_ACTION_CHILDREN_PER_CATEGORY} items"
            )
        parsed = []
        for child in children:
            item = SessionActionCandidate.from_payload(child)
            parsed.append(
                SessionActionCandidate(
                    candidate_id=item.candidate_id,
                    action_id=item.action_id,
                    source_label=item.source_label,
                    short_definition=item.short_definition,
                    execution_binding=dict(item.execution_binding),
                    category_id=category_id.strip(),
                    proactive_expression=item.proactive_expression,
                    user_reaction_expression=item.user_reaction_expression,
                )
            )
        return cls(
            category_id=category_id.strip(),
            source_label=source_label.strip(),
            short_definition=short_definition.strip(),
            category_path=tuple(item.strip() for item in category_path_value),
            children=tuple(parsed),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "source_label": self.source_label,
            "short_definition": self.short_definition,
            "category_path": list(self.category_path),
            "children": [item.as_dict() for item in self.children],
        }

@dataclass(slots=True)
class PerformanceTurn:
    turn_id: str
    request_base: str
    trace_id: str
    text: str | None = None
    turn_origin: Literal["user", "proactive"] = "user"
    text_role: Literal["user_input", "character_reply"] = "user_input"
    intent: Any | None = None
