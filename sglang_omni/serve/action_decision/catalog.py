"""Load immutable action artifacts and derive exact per-channel candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ActionDecisionRequest, DecisionChannel


_BODY_NO_ACTION = "IB0"
_EXPRESSION_KEEP = "IF0"
_UNSUPPORTED = "000"
_FACE_OCCUPYING_BODY_IDS = frozenset(
    {"277", "287", "322", "324", "335", "336", "354", "444", "449", "450", "455", "468"}
)


@dataclass(frozen=True)
class ActionEntry:
    candidate_id: str
    action_id: str | None
    category_id: str | None
    label: str
    definition: str
    code: str
    token_ids: tuple[int, int]
    kind: str
    occupies_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedCatalog:
    entries: tuple[ActionEntry, ...]
    runtime_context: dict[str, Any]
    hard_bound: ActionEntry | None
    catalog_version: str
    catalog_hash: str
    mapping_version: str
    mapping_hash: str

    @property
    def by_code(self) -> dict[str, ActionEntry]:
        return {entry.code: entry for entry in self.entries}

    @property
    def by_candidate_id(self) -> dict[str, ActionEntry]:
        return {entry.candidate_id: entry for entry in self.entries}


class ActionCatalogRegistry:
    """Validated product catalog plus registered orchestrator extensions."""

    def __init__(
        self,
        *,
        mapping_path: str | Path,
        full_mapping_path: str | Path,
        product_catalog_path: str | Path,
        agent_policy_path: str | Path,
    ) -> None:
        current = _object(mapping_path)
        full = _object(full_mapping_path)
        product = _object(product_catalog_path)
        policy = _object(agent_policy_path)
        self.catalog_version = str(product["action_catalog_version"])
        self.catalog_hash = str(product["action_catalog_hash"])
        self.mapping_version = str(current["mapping_version"])
        self.mapping_hash = str(current["assignment_sha256"])
        self._mapping_by_id = _mapping_entries(current)
        self._full_mapping_by_id = _mapping_entries(full)
        self._product_by_id: dict[str, dict[str, Any]] = {}
        self._product_channel: dict[str, DecisionChannel] = {}
        for category in product.get("categories", []):
            category_id = str(category["category_id"])
            channel = (
                DecisionChannel.EXPRESSION
                if category_id == "19"
                else DecisionChannel.BODY
            )
            for child in category.get("children", []):
                candidate_id = str(child["candidate_id"])
                if candidate_id in self._product_by_id:
                    raise ValueError(f"duplicate product candidate_id: {candidate_id}")
                self._product_by_id[candidate_id] = {
                    **child,
                    "category_id": category_id,
                }
                self._product_channel[candidate_id] = channel
        bindings = policy.get("bindings", {})
        if not isinstance(bindings, dict):
            raise ValueError("agent action policy bindings must be an object")
        self._agent_bindings = bindings
        self._validate_artifacts()

    def resolve(self, request: ActionDecisionRequest) -> ResolvedCatalog:
        context = _normalize_context(request.runtime_context)
        if request.reply_prefix:
            context.setdefault("reply_prefix", request.reply_prefix)
        agent_event = context.get("agent_event")
        injected: set[str] = set()
        if agent_event:
            binding = self._agent_bindings.get(str(agent_event))
            if not isinstance(binding, dict):
                raise ValueError(f"unregistered agent_event: {agent_event}")
            injected.update(str(item) for item in binding.get("inject_candidate_ids", []))
            registered_context = _normalize_context(binding.get("runtime_context", {}))
            for key, value in registered_context.items():
                if key in context and context[key] != value:
                    raise ValueError(f"runtime_context conflicts with registered {key}")
                context[key] = value

        product_ids = {
            candidate_id
            for candidate_id, channel in self._product_channel.items()
            if channel is request.channel
        }
        injected_for_channel = {
            item
            for item in injected
            if self._channel_for_mapping(item) is request.channel
        }
        allowed = product_ids | injected_for_channel
        context_allowed = _ids(context.get("allowed_candidate_ids"))
        request_allowed = set(request.allowed_candidate_ids)
        if context_allowed:
            allowed.intersection_update(context_allowed)
        if request_allowed:
            allowed.intersection_update(request_allowed)
        excluded = _ids(context.get("excluded_candidate_ids")) | set(
            request.excluded_candidate_ids
        )
        allowed.difference_update(excluded)

        entries = [self._entry(candidate_id) for candidate_id in sorted(allowed)]
        controls = (
            (_BODY_NO_ACTION, _UNSUPPORTED)
            if request.channel is DecisionChannel.BODY
            else (_EXPRESSION_KEEP, _UNSUPPORTED)
        )
        entries.extend(self._entry(candidate_id) for candidate_id in controls)

        required = context.get("required_action_candidate_id")
        hard_bound = None
        if required is not None:
            required_id = str(required)
            required_channel = self._channel_for_mapping(required_id)
            if required_channel is request.channel:
                if required_id in excluded:
                    raise ValueError(f"required candidate {required_id} is excluded")
                if required_id not in allowed:
                    raise ValueError(
                        f"required candidate {required_id} is outside the {request.channel.value} catalog"
                    )
                hard_bound = self._entry(required_id)
            else:
                # The same trusted decision-point context is sent to both
                # independent calls. A body binding must not constrain or
                # invalidate the expression call, and vice versa.
                context.pop("required_action_candidate_id", None)
                context.pop("required_action_semantics", None)
        _validate_entries(entries)
        return ResolvedCatalog(
            entries=tuple(entries),
            runtime_context=context,
            hard_bound=hard_bound,
            catalog_version=self.catalog_version,
            catalog_hash=self.catalog_hash,
            mapping_version=self.mapping_version,
            mapping_hash=self.mapping_hash,
        )

    def control(self, channel: DecisionChannel, *, unsupported: bool = False) -> ActionEntry:
        candidate_id = _UNSUPPORTED if unsupported else (
            _BODY_NO_ACTION if channel is DecisionChannel.BODY else _EXPRESSION_KEEP
        )
        return self._entry(candidate_id)

    def verify_tokenizer(self, tokenizer_path: str | Path) -> None:
        """Fail startup when artifact codes no longer encode to recorded pairs."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(tokenizer_path).expanduser().resolve()),
            local_files_only=True,
            trust_remote_code=True,
        )
        errors: list[str] = []
        for raw in self._full_mapping_by_id.values():
            expected = tuple(int(item) for item in raw["token_ids"])
            actual = tuple(
                tokenizer.encode(str(raw["output_text"]), add_special_tokens=False)
            )
            if actual != expected:
                errors.append(
                    f"{raw['candidate_id']}: {raw['output_text']!r} -> {actual}, expected {expected}"
                )
                if len(errors) == 10:
                    break
        if errors:
            raise ValueError("action tokenizer does not match mapping:\n" + "\n".join(errors))

    def _entry(self, candidate_id: str) -> ActionEntry:
        raw = self._mapping_by_id.get(candidate_id) or self._full_mapping_by_id.get(candidate_id)
        if raw is None:
            raise ValueError(f"candidate {candidate_id} is absent from the action mapping")
        product = self._product_by_id.get(candidate_id, {})
        category_id = raw.get("canonical_category_id")
        return ActionEntry(
            candidate_id=candidate_id,
            action_id=str(raw.get("action_id")) if raw.get("action_id") is not None else None,
            category_id=str(category_id) if category_id is not None else None,
            label=str(product.get("source_label") or raw.get("source_label") or candidate_id),
            definition=str(
                product.get("user_reaction_expression")
                or product.get("short_definition")
                or raw.get("short_definition")
                or ""
            ),
            code=str(raw["output_text"]),
            token_ids=tuple(int(item) for item in raw["token_ids"]),
            kind=str(raw.get("entry_kind", "action")),
            occupies_channels=("face",) if candidate_id in _FACE_OCCUPYING_BODY_IDS else (),
        )

    def _channel_for_mapping(self, candidate_id: str) -> DecisionChannel:
        raw = self._full_mapping_by_id.get(candidate_id)
        if raw is None:
            raise ValueError(f"injected candidate {candidate_id} is absent from full mapping")
        return (
            DecisionChannel.EXPRESSION
            if str(raw.get("canonical_category_id")) == "19"
            else DecisionChannel.BODY
        )

    def _validate_artifacts(self) -> None:
        missing = set(self._product_by_id) - set(self._full_mapping_by_id)
        if missing:
            raise ValueError(f"product candidates missing from full mapping: {sorted(missing)}")
        for control in (_BODY_NO_ACTION, _EXPRESSION_KEEP, _UNSUPPORTED):
            entry = self._entry(control)
            if len(entry.token_ids) != 2:
                raise ValueError(f"control {control} is not a two-token output")


def _object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"action artifact must be an object: {path}")
    return value


def _mapping_entries(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in [*raw.get("action_entries", []), *raw.get("control_entries", [])]:
        if item.get("active", True):
            result[str(item["candidate_id"])] = dict(item)
    return result


def _ids(value: Any) -> set[str]:
    if value in (None, "", []):
        return set()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("candidate constraints must be arrays")
    return {str(item) for item in value}


def _normalize_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("runtime_context must be an object")
    allowed = {
        "turn_origin", "trigger", "reply_status", "reply_prefix", "body_intent",
        "body_task", "visual_route", "scene_context", "scene_reply_guidance",
        "proactive_scene", "avatar_state_description", "last_executed_action_id",
        "allowed_candidate_ids", "excluded_candidate_ids", "knowledge_state",
        "entity_state", "script_state", "agent_event", "agent_task", "agent_phase",
        "tool_name", "tool_status", "display_state", "required_action_candidate_id",
        "required_action_semantics", "expression_priority", "prompt_profile",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("unsupported runtime_context fields: " + ", ".join(unknown))
    return {str(key): item for key, item in value.items() if item not in (None, "", [])}


def _validate_entries(entries: Iterable[ActionEntry]) -> None:
    ids: set[str] = set()
    codes: set[str] = set()
    token_pairs: set[tuple[int, int]] = set()
    for entry in entries:
        if len(entry.token_ids) != 2:
            raise ValueError(f"candidate {entry.candidate_id} is not a two-token output")
        if entry.candidate_id in ids or entry.code in codes or entry.token_ids in token_pairs:
            raise ValueError(f"duplicate action output: {entry.candidate_id}")
        ids.add(entry.candidate_id)
        codes.add(entry.code)
        token_pairs.add(entry.token_ids)
