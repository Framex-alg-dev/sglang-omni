# SPDX-License-Identifier: Apache-2.0
"""Offline single-token identifiers for realtime action catalogs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal


MAPPING_SCHEMA_VERSION = 1
RUNTIME_CATALOG_SCHEMA_VERSION = 1
DEFAULT_ID_PATTERN = r"[A-Z]{2}"
DEFAULT_ALLOCATION_SALT = "sglang-omni-action-token-v1"


@dataclass(frozen=True, slots=True)
class SingleTokenId:
    text: str
    token_id: int


@dataclass(frozen=True, slots=True)
class CatalogEntity:
    kind: Literal["category", "action"]
    stable_key: str
    source_id: str
    source_label: str
    short_definition: str
    category_key: str | None = None
    action_id: str | None = None
    execution_binding: dict[str, str] | None = None


@dataclass(slots=True)
class ParsedCatalog:
    entities: list[CatalogEntity]
    category_order: list[str]
    action_order_by_category: dict[str, list[str]]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False)
    ]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint the effective vocabulary and special-token contract."""
    vocab = tokenizer.get_vocab()
    rows = sorted((int(token_id), str(token)) for token, token_id in vocab.items())
    payload = {
        "class": type(tokenizer).__name__,
        "vocab": rows,
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    return sha256_json(payload)


def tokenizer_metadata(tokenizer: Any, model_path: str) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    return {
        "model_path": model_path,
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "fingerprint": tokenizer_fingerprint(tokenizer),
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def discover_single_token_ids(
    tokenizer: Any,
    *,
    pattern: str = DEFAULT_ID_PATTERN,
    min_token_id: int = 0,
    max_token_id: int | None = None,
) -> list[SingleTokenId]:
    """Return protocol-safe strings that round-trip to one existing token."""
    if min_token_id < 0:
        raise ValueError("min_token_id must be non-negative")
    compiled = re.compile(rf"^(?:{pattern})$")
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    vocab = tokenizer.get_vocab()
    result: list[SingleTokenId] = []
    seen_text: set[str] = set()
    token_ids = sorted({int(value) for value in vocab.values()})
    for token_id in token_ids:
        if token_id < min_token_id:
            continue
        if max_token_id is not None and token_id > max_token_id:
            break
        if token_id in special_ids:
            continue
        text = _decode_one(tokenizer, token_id)
        if (
            not text
            or text in seen_text
            or text.strip() != text
            or not text.isprintable()
            or compiled.fullmatch(text) is None
            or _encode(tokenizer, text) != [token_id]
        ):
            continue
        seen_text.add(text)
        result.append(SingleTokenId(text=text, token_id=token_id))
    return sorted(result, key=lambda item: (item.text, item.token_id))


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _category_key(path: Iterable[str]) -> str:
    return "category:" + canonical_json(list(path))


def _action_key(action_id: str) -> str:
    return f"action:{action_id}"


def parse_canonical_action_catalog(
    document: dict[str, Any],
    *,
    category_depth: int = 2,
    add_no_action: bool = True,
) -> ParsedCatalog:
    """Convert the canonical ``actions`` document into stable mapping entities."""
    if category_depth <= 0:
        raise ValueError("category_depth must be positive")
    actions = document.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("catalog must contain a non-empty actions list")

    categories: dict[str, CatalogEntity] = {}
    category_paths: dict[str, tuple[str, ...]] = {}
    action_order_by_category: dict[str, list[str]] = {}
    action_entities: list[CatalogEntity] = []
    action_ids: set[str] = set()
    has_no_action = False

    for index, raw in enumerate(actions):
        if not isinstance(raw, dict):
            raise ValueError(f"actions[{index}] must be an object")
        action_id = _nonempty_string(raw.get("action_id"), f"actions[{index}].action_id")
        if action_id in action_ids:
            raise ValueError(f"duplicate action_id: {action_id!r}")
        action_ids.add(action_id)
        has_no_action = has_no_action or action_id == "no_action"

        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        source_label = _nonempty_string(
            source.get("source_label") or raw.get("source_label") or action_id,
            f"actions[{index}].source_label",
        )
        short_definition = _nonempty_string(
            raw.get("short_definition") or raw.get("prompt") or source_label,
            f"actions[{index}].short_definition",
        )
        raw_path = raw.get("category_path")
        if not isinstance(raw_path, list) or not raw_path:
            raw_path = ["未分类"]
        path = tuple(
            _nonempty_string(value, f"actions[{index}].category_path")
            for value in raw_path[:category_depth]
        )
        if len(path) < category_depth:
            path = (*path, *((path[-1],) * (category_depth - len(path))))
        category_key = _category_key(path)
        category_paths[category_key] = path
        categories.setdefault(
            category_key,
            CatalogEntity(
                kind="category",
                stable_key=category_key,
                source_id=" / ".join(path),
                source_label=path[-1],
                short_definition=" / ".join(path),
            ),
        )
        binding = raw.get("execution_binding")
        if binding is None:
            binding = {}
        if not isinstance(binding, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in binding.items()
        ):
            raise ValueError(
                f"actions[{index}].execution_binding must be a string dictionary"
            )
        stable_key = _action_key(action_id)
        action_entities.append(
            CatalogEntity(
                kind="action",
                stable_key=stable_key,
                source_id=action_id,
                source_label=source_label,
                short_definition=short_definition,
                category_key=category_key,
                action_id=action_id,
                execution_binding=dict(binding),
            )
        )
        action_order_by_category.setdefault(category_key, []).append(stable_key)

    if add_no_action and not has_no_action:
        path = ("系统动作",) * category_depth
        category_key = _category_key(path)
        category_paths[category_key] = path
        categories[category_key] = CatalogEntity(
            kind="category",
            stable_key=category_key,
            source_id=" / ".join(path),
            source_label="系统动作",
            short_definition="保持当前姿态",
        )
        stable_key = _action_key("no_action")
        action_entities.append(
            CatalogEntity(
                kind="action",
                stable_key=stable_key,
                source_id="no_action",
                source_label="不做动作",
                short_definition="保持当前姿态",
                category_key=category_key,
                action_id="no_action",
                execution_binding={},
            )
        )
        action_order_by_category[category_key] = [stable_key]

    category_order = sorted(categories, key=lambda key: category_paths[key])
    entities = [categories[key] for key in category_order] + action_entities
    stable_keys = [entity.stable_key for entity in entities]
    if len(stable_keys) != len(set(stable_keys)):
        raise ValueError("catalog entity stable keys must be globally unique")
    return ParsedCatalog(
        entities=entities,
        category_order=category_order,
        action_order_by_category=action_order_by_category,
    )


def _pool_by_text(pool: Iterable[SingleTokenId]) -> dict[str, SingleTokenId]:
    result: dict[str, SingleTokenId] = {}
    token_ids: set[int] = set()
    for item in pool:
        if item.text in result or item.token_id in token_ids:
            raise ValueError("single-token pool must have unique strings and token IDs")
        result[item.text] = item
        token_ids.add(item.token_id)
    return result


def _assignment_candidates(
    stable_key: str,
    pool: Iterable[SingleTokenId],
    allocation_salt: str,
) -> list[SingleTokenId]:
    return sorted(
        pool,
        key=lambda item: hashlib.sha256(
            f"{allocation_salt}\0{stable_key}\0{item.text}\0{item.token_id}".encode(
                "utf-8"
            )
        ).digest(),
    )


def allocate_single_token_ids(
    entities: Iterable[CatalogEntity],
    pool: list[SingleTokenId],
    *,
    allocation_salt: str = DEFAULT_ALLOCATION_SALT,
    previous_mapping: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Assign globally unique tokens while preserving previous and retired IDs."""
    entity_list = list(entities)
    entity_by_key = {entity.stable_key: entity for entity in entity_list}
    if len(entity_by_key) != len(entity_list):
        raise ValueError("catalog entity stable keys must be unique")
    pool_by_text = _pool_by_text(pool)
    occupied_text: set[str] = set()
    occupied_token_ids: set[int] = set()
    records: dict[str, dict[str, Any]] = {}

    previous_entries = (
        previous_mapping.get("entries", [])
        if isinstance(previous_mapping, dict)
        else []
    )
    if not isinstance(previous_entries, list):
        raise ValueError("previous mapping entries must be a list")
    for raw in previous_entries:
        if not isinstance(raw, dict):
            raise ValueError("previous mapping entry must be an object")
        stable_key = _nonempty_string(raw.get("stable_key"), "stable_key")
        short_id = _nonempty_string(raw.get("short_id"), "short_id")
        token_id = raw.get("token_id")
        pool_item = pool_by_text.get(short_id)
        if (
            not isinstance(token_id, int)
            or pool_item is None
            or pool_item.token_id != token_id
        ):
            raise ValueError(
                f"previous mapping token is not valid under current policy: {stable_key!r}"
            )
        if stable_key in records:
            raise ValueError(f"duplicate previous stable_key: {stable_key!r}")
        if short_id in occupied_text or token_id in occupied_token_ids:
            raise ValueError("previous mapping reuses a short ID or token ID")
        occupied_text.add(short_id)
        occupied_token_ids.add(token_id)
        records[stable_key] = dict(raw)

    required_new = sum(key not in records for key in entity_by_key)
    free_count = len(pool) - len(occupied_text)
    if required_new > free_count:
        raise ValueError(
            "single-token pool capacity is insufficient: "
            f"need {required_new} new IDs, have {free_count} free "
            f"({len(pool)} total, {len(occupied_text)} reserved)"
        )

    for stable_key in sorted(entity_by_key):
        if stable_key in records:
            continue
        for item in _assignment_candidates(stable_key, pool, allocation_salt):
            if item.text in occupied_text or item.token_id in occupied_token_ids:
                continue
            occupied_text.add(item.text)
            occupied_token_ids.add(item.token_id)
            records[stable_key] = {
                "stable_key": stable_key,
                "short_id": item.text,
                "token_id": item.token_id,
            }
            break

    output: list[dict[str, Any]] = []
    for stable_key in sorted(records):
        record = records[stable_key]
        entity = entity_by_key.get(stable_key)
        if entity is None:
            output.append({**record, "active": False})
            continue
        output.append(
            {
                "kind": entity.kind,
                "stable_key": stable_key,
                "source_id": entity.source_id,
                "short_id": record["short_id"],
                "token_id": int(record["token_id"]),
                "active": True,
            }
        )
    return output


def assignment_hash(entries: Iterable[dict[str, Any]]) -> str:
    rows = [
        {
            "stable_key": entry["stable_key"],
            "short_id": entry["short_id"],
            "token_id": entry["token_id"],
            "active": bool(entry.get("active", True)),
        }
        for entry in entries
    ]
    return sha256_json(sorted(rows, key=lambda row: row["stable_key"]))


def build_mapping_manifest(
    parsed: ParsedCatalog,
    entries: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model_path: str,
    source_catalog_path: str,
    source_catalog_sha256: str,
    pattern: str,
    min_token_id: int,
    max_token_id: int | None,
    allocation_salt: str,
    pool_size: int,
    category_depth: int,
    add_no_action: bool,
) -> dict[str, Any]:
    active_count = sum(bool(entry.get("active", True)) for entry in entries)
    category_count = sum(entity.kind == "category" for entity in parsed.entities)
    action_count = sum(entity.kind == "action" for entity in parsed.entities)
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "tokenizer": tokenizer_metadata(tokenizer, model_path),
        "source_catalog": {
            "path": source_catalog_path,
            "sha256": source_catalog_sha256,
        },
        "policy": {
            "pattern": pattern,
            "min_token_id": min_token_id,
            "max_token_id": max_token_id,
            "allocation_salt": allocation_salt,
            "single_token_required": True,
            "global_uniqueness": "category_id+candidate_id",
            "category_depth": category_depth,
            "add_no_action": add_no_action,
        },
        "counts": {
            "categories": category_count,
            "actions": action_count,
            "active": active_count,
            "retired": len(entries) - active_count,
            "pool_size": pool_size,
            "pool_remaining": pool_size - len(entries),
        },
        "assignment_sha256": assignment_hash(entries),
        "entries": entries,
        "warnings": [
            "Single-token IDs reduce suffix length but retain pretrained token-prior bias.",
            "Run action-selection accuracy evaluation before production rollout.",
            "The scorer appends a terminal token, so a one-token ID usually yields two scored suffix tokens.",
        ],
    }


def build_runtime_catalog(
    parsed: ParsedCatalog,
    mapping_manifest: dict[str, Any],
) -> dict[str, Any]:
    entries = mapping_manifest.get("entries", [])
    active_by_key = {
        entry["stable_key"]: entry
        for entry in entries
        if entry.get("active", True)
    }
    entity_by_key = {entity.stable_key: entity for entity in parsed.entities}
    missing = sorted(set(entity_by_key) - set(active_by_key))
    if missing:
        raise ValueError(f"mapping is missing active catalog entities: {missing[:3]!r}")

    categories: list[dict[str, Any]] = []
    for category_key in parsed.category_order:
        category = entity_by_key[category_key]
        children: list[dict[str, Any]] = []
        for action_key in parsed.action_order_by_category.get(category_key, []):
            action = entity_by_key[action_key]
            child = {
                "candidate_id": active_by_key[action_key]["short_id"],
                "action_id": action.action_id,
                "source_label": action.source_label,
                "short_definition": action.short_definition,
            }
            if action.execution_binding:
                child["execution_binding"] = dict(action.execution_binding)
            children.append(child)
        categories.append(
            {
                "category_id": active_by_key[category_key]["short_id"],
                "source_label": category.source_label,
                "short_definition": category.short_definition,
                "children": children,
            }
        )
    action_candidates_hash = sha256_json(categories)
    return {
        "schema_version": RUNTIME_CATALOG_SCHEMA_VERSION,
        "source_catalog": dict(mapping_manifest["source_catalog"]),
        "tokenizer_fingerprint": mapping_manifest["tokenizer"]["fingerprint"],
        "assignment_sha256": mapping_manifest["assignment_sha256"],
        "action_candidates_sha256": action_candidates_hash,
        "action_candidates": categories,
    }


def validate_mapping_manifest(
    tokenizer: Any,
    manifest: dict[str, Any],
    *,
    parsed: ParsedCatalog | None = None,
    require_fingerprint_match: bool = True,
) -> dict[str, Any]:
    if manifest.get("schema_version") != MAPPING_SCHEMA_VERSION:
        raise ValueError("unsupported action token mapping schema_version")
    tokenizer_info = manifest.get("tokenizer")
    if not isinstance(tokenizer_info, dict):
        raise ValueError("mapping tokenizer metadata must be an object")
    actual_fingerprint = tokenizer_fingerprint(tokenizer)
    expected_fingerprint = tokenizer_info.get("fingerprint")
    if require_fingerprint_match and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "tokenizer fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("mapping entries must be a non-empty list")
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("mapping policy must be an object")
    pattern = policy.get("pattern")
    min_token_id = policy.get("min_token_id", 0)
    max_token_id = policy.get("max_token_id")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("mapping policy pattern must be a non-empty string")
    if not isinstance(min_token_id, int) or min_token_id < 0:
        raise ValueError("mapping policy min_token_id must be non-negative")
    if max_token_id is not None and not isinstance(max_token_id, int):
        raise ValueError("mapping policy max_token_id must be an integer or null")
    compiled_pattern = re.compile(rf"^(?:{pattern})$")
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    stable_keys: set[str] = set()
    short_ids: set[str] = set()
    token_ids: set[int] = set()
    failures: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("mapping entry must be an object")
        stable_key = entry.get("stable_key")
        short_id = entry.get("short_id")
        token_id = entry.get("token_id")
        if not isinstance(stable_key, str) or not stable_key:
            failures.append("entry has invalid stable_key")
            continue
        if not isinstance(short_id, str) or not short_id:
            failures.append(f"{stable_key}: invalid short_id")
            continue
        if not isinstance(token_id, int):
            failures.append(f"{stable_key}: invalid token_id")
            continue
        if compiled_pattern.fullmatch(short_id) is None:
            failures.append(f"{stable_key}: short_id violates mapping pattern")
        if token_id < min_token_id or (
            max_token_id is not None and token_id > max_token_id
        ):
            failures.append(f"{stable_key}: token_id violates mapping range")
        if token_id in special_ids:
            failures.append(f"{stable_key}: token_id is special")
        if stable_key in stable_keys:
            failures.append(f"duplicate stable_key: {stable_key}")
        if short_id in short_ids:
            failures.append(f"duplicate short_id: {short_id}")
        if token_id in token_ids:
            failures.append(f"duplicate token_id: {token_id}")
        stable_keys.add(stable_key)
        short_ids.add(short_id)
        token_ids.add(token_id)
        encoded = _encode(tokenizer, short_id)
        if encoded != [token_id]:
            failures.append(
                f"{stable_key}: {short_id!r} encodes to {encoded}, expected [{token_id}]"
            )
        elif _decode_one(tokenizer, token_id) != short_id:
            failures.append(f"{stable_key}: token decode is not an exact round trip")

    expected_assignment_hash = manifest.get("assignment_sha256")
    actual_assignment_hash = assignment_hash(entries)
    if expected_assignment_hash != actual_assignment_hash:
        failures.append(
            "assignment_sha256 mismatch: "
            f"expected {expected_assignment_hash}, got {actual_assignment_hash}"
        )
    if parsed is not None:
        expected_active = {entity.stable_key for entity in parsed.entities}
        actual_active = {
            entry["stable_key"]
            for entry in entries
            if entry.get("active", True)
        }
        if expected_active != actual_active:
            missing = sorted(expected_active - actual_active)
            extra = sorted(actual_active - expected_active)
            failures.append(
                f"active catalog keys mismatch: missing={missing[:3]!r} extra={extra[:3]!r}"
            )
    if failures:
        raise ValueError("action token mapping validation failed: " + "; ".join(failures))
    return {
        "valid": True,
        "entry_count": len(entries),
        "active_count": sum(entry.get("active", True) for entry in entries),
        "retired_count": sum(not entry.get("active", True) for entry in entries),
        "tokenizer_fingerprint": actual_fingerprint,
        "assignment_sha256": actual_assignment_hash,
    }


__all__ = [
    "CatalogEntity",
    "DEFAULT_ALLOCATION_SALT",
    "DEFAULT_ID_PATTERN",
    "MAPPING_SCHEMA_VERSION",
    "ParsedCatalog",
    "SingleTokenId",
    "allocate_single_token_ids",
    "assignment_hash",
    "build_mapping_manifest",
    "build_runtime_catalog",
    "canonical_json",
    "discover_single_token_ids",
    "parse_canonical_action_catalog",
    "sha256_json",
    "tokenizer_fingerprint",
    "tokenizer_metadata",
    "validate_mapping_manifest",
]
