"""Versioned scene semantics layered over immutable V116 action identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .catalog import ActionEntry, ResolvedCatalog


USAGE_SCENARIOS_PATH = (
    Path(__file__).resolve().parents[2] / "assets/action_usage_scenarios_v116.json"
)
BODY_USAGE_SCENARIOS_PATH = USAGE_SCENARIOS_PATH.with_name("action_usage_scenarios_v116_r4.json")


@dataclass(frozen=True)
class ActionUsageScenario:
    candidate_id: str
    label: str
    category_id: str
    code: str
    token_ids: tuple[int, int]
    meaning: str
    recall_summary: str
    required_evidence: str
    applicable_scenarios: tuple[str, ...]
    negative_scenarios: tuple[str, ...]
    confusable_candidate_ids: tuple[str, ...]
    distinction: str
    exclusion_summary: str = ""


@dataclass(frozen=True)
class UsageScenarioCatalog:
    version: str
    sha256: str
    base_catalog_version: str
    base_catalog_hash: str
    mapping_sha256: str
    entries: Mapping[str, ActionUsageScenario]

    def validate_catalog(self, catalog: ResolvedCatalog) -> None:
        if (
            catalog.catalog_version != self.base_catalog_version
            or catalog.catalog_hash != self.base_catalog_hash
            or catalog.mapping_hash != self.mapping_sha256
        ):
            raise ValueError("persona window scene semantics require the pinned V116 catalog and mapping")
        for entry in catalog.entries:
            self.for_entry(entry, catalog)

    def for_entry(
        self, entry: ActionEntry, catalog: ResolvedCatalog,
    ) -> ActionUsageScenario | None:
        if entry.candidate_id in {"IB0", "IF0", "000"}:
            return None
        scenario = self.entries.get(entry.candidate_id)
        if scenario is None:
            # Registered business extensions are not extra V116 body gestures.
            # Their authoritative binding remains owned by the policy registry.
            if (
                catalog.runtime_context.get("agent_event")
                and catalog.hard_bound is not None
                and entry.candidate_id == catalog.hard_bound.candidate_id
            ):
                return None
            raise ValueError(f"missing V116 action usage scenario: {entry.candidate_id}")
        if (
            scenario.code != entry.code
            or scenario.token_ids != entry.token_ids
            or scenario.label != entry.label
            or scenario.category_id != entry.category_id
        ):
            raise ValueError(f"V116 action identity differs from usage scenarios: {entry.candidate_id}")
        return scenario


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\x00|"):
        raise ValueError(f"action usage scenario {key} must be nonempty single-line text")
    return value


def parse_usage_scenarios(payload: bytes) -> UsageScenarioCatalog:
    raw = json.loads(payload)
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 2:
        raise ValueError("unsupported action usage scenario schema")
    entries = {}
    for row in raw.get("entries", []):
        if not isinstance(row, dict):
            raise ValueError("action usage scenario entries must be objects")
        fields = {
            key: _text(row, key) for key in (
                "candidate_id", "label", "category_id", "code", "meaning",
                "recall_summary", "required_evidence", "distinction",
            )
        }
        if raw.get("version") in {"v116-body-scenes-r3", "v116-body-scenes-r4"} and row.get("category_id") != "19":
            fields["exclusion_summary"] = _text(row, "exclusion_summary")
        lists = {}
        for key in ("applicable_scenarios", "negative_scenarios", "confusable_candidate_ids"):
            items = row.get(key)
            minimum = 1 if key == "confusable_candidate_ids" else 2
            if not isinstance(items, list) or not minimum <= len(items) <= 8:
                raise ValueError(f"action usage scenario {key} needs {minimum} to 8 entries")
            lists[key] = tuple(_text({"item": item}, "item") for item in items)
            if len(set(lists[key])) != len(items):
                raise ValueError(f"duplicate entries in action usage scenario {key}")
        pair = row.get("token_ids")
        if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int or v < 0 for v in pair):
            raise ValueError("action usage scenario requires two token IDs")
        entry = ActionUsageScenario(**fields, **lists, token_ids=tuple(pair))
        if entry.candidate_id in entries:
            raise ValueError(f"duplicate action usage scenario: {entry.candidate_id}")
        entries[entry.candidate_id] = entry
    if len(entries) != 116 or any(not cid.isdecimal() or cid == "000" for cid in entries):
        raise ValueError("V116 usage scenarios must describe exactly 116 concrete actions")
    for entry in entries.values():
        if any(cid not in entries or cid == entry.candidate_id for cid in entry.confusable_candidate_ids):
            raise ValueError(f"invalid similar-action reference: {entry.candidate_id}")
    return UsageScenarioCatalog(
        version=_text(raw, "version"), sha256=hashlib.sha256(payload).hexdigest(),
        base_catalog_version=_text(raw, "base_catalog_version"),
        base_catalog_hash=_text(raw, "base_catalog_hash"),
        mapping_sha256=_text(raw, "mapping_sha256"),
        entries=MappingProxyType(entries),
    )


@lru_cache(maxsize=1)
def load_usage_scenarios() -> UsageScenarioCatalog:
    return parse_usage_scenarios(USAGE_SCENARIOS_PATH.read_bytes())


@lru_cache(maxsize=1)
def load_body_usage_scenarios() -> UsageScenarioCatalog:
    """The revised body scenarios are opt-in through the v2 body renderer."""
    return parse_usage_scenarios(BODY_USAGE_SCENARIOS_PATH.read_bytes())
