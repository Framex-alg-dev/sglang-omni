"""Validate immutable prompt snapshots supplied by the Agent service."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

ACTION_CONSUMER = "d_video_call.action.decision"
ACTION_ASSEMBLY_CONTRACT = "d-action-decision-v1"
ACTION_OUTPUT_PROTOCOL = "d-action-decision-response-v1"
TIMELINE_CONSUMER = "d_video_call.timeline.detection"
TIMELINE_ASSEMBLY_CONTRACT = "d-timeline-detection-v1"
TIMELINE_OUTPUT_PROTOCOL = "d-timeline-observation-v2"
ROUTER_CONSUMER = "sglang_omni.router.v1_0"
ROUTER_ASSEMBLY_CONTRACT = "turn-router-v1.0"
ROUTER_OUTPUT_PROTOCOL = "turn-router-code-v1.0"

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AgentPromptBinding:
    assembly_contract: str
    output_protocol: str
    model_id: str
    model_version: str


@dataclass(frozen=True)
class ActionPromptSnapshot:
    release_id: str
    bundle_sha256: str
    binding: AgentPromptBinding
    system_prompt: str
    system_prompt_sha256: str

    @property
    def rules(self) -> Mapping[str, str]:
        return {"system_prompt": self.system_prompt}


@dataclass(frozen=True)
class TimelinePromptSnapshot:
    release_id: str
    bundle_sha256: str
    binding: AgentPromptBinding
    system_prompt: str
    event_registry_json: str
    event_codebook_csv: str
    item_sha256: Mapping[str, str]


@dataclass(frozen=True)
class RouterPromptSnapshot:
    """One immutable Router v1.0 prompt supplied by the Agent service."""

    release_id: str
    bundle_sha256: str
    binding: AgentPromptBinding
    system_prompt: str
    mcp_scope: str
    multimodal_scope: str
    item_sha256: Mapping[str, str]

    @property
    def rules(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "system_prompt": self.system_prompt,
                "mcp_scope": self.mcp_scope,
                "multimodal_scope": self.multimodal_scope,
            }
        )


def decode_action_prompt_snapshot(
    raw: Any,
) -> ActionPromptSnapshot:
    expected = {
        "consumer",
        "release_id",
        "bundle_sha256",
        "binding",
        "system_prompt",
        "system_prompt_sha256",
    }
    value = _object_with_exact_keys(raw, expected, "action prompt_snapshot")
    binding = _binding(
        value["binding"],
        assembly_contract=ACTION_ASSEMBLY_CONTRACT,
        output_protocol=ACTION_OUTPUT_PROTOCOL,
    )
    if value["consumer"] != ACTION_CONSUMER:
        raise ValueError("action prompt_snapshot consumer mismatch")
    prompt = _prompt_text(value["system_prompt"], "system_prompt")
    prompt_sha256 = _digest(value["system_prompt_sha256"], "system_prompt_sha256")
    if hashlib.sha256(prompt.encode()).hexdigest() != prompt_sha256:
        raise ValueError("action prompt_snapshot content checksum mismatch")
    return ActionPromptSnapshot(
        release_id=_release(value["release_id"]),
        bundle_sha256=_digest(value["bundle_sha256"], "bundle_sha256"),
        binding=binding,
        system_prompt=prompt,
        system_prompt_sha256=prompt_sha256,
    )


def decode_timeline_prompt_snapshot(
    raw: Any,
) -> TimelinePromptSnapshot:
    expected = {
        "consumer",
        "release_id",
        "bundle_sha256",
        "binding",
        "system_prompt",
        "event_registry_json",
        "event_codebook_csv",
        "item_sha256",
    }
    value = _object_with_exact_keys(raw, expected, "timeline prompt_snapshot")
    binding = _binding(
        value["binding"],
        assembly_contract=TIMELINE_ASSEMBLY_CONTRACT,
        output_protocol=TIMELINE_OUTPUT_PROTOCOL,
    )
    if value["consumer"] != TIMELINE_CONSUMER:
        raise ValueError("timeline prompt_snapshot consumer mismatch")
    texts = {
        key: _prompt_text(value[key], key)
        for key in ("system_prompt", "event_registry_json", "event_codebook_csv")
    }
    hashes = _object_with_exact_keys(
        value["item_sha256"],
        set(texts),
        "timeline prompt item_sha256",
    )
    checked_hashes: dict[str, str] = {}
    for key, content in texts.items():
        digest = _digest(hashes[key], f"item_sha256.{key}")
        if hashlib.sha256(content.encode()).hexdigest() != digest:
            raise ValueError(f"timeline prompt_snapshot {key} checksum mismatch")
        checked_hashes[key] = digest
    return TimelinePromptSnapshot(
        release_id=_release(value["release_id"]),
        bundle_sha256=_digest(value["bundle_sha256"], "bundle_sha256"),
        binding=binding,
        item_sha256=MappingProxyType(checked_hashes),
        **texts,
    )


def decode_router_prompt_snapshot(raw: Any) -> RouterPromptSnapshot:
    """Validate an Agent-owned Router snapshot without owning Knowledge I/O."""

    expected = {
        "consumer",
        "release_id",
        "bundle_sha256",
        "binding",
        "system_prompt",
        "mcp_scope",
        "multimodal_scope",
        "item_sha256",
    }
    value = _object_with_exact_keys(raw, expected, "router prompt_snapshot")
    binding_value = _object_with_exact_keys(
        value["binding"],
        {"assembly_contract", "output_protocol", "model_id", "model_version"},
        "router prompt binding",
    )
    if (
        binding_value["assembly_contract"] != ROUTER_ASSEMBLY_CONTRACT
        or binding_value["output_protocol"] != ROUTER_OUTPUT_PROTOCOL
    ):
        raise ValueError("router prompt_snapshot protocol binding mismatch")
    binding = AgentPromptBinding(
        assembly_contract=ROUTER_ASSEMBLY_CONTRACT,
        output_protocol=ROUTER_OUTPUT_PROTOCOL,
        model_id=_prompt_text(binding_value["model_id"], "binding.model_id"),
        model_version=_prompt_text(
            binding_value["model_version"], "binding.model_version"
        ),
    )
    if value["consumer"] != ROUTER_CONSUMER:
        raise ValueError("router prompt_snapshot consumer mismatch")
    texts = {
        key: _prompt_text(value[key], key)
        for key in ("system_prompt", "mcp_scope", "multimodal_scope")
    }
    hashes = _object_with_exact_keys(
        value["item_sha256"],
        set(texts),
        "router prompt item_sha256",
    )
    checked_hashes: dict[str, str] = {}
    for key, content in texts.items():
        digest = _digest(hashes[key], f"item_sha256.{key}")
        if hashlib.sha256(content.encode()).hexdigest() != digest:
            raise ValueError(f"router prompt_snapshot {key} checksum mismatch")
        checked_hashes[key] = digest
    return RouterPromptSnapshot(
        release_id=_release(value["release_id"]),
        bundle_sha256=_digest(value["bundle_sha256"], "bundle_sha256"),
        binding=binding,
        item_sha256=MappingProxyType(checked_hashes),
        **texts,
    )


def _binding(
    raw: Any,
    *,
    assembly_contract: str,
    output_protocol: str,
) -> AgentPromptBinding:
    value = _object_with_exact_keys(
        raw,
        {"assembly_contract", "output_protocol", "model_id", "model_version"},
        "prompt binding",
    )
    if (
        value["assembly_contract"] != assembly_contract
        or value["output_protocol"] != output_protocol
    ):
        raise ValueError("prompt_snapshot protocol binding mismatch")
    return AgentPromptBinding(
        assembly_contract=assembly_contract,
        output_protocol=output_protocol,
        model_id=_prompt_text(value["model_id"], "binding.model_id"),
        model_version=_prompt_text(
            value["model_version"], "binding.model_version"
        ),
    )


def _object_with_exact_keys(raw: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"{name} has an invalid shape")
    return raw


def _release(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value == "active"
        or len(value) > 256
    ):
        raise ValueError("prompt_snapshot release_id must be an immutable release")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"prompt_snapshot {name} must be a SHA-256 digest")
    return value


def _prompt_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 200_000
        or "\x00" in value
    ):
        raise ValueError(f"prompt_snapshot {name} is invalid")
    return value
