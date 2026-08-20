# SPDX-License-Identifier: Apache-2.0
"""Cross-stage timing metadata for Qwen3-Omni action scoring."""

from __future__ import annotations

from typing import Any, Iterable

from sglang_omni.proto import StagePayload


ACTION_STAGE_TIMING_KEY = "_action_stage_timing"


def _metadata(value: StagePayload | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(value, StagePayload):
        metadata = getattr(value.request, "metadata", None)
    else:
        metadata = value
    if not isinstance(metadata, dict) or metadata.get("task") != "action_suffix_scoring":
        return None
    return metadata


def record_action_stage_timing(
    value: StagePayload | dict[str, Any], stage: str, **fields: Any
) -> None:
    metadata = _metadata(value)
    if metadata is None:
        return
    timings = metadata.setdefault(ACTION_STAGE_TIMING_KEY, {})
    if not isinstance(timings, dict):
        timings = {}
        metadata[ACTION_STAGE_TIMING_KEY] = timings
    current = timings.setdefault(stage, {})
    if not isinstance(current, dict):
        current = {}
        timings[stage] = current
    current.update(fields)


def merge_action_stage_timings(
    payloads: Iterable[StagePayload], target: StagePayload
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        metadata = _metadata(payload)
        timings = metadata.get(ACTION_STAGE_TIMING_KEY) if metadata is not None else None
        if not isinstance(timings, dict):
            continue
        for stage, fields in timings.items():
            if isinstance(fields, dict):
                combined.setdefault(str(stage), {}).update(fields)
    target_metadata = _metadata(target)
    if target_metadata is not None:
        target_metadata[ACTION_STAGE_TIMING_KEY] = combined
    return combined


def get_action_stage_timings(
    value: StagePayload | dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metadata = _metadata(value)
    timings = metadata.get(ACTION_STAGE_TIMING_KEY) if metadata is not None else None
    if not isinstance(timings, dict):
        return {}
    return {
        str(stage): dict(fields)
        for stage, fields in timings.items()
        if isinstance(fields, dict)
    }


__all__ = [
    "ACTION_STAGE_TIMING_KEY",
    "get_action_stage_timings",
    "merge_action_stage_timings",
    "record_action_stage_timing",
]
