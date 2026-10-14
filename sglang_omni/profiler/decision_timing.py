# SPDX-License-Identifier: Apache-2.0
"""Bounded structured timing events for decision-model requests.

The turn router and task Brain share the same model runtime.  This module keeps
their correlation fields consistent across the API, pipeline, and scheduler
without ever copying prompt text or media payloads into logs.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from sglang_omni.utils.structured_logs import emit_structured_log


_DECISION_TASKS = frozenset({"turn_router", "task_brain"})
logger = logging.getLogger(__name__)
_EMIT_FAILURE_REPORTED = False


def decision_correlation_fields(
    *,
    request_id: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return safe correlation/cardinality fields for a decision request."""

    source = metadata if isinstance(metadata, Mapping) else {}
    task = _nonempty_text(source.get("task"))
    if task not in _DECISION_TASKS:
        return None

    operation_id = _nonempty_text(source.get("operation_id")) or request_id
    trace_id = _nonempty_text(source.get("trace_id")) or operation_id
    turn_id = _nonempty_text(source.get("turn_id")) or _nonempty_text(
        source.get("logical_request_id")
    )
    fields: dict[str, Any] = {
        "trace_id": trace_id,
        "request_id": request_id,
        "operation_id": operation_id,
        "task": task,
    }
    if task == "turn_router":
        fields["classification_id"] = (
            _nonempty_text(source.get("classification_id")) or operation_id
        )
    for output_name, source_names in (
        ("session_id", ("session_id", "session_instance_id")),
        ("turn_id", ("turn_id", "logical_request_id")),
        ("model_name", ("model", "model_name")),
        ("model_version", ("model_version",)),
        ("stage", ("stage",)),
    ):
        value = next(
            (
                candidate
                for name in source_names
                if (candidate := _nonempty_text(source.get(name))) is not None
            ),
            None,
        )
        if value is not None:
            fields[output_name] = value
    for name in ("identity_epoch", "input_revision"):
        value = source.get(name)
        if type(value) is int:
            fields[name] = value

    media_counts = {
        "audio_count": _media_count(source, "audio_count", "audios"),
        "image_count": _media_count(source, "image_count", "images"),
        "video_count": _media_count(source, "video_count", "videos"),
    }
    fields.update(media_counts)
    explicit_media_count = source.get("media_count")
    fields["media_count"] = (
        explicit_media_count
        if type(explicit_media_count) is int and explicit_media_count >= 0
        else sum(media_counts.values())
    )
    return fields


def emit_decision_timing(
    event: str,
    *,
    request_id: str,
    metadata: Mapping[str, Any] | None,
    component: str,
    level: str = "info",
    **fields: Any,
) -> bool:
    """Emit one non-blocking event when ``metadata`` marks decision work."""

    correlation = decision_correlation_fields(
        request_id=request_id,
        metadata=metadata,
    )
    if correlation is None:
        return False
    try:
        return emit_structured_log(
            "performance",
            event,
            level=level,
            component=component,
            **correlation,
            **fields,
        )
    except Exception:
        # Timing must never become part of the inference failure domain.  Warn
        # once so a broken sink remains visible without flooding stderr.
        global _EMIT_FAILURE_REPORTED
        if not _EMIT_FAILURE_REPORTED:
            _EMIT_FAILURE_REPORTED = True
            logger.warning("decision timing event sink failed", exc_info=True)
        return False


def request_metadata(value: Any) -> Mapping[str, Any]:
    """Safely find ``StagePayload.request.metadata`` from scheduler data."""

    stage_payload = getattr(value, "stage_payload", value)
    request = getattr(stage_payload, "request", None)
    metadata = getattr(request, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _sequence_length(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


def _media_count(source: Mapping[str, Any], count_key: str, items_key: str) -> int:
    value = source.get(count_key)
    if type(value) is int and value >= 0:
        return value
    return _sequence_length(source.get(items_key))


__all__ = [
    "decision_correlation_fields",
    "emit_decision_timing",
    "request_metadata",
]
