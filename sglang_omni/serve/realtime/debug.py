# SPDX-License-Identifier: Apache-2.0
"""Read-only realtime session diagnostics backed by structured JSONL logs."""

from __future__ import annotations

import asyncio
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from sglang_omni.http.admin_auth import make_admin_auth_dependency

REALTIME_LOG_DIR_ENV = "SGLANG_OMNI_REALTIME_LOG_DIR"
DEFAULT_REALTIME_LOG_DIR = "/tmp/sglang-omni-realtime-logs"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
DEBUG_LOG_PREFIXES = (
    "action_",
    "diagnostic_",
    "error_",
    "lifecycle_",
    "performance_",
    "protocol_",
    "reply_",
)
MAX_MATCHED_RECORDS = 20_000
MAX_PROMPT_CHARS = 400_000


def realtime_log_root() -> Path:
    return Path(os.environ.get(REALTIME_LOG_DIR_ENV, DEFAULT_REALTIME_LOG_DIR))


def _source(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": record.get("_source_file"),
        "line": record.get("_source_line"),
    }


def _record_session_id(record: dict[str, Any]) -> str | None:
    session_id = record.get("session_id")
    if isinstance(session_id, str):
        return session_id
    logical = record.get("full_logical_input")
    if isinstance(logical, dict):
        session_id = logical.get("session_id")
        if isinstance(session_id, str):
            return session_id
        metadata = logical.get("metadata")
        if isinstance(metadata, dict):
            session_id = metadata.get("session_id")
            if isinstance(session_id, str):
                return session_id
    return None


def _record_turn_id(record: dict[str, Any], known_turn_ids: set[str]) -> str | None:
    turn_id = record.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        return turn_id
    logical = record.get("full_logical_input")
    candidates: list[Any] = [
        record.get("request_id"),
        record.get("logical_request_id"),
    ]
    if isinstance(logical, dict):
        candidates.extend(
            [logical.get("request_id"), logical.get("logical_request_id")]
        )
        metadata = logical.get("metadata")
        if isinstance(metadata, dict):
            turn_id = metadata.get("turn_id")
            if isinstance(turn_id, str) and turn_id:
                return turn_id
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        matches = [item for item in known_turn_ids if item in candidate]
        if matches:
            return max(matches, key=len)
    return None


def _iter_session_records(
    root: Path, session_id: str
) -> tuple[list[dict[str, Any]], int]:
    if not root.is_dir():
        return [], 0
    marker = session_id
    records: list[dict[str, Any]] = []
    scanned_files = 0
    for path in sorted(root.rglob("*.jsonl")):
        if not path.name.startswith(DEBUG_LOG_PREFIXES):
            continue
        scanned_files += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if marker not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # The active writer can expose a final partial line briefly.
                        continue
                    if _record_session_id(record) != session_id:
                        continue
                    record["_source_file"] = str(path)
                    record["_source_line"] = line_number
                    records.append(record)
                    if len(records) > MAX_MATCHED_RECORDS:
                        raise ValueError(
                            f"session log exceeds {MAX_MATCHED_RECORDS} records"
                        )
        except OSError:
            # A rotated file may disappear between discovery and open.
            continue
    records.sort(
        key=lambda item: (
            int(item.get("timestamp_unix_ms") or 0),
            str(item.get("event") or ""),
            str(item.get("_source_file") or ""),
            int(item.get("_source_line") or 0),
        )
    )
    return records, scanned_files


def _limited_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= MAX_PROMPT_CHARS:
        return value
    return value[:MAX_PROMPT_CHARS] + "\n<debug output truncated>"


def _action_prompt(record: dict[str, Any]) -> dict[str, Any]:
    logical = record.get("full_logical_input")
    if not isinstance(logical, dict):
        logical = {}
    messages = logical.get("messages")
    if not isinstance(messages, list):
        messages = []
    return {
        "available": True,
        "stage": logical.get("stage"),
        "request_id": logical.get("request_id"),
        "system_prompt": _limited_text(logical.get("system_prompt")),
        "dynamic_prompt": _limited_text(logical.get("prefix")),
        "messages": messages,
        "avatar_state": (
            (logical.get("metadata") or {}).get("avatar_state")
            if isinstance(logical.get("metadata"), dict)
            else None
        ),
        "rendered_prompt": None,
        "prompt_tokens": None,
        "scores": [],
        "source": _source(record),
    }


def _reply_prompt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "request_id": record.get("request_id"),
        "system_prompt": _limited_text(record.get("effective_system_prompt")),
        "messages": (
            record.get("messages") if isinstance(record.get("messages"), list) else []
        ),
        "selected_category": record.get("selected_category"),
        "support_status": record.get("support_status"),
        "redacted": record.get("effective_system_prompt") is None
        and bool(record.get("system_prompt_present")),
        "source": _source(record),
    }


def _new_turn(turn_id: str) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "trace_id": None,
        "origin": None,
        "trigger": None,
        "status": None,
        "started_at": None,
        "committed_at": None,
        "completed_at": None,
        "reply": {
            "source": None,
            "text": None,
            "first_delta_after_commit_ms": None,
            "text_done_after_commit_ms": None,
            "response_done_after_commit_ms": None,
            "prompt": None,
        },
        "action": {
            "category_id": None,
            "category_label": None,
            "candidate_id": None,
            "action_id": None,
            "source_label": None,
            "execute": None,
            "support_status": None,
            "fallback_applied": None,
            "ready_at": None,
            "ready_after_commit_ms": None,
            "category_compute_ms": None,
            "child_compute_ms": None,
            "category_prompt": None,
            "child_prompt": None,
        },
        "timing": {},
        "errors": [],
    }


def _monotonic_elapsed_ms(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> float | None:
    """Subtract monotonic points only when they came from the same process."""
    if start is None or end is None or start.get("pid") != end.get("pid"):
        return None
    start_ns = start.get("monotonic_ns")
    end_ns = end.get("monotonic_ns")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int):
        return None
    return round(max(0, end_ns - start_ns) / 1_000_000, 3)


def load_realtime_session_debug(
    session_id: str, *, log_root: Path | None = None
) -> dict[str, Any] | None:
    """Aggregate every structured log record associated with one session."""
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "session_id must be 1-128 characters using letters, digits, _, ., :, or -"
        )
    root = log_root or realtime_log_root()
    records, scanned_files = _iter_session_records(root, session_id)
    if not records:
        return None

    session: dict[str, Any] = {
        "session_id": session_id,
        "started_at": None,
        "locale": None,
        "modalities": [],
        "selection_mode": None,
        "action_candidate_count": None,
        "action_category_count": None,
        "instructions": None,
        "instructions_redacted": False,
        "action_profile": None,
    }
    turns: dict[str, dict[str, Any]] = {}
    rendered_prompts: dict[str, dict[str, Any]] = {}
    action_scores: dict[str, list[dict[str, Any]]] = {}
    timing_points: dict[str, dict[str, dict[str, Any]]] = {}
    known_turn_ids = {
        str(record["turn_id"])
        for record in records
        if isinstance(record.get("turn_id"), str) and record.get("turn_id")
    }

    for record in records:
        event = record.get("event")
        request_id = record.get("request_id")
        if event == "action_scoring_prompt_rendered" and isinstance(request_id, str):
            rendered_prompts[request_id] = record
        elif (
            event == "action_scoring_completed"
            and isinstance(request_id, str)
            and isinstance(record.get("scores"), list)
        ):
            action_scores[request_id] = record["scores"]

    for record in records:
        event = record.get("event")
        if event == "session_started":
            session.update(
                {
                    "started_at": record.get("timestamp"),
                    "locale": record.get("locale"),
                    "modalities": record.get("modalities")
                    or record.get("outputs")
                    or [],
                    "selection_mode": record.get("action_selection_mode"),
                    "action_candidate_count": record.get("action_candidate_count"),
                    "action_category_count": record.get("action_category_count"),
                }
            )
        elif event == "session_instructions_received":
            session["instructions"] = _limited_text(record.get("instructions"))
            session["instructions_redacted"] = record.get("instructions") is None
        elif event == "session_action_profile_received":
            session["action_profile"] = record.get("action_profile")

        turn_id = _record_turn_id(record, known_turn_ids)
        if turn_id is None:
            continue
        turn = turns.setdefault(turn_id, _new_turn(turn_id))
        timing_points.setdefault(turn_id, {})[str(event)] = record
        turn["trace_id"] = turn["trace_id"] or record.get("trace_id")

        if event == "turn_started":
            turn.update(
                {
                    "origin": record.get("turn_origin"),
                    "trigger": record.get("trigger"),
                    "started_at": record.get("timestamp"),
                }
            )
        elif event == "turn_commit_received":
            turn["committed_at"] = record.get("timestamp")
            turn["_committed_unix_ms"] = record.get("timestamp_unix_ms")
            turn["origin"] = turn["origin"] or record.get("turn_origin")
        elif event == "turn_completed":
            turn["completed_at"] = record.get("timestamp")
            turn["status"] = record.get("status")
        elif event in {"provided_reply_used", "reply_completed"}:
            turn["reply"].update(
                {
                    "source": (
                        "provided" if event == "provided_reply_used" else "generated"
                    ),
                    "text": record.get("output_text"),
                    "first_delta_after_commit_ms": record.get(
                        "first_delta_after_commit_ms"
                    ),
                    "text_done_after_commit_ms": record.get(
                        "text_done_after_commit_ms"
                    ),
                    "response_done_after_commit_ms": record.get(
                        "response_done_after_commit_ms"
                    ),
                }
            )
        elif event == "reply_logical_input":
            turn["reply"]["prompt"] = _reply_prompt(record)
        elif event == "category_selected":
            turn["action"].update(
                {
                    "category_id": record.get("category_id"),
                    "category_label": record.get("category_label"),
                    "support_status": record.get("support_status"),
                }
            )
        elif event == "action_execution_assumed":
            turn["action"].update(
                {
                    "category_id": record.get("category_id"),
                    "candidate_id": record.get("candidate_id"),
                    "action_id": record.get("action_id"),
                    "source_label": record.get("source_label"),
                    "execute": record.get("execute"),
                }
            )
        elif event == "action_scoring_started" and isinstance(
            record.get("full_logical_input"), dict
        ):
            prompt = _action_prompt(record)
            prompt_request_id = prompt.get("request_id")
            rendered = rendered_prompts.get(prompt_request_id)
            if rendered is not None:
                prompt["rendered_prompt"] = _limited_text(rendered.get("full_prompt"))
                prompt["prompt_tokens"] = rendered.get("prompt_tokens")
                prompt["rendered_source"] = _source(rendered)
            prompt["scores"] = action_scores.get(prompt_request_id, [])
            stage = prompt.get("stage")
            if stage == "category":
                turn["action"]["category_prompt"] = prompt
            elif stage == "child":
                turn["action"]["child_prompt"] = prompt
        elif event == "turn_timing":
            timing_fields = {
                key: value
                for key, value in record.items()
                if key.endswith("_ms")
                or key
                in {
                    "status",
                    "action_support_status",
                    "action_fallback_applied",
                    "category_decision_id",
                    "child_decision_id",
                }
            }
            turn["timing"] = timing_fields
            turn["reply"]["first_delta_after_commit_ms"] = record.get(
                "reply_first_delta_after_commit_ms"
            )
            turn["reply"]["text_done_after_commit_ms"] = record.get(
                "reply_text_done_after_commit_ms"
            )
            turn["reply"]["response_done_after_commit_ms"] = record.get(
                "reply_response_done_after_commit_ms"
            )
            turn["action"]["category_compute_ms"] = record.get("category_compute_ms")
            turn["action"]["child_compute_ms"] = record.get("child_compute_ms")
            turn["action"]["support_status"] = record.get("action_support_status")
            turn["action"]["fallback_applied"] = record.get("action_fallback_applied")
            turn["status"] = turn["status"] or record.get("status")
        elif (
            event == "ws_event_sent"
            and record.get("ws_event_type") == "turn.action.ready"
        ):
            turn["action"]["ready_at"] = record.get("timestamp")
            turn["action"]["_ready_unix_ms"] = record.get("timestamp_unix_ms")

        if record.get("level") in {"warning", "error", "critical"}:
            turn["errors"].append(
                {
                    "timestamp": record.get("timestamp"),
                    "event": event,
                    "level": record.get("level"),
                    "message": record.get("error_message") or record.get("message"),
                    "source": _source(record),
                }
            )

    configured_instructions = session.get("instructions")
    for turn in turns.values():
        action = turn["action"]
        committed_ms = turn.pop("_committed_unix_ms", None)
        ready_ms = action.pop("_ready_unix_ms", None)
        if isinstance(committed_ms, (int, float)) and isinstance(
            ready_ms, (int, float)
        ):
            action["ready_after_commit_ms"] = round(
                max(0.0, ready_ms - committed_ms), 3
            )
        if turn["reply"]["prompt"] is None:
            source = turn["reply"].get("source")
            turn["reply"]["prompt"] = {
                "available": False,
                "reason": (
                    "provided_reply"
                    if source == "provided"
                    else "reply_prompt_not_logged"
                ),
                "configured_system_prompt": configured_instructions,
                "configured_system_prompt_applied": (
                    False if source == "provided" else None
                ),
            }
        points = timing_points.get(turn["turn_id"], {})
        first_text = points.get("reply_first_token") or points.get(
            "provisional_reply_first_token"
        )
        commit = points.get("turn_commit_received")
        turn["timing"].update(
            {
                "commit_to_first_text_ms": _monotonic_elapsed_ms(commit, first_text),
                "first_text_to_tts_append_ms": _monotonic_elapsed_ms(
                    first_text, points.get("tts_first_append_sent")
                ),
                "tts_first_audio_ms": _monotonic_elapsed_ms(
                    points.get("tts_first_append_sent"),
                    points.get("tts_first_audio_received"),
                ),
                "tts_text_stream_ms": _monotonic_elapsed_ms(
                    points.get("tts_first_append_sent"),
                    points.get("tts_commit_sent"),
                ),
                "tts_provider_queue_ms": _monotonic_elapsed_ms(
                    points.get("tts_commit_sent"),
                    points.get("tts_response_created"),
                ),
                "tts_provider_first_pcm_ms": _monotonic_elapsed_ms(
                    points.get("tts_response_created"),
                    points.get("tts_first_audio_received"),
                ),
                "tts_commit_to_first_pcm_ms": _monotonic_elapsed_ms(
                    points.get("tts_commit_sent"),
                    points.get("tts_first_audio_received"),
                ),
                "commit_to_first_audio_ms": _monotonic_elapsed_ms(
                    commit, points.get("response_first_audio_delta_sent")
                ),
                "commit_to_playable_250ms": _monotonic_elapsed_ms(
                    commit, points.get("tts_playable_250ms_ready")
                ),
                "response_total_ms": _monotonic_elapsed_ms(
                    commit, points.get("response_done_sent")
                ),
                "turn_total_ms": _monotonic_elapsed_ms(
                    commit,
                    points.get("turn_result_sent") or points.get("turn_completed"),
                ),
                "cancel_total_ms": _monotonic_elapsed_ms(
                    points.get("turn_cancel_received"),
                    points.get("turn_cancelled_sent"),
                ),
            }
        )

    ordered_turns = sorted(
        turns.values(),
        key=lambda item: (
            item.get("started_at") or item.get("committed_at") or "",
            item["turn_id"],
        ),
    )
    return {
        "session": session,
        "turns": ordered_turns,
        "diagnostics": {
            "log_root": str(root),
            "scanned_files": scanned_files,
            "matched_records": len(records),
        },
    }


def register_realtime_debug_routes(
    app: FastAPI, admin_api_key: str | None = None
) -> None:
    """Register the browser page and authenticated read-only data endpoint."""
    auth = make_admin_auth_dependency(admin_api_key)

    @app.get("/debug/realtime", response_class=HTMLResponse, include_in_schema=False)
    async def realtime_debug_page() -> HTMLResponse:
        resource = files("sglang_omni").joinpath("assets/realtime_session_debug.html")
        return HTMLResponse(resource.read_text(encoding="utf-8"))

    @app.get(
        "/debug/realtime/api/session/{session_id}",
        dependencies=[Depends(auth)],
        tags=["debug"],
    )
    async def realtime_debug_session(session_id: str) -> JSONResponse:
        try:
            result = await asyncio.to_thread(load_realtime_session_debug, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No structured logs found for session_id={session_id}",
            )
        return JSONResponse(result)
