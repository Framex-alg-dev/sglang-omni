#!/usr/bin/env python3
"""Run one 24-turn audio+image user journey through D_video_call and SGLang."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "benchmarks/eval/realtime_multimodal_user_journey_cases.json"
DEFAULT_ASSETS = ROOT / "reports/realtime_multimodal_user_journey/assets"
DEFAULT_REPORT_ROOT = ROOT / "reports/realtime_multimodal_user_journey"
NUMERIC_ACTION_ID = re.compile(r"^[0-9]{3}$")
NUMERIC_CATEGORY_ID = re.compile(r"^[0-9]{2}$")
TERMINAL_TYPES = {"character_session_turn_result", "character_reply_error"}
REDACTED_KEYS = {"ticket", "token", "authorization", "api_key", "apikey"}


@dataclass
class TurnRecord:
    case_id: str
    request_id: str
    prompt: str
    scene: str
    image_file: str
    audio_file: str
    image_bytes: int
    audio_bytes: int
    audio_chunks: int
    started_at: float
    turn_id: str | None = None
    transcript: str = ""
    reply: str = ""
    result: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    expression: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    completed_at: float | None = None
    transport_pass: bool = False
    identifier_pass: bool = False
    semantic_pass: bool = False
    motion_pass: bool = False
    semantic_failures: list[str] = field(default_factory=list)
    motion_failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        duration_ms = None
        if self.completed_at is not None:
            duration_ms = round((self.completed_at - self.started_at) * 1000, 1)
        return {
            "case_id": self.case_id,
            "request_id": self.request_id,
            "turn_id": self.turn_id,
            "prompt": self.prompt,
            "scene": self.scene,
            "image_file": self.image_file,
            "audio_file": self.audio_file,
            "image_bytes": self.image_bytes,
            "audio_bytes": self.audio_bytes,
            "audio_chunks": self.audio_chunks,
            "duration_ms": duration_ms,
            "transcript": self.transcript,
            "reply": self.reply,
            "result": self.result,
            "action": self.action,
            "expression": self.expression,
            "errors": self.errors,
            "events": self.events,
            "transport_pass": self.transport_pass,
            "identifier_pass": self.identifier_pass,
            "semantic_pass": self.semantic_pass,
            "motion_pass": self.motion_pass,
            "semantic_failures": self.semantic_failures,
            "motion_failures": self.motion_failures,
        }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in REDACTED_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _post_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    turns = payload.get("turns")
    if not isinstance(turns, list) or len(turns) < 20:
        raise ValueError("the user journey must contain at least 20 turns")
    ids = [str(turn.get("id") or "") for turn in turns]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("all journey turns require unique non-empty ids")
    for turn in turns:
        if not str(turn.get("prompt") or "").strip():
            raise ValueError(f"turn {turn.get('id')} has no spoken prompt")
        if not str(turn.get("scene") or "").strip():
            raise ValueError(f"turn {turn.get('id')} has no camera scene")
    return turns


def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != 16000
        ):
            raise ValueError(f"{path} must be mono 16-bit PCM at 16 kHz")
        return wav.readframes(wav.getnframes())


def _require_assets(turns: list[dict[str, Any]], assets: Path) -> None:
    missing: list[str] = []
    for turn in turns:
        image = assets / f"{turn['scene']}.jpg"
        audio = assets / f"turn_{turn['id']}.wav"
        if not image.exists():
            missing.append(str(image))
        if not audio.exists():
            missing.append(str(audio))
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(
            "journey assets are missing; run "
            "scripts/prepare_realtime_multimodal_user_journey.py first:\n"
            f"{preview}"
        )


async def _receive_json(websocket, timeout: float) -> dict[str, Any]:
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    if not isinstance(raw, str):
        raise RuntimeError("D_video_call returned a non-JSON binary browser event")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("D_video_call returned a non-object browser event")
    return value


async def _wait_until_ready(websocket, timeout: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    session_init_seen = False
    playback_ready_sent = False
    while time.monotonic() < deadline:
        event = await _receive_json(websocket, deadline - time.monotonic())
        events.append(_redact(event))
        event_type = event.get("type")
        if event_type == "error":
            raise RuntimeError(f"session startup failed: {event}")
        if event_type == "session_init":
            session_init_seen = True
        if session_init_seen and not playback_ready_sent:
            await websocket.send(
                json.dumps(
                    {
                        "type": "dh_playback_ready",
                        "reason": "headless_evaluation_ready",
                        "playback_actual": "headless",
                        "client_ts": int(time.time() * 1000),
                    }
                )
            )
            playback_ready_sent = True
        if session_init_seen and event_type == "input_locked" and event.get("locked") is False:
            return events
    raise TimeoutError("session did not become input-ready")


async def _send_media_turn(
    websocket,
    case: dict[str, Any],
    assets: Path,
    *,
    run_id: str,
    audio_realtime_factor: float,
    timeout: float,
) -> TurnRecord:
    case_id = str(case["id"])
    request_id = f"journey-{run_id}-{case_id}"
    image_path = assets / f"{case['scene']}.jpg"
    audio_path = assets / f"turn_{case_id}.wav"
    image = image_path.read_bytes()
    pcm = _read_pcm(audio_path)
    chunk_size = 3200
    chunks = [pcm[offset : offset + chunk_size] for offset in range(0, len(pcm), chunk_size)]
    record = TurnRecord(
        case_id=case_id,
        request_id=request_id,
        prompt=str(case["prompt"]),
        scene=str(case["scene"]),
        image_file=str(image_path),
        audio_file=str(audio_path),
        image_bytes=len(image),
        audio_bytes=len(pcm),
        audio_chunks=len(chunks),
        started_at=time.monotonic(),
    )

    await websocket.send(
        json.dumps(
            {
                "type": "ptt",
                "state": "down",
                "request_id": request_id,
                "client_ts": int(time.time() * 1000),
            }
        )
    )
    await websocket.send(
        json.dumps(
            {
                "type": "image",
                "request_id": request_id,
                "frame_id": f"frame-{run_id}-{case_id}",
                "client_ts": int(time.time() * 1000),
                "mime_type": "image/jpeg",
                "reason": "user_camera",
                "image": base64.b64encode(image).decode(),
            }
        )
    )
    for index, chunk in enumerate(chunks, 1):
        await websocket.send(
            json.dumps(
                {
                    "type": "audio",
                    "request_id": request_id,
                    "chunk_id": f"{case_id}-{index}",
                    "client_ts": int(time.time() * 1000),
                    "audio": base64.b64encode(chunk).decode(),
                }
            )
        )
        if audio_realtime_factor > 0:
            await asyncio.sleep(0.1 * audio_realtime_factor)
    await websocket.send(
        json.dumps(
            {
                "type": "ptt",
                "state": "up",
                "request_id": request_id,
                "client_ts": int(time.time() * 1000),
            }
        )
    )

    deadline = time.monotonic() + timeout
    result_seen = False
    unlocked_after_result = False
    action_events: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        try:
            event = await _receive_json(websocket, deadline - time.monotonic())
        except TimeoutError:
            record.errors.append(
                {"type": "timeout", "message": f"turn exceeded {timeout:.1f}s"}
            )
            break
        safe_event = _redact(event)
        event_type = str(event.get("type") or "")
        event_turn_id = str(event.get("turn_id") or "") or None
        # A PTT request may own a short filler/delivery turn before the provider
        # creates the semantic user turn.  The ASR final or provider result is
        # authoritative; a worker_ack turn_id is not.
        if record.turn_id is None and event_turn_id and event_type in {
            "user_text_done",
            "character_session_turn_result",
        }:
            record.turn_id = event_turn_id
        belongs_to_turn = (
            event.get("request_id") == request_id
            or (record.turn_id is not None and event_turn_id == record.turn_id)
        )
        if belongs_to_turn:
            record.events.append(safe_event)
        if event_type == "error" and (
            event.get("request_id") in {None, request_id} or belongs_to_turn
        ):
            record.errors.append(safe_event)
            if event.get("request_id") == request_id:
                break
        if event_type == "user_text_done" and belongs_to_turn:
            record.transcript = str(event.get("text") or "").strip()
        elif event_type == "character_session_action_ready" and event_turn_id:
            action_events[event_turn_id] = safe_event
            if belongs_to_turn:
                record.action = safe_event
            expression = event.get("expression")
            if belongs_to_turn and isinstance(expression, dict):
                record.expression = _redact(expression)
        elif event_type == "character_session_turn_result" and (
            record.turn_id is None or event_turn_id == record.turn_id
        ):
            record.turn_id = event_turn_id
            record.result = safe_event
            record.reply = str(event.get("reply") or "").strip()
            if event_turn_id in action_events:
                record.action = action_events[event_turn_id]
            expression = event.get("expression")
            if isinstance(expression, dict):
                record.expression = _redact(expression)
            result_seen = True
        elif event_type == "input_locked" and result_seen and event.get("locked") is False:
            unlocked_after_result = True
        if result_seen and unlocked_after_result:
            break
    if not result_seen and not any(error.get("type") == "timeout" for error in record.errors):
        record.errors.append({"type": "timeout", "message": "turn result not received"})
    record.completed_at = time.monotonic()
    _evaluate_turn(record, case)
    return record


def _evaluate_turn(record: TurnRecord, case: dict[str, Any]) -> None:
    record.transport_pass = bool(
        record.result
        and record.result.get("status") == "completed"
        and record.turn_id
        and record.transcript
        and not record.errors
        and record.image_bytes > 0
        and record.audio_bytes > 0
        and record.audio_chunks > 0
    )

    action = record.action or {}
    candidate_id = str(action.get("candidate_id") or "")
    category_id = str(action.get("category_id") or "")
    selected_action = action.get("selected_action") or {}
    fallback_action = action.get("fallback_action") or {}
    expression = record.expression or {}
    candidate_ids = [
        str(value)
        for value in (
            candidate_id,
            selected_action.get("candidate_id"),
            fallback_action.get("candidate_id"),
            expression.get("candidate_id"),
            expression.get("expression_id"),
        )
        if value
    ]
    category_ids = [
        str(value)
        for value in (category_id, expression.get("category_id"))
        if value
    ]
    # D_video_call intentionally maps a catalog candidate to an opaque local
    # action_id such as act_<uuid>, or to the sentinel no_action.  The migrated
    # wire contract concerns candidate_id and category_id, not that local ID.
    record.identifier_pass = bool(
        all(NUMERIC_ACTION_ID.fullmatch(value) for value in candidate_ids)
        and all(NUMERIC_CATEGORY_ID.fullmatch(value) for value in category_ids)
    )

    reply_lower = record.reply.lower()
    for group in case.get("expected_reply_groups", []):
        alternatives = [str(item).lower() for item in group]
        if not any(item in reply_lower for item in alternatives):
            record.semantic_failures.append("missing one of: " + " | ".join(alternatives))
    record.semantic_pass = not record.semantic_failures

    expected_actions = {str(item) for item in case.get("expected_action_candidates", [])}
    if expected_actions and candidate_id not in expected_actions:
        record.motion_failures.append(
            f"action {candidate_id or '<none>'} not in {sorted(expected_actions)}"
        )
    expected_expressions = {
        str(item) for item in case.get("expected_expression_candidates", [])
    }
    expression_candidate = str((record.expression or {}).get("candidate_id") or "")
    if expected_expressions and expression_candidate not in expected_expressions:
        record.motion_failures.append(
            f"expression {expression_candidate or '<none>'} not in {sorted(expected_expressions)}"
        )
    record.motion_pass = not record.motion_failures


def _write_report(
    output: Path,
    *,
    run_id: str,
    character_id: str,
    session_id: str,
    startup_events: list[dict[str, Any]],
    records: list[TurnRecord],
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "character_id": character_id,
        "session_id": session_id,
        "turn_count": len(records),
        "unique_turn_ids": len({record.turn_id for record in records if record.turn_id}),
        "every_turn_sent_audio": all(record.audio_bytes > 0 for record in records),
        "every_turn_sent_image": all(record.image_bytes > 0 for record in records),
        "transport_pass_count": sum(record.transport_pass for record in records),
        "identifier_pass_count": sum(record.identifier_pass for record in records),
        "identifier_observed_count": sum(
            bool(
                (record.action or {}).get("candidate_id")
                or (record.action or {}).get("category_id")
                or (record.expression or {}).get("candidate_id")
                or (record.expression or {}).get("category_id")
            )
            for record in records
        ),
        "semantic_pass_count": sum(record.semantic_pass for record in records),
        "motion_pass_count": sum(record.motion_pass for record in records),
        "all_transport_pass": all(record.transport_pass for record in records),
        "all_identifiers_numeric": all(record.identifier_pass for record in records),
        "single_session_pass": len({record.turn_id for record in records if record.turn_id}) == len(records),
    }
    payload = {
        "summary": summary,
        "startup_events": startup_events,
        "turns": [record.as_dict() for record in records],
    }
    (output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "# 24-turn audio + image user journey",
        "",
        f"- Run: `{run_id}`",
        f"- Character: `{character_id}`",
        f"- Session: `{session_id}`",
        f"- Turns: {len(records)} in one continuous WebSocket session",
        f"- Transport completed: {summary['transport_pass_count']}/{len(records)}",
        f"- Numeric action/category IDs: {summary['identifier_pass_count']}/{len(records)}",
        f"- Semantic checks: {summary['semantic_pass_count']}/{len(records)}",
        f"- Explicit motion checks: {summary['motion_pass_count']}/{len(records)}",
        "",
        "| Turn | Media | Transcript | Reply | Action / category | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        action = record.action or {}
        result_mark = "PASS" if record.transport_pass else "FAIL"
        if not record.semantic_pass or not record.motion_pass:
            result_mark += " / quality review"
        transcript = record.transcript.replace("|", "\\|").replace("\n", " ")[:100]
        reply = record.reply.replace("|", "\\|").replace("\n", " ")[:140]
        lines.append(
            f"| {record.case_id} | 1 image + {record.audio_chunks} audio chunks | "
            f"{transcript} | {reply} | "
            f"{action.get('candidate_id', '-')} / {action.get('category_id', '-')} | {result_mark} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return summary


def _write_progress(
    output: Path,
    *,
    run_id: str,
    character_id: str,
    session_id: str,
    startup_events: list[dict[str, Any]],
    records: list[TurnRecord],
) -> None:
    (output / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "character_id": character_id,
                "session_id": session_id,
                "completed_records": len(records),
                "startup_events": startup_events,
                "turns": [record.as_dict() for record in records],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def _run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    turns = _load_cases(args.cases)
    _require_assets(turns, args.assets)
    run_id = time.strftime("journey-%Y%m%d-%H%M%S")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)

    created = await asyncio.to_thread(
        _post_json,
        f"{args.base_url}/api/characters/{args.character_id}/entry-sessions",
        {"entry_type": "chat"},
    )
    session_id = str(created["session_id"])
    started = await asyncio.to_thread(
        _post_json,
        f"{args.base_url}/api/character-sessions/{session_id}/start",
        None,
    )
    realtime = started.get("realtime") or {}
    websocket_path = str(realtime.get("websocket_path") or "")
    ticket = str(realtime.get("ticket") or "")
    if not websocket_path or not ticket:
        raise RuntimeError("session start did not return a WebSocket path and ticket")
    parsed = urllib.parse.urlsplit(args.base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urllib.parse.urlencode({"ticket": ticket, "log_mode": "full"})
    websocket_url = f"{ws_scheme}://{parsed.netloc}{websocket_path}?{query}"

    startup_events: list[dict[str, Any]] = []
    records: list[TurnRecord] = []
    _write_progress(
        output,
        run_id=run_id,
        character_id=args.character_id,
        session_id=session_id,
        startup_events=startup_events,
        records=records,
    )
    async with websockets.connect(
        websocket_url,
        open_timeout=args.startup_timeout,
        max_size=32 * 1024 * 1024,
    ) as websocket:
        startup_events = await _wait_until_ready(websocket, args.startup_timeout)
        _write_progress(
            output,
            run_id=run_id,
            character_id=args.character_id,
            session_id=session_id,
            startup_events=startup_events,
            records=records,
        )
        for index, case in enumerate(turns, 1):
            record = await _send_media_turn(
                websocket,
                case,
                args.assets,
                run_id=run_id,
                audio_realtime_factor=args.audio_realtime_factor,
                timeout=args.turn_timeout,
            )
            records.append(record)
            _write_progress(
                output,
                run_id=run_id,
                character_id=args.character_id,
                session_id=session_id,
                startup_events=startup_events,
                records=records,
            )
            print(
                json.dumps(
                    {
                        "turn": f"{index}/{len(turns)}",
                        "case_id": record.case_id,
                        "turn_id": record.turn_id,
                        "transport_pass": record.transport_pass,
                        "identifier_pass": record.identifier_pass,
                        "semantic_pass": record.semantic_pass,
                        "motion_pass": record.motion_pass,
                        "transcript": record.transcript[:100],
                        "reply": record.reply[:140],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if record.result is None:
                break
            if index != len(turns) and args.inter_turn_gap > 0:
                await asyncio.sleep(args.inter_turn_gap)

    summary = _write_report(
        output,
        run_id=run_id,
        character_id=args.character_id,
        session_id=session_id,
        startup_events=startup_events,
        records=records,
    )
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:40004")
    parser.add_argument(
        "--character-id",
        default="character_4ef8a49978444c9aba77fc7293fb5984",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--turn-timeout", type=float, default=180.0)
    parser.add_argument("--inter-turn-gap", type=float, default=1.0)
    parser.add_argument(
        "--audio-realtime-factor",
        type=float,
        default=1.0,
        help="1.0 sends audio in real time; 0 sends chunks without pacing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail the process for semantic or explicit motion misses",
    )
    args = parser.parse_args()
    if args.audio_realtime_factor < 0:
        parser.error("--audio-realtime-factor must be non-negative")
    try:
        output, summary = asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2))
    structurally_passed = bool(
        summary["turn_count"] >= 20
        and summary["every_turn_sent_audio"]
        and summary["every_turn_sent_image"]
        and summary["all_transport_pass"]
        and summary["all_identifiers_numeric"]
        and summary["identifier_observed_count"] > 0
        and summary["single_session_pass"]
    )
    quality_passed = bool(
        summary["semantic_pass_count"] == summary["turn_count"]
        and summary["motion_pass_count"] == summary["turn_count"]
    )
    return 0 if structurally_passed and (quality_passed or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
