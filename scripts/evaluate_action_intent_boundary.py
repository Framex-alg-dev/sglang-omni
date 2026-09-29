#!/usr/bin/env python3
"""Measure grouped body-intent accuracy and action latency on text/audio pairs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import math
import time
import uuid
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from scipy.signal import resample_poly

from sglang_omni.serve.realtime.embedded_tts import (
    EmbeddedTTSConfig,
    EmbeddedTTSConnection,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    ROOT / "tests/unit_test/fixtures/realtime_action_intent_boundary_cases.json"
)
DEFAULT_CATALOG = (
    ROOT / "sglang_omni/assets/character_limited_action_global_catalog.json"
)
DEFAULT_SESSION = ROOT / "reports/soo_multimodal_506/session_start.json"
DEFAULT_AUDIO_DIR = ROOT / "reports/action_intent_boundary_audio"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getcomptype(),
        ) != (1, 2, 16000, "NONE"):
            raise ValueError(f"{path} must be mono 16 kHz PCM16 WAV")
        return wav.readframes(wav.getnframes())


async def prepare_audio(
    cases: list[dict[str, Any]],
    audio_dir: Path,
    *,
    tts_url: str,
    voice: str,
) -> None:
    audio_cases = [case for case in cases if case["modality"] == "audio"]
    missing = [case for case in audio_cases if not (audio_dir / f"{case['id']}.wav").exists()]
    if not missing:
        return
    audio_dir.mkdir(parents=True, exist_ok=True)
    connection = EmbeddedTTSConnection(
        EmbeddedTTSConfig(url=tts_url, voice=voice, turn_timeout_seconds=120),
        session_id="action-intent-boundary-audio-fixtures",
    )
    try:
        for case in missing:
            pcm_24k = bytearray()

            async def chunks():
                yield case["text"]

            async def sink(chunk: bytes) -> None:
                pcm_24k.extend(chunk)

            await connection.synthesize_streaming(
                turn_id=f"intent-boundary-{case['id']}",
                text_chunks=chunks(),
                audio_sink=sink,
                instruct="Read naturally as a real user speaking Chinese.",
            )
            samples = np.frombuffer(pcm_24k, dtype="<i2").astype(np.float32)
            samples_16k = np.clip(
                resample_poly(samples, 2, 3), -32768, 32767
            ).astype("<i2")
            target = audio_dir / f"{case['id']}.wav"
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples_16k.tobytes())
            print(
                json.dumps(
                    {
                        "prepared_audio": case["id"],
                        "seconds": round(len(samples_16k) / 16000, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await connection.close()


async def receive_until(
    websocket: Any,
    terminal_types: set[str],
    timeout: float,
    *,
    started: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.perf_counter() + timeout
    events: list[dict[str, Any]] = []
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {sorted(terminal_types)}")
        event = json.loads(
            await asyncio.wait_for(websocket.recv(), timeout=remaining)
        )
        if started is not None:
            event["_client_after_start_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
        events.append(event)
        if event.get("type") in terminal_types:
            return event, events


def build_session_start(
    template: dict[str, Any],
    catalog: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    start = copy.deepcopy(template)
    start["session_id"] = session_id
    # Keep the character's reply instructions, action guidance, visual profile,
    # locale and input-audio contract. Text+action isolates the measured intent
    # and action path while avoiding TTS/playback latency in the terminal event.
    start["outputs"] = ["text", "action"]
    start.pop("output_audio", None)
    candidate_ids = [
        action["candidate_id"]
        for category in catalog["categories"]
        for action in category["children"]
    ]
    fallback_ids = [
        category["category_id"]
        for category in catalog["categories"]
        if "silent_accompaniment" in category.get("semantic_tags", [])
    ]
    action = dict(start.get("action") or {})
    action["allowed_candidates"] = [
        {"candidate_id": candidate_id} for candidate_id in candidate_ids
    ]
    action["fallback_category_ids"] = fallback_ids
    start["action"] = action
    start["diagnostics"] = {"include_action_scores": True}
    return start


def extract_decision(result: dict[str, Any]) -> dict[str, Any]:
    media = result.get("media_summary") or {}
    context = media.get("action_context") or {}
    decision = context.get("action_decision") or {}
    return decision if isinstance(decision, dict) else {}


def judge(
    case: dict[str, Any],
    result: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, bool]:
    expected = case["expected"]
    action = result.get("action") or {}
    checks = {
        "completed": result.get("status") == "completed",
        "body_mode": decision.get("body_mode") == expected["body_mode"],
        "execute": action.get("execute") is expected["execute"],
    }
    if "candidate_id" in expected:
        checks["candidate_id"] = action.get("candidate_id") == expected["candidate_id"]
    if "support_status" in expected:
        checks["support_status"] = (
            action.get("support_status") == expected["support_status"]
        )
    return checks


async def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    template: dict[str, Any],
    catalog: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    session_id = f"intent-boundary-{case['id']}-{uuid.uuid4().hex[:8]}"[:128]
    turn_id = f"turn-{uuid.uuid4().hex[:12]}"
    row: dict[str, Any] = {
        "id": case["id"],
        "pair": case["pair"],
        "modality": case["modality"],
        "text": case["text"],
        "iteration": iteration,
        "expected": case["expected"],
        "session_id": session_id,
        "turn_id": turn_id,
    }
    started = time.perf_counter()
    try:
        async with asyncio.timeout(args.timeout):
            async with websockets.connect(
                args.url,
                max_size=64 * 1024 * 1024,
                open_timeout=args.timeout,
                close_timeout=args.timeout,
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        build_session_start(
                            template, catalog, session_id=session_id
                        ),
                        ensure_ascii=False,
                    )
                )
                session_started, _ = await receive_until(
                    websocket, {"session.started", "error"}, args.timeout
                )
                if session_started.get("type") != "session.started":
                    raise RuntimeError(str(session_started))
                row["decision_mode"] = session_started.get(
                    "action_decision_batch_mode"
                )
                row["catalog_hash"] = session_started.get(
                    "global_action_catalog_hash"
                )

                await websocket.send(
                    json.dumps(
                        {"type": "turn.start", "turn_id": turn_id, "origin": "user"}
                    )
                )
                turn_started, _ = await receive_until(
                    websocket, {"turn.started", "error"}, args.timeout
                )
                if turn_started.get("type") != "turn.started":
                    raise RuntimeError(str(turn_started))

                if case["modality"] == "text":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input.text.set",
                                "turn_id": turn_id,
                                "text": case["text"],
                            },
                            ensure_ascii=False,
                        )
                    )
                    ack, _ = await receive_until(
                        websocket, {"input.text.ack", "error"}, args.timeout
                    )
                    if ack.get("type") != "input.text.ack":
                        raise RuntimeError(str(ack))
                else:
                    pcm = read_pcm16(args.audio_dir / f"{case['id']}.wav")
                    for seq, offset in enumerate(range(0, len(pcm), 32000), 1):
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "input.audio.append",
                                    "turn_id": turn_id,
                                    "seq": seq,
                                    "data": base64.b64encode(
                                        pcm[offset : offset + 32000]
                                    ).decode(),
                                }
                            )
                        )
                        ack, _ = await receive_until(
                            websocket, {"input.ack", "error"}, args.timeout
                        )
                        if ack.get("type") != "input.ack":
                            raise RuntimeError(str(ack))

                committed = time.perf_counter()
                await websocket.send(
                    json.dumps({"type": "turn.commit", "turn_id": turn_id})
                )
                result, events = await receive_until(
                    websocket,
                    {"turn.result", "turn.failed", "error"},
                    args.timeout,
                    started=committed,
                )
                if result.get("type") != "turn.result":
                    raise RuntimeError(str(result))
                action_ready_ms: float | None = None
                elapsed_cursor = result.get("timing", {}).get(
                    "server_total_after_commit_ms"
                )
                for event in events:
                    if event.get("type") == "turn.action.ready":
                        action_ready_ms = event.get("_client_after_start_ms")
                        break
                decision = extract_decision(result)
                row.update(
                    {
                        "action": result.get("action"),
                        "decision": decision,
                        "timing": result.get("timing"),
                        "result_ms": round(
                            (time.perf_counter() - committed) * 1000, 3
                        ),
                        "action_ready_ms": action_ready_ms,
                        "server_total_after_commit_ms": elapsed_cursor,
                    }
                )
                row["checks"] = judge(case, result, decision)
                row["passed"] = all(row["checks"].values())
                await websocket.send(
                    json.dumps(
                        {"type": "session.close", "reason": "evaluation_complete"}
                    )
                )
                await receive_until(
                    websocket, {"session.closed", "error"}, args.timeout
                )
    except Exception as exc:
        row["passed"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    check_names = sorted(
        {name for row in rows for name in row.get("checks", {})}
    )
    latency_fields = (
        "action_ready_ms",
        "result_ms",
        "server_total_after_commit_ms",
    )
    action_passed = sum(
        "error" not in row
        and row.get("checks", {}).get("execute") is True
        and all(
            row.get("checks", {}).get(name) is True
            for name in ("candidate_id", "support_status")
            if name in row.get("checks", {})
        )
        for row in rows
    )
    summary: dict[str, Any] = {
        "samples": len(rows),
        "passed": sum(row.get("passed") is True for row in rows),
        "errors": sum("error" in row for row in rows),
        "action_accuracy": {
            "evaluated": len(rows),
            "passed": action_passed,
            "rate": action_passed / len(rows) if rows else None,
        },
        "checks": {
            name: {
                "evaluated": sum(name in row.get("checks", {}) for row in rows),
                "passed": sum(
                    row.get("checks", {}).get(name) is True for row in rows
                ),
            }
            for name in check_names
        },
        "latency_ms": {},
    }
    for field in latency_fields:
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float))
        ]
        summary["latency_ms"][field] = {
            "samples": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    intent_values = [
        float(row["timing"]["intent_ms"])
        for row in rows
        if isinstance(row.get("timing"), dict)
        and isinstance(row["timing"].get("intent_ms"), (int, float))
    ]
    action_values = [
        float(row["timing"]["server_action_compute_ms"])
        for row in rows
        if isinstance(row.get("timing"), dict)
        and isinstance(
            row["timing"].get("server_action_compute_ms"), (int, float)
        )
    ]
    summary["latency_ms"]["intent_ms"] = {
        "samples": len(intent_values),
        "p50": percentile(intent_values, 0.5),
        "p95": percentile(intent_values, 0.95),
        "max": max(intent_values) if intent_values else None,
    }
    summary["latency_ms"]["server_action_compute_ms"] = {
        "samples": len(action_values),
        "p50": percentile(action_values, 0.5),
        "p95": percentile(action_values, 0.95),
        "max": max(action_values) if action_values else None,
    }
    return summary


async def main(args: argparse.Namespace) -> int:
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    if not cases:
        raise ValueError("no matching cases")
    await prepare_audio(
        cases,
        args.audio_dir,
        tts_url=args.tts_url,
        voice=args.voice,
    )
    template = json.loads(args.session.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(case: dict[str, Any], iteration: int) -> dict[str, Any]:
        async with semaphore:
            row: dict[str, Any] | None = None
            for attempt in range(args.busy_retries + 1):
                row = await run_case(args, case, template, catalog, iteration)
                if "session_busy" not in row.get("error", ""):
                    break
                if attempt < args.busy_retries:
                    await asyncio.sleep(args.busy_retry_delay * (attempt + 1))
            assert row is not None
            row["attempts"] = attempt + 1
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "iteration": iteration,
                        "passed": row.get("passed"),
                        "body_mode": row.get("decision", {}).get("body_mode"),
                        "action": row.get("action"),
                        "intent_ms": row.get("timing", {}).get("intent_ms"),
                        "action_ms": row.get("timing", {}).get(
                            "server_action_compute_ms"
                        ),
                        "error": row.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return row

    rows = await asyncio.gather(
        *(
            bounded(case, iteration)
            for iteration in range(args.repeat)
            for case in cases
        )
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["modality"]].append(row)
    report = {
        "url": args.url,
        "character_session_template": str(args.session),
        "catalog": str(args.catalog),
        "fixture": str(args.cases),
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "minimum_action_accuracy": args.minimum_action_accuracy,
        "summary": summarize(rows),
        "modalities": {
            modality: summarize(group_rows)
            for modality, group_rows in sorted(groups.items())
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.minimum_action_accuracy is not None:
        action_accuracy = report["summary"]["action_accuracy"]
        return 0 if (
            report["summary"]["errors"] == 0
            and action_accuracy["rate"] is not None
            and action_accuracy["rate"] >= args.minimum_action_accuracy
        ) else 1
    return 0 if all(row.get("passed") is True for row in rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18004/v1/session/realtime"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/action_intent_boundary_18004.json",
    )
    parser.add_argument(
        "--tts-url", default="ws://127.0.0.1:40001/api-ws/v1/realtime"
    )
    parser.add_argument("--voice", default="spk_691b97a24dcc")
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--busy-retries", type=int, default=20)
    parser.add_argument("--busy-retry-delay", type=float, default=0.25)
    parser.add_argument(
        "--minimum-action-accuracy",
        type=float,
        default=None,
        help=(
            "Fail unless candidate/execute/support action accuracy reaches "
            "this 0..1 threshold; body_mode remains reported separately."
        ),
    )
    args = parser.parse_args()
    if (
        args.repeat < 1
        or args.concurrency < 1
        or args.timeout <= 0
        or args.busy_retries < 0
        or args.busy_retry_delay <= 0
        or (
            args.minimum_action_accuracy is not None
            and not 0.0 <= args.minimum_action_accuracy <= 1.0
        )
    ):
        parser.error("repeat/concurrency/timeout/delay must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
