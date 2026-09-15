#!/usr/bin/env python3
"""Evaluate a current Session Realtime server; report wire latency, not playback.

Audio cases use only WAV input (no transcript leakage). Each case has its own
session; optional history turns stay in that session. No generated model judge.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/unit_test/fixtures/realtime_mixed_instruction_cases.json"
ALIASES = {"挥手": {"288", "289", "217", "218"}, "摇头": {"135"}}


def normalize_text(text):
    # Do not erase acknowledgements, internal punctuation, or spoken words.
    return text.strip().rstrip("。.!！?？")


def target_ids(targets, catalog):
    ids = set()
    for target in targets:
        matches = ALIASES.get(target, set()) | {
            a["candidate_id"] for c in catalog["categories"] for a in c["children"]
            if a["source_label"] == target
        }
        if not matches:
            raise ValueError(f"unknown semantic target: {target}")
        ids.update(matches)
    return ids


def judge(case, result, catalog):
    expected = case["expected"]
    checks = {}
    reply = (result.get("reply") or {}).get("text", "")
    if expected.get("reply_exact") is not None:
        checks["reply"] = normalize_text(reply) == normalize_text(expected["reply_exact"])
    elif expected.get("language_required"):
        checks["reply"] = bool(reply.strip())
    if "language_required" in expected:
        checks["language_route"] = result.get("timing", {}).get("reply_mode") == (
            "LANGUAGE_REQUIRED" if expected["language_required"] else "PURE_ACTION"
        )
    action = result.get("action") or {}
    if expected.get("body_targets"):
        checks["body"] = (
            action.get("candidate_id") in target_ids(expected["body_targets"], catalog)
            and action.get("execute") is True and action.get("support_status") == "supported"
        )
    if expected.get("forbidden_body_targets"):
        checks["body_prohibition"] = not action.get("execute") or action.get("candidate_id") not in target_ids(expected["forbidden_body_targets"], catalog)
    if expected.get("body_support"):
        checks["body_support"] = action.get("support_status") == expected["body_support"]
    if expected.get("expression_target"):
        checks["expression"] = (result.get("expression") or {}).get("candidate_id") in target_ids([expected["expression_target"]], catalog)
    if expected.get("request_scope") is not None:
        checks["scope"] = result.get("timing", {}).get("request_scope") == expected["request_scope"]
    checks["completed"] = result.get("status") == "completed"
    return checks


def load_audio(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("audio must be uncompressed mono 16 kHz PCM16 WAV")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError("empty audio")
    return pcm


async def receive(ws, expected=None):
    event = json.loads(await ws.recv())
    if event.get("type") == "error":
        raise RuntimeError(str(event))
    if expected and event.get("type") != expected:
        raise RuntimeError(f"expected {expected}, got {event.get('type')}")
    return event


async def run_turn(ws, case, base, timeout):
    turn_id = uuid.uuid4().hex
    await ws.send(json.dumps({"type": "turn.start", "turn_id": turn_id, "origin": "user"}))
    await receive(ws, "turn.started")
    if case.get("audio"):
        pcm = load_audio(base / case["audio"])
        for seq, offset in enumerate(range(0, len(pcm), 32000), 1):
            await ws.send(json.dumps({"type": "input.audio.append", "turn_id": turn_id, "seq": seq,
                                     "data": base64.b64encode(pcm[offset:offset + 32000]).decode()}))
            await receive(ws, "input.ack")
    else:
        await ws.send(json.dumps({"type": "input.text.set", "turn_id": turn_id, "text": case["text"]}))
        await receive(ws, "input.text.ack")
    start = time.perf_counter()
    await ws.send(json.dumps({"type": "turn.commit", "turn_id": turn_id}))
    first = {}
    events = []
    while True:
        event = await receive(ws)
        elapsed = (time.perf_counter() - start) * 1000
        kind = event.get("type")
        if event.get("turn_id") != turn_id:
            continue
        if kind in {"response.text.delta", "response.provisional.text.delta", "response.audio.delta", "turn.action.ready", "turn.expression.ready", "turn.committed"}:
            first.setdefault(kind, elapsed)
        # Persist summaries, not large audio payloads.
        events.append({"type": kind, "after_send_commit_ms": elapsed})
        if kind == "turn.result":
            return event, first, events
        if kind in {"turn.failed", "turn.cancelled"}:
            raise RuntimeError(str(event))


async def run_case(args, case, catalog, index):
    import websockets
    all_ids = sorted({a["candidate_id"] for c in catalog["categories"] for a in c["children"]})
    excluded = set(case.get("excluded_candidate_ids", []))
    fallback = next(c["category_id"] for c in catalog["categories"] if "silent_accompaniment" in c.get("semantic_tags", []))
    start = {"type": "session.start", "protocol_version": 1, "session_id": uuid.uuid4().hex,
             "locale": "zh-CN", "outputs": ["text", "expression", "action"],
             "reply": {"instructions": "自然简洁地回答。", "unsupported_action_text": "这个动作暂时无法执行。"},
             "action": {"fallback_category_ids": [fallback], "allowed_candidates": [
                 {"candidate_id": x} for x in all_ids if x not in excluded]}}
    if args.audio_output:
        start["outputs"].insert(1, "audio")
    if args.session:
        start.update(json.loads(args.session.read_text()))
        start["session_id"] = uuid.uuid4().hex
    row = {"id": case["id"], "iteration": index, "group": case.get("group", "mixed"),
           "input_kind": "audio" if case.get("audio") else "text"}
    try:
        async with asyncio.timeout(args.timeout):
            async with websockets.connect(args.url, max_size=16 * 1024 * 1024) as ws:
                await ws.send(json.dumps(start, ensure_ascii=False))
                await receive(ws, "session.started")
                for previous in case.get("history", []):
                    await run_turn(ws, previous, args.cases.parent, args.timeout)
                result, first, events = await run_turn(ws, case, args.cases.parent, args.timeout)
                # A closing owner still occupies a slot. Await server cleanup
                # before admitting the next case under the four-session limit.
                await ws.send(json.dumps({"type": "session.close", "reason": "evaluation_complete"}))
                await receive(ws, "session.closed")
                await ws.wait_closed()
        row.update(result=result, wire_first_ms=first, events=events, checks=judge(case, result, catalog))
        row["passed"] = all(row["checks"].values())
    except Exception as exc:
        row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    return row


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)] if values else None


def summarize(rows):
    names = {name for row in rows for name in row.get("wire_first_ms", {})}
    check_names = {name for row in rows for name in row.get("checks", {})}
    return {"samples": len(rows), "passed": sum(r["passed"] for r in rows),
            "errors": sum("error" in r for r in rows),
            "checks": {name: {"evaluated": sum(name in r.get("checks", {}) for r in rows),
                              "passed": sum(r.get("checks", {}).get(name, False) for r in rows)} for name in sorted(check_names)},
            "wire_latency_ms": {name: {"samples": len(v := [r["wire_first_ms"][name] for r in rows if name in r.get("wire_first_ms", {})]),
                                       "p50": percentile(v, .5), "p95": percentile(v, .95), "p99": percentile(v, .99)} for name in sorted(names)}}


async def main(args):
    cases = json.loads(args.cases.read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if not cases:
        raise ValueError("no matching cases")
    catalog = json.loads(args.catalog.read_text())
    semaphore = asyncio.Semaphore(args.concurrency)
    async def bounded(case, index):
        async with semaphore:
            return await run_case(args, case, catalog, index)
    warmup = [await bounded(cases[0], -1) for _ in range(args.warmup)]
    rows = await asyncio.gather(*(bounded(c, i) for i in range(args.repeat) for c in cases))
    report = {"measurement": "client receive time from commit send; not server compute or audio playback", "url": args.url,
              "catalog_hash": catalog.get("action_catalog_hash"), "warmup": warmup,
              "summary": summarize(rows), "groups": {g: summarize([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=ROOT / "sglang_omni/assets/character_action_global_catalog.json")
    parser.add_argument("--session", type=Path, help="Optional session.start fields, e.g. TTS configuration")
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mixed_instructions.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--audio-output", action="store_true")
    args = parser.parse_args()
    if min(args.repeat, args.concurrency, args.timeout) <= 0 or args.warmup < 0:
        parser.error("repeat/concurrency/timeout must be positive; warmup must be nonnegative")
    raise SystemExit(asyncio.run(main(args)))
