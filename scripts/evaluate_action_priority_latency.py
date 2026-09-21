#!/usr/bin/env python3
"""Compare identical ordered workloads on one and two realtime sessions.

Measures client commit-send to event-receive, not server-only or playback latency.
Use --session and --cases to replay a specific role's sanitized configuration.
No server restart or active-session eviction is performed by this tool.
"""
import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path

import websockets

from evaluate_mixed_instructions import ROOT, receive, run_turn, judge


async def pace_turn(previous, interval):
    if previous is not None:
        remaining = interval - (time.perf_counter() - previous)
        if remaining > 0:
            await asyncio.sleep(remaining)
    started = time.perf_counter()
    return started, None if previous is None else started - previous


def summarize(rows):
    summary = {}
    for concurrency in (1, 2):
        subset = [r for r in rows if r["concurrency"] == concurrency]
        metrics = {}
        for name in ("turn.action.ready", "response.audio.delta"):
            values = sorted(r["wire_first_ms"][name] for r in subset if name in r["wire_first_ms"])
            metrics[name] = {
                "count": len(values), "missing": len(subset) - len(values),
                "p50_ms": statistics.median(values) if values else None,
                "p95_ms": values[max(0, math.ceil(len(values) * .95) - 1)] if values else None,
                "max_ms": max(values) if values else None,
            }
        summary[str(concurrency)] = metrics
    return summary


async def evaluate(args):
    catalog = json.loads((ROOT / "sglang_omni/assets/character_action_global_catalog.json").read_text())
    fallback = next(c["category_id"] for c in catalog["categories"] if "silent_accompaniment" in c.get("semantic_tags", []))
    session = {
        "type": "session.start", "protocol_version": 1, "locale": "zh-CN",
        "outputs": ["text", "audio", "expression", "action"],
        "reply": {"instructions": "自然简洁地回答。", "unsupported_action_text": "这个动作暂时无法执行。"},
        "action": {"fallback_category_ids": [fallback], "allowed_candidates": [
            {"candidate_id": candidate_id} for candidate_id in sorted({
                a["candidate_id"] for c in catalog["categories"] for a in c["children"]})]},
    }
    if args.session:
        session.update(json.loads(args.session.read_text()))
    cases = json.loads(args.cases.read_text()) if args.cases else [
        {"id": "chat", "text": "你好，简单介绍一下自己。"},
        {"id": "wave", "text": "挥挥手，再说你好。"},
        {"id": "three", "text": "做出数字三手势，再说三。"},
        {"id": "raise", "text": "抬起右手，再说好的。"},
    ]
    if isinstance(cases, dict):
        cases = cases["cases"]
    if args.image:
        cases = [{**case, "image": str(args.image.resolve())} for case in cases]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    rows = []
    report = {
        "label": args.label, "measurement": "client_commit_send_to_event_receive",
        "profile": "provided" if args.session else "synthetic_full_catalog_not_character_specific",
        "workload_sha256": hashlib.sha256(json.dumps([session, cases], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "rows": rows,
        "turns_per_session": len(cases) * args.repeats,
        "turn_start_interval_seconds": args.turn_interval,
        "media_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for case in cases for key in ("audio", "image") if case.get(key)
            for path in [args.case_base / case[key]]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["summary"] = summarize(rows)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    try:
        for concurrency in (1, 2):
            sockets = []
            previous_starts = {}
            try:
                for index in range(concurrency):
                    ws = await websockets.connect(args.url, max_size=32 * 1024 * 1024)
                    sockets.append(ws)
                    sid = f"latency-{uuid.uuid4().hex}"
                    started = time.perf_counter()
                    await ws.send(json.dumps({**session, "session_id": sid}, ensure_ascii=False))
                    async with asyncio.timeout(args.timeout):
                        await receive(ws, "session.started")
                    report.setdefault("startups", []).append({"concurrency": concurrency,
                        "slot": index, "session_id": sid, "elapsed_ms": (time.perf_counter() - started) * 1000})
                for iteration in range(args.repeats):
                    for case in cases:
                        async def one(index, ws):
                            previous = previous_starts.get(index)
                            turn_started, actual_interval = await pace_turn(previous, args.turn_interval)
                            previous_starts[index] = turn_started
                            async with asyncio.timeout(args.timeout):
                                result, first, events = await run_turn(ws, case, args.case_base, args.timeout)
                            return {"concurrency": concurrency, "slot": index,
                                "actual_start_interval_seconds": actual_interval,
                                "turn_elapsed_seconds": time.perf_counter() - turn_started,
                                "iteration": iteration, "case_id": case["id"],
                                "session_id": result["session_id"], "turn_id": result["turn_id"],
                                "status": result["status"], "action": result.get("action"),
                                "checks": judge(case, result, catalog) if case.get("expected") else {},
                                "timing": result.get("timing"), "wire_first_ms": first, "events": events}
                        results = await asyncio.gather(*(one(i, ws) for i, ws in enumerate(sockets)), return_exceptions=True)
                        rows.extend(r for r in results if not isinstance(r, BaseException))
                        save()
                        failure = next((r for r in results if isinstance(r, BaseException)), None)
                        if failure is not None:
                            raise failure
            finally:
                await asyncio.gather(*(ws.close() for ws in sockets), return_exceptions=True)
            # Allow asynchronous server-side session cleanup between groups.
            await asyncio.sleep(2)
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        save()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18007/v1/session/realtime")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--image", type=Path, help="Append one fixed JPEG to every turn; no video pacing")
    parser.add_argument("--case-base", type=Path, default=ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turn-interval", type=float, default=0,
                        help="Minimum start-to-start seconds per session; never overlap turns")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0 or not math.isfinite(args.turn_interval) or args.turn_interval < 0:
        parser.error("repeats/timeout must be positive; turn-interval must be finite and nonnegative")
    asyncio.run(evaluate(args))
