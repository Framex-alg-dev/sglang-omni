#!/usr/bin/env python3
"""Evaluate text-only action recall against a live Realtime service."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import websockets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    ROOT / "tests/unit_test/fixtures/realtime_text_action_recall_cases.json"
)
DEFAULT_CATALOG = (
    ROOT / "sglang_omni/assets/character_limited_action_global_catalog.json"
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


async def receive_until(
    websocket: Any, terminal_types: set[str], timeout: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.perf_counter() + timeout
    events = []
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {sorted(terminal_types)}")
        event = json.loads(
            await asyncio.wait_for(websocket.recv(), timeout=remaining)
        )
        events.append(event)
        if event.get("type") in terminal_types:
            return event, events


def normalize_ranking(
    items: list[dict[str, Any]], valid_candidate_ids: set[str]
) -> list[dict[str, Any]]:
    ranking = []
    for score in items:
        candidate_id = score.get("candidate_id")
        if candidate_id == "000":
            candidate_id = "UNSUPPORTED"
        if candidate_id not in valid_candidate_ids | {"UNSUPPORTED"}:
            continue
        ranking.append(
            {
                "candidate_id": candidate_id,
                "source_label": score.get("source_label"),
                "mean_logprob": score.get("mean_logprob"),
                "ppl": score.get("ppl"),
                "background_centered_bias": score.get(
                    "background_centered_bias"
                ),
                "background_calibrated_score": score.get(
                    "background_calibrated_score"
                ),
                "background_raw_rank": score.get("background_raw_rank"),
                "background_calibrated_rank": score.get(
                    "background_calibrated_rank"
                ),
            }
        )
    return ranking


def action_ranking(
    result: dict[str, Any], valid_candidate_ids: set[str]
) -> list[dict[str, Any]]:
    return normalize_ranking(result.get("scores") or [], valid_candidate_ids)


def single_token_shadow_result(result: dict[str, Any]) -> dict[str, Any] | None:
    media = result.get("media_summary")
    context = media.get("action_context") if isinstance(media, dict) else None
    shadow = context.get("single_token_shadow") if isinstance(context, dict) else None
    return shadow if isinstance(shadow, dict) else None


def judge_shadow(
    case: dict[str, Any], ranking: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool | None]]:
    selected_id = ranking[0]["candidate_id"] if ranking else None
    action = {
        "candidate_id": selected_id,
        "support_status": (
            "unsupported" if selected_id == "UNSUPPORTED" else "supported"
        ),
    }
    expected = case["expected"]
    expected_ids = set(expected.get("candidate_ids") or [])
    ranked_ids = [item["candidate_id"] for item in ranking]
    checks: dict[str, bool | None] = {
        "candidate_top1": selected_id in expected_ids if expected_ids else None,
        "candidate_top3": (
            bool(expected_ids.intersection(ranked_ids[:3]))
            if expected_ids and ranked_ids else None
        ),
        "candidate_top5": (
            bool(expected_ids.intersection(ranked_ids[:5]))
            if expected_ids and ranked_ids else None
        ),
        "support_status": (
            action["support_status"] == expected["support_status"]
            if "support_status" in expected else None
        ),
    }
    return action, checks


def judge(
    case: dict[str, Any], result: dict[str, Any], ranking: list[dict[str, Any]]
) -> dict[str, bool | None]:
    expected = case["expected"]
    action = result.get("action") or {}
    selected_id = action.get("candidate_id")
    expected_ids = set(expected.get("candidate_ids") or [])
    ranked_ids = [item["candidate_id"] for item in ranking]
    checks: dict[str, bool | None] = {
        "completed": result.get("status") == "completed",
        "candidate_top1": (
            selected_id in expected_ids if expected_ids else None
        ),
        "candidate_top3": (
            bool(expected_ids.intersection(ranked_ids[:3]))
            if expected_ids and ranked_ids
            else None
        ),
        "candidate_top5": (
            bool(expected_ids.intersection(ranked_ids[:5]))
            if expected_ids and ranked_ids
            else None
        ),
        "execute": (
            action.get("execute") is expected["execute"]
            if "execute" in expected
            else None
        ),
        "support_status": (
            action.get("support_status") == expected["support_status"]
            if "support_status" in expected
            else None
        ),
    }
    return checks


def checks_pass(checks: dict[str, bool | None]) -> bool:
    return all(value for value in checks.values() if value is not None)


async def run_case(
    *,
    url: str,
    case: dict[str, Any],
    candidate_ids: list[str],
    valid_candidate_ids: set[str],
    timeout: float,
    include_expression: bool,
    save_all_scores: bool,
) -> dict[str, Any]:
    session_id = f"text-recall-{case['id']}-{uuid.uuid4().hex[:8]}"[:128]
    turn_id = f"turn-{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": case["id"],
        "group": case["group"],
        "text": case["text"],
        "expected": case["expected"],
        "session_id": session_id,
        "turn_id": turn_id,
    }
    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(
                url,
                max_size=64 * 1024 * 1024,
                open_timeout=timeout,
                close_timeout=timeout,
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "protocol_version": 1,
                            "session_id": session_id,
                            "locale": "zh-CN",
                            "outputs": (
                                ["expression", "action"]
                                if include_expression
                                else ["action"]
                            ),
                            "action": {
                                "locale": "zh-CN",
                                "allowed_candidates": [
                                    {"candidate_id": candidate_id}
                                    for candidate_id in candidate_ids
                                ],
                            },
                            "diagnostics": {"include_action_scores": True},
                        },
                        ensure_ascii=False,
                    )
                )
                session_started, _ = await receive_until(
                    websocket, {"session.started", "error"}, timeout
                )
                if session_started.get("type") != "session.started":
                    raise RuntimeError(str(session_started))
                row["catalog_hash"] = session_started.get(
                    "global_action_catalog_hash"
                )
                row["session_catalog_hash"] = session_started.get(
                    "session_action_catalog_hash"
                )
                row["action_single_token_mode"] = session_started.get(
                    "action_single_token_mode", "off"
                )
                row["action_selection_mapping_hash"] = session_started.get(
                    "action_selection_mapping_hash"
                )
                row["action_selection_calibration_hash"] = session_started.get(
                    "action_selection_calibration_hash"
                )
                row["session_start_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 3
                )

                await websocket.send(
                    json.dumps(
                        {
                            "type": "turn.start",
                            "turn_id": turn_id,
                            "origin": "user",
                        }
                    )
                )
                turn_started, _ = await receive_until(
                    websocket, {"turn.started", "error"}, timeout
                )
                if turn_started.get("type") != "turn.started":
                    raise RuntimeError(str(turn_started))
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
                text_ack, _ = await receive_until(
                    websocket, {"input.text.ack", "error"}, timeout
                )
                if text_ack.get("type") != "input.text.ack":
                    raise RuntimeError(str(text_ack))

                committed = time.perf_counter()
                await websocket.send(
                    json.dumps({"type": "turn.commit", "turn_id": turn_id})
                )
                result, events = await receive_until(
                    websocket, {"turn.result", "turn.failed", "error"}, timeout
                )
                row["commit_to_result_ms"] = round(
                    (time.perf_counter() - committed) * 1000.0, 3
                )
                row["event_types"] = [event.get("type") for event in events]
                if result.get("type") != "turn.result":
                    raise RuntimeError(str(result))
                ranking = action_ranking(result, valid_candidate_ids)
                row["action"] = result.get("action")
                row["ranking_top10"] = ranking[:10]
                if save_all_scores:
                    row["ranking_all"] = ranking
                row["score_count"] = len(result.get("scores") or [])
                row["timing"] = result.get("timing")
                row["checks"] = judge(case, result, ranking)
                row["passed"] = checks_pass(row["checks"])
                shadow = single_token_shadow_result(result)
                if shadow is not None:
                    shadow_ranking = normalize_ranking(
                        shadow.get("ranking") or [], valid_candidate_ids
                    )
                    shadow_action, shadow_checks = judge_shadow(
                        case, shadow_ranking
                    )
                    row["single_token_shadow_action"] = shadow_action
                    row["single_token_shadow_ranking_top10"] = shadow_ranking[:10]
                    if save_all_scores:
                        row["single_token_shadow_ranking_all"] = shadow_ranking
                    row["single_token_shadow_checks"] = shadow_checks
                    row["single_token_shadow_passed"] = checks_pass(
                        shadow_checks
                    )
                    row["single_token_shadow_metadata"] = {
                        key: shadow.get(key)
                        for key in (
                            "mapping_version",
                            "mapping_hash",
                            "calibration_version",
                            "calibration_hash",
                            "decision",
                            "winners",
                            "legacy_winners",
                            "agreement",
                            "legacy_winner_rank",
                        )
                    }
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.close",
                            "reason": "text_recall_complete",
                        }
                    )
                )
                await receive_until(
                    websocket, {"session.closed", "error"}, timeout
                )
    except Exception as exc:
        row["passed"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_checks: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for name, value in row.get("checks", {}).items():
            if value is not None:
                evaluated_checks[name].append(bool(value))
    latencies = [
        row["commit_to_result_ms"]
        for row in rows
        if "commit_to_result_ms" in row
    ]
    confusions = Counter(
        (
            tuple(row["expected"].get("candidate_ids") or []),
            (row.get("action") or {}).get("candidate_id"),
        )
        for row in rows
        if row.get("checks", {}).get("candidate_top1") is False
    )
    return {
        "cases": len(rows),
        "passed": sum(row.get("passed") is True for row in rows),
        "errors": sum("error" in row for row in rows),
        "checks": {
            name: {
                "evaluated": len(values),
                "passed": sum(values),
                "rate": round(sum(values) / len(values), 6),
            }
            for name, values in sorted(evaluated_checks.items())
        },
        "commit_to_result_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "top_confusions": [
            {
                "expected": list(expected),
                "selected": selected,
                "count": count,
            }
            for (expected, selected), count in confusions.most_common(15)
        ],
    }


def summarize_single_token_shadow(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    shadow_rows = [row for row in rows if "single_token_shadow_checks" in row]
    if not shadow_rows:
        return None
    projected = [
        {
            **row,
            "checks": row["single_token_shadow_checks"],
            "action": row.get("single_token_shadow_action"),
            "passed": row.get("single_token_shadow_passed", False),
        }
        for row in shadow_rows
    ]
    summary = summarize(projected)
    agreements: dict[str, list[bool]] = defaultdict(list)
    for row in shadow_rows:
        agreement = row.get("single_token_shadow_metadata", {}).get(
            "agreement", {}
        )
        if not isinstance(agreement, dict):
            continue
        for group, value in agreement.items():
            if isinstance(value, bool):
                agreements[str(group)].append(value)
    summary["legacy_agreement"] = {
        group: {
            "evaluated": len(values),
            "agreed": sum(values),
            "rate": round(sum(values) / len(values), 6),
        }
        for group, values in sorted(agreements.items())
        if values
    }
    return summary


async def evaluate(args: argparse.Namespace) -> int:
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    candidate_ids = list(
        dict.fromkeys(
            action["candidate_id"]
            for category in catalog["categories"]
            for action in category["children"]
        )
    )
    cases = fixture["cases"]
    if args.group:
        cases = [case for case in cases if case["group"] in args.group]
    if args.category_id:
        cases = [
            case
            for case in cases
            if case.get("category_id") in args.category_id
        ]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no matching cases")

    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    rows: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    def save() -> None:
        by_group = {
            group: summarize([row for row in rows if row["group"] == group])
            for group in sorted({row["group"] for row in rows})
        }
        report = {
            "url": args.url,
            "fixture": str(args.cases),
            "fixture_catalog_hash": fixture.get("catalog_hash"),
            "isolated_session_per_case": True,
            "concurrency": args.concurrency,
            "include_expression": args.include_expression,
            "save_all_scores": args.save_all_scores,
            "summary": summarize(rows),
            "single_token_shadow_summary": summarize_single_token_shadow(rows),
            "groups": by_group,
            "single_token_shadow_groups": {
                group: summarize_single_token_shadow(
                    [row for row in rows if row["group"] == group]
                )
                for group in sorted({row["group"] for row in rows})
            },
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def bounded(case: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        row = None
        for attempt in range(args.busy_retries + 1):
            async with semaphore:
                row = await run_case(
                    url=args.url,
                    case=case,
                    candidate_ids=candidate_ids,
                    valid_candidate_ids=set(candidate_ids),
                    timeout=args.timeout,
                    include_expression=args.include_expression,
                    save_all_scores=args.save_all_scores,
                )
            if "session_busy" not in row.get("error", ""):
                break
            if attempt < args.busy_retries:
                await asyncio.sleep(args.busy_retry_delay * (attempt + 1))
        assert row is not None
        row["attempts"] = attempt + 1
        async with lock:
            rows.append(row)
            completed += 1
            save()
            if completed % 10 == 0 or completed == len(cases):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(cases),
                            "passed": sum(
                                item.get("passed") is True for item in rows
                            ),
                            "errors": sum("error" in item for item in rows),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return row

    await asyncio.gather(*(bounded(case) for case in cases))
    rows.sort(key=lambda row: next(
        index for index, case in enumerate(cases) if case["id"] == row["id"]
    ))
    save()
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return 0 if all(row.get("passed") is True for row in rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18004/v1/session/realtime"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/text_action_recall_18004.json",
    )
    parser.add_argument("--group", action="append")
    parser.add_argument("--category-id", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--busy-retries", type=int, default=20)
    parser.add_argument("--busy-retry-delay", type=float, default=0.25)
    parser.add_argument("--include-expression", action="store_true")
    parser.add_argument(
        "--save-all-scores",
        action="store_true",
        help="Persist every concrete candidate score for offline reranking.",
    )
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.timeout <= 0
        or args.busy_retries < 0
        or args.busy_retry_delay <= 0
    ):
        parser.error("concurrency/timeout/delay must be positive; retries nonnegative")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(evaluate(parse_args())))
