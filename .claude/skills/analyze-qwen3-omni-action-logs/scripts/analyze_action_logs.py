#!/usr/bin/env python3
"""Summarize Qwen3-Omni realtime action-scoring logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

COMMIT_RE = re.compile(
    r"turn\.commit input session_id=(\S+) turn_id=(\S+) payload=(\{.*\})$"
)
REPLY_RE = re.compile(
    r"reply completed session_id=(\S+) turn_id=(\S+) "
    r"elapsed_ms=([0-9.]+) reply_chars=(\d+)"
)
ACTION_RE = re.compile(
    r"action completed session_id=(\S+) turn_id=(\S+) "
    r"elapsed_ms=([0-9.]+) top_action=(\S+)"
)
TURN_RE = re.compile(r"session-(.+?)-turn-(turn_[^-]+)-action-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--debug-log", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--max-seq-len", type=int, default=20000)
    parser.add_argument("--token-breakdown", action="store_true")
    parser.add_argument(
        "--model-path", default="/data/models/Qwen3-Omni-30B-A3B-FP8"
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--require-reply", action="store_true")
    return parser.parse_args()


def load_logs(args: argparse.Namespace) -> tuple[Path, Path]:
    server = args.server_log or args.repo_root / "logs/qwen3_omni_local.log"
    debug = args.debug_log or args.repo_root / "logs/session_action_debug.jsonl"
    if not server.exists():
        raise SystemExit(f"server log not found: {server}")
    if not debug.exists():
        raise SystemExit(f"debug log not found: {debug}")
    return server, debug


def parse_server_log(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    turns: dict[str, dict[str, Any]] = {}
    session_order: list[str] = []
    for line_no, line in enumerate(path.open(errors="replace"), 1):
        text = line.rstrip()
        match = COMMIT_RE.search(text)
        if match:
            session_id, turn_id, payload_text = match.groups()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = {}
            turns.setdefault(turn_id, {}).update(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "commit_line": line_no,
                    "history_turn_count": payload.get("history_turn_count"),
                    "current_audio_chunks": payload.get("audio_chunk_count"),
                    "current_image_frames": payload.get("image_frame_count"),
                    "text_present": payload.get("text") is not None,
                }
            )
            if session_id not in session_order:
                session_order.append(session_id)
            continue
        match = REPLY_RE.search(text)
        if match:
            session_id, turn_id, elapsed, chars = match.groups()
            turns.setdefault(turn_id, {}).update(
                {
                    "session_id": session_id,
                    "reply_ms": float(elapsed),
                    "reply_chars": int(chars),
                    "reply_line": line_no,
                }
            )
            continue
        match = ACTION_RE.search(text)
        if match:
            session_id, turn_id, elapsed, action = match.groups()
            turns.setdefault(turn_id, {}).update(
                {
                    "session_id": session_id,
                    "action_ms": float(elapsed),
                    "action": action,
                    "action_line": line_no,
                }
            )
    return turns, session_order


def stage_name(request_id: str) -> str:
    if request_id.endswith("-category"):
        return "category"
    if request_id.endswith("-child"):
        return "child"
    return "single"


def parse_debug_log(path: Path, turns: dict[str, dict[str, Any]]) -> None:
    for line_no, line in enumerate(path.open(errors="replace"), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = event.get("request_id", "")
        match = TURN_RE.search(request_id)
        if not match:
            continue
        session_id, turn_id = match.groups()
        turn = turns.setdefault(
            turn_id, {"session_id": session_id, "turn_id": turn_id}
        )
        stage = stage_name(request_id)
        target = turn.setdefault("stages", {}).setdefault(stage, {})
        if event.get("event") == "action_scoring_prompt_rendered":
            target.update(
                {
                    "prompt_tokens": event.get("prompt_tokens"),
                    "full_prompt": event.get("full_prompt", ""),
                    "rendered_line": line_no,
                }
            )
        elif event.get("event") == "action_scoring_started":
            logical = event.get("full_logical_input", {})
            target.update(
                {
                    "audio_count": len(logical.get("audios", [])),
                    "image_count": len(logical.get("images", [])),
                    "history_audio_count": len(
                        logical.get("history_audios", [])
                    ),
                    "history_image_count": len(
                        logical.get("history_images", [])
                    ),
                }
            )
        elif event.get("event") == "action_scoring_completed":
            scores = event.get("scores") or []
            top = max(
                scores,
                key=lambda item: item.get("mean_logprob", float("-inf")),
                default={},
            )
            stats = event.get("stats") or {}
            target.update(
                {
                    "elapsed_ms": event.get("elapsed_ms"),
                    "prefix_cached": event.get("prefix_cached"),
                    "prefix_tokens": stats.get("prefix_token_count"),
                    "candidate_count": len(scores),
                    "top_candidate": top.get("candidate_id"),
                    "top_ppl": top.get("ppl"),
                    "completed_line": line_no,
                }
            )


def choose_session(
    turns: dict[str, dict[str, Any]], order: list[str], requested: str | None
) -> str:
    if requested:
        return requested
    for session_id in reversed(order):
        if any(turn.get("session_id") == session_id for turn in turns.values()):
            return session_id
    raise SystemExit("no session records found")


def successful_turn(turn: dict[str, Any], require_reply: bool) -> bool:
    if "action_ms" not in turn:
        return False
    if require_reply and "reply_ms" not in turn:
        return False
    stages = turn.get("stages", {})
    if "single" in stages:
        return "elapsed_ms" in stages["single"]
    return all(
        "elapsed_ms" in stages.get(stage, {})
        for stage in ("category", "child")
    )


def metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    data = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]
    if not data:
        return None
    return {
        "count": len(data),
        "min": min(data),
        "avg": statistics.fmean(data),
        "max": max(data),
    }


def token_breakdown(
    turn: dict[str, Any], model_path: str
) -> dict[str, Any] | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {
            "error": (
                "transformers is unavailable; use the local "
                "sglang-omni environment"
            )
        }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": f"tokenizer load failed: {exc}"}

    def encoded(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    result: dict[str, Any] = {}
    for stage, data in turn.get("stages", {}).items():
        prompt = data.get("full_prompt")
        logged = data.get("prompt_tokens")
        if not prompt or logged is None:
            continue
        segments = re.findall(
            r"<\|im_start\|>.*?(?=<\|im_start\|>|$)",
            prompt,
            re.S,
        )
        if len(segments) < 2:
            continue
        system = encoded(segments[0])
        history_assistant = sum(
            encoded(segment)
            for segment in segments[:-1]
            if "<|im_start|>assistant" in segment
        )
        current_user = segments[-2]
        before, separator, after = current_user.partition(
            "<|audio_end|>"
        )
        current_instruction = encoded(after) if separator else 0
        residual = int(logged) - system - history_assistant - current_instruction
        result[stage] = {
            "prompt_tokens": int(logged),
            "system_tokens": system,
            "history_assistant_tokens": history_assistant,
            "current_instruction_tokens": current_instruction,
            "media_and_message_structure_residual": residual,
        }
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    server, debug = load_logs(args)
    turns, order = parse_server_log(server)
    parse_debug_log(debug, turns)
    session_id = choose_session(turns, order, args.session_id)
    session_turns = sorted(
        [
            turn
            for turn in turns.values()
            if turn.get("session_id") == session_id
        ],
        key=lambda item: item.get("commit_line", 0),
    )
    successful = [
        turn
        for turn in session_turns
        if successful_turn(turn, args.require_reply)
    ]
    stage_rows: dict[str, list[dict[str, Any]]] = {
        "category": [],
        "child": [],
        "single": [],
    }
    for turn in successful:
        for stage in stage_rows:
            elapsed = turn.get("stages", {}).get(stage, {}).get("elapsed_ms")
            if elapsed is not None:
                stage_rows[stage].append({"elapsed_ms": elapsed})
    max_prompt = max(
        (
            stage.get("prompt_tokens", 0)
            for turn in successful
            for stage in turn.get("stages", {}).values()
        ),
        default=0,
    )
    report: dict[str, Any] = {
        "session_id": session_id,
        "server_log": str(server),
        "debug_log": str(debug),
        "max_seq_len": args.max_seq_len,
        "turn_count": len(session_turns),
        "successful_action_turn_count": len(successful),
        "aggregate": {
            "action_ms": metric_stats(successful, "action_ms"),
            "category_ms": metric_stats(stage_rows["category"], "elapsed_ms"),
            "child_ms": metric_stats(stage_rows["child"], "elapsed_ms"),
            "single_ms": metric_stats(stage_rows["single"], "elapsed_ms"),
            "max_single_prompt_tokens": max_prompt,
        },
        "turns": [],
    }
    for turn in session_turns:
        stage_data = turn.get("stages", {})
        row = {
            key: turn.get(key)
            for key in (
                "turn_id",
                "history_turn_count",
                "current_audio_chunks",
                "current_image_frames",
                "reply_ms",
                "reply_chars",
                "action_ms",
                "action",
            )
        }
        row["success"] = turn in successful
        row["stages"] = {}
        for name, data in stage_data.items():
            row["stages"][name] = {
                key: data.get(key)
                for key in (
                    "elapsed_ms",
                    "prompt_tokens",
                    "candidate_count",
                    "prefix_cached",
                    "prefix_tokens",
                    "top_candidate",
                    "top_ppl",
                    "audio_count",
                    "image_count",
                    "history_audio_count",
                    "history_image_count",
                )
                if data.get(key) is not None
            }
        report["turns"].append(row)
    if args.token_breakdown and successful:
        report["latest_token_breakdown"] = token_breakdown(
            successful[-1], args.model_path
        )
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"session: {report['session_id']}")
    print(
        "action-successful turns: "
        f"{report['successful_action_turn_count']}/{report['turn_count']}"
    )
    aggregate = report["aggregate"]
    max_prompt = aggregate["max_single_prompt_tokens"]
    print(
        f"max single-stage prompt: {max_prompt} / "
        f"{report['max_seq_len']} tokens"
    )
    print(
        "status:",
        "within limit" if max_prompt < report["max_seq_len"]
        else "EXCEEDS LIMIT",
    )
    print()
    print(
        "turn | history | category_ms | child_ms | action_ms | "
        "action | prompt_tokens | prefix_cached"
    )
    for row in report["turns"]:
        stages = row["stages"]
        category = stages.get("category", {})
        child = stages.get("child", {})
        single = stages.get("single", {})
        prompts = [
            stage.get("prompt_tokens")
            for stage in stages.values()
            if stage.get("prompt_tokens") is not None
        ]
        cached = [
            stage.get("prefix_cached")
            for stage in stages.values()
            if stage.get("prefix_cached") is not None
        ]
        print(
            f"{row.get('turn_id', '-')} | "
            f"{row.get('history_turn_count', '-')} | "
            f"{category.get('elapsed_ms', single.get('elapsed_ms', '-'))} | "
            f"{child.get('elapsed_ms', '-')} | "
            f"{row.get('action_ms', '-')} | "
            f"{row.get('action', '-')} | "
            f"{','.join(map(str, prompts)) or '-'} | "
            f"{all(cached) if cached else '-'}"
        )
    print()
    for name, data in aggregate.items():
        if isinstance(data, dict) and {"min", "avg", "max"} <= data.keys():
            print(
                f"{name}: min={data['min']:.3f} "
                f"avg={data['avg']:.3f} max={data['max']:.3f} ms"
            )
    if "latest_token_breakdown" in report:
        print("\nlatest token breakdown:")
        print(
            json.dumps(
                report["latest_token_breakdown"],
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
