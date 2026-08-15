#!/usr/bin/env python3
"""Benchmark realtime action scoring at several candidate micro-batch sizes.

The benchmark uses the external session WebSocket protocol and the same nested
catalog for both selection modes. In flat_children mode the server flattens the
children; in hierarchical mode it scores categories and then selected children.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import websockets


DEFAULT_BASE_URL = "ws://127.0.0.1:18001/v1/session/realtime"
DEFAULT_CATALOG = "tests/data/actions/character_action_catalog.json"
DEFAULT_AUDIO = "/data/models/hehy/D_human_train/examples/test/bench20_examples/none_4.wav"
DEFAULT_IMAGE = "/data/models/xingmt/wan_export_step651_speedtest/ref.png"
DEFAULT_MODEL = "Qwen3-Omni-30B-A3B-Instruct"


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise ValueError(f"audio must be mono PCM16 16kHz: {path}")
        return stream.readframes(stream.getnframes())


def encode_image(path: str) -> tuple[str, str]:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ValueError(f"unsupported image suffix: {suffix}")
    return mime, base64.b64encode(data).decode("ascii")


def build_nested_catalog(path: str) -> tuple[list[dict[str, Any]], int, int]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = document.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("catalog has no actions")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for action_index, item in enumerate(actions):
        action_id = item.get("action_id")
        source = item.get("source") or {}
        label = source.get("source_label")
        definition = item.get("short_definition") or item.get("prompt") or label
        path_parts = item.get("category_path") or ["未分类", "未分类"]
        if not isinstance(action_id, str) or not isinstance(label, str) or not isinstance(definition, str):
            raise ValueError(f"invalid action catalog item: {item!r}")
        key = (str(path_parts[0]), str(path_parts[1] if len(path_parts) > 1 else path_parts[0]))
        # Keep the externally scored suffix short while preserving the
        # catalog's long canonical action_id for the execution result.
        grouped.setdefault(key, []).append({
            "candidate_id": f"A{action_index:03d}",
            "action_id": action_id,
            "source_label": label,
            "short_definition": definition,
        })

    categories: list[dict[str, Any]] = []
    for index, (key, children) in enumerate(sorted(grouped.items())):
        categories.append({
            "category_id": f"B{index:03d}",
            "source_label": key[1],
            "short_definition": f"{key[0]} / {key[1]}",
            "children": children,
        })
    categories.append({
        "category_id": "B_NO_ACTION",
        "source_label": "系统动作",
        "short_definition": "保持当前姿态",
        "children": [{
            "candidate_id": "A_NO_ACTION",
            "action_id": "no_action",
            "source_label": "不做动作",
            "short_definition": "保持当前姿态",
        }],
    })
    child_count = sum(len(category["children"]) for category in categories)
    return categories, len(categories), child_count


async def recv_until(ws: Any, terminal_types: set[str], timeout_s: float) -> tuple[dict[str, Any], list[str]]:
    event_types: list[str] = []
    deadline = time.perf_counter() + timeout_s
    while True:
        remaining = max(deadline - time.perf_counter(), 0.1)
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = json.loads(raw)
        event_type = event.get("type", "<missing>")
        event_types.append(event_type)
        if event_type in terminal_types:
            return event, event_types


async def run_once(
    base_url: str,
    mode: str,
    batch_size: int,
    repeat_index: int,
    catalog: list[dict[str, Any]],
    pcm: bytes,
    image_mime: str,
    image_b64: str,
    timeout_s: float,
) -> dict[str, Any]:
    session_id = f"batch-bench-{mode}-{batch_size}-{repeat_index}-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{repeat_index}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "session_id": session_id,
        "turn_id": turn_id,
        "mode": mode,
        "batch_size": batch_size,
        "repeat_index": repeat_index,
        "status": "error",
        "event_types": [],
    }
    async with websockets.connect(base_url, open_timeout=timeout_s, close_timeout=timeout_s, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "session.start",
            "session_id": session_id,
            "model": DEFAULT_MODEL,
            "selection_mode": mode,
            "include_scores": True,
            "input_audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
            "action_candidates": catalog,
        }, ensure_ascii=False))
        session_started, events = await recv_until(ws, {"session.started", "error"}, timeout_s)
        result["event_types"].extend(events)
        if session_started.get("type") != "session.started":
            result["error"] = session_started
            return result

        await ws.send(json.dumps({"type": "turn.start", "session_id": session_id, "turn_id": turn_id}))
        turn_started, events = await recv_until(ws, {"turn.started", "error"}, timeout_s)
        result["event_types"].extend(events)
        if turn_started.get("type") != "turn.started":
            result["error"] = turn_started
            return result

        chunk_bytes = 16000 * 200 // 1000 * 2
        for seq, start in enumerate(range(0, len(pcm), chunk_bytes), start=1):
            chunk = pcm[start : start + chunk_bytes]
            await ws.send(json.dumps({
                "type": "input_audio.append",
                "session_id": session_id,
                "turn_id": turn_id,
                "seq": seq,
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
            ack, events = await recv_until(ws, {"input.ack", "error"}, timeout_s)
            result["event_types"].extend(events)
            if ack.get("type") != "input.ack" or ack.get("error"):
                result["error"] = ack
                return result

        await ws.send(json.dumps({
            "type": "input_image.append",
            "session_id": session_id,
            "turn_id": turn_id,
            "seq": 1,
            "timestamp_ms": 1000,
            "mime_type": image_mime,
            "image": image_b64,
        }))
        image_ack, events = await recv_until(ws, {"input.ack", "error"}, timeout_s)
        result["event_types"].extend(events)
        if image_ack.get("type") != "input.ack" or image_ack.get("error"):
            result["error"] = image_ack
            return result

        commit_sent = time.perf_counter()
        await ws.send(json.dumps({
            "type": "turn.commit",
            "session_id": session_id,
            "turn_id": turn_id,
            "text": "你可以开心一点吗？",
            "avatar_state": {"pose": "seated", "gaze": "camera", "hands": "resting"},
        }, ensure_ascii=False))
        turn_result, events = await recv_until(ws, {"turn.result", "error"}, timeout_s)
        result["event_types"].extend(events)
        result["client_commit_to_result_ms"] = round((time.perf_counter() - commit_sent) * 1000.0, 3)
        result["wall_total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        if turn_result.get("type") != "turn.result":
            result["error"] = turn_result
            return result
        result["status"] = "success"
        result["turn_result"] = turn_result
        result["action"] = turn_result.get("action")
        result["timing"] = turn_result.get("timing")
        result["score_count"] = len(turn_result.get("scores") or [])
        result["prefix_cached"] = turn_result.get("media_summary", {}).get("action_context", {}).get("prefix_cached")
        result["media_summary"] = turn_result.get("media_summary")
        result["session_started"] = session_started
        await ws.send(json.dumps({"type": "session.close", "session_id": session_id}))
    return result


async def main(args: argparse.Namespace) -> None:
    catalog, category_count, child_count = build_nested_catalog(args.catalog)
    pcm = load_pcm16(args.audio)
    image_mime, image_b64 = encode_image(args.image)
    results: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeats + 1):
        results.append(await run_once(
            args.base_url, args.mode, args.batch_size, repeat_index,
            catalog, pcm, image_mime, image_b64, args.timeout,
        ))
    output = {
        "benchmark": "qwen3_omni_action_batch",
        "base_url": args.base_url,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "catalog_path": args.catalog,
        "category_count": category_count,
        "child_count_including_no_action": child_count,
        "audio": args.audio,
        "image": args.image,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "category_count": category_count,
        "child_count": child_count,
        "statuses": dict(Counter(item["status"] for item in results)),
        "server_action_compute_ms": [item.get("timing", {}).get("server_action_compute_ms") for item in results],
        "actions": [item.get("action", {}).get("action_id") for item in results],
    }, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--mode", choices=("flat_children", "hierarchical"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
