#!/usr/bin/env python3
"""Measure avatar-image prefetch on the realtime action critical path."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import websockets
from PIL import Image, ImageDraw


DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "sglang_omni/assets/character_limited_action_global_catalog.json"
)


def allowed_candidates(path: Path) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"candidate_id": str(child["candidate_id"])}
        for category in document["categories"]
        for child in category["children"]
    ]


def avatar_jpeg(index: int) -> str:
    """Create a stable-size but content-unique 640x480 avatar placeholder."""

    image = Image.new(
        "RGB",
        (640, 480),
        color=((37 * index) % 255, (71 * index) % 255, (113 * index) % 255),
    )
    draw = ImageDraw.Draw(image)
    draw.ellipse((220, 45, 420, 245), fill=(225, 190, 165))
    draw.rectangle((260, 245, 380, 470), fill=(40, 80, 170))
    draw.text((16, 16), f"avatar-prefetch-{index}", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def receive(ws: Any, timeout: float) -> dict[str, Any]:
    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
    if event.get("type") == "error":
        raise RuntimeError(json.dumps(event, ensure_ascii=False))
    return event


async def expect(ws: Any, event_type: str, timeout: float) -> dict[str, Any]:
    while True:
        event = await receive(ws, timeout)
        if event.get("type") == event_type:
            return event


async def run(args: argparse.Namespace) -> dict[str, Any]:
    session_id = f"avatar-prefetch-bench-{uuid.uuid4().hex[:10]}"
    rows: list[dict[str, Any]] = []
    async with websockets.connect(
        args.url,
        open_timeout=args.timeout,
        close_timeout=args.timeout,
        max_size=None,
        proxy=None,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "protocol_version": 1,
                    "session_id": session_id,
                    "outputs": ["action"],
                    "locale": "zh-CN",
                    "action": {
                        "allowed_candidates": allowed_candidates(args.catalog),
                    },
                    "diagnostics": {"include_action_scores": False},
                },
                ensure_ascii=False,
            )
        )
        started = await expect(ws, "session.started", args.timeout)

        for index in range(1, args.turns + 1):
            turn_id = f"turn-avatar-prefetch-{index}"
            await ws.send(
                json.dumps(
                    {"type": "turn.start", "turn_id": turn_id, "origin": "user"}
                )
            )
            await expect(ws, "turn.started", args.timeout)

            await ws.send(
                json.dumps(
                    {
                        "type": "input.image.append",
                        "turn_id": turn_id,
                        "seq": 1,
                        "capture_timestamp_ms": 1,
                        "image_source": "avatar_current",
                        "media_type": "image/jpeg",
                        "data": avatar_jpeg(index),
                    }
                )
            )
            await expect(ws, "input.ack", args.timeout)
            await asyncio.sleep(args.prefetch_lead_seconds)

            await ws.send(
                json.dumps(
                    {
                        "type": "input.text.set",
                        "turn_id": turn_id,
                        "text": args.text,
                    },
                    ensure_ascii=False,
                )
            )
            await expect(ws, "input.text.ack", args.timeout)
            committed_at = time.perf_counter()
            await ws.send(
                json.dumps(
                    {
                        "type": "turn.commit",
                        "turn_id": turn_id,
                        "avatar_state": {
                            "pose": "seated",
                            "gaze": "camera",
                            "hands": "resting",
                        },
                    },
                    ensure_ascii=False,
                )
            )

            action_ready_ms = None
            result = None
            while result is None:
                event = await receive(ws, args.timeout)
                event_type = event.get("type")
                if event_type == "turn.action.ready" and action_ready_ms is None:
                    action_ready_ms = round(
                        (time.perf_counter() - committed_at) * 1000.0, 3
                    )
                elif event_type == "turn.result":
                    result = event
            rows.append(
                {
                    "turn_id": turn_id,
                    "action_ready_ms": action_ready_ms,
                    "result_ms": round(
                        (time.perf_counter() - committed_at) * 1000.0, 3
                    ),
                    "action": result.get("action"),
                    "server_timing": result.get("timing"),
                }
            )

        await ws.send(json.dumps({"type": "session.close", "reason": "benchmark"}))

    ready = [float(row["action_ready_ms"]) for row in rows]
    output = {
        "session_id": session_id,
        "server_prefetch_enabled": started.get(
            "avatar_image_encoder_prefetch_enabled"
        ),
        "prefetch_lead_seconds": args.prefetch_lead_seconds,
        "text": args.text,
        "turns": rows,
        "summary": {
            "count": len(ready),
            "min_ms": round(min(ready), 3),
            "median_ms": round(statistics.median(ready), 3),
            "mean_ms": round(statistics.mean(ready), 3),
            "max_ms": round(max(ready), 3),
        },
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18004/v1/session/realtime"
    )
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--prefetch-lead-seconds", type=float, default=1.0)
    parser.add_argument("--text", default="手臂交叠")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.turns < 1:
        parser.error("--turns must be positive")
    if args.prefetch_lead_seconds < 0:
        parser.error("--prefetch-lead-seconds must be non-negative")
    return args


if __name__ == "__main__":
    result = asyncio.run(run(parse_args()))
    print(
        json.dumps(
            {
                "session_id": result["session_id"],
                "server_prefetch_enabled": result[
                    "server_prefetch_enabled"
                ],
                "summary": result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
