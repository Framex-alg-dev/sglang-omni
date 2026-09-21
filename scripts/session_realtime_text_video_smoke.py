# SPDX-License-Identifier: Apache-2.0
"""Send one text-plus-video turn to the Session Realtime WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import av
import websockets


def sample_video_frames(
    video_path: Path,
    *,
    max_frames: int,
    interval_seconds: float,
) -> list[tuple[int, str]]:
    """Return uniformly spaced JPEG frames as ``(timestamp_ms, base64)``."""
    sampled: list[tuple[int, str]] = []
    next_sample_seconds = 0.0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fallback_rate = float(stream.average_rate or 25)
        for index, frame in enumerate(container.decode(stream)):
            frame_seconds = (
                float(frame.time)
                if frame.time is not None
                else index / fallback_rate
            )
            if sampled and frame_seconds < next_sample_seconds:
                continue
            image = frame.to_image().convert("RGB")
            image.thumbnail((640, 480))
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=85)
            sampled.append(
                (
                    round(frame_seconds * 1000),
                    base64.b64encode(encoded.getvalue()).decode("ascii"),
                )
            )
            if len(sampled) >= max_frames:
                break
            next_sample_seconds = frame_seconds + interval_seconds
    if not sampled:
        raise ValueError(f"video contains no decodable frames: {video_path}")
    return sampled


async def receive(websocket: Any, timeout_seconds: float) -> dict[str, Any]:
    event = json.loads(
        await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
    )
    if event.get("type") == "error":
        raise RuntimeError(f"server returned an error: {event}")
    return event


async def expect(
    websocket: Any,
    event_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    event = await receive(websocket, timeout_seconds)
    if event.get("type") != event_type:
        raise RuntimeError(f"expected {event_type}, received {event}")
    return event


async def run(args: argparse.Namespace) -> None:
    frames = sample_video_frames(
        args.video,
        max_frames=args.max_frames,
        interval_seconds=args.frame_interval,
    )
    session_id = f"infra-text-video-{uuid.uuid4().hex[:12]}"
    turn_id = "turn-1"
    async with websockets.connect(
        args.url,
        open_timeout=args.timeout,
        close_timeout=args.timeout,
        max_size=None,
        proxy=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "session.start",
                    "protocol_version": 1,
                    "session_id": session_id,
                    "outputs": ["text"],
                    "locale": "zh-CN",
                    "reply": {"instructions": "使用中文简洁回答当前问题。"},
                },
                ensure_ascii=False,
            )
        )
        await expect(websocket, "session.started", args.timeout)

        await websocket.send(
            json.dumps(
                {"type": "turn.start", "turn_id": turn_id, "origin": "user"}
            )
        )
        await expect(websocket, "turn.started", args.timeout)

        await websocket.send(
            json.dumps(
                {"type": "input.text.set", "turn_id": turn_id, "text": args.text},
                ensure_ascii=False,
            )
        )
        await expect(websocket, "input.text.ack", args.timeout)

        for seq, (timestamp_ms, image_data) in enumerate(frames, start=1):
            await websocket.send(
                json.dumps(
                    {
                        "type": "input.image.append",
                        "turn_id": turn_id,
                        "seq": seq,
                        "capture_timestamp_ms": timestamp_ms,
                        "media_type": "image/jpeg",
                        "image_source": "user_camera",
                        "data": image_data,
                    }
                )
            )
            await expect(websocket, "input.image.ack", args.timeout)

        commit_started = time.perf_counter()
        await websocket.send(
            json.dumps({"type": "turn.commit", "turn_id": turn_id})
        )

        first_text_ms: float | None = None
        text_parts: list[str] = []
        result: dict[str, Any] | None = None
        while result is None:
            event = await receive(websocket, args.timeout)
            event_type = event.get("type")
            if event_type in {
                "response.text.delta",
                "response.provisional.text.delta",
            }:
                if first_text_ms is None:
                    first_text_ms = (time.perf_counter() - commit_started) * 1000
                text_parts.append(str(event.get("text") or event.get("delta") or ""))
            elif event_type == "turn.result":
                result = event

        total_ms = (time.perf_counter() - commit_started) * 1000
        reply = result.get("reply")
        result_text = reply.get("text", "") if isinstance(reply, dict) else ""
        streamed_text = "".join(text_parts)
        print(f"session_id={session_id}")
        print(f"frames={len(frames)}")
        print(f"first_text_after_commit_ms={first_text_ms}")
        print(f"turn_result_after_commit_ms={total_ms:.3f}")
        print(f"reply={result_text or streamed_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:18008/v1/session/realtime",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", default="请简要描述视频中发生了什么。")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.frame_interval <= 0:
        parser.error("--frame-interval must be positive")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
