# SPDX-License-Identifier: Apache-2.0
"""Process-external smoke test for the Session Realtime development server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path
from typing import Any

DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "tests/data/query_to_draw.wav"


def load_pcm16_16k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getframerate() != 16000
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("audio must be uncompressed mono 16 kHz PCM16 WAV")
        pcm = audio.readframes(audio.getnframes())
    if not pcm:
        raise ValueError("audio must contain at least one PCM frame")
    return pcm


async def _receive(websocket: Any, timeout: float) -> dict[str, Any]:
    event = json.loads(await asyncio.wait_for(websocket.recv(), timeout))
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise AssertionError(f"invalid event: {event!r}")
    if event["type"] == "error":
        raise AssertionError(f"server error: {event!r}")
    return event


async def _expect(websocket: Any, event_type: str, timeout: float) -> dict[str, Any]:
    event = await _receive(websocket, timeout)
    if event["type"] != event_type:
        raise AssertionError(f"expected {event_type}, got {event!r}")
    return event


def _session_start(mode: str) -> dict[str, Any]:
    outputs = ["text", "action"] if mode == "fusion" else [mode]
    event: dict[str, Any] = {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": f"dev-smoke-{mode}",
        "outputs": outputs,
        "locale": "zh-CN",
    }
    if "text" in outputs:
        event["reply"] = {"instructions": "请简短回复。"}
    if "action" in outputs:
        event["action"] = {
            "fallback_category_ids": ["BDEV"],
            "allowed_candidates": [{"candidate_id": "ADEV"}],
        }
    if mode == "fusion":
        event["reply"]["unsupported_action_text"] = "该动作暂不支持。"
    return event


async def _send_turn_input(
    websocket: Any,
    *,
    turn_id: str,
    text: str | None,
    pcm: bytes | None,
    timeout: float,
) -> None:
    await websocket.send(
        json.dumps({"type": "turn.start", "turn_id": turn_id, "origin": "user"})
    )
    await _expect(websocket, "turn.started", timeout)
    if text is not None:
        await websocket.send(
            json.dumps({"type": "input.text.set", "turn_id": turn_id, "text": text})
        )
        await _expect(websocket, "input.text.ack", timeout)
    if pcm is not None:
        await websocket.send(
            json.dumps(
                {
                    "type": "input.audio.append",
                    "turn_id": turn_id,
                    "seq": 1,
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        await _expect(websocket, "input.audio.ack", timeout)


async def run_smoke(
    *,
    url: str,
    mode: str,
    response_text: str,
    text: str | None,
    pcm: bytes | None,
    timeout: float,
    connect=None,
) -> list[dict[str, Any]]:
    if connect is None:
        import websockets

        connect = websockets.connect
    events: list[dict[str, Any]] = []
    async with connect(url, open_timeout=timeout) as websocket:
        await websocket.send(json.dumps(_session_start(mode)))
        events.append(await _expect(websocket, "session.started", timeout))
        await _send_turn_input(
            websocket, turn_id="turn-1", text=text, pcm=pcm, timeout=timeout
        )
        await websocket.send(json.dumps({"type": "turn.commit", "turn_id": "turn-1"}))
        while True:
            event = await _receive(websocket, timeout)
            events.append(event)
            if event["type"] == "turn.result":
                break
        _validate_turn_events(events, mode=mode, response_text=response_text)
        # A second turn proves terminal cleanup and deterministic reuse.
        await _send_turn_input(
            websocket, turn_id="turn-cancel", text="取消", pcm=None, timeout=timeout
        )
        await websocket.send(
            json.dumps({"type": "turn.cancel", "turn_id": "turn-cancel"})
        )
        while True:
            event = await _receive(websocket, timeout)
            events.append(event)
            if event["type"] == "turn.cancelled":
                break
    return events


def _validate_turn_events(
    events: list[dict[str, Any]], *, mode: str, response_text: str
) -> None:
    types = [event["type"] for event in events]
    required = ["session.started", "turn.committed"]
    if mode in {"text", "fusion"}:
        required.extend(
            ["response.created", "response.text.done", "response.done"]
        )
    if mode in {"action", "fusion"}:
        required.append("turn.action.ready")
    required.append("turn.result")
    missing = [event_type for event_type in required if event_type not in types]
    if missing:
        raise AssertionError(f"missing events {missing}; received {types}")
    order_constraints = [
        ("session.started", "turn.committed"),
        ("turn.committed", "turn.result"),
    ]
    if mode in {"text", "fusion"}:
        order_constraints.extend(
            [
                ("turn.committed", "response.created"),
                ("response.created", "response.text.done"),
                ("response.text.done", "response.done"),
                ("response.done", "turn.result"),
            ]
        )
    if mode in {"action", "fusion"}:
        order_constraints.extend(
            [
                ("turn.committed", "turn.action.ready"),
                ("turn.action.ready", "turn.result"),
            ]
        )
    violated = [
        f"{before} < {after}"
        for before, after in order_constraints
        if types.index(before) >= types.index(after)
    ]
    if violated:
        raise AssertionError(
            f"events out of order: violated {violated}; received {types}"
        )
    if mode in {"text", "fusion"}:
        created_at = types.index("response.created")
        text_done_at = types.index("response.text.done")
        misplaced_deltas = [
            index
            for index, event_type in enumerate(types)
            if event_type == "response.text.delta"
            and not created_at < index < text_done_at
        ]
        if misplaced_deltas:
            raise AssertionError(
                "events out of order: response.text.delta must be between "
                f"response.created and response.text.done; received {types}"
            )

    result = next(event for event in events if event["type"] == "turn.result")
    if mode in {"text", "fusion"}:
        deltas = "".join(
            event.get("delta", "")
            for event in events
            if event["type"] == "response.text.delta"
        )
        done = next(event for event in events if event["type"] == "response.text.done")
        if (
            deltas != response_text
            or done.get("text") != response_text
            or result.get("reply", {}).get("text") != response_text
        ):
            raise AssertionError(f"text result mismatch: {events!r}")
    if mode in {"action", "fusion"}:
        ready = next(event for event in events if event["type"] == "turn.action.ready")
        if (
            ready.get("action", {}).get("candidate_id") != "ADEV"
            or result.get("action", {}).get("candidate_id") != "ADEV"
        ):
            raise AssertionError(f"action result mismatch: {events!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/session/realtime")
    parser.add_argument("--mode", choices=("text", "action", "fusion"), default="text")
    parser.add_argument("--response-text", default="这是本地开发模型返回的固定回复。")
    parser.add_argument("--text", default="你好")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--timeout", type=_positive_float, default=15.0)
    return parser


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        events = asyncio.run(
            run_smoke(
                url=args.url,
                mode=args.mode,
                response_text=args.response_text,
                text=args.text,
                pcm=load_pcm16_16k_mono(args.audio) if args.audio else None,
                timeout=args.timeout,
            )
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: Session Realtime {args.mode} and cancel ({len(events)} events).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
