# SPDX-License-Identifier: Apache-2.0
"""Process-external smoke test for the Session Realtime development server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
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
    outputs_by_mode = {
        "text": ["text"],
        "action": ["action"],
        "fusion": ["text", "action"],
        "text-audio": ["text", "audio"],
        "fusion-audio": ["text", "audio", "action"],
    }
    outputs = outputs_by_mode[mode]
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
    if "action" in outputs and "text" in outputs:
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
        # A committed second turn exercises cancellation and late-terminal cleanup.
        await _send_turn_input(
            websocket, turn_id="turn-cancel", text="取消", pcm=None, timeout=timeout
        )
        await websocket.send(
            json.dumps({"type": "turn.commit", "turn_id": "turn-cancel"})
        )
        events.append(await _expect(websocket, "turn.committed", timeout))
        await websocket.send(
            json.dumps({"type": "turn.cancel", "turn_id": "turn-cancel"})
        )
        while True:
            event = await _receive(websocket, timeout)
            events.append(event)
            if event["type"] == "turn.cancelled":
                break
        grace_deadline = time.monotonic() + min(timeout, 0.2)
        while True:
            remaining = grace_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                late_event = await _receive(websocket, remaining)
            except asyncio.TimeoutError:
                break
            if late_event.get("turn_id") == "turn-cancel" and late_event["type"] in {
                "response.text.done",
                "response.audio.done",
                "response.done",
                "turn.result",
            }:
                raise AssertionError(
                    f"late terminal event after turn.cancelled: {late_event!r}"
                )
    return events


def _validate_turn_events(
    events: list[dict[str, Any]], *, mode: str, response_text: str
) -> None:
    types = [event["type"] for event in events]
    required = ["session.started", "turn.committed"]
    has_text = mode in {"text", "fusion", "text-audio", "fusion-audio"}
    has_audio = mode in {"text-audio", "fusion-audio"}
    has_action = mode in {"action", "fusion", "fusion-audio"}
    expected_outputs = [
        output
        for output, enabled in (
            ("text", has_text),
            ("audio", has_audio),
            ("action", has_action),
        )
        if enabled
    ]
    if has_audio and events[0].get("outputs") != expected_outputs:
        raise AssertionError(f"session outputs mismatch: {events[0]!r}")
    if has_text:
        required.extend(["response.created", "response.text.done", "response.done"])
    if has_audio:
        required.extend(["response.audio.delta", "response.audio.done"])
    if has_action:
        required.append("turn.action.ready")
    required.append("turn.result")
    missing = [event_type for event_type in required if event_type not in types]
    if missing:
        raise AssertionError(f"missing events {missing}; received {types}")
    order_constraints = [
        ("session.started", "turn.committed"),
        ("turn.committed", "turn.result"),
    ]
    if has_text:
        order_constraints.extend(
            [
                ("turn.committed", "response.created"),
                ("response.created", "response.text.done"),
                ("response.text.done", "response.done"),
                ("response.done", "turn.result"),
            ]
        )
    if has_audio:
        order_constraints.extend(
            [
                ("response.created", "response.audio.delta"),
                ("response.audio.delta", "response.audio.done"),
                ("response.audio.done", "response.done"),
            ]
        )
    if has_action:
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
    if has_text:
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
    session_id = events[0].get("session_id")
    if has_audio and (
        result.get("session_id") != session_id
        or result.get("turn_id") != "turn-1"
        or result.get("status") != "completed"
        or result.get("outputs") != {output: "completed" for output in expected_outputs}
    ):
        raise AssertionError(f"turn result terminal mismatch: {result!r}")
    if has_text:
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
    if has_audio:
        created = next(event for event in events if event["type"] == "response.created")
        response = created.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not response_id or any(
            created.get(key) != value
            for key, value in {"session_id": session_id, "turn_id": "turn-1"}.items()
        ):
            raise AssertionError(f"response correlation mismatch: {created!r}")
        audio_events = [
            event for event in events if event["type"] == "response.audio.delta"
        ]
        if [event.get("seq") for event in audio_events] != list(
            range(1, len(audio_events) + 1)
        ):
            raise AssertionError(f"audio seq is not monotonic: {audio_events!r}")
        for event in audio_events:
            if (
                event.get("session_id") != session_id
                or event.get("turn_id") != "turn-1"
                or event.get("response_id") != response_id
            ):
                raise AssertionError(f"audio correlation mismatch: {event!r}")
            try:
                pcm = base64.b64decode(event.get("delta", ""), validate=True)
            except ValueError as exc:
                raise AssertionError("audio delta is not valid base64") from exc
            if not pcm or len(pcm) % 2:
                raise AssertionError("audio delta is not non-empty PCM16LE")
            if event.get("audio") != {
                "format": "pcm16le",
                "sample_rate_hz": 24000,
                "channels": 1,
            }:
                raise AssertionError(f"audio format mismatch: {event!r}")
        audio_done = next(
            event for event in events if event["type"] == "response.audio.done"
        )
        if any(
            audio_done.get(key) != value
            for key, value in {
                "session_id": session_id,
                "turn_id": "turn-1",
                "response_id": response_id,
                "seq": len(audio_events),
            }.items()
        ):
            raise AssertionError(f"audio done correlation mismatch: {audio_done!r}")
        response_done = next(
            event for event in events if event["type"] == "response.done"
        )
        done_response = response_done.get("response")
        if (
            response_done.get("session_id") != session_id
            or response_done.get("turn_id") != "turn-1"
            or not isinstance(done_response, dict)
            or done_response.get("id") != response_id
            or done_response.get("status") != "completed"
        ):
            raise AssertionError(
                f"response done correlation mismatch: {response_done!r}"
            )
    if has_action:
        ready = next(event for event in events if event["type"] == "turn.action.ready")
        if (
            ready.get("action", {}).get("candidate_id") != "ADEV"
            or result.get("action", {}).get("candidate_id") != "ADEV"
        ):
            raise AssertionError(f"action result mismatch: {events!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/session/realtime")
    parser.add_argument(
        "--mode",
        choices=("text", "action", "fusion", "text-audio", "fusion-audio"),
        default="text",
    )
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
