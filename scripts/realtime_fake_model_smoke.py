# SPDX-License-Identifier: Apache-2.0
"""Process-external smoke test for the Realtime development model server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "tests/data/query_to_draw.wav"
REQUIRED_EVENTS = (
    "session.created",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "response.created",
    "response.text.delta",
    "response.text.done",
    "response.done",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
)


def load_pcm16_16k_mono(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"audio fixture does not exist: {path}")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getframerate() != 16000
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("audio fixture must be uncompressed mono 16 kHz PCM16 WAV")
        pcm = audio.readframes(audio.getnframes())
    if not pcm:
        raise ValueError("audio fixture must contain at least one PCM frame")
    return pcm + b"\x00\x00" * 16000


async def receive_event(websocket: Any, *, timeout: float) -> dict[str, Any]:
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    event = json.loads(raw)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise AssertionError(f"invalid Realtime event: {event!r}")
    return event


async def stream_audio(websocket: Any, pcm: bytes, *, chunk_ms: int = 200) -> None:
    chunk_bytes = 16000 * chunk_ms // 1000 * 2
    for offset in range(0, len(pcm), chunk_bytes):
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm[offset : offset + chunk_bytes]
                    ).decode("ascii"),
                }
            )
        )


def validate_events(
    events: list[dict[str, Any]], *, response_text: str, transcript_text: str
) -> None:
    types = [event["type"] for event in events]
    missing = [event_type for event_type in REQUIRED_EVENTS if event_type not in types]
    if missing:
        raise AssertionError(f"missing events {missing}; received {types}")
    positions = [types.index(event_type) for event_type in REQUIRED_EVENTS]
    if positions != sorted(positions):
        raise AssertionError(
            "Realtime events are out of order; expected "
            f"{list(REQUIRED_EVENTS)}, received {types}"
        )

    response_done = next(event for event in events if event["type"] == "response.done")
    actual_response = response_done["response"]["output"][0]["content"][0]["text"]
    response_deltas = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.text.delta"
    )
    response_text_done = next(
        event for event in events if event["type"] == "response.text.done"
    ).get("text")
    completed = next(
        event
        for event in events
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    )
    if actual_response != response_text:
        raise AssertionError(
            f"response mismatch: expected {response_text!r}, got {actual_response!r}"
        )
    if response_deltas != response_text or response_text_done != response_text:
        raise AssertionError(
            "response stream mismatch: "
            f"expected {response_text!r}, deltas produced {response_deltas!r}, "
            f"text.done produced {response_text_done!r}"
        )
    transcript_deltas = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "conversation.item.input_audio_transcription.delta"
    )
    if completed.get("transcript") != transcript_text:
        raise AssertionError(
            "transcription mismatch: "
            f"expected {transcript_text!r}, got {completed.get('transcript')!r}"
        )
    if transcript_deltas != transcript_text:
        raise AssertionError(
            "transcription stream mismatch: "
            f"expected {transcript_text!r}, deltas produced {transcript_deltas!r}"
        )


async def run_smoke(
    *,
    url: str,
    pcm: bytes,
    response_text: str,
    transcript_text: str,
    timeout: float,
    connect: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    if connect is None:
        import websockets

        connect = websockets.connect
    events: list[dict[str, Any]] = []
    async with connect(url, open_timeout=timeout) as websocket:
        created = await receive_event(websocket, timeout=timeout)
        events.append(created)
        if created["type"] != "session.created":
            raise AssertionError(
                f"first event must be session.created, got {created!r}"
            )
        await stream_audio(websocket, pcm)
        for _ in range(300):
            event = await receive_event(websocket, timeout=timeout)
            events.append(event)
            if event["type"] == "error":
                raise AssertionError(f"server returned error event: {event!r}")
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                break
        else:
            raise AssertionError(
                "terminal transcription event not received; events="
                f"{[event['type'] for event in events]}"
            )
    validate_events(
        events, response_text=response_text, transcript_text=transcript_text
    )
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--response-text", required=True)
    parser.add_argument("--transcript-text", required=True)
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
                pcm=load_pcm16_16k_mono(args.audio),
                response_text=args.response_text,
                transcript_text=args.transcript_text,
                timeout=args.timeout,
            )
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: Realtime fake model emitted the expected response and "
        f"transcription ({len(events)} events)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
