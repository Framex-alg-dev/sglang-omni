# SPDX-License-Identifier: Apache-2.0
"""Deterministic process-external WebSocket provider for embedded TTS smoke tests."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

logger = logging.getLogger(__name__)

DEFAULT_PCM = b"\x00\x00\x10\x00\x00\x00\xf0\xff"


async def handle_connection(websocket: ServerConnection) -> None:
    """Serve sequential synthesis requests on one reusable connection."""
    await websocket.send(json.dumps({"type": "session.created"}))
    parts: list[str] = []
    response_id: str | None = None
    response_number = 0
    async for message in websocket:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "code": "invalid_json"}))
            continue
        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type == "input_text_buffer.append":
            text = event.get("text")
            if not isinstance(text, str) or not text:
                await websocket.send(
                    json.dumps({"type": "error", "code": "invalid_text"})
                )
                continue
            if response_id is None:
                response_number += 1
                response_id = f"fake-tts-response-{response_number}"
                await websocket.send(
                    json.dumps({"type": "response.created", "response_id": response_id})
                )
            parts.append(text)
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(DEFAULT_PCM).decode("ascii"),
                    }
                )
            )
        elif event_type == "input_text_buffer.commit":
            if response_id is None or not parts:
                await websocket.send(
                    json.dumps({"type": "error", "code": "commit_without_text"})
                )
                continue
            await websocket.send(json.dumps({"type": "response.audio.done"}))
            await websocket.send(json.dumps({"type": "response.done"}))
            logger.info(
                "Completed fake TTS response %s (%d text chunks)",
                response_id,
                len(parts),
            )
            parts.clear()
            response_id = None
        elif event_type == "response.cancel":
            parts.clear()
            response_id = None
            logger.info("Cancelled active fake TTS response")
        else:
            await websocket.send(
                json.dumps({"type": "error", "code": "unsupported_event"})
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="info")
    return parser


async def run(host: str, port: int) -> None:
    async with serve(handle_connection, host, port):
        logger.info("Fake TTS provider listening on ws://%s:%d", host, port)
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    try:
        asyncio.run(run(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
