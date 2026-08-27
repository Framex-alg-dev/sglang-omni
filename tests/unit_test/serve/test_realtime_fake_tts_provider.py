# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from scripts import realtime_fake_tts_provider as provider


class FakeServerWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self._events = iter(json.dumps(event) for event in events)
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))


@pytest.mark.asyncio
async def test_provider_reuses_connection_for_two_responses() -> None:
    websocket = FakeServerWebSocket(
        [
            {"type": "input_text_buffer.append", "text": "one"},
            {"type": "input_text_buffer.commit"},
            {"type": "input_text_buffer.append", "text": "two"},
            {"type": "input_text_buffer.commit"},
        ]
    )
    await provider.handle_connection(websocket)

    assert [event["type"] for event in websocket.sent] == [
        "session.created",
        "response.created",
        "response.audio.delta",
        "response.audio.done",
        "response.done",
        "response.created",
        "response.audio.delta",
        "response.audio.done",
        "response.done",
    ]
    assert [
        event["response_id"]
        for event in websocket.sent
        if event["type"] == "response.created"
    ] == ["fake-tts-response-1", "fake-tts-response-2"]


@pytest.mark.asyncio
async def test_provider_cancel_discards_active_response_and_accepts_next() -> None:
    websocket = FakeServerWebSocket(
        [
            {"type": "input_text_buffer.append", "text": "cancel me"},
            {"type": "response.cancel"},
            {"type": "input_text_buffer.append", "text": "next"},
            {"type": "input_text_buffer.commit"},
        ]
    )
    await provider.handle_connection(websocket)

    types = [event["type"] for event in websocket.sent]
    assert types.count("response.done") == 1
    assert types.count("response.created") == 2
    assert types[-2:] == ["response.audio.done", "response.done"]
