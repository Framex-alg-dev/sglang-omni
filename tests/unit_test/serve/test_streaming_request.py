from __future__ import annotations

import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from sglang_omni.serve.streaming_request import run_until_websocket_disconnect


class _DisconnectingSocket:
    async def receive(self):
        await asyncio.sleep(0)
        return {"type": "websocket.disconnect"}


def test_disconnect_cancels_model_task_and_calls_abort() -> None:
    aborted = False
    operation_cancelled = False

    async def operation():
        nonlocal operation_cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            operation_cancelled = True
            raise

    async def abort():
        nonlocal aborted
        aborted = True

    async def drive() -> None:
        with pytest.raises(WebSocketDisconnect):
            await run_until_websocket_disconnect(
                _DisconnectingSocket(),
                operation(),
                abort=abort,
            )

    asyncio.run(drive())
    assert aborted
    assert operation_cancelled


def test_completed_operation_owns_and_cleans_disconnect_watcher() -> None:
    class WaitingSocket:
        cancelled = False

        async def receive(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    socket = WaitingSocket()

    async def drive():
        return await run_until_websocket_disconnect(socket, asyncio.sleep(0, result=7))

    assert asyncio.run(drive()) == 7
    assert socket.cancelled
