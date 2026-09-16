from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from sglang_omni.serve.realtime.protocol.models import ProvisionalReplyState
from sglang_omni.serve.realtime.reply.provisional import ProvisionalReplyComponent


class ProvisionalHarness(ProvisionalReplyComponent):
    session_id = "test-session"

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send(self, event: dict) -> None:
        self.events.append(event)

    @staticmethod
    def _after_commit_ms(turn) -> float:
        return 1.0

    @staticmethod
    def _ensure_turn_processing(turn) -> None:
        return None


def _turn():
    return SimpleNamespace(
        turn_id="turn-1",
        trace_id="trace-1",
        request_base="request-1",
    )


def _state(*parts: str) -> ProvisionalReplyState:
    state = ProvisionalReplyState(
        response_id="reply-1",
        source="generated",
        started_at=time.perf_counter(),
        created_after_commit_ms=0.0,
    )
    state.text_parts.extend(parts)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "expected_text", "expected_status"),
    [
        (("等于", "三"), "等于三", "completed"),
        ((), "", "empty"),
    ],
)
async def test_complete_reply_signal_publishes_final_immutable_text(
    parts: tuple[str, ...],
    expected_text: str,
    expected_status: str,
) -> None:
    harness = ProvisionalHarness()
    turn = _turn()
    state = _state(*parts)

    waiter = asyncio.create_task(
        harness._resolve_complete_provisional_reply(turn, state)
    )
    await harness._finish_provisional_reply(
        turn,
        state,
        finish_reason="stop",
        usage=None,
    )
    text, status, _ = await asyncio.wait_for(waiter, timeout=0.1)

    assert text == expected_text
    assert status == expected_status
    assert state.text_completed_at is not None
    state.text_parts.append("不会改变终态")
    assert state.complete_text == expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed", "cancelled", "expected_status"),
    [
        (True, False, "failed"),
        (False, True, "cancelled"),
    ],
)
async def test_failed_or_cancelled_reply_releases_complete_text_waiter(
    failed: bool,
    cancelled: bool,
    expected_status: str,
) -> None:
    harness = ProvisionalHarness()
    turn = _turn()
    state = _state("partial")

    waiter = asyncio.create_task(
        harness._resolve_complete_provisional_reply(turn, state)
    )
    await harness._mark_provisional_reply_terminal(
        state,
        failed=failed,
        cancelled=cancelled,
    )
    text, status, _ = await asyncio.wait_for(waiter, timeout=0.1)

    assert text is None
    assert status == expected_status
