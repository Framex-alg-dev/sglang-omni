from __future__ import annotations

import asyncio

import pytest

from sglang_omni.serve import request_log_context
from sglang_omni.serve.request_log_context import (
    model_request_log,
    payload_request_log_context,
)
from sglang_omni.utils.framex_service_logs import (
    bind_service_log_context,
    current_service_log_context,
)


def test_payload_context_is_bound_to_the_complete_model_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        request_log_context,
        "emit_service_event",
        lambda _service, event, **fields: events.append((event, fields)),
    )
    context = payload_request_log_context(
        {
            "log_context": {
                "request_id": "req_0123456789abcdfg",
                "root_session_id": "root-1",
                "turn_id": "turn-1",
                "agent_session_id": "agent-1",
            }
        },
        fallback_request_id="operation-1",
    )

    async def run() -> None:
        async with model_request_log(
            "omni-reply-worker",
            "task_brain",
            context,
            operation_id="operation-1",
            logger_name=__name__,
        ):
            assert current_service_log_context().request_id == (
                "req_0123456789abcdfg"
            )
            assert current_service_log_context().root_session_id == "root-1"
            assert current_service_log_context().turn_id == "turn-1"

    asyncio.run(run())

    assert [event for event, _ in events] == [
        "inference.started",
        "inference.completed",
    ]
    assert events[-1][1]["status"] == 200
    assert current_service_log_context().request_id is None


def test_bound_http_context_wins_over_payload_context() -> None:
    with bind_service_log_context(
        request_id="http-request",
        root_session_id="http-root",
        turn_id="http-turn",
    ):
        context = payload_request_log_context(
            {
                "log_context": {
                    "request_id": "payload-request",
                    "root_session_id": "payload-root",
                    "turn_id": "payload-turn",
                }
            },
            fallback_request_id="operation-1",
        )

    assert context.request_id == "http-request"
    assert context.root_session_id == "http-root"
    assert context.turn_id == "http-turn"


def test_model_lifecycle_has_exactly_one_cancelled_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        request_log_context,
        "emit_service_event",
        lambda _service, event, **_fields: events.append(event),
    )
    context = payload_request_log_context(
        {},
        fallback_request_id="operation-1",
    )

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            async with model_request_log(
                "omni-action-worker",
                "performance_control",
                context,
                operation_id="operation-1",
                logger_name=__name__,
            ):
                raise asyncio.CancelledError

    asyncio.run(run())
    assert events == ["inference.started", "inference.cancelled"]
