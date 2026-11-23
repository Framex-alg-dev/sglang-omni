# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.serve import http_request_logs
from sglang_omni.serve.http_request_logs import FramexHTTPRequestLogMiddleware
from sglang_omni.utils.framex_service_logs import current_service_log_context


def _http_scope(*, headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"ignored=secret",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 41035),
    }


def test_http_log_waits_for_final_stream_body_and_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        order: list[str] = []
        sent: list[dict[str, Any]] = []
        events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record_event(*args: Any, **kwargs: Any) -> None:
            events.append((args, kwargs))
            order.append("emit")
            assert current_service_log_context().request_id == "client-request-1"

        monkeypatch.setattr(http_request_logs, "emit_service_event", record_event)

        async def app(scope, receive, send) -> None:
            assert current_service_log_context().root_session_id == "root-1"
            assert current_service_log_context().turn_id == "turn-1"
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"x-request-id", b"stale"),
                    ],
                }
            )
            assert not events
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: first\n\n",
                    "more_body": True,
                }
            )
            assert not events
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": False,
                }
            )

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            order.append(message["type"])

        middleware = FramexHTTPRequestLogMiddleware(app)
        await middleware(
            _http_scope(
                headers=[
                    (b"x-request-id", b"client-request-1"),
                    (b"x-root-session-id", b"root-1"),
                    (b"x-turn-id", b"turn-1"),
                ]
            ),
            receive,
            send,
        )

        assert order == [
            "http.response.start",
            "http.response.body",
            "http.response.body",
            "emit",
        ]
        response_headers = sent[0]["headers"]
        assert (b"content-type", b"text/event-stream") in response_headers
        assert [item for item in response_headers if item[0] == b"x-request-id"] == [
            (b"x-request-id", b"client-request-1")
        ]
        assert len(events) == 1
        assert events[0][0] == (
            "omni-reply-worker",
            "http.request.completed",
        )
        assert events[0][1]["outcome"] == "completed"
        assert events[0][1]["status"] == 200
        assert events[0][1]["response_started"] is True
        assert current_service_log_context().request_id is None

    asyncio.run(run())


def test_http_log_records_stream_cancellation_once_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            http_request_logs,
            "emit_service_event",
            lambda *args, **kwargs: events.append(kwargs),
        )

        async def app(scope, receive, send) -> None:
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"partial",
                    "more_body": True,
                }
            )
            raise asyncio.CancelledError

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            return None

        with pytest.raises(asyncio.CancelledError):
            await FramexHTTPRequestLogMiddleware(app)(
                _http_scope(), receive, send
            )

        assert len(events) == 1
        assert events[0]["outcome"] == "cancelled"
        assert events[0]["status"] == 200
        assert events[0]["error_type"] == "CancelledError"

    asyncio.run(run())


def test_http_log_records_disconnect_without_response_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            http_request_logs,
            "emit_service_event",
            lambda *args, **kwargs: events.append(kwargs),
        )

        async def app(scope, receive, send) -> None:
            assert (await receive())["type"] == "http.disconnect"

        async def receive() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            raise AssertionError("disconnecting request must not send a response")

        await FramexHTTPRequestLogMiddleware(app)(_http_scope(), receive, send)

        assert len(events) == 1
        assert events[0]["outcome"] == "cancelled"
        assert events[0]["status"] == 499
        assert events[0]["response_started"] is False

    asyncio.run(run())


def test_http_log_records_send_failure_once_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            http_request_logs,
            "emit_service_event",
            lambda *args, **kwargs: events.append(kwargs),
        )

        async def app(scope, receive, send) -> None:
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            raise ConnectionResetError("client vanished")

        with pytest.raises(ConnectionResetError, match="client vanished"):
            await FramexHTTPRequestLogMiddleware(app)(
                _http_scope(), receive, send
            )

        assert len(events) == 1
        assert events[0]["outcome"] == "cancelled"
        assert events[0]["status"] == 499
        assert events[0]["response_started"] is False

    asyncio.run(run())


def test_http_log_emission_failure_does_not_break_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        def fail_to_log(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("log queue unavailable")

        monkeypatch.setattr(
            http_request_logs, "emit_service_event", fail_to_log
        )

        async def app(scope, receive, send) -> None:
            await send(
                {"type": "http.response.start", "status": 204, "headers": []}
            )
            await send({"type": "http.response.body", "body": b""})

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict[str, Any]] = []

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await FramexHTTPRequestLogMiddleware(app)(_http_scope(), receive, send)
        assert [message["type"] for message in sent] == [
            "http.response.start",
            "http.response.body",
        ]

    asyncio.run(run())


def test_http_log_does_not_intercept_websocket_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        events: list[dict[str, Any]] = []
        downstream_scopes: list[dict[str, Any]] = []
        monkeypatch.setattr(
            http_request_logs,
            "emit_service_event",
            lambda *args, **kwargs: events.append(kwargs),
        )

        async def app(scope, receive, send) -> None:
            downstream_scopes.append(scope)

        async def receive() -> dict[str, Any]:
            return {"type": "websocket.connect"}

        async def send(message: dict[str, Any]) -> None:
            return None

        scope = {"type": "websocket", "path": "/v1/realtime", "headers": []}
        await FramexHTTPRequestLogMiddleware(app)(scope, receive, send)

        assert downstream_scopes == [scope]
        assert events == []

    asyncio.run(run())
