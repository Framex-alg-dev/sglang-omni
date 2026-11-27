from __future__ import annotations

import json
import re

import pytest

from sglang_omni.utils.framex_service_logs import (
    REQUEST_ID_ALPHABET,
    adopt_request_id,
    bind_service_log_context,
    configure_service_logging,
    correlation_id_or_digest,
    emit_service_event,
    shutdown_service_logging,
)


def _records(tmp_path, service: str) -> list[dict]:
    path = tmp_path / service / "app.log"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_framex_service_log_contract_and_context(tmp_path) -> None:
    service = "omni-action-worker"
    configure_service_logging(
        service,
        log_root=tmp_path,
        version="abc123def456",
        host="worker-1",
        include_stdout=False,
    )
    try:
        with bind_service_log_context(
            request_id="req_0123456789abcdfg",
            root_session_id="ses_0123456789abcdfg",
            turn_id="turn-1",
        ):
            emit_service_event(
                service,
                "dependency.call",
                logger_name="tests.action",
                peer="omni_reply_worker",
                operation="reply",
                status=200,
                duration_ms=12.5,
                agent_session_id="agent-1",
            )
    finally:
        shutdown_service_logging(service)

    assert _records(tmp_path, service) == [
        {
            "timestamp": _records(tmp_path, service)[0]["timestamp"],
            "level": "INFO",
            "service": service,
            "env": "shared",
            "version": "abc123def456",
            "host": "worker-1",
            "logger": "tests.action",
            "event": "dependency.call",
            "request_id": "req_0123456789abcdfg",
            "root_session_id": "ses_0123456789abcdfg",
            "turn_id": "turn-1",
            "peer": "omni_reply_worker",
            "operation": "reply",
            "status": 200,
            "duration_ms": 12.5,
            "agent_session_id": "agent-1",
        }
    ]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        _records(tmp_path, service)[0]["timestamp"],
    )


def test_framex_service_log_redacts_content_and_rejects_session_id(tmp_path) -> None:
    service = "omni-reply-worker"
    configure_service_logging(
        service,
        log_root=tmp_path,
        version="deadbeef0000",
        include_stdout=False,
    )
    try:
        emit_service_event(
            service,
            "inference.completed",
            user_text="private",
            system_prompt="private",
            authorization_token="private",
            status=200,
        )
        with pytest.raises(ValueError, match="ambiguous session_id"):
            emit_service_event(
                service, "inference.completed", session_id="ambiguous"
            )
    finally:
        shutdown_service_logging(service)

    record = _records(tmp_path, service)[0]
    assert "user_text" not in record
    assert "system_prompt" not in record
    assert "authorization_token" not in record
    assert record["redacted_fields"] == [
        "authorization_token",
        "system_prompt",
        "user_text",
    ]


def test_request_id_adoption_contract() -> None:
    assert adopt_request_id("client:request-1") == ("client:request-1", False)
    generated, invalid = adopt_request_id("contains spaces")
    assert invalid is True
    assert generated.startswith("req_")
    assert len(generated) == 20
    assert set(generated[4:]) <= set(REQUEST_ID_ALPHABET)


def test_legacy_correlation_id_is_stable_and_non_reversible() -> None:
    assert correlation_id_or_digest("already-valid", prefix="legacy") == (
        "already-valid"
    )
    first = correlation_id_or_digest("legacy id with spaces", prefix="legacy")
    second = correlation_id_or_digest("legacy id with spaces", prefix="legacy")
    assert first == second
    assert first.startswith("legacy_")
    assert "legacy id with spaces" not in first


def test_secret_values_are_redacted_from_message_and_error(tmp_path) -> None:
    service = "omni-event-worker"
    configure_service_logging(
        service,
        log_root=tmp_path,
        version="deadbeef0000",
        include_stdout=False,
    )
    try:
        emit_service_event(
            service,
            "dependency.call",
            message="Authorization: Bearer very-secret-value",
            error_message="token=another-secret 13812345678",
        )
    finally:
        shutdown_service_logging(service)

    record = _records(tmp_path, service)[0]
    assert "very-secret-value" not in record["message"]
    assert "another-secret" not in record["error_message"]
    assert "13812345678" not in record["error_message"]
    assert "138****5678" in record["error_message"]
