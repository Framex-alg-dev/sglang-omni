from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType

from sglang_omni.utils.framex_service_logs import (
    configure_service_logging,
    shutdown_service_logging,
)
from sglang_omni.utils.model_trace_logs import cls_safe, emit_model_trace


@dataclass(frozen=True)
class _Snapshot:
    prompt: str
    hashes: object


def test_cls_safe_preserves_text_and_omits_binary_media() -> None:
    value = cls_safe(
        {
            "system_prompt": "完整中文提示词",
            "authorization": "Bearer secret",
            "audio": {"data": "YWJjZA=="},
            "image": "data:image/jpeg;base64,YWJjZA==",
            "payload": b"abcd",
        }
    )

    assert value["system_prompt"] == "完整中文提示词"
    assert value["authorization"] == "[REDACTED]"
    assert value["audio"]["data"] == {
        "binary_omitted": True,
        "media_type": "audio",
        "base64_characters": 8,
    }
    assert value["image"]["binary_omitted"] is True
    assert value["payload"]["bytes"] == 4


def test_cls_safe_handles_immutable_prompt_snapshot_mappings() -> None:
    value = cls_safe(
        _Snapshot("提示词", MappingProxyType({"system_prompt": "sha256"}))
    )

    assert value == {
        "prompt": "提示词",
        "hashes": {"system_prompt": "sha256"},
    }


def test_cls_safe_redacts_secrets_embedded_in_text() -> None:
    cloud_id = "AKID" + "A" * 20
    github_token = "ghp_" + "B" * 24
    value = cls_safe(
        {
            "system_prompt": (
                "Authorization: Bearer abc.def and phone 13812345678 "
                f"cloud {cloud_id} github {github_token} "
                "url https://example.test/?token=visible"
            ),
        }
    )

    assert value["system_prompt"] == (
        "Authorization: Bearer *** and phone 138****5678 "
        "cloud [REDACTED] github [REDACTED] "
        "url https://example.test/?token=***"
    )


def test_emit_model_trace_adds_shared_correlation_fields(monkeypatch) -> None:
    captured = {}
    captured_service = {}

    def emit(log_type, event, **fields):
        captured.update(log_type=log_type, event=event, fields=fields)
        return True

    monkeypatch.setattr(
        "sglang_omni.utils.model_trace_logs.emit_structured_log", emit
    )
    monkeypatch.setattr(
        "sglang_omni.utils.model_trace_logs._emit_cls_service_log",
        lambda *args, **kwargs: captured_service.update(
            event=args[1], fields=kwargs
        )
        or True,
    )

    assert emit_model_trace(
        "reply",
        "model.response.received",
        component="turn_router",
        request_id="request-1",
        sequence=4,
        model_output="0",
    )
    assert captured["event"] == "model.response.received"
    assert captured["fields"]["invocation_id"] == "request-1"
    assert captured["fields"]["sequence"] == 4
    assert captured["fields"]["model_output"] == "0"
    assert captured_service["event"] == "model.response.received"
    assert captured_service["fields"]["invocation_id"] == "request-1"
    assert captured_service["fields"]["sequence"] == 4


def test_emit_model_trace_is_fail_open(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("sink unavailable")

    monkeypatch.setattr(
        "sglang_omni.utils.model_trace_logs.emit_structured_log", fail
    )
    monkeypatch.setattr(
        "sglang_omni.utils.model_trace_logs._emit_cls_service_log",
        lambda *_args, **_kwargs: False,
    )

    assert not emit_model_trace(
        "reply",
        "model.response.received",
        component="turn_router",
        request_id="request-1",
        sequence=4,
        model_output="0",
    )


def test_emit_model_trace_mirrors_full_text_into_cls_app_log(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sglang_omni.utils.model_trace_logs.emit_structured_log",
        lambda *_args, **_kwargs: True,
    )
    configure_service_logging(
        "omni-reply-worker",
        log_root=tmp_path,
        version="test",
        host="test",
        include_stdout=False,
    )
    try:
        assert emit_model_trace(
            "reply",
            "model.input.prepared",
            component="turn_router",
            request_id="request-1",
            sequence=2,
            root_session_id="session-1",
            session_id="session-1",
            turn_id="turn-1",
            model_input={"system_prompt": "完整中文提示词"},
        )
    finally:
        shutdown_service_logging("omni-reply-worker")

    record = json.loads(
        (tmp_path / "omni-reply-worker" / "app.log").read_text()
    )
    assert record["event"] == "model.input.prepared"
    assert record["sequence"] == 2
    assert record["root_session_id"] == "session-1"
    assert record["model_session_id"] == "session-1"
    assert record["model_input"]["system_prompt"] == "完整中文提示词"
