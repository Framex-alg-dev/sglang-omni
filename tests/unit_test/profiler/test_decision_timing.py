from __future__ import annotations

from sglang_omni.profiler import decision_timing


def test_decision_correlation_fields_are_bounded_and_content_free() -> None:
    metadata = {
        "task": "turn_router",
        "trace_id": "trace-1",
        "operation_id": "classification-1",
        "classification_id": "classification-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "model": "decision-model",
        "model_version": "router-v1",
        "identity_epoch": 2,
        "input_revision": 4,
        "audios": ["data:audio/wav;base64,private-audio"],
        "prompt": "private prompt",
    }

    fields = decision_timing.decision_correlation_fields(
        request_id="classification-1",
        metadata=metadata,
    )

    assert fields == {
        "trace_id": "trace-1",
        "request_id": "classification-1",
        "operation_id": "classification-1",
        "task": "turn_router",
        "classification_id": "classification-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "model_name": "decision-model",
        "model_version": "router-v1",
        "identity_epoch": 2,
        "input_revision": 4,
        "audio_count": 1,
        "image_count": 0,
        "video_count": 0,
        "media_count": 1,
    }
    assert "prompt" not in fields
    assert "audios" not in fields


def test_emit_decision_timing_ignores_non_decision_requests(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        decision_timing,
        "emit_structured_log",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )

    emitted = decision_timing.emit_decision_timing(
        "decision_scheduler_enqueued",
        request_id="request-1",
        metadata={"task": "speech"},
        component="test",
    )

    assert emitted is False
    assert calls == []


def test_emit_decision_timing_forwards_only_safe_correlation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        decision_timing,
        "emit_structured_log",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )

    emitted = decision_timing.emit_decision_timing(
        "decision_request_received",
        request_id="request-1",
        metadata={
            "task": "turn_router",
            "session_id": "session-1",
            "logical_request_id": "turn-1",
            "audios": ["private-audio"],
        },
        component="turn_router",
        server_total_ms=12.5,
    )

    assert emitted is True
    args, fields = calls[0]
    assert args == ("performance", "decision_request_received")
    assert fields["component"] == "turn_router"
    assert fields["trace_id"] == "request-1"
    assert fields["turn_id"] == "turn-1"
    assert fields["audio_count"] == 1
    assert fields["server_total_ms"] == 12.5
    assert "audios" not in fields


def test_emit_decision_timing_never_breaks_inference_on_sink_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(decision_timing, "_EMIT_FAILURE_REPORTED", False)
    monkeypatch.setattr(
        decision_timing,
        "emit_structured_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert (
        decision_timing.emit_decision_timing(
            "decision_request_received",
            request_id="request-1",
            metadata={"task": "turn_router"},
            component="turn_router",
        )
        is False
    )
