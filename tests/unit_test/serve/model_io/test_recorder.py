import base64
import json
import threading
import time
from pathlib import Path

from sglang_omni.serve.model_io.recorder import ModelIORecorder


def _wait_for_ready(root: Path) -> Path:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        matches = list(
            root.glob("*/dt=*/root=*/turn=*/request=*/invocation=*/ready.json")
        )
        if matches:
            return matches[0]
        time.sleep(0.01)
    raise AssertionError("archive writer did not produce ready.json")


def test_recorder_extracts_inline_media_and_commits_manifest_last(
    tmp_path: Path,
) -> None:
    recorder = ModelIORecorder(tmp_path, queue_size=2, max_bytes=1_000_000)
    audio = b"RIFF-test-audio"
    assert recorder.record(
        stage="router",
        input_value={
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {
                "audios": ["data:audio/wav;base64," + base64.b64encode(audio).decode()]
            },
        },
        output_value={
            "text": "0",
            "audio": {"data": base64.b64encode(audio).decode()},
        },
        context={
            "request_id": "req_1",
            "root_session_id": "session_1",
            "turn_id": "turn_1",
            "model_id": "router-model",
            "model_version": "v1",
        },
        started_at=1_800_000_000,
        completed_at=1_800_000_001,
        invocation_id="inv_1",
    )
    ready_path = _wait_for_ready(tmp_path)
    call_dir = ready_path.parent
    ready = json.loads(ready_path.read_text())
    manifest = json.loads((call_dir / "manifest.json").read_text())
    request = json.loads((call_dir / "input/request.json").read_text())
    result = json.loads((call_dir / "output/result.json").read_text())
    media_ref = request["metadata"]["audios"][0]

    assert manifest["schema_version"] == "omni-model-io.v1"
    assert manifest["root_session_id"] == "session_1"
    assert manifest["turn_id"] == "turn_1"
    assert manifest["duration_ms"] == 1000.0
    assert ready["manifest_path"] == "manifest.json"
    assert (call_dir / media_ref["$media"]).read_bytes() == audio
    assert (call_dir / result["audio"]["data"]["$media"]).read_bytes() == audio
    assert "base64" not in (call_dir / "input/request.json").read_text()
    recorder.close()


def test_disabled_recorder_is_fail_open() -> None:
    recorder = ModelIORecorder(None)
    assert not recorder.record(
        stage="reply",
        input_value={},
        output_value={},
        context={},
    )


def test_close_drains_a_full_queue_and_joins_writer(tmp_path, monkeypatch) -> None:
    recorder = ModelIORecorder(tmp_path, queue_size=1)
    write_started = threading.Event()
    release_write = threading.Event()
    original_write = recorder._write

    def blocked_write(item):
        write_started.set()
        assert release_write.wait(2)
        original_write(item)

    monkeypatch.setattr(recorder, "_write", blocked_write)
    assert recorder.record(
        stage="reply", input_value={"index": 1}, output_value={},
        context={"session_id": "session", "turn_id": "turn"},
    )
    assert write_started.wait(2)
    assert recorder.record(
        stage="reply", input_value={"index": 2}, output_value={},
        context={"session_id": "session", "turn_id": "turn"},
    )

    closer = threading.Thread(target=lambda: recorder.close(timeout=2))
    closer.start()
    release_write.set()
    closer.join(2)

    assert not closer.is_alive()
    assert recorder._thread is not None and not recorder._thread.is_alive()
    assert recorder._queue.empty()
    assert len(list(tmp_path.rglob("ready.json"))) == 2


def test_close_reports_timeout_without_losing_stop_request(tmp_path, monkeypatch) -> None:
    recorder = ModelIORecorder(tmp_path, queue_size=1)
    write_started = threading.Event()
    release_write = threading.Event()
    original_write = recorder._write

    def blocked_write(item):
        write_started.set()
        assert release_write.wait(2)
        original_write(item)

    monkeypatch.setattr(recorder, "_write", blocked_write)
    assert recorder.record(
        stage="reply", input_value={}, output_value={},
        context={"session_id": "session", "turn_id": "turn"},
    )
    assert write_started.wait(2)

    assert recorder.close(timeout=0.01) is False
    release_write.set()
    assert recorder._thread is not None
    recorder._thread.join(2)
    assert not recorder._thread.is_alive()
