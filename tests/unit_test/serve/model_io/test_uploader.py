import hashlib
import json
from pathlib import Path

from sglang_omni.serve.model_io.uploader import upload_pending


class FakeCOSClient:
    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []

    def upload_file(self, **kwargs: object) -> None:
        self.uploads.append(kwargs)


def _write_bundle(root: Path) -> Path:
    call_dir = (
        root
        / "action"
        / "dt=2027-01-15"
        / "root=session_1"
        / "turn=turn_1"
        / "request=req_1"
        / "invocation=inv_1"
    )
    payload_path = call_dir / "input/request.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text('{"prompt":"hello"}\n')
    payload = payload_path.read_bytes()
    manifest = {
        "schema_version": "omni-model-io.v1",
        "artifact_id": "inv_1",
        "stage": "action",
        "request_id": "req_1",
        "root_session_id": "session_1",
        "turn_id": "turn_1",
        "files": [
            {
                "path": "input/request.json",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "content_type": "application/json",
            }
        ],
    }
    manifest_path = call_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    manifest_payload = manifest_path.read_bytes()
    (call_dir / "ready.json").write_text(
        json.dumps(
            {
                "schema_version": "omni-model-io-ready.v1",
                "artifact_id": "inv_1",
                "manifest_path": "manifest.json",
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "object_count": 2,
                "total_bytes": len(payload) + len(manifest_payload),
            }
        )
    )
    return call_dir


def test_uploader_sends_manifest_last_and_is_idempotent(tmp_path: Path) -> None:
    call_dir = _write_bundle(tmp_path)
    client = FakeCOSClient()

    first = upload_pending(
        tmp_path,
        client=client,
        bucket="bucket-1",
        now=100,
        local_retention_seconds=1000,
    )

    assert first == {"uploaded": 1, "failed": 0, "skipped": 0, "pruned": 0}
    keys = [str(item["Key"]) for item in client.uploads]
    expected_root = (
        "omni/model-io/v1/session_1/2027-01-15/action/"
        "turn_1__req_1__inv_1"
    )
    assert keys == [f"{expected_root}.input.json", f"{expected_root}.manifest.json"]
    assert all(item["ServerSideEncryption"] == "AES256" for item in client.uploads)
    assert json.loads((call_dir / ".cos-uploaded.json").read_text())[
        "manifest_key"
    ] == f"{expected_root}.manifest.json"

    second = upload_pending(
        tmp_path,
        client=client,
        bucket="bucket-1",
        now=101,
        local_retention_seconds=1000,
    )
    assert second["skipped"] == 1
    assert len(client.uploads) == 2


def test_uploader_retries_checksum_failure_without_remote_commit(
    tmp_path: Path,
) -> None:
    call_dir = _write_bundle(tmp_path)
    (call_dir / "input/request.json").write_text("corrupted")
    client = FakeCOSClient()

    counters = upload_pending(
        tmp_path,
        client=client,
        bucket="bucket-1",
        now=100,
    )

    assert counters["failed"] == 1
    assert client.uploads == []
    assert not (call_dir / ".cos-uploaded.json").exists()
    retry = json.loads((call_dir / ".cos-retry.json").read_text())
    assert retry["attempts"] == 1
    assert retry["next_attempt_at"] > 100
