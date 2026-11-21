"""Upload completed event inference bundles without occupying inference workers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _atomic_json(path: Path, value: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def upload_pending(
    root: Path, *, client: Any, bucket: str, prefix: str,
    now: float | None = None, local_retention_seconds: float = 86_400,
) -> dict[str, int]:
    root = root.resolve(strict=True)
    observed_at = time.time() if now is None else now
    counters = {"uploaded": 0, "failed": 0, "skipped": 0, "pruned": 0}
    with (root / ".cos-upload.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return counters
        for ready in sorted(root.glob("*/*/ready.json")):
            call_dir = ready.parent
            marker = call_dir / ".cos-uploaded.json"
            retry = call_dir / ".cos-retry.json"
            if marker.exists():
                receipt = json.loads(marker.read_text())
                if observed_at - receipt["uploaded_at"] >= local_retention_seconds:
                    shutil.rmtree(call_dir)
                    counters["pruned"] += 1
                else:
                    counters["skipped"] += 1
                continue
            state = json.loads(retry.read_text()) if retry.exists() else {}
            if observed_at < state.get("next_attempt_at", 0):
                counters["skipped"] += 1
                continue
            try:
                manifest_bytes = ready.read_bytes()
                manifest = json.loads(manifest_bytes)
                if manifest.get("schema_version") != "timeline-artifact-bundle.v1":
                    raise ValueError("unsupported event artifact manifest")
                files = [ready]
                for record in manifest["files"]:
                    path = call_dir / record["path"]
                    if path.is_symlink() or not path.resolve().is_relative_to(call_dir.resolve()):
                        raise ValueError("unsafe event artifact path")
                    if path.stat().st_size != record["bytes"]:
                        raise ValueError("event artifact size mismatch")
                    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                        raise ValueError("event artifact checksum mismatch")
                    files.append(path)
                session_file = call_dir.parent / "session.json"
                session = json.loads(session_file.read_text())
                manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
                date = session["created_at"][:10].replace("-", "/")
                object_key = (
                    f"{prefix.strip('/')}/{date}/{call_dir.parent.name}/"
                    f"{call_dir.name}-{manifest_sha}.tar.gz"
                ).lstrip("/")
                with tempfile.TemporaryDirectory(prefix=".cos-package-", dir=root) as directory:
                    archive = Path(directory) / "bundle.tar.gz"
                    with tarfile.open(archive, "w:gz") as bundle:
                        bundle.add(session_file, arcname="session.json", recursive=False)
                        for path in files:
                            bundle.add(path, arcname=str(path.relative_to(call_dir)), recursive=False)
                    client.upload_file(
                        Bucket=bucket, Key=object_key, LocalFilePath=str(archive),
                        EnableMD5=True, MAXThread=1,
                    )
                _atomic_json(
                    marker,
                    {
                        "bucket": bucket, "key": object_key,
                        "manifest_sha256": manifest_sha,
                        "uploaded_at": observed_at,
                    },
                )
                retry.unlink(missing_ok=True)
                counters["uploaded"] += 1
            except Exception as exc:
                attempts = state.get("attempts", 0) + 1
                _atomic_json(
                    retry,
                    {
                        "attempts": attempts, "error_type": type(exc).__name__,
                        "next_attempt_at": observed_at + min(3_600, 5 * 2 ** min(attempts, 10)),
                    },
                )
                logger.warning(
                    "event COS upload failed call=%s error_type=%s",
                    call_dir.name, type(exc).__name__,
                )
                counters["failed"] += 1
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bucket", default=os.getenv("SGLANG_OMNI_TIMELINE_COS_BUCKET", ""))
    parser.add_argument("--region", default=os.getenv("SGLANG_OMNI_TIMELINE_COS_REGION", ""))
    parser.add_argument("--prefix", default="event-inference/v1")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--local-retention-seconds", type=float, default=86_400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.bucket or not args.region or args.interval <= 0 or args.local_retention_seconds < 0:
        parser.error("bucket, region, positive interval and non-negative retention are required")
    from qcloud_cos import CosConfig, CosS3Client

    credential = json.loads(args.credentials_file.read_text()) if args.credentials_file else {}
    secret_id = credential.get("secret_id") or os.environ.get("COS_SECRET_ID")
    secret_key = credential.get("secret_key") or os.environ.get("COS_SECRET_KEY")
    if not secret_id or not secret_key:
        parser.error("COS credentials are required")
    client = CosS3Client(
        CosConfig(
            Region=args.region, SecretId=secret_id, SecretKey=secret_key,
            Token=credential.get("token") or os.environ.get("COS_TOKEN"),
            Timeout=30,
        )
    )
    logging.basicConfig(level=logging.INFO)
    while True:
        counters = upload_pending(
            args.root, client=client, bucket=args.bucket, prefix=args.prefix,
            local_retention_seconds=args.local_retention_seconds,
        )
        logger.info("event COS upload progress %s", json.dumps(counters))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
