"""Upload completed Omni model I/O bundles to COS from a sidecar process."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sglang_omni.utils.framex_service_logs import (
    configure_service_logging,
    emit_service_event,
    shutdown_service_logging,
)

logger = logging.getLogger(__name__)


def _atomic_json(path: Path, value: Any) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _service(stage: str) -> str:
    return {
        "event": "omni-event-worker",
        "router": "omni-reply-worker",
        "reply": "omni-reply-worker",
        "action": "omni-action-worker",
    }[stage]


def _remote_bundle(
    prefix: str,
    stage: str,
    relative_call: Path,
    *,
    archive_day: str | None = None,
) -> tuple[str, str]:
    """Return the console-friendly COS directory and invocation file stem.

    The local spool keeps every correlation dimension as its own directory for
    recovery and pruning. COS is also an operator-facing index, so make the
    root session the first lookup dimension and flatten invocations to files:

      {prefix}/{session}/{date}/
        {stage}__{turn}__{request}__{invocation}.{kind}
    """

    parts = relative_call.parts
    expected = ("dt=", "root=", "turn=", "request=", "invocation=")
    if len(parts) != len(expected):
        raise ValueError("unsupported local model I/O layout")
    values: list[str] = []
    for part, marker in zip(parts, expected, strict=True):
        if not part.startswith(marker) or not part[len(marker) :]:
            raise ValueError("unsupported local model I/O component")
        values.append(part[len(marker) :])
    local_day, root_session_id, turn_id, request_id, invocation_id = values
    day = archive_day or local_day
    bundle = f"{stage}__{turn_id}__{request_id}__{invocation_id}"
    directory = f"{prefix.strip('/')}/{root_session_id}/{day}"
    return directory, bundle


def _manifest_day(
    manifest: dict[str, Any], *, timezone: ZoneInfo
) -> str | None:
    started_at = manifest.get("started_at")
    if not isinstance(started_at, str):
        return None
    value = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("model I/O started_at must include a timezone")
    return value.astimezone(timezone).date().isoformat()


def _remote_file_name(bundle: str, relative_path: str) -> str:
    suffix = {
        "input/request.json": "input.json",
        "output/result.json": "output.json",
    }.get(relative_path, relative_path.replace("/", "."))
    return f"{bundle}.{suffix}"


def _emit(stage: str, event: str, manifest: dict[str, Any], **fields: Any) -> None:
    try:
        emit_service_event(
            _service(stage),
            event,
            logger_name=__name__,
            request_id=manifest.get("request_id"),
            root_session_id=manifest.get("root_session_id"),
            turn_id=manifest.get("turn_id"),
            artifact_id=manifest.get("artifact_id"),
            stage=stage,
            **fields,
        )
    except Exception:
        # CLS availability or malformed legacy correlation values must not
        # alter an otherwise successful COS commit/retry decision.
        logger.exception("COS model I/O status log emission failed")


def upload_pending(
    root: Path,
    *,
    client: Any,
    bucket: str,
    prefix: str = "omni",
    date_timezone: str = "Asia/Shanghai",
    now: float | None = None,
    local_retention_seconds: float = 86_400,
) -> dict[str, int]:
    root = root.resolve(strict=True)
    observed_at = time.time() if now is None else now
    archive_timezone = ZoneInfo(date_timezone)
    counters = {"uploaded": 0, "failed": 0, "skipped": 0, "pruned": 0}
    with (root / ".cos-upload.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return counters
        for ready in sorted(
            root.glob("*/dt=*/root=*/turn=*/request=*/invocation=*/ready.json")
        ):
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
            manifest: dict[str, Any] = {}
            try:
                ready_value = json.loads(ready.read_text())
                if ready_value.get("schema_version") != "omni-model-io-ready.v1":
                    raise ValueError("unsupported ready marker")
                manifest_path = call_dir / ready_value["manifest_path"]
                if (
                    manifest_path.is_symlink()
                    or not manifest_path.resolve().is_relative_to(call_dir.resolve())
                ):
                    raise ValueError("unsafe manifest path")
                manifest_bytes = manifest_path.read_bytes()
                if (
                    hashlib.sha256(manifest_bytes).hexdigest()
                    != ready_value["manifest_sha256"]
                ):
                    raise ValueError("manifest checksum mismatch")
                manifest = json.loads(manifest_bytes)
                if manifest.get("schema_version") != "omni-model-io.v1":
                    raise ValueError("unsupported model I/O manifest")
                if manifest.get("artifact_id") != ready_value.get("artifact_id"):
                    raise ValueError("artifact identity mismatch")
                expected_count = len(manifest["files"]) + 1
                expected_bytes = len(manifest_bytes) + sum(
                    int(record["bytes"]) for record in manifest["files"]
                )
                if ready_value.get("object_count") != expected_count:
                    raise ValueError("artifact object count mismatch")
                if ready_value.get("total_bytes") != expected_bytes:
                    raise ValueError("artifact total byte count mismatch")
                stage = manifest["stage"]
                relative_call = call_dir.relative_to(root / stage)
                object_root, object_stem = _remote_bundle(
                    prefix,
                    stage,
                    relative_call,
                    archive_day=_manifest_day(
                        manifest, timezone=archive_timezone
                    ),
                )
                records = list(manifest["files"])
                for record in records:
                    path = call_dir / record["path"]
                    resolved = path.resolve()
                    if path.is_symlink() or not resolved.is_relative_to(
                        call_dir.resolve()
                    ):
                        raise ValueError("unsafe artifact path")
                    if path.stat().st_size != record["bytes"]:
                        raise ValueError("artifact size mismatch")
                    if _sha256_file(path) != record["sha256"]:
                        raise ValueError("artifact checksum mismatch")
                    client.upload_file(
                        Bucket=bucket,
                        Key=(
                            f"{object_root}/"
                            f"{_remote_file_name(object_stem, record['path'])}"
                        ),
                        LocalFilePath=str(path),
                        EnableMD5=True,
                        MAXThread=1,
                        ServerSideEncryption="AES256",
                    )
                # The manifest is the remote commit marker and must be last.
                manifest_key = f"{object_root}/{object_stem}.manifest.json"
                client.upload_file(
                    Bucket=bucket,
                    Key=manifest_key,
                    LocalFilePath=str(manifest_path),
                    EnableMD5=True,
                    MAXThread=1,
                    ServerSideEncryption="AES256",
                )
                receipt = {
                    "bucket": bucket,
                    "manifest_key": manifest_key,
                    "manifest_sha256": ready_value["manifest_sha256"],
                    "object_count": ready_value["object_count"],
                    "total_bytes": ready_value["total_bytes"],
                    "uploaded_at": observed_at,
                }
                _atomic_json(marker, receipt)
                retry.unlink(missing_ok=True)
                _emit(
                    stage,
                    "artifact.upload.completed",
                    manifest,
                    cos_manifest_key=manifest_key,
                    manifest_sha256=ready_value["manifest_sha256"],
                    object_count=ready_value["object_count"],
                    total_bytes=ready_value["total_bytes"],
                )
                counters["uploaded"] += 1
            except Exception as exc:
                attempts = state.get("attempts", 0) + 1
                _atomic_json(
                    retry,
                    {
                        "attempts": attempts,
                        "error_type": type(exc).__name__,
                        "next_attempt_at": observed_at
                        + min(3_600, 5 * 2 ** min(attempts, 10)),
                    },
                )
                stage = manifest.get("stage")
                if stage in {"event", "router", "reply", "action"}:
                    _emit(
                        stage,
                        "artifact.upload.failed",
                        manifest,
                        error_type=type(exc).__name__,
                        attempts=attempts,
                    )
                logger.warning(
                    "COS model I/O upload failed call=%s error_type=%s",
                    call_dir.name,
                    type(exc).__name__,
                )
                counters["failed"] += 1
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--bucket", default=os.getenv("SGLANG_OMNI_MODEL_IO_COS_BUCKET", "")
    )
    parser.add_argument(
        "--region", default=os.getenv("SGLANG_OMNI_MODEL_IO_COS_REGION", "")
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("SGLANG_OMNI_MODEL_IO_COS_PREFIX", "omni"),
    )
    parser.add_argument(
        "--date-timezone",
        default=os.getenv(
            "SGLANG_OMNI_MODEL_IO_DATE_TIMEZONE", "Asia/Shanghai"
        ),
    )
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--local-retention-seconds", type=float, default=86_400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        not args.bucket
        or not args.region
        or args.interval <= 0
        or args.local_retention_seconds < 0
    ):
        parser.error(
            "bucket, region, positive interval and non-negative retention are required"
        )
    from qcloud_cos import CosConfig, CosS3Client

    credential = (
        json.loads(args.credentials_file.read_text()) if args.credentials_file else {}
    )
    secret_id = credential.get("secret_id") or os.environ.get("COS_SECRET_ID")
    secret_key = credential.get("secret_key") or os.environ.get("COS_SECRET_KEY")
    if not secret_id or not secret_key:
        parser.error("COS credentials are required")
    client = CosS3Client(
        CosConfig(
            Region=args.region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Token=credential.get("token") or os.environ.get("COS_TOKEN"),
            Timeout=30,
        )
    )
    logging.basicConfig(level=logging.INFO)
    services = (
        "omni-event-worker",
        "omni-reply-worker",
        "omni-action-worker",
    )
    for service in services:
        configure_service_logging(service, include_stdout=False)
    try:
        while True:
            counters = upload_pending(
                args.root,
                client=client,
                bucket=args.bucket,
                prefix=args.prefix,
                date_timezone=args.date_timezone,
                local_retention_seconds=args.local_retention_seconds,
            )
            logger.info("COS model I/O upload progress %s", json.dumps(counters))
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        for service in services:
            shutdown_service_logging(service)


if __name__ == "__main__":
    main()
