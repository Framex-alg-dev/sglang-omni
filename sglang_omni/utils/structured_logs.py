# SPDX-License-Identifier: Apache-2.0
"""Partitioned, non-blocking JSONL logs for realtime sessions.

Records are timestamped on the caller thread and written by one daemon thread
per process.  The sink partitions by local hour and log type, then rolls files
by size.  Request handling never waits for disk I/O.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


logger = logging.getLogger(__name__)

_STOP = object()
_WRITER: "PartitionedJSONLWriter | None" = None
_WRITER_LOCK = threading.Lock()
_LOG_TYPES = {
    "lifecycle",
    "protocol",
    "reply",
    "action",
    "performance",
    "resource",
    "error",
    "diagnostic",
}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive; using %d", name, default)
        return default
    return value


def _timezone() -> ZoneInfo:
    name = os.environ.get("SGLANG_OMNI_REALTIME_LOG_TIMEZONE", "Asia/Shanghai")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning("unknown realtime log timezone %r; using UTC", name)
        return ZoneInfo("UTC")


class PartitionedJSONLWriter:
    """Route bounded asynchronous records to hourly, typed JSONL files."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_queue_size: int = 8192,
        max_file_bytes: int = 128 * 1024 * 1024,
        timezone: ZoneInfo | None = None,
    ) -> None:
        if max_queue_size <= 0 or max_file_bytes <= 0:
            raise ValueError("queue and file limits must be positive")
        self.root = Path(root)
        self.max_file_bytes = max_file_bytes
        self.timezone = timezone or _timezone()
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.instance_id = os.environ.get(
            "SGLANG_OMNI_SERVICE_INSTANCE_ID", f"{self.hostname}-{self.pid}"
        )
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max_queue_size
        )
        self._closed = False
        self._lock = threading.Lock()
        self._dropped = 0
        self._written = 0
        self._write_errors = 0
        self._segments: dict[tuple[str, str, str], tuple[int, int]] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-structured-log-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped

    @property
    def written_records(self) -> int:
        return self._written

    def write(self, record: dict[str, Any]) -> bool:
        with self._lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 100 == 0:
                    logger.warning(
                        "realtime structured log queue full dropped_records=%d",
                        self._dropped,
                    )
                return False
        return True

    def emit(
        self,
        log_type: str,
        event: str,
        *,
        level: str = "info",
        **fields: Any,
    ) -> bool:
        if log_type not in _LOG_TYPES:
            raise ValueError(f"unsupported structured log type: {log_type!r}")
        now_ns = time.time_ns()
        timestamp = datetime.fromtimestamp(
            now_ns / 1_000_000_000, tz=self.timezone
        )
        record = {
            "schema_version": "1.0",
            "timestamp": timestamp.isoformat(timespec="microseconds"),
            "timestamp_unix_ms": now_ns // 1_000_000,
            "log_hour": timestamp.strftime("%Y-%m-%d/%H"),
            "log_type": log_type,
            "event": event,
            "level": level,
            "service_instance_id": self.instance_id,
            "component": str(fields.pop("component", "api")),
            "hostname": self.hostname,
            "pid": self.pid,
            **fields,
        }
        return self.write(record)

    def health(self) -> dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "written_records": self._written,
            "dropped_records": self._dropped,
            "write_errors": self._write_errors,
        }

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._append(item)
            except Exception:
                self._write_errors += 1
                logger.exception("failed to write realtime structured log")
            finally:
                self._queue.task_done()

    def _append(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        log_hour = str(record["log_hour"])
        log_type = str(record["log_type"])
        component = str(record.get("component") or "api").replace("/", "_")
        key = (log_hour, log_type, component)
        segment, current_size = self._segments.get(key, (0, 0))
        if current_size and current_size + len(encoded) > self.max_file_bytes:
            segment += 1
            current_size = 0
        directory = self.root / log_hour
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{log_type}_{component}_{self.pid}_{segment:03d}.jsonl"
        with path.open("ab", buffering=0) as handle:
            handle.write(encoded)
        self._segments[key] = (segment, current_size + len(encoded))
        self._written += 1
        # Keep routing metadata bounded to the current and immediately previous
        # hour.  Files are opened per append, so dropping metadata is safe.
        active_hours = sorted({item[0] for item in self._segments})
        if len(active_hours) > 2:
            expired = set(active_hours[:-2])
            for old_key in [item for item in self._segments if item[0] in expired]:
                self._segments.pop(old_key, None)


def _reset_after_fork() -> None:
    global _WRITER, _WRITER_LOCK
    _WRITER = None
    _WRITER_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def get_structured_log_writer() -> PartitionedJSONLWriter:
    global _WRITER
    with _WRITER_LOCK:
        if _WRITER is None:
            root = os.environ.get(
                "SGLANG_OMNI_REALTIME_LOG_DIR",
                "/tmp/sglang-omni-realtime-logs",
            )
            _WRITER = PartitionedJSONLWriter(
                root,
                max_queue_size=_positive_int_env(
                    "SGLANG_OMNI_REALTIME_LOG_QUEUE_SIZE", 8192
                ),
                max_file_bytes=(
                    _positive_int_env("SGLANG_OMNI_REALTIME_LOG_MAX_FILE_MB", 128)
                    * 1024
                    * 1024
                ),
            )
        return _WRITER


def emit_structured_log(
    log_type: str,
    event: str,
    *,
    level: str = "info",
    **fields: Any,
) -> bool:
    return get_structured_log_writer().emit(
        log_type, event, level=level, **fields
    )


def new_trace_id(session_id: str | None, turn_id: str | None) -> str:
    prefix = ":".join(item for item in (session_id, turn_id) if item)
    return f"{prefix}:{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex


def shutdown_structured_log_writer() -> None:
    global _WRITER
    with _WRITER_LOCK:
        writer = _WRITER
        _WRITER = None
    if writer is not None:
        writer.close()


atexit.register(shutdown_structured_log_writer)


__all__ = [
    "PartitionedJSONLWriter",
    "emit_structured_log",
    "get_structured_log_writer",
    "new_trace_id",
    "shutdown_structured_log_writer",
]
