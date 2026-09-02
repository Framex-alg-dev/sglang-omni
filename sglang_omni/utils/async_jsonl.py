# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous JSONL diagnostics.

The request path only enqueues Python objects. JSON serialization, directory
creation, file locking, and disk writes run on one daemon thread per process
and output path.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_STOP = object()
_WRITERS: dict[tuple[int, str], "AsyncJSONLWriter"] = {}
_WRITERS_LOCK = threading.Lock()


def _reset_after_fork() -> None:
    """Drop inherited thread objects; child processes create their own writers."""
    global _WRITERS_LOCK
    _WRITERS.clear()
    _WRITERS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


class AsyncJSONLWriter:
    """Write JSONL records off the caller thread with bounded memory use."""

    def __init__(self, path: str | Path, *, max_queue_size: int = 2048) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.path = Path(path)
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max_queue_size
        )
        self._closed = False
        self._dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"jsonl-writer-{self.path.name}",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped

    def write(self, record: dict[str, Any]) -> bool:
        """Queue a record without blocking; return false if it was dropped."""
        with self._lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 100 == 0:
                    logger.warning(
                        "async JSONL queue full path=%s dropped_records=%d",
                        self.path,
                        self._dropped,
                    )
                return False
        return True

    def flush(self) -> None:
        """Wait until all records queued before this call have been handled."""
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # A full queue is expected only during overload. Waiting here is safe:
        # close is a shutdown path, never a request path.
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
                logger.exception("failed to write async JSONL record path=%s", self.path)
            finally:
                self._queue.task_done()

    def _append(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Different pipeline processes may share the same action debug file.
        # Serialize complete records so large JSON lines cannot interleave.
        with self.path.open("ab", buffering=0) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(encoded)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def get_async_jsonl_writer(
    path: str | Path, *, max_queue_size: int = 2048
) -> AsyncJSONLWriter:
    resolved = str(Path(path))
    key = (os.getpid(), resolved)
    with _WRITERS_LOCK:
        writer = _WRITERS.get(key)
        if writer is None:
            writer = AsyncJSONLWriter(resolved, max_queue_size=max_queue_size)
            _WRITERS[key] = writer
        return writer


def enqueue_jsonl(path: str | Path, record: dict[str, Any]) -> bool:
    return get_async_jsonl_writer(path).write(record)


def shutdown_async_jsonl_writers() -> None:
    with _WRITERS_LOCK:
        writers = list(_WRITERS.values())
        _WRITERS.clear()
    for writer in writers:
        writer.close()


atexit.register(shutdown_async_jsonl_writers)


__all__ = [
    "AsyncJSONLWriter",
    "enqueue_jsonl",
    "get_async_jsonl_writer",
    "shutdown_async_jsonl_writers",
]
