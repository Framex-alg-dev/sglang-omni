"""Persist exact model calls locally without blocking inference workers.

The inference path only performs a bounded ``put_nowait``.  JSON encoding,
inline-media extraction, hashing and filesystem I/O all run on a daemon writer
thread.  COS is deliberately handled by a separate uploader process.
"""

from __future__ import annotations

import atexit
import base64
import binascii
import dataclasses
import hashlib
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sglang_omni.utils.framex_service_logs import emit_service_event

logger = logging.getLogger(__name__)

_DATA_URI = re.compile(r"^data:([^;,]+)?(?:;[^,]*)?;base64,(.*)$", re.DOTALL)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.:-]+")
_SERVICE_BY_STAGE = {
    "event": "omni-event-worker",
    "router": "omni-reply-worker",
    "reply": "omni-reply-worker",
    "action": "omni-action-worker",
}


def _component(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    normalized = _SAFE_COMPONENT.sub("_", raw).strip("._")
    if not normalized:
        normalized = fallback
    if len(normalized) > 160:
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        normalized = f"{normalized[:140]}-{digest}"
    return normalized


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return (
            "data:application/octet-stream;base64," + base64.b64encode(value).decode()
        )
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    method = getattr(value, "model_dump", None)
    if callable(method):
        return _jsonable(method())
    method = getattr(value, "to_dict", None)
    if callable(method):
        return _jsonable(method())
    return repr(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _atomic_bytes(path, payload)
    return payload


def _media_suffix(content_type: str) -> str:
    return {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "video/mp4": ".mp4",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type.lower(), ".bin")


def _extract_media(
    value: Any,
    *,
    directory: Path,
    stem: str,
    files: list[dict[str, Any]],
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, str):
        match = _DATA_URI.match(value)
        content_type = match.group(1) or "application/octet-stream" if match else None
        encoded = match.group(2) if match else None
        # Completion audio uses a plain base64 ``audio.data`` value rather than
        # a data URI. Keep the matcher path-specific so ordinary text named
        # ``data`` is never treated as media.
        if match is None and len(path) >= 2 and path[-2:] == ("audio", "data"):
            content_type = "audio/wav"
            encoded = value
        if encoded is None or content_type is None:
            return value
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return value
        digest = hashlib.sha256(payload).hexdigest()
        relative = (
            Path("media")
            / f"{stem}-{len(files):04d}-{digest[:12]}{_media_suffix(content_type)}"
        )
        _atomic_bytes(directory / relative, payload)
        files.append(
            {
                "path": str(relative),
                "bytes": len(payload),
                "sha256": digest,
                "content_type": content_type,
            }
        )
        return {
            "$media": str(relative),
            "content_type": content_type,
            "bytes": len(payload),
            "sha256": digest,
        }
    if isinstance(value, dict):
        return {
            key: _extract_media(
                item,
                directory=directory,
                stem=stem,
                files=files,
                path=(*path, key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _extract_media(
                item,
                directory=directory,
                stem=stem,
                files=files,
                path=(*path, str(index)),
            )
            for index, item in enumerate(value)
        ]
    return value


@dataclass(frozen=True)
class _Record:
    stage: str
    input_value: Any
    output_value: Any
    context: dict[str, Any]
    status: str
    started_at: float
    completed_at: float
    invocation_id: str


class ModelIORecorder:
    """A bounded, fail-open local recorder for one Omni model stage."""

    def __init__(
        self, root: Path | None, *, queue_size: int = 128, max_bytes: int = 8 * 1024**3
    ) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self._queue: queue.Queue[_Record | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._closed = False
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._run, name="omni-model-io-writer", daemon=True
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.root is not None and not self._closed

    def record(
        self,
        *,
        stage: str,
        input_value: Any,
        output_value: Any,
        context: dict[str, Any],
        status: str = "success",
        started_at: float | None = None,
        completed_at: float | None = None,
        invocation_id: str | None = None,
    ) -> bool:
        """Enqueue a record and return immediately; never raise into inference."""

        if not self.enabled:
            return False
        completed = time.time() if completed_at is None else completed_at
        item = _Record(
            stage=stage,
            input_value=input_value,
            output_value=output_value,
            context=dict(context),
            status=status,
            started_at=completed if started_at is None else started_at,
            completed_at=completed,
            invocation_id=invocation_id or uuid.uuid4().hex,
        )
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._emit(item, "artifact.archive.dropped", reason="queue_full")
            return False
        except Exception:
            logger.exception("model I/O enqueue failed")
            return False

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write(item)
            except Exception as exc:
                if item is not None:
                    self._emit(
                        item, "artifact.archive.failed", reason=type(exc).__name__
                    )
                logger.exception("model I/O local archive failed")
            finally:
                self._queue.task_done()

    def _write(self, item: _Record) -> None:
        assert self.root is not None
        if self.max_bytes > 0 and _tree_bytes(self.root) >= self.max_bytes:
            self._emit(item, "artifact.archive.dropped", reason="local_byte_cap")
            return
        context = item.context
        root_session_id = _component(
            context.get("root_session_id") or context.get("session_id"), "unknown"
        )
        turn_id = _component(context.get("turn_id"), "system")
        request_id = _component(context.get("request_id"), item.invocation_id)
        day = (
            datetime.fromtimestamp(item.started_at, tz=timezone.utc).date().isoformat()
        )
        call_dir = (
            self.root
            / item.stage
            / f"dt={day}"
            / f"root={root_session_id}"
            / f"turn={turn_id}"
            / f"request={request_id}"
            / f"invocation={_component(item.invocation_id, uuid.uuid4().hex)}"
        )
        call_dir.mkdir(parents=True, exist_ok=False)
        files: list[dict[str, Any]] = []
        input_value = _extract_media(
            _jsonable(item.input_value), directory=call_dir, stem="input", files=files
        )
        output_value = _extract_media(
            _jsonable(item.output_value), directory=call_dir, stem="output", files=files
        )
        for relative, value in (
            (Path("input/request.json"), input_value),
            (Path("output/result.json"), output_value),
        ):
            payload = _atomic_json(call_dir / relative, value)
            files.append(
                {
                    "path": str(relative),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "content_type": "application/json",
                }
            )
        manifest = {
            "schema_version": "omni-model-io.v1",
            "artifact_id": item.invocation_id,
            "invocation_id": item.invocation_id,
            "parent_invocation_id": context.get("parent_invocation_id"),
            "service": _SERVICE_BY_STAGE[item.stage],
            "stage": item.stage,
            "root_session_id": context.get("root_session_id")
            or context.get("session_id"),
            "agent_session_id": context.get("agent_session_id"),
            "turn_id": context.get("turn_id"),
            "source_turn_id": context.get("source_turn_id"),
            "request_id": context.get("request_id"),
            "parent_request_id": context.get("parent_request_id"),
            "provider_request_id": context.get("provider_request_id"),
            "operation_id": context.get("operation_id"),
            "trace_id": context.get("trace_id"),
            "observation_id": context.get("observation_id"),
            "task_id": context.get("task_id"),
            "window_id": context.get("window_id"),
            "decision_point_id": context.get("decision_point_id"),
            "attempt": context.get("attempt", 1),
            "model_id": context.get("model_id"),
            "model_version": context.get("model_version"),
            "release_id": context.get("release_id"),
            "prompt_snapshot_sha256": context.get("prompt_snapshot_sha256"),
            "status": item.status,
            "started_at": datetime.fromtimestamp(
                item.started_at, tz=timezone.utc
            ).isoformat(),
            "completed_at": datetime.fromtimestamp(
                item.completed_at, tz=timezone.utc
            ).isoformat(),
            "duration_ms": round((item.completed_at - item.started_at) * 1000, 3),
            "files": files,
        }
        manifest_payload = _atomic_json(call_dir / "manifest.json", manifest)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        total_bytes = sum(record["bytes"] for record in files) + len(manifest_payload)
        _atomic_json(
            call_dir / "ready.json",
            {
                "schema_version": "omni-model-io-ready.v1",
                "artifact_id": item.invocation_id,
                "manifest_path": "manifest.json",
                "manifest_sha256": manifest_sha,
                "object_count": len(files) + 1,
                "total_bytes": total_bytes,
            },
        )
        self._emit(
            item,
            "artifact.archive.persisted",
            artifact_id=item.invocation_id,
            manifest_sha256=manifest_sha,
            object_count=len(files) + 1,
            total_bytes=total_bytes,
        )

    def _emit(self, item: _Record, event: str, **fields: Any) -> None:
        service = _SERVICE_BY_STAGE.get(item.stage, "omni-reply-worker")
        try:
            emit_service_event(
                service,
                event,
                logger_name=__name__,
                request_id=item.context.get("request_id"),
                root_session_id=item.context.get("root_session_id")
                or item.context.get("session_id"),
                turn_id=item.context.get("turn_id"),
                stage=item.stage,
                **fields,
            )
        except Exception:
            logger.exception("model I/O archive event emission failed")


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


_RECORDERS: dict[str, ModelIORecorder] = {}
_RECORDERS_LOCK = threading.Lock()


def get_model_io_recorder(stage: str) -> ModelIORecorder:
    if stage not in _SERVICE_BY_STAGE:
        raise ValueError(f"unsupported model I/O stage: {stage}")
    with _RECORDERS_LOCK:
        recorder = _RECORDERS.get(stage)
        if recorder is not None:
            return recorder
        try:
            raw_root = os.environ.get("SGLANG_OMNI_MODEL_IO_ROOT", "").strip()
            root = Path(raw_root) if raw_root else None
            queue_size = int(os.environ.get("SGLANG_OMNI_MODEL_IO_QUEUE_SIZE", "128"))
            max_bytes = int(
                os.environ.get("SGLANG_OMNI_MODEL_IO_MAX_BYTES", str(8 * 1024**3))
            )
            recorder = ModelIORecorder(
                root, queue_size=max(1, queue_size), max_bytes=max_bytes
            )
        except Exception:
            # A bad mount or deployment setting must never make model serving
            # unavailable. The disabled instance is cached to avoid repeated
            # filesystem/config work on every model call.
            logger.exception(
                "model I/O recorder initialization failed; disabling archive"
            )
            recorder = ModelIORecorder(None)
        _RECORDERS[stage] = recorder
        return recorder


@atexit.register
def _close_recorders() -> None:
    for recorder in tuple(_RECORDERS.values()):
        recorder.close(timeout=1.0)
