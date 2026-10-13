"""Durable, request-scoped evidence for timeline model calls."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Any

from sglang_omni.client.types import GenerateRequest

from .contracts import ObservationEvent, TimelineSessionStart
from .window import EventDetectionWindow


logger = logging.getLogger(__name__)

_ARTIFACT_ROOT_ENV = "SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT"
_ARTIFACT_MAX_BYTES_ENV = "SGLANG_OMNI_TIMELINE_ARTIFACT_MAX_BYTES"
_DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024
_DEFAULT_QUEUE_SIZE = 256
_MEDIA_METADATA_KEYS = frozenset({"images", "videos", "audios"})
_WriteItem = tuple[str, tuple[Any, ...], dict[str, Any]] | None


class TimelineArtifactRecorder:
    """Persist exact model media, prompts, outputs and admission results."""

    def __init__(
        self,
        *,
        root: Path | None,
        start: TimelineSessionStart,
        model_version: str,
        prompt_version: str,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("timeline artifact byte limit must be positive")
        self._lock = threading.Lock()
        self._root = root
        self._max_bytes = max_bytes
        self._start = start
        self._session_dir: Path | None = None
        self._write_queue: Queue[_WriteItem] | None = None
        self._writer: threading.Thread | None = None
        if root is None:
            return
        session_name = (
            f"{_safe_segment(start.session_id)}__observer-{start.observer_epoch}__"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
        )
        self._session_dir = root / session_name
        self._model_version = model_version
        self._prompt_version = prompt_version
        self._write_queue = Queue(maxsize=_DEFAULT_QUEUE_SIZE)
        self._writer = threading.Thread(
            target=self._run_writer,
            name=f"timeline-artifacts-{_safe_segment(start.session_id)}",
            daemon=True,
        )
        self._writer.start()
        logger.info(
            "timeline event-only artifact recording configured session_id=%s "
            "root=%s max_bytes=%s",
            start.session_id,
            self._session_dir,
            max_bytes,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        start: TimelineSessionStart,
        model_version: str,
        prompt_version: str,
    ) -> "TimelineArtifactRecorder":
        raw_root = os.environ.get(_ARTIFACT_ROOT_ENV, "").strip()
        root = Path(raw_root).expanduser().resolve() if raw_root else None
        max_bytes = int(
            os.environ.get(_ARTIFACT_MAX_BYTES_ENV, str(_DEFAULT_MAX_BYTES))
        )
        return cls(
            root=root,
            start=start,
            model_version=model_version,
            prompt_version=prompt_version,
            max_bytes=max_bytes,
        )

    @property
    def enabled(self) -> bool:
        return self._session_dir is not None

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def record_window_event(self, event: dict[str, Any]) -> None:
        # Window diagnostics remain in the service log.  Persisting every
        # ready/discarded window would violate the event-only evidence policy.
        return

    def record_event_call(
        self,
        *,
        stream_epoch: int,
        window: EventDetectionWindow,
        attempt: int,
        request: GenerateRequest,
        video_mp4: bytes | None,
        image_jpegs: tuple[bytes, ...],
        audio_wav: bytes | None,
        result: dict[str, Any],
    ) -> Path | None:
        """Queue one complete model call only after an event was parsed."""

        if self._session_dir is None:
            return None
        call_dir = self._session_dir / (
            f"stream-{stream_epoch:04d}__end-{window.end_ms:012d}__"
            f"attempt-{attempt:02d}"
        )
        queued = self._enqueue(
            "_write_event_call",
            call_dir,
            stream_epoch,
            window,
            request,
            video_mp4,
            image_jpegs,
            audio_wav,
            result,
        )
        return call_dir if queued else None

    def _write_event_call(
        self,
        call_dir: Path,
        stream_epoch: int,
        window: EventDetectionWindow,
        request: GenerateRequest,
        video_mp4: bytes | None,
        image_jpegs: tuple[bytes, ...],
        audio_wav: bytes | None,
        result: dict[str, Any],
    ) -> None:
        self._ensure_session_dir()
        self._write_model_input(
            call_dir,
            stream_epoch,
            window,
            request,
            video_mp4,
            image_jpegs,
            audio_wav,
        )
        self._write_model_result(call_dir, result)

    def _write_model_input(
        self,
        call_dir: Path,
        stream_epoch: int,
        window: EventDetectionWindow,
        request: GenerateRequest,
        video_mp4: bytes | None,
        image_jpegs: tuple[bytes, ...],
        audio_wav: bytes | None,
    ) -> None:
        input_dir = call_dir / "input"
        with self._lock:
            input_dir.mkdir(parents=True, exist_ok=False)
            media: dict[str, list[dict[str, Any]]] = {
                "videos": [],
                "images": [],
                "audios": [],
            }
            if video_mp4 is not None:
                path = input_dir / "video.mp4"
                _atomic_bytes(path, video_mp4)
                media["videos"].append(_media_record(path, video_mp4))
            for index, payload in enumerate(image_jpegs):
                path = input_dir / f"frame-{index:02d}.jpg"
                _atomic_bytes(path, payload)
                media["images"].append(_media_record(path, payload))
            if audio_wav is not None:
                path = input_dir / "audio.wav"
                _atomic_bytes(path, audio_wav)
                media["audios"].append(_media_record(path, audio_wav))
            frame_dir = input_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=False)
            history_records = _store_window_frames(
                frame_dir,
                "history",
                window.history_frames,
            )
            current_records = _store_window_frames(
                frame_dir,
                "current",
                window.current_frames,
            )
            _atomic_json(
                input_dir / "frames.json",
                {
                    "history": history_records,
                    "current": current_records,
                },
            )
            _atomic_json(
                call_dir / "request.json",
                {
                    "schema_version": "timeline-model-call.v1",
                    "recorded_at": _now(),
                    "stream_epoch": stream_epoch,
                    "window": _window_record(window),
                    "request": _request_record(request, media),
                },
            )

    def _write_model_result(
        self,
        call_dir: Path,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            _atomic_json(
                call_dir / "result.json",
                {
                    "schema_version": "timeline-model-result.v1",
                    "recorded_at": _now(),
                    **result,
                },
            )

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        """Flush queued artifacts without ever joining from inference."""

        queue = self._write_queue
        writer = self._writer
        if queue is None or writer is None:
            return
        try:
            queue.put(None, timeout=1.0)
        except Full:
            logger.warning(
                "timeline artifact queue full during close session_id=%s",
                self._start.session_id,
            )
            return
        writer.join(timeout=max(0.0, timeout_seconds))
        if writer.is_alive():
            logger.warning(
                "timeline artifact flush timed out session_id=%s pending=%s",
                self._start.session_id,
                queue.qsize(),
            )
        self._writer = None
        self._write_queue = None

    def _enqueue(self, method: str, *args: Any, **kwargs: Any) -> bool:
        queue = self._write_queue
        if queue is None:
            return False
        try:
            queue.put_nowait((method, args, kwargs))
            return True
        except Full:
            logger.warning(
                "timeline artifact queue full; dropping record "
                "session_id=%s method=%s pending=%s",
                self._start.session_id,
                method,
                queue.qsize(),
            )
            return False

    def _run_writer(self) -> None:
        queue = self._write_queue
        assert queue is not None
        while True:
            item = queue.get()
            try:
                if item is None:
                    return
                method, args, kwargs = item
                getattr(self, method)(*args, **kwargs)
            except Exception:
                logger.exception(
                    "timeline artifact background write failed session_id=%s",
                    self._start.session_id,
                )
            finally:
                queue.task_done()

    def _ensure_session_dir(self) -> None:
        assert self._root is not None
        assert self._session_dir is not None
        if self._session_dir.exists():
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._session_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            self._session_dir / "session.json",
            {
                "schema_version": "timeline-artifacts.v1",
                "created_at": _now(),
                "session": asdict(self._start),
                "model_version": self._model_version,
                "prompt_version": self._prompt_version,
                "retention_max_bytes": self._max_bytes,
                "recording_policy": "event_outputs_only",
            },
        )
        self._prune_old_sessions()
        logger.info(
            "timeline event artifact recording activated session_id=%s root=%s",
            self._start.session_id,
            self._session_dir,
        )

    def _prune_old_sessions(self) -> None:
        assert self._root is not None
        assert self._session_dir is not None
        sessions = [
            item
            for item in self._root.iterdir()
            if item.is_dir() and item != self._session_dir
        ]
        sizes = {item: _tree_size(item) for item in sessions}
        total = sum(sizes.values())
        for item in sorted(sessions, key=lambda path: path.stat().st_mtime):
            if total <= self._max_bytes:
                break
            size = sizes[item]
            _remove_tree(item)
            total -= size
            logger.info("pruned timeline artifact session path=%s bytes=%s", item, size)


def observation_records(events: tuple[ObservationEvent, ...]) -> list[dict[str, Any]]:
    return [asdict(event) for event in events]


def _request_record(
    request: GenerateRequest,
    media: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    value = request.to_dict()
    metadata = dict(value.get("metadata") or {})
    for key in _MEDIA_METADATA_KEYS:
        if key in metadata:
            metadata[key] = media[key]
    value["metadata"] = metadata
    return value


def _window_record(window: EventDetectionWindow) -> dict[str, Any]:
    return {
        "start_ms": window.start_ms,
        "current_start_ms": window.current_start_ms,
        "end_ms": window.end_ms,
        "history_frame_count": len(window.history_frames),
        "current_frame_count": len(window.current_frames),
        "audio_status": window.audio_status.value,
        "audio_start_ms": window.audio_start_ms,
        "evidence_mode": window.evidence_mode,
    }


def _store_window_frames(
    directory: Path,
    role: str,
    frames: tuple[Any, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        extension = _frame_extension(frame.encoding)
        filename = (
            f"{role}-{index:02d}__seq-{frame.sequence:08d}."
            f"{extension}"
        )
        path = directory / filename
        _atomic_bytes(path, frame.payload)
        records.append(
            _frame_record(
                frame,
                stored_as=f"frames/{filename}",
            )
        )
    return records


def _frame_record(frame: Any, *, stored_as: str) -> dict[str, Any]:
    return {
        "sequence": frame.sequence,
        "stream_epoch": frame.stream_epoch,
        "start_ms": frame.start_ms,
        "end_ms": frame.end_ms,
        "encoding": frame.encoding,
        "payload_bytes": len(frame.payload),
        "sha256": hashlib.sha256(frame.payload).hexdigest(),
        "stored_as": stored_as,
    }


def _frame_extension(encoding: str) -> str:
    normalized = encoding.strip().lower().split(";", 1)[0]
    if normalized in {"image/jpeg", "jpeg", "jpg"}:
        return "jpg"
    if normalized in {"image/png", "png"}:
        return "png"
    if normalized in {"image/webp", "webp"}:
        return "webp"
    return "bin"


def _media_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "stored_as": str(path.name),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if normalized:
        return normalized[:160]
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _tree_size(root: Path) -> int:
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def _remove_tree(root: Path) -> None:
    for item in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    root.rmdir()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
