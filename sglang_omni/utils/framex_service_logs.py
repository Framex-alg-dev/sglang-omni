# SPDX-License-Identifier: Apache-2.0
"""FrameX v1.0 service logs for CLS collection.

This module is deliberately separate from ``structured_logs``.  The latter is
an internal diagnostic/replay stream and may contain model-specific details;
only the bounded records produced here are safe to collect into CLS.
"""

from __future__ import annotations

import atexit
import contextvars
import fcntl
import hashlib
import json
import logging
import logging.handlers
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


LOG_SPEC_VERSION = "1.0"
DEFAULT_LOG_ROOT = Path("/var/log/framex")
MAX_ERROR_MESSAGE_BYTES = 2 * 1024
REQUEST_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
EVENT_RE = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+)+$")
SERVICES = frozenset(
    {"omni-event-worker", "omni-reply-worker", "omni-action-worker"}
)

_FORBIDDEN_FIELD_PARTS = frozenset(
    {
        "authorization",
        "cookie",
        "content",
        "messages",
        "prompt",
        "secret",
        "signature",
        "signed_url",
        "token",
        "user_text",
        "assistant_text",
        "audio",
        "video",
        "image",
        "payload",
    }
)
_SECRET_TEXT_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+|"
    r"(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"((?:cookie|secret|token|api[_-]?key)\s*[:=]\s*)[^\s,;]+"
)
_PHONE_RE = re.compile(r"(?<!\d)(1\d{2})\d{4}(\d{4})(?!\d)")

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "framex_request_id", default=None
)
_root_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "framex_root_session_id", default=None
)
_turn_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "framex_turn_id", default=None
)


def new_request_id() -> str:
    """Return the frozen FrameX ``req_`` plus 16-character identifier."""

    return "req_" + "".join(secrets.choice(REQUEST_ID_ALPHABET) for _ in range(16))


def valid_correlation_id(value: str | None) -> bool:
    return isinstance(value, str) and CORRELATION_ID_RE.fullmatch(value) is not None


def adopt_request_id(value: str | None) -> tuple[str, bool]:
    """Adopt a valid inbound ID, otherwise generate one and flag rejection."""

    if valid_correlation_id(value):
        return str(value), False
    return new_request_id(), value is not None


def correlation_id_or_digest(value: str, *, prefix: str) -> str:
    """Keep a valid legacy ID or map it to a stable, non-reversible ID."""

    if valid_correlation_id(value):
        return value
    if CORRELATION_ID_RE.fullmatch(prefix) is None or len(prefix) > 96:
        raise ValueError("correlation ID digest prefix is invalid")
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _optional_correlation_id(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not valid_correlation_id(value):
        raise ValueError(f"{name} must match {CORRELATION_ID_RE.pattern}")
    return value


@dataclass(frozen=True)
class ServiceLogContext:
    request_id: str | None = None
    root_session_id: str | None = None
    turn_id: str | None = None


def current_service_log_context() -> ServiceLogContext:
    return ServiceLogContext(
        request_id=_request_id_var.get(),
        root_session_id=_root_session_id_var.get(),
        turn_id=_turn_id_var.get(),
    )


@contextmanager
def bind_service_log_context(
    *,
    request_id: str | None = None,
    root_session_id: str | None = None,
    turn_id: str | None = None,
) -> Iterator[ServiceLogContext]:
    """Bind correlation IDs to the current async/task context."""

    request_id = _optional_correlation_id("request_id", request_id)
    root_session_id = _optional_correlation_id("root_session_id", root_session_id)
    turn_id = _optional_correlation_id("turn_id", turn_id)
    tokens = (
        (_request_id_var, _request_id_var.set(request_id)),
        (_root_session_id_var, _root_session_id_var.set(root_session_id)),
        (_turn_id_var, _turn_id_var.set(turn_id)),
    )
    try:
        yield current_service_log_context()
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def _git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        version = result.stdout.strip()
        if version:
            return version
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _truncate_utf8(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")


def _redact_text(value: str) -> str:
    def replace_secret(match: re.Match[str]) -> str:
        prefix = match.group(1) or match.group(2) or match.group(3) or ""
        return prefix + "***"

    redacted = _SECRET_TEXT_RE.sub(replace_secret, value)
    return _PHONE_RE.sub(r"\1****\2", redacted)


def _forbidden_field(name: str) -> bool:
    normalized = name.strip().lower()
    return any(part in normalized for part in _FORBIDDEN_FIELD_PARTS)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value[:100]]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in list(value.items())[:100]
            if not _forbidden_field(str(key))
        }
    return f"<{type(value).__name__}>"


def _safe_fields(fields: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if "session_id" in fields:
        raise ValueError(
            "ambiguous session_id is forbidden in service logs; use "
            "root_session_id or a service-specific session ID"
        )
    safe: dict[str, Any] = {}
    redacted: list[str] = []
    for name, value in fields.items():
        if value is None:
            continue
        if _forbidden_field(name):
            redacted.append(name)
            continue
        safe[name] = _safe_value(value)
    return safe, sorted(redacted)


class _FramexJsonFormatter(logging.Formatter):
    def __init__(self, *, service: str, version: str, host: str) -> None:
        super().__init__()
        self.service = service
        self.version = version
        self.host = host

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp_value = timestamp.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        fields = dict(getattr(record, "framex_fields", {}))
        value: dict[str, Any] = {
            "timestamp": timestamp_value,
            "level": record.levelname.upper(),
            "service": self.service,
            "env": "shared",
            "version": self.version,
            "host": self.host,
            "logger": str(getattr(record, "framex_logger", record.name)),
            "event": str(getattr(record, "framex_event")),
        }
        message = record.getMessage()
        if message:
            value["message"] = _redact_text(message)
        for name in ("request_id", "root_session_id", "turn_id"):
            item = getattr(record, f"framex_{name}", None)
            if item is not None:
                value[name] = item
        value.update(fields)
        if record.exc_info:
            value.setdefault("error_type", record.exc_info[0].__name__)
            value["exception_present"] = True
        if "error_message" in value:
            value["error_message"] = _truncate_utf8(
                _redact_text(str(value["error_message"])),
                MAX_ERROR_MESSAGE_BYTES,
            )
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = current_service_log_context()
        for name in ("request_id", "root_session_id", "turn_id"):
            explicit = getattr(record, f"framex_{name}", None)
            if explicit is None:
                setattr(record, f"framex_{name}", getattr(context, name))
        return True


class _DropAwareQueueHandler(logging.handlers.QueueHandler):
    def __init__(self, target: queue.Queue[logging.LogRecord]) -> None:
        super().__init__(target)
        self.dropped_records = 0
        self._drop_lock = threading.Lock()

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self.dropped_records += 1


class _DrainingQueueListener(logging.handlers.QueueListener):
    def enqueue_sentinel(self) -> None:
        # QueueListener's default put_nowait can fail exactly when a busy
        # process is shutting down. Wait for the consumer instead so accepted
        # records are flushed before the file handlers close.
        self.queue.put(self._sentinel)


class _LockedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Serialize append and rollover across co-located Worker processes."""

    def __init__(self, filename: Path, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self._lock_stream = Path(f"{filename}.lock").open("a+")

    def emit(self, record: logging.LogRecord) -> None:
        fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            self._reopen_if_rotated_by_peer()
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)
        finally:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)

    def _reopen_if_rotated_by_peer(self) -> None:
        if self.stream is None:
            self.stream = self._open()
            return
        try:
            current = os.stat(self.baseFilename)
            opened = os.fstat(self.stream.fileno())
        except FileNotFoundError:
            current = None
            opened = None
        if current is None or opened is None or (
            current.st_dev,
            current.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            self.stream.close()
            self.stream = self._open()

    def close(self) -> None:
        try:
            self._lock_stream.close()
        finally:
            super().close()


@dataclass
class _LoggerState:
    logger: logging.Logger
    handler: _DropAwareQueueHandler
    listener: _DrainingQueueListener


_LOGGER_STATES: dict[str, _LoggerState] = {}
_LOGGER_LOCK = threading.Lock()


def configure_service_logging(
    service: str,
    *,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    version: str | None = None,
    host: str | None = None,
    include_stdout: bool = True,
    queue_size: int = 8192,
) -> logging.Logger:
    """Configure one non-blocking JSON logger for a Worker service."""

    if service not in SERVICES:
        raise ValueError(f"unsupported FrameX service: {service!r}")
    if queue_size <= 0:
        raise ValueError("service log queue_size must be positive")
    with _LOGGER_LOCK:
        existing = _LOGGER_STATES.get(service)
        if existing is not None:
            return existing.logger

        resolved_version = version or _git_version()
        resolved_host = host or socket.gethostname()
        formatter = _FramexJsonFormatter(
            service=service, version=resolved_version, host=resolved_host
        )
        sinks: list[logging.Handler] = []
        if include_stdout:
            stdout = logging.StreamHandler(sys.stdout)
            stdout.setFormatter(formatter)
            sinks.append(stdout)

        log_path = Path(log_root) / service / "app.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _LockedRotatingFileHandler(
            log_path,
            maxBytes=100 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        sinks.append(file_handler)

        records: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=queue_size)
        queue_handler = _DropAwareQueueHandler(records)
        queue_handler.addFilter(_ContextFilter())
        logger = logging.getLogger(f"framex.{service}")
        logger.handlers.clear()
        logger.addHandler(queue_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        listener = _DrainingQueueListener(
            records, *sinks, respect_handler_level=True
        )
        listener.start()
        _LOGGER_STATES[service] = _LoggerState(logger, queue_handler, listener)
        return logger


def emit_service_event(
    service: str,
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    logger_name: str | None = None,
    request_id: str | None = None,
    root_session_id: str | None = None,
    turn_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit a CLS-safe event using the v1.0 field contract."""

    if EVENT_RE.fullmatch(event) is None:
        raise ValueError("event must be a stable dotted lowercase name")
    for name, value in (
        ("request_id", request_id),
        ("root_session_id", root_session_id),
        ("turn_id", turn_id),
    ):
        _optional_correlation_id(name, value)
    safe, redacted = _safe_fields(fields)
    if redacted:
        safe["redacted_fields"] = redacted
    # Production entry points configure the service before accepting traffic.
    # Keeping emission a no-op before configuration prevents observability from
    # breaking library users, unit tests, or readiness probes on hosts where
    # /var/log/framex is intentionally not mounted yet.
    with _LOGGER_LOCK:
        state = _LOGGER_STATES.get(service)
    if state is None:
        return
    logger = state.logger
    logger.log(
        level,
        message or "",
        extra={
            "framex_event": event,
            "framex_logger": logger_name or logger.name,
            "framex_request_id": request_id,
            "framex_root_session_id": root_session_id,
            "framex_turn_id": turn_id,
            "framex_fields": safe,
        },
    )


def shutdown_service_logging(service: str | None = None) -> None:
    with _LOGGER_LOCK:
        names = [service] if service is not None else list(_LOGGER_STATES)
        states = [
            _LOGGER_STATES.pop(name)
            for name in names
            if name in _LOGGER_STATES
        ]
    for state in states:
        state.listener.stop()
        for handler in tuple(state.listener.handlers):
            handler.flush()
            handler.close()
        state.logger.handlers.clear()


atexit.register(shutdown_service_logging)


__all__ = [
    "CORRELATION_ID_RE",
    "LOG_SPEC_VERSION",
    "SERVICES",
    "ServiceLogContext",
    "adopt_request_id",
    "bind_service_log_context",
    "configure_service_logging",
    "correlation_id_or_digest",
    "current_service_log_context",
    "emit_service_event",
    "new_request_id",
    "shutdown_service_logging",
    "valid_correlation_id",
]
