"""CLS model-call traces with text preserved and binary payloads omitted."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .structured_logs import emit_structured_log


logger = logging.getLogger(__name__)

_SECRET_FIELDS = {
    "api_key",
    "authorization",
    "secret",
    "secret_id",
    "secret_key",
    "access_token",
    "token",
}
_SERVICE_BY_LOG_TYPE = {
    "action": "omni-action-worker",
    "diagnostic": "omni-event-worker",
    "reply": "omni-reply-worker",
}
_SECRET_TEXT_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+|"
    r"(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"((?:cookie|secret|token|api[_-]?key)\s*[:=]\s*)[^\s,;]+"
)
_PHONE_RE = re.compile(r"(?<!\d)(1\d{2})\d{4}(\d{4})(?!\d)")


def _redact_text(value: str) -> str:
    def replace_secret(match: re.Match[str]) -> str:
        prefix = match.group(1) or match.group(2) or match.group(3) or ""
        return prefix + "***"

    return _PHONE_RE.sub(r"\1****\2", _SECRET_TEXT_RE.sub(replace_secret, value))


def cls_safe(value: Any, *, _path: tuple[str, ...] = ()) -> Any:
    """Keep JSON text while replacing secrets and binary media with metadata."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            separator = value.find(",")
            descriptor = value[:separator]
            return {
                "binary_omitted": True,
                "media_type": descriptor[5:].removesuffix(";base64"),
                "base64_characters": len(value) - separator - 1,
            }
        if len(_path) >= 2 and _path[-2:] == ("audio", "data"):
            return {
                "binary_omitted": True,
                "media_type": "audio",
                "base64_characters": len(value),
            }
        return _redact_text(value)
    if isinstance(value, bytes):
        return {
            "binary_omitted": True,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return cls_safe(
            {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            },
            _path=_path,
        )
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return cls_safe(model_dump(), _path=_path)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return cls_safe(to_dict(), _path=_path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            result[name] = (
                "[REDACTED]"
                if name.lower() in _SECRET_FIELDS
                else cls_safe(item, _path=(*_path, name))
            )
        return result
    if isinstance(value, (list, tuple, set)):
        return [
            cls_safe(item, _path=(*_path, str(index)))
            for index, item in enumerate(value)
        ]
    return repr(value)


def emit_model_trace(
    log_type: str,
    event: str,
    *,
    component: str,
    request_id: str,
    sequence: int,
    invocation_id: str | None = None,
    **fields: Any,
) -> bool:
    """Emit one non-blocking, correlated CLS stage record."""

    safe_fields = cls_safe(fields)
    invocation = invocation_id or request_id
    structured_ok = False
    try:
        structured_ok = emit_structured_log(
            log_type,
            event,
            component=component,
            request_id=request_id,
            invocation_id=invocation,
            sequence=sequence,
            **safe_fields,
        )
    except Exception:
        logger.warning(
            "model trace realtime log dropped event=%s", event, exc_info=True
        )

    service_ok = False
    try:
        service_ok = _emit_cls_service_log(
            log_type,
            event,
            component=component,
            request_id=request_id,
            invocation_id=invocation,
            sequence=sequence,
            fields=safe_fields,
        )
    except Exception:
        logger.warning("model trace CLS log dropped event=%s", event, exc_info=True)
    return structured_ok or service_ok


def _emit_cls_service_log(
    log_type: str,
    event: str,
    *,
    component: str,
    request_id: str,
    invocation_id: str,
    sequence: int,
    fields: Mapping[str, Any],
) -> bool:
    """Mirror a pre-sanitized trace into the app.log files collected by CLS."""

    service = _SERVICE_BY_LOG_TYPE.get(log_type)
    if service is None:
        return False
    service_logger = logging.getLogger(f"framex.{service}")
    if not service_logger.handlers:
        return False
    app_fields = dict(fields)
    root_session_id = app_fields.pop("root_session_id", None)
    turn_id = app_fields.pop("turn_id", None)
    if "session_id" in app_fields:
        app_fields["model_session_id"] = app_fields.pop("session_id")
    service_logger.info(
        "",
        extra={
            "framex_event": event,
            "framex_logger": __name__,
            "framex_request_id": request_id,
            "framex_root_session_id": root_session_id,
            "framex_turn_id": turn_id,
            "framex_fields": {
                "log_type": log_type,
                "component": component,
                "invocation_id": invocation_id,
                "sequence": sequence,
                **app_fields,
            },
        },
    )
    return True


__all__ = ["cls_safe", "emit_model_trace"]
