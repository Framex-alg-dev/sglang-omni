"""Validate OpenAI response-format envelopes for constrained decoding."""

from __future__ import annotations

import json
from typing import Any


def response_json_schema(response_format: dict[str, Any] | None) -> str | None:
    """Translate an OpenAI response-format envelope into an SGLang schema."""

    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    response_type = response_format.get("type")
    if response_type == "json_object":
        if set(response_format) != {"type"}:
            raise ValueError(
                "json_object response_format contains unsupported fields"
            )
        schema: dict[str, Any] = {"type": "object"}
    elif response_type == "json_schema":
        if set(response_format) != {"type", "json_schema"}:
            raise ValueError(
                "json_schema response_format contains unsupported fields"
            )
        envelope = response_format.get("json_schema")
        if not isinstance(envelope, dict):
            raise ValueError("response_format.json_schema must be an object")
        if set(envelope).difference({"name", "description", "strict", "schema"}):
            raise ValueError(
                "response_format.json_schema contains unsupported fields"
            )
        name = envelope.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("response_format.json_schema.name must be non-empty")
        strict = envelope.get("strict")
        if strict is not None and type(strict) is not bool:
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = envelope.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("response_format.json_schema.schema must be an object")
    else:
        raise ValueError("unsupported response_format type")
    try:
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("response_format JSON schema is not serializable") from exc
