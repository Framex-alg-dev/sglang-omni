# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible error helpers for TTS serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse

from sglang_omni.utils.framex_service_logs import current_service_log_context


@dataclass
class SpeechAPIError(Exception):
    """A user-visible TTS API error."""

    message: str
    status_code: int
    error_type: str
    param: str | None = None
    code: int | str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


def openai_error_payload(
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: int | str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-style error envelope."""

    error = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    request_id = current_service_log_context().request_id
    if request_id is not None:
        error["request_id"] = request_id
    return {"error": error}


def openai_error_response(
    message: str,
    *,
    status_code: int,
    error_type: str,
    param: str | None = None,
    code: int | str | None = None,
) -> JSONResponse:
    """Return an OpenAI-style JSON error response."""

    return JSONResponse(
        status_code=status_code,
        content=openai_error_payload(
            message,
            error_type=error_type,
            param=param,
            code=code,
        ),
    )


def speech_error_response(error: SpeechAPIError) -> JSONResponse:
    return openai_error_response(
        error.message,
        status_code=error.status_code,
        error_type=error.error_type,
        param=error.param,
        code=error.code,
    )


def bad_request(message: str, *, param: str | None = None) -> SpeechAPIError:
    return SpeechAPIError(
        message=message,
        status_code=400,
        error_type="BadRequestError",
        param=param,
        code=400,
    )


def internal_error(message: str, *, param: str | None = None) -> SpeechAPIError:
    return SpeechAPIError(
        message=message,
        status_code=500,
        error_type="server_error",
        param=param,
        code=None,
    )


def service_unavailable(message: str, *, param: str | None = None) -> SpeechAPIError:
    return SpeechAPIError(
        message=message,
        status_code=503,
        error_type="server_error",
        param=param,
        code=None,
    )
