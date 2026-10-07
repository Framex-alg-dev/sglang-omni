"""Strict direct/delegate routing pipeline around model-1 implementations."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol

from .contracts import (
    MediaDirective,
    OutputDirective,
    ROUTE_BY_TOKEN,
    RequestCuePlan,
    RouteToken,
    TaskClassificationRequest,
    TaskClassificationResult,
    TaskDirective,
)


class TaskClassificationModel(Protocol):
    """Stable seam for the prompted model today and a trained model later."""

    model_id: str
    model_version: str

    async def classify(self, request: TaskClassificationRequest) -> str: ...


class InvalidRouteOutput(ValueError):
    """The model responded, but not with the closed route vocabulary."""


class TaskClassificationPipeline:
    """Enforce the model-1 contract's closed routing vocabulary."""

    def __init__(self, model: TaskClassificationModel) -> None:
        self._model = model

    async def prewarm(self) -> None:
        prewarm = getattr(self._model, "prewarm", None)
        if callable(prewarm):
            await prewarm()

    async def classify(
        self,
        request: TaskClassificationRequest,
    ) -> TaskClassificationResult:
        raw = await self._model.classify(request)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, Mapping):
                raise ValueError("router output must be an object")
            allowed = {
                "route_token",
                "output_directive",
                "task_directive",
                "media_directive",
                "response_locale",
                "request_cue",
            }
            if set(parsed) != allowed:
                raise ValueError("router output fields are invalid")
            route_token = RouteToken(_string(parsed, "route_token"))
            output_directive = OutputDirective(_string(parsed, "output_directive"))
            task_directive = TaskDirective(_string(parsed, "task_directive"))
            media_directive = MediaDirective(_string(parsed, "media_directive"))
            response_locale = _string(parsed, "response_locale")
            request_cue = _request_cue(parsed.get("request_cue"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidRouteOutput(
                "turn-router model must output the exact v5 JSON contract"
            ) from exc
        try:
            return TaskClassificationResult(
                request_id=request.request_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
                identity_epoch=request.identity_epoch,
                input_revision=request.input_revision,
                route_token=route_token,
                route=ROUTE_BY_TOKEN[route_token],
                output_directive=output_directive,
                task_directive=task_directive,
                media_directive=media_directive,
                response_locale=response_locale,
                request_cue=request_cue,
                model_id=self._model.model_id,
                model_version=self._model.model_version,
            )
        except ValueError as exc:
            raise InvalidRouteOutput(
                "turn-router model must output the exact v5 JSON contract"
            ) from exc


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _request_cue(raw: object) -> RequestCuePlan | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"verb", "object", "language"}:
        raise ValueError("request_cue must use the exact cue contract")
    return RequestCuePlan(
        verb=_string(raw, "verb"),
        object=_string(raw, "object"),
        language=_string(raw, "language"),
    )
