"""Strict four-code capability routing pipeline around model-1."""

from __future__ import annotations

import re
from typing import Protocol

from .contracts import (
    BrainRoute,
    MediaDirective,
    OutputDirective,
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
    """Enforce Router v0.7's exact one-character output vocabulary."""

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
        if raw not in {"0", "1", "2", "3"}:
            raise InvalidRouteOutput(
                "turn-router model must output exactly one of 0, 1, 2, or 3"
            )
        route_code = int(raw)
        needs_mcp = bool(route_code & 1)
        needs_multimodal = bool(route_code & 2)
        route_token = RouteToken.DELEGATE if needs_mcp else RouteToken.DIRECT
        route = BrainRoute.BRAIN2 if needs_mcp else BrainRoute.BRAIN1
        try:
            return TaskClassificationResult(
                request_id=request.request_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
                identity_epoch=request.identity_epoch,
                input_revision=request.input_revision,
                route_token=route_token,
                route=route,
                route_code=route_code,
                needs_mcp=needs_mcp,
                needs_multimodal=needs_multimodal,
                output_directive=OutputDirective.KEEP,
                task_directive=TaskDirective.KEEP,
                media_directive=MediaDirective.NONE,
                response_locale=_response_locale(request.text),
                request_cue=None,
                model_id=self._model.model_id,
                model_version=self._model.model_version,
            )
        except ValueError as exc:
            raise InvalidRouteOutput(
                "turn-router model returned an inconsistent capability code"
            ) from exc


def _response_locale(text: str | None) -> str:
    """Keep the legacy locale field outside Router v0.7's model authority."""

    if text and re.search(r"[\u3400-\u9fff]", text) is None:
        return "en-US"
    return "zh-CN"
