"""Strict direct/delegate routing pipeline around model-1 implementations."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    MediaDirective,
    OutputDirective,
    ROUTE_BY_TOKEN,
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

    async def classify(
        self,
        request: TaskClassificationRequest,
    ) -> TaskClassificationResult:
        raw = await self._model.classify(request)
        try:
            route_raw, output_raw, task_raw, media_raw = raw.split("|")
            route_token = RouteToken(route_raw)
            output_directive = OutputDirective(output_raw)
            task_directive = TaskDirective(task_raw)
            media_directive = MediaDirective(media_raw)
        except (ValueError, AttributeError) as exc:
            raise InvalidRouteOutput(
                "turn-router model must output exactly "
                "route|output_directive|task_directive|media_directive"
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
                model_id=self._model.model_id,
                model_version=self._model.model_version,
            )
        except ValueError as exc:
            raise InvalidRouteOutput(
                "turn-router model must output exactly "
                "route|output_directive|task_directive|media_directive"
            ) from exc
