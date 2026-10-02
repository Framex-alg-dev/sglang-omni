from __future__ import annotations

import pytest

from sglang_omni.serve.task_classification.contracts import (
    BrainRoute,
    MediaDirective,
    OutputDirective,
    RouteToken,
    TaskClassificationRequest,
    TaskDirective,
)
from sglang_omni.serve.task_classification.pipeline import (
    InvalidRouteOutput,
    TaskClassificationPipeline,
)


class _Model:
    model_id = "turn-router"
    model_version = "2026-09-25"

    def __init__(self, output: str) -> None:
        self.output = output

    async def classify(self, request: TaskClassificationRequest) -> str:
        del request
        return self.output


def _request() -> TaskClassificationRequest:
    return TaskClassificationRequest(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        identity_epoch=2,
        input_revision=4,
        text="帮我订明天的火车票",
        media=(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "token", "route", "output", "task", "media"),
    [
        (
            "direct|keep|keep|none",
            RouteToken.DIRECT,
            BrainRoute.BRAIN1,
            OutputDirective.KEEP,
            TaskDirective.KEEP,
            MediaDirective.NONE,
        ),
        (
            "delegate|suppress_reply|keep|stop",
            RouteToken.DELEGATE,
            BrainRoute.BRAIN2,
            OutputDirective.SUPPRESS_REPLY,
            TaskDirective.KEEP,
            MediaDirective.STOP,
        ),
        (
            "control|suppress_reply|cancel_all|none",
            RouteToken.CONTROL,
            BrainRoute.CONTROL,
            OutputDirective.SUPPRESS_REPLY,
            TaskDirective.CANCEL_ALL,
            MediaDirective.NONE,
        ),
    ],
)
async def test_accepts_only_closed_router_vocabulary(
    raw, token, route, output, task, media
) -> None:
    result = await TaskClassificationPipeline(_Model(raw)).classify(_request())
    assert result.route_token is token
    assert result.route is route
    assert result.output_directive is output
    assert result.task_directive is task
    assert result.media_directive is media
    assert result.input_revision == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        " delegate|keep|keep|none",
        "Delegate|keep|keep|none",
        "delegate because tools",
        "control|keep|keep|none",
        "direct|keep|cancel_all|none",
        "control|suppress_reply|keep|stop",
    ],
)
async def test_rejects_whitespace_case_or_explanation(raw: str) -> None:
    with pytest.raises(
        InvalidRouteOutput,
        match=r"route\|output_directive\|task_directive\|media_directive",
    ):
        await TaskClassificationPipeline(_Model(raw)).classify(_request())
