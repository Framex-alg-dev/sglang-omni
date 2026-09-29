from __future__ import annotations

import pytest

from sglang_omni.serve.task_classification.contracts import (
    BrainRoute,
    RouteToken,
    TaskClassificationRequest,
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
    ("raw", "token", "route"),
    [
        ("direct", RouteToken.DIRECT, BrainRoute.BRAIN1),
        ("delegate", RouteToken.DELEGATE, BrainRoute.BRAIN2),
        ("cancel", RouteToken.CANCEL, BrainRoute.CONTROL),
    ],
)
async def test_accepts_only_closed_router_vocabulary(raw, token, route) -> None:
    result = await TaskClassificationPipeline(_Model(raw)).classify(_request())
    assert result.route_token is token
    assert result.route is route
    assert result.input_revision == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [" delegate\n", "Delegate", "delegate because tools"])
async def test_rejects_whitespace_case_or_explanation(raw: str) -> None:
    with pytest.raises(
        InvalidRouteOutput,
        match="exactly direct, delegate, or cancel",
    ):
        await TaskClassificationPipeline(_Model(raw)).classify(_request())
