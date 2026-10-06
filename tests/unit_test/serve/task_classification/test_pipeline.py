from __future__ import annotations

import json

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


def _output(
    route: str,
    output: str = "keep",
    task: str = "keep",
    media: str = "none",
    *,
    cue: dict[str, str] | None = None,
) -> str:
    return json.dumps(
        {
            "route_token": route,
            "output_directive": output,
            "task_directive": task,
            "media_directive": media,
            "response_locale": "zh-CN",
            "request_cue": cue,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


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
            _output("direct"),
            RouteToken.DIRECT,
            BrainRoute.BRAIN1,
            OutputDirective.KEEP,
            TaskDirective.KEEP,
            MediaDirective.NONE,
        ),
        (
            _output("control", "suppress_reply", media="stop"),
            RouteToken.CONTROL,
            BrainRoute.CONTROL,
            OutputDirective.SUPPRESS_REPLY,
            TaskDirective.KEEP,
            MediaDirective.STOP,
        ),
        (
            _output("control", "suppress_reply", "cancel_all"),
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
async def test_delegate_requires_and_returns_grounded_request_cue() -> None:
    raw = _output(
        "delegate",
        cue={"verb": "订", "object": "明天的火车票", "language": "zh-CN"},
    )
    result = await TaskClassificationPipeline(_Model(raw)).classify(_request())

    assert result.request_cue is not None
    assert result.request_cue.verb == "订"
    assert result.request_cue.object == "明天的火车票"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "object_text",
    ["你会唱什么歌", "你能查天气吗", "帮我查一下新闻", "what can you sing?"],
)
async def test_delegate_rejects_question_or_request_clause_as_cue_object(
    object_text: str,
) -> None:
    language = "en-US" if object_text.startswith("what") else "zh-CN"
    raw = _output(
        "delegate",
        cue={"verb": "查", "object": object_text, "language": language},
    )

    with pytest.raises(InvalidRouteOutput, match="exact v5 JSON contract"):
        await TaskClassificationPipeline(_Model(raw)).classify(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        _output("Delegate"),
        "delegate because tools",
        _output("control"),
        _output("direct", task="cancel_all"),
        _output("delegate"),
        _output(
            "direct",
            cue={"verb": "查", "object": "天气", "language": "zh-CN"},
        ),
    ],
)
async def test_rejects_whitespace_case_or_explanation(raw: str) -> None:
    with pytest.raises(
        InvalidRouteOutput,
        match="exact v5 JSON contract",
    ):
        await TaskClassificationPipeline(_Model(raw)).classify(_request())
