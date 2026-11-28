from __future__ import annotations

import hashlib

import pytest

from sglang_omni.serve.agent_prompt_snapshot import decode_router_prompt_snapshot
from sglang_omni.serve.task_classification.contracts import (
    CONTRACT_VERSION,
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
    model_version = "v1.0"

    def __init__(self, output: str) -> None:
        self.output = output

    async def classify(self, request: TaskClassificationRequest) -> str:
        del request
        return self.output


def _prompt_snapshot():
    rules = {
        "system_prompt": "Agent Router",
        "mcp_scope": "MCP scope",
        "multimodal_scope": "multimodal scope",
    }
    return decode_router_prompt_snapshot({
        "consumer": "sglang_omni.router.v1_0",
        "release_id": "router-release-v1",
        "bundle_sha256": "a" * 64,
        "binding": {
            "assembly_contract": "turn-router-v1.0",
            "output_protocol": "turn-router-code-v1.0",
            "model_id": "turn-router",
            "model_version": "v1.0",
        },
        **rules,
        "item_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in rules.items()
        },
    })


def _request(
    text: str = "帮我订明天的火车票",
    *,
    router_profile: str = "v1.0",
    has_active_agent: bool = False,
) -> TaskClassificationRequest:
    return TaskClassificationRequest(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        identity_epoch=2,
        input_revision=4,
        text=text,
        media=(),
        router_profile=router_profile,
        has_active_agent=has_active_agent,
        contract_version=CONTRACT_VERSION,
        prompt_snapshot=_prompt_snapshot(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "needs_mcp", "needs_multimodal", "token", "route"),
    [
        ("0", False, False, RouteToken.DIRECT, BrainRoute.BRAIN1),
        ("1", True, False, RouteToken.DELEGATE, BrainRoute.BRAIN2),
        ("2", False, True, RouteToken.DIRECT, BrainRoute.BRAIN1),
        ("3", True, True, RouteToken.DELEGATE, BrainRoute.BRAIN2),
    ],
)
async def test_maps_the_four_codes_to_capability_requirements(
    raw, needs_mcp, needs_multimodal, token, route
) -> None:
    result = await TaskClassificationPipeline(_Model(raw)).classify(_request())
    assert result.route_code == int(raw)
    assert result.needs_mcp is needs_mcp
    assert result.needs_multimodal is needs_multimodal
    assert result.route_token is token
    assert result.route is route
    assert result.output_directive is OutputDirective.KEEP
    assert result.task_directive is TaskDirective.KEEP
    assert result.media_directive is MediaDirective.NONE
    assert result.request_cue is None
    assert result.contract_version == CONTRACT_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["", "4", " 1", "1\n", "direct", '{"route":1}'])
async def test_rejects_everything_except_one_exact_code(raw: str) -> None:
    with pytest.raises(InvalidRouteOutput, match="exactly one of 0, 1, 2, 3"):
        await TaskClassificationPipeline(_Model(raw)).classify(_request())


@pytest.mark.asyncio
async def test_rejects_retired_relation_codes() -> None:
    with pytest.raises(InvalidRouteOutput, match="exactly one of 0, 1, 2, 3"):
        await TaskClassificationPipeline(_Model("4")).classify(
            _request(router_profile="v1.0")
        )


@pytest.mark.asyncio
async def test_legacy_locale_is_derived_outside_model_authority() -> None:
    zh = await TaskClassificationPipeline(_Model("0")).classify(_request("你好"))
    en = await TaskClassificationPipeline(_Model("0")).classify(_request("hello"))
    assert zh.response_locale == "zh-CN"
    assert en.response_locale == "en-US"
