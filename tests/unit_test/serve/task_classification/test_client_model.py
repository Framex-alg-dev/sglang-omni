from __future__ import annotations

import hashlib

import pytest

from sglang_omni.client.types import CompletionResult
from sglang_omni.serve.task_classification.client_model import (
    SglangClientTaskClassificationModel,
)
from sglang_omni.serve.task_classification.contracts import (
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from sglang_omni.serve.task_classification import prompt_v09, prompt_v11, prompt_v12
from sglang_omni.serve.task_classification.prompt_v10 import (
    SINGLE_TURN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)


class _Client:
    def __init__(self, output: str = "0") -> None:
        self.calls = []
        self.output = output

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id, audio_format))
        return CompletionResult(request_id=request_id, text=self.output)


class _SequenceClient(_Client):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = iter(outputs)

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id, audio_format))
        return CompletionResult(request_id=request_id, text=next(self.outputs))


def _model(client: _Client) -> SglangClientTaskClassificationModel:
    return SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="router-v0.6",
    )


def _request(
    *,
    text="你好",
    media=(),
    history=(),
    has_active_agent=False,
    pending_confirmation=False,
    follow_up_required=False,
    active_task_goal=None,
    active_task_status=None,
    active_task_family=None,
    active_task_phase=None,
    active_task_awaiting_input=None,
    router_profile="v1.0",
) -> TaskClassificationRequest:
    return TaskClassificationRequest(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        identity_epoch=1,
        input_revision=2,
        text=text,
        media=media,
        router_history=history,
        has_active_agent=has_active_agent,
        pending_confirmation=pending_confirmation,
        follow_up_required=follow_up_required,
        active_task_goal=active_task_goal,
        active_task_status=active_task_status,
        active_task_family=active_task_family,
        active_task_phase=active_task_phase,
        active_task_awaiting_input=active_task_awaiting_input,
        router_profile=router_profile,
    )


def _audio(payload: bytes = b"audio") -> ClassificationMediaRef:
    return ClassificationMediaRef(
        media_id="audio-1",
        kind="audio",
        start_ms=0,
        end_ms=100,
        encoding="wav",
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


@pytest.mark.asyncio
async def test_single_turn_uses_exact_v10_prompt_and_one_token_decode() -> None:
    client = _Client("1")
    output = await _model(client).classify(_request(text="你会唱歌吗"))

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert request.messages[1].content[-1]["text"].endswith(
        "[输出]\n只输出 0、1、2 或 3："
    )
    assert request.sampling.temperature == 0.0
    assert request.sampling.top_p == 1.0
    assert request.sampling.max_new_tokens == 1
    assert request.max_tokens == 1
    assert request.sampling.stop == ["\n"]
    assert request.metadata["prefix_cache_namespace"] == "turn-router:v10"
    assert request.metadata["cache_static_system_only"] is True
    assert request.metadata["trace_id"] == "request-1"
    assert request.metadata["operation_id"] == "request-1"
    assert request.metadata["classification_id"] == "request-1"
    assert request.metadata["session_id"] == "session-1"
    assert request.metadata["turn_id"] == "turn-1"
    assert request.metadata["input_revision"] == 2
    assert request.metadata["model"] == "shared-base"
    assert request.metadata["model_version"] == "router-v0.6"


@pytest.mark.asyncio
async def test_v09_profile_uses_v09_prompt_and_separate_cache_namespace() -> None:
    client = _Client("0")
    model = _model(client)
    await model.classify(_request(router_profile="v0.9"))

    request = client.calls[0][0]
    assert request.messages[0].content == prompt_v09.SINGLE_TURN_SYSTEM_PROMPT
    assert request.metadata["router_profile"] == "v0.9"
    assert request.metadata["prefix_cache_namespace"] == "turn-router:v09"
    assert model.model_version_for(_request(router_profile="v0.9")).endswith(
        "+prompt-v0.9"
    )



@pytest.mark.asyncio
async def test_v11_profile_carries_active_task_goal_and_status() -> None:
    client = _Client("4")
    model = _model(client)
    output = await model.classify(
        _request(
            text="继续",
            has_active_agent=True,
            pending_confirmation=True,
            follow_up_required=True,
            active_task_goal="唱一首歌",
            active_task_status="waiting_user",
            router_profile="v1.1",
        )
    )

    assert output == "4"
    request = client.calls[0][0]
    assert request.messages[0].content == prompt_v11.SINGLE_TURN_SYSTEM_PROMPT
    prompt_text = str(request.messages[1].content)
    assert "目标=\"唱一首歌\"" in prompt_text
    assert "状态=waiting_user" in prompt_text
    assert "等待确认=是" in prompt_text
    assert request.metadata["router_profile"] == "v1.1"
    assert request.metadata["prefix_cache_namespace"] == "turn-router:v11"

@pytest.mark.asyncio
async def test_v12_profile_carries_structured_task_context_and_constrains_decode() -> None:
    client = _Client("4")
    await _model(client).classify(
        _request(
            text="分手快乐",
            has_active_agent=True,
            follow_up_required=True,
            active_task_goal="等待用户选择歌曲",
            active_task_status="waiting_user",
            active_task_family="singing",
            active_task_phase="waiting_user",
            active_task_awaiting_input="selection",
            router_profile="v1.2",
        )
    )

    request = client.calls[0][0]
    assert request.messages[0].content == prompt_v12.SINGLE_TURN_SYSTEM_PROMPT
    prompt_text = str(request.messages[1].content)
    assert "类型=singing" in prompt_text
    assert "阶段=waiting_user" in prompt_text
    assert "等待内容=selection" in prompt_text
    assert request.sampling.seed == 0
    assert request.sampling.regex == "[0-5]"
    assert request.metadata["prefix_cache_namespace"] == "turn-router:v12"


@pytest.mark.asyncio
async def test_v12_without_active_task_constrains_decode_to_new_or_direct() -> None:
    client = _Client("0")
    await _model(client).classify(_request(router_profile="v1.2"))

    request = client.calls[0][0]
    assert request.sampling.regex == "[0-3]"


@pytest.mark.asyncio
async def test_multi_turn_uses_only_last_complete_assistant_reply() -> None:
    client = _Client("1")
    history = (
        {"role": "user", "text": "derived ASR must be ignored"},
        {"role": "assistant", "text": "第一条"},
        {"role": "assistant", "text": "要播放这首歌吗？"},
    )
    await _model(client).classify(_request(text="可以", history=history))

    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    prompt_text = str(request.messages[1].content)
    assert "要播放这首歌吗？" in prompt_text
    assert "第一条" not in prompt_text
    assert "derived ASR" not in prompt_text


@pytest.mark.asyncio
async def test_interrupted_history_reaches_prompt_with_runtime_task_flags() -> None:
    client = _Client("1")
    history = ({
        "role": "assistant",
        "text": "要播放",
        "playback_status": "interrupted",
    },)
    await _model(client).classify(_request(
        text="继续",
        history=history,
        has_active_agent=True,
        pending_confirmation=True,
        follow_up_required=True,
    ))

    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    prompt_text = str(request.messages[1].content)
    assert "播放状态=interrupted" in prompt_text
    assert "interrupted" in prompt_text
    assert "存在活动 MCP 任务=是" in prompt_text
    assert "正在等待用户确认=是" in prompt_text
    assert "正在等待用户补充=是" in prompt_text


@pytest.mark.asyncio
async def test_active_task_without_reply_history_uses_single_turn_prompt() -> None:
    client = _Client("1")
    await _model(client).classify(_request(
        text="继续",
        has_active_agent=True,
        follow_up_required=True,
    ))

    request = client.calls[0][0]
    assert request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    prompt_text = str(request.messages[1].content)
    assert "存在活动 MCP 任务=是" in prompt_text
    assert "正在等待用户补充=是" in prompt_text


@pytest.mark.asyncio
async def test_original_audio_has_no_asr_text_and_uses_same_policy() -> None:
    client = _Client("1")
    output = await _model(client).classify(_request(text=None, media=(_audio(),)))

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert request.messages[1].content[1] == {"type": "audio"}
    assert "CURRENT_USER_TEXT" not in str(request.messages[1].content)
    assert "MCP=唱歌、搜索（含天气）、查交通" in str(request.messages[1].content)
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")


@pytest.mark.asyncio
async def test_prewarm_populates_all_profile_prefixes() -> None:
    client = _SequenceClient(["0", "1", "0", "1", "0", "4", "0", "4"])
    await _model(client).prewarm()
    assert len(client.calls) == 8
    (
        v10_single,
        v10_multi,
        v09_single,
        v09_multi,
        v11_single,
        v11_multi,
        v12_single,
        v12_multi,
    ) = (item[0] for item in client.calls)
    assert v10_single.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert v10_multi.messages[0].content == SYSTEM_PROMPT
    assert v09_single.messages[0].content == prompt_v09.SINGLE_TURN_SYSTEM_PROMPT
    assert v09_multi.messages[0].content == prompt_v09.SYSTEM_PROMPT
    assert v11_single.messages[0].content == prompt_v11.SINGLE_TURN_SYSTEM_PROMPT
    assert v11_multi.messages[0].content == prompt_v11.SYSTEM_PROMPT
    assert v12_single.messages[0].content == prompt_v12.SINGLE_TURN_SYSTEM_PROMPT
    assert v12_multi.messages[0].content == prompt_v12.SYSTEM_PROMPT
    assert v10_single.sampling.max_new_tokens == 1
    assert v09_single.sampling.max_new_tokens == 1
    assert v11_single.sampling.max_new_tokens == 1
    assert v12_single.sampling.max_new_tokens == 1
    assert v12_single.sampling.regex == "[0-3]"
    assert v12_multi.sampling.regex == "[0-5]"
    assert v11_single.metadata["prefix_cache_namespace"] == "turn-router:v11"
    assert "需要我查询当前天气吗？" in str(v09_multi.messages[1].content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (["1"], "ordinary conversation"),
        (["0", "0"], "inherited MCP request"),
    ],
)
async def test_prewarm_rejects_broken_route_invariants(
    outputs: list[str], message: str
) -> None:
    client = _SequenceClient(outputs)
    with pytest.raises(RuntimeError, match=message):
        await _model(client).prewarm()


def test_rejects_visual_media() -> None:
    payload = b"image"
    with pytest.raises(ValueError, match="original user audio"):
        ClassificationMediaRef(
            media_id="image-1",
            kind="image",
            start_ms=0,
            end_ms=100,
            encoding="jpeg",
            checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )
