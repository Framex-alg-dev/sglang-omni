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
from sglang_omni.serve.task_classification.prompt_v09 import (
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
async def test_single_turn_uses_exact_v09_prompt_and_one_token_decode() -> None:
    client = _Client("1")
    output = await _model(client).classify(_request(text="你会唱歌吗"))

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert request.messages[1].content[-1]["text"] == "[OUTPUT]\n只输出 0、1、2 或 3："
    assert request.sampling.temperature == 0.0
    assert request.sampling.top_p == 1.0
    assert request.sampling.max_new_tokens == 1
    assert request.max_tokens == 1
    assert request.sampling.stop == ["\n"]
    assert request.metadata["prefix_cache_namespace"] == "turn-router:v09"
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
    assert "playback_status" in prompt_text
    assert "interrupted" in prompt_text
    assert "has_active_agent=true" in prompt_text
    assert "pending_confirmation=true" in prompt_text
    assert "follow_up_required=true" in prompt_text


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
    assert "has_active_agent=true" in prompt_text
    assert "follow_up_required=true" in prompt_text
    assert "等待确认或继续的活动 Agent" in prompt_text


@pytest.mark.asyncio
async def test_original_audio_has_no_asr_text_and_uses_same_policy() -> None:
    client = _Client("1")
    output = await _model(client).classify(_request(text=None, media=(_audio(),)))

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert request.messages[1].content[1] == {"type": "audio"}
    assert "CURRENT_USER_TEXT" not in str(request.messages[1].content)
    assert "mcp=唱歌、搜索（含天气）、查交通" in str(request.messages[1].content)
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")


@pytest.mark.asyncio
async def test_prewarm_populates_single_and_multi_turn_prefixes() -> None:
    client = _SequenceClient(["0", "1"])
    await _model(client).prewarm()
    assert len(client.calls) == 2
    single_request = client.calls[0][0]
    multi_request = client.calls[1][0]
    assert single_request.messages[0].content == SINGLE_TURN_SYSTEM_PROMPT
    assert multi_request.messages[0].content == SYSTEM_PROMPT
    assert single_request.sampling.max_new_tokens == 1
    assert multi_request.sampling.max_new_tokens == 1
    assert "需要我查询当前天气吗？" in str(multi_request.messages[1].content)


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
