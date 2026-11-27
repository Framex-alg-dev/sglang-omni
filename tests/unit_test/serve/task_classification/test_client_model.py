from __future__ import annotations

import hashlib

import pytest

from sglang_omni.client.types import CompletionResult
from sglang_omni.serve.agent_prompt_snapshot import decode_router_prompt_snapshot
from sglang_omni.serve.task_classification import client_model as client_model_module
from sglang_omni.serve.task_classification.client_model import (
    SglangClientTaskClassificationModel,
)
from sglang_omni.serve.task_classification.contracts import (
    CONTRACT_VERSION,
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from sglang_omni.utils.framex_service_logs import bind_service_log_context


class _Client:
    def __init__(self, output: str = "0") -> None:
        self.calls = []
        self.output = output

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id, audio_format))
        return CompletionResult(request_id=request_id, text=self.output)


def _model(client: _Client) -> SglangClientTaskClassificationModel:
    return SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="router-v1.0",
    )


def _prompt_snapshot():
    rules = {
        "system_prompt": "AGENT ROUTER SYSTEM",
        "mcp_scope": "MCP SCOPE",
        "multimodal_scope": "MULTIMODAL SCOPE",
    }
    return decode_router_prompt_snapshot(
        {
            "consumer": "sglang_omni.router.v1_0",
            "release_id": "router-release-v1",
            "bundle_sha256": "a" * 64,
            "binding": {
                "assembly_contract": "turn-router-v1.0",
                "output_protocol": "turn-router-code-v1.0",
                "model_id": "agent-logical-router",
                "model_version": "omni-turn-router-v1.0",
            },
            **rules,
            "item_sha256": {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in rules.items()
            },
        }
    )


def _request(
    *,
    text="你好",
    media=(),
    history=(),
    has_active_agent=False,
    pending_confirmation=False,
    follow_up_required=False,
    router_profile="v1.0",
    contract_version=CONTRACT_VERSION,
    prompt_snapshot=_prompt_snapshot(),
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
        router_profile=router_profile,
        contract_version=contract_version,
        prompt_snapshot=prompt_snapshot,
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
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    assert (
        request.messages[1]
        .content[-1]["text"]
        .endswith("[输出]\n只输出 0、1、2 或 3：")
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
    assert request.metadata["model_version"] == "router-v1.0"


@pytest.mark.asyncio
async def test_model_logs_and_archive_use_agent_lineage(monkeypatch) -> None:
    archived = []
    traces = []

    class _Recorder:
        def record(self, **values):
            archived.append(values)

    monkeypatch.setattr(
        client_model_module,
        "get_model_io_recorder",
        lambda _stage: _Recorder(),
    )
    monkeypatch.setattr(
        client_model_module,
        "emit_model_trace",
        lambda stream, event, **fields: traces.append((stream, event, fields)),
    )
    client = _Client("0")
    with bind_service_log_context(
        request_id="trace-request-1",
        root_session_id="root-1",
        agent_session_id="agent-1",
        turn_id="trace-turn-1",
    ):
        assert await _model(client).classify(_request()) == "0"

    assert client.calls[0][1] == "request-1"
    assert archived[0]["context"] == {
        "request_id": "trace-request-1",
        "provider_request_id": "request-1",
        "root_session_id": "root-1",
        "agent_session_id": "agent-1",
        "turn_id": "trace-turn-1",
        "model_id": "shared-base",
        "model_version": "router-v1.0",
        "release_id": "router-release-v1",
        "prompt_snapshot_sha256": "a" * 64,
    }
    assert all(item[2]["request_id"] == "trace-request-1" for item in traces)
    assert all(item[2]["agent_session_id"] == "agent-1" for item in traces)


@pytest.mark.asyncio
async def test_v7_uses_agent_snapshot_and_model_drift_is_advisory(caplog) -> None:
    client = _Client("1")

    with caplog.at_level("WARNING"):
        output = await _model(client).classify(
            _request(
                text="帮我查天气",
                contract_version=CONTRACT_VERSION,
                prompt_snapshot=_prompt_snapshot(),
            )
        )

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    assert "MCP SCOPE" in str(request.messages[1].content)
    assert "Router prompt serving model drift" in caplog.text


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
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    prompt_text = str(request.messages[1].content)
    assert "要播放这首歌吗？" in prompt_text
    assert "第一条" not in prompt_text
    assert "derived ASR" not in prompt_text


@pytest.mark.asyncio
async def test_interrupted_history_reaches_prompt_with_runtime_task_flags() -> None:
    client = _Client("1")
    history = (
        {
            "role": "assistant",
            "text": "要播放",
            "playback_status": "interrupted",
        },
    )
    await _model(client).classify(
        _request(
            text="继续",
            history=history,
            has_active_agent=True,
            pending_confirmation=True,
            follow_up_required=True,
        )
    )

    request = client.calls[0][0]
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    prompt_text = str(request.messages[1].content)
    assert "播放状态=interrupted" in prompt_text
    assert "interrupted" in prompt_text
    assert "存在活动 MCP 任务=是" in prompt_text
    assert "正在等待用户确认=是" in prompt_text
    assert "正在等待用户补充=是" in prompt_text


@pytest.mark.asyncio
async def test_active_task_without_reply_history_uses_single_turn_prompt() -> None:
    client = _Client("1")
    await _model(client).classify(
        _request(
            text="继续",
            has_active_agent=True,
            follow_up_required=True,
        )
    )

    request = client.calls[0][0]
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    prompt_text = str(request.messages[1].content)
    assert "存在活动 MCP 任务=是" in prompt_text
    assert "正在等待用户补充=是" in prompt_text


@pytest.mark.asyncio
async def test_original_audio_has_no_asr_text_and_uses_same_policy() -> None:
    client = _Client("1")
    output = await _model(client).classify(_request(text=None, media=(_audio(),)))

    assert output == "1"
    request = client.calls[0][0]
    assert request.messages[0].content == "AGENT ROUTER SYSTEM"
    assert request.messages[1].content[1] == {"type": "audio"}
    assert "CURRENT_USER_TEXT" not in str(request.messages[1].content)
    assert "MCP=MCP SCOPE" in str(request.messages[1].content)
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")


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
