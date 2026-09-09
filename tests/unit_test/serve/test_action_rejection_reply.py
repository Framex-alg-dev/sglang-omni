from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.serve.realtime.reply import action_rejection
from sglang_omni.serve.realtime.reply.pipeline import ReplyPipeline


class _Client:
    def __init__(self, *, text="我先坐着陪你聊天吧。", error=None, wait=False):
        self.text, self.error, self.wait = text, error, wait
        self.requests = []
        self.aborted = []
        self.started = asyncio.Event()

    async def completion(self, request, *, request_id):
        self.requests.append((request_id, request))
        self.started.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)

    async def abort(self, request_id):
        self.aborted.append(request_id)


class _Session(ReplyPipeline):
    def __init__(self, client):
        self.client = client
        self.model_name = "test-model"
        self.language = "Chinese"
        self.instructions = "你是一个说话温柔的角色。"
        self.session_id = "session-test"
        self.unsupported_action_text = "这个动作暂时做不了。"
        self.requests = set()

    def _ensure_turn_processing(self, turn):
        if turn.closed:
            raise asyncio.CancelledError()

    def _register_turn_request(self, turn, request_id):
        self.requests.add(request_id)

    def _unregister_turn_request(self, turn, request_id):
        self.requests.discard(request_id)

    def _prompt(self, *, zh, en):
        return zh


def _turn():
    return SimpleNamespace(
        turn_id="turn-test", request_base="request-test", closed=False,
        text="请站起来", reply_context="Settled backend posture at user commit: seated.",
    )


@pytest.mark.asyncio
async def test_rejection_reuses_current_audio_and_latest_avatar_not_user_camera():
    client = _Client()
    session = _Session(client)
    text, timing = await session._run_action_rejection_reply(
        _turn(), ["current-audio"], ["old-avatar", "user-camera", "latest-avatar"],
        ["avatar_state", "user_camera", "avatar_state"],
    )
    request_id, request = client.requests[0]
    assert request_id == "request-test-action-rejection"
    assert request.metadata["images"] == ["latest-avatar"]
    assert request.metadata["audios"] == ["current-audio"]
    assert request.output_modalities == ["text"]
    assert text == client.text
    assert timing["forwarded_image_roles"] == ["avatar_state"]
    assert timing["fallback_reason"] is None
    prompt = request.messages[0].content
    assert session.instructions in prompt
    assert "[Action rejection reply]" in prompt
    assert "Do not refuse a physical action" not in prompt
    parts = request.messages[-1].content
    assert {"type": "text", "text": "请站起来"} in parts
    assert sum(p["type"] == "image" for p in parts) == 1
    assert any("Settled backend posture" in p.get("text", "") for p in parts)
    assert not session.requests
    assert not client.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["empty", "error", "timeout"])
async def test_generation_failure_returns_exact_configured_fallback_and_cleans_up(cause, monkeypatch):
    monkeypatch.setattr(action_rejection, "REJECTION_TIMEOUT_S", 0.02)
    client = _Client(text="" if cause == "empty" else "unused",
                     error=RuntimeError("failed") if cause == "error" else None,
                     wait=cause == "timeout")
    session = _Session(client)
    turn = _turn()
    turn.text = None  # Audio-only input does not require a local transcript.
    text, timing = await session._run_action_rejection_reply(turn, ["audio"], [], [])
    assert text == session.unsupported_action_text
    assert timing["fallback_reason"]
    assert client.aborted == ["request-test-action-rejection"]
    assert not session.requests


@pytest.mark.asyncio
async def test_interrupt_cancels_remote_request_without_returning_fallback():
    client = _Client(wait=True)
    session = _Session(client)
    task = asyncio.create_task(session._run_action_rejection_reply(_turn(), [], [], []))
    await client.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.aborted == ["request-test-action-rejection"]
    assert not session.requests


@pytest.mark.asyncio
async def test_stream_is_buffered_and_closed_on_interrupt():
    closed = asyncio.Event()
    client = _Client()

    async def stream(request, *, request_id):
        try:
            yield SimpleNamespace(modality="text", text="未完成的句子")
            client.started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    client.completion_stream = stream
    session = _Session(client)
    task = asyncio.create_task(session._run_action_rejection_reply(_turn(), [], [], []))
    await client.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert client.aborted == ["request-test-action-rejection"]
    assert not session.requests
