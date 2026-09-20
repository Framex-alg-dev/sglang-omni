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
    def __init__(self, client, language="zh"):
        self.client = client
        self.model_name = "test-model"
        self.language = language
        self.instructions = "你是一个说话温柔的角色。"
        self.session_id = "session-test"
        self.session_instance_id = "instance-test"
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
        return zh if self.language == "zh" else en


def _turn():
    return SimpleNamespace(
        turn_id="turn-test", request_base="request-test", closed=False,
        text="请站起来", reply_context="Settled backend posture at user commit: seated.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["zh", "en"])
async def test_rejection_reuses_current_audio_and_latest_avatar_not_user_camera(language):
    client = _Client()
    session = _Session(client, language=language)
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
    if language == "en":
        assert "[Action Rejection Response]" in prompt
        assert 'The stance is always "I don\'t feel like it", never "I can\'t"' in prompt
        assert "capability limitations" in prompt
        assert "Never bring up being an AI" in prompt
        assert "The reason must always be a human-like one" in prompt
        assert "neither use it as the excuse nor claim to be human" in prompt
        assert "Do not refuse a physical action" not in prompt
        assert "[动作拒绝回复]" not in prompt
    else:
        assert "[动作拒绝回复]" in prompt
        assert "不是做不到" in prompt or "我做不到" in prompt
        assert "[Action Rejection Response]" not in prompt
        assert "不能把动作预期结束状态当成当前事实" in prompt
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


@pytest.mark.parametrize("language", ["zh", "en"])
def test_independent_speech_and_image_labels_follow_reply_language(language):
    session = _Session(_Client(), language=language)
    turn = _turn()
    turn.intent = SimpleNamespace(
        speech_independent_of_body=True, speech="verbatim", text="Cheese~!"
    )
    request, roles = session._build_action_rejection_request(
        turn, [], ["avatar", "camera"], ["avatar_state", "user_camera"]
    )
    prompt = request.messages[0].content
    parts = request.messages[-1].content
    if language == "en":
        assert "Say the requested verbatim text exactly once" in prompt
        assert parts[0]["text"] == "Character image (not the user's camera):"
        assert any("[Parsed independent language task" in p.get("text", "") for p in parts)
    else:
        assert "将用户要求原样说出的文字准确说一次" in prompt
        assert parts[0]["text"] == "角色图片（不是用户摄像头）："
        assert any("[解析出的独立语言任务" in p.get("text", "") for p in parts)
    assert any("Cheese~!" in p.get("text", "") for p in parts)
    assert request.metadata["images"] == ["avatar"]
    assert roles == ["avatar_state"]
