from __future__ import annotations

import asyncio
import base64
import json

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionResult
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
from sglang_omni.serve.realtime.multimodal import (
    ACTION_CATEGORY_TOP_K_ENV,
    ACTION_MICRO_BATCH_SIZE_ENV,
    MultimodalSession,
    MultimodalSessionManager,
)


class FakeWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.close_calls = 0

    async def send_text(self, value: str) -> None:
        self.events.append(json.loads(value))

    async def close(self) -> None:
        self.close_calls += 1
        self.client_state = WebSocketState.DISCONNECTED
        self.application_state = WebSocketState.DISCONNECTED


class DisconnectingFakeWebSocket(FakeWebSocket):
    async def receive(self) -> dict:
        self.client_state = WebSocketState.DISCONNECTED
        return {"type": "websocket.disconnect"}


class FakeClient:
    def __init__(self, *, select_none: bool = False) -> None:
        self.chat_requests = []
        self.score_requests = []
        self.select_none = select_none

    async def completion(self, request, *, request_id: str) -> CompletionResult:
        self.chat_requests.append(request)
        return CompletionResult(request_id=request_id, text="好的，我来看看。")

    async def score_action_suffixes(
        self, request
    ) -> ActionSuffixScoreResult:
        self.score_requests.append(request)
        top_score = -1.0 if self.select_none else -0.1
        other_score = -0.1 if self.select_none else -1.0
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id="a01",
                    token_count=1,
                    mean_logprob=top_score,
                    mean_nll=-top_score,
                    ppl=2.718281 if self.select_none else 1.105170,
                    token_scores=[TokenScore(token_id=101, logprob=top_score)],
                ),
                CandidateScore(
                    candidate_id="none",
                    token_count=1,
                    mean_logprob=other_score,
                    mean_nll=-other_score,
                    ppl=1.105170 if self.select_none else 2.718281,
                    token_scores=[TokenScore(token_id=102, logprob=other_score)],
                ),
            ],
        )


class BlockingActionClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.started_requests = []
        self.aborted: list[str] = []

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        self.started_requests.append(request)
        if len(self.started_requests) == 1:
            self.started.set()
            await self.release.wait()
        return await super().score_action_suffixes(request)

    async def abort(self, request_id: str):
        self.aborted.append(request_id)
        self.release.set()
        return None


class PrefillFakeClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_requests = []

    async def prefill_action_catalog(self, **kwargs):
        self.prefill_requests.append(kwargs)
        return True


class FailingOnceClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        if self.fail:
            self.fail = False
            raise RuntimeError(
                "action scoring prefix selected-token logprobs are missing"
            )
        return await super().score_action_suffixes(request)


def make_session(
    ws: FakeWebSocket,
    client: FakeClient,
    action_selection_mode: str | None = None,
    action_category_top_k: int | None = None,
) -> MultimodalSession:
    claimed = {}

    def claim(session_id: str, session: MultimodalSession) -> None:
        if session_id in claimed:
            raise ValueError("duplicate session")
        claimed[session_id] = session

    def release(session_id: str, session: MultimodalSession) -> None:
        if claimed.get(session_id) is session:
            del claimed[session_id]

    return MultimodalSession(
        ws,
        client=client,
        model_name="Qwen3-Omni-30B-A3B-Instruct",
        action_selection_mode=action_selection_mode,
        action_category_top_k=action_category_top_k,
        claim_session=claim,
        release_session=release,
    )


def user_turn_start(turn_id: str | None) -> dict:
    event = {
        "type": "turn.start",
        "turn_origin": "user",
        "text_role": "user_input",
    }
    if turn_id is not None:
        event["turn_id"] = turn_id
    return event


def user_turn_commit(turn_id: str, **fields) -> dict:
    return {
        "type": "turn.commit",
        "turn_id": turn_id,
        "turn_origin": "user",
        "text_role": "user_input",
        **fields,
    }


@pytest.mark.asyncio
async def test_cleanup_does_not_send_duplicate_close_frame() -> None:
    ws = FakeWebSocket()
    ws.application_state = WebSocketState.DISCONNECTED
    session = make_session(ws, FakeClient())

    await session._close_websocket()

    assert ws.close_calls == 0


@pytest.mark.asyncio
async def test_send_ignores_already_closed_websocket() -> None:
    ws = FakeWebSocket()
    ws.application_state = WebSocketState.DISCONNECTED
    session = make_session(ws, FakeClient())

    await session.send({"type": "turn.result"})

    assert ws.events == []


@pytest.mark.asyncio
async def test_manual_turn_collects_multiple_audio_and_images() -> None:
    ws = FakeWebSocket()
    client = FakeClient()
    session = make_session(ws, client)

    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-1",
            "include_scores": True,
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave_left",
                    "source_label": "左手挥手",
                    "short_definition": "使用左手抬起并左右摆动",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-1"))

    pcm = base64.b64encode(b"\x00\x00" * 160).decode()
    await session.handle_audio_append(
        {
            "type": "input_audio.append",
            "turn_id": "turn-1",
            "seq": 1,
            "audio": pcm,
        }
    )
    await session.handle_audio_append(
        {
            "type": "input_audio_buffer.append",
            "turn_id": "turn-1",
            "seq": 2,
            "audio": pcm,
        }
    )
    image = base64.b64encode(b"image-bytes").decode()
    for seq, timestamp, image_role in (
        (1, 2000, "avatar_state"),
        (2, 1000, "user_camera"),
    ):
        await session.handle_image_append(
            {
                "type": "input_image.append",
                "turn_id": "turn-1",
                "seq": seq,
                "timestamp_ms": timestamp,
                "image_role": image_role,
                "mime_type": "image/jpeg",
                "image": image,
            }
        )

    await session.handle_turn_commit(
        user_turn_commit("turn-1", avatar_state={"pose": "seated"})
    )

    assert len(client.chat_requests) == 0
    assert len(client.score_requests) == 1
    request = client.score_requests[0]
    assert len(request.audios) == 1
    assert len(request.images) == 2
    assert request.history == []
    assert "图片1=用户摄像头画面" in request.prefix
    assert "图片2=数字人当前状态画面" in request.prefix
    assert "不得混淆两类图片" in request.prefix
    assert "只有本轮标记为 avatar_state 的图片" in request.prefix
    assert request.candidates[0].suffix == "a01"
    assert "a01=wave_left" in request.system_prompt
    assert "none=no_action" in request.system_prompt

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "wave_left"
    assert result["action"]["execute"] is True
    assert result["media_summary"]["audio_chunk_count"] == 2
    assert result["media_summary"]["image_frame_count"] == 2
    assert result["media_summary"]["user_camera_image_count"] == 1
    assert result["media_summary"]["avatar_state_image_count"] == 1
    assert result["timing"]["server_action_compute_ms"] >= 0
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"
    assert "action_id=wave_left" in session.history[1]["content"]

    image_acks = [
        event
        for event in ws.events
        if event["type"] == "input.ack" and event["media_type"] == "image"
    ]
    assert [event["image_role"] for event in image_acks] == [
        "avatar_state",
        "user_camera",
    ]

    await session.handle_turn_start(user_turn_start("turn-2"))
    await session.handle_turn_commit(user_turn_commit("turn-2"))
    second_request = client.score_requests[1]
    historical_parts = second_request.history[0]["content"]
    historical_labels = [
        part["text"]
        for part in historical_parts
        if part.get("type") == "text" and part["text"].startswith("[image_role]")
    ]
    assert historical_labels == [
        "[image_role] 用户摄像头画面（用于观察用户及其环境）：",
    ]
    assert len(second_request.history_images) == 1
    assert "数字人当前视觉姿态未知" in second_request.prefix
    assert "不得根据历史图片推断" in second_request.prefix
    second_result = next(
        event
        for event in ws.events
        if event["type"] == "turn.result" and event["turn_id"] == "turn-2"
    )
    assert (
        second_result["media_summary"]["action_context"][
            "ignored_history_avatar_image_count"
        ]
        == 1
    )


@pytest.mark.asyncio
async def test_image_role_defaults_by_turn_origin_and_rejects_unknown_role() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-image-role",
            "action_candidates": [
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                }
            ],
        }
    )
    image = base64.b64encode(b"image-bytes").decode()

    await session.handle_turn_start(user_turn_start("user-image"))
    await session.handle_image_append(
        {
            "type": "input_image.append",
            "turn_id": "user-image",
            "seq": 1,
            "image": image,
        }
    )
    assert session.active_turn.images[0].image_role == "user_camera"
    with pytest.raises(ValueError, match="image_role must be"):
        await session.handle_image_append(
            {
                "type": "input_image.append",
                "turn_id": "user-image",
                "seq": 2,
                "image_role": "unknown",
                "image": image,
            }
        )
    await session.handle_turn_cancel(
        {"type": "turn.cancel", "turn_id": "user-image"}
    )

    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-image",
            "turn_origin": "proactive",
            "text_role": "character_reply",
        }
    )
    await session.handle_image_append(
        {
            "type": "input_image.append",
            "turn_id": "proactive-image",
            "seq": 1,
            "image": image,
        }
    )
    assert session.active_turn.images[0].image_role == "avatar_state"


@pytest.mark.asyncio
async def test_action_score_failure_returns_turn_error_and_session_recovers() -> None:
    ws = FakeWebSocket()
    client = FailingOnceClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-score-error",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave_left",
                    "source_label": "左手挥手",
                    "short_definition": "左手挥手",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                },
            ],
        }
    )

    await session.handle_turn_start(user_turn_start("turn-failed"))
    await session.handle_turn_commit(user_turn_commit("turn-failed"))

    error = next(event for event in ws.events if event["type"] == "error")
    assert error["error"]["type"] == "action_score_error"
    assert error["error"]["code"] == "action_score_logprob_unavailable"
    assert error["session_id"] == "session-score-error"
    assert error["turn_id"] == "turn-failed"
    assert session.active_turn is None

    await session.handle_turn_start(user_turn_start("turn-recovered"))
    await session.handle_turn_commit(user_turn_commit("turn-recovered"))
    result = next(
        event
        for event in ws.events
        if event["type"] == "turn.result" and event["turn_id"] == "turn-recovered"
    )
    assert result["action"]["action_id"] == "wave_left"


@pytest.mark.asyncio
async def test_current_turn_is_not_sent_as_completed_assistant_history() -> None:
    ws = FakeWebSocket()
    client = FakeClient()
    session = make_session(ws, client)
    candidates = [
        {
            "candidate_id": "a01",
            "action_id": "wave_left",
            "source_label": "左手挥手",
            "short_definition": "左手挥手",
        },
        {
            "candidate_id": "none",
            "action_id": "no_action",
            "source_label": "不做动作",
            "short_definition": "不做动作",
        },
    ]
    await session.handle_session_start(
        {"type": "session.start", "session_id": "session-2", "action_candidates": candidates}
    )

    for turn_id in ("turn-1", "turn-2"):
        await session.handle_turn_start(user_turn_start(turn_id))
        await session.handle_turn_commit(user_turn_commit(turn_id))

    assert len(client.score_requests) == 2
    second_request = client.score_requests[1]
    assert [item["role"] for item in second_request.history] == [
        "user",
        "assistant",
    ]
    assert "action_id=wave_left" in second_request.history[1]["content"]
    assert "动作名称=左手挥手" in second_request.history[1]["content"]
    assert second_request.system_prompt == client.score_requests[0].system_prompt

@pytest.mark.asyncio
async def test_committed_turn_can_be_aborted_and_next_turn_runs() -> None:
    ws = FakeWebSocket()
    client = BlockingActionClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-cancel-committed",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "抬手挥手",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前状态",
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("low-priority"))
    await session.dispatch(
        user_turn_commit(
            "low-priority",
            text="稍后向用户挥手",
            avatar_state={
                "current_action_id": None,
                "state_description": "低优先级欢迎动作",
            },
        )
    )
    await asyncio.wait_for(client.started.wait(), timeout=1)

    turn = session.active_turn
    assert turn is not None
    assert turn.phase == "processing"
    assert turn.current_request_id is not None
    assert turn.current_request_id.endswith("-single")
    with pytest.raises(ValueError, match="already committed"):
        await session.handle_text_update(
            {
                "type": "turn.text.update",
                "turn_id": "low-priority",
                "text": "不能再修改",
            }
        )

    request_id = turn.current_request_id
    with pytest.raises(ValueError, match="already committed"):
        await session.handle_audio_append(
            {"type": "input_audio.append", "turn_id": "low-priority", "seq": 1}
        )
    with pytest.raises(ValueError, match="already committed"):
        await session.handle_image_append(
            {"type": "input_image.append", "turn_id": "low-priority", "seq": 1}
        )
    with pytest.raises(ValueError, match="already committed"):
        await session.handle_turn_commit(
            user_turn_commit("low-priority", text="不能重复提交")
        )

    await session.handle_turn_cancel(
        {"type": "turn.cancel", "turn_id": "low-priority"}
    )

    assert client.aborted == [request_id]
    assert session.active_turn is None
    assert session.history == []
    assert session.history_turns == []
    assert session.last_avatar_state == {}
    assert not any(event["type"] == "turn.result" for event in ws.events)
    assert not any(event["type"] == "error" for event in ws.events)
    assert ws.events[-1] == {
        "type": "turn.cancelled",
        "session_id": "session-cancel-committed",
        "turn_id": "low-priority",
    }

    await session.handle_turn_start(user_turn_start("high-priority"))
    await session.handle_turn_commit(
        user_turn_commit("high-priority", text="现在挥手")
    )
    assert ws.events[-1]["type"] == "turn.result"
    assert ws.events[-1]["turn_id"] == "high-priority"
    assert len(session.history_turns) == 1


@pytest.mark.asyncio
async def test_session_close_aborts_committed_turn() -> None:
    client = BlockingActionClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-close-abort",
            "action_candidates": [{"candidate_id": "none", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前状态"}],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-close"))
    await session.dispatch(user_turn_commit("turn-close"))
    await asyncio.wait_for(client.started.wait(), timeout=1)
    request_id = session.active_turn.current_request_id

    await session.handle_session_close({"type": "session.close"})

    assert client.aborted == [request_id]
    assert session.active_turn is None
    assert session.closed is True
    assert session.websocket.events[-1]["type"] == "session.closed"


@pytest.mark.asyncio
async def test_websocket_disconnect_aborts_committed_turn() -> None:
    ws = DisconnectingFakeWebSocket()
    client = BlockingActionClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-disconnect-abort",
            "action_candidates": [
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前状态",
                }
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-disconnect"))
    await session.dispatch(user_turn_commit("turn-disconnect"))
    await asyncio.wait_for(client.started.wait(), timeout=1)
    request_id = session.active_turn.current_request_id

    await session.run()

    assert client.aborted == [request_id]
    assert session.active_turn is None
    assert session.closed is True
    assert ws.close_calls == 0


@pytest.mark.asyncio
async def test_turn_id_is_required_and_cannot_be_reused() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-required-id",
            "action_candidates": [
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="generated by the caller"):
        await session.handle_turn_start(user_turn_start(None))
    await session.handle_turn_start(user_turn_start("turn-1"))
    await session.handle_turn_cancel({"type": "turn.cancel", "turn_id": "turn-1"})
    with pytest.raises(ValueError, match="already been used"):


        await session.handle_turn_start(user_turn_start("turn-1"))
@pytest.mark.asyncio
async def test_turn_semantics_are_required_and_must_match_commit() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-turn-semantics",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "抬手向用户挥手",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前状态",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="turn_origin"):
        await session.handle_turn_start({"type": "turn.start", "turn_id": "missing"})
    with pytest.raises(ValueError, match="text_role"):
        await session.handle_turn_start(
            {
                "type": "turn.start",
                "turn_id": "invalid-pair",
                "turn_origin": "proactive",
                "text_role": "user_input",
            }
        )

    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-1",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
        }
    )
    with pytest.raises(ValueError, match="must match turn.start"):
        await session.handle_turn_commit(
            {
                "type": "turn.commit",
                "turn_id": "proactive-1",
                "turn_origin": "proactive",
                "text_role": "character_reply",
                "trigger": "different_trigger",
                "text": "Hello，你回来啦！",
            }
        )
    assert session.active_turn is not None

    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "proactive-1",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
            "text": "Hello，你回来啦！",
            "user_input": None,
        }
    )
    assert session.active_turn is None
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-missing-text",
            "turn_origin": "proactive",
            "text_role": "character_reply",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "proactive-missing-text",
            "turn_origin": "proactive",
            "text_role": "character_reply",
        }
    )
    assert session.active_turn is None
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-invalid-user-input",
            "turn_origin": "proactive",
            "text_role": "character_reply",
        }
    )
    with pytest.raises(ValueError, match="user_input must be null"):
        await session.handle_turn_commit(
            {
                "type": "turn.commit",
                "turn_id": "proactive-invalid-user-input",
                "turn_origin": "proactive",
                "text_role": "character_reply",
                "text": "欢迎回来",
                "user_input": "不应使用",
            }
        )
    await session.handle_turn_cancel(
        {"type": "turn.cancel", "turn_id": "proactive-invalid-user-input"}
    )
    assert (
        session._classify_error(
            {"type": "turn.commit"}, ValueError("text_role does not match")
        )
        == "invalid_turn_semantics"
    )


@pytest.mark.asyncio
async def test_proactive_without_text_uses_state_only_context() -> None:
    client = FakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-proactive-without-text",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "友好地挥手",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前状态",
                },
            ],
        }
    )
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-without-text",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "action_finished",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "proactive-without-text",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "action_finished",
            "avatar_state": {
                "current_action_id": "wave",
                "state_description": "上一动作已结束，保持当前状态。",
            },
        }
    )

    request = client.score_requests[0]
    assert request.history == []
    assert "未提供本轮待播文本" in request.prefix
    assert "不依赖当前语言文本" in request.prefix
    assert "上一条 assistant 消息是数字人已经准备好" not in request.prefix
    assert session.history[0]["role"] == "assistant"
    assert session.history[0]["content"].startswith("[action_state]")
    assert "None" not in session.history[0]["content"]


@pytest.mark.asyncio
async def test_proactive_text_is_assistant_context_and_persists_for_next_turn() -> None:
    client = FakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-proactive-context",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "友好地抬手挥手欢迎用户",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前状态",
                },
            ],
        }
    )

    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "proactive-welcome",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "proactive-welcome",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
            "text": "Hello，你回来啦！",
            "avatar_state": {
                "current_action_id": None,
                "state_description": "用户刚刚回来，动作应轻量友好。",
                "pose": "seated",
                "conversation_phase": "greeting",
            },
        }
    )

    proactive_request = client.score_requests[0]
    assert proactive_request.turn_origin == "proactive"
    assert proactive_request.text_role == "character_reply"
    assert proactive_request.trigger == "user_returned"
    assert proactive_request.history[-1] == {
        "role": "assistant",
        "content": "Hello，你回来啦！",
    }
    assert "上一条 assistant 消息" in proactive_request.prefix
    assert "上一条 assistant 消息与 avatar_state 是动作选择的共同核心约束" in proactive_request.prefix
    assert "待播文本的语义、语气和表达目标直接相关" in proactive_request.prefix
    assert "state_description 描述的场景目标、动作要求和禁止项" in proactive_request.prefix
    assert "current_action_id 表示当前或刚结束的动作" in proactive_request.prefix
    assert "没有同时满足文本和状态约束的动作时选择 no_action" in proactive_request.prefix
    assert proactive_request.prefix.count("Hello，你回来啦！") == 0
    assert "不要生成新的回复" in proactive_request.prefix
    assert "当前用户文本" not in proactive_request.prefix
    assert proactive_request.avatar_state["pose"] == "seated"
    assert proactive_request.avatar_state["conversation_phase"] == "greeting"

    assert len(session.history_turns) == 1
    assert session.history_turns[0].turn_origin == "proactive"
    assert [message["role"] for message in session.history] == ["assistant"]
    assert "Hello，你回来啦！" in session.history[0]["content"]
    assert "[action_state]" in session.history[0]["content"]
    assert "action_id=wave" in session.history[0]["content"]

    await session.handle_turn_start(user_turn_start("user-after-proactive"))
    await session.handle_turn_commit(
        user_turn_commit("user-after-proactive", text="我们继续聊吧")
    )
    next_request = client.score_requests[1]
    assert len(next_request.history) == 1
    assert next_request.history[0]["role"] == "assistant"
    assert "Hello，你回来啦！" in next_request.history[0]["content"]
    assert "[action_state]" in next_request.history[0]["content"]
    assert "本轮来源是 user" in next_request.prefix


@pytest.mark.asyncio
async def test_audio_sequence_is_strict_and_duplicate_is_idempotent() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-seq",
            "action_candidates": [
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                }
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-seq"))
    pcm = base64.b64encode(b"\x00\x00" * 8).decode()
    event = {
        "type": "input_audio.append",
        "turn_id": "turn-seq",
        "seq": 1,
        "audio": pcm,
    }
    await session.handle_audio_append(event)
    await session.handle_audio_append(event)
    assert session.active_turn is not None
    assert session.active_turn.audio_chunk_count == 1
    assert session.active_turn.duplicate_audio_chunks == 1
    with pytest.raises(ValueError, match="expected 2, got 3"):
        await session.handle_audio_append({**event, "seq": 3})


@pytest.mark.asyncio
async def test_candidate_list_is_fixed_and_no_action_returns_execute_false() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient(select_none=True))
    candidates = [
        {
            "candidate_id": "a01",
            "action_id": "wave_left",
            "source_label": "左手挥手",
            "short_definition": "左手挥手",
        },
        {
            "candidate_id": "none",
            "action_id": "no_action",
            "source_label": "不做动作",
            "short_definition": "保持当前姿态",
        },
    ]
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-fixed-catalog",
            "include_scores": True,
            "action_candidates": candidates,
        }
    )
    with pytest.raises(ValueError, match="only be sent once"):
        await session.handle_session_start(
            {
                "type": "session.start",
                "session_id": "session-fixed-catalog",
                "action_candidates": [candidates[1]],
            }
        )
    await session.handle_turn_start(user_turn_start("turn-none"))
    await session.handle_turn_commit(user_turn_commit("turn-none"))
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "no_action"
    assert result["action"]["execute"] is False
    assert result["media_summary"]["received_image_count"] == 0
    assert result["media_summary"]["scored_image_count"] == 0



class NestedFakeClient(FakeClient):
    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        self.score_requests.append(request)
        ids = [item.candidate_id for item in request.candidates]
        selected = ids[0]
        scores = []
        for index, candidate_id in enumerate(ids):
            value = -0.1 if candidate_id == selected else -1.0 - index
            scores.append(CandidateScore(candidate_id=candidate_id, token_count=1, mean_logprob=value, mean_nll=-value, ppl=2.718281 if value == -1.0 else 1.105170, token_scores=[TokenScore(token_id=100 + index, logprob=value)]))
        return ActionSuffixScoreResult(request_id=request.request_id, model=request.model, prefix_cached=True, scores=scores)


class NestedPrefillFakeClient(NestedFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_requests = []

    async def prefill_action_catalog(self, **kwargs):
        self.prefill_requests.append(kwargs)
        return True


class BlockingNestedPrefillClient(NestedFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_requests = []
        self.child_prefill_started = asyncio.Event()
        self.release_child_prefill = asyncio.Event()
        self.aborted: list[str] = []

    async def prefill_action_catalog(self, **kwargs):
        self.prefill_requests.append(kwargs)
        if kwargs["stage"] == "child":
            self.child_prefill_started.set()
            await self.release_child_prefill.wait()
        return True

    async def abort(self, request_id: str):
        self.aborted.append(request_id)
        self.release_child_prefill.set()
        return None


@pytest.mark.asyncio
async def test_hierarchical_child_prefill_can_be_aborted() -> None:
    ws = FakeWebSocket()
    client = BlockingNestedPrefillClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-cancel-child-prefill",
            "action_candidates": [
                {
                    "category_id": "B1",
                    "source_label": "问候",
                    "short_definition": "问候动作",
                    "children": [
                        {
                            "candidate_id": "A1",
                            "action_id": "wave",
                            "source_label": "挥手",
                            "short_definition": "挥手问候",
                        },
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前状态",
                        },
                    ],
                }
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-child-prefill"))
    await session.dispatch(
        user_turn_commit("turn-child-prefill", text="向用户问好")
    )
    await asyncio.wait_for(client.child_prefill_started.wait(), timeout=1)

    request_id = session.active_turn.current_request_id
    assert request_id.endswith("-child-prefill")
    await session.handle_turn_cancel(
        {"type": "turn.cancel", "turn_id": "turn-child-prefill"}
    )

    assert client.aborted == [request_id]
    assert len(client.score_requests) == 1
    assert client.score_requests[0].stage == "category"
    assert session.active_turn is None
    assert session.history_turns == []
    assert [event["type"] for event in ws.events[-2:]] == [
        "turn.committed",
        "turn.cancelled",
    ]


@pytest.mark.asyncio
async def test_nested_catalog_runs_two_stages_in_one_turn() -> None:
    ws = FakeWebSocket()
    client = NestedFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start({
        "type": "session.start", "session_id": "session-nested",
        "include_scores": True,
        "action_candidates": [
            {"category_id": "B1", "source_label": "基础姿态", "short_definition": "姿态变化", "children": [
                {"candidate_id": "A1", "action_id": "A1", "source_label": "正式站立", "short_definition": "站立"},
                {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"},
            ]},
            {"category_id": "B2", "source_label": "重心变化", "short_definition": "重心变化", "children": [
                {"candidate_id": "A15", "action_id": "A15", "source_label": "重心左移", "short_definition": "左移"},
            ]},
        ],
    })
    await session.handle_turn_start(user_turn_start("turn-nested"))
    await session.handle_turn_commit(user_turn_commit("turn-nested", text="请站起来"))
    assert len(client.score_requests) == 2
    category_request, child_request = client.score_requests
    assert [item.candidate_id for item in category_request.candidates] == ["B1", "B2"]
    assert [item.candidate_id for item in child_request.candidates] == ["A1", "A0"]
    assert category_request.suffix_tokenization_mode == "short_id"
    assert child_request.suffix_tokenization_mode == "short_id"
    assert category_request.action_context_cache_key == child_request.action_context_cache_key
    assert category_request.action_context_cache_key == category_request.logical_request_id
    assert category_request.micro_batch_size == 64
    assert child_request.micro_batch_size == 64
    assert "B1=基础姿态；姿态变化" in category_request.system_prompt
    assert "B2=重心变化；重心变化" in category_request.system_prompt
    assert "A1=A1" in child_request.system_prompt
    assert "B2=重心变化" not in child_request.system_prompt
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "A1"
    assert result["media_summary"]["action_context"]["selection_stages"] == 2
    assert result["media_summary"]["action_context"]["selected_category_id"] == "B1"
    assert result["media_summary"]["action_context"]["selected_category_ids"] == ["B1"]
    assert result["media_summary"]["action_context"]["category_top_k"] == 1

@pytest.mark.asyncio
async def test_hierarchical_proactive_turn_uses_character_reply_in_both_stages() -> None:
    client = NestedFakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-nested-proactive",
            "action_candidates": [
                {
                    "category_id": "B1",
                    "source_label": "问候",
                    "short_definition": "欢迎或告别用户",
                    "children": [
                        {
                            "candidate_id": "A1",
                            "action_id": "wave",
                            "source_label": "挥手",
                            "short_definition": "友好地挥手欢迎用户",
                        },
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前状态",
                        },
                    ],
                }
            ],
        }
    )
    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "nested-proactive",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
        }
    )
    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "nested-proactive",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "user_returned",
            "text": "Hello，你回来啦！",
        }
    )

    assert len(client.score_requests) == 2
    category_request, child_request = client.score_requests
    for request in (category_request, child_request):
        assert request.turn_origin == "proactive"
        assert request.text_role == "character_reply"
        assert request.trigger == "user_returned"
        assert request.history[-1] == {
            "role": "assistant",
            "content": "Hello，你回来啦！",
        }
        assert "上一条 assistant 消息与 avatar_state 是动作选择的共同核心约束" in request.prefix
        assert "待播文本的语义、语气和表达目标直接相关" in request.prefix
        assert "state_description 描述的场景目标、动作要求和禁止项" in request.prefix
        assert "current_action_id 表示当前或刚结束的动作" in request.prefix
        assert "没有同时满足文本和状态约束的动作时选择 no_action" in request.prefix
        assert request.prefix.count("Hello，你回来啦！") == 0
    assert session.history_turns[-1].turn_origin == "proactive"
    assert session.history[-1]["role"] == "assistant"
    assert "action_id=wave" in session.history[-1]["content"]

@pytest.mark.asyncio
async def test_hierarchical_child_prefix_is_lazily_prefilled_once_per_namespace() -> None:
    ws = FakeWebSocket()
    client = NestedPrefillFakeClient()
    session = make_session(ws, client)
    candidates = [
        {
            "category_id": "B1",
            "source_label": "基础姿态",
            "short_definition": "姿态变化",
            "children": [
                {"candidate_id": "A1", "action_id": "A1", "source_label": "站立", "short_definition": "站立"},
                {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"},
            ],
        },
        {
            "category_id": "B2",
            "source_label": "情绪",
            "short_definition": "情绪变化",
            "children": [
                {"candidate_id": "A2", "action_id": "A2", "source_label": "微笑", "short_definition": "微笑"},
            ],
        },
    ]
    await session.handle_session_start({
        "type": "session.start",
        "session_id": "session-lazy-child-prefill",
        "include_scores": True,
        "action_candidates": candidates,
    })
    assert [item["stage"] for item in client.prefill_requests] == ["category"]

    await session.handle_turn_start(user_turn_start("turn-1"))
    await session.handle_turn_commit(user_turn_commit("turn-1", text="请站起来"))
    assert [item["stage"] for item in client.prefill_requests] == ["category", "child"]
    await session.handle_turn_start(user_turn_start("turn-2"))
    await session.handle_turn_commit(user_turn_commit("turn-2", text="再来一次"))
    assert [item["stage"] for item in client.prefill_requests] == ["category", "child"]
    results = [
        event for event in ws.events
        if event["type"] == "turn.result"
    ]
    assert results[-2]["media_summary"]["action_context"]["child_prefix_prefilled"] is True
    assert results[-1]["media_summary"]["action_context"]["child_prefix_prefilled"] is False


@pytest.mark.asyncio
async def test_hierarchical_top_k_scores_children_from_multiple_categories() -> None:
    ws = FakeWebSocket()
    client = NestedFakeClient()
    session = make_session(ws, client, action_category_top_k=2)
    await session.handle_session_start({
        "type": "session.start", "session_id": "session-nested-top-k",
        "include_scores": True,
        "action_candidates": [
            {"category_id": "B1", "source_label": "基础姿态", "short_definition": "姿态变化", "children": [
                {"candidate_id": "A1", "action_id": "A1", "source_label": "正式站立", "short_definition": "站立"},
                {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"},
            ]},
            {"category_id": "B2", "source_label": "重心变化", "short_definition": "重心变化", "children": [
                {"candidate_id": "A15", "action_id": "A15", "source_label": "重心左移", "short_definition": "左移"},
            ]},
        ],
    })
    await session.handle_turn_start(user_turn_start("turn-nested-top-k"))
    await session.handle_turn_commit(user_turn_commit("turn-nested-top-k", text="请站起来"))

    assert len(client.score_requests) == 2
    category_request, child_request = client.score_requests
    assert [item.candidate_id for item in category_request.candidates] == ["B1", "B2"]
    assert [item.candidate_id for item in child_request.candidates] == ["A1", "A0", "A15"]
    assert "B1=基础姿态；姿态变化" in child_request.system_prompt
    assert "B2=重心变化；重心变化" in child_request.system_prompt
    assert "A15=A15" in child_request.system_prompt
    context = next(event for event in ws.events if event["type"] == "turn.result")["media_summary"]["action_context"]
    assert context["selected_category_id"] == "B1"
    assert context["selected_category_ids"] == ["B1", "B2"]
    assert context["category_top_k"] == 2


def test_action_micro_batch_size_reads_environment_and_is_fixed_on_manager(monkeypatch) -> None:
    monkeypatch.setenv(ACTION_MICRO_BATCH_SIZE_ENV, "128")
    manager = MultimodalSessionManager(
        client=FakeClient(),
        model_name="Qwen3-Omni-30B-A3B-Instruct",
    )
    assert manager.action_micro_batch_size == 128
    session = manager.create(FakeWebSocket())
    assert session.action_micro_batch_size == 128


@pytest.mark.parametrize("value", ["0", "257", "not-an-int"])
def test_invalid_action_micro_batch_size_fails_manager_startup(monkeypatch, value) -> None:
    monkeypatch.setenv(ACTION_MICRO_BATCH_SIZE_ENV, value)
    with pytest.raises(ValueError, match=ACTION_MICRO_BATCH_SIZE_ENV):
        MultimodalSessionManager(
            client=FakeClient(),
            model_name="Qwen3-Omni-30B-A3B-Instruct",
        )

def test_action_category_top_k_reads_environment_and_is_fixed_on_manager(monkeypatch) -> None:
    monkeypatch.setenv(ACTION_CATEGORY_TOP_K_ENV, "2")
    manager = MultimodalSessionManager(
        client=FakeClient(),
        model_name="Qwen3-Omni-30B-A3B-Instruct",
    )
    assert manager.action_category_top_k == 2
    assert manager.create(FakeWebSocket()).action_category_top_k == 2


@pytest.mark.parametrize("value", ["0", "4", "not-an-int"])
def test_invalid_action_category_top_k_fails_manager_startup(monkeypatch, value) -> None:
    monkeypatch.setenv(ACTION_CATEGORY_TOP_K_ENV, value)
    with pytest.raises(ValueError, match=ACTION_CATEGORY_TOP_K_ENV):
        MultimodalSessionManager(
            client=FakeClient(),
            model_name="Qwen3-Omni-30B-A3B-Instruct",
        )


def test_action_selection_mode_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ACTION_SELECTION_MODE", "flat_children")
    manager = MultimodalSessionManager(
        client=FakeClient(),
        model_name="Qwen3-Omni-30B-A3B-Instruct",
    )
    assert manager.action_selection_mode == "flat_children"


@pytest.mark.asyncio
async def test_session_start_invalid_mode_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ACTION_SELECTION_MODE", "flat_children")
    ws = FakeWebSocket()
    session = make_session(ws, NestedFakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-invalid-mode",
            "selection_mode": "not-a-mode",
            "action_candidates": [
                {
                    "category_id": "B1",
                    "source_label": "基础姿态",
                    "short_definition": "姿态变化",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前姿态",
                        }
                    ],
                }
            ],
        }
    )
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_selection_mode"] == "flat_children"
    assert started["action_selection_stages"] == 1


@pytest.mark.asyncio
async def test_session_start_mode_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ACTION_SELECTION_MODE", "hierarchical")
    ws = FakeWebSocket()
    session = make_session(ws, NestedFakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-mode-override",
            "selection_mode": "flat_children",
            "action_candidates": [
                {
                    "category_id": "B1",
                    "source_label": "基础姿态",
                    "short_definition": "姿态变化",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前姿态",
                        }
                    ],
                }
            ],
        }
    )
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_selection_mode"] == "flat_children"
    assert started["action_selection_stages"] == 1


@pytest.mark.asyncio
async def test_nested_catalog_flat_children_runs_one_stage() -> None:
    ws = FakeWebSocket()
    client = NestedFakeClient()
    session = make_session(ws, client, action_selection_mode="flat_children")
    await session.handle_session_start({
        "type": "session.start",
        "session_id": "session-flat-children",
        "include_scores": True,
        "action_candidates": [
            {
                "category_id": "B1",
                "source_label": "基础姿态",
                "short_definition": "姿态变化",
                "children": [
                    {
                        "candidate_id": "A1",
                        "action_id": "A1",
                        "source_label": "正式站立",
                        "short_definition": "站立",
                    },
                    {
                        "candidate_id": "A0",
                        "action_id": "no_action",
                        "source_label": "不做动作",
                        "short_definition": "保持当前姿态",
                    },
                ],
            },
            {
                "category_id": "B2",
                "source_label": "重心变化",
                "short_definition": "重心变化",
                "children": [
                    {
                        "candidate_id": "A15",
                        "action_id": "A15",
                        "source_label": "重心左移",
                        "short_definition": "左移",
                    },
                ],
            },
        ],
    })
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_selection_mode"] == "flat_children"
    assert started["action_selection_stages"] == 1

    await session.handle_turn_start(user_turn_start("turn-flat"))
    await session.handle_turn_commit(
        user_turn_commit("turn-flat", text="请站起来")
    )

    assert len(client.score_requests) == 1
    request = client.score_requests[0]
    assert [item.candidate_id for item in request.candidates] == ["A1", "A0", "A15"]
    assert "A1=A1" in request.system_prompt
    assert "A15=A15" in request.system_prompt
    assert "B1=基础姿态" not in request.system_prompt

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "A1"
    assert result["action"]["candidate_id"] == "A1"
    assert result["action"]["category_id"] == "B1"
    assert result["scores"][0]["category_id"] == "B1"
    assert result["media_summary"]["action_context"]["selection_stages"] == 1
    assert result["media_summary"]["action_context"]["selection_mode"] == "flat_children"
    assert result["media_summary"]["action_context"]["flattened_child_count"] == 3

@pytest.mark.asyncio
async def test_turn_result_defaults_to_compact_action_payload() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-compact-result",
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave_left",
                    "source_label": "wave left",
                    "short_definition": "wave left",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "no action",
                    "short_definition": "keep pose",
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-compact"))
    await session.handle_turn_commit(user_turn_commit("turn-compact"))

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert set(result) == {
        "type",
        "session_id",
        "turn_id",
        "action_catalog_hash",
        "action",
        "timing",
    }
    assert result["action"] == {"action_id": "wave_left", "candidate_id": "a01", "execute": True}
    assert "scores" not in result
    assert "media_summary" not in result
    assert "reply" not in result


@pytest.mark.asyncio
async def test_include_scores_returns_diagnostic_fields() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-detailed-result",
            "include_scores": True,
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave_left",
                    "source_label": "wave left",
                    "short_definition": "wave left",
                },
                {
                    "candidate_id": "none",
                    "action_id": "no_action",
                    "source_label": "no action",
                    "short_definition": "keep pose",
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-detailed"))
    await session.handle_turn_commit(user_turn_commit("turn-detailed"))

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"] == {"action_id": "wave_left", "candidate_id": "a01", "execute": True}
    assert len(result["scores"]) == 2
    assert result["media_summary"]["action_context"]["selection_stages"] == 1


@pytest.mark.asyncio
async def test_session_start_prefills_hierarchical_category_catalog() -> None:
    ws = FakeWebSocket()
    client = PrefillFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start({
        "type": "session.start",
        "session_id": "session-prefill-hierarchical",
        "action_candidates": [
            {
                "category_id": "B1",
                "source_label": "基础姿态",
                "short_definition": "姿态变化",
                "children": [
                    {"candidate_id": "A1", "action_id": "A1", "source_label": "站立", "short_definition": "站立"},
                ],
            },
            {
                "category_id": "B2",
                "source_label": "情绪",
                "short_definition": "情绪变化",
                "children": [
                    {"candidate_id": "A2", "action_id": "A2", "source_label": "微笑", "short_definition": "微笑"},
                    {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"},
                ],
            },
        ],
    })

    assert len(client.prefill_requests) == 1
    prefill = client.prefill_requests[0]
    assert prefill["stage"] == "category"
    assert [item.candidate_id for item in prefill["candidates"]] == ["B1", "B2"]
    assert "B1=基础姿态" in prefill["system_prompt"]
    assert "A1" not in prefill["system_prompt"]
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_prefix_prefilled"] is True


@pytest.mark.asyncio
async def test_session_start_prefills_flat_children_without_category_stage() -> None:
    ws = FakeWebSocket()
    client = PrefillFakeClient()
    session = make_session(ws, client, action_selection_mode="flat_children")
    await session.handle_session_start({
        "type": "session.start",
        "session_id": "session-prefill-flat",
        "action_candidates": [
            {
                "category_id": "B1",
                "source_label": "基础姿态",
                "short_definition": "姿态变化",
                "children": [
                    {"candidate_id": "A1", "action_id": "A1", "source_label": "站立", "short_definition": "站立"},
                ],
            },
            {
                "category_id": "B2",
                "source_label": "情绪",
                "short_definition": "情绪变化",
                "children": [
                    {"candidate_id": "A2", "action_id": "A2", "source_label": "微笑", "short_definition": "微笑"},
                    {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"},
                ],
            },
        ],
    })

    assert len(client.prefill_requests) == 1
    prefill = client.prefill_requests[0]
    assert prefill["stage"] == "single"
    assert [item.candidate_id for item in prefill["candidates"]] == ["A1", "A2", "A0"]
    assert "A1=A1" in prefill["system_prompt"]
    assert "B1=" not in prefill["system_prompt"]
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_selection_mode"] == "flat_children"
    assert started["action_selection_stages"] == 1
    assert started["action_prefix_prefilled"] is True
