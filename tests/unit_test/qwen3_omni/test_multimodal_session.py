from __future__ import annotations

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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-1"})

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
    for seq, timestamp in ((1, 2000), (2, 1000)):
        await session.handle_image_append(
            {
                "type": "input_image.append",
                "turn_id": "turn-1",
                "seq": seq,
                "timestamp_ms": timestamp,
                "mime_type": "image/jpeg",
                "image": image,
            }
        )

    await session.handle_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-1",
            "avatar_state": {"pose": "seated"},
        }
    )

    assert len(client.chat_requests) == 0
    assert len(client.score_requests) == 1
    request = client.score_requests[0]
    assert len(request.audios) == 1
    assert len(request.images) == 2
    assert request.history == []
    assert request.candidates[0].suffix == "a01"
    assert "a01=wave_left" in request.system_prompt
    assert "none=no_action" in request.system_prompt

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "wave_left"
    assert result["action"]["execute"] is True
    assert result["media_summary"]["audio_chunk_count"] == 2
    assert result["media_summary"]["image_frame_count"] == 2
    assert result["timing"]["server_action_compute_ms"] >= 0
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"
    assert "action_id=wave_left" in session.history[1]["content"]


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

    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-failed"})
    await session.handle_turn_commit(
        {"type": "turn.commit", "turn_id": "turn-failed"}
    )

    error = next(event for event in ws.events if event["type"] == "error")
    assert error["error"]["type"] == "action_score_error"
    assert error["error"]["code"] == "action_score_logprob_unavailable"
    assert error["session_id"] == "session-score-error"
    assert error["turn_id"] == "turn-failed"
    assert session.active_turn is None

    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-recovered"})
    await session.handle_turn_commit(
        {"type": "turn.commit", "turn_id": "turn-recovered"}
    )
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
        await session.handle_turn_start({"type": "turn.start", "turn_id": turn_id})
        await session.handle_turn_commit({"type": "turn.commit", "turn_id": turn_id})

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
        await session.handle_turn_start({"type": "turn.start"})
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-1"})
    await session.handle_turn_cancel({"type": "turn.cancel", "turn_id": "turn-1"})
    with pytest.raises(ValueError, match="already been used"):
        await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-1"})


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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-seq"})
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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-none"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-none"})
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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-nested"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-nested", "text": "请站起来"})
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

    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-1"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-1", "text": "请站起来"})
    assert [item["stage"] for item in client.prefill_requests] == ["category", "child"]
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-2"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-2", "text": "再来一次"})
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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-nested-top-k"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-nested-top-k", "text": "请站起来"})

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

    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-flat"})
    await session.handle_turn_commit({
        "type": "turn.commit",
        "turn_id": "turn-flat",
        "text": "请站起来",
    })

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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-compact"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-compact"})

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
    await session.handle_turn_start({"type": "turn.start", "turn_id": "turn-detailed"})
    await session.handle_turn_commit({"type": "turn.commit", "turn_id": "turn-detailed"})

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
