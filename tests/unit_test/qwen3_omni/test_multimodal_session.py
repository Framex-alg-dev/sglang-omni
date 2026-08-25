from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO

import pytest
from PIL import Image
from starlette.websockets import WebSocketState

import sglang_omni.serve.realtime.multimodal as multimodal_module
from sglang_omni.client.client import Client
from sglang_omni.client.types import CompletionResult, CompletionStreamChunk, UsageInfo
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    GlobalActionCatalog,
    load_global_action_catalog,
)
from sglang_omni.preprocessing.image import is_prepared_image_wire
from sglang_omni.serve.realtime.multimodal import (
    ACTION_CATEGORY_TOP_K_ENV,
    ACTION_MICRO_BATCH_SIZE_ENV,
    FULL_INSTRUCTIONS_LOG_ENV,
    MultimodalSession,
    MultimodalSessionManager,
    SessionActionCandidate,
    SessionActionProfile,
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
    resource_sample_requester=None,
    global_action_catalog: GlobalActionCatalog | None = None,
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
        global_action_catalog=global_action_catalog,
        claim_session=claim,
        release_session=release,
        request_resource_sample=resource_sample_requester,
    )


def protocol_v1_session_start(
    session_id: str,
    *,
    outputs: list[str] | None = None,
    **fields,
) -> dict:
    event = {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": session_id,
        "outputs": outputs or ["text"],
        "locale": "zh-CN",
    }
    event.update(fields)
    return event


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
async def test_protocol_v1_expands_compact_action_whitelist() -> None:
    catalog = load_global_action_catalog()
    fallback_category = catalog.category_by_id["B008"]
    fallback_candidate = fallback_category.children[0]
    non_system = next(
        candidate
        for candidate in catalog.candidate_by_id.values()
        if candidate.category_id != fallback_category.category_id
    )
    session = make_session(
        FakeWebSocket(),
        FakeClient(),
        global_action_catalog=catalog,
    )
    await session.dispatch(
        protocol_v1_session_start(
            "protocol-v1-actions",
            outputs=["text", "action"],
            character_profile={
                "role": "海洋科学家",
                "personality": "沉稳克制",
            },
            reply={
                "instructions": "使用简洁中文回复。",
                "unsupported_action_text": (
                    "这个动作暂时做不了，我们换个互动方式吧"
                ),
            },
            action={
                "category_guidance": "优先低打扰类别",
                "candidate_guidance": "避免大幅位移",
                "fallback_category_ids": [fallback_category.category_id],
                "allowed_candidates": [
                    {
                        "candidate_id": non_system.candidate_id,
                        "execution_binding": {"asset_id": "test-asset"},
                    },
                    {"candidate_id": fallback_candidate.candidate_id},
                ],
            },
            input_audio={
                "format": "pcm16le",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
            diagnostics={"include_action_scores": True},
        )
    )

    assert session.protocol_version == 1
    assert session.modalities == ("text", "action")
    assert session.locale == "zh-CN"
    assert session.instructions == "使用简洁中文回复。"
    assert "海洋科学家" not in session.instructions
    assert {item.candidate_id for item in session.candidates} == {
        non_system.candidate_id,
        fallback_candidate.candidate_id,
    }
    selected = session.candidate_by_id[non_system.candidate_id]
    assert selected.action_id == non_system.action_id
    assert selected.execution_binding == {"asset_id": "test-asset"}
    assert session.action_profile is not None
    assert session.action_profile.as_dict() == {
        "persona": {
            "role": "海洋科学家",
            "personality": "沉稳克制",
        },
        "category_preferences": "优先低打扰类别",
        "action_preferences": "避免大幅位移",
    }
    started = session.websocket.events[-1]
    assert started["type"] == "session.started"
    assert started["protocol_version"] == 1
    assert started["outputs"] == ["text", "action"]
    assert started["fallback_category_ids"] == ["B008"]
    assert started["unsupported_action_text_configured"] is True
    assert started["unsupported_action_text_sha256"].startswith("sha256:")
    assert "modalities" not in started


@pytest.mark.asyncio
async def test_protocol_v1_defaults_locale_to_english() -> None:
    client = FakeClient()
    session = make_session(FakeWebSocket(), client)
    event = protocol_v1_session_start("protocol-v1-default-locale")
    event.pop("locale")
    event["reply"] = {"instructions": "客户端中文原文，不得翻译。"}

    await session.dispatch(event)

    assert session.locale == "en-US"
    assert session.language == "en"
    started = session.websocket.events[-1]
    assert started["locale"] == "en-US"
    assert session.instructions == "客户端中文原文，不得翻译。"

    await session.handle_turn_start(user_turn_start("turn-default-en"))
    await session.handle_turn_commit(
        user_turn_commit("turn-default-en", text="你好")
    )
    reply_request = client.chat_requests[-1]
    assert reply_request.messages[0].content == "客户端中文原文，不得翻译。"
    assert {"type": "text", "text": "你好"} in reply_request.messages[-1].content
    priority = session._state_description_priority_instruction(
        "category", enabled=True
    )
    assert "highest-priority basis for category selection" in priority
    assert "default action category only when it does not conflict" in priority
    assert (
        session._state_description_priority_instruction(
            "category", enabled=False
        )
        == ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", [None, "zh", "en", "fr-FR"])
async def test_protocol_v1_rejects_invalid_locale(locale) -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    event = protocol_v1_session_start("protocol-v1-invalid-locale")
    event["locale"] = locale

    with pytest.raises(ValueError, match="locale must be 'zh-CN' or 'en-US'"):
        await session.dispatch(event)


@pytest.mark.asyncio
async def test_protocol_v1_does_not_build_reply_prompt_from_character_profile() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    await session.dispatch(
        protocol_v1_session_start(
            "protocol-v1-client-owned-reply-prompt",
            outputs=["text"],
            character_profile={
                "role": "海洋科学家",
                "personality": "沉稳克制",
            },
        )
    )

    assert session.instructions == ""
    assert session.action_profile is None


@pytest.mark.asyncio
async def test_protocol_v1_requires_unsupported_text_only_for_fusion() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    with pytest.raises(ValueError, match="unsupported_action_text is required"):
        await session.dispatch(
            protocol_v1_session_start(
                "missing-unsupported-text",
                outputs=["text", "action"],
            )
        )

    session = make_session(FakeWebSocket(), FakeClient())
    with pytest.raises(
        ValueError, match="requires both text and action outputs"
    ):
        await session.dispatch(
            protocol_v1_session_start(
                "unexpected-unsupported-text",
                outputs=["text"],
                reply={"unsupported_action_text": "暂时做不了。"},
            )
        )


@pytest.mark.parametrize("send_empty_allowed_candidates", [False, True])
@pytest.mark.asyncio
async def test_protocol_v1_defaults_to_all_fallback_category_actions(
    send_empty_allowed_candidates: bool,
) -> None:
    catalog = load_global_action_catalog()
    fallback_category = catalog.category_by_id["B008"]
    action = {"fallback_category_ids": [fallback_category.category_id]}
    if send_empty_allowed_candidates:
        action["allowed_candidates"] = []
    session = make_session(
        FakeWebSocket(),
        FakeClient(),
        global_action_catalog=catalog,
    )

    await session.dispatch(
        protocol_v1_session_start(
            f"default-fallback-actions-{send_empty_allowed_candidates}",
            outputs=["action"],
            action=action,
        )
    )

    assert session.fallback_category_ids == (fallback_category.category_id,)
    assert {item.candidate_id for item in session.candidates} == {
        item.candidate_id for item in fallback_category.children
    }
    assert {item.category_id for item in session.candidates} == {
        fallback_category.category_id
    }
    assert all(not item.execution_binding for item in session.candidates)


@pytest.mark.asyncio
async def test_protocol_v1_requires_fallback_categories_for_action() -> None:
    catalog = load_global_action_catalog()
    candidate = next(iter(catalog.candidate_by_id.values()))
    session = make_session(
        FakeWebSocket(),
        FakeClient(),
        global_action_catalog=catalog,
    )

    with pytest.raises(
        ValueError, match="action is missing required fields: fallback_category_ids"
    ):
        await session.dispatch(
            protocol_v1_session_start(
                "missing-fallback-categories",
                outputs=["action"],
                action={
                    "allowed_candidates": [
                        {"candidate_id": candidate.candidate_id}
                    ]
                },
            )
        )


@pytest.mark.asyncio
async def test_protocol_v1_rejects_legacy_and_unknown_fields() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    with pytest.raises(ValueError, match="unsupported protocol_version"):
        await session.dispatch(
            {
                "type": "session.start",
                "protocol_version": 2,
                "session_id": "future-version",
                "outputs": ["text"],
            }
        )
    with pytest.raises(ValueError, match="unsupported fields: modalities"):
        await session.dispatch(
            {
                "type": "session.start",
                "protocol_version": 1,
                "session_id": "legacy-field",
                "modalities": ["text"],
            }
        )
    with pytest.raises(ValueError, match="unsupported event type"):
        await session.dispatch(
            {
                "type": "input_audio_buffer.append",
                "turn_id": "turn-1",
                "seq": 1,
                "audio": "AA==",
            }
        )


@pytest.mark.asyncio
async def test_protocol_v1_empty_provided_reply_skips_generation() -> None:
    ws = FakeWebSocket()
    client = FakeClient()
    session = make_session(ws, client)
    await session.dispatch(protocol_v1_session_start("empty-reply"))
    await session.dispatch(
        {
            "type": "turn.start",
            "turn_id": "turn-empty",
            "origin": "proactive",
            "trigger_type": "action_request",
        }
    )
    await session.dispatch(
        {
            "type": "turn.commit",
            "turn_id": "turn-empty",
            "reply": {"provided_text": ""},
        }
    )
    for _ in range(100):
        if session.active_turn is None:
            break
        await asyncio.sleep(0)

    assert client.chat_requests == []
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["reply"] == {"text": "", "source": "provided"}
    assert result["status"] == "completed"
    assert result["outputs"] == {"text": "completed"}
    assert "modalities" not in result


@pytest.mark.asyncio
async def test_protocol_v1_rejects_conflicting_media_retries() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    await session.dispatch(protocol_v1_session_start("media-retry"))
    await session.dispatch(
        {"type": "turn.start", "turn_id": "turn-media", "origin": "user"}
    )
    first = base64.b64encode(b"\x00\x00").decode()
    second = base64.b64encode(b"\x01\x00").decode()
    await session.dispatch(
        {
            "type": "input.audio.append",
            "turn_id": "turn-media",
            "seq": 1,
            "data": first,
        }
    )
    await session.dispatch(
        {
            "type": "input.audio.append",
            "turn_id": "turn-media",
            "seq": 1,
            "data": first,
        }
    )
    with pytest.raises(ValueError, match="conflicts"):
        await session.dispatch(
            {
                "type": "input.audio.append",
                "turn_id": "turn-media",
                "seq": 1,
                "data": second,
            }
        )


def test_action_profile_validation_and_normalization() -> None:
    profile = SessionActionProfile.from_payload(
        {
            "persona": {
                "gender_expression": " 男性 ",
                "visual_style": "写实",
                "role": "海洋科学家",
                "personality": "沉稳克制",
            },
            "category_preferences": " 优先低打扰类别 ",
            "action_preferences": "避免大幅位移",
        }
    )
    assert profile.as_dict() == {
        "persona": {
            "gender_expression": "男性",
            "visual_style": "写实",
            "role": "海洋科学家",
            "personality": "沉稳克制",
        },
        "category_preferences": "优先低打扰类别",
        "action_preferences": "避免大幅位移",
    }

    with pytest.raises(ValueError, match="at least one non-empty field"):
        SessionActionProfile.from_payload({})
    with pytest.raises(ValueError, match="unsupported fields"):
        SessionActionProfile.from_payload({"persona": {"age": "青年"}})
    with pytest.raises(ValueError, match="category_preferences must be a string"):
        SessionActionProfile.from_payload({"category_preferences": ["低打扰"]})


@pytest.mark.asyncio
async def test_action_profile_requires_action_modality() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    with pytest.raises(ValueError, match="requires the action modality"):
        await session.handle_session_start(
            {
                "type": "session.start",
                "session_id": "text-only-action-profile",
                "modalities": ["text"],
                "action_profile": {
                    "category_preferences": "优先低打扰类别"
                },
            }
        )


@pytest.mark.asyncio
async def test_full_diagnostic_logging_records_normalized_action_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    monkeypatch.setenv(FULL_INSTRUCTIONS_LOG_ENV, "1")
    session = make_session(FakeWebSocket(), FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "action-profile-log",
            "modalities": ["action"],
            "action_profile": {
                "persona": {"role": " 海洋科学家 "},
                "action_preferences": " 避免大幅位移 ",
            },
            "action_candidates": [
                {
                    "candidate_id": "a01",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "挥手问候",
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

    profile_record = next(
        record
        for record in structured_records
        if record["event"] == "session_action_profile_received"
    )
    assert profile_record["action_profile"] == {
        "persona": {"role": "海洋科学家"},
        "action_preferences": "避免大幅位移",
    }
    assert profile_record["action_profile_sha256"].startswith("sha256:")


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
            "modalities": ["action"],
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
    assert request.image_roles == ["user_camera", "avatar_state"]
    assert request.avatar_state == {}
    assert request.history == []
    assert "本轮图片2是时间最新的数字人状态照片" not in request.prefix
    assert request.candidates[0].suffix == "a01"
    assert (
        "candidate_id=a01｜动作=左手挥手｜说明=使用左手抬起并左右摆动"
        in request.system_prompt
    )
    assert (
        "candidate_id=none｜动作=不做动作｜说明=保持当前姿态"
        in request.system_prompt
    )
    omni_request = Client._build_action_scoring_request(request)
    current_parts = omni_request.inputs["messages"][-1]["content"]
    role_map_index = next(
        index
        for index, part in enumerate(current_parts)
        if part.get("type") == "text"
        and part["text"].startswith("[当前图片用途]")
    )
    image_indices = [
        index for index, part in enumerate(current_parts) if part.get("type") == "image"
    ]
    assert role_map_index < min(image_indices)
    assert "用户摄像头画面=1" in current_parts[role_map_index]["text"]
    assert "数字人当前状态画面=2" in current_parts[role_map_index]["text"]
    assert "画面2是本轮时间最新的数字人照片" in current_parts[role_map_index]["text"]
    assert current_parts[role_map_index]["text"].count("用户摄像头画面") == 1
    assert current_parts[role_map_index]["text"].count("数字人当前状态画面") == 1
    assert "当前数字人状态：" not in current_parts[-1]["text"]
    assert '"pose":"seated"' not in current_parts[-1]["text"]

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
        if part.get("type") == "text"
        and part["text"].startswith("[当前图片用途]")
    ]
    assert historical_labels == [
        "[当前图片用途] 用户摄像头画面=1（只描述用户及其环境）；"
        "本轮没有数字人当前状态画面，数字人姿态只能使用结构化的"
        "数字人当前状态信息，不得从用户摄像头画面或历史图片推断。",
    ]
    assert len(second_request.history_images) == 1
    assert second_request.avatar_state == {}
    assert "本轮没有可用的数字人当前状态信息" in second_request.prefix
    assert "不得从历史图片或用户摄像头画面推断数字人状态" in second_request.prefix
    assert "以结构化 数字人当前状态信息为准" not in second_request.prefix
    assert "候选动作必须与当前可视姿态兼容" not in second_request.prefix
    second_omni_request = Client._build_action_scoring_request(second_request)
    assert "当前结构化数字人状态：" not in second_omni_request.inputs["messages"][-1]["content"]
    second_result = next(
        event
        for event in ws.events
        if event["type"] == "turn.result" and event["turn_id"] == "turn-2"
    )
    assert (
        second_result["media_summary"]["action_context"][
            "ignored_history_avatar_image_count"
        ]
        == 0
    )
    assert session.history_image_roles == ["user_camera"]


def test_current_image_role_map_is_compact_and_precedes_images() -> None:
    parts = Client._current_image_content_parts(["user_camera"] * 6)

    assert parts[0] == {
        "type": "text",
        "text": (
            "[当前图片用途] 用户摄像头画面=1-6（只描述用户及其环境）；"
            "本轮没有数字人当前状态画面，数字人姿态只能使用结构化的"
            "数字人当前状态信息，不得从用户摄像头画面或历史图片推断。"
        ),
    }
    assert [part["type"] for part in parts[1:]] == ["image"] * 6

    audio_only = Client._action_instruction_content(
        [],
        "选择动作",
        audios=["pcm"],
        images=[],
        image_roles=[],
    )
    assert not any(
        part.get("type") == "text"
        and part["text"].startswith("[当前图片用途]")
        for part in audio_only
    )

    english_parts = Client._current_image_content_parts(
        ["user_camera", "avatar_state"], language="en"
    )
    assert english_parts[0]["text"].startswith("[Current image purposes]")
    assert "User camera view=1" in english_parts[0]["text"]
    assert "Current digital character state view=2" in english_parts[0]["text"]


def test_avatar_image_replaces_visual_state_but_keeps_action_context() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    session.last_avatar_state = {"pose": "standing", "gaze": "camera"}
    explicit = {
        "pose": "seated",
        "current_action_id": "wave",
        "state_description": "用户刚回来，本次应轻量欢迎。",
    }

    proactive_state = session._effective_avatar_state(
        explicit,
        turn_origin="proactive",
        has_avatar_image=True,
    )
    assert proactive_state == {
        "current_action_id": "wave",
        "state_description": "用户刚回来，本次应轻量欢迎。",
    }
    passive_state = session._effective_avatar_state(
        explicit,
        turn_origin="user",
        has_avatar_image=True,
    )
    assert passive_state == {"current_action_id": "wave"}
    assert session._avatar_state_source(proactive_state, ["avatar_state"]) == "image"
    assert session._avatar_state_source(proactive_state, []) == "unknown"
    assert session._avatar_state_source(
        {"pose": " ", "gaze": None, "hands": []}, []
    ) == "unknown"

    messages = Client._build_action_context_messages(
        [],
        "选择动作",
        avatar_state=proactive_state,
        system_prompt=None,
        audios=[],
        images=["image"],
        image_roles=["avatar_state"],
    )
    instruction = messages[-1]["content"][-1]["text"]
    assert instruction.startswith("本轮动作选择补充信息：")
    assert '"当前实际动作 ID":"wave"' in instruction
    assert '"本轮主动场景约束":"用户刚回来，本次应轻量欢迎。"' in instruction
    assert '"pose"' not in instruction

    session._persist_avatar_state(explicit, has_avatar_image=False)
    assert session.last_avatar_state == {"pose": "seated"}
    inherited = session._effective_avatar_state(
        None,
        turn_origin="proactive",
        has_avatar_image=False,
    )
    assert inherited == {"pose": "seated"}
    assert session._avatar_state_source(inherited, []) == "structured"
    session._persist_avatar_state(explicit, has_avatar_image=True)
    assert session.last_avatar_state == {}


def test_bounded_context_keeps_only_latest_current_avatar_image() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    images = [
        "avatar-old",
        "avatar-latest",
        *(f"user-{index}" for index in range(1, 10)),
    ]
    roles = ["avatar_state", "avatar_state", *(["user_camera"] * 9)]

    (
        _,
        _,
        _,
        bounded_images,
        bounded_roles,
        context,
    ) = session._build_bounded_action_context([], images, roles)

    assert len(bounded_images) == 8
    assert bounded_images[0] == "avatar-latest"
    assert "avatar-old" not in bounded_images
    assert bounded_roles.count("avatar_state") == 1
    assert context["received_current_image_count"] == 11
    assert context["scored_current_image_count"] == 8
    assert context["truncated"] is True


def test_action_history_does_not_retain_avatar_images() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    candidate = SessionActionCandidate(
        candidate_id="A1",
        action_id="wave",
        source_label="挥手",
        short_definition="挥手问候",
        execution_binding={},
        category_id="B1",
    )
    session.candidate_by_id = {candidate.candidate_id: candidate}

    session._append_action_history(
        [],
        ["user-camera", "avatar-old", "avatar-latest"],
        ["user_camera", "avatar_state", "avatar_state"],
        "你好",
        turn_id="turn-action-history-images",
        turn_origin="user",
        text_role="user_input",
        action={
            "candidate_id": "A1",
            "action_id": "wave",
            "category_id": "B1",
            "execute": True,
        },
    )

    assert session.history_images == ["user-camera"]
    assert session.history_image_roles == ["user_camera"]
    assert session.history_turns[-1].images == ["user-camera"]
    content = session.history_turns[-1].messages[0]["content"]
    assert [part["type"] for part in content] == ["image", "text"]


@pytest.mark.asyncio
async def test_image_append_preprocesses_before_action_scoring() -> None:
    ws = FakeWebSocket()
    client = FakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-image-preprocess",
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
    await session.handle_turn_start(user_turn_start("turn-image-preprocess"))
    image = Image.new("RGB", (3, 2), color=(17, 34, 51))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    original_b64 = base64.b64encode(encoded.getvalue()).decode()

    await session.handle_image_append(
        {
            "type": "input_image.append",
            "turn_id": "turn-image-preprocess",
            "seq": 1,
            "timestamp_ms": 1,
            "mime_type": "image/png",
            "image": original_b64,
        }
    )
    assert session.active_turn is not None
    assert session.active_turn.images[0].preprocess_task is not None

    await session.handle_turn_commit(user_turn_commit("turn-image-preprocess"))

    assert len(client.score_requests) == 1
    prepared = client.score_requests[0].images[0]
    assert is_prepared_image_wire(prepared)
    assert prepared["width"] == 3
    assert prepared["height"] == 2
    assert session.history_images == [f"data:image/png;base64,{original_b64}"]
    result = next(event for event in ws.events if event["type"] == "turn.result")
    stats = result["timing"]["image_preprocessing"]
    assert stats["scheduled_count"] == 1
    assert stats["prepared_count"] == 1
    assert stats["fallback_count"] == 0
    assert stats["prepared_bytes"] == 18
    assert stats["statuses"] == ["prepared"]
    assert stats["commit_wait_ms"] >= 0


@pytest.mark.asyncio
async def test_image_role_defaults_by_turn_origin_and_rejects_unknown_role() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
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
            "modalities": ["action"],
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
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-2",
            "language": "zh",
            "action_candidates": candidates,
        }
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
    assert "处理结果=已按执行处理" in second_request.history[1]["content"]
    assert "动作=左手挥手" in second_request.history[1]["content"]
    assert second_request.system_prompt == client.score_requests[0].system_prompt

@pytest.mark.asyncio
async def test_committed_turn_can_be_aborted_and_next_turn_runs() -> None:
    ws = FakeWebSocket()
    client = BlockingActionClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
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
    await session._dispatch_turn_commit(
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
            "modalities": ["action"],
            "session_id": "session-close-abort",
            "action_candidates": [{"candidate_id": "none", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前状态"}],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-close"))
    await session._dispatch_turn_commit(user_turn_commit("turn-close"))
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
            "modalities": ["action"],
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
    await session._dispatch_turn_commit(user_turn_commit("turn-disconnect"))
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
            "modalities": ["action"],
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
            "modalities": ["action"],
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
                "modalities": ["action"],
                "session_id": "session-proactive-without-text",
                "language": "zh",
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
    assert "未提供数字人本轮将要说出的文本" in request.prefix
    assert "已提供数字人本轮将要说出的文本" not in request.prefix
    assert "本轮新增的数字人消息" not in request.prefix
    assert "不要把历史中的数字人回复当成本轮将要说出的文本" in request.prefix
    assert "本轮没有可用的数字人当前状态信息" in request.prefix
    assert "以结构化 数字人当前状态信息为准" not in request.prefix
    assert "候选动作必须与当前可视姿态兼容" not in request.prefix
    assert "本轮主动场景约束中给出的目标、指引、要求和禁止项" in request.prefix
    assert "本轮提供的“当前实际动作 ID”所表示的动作" in request.prefix
    assert request.avatar_state == {
        "current_action_id": "wave",
        "state_description": "上一动作已结束，保持当前状态。",
    }
    assert session.last_avatar_state == {}
    assert "上一条 assistant 消息是数字人已经准备好" not in request.prefix
    assert session.history[0]["role"] == "assistant"
    assert session.history[0]["content"].startswith("[历史动作记录]")
    assert "turn_id=" not in session.history[0]["content"]
    assert "None" not in session.history[0]["content"]


@pytest.mark.asyncio
async def test_proactive_text_is_assistant_context_and_persists_for_next_turn() -> None:
    client = FakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
                "modalities": ["action"],
                "session_id": "session-proactive-context",
                "language": "zh",
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
    assert "已提供数字人本轮将要说出的文本" in proactive_request.prefix
    assert "本轮新增的数字人消息" in proactive_request.prefix
    assert (
        "判断候选动作可执行性时，以结构化 数字人当前状态信息为准"
        in proactive_request.prefix
    )
    assert "将要说出的文本，其语义、语气和表达目标是本轮核心约束" in proactive_request.prefix
    assert "该文本的语义、语气和表达目标直接相关" in proactive_request.prefix
    assert "本轮主动场景约束中给出的目标、指引、要求和禁止项" in proactive_request.prefix
    assert "结合历史动作判断衔接关系" in proactive_request.prefix
    assert "选择与表达目标和状态约束最匹配的候选项" in proactive_request.prefix
    assert "candidate_id=none" not in proactive_request.prefix
    assert proactive_request.prefix.count("Hello，你回来啦！") == 0
    assert "不要生成回复" in proactive_request.prefix
    assert "当前用户文本" not in proactive_request.prefix
    assert proactive_request.avatar_state["pose"] == "seated"
    assert proactive_request.avatar_state["conversation_phase"] == "greeting"
    image_state_prompt = session._build_turn_action_instruction(
        "Hello，你回来啦！",
        turn_origin="proactive",
        trigger="user_returned",
        image_roles=["avatar_state"],
        has_current_action_id=True,
        has_state_description=True,
    )
    assert (
        "以本轮最新数字人照片中的当前可视姿态和行为为准"
        in image_state_prompt
    )
    assert "将要说出的文本，其语义、语气和表达目标是本轮核心约束" in image_state_prompt
    assert "本轮主动场景约束中给出的目标、指引、要求和禁止项" in image_state_prompt
    assert "本轮提供的“当前实际动作 ID”所表示的动作" in image_state_prompt
    assert session.last_avatar_state == {
        "pose": "seated",
        "conversation_phase": "greeting",
    }

    assert len(session.history_turns) == 1
    assert session.history_turns[0].turn_origin == "proactive"
    assert [message["role"] for message in session.history] == ["assistant"]
    assert "Hello，你回来啦！" in session.history[0]["content"]
    assert "[历史动作记录]" in session.history[0]["content"]
    assert "action_id=wave" in session.history[0]["content"]

    await session.handle_turn_start(user_turn_start("user-after-proactive"))
    await session.handle_turn_commit(
        user_turn_commit("user-after-proactive", text="我们继续聊吧")
    )
    next_request = client.score_requests[1]
    assert len(next_request.history) == 1
    assert next_request.history[0]["role"] == "assistant"
    assert "Hello，你回来啦！" in next_request.history[0]["content"]
    assert "[历史动作记录]" in next_request.history[0]["content"]
    assert "本轮由用户输入触发" in next_request.prefix


@pytest.mark.asyncio
async def test_audio_sequence_is_strict_and_duplicate_is_idempotent() -> None:
    ws = FakeWebSocket()
    session = make_session(ws, FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
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
            "modalities": ["action"],
            "session_id": "session-fixed-catalog",
            "include_scores": True,
            "action_candidates": candidates,
        }
    )
    with pytest.raises(ValueError, match="only be sent once"):
        await session.handle_session_start(
            {
                "type": "session.start",
                "modalities": ["action"],
                "session_id": "session-fixed-catalog",
                "action_candidates": [candidates[1]],
            }
        )
    await session.handle_turn_start(user_turn_start("turn-none"))
    await session.handle_turn_commit(user_turn_commit("turn-none"))
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "no_action"
    assert result["action"]["execute"] is False
    assert session.last_executed_action is not None
    assert session.last_executed_action.action_id == "no_action"
    assert session.last_executed_action.execute is False
    assert result["media_summary"]["received_image_count"] == 0
    assert result["media_summary"]["scored_image_count"] == 0
    assert "处理结果=未执行新动作（保持当前姿态）" in (
        session.history[-1]["content"]
    )



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


class FusionFakeClient(NestedFakeClient):
    def __init__(self, *, block_child: bool = False) -> None:
        super().__init__()
        self.reply_requests = []
        self.reply_started = asyncio.Event()
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.block_child = block_child

    async def completion_stream(self, request, *, request_id: str):
        self.reply_requests.append(request)
        self.reply_started.set()
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="text",
            text="你好呀，",
        )
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="text",
            text="今天过得怎么样？",
            finish_reason="stop",
            usage=UsageInfo(
                prompt_tokens=20,
                completion_tokens=9,
                total_tokens=29,
            ),
        )

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        if request.stage == "child":
            self.child_started.set()
            if self.block_child:
                await self.release_child.wait()
        return await super().score_action_suffixes(request)


class ScriptedChildFusionClient(FusionFakeClient):
    def __init__(self, child_candidate_ids: list[str]) -> None:
        super().__init__()
        self.child_candidate_ids = iter(child_candidate_ids)

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        if request.stage != "child":
            return await super().score_action_suffixes(request)
        self.score_requests.append(request)
        selected = next(self.child_candidate_ids)
        scores = []
        for index, candidate in enumerate(request.candidates):
            value = -0.1 if candidate.candidate_id == selected else -1.0 - index
            scores.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    token_count=1,
                    mean_logprob=value,
                    mean_nll=-value,
                    ppl=1.105170 if value == -0.1 else 2.718281,
                    token_scores=[
                        TokenScore(token_id=100 + index, logprob=value)
                    ],
                )
            )
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=scores,
        )


class BlockingFusionClient(FusionFakeClient):
    def __init__(self) -> None:
        super().__init__(block_child=True)
        self.release_reply = asyncio.Event()
        self.aborted: list[str] = []

    async def completion_stream(self, request, *, request_id: str):
        self.reply_requests.append(request)
        self.reply_started.set()
        await self.release_reply.wait()
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="text",
            text="不会在取消后发送",
            finish_reason="stop",
        )

    async def abort(self, request_id: str):
        self.aborted.append(request_id)
        self.release_child.set()
        self.release_reply.set()


class FailingChildFusionClient(FusionFakeClient):
    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        if request.stage == "child":
            self.score_requests.append(request)
            raise RuntimeError("synthetic child scoring failure")
        return await super().score_action_suffixes(request)


class ToggleChildFailureFusionClient(FusionFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.fail_child = False

    async def score_action_suffixes(self, request) -> ActionSuffixScoreResult:
        if request.stage == "child" and self.fail_child:
            self.score_requests.append(request)
            raise RuntimeError("synthetic child scoring failure")
        return await super().score_action_suffixes(request)


def fusion_catalog() -> list[dict]:
    return [
        {
            "category_id": "B010",
            "source_label": "问候",
            "short_definition": "打招呼、欢迎和告别等互动",
            "children": [
                {
                    "candidate_id": "A123",
                    "action_id": "wave",
                    "source_label": "挥手",
                    "short_definition": "自然挥手问候",
                },
                {
                    "candidate_id": "A124",
                    "action_id": "wave_both",
                    "source_label": "双手问候",
                    "short_definition": "自然进行双手问候",
                }
            ],
        },
        {
            "category_id": "B000",
            "source_label": "系统动作",
            "short_definition": "没有合适动作时保持当前姿态",
            "children": [
                {
                    "candidate_id": "A000",
                    "action_id": "no_action",
                    "source_label": "不做动作",
                    "short_definition": "保持当前姿态",
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_default_modalities_run_reply_and_action_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    monkeypatch.delenv(FULL_INSTRUCTIONS_LOG_ENV, raising=False)
    ws = FakeWebSocket()
    client = FusionFakeClient(block_child=True)
    session = make_session(ws, client)
    await session.handle_session_start(
        {
                "type": "session.start",
                "session_id": "session-fusion-default",
                "language": "zh",
            "instructions": "自然回复，不要复述动作。",
            "unsupported_action_text": "这个动作暂时做不了。",
            "fallback_category_ids": ["B000"],
            "action_profile": {
                "persona": {
                    "gender_expression": "男性",
                    "visual_style": "写实",
                    "role": "海洋科学家",
                    "personality": "沉稳克制",
                },
                "category_preferences": "优先自然交流和低打扰类别",
                "action_preferences": "避免夸张舞蹈和大幅位移",
            },
            "action_candidates": fusion_catalog(),
        }
    )
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["modalities"] == ["text", "action"]
    assert started["action_profile_applied"] is True
    assert started["action_profile_sha256"].startswith("sha256:")

    await session.handle_turn_start(user_turn_start("turn-fusion"))
    await session._dispatch_turn_commit(
        user_turn_commit(
            "turn-fusion",
            text="你可以给我打个招呼吗？",
            reply_context="回复要自然，不要说明动作。",
        )
    )
    turn_task = session.active_turn.inference_task
    await asyncio.wait_for(client.child_started.wait(), timeout=1)
    await asyncio.wait_for(client.reply_started.wait(), timeout=1)
    assert not turn_task.done()
    client.release_child.set()
    await asyncio.wait_for(turn_task, timeout=1)

    assert [request.stage for request in client.score_requests] == [
        "category",
        "child",
    ]
    reply_request = client.reply_requests[0]
    assert reply_request.messages[0].role == "system"
    assert reply_request.messages[0].content == "自然回复，不要复述动作。"
    assert "海洋科学家" not in reply_request.messages[0].content
    current_reply_content = reply_request.messages[-1].content
    rendered_controls = [
        part["text"]
        for part in current_reply_content
        if part.get("type") == "text"
    ]
    assert "回复要自然，不要说明动作。" in rendered_controls
    assert all("[已执行动作事实]" not in item for item in rendered_controls)
    assert all("category_id=B010" not in item for item in rendered_controls)
    assert all("A123" not in item for item in rendered_controls)
    category_request, child_request = client.score_requests
    persona_prompt = (
        "数字人人设：性别表达=男性；画风=写实；"
        "职业或角色定位=海洋科学家；性格基调=沉稳克制"
    )
    assert persona_prompt in category_request.prefix
    assert (
        "类别偏好（动作类别选择的主要约束）：优先自然交流和低打扰类别"
        in category_request.prefix
    )
    assert "动作偏好（动作类别选择的可行性约束）" in category_request.prefix
    assert persona_prompt in child_request.prefix
    assert "类别偏好（具体动作选择的背景约束）" in child_request.prefix
    assert (
        "动作偏好（具体动作选择的主要约束）：避免夸张舞蹈和大幅位移"
        in child_request.prefix
    )
    assert "海洋科学家" not in category_request.system_prompt
    assert "海洋科学家" not in child_request.system_prompt
    assert [item.candidate_id for item in child_request.candidates] == [
        "A123",
        "A124",
    ]
    assert "candidate_id=A000" not in child_request.system_prompt
    assert all(
        "回复要自然" not in request.prefix
        and "回复要自然" not in request.system_prompt
        for request in client.score_requests
    )

    event_types = [event["type"] for event in ws.events]
    assert "response.provisional.created" in event_types
    assert "response.provisional.text.delta" in event_types
    assert "response.provisional.resolved" in event_types
    assert "response.text.delta" in event_types
    assert "turn.action.ready" in event_types
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["reply"] == {
        "text": "你好呀，今天过得怎么样？",
        "source": "generated",
    }
    assert result["action"]["action_id"] == "wave"
    assert result["modalities"] == {"text": "completed", "action": "completed"}
    resolved = next(
        event
        for event in ws.events
        if event["type"] == "response.provisional.resolved"
    )
    assert resolved["status"] == "promoted"
    assert resolved["reason"] == "action_supported"
    completed = next(
        record
        for record in structured_records
        if record["event"] == "reply_completed"
    )
    logical_input = next(
        record
        for record in structured_records
        if record["event"] == "reply_logical_input"
    )
    assert logical_input["instructions_applied"] is True
    assert logical_input["effective_system_prompt"] is None
    assert logical_input["messages"][0]["content"].startswith("<redacted;")
    assert not any(
        record["event"] == "session_instructions_received"
        for record in structured_records
    )
    assert not any(
        record["event"] == "session_action_profile_received"
        for record in structured_records
    )
    started_record = next(
        record
        for record in structured_records
        if record["event"] == "session_started"
    )
    assert started_record["action_profile_present"] is True
    assert started_record["action_profile_sha256"].startswith("sha256:")
    assert 0 <= completed["created_after_commit_ms"]
    assert (
        completed["created_after_commit_ms"]
        <= completed["first_delta_after_commit_ms"]
        <= completed["provisional_done_after_commit_ms"]
    )
    assert completed["stream_duration_ms"] >= 0
    assert completed["delta_count"] == 2
    assert completed["completion_tokens"] == 9
    turn_timing = next(
        record
        for record in structured_records
        if record["event"] == "turn_timing"
    )
    assert (
        turn_timing["reply_first_delta_after_commit_ms"]
        == completed["first_delta_after_commit_ms"]
    )
    assert (
        turn_timing["reply_text_done_after_commit_ms"]
        >= completed["provisional_done_after_commit_ms"]
    )
    assert turn_timing["reply_response_done_after_commit_ms"] is not None
    assert turn_timing["reply_delta_count"] == 2
    assert turn_timing["reply_completion_tokens"] == 9


@pytest.mark.asyncio
async def test_turn_resource_samples_cover_successful_inference_boundaries() -> None:
    requests: list[tuple[str, dict]] = []

    def request_sample(sample_trigger: str, **fields) -> bool:
        requests.append((sample_trigger, fields))
        return True

    ws = FakeWebSocket()
    client = FusionFakeClient()
    session = make_session(
        ws,
        client,
        resource_sample_requester=request_sample,
    )
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-turn-resource-success",
            "action_candidates": fusion_catalog(),
        }
    )
    await session.handle_turn_start(user_turn_start("turn-resource-success"))
    await session.handle_turn_commit(
        user_turn_commit("turn-resource-success", text="向我打招呼")
    )

    assert [trigger for trigger, _ in requests] == [
        "turn_before_inference",
        "turn_after_terminal",
    ]
    before = requests[0][1]
    after = requests[1][1]
    assert before["session_id"] == "session-turn-resource-success"
    assert before["turn_id"] == "turn-resource-success"
    assert before["turn_origin"] == "user"
    assert before["modalities"] == ["text", "action"]
    assert before["audio_chunk_count"] == 0
    assert after["turn_outcome"] == "completed"
    assert after["elapsed_after_commit_ms"] >= 0


@pytest.mark.asyncio
async def test_turn_resource_samples_cover_failed_inference() -> None:
    requests: list[tuple[str, dict]] = []

    def request_sample(sample_trigger: str, **fields) -> bool:
        requests.append((sample_trigger, fields))
        return True

    ws = FakeWebSocket()
    session = make_session(
        ws,
        FailingOnceClient(),
        resource_sample_requester=request_sample,
    )
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-turn-resource-failed",
            "action_candidates": fusion_catalog(),
        }
    )
    await session.handle_turn_start(user_turn_start("turn-resource-failed"))
    await session.handle_turn_commit(
        user_turn_commit("turn-resource-failed", text="向我打招呼")
    )

    assert [trigger for trigger, _ in requests] == [
        "turn_before_inference",
        "turn_after_terminal",
    ]
    assert requests[-1][1]["turn_outcome"] == "failed"
    assert any(event["type"] == "error" for event in ws.events)


@pytest.mark.asyncio
async def test_successful_action_is_not_automatically_injected_into_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    ws = FakeWebSocket()
    client = FusionFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
                "type": "session.start",
                "session_id": "session-executed-action-reply-fact",
                "language": "zh",
            "instructions": "只输出自然语言台词。",
            "action_candidates": fusion_catalog(),
        }
    )

    await session.handle_turn_start(user_turn_start("turn-action-source"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-action-source", text="向我打个招呼")
    )
    first_task = session.active_turn.inference_task
    await asyncio.wait_for(first_task, timeout=1)

    assert session.last_executed_action is not None
    assert session.last_executed_action.turn_id == "turn-action-source"
    assert session.last_executed_action.source_label == "挥手"
    assert session.last_executed_action.execute is True
    assert len(session.executed_action_history) == 1
    assumed = next(
        record
        for record in structured_records
        if record["event"] == "action_execution_assumed"
    )
    assert assumed["source_turn_id"] == "turn-action-source"
    assert assumed["action_id"] == "wave"

    await session.handle_turn_start(user_turn_start("turn-ask-previous-action"))
    await session._dispatch_turn_commit(
        user_turn_commit(
            "turn-ask-previous-action",
            text="你上一个做的动作是什么？",
            reply_context="用户正在询问历史动作；最近一次已执行动作：挥手。",
        )
    )
    second_task = session.active_turn.inference_task
    await asyncio.wait_for(second_task, timeout=1)

    reply_controls = [
        part["text"]
        for part in client.reply_requests[1].messages[-1].content
        if part.get("type") == "text"
    ]
    assert all("[已执行动作事实]" not in item for item in reply_controls)
    assert "用户正在询问历史动作；最近一次已执行动作：挥手。" in reply_controls
    assert all("仅当用户明确询问之前的动作" not in item for item in reply_controls)
    assert all("A123" not in item for item in reply_controls)
    assert all("wave" not in item for item in reply_controls)
    assert all("自然挥手问候" not in item for item in reply_controls)


@pytest.mark.asyncio
async def test_failed_child_does_not_replace_last_executed_action_fact() -> None:
    ws = FakeWebSocket()
    client = ToggleChildFailureFusionClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-executed-action-child-failure",
            "action_candidates": fusion_catalog(),
        }
    )

    await session.handle_turn_start(user_turn_start("turn-successful-action"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-successful-action", text="向我打招呼")
    )
    first_task = session.active_turn.inference_task
    await asyncio.wait_for(first_task, timeout=1)
    assert session.last_executed_action is not None
    assert session.last_executed_action.turn_id == "turn-successful-action"

    client.fail_child = True
    await session.handle_turn_start(user_turn_start("turn-failed-action"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-failed-action", text="再向我打个招呼")
    )
    failed_task = session.active_turn.inference_task
    await asyncio.wait_for(failed_task, timeout=1)

    assert session.last_executed_action is not None
    assert session.last_executed_action.turn_id == "turn-successful-action"
    assert len(session.executed_action_history) == 1
    failed_result = next(
        event
        for event in ws.events
        if event.get("type") == "turn.result"
        and event.get("turn_id") == "turn-failed-action"
    )
    assert failed_result["status"] == "partial"

    await session.handle_turn_start(user_turn_start("turn-after-action-failure"))
    await session._dispatch_turn_commit(
        user_turn_commit(
            "turn-after-action-failure",
            text="你最近一次成功做了什么？",
            reply_context="最近一次成功执行的动作：挥手。",
        )
    )
    third_task = session.active_turn.inference_task
    await asyncio.wait_for(third_task, timeout=1)
    reply_controls = [
        part["text"]
        for part in client.reply_requests[-1].messages[-1].content
        if part.get("type") == "text"
    ]
    assert "最近一次成功执行的动作：挥手。" in reply_controls
    assert all("[已执行动作事实]" not in item for item in reply_controls)


@pytest.mark.asyncio
async def test_user_action_reference_survives_later_proactive_action() -> None:
    ws = FakeWebSocket()
    client = ScriptedChildFusionClient(["A123", "A124", "A123"])
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-user-action-reference",
            "language": "zh",
            "modalities": ["action"],
            "action_candidates": fusion_catalog(),
        }
    )

    await session.handle_turn_start(user_turn_start("turn-user-wave"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-user-wave", text="向我挥挥手")
    )
    await asyncio.wait_for(session.active_turn.inference_task, timeout=1)

    await session.handle_turn_start(
        {
            "type": "turn.start",
            "turn_id": "turn-proactive-wave",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "action_finished",
        }
    )
    await session._dispatch_turn_commit(
        {
            "type": "turn.commit",
            "turn_id": "turn-proactive-wave",
            "turn_origin": "proactive",
            "text_role": "character_reply",
            "trigger": "action_finished",
            "text": "我们继续聊吧。",
        }
    )
    await asyncio.wait_for(session.active_turn.inference_task, timeout=1)

    assert session.last_executed_action is not None
    assert session.last_executed_action.turn_id == "turn-proactive-wave"
    assert session.last_executed_action.candidate_id == "A124"
    assert session.last_executed_action.turn_origin == "proactive"
    assert session.last_user_executed_action is not None
    assert session.last_user_executed_action.turn_id == "turn-user-wave"
    assert session.last_user_executed_action.candidate_id == "A123"
    assert session.last_user_executed_action.turn_origin == "user"

    await session.handle_turn_start(user_turn_start("turn-repeat-user-action"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-repeat-user-action", text="做一下刚刚那个动作")
    )
    await asyncio.wait_for(session.active_turn.inference_task, timeout=1)

    category_request = client.score_requests[-2]
    history_content = category_request.history[0]["content"]
    assert "[当前实际动作状态]" in history_content
    assert "candidate_id=A124" in history_content
    assert "[最近一次用户触发动作]" in history_content
    assert "candidate_id=A123" in history_content
    assert (
        "不得用后来由数字人主动触发的动作替代"
        in category_request.system_prompt
    )
    result = next(
        event
        for event in ws.events
        if event.get("type") == "turn.result"
        and event.get("turn_id") == "turn-repeat-user-action"
    )
    assert result["action"]["candidate_id"] == "A123"


@pytest.mark.asyncio
async def test_reply_without_instructions_has_no_server_system_prompt() -> None:
    ws = FakeWebSocket()
    client = FusionFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-no-reply-instructions",
            "action_candidates": fusion_catalog(),
        }
    )
    await session.handle_turn_start(user_turn_start("turn-no-reply-instructions"))
    await session._dispatch_turn_commit(
        user_turn_commit(
            "turn-no-reply-instructions",
            text="你好。",
        )
    )
    await asyncio.wait_for(session.active_turn.inference_task, timeout=1)

    reply_request = client.reply_requests[0]
    assert all(message.role != "system" for message in reply_request.messages)
    assert reply_request.messages[0].role == "user"
    assert reply_request.messages[0].content == [
        {"type": "text", "text": "你好。"},
        {
            "type": "text",
            "text": (
                "No user-camera image is provided in this interaction. "
                "Do not claim to see the user or make judgments based on "
                "the user's appearance."
            ),
        },
    ]


@pytest.mark.asyncio
async def test_reply_uses_latest_user_camera_and_drops_images_from_history() -> None:
    session = make_session(FakeWebSocket(), FakeClient())
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-reply-image-isolation",
            "language": "zh",
            "modalities": ["text"],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-reply-image-isolation"))
    turn = session.active_turn
    assert turn is not None
    turn.text = "看看我"

    request, forwarded_roles = session._build_reply_request(
        turn,
        ["audio-current"],
        ["user-camera-old", "avatar-current", "user-camera-current"],
        ["user_camera", "avatar_state", "user_camera"],
        None,
    )
    assert request.sampling.temperature == 0.4
    assert request.sampling.top_p == 1.0
    assert forwarded_roles == ["user_camera"]
    assert request.metadata["images"] == ["user-camera-current"]
    current_content = request.messages[-1].content
    assert current_content == [
        {
            "type": "text",
            "text": (
                "[用户摄像头画面，仅作为回答当前问题时的视觉依据；"
                "不要主动描述正在观看用户]"
            ),
        },
        {"type": "image"},
        {"type": "audio"},
        {"type": "text", "text": "看看我"},
    ]
    assert "avatar-current" not in request.metadata["images"]
    assert "user-camera-old" not in request.metadata["images"]

    action_context = session._build_bounded_action_context(
        ["audio-current"],
        ["user-camera-current", "avatar-current"],
        ["user_camera", "avatar_state"],
        include_history=False,
    )
    assert action_context[3] == ["user-camera-current", "avatar-current"]
    assert action_context[4] == ["user_camera", "avatar_state"]

    session._append_reply_history(
        turn,
        ["audio-history"],
        ["user-camera-history", "avatar-history"],
        ["user_camera", "avatar_state"],
        "历史回复",
    )
    history_turn = session.reply_history_turns[-1]
    assert history_turn.images == []
    assert history_turn.image_roles == []
    assert history_turn.messages[0]["content"] == [
        {"type": "audio"},
        {"type": "text", "text": "看看我"},
    ]

    next_request, next_forwarded_roles = session._build_reply_request(
        turn,
        [],
        ["avatar-next"],
        ["avatar_state"],
        None,
    )
    assert next_forwarded_roles == []
    assert next_request.metadata["images"] == []
    assert next_request.messages[-1].content[-1] == {
        "type": "text",
        "text": (
            "本轮未提供用户摄像头画面，不能声称看见用户或"
            "根据用户外观作出判断。"
        ),
    }
    assert "avatar-history" not in next_request.metadata["images"]
    assert "avatar-next" not in next_request.metadata["images"]


@pytest.mark.asyncio
async def test_reply_image_isolation_is_visible_in_diagnostic_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    client = FusionFakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-reply-image-diagnostics",
            "language": "zh",
            "modalities": ["text"],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-reply-image-diagnostics"))
    turn = session.active_turn
    assert turn is not None
    turn.phase = "processing"
    turn.request_base = "reply-image-diagnostics"

    await session._run_generated_reply(
        turn,
        [],
        ["prepared-avatar-image"],
        ["avatar_state"],
        None,
    )

    assert client.reply_requests[0].metadata["images"] == []
    logical_input = next(
        record
        for record in structured_records
        if record["event"] == "reply_logical_input"
    )
    assert logical_input["received_image_roles"] == ["avatar_state"]
    assert logical_input["reply_forwarded_image_roles"] == []
    assert logical_input["reply_filtered_avatar_image_count"] == 1
    assert logical_input["reply_filtered_stale_user_camera_image_count"] == 0
    assert logical_input["user_camera_present"] is False


@pytest.mark.asyncio
async def test_proactive_generated_reply_keeps_reply_and_action_prompts_isolated() -> None:
    ws = FakeWebSocket()
    client = FusionFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-proactive-generated",
            "language": "zh",
            "instructions": "回复要简洁亲切。",
            "action_candidates": fusion_catalog(),
        }
    )
    start = {
        "type": "turn.start",
        "turn_id": "turn-proactive-generated",
        "turn_origin": "proactive",
        "text_role": "character_reply",
        "trigger": "user_returned",
    }
    await session.handle_turn_start(start)
    await session.handle_turn_commit(
        {
            **start,
            "type": "turn.commit",
            "reply_context": "用户刚刚回来，自然地欢迎用户。",
            "avatar_state": {
                "current_action_id": "idle",
                "state_description": "避免重复上一个动作。",
            },
        }
    )

    reply_request = client.reply_requests[0]
    assert "回复要简洁亲切" in reply_request.messages[0].content
    reply_controls = [
        part["text"]
        for part in reply_request.messages[-1].content
        if part.get("type") == "text"
    ]
    assert "用户刚刚回来，自然地欢迎用户。" in reply_controls
    assert all("避免重复上一个动作" not in item for item in reply_controls)
    for action_request in client.score_requests:
        assert "回复要简洁亲切" not in action_request.system_prompt
        assert "用户刚刚回来" not in action_request.prefix
        assert action_request.avatar_state["state_description"] == "避免重复上一个动作。"
        assert "[本轮主动场景约束优先级]" in action_request.prefix
        assert action_request.prefix.rfind(
            "[本轮主动场景约束优先级]"
        ) > action_request.prefix.rfind("本轮主动场景约束中给出的目标")
        assert action_request.prefix.rfind(
            "[本轮主动场景约束优先级]"
        ) < action_request.prefix.rfind("最合适的")
    category_request, child_request = client.score_requests
    assert "默认动作类别仅在不与该约束冲突" in category_request.prefix
    assert "任何违反明确禁止项的 candidate_id 都不得选择" in child_request.prefix


@pytest.mark.asyncio
async def test_proactive_provided_text_skips_reply_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    ws = FakeWebSocket()
    client = FusionFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-provided-reply",
            "instructions": "回复保持简短。",
            "action_candidates": fusion_catalog(),
        }
    )
    start = {
        "type": "turn.start",
        "turn_id": "turn-provided",
        "turn_origin": "proactive",
        "text_role": "character_reply",
        "trigger": "user_returned",
    }
    await session.handle_turn_start(start)
    await session.handle_turn_commit(
        {
            **start,
            "type": "turn.commit",
            "text": "你回来啦，今天过得怎么样？",
            "avatar_state": {
                "state_description": "选择轻量友好的欢迎动作。"
            },
        }
    )

    assert client.reply_requests == []
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["reply"] == {
        "text": "你回来啦，今天过得怎么样？",
        "source": "provided",
    }
    assert result["action"]["action_id"] == "wave"
    provided = next(
        record
        for record in structured_records
        if record["event"] == "provided_reply_used"
    )
    assert provided["instructions_applied"] is False
    assert provided["instructions_present"] is True
    assert provided["instructions_chars"] == len("回复保持简短。")
    assert provided["instructions_sha256"].startswith("sha256:")


@pytest.mark.asyncio
async def test_fusion_cancel_aborts_reply_and_child_requests() -> None:
    resource_requests: list[tuple[str, dict]] = []

    def request_sample(sample_trigger: str, **fields) -> bool:
        resource_requests.append((sample_trigger, fields))
        return True

    ws = FakeWebSocket()
    client = BlockingFusionClient()
    session = make_session(
        ws,
        client,
        resource_sample_requester=request_sample,
    )
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-fusion-cancel",
            "action_candidates": fusion_catalog(),
        }
    )
    await session.handle_turn_start(user_turn_start("turn-fusion-cancel"))
    await session._dispatch_turn_commit(
        user_turn_commit("turn-fusion-cancel", text="向我打招呼")
    )
    await asyncio.wait_for(client.reply_started.wait(), timeout=1)
    await asyncio.wait_for(client.child_started.wait(), timeout=1)

    await session.handle_turn_cancel(
        {"type": "turn.cancel", "turn_id": "turn-fusion-cancel"}
    )

    assert len(client.aborted) == 2
    assert any(request_id.endswith("-reply") for request_id in client.aborted)
    assert any(request_id.endswith("-child") for request_id in client.aborted)
    assert any(event["type"] == "turn.cancelled" for event in ws.events)
    assert not any(event["type"] == "turn.result" for event in ws.events)
    assert session.reply_history_turns == []
    assert [trigger for trigger, _ in resource_requests] == [
        "turn_before_inference",
        "turn_after_terminal",
    ]
    assert resource_requests[-1][1]["turn_outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_child_failure_keeps_reply_and_returns_partial_no_action() -> None:
    resource_requests: list[tuple[str, dict]] = []

    def request_sample(sample_trigger: str, **fields) -> bool:
        resource_requests.append((sample_trigger, fields))
        return True

    ws = FakeWebSocket()
    client = FailingChildFusionClient()
    session = make_session(
        ws,
        client,
        resource_sample_requester=request_sample,
    )
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-child-failure",
            "action_candidates": fusion_catalog(),
        }
    )
    await session.handle_turn_start(user_turn_start("turn-child-failure"))
    await session.handle_turn_commit(
        user_turn_commit("turn-child-failure", text="向我打招呼")
    )

    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["status"] == "partial"
    assert [trigger for trigger, _ in resource_requests] == [
        "turn_before_inference",
        "turn_after_terminal",
    ]
    assert resource_requests[-1][1]["turn_outcome"] == "partial"
    assert result["modalities"] == {"text": "completed", "action": "failed"}
    assert result["reply"]["text"] == "你好呀，今天过得怎么样？"
    assert result["action"]["action_id"] == "no_action"
    assert result["action"]["execute"] is False
    assert result["errors"] == {
        "action": {"message": "synthetic child scoring failure"}
    }


@pytest.mark.asyncio
async def test_text_only_session_does_not_require_action_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured_records: list[dict] = []

    def capture_structured_log(log_type, event, **fields):
        structured_records.append(
            {"log_type": log_type, "event": event, **fields}
        )
        return True

    monkeypatch.setattr(
        multimodal_module, "emit_structured_log", capture_structured_log
    )
    monkeypatch.setenv(FULL_INSTRUCTIONS_LOG_ENV, "1")
    ws = FakeWebSocket()
    client = FakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "session_id": "session-text-only",
            "modalities": ["text"],
            "language": "zh",
            "instructions": "简洁回复用户。",
        }
    )
    await session.handle_turn_start(user_turn_start("turn-text-only"))
    await session.handle_turn_commit(
        user_turn_commit("turn-text-only", text="你好")
    )

    assert len(client.chat_requests) == 1
    assert client.score_requests == []
    session_start = next(
        record
        for record in structured_records
        if record["event"] == "session_start_received"
    )
    assert session_start["instructions_provided"] is True
    assert session_start["instructions_present"] is True
    assert session_start["instructions_chars"] == len("简洁回复用户。")
    assert session_start["instructions_sha256"].startswith("sha256:")
    instructions_record = next(
        record
        for record in structured_records
        if record["event"] == "session_instructions_received"
    )
    assert instructions_record["instructions"] == "简洁回复用户。"
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["reply"]["text"] == "好的，我来看看。"
    assert "action" not in result
    logical_input = next(
        record
        for record in structured_records
        if record["event"] == "reply_logical_input"
    )
    assert logical_input["messages"][0]["role"] == "system"
    assert logical_input["messages"][0]["content"] == "简洁回复用户。"
    assert logical_input["instructions_applied"] is True
    assert logical_input["effective_system_prompt"] == "简洁回复用户。"
    assert logical_input["system_prompt_chars"] == len("简洁回复用户。")
    assert logical_input["system_prompt_sha256"].startswith("sha256:")
    assert "本 Session 未启用动作输出" not in str(logical_input["messages"])
    assert "请仅通过自然语言完成本轮交流" not in str(logical_input["messages"])
    assert logical_input["current_audio"] == []
    assert logical_input["current_images"] == []
    assert logical_input["received_image_roles"] == []
    assert logical_input["reply_forwarded_image_roles"] == []
    assert logical_input["reply_filtered_avatar_image_count"] == 0
    assert logical_input["user_camera_present"] is False
    completed = next(
        record
        for record in structured_records
        if record["event"] == "reply_completed"
    )
    assert completed["output_text"] == "好的，我来看看。"
    assert completed["finish_reason"] == "stop"
    assert completed["usage"] is None
    assert 0 <= completed["created_after_commit_ms"]
    assert (
        completed["created_after_commit_ms"]
        <= completed["first_delta_after_commit_ms"]
        <= completed["text_done_after_commit_ms"]
        <= completed["response_done_after_commit_ms"]
    )
    assert completed["delta_count"] == 1
    assert completed["completion_tokens"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("modalities", "message"),
    [
        ([], "non-empty list"),
        (["text", "text"], "must not contain duplicates"),
        (["audio"], "unsupported output modalities: audio"),
    ],
)
async def test_session_rejects_invalid_output_modalities(
    modalities: list[str], message: str
) -> None:
    session = make_session(FakeWebSocket(), FusionFakeClient())
    with pytest.raises(ValueError, match=message):
        await session.handle_session_start(
            {
                "type": "session.start",
                "session_id": "session-invalid-modalities",
                "modalities": modalities,
            }
        )


@pytest.mark.asyncio
async def test_default_modalities_require_action_catalog() -> None:
    session = make_session(FakeWebSocket(), FusionFakeClient())
    with pytest.raises(ValueError, match="when action modality is enabled"):
        await session.handle_session_start(
            {"type": "session.start", "session_id": "session-default-no-catalog"}
        )


@pytest.mark.asyncio
async def test_fusion_rejects_flat_children_until_todo_is_implemented() -> None:
    session = make_session(
        FakeWebSocket(), FusionFakeClient(), action_selection_mode="flat_children"
    )
    with pytest.raises(ValueError, match="requires selection_mode=hierarchical"):
        await session.handle_session_start(
            {
                "type": "session.start",
                "session_id": "session-flat-fusion",
                "action_candidates": fusion_catalog(),
            }
        )


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
            "modalities": ["action"],
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
    await session._dispatch_turn_commit(
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
        "type": "session.start", "modalities": ["action"], "session_id": "session-nested",
        "language": "zh",
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
    assert category_request.prefix_cache_namespace == session.action_prefix_cache_namespace
    assert child_request.prefix_cache_namespace == (
        f"{session.action_prefix_cache_namespace}:child:B1"
    )
    assert category_request.micro_batch_size == 64
    assert child_request.micro_batch_size == 64
    assert (
        "category_id=B1｜类别=基础姿态｜说明=姿态变化"
        in category_request.system_prompt
    )
    assert (
        "category_id=B2｜类别=重心变化｜说明=重心变化"
        in category_request.system_prompt
    )
    assert (
        "candidate_id=A1｜动作=正式站立｜说明=站立"
        in child_request.system_prompt
    )
    assert (
        "candidate_id=A0｜动作=不做动作｜说明=保持当前姿态"
        in child_request.system_prompt
    )
    assert child_request.system_prompt.count("已选类别：category_id=B1") == 1
    assert "已选择 category_id=" not in child_request.prefix
    assert "candidate_id=A0" not in category_request.prefix
    assert "category_id=B1" not in child_request.prefix
    assert "选择与输入和状态约束最匹配的候选项" in category_request.prefix
    assert "选择与输入和状态约束最匹配的候选项" in child_request.prefix
    assert "兜底 category_id=B1" in category_request.system_prompt
    assert category_request.system_prompt.index("兜底 category_id=B1") < (
        category_request.system_prompt.index("固定类别集合如下：")
    )
    assert category_request.system_prompt.endswith(
        "请根据当前输入选择最匹配的 category_id；只输出一个 category_id，"
        "输出后立即结束，不要解释。"
    )
    assert "不得引入其他类别或系统兜底动作" in child_request.system_prompt
    assert category_request.prefix.removesuffix("最合适的 category_id：") == (
        child_request.prefix.removesuffix("最合适的 candidate_id：")
    )
    assert "category_id=B2｜类别=重心变化" not in child_request.system_prompt
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "A1"
    assert result["media_summary"]["action_context"]["selection_stages"] == 2
    assert result["media_summary"]["action_context"]["selected_category_id"] == "B1"
    assert result["media_summary"]["action_context"]["selected_category_ids"] == ["B1"]
    assert result["media_summary"]["action_context"]["category_top_k"] == 1


@pytest.mark.asyncio
async def test_hierarchical_single_child_skips_child_without_global_fallback() -> None:
    ws = FakeWebSocket()
    client = NestedPrefillFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-single-child-fast-path",
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
                        }
                    ],
                },
                {
                    "category_id": "B0",
                    "source_label": "静止",
                    "short_definition": "不做动作",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前状态",
                        }
                    ],
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-single-child"))
    await session.handle_turn_commit(
        user_turn_commit("turn-single-child", text="向用户问好")
    )

    assert len(client.score_requests) == 1
    assert client.score_requests[0].stage == "category"
    assert [item["stage"] for item in client.prefill_requests] == ["category"]
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"] == {
        "action_id": "wave",
        "candidate_id": "A1",
        "category_id": "B1",
        "execute": True,
    }
    assert result["timing"]["server_result_finalize_ms"] >= 0.0
    assert result["timing"]["server_total_after_commit_ms"] >= 0.0
    breakdown = result["timing"]["action_breakdown"]
    assert breakdown["category"]["client"]["total_ms"] == 0.0
    assert breakdown["child"] == {"skipped": True, "reason": "single_child"}
    assert breakdown["child_catalog_prefill_ms"] == 0.0


@pytest.mark.asyncio
async def test_hierarchical_single_no_action_child_returns_execute_false() -> None:
    ws = FakeWebSocket()
    client = NestedFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-single-no-action",
            "action_candidates": [
                {
                    "category_id": "B0",
                    "source_label": "静止",
                    "short_definition": "不做动作",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前状态",
                        }
                    ],
                },
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
                        }
                    ],
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-single-none"))
    await session.handle_turn_commit(user_turn_commit("turn-single-none"))

    assert len(client.score_requests) == 1
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert result["action"]["action_id"] == "no_action"
    assert result["action"]["execute"] is False


@pytest.mark.asyncio
async def test_hierarchical_single_child_keeps_scoring_for_diagnostics() -> None:
    ws = FakeWebSocket()
    client = NestedFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-single-child-diagnostics",
            "include_scores": True,
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
                        }
                    ],
                },
                {
                    "category_id": "B0",
                    "source_label": "静止",
                    "short_definition": "不做动作",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前状态",
                        }
                    ],
                },
            ],
        }
    )
    await session.handle_turn_start(user_turn_start("turn-single-diagnostics"))
    await session.handle_turn_commit(user_turn_commit("turn-single-diagnostics"))

    assert [request.stage for request in client.score_requests] == [
        "category",
        "child",
    ]
    result = next(event for event in ws.events if event["type"] == "turn.result")
    assert len(result["scores"]) == 1
    assert result["timing"]["action_breakdown"]["child"]["server_total_ms"] == 0.0

@pytest.mark.asyncio
async def test_hierarchical_proactive_turn_uses_character_reply_in_both_stages() -> None:
    client = NestedFakeClient()
    session = make_session(FakeWebSocket(), client)
    await session.handle_session_start(
        {
            "type": "session.start",
                "modalities": ["action"],
                "session_id": "session-nested-proactive",
                "language": "zh",
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
        assert "本轮没有可用的数字人当前状态信息" in request.prefix
        assert "以结构化 数字人当前状态信息为准" not in request.prefix
        assert "该文本的语义、语气和表达目标直接相关" in request.prefix
        assert "本轮主动场景约束中给出的目标" not in request.prefix
        assert "[本轮主动场景约束优先级]" not in request.prefix
        assert "结合历史动作判断衔接关系" in request.prefix
        assert "选择与表达目标和状态约束最匹配的候选项" in request.prefix
        assert "candidate_id=A0" not in request.prefix
        assert request.prefix.count("Hello，你回来啦！") == 0
    assert "已选择 category_id=" not in child_request.prefix
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
        "modalities": ["action"],
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
        "type": "session.start", "modalities": ["action"], "session_id": "session-nested-top-k",
        "language": "zh",
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
    assert (
        "已选类别：category_id=B1｜类别=基础姿态｜说明=姿态变化"
        in child_request.system_prompt
    )
    assert (
        "已选类别：category_id=B2｜类别=重心变化｜说明=重心变化"
        in child_request.system_prompt
    )
    assert (
        "candidate_id=A15｜动作=重心左移｜说明=左移"
        in child_request.system_prompt
    )
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
            "modalities": ["action"],
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
            "modalities": ["action"],
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
        "modalities": ["action"],
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
    assert (
        "candidate_id=A1｜动作=正式站立｜说明=站立"
        in request.system_prompt
    )
    assert (
        "candidate_id=A15｜动作=重心左移｜说明=左移"
        in request.system_prompt
    )
    assert "category_id=B1｜类别=基础姿态" not in request.system_prompt

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
            "modalities": ["action"],
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
            "modalities": ["action"],
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
        "modalities": ["action"],
        "session_id": "session-prefill-hierarchical",
        "language": "zh",
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
    assert (
        "category_id=B1｜类别=基础姿态｜说明=姿态变化"
        in prefill["system_prompt"]
    )
    assert "candidate_id=A1" not in prefill["system_prompt"]
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_prefix_prefilled"] is True


@pytest.mark.asyncio
async def test_session_start_prewarms_selected_child_catalog_without_fallback() -> None:
    ws = FakeWebSocket()
    client = NestedPrefillFakeClient()
    session = make_session(ws, client)
    await session.handle_session_start(
        {
            "type": "session.start",
            "modalities": ["action"],
            "session_id": "session-prewarm-child",
            "prewarm_child_category_ids": ["B1"],
            "action_candidates": [
                {
                    "category_id": "B1",
                    "source_label": "问候",
                    "short_definition": "问候动作",
                    "children": [
                        {
                            "candidate_id": "A1",
                            "action_id": "wave_left",
                            "source_label": "左手问候",
                            "short_definition": "使用左手问候",
                        },
                        {
                            "candidate_id": "A2",
                            "action_id": "wave_both",
                            "source_label": "双手问候",
                            "short_definition": "使用双手问候",
                        },
                    ],
                },
                {
                    "category_id": "B0",
                    "source_label": "系统动作",
                    "short_definition": "系统兜底",
                    "children": [
                        {
                            "candidate_id": "A0",
                            "action_id": "no_action",
                            "source_label": "不做动作",
                            "short_definition": "保持当前姿态",
                        }
                    ],
                },
            ],
        }
    )

    assert [item["stage"] for item in client.prefill_requests] == [
        "category",
        "child",
    ]
    child_prewarm = client.prefill_requests[1]
    assert child_prewarm["request_id"] == "session-session-prewarm-child-child-prewarm-B1"
    assert [item.candidate_id for item in child_prewarm["candidates"]] == [
        "A1",
        "A2",
    ]
    assert "candidate_id=A0" not in child_prewarm["system_prompt"]
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["prewarmed_child_category_ids"] == ["B1"]

    await session.handle_turn_start(user_turn_start("turn-prewarmed-child"))
    await session.handle_turn_commit(
        user_turn_commit("turn-prewarmed-child", text="向用户问好")
    )
    assert [item["stage"] for item in client.prefill_requests] == [
        "category",
        "child",
    ]
    child_request = client.score_requests[-1]
    assert child_request.stage == "child"
    assert [item.candidate_id for item in child_request.candidates] == [
        "A1",
        "A2",
    ]


@pytest.mark.asyncio
async def test_session_start_rejects_unknown_child_prewarm_category() -> None:
    session = make_session(FakeWebSocket(), NestedPrefillFakeClient())
    with pytest.raises(ValueError, match="unknown category IDs: B9"):
        await session.handle_session_start(
            {
                "type": "session.start",
                "modalities": ["action"],
                "session_id": "session-unknown-child-prewarm",
                "prewarm_child_category_ids": ["B9"],
                "action_candidates": [
                    {
                        "category_id": "B0",
                        "source_label": "系统动作",
                        "short_definition": "系统兜底",
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


@pytest.mark.asyncio
async def test_session_start_prefills_flat_children_without_category_stage() -> None:
    ws = FakeWebSocket()
    client = PrefillFakeClient()
    session = make_session(ws, client, action_selection_mode="flat_children")
    await session.handle_session_start({
        "type": "session.start",
        "language": "zh",
        "modalities": ["action"],
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
    assert "candidate_id=A1｜动作=站立｜说明=站立" in prefill["system_prompt"]
    assert "category_id=B1｜" not in prefill["system_prompt"]
    started = next(event for event in ws.events if event["type"] == "session.started")
    assert started["action_selection_mode"] == "flat_children"
    assert started["action_selection_stages"] == 1
    assert started["action_prefix_prefilled"] is True
