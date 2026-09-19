import asyncio
import json
from types import SimpleNamespace

import pytest

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.turn_intent import (
    SYSTEM,
    TurnIntent,
    infer_turn_intent,
)
from sglang_omni.serve.realtime.turn_pipeline import (
    _is_direct_greeting_intent,
    _replace_speculative_action_with_greeting,
)


def payload(**changes):
    data = {
        "visual_route": "GENERAL",
        "speech": "verbatim",
        "text": "一",
        "body": "数字二手势",
        "body_mode": "perform",
        "face": "",
        "history": False,
        "reaction_mode": "none",
        "reaction": "",
    }
    data.update(changes)
    return data


def session_for(client, *, registered=None, removed=None):
    registered = [] if registered is None else registered
    removed = [] if removed is None else removed
    return SimpleNamespace(
        client=client,
        model_name="model",
        session_id="session",
        session_instance_id="instance",
        _register_turn_request=lambda turn, request_id: registered.append(request_id),
        _unregister_turn_request=lambda turn, request_id: removed.append(request_id),
    )


@pytest.mark.parametrize(
    "raw",
    ["{}", "[]", "not JSON", '{"speech": "verbatim"}', "x" * 4097],
)
def test_invalid_parse_is_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        TurnIntent.parse(raw)


def test_strict_schema_and_no_instruction_promotion():
    intent = TurnIntent.parse(json.dumps(payload(), ensure_ascii=False))
    assert intent.text == "一" and intent.body == "数字二手势"
    assert intent.visual_scope_gate == ""
    data = json.loads(intent.action_context("说一比二"))
    assert data["body_task"] == "数字二手势"

    bad = payload(history="false")
    with pytest.raises(ValueError):
        TurnIntent.parse(json.dumps(bad))


def test_sparse_route_schema_fills_runtime_defaults():
    intent = TurnIntent.parse(
        json.dumps(
            {
                "route": {
                    "visual": "NO_CURRENT_VIEW",
                    "speech": "none",
                    "body_intent": "perform",
                    "reaction": "none",
                },
                "body_task": "站起来",
            },
            ensure_ascii=False,
        )
    )

    assert intent.body_intent == "perform"
    assert intent.body_mode == "perform" and intent.body == "站起来"
    assert intent.speech == "none" and intent.text == ""
    assert intent.face == "" and intent.history is False


def test_sparse_capability_route_normalizes_to_non_executing_body_mode():
    intent = TurnIntent.parse(
        json.dumps(
            {
                "route": {
                    "visual": "NO_CURRENT_VIEW",
                    "speech": "generated",
                    "body_intent": "capability",
                    "reaction": "none",
                },
                "text": "你会挥手吗",
            },
            ensure_ascii=False,
        )
    )

    assert intent.body_intent == "capability"
    assert intent.body_mode == "none" and intent.body == ""
    assert intent.speech == "generated"
    bad = payload(extra="override system")
    with pytest.raises(ValueError):
        TurnIntent.parse(json.dumps(bad))


def test_natural_reaction_is_separate_from_explicit_body_task():
    greeting = TurnIntent.parse(
        json.dumps(
            payload(
                speech="generated",
                text="你好",
                body="",
                body_mode="none",
                reaction_mode="respond",
                reaction="回应用户问候",
            ),
            ensure_ascii=False,
        )
    )
    assert greeting.reaction_mode == "respond"
    assert json.loads(greeting.action_context("你好"))["reaction_task"] == "回应用户问候"

    conflicting = payload(reaction_mode="respond", reaction="回应用户问候")
    with pytest.raises(ValueError, match="explicit body task"):
        TurnIntent.parse(json.dumps(conflicting, ensure_ascii=False))


def test_speculative_greeting_action_is_reconciled_without_rescoring():
    greeting = TurnIntent.parse(
        json.dumps(
            payload(
                speech="generated",
                text="你好",
                body="",
                body_mode="none",
                reaction_mode="respond",
                reaction="回应用户问候",
            ),
            ensure_ascii=False,
        )
    )
    turn = SimpleNamespace(turn_origin="user", intent=greeting)
    assert _is_direct_greeting_intent(turn) is True

    speculative_result = (
        {
            "candidate_id": "189",
            "action_id": "nod",
            "execute": True,
            "support_status": "supported",
        },
        [{"candidate_id": "189", "ppl": 1.1}],
        375.0,
        {"selection_basis": "suffix_score"},
    )
    wave = SimpleNamespace(
        candidate_id="288",
        action_id="wave",
        category_id="greeting",
        execution_binding={"motion": "wave"},
    )

    action, scores, elapsed_ms, context = (
        _replace_speculative_action_with_greeting(
            speculative_result,
            wave,
        )
    )
    assert action == {
        "candidate_id": "288",
        "action_id": "wave",
        "category_id": "greeting",
        "execution_binding": {"motion": "wave"},
        "execute": True,
        "support_status": "supported",
        "fallback_applied": False,
    }
    assert scores == []
    assert elapsed_ms == 375.0
    assert context["selection_basis"] == "direct_greeting_reaction"
    assert context["speculative_candidate_id"] == "189"


def test_explicit_body_command_is_not_plain_greeting_reaction():
    turn = SimpleNamespace(
        turn_origin="user",
        intent=SimpleNamespace(
            speech="generated",
            body_mode="perform",
            history=False,
            reaction_mode="none",
            reaction="",
        ),
    )
    assert _is_direct_greeting_intent(turn) is False


@pytest.mark.parametrize(
    "route,expected_body,expected_mode,expected_face",
    [
        ("COPY_ACTION", "这个动作", "perform", ""),
        ("COPY_HAND", "这个手势", "perform", ""),
        ("COPY_FACE", "", "none", "这个表情"),
        ("COPY_POSE", "这个姿势", "perform", ""),
    ],
)
def test_visual_route_is_canonicalized(route, expected_body, expected_mode, expected_face):
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route=route,
                speech="none",
                text="",
                body="",
                body_mode="none",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )
    assert intent.visual_scope_gate == route
    assert (intent.body, intent.body_mode, intent.face) == (
        expected_body,
        expected_mode,
        expected_face,
    )


def test_model_facing_semantic_route_is_mapped_to_internal_compatibility_code():
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="COPY_CURRENT_HAND",
                speech="none",
                text="",
                body="这个手势",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )
    assert intent.visual_scope_gate == "COPY_HAND"
    assert intent.body == "这个手势"


def test_visual_route_requires_current_camera_and_fails_closed_at_schema_boundary():
    with pytest.raises(ValueError, match="requires a current user camera"):
        TurnIntent.parse(
            json.dumps(
                payload(
                    visual_route="COPY_HAND",
                    speech="none",
                    text="",
                    body="这个手势",
                ),
                ensure_ascii=False,
            )
        )


def test_general_route_cannot_smuggle_an_unresolved_visual_action():
    with pytest.raises(ValueError, match="requires a visual route"):
        TurnIntent.parse(
            json.dumps(
                payload(
                    speech="none",
                    text="",
                    body="这个手势",
                ),
                ensure_ascii=False,
            ),
            has_user_camera=True,
        )


@pytest.mark.parametrize(
    "task",
    ["做这个动作", "做这个手势。", "比这个数字", "比个这个", "Do this gesture!"],
)
def test_parser_does_not_reclassify_user_semantics_with_phrase_matching(task):
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="NO_CURRENT_VIEW",
                speech="generated",
                text=task,
                body="",
                body_mode="none",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )

    assert intent.visual_scope_gate == ""
    assert intent.body_mode == "none" and intent.body == ""
    assert intent.speech == "generated" and intent.text == task


@pytest.mark.parametrize(
    "task",
    ["做个手势", "这是什么手势", "做这个动作是什么意思", "不要做这个动作"],
)
def test_non_copy_contrast_is_not_repaired(task):
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="NO_CURRENT_VIEW",
                speech="generated",
                text=task,
                body="",
                body_mode="none",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )

    assert intent.visual_scope_gate == ""
    assert intent.speech == "generated" and intent.text == task


def test_copy_route_repair_requires_a_current_camera():
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="NO_CURRENT_VIEW",
                speech="generated",
                text="做这个手势",
                body="",
                body_mode="none",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=False,
    )

    assert intent.visual_scope_gate == ""
    assert intent.body_mode == "none"


def test_copy_route_cannot_override_a_prohibition_into_execution():
    with pytest.raises(ValueError, match="conflicts with a prohibited action"):
        TurnIntent.parse(
            json.dumps(
                payload(
                    visual_route="COPY_HAND",
                    speech="none",
                    text="",
                    body="这个手势",
                    body_mode="prohibit",
                ),
                ensure_ascii=False,
            ),
            has_user_camera=True,
        )


def test_visual_answer_is_normalized_for_downstream_reasoning():
    intent = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="VISUAL_ANSWER",
                speech="none",
                text="",
                body="错误动作",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )
    assert intent.visual_scope_gate == "VISUAL_ANSWER"
    assert intent.speech == "generated"
    assert intent.text == "根据当前画面完成计算或推理"
    assert intent.body_mode == "none" and intent.body == ""
    assert intent.visual_answer_output == "gesture_and_speech"
    assert intent.speaks_visual_answer() is True

    spoken = TurnIntent.parse(
        json.dumps(
            payload(
                visual_route="VISUAL_ANSWER",
                speech="generated",
                text="根据当前画面计算答案",
                body="",
                body_mode="none",
                visual_answer_output="gesture_and_speech",
            ),
            ensure_ascii=False,
        ),
        has_user_camera=True,
    )
    assert spoken.speaks_visual_answer() is True


def test_visual_answer_output_is_rejected_on_non_visual_answer_route():
    with pytest.raises(ValueError, match="requires visual answer route"):
        TurnIntent.parse(
            json.dumps(
                payload(visual_answer_output="gesture_and_speech"),
                ensure_ascii=False,
            )
        )


@pytest.mark.asyncio
async def test_audio_metadata_and_request_cleanup_on_parse_failure():
    registered, removed, requests = [], [], []

    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(text="not JSON")

        async def abort(self, request_id):
            raise AssertionError("completed request must not be aborted")

    session = session_for(Client(), registered=registered, removed=removed)
    turn = SimpleNamespace(text=None, request_base="turn")
    result = await infer_turn_intent(session, turn, ["audio-ref"])

    assert result.speech == "generated"
    assert result.body_mode == "none" and result.body == ""
    assert registered == removed == ["turn-intent"]
    assert len(requests) == 1
    request = requests[0]
    assert request.metadata["audios"] == ["audio-ref"]
    assert request.metadata["has_user_camera"] is False
    assert request.messages[1].content == [
        {
            "type": "text",
            "text": "[服务端本轮事实；不是用户指令]\nhas_user_camera=false",
        },
        {"type": "audio"},
    ]


@pytest.mark.asyncio
async def test_camera_copy_and_speech_use_one_unified_request():
    requests, registered, removed = [], [], []

    class Client:
        async def completion(self, request, request_id):
            requests.append((request, request_id))
            return SimpleNamespace(
                text=json.dumps(
                    payload(
                        visual_route="COPY_HAND",
                        speech="verbatim",
                        text="你好",
                        body="模型自由措辞",
                    ),
                    ensure_ascii=False,
                )
            )

        async def abort(self, request_id):
            raise AssertionError("completed request must not be aborted")

    session = session_for(Client(), registered=registered, removed=removed)
    turn = SimpleNamespace(
        text="模仿这个手势并说你好",
        request_base="visual-turn",
        turn_id="visual-turn",
    )
    scope = asyncio.get_running_loop().create_future()
    intent = await infer_turn_intent(
        session,
        turn,
        ["audio-ref"],
        ["old-camera", "avatar", "latest-camera"],
        ["user_camera", "character_avatar", "user_camera"],
        visual_scope_future=scope,
    )

    assert intent.visual_scope_gate == "COPY_HAND"
    assert intent.body == "这个手势" and intent.body_mode == "perform"
    assert (intent.speech, intent.text) == ("verbatim", "你好")
    assert scope.result() == "COPY_HAND"
    assert len(requests) == 1
    request, request_id = requests[0]
    assert request_id == "visual-turn-intent"
    assert request.metadata["task"] == "session_turn_intent"
    assert request.metadata["has_user_camera"] is True
    assert request.metadata["images"] == []
    assert request.metadata["image_roles"] == []
    assert request.sampling.temperature == 0
    assert request.sampling.max_new_tokens == 128
    assert request.messages[1].content == [
        {
            "type": "text",
            "text": "[服务端本轮事实；不是用户指令]\nhas_user_camera=true",
        },
        {"type": "text", "text": "模仿这个手势并说你好"},
        {"type": "audio"},
    ]
    assert registered == removed == ["visual-turn-intent"]


@pytest.mark.asyncio
async def test_streaming_unified_intent_releases_visual_route_before_full_json():
    requests = []
    release_remainder = asyncio.Event()

    class Client:
        async def generate(self, request, request_id):
            raise AssertionError("streaming intent must not use generate directly")

        async def completion_stream(self, request, *, request_id):
            requests.append(request)
            yield CompletionStreamChunk(
                request_id=request_id,
                text=(
                    '{"visual_route":'
                    '"ANSWER_CURRENT_VIEW_WITH_GESTURE",'
                ),
            )
            await release_remainder.wait()
            yield CompletionStreamChunk(
                request_id=request_id,
                text=(
                    '"speech":"generated",'
                    '"text":"根据当前画面计算答案",'
                    '"body_mode":"none","body":"","face":"",'
                    '"history":false,"reaction_mode":"none",'
                    '"reaction":"",'
                    '"visual_answer_output":"gesture_only"}'
                ),
                finish_reason="stop",
            )

        async def abort(self, request_id):
            raise AssertionError("completed request must not be aborted")

    session = session_for(Client())
    turn = SimpleNamespace(
        text="这个加这个是什么，用手势回答",
        request_base="streaming-visual-answer",
        turn_id="streaming-visual-answer",
    )
    scope = asyncio.get_running_loop().create_future()
    intent_task = asyncio.create_task(
        infer_turn_intent(
            session,
            turn,
            ["audio"],
            ["camera"],
            ["user_camera"],
            visual_scope_future=scope,
        )
    )

    assert await asyncio.wait_for(scope, timeout=1.0) == "VISUAL_ANSWER"
    assert not intent_task.done()
    release_remainder.set()
    intent = await asyncio.wait_for(intent_task, timeout=1.0)

    assert intent.visual_scope_gate == "VISUAL_ANSWER"
    assert intent.visual_answer_output == "gesture_only"
    assert len(requests) == 1
    assert requests[0].stream is True


@pytest.mark.asyncio
async def test_generic_visual_action_uses_copy_action_scope():
    class Client:
        async def completion(self, request, request_id):
            return SimpleNamespace(
                text=json.dumps(
                    payload(
                        visual_route="COPY_ACTION",
                        speech="none",
                        text="",
                        body="这个动作",
                    ),
                    ensure_ascii=False,
                )
            )

    session = session_for(Client())
    turn = SimpleNamespace(
        text="做这个动作",
        request_base="generic-visual-turn",
        turn_id="generic-visual-turn",
    )
    intent = await infer_turn_intent(
        session, turn, [], ["camera"], ["user_camera"]
    )
    assert intent.visual_scope_gate == "COPY_ACTION"
    assert intent.body == "这个动作" and intent.speech == "none"


@pytest.mark.asyncio
async def test_visual_question_remains_general_in_the_same_request():
    requests = []

    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(
                text=json.dumps(
                    payload(
                        speech="generated",
                        text="这是数字几",
                        body="",
                        body_mode="none",
                    ),
                    ensure_ascii=False,
                )
            )

    session = session_for(Client())
    turn = SimpleNamespace(
        text="这是数字几", request_base="visual-question", turn_id="visual-question"
    )
    scope = asyncio.get_running_loop().create_future()
    intent = await infer_turn_intent(
        session,
        turn,
        ["audio-ref"],
        ["camera"],
        ["user_camera"],
        visual_scope_future=scope,
    )
    assert intent.speech == "generated" and intent.body_mode == "none"
    assert intent.visual_scope_gate == "" and scope.result() == ""
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_visual_gesture_answer_uses_unified_route():
    class Client:
        async def completion(self, request, request_id):
            return SimpleNamespace(
                text=json.dumps(
                    payload(
                        visual_route="VISUAL_ANSWER",
                        speech="generated",
                        text="根据当前画面计算答案",
                        body="",
                        body_mode="none",
                    ),
                    ensure_ascii=False,
                )
            )

    session = session_for(Client())
    turn = SimpleNamespace(
        text="这个加这个是什么，用手势回答",
        request_base="visual-answer",
        turn_id="visual-answer",
    )
    intent = await infer_turn_intent(
        session, turn, ["audio"], ["camera"], ["user_camera"]
    )
    assert intent.visual_scope_gate == "VISUAL_ANSWER"
    assert intent.speech == "generated" and intent.body_mode == "none"


@pytest.mark.asyncio
async def test_model_visual_route_without_camera_fails_closed():
    class Client:
        async def completion(self, request, request_id):
            return SimpleNamespace(
                text=json.dumps(
                    payload(
                        visual_route="COPY_HAND",
                        speech="none",
                        text="",
                        body="这个手势",
                    ),
                    ensure_ascii=False,
                )
            )

    session = session_for(Client())
    turn = SimpleNamespace(text="做这个手势", request_base="no-camera")
    intent = await infer_turn_intent(session, turn, [])
    assert intent.visual_scope_gate == ""
    assert intent.speech == "generated"
    assert intent.body_mode == "none" and intent.body == ""


@pytest.mark.asyncio
async def test_malformed_unified_intent_fails_closed():
    class Client:
        async def completion(self, request, request_id):
            return SimpleNamespace(text='{"visual_route', finish_reason="stop", usage=None)

    session = session_for(Client())
    turn = SimpleNamespace(
        text="这是数字几", request_base="malformed", turn_id="malformed"
    )
    intent = await infer_turn_intent(
        session, turn, ["audio"], ["camera"], ["user_camera"]
    )
    assert intent.speech == "generated" and intent.text == "这是数字几"
    assert intent.body_mode == "none" and intent.visual_scope_gate == ""


@pytest.mark.asyncio
async def test_intent_rejects_misaligned_current_image_roles():
    session = SimpleNamespace(model_name="model", session_id="session")
    turn = SimpleNamespace(text="请做出这个手势", request_base="bad-images")
    with pytest.raises(ValueError, match="images and image_roles"):
        await infer_turn_intent(session, turn, [], ["camera"], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pending_parse_aborts_on_timeout_or_cancellation(monkeypatch, cancel):
    import sglang_omni.serve.realtime.turn_intent as module

    aborted, removed = [], []
    entered = asyncio.Event()

    class Client:
        async def completion(self, request, request_id):
            entered.set()
            await asyncio.Event().wait()

        async def abort(self, request_id):
            aborted.append(request_id)

    session = session_for(Client(), removed=removed)
    turn = SimpleNamespace(text="说一比二", request_base="pending")
    if not cancel:
        monkeypatch.setattr(module, "TURN_INTENT_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(infer_turn_intent(session, turn, []))
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        intent = await task
        assert intent.speech == "generated" and intent.body_mode == "none"
    assert aborted == removed == ["pending-intent"]


def test_generated_reply_context_preserves_original_question_not_predicted_answer():
    intent = TurnIntent(
        speech="generated",
        text="我不知道你的名字",
        body="",
        body_mode="none",
        face="",
        history=True,
    )
    data = json.loads(intent.action_context("你知道我叫什么名字吗"))
    assert data["speech_task"] == "你知道我叫什么名字吗"


def test_unified_prompt_has_consistent_visual_and_action_boundaries():
    assert "模仿与说话可以同时存在" in SYSTEM
    assert "挥手、点头、比个心、比数字二、做个手势" in SYSTEM
    assert "“这是什么手势/这是数字几”选NO_CURRENT_VIEW" in SYSTEM
    assert "“这个加这个等于多少”依赖当前画面" in SYSTEM
    assert "未指定回答方式或明确“用手势回答并说出来/手势加语音”=gesture_and_speech" in SYSTEM
    assert "能挥挥手吗”" in SYSTEM and "是perform" in SYSTEM
    assert "你会挥手吗”" in SYSTEM and "body_intent=capability" in SYSTEM
    assert "gesture_only" in SYSTEM and "gesture_and_speech" in SYSTEM
    assert "模仿同时说话则输出 GENERAL" not in SYSTEM
    assert len(SYSTEM) < 5000
