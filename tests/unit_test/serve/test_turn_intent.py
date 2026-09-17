import json
from types import SimpleNamespace

import pytest
from sglang_omni.serve.realtime.turn_intent import TurnIntent, infer_turn_intent


def payload(**changes):
    return dict(speech='verbatim', text='一', body='数字二手势', body_mode='perform', face='', history=False, **changes)


@pytest.mark.parametrize('raw', ['{}', '[]', 'not JSON', '{"speech": "verbatim"}', 'x' * 4097])
def test_invalid_parse_is_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        TurnIntent.parse(raw)


def test_strict_schema_and_no_instruction_promotion():
    intent = TurnIntent.parse(json.dumps(payload()))
    assert intent.text == '一' and intent.body == '数字二手势'
    data = json.loads(intent.action_context('说一比二'))
    assert data['body_task'] == '数字二手势'
    bad = payload()
    bad['history'] = 'false'
    with pytest.raises(ValueError):
        TurnIntent.parse(json.dumps(bad))


def test_natural_reaction_is_separate_from_explicit_body_task():
    greeting = TurnIntent.parse(json.dumps({
        'speech': 'generated',
        'text': '你好',
        'body': '',
        'body_mode': 'none',
        'face': '',
        'history': False,
        'reaction_mode': 'respond',
        'reaction': '回应用户问候',
    }, ensure_ascii=False))
    assert greeting.reaction_mode == 'respond'
    assert json.loads(greeting.action_context('你好'))['reaction_task'] == '回应用户问候'

    legacy = TurnIntent.parse(json.dumps(payload()))
    assert legacy.reaction_mode == 'none'
    assert legacy.reaction == ''

    conflicting = payload(
        reaction_mode='respond',
        reaction='回应用户问候',
    )
    with pytest.raises(ValueError, match='explicit body task'):
        TurnIntent.parse(json.dumps(conflicting, ensure_ascii=False))
    bad = payload()
    bad['extra'] = 'override system'
    with pytest.raises(ValueError):
        TurnIntent.parse(json.dumps(bad))


@pytest.mark.asyncio
async def test_audio_metadata_and_request_cleanup_on_parse_failure():
    registered, removed, aborted, requests = [], [], [], []
    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(text='not JSON')
        async def abort(self, request_id):
            aborted.append(request_id)
    session = SimpleNamespace(client=Client(), model_name='model', session_id='session',
                              _register_turn_request=lambda t, r: registered.append(r),
                              _unregister_turn_request=lambda t, r: removed.append(r))
    turn = SimpleNamespace(text=None, request_base='turn')
    assert await infer_turn_intent(session, turn, ['audio-ref']) is None
    assert registered == removed == ['turn-intent']
    assert aborted == []
    assert len(requests) == 1
    assert requests[0].metadata['audios'] == ['audio-ref']
    assert requests[0].messages[1].content == [{'type':'audio'}]


@pytest.mark.asyncio
@pytest.mark.parametrize("body,speech,text", [
    ("单手挥手", "none", ""),
    ("单手摊开展示", "verbatim", "这是礼物"),
    ("这个手势", "none", ""),
    ("这个手势", "verbatim", "你好"),
])
async def test_shared_intent_preserves_body_and_speech_with_camera(body, speech, text):
    requests = []
    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            if request.metadata['task'] == 'session_visual_scope_gate':
                return SimpleNamespace(text='V00')
            assert request.metadata['task'] == 'session_turn_intent'
            return SimpleNamespace(text=json.dumps(dict(
                body=body, speech=speech, text=text, body_mode='perform', face='', history=False)))
    session = SimpleNamespace(client=Client(), model_name='model', session_id='session',
        _register_turn_request=lambda *a: None, _unregister_turn_request=lambda *a: None)
    turn = SimpleNamespace(text=None, request_base='turn', turn_id='turn')
    result = await infer_turn_intent(session, turn, ['audio'], ['camera'], ['user_camera'])
    assert (result.body, result.speech, result.text) == (body, speech, text)
    assert len(requests) == 2
    assert all(request.metadata['images'] == [] for request in requests)
    assert not result.visual_scope_gate


@pytest.mark.asyncio
@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_visual_scope_gate_precedes_free_form_intent_and_uses_language_only(
    language,
):
    completion_requests, registered, removed = [], [], []

    class Client:
        async def completion(self, request, request_id):
            completion_requests.append((request, request_id))
            assert request.metadata['task'] == 'session_visual_scope_gate'
            return SimpleNamespace(text='V01')

        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def abort(self, request_id):
            raise AssertionError('completed request must not be aborted')

    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        session_instance_id='instance',
        language=language,
        _register_turn_request=lambda turn, request_id: registered.append(request_id),
        _unregister_turn_request=lambda turn, request_id: removed.append(request_id),
    )
    turn = SimpleNamespace(
        text=None,
        request_base='visual-turn',
        turn_id='visual-turn',
    )

    intent = await infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['old-camera', 'avatar', 'latest-camera'],
        ['user_camera', 'character_avatar', 'user_camera'],
    )

    assert intent is not None
    assert intent.body_mode == 'perform'
    assert intent.body == '这个手势'
    assert intent.speech == 'none'
    assert intent.visual_scope_gate == 'V01'
    assert len(completion_requests) == 1
    request, request_id = completion_requests[0]
    assert request_id == 'visual-turn-visual-scope-gate'
    assert [message.role for message in request.messages] == ['system', 'user']
    assert '听取当前用户音频' in request.messages[0].content
    assert "Listen to the current user's audio" in request.messages[0].content
    assert '"Do this gesture" => V01' in request.messages[0].content
    assert request.messages[1].content == [{'type': 'audio'}]
    assert request.metadata['audios'] == ['audio-ref']
    assert request.metadata['images'] == []
    assert request.metadata['image_roles'] == []
    assert request.sampling.temperature == 0
    assert request.sampling.max_new_tokens == 8
    assert request.stream is False
    assert request.output_modalities == ['text']
    assert registered == removed == ['visual-turn-visual-scope-gate']


@pytest.mark.asyncio
async def test_non_visual_gate_falls_through_to_language_only_general_intent():
    completion_requests = []

    class Client:
        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def completion(self, request, request_id):
            completion_requests.append(request)
            if request.metadata['task'] == 'session_visual_scope_gate':
                return SimpleNamespace(text='V00')
            return SimpleNamespace(text=json.dumps({
                'speech': 'generated',
                'text': '这是什么手势',
                'body': '',
                'body_mode': 'none',
                'face': '',
                'history': False,
            }, ensure_ascii=False))

        async def abort(self, request_id):
            raise AssertionError('completed request must not be aborted')

    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        session_instance_id='instance',
        language='zh',
        _register_turn_request=lambda *args: None,
        _unregister_turn_request=lambda *args: None,
    )
    turn = SimpleNamespace(
        text=None,
        request_base='visual-question',
        turn_id='visual-question',
    )

    intent = await infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['camera'],
        ['user_camera'],
    )

    assert intent is not None
    assert intent.speech == 'generated'
    assert intent.body_mode == 'none'
    assert len(completion_requests) == 2
    assert completion_requests[0].metadata['task'] == 'session_visual_scope_gate'
    request = completion_requests[1]
    assert request.metadata['task'] == 'session_turn_intent'
    assert request.metadata['images'] == []
    assert request.metadata['image_roles'] == []
    assert request.messages[1].content == [{'type': 'audio'}]


@pytest.mark.asyncio
async def test_visual_scope_gate_places_optional_text_before_audio():
    requests = []

    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(text='V01')

        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def abort(self, request_id):
            raise AssertionError('completed request must not be aborted')

    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        language='zh',
        _register_turn_request=lambda *args: None,
        _unregister_turn_request=lambda *args: None,
    )
    turn = SimpleNamespace(
        text='请做出这个手势',
        request_base='visual-text-audio',
        turn_id='visual-text-audio',
    )

    intent = await infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['camera'],
        ['user_camera'],
    )

    assert intent is not None and intent.visual_scope_gate == 'V01'
    assert requests[0].messages[1].content == [
        {'type': 'text', 'text': '请做出这个手势'},
        {'type': 'audio'},
    ]


@pytest.mark.asyncio
async def test_visual_gesture_answer_gate_requests_hidden_multimodal_reasoning():
    requests = []

    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(text='V11')

        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def abort(self, request_id):
            raise AssertionError('completed request must not be aborted')

    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        language='zh',
        _register_turn_request=lambda *args: None,
        _unregister_turn_request=lambda *args: None,
    )
    turn = SimpleNamespace(
        text='这个加这个是什么，用手势回答',
        request_base='visual-gesture-answer',
        turn_id='visual-gesture-answer',
    )

    intent = await infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['camera-one', 'camera-two'],
        ['user_camera', 'user_camera'],
    )

    assert intent is not None
    assert intent.visual_scope_gate == 'V11'
    assert intent.speech == 'generated'
    assert intent.body_mode == 'none'
    assert intent.body == ''
    assert len(requests) == 1
    assert '"Answer with a gesture" is not imitation' in requests[0].messages[0].content
    assert requests[0].metadata['images'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('gate_outcome', ['invalid', 'error', 'timeout'])
async def test_visual_scope_gate_failure_aborts_and_falls_through(
    monkeypatch,
    gate_outcome,
):
    import asyncio
    import sglang_omni.serve.realtime.turn_intent as module

    requests, aborted, registered, removed = [], [], [], []

    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            if request.metadata['task'] == 'session_visual_scope_gate':
                if gate_outcome == 'invalid':
                    return SimpleNamespace(text='The result is V01')
                if gate_outcome == 'error':
                    raise RuntimeError('gate failed')
                await asyncio.Event().wait()
            return SimpleNamespace(text=json.dumps({
                'speech': 'generated',
                'text': 'fallback',
                'body': '',
                'body_mode': 'none',
                'face': '',
                'history': False,
            }))

        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def abort(self, request_id):
            aborted.append(request_id)

    if gate_outcome == 'timeout':
        monkeypatch.setattr(module, 'VISUAL_SCOPE_GATE_TIMEOUT_SECONDS', .01)
    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        _register_turn_request=lambda turn, request_id: registered.append(request_id),
        _unregister_turn_request=lambda turn, request_id: removed.append(request_id),
    )
    turn = SimpleNamespace(
        text=None,
        request_base='gate-fallback',
        turn_id='gate-fallback',
    )

    intent = await infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['camera'],
        ['user_camera'],
    )

    assert intent is not None and intent.text == 'fallback'
    assert [request.metadata['task'] for request in requests] == [
        'session_visual_scope_gate',
        'session_turn_intent',
    ]
    assert aborted == ['gate-fallback-visual-scope-gate']
    assert registered == removed == [
        'gate-fallback-visual-scope-gate',
        'gate-fallback-intent',
    ]


@pytest.mark.asyncio
async def test_visual_scope_gate_cancellation_aborts_and_unregisters():
    import asyncio

    entered = asyncio.Event()
    aborted, removed = [], []

    class Client:
        async def completion(self, request, request_id):
            assert request.metadata['task'] == 'session_visual_scope_gate'
            entered.set()
            await asyncio.Event().wait()

        async def score_action_suffixes(self, request):
            raise AssertionError('visual scope generation must not score suffixes')

        async def abort(self, request_id):
            aborted.append(request_id)

    session = SimpleNamespace(
        client=Client(),
        model_name='model',
        session_id='session',
        _register_turn_request=lambda *args: None,
        _unregister_turn_request=lambda turn, request_id: removed.append(request_id),
    )
    turn = SimpleNamespace(
        text=None,
        request_base='gate-cancelled',
        turn_id='gate-cancelled',
    )
    task = asyncio.create_task(infer_turn_intent(
        session,
        turn,
        ['audio-ref'],
        ['camera'],
        ['user_camera'],
    ))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert aborted == removed == ['gate-cancelled-visual-scope-gate']


@pytest.mark.asyncio
async def test_intent_rejects_misaligned_current_image_roles():
    session = SimpleNamespace(model_name='model', session_id='session')
    turn = SimpleNamespace(text='请做出这个手势', request_base='bad-images')

    with pytest.raises(ValueError, match='images and image_roles'):
        await infer_turn_intent(session, turn, [], ['camera'], [])


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_pending_parse_aborts_on_timeout_or_cancellation(monkeypatch, cancel):
    import asyncio
    import sglang_omni.serve.realtime.turn_intent as module
    calls, removed = [], []
    entered = asyncio.Event()
    class Client:
        async def completion(self, request, request_id):
            entered.set()
            await asyncio.Event().wait()
        async def abort(self, request_id):
            calls.append(request_id)
    session = SimpleNamespace(client=Client(), model_name='model', session_id='session',
                              _register_turn_request=lambda *args: None,
                              _unregister_turn_request=lambda t, r: removed.append(r))
    turn = SimpleNamespace(text='说一比二', request_base='pending')
    if not cancel:
        monkeypatch.setattr(module, 'TURN_INTENT_TIMEOUT_SECONDS', .01)
    task = asyncio.create_task(infer_turn_intent(session, turn, []))
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task is None
    assert calls == removed == ['pending-intent']


def test_generated_reply_context_preserves_original_question_not_predicted_answer():
    intent = TurnIntent(speech='generated', text='我不知道你的名字', body='', body_mode='none', face='', history=True)
    data = json.loads(intent.action_context('你知道我叫什么名字吗'))
    assert data['speech_task'] == '你知道我叫什么名字吗'


def test_visual_imitation_prompt_preserves_the_unresolved_image_reference():
    from sglang_omni.serve.realtime.turn_intent import SYSTEM

    assert '请做出这个手势' in SYSTEM
    assert '"body":"这个手势"' in SYSTEM
    assert '请做出手势' in SYSTEM
    assert '"body":"做出手势"' in SYSTEM
    assert '请做出这个表情' in SYSTEM
    assert '"face":"这个表情"' in SYSTEM
    assert '请做出表情' in SYSTEM
    assert '"face":"做出表情"' in SYSTEM
    assert '请做出这个动作' in SYSTEM
    assert '"body":"这个动作"' in SYSTEM
    assert '这个动作叫什么' in SYSTEM
    assert '"body_mode":"none"' in SYSTEM


def test_permission_shaped_body_command_is_distinct_from_capability_question():
    from sglang_omni.serve.realtime.turn_intent import SYSTEM

    assert '你可以站起来 -> {"speech":"none"' in SYSTEM
    assert '你可以站起来吗？ -> {"speech":"generated"' in SYSTEM
    assert '你不可以站起来 -> {"speech":"none"' in SYSTEM
    assert 'You can stand up now. -> {"speech":"none"' in SYSTEM
    assert 'Can you stand up? -> {"speech":"generated"' in SYSTEM
