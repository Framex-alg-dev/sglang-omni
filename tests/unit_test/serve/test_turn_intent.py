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
            assert request.metadata['task'] == 'session_turn_intent'
            return SimpleNamespace(text=json.dumps(dict(
                body=body, speech=speech, text=text, body_mode='perform', face='', history=False)))
    session = SimpleNamespace(client=Client(), model_name='model', session_id='session',
        _register_turn_request=lambda *a: None, _unregister_turn_request=lambda *a: None)
    turn = SimpleNamespace(text=None, request_base='turn', turn_id='turn')
    result = await infer_turn_intent(session, turn, ['audio'], ['camera'], ['user_camera'])
    assert (result.body, result.speech, result.text) == (body, speech, text)
    assert len(requests) == 1
    assert requests[0].metadata['images'] == []
    assert not result.visual_scope_gate


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
