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


@pytest.mark.parametrize('allowed,excluded,expected', [
    ((), (), ['弹吉他', '弹钢琴', '打篮球']),
    (('A426', 'A425'), ('A425',), ['弹吉他']),
    (('missing',), (), []),
])
def test_capability_context_is_filtered_deduplicated_and_at_system_end(allowed, excluded, expected):
    from sglang_omni.serve.realtime.turn_intent import SYSTEM, CAPABILITY_INTENT_RULES, _intent_system_prompt
    def candidate(cid, label, action_id='action'):
        return SimpleNamespace(candidate_id=cid, source_label=label, action_id=action_id)
    session = SimpleNamespace(candidates=[candidate('A000', '不执行', 'no_action'),
        candidate('A426', '弹吉他'), candidate('A425', '弹钢琴'), candidate('A431', '打篮球'),
        candidate('duplicate', '弹吉他'), candidate('invalid_no_action', '不执行', 'no_action')])
    turn = SimpleNamespace(action_allowed_candidate_ids=allowed, action_excluded_candidate_ids=excluded)
    prompt = _intent_system_prompt(session, turn)
    prefix = SYSTEM + '\n' + CAPABILITY_INTENT_RULES + '\n[角色动作白名单数据] '
    assert prompt.startswith(prefix)
    assert json.loads(prompt[len(prefix):]) == {'supported_action_names': expected}


def test_no_candidates_still_preserves_classification_rules_without_mutating_system():
    from sglang_omni.serve.realtime.turn_intent import SYSTEM, _intent_system_prompt
    assert _intent_system_prompt(SimpleNamespace(), SimpleNamespace()).endswith('{"supported_action_names": []}')
    assert '[角色动作白名单数据]' not in SYSTEM


@pytest.mark.asyncio
async def test_live_request_uses_system_tail_and_preserves_user_audio():
    from sglang_omni.serve.realtime.turn_intent import _intent_system_prompt
    requests = []
    class Client:
        async def completion(self, request, request_id):
            requests.append(request)
            return SimpleNamespace(text=json.dumps(payload()))
    session = SimpleNamespace(client=Client(), model_name='model', session_id='session', candidates=[],
        _register_turn_request=lambda *args: None, _unregister_turn_request=lambda *args: None)
    turn = SimpleNamespace(text='说一比二', request_base='turn', turn_id='turn')
    assert await infer_turn_intent(session, turn, ['audio-ref']) is not None
    assert requests[0].messages[0].content == _intent_system_prompt(session, turn)
    assert requests[0].messages[1].content == [{'type':'text', 'text':'说一比二'}, {'type':'audio'}]
    assert requests[0].metadata['audios'] == ['audio-ref']
