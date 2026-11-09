"""Verify role attribution and text/audio boundaries of the unified recall API."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.action_decision.model_input import user_message_content
from sglang_omni.serve.action_decision.prompt import build_system_prompt, build_user_prompt
from sglang_omni.serve.action_decision.service import create_action_decision_app
from sglang_omni.serve.action_decision.simple_audio import parse_simple_audio_request
from tests.unit_test.serve.test_persona_audio_window import Client
from tests.unit_test.serve.test_persona_window_prompt import config, registry  # noqa: F401
from tests.unit_test.serve.test_simple_action_recall import payload


def content(kind, text):
    return {'text': text} if kind == 'text' else {'audio': payload()['audio']}


def named(channel='body'):
    return {'mode':'avatar_speech', 'channel':channel, 'character_prompt':'温柔耐心的朋友',
            'allowed_candidate_ids':['661','237'] if channel=='body' else ['154','161'],
            'avatar_current':{'text':'别着急，我们可以一步一步解决。'}}


@pytest.mark.parametrize('channel', ['body','expression'])
@pytest.mark.parametrize('focus', ['text','audio'])
@pytest.mark.parametrize('user', [None,'text','audio'])
@pytest.mark.parametrize('history', [None,'text','audio'])
def test_all_modalities_keep_current_avatar_as_the_only_target(registry, channel, focus, user, history):
    raw=named(channel); raw['avatar_current']=content(focus,'别着急，我们可以一步一步解决。')
    if user: raw['user_input']=content(user,'我今天很生气，不想继续了。')
    if history: raw['avatar_history']=content(history,'刚才已经庆祝过成功了。')
    req,_=parse_simple_audio_request(raw,registry);cat=registry.resolve(req)
    roles=(['user_context_audio'] if user=='audio' else [])+(['avatar_speech_context'] if history=='audio' else [])+(['avatar_speech_audio'] if focus=='audio' else [])
    assert [item.evidence_role for item in req.media]==roles
    assert (req.text=='我今天很生气，不想继续了。') == (user=='text')
    assert (req.avatar_current_text is not None)==(focus=='text')
    assert ('speech_window' in req.runtime_context)==(focus=='audio')
    text=build_user_prompt(req,cat);system=build_system_prompt(req,cat)
    assert '[DECISION_TASK speech_window]' in system
    assert '[USER_OBSERVATION]' not in text
    if focus=='text':
        assert '[AVATAR_CURRENT_TEXT]' in text and '别着急，我们可以一步一步解决。' in text
        final=text.split('[FINAL_TASK]')[1]
        assert 'AVATAR_CURRENT_TEXT' in final and '目标音频' not in final
        assert '不臆测声调' in system
    if history=='text':assert '[AVATAR_HISTORY_TEXT]' in text and '刚才已经庆祝过成功了。' in text
    if user=='text':assert '[ORIGINAL_USER_CONTEXT]' in text and '我今天很生气，不想继续了。' in text
    assert all(key not in str(user_message_content(req,cat)) for key in ['start_ms','end_ms','[AVATAR_SPEECH_WINDOW]'])
    model=Client('_q_d' if channel=='body' else '_l_a')
    with TestClient(create_action_decision_app(model,config=config())) as client:
        response=client.post('/v1/action-recall',json=raw,headers={'Authorization':'Bearer offline-test'})
    assert response.status_code==200,response.text
    assert response.json()['candidate_id'] in raw['allowed_candidate_ids']
    assert len(model.calls[0].metadata.get('audios',[]))==len(roles)


@pytest.mark.parametrize('channel',['body','expression'])
@pytest.mark.parametrize('source',['user_text','user_audio','avatar_speech_audio'])
def test_named_existing_paths_preserve_the_legacy_model_input(registry,channel,source):
    old=payload('user_audio' if source.startswith('user') else source)
    old.update(channel=channel,request_id='same-semantic-input')
    if channel=='expression':old['allowed_candidate_ids']=['154','161']
    new={k:v for k,v in old.items() if k not in {'audio','audio_source'}}
    if source=='user_text':
        old.pop('audio');old.pop('audio_source');old['text']='请温柔地回应我。'
        new.update(mode='user_reaction',user_input={'text':old['text']})
    elif source=='user_audio':new.update(mode='user_reaction',user_input={'audio':old['audio']})
    else:
        old.update(user_context='原始用户问题',context_audio=copy.deepcopy(old['audio']))
        new.update(mode='avatar_speech',avatar_current={'audio':old['audio']},user_input={'text':old['user_context']},avatar_history={'audio':old['context_audio']})
    a,_=parse_simple_audio_request(old,registry);b,_=parse_simple_audio_request(new,registry)
    ca,cb=registry.resolve(a),registry.resolve(b)
    assert build_system_prompt(a,ca)==build_system_prompt(b,cb)
    assert user_message_content(a,ca)==user_message_content(b,cb)


@pytest.mark.parametrize('change',[
    {'mode':'unknown'}, {'avatar_current':None}, {'avatar_current':{}},
    {'avatar_current':{'text':'a','audio':{'data':'abc'}}},
    {'avatar_current':{'text':' '}}, {'avatar_current':{'text':'x'*4001}},
    {'avatar_current':{'text':'bad\x00text'}}, {'avatar_current':{'audio':None}},
    {'user_input':{'text':'背景','extra':True}}, {'avatar_history':{'text':''}},
    {'text':'ambiguous old text'}, {'audio_source':'user_audio'},
    {'audio':{'data':'invalid'}}, {'context_audio':{}}, {'user_context':'old background'},
    {'allowed_candidate_ids':[]}, {'allowed_candidate_ids':['154']},
    {'mode':'user_reaction'},
])
def test_invalid_or_mixed_contracts_fail_before_inference(registry,change):
    raw=named();raw.update(change)
    with pytest.raises(ValueError):parse_simple_audio_request(raw,registry)


def test_text_focus_and_history_changes_participate_in_replay_identity():
    raw=named();raw.update(request_id='unified-text-retry',avatar_history={'text':'前文内容'})
    model=Client('_q_d')
    with TestClient(create_action_decision_app(model,config=config())) as client:
        call=lambda x:client.post('/v1/action-recall',json=x,headers={'Authorization':'Bearer offline-test'})
        first=call(raw);assert first.status_code==200
        assert call(raw).json()==first.json()
        for key in ['avatar_current','avatar_history']:
            changed=copy.deepcopy(raw);changed[key]['text']='changed content'
            assert call(changed).status_code==422
        assert len(model.calls)==1
