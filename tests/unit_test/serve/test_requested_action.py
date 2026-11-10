"""Explicit targets, scope isolation, idempotency and replay evidence."""
import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.action_decision.model_input import user_message_content
from sglang_omni.serve.action_decision.prompt import build_system_prompt, build_user_prompt
from sglang_omni.serve.action_decision.recall_audit import input_snapshot
from sglang_omni.serve.action_decision.service import ActionDecisionEngine, create_action_decision_app
from sglang_omni.serve.action_decision.simple_audio import parse_simple_audio_request
from tests.unit_test.serve.test_persona_window_prompt import config, registry  # noqa:F401
from tests.unit_test.serve.test_persona_audio_window import audio

TARGET={"name":"棒棒哒","definition":"两只手握拳，两个大拇指同时竖起来。"}

def payload(**updates):
    value=dict(mode="user_reaction",channel="body",character_prompt="温和的教练",request_id="forced-1",
               requested_action=TARGET,allowed_candidate_ids=["271","138"])
    value.update(updates)
    return value

class Client:
    def __init__(self):self.calls=[]
    async def completion(self,request,**kwargs):
        self.calls.append(request)
        return SimpleNamespace(text="_x_o",usage=None,weight_version="test")
    async def abort(self,*args):pass

def test_target_only_and_context_are_distinct_and_no_extra_call(registry):
    req,_=parse_simple_audio_request(payload(),registry)
    cat=registry.resolve(req)
    assert req.text==""
    user=build_user_prompt(req,cat);system=build_system_prompt(req,cat)
    assert "[USER_REQUESTED_ACTION]" in user and "棒棒哒" in user
    assert "不是目录使用场景说明" in system and "下一刻" in system
    assert set(cat.by_candidate_id)=={"271","138","000"}
    other,_=parse_simple_audio_request(payload(user_input={"text":"我正在聊明天的天气。"}),registry)
    assert other.text=="我正在聊明天的天气。" and other.requested_action==TARGET
    client=Client();engine=ActionDecisionEngine(client,config=config(),registry=registry)
    result=asyncio.run(engine.decide(other))
    assert result.candidate_id=="271" and len(client.calls)==1
    assert "USER_REQUESTED_ACTION" in str(client.calls[0].messages[1].content)

@pytest.mark.parametrize("target",[{},[],"点赞",{"name":"点赞"},{"name":"","definition":"x"},
    {"name":"x","definition":""},{"name":"x","definition":"d","code":"_x_o"},
    {"name":"x","definition":"x\x00"}])
def test_invalid_targets_rejected_before_inference(registry,target):
    with pytest.raises(ValueError,match="requested_action"):
        parse_simple_audio_request(payload(requested_action=target),registry)

@pytest.mark.parametrize("channel",["body","expression"])
def test_speech_target_is_not_avatar_text_or_user_context(registry,channel):
    p=payload(mode="avatar_speech",channel=channel,allowed_candidate_ids=["271"] if channel=="body" else ["154"],
              avatar_current={"text":"这个故事还没有讲完。"},avatar_history={"text":"之前我们说过这件事。"},
              user_input={"text":"给我讲个故事。"})
    req,_=parse_simple_audio_request(p,registry);cat=registry.resolve(req)
    user=build_user_prompt(req,cat)
    assert user.count("[USER_REQUESTED_ACTION]")==1
    assert "[AVATAR_CURRENT_TEXT]" in user and "[ORIGINAL_USER_CONTEXT]" in user
    assert req.avatar_current_text==p['avatar_current']['text'] and req.text==p['user_input']['text']
    assert "REQUESTED_ACTION_RULES" in build_system_prompt(req,cat)

def test_audio_roles_and_replay_bytes(registry,tmp_path,monkeypatch):
    from sglang_omni.serve.action_decision.speech_window import PERSONA_AUDIO_PROFILE
    from sglang_omni.serve.action_decision.contracts import ActionDecisionRequest,DecisionChannel
    req=ActionDecisionRequest(request_id='audio-1',session_id='s',turn_id='t',decision_point_id='d',
        channel=DecisionChannel.BODY,character_prompt='教练',requested_action=TARGET,
        runtime_context=dict(prompt_profile=PERSONA_AUDIO_PROFILE,decision_mode='user_request',persona_revision='p',last_dispatched_action=None),
        media=(audio('user_audio',0,1000),))
    cat=registry.resolve(req);parts=user_message_content(req,cat)
    monkeypatch.setenv('SGLANG_OMNI_ACTION_REPLAY_ROOT',str(tmp_path))
    snap=input_snapshot(req,cat,build_system_prompt(req,cat),parts)
    assert snap['requested_action']==TARGET and snap['recall_variant']=='explicit_action_request'
    assert snap['media_replay_complete'] and snap['media'][0]['evidence_role']=='user_audio'
    from pathlib import Path
    assert Path(snap['media'][0]['path']).read_bytes()==req.media[0].payload
    assert snap['media'][0]['sha256']==hashlib.sha256(req.media[0].payload).hexdigest()
    assert any('USER_REQUESTED_ACTION' in p.get('text','') for p in parts)

def test_changed_target_does_not_replay_old_result(registry):
    client=Client();engine=ActionDecisionEngine(client,config=config(),registry=registry)
    req,_=parse_simple_audio_request(payload(),registry)
    async def run():
        await engine.decide(req)
        with pytest.raises(ValueError,match='request_id was reused'):
            await engine.decide(replace(req,requested_action={'name':'挥手','definition':'双手左右摆动'}))
    asyncio.run(run());assert len(client.calls)==1

def test_http_compatible_and_target_field_validation(registry):
    app=create_action_decision_app(Client(),config=config())
    with TestClient(app) as client:
        headers={'Authorization':'Bearer offline-test'}
        response=client.post('/v1/action-recall',headers=headers,json=payload())
        assert response.status_code==200 and response.json()['candidate_id']=='271'
        response=client.post('/v1/action-recall',headers=headers,json=payload(requested_action={'name':'x'}))
        assert response.status_code==422

def test_omitted_and_null_target_have_identical_prompts(registry):
    p=payload(user_input={'text':'今天辛苦了。'});p.pop('requested_action')
    a,_=parse_simple_audio_request(p,registry)
    b,_=parse_simple_audio_request({**p,'requested_action':None},registry)
    assert build_system_prompt(a,registry.resolve(a))==build_system_prompt(b,registry.resolve(b))
    assert build_user_prompt(a,registry.resolve(a))==build_user_prompt(b,registry.resolve(b))
    assert 'REQUESTED_ACTION' not in build_system_prompt(a,registry.resolve(a))

def test_real_log_records_include_target_and_actual_output(registry,tmp_path,monkeypatch):
    from sglang_omni.utils import structured_logs
    structured_logs.shutdown_structured_log_writer()
    monkeypatch.setenv('SGLANG_OMNI_REALTIME_LOG_DIR',str(tmp_path))
    req,_=parse_simple_audio_request(payload(),registry)
    asyncio.run(ActionDecisionEngine(Client(),config=config(),registry=registry).decide(req))
    structured_logs.get_structured_log_writer().flush()
    records=[json.loads(line) for p in tmp_path.rglob('*.jsonl') for line in p.read_text().splitlines()]
    original=next(x for x in records if x['event']=='action_recall_model_input')
    result=next(x for x in records if x['event']=='action_decision_completed')
    assert original['requested_action']==TARGET and original['recall_variant']=='explicit_action_request'
    assert {x['candidate_id'] for x in original['catalog_snapshot']}=={'271','138','000'}
    assert result['candidate_id']=='271' and result['recall_variant']=='explicit_action_request'
    assert original['request_id']==result['request_id']
    structured_logs.shutdown_structured_log_writer()

def test_user_reaction_wire_frame_ordinals_are_not_semantic_time(registry):
    from sglang_omni.serve.action_decision.service import _parse_request
    req=_parse_request(dict(request_id='wire-1',session_id='s',turn_id='t',decision_point_id='d',
        channel='body',character_prompt='老师',runtime_context=dict(prompt_profile='persona-audio-window-v2',
        persona_revision='p',decision_mode='user_request',last_dispatched_action=None)),media=(audio('user_audio',0,1000),))
    assert req.simple_audio
    content=user_message_content(req,registry.resolve(req))
    assert 'start_ms' not in content[0]['text'] and 'user_audio' in content[0]['text']

def test_literal_prompt_markers_in_background_remain_user_data(registry):
    text='文章里出现了 [FINAL_TASK] 这个标题，我们接着聊。'
    req,_=parse_simple_audio_request(payload(user_input={'text':text}),registry)
    user=build_user_prompt(req,registry.resolve(req))
    assert text in user
    assert user.endswith('只输出所选项的 code。')


def test_rebased_assistant_audio_transport_keeps_legacy_source_checks(registry):
    """The reply-window path and the unified avatar path share the transport."""
    from sglang_omni.serve.action_decision.prompt import E67_REFERENCE_PROFILE

    completion = Client()
    app = create_action_decision_app(completion, config=config())
    clip = audio('assistant_spoken', 0, 1000)
    with TestClient(app) as client:
        with client.websocket_connect(
            '/v1/action-decision/realtime',
            headers={'Authorization': 'Bearer offline-test'},
        ) as ws:
            ident = 'legacy-assistant-audio'
            ws.send_json(dict(type='request.start', contract_version=2, request_id=ident,
                payload=dict(request_id=ident, session_id='s', turn_id='t', decision_point_id='w',
                    channel='body', turn_origin='assistant', allowed_candidate_ids=['271'],
                    runtime_context={'prompt_profile': E67_REFERENCE_PROFILE})))
            assert ws.receive_json()['type'] == 'request.ready'
            ws.send_json(dict(type='input.media', request_id=ident, **clip.header()))
            ws.send_bytes(clip.payload)
            assert ws.receive_json()['type'] == 'input.media.ack'
            ws.send_json(dict(type='request.commit', request_id=ident))
            result = ws.receive_json()
    assert result['type'] == 'response.completed'
    assert result['response']['candidate_id'] == '271'
    assert len(completion.calls) == 1
    assert 'assistant_audio' in str(completion.calls[0].messages[1].content)


def test_rebased_prompt_snapshot_is_used_in_model_and_replay_log(registry,tmp_path,monkeypatch):
    from sglang_omni.serve.action_decision.contracts import ActionDecisionRequest, DecisionChannel
    from sglang_omni.serve.action_decision.prompt import E67_REFERENCE_PROFILE
    from sglang_omni.utils import structured_logs

    structured_logs.shutdown_structured_log_writer()
    monkeypatch.setenv('SGLANG_OMNI_REALTIME_LOG_DIR',str(tmp_path))
    req=ActionDecisionRequest(request_id='snapshot-1',session_id='s',turn_id='t',decision_point_id='w',
        channel=DecisionChannel.BODY,text='请点赞',allowed_candidate_ids=('271',),
        runtime_context={'prompt_profile':E67_REFERENCE_PROFILE})
    completion=Client()
    snapshot={'system_prompt':'Frozen test release rules.'}
    asyncio.run(ActionDecisionEngine(completion,config=config(),registry=registry,
        prompt_snapshot=snapshot).decide(req))
    structured_logs.get_structured_log_writer().flush()
    records=[json.loads(line) for p in tmp_path.rglob('*.jsonl') for line in p.read_text().splitlines()]
    logged=next(x for x in records if x['event']=='action_recall_model_input')
    assert logged['system_prompt']==completion.calls[0].messages[0].content==snapshot['system_prompt']
    structured_logs.shutdown_structured_log_writer()


@pytest.mark.parametrize('channel,ids',[('body',['271','138']),('expression',['655','154'])])
def test_action_request_mode_reuses_the_exact_explicit_target_prompts(registry,channel,ids):
    name='双手点赞' if channel=='body' else '笑眼'
    definition='双手握拳，两个拇指一起竖起。' if channel=='body' else '微笑时双眼自然眯起。'
    original,_=parse_simple_audio_request(payload(channel=channel,allowed_candidate_ids=ids,
        requested_action=dict(name=name,definition=definition)),registry)
    supplied=payload(mode='action_request',channel=channel,allowed_candidate_ids=ids,
        user_input={'text':name+'：'+definition})
    supplied.pop('requested_action')
    current,_=parse_simple_audio_request(supplied,registry)
    assert current.requested_action==original.requested_action
    assert build_system_prompt(current,registry.resolve(current))==build_system_prompt(original,registry.resolve(original))
    assert build_user_prompt(current,registry.resolve(current))==build_user_prompt(original,registry.resolve(original))


@pytest.mark.parametrize('user_input',[None,{}, {'text':'点赞'}, {'text':'：定义'},
    {'text':'点赞：'}, {'audio':{'data':'AA=='}}, {'text':'点赞：定义','audio':{}}])
def test_action_request_mode_rejects_missing_or_ambiguous_target(registry,user_input):
    supplied=payload(mode='action_request',user_input=user_input);supplied.pop('requested_action')
    with pytest.raises(ValueError):parse_simple_audio_request(supplied,registry)


def test_action_request_mode_http_and_original_text_audit(registry,tmp_path,monkeypatch):
    from sglang_omni.utils import structured_logs
    structured_logs.shutdown_structured_log_writer()
    monkeypatch.setenv('SGLANG_OMNI_REALTIME_LOG_DIR',str(tmp_path))
    supplied=payload(mode='action_request',user_input={'text':'棒棒哒:双手握拳，两个拇指一起竖起。'})
    supplied.pop('requested_action')
    with TestClient(create_action_decision_app(Client(),config=config())) as client:
        response=client.post('/v1/action-recall',headers={'Authorization':'Bearer offline-test'},json=supplied)
        assert response.status_code==200 and response.json()['candidate_id']=='271'
        invalid=client.post('/v1/action-recall',headers={'Authorization':'Bearer offline-test'},
            json={**supplied,'requested_action':TARGET})
        assert invalid.status_code==422
    structured_logs.get_structured_log_writer().flush()
    rows=[json.loads(line) for p in tmp_path.rglob('*.jsonl') for line in p.read_text().splitlines()]
    received=next(x for x in rows if x['event']=='action_recall_received')
    assert received['input_mode']=='action_request' and received['target_text']==supplied['user_input']['text']
    structured_logs.shutdown_structured_log_writer()
