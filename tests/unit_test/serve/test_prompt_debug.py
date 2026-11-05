from types import SimpleNamespace
from fastapi.testclient import TestClient
from sglang_omni.serve.task_classification.client_model import SglangClientTaskClassificationModel
from sglang_omni.serve.task_classification.pipeline import TaskClassificationPipeline
from sglang_omni.serve.task_classification.service import create_task_classification_app
from sglang_omni.serve.action_decision.service import create_action_decision_app
from tests.unit_test.serve.test_action_decision_service import _config,_FakeClient


def router_payload():
    return dict(contract_version=6,request_id='debug-router',session_id='debug-session',turn_id='debug-turn',identity_epoch=0,input_revision=0,
        text='查天气',media=[],router_history=[],brain1_capabilities='聊天',brain2_capabilities='查询天气',has_active_agent=False,pending_confirmation=False,follow_up_required=False,router_profile='v1.0')


def test_router_debug_uses_live_assembler_and_does_not_change_defaults():
    model_client=_FakeClient(code='1')
    app=create_task_classification_app(TaskClassificationPipeline(SglangClientTaskClassificationModel(model_client,model_id='router',model_version='omni-turn-router-v1.0')),token='secret')
    client=TestClient(app);headers={'Authorization':'Bearer secret'}
    assert client.get('/v1/prompt-debug/state').status_code==404
    state=client.get('/v1/prompt-debug/state',headers=headers).json()
    body={'payload':router_payload(),'expected_state_sha256':state['state_sha256'],'rules':dict(state['rules']),'operation':'preview'}
    response=client.post('/v1/prompt-debug/evaluate',headers=headers,json=body)
    assert response.status_code==200,response.text
    assert not model_client.requests
    assert response.json()['requests'][0]['payload']['messages'][0]['content']==state['rules']['system_prompt']
    body['rules']['system_prompt']+='\nUnique draft instruction.';body['rules']['mcp_scope']='changed scope';body['operation']='run'
    result=client.post('/v1/prompt-debug/evaluate',headers=headers,json=body)
    assert result.status_code==200,result.text
    assert result.json()['result']['route_code']==1
    assert 'changed scope' in str(model_client.requests[-1][1].messages)
    assert client.get('/v1/prompt-debug/state',headers=headers).json()==state
    body['expected_state_sha256']='stale'
    assert client.post('/v1/prompt-debug/evaluate',headers=headers,json=body).status_code==409


def test_action_debug_uses_real_engine_and_keeps_reference_prefix_immutable():
    model_client=_FakeClient();client=TestClient(create_action_decision_app(model_client,config=_config()));headers={'Authorization':'Bearer secret'}
    state=client.get('/v1/prompt-debug/state?profile=e67',headers=headers).json()
    payload=dict(request_id='debug-body',session_id='debug-session',turn_id='debug-turn',decision_point_id='debug-point',channel='body',text='挥挥手',runtime_context={'prompt_profile':'e67_v116_step695_fp8'})
    body={'payload':payload,'rules':dict(state['rules']),'expected_state_sha256':state['state_sha256'],'operation':'preview'}
    response=client.post('/v1/prompt-debug/evaluate?profile=e67',json=body,headers=headers)
    assert response.status_code==200,response.text
    assert not model_client.requests
    assert response.json()['requests'][0]['payload']['messages'][0]['content']==state['rules']['system_prompt']
    body['rules']['system_prompt']+='\nUnique body draft.';body['operation']='run'
    response=client.post('/v1/prompt-debug/evaluate?profile=e67',json=body,headers=headers)
    assert response.status_code==200,response.text
    assert 'Unique body draft.' in model_client.requests[-1][1].messages[0].content
    assert response.json()['result']['catalog_version']==state['catalog_version']
    assert client.get('/v1/prompt-debug/state?profile=e67',headers=headers).json()==state
    assert client.post('/v1/prompt-debug/evaluate?profile=generic',json=body,headers=headers).status_code==409


def test_gateway_debug_proxy_uses_only_configured_target_and_dedicated_auth(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from sglang_omni.serve.prompt_debug_proxy import install_prompt_debug_proxy
    import sglang_omni.serve.prompt_debug_proxy as proxy
    captured=[]
    class Client:
        def __init__(self,**kwargs):
            assert kwargs['trust_env'] is False
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def request(self,method,url,**kwargs):
            captured.append((method,url,kwargs))
            return httpx.Response(200,json={'model_id':'active-router'})
    monkeypatch.setattr(proxy.httpx,'AsyncClient',Client)
    app=FastAPI()
    config=SimpleNamespace(prompt_debug_token='debug-only-token',stages={'classifier':SimpleNamespace(url='ws://127.0.0.1:18006/classifier/v1/classify',authorization='Bearer model-service-token')})
    install_prompt_debug_proxy(app,config)
    client=TestClient(app)
    assert client.get('/v1/prompt-debug/router/state?profile=v1.0').status_code==404
    assert not captured
    result=client.get('/v1/prompt-debug/router/state?profile=v1.0',headers={'Authorization':'Bearer debug-only-token'})
    assert result.status_code==200
    method,url,kwargs=captured[0]
    assert url=='http://127.0.0.1:18006/classifier/v1/prompt-debug/state'
    assert kwargs['headers']['Authorization']=='Bearer model-service-token'
    assert kwargs['params']=={'profile':'v1.0'}
    assert 'token' not in result.text
    assert client.get('/v1/prompt-debug/arbitrary/state',headers={'Authorization':'Bearer debug-only-token'}).status_code==404
    assert len(captured)==1


def test_e67_uses_character_behavior_in_final_model_request_and_keeps_snapshot_isolated():
    model=_FakeClient();client=TestClient(create_action_decision_app(model,config=_config()));headers={'Authorization':'Bearer secret'}
    state=client.get('/v1/prompt-debug/state?profile=e67',headers=headers).json()
    payload=dict(request_id='behavior-check',session_id='behavior-check',turn_id='turn',decision_point_id='point',channel='body',text='普通聊天',
        character_prompt='## 1. 身份与场景\n老师\n## 2. 性格\n克制\n## 5. 行为规则\n倾听时不挥手',runtime_context={'prompt_profile':'e67_v116_step695_fp8'})
    response=client.post('/v1/prompt-debug/evaluate?profile=e67',headers=headers,json={'payload':payload,'rules':state['rules'],'expected_state_sha256':state['state_sha256'],'operation':'preview'})
    assert response.status_code==200,response.text
    final=response.json()['requests'][0]['payload']['messages'][0]['content']
    assert final.startswith(state['rules']['system_prompt']) and payload['character_prompt'] in final
    assert 'TRUSTED_CHARACTER_BEHAVIOR' in final
    assert not model.requests
    assert client.get('/v1/prompt-debug/state?profile=e67',headers=headers).json()==state


def test_performance_debug_runs_native_scoring_and_preview_never_calls_model():
    from tests.unit_test.serve.test_performance_control_service import _Client, _payload, _app
    from types import SimpleNamespace
    model=_Client();app=_app(model)
    app.state.action_decision_engine=SimpleNamespace(config=SimpleNamespace(model_id='qwen3-omni',model_version='test'),registry=SimpleNamespace(catalog_version='test'))
    client=TestClient(app);headers={'Authorization':'Bearer secret'}
    path='/v1/performance-control/prompt-debug/'
    assert client.get(path+'state').status_code==404
    state=client.get(path+'state',headers=headers).json();assert state['consumer']=='sglang_omni.performance' and len(state['rules'])==4
    payload=_payload();payload['performance_intent']={'response_purpose':'回应用户','reply_mode':'normal_turn','voice_style':'温柔','spoken_text_preview':'你好','required_behavior_contract':'expression'}
    body={'payload':payload,'rules':state['rules'],'expected_state_sha256':state['state_sha256'],'operation':'preview'}
    r=client.post(path+'evaluate',headers=headers,json=body);assert r.status_code==200,r.text
    assert not model.requests and r.json()['result'] is None
    trace=r.json()['requests'][0];assert trace['request_type']=='candidate_scoring'
    assert '自然克制' in trace['payload']['system_prompt']
    assert '你好' in trace['payload']['current_text']
    body['operation']='run';body['rules']={**state['rules'],'system_rules.zh-CN':state['rules']['system_rules.zh-CN']+'\nOnly this request marker.'}
    r=client.post(path+'evaluate',headers=headers,json=body);assert r.status_code==200,r.text
    assert r.json()['result']['expression_id']=='154' and not r.json()['result']['degraded']
    assert 'Only this request marker.' in model.requests[-1].system_prompt
    assert client.get(path+'state',headers=headers).json()==state
    # Production without a snapshot continues using the original rules.
    r=client.post('/v1/performance-control',headers=headers,json=payload);assert r.status_code==200
    assert 'Only this request marker.' not in model.requests[-1].system_prompt
    body['expected_state_sha256']='stale'
    assert client.post(path+'evaluate',headers=headers,json=body).status_code==409
