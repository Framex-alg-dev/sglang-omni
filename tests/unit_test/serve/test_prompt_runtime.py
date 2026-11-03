import hashlib,json,msgpack
from types import SimpleNamespace,MappingProxyType
import httpx,pytest
from fastapi.testclient import TestClient
from sglang_omni.serve.prompt_runtime import SPECS,decode_bundle,load_runtime_bundle,RuntimePromptBundle

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bundle(profile='generic',release='release-1',marker='FIRST'):
 consumer,contract,protocol,keys=SPECS[profile]
 binding={'assembly_contract':contract,'output_protocol':protocol,'model_id':'model','model_version':'weights','catalog_version':None,'mapping_version':None}
 value={'tenant_id':'tenant','consumer':consumer,'release_id':release,'contract_version':contract,'binding':binding,
        'items':[{'key':key,'revision':1,'content':marker+' '+key,'sha256':hashlib.sha256((marker+' '+key).encode()).hexdigest()} for key in keys]}
 return {**value,'bundle_sha256':digest(value)}
EXPECTED={'model_id':'model','model_version':'weights','catalog_version':None,'mapping_version':None}

@pytest.mark.asyncio
async def test_selected_release_is_only_resolved_at_startup():
 current=bundle();calls=[]
 def respond(request):
  calls.append(request);assert request.headers['authorization']=='Bearer secret'
  return httpx.Response(200,json=current)
 env={'KNOWLEDGE_PROMPT_ENABLED':'true','KNOWLEDGE_PROMPT_URL':'http://knowledge','KNOWLEDGE_PROMPT_TENANT_ID':'tenant','KNOWLEDGE_PROMPT_RUNTIME_TOKEN':'secret'}
 async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
  loaded=await load_runtime_bundle('generic',expected=EXPECTED,client=client,environment=env)
  current=bundle(release='release-2',marker='SECOND')
  assert loaded.release_id=='release-1' and loaded.rules['system_rules'].startswith('FIRST') and len(calls)==1
  restarted=await load_runtime_bundle('generic',expected=EXPECTED,client=client,environment=env)
  assert restarted.release_id=='release-2' and restarted.rules['system_rules'].startswith('SECOND')
  assert 'secret' not in str(restarted.status())

@pytest.mark.parametrize('change',['scope','digest','model','item','keys','pin'])
def test_invalid_published_snapshot_is_rejected(change):
 value=bundle();selector='active'
 if change=='scope':value['tenant_id']='wrong'
 if change=='model':value['binding']['model_version']='wrong'
 if change=='item':value['items'][0]['content']='tampered'
 if change=='keys':value['items'].pop()
 if change=='pin':selector='other'
 value['bundle_sha256']=digest({k:v for k,v in value.items() if k!='bundle_sha256'}) if change!='digest' else 'bad'
 with pytest.raises(ValueError):decode_bundle(value,profile='generic',tenant='tenant',selector=selector,expected=EXPECTED)

@pytest.mark.asyncio
async def test_unavailable_or_unselected_bundle_has_no_default_fallback():
 env={'KNOWLEDGE_PROMPT_ENABLED':'true','KNOWLEDGE_PROMPT_URL':'http://knowledge','KNOWLEDGE_PROMPT_TENANT_ID':'tenant','KNOWLEDGE_PROMPT_RUNTIME_TOKEN':'secret'}
 async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(404))) as client:
  with pytest.raises(httpx.HTTPStatusError):await load_runtime_bundle('e67',expected=EXPECTED,client=client,environment=env)

def test_normal_action_and_performance_use_frozen_releases_until_restart(monkeypatch):
 from tests.unit_test.serve.test_action_decision_service import _FakeClient,_config
 from tests.unit_test.serve.test_performance_control_service import _Client,_payload
 from sglang_omni.serve.action_decision.service import create_action_decision_app
 import sglang_omni.serve.prompt_runtime as runtime
 from sglang_omni.serve.realtime.performance.prompts import PERFORMANCE_RULES
 generation=['FIRST'];scorer=_Client();model=_FakeClient();model.score_action_suffixes=scorer.score_action_suffixes
 async def load(profile,**kwargs):
  rules=dict(PERFORMANCE_RULES) if profile=='performance' else {'system_prompt':'ACTUAL-'+generation[0]} if profile=='e67' else {'system_rules':'GENERIC-'+generation[0],'final_constraints':'constraints'}
  if profile=='performance':rules['system_rules.zh-CN']+='\nPERFORMANCE-'+generation[0]
  return RuntimePromptBundle(SPECS[profile][0],'release-'+generation[0],'active','hash',MappingProxyType(rules),MappingProxyType({'assembly_contract':SPECS[profile][1],'output_protocol':SPECS[profile][2],**EXPECTED}),'now')
 monkeypatch.setattr(runtime,'load_runtime_bundle',load)
 def app():return create_action_decision_app(model,config=_config(),performance_token='secret')
 payload=dict(request_id='one',session_id='session',turn_id='one',decision_point_id='point',channel='body',text='挥手',runtime_context={'prompt_profile':'e67_v116_step695_fp8'})
 with TestClient(app()) as client:
  r=client.post('/v1/action-decision',headers={'Authorization':'Bearer secret'},json=payload);assert r.status_code==200,r.text
  assert 'ACTUAL-FIRST' in model.requests[-1][1].messages[0].content
  generation[0]='SECOND';payload.update(request_id='two',turn_id='two')
  r=client.post('/v1/action-decision',headers={'Authorization':'Bearer secret'},json=payload);assert r.status_code==200
  assert 'ACTUAL-FIRST' in model.requests[-1][1].messages[0].content and 'SECOND' not in model.requests[-1][1].messages[0].content
  generic={**payload,'request_id':'generic-first','turn_id':'generic-first','runtime_context':{'prompt_profile':'generic'}}
  r=client.post('/v1/action-decision',headers={'Authorization':'Bearer secret'},json=generic);assert r.status_code==200,r.text
  assert 'GENERIC-FIRST' in model.requests[-1][1].messages[0].content and 'SECOND' not in model.requests[-1][1].messages[0].content
  r=client.post('/v1/performance-control',headers={'Authorization':'Bearer secret'},json=_payload());assert r.status_code==200
  assert 'PERFORMANCE-FIRST' in scorer.requests[-1].system_prompt
  assert client.get('/v1/prompt-debug/state?profile=e67',headers={'Authorization':'Bearer secret'}).json()['runtime']['release_id']=='release-FIRST'
 with TestClient(app()) as client:
  payload.update(request_id='three',turn_id='three')
  r=client.post('/v1/action-decision',headers={'Authorization':'Bearer secret'},json=payload);assert r.status_code==200
  assert 'ACTUAL-SECOND' in model.requests[-1][1].messages[0].content
  generic.update(request_id='generic-second',turn_id='generic-second')
  r=client.post('/v1/action-decision',headers={'Authorization':'Bearer secret'},json=generic);assert r.status_code==200,r.text
  assert 'GENERIC-SECOND' in model.requests[-1][1].messages[0].content
  r=client.post('/v1/performance-control',headers={'Authorization':'Bearer secret'},json=_payload());assert r.status_code==200
  assert 'PERFORMANCE-SECOND' in scorer.requests[-1].system_prompt


def test_router_loads_only_v10_and_keeps_legacy_profiles_working(monkeypatch):
 from tests.unit_test.serve.test_prompt_debug import router_payload,_FakeClient
 from sglang_omni.serve.task_classification.service import create_task_classification_app
 from sglang_omni.serve.task_classification.pipeline import TaskClassificationPipeline
 from sglang_omni.serve.task_classification.client_model import SglangClientTaskClassificationModel
 import sglang_omni.serve.prompt_runtime as runtime
 selected=['FIRST'];client_model=_FakeClient(code='0')
 async def load(profile,**kwargs):
  return decode_bundle(bundle('router','router-'+selected[0],selected[0]),profile='router',tenant='tenant',selector='active',expected=EXPECTED)
 monkeypatch.setattr(runtime,'load_runtime_bundle',load);monkeypatch.setenv('KNOWLEDGE_PROMPT_ENABLED','true')
 def app():return create_task_classification_app(TaskClassificationPipeline(SglangClientTaskClassificationModel(client_model,model_id='model',model_version='weights')),token='secret')
 with TestClient(app()) as client:
  payload=router_payload();r=client.post('/v1/task-classification',headers={'Authorization':'Bearer secret'},content=msgpack.packb(payload,use_bin_type=True));assert r.status_code==200,r.text
  assert 'FIRST' in client_model.requests[-1][1].messages[0].content
  selected[0]='SECOND';payload.update(request_id='next',turn_id='next');r=client.post('/v1/task-classification',headers={'Authorization':'Bearer secret'},content=msgpack.packb(payload,use_bin_type=True));assert r.status_code==200
  assert 'FIRST' in client_model.requests[-1][1].messages[0].content
  payload.update(router_profile='v1.1',request_id='legacy',turn_id='legacy',active_task_goal='查天气',active_task_status='running')
  r=client.post('/v1/task-classification',headers={'Authorization':'Bearer secret'},content=msgpack.packb(payload,use_bin_type=True));assert r.status_code==200,r.text
  assert 'FIRST' not in client_model.requests[-1][1].messages[0].content
 with TestClient(app()) as client:
  payload=router_payload();r=client.post('/v1/task-classification',headers={'Authorization':'Bearer secret'},content=msgpack.packb(payload,use_bin_type=True));assert r.status_code==200
  assert 'SECOND' in client_model.requests[-1][1].messages[0].content
