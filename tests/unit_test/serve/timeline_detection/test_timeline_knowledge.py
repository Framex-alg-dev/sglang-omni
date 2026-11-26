import asyncio,base64,io
from types import SimpleNamespace
from dataclasses import replace
import av,httpx,pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sglang_omni.serve.prompt_runtime import decode_bundle,load_runtime_bundle
from tests.unit_test.serve.test_prompt_runtime import bundle,EXPECTED
from tests.unit_test.serve.timeline_detection.test_partial30 import window,profile
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile
from sglang_omni.serve.timeline_detection.client_model import SglangClientTimelineDetectionModel,_WindowWork
from sglang_omni.serve.timeline_detection.contracts import TimelineSessionStart
from sglang_omni.serve.timeline_prompt_debug import decode_window,install_timeline_prompt_debug

class Client:
 def __init__(self):self.prefills=[];self.requests=[]
 async def prefill_completion_prefix(self,request,**kwargs):self.prefills.append(request);return True
 async def completion(self,request,**kwargs):self.requests.append(request);return SimpleNamespace(text='N')
 async def abort(self,*args):pass

def movie(seconds=6):
 out=io.BytesIO()
 with av.open(out,mode='w',format='mp4') as target:
  stream=target.add_stream('libx264',rate=2);stream.width=320;stream.height=240;stream.pix_fmt='yuv420p'
  for i in range(seconds*2):
   frame=av.VideoFrame.from_image(Image.new('RGB',(320,240),'gray'))
   for packet in stream.encode(frame):target.mux(packet)
  for packet in stream.encode():target.mux(packet)
 return out.getvalue()

@pytest.mark.asyncio
async def test_partial_bundle_pins_runtime_and_no_external_file_reads(monkeypatch):
 generation=['one']
 env={'KNOWLEDGE_PROMPT_ENABLED':'true','KNOWLEDGE_PROMPT_URL':'http://knowledge','KNOWLEDGE_PROMPT_TENANT_ID':'tenant','KNOWLEDGE_PROMPT_RUNTIME_TOKEN':'secret'}
 async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=bundle('partial30',generation[0],generation[0])))) as http:
  loaded=await load_runtime_bundle('partial30',expected=EXPECTED,environment=env,client=http)
  generation[0]='two'
  monkeypatch.setenv('SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL','direct-partial30')
  monkeypatch.setenv('SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH','/does/not/exist')
  p=DirectE50Profile.from_environment(system_prompt_override=loaded.rules['system_prompt'],prompt_version_override=loaded.release_id)
  client=Client();start=TimelineSessionStart('test',0,0,'pcm16','jpeg','model')
  model=SglangClientTimelineDetectionModel(client,start,model_version='weights',direct_profile=p)
  try:
   assert await model._prewarm_direct_e50_prefix()
   await model._detect(_WindowWork(0,0,window()))
   assert client.prefills[0].messages[0].content==client.requests[0].messages[0].content==loaded.rules['system_prompt']
   assert model.prompt_version=='one'
  finally:await model.close()
  new=await load_runtime_bundle('partial30',expected=EXPECTED,environment=env,client=http)
  assert new.release_id=='two' and model.prompt_version=='one'


def test_uploaded_video_builds_native_window_and_fences_debug_before_inference():
 video=movie();w=decode_window(video,'test')
 assert w.end_ms==6000 and len(w.current_frames)==6 and len(w.history_frames)>=5
 app=FastAPI();client=Client();p=profile()
 loaded=decode_bundle(bundle('partial30'),profile='partial30',tenant='tenant',selector='active',expected=EXPECTED)
 install_timeline_prompt_debug(app,client=client,profile=p,bundle=loaded,model_id='model',model_version='weights',token='secret')
 with TestClient(app) as http:
  assert http.get('/v1/prompt-debug/state').status_code==404
  headers={'Authorization':'Bearer secret'};state=http.get('/v1/prompt-debug/state',headers=headers).json()
  body=dict(expected_state_sha256=state['state_sha256'],rules={'system_prompt':p.system_prompt+' DRAFT'},before_rules=state['rules'],video_base64=base64.b64encode(video).decode(),operation='preview')
  assert http.post('/v1/prompt-debug/evaluate',headers=headers,json={**body,'expected_state_sha256':'0'*64}).status_code==409
  result=http.post('/v1/prompt-debug/evaluate',headers=headers,json=body)
  assert result.status_code==200,result.text
  assert not client.prefills and not client.requests
  variants=result.json()['variants']
  for name,row in variants.items():
   assert row['prefills'][0]['request']['messages'][0]['content']==row['requests'][0]['payload']['messages'][0]['content']
  assert 'DRAFT' not in str(variants['before']) and 'DRAFT' in str(variants['draft'])
  assert http.post('/v1/prompt-debug/evaluate',headers=headers,json={**body,'operation':'run'}).status_code==200
  assert len(client.prefills)==2 and len(client.requests)==2
  assert http.post('/v1/prompt-debug/evaluate',headers=headers,json={**body,'video_base64':'bad'}).status_code==422
  assert http.get('/v1/prompt-debug/state',headers=headers).json()['rules']==state['rules']
