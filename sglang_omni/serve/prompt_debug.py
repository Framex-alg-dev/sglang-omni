"""Authenticated per-request prompt evaluation using the live service assemblers."""
from __future__ import annotations
import hashlib
import json
import time
import asyncio
from types import SimpleNamespace
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from .task_classification.client_model import SglangClientTaskClassificationModel
from .task_classification.pipeline import TaskClassificationPipeline
from .action_decision import prompt as action_prompt


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PreviewCaptured(RuntimeError):
    pass


class CaptureClient:
    def __init__(self, client, *, preview):
        self.client,self.preview,self.requests=client,preview,[]
    async def completion(self, request, **kwargs):
        record={'payload':jsonable_encoder(request)}
        self.requests.append(record)
        if self.preview:raise PreviewCaptured()
        started=time.monotonic()
        result=await self.client.completion(request,**kwargs)
        record.update(output=str(result.text),elapsed_ms=round((time.monotonic()-started)*1000))
        return result
    async def abort(self,*args,**kwargs):
        return await self.client.abort(*args,**kwargs)


def _state(kind,rules,model_id,model_version,**extra):
    value={'kind':kind,'rules':rules,'model_id':model_id,'model_version':model_version,**extra}
    value['state_sha256']=digest(value)
    return value


async def _body(request,state,required_rule_keys):
    raw=await request.body()
    if len(raw)>2_500_000:raise HTTPException(413,'debug request too large')
    try:data=json.loads(raw)
    except ValueError as exc:raise HTTPException(422,'invalid JSON') from exc
    if not isinstance(data,dict) or set(data)-{'payload','rules','operation','expected_state_sha256','profile'}:raise HTTPException(422,'invalid debug request')
    if data.get('expected_state_sha256')!=state['state_sha256']:raise HTTPException(409,'runtime prompt state changed')
    rules=data.get('rules')
    if not isinstance(rules,dict) or set(rules)!=required_rule_keys or not all(isinstance(v,str) and v.strip() and '\x00' not in v and len(v)<=100000 for v in rules.values()):raise HTTPException(422,'complete Agent prompt rules required')
    if data.get('operation') not in {'preview','run'}:raise HTTPException(422,'invalid operation')
    if not isinstance(data.get('payload'),dict):raise HTTPException(422,'native payload required')
    return data


def install_router_debug(app,*,pipeline,authenticate,invoke):
    limit=asyncio.Semaphore(2)
    model=getattr(pipeline,"_model",None)
    if not isinstance(model,SglangClientTaskClassificationModel):return
    def state():
        return _state('router',{},model.model_id,model.model_version_for(SimpleNamespace(router_profile='v1.0')),consumer='sglang_omni.router.v1_0',prompt_owner='agent')
    @app.get('/v1/prompt-debug/state')
    async def current(request:Request):
        authenticate(request);return state()
    @app.post('/v1/prompt-debug/evaluate')
    async def evaluate(request:Request):
        authenticate(request);current=state();data=await _body(request,current,{'system_prompt','mcp_scope','multimodal_scope'})
        if data['payload'].get('router_profile')!='v1.0':raise HTTPException(422,'Router v1.0 required')
        rules=data['rules'];payload=dict(data['payload'])
        payload['contract_version']=7
        payload['prompt_snapshot']={
            'consumer':'sglang_omni.router.v1_0','release_id':'knowledge-debug',
            'bundle_sha256':digest(rules),
            'binding':{'assembly_contract':'turn-router-v1.0','output_protocol':'turn-router-code-v1.0',
                'model_id':model.model_id,'model_version':model.model_version_for(SimpleNamespace(router_profile='v1.0'))},
            **rules,'item_sha256':{key:hashlib.sha256(value.encode()).hexdigest() for key,value in rules.items()},
        }
        capture=CaptureClient(model._client,preview=data['operation']=='preview')
        local=SglangClientTaskClassificationModel(capture,model_id=model.model_id,model_version=model.model_version)
        result=None
        async with limit:
            try:result=await invoke(TaskClassificationPipeline(local),payload,transport='knowledge_debug')
            except PreviewCaptured:pass
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        return {'state':current,'requests':capture.requests,'result':result,'preview_only':data['operation']=='preview'}


def install_action_debug(app,*,engine,authenticate,parser,factory):
    limit=asyncio.Semaphore(2)
    def state(profile):
        if profile=='e67':consumer='sglang_omni.action.e67'
        elif profile=='generic':consumer='sglang_omni.action.generic'
        else:raise HTTPException(422,'unsupported action profile')
        return _state('action',{},engine.config.model_id,engine.config.model_version,consumer=consumer,profile=profile,prompt_owner='agent',
            catalog_version=engine.registry.catalog_version,mapping_version=engine.registry.mapping_version,mapping_hash=engine.registry.mapping_hash,assembly_version="behavior-role-125-v1")
    @app.get('/v1/prompt-debug/state')
    async def current(request:Request,profile:str='e67'):
        authenticate(request);return state(profile)
    @app.post('/v1/prompt-debug/evaluate')
    async def evaluate(request:Request):
        authenticate(request);profile=request.query_params.get('profile','e67');current=state(profile);data=await _body(request,current,{'system_prompt'})
        native=parser(data['payload'],media=())
        selected=action_prompt.reference_prompt_profile(native)
        if (profile=='e67' and selected!=action_prompt.E67_REFERENCE_PROFILE) or (profile=='generic' and selected is not None):raise HTTPException(422,'prompt profile mismatch')
        capture=CaptureClient(engine._client,preview=data['operation']=='preview');local=factory(capture,data['rules']);result=None
        async with limit:
            try:result=(await local.decide(native)).to_dict()
            except PreviewCaptured:pass
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        return {'state':current,'requests':capture.requests,'result':result,'preview_only':data['operation']=='preview'}
