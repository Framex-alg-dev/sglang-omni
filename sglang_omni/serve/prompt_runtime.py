"""Resolve selected Knowledge releases once, before a model endpoint becomes ready."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from types import MappingProxyType
from urllib.parse import quote, urlsplit
import httpx

SPECS = {
    'partial30': ('sglang_omni.timeline.partial30','direct-partial30','partial30-event-code-v1',('system_prompt',)),
    'e67': ('sglang_omni.action.e67', 'e67_v116_step695_fp8', 'action-two-token-code-v1', ('system_prompt',)),
    'generic': ('sglang_omni.action.generic', 'action-generic-v1', 'action-two-token-code-v1', ('system_rules','final_constraints')),
    'performance': ('sglang_omni.performance', 'performance-control-v1', 'performance-decision-v1', ('system_rules.zh-CN','system_rules.en-US','host_intent_rules.zh-CN','host_intent_rules.en-US')),
    'router': ('sglang_omni.router.v1_0', 'turn-router-v1.0', 'turn-router-code-v1.0', ('system_prompt','mcp_scope','multimodal_scope')),
}
SELECTORS = {'partial30':'TIMELINE_PARTIAL30','e67':'ACTION_E67', 'generic':'ACTION_GENERIC', 'performance':'PERFORMANCE', 'router':'ROUTER'}

@dataclass(frozen=True)
class RuntimePromptBundle:
    consumer: str
    release_id: str
    selector: str
    sha256: str
    rules: object
    binding: object
    loaded_at: str

    def status(self):
        return {'integration_enabled':True,'source':'knowledge','consumer':self.consumer,'release_id':self.release_id,
                'release_selector':self.selector,'bundle_sha256':self.sha256,'loaded_at':self.loaded_at,
                **{k:self.binding[k] for k in ['assembly_contract','output_protocol','model_id','model_version','catalog_version','mapping_version']}}

def decode_bundle(bundle, *, profile, tenant, selector, expected):
    consumer,contract,protocol,keys=SPECS[profile]
    if not isinstance(bundle,dict) or set(bundle)!={'tenant_id','consumer','release_id','contract_version','items','binding','bundle_sha256'}:
        raise ValueError('invalid runtime bundle shape')
    if bundle['tenant_id']!=tenant or bundle['consumer']!=consumer or bundle['contract_version']!=contract:
        raise ValueError('runtime bundle scope or contract mismatch')
    release=bundle['release_id']
    if not isinstance(release,str) or not release or release=='active' or (selector!='active' and selector!=release):
        raise ValueError('runtime release selector mismatch')
    canonical={k:v for k,v in bundle.items() if k!='bundle_sha256'}
    digest=hashlib.sha256(json.dumps(canonical,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if digest!=bundle['bundle_sha256']:raise ValueError('runtime bundle checksum mismatch')
    binding=bundle['binding']
    if not isinstance(binding,dict) or binding.get('assembly_contract')!=contract or binding.get('output_protocol')!=protocol:
        raise ValueError('runtime assembly binding mismatch')
    if any(binding.get(k)!=v for k,v in expected.items()):raise ValueError('runtime model/catalog binding mismatch')
    rows=bundle['items'];rules={}
    if not isinstance(rows,list) or len(rows)!=len(keys):raise ValueError('incomplete runtime bundle')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'key','content','revision','sha256'}:raise ValueError('invalid runtime item')
        key,text=row['key'],row['content']
        if key not in keys or key in rules or not isinstance(text,str) or not text.strip() or len(text)>100000 or '\x00' in text:
            raise ValueError('invalid runtime content')
        if type(row['revision']) is not int or row['revision']<1 or hashlib.sha256(text.encode()).hexdigest()!=row['sha256']:
            raise ValueError('runtime item checksum or revision mismatch')
        rules[key]=text
    return RuntimePromptBundle(consumer,release,selector,digest,MappingProxyType(rules),MappingProxyType(dict(binding)),datetime.now(timezone.utc).isoformat())

async def load_runtime_bundle(profile, *, expected, client=None, environment=None):
    env=os.environ if environment is None else environment
    if env.get('KNOWLEDGE_PROMPT_ENABLED','false').lower()!='true':return None
    base=env.get('KNOWLEDGE_PROMPT_URL','');tenant=env.get('KNOWLEDGE_PROMPT_TENANT_ID','');token=env.get('KNOWLEDGE_PROMPT_RUNTIME_TOKEN','')
    parsed=urlsplit(base)
    if parsed.scheme not in {'http','https'} or not parsed.netloc or parsed.username or parsed.password or not tenant or not token:
        raise ValueError('Knowledge runtime startup configuration is incomplete')
    selector=env.get('KNOWLEDGE_PROMPT_'+SELECTORS[profile]+'_RELEASE','active')
    owned=client is None;client=client or httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=20)
    async def download():
        async with client.stream('GET',base.rstrip('/')+'/v1/runtime/prompts/releases/'+quote(selector,safe='')+'/bundle',
            params={'tenant_id':tenant,'consumer':SPECS[profile][0]},headers={'Authorization':'Bearer '+token}) as response:
            response.raise_for_status();raw=bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw)>1500000:raise ValueError('runtime bundle too large')
            return decode_bundle(json.loads(raw),profile=profile,tenant=tenant,selector=selector,expected=expected)
    try:return await asyncio.wait_for(download(),20)
    finally:
        if owned:await client.aclose()

async def install_action_runtime(app, engine, *, include_performance):
    profiles=['e67','generic']+(['performance'] if include_performance else [])
    loaded={}
    for profile in profiles:
        expected={'model_id':engine.config.model_id,'model_version':engine.config.model_version,
                  'catalog_version':engine.registry.catalog_version,
                  'mapping_version':None if profile=='performance' else engine.registry.mapping_version}
        bundle=await load_runtime_bundle(profile,expected=expected)
        if bundle is not None:loaded[profile]=bundle
    # Install only after every required bundle has validated; never partly activate.
    engine._runtime_prompt_bundles=MappingProxyType(loaded)
    app.state.performance_prompt_runtime=loaded.get('performance')

async def install_router_runtime(pipeline):
    model=pipeline._model
    from types import SimpleNamespace
    expected={'model_id':model.model_id,'model_version':model.model_version_for(SimpleNamespace(router_profile='v1.0')),
              'catalog_version':None,'mapping_version':None}
    bundle=await load_runtime_bundle('router',expected=expected)
    if bundle is not None:
        model._runtime_prompt_bundle=bundle
    return bundle
