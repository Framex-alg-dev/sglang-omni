"""Fixed-target management proxy; credentials never reach the browser."""
from fastapi import HTTPException, Request
from fastapi.responses import Response
import hmac
import httpx


def install_prompt_debug_proxy(app, config):
    def authorize(request):
        if not hmac.compare_digest(request.headers.get('authorization',''), 'Bearer '+config.prompt_debug_token):
            raise HTTPException(404,'not found')
    async def forward(kind,operation,request):
        authorize(request)
        if kind not in {'router','action','performance'} or operation not in {'state','evaluate'}:raise HTTPException(404,'not found')
        profile=request.query_params.get('profile','e67')
        if profile not in {'e67','generic','v1.0','performance'}:raise HTTPException(422,'invalid profile')
        target=config.stages['classifier' if kind=='router' else 'performance' if kind=='performance' else 'body']
        url=target.url.rsplit('/v1/',1)[0]+('/v1/performance-control/prompt-debug/' if kind=='performance' else '/v1/prompt-debug/')+operation
        url=url.replace('ws://','http://',1).replace('wss://','https://',1)
        raw=await request.body()
        if len(raw)>2_500_000:raise HTTPException(413,'debug request too large')
        async with httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=60) as client:
            try:
                response=await client.request(request.method,url,params={'profile':profile},content=raw if raw else None,
                    headers={'Authorization':target.authorization,'Content-Type':'application/json'})
            except httpx.HTTPError as exc:raise HTTPException(502,'decision debug upstream unavailable') from exc
        return Response(content=response.content,status_code=response.status_code,media_type='application/json')
    @app.get('/v1/prompt-debug/{kind}/state')
    async def state(kind:str,request:Request):return await forward(kind,'state',request)
    @app.post('/v1/prompt-debug/{kind}/evaluate')
    async def evaluate(kind:str,request:Request):return await forward(kind,'evaluate',request)
