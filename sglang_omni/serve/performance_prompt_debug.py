"""Authorized snapshots through the real stateless Performance scoring path."""
from __future__ import annotations
import asyncio
import hmac
import time
from types import SimpleNamespace
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from .prompt_debug import _body, _state
from .performance_control.prompts import PERFORMANCE_RULES


class ScoringPreviewCaptured(BaseException):
    """Stop preview before scoring, outside the production fail-soft handler."""


class CaptureScoringClient:
    def __init__(self, client, preview):
        self.client, self.preview, self.requests = client, preview, []

    async def score_action_suffixes(self, request):
        row = {'request_type': 'candidate_scoring', 'payload': jsonable_encoder(request)}
        self.requests.append(row)
        if self.preview:
            raise ScoringPreviewCaptured()
        started = time.monotonic()
        result = await self.client.score_action_suffixes(request)
        row.update(elapsed_ms=round((time.monotonic()-started)*1000), scores=jsonable_encoder(result.scores))
        return result


def install_performance_debug(app, *, token, response_factory):
    limit = asyncio.Semaphore(2)

    def authenticate(request):
        if not token or not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer '+token):
            raise HTTPException(404, 'not found')

    def state():
        engine = getattr(app.state, 'action_decision_engine', None)
        if engine is None:
            raise HTTPException(503, 'active model provenance unavailable')
        bundle=getattr(app.state,'performance_prompt_runtime',None)
        return _state('performance', dict(bundle.rules) if bundle is not None else dict(PERFORMANCE_RULES), engine.config.model_id, engine.config.model_version,
                      consumer='sglang_omni.performance', profile='performance', assembly_version='performance-control-behavior-125-v1',
                      catalog_version=engine.registry.catalog_version, runtime=bundle.status() if bundle is not None else {'integration_enabled':False,'release_id':None})

    @app.get('/v1/performance-control/prompt-debug/state')
    async def current(request: Request):
        authenticate(request)
        return state()

    @app.post('/v1/performance-control/prompt-debug/evaluate')
    async def evaluate(request: Request):
        authenticate(request)
        current = state()
        data = await _body(request, current)
        from .performance_control.service import PerformanceControlRequest
        try:
            native = PerformanceControlRequest.model_validate(data['payload'])
        except ValueError as exc:
            raise HTTPException(422, 'invalid performance request') from exc
        if native.audios or native.images:
            raise HTTPException(422, 'workbench requires text-only evidence')
        capture = CaptureScoringClient(app.state.client, data['operation']=='preview')
        local = SimpleNamespace(state=SimpleNamespace(client=capture))
        result = None
        async with limit:
            try:
                result = (await response_factory(local, native, prompt_snapshot=data['rules'])).model_dump()
            except ScoringPreviewCaptured:
                pass
        return {'state': current, 'requests': capture.requests, 'result': result, 'preview_only': data['operation']=='preview'}
