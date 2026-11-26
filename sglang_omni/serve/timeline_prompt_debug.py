"""Private Partial30 scene trials through the native window, prefill and detector."""
from __future__ import annotations
import asyncio
import base64
from dataclasses import asdict, replace
import hashlib
import hmac
import io
import json
import time
import uuid
import wave

import av
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .timeline_detection.client_model import SglangClientTimelineDetectionModel, _WindowWork
from .timeline_detection.contracts import TimelineSessionStart, TimelineMediaChunk, MediaKind
from .timeline_detection.window import EventWindowAssembler
from .timeline_detection.partial30 import PARTIAL30_PROTOCOL, PARTIAL30_EVENT_SUMMARIES
from .timeline_detection.direct_e50 import parse_direct_e50_output

class Trial(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    expected_state_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    rules:dict[str,str]
    before_rules:dict[str,str]|None=None
    video_base64:str=Field(min_length=1,max_length=2800000)
    operation:str='preview'

class Captured(BaseException):pass


def public_request(request):
    metadata={k:v for k,v in request.metadata.items() if k not in {'videos','audios'}}
    media={kind:[{'bytes':len(base64.b64decode(item.split(',',1)[1])),
                 'sha256':hashlib.sha256(base64.b64decode(item.split(',',1)[1])).hexdigest()}
                for item in request.metadata.get(kind,[])] for kind in ['videos','audios']}
    return {'model':request.model, 'messages':[{'role':m.role,'content':m.content} for m in request.messages],
            'metadata':metadata,'media':media,'max_tokens':request.max_tokens}


class DebugClient:
    def __init__(self, client, operation, trace):
        self.client,self.operation,self.trace=client,operation,trace
        self.prefills=[]
    async def prefill_completion_prefix(self,request,*,request_id):
        row={'request':public_request(request),'invoked':self.operation=='run','ready':False}
        self.prefills.append(row)
        if self.operation!='run':return True
        row['ready']=bool(await self.client.prefill_completion_prefix(request,request_id=request_id))
        return row['ready']
    async def completion(self,request,*,request_id,**kwargs):
        row={'payload':public_request(request)};self.trace.append(row)
        if self.operation!='run':raise Captured()
        started=time.monotonic()
        try:result=await self.client.completion(request,request_id=request_id,**kwargs)
        except asyncio.CancelledError:
            abort=getattr(self.client,'abort',None)
            if abort is not None:await abort(request_id)
            raise
        row.update(output=str(result.text),elapsed_ms=round((time.monotonic()-started)*1000))
        return result
    async def abort(self,request_id):
        if self.operation=='run':await self.client.abort(request_id)


def decode_window(raw,session_id):
    if len(raw)>2100000:raise ValueError('视频不能超过 2 MiB。')
    selected=[];last=None;next_ms=0;duration_ms=None
    with av.open(io.BytesIO(raw)) as container:
        stream=container.streams.video[0]
        if stream.duration is not None:duration_ms=round(float(stream.duration*stream.time_base)*1000)
        rate=float(stream.average_rate or 25)
        for index,frame in enumerate(container.decode(stream)):
            if index>1024:raise ValueError('视频帧数过多。')
            if frame.width*frame.height>2073600:raise ValueError('视频分辨率不能超过 1920×1080。')
            stamp=round(float(frame.time or 0)*1000)
            if stamp>10500:raise ValueError('请选择 3 至 10 秒短视频。')
            last=(stamp,frame)
            if stamp>=next_ms and stamp<=10000:
                selected.append((stamp,frame));next_ms+=500
        if last is None:raise ValueError('视频没有画面。')
        duration_ms=duration_ms or round(last[0]+1000/rate)
        if not 2900<=duration_ms<=10500:raise ValueError('请选择 3 至 10 秒短视频。')
        if last[0]-selected[-1][0]>100:selected.append(last)
    end_ms=min(10000,round(duration_ms/500)*500)
    assembler=EventWindowAssembler(audio_format='pcm16/16000/mono')
    pcm=[];audio_start=None
    with av.open(io.BytesIO(raw)) as source:
        if source.streams.audio:
            resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
            for index,frame in enumerate(source.decode(audio=0)):
                if index>4096:raise ValueError('音频帧数过多。')
                for normalized in resampler.resample(frame):
                    stamp=round(float(normalized.time or 0)*1000)
                    if stamp>10500:break
                    if audio_start is None:audio_start=stamp
                    pcm.append(normalized.to_ndarray().tobytes())
    if pcm:
        audio=b''.join(pcm)
        assembler.append(TimelineMediaChunk(session_id,0,0,1,MediaKind.AUDIO,max(0,audio_start or 0),
            max(0,audio_start or 0)+len(audio)//32,'pcm16',audio))
    for i,(stamp,frame) in enumerate(selected):
        output=io.BytesIO();frame.to_image().convert('RGB').save(output,format='JPEG',quality=85)
        assembler.append(TimelineMediaChunk(session_id,0,0,i+2,MediaKind.VIDEO,stamp,stamp+1,'jpeg',output.getvalue()))
    window=assembler._build_window(end_ms)
    if window is None:raise ValueError('素材无法形成有效的 H/C 窗口。')
    return window


def install_timeline_prompt_debug(app, *, client, profile, bundle, model_id, model_version, token):
    lock=asyncio.Semaphore(1)
    binding={'assembly_contract':PARTIAL30_PROTOCOL,'output_protocol':'partial30-event-code-v1',
             'model_id':model_id,'model_version':model_version,'catalog_version':None,'mapping_version':None}
    runtime=bundle.status() if bundle else {'integration_enabled':False,'consumer':'sglang_omni.timeline.partial30',
        'release_id':None,**binding}
    rules={'system_prompt':profile.system_prompt} if profile is not None else {}
    fingerprint=hashlib.sha256(json.dumps({'rules':rules,'binding':binding,'release_id':runtime.get('release_id')},
        sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    state={'runtime':runtime,'rules':rules,'state_sha256':fingerprint,
           'events':[{'code':code,'title':PARTIAL30_EVENT_SUMMARIES[code]} for code in sorted(PARTIAL30_EVENT_SUMMARIES,key=lambda v:int(v[1:]))],
           'protocol':PARTIAL30_PROTOCOL,'audio_policy':'optional_synchronized_mono_16k_pcm16',
           'scope':'detector_only_no_role_or_reply_or_action'}
    def auth(request):
        if not hmac.compare_digest(request.headers.get('authorization',''), 'Bearer '+token):raise HTTPException(404,'not found')
        if profile is None or profile.protocol!=PARTIAL30_PROTOCOL:raise HTTPException(409,'Partial30 profile is not active')
    @app.get('/v1/prompt-debug/state')
    async def read(request:Request):auth(request);return state
    @app.post('/v1/prompt-debug/evaluate')
    async def evaluate(request:Request,payload:Trial):
        auth(request)
        if payload.operation not in {'run','preview'}:raise HTTPException(422,'无效操作。')
        if payload.expected_state_sha256!=fingerprint:raise HTTPException(409,'Timeline 运行规则已变化，请重新载入。')
        variants={'draft':payload.rules}
        if payload.before_rules is not None:variants={'before':payload.before_rules,**variants}
        for texts in variants.values():
            if set(texts)!={'system_prompt'} or any(not isinstance(v,str) or not v.strip() or len(v)>100000 or '\x00' in v for v in texts.values()):raise HTTPException(422,'需要完整的 Partial30 规则。')
        try:
            raw=base64.b64decode(payload.video_base64,validate=True)
            source_window=await asyncio.to_thread(decode_window,raw,'decode-only')
        except (ValueError,IndexError,av.error.FFmpegError) as exc:raise HTTPException(422,str(exc)[:200]) from exc
        result={'variants':{},'scope':'isolated_native_detector_no_avatar_output'}
        async with lock:
            for name,texts in variants.items():
                session_id='knowledge-timeline-'+uuid.uuid4().hex
                start=TimelineSessionStart(session_id,0,0,'pcm16/16000/mono','jpeg',model_id)
                trace=[];adapter=DebugClient(client,payload.operation,trace)
                frozen=replace(profile,system_prompt=texts['system_prompt'])
                model=SglangClientTimelineDetectionModel(adapter,start,model_version=model_version,direct_profile=frozen)
                observations=();error=None
                try:
                    async with asyncio.timeout(75):
                        await model._prewarm_direct_e50_prefix()
                        observations=await model._detect(_WindowWork(0,0,source_window))
                except Captured:pass
                except (ValueError,RuntimeError,TimeoutError) as exc:error=type(exc).__name__
                finally:await model.close()
                codes=[]
                if trace and trace[-1].get('output') and error is None:
                    codes=list(parse_direct_e50_output(trace[-1]['output'],allowed_event_ids=frozen.allowed_event_ids))
                result['variants'][name]={'requests':trace,'prefills':adapter.prefills,'error':error,
                    'detected_events':[{'code':c,'title':PARTIAL30_EVENT_SUMMARIES[c]} for c in codes],
                    'emitted_events':[asdict(v) for v in observations],
                    'window':{'end_ms':source_window.end_ms,'history_frames':len(source_window.history_frames),
                              'current_frames':len(source_window.current_frames),'evidence_mode':source_window.evidence_mode}}
        return result
