"""Rebuild per-turn latency data from the captured structured-log snapshot."""
import csv,json,statistics,collections,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
es=sorted((json.loads(l) for l in (ROOT/'events.jsonl').open()),key=lambda d:d['monotonic_ns'])
turns=collections.defaultdict(list); bases={}
for d in es:
 if d.get('turn_id'):turns[d['session_id'],d['turn_id']].append(d)
 if d['event']=='turn_commit_received':bases[d['logical_request_id']]=(d['session_id'],d['turn_id'])
for d in es:
 if not d.get('turn_id') and d.get('request_id'):
  key=next((v for k,v in bases.items() if d['request_id'].startswith(k+'-')),None)
  if key:turns[key].append(d)
texts={}
for l in (ROOT.parent.parent/'logs/sglang-omni.log').open():
 if 'turn.commit input' not in l:continue
 m=re.search(r'session_id=(sess_939ac65d|sess_dee23864) turn_id=(\S+) payload=(.*)',l)
 if m:texts[m[1],m[2]]=json.loads(m[3])
cat=json.loads((ROOT.parent.parent/'sglang_omni/assets/character_action_global_catalog.json').read_text())
labels={c['candidate_id']:c['source_label'] for g in cat['categories'] for c in g['children']}
rows=[]; stages=[]
def diff(a,b):return round((b['monotonic_ns']-a['monotonic_ns'])/1e6,3) if a and b else None
for key,ds in turns.items():
 ds.sort(key=lambda d:d['monotonic_ns'])
 def first(e):return next((d for d in ds if d['event']==e),{})
 def ws(e):return next((d for d in ds if d['event']=='ws_event_sent' and d.get('ws_event_type')==e),{})
 start=first('turn_start_received'); commit=first('turn_commit_received')
 if not start:continue
 timing=first('turn_timing'); perf=first('turn_performance_control_ready'); payload=texts.get(key,{})
 r={'session':key[0],'turn':key[1],'time':start['timestamp'],'pid':start['pid'],'origin':commit.get('turn_origin'),'trigger':first('turn_started').get('trigger'),'text':payload.get('text',''),'audio_chunks':commit.get('audio_chunk_count'),'image_frames':commit.get('image_frame_count'),'scope':perf.get('request_scope'),'action':first('action_child_ready').get('candidate_id'),'action_status':timing.get('action_support_status'),'expression':perf.get('expression_candidate_id'),'reply':first('provided_reply_used').get('output_text') or first('reply_completed').get('output_text'),'reply_mode':first('reply_history_route_completed').get('reply_mode'),'outputs':json.dumps(first('turn_result_sent').get('outputs',{}),ensure_ascii=False),'ingest_ms':diff(start,commit)}
 points={'category_ready':first('action_category_ready'),'performance_ready':perf,'body_scored':next((d for d in ds if d['event']=='action_scoring_completed' and d.get('stage')=='child' and d['component']=='api'),{}),'action_fused':first('action_child_ready'),'action':ws('turn.action.ready'),'expression':ws('turn.expression.ready'),'first_text':first('provisional_reply_first_token') or first('reply_first_token'),'first_public_text':ws('response.text.delta'),'route':first('reply_history_route_completed'),'tts_queued':first('tts_first_text_queued'),'tts_append':first('tts_first_append_sent'),'tts_commit':first('tts_commit_sent'),'tts_created':first('tts_response_created'),'tts_pcm':first('tts_first_audio_received'),'audio':ws('response.audio.delta'),'playable250':first('tts_playable_250ms_ready'),'result':first('turn_result_sent'),'promotion':first('provisional_reply_resolved'),'validation':first('pure_action_reply_semantic_validation_completed')}
 for name,p in points.items():r['commit_'+name+'_ms']=diff(commit,p)
 for name in ['action','expression','audio','result']:r['start_'+name+'_ms']=diff(start,points[name])
 for name,a,b in [('expr_hold','performance_ready','expression'),('body_to_action','body_scored','action'),('first_text_to_append','first_text','tts_append'),('append_to_commit','tts_append','tts_commit'),('commit_to_created','tts_commit','tts_created'),('created_to_pcm','tts_created','tts_pcm'),('pcm_to_ws','tts_pcm','audio'),('append_to_audio','tts_append','audio'),('performance_to_append','performance_ready','tts_append')]:r[name+'_ms']=diff(points[a],points[b])
 for f in ['category_compute_ms','child_compute_ms','reply_prefix_wait_ms','reply_ttft_ms','reply_stream_duration_ms','reply_total_ms','reply_delta_count','reply_completion_tokens']:r[f]=timing.get(f)
 r['route_compute_ms']=first('reply_history_route_completed').get('classification_ms');r['validation_compute_ms']=first('pure_action_reply_semantic_validation_completed').get('elapsed_ms');r['pure_reply_reported_generation_ms']=first('pure_action_short_reply_completed').get('generation_ms')
 r['action_label']=labels.get(r['action'],r['action']);r['expression_label']=labels.get(r['expression'],r['expression'])
 rows.append(r)
 for d in ds:
  if d['event']!='action_scoring_completed' or d['component']!='client':continue
  rid=d['request_id'];base=commit.get('logical_request_id','');stage=rid[len(base)+1:];s=d['stats'];phase=d.get('phase',{})
  stages.append({'session':key[0],'turn':key[1],'stage':stage,'end_after_commit_ms':diff(commit,d),'elapsed_ms':d['elapsed_ms'],'slot_wait_ms':s.get('action_slot_wait_ms'),'scheduler_wait_ms':s.get('scheduler_wait_ms'),'preprocessing_ms':s.get('preprocessing_ms'),'audio_encoder_ms':s.get('audio_encoder_ms'),'image_encoder_ms':s.get('image_encoder_ms'),'prefix_prefill_ms':s.get('prefix_prefill_ms'),'suffix_ms':sum(s.get('suffix_batch_ms',[])),'suffix_queue_ms':sum(s.get('suffix_batch_queue_wait_ms',[])),'tokens':s.get('prefix_token_count'),'cached_tokens':s.get('parent_radix_cached_token_count'),'computed_tokens':s.get('parent_computed_token_count'),'cache_ratio':s.get('parent_cache_hit_ratio'),'candidates':len(d.get('scores',[])),'batches':s.get('suffix_batch_count')})
rows.sort(key=lambda r:r['time'])
def write_csv(name,rs):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
write_csv('turns.csv',rows);write_csv('stages.csv',stages)
(ROOT/'turns.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
def stat(v):
 v=sorted(x for x in v if isinstance(x,(float,int)))
 if not v:return {'n':0}
 def q(p):
  idx=(len(v)-1)*p;lo=int(idx);hi=min(lo+1,len(v)-1);return round(v[lo]+(v[hi]-v[lo])*(idx-lo),3)
 return {'n':len(v),'mean':round(statistics.mean(v),3),'p50':q(.5),'p95':q(.95),'min':v[0],'max':v[-1]}
metrics=[k for k,v in rows[0].items() if k.endswith('_ms')]
summary={}
for sid in sorted(set(r['session'] for r in rows)):
 for group in ['all','user','proactive','action_finished','expression_only']:
  rs=[r for r in rows if r['session']==sid and (group=='all' or r['origin']==group or r['trigger']==group or r['scope']==group)]
  summary[sid+'/'+group]={'turns':len(rs),**{m:stat([r[m] for r in rs]) for m in metrics}}
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for k,s in summary.items():
 print(k,s['turns'],{m:s[m] for m in ['commit_action_ms','commit_expression_ms','commit_audio_ms']})
print('STAGES')
for stage in sorted(set(s['stage'] for s in stages)):
 ss=[s for s in stages if s['stage']==stage]
 print(stage,{k:stat([s[k] for s in ss]) for k in ['elapsed_ms','slot_wait_ms','prefix_prefill_ms','preprocessing_ms','tokens']})
