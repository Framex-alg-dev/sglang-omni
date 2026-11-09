"""Call the text/audio recall endpoint using Python standard library only."""
from pathlib import Path
import argparse, base64, json, os, urllib.request
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('audio',nargs='?',type=Path)
p.add_argument('--text')
p.add_argument('--audio-source',choices=['user_audio','avatar_speech_audio'])
p.add_argument('--channel',choices=['body','expression'],required=True)
p.add_argument('--persona',required=True)
p.add_argument('--candidates',nargs='+',help='Omit to use the full current channel catalog')
p.add_argument('--context-audio',type=Path)
p.add_argument('--user-context')
p.add_argument('--user-context-audio',type=Path,help='Optional original user audio background for avatar speech')
p.add_argument('--last-candidate-id')
p.add_argument('--last-status',choices=['sent','running','completed','failed','cancelled'])
p.add_argument('--request-id')
p.add_argument('--url',default='http://10.199.0.4:18048/v1/action-recall')
a=p.parse_args()
if (a.audio is None)==(a.text is None):p.error('Provide an audio file or --text, exclusively')
body={'channel':a.channel,'character_prompt':a.persona}
clip=lambda path:{'data':base64.b64encode(path.read_bytes()).decode(),'format':'wav'}
if a.text is not None:
 if a.audio_source is not None or a.context_audio is not None or a.user_context is not None or a.user_context_audio is not None:p.error('Text mode does not accept audio source or avatar context')
 body['text']=a.text
else:
 if a.audio_source is None:p.error('--audio-source is required for audio')
 body.update(audio=clip(a.audio),audio_source=a.audio_source)
 if a.context_audio is not None:body['context_audio']=clip(a.context_audio)
 if a.user_context is not None and a.user_context_audio is not None:p.error('Use --user-context or --user-context-audio, exclusively')
 if a.audio_source!='avatar_speech_audio' and (a.context_audio is not None or a.user_context is not None or a.user_context_audio is not None):p.error('Background fields require avatar_speech_audio')
 if a.user_context is not None:body['user_context']=a.user_context
 if a.user_context_audio is not None:body['user_context_audio']=clip(a.user_context_audio)
if a.candidates is not None:body['allowed_candidate_ids']=a.candidates
if bool(a.last_candidate_id)!=bool(a.last_status):p.error('Last candidate and status must be supplied together')
if a.last_candidate_id:body['last_dispatched_action']={'candidate_id':a.last_candidate_id,'status':a.last_status}
if a.request_id is not None:body['request_id']=a.request_id
token=os.environ.get('E68_ACTION_TOKEN')
if not token:p.error('Set E68_ACTION_TOKEN in the caller environment')
req=urllib.request.Request(a.url,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
with urllib.request.urlopen(req,timeout=120) as response:print(json.dumps(json.load(response),ensure_ascii=False,indent=2))
