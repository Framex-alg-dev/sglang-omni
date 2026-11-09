"""Call unified text/audio recall using Python's standard library."""
from pathlib import Path
import argparse
import base64
import json
import os
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['user_reaction', 'avatar_speech'], required=True)
p.add_argument('--channel', choices=['body', 'expression'], required=True)
p.add_argument('--persona', required=True)
for group in ['user', 'avatar', 'history']:
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--' + group + '-text')
    choice.add_argument('--' + group + '-audio', type=Path, help='16kHz mono PCM16 WAV')
p.add_argument('--candidates', nargs='+', help='Omit to use all candidates in the current channel')
p.add_argument('--last-candidate-id')
p.add_argument('--last-status', choices=['sent', 'running', 'completed', 'failed', 'cancelled'])
p.add_argument('--request-id')
p.add_argument('--url', default='http://10.0.0.45:18048/v1/action-recall')
a = p.parse_args()
body = {'mode': a.mode, 'channel': a.channel, 'character_prompt': a.persona}
for group, field in [('user', 'user_input'), ('avatar', 'avatar_current'), ('history', 'avatar_history')]:
    text, audio = getattr(a, group + '_text'), getattr(a, group + '_audio')
    if text is not None:
        body[field] = {'text': text}
    elif audio is not None:
        body[field] = {'audio': {'data': base64.b64encode(audio.read_bytes()).decode(), 'format': 'wav'}}
if a.mode == 'user_reaction':
    if 'user_input' not in body or 'avatar_current' in body or 'avatar_history' in body:
        p.error('user_reaction requires user input and does not accept avatar current/history')
elif 'avatar_current' not in body:
    p.error('avatar_speech requires --avatar-text or --avatar-audio')
if a.candidates is not None:
    body['allowed_candidate_ids'] = a.candidates
if bool(a.last_candidate_id) != bool(a.last_status):
    p.error('Last candidate and status must be supplied together')
if a.last_candidate_id:
    if a.channel != 'body':
        p.error('Last dispatched body action cannot be used as expression state')
    body['last_dispatched_action'] = {'candidate_id': a.last_candidate_id, 'status': a.last_status}
if a.request_id is not None:
    body['request_id'] = a.request_id
token = os.environ.get('E68_ACTION_TOKEN')
if not token:
    p.error('Set E68_ACTION_TOKEN in the caller environment')
req = urllib.request.Request(a.url, data=json.dumps(body, ensure_ascii=False).encode(),
                             headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
with urllib.request.urlopen(req, timeout=120) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
