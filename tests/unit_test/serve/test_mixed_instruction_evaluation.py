"""Check the evaluator cannot pass wrong words, forbidden actions or lost faces."""
import asyncio
import json
import wave
from pathlib import Path

import pytest
from scripts.evaluate_mixed_instructions import judge, load_audio, normalize_text, run_turn, summarize

CATALOG = json.loads((Path(__file__).resolve().parents[3] / 'sglang_omni/assets/character_action_global_catalog.json').read_text())


def result():
    return {'status': 'completed', 'reply': {'text': '一。'},
            'action': {'candidate_id': '259', 'execute': True, 'support_status': 'supported'},
            'timing': {'reply_mode': 'LANGUAGE_REQUIRED', 'request_scope': 'body_only'}}


def test_joint_checks_fail_wrong_reply_body_or_scope():
    case = {'expected': {'reply_exact': '一', 'body_targets': ['数字二手势'],
                         'language_required': True, 'request_scope': 'body_only'}}
    actual = result()
    assert all(judge(case, actual, CATALOG).values())
    actual['reply']['text'] = '好的，一。'
    actual['action']['candidate_id'] = '258'
    actual['timing']['request_scope'] = 'expression_only'
    checks = judge(case, actual, CATALOG)
    assert checks['reply'] is checks['body'] is checks['scope'] is False


def test_optional_face_and_explicit_face_have_different_requirements():
    actual = result()
    assert all(judge({'expected': {'expression_target': None}}, actual, CATALOG).values())
    assert not judge({'expected': {'expression_target': '微笑'}}, actual, CATALOG)['expression']
    actual['action']['candidate_id'] = '288'
    assert not judge({'expected': {'forbidden_body_targets': ['挥手']}}, actual, CATALOG)['body_prohibition']


def test_report_keeps_errors_in_denominator_and_missing_latency_out_of_percentiles():
    report = summarize([{'passed': True, 'checks': {'reply': True}, 'wire_first_ms': {'response.text.delta': 100}},
                        {'passed': False, 'error': 'timeout'}])
    assert report['samples'] == 2 and report['passed'] == report['errors'] == 1
    assert report['wire_latency_ms']['response.text.delta']['p95'] == 100
    assert report['wire_latency_ms']['response.text.delta']['samples'] == 1
    assert normalize_text('一，比二。') != normalize_text('一比二')


@pytest.mark.asyncio
async def test_audio_turn_does_not_send_transcript(tmp_path):
    path = tmp_path / 'one.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\x00\x00' * 100)
    assert len(load_audio(path)) == 200

    class Socket:
        def __init__(self):
            self.sent = []
            self.queue = asyncio.Queue()
        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            kind = {'turn.start': 'turn.started', 'input.audio.append': 'input.ack', 'turn.commit': 'turn.result'}[event['type']]
            await self.queue.put(json.dumps({'type': kind, 'turn_id': event['turn_id']}))
        async def recv(self):
            return await self.queue.get()
    socket = Socket()
    await run_turn(socket, {'audio': 'one.wav', 'text': 'SECRET TRANSCRIPT'}, tmp_path, 1)
    assert [e['type'] for e in socket.sent] == ['turn.start', 'input.audio.append', 'turn.commit']
    assert 'SECRET TRANSCRIPT' not in json.dumps(socket.sent)
