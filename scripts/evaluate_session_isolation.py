"""Live public-prefix/session-lifecycle smoke test; CPU tests prove KV ownership."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid

import websockets
from evaluate_mixed_instructions import ROOT, receive, run_turn


async def main(args):
    catalog = json.loads((ROOT / 'sglang_omni/assets/character_action_global_catalog.json').read_text())
    ids = dict.fromkeys(child['candidate_id'] for category in catalog['categories'] for child in category['children'])
    allowed = [{'candidate_id': candidate_id} for candidate_id in ids]
    fallback = next(c['category_id'] for c in catalog['categories'] if 'silent_accompaniment' in c.get('semantic_tags', []))
    run = uuid.uuid4().hex[:12]
    rows = []

    async def open_session(name):
        ws = await websockets.connect(args.url, max_size=16 * 1024 * 1024)
        await ws.send(json.dumps({'type': 'session.start', 'protocol_version': 1,
            'session_id': run + name, 'locale': 'zh-CN', 'outputs': ['text', 'audio', 'action'] if args.audio_output else ['text', 'action'],
            'reply': {'instructions': '这是隔离验收使用的相同测试资料。自然简洁地回答，准确执行明确指令。',
                      'unsupported_action_text': '这个动作暂时无法执行。'},
            'action': {'fallback_category_ids': [fallback], 'allowed_candidates': allowed}}, ensure_ascii=False))
        try:
            await receive(ws, 'session.started')
        except BaseException:
            await ws.close()
            raise
        return ws

    async def turn(ws, label):
        result, first, _ = await run_turn(ws, {'text': '说一比二'}, ROOT, 30)
        assert result['status'] == 'completed', result
        assert result['reply']['text'] == '一', result
        assert result['action']['candidate_id'] == 'A259' and result['action']['execute'], result
        rows.append({'label': label, 'session_id': result['session_id'], 'turn_id': result['turn_id'],
            'wire_first_ms': first, 'action_breakdown': result['timing']['action_breakdown'], 'passed': True})

    async def cancel(ws):
        turn_id = uuid.uuid4().hex
        for event, expected in [({'type': 'turn.start', 'turn_id': turn_id, 'origin': 'user'}, 'turn.started'),
            ({'type': 'input.text.set', 'turn_id': turn_id, 'text': '解释一下春夏秋冬各自的特点'}, 'input.text.ack')]:
            await ws.send(json.dumps(event, ensure_ascii=False))
            await receive(ws, expected)
        await ws.send(json.dumps({'type': 'turn.commit', 'turn_id': turn_id}))
        await ws.send(json.dumps({'type': 'turn.cancel', 'turn_id': turn_id}))
        while True:
            event = await receive(ws)
            if event.get('type') == 'turn.cancelled' and event.get('turn_id') == turn_id:
                break
        rows.append({'label': 'b_cancelled', 'passed': True})

    a = b = b2 = None
    try:
        async with asyncio.timeout(120):
            a, b = await open_session('a'), await open_session('b')
            await asyncio.gather(turn(a, 'a_first'), turn(b, 'b_same_private_input'))
            await asyncio.gather(turn(a, 'a_while_b_cancelled'), cancel(b))
            await b.close()
            await turn(a, 'a_after_b_closed')
            b2 = await open_session('b')
            await turn(b2, 'b_new_instance_same_business_id')
    finally:
        for ws in (a, b, b2):
            if ws is not None:
                await ws.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'note': 'Functional live checks; inspect scheduler stats and CPU ownership tests for private cache isolation.', 'rows': rows}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'checks': len(rows), 'passed': all(r['passed'] for r in rows), 'output': str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='ws://127.0.0.1:18007/v1/session/realtime')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/sess_e75a0e7d/optimization_isolation.json')
    parser.add_argument('--audio-output', action='store_true')
    asyncio.run(main(parser.parse_args()))
