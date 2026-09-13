"""Live acceptance for the fixed four-session limit; keeps admitted sockets open."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid

import websockets
from evaluate_mixed_instructions import ROOT, receive, run_turn


async def main(args):
    sockets = []
    rows = []
    run_id = uuid.uuid4().hex[:12]

    async def connect():
        ws = await websockets.connect(args.url)
        sockets.append(ws)
        return ws

    async def start(ws, sid):
        await ws.send(json.dumps({'type': 'session.start', 'protocol_version': 1,
                                 'session_id': sid, 'outputs': ['text'], 'locale': 'zh-CN'}))
        return json.loads(await ws.recv())

    async def attempt(index):
        ws = await connect()
        sid = f'capacity-{run_id}-{index}'
        result = await start(ws, sid)
        if result['type'] == 'error':
            assert result == {'type': 'error', 'session_id': sid, 'error': {
                'type': 'server_error', 'code': 'session_busy',
                'message': 'Maximum concurrent sessions reached (4). Please retry later.'}}, result
            await ws.wait_closed()
            assert ws.close_code == 1013, ws.close_code
        else:
            assert result['type'] == 'session.started', result
        return ws, sid, result['type']

    try:
        async with asyncio.timeout(120):
            idle = await connect()  # No session.start: must not consume a slot.
            attempts = await asyncio.gather(*(attempt(i) for i in range(8)))
            admitted = [(ws, sid) for ws, sid, kind in attempts if kind == 'session.started']
            assert len(admitted) == 4, attempts
            rows.append({'check': 'eight_starts_four_admitted_four_busy_1013', 'passed': True})
            duplicate = await connect()
            result = await start(duplicate, admitted[0][1])
            assert result['type'] == 'error' and result['error']['code'] != 'session_busy', result
            assert 'already active' in result['error']['message'], result
            await duplicate.close()
            rows.append({'check': 'duplicate_id_keeps_existing_error', 'passed': True})
            for ws, sid in admitted:
                result, _, _ = await run_turn(ws, {'text': '说你好'}, ROOT, 30)
                assert result['status'] == 'completed' and result['reply']['text'], result
            rows.append({'check': 'all_four_existing_sessions_still_reply', 'passed': True})
            ws, sid = admitted[0]
            await ws.send(json.dumps({'type': 'session.close', 'reason': 'capacity_acceptance'}))
            await receive(ws, 'session.closed')
            await ws.wait_closed()
            result = await start(idle, sid)  # Same ID, new owner after release.
            assert result['type'] == 'session.started', result
            result, _, _ = await run_turn(idle, {'text': '说你好'}, ROOT, 30)
            assert result['status'] == 'completed', result
            rows.append({'check': 'unstarted_socket_reuses_released_slot_and_business_id', 'passed': True})
    finally:
        await asyncio.gather(*(ws.close() for ws in sockets), return_exceptions=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'url': args.url, 'rows': rows,
            'passed': len(rows) == 4 and all(row['passed'] for row in rows)}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'checks': len(rows), 'passed': True, 'output': str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:18007/v1/session/realtime')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/session_capacity_acceptance.json')
    asyncio.run(main(parser.parse_args()))
