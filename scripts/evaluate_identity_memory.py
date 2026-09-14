"""Live identity statement/recall/correction/retraction regression."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid
import websockets
from evaluate_mixed_instructions import ROOT, receive, run_turn


async def main(args):
    sid = 'identity-' + uuid.uuid4().hex
    rows = []
    turns = [('self_report', '我叫范世德'), ('immediate_recall', '你知道我叫什么名字吗'),
             ('filler1', '用一句话介绍春天'), ('filler2', '用一句话解释彩虹'),
             ('filler3', '用一句话描述星空'), ('delayed_recall', '你知道我的名字吗'),
             ('correction', '更正一下，我叫林海，不是范世德。'), ('corrected_recall', '我现在叫什么名字？'),
             ('retraction', '请忘记我的姓名。'), ('after_forget', '你知道我的名字吗')]
    try:
        async with websockets.connect(args.url, max_size=16*1024*1024) as ws:
            await ws.send(json.dumps({'type':'session.start','protocol_version':1,'session_id':sid,
                'outputs':['text','audio'],'locale':'zh-CN','reply':{'instructions':'自然简洁地回答。'}}))
            await receive(ws,'session.started')
            for label,text in turns:
                async with asyncio.timeout(60):
                    result,first,events = await run_turn(ws, {'text':text}, ROOT, 60)
                reply = result.get('reply',{}).get('text','')
                checks = {'completed':result['status']=='completed'}
                if label=='self_report':checks['not_echo']=reply.strip('。！？!?. ')!='我叫范世德'
                if label in ['immediate_recall','delayed_recall']:checks['name_recalled']='范世德' in reply
                if label=='corrected_recall':checks['corrected_name']='林海' in reply and '范世德' not in reply
                if label=='after_forget':checks['name_not_repeated']='林海' not in reply and '范世德' not in reply
                row={'label':label,'input':text,'reply':reply,'result':result,'wire_first_ms':first,
                     'checks':checks,'passed':all(checks.values())};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['result','wire_first_ms']},ensure_ascii=False),flush=True)
            # Background extraction is asynchronous. Keep the session open briefly
            # so the final memory batch can be inspected in server diagnostics.
            # This pause is outside all measured turn timings.
            await asyncio.sleep(2)
            await ws.send(json.dumps({'type':'session.close'}));await receive(ws,'session.closed');await ws.wait_closed()
    finally:
        args.output.write_text(json.dumps({'session_id':sid,'rows':rows,'passed':len(rows)==len(turns) and all(r['passed'] for r in rows)},ensure_ascii=False,indent=2)+'\n')

    return len(rows)==len(turns) and all(r['passed'] for r in rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='ws://127.0.0.1:18007/v1/session/realtime');p.add_argument('--output',type=Path,default=ROOT/'reports/identity_memory_fix.json');raise SystemExit(0 if asyncio.run(main(p.parse_args())) else 1)
