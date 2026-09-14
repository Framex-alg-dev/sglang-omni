"""Measure repeated turns on ONE connection, without clearing session history."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import websockets
from evaluate_mixed_instructions import ROOT, judge, receive, run_turn, summarize


async def main(args):
    catalog = json.loads((ROOT / 'sglang_omni/assets/character_action_global_catalog.json').read_text())
    audio_base = ROOT / 'reports/mixed_audio'
    cases = json.loads((audio_base / 'cases.json').read_text())['cases']
    prompts = ['用不超过四十个字解释为什么天空是蓝色的。', '用不超过四十个字给我一个缓解工作疲劳的建议。',
               '用不超过四十个字描述春天的公园。', '用不超过四十个字解释彩虹是怎么形成的。']
    schedule = [(f'audio_cycle_{i+1}', c) for i in range(10) for c in cases]
    schedule += [(f'generated_cycle_{i+1}', {'id': f'generated_{j+1}', 'text': prompt,
                  'expected': {'language_required': True}, 'group': 'generated'})
                 for i in range(3) for j, prompt in enumerate(prompts)]
    if args.history_suite:
        schedule = [('history', {'id': f'history_{i+1}',
            'text': '我们一起写一个连续故事。请先用一句不超过二十个字的话写开头。' if i == 0 else '接着我们刚才的故事往下写，只增加一句不超过二十个字的新情节，不重复前文。',
            'expected': {'language_required': True}, 'group': 'history'}) for i in range(12)]
    ids = sorted({a['candidate_id'] for c in catalog['categories'] for a in c['children']})
    fallback = next(c['category_id'] for c in catalog['categories'] if 'silent_accompaniment' in c.get('semantic_tags', []))
    session_id = 'multiturn-' + uuid.uuid4().hex
    start = {'type': 'session.start', 'protocol_version': 1, 'session_id': session_id,
             'locale': 'zh-CN', 'outputs': ['text', 'audio', 'expression', 'action'],
             'reply': {'instructions': '自然简洁地回答。', 'unsupported_action_text': '这个动作暂时无法执行。'},
             'action': {'fallback_category_ids': [fallback], 'allowed_candidates': [{'candidate_id': x} for x in ids]}}
    rows = []
    report = {'session_id': session_id, 'url': args.url, 'start_unix_ms': time.time()*1000,
              'measurement': 'one connection, sequential turns; no history reset; wire commit-send to event receive; no playback',
              'schedule': '12 history-dependent story turns' if args.history_suite else '30 synthetic audio turns (3 cases x10), then 12 generated text turns (4 prompts x3); no discarded warmup', 'rows': rows}
    try:
        async with websockets.connect(args.url, max_size=16*1024*1024) as ws:
            t = time.perf_counter()
            await ws.send(json.dumps(start, ensure_ascii=False))
            await asyncio.wait_for(receive(ws, 'session.started'), 60)
            report['session_start_ms'] = (time.perf_counter()-t)*1000
            for n, (cycle, case) in enumerate(schedule, 1):
                row = {'ordinal': n, 'cycle': cycle, 'id': case['id'], 'input_kind': 'audio' if case.get('audio') else 'text',
                       'input_description': case['text'], 'group': 'audio' if case.get('audio') else 'generated', 'start_unix_ms': time.time()*1000}
                rows.append(row)
                try:
                    async with asyncio.timeout(60):
                        result, first, events = await run_turn(ws, case, audio_base, 60)
                    checks = judge(case, result, catalog)
                    row.update(result=result, wire_first_ms=first, events=events, checks=checks, passed=all(checks.values()))
                except Exception as exc:
                    row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                print(json.dumps({'turn': n, 'case': case['id'], 'passed': row['passed'], 'first_audio_ms': first.get('response.audio.delta'),
                                  'action_ms': first.get('turn.action.ready')}, ensure_ascii=False), flush=True)
            await ws.send(json.dumps({'type': 'session.close', 'reason': 'multiturn_benchmark_complete'}))
            await asyncio.wait_for(receive(ws, 'session.closed'), 30)
            await ws.wait_closed()
    finally:
        report['summary'] = summarize(rows)
        report['end_unix_ms'] = time.time()*1000
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return 0 if all(row['passed'] for row in rows) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-suite', action='store_true')
    parser.add_argument('--url', default='ws://127.0.0.1:18007/v1/session/realtime')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/session_multiturn_latency.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(asyncio.run(main(args)))
