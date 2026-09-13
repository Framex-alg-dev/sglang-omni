import asyncio
import json
import pytest

from sglang_omni.client.client import _PriorityAdmissionGate
from sglang_omni.serve.realtime.turn_intent import TurnIntent
from sglang_omni.utils.prepared_media_cache import PreparedMediaCache
from sglang_omni.profiler.event_recorder import _json_default


def test_private_media_budgets_expiry_and_one_shot():
    now = [0.0]
    cache = PreparedMediaCache(max_entries=3, max_bytes=12, per_session_bytes=8, ttl_seconds=2, clock=lambda: now[0])
    cache.put(('a', '1'), b'12345')
    cache.put(('b', '1'), b'12345')
    cache.put(('a', '2'), b'67890')
    assert cache.pop(('a', '1')) is None
    assert cache.pop(('b', '1')) == b'12345'
    assert cache.pop(('b', '1')) is None
    now[0] = 3
    assert cache.pop(('a', '2')) is None
    assert cache.current_bytes == 0
    cache.put(('a', '3'), b'x' * 9)
    assert cache.pop(('a', '3')) is None
    cache.put(('b', '3'), b'12')
    cache.release_session('a')
    assert cache.pop(('b', '3')) == b'12'


def test_voice_plan_is_independent_of_face_and_preserves_contractions():
    intent = TurnIntent.parse(json.dumps(dict(speech='verbatim', text="What's new?", body='', body_mode='none', face='严肃', history=False, voice_tone='cheerful', voice_pace='fast')))
    assert intent.text == "What's new?"
    assert 'cheerful' in intent.tts_instruction()
    assert 'serious' not in intent.tts_instruction()
    with pytest.raises(ValueError):
        TurnIntent.parse(json.dumps(dict(speech='none', text='', body='', body_mode='none', face='', history=False, voice_tone='execute arbitrary code')))


def test_gpu_scalar_logging_never_synchronizes():
    class Scalar:
        shape, dtype, device = (), 'float32', 'cuda:0'
        def item(self):
            raise AssertionError('GPU synchronization')
    assert _json_default(Scalar())['shape'] == []


@pytest.mark.asyncio
async def test_fair_admission_round_robin_and_cancelled_waiter():
    gate = _PriorityAdmissionGate(1)
    await gate.acquire(0, 'holder')
    order = []
    async def run(session):
        async with gate.slot(0, session):
            order.append(session)
            await asyncio.sleep(0)
    tasks = [asyncio.create_task(run(s)) for s in ['a', 'a', 'a', 'b', 'b', 'c', 'c']]
    cancelled = asyncio.create_task(run('cancelled'))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await gate.release()
    await asyncio.gather(*tasks)
    assert order == ['a', 'b', 'c', 'a', 'b', 'c', 'a']
    assert gate._active == 0 and not gate._waiters


@pytest.mark.asyncio
async def test_admission_session_queue_cap_does_not_block_other_session():
    gate = _PriorityAdmissionGate(1, max_session_waiting=1)
    await gate.acquire(0, 'holder')
    waiting = asyncio.create_task(gate.acquire(0, 'a'))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match='queue is full'):
        await gate.acquire(0, 'a')
    other = asyncio.create_task(gate.acquire(0, 'b'))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await gate.release()
    await other
    await gate.release()


@pytest.mark.asyncio
async def test_pending_reply_is_not_silent_and_cancellation_is_preserved():
    from sglang_omni.serve.realtime.reply.provisional import ProvisionalReplyComponent
    from sglang_omni.serve.realtime.protocol.models import ProvisionalReplyState
    from types import SimpleNamespace
    state = ProvisionalReplyState('reply', 'generated', 0, 0)
    session = SimpleNamespace(_ensure_turn_processing=lambda turn: None, _provisional_reply_prefix=lambda text: (text, False))
    prefix, status, elapsed = await ProvisionalReplyComponent._resolve_provisional_reply_prefix(session, object(), state)
    assert (prefix, status) == ('', 'reply_pending')
    assert 80 <= elapsed < 300
    state.completed = True
    state.content_available.set()
    assert (await ProvisionalReplyComponent._resolve_provisional_reply_prefix(session, object(), state))[1] == 'empty_completed'
    state.failed = True
    assert (await ProvisionalReplyComponent._resolve_provisional_reply_prefix(session, object(), state))[1] == 'reply_failed'
    fresh = ProvisionalReplyState('reply2', 'generated', 0, 0)
    task = asyncio.create_task(ProvisionalReplyComponent._resolve_provisional_reply_prefix(session, object(), fresh))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_public_prefix_requires_published_source_and_exact_tokens(monkeypatch):
    from sglang_omni.models.qwen3_omni import public_prefix
    monkeypatch.setattr(public_prefix, 'published_prompts', lambda: frozenset({'catalog'}))
    class Tokenizer:
        calls = 0
        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return [1, 2, 3]
    tokenizer = Tokenizer()
    verifier = public_prefix.PublicPrefixVerifier(tokenizer)
    assert verifier.boundary('private system with whitelist', [1, 2, 3]) == 0
    assert verifier.boundary('catalog', [1, 2, 4]) == 0
    assert verifier.boundary('catalog', [1, 2, 3, 4]) == 3
    assert tokenizer.calls == 1


def test_entity_projection_is_explicit_and_preserves_constraints(monkeypatch):
    from sglang_omni.serve.realtime.action.entity_view import action_entity_text
    raw = '{"name":"箱子","constraints":["不能举起"],"news_body":"长新闻","unknown_fact":42}'
    monkeypatch.delenv('SGLANG_OMNI_ACTION_ENTITY_OMIT_FIELDS', raising=False)
    assert action_entity_text(raw) == raw
    monkeypatch.setenv('SGLANG_OMNI_ACTION_ENTITY_OMIT_FIELDS', '["news_body"]')
    assert json.loads(action_entity_text(raw)) == {'name':'箱子','constraints':['不能举起'],'unknown_fact':42}
    assert action_entity_text('非结构化资料必须保留全部约束。') == '非结构化资料必须保留全部约束。'
    monkeypatch.setenv('SGLANG_OMNI_ACTION_ENTITY_OMIT_FIELDS', '["constraints"]')
    with pytest.raises(ValueError):
        action_entity_text(raw)


def test_standalone_speech_segmentation_preserves_words_numbers_and_urls():
    from types import SimpleNamespace
    from sglang_omni.serve.speech_ws import SpeechWebSocketSession
    for text in ["What's new?", '3.14 is pi.', 'Dr. Smith is here.', 'https://example.com/a is a link.']:
        for split in range(len(text) + 1):
            session = object.__new__(SpeechWebSocketSession)
            session.config = SimpleNamespace(split_granularity='clause')
            session.buffer = ''
            pieces = []
            for chunk in [text[:split], text[split:]]:
                session.buffer += chunk
                pieces.extend(session._pop_complete_segments())
            if session.buffer.strip():
                pieces.append(session.buffer.strip())
            assert ' '.join(pieces) == text


def test_body_view_preserves_qualifiers_and_does_not_repeat_spoken_commands():
    import json
    intent = TurnIntent(speech='verbatim', text='别挥手', body='用左手挥手，右手保持不动', body_mode='perform', face='微笑', history=False)
    body = json.loads(intent.body_context('用左手挥手并说别挥手，右手不动'))
    assert body['body_task'] == intent.body
    assert body['body_mode'] == 'perform'
    assert 'speech_task' not in body and 'original_text' not in body
    assert json.loads(intent.action_context('original'))['speech_task'] == '别挥手'
    from dataclasses import replace
    unresolved = replace(intent, history=True)
    assert json.loads(unresolved.body_context('原话'))['original_text'] == '原话'
    automatic = replace(intent, body='', body_mode='none')
    assert json.loads(automatic.body_context('原话'))['speech_task'] == '别挥手'
    prohibited = replace(intent, body='挥手', body_mode='prohibit')
    assert json.loads(prohibited.body_context('禁止挥手说你好'))['body_task'] == '禁止挥手'
