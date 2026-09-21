import asyncio
import time

import pytest

from sglang_omni.serve.realtime.tts_buffer import buffered_appends, text_boundary
from sglang_omni.serve.realtime.tts_text import TTSTextAppend


def item(text, seq=1):
    return TTSTextAppend(text, seq, seq, time.monotonic(), 'delta')


@pytest.mark.asyncio
async def test_timer_sends_without_next_delta_and_resets_per_batch():
    q = asyncio.Queue()
    q.put_nowait(item('Hello'))
    stream = buffered_appends(q, first_wait=.03, later_wait=.04)
    started = time.monotonic()
    result = await asyncio.wait_for(anext(stream), .3)
    assert result.text == 'Hello' and result.reason == 'buffer_deadline'
    assert time.monotonic() - started >= .02
    q.put_nowait(item(' world'))
    q.put_nowait(None)
    assert (await anext(stream)).text == ' world'
    await stream.aclose()


@pytest.mark.asyncio
async def test_arrivals_do_not_reset_deadline():
    q = asyncio.Queue()
    q.put_nowait(item('a'))
    async def feed():
        for _ in range(20):
            await asyncio.sleep(.01)
            q.put_nowait(item('b'))
    producer = asyncio.create_task(feed())
    stream = buffered_appends(q, first_wait=.04, later_wait=.06)
    try:
        result = await asyncio.wait_for(anext(stream), .15)
        assert result.reason == 'buffer_deadline'
        assert not producer.done()
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        await stream.aclose()


@pytest.mark.asyncio
async def test_eof_coalesces_and_preserves_exact_text_and_source():
    q = asyncio.Queue()
    chunks = ['Hey', ' there', '!', ' How', '’s', ' it', ' going', '?']
    for i, text in enumerate(chunks, 1):q.put_nowait(item(text, i))
    q.put_nowait(None)
    out = [v async for v in buffered_appends(q, first_wait=.06, later_wait=.1)]
    assert ''.join(v.text for v in out) == ''.join(chunks)
    assert len(out) < len(chunks)
    assert out[0].source_first == 1 and out[-1].source_last == 8


@pytest.mark.parametrize('text', ['3.14159', "What's", 'https://example.org/a?b=1', 'U.S.A.', 'e\u0301'])
def test_protected_atoms_do_not_trigger_punctuation(text):
    assert text_boundary(text, first=True) is None


def test_boundary_includes_punctuation_and_closing_quote():
    text = '这是一个完整的句子。”下一句'
    cut, reason = text_boundary(text, first=True)
    assert text[:cut] == '这是一个完整的句子。”'
    assert reason == 'punctuation'


@pytest.mark.asyncio
async def test_cancel_does_not_leave_queue_get_task():
    q = asyncio.Queue()
    q.put_nowait(item('hello'))
    stream = buffered_appends(q, first_wait=1, later_wait=1)
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    q.put_nowait(item('new'))
    await asyncio.sleep(0)
    assert q.qsize() == 1


def test_cjk_length_boundary_and_english_word_boundary():
    assert text_boundary('汉' * 50, first=True) == (16, 'length')
    cut, _ = text_boundary('A sufficiently long English sentence with words', first=True)
    assert 'A sufficiently long English sentence with words'[cut - 1].isspace()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['好。', 'OK!', 'hello world 3.14 https://a.b/x?q=1', '中英文 hello e\u0301 👩\u200d💻 混合。下一句！'])
async def test_arbitrary_delta_splits_are_lossless(text):
    for step in (1, 3, 7):
        q = asyncio.Queue()
        for i in range(0, len(text), step):q.put_nowait(item(text[i:i+step]))
        q.put_nowait(None)
        out = [v.text async for v in buffered_appends(q, first_wait=.06, later_wait=.1)]
        assert ''.join(out) == text
