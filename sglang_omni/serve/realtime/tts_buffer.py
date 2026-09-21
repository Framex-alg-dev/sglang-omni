"""Bounded text batching for ordinary TTS appends, not synthesis boundaries."""
from __future__ import annotations

import asyncio
import time
import unicodedata
from collections import deque
from dataclasses import replace
from typing import AsyncIterator

from .tts_text import TTSTextAppend


def text_boundary(text: str, *, first: bool) -> tuple[int, str] | None:
    """Use Unicode content/whitespace/punctuation, never phrase dictionaries.

    ASCII periods are deliberately ambiguous (decimal/abbreviation/URL). They
    do not independently trigger an early flush. The deadline still applies.
    """
    target = 16 if first else 40
    minimum = 4 if first else 8
    content = 0
    last_space = 0
    closing = '\"\'”’）)]}」』'
    for i, char in enumerate(text):
        content += int(char.isalnum())
        cut = i + 1
        if cut < len(text) and (unicodedata.category(text[cut]).startswith('M') or text[cut] in '\u200d\ufe0f'):
            continue
        if char.isspace():
            last_space = cut
        strong = char in '。！？\n' or (char in '!?' and cut < len(text) and text[cut].isspace())
        weak = char in '，；、' or (char in ',;:' and cut < len(text) and text[cut].isspace())
        if strong or weak:
            while cut < len(text) and text[cut] in closing + '。！？!?':
                cut += 1
            if content >= minimum and (strong or content >= target):
                return cut, 'punctuation'
        cjk = '\u3400' <= char <= '\u9fff'
        next_cjk = cut < len(text) and '\u3400' <= text[cut] <= '\u9fff'
        if content >= target and cjk and next_cjk:
            return cut, 'length'
        if content >= target and char.isspace():
            return cut, 'length'
        if content >= target * 2 and last_space:
            return last_space, 'length'
    return None


async def buffered_appends(
    queue: asyncio.Queue[TTSTextAppend | None], *, first_wait: float, later_wait: float,
) -> AsyncIterator[TTSTextAppend]:
    pending: deque[TTSTextAppend] = deque()
    first = True
    eof = False

    def consume(count: int, reason: str) -> TTSTextAppend:
        head = pending[0]
        pieces = []
        last = head
        while count:
            last = pending.popleft()
            used = min(count, len(last.text))
            pieces.append(last.text[:used])
            count -= used
            if used < len(last.text):
                pending.appendleft(replace(last, text=last.text[used:]))
        return TTSTextAppend(''.join(pieces), head.source_first, last.source_last,
                             head.received_at, reason)

    while True:
        if not pending:
            if eof:
                return
            item = await queue.get()
            if item is None:
                return
            pending.append(item)
        deadline = pending[0].received_at + (first_wait if first else later_wait)
        # Drain already available deltas before inspecting punctuation. Bound
        # this work by the original deadline; arrivals never reset it.
        while not eof and time.monotonic() < deadline:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                eof = True
            else:
                pending.append(item)
        text = ''.join(item.text for item in pending)
        boundary = text_boundary(text, first=first)
        if boundary:
            count, reason = boundary
        elif eof:
            count, reason = len(text), 'eof'
        elif time.monotonic() >= deadline:
            count, reason = len(text), 'buffer_deadline'
        else:
            try:
                item = await asyncio.wait_for(queue.get(), max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                item = None
                # Timeout flushes existing text, but is not producer EOF.
                yield consume(len(text), 'buffer_deadline')
                first = False
                continue
            if item is None:
                eof = True
            else:
                pending.append(item)
            continue
        yield consume(count, reason)
        first = False
