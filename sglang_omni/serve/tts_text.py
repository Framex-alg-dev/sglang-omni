"""Streaming whitespace normalization for ordinary TTS append messages.

An append is NOT a synthesis boundary. This module never inserts punctuation
or waits for a complete sentence. State belongs to one synthesis invocation.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


class StreamingTTSWhitespace:
    def __init__(self) -> None:
        self.started = False
        self.pending = ""
        self.input_chars = 0
        self.output_chars = 0
        self.leading_whitespace_chars = 0
        self.trailing_whitespace_chars = 0
        self.collapsed_whitespace_chars = 0
        self.cr_chars = 0
        self._pending_count = 0

    def feed(self, text: str) -> str:
        self.input_chars += len(text)
        out: list[str] = []
        for char in text:
            if char in " \t\r\n":
                self.cr_chars += int(char == "\r")
                if self.started:
                    self._pending_count += 1
                    if char in "\r\n":
                        self.pending = "\n"
                    elif not self.pending:
                        self.pending = " "
                else:
                    self.leading_whitespace_chars += 1
                continue
            if self.pending:
                self.collapsed_whitespace_chars += self._pending_count - 1
                self._pending_count = 0
                out.append(self.pending)
                self.pending = ""
            out.append(char)
            self.started = True
        result = "".join(out)
        self.output_chars += len(result)
        return result

    def finish(self) -> None:
        # Only a separator awaiting following content remains. EOF makes it
        # trailing whitespace. Never append it just to force gateway synthesis.
        self.pending = ""
        self.trailing_whitespace_chars += self._pending_count
        self._pending_count = 0


def split_tts_append(text: str, target: int, hard_limit: int) -> list[str]:
    """Prefer existing separators; fail rather than cut a long protected atom.

    Small incoming model deltas may already contain partial words: they remain
    ordinary appends, which the gateway concatenates. We don't add new cuts
    inside words, URLs, apostrophes, or combining sequences in large deltas.
    """
    if not text:
        return []
    if target == 0:
        if len(text) > hard_limit:
            raise ValueError("TTS append character budget exceeded")
        return [text]
    pieces: list[str] = []
    while len(text) > target:
        boundaries = [
            i for i in range(1, min(len(text), hard_limit) + 1)
            if text[i - 1] in " \t\r\n。！？；，、"
            and (i == len(text) or not (
                unicodedata.category(text[i]).startswith("M")
                or text[i] in "\u200d\ufe0f"
            ))
        ]
        before = [i for i in boundaries if i <= target]
        cut = max(before) if before else next(iter(boundaries), None)
        if cut is None:
            if len(text) > hard_limit:
                raise ValueError("TTS indivisible append exceeds character budget")
            break
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


@dataclass(frozen=True)
class TTSTextAppend:
    text: str
    source_first: int
    source_last: int
    received_at: float
    reason: str
