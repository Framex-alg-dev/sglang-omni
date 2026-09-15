import random

import pytest

from sglang_omni.serve.realtime.tts_text import StreamingTTSWhitespace, split_tts_append


def normalize(parts):
    processor = StreamingTTSWhitespace()
    result = "".join(processor.feed(part) for part in parts)
    processor.finish()
    assert processor.pending == ""
    return result


@pytest.mark.parametrize("text,expected", [
    ("  Hello\t  world\r\n\n Next.  ", "Hello world\nNext."),
    ("\n \t\r", ""),
    ("what's new?  Don't change 3.14 or v1.2.3!", "what's new? Don't change 3.14 or v1.2.3!"),
    (" 你好\n\n world\t你好 ", "你好\nworld 你好"),
    ("e\u0301 👨\u200d👩\u200d👧\u200d👦\u00a0x", "e\u0301 👨\u200d👩\u200d👧\u200d👦\u00a0x"),
])
def test_normalization_independent_of_delta_partition(text, expected):
    assert normalize([text]) == expected
    assert normalize(list(text)) == expected
    for cut in range(len(text) + 1):
        assert normalize([text[:cut], "", text[cut:]]) == expected
    rng = random.Random(91)
    for _ in range(100):
        cuts = sorted({0, len(text), *[rng.randrange(len(text) + 1) for _ in range(10)]})
        assert normalize([text[a:b] for a, b in zip(cuts, cuts[1:])]) == expected


def test_no_sentence_wait_and_no_cross_turn_separator():
    a = StreamingTTSWhitespace()
    assert a.feed("  Hel") == "Hel"
    assert a.feed("lo ") == "lo"
    assert a.feed(" \r") == ""
    assert a.feed("\nworld") == "\nworld"
    assert StreamingTTSWhitespace().feed(" new") == "new"


def test_whitespace_accounting():
    processor = StreamingTTSWhitespace()
    result = processor.feed("  Hello \r") + processor.feed("\n  world  ")
    processor.finish()
    assert result == "Hello\nworld"
    assert processor.input_chars - processor.output_chars == (
        processor.leading_whitespace_chars + processor.trailing_whitespace_chars
        + processor.collapsed_whitespace_chars
    )
    assert processor.cr_chars == 1


def test_append_splitting_preserves_words_and_unicode():
    text = "What's new? 3.14 https://example.org/a/b e\u0301 👨\u200d👩\u200d👧\u200d👦。下一句，结束。" * 20
    pieces = split_tts_append(text, 32, 128)
    assert "".join(pieces) == text
    assert max(map(len, pieces)) <= 128
    for token in ["What's", "3.14", "https://example.org/a/b", "e\u0301", "👨\u200d👩\u200d👧\u200d👦"]:
        assert sum(p.count(token) for p in pieces) == text.count(token)


def test_indivisible_atom_fails_instead_of_truncation():
    with pytest.raises(ValueError, match="budget"):
        split_tts_append("x" * 129, 32, 128)
    assert split_tts_append("what'", 0, 128) == ["what'"]
    assert split_tts_append("x" * 80, 32, 128) == ["x" * 80]
