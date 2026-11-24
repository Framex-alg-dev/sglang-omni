import base64
import random

import pytest

from sglang_omni.preprocessing import image


def test_decoder_matches_stdlib_for_valid_payloads():
    rng = random.Random(42)
    for n in [0, 1, 2, 3, 17, 1024, 65537]:
        raw = rng.randbytes(n)
        encoded = base64.b64encode(raw).decode()
        assert image._decode_image_base64(encoded) == raw
        assert image._decode_image_base64('\n'.join(encoded[i:i+76] for i in range(0, len(encoded), 76))) == raw


def test_fallback_without_optional_decoder(monkeypatch):
    monkeypatch.setattr(image, '_fast_b64decode', None)
    assert image._decode_image_base64('aGVsbG8=') == b'hello'


def test_stdlib_errors_and_permissive_inputs():
    rng = random.Random(13)
    samples = ['a', 'a===', 'abcd===', '中文', 'aGVsbG8=garbage', '!!!!', '=a=b=', 'aGVs\x00bG8=']
    alphabet = 'ABCDabcd0123+/=! \n'
    samples += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80))) for _ in range(1000)]
    for data in samples:
        try:
            expected = base64.b64decode(data)
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error)) as caught:
                image._decode_image_base64(data)
            assert str(caught.value) == str(error)
        else:
            assert image._decode_image_base64(data) == expected
