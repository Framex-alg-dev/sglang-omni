from __future__ import annotations

import hashlib
from sglang_omni.models.qwen3_omni import public_prefix


def test_agent_owned_timeline_e50_digest_is_approved() -> None:
    expected = "ccd76772db751450205dae0f82beea877f0bec5f4c0d1d98bf03ef1feef90a44"
    assert expected in public_prefix._PUBLISHED_EXTERNAL_PROMPT_SHA256


def test_public_prefix_boundary_still_rejects_unpublished_agent_text() -> None:
    system_prompt = "synthetic Agent snapshot that is not a reviewed public prefix"
    assert hashlib.sha256(system_prompt.encode()).hexdigest() not in (
        public_prefix._PUBLISHED_EXTERNAL_PROMPT_SHA256
    )
    assert not public_prefix.is_published_prompt(system_prompt)

    class Tokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return list(text.encode("utf-8"))

    class Processor:
        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, object]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is False
            content = messages[0]["content"]
            assert isinstance(content, list)
            text = content[0]["text"]
            assert isinstance(text, str)
            return text

    full_ids = Tokenizer.encode(system_prompt)
    verifier = public_prefix.PublicPrefixVerifier(Tokenizer(), Processor())
    assert verifier.boundary(system_prompt, full_ids) == 0
