from __future__ import annotations

import hashlib
from pathlib import Path

from sglang_omni.models.qwen3_omni.public_prefix import (
    PublicPrefixVerifier,
    is_published_prompt,
)


def test_timeline_e50_direct_prompt_is_an_approved_public_prefix() -> None:
    prompt_path = (
        Path(__file__).parents[3]
        / "deploy/prompts/e50-direct-checkpoint1500/system_prompt_e50_direct.txt"
    )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    assert hashlib.sha256(system_prompt.encode()).hexdigest() == (
        "ccd76772db751450205dae0f82beea877f0bec5f4c0d1d98bf03ef1feef90a44"
    )
    assert is_published_prompt(system_prompt)
    assert not is_published_prompt(system_prompt + "\nchanged")

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
    verifier = PublicPrefixVerifier(Tokenizer(), Processor())
    assert verifier.boundary(system_prompt, full_ids) == len(full_ids)
