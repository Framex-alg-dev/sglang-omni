# SPDX-License-Identifier: Apache-2.0
"""Language selection helpers for server-authored Qwen3-Omni prompts."""

from __future__ import annotations


SUPPORTED_PROMPT_LOCALES = ("zh-CN", "en-US")
DEFAULT_PROMPT_LOCALE = "en-US"
PROMPT_LANGUAGE_BY_LOCALE = {"zh-CN": "zh", "en-US": "en"}


def normalize_prompt_language(value: str) -> str:
    """Normalize an internal language or public locale to ``zh`` or ``en``."""

    if value in {"zh", "zh-CN"}:
        return "zh"
    if value in {"en", "en-US"}:
        return "en"
    raise ValueError(f"unsupported prompt language or locale: {value!r}")


def localized_prompt(value: str, *, zh: str, en: str) -> str:
    """Select server-authored prompt text without touching client content."""

    return zh if normalize_prompt_language(value) == "zh" else en


__all__ = [
    "DEFAULT_PROMPT_LOCALE",
    "PROMPT_LANGUAGE_BY_LOCALE",
    "SUPPORTED_PROMPT_LOCALES",
    "localized_prompt",
    "normalize_prompt_language",
]
