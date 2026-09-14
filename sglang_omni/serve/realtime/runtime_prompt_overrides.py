"""Hot-reloadable development overrides for realtime prompt policy text."""

from __future__ import annotations

import os
from pathlib import Path


_RUNTIME_DIR_ENV = "SGLANG_OMNI_RUNTIME_PROMPT_DIR"
_MAX_PROMPT_CHARS = 100_000
_FILENAMES = {
    "reply_rules": "reply_rules.txt",
    "action_rules": "action_rules.txt",
    "proactive_reply_rules": "proactive_reply_rules.txt",
    "proactive_action_rules": "proactive_action_rules.txt",
}
_PROACTIVE_SECTION_NAMES = frozenset(
    {
        "session_enter",
        "idle_timeout",
        "user_returned",
        "character_proactive",
        "session_ending",
    }
)
_SECTION_NAMES = {
    "proactive_reply_rules": _PROACTIVE_SECTION_NAMES,
    "proactive_action_rules": _PROACTIVE_SECTION_NAMES,
}


def runtime_prompt_dir() -> Path:
    configured = os.getenv(_RUNTIME_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "runtime"


def read_runtime_prompt(key: str) -> str | None:
    path = _prompt_path(key)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    content = content.strip()
    return content or None


def effective_runtime_prompt(key: str, default: str) -> str:
    return read_runtime_prompt(key) or default


def read_runtime_prompt_section(key: str, section: str) -> str | None:
    content = read_runtime_prompt(key)
    if content is None:
        return None
    sections = _parse_sections(content, _SECTION_NAMES.get(key, frozenset()))
    if not sections:
        return content
    return sections.get(section)


def write_runtime_prompt(key: str, content: str) -> None:
    if not isinstance(content, str):
        raise ValueError("runtime prompt content must be a string")
    if len(content) > _MAX_PROMPT_CHARS:
        raise ValueError(
            f"runtime prompt content cannot exceed {_MAX_PROMPT_CHARS} characters"
        )
    path = _prompt_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prompt_slot_payloads(defaults: dict[str, str]) -> list[dict[str, object]]:
    if set(defaults) != set(_FILENAMES):
        raise ValueError("runtime prompt defaults must contain every supported key")
    descriptions = {
        "reply_rules": "sglang-omni 对主动和用户回复都生效的服务端通用回复规则",
        "action_rules": "sglang-omni 对主动和用户动作都生效的通用动作选择规则",
        "proactive_reply_rules": "sglang-omni 对所有主动场景生效的强制回复规则",
        "proactive_action_rules": "sglang-omni 对所有主动场景生效的动作目标指导",
    }
    payloads: list[dict[str, object]] = []
    for key, filename in _FILENAMES.items():
        override = read_runtime_prompt(key)
        default = defaults[key]
        payloads.append(
            {
                "owner": "sglang_omni",
                "key": key,
                "filename": filename,
                "description": descriptions[key],
                "source": "runtime" if override is not None else "repository_default",
                "override_content": override or "",
                "effective_content": override or default,
                "default_content": default,
                "activation": "new_turn",
            }
        )
    return payloads


def _prompt_path(key: str) -> Path:
    try:
        filename = _FILENAMES[key]
    except KeyError as exc:
        raise ValueError(f"unknown runtime prompt key: {key}") from exc
    return runtime_prompt_dir() / filename


def _parse_sections(content: str, allowed_names: frozenset[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip()
            if name in allowed_names:
                current = name
                sections.setdefault(name, [])
                continue
        if current is not None:
            sections[current].append(line)
    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if "\n".join(lines).strip()
    }
