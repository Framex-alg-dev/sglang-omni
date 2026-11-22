"""Frozen event codebook and output contract."""

import csv
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import re

from .parser import ModelOutputError


EVENT1115_PROTOCOL = "direct-1115-v1"
EVENT1115_IDS = frozenset(f"E{index}" for index in range(96))


@lru_cache(maxsize=1)
def event1115_names(system_prompt: str) -> dict[str, str]:
    directory = Path(os.environ["SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH"]).parent
    names = ("system_prompt.txt", "registry_1115.json", "codebook.csv")
    manifest = {
        filename.lstrip("*"): digest
        for digest, filename in (line.split() for line in (directory / "SHA256SUMS").read_text().splitlines())
    }
    for filename in names:
        content = (directory / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest.get(filename):
            raise ValueError(f"frozen event checksum mismatch: {filename}")
    if system_prompt.encode() != (directory / "system_prompt.txt").read_bytes():
        raise ValueError("system prompt differs from the frozen copy")
    registry = json.loads((directory / "registry_1115.json").read_text())
    with (directory / "codebook.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    codebook = {row["new_code"]: row["event_name"] for row in rows}
    event_ids = registry["codes"]
    if (
        not event_ids
        or len(event_ids) != len(set(event_ids))
        or not set(event_ids) <= EVENT1115_IDS
        or event_ids != sorted(event_ids, key=lambda code: int(code[1:]))
        or len(rows) != len(codebook)
        or list(codebook) != event_ids
        or registry["names"] != codebook
    ):
        raise ValueError("frozen event codebook and registry disagree")
    if re.findall(r"^(E\d+)\s", system_prompt, flags=re.MULTILINE) != event_ids:
        raise ValueError("frozen event prompt and codebook disagree")
    return codebook


def event1115_names_from_agent_snapshot(
    system_prompt: str,
    registry_json: str,
    codebook_csv: str,
) -> dict[str, str]:
    """Validate the Agent-owned E0..E95 registry without reading local files."""
    try:
        registry = json.loads(registry_json)
        if not isinstance(registry, dict) or set(registry) != {"events"}:
            raise ValueError("invalid registry shape")
        events = registry["events"]
        if not isinstance(events, list):
            raise ValueError("invalid registry events")
        registry_pairs = []
        for row in events:
            if (
                not isinstance(row, dict)
                or set(row) != {"code", "name"}
                or not isinstance(row["code"], str)
                or not isinstance(row["name"], str)
                or not row["name"].strip()
            ):
                raise ValueError("invalid registry row")
            registry_pairs.append((row["code"], row["name"]))
        reader = csv.DictReader(io.StringIO(codebook_csv, newline=""))
        if reader.fieldnames != ["code", "name"]:
            raise ValueError("invalid codebook header")
        codebook_pairs = [(row["code"], row["name"]) for row in reader]
        expected = [(f"E{index}", name) for index, (_code, name) in enumerate(registry_pairs)]
        if (
            len(registry_pairs) != 96
            or registry_pairs != expected
            or registry_pairs != codebook_pairs
            or registry_json not in system_prompt
        ):
            raise ValueError("registry, codebook and prompt disagree")
    except (csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Agent event prompt, registry and codebook disagree") from exc
    return dict(registry_pairs)


@lru_cache(maxsize=1)
def event1115_allowed_ids() -> frozenset[str]:
    prompt = Path(os.environ["SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH"]).read_text()
    return frozenset(event1115_names(prompt))


def parse_event1115_output(
    raw_output: str,
    *,
    allowed_event_ids: frozenset[str] | None = None,
) -> list[str]:
    text = raw_output.strip("\r\n")
    if text == "N":
        return []
    codes = text.split(",")
    allowed = event1115_allowed_ids() if allowed_event_ids is None else allowed_event_ids
    if any(code not in allowed for code in codes):
        raise ModelOutputError("output contains an event code outside the frozen registry")
    if codes != sorted(codes, key=lambda code: int(code[1:])):
        raise ModelOutputError("event codes must be in numeric order")
    if any(codes.count(code) > 1 for code in set(codes) - {"E22", "E38", "E44"}):
        raise ModelOutputError("output contains unexpected repeated codes")
    if {"E8", "E50"}.issubset(codes):
        raise ModelOutputError("E8 and E50 are mutually exclusive")
    return codes
