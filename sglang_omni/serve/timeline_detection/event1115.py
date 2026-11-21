"""Frozen event codebook and output contract."""

import csv
from functools import lru_cache
import hashlib
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


@lru_cache(maxsize=1)
def event1115_allowed_ids() -> frozenset[str]:
    prompt = Path(os.environ["SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH"]).read_text()
    return frozenset(event1115_names(prompt))


def parse_event1115_output(raw_output: str) -> list[str]:
    text = raw_output.strip("\r\n")
    if text == "N":
        return []
    codes = text.split(",")
    if any(code not in event1115_allowed_ids() for code in codes):
        raise ModelOutputError("output contains an event code outside the frozen registry")
    if codes != sorted(codes, key=lambda code: int(code[1:])):
        raise ModelOutputError("event codes must be in numeric order")
    if any(codes.count(code) > 1 for code in set(codes) - {"E22", "E38", "E44"}):
        raise ModelOutputError("output contains unexpected repeated codes")
    if {"E8", "E50"}.issubset(codes):
        raise ModelOutputError("E8 and E50 are mutually exclusive")
    return codes
