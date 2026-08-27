# SPDX-License-Identifier: Apache-2.0
"""Canonical Session Realtime output capability parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OUTPUT_ORDER = ("text", "audio", "action")
SUPPORTED_OUTPUTS = frozenset(OUTPUT_ORDER)
DEFAULT_OUTPUTS = ("text", "action")


@dataclass(frozen=True)
class SessionOutputCapabilities:
    """Capabilities derived exclusively from ``session.start.outputs``."""

    outputs: tuple[str, ...]
    text_enabled: bool
    audio_enabled: bool
    action_enabled: bool

    @classmethod
    def parse(cls, value: Any) -> "SessionOutputCapabilities":
        if value is None:
            outputs = DEFAULT_OUTPUTS
        else:
            if not isinstance(value, list) or not value:
                raise ValueError("outputs must be a non-empty list")
            if not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError("outputs entries must be non-empty strings")
            if len(set(value)) != len(value):
                raise ValueError("outputs must not contain duplicates")
            unsupported = sorted(set(value) - SUPPORTED_OUTPUTS)
            if unsupported:
                raise ValueError("unsupported outputs: " + ", ".join(unsupported))
            outputs = tuple(item for item in OUTPUT_ORDER if item in value)

        if "audio" in outputs and "text" not in outputs:
            raise ValueError("audio output requires the text output")
        return cls(
            outputs=outputs,
            text_enabled="text" in outputs,
            audio_enabled="audio" in outputs,
            action_enabled="action" in outputs,
        )
