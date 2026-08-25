# SPDX-License-Identifier: Apache-2.0
"""Realtime session APIs and shared session primitives.

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.serve.realtime.manager import RealtimeSessionManager
    from sglang_omni.serve.realtime.multimodal import (
        MultimodalSession,
        MultimodalSessionManager,
    )
    from sglang_omni.serve.realtime.session import RealtimeSession

__all__ = [
    "RealtimeSession",
    "RealtimeSessionManager",
    "MultimodalSession",
    "MultimodalSessionManager",
]


def __getattr__(name: str):
    """Keep legacy and multimodal implementations isolated until requested."""
    if name == "RealtimeSession":
        from sglang_omni.serve.realtime.session import RealtimeSession

        return RealtimeSession
    if name == "RealtimeSessionManager":
        from sglang_omni.serve.realtime.manager import RealtimeSessionManager

        return RealtimeSessionManager
    if name in {"MultimodalSession", "MultimodalSessionManager"}:
        from sglang_omni.serve.realtime.multimodal import (
            MultimodalSession,
            MultimodalSessionManager,
        )

        return {
            "MultimodalSession": MultimodalSession,
            "MultimodalSessionManager": MultimodalSessionManager,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
