# SPDX-License-Identifier: Apache-2.0
"""Instance-local FA4 selection for the supported Qwen3 vision inference path."""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

logger = logging.getLogger(__name__)


def _load_fa4():
    # Earlier installed beta builds failed on this SM120/head-dim=72 path.
    if Version(version("flash-attn-4")) < Version("4.0.0b38"):
        raise ImportError("vision FA4 requires flash-attn-4 >= 4.0.0b38")
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


def select_vision_attention(state, fallback):
    if state["backend"] == "sdpa":
        return fallback
    if not state.get("fa4_checked", False):
        state["fa4_checked"] = True
        try:
            state["fa4_func"] = _load_fa4()
        except (ImportError, PackageNotFoundError) as exc:
            state["fa4_func"] = None
            logger.warning("Vision FA4 unavailable; using SDPA: %s", exc)
        else:
            logger.info("Qwen3 vision attention backend: FA4")
    fn = state.get("fa4_func")
    if fn is None:
        return fallback

    def attention(
        module, q, k, v, attention_mask=None, scaling=None,
        dropout=0.0, is_causal=False, **kwargs,
    ):
        if attention_mask is not None or dropout != 0.0 or is_causal:
            return fallback(
                module, q, k, v, attention_mask=attention_mask,
                scaling=scaling, dropout=dropout, is_causal=is_causal, **kwargs,
            )
        # Preserve the tested B,S,H,D layout and scaling. Execution/compilation
        # errors are not swallowed (in particular CUDA OOM and illegal access).
        result = fn(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            softmax_scale=scaling, causal=False,
        )
        return (result[0] if isinstance(result, tuple) else result), None

    return attention
