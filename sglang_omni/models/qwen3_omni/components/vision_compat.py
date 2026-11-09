# SPDX-License-Identifier: Apache-2.0
"""Transformers compatibility boundary for the Qwen3-Omni vision encoder."""

from __future__ import annotations

import logging
import os
import subprocess
import types
from collections import OrderedDict

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

logger = logging.getLogger(__name__)


def _vision_attention_forward(
    self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs
):
    # Retain upstream behavior for training, CPU, other precisions/backends and
    # deprecated call signatures. No HF module globals or checkpoint keys change.
    if (
        not self._omni_rope_state["enabled"]
        or self.training
        or torch.is_grad_enabled()
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        or self.config._attn_implementation != "sdpa"
        or position_embeddings is None
        or "rotary_pos_emb" in kwargs
        or self.head_dim != 72
        or torch.cuda.get_device_capability(hidden_states.device) != (12, 0)
        or hidden_states.shape[0] == 0
    ):
        return hf_modeling.Qwen3OmniMoeVisionAttention.forward(
            self,
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    cos, sin = position_embeddings
    if (
        cos.shape != (hidden_states.shape[0], self.head_dim)
        or sin.shape != cos.shape
        or cos.device != hidden_states.device
        or sin.device != hidden_states.device
        or cos.dtype not in (torch.bfloat16, torch.float32)
        or sin.dtype != cos.dtype
    ):
        return hf_modeling.Qwen3OmniMoeVisionAttention.forward(
            self,
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    try:
        from triton.compiler.errors import CompilationError

        from .vision_rope import fused_vision_rope
    except ImportError:
        self._omni_rope_state["enabled"] = False
        logger.warning("Triton unavailable; falling back to HF vision RoPE")
        return hf_modeling.Qwen3OmniMoeVisionAttention.forward(
            self,
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    try:
        q, k = fused_vision_rope(q, k, cos, sin)
    except (CompilationError, subprocess.CalledProcessError) as exc:
        # Disable for this encoder after a compiler failure. Do not swallow GPU
        # execution errors, OOMs, or unexpected programming errors.
        self._omni_rope_state["enabled"] = False
        logger.warning("Vision RoPE compilation failed; using HF fallback: %s", exc)
        q, k = hf_modeling.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    attention = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, hf_modeling.eager_attention_forward
    )
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
    chunks = [
        attention(
            self,
            qi,
            ki,
            vi,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0,
            is_causal=False,
            **kwargs,
        )[0]
        for qi, ki, vi in zip(*splits)
    ]
    output = torch.cat(chunks, dim=1).reshape(seq_length, -1).contiguous()
    return self.proj(output)


class Qwen3OmniMoeVisionEncoderCompat(hf_modeling.Qwen3OmniMoeVisionEncoder):
    """HF vision encoder with the Transformers 5.6 interpolation arithmetic."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        enabled = os.environ.get("SGLANG_OMNI_FUSED_VISION_ROPE", "1") != "0"
        state = {"enabled": enabled}
        for block in self.blocks:
            block.attn._omni_rope_state = state
            block.attn.forward = types.MethodType(_vision_attention_forward, block.attn)

    # Cache only geometry, never learned embeddings or image-dependent features.
    # Bound both shape count and retained device memory for variable-resolution input.
    _interpolation_cache_max_entries = 8
    _interpolation_cache_max_bytes = 16 * 1024 * 1024

    def clear_interpolation_cache(self) -> None:
        self._interpolation_cache = OrderedDict()

    def _apply(self, fn, recurse=True):
        self.clear_interpolation_cache()
        return super()._apply(fn, recurse=recurse)

    def _interpolation_geometry(self, grid_thw_list):
        device = self.pos_embed.weight.device
        # Separate streams and inference tensors: cached tensors must not be
        # consumed before their producer stream, or saved by an autograd graph.
        cacheable = not torch.is_grad_enabled()
        stream = (
            torch.cuda.current_stream(device).cuda_stream
            if device.type == "cuda"
            else None
        )
        key = (
            tuple(tuple(row) for row in grid_thw_list),
            self.num_grid_per_side,
            device,
            self.pos_embed.weight.dtype,
            torch.get_default_dtype(),
            torch.is_inference_mode_enabled(),
            stream,
        )
        if not hasattr(self, "_interpolation_cache"):
            self.clear_interpolation_cache()
        cache = self._interpolation_cache
        if cacheable and key in cache:
            cache.move_to_end(key)
            return cache[key]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # Keep these coordinate and weight operations byte-for-byte equivalent
        # to the Transformers 5.6 implementation. In particular, linspace runs
        # on CPU and the materialized weights use the embedding table dtype.
        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for corner in range(4):
                idx_list[corner].extend(indices[corner].tolist())
                weight_list[corner].extend(weights[corner].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=device,
        )

        value = (idx_tensor, weight_tensor)
        size = sum(t.numel() * t.element_size() for t in value)
        if cacheable and size <= self._interpolation_cache_max_bytes:
            cache[key] = value
            while (
                len(cache) > self._interpolation_cache_max_entries
                or sum(
                    t.numel() * t.element_size()
                    for pair in cache.values()
                    for t in pair
                )
                > self._interpolation_cache_max_bytes
            ):
                cache.popitem(last=False)
        return value

    def _legacy_pos_embed_interpolate(
        self,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_tensor, weight_tensor = self._interpolation_geometry(grid_thw_list)
        corners = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = corners[0] + corners[1] + corners[2] + corners[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf_modeling.BaseModelOutputWithDeepstackFeatures:
        """Run the HF 5.12.1 encoder with legacy positional interpolation."""
        # HF 5.12 may pass its device-generated interpolation tensors through
        # kwargs. This compatibility path always recomputes the legacy values.
        kwargs.pop("bilinear_indices", None)
        kwargs.pop("bilinear_weights", None)
        position_ids = hf_modeling.get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens = hf_modeling.get_vision_cu_seqlens(grid_thw, kwargs=kwargs)

        # HF's non-FlashAttention path only uses these boundaries to split Q/K/V.
        # Keep them on CPU so lengths.tolist() does not synchronize CUDA three
        # times per layer. FlashAttention still needs device-resident boundaries.
        if not hf_modeling.is_flash_attention_requested(self.config):
            cu_seqlens = cu_seqlens.cpu()

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self._legacy_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_index](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return hf_modeling.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
