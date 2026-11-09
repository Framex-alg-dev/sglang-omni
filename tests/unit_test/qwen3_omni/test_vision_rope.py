# SPDX-License-Identifier: Apache-2.0
import subprocess
from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

from sglang_omni.models.qwen3_omni.components import vision_compat


def config():
    return hf.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=1152,
        intermediate_size=32,
        num_heads=16,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=(0,),
        _attn_implementation="sdpa",
    )


def test_binding_preserves_checkpoint_and_does_not_patch_hf(monkeypatch):
    original = hf.Qwen3OmniMoeVisionAttention.forward
    monkeypatch.setenv("SGLANG_OMNI_FUSED_VISION_ROPE", "0")
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config())
    reference = hf.Qwen3OmniMoeVisionEncoder(config())
    assert encoder.state_dict().keys() == reference.state_dict().keys()
    assert hf.Qwen3OmniMoeVisionAttention.forward is original
    assert not encoder.blocks[0].attn._omni_rope_state["enabled"]
    assert (
        encoder.blocks[0].attn._omni_rope_state
        is encoder.blocks[1].attn._omni_rope_state
    )
    reference.load_state_dict(encoder.state_dict(), strict=True)


def test_cpu_autograd_fallback():
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config())
    attn = encoder.blocks[0].attn
    x = torch.randn(8, 1152, requires_grad=True)
    pos = torch.randn(8, 72)
    lengths = torch.tensor([0, 4, 8], dtype=torch.int32)
    actual = attn(x, lengths, (pos.cos(), pos.sin()))
    expected = hf.Qwen3OmniMoeVisionAttention.forward(
        attn, x, lengths, (pos.cos(), pos.sin())
    )
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Optimized path targets SM120",
)


@cuda
@pytest.mark.parametrize("length", [1, 17, 257, 1024])
@pytest.mark.parametrize("position_dtype", [torch.float32, torch.bfloat16])
def test_fused_rope_bit_exact_strided_qkv(length, position_dtype):
    from sglang_omni.models.qwen3_omni.components.vision_rope import fused_vision_rope

    torch.manual_seed(19)
    # Match the non-contiguous Q/K views produced by the interleaved QKV projection.
    qkv = torch.randn(length, 3, 16, 72, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.permute(1, 0, 2, 3).unbind(0)
    pos = torch.randn(length, 144, device="cuda")[:, ::2].to(position_dtype)
    cos, sin = pos.cos(), pos.sin()
    with torch.inference_mode():
        expected = hf.apply_rotary_pos_emb_vision(q, k, cos, sin)
        actual = fused_vision_rope(q, k, cos, sin)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@cuda
def test_attention_bit_exact_and_compiler_fallback():
    from sglang_omni.models.qwen3_omni.components import vision_rope

    encoder = (
        vision_compat.Qwen3OmniMoeVisionEncoderCompat(config()).cuda().bfloat16().eval()
    )
    attn = encoder.blocks[0].attn
    x = torch.randn(64, 1152, device="cuda", dtype=torch.bfloat16)
    pos = torch.randn(64, 72, device="cuda")
    lengths = torch.tensor([0, 16, 64], dtype=torch.int32)
    with torch.inference_mode():
        expected = hf.Qwen3OmniMoeVisionAttention.forward(
            attn, x, lengths, (pos.cos(), pos.sin())
        )
        with patch.object(
            vision_rope, "fused_vision_rope", wraps=vision_rope.fused_vision_rope
        ) as fn:
            actual = attn(x, lengths, (pos.cos(), pos.sin()))
            assert fn.call_count == 1
        assert torch.equal(actual, expected)
        with patch.object(
            vision_rope,
            "fused_vision_rope",
            side_effect=subprocess.CalledProcessError(1, "compiler"),
        ):
            fallback = attn(x, lengths, (pos.cos(), pos.sin()))
        assert torch.equal(fallback, expected)
        assert not encoder.blocks[1].attn._omni_rope_state["enabled"]
        assert torch.equal(attn(x, lengths, (pos.cos(), pos.sin())), expected)


@cuda
@pytest.mark.parametrize("mode", ["training", "float32", "disabled"])
def test_gpu_fallback_modes(mode):
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config()).cuda().eval()
    if mode != "float32":
        encoder.bfloat16()
    attn = encoder.blocks[0].attn
    if mode == "training":
        attn.train()
    if mode == "disabled":
        attn._omni_rope_state["enabled"] = False
    x = torch.randn(8, 1152, device="cuda", dtype=attn.qkv.weight.dtype)
    pos = torch.randn(8, 72, device="cuda")
    lengths = torch.tensor([0, 8], dtype=torch.int32)
    with torch.no_grad():
        expected = hf.Qwen3OmniMoeVisionAttention.forward(
            attn, x, lengths, (pos.cos(), pos.sin())
        )
        actual = attn(x, lengths, (pos.cos(), pos.sin()))
    assert torch.equal(actual, expected)


@cuda
def test_missing_triton_falls_back(monkeypatch):
    import sys

    encoder = (
        vision_compat.Qwen3OmniMoeVisionEncoderCompat(config()).cuda().bfloat16().eval()
    )
    attn = encoder.blocks[0].attn
    x = torch.randn(8, 1152, device="cuda", dtype=torch.bfloat16)
    pos = torch.randn(8, 72, device="cuda")
    lengths = torch.tensor([0, 8], dtype=torch.int32)
    with torch.inference_mode():
        expected = hf.Qwen3OmniMoeVisionAttention.forward(
            attn, x, lengths, (pos.cos(), pos.sin())
        )
        monkeypatch.setitem(sys.modules, "triton.compiler.errors", None)
        actual = attn(x, lengths, (pos.cos(), pos.sin()))
    assert torch.equal(actual, expected)
    assert not encoder.blocks[1].attn._omni_rope_state["enabled"]


@cuda
def test_eval_with_gradients_uses_upstream():
    encoder = (
        vision_compat.Qwen3OmniMoeVisionEncoderCompat(config()).cuda().bfloat16().eval()
    )
    attn = encoder.blocks[0].attn
    x = torch.randn(8, 1152, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pos = torch.randn(8, 72, device="cuda")
    lengths = torch.tensor([0, 8], dtype=torch.int32)
    actual = attn(x, lengths, (pos.cos(), pos.sin()))
    expected = hf.Qwen3OmniMoeVisionAttention.forward(
        attn, x, lengths, (pos.cos(), pos.sin())
    )
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
