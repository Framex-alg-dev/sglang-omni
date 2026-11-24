# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang_omni.models.qwen3_omni.components.image_encoder import (
    _prepare_visual_pixels,
)


def test_cpu_path_preserves_conversion_and_identity():
    x = torch.randn(17, 32)
    assert _prepare_visual_pixels(x, "cpu", torch.float32) is x
    assert torch.equal(_prepare_visual_pixels(x, "cpu", torch.bfloat16), x.bfloat16())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_input_transfer_is_exact_on_consumer_stream(
    monkeypatch, pinned, enabled, strided
):
    monkeypatch.setenv("SGLANG_OMNI_ASYNC_VISION_H2D", "1" if enabled else "0")
    torch.manual_seed(23)
    x = torch.randn(257, 144, pin_memory=pinned)
    if strided:
        x = x[:, ::2]
    original = x.clone()
    reference = x.to("cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = _prepare_visual_pixels(x, "cuda", torch.bfloat16)
        # Consumer runs without an intermediate CPU synchronize.
        consumed = actual * 2
    stream.synchronize()
    assert torch.equal(actual, reference)
    assert torch.equal(consumed, reference * 2)
    assert torch.equal(x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_and_other_dtypes_fall_back():
    x = torch.randn(17, 72, device="cuda", dtype=torch.bfloat16)
    assert _prepare_visual_pixels(x, "cuda", torch.bfloat16) is x
    cpu = torch.randn(17, 72, dtype=torch.float64)
    assert torch.equal(
        _prepare_visual_pixels(cpu, "cuda", torch.float32), cpu.cuda().float()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bfloat16_rounding_boundaries(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_ASYNC_VISION_H2D", "1")
    # Sweep every BF16 exponent/mantissa/sign around the discarded FP32 bits'
    # midpoint. Include subnormals, signed zero and large finite values.
    high = torch.arange(65536, dtype=torch.int64) << 16
    bits = torch.stack([high | low for low in (0, 0x7FFF, 0x8000, 0x8001, 0xFFFF)])
    x = bits.to(torch.int32).view(torch.float32).flatten()
    x = x[torch.isfinite(x)]
    expected = x.to("cuda", dtype=torch.bfloat16)
    actual = _prepare_visual_pixels(x, "cuda", torch.bfloat16)
    torch.cuda.synchronize()
    # Compare storage bits as well, including the sign of zero.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
