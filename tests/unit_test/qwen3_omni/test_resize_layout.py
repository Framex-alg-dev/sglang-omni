# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import SizeDict

from sglang_omni.models.qwen3_omni.components.image_preprocess import _resize_image


@pytest.mark.parametrize("pattern", ["random", "binary", "checker", "border"])
@pytest.mark.parametrize(
    "src,dst",
    [
        ((513, 769), (512, 768)),
        ((512, 768), (1024, 1536)),
        ((513, 769), (127, 213)),
        ((1024, 1024), (32, 32)),
        ((33, 12001), (64, 4096)),
    ],
)
def test_layout_resize_bit_exact(pattern, src, dst):
    torch.manual_seed(42)
    x = torch.randint(0, 256, (1, 3, *src), dtype=torch.uint8)
    if pattern == "binary":
        x = (x > 127).to(torch.uint8) * 255
    elif pattern == "checker":
        x.zero_()
        x[:, :, ::2, ::2] = 255
    elif pattern == "border":
        x[:, :, 0, :] = 255
        x[:, :, -1, :] = 0
    original = x.clone()
    p = Qwen2VLImageProcessor()
    expected = p.resize(
        image=x, size=SizeDict(height=dst[0], width=dst[1]), resample=p.resample
    )
    actual = _resize_image(p, x, *dst, p.resample)
    assert torch.equal(actual, expected)
    assert torch.equal(x, original)


@pytest.mark.parametrize(
    "scenario",
    ["enabled", "disabled", "small", "float", "bilinear", "other_cpu", "same_size"],
)
def test_layout_selection_and_fallback(monkeypatch, scenario):
    monkeypatch.setenv(
        "SGLANG_OMNI_CHANNELS_LAST_RESIZE", "0" if scenario == "disabled" else "1"
    )
    monkeypatch.setattr(
        torch.backends.cpu,
        "get_cpu_capability",
        lambda: "DEFAULT" if scenario == "other_cpu" else "AVX512",
    )
    p = Qwen2VLImageProcessor()
    x = torch.randint(0, 256, (1, 3, 512, 768), dtype=torch.uint8)
    if scenario == "small":
        x = x[:, :, :32, :32].contiguous()
    if scenario == "float":
        x = x.float()
    mode = Image.Resampling.BILINEAR if scenario == "bilinear" else p.resample
    with patch.object(p, "resize", wraps=p.resize) as resize:
        target = x.shape[-2:] if scenario == "same_size" else (256, 384)
        _resize_image(p, x, *target, mode)
        transferred = resize.call_args.kwargs["image"]
    if scenario == "enabled":
        assert transferred.is_contiguous(memory_format=torch.channels_last)
        assert torch.equal(transferred, x)
    else:
        assert transferred is x
