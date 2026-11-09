# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from sglang_omni.models.qwen3_omni.components.image_preprocess import (
    optimize_image_processor,
)


def processors():
    kwargs = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 1024,
        "max_pixels": 65536,
    }
    base, fast = Qwen2VLImageProcessor(**kwargs), Qwen2VLImageProcessor(**kwargs)
    assert optimize_image_processor(fast)
    return base, fast


@pytest.mark.parametrize(
    "kind",
    ["pil", "numpy", "tensor", "float", "grayscale", "rgba", "mixed", "same_size"],
)
@pytest.mark.parametrize("grouping", [None, False, True])
def test_exact_inputs_and_order(kind, grouping):
    base, fast = processors()
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, (127, 213, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    value = {
        "pil": image,
        "numpy": array,
        "tensor": torch.from_numpy(array).permute(2, 0, 1),
        "float": array.astype(np.float32),
        "grayscale": image.convert("L"),
        "rgba": image.convert("RGBA"),
        "mixed": [image, image.resize((93, 131)), image],
        "same_size": [image, image],
    }[kind]
    before = array.copy()
    expected = base(images=value, return_tensors="pt", disable_grouping=grouping)
    actual = fast(images=value, return_tensors="pt", disable_grouping=grouping)
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    actual["pixel_values"].zero_()
    assert np.array_equal(array, before)
    # No shared output scratch buffer may corrupt an earlier request.
    again = fast(images=value, return_tensors="pt", disable_grouping=grouping)
    assert torch.equal(again["pixel_values"], expected["pixel_values"])
    assert torch.count_nonzero(actual["pixel_values"]) == 0


@pytest.mark.parametrize(
    "options",
    [
        {"do_resize": False},
        {"do_rescale": False},
        {"do_normalize": False},
        {
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.3, 0.4, 0.5],
            "rescale_factor": 0.0123,
        },
        {"temporal_patch_size": 1},
        {"return_tensors": "np"},
        {"return_tensors": None},
    ],
)
def test_options_and_fallback(options):
    base, fast = processors()
    value = torch.arange(3 * 64 * 96).reshape(3, 64, 96).to(torch.uint8)
    options = {"return_tensors": "pt", **options}
    expected, actual = base(images=value, **options), fast(images=value, **options)
    for key in expected:
        assert np.array_equal(np.asarray(actual[key]), np.asarray(expected[key]))


def test_config_and_opt_out(monkeypatch):
    p = Qwen2VLImageProcessor()
    config = p.to_dict()
    monkeypatch.setenv("SGLANG_OMNI_FAST_IMAGE_PREPROCESS", "0")
    assert not optimize_image_processor(p)
    monkeypatch.delenv("SGLANG_OMNI_FAST_IMAGE_PREPROCESS")
    assert optimize_image_processor(p)
    assert (
        p.rescale_and_normalize.__func__ is Qwen2VLImageProcessor.rescale_and_normalize
    )
    assert p.to_dict() == config
    assert Qwen2VLImageProcessor.from_dict(p.to_dict()).to_dict() == config
    assert "_preprocess" not in p.to_json_string()
    assert not optimize_image_processor(None)
