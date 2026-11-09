# SPDX-License-Identifier: Apache-2.0
"""Avoid redundant CPU copies in the pinned HF Qwen2-VL image processor."""

import os
import types

import torch
from torchvision.transforms.v2 import functional as tvf
from transformers.models.qwen2_vl import image_processing_qwen2_vl as hf


def _rescale_and_normalize(
    self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
):
    if images.device.type != "cpu" or images.dtype != torch.uint8 or not do_normalize:
        return hf.Qwen2VLImageProcessor.rescale_and_normalize(
            self,
            images,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
        )
    mean, std, _ = self._fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    # uint8 -> float32 always allocates private storage. The same torchvision
    # subtract/divide operations may safely reuse it without mutating the input.
    return tvf.normalize(images.to(dtype=torch.float32), mean, std, inplace=True)


def _preprocess(
    self,
    images,
    do_resize,
    size,
    resample,
    do_rescale,
    rescale_factor,
    do_normalize,
    image_mean,
    image_std,
    patch_size,
    temporal_patch_size,
    merge_size,
    disable_grouping,
    return_tensors,
    **kwargs,
):
    # Only optimize the common single-image CPU uint8 path. Preserve upstream behavior
    # (including gradients and storage semantics) for float/GPU/unusual inputs.
    if (
        len(images) != 1
        or not do_normalize
        or any(
            image.device.type != "cpu" or image.dtype != torch.uint8 for image in images
        )
    ):
        return hf.Qwen2VLImageProcessor._preprocess(
            self,
            images,
            do_resize,
            size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            patch_size,
            temporal_patch_size,
            merge_size,
            disable_grouping,
            return_tensors,
            **kwargs,
        )
    grouped, indices = hf.group_images_by_shape(
        images, disable_grouping=disable_grouping
    )
    resized = {}
    for shape, batch in grouped.items():
        height, width = batch.shape[-2:]
        if do_resize:
            height, width = hf.smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size.shortest_edge,
                max_pixels=size.longest_edge,
            )
            batch = self.resize(
                image=batch,
                size=hf.SizeDict(height=height, width=width),
                resample=resample,
            )
        resized[shape] = batch
    images = hf.reorder_images(resized, indices)
    grouped, indices = hf.group_images_by_shape(
        images, disable_grouping=disable_grouping
    )
    processed, grids = {}, {}
    for shape, batch in grouped.items():
        height, width = batch.shape[-2:]
        patches = _rescale_and_normalize(
            self, batch, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        batch_size, channels = patches.shape[:2]
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = patches.reshape(
            batch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        processed[shape] = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
        )
        grids[shape] = [[1, grid_h, grid_w]] * batch_size
    outputs = hf.reorder_images(processed, indices)
    # The normalized/packed tensor already owns private storage. torch.cat on
    # one image otherwise copies the entire (often ~148 MB) output again.
    pixels = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    return hf.BatchFeature(
        data={
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor(
                hf.reorder_images(grids, indices), dtype=torch.long
            ),
        },
        tensor_type=return_tensors,
    )


def _to_dict(self):
    result = hf.Qwen2VLImageProcessor.to_dict(self)
    # HF serializes instance attributes. Per-instance method bindings are runtime
    # details, not checkpoint configuration, and must not enter saved processors.
    for name in ("_preprocess", "rescale_and_normalize", "to_dict"):
        result.pop(name, None)
    return result


def optimize_image_processor(processor) -> bool:
    """Bind to this processor only; preserve its config and serialization type."""
    if (
        os.environ.get("SGLANG_OMNI_FAST_IMAGE_PREPROCESS", "1") == "0"
        or type(processor) is not hf.Qwen2VLImageProcessor
        or any(
            name in processor.__dict__
            for name in ("_preprocess", "rescale_and_normalize", "to_dict")
        )
    ):
        return False
    processor.to_dict = types.MethodType(_to_dict, processor)
    processor._preprocess = types.MethodType(_preprocess, processor)
    return True
