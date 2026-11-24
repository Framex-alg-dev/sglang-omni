# Further lossless patch packing and input transfer optimization

This change builds on b561e5f (fused RoPE and reduced CPU preprocessing copies).
It does not change image resolution, interpolation, model weights, attention
backend, token ordering or numerical precision.

## Pack large uint8 images before normalization

For a single normalized CPU uint8 image with at least 1,048,576 elements after
resize, pack spatial patches and duplicate the temporal dimension before FP32
conversion. The large strided packing operation moves one byte per value rather
than four. Reshape the packed samples to `(patches, channels, temporal*patch,
patch)` before normalization so the original per-channel coefficients and
subtract/divide order are preserved. Small images retain normalize-then-pack;
multiple images, float/GPU inputs and unsupported processor classes retain their
existing paths. Input storage remains unmodified and no shared scratch buffers
are introduced. Existing `SGLANG_OMNI_FAST_IMAGE_PREPROCESS=0` disables the CPU
fast path when the processor is constructed.

## Transfer then convert on GPU

`_prepare_visual_pixels` uses nonblocking CPU-FP32 to CUDA-BF16 transfer for
pageable inputs. In the tested PyTorch build, this transfers FP32 and performs
conversion on GPU rather than doing the conversion on CPU before transfer.
Consumers execute on the same CUDA stream and do not need an extra host wait.
This does not increase request concurrency. Pinned input buffers, CPU targets,
GPU inputs and other dtypes retain the previous copy path. Pinned inputs are
excluded to preserve the caller's previous buffer-reuse behavior.

Set `SGLANG_OMNI_ASYNC_VISION_H2D=0` to disable the new transfer path. CUDA stream
ordering is respected, but this path trades memory for latency: for cars.jpg the
measured peak allocated memory during transfer increased from 74,244,096 bytes
to 222,732,288 bytes (about 141.6 MiB extra). Disable it when that headroom is not
available. The FP32 staging tensor is temporary; PyTorch may retain freed allocator memory
for reuse. No persistent input-data cache is introduced.

## Validation (2026-11-09)

Host 121.4.54.133, GPU 5 RTX PRO 5000 72GB, PyTorch 2.11.0+cu130,
Transformers 5.12.1, four CPU threads. Model is the public Qwen3-Omni visual
checkpoint in BF16, with the existing fused RoPE and SDPA on both sides.

47 regression tests pass (image preprocessing plus input transfer). These cover
large per-channel patterns, grouping, temporal patch sizes, input ownership,
CPU/GPU/pinned/strided fallbacks, nondefault consumer streams, disable switch,
and FP32 values around every BF16 rounding boundary, including finite subnormals,
signed zeros and large finite values. Bitwise comparisons use integer views for
the conversion-boundary test. Full Qwen processor pixel/grid/input IDs/attention
mask equality also passes.

```sh
pytest tests/unit_test/qwen3_omni/test_image_preprocess.py \
       tests/unit_test/qwen3_omni/test_vision_input_transfer.py
```

Serial, alternating before/after tests, warmed; transfer and combined timings
include `torch.cuda.synchronize()` so early API return is not counted as a gain:

| Measurement | Before | After |
|---|---:|---:|
| Large-image normalization + packing microbenchmark | 36.41 ms | 31.70 ms |
| Large-image CPU preprocessing, full processor image path | 70.35 ms | 65.85 ms |
| Input transfer + dtype conversion, standalone | 15.04 ms | 6.24 ms |
| Large-image preprocess + transfer + complete ViT | 681.05 ms | 672.90 ms |

The combined measurement saves 8.15 ms (1.2%) relative to the already optimized
b561e5f baseline. Component microbenchmarks do not add linearly to the combined
measurement. Absolute timings differ from earlier sessions due to hardware and
CPU state; comparisons above are paired within each run. Small-image combined
benefit was negligible; the new pack order is restricted to large images.

Preprocessed tensors and all five ViT outputs were bitwise equal for the real
cars.jpg and a smaller image. These are validation results for the tested inputs
and environment, not a claim of every future compiler/hardware configuration.
This benchmark excludes HTTP handling, queueing, text tokenization and Thinker
inference; it is not a complete Omni-service latency measurement.
