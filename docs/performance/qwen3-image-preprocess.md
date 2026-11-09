# Lossless single-image CPU preprocessing

Qwen3OmniPreprocessor enables a per-instance optimization for the pinned
Transformers 5.12.1 Qwen2VLImageProcessor. It only takes the fast path for one
CPU uint8 image with normalization enabled. Multiple images, float/GPU inputs,
other processor classes and disabled normalization use the original HF path.
The video processor is unchanged.

Two copies are removed:

- Normalize in place on the newly allocated uint8-to-FP32 conversion buffer.
  The original caller's image is never modified. The same HF coefficient fusion
  and torchvision subtraction/division operations are used.
- Return the already materialized single-image patch tensor instead of copying
  the entire output through a one-input `torch.cat`. For cars.jpg this removes
  another approximately 148 MB output copy.

Resize, antialiasing, interpolation, normalization coefficients, output dtype,
patch order, image grid and image token count retain their original definitions.
No shared scratch buffer or image-content cache is introduced. Runtime method
bindings are excluded from serialization so the processor config and HF class
name are preserved. Processors with existing custom method overrides are skipped.

Set `SGLANG_OMNI_FAST_IMAGE_PREPROCESS=0` before constructing the preprocessor to
disable this optimization. No new runtime compiler dependency is required.
Review the local `_preprocess` compatibility implementation when upgrading the
pinned Transformers version.

## Verification (2026-11-09)

Host 121.4.54.133, PyTorch 2.11.0+cu130, Transformers 5.12.1, four CPU threads.
The original cars.jpg is opened before timing. Timings include image-processor
conversion, resize, normalize and patch preparation, but exclude disk decoding,
text tokenization, H2D, ViT and downstream generation. Original/optimized calls
alternate order, four warmups and twenty measured calls per variant. CPU load
causes absolute timings to vary between runs; compare paired runs.

| Case | Original median | Optimized median | Reduction |
|---|---:|---:|---:|
| cars.jpg | 113.84 ms | 84.27 ms | 26.0% |
| Small image | 0.478 ms | 0.453 ms | 5.4% |
| Mixed images (original path) | 1.236 ms | 1.253 ms | No demonstrated benefit |

32 tests pass, covering PIL/NumPy/Tensor, float fallback, grayscale/RGBA,
grouping settings, multiple-image ordering, resize/rescale/normalize options,
custom mean/std, temporal patch size, numpy/torch output, input/output ownership,
serialization and the disable switch. Full Qwen3OmniMoeProcessor comparison on
cars.jpg also verifies bitwise equality of pixel_values, image_grid_thw,
input_ids and attention_mask.

```sh
pytest tests/unit_test/qwen3_omni/test_image_preprocess.py
```

Profiling confirms the full-size final output concatenation and allocating
subtraction are removed. Resize and patch materialization remain the main CPU
costs. These numbers are not end-to-end request speedups and should not be added
as percentages to the separate RoPE optimization.
