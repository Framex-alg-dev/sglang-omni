# Bit-exact CPU resize layout optimization

The single-image CPU fast path now supplies large uint8 RGB images to the
existing torchvision bicubic resize operation in channels-last memory layout.
This changes storage, not logical image layout, interpolation coefficients,
antialiasing, output resolution or normalization. No GPU resize or alternative
image library is introduced.

Enable conditions: CPU backend reports AVX512, dtype uint8, batch shape NCHW
with three channels, at least 1,048,576 elements, requested resampling is PIL
BICUBIC, and the target size differs from the input size. Other cases retain
the previous resize path. The existing fast-path restrictions (single CPU
image with normalization enabled and exact HF processor class) still apply.

`SGLANG_OMNI_CHANNELS_LAST_RESIZE=0` disables just the layout change. The existing
`SGLANG_OMNI_FAST_IMAGE_PREPROCESS=0` disables the whole CPU preprocessing fast
path. Inputs are not modified, no shared buffer/cache is added, and no global
thread-count or model settings are changed. Revalidate on future torch or
CPU backend upgrades before assuming bitwise equivalence.

## Validation, 2026-11-09

PyTorch 2.11.0+cu130, Transformers 5.12.1, AVX512 CPU on 121.4.54.133. Tests
compare exact uint8 pixels for random, binary, checkerboard and border patterns,
including upsampling, severe downsampling, almost-identical dimensions and very
wide images (20 combinations). The existing full preprocessing tests cover
normalization, patch order, mixed-image fallback, storage ownership and config
serialization. Additional tests check the disable switch, small/float/bilinear/
other-CPU fallback, layout selection and no-op resize. Total: 63 passing tests.

```sh
pytest tests/unit_test/qwen3_omni/test_image_preprocess.py \
       tests/unit_test/qwen3_omni/test_resize_layout.py
```

The original cars.jpg was evaluated without changing resolution. Full Qwen
processor input IDs, attention mask, image grid and pixel values are bitwise
identical. All five complete ViT outputs are also bitwise identical with real
public Qwen3-Omni visual weights in BF16. Attention remains SDPA, with the prior
RoPE, patch packing and async transfer optimizations enabled on both sides.

| Measurement | Before | After |
|---|---:|---:|
| Bicubic resize, 4 CPU threads, including layout conversion | 35.08 ms | 16.19 ms |
| CPU preprocessing, separate 4-thread run | 79.83 ms | 60.65 ms |
| CPU preprocessing, separate 8-thread run | 73.43 ms | 51.24 ms |
| CPU preprocessing in final 8-thread run | 74.91 ms | 53.53 ms |
| Preprocess + transfer + ViT, final 8-thread run | 684.97 ms | 665.29 ms |

The final combined measurement saves 19.68 ms (2.9%) beyond the previous local
optimization. The 8-thread setting matches Qwen3-Omni's default OMP setting; it
is set only inside the isolated benchmark, not in production code. GPU 5 was
used for the combined test. Original/optimized calls alternate order, with four
warmups and 16 timed forwards per variant; CUDA completion is awaited before
stopping the timer. CPU-only tests use four warmups and twenty measured calls.

Absolute timings vary with CPU/GPU state, so compare within each paired run.
Small-image combined time (about 7 ms) showed no meaningful benefit and the
layout fast path is not enabled for those inputs. These are not full-service
HTTP/queue/Thinker latency measurements. No production service was changed.
