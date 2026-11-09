# Qwen3-Omni fused vision RoPE

The vision compatibility encoder uses a fused Triton RoPE kernel on NVIDIA SM120,
with BF16 hidden states, head dimension 72, SDPA attention, evaluation mode and
no autograd. Other devices, precisions, head dimensions, attention backends and
training/gradient-enabled calls retain the Transformers implementation. Model
weights, checkpoint keys, image resolution and attention backend are unchanged.
The binding is per encoder instance; Transformers module globals are not patched.

The kernel combines Q/K conversion, half rotation, multiplication, addition and
BF16 output conversion. Intermediate operations retain FP32 precision. Explicit
`mul.rn.f32` and `add.rn.f32` instructions preserve separate rounding even for FP32
position tensors; disabling FMA alone was insufficient on some test inputs.

Set `SGLANG_OMNI_FUSED_VISION_ROPE=0` **before constructing the encoder** to disable
fusion. The default is enabled for the supported path. Missing Triton or a caught
Triton compilation / compiler subprocess error logs a warning and disables fusion
for that encoder, falling back to HF. Unexpected CUDA execution errors and OOMs
are not swallowed. The first eligible call compiles the kernel; warm it before
latency measurement. CUDA device/stream selection is preserved.

The runtime needs a working Triton JIT environment, including a C compiler and
matching Python development headers. For example, a Python 3.12 environment
without `Python.h` cannot build Triton's launcher. This change does not install
system packages. The benchmark used isolated headers via `CPATH`; serving
images should supply the matching development headers or a working prebuilt
launcher cache. Check logs for fallback warnings before claiming an enabled
optimization.

## Validation

Tested with PyTorch 2.11.0+cu130, Transformers 5.12.1, RTX PRO 5000 72GB (SM120),
public Qwen3-Omni-30B-A3B-FP8 visual weights evaluated in BF16. The attention
interface mirrors the pinned Transformers 5.12.1 SDPA path; review it when
upgrading Transformers.

Tests cover strided Q/K tensors, odd sequence lengths, BF16/FP32 position tensors,
full attention output equality, strict state-dict loading, model isolation,
CPU/GPU autograd fallback, disabled mode, unsupported precision, missing Triton,
and compilation failure. Run:

```sh
pytest tests/unit_test/qwen3_omni/test_vision_compat.py \
       tests/unit_test/qwen3_omni/test_vision_rope.py
```

SM120 tests skip on other hardware. Four complete-encoder cases (original
cars.jpg, small image, video, mixed image sizes) were also compared using real
weights. Last hidden state, pooled visual output and all three DeepStack outputs
were bitwise equal. This is evidence for the tested inputs/environment, not a
claim of identical outputs on every future hardware/compiler version.

The serial warmed benchmark alternated original/fused runs, with 4 warmups and
12 timed runs per variant, and CUDA synchronization around wall-clock timing.
It measures the complete ViT, not HTTP or downstream Thinker latency.

| Input | Original ms | Fused ms | Reduction |
|---|---:|---:|---:|
| cars | 682.13 | 611.86 | 10.3% |
| small | 10.74 | 8.99 | 16.3% |
| video | 17.49 | 14.75 | 15.7% |
| mixed | 17.61 | 14.87 | 15.6% |
