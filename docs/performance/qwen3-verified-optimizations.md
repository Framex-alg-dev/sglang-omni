# Verified Qwen3-Omni optimization bundle

This bundle integrates the source snapshot deployed and tested on 2026-11-18/19,
relative to `b561e5f2c1665d227731446a2e50cede59470f74`:

- uint8 patch packing, AVX512 channels-last resize and image Base64 decoding;
- asynchronous pageable vision input transfer, GPU dtype conversion and host-grid reuse;
- single-segment vision copy avoidance and the supported FA4 vision path;
- RTX PRO 5000 FP8 MoE prefill configurations for 4096–8192 tokens;
- configurable image encoder batch limit (default remains 32).

The deployment's post-Thinker CPU thread hook is now available as
`create_decode_executor(..., cpu_num_threads=8)`. In the pipeline configuration,
add `cpu_num_threads: 8` to the decode stage's `factory_args`. That stage must be
constructed **after Thinker in the same process**, as in the validated colocated
pipeline. SGLang initialization otherwise resets PyTorch's intra-op pool to one.
This is a process-wide setting, including colocated preprocessing; it is not eight
request workers. Omitting the argument preserves the existing thread setting.
The appropriate thread count depends on available CPU capacity and concurrency.

FA4 requires the dependencies described in [the FA4 guide](qwen3-vision-fa4.md).
A source merge alone does not upgrade installed kernels or apply deployment
configuration. Use `SGLANG_OMNI_VISION_ATTENTION_BACKEND=sdpa` to select SDPA.

## Validation boundaries

The historical controlled same-GPU, CPU-one-thread comparison measured fresh
request medians of 772→673 ms (event), 225→225 ms (reply), and 624→536 ms
(action). Cached requests saw little improvement from the code changes alone.
A separate CPU 1→8 comparison measured cached medians of 383→201 ms, 149→135 ms,
and 83→85 ms respectively. These are sample-specific results, not latency SLAs.

FA4 is **not bitwise equivalent** to SDPA; the large-image vision experiment
measured approximately 1.94% relative L2 feature difference. Identical outputs in
240 serial controlled requests do not establish general business accuracy
parity. Concurrent/stochastic outputs can differ. Smaller images can be slower.

The experimental CPU media-preprocessing cache, deferred video RGB conversion,
and audio/video Base64 expansion were not deployed and are excluded. Model
weights, prompts, token limits and business routing are unchanged by this bundle.

## Merge verification (2026-11-24)

Against the deployed source snapshot, the integration adds the opt-in CPU-thread
factory argument and documentation. On macOS CPU with PyTorch 2.11.0,
torchvision 0.26.0 and Transformers 5.12.1, the focused preprocessing, Base64,
vision fallback/selection, transfer, host-grid, CPU-thread and kernel-configuration
checks passed: **89 passed, 31 CUDA-dependent tests skipped**. The SGLang-dependent
image batch configuration tests could not be collected locally because SGLang
was unavailable. PRO006 SSH was unavailable, so no new GPU/service benchmark was
run during this merge. Historical GPU results above are not a fresh deployment
verification.
