# Single-segment vision attention copy removal (2026-11-17)

The optimized SDPA vision path concatenated a list even when it contained only
one attention result. Transformers SDPA already returns contiguous B,S,H,D
output. Use that result directly for a single segment, retaining the final
reshape/contiguous call. Multi-segment image/video inputs still concatenate.
Compute CPU split sizes once rather than three times per attention layer.
The attention implementation, RoPE arithmetic, GEMMs, weights and input pixels
are unchanged. Existing training/dtype/device/backend fallbacks remain intact.

RTX PRO 5000 72GB, public Qwen3-Omni FP8 checkpoint with BF16 vision, SDPA,
current local optimizations. One model, before/after attention methods alternated
in the same process; four warmup pairs and sixteen measured pairs per shape.
CUDA synchronized before and after each timed forward. Inputs preloaded on GPU.

| Input | Before ViT P50 | After ViT P50 | Change |
|---|---:|---:|---:|
| Original cars.jpg, grid 1x114x212 | 610.986ms | 609.882ms | -1.104ms / -0.18% |
| Small image, 448x256 | 6.245ms | 5.878ms | -0.367ms / -5.88% |
| Two small images | 9.207ms | 9.209ms | essentially unchanged |
| Two-frame small video grid | 9.122ms | 9.119ms | essentially unchanged |

All five visual outputs (last hidden state, pooled output, three DeepStack
features) are bitwise equal for all four cases. The compatibility suite passed
8 tests and the updated RoPE/attention suite passed 17 tests, including explicit
single- and multi-segment comparisons against upstream and compiler fallback.
This is validation of tested inputs, not a universal accuracy guarantee.

An independent nsys capture with three A/B pairs confirms 27 fewer device-to-
device copies per large image, eliminating 1,503,442,944 copied bytes. Total
memcpy duration drops from about 1.454ms to 0.102ms. Kernel count remains 340:
the removed operations are CUDA copies, not compute kernels. Under profiler the
compute timing varies enough to obscure the small end-to-end benefit, so use
the unprofiled paired results for performance reporting.

A separate full-service trace was collected after warmup, using isolated
instrumentation that labels encoder and Thinker execution with NVTX. GPU activity
is attributed through CUDA API launch correlation, including memcpy and CUDA
graph events; asynchronous CPU range durations are not treated as GPU time.
Three distinct-image requests, 6058 prompt tokens, two output tokens:

| Stage | Mean GPU activity span | Mean GPU busy time |
|---|---:|---:|
| Vision encoder including H2D/conversion | 609.56ms | 606.98ms |
| Thinker Prefill | 291.39ms | 288.44ms |
| Second-token Decode | 6.85ms | 5.52ms |

These are profiled stage activity spans, not HTTP latency or sum-of-exclusive
CPU buckets. The Thinker CPU execution range also waits for queued ViT work;
its CPU duration must not be mistaken for Thinker GPU compute. Remaining GPU
activity gaps are only about 2.6ms in vision and 2.9ms in prefill for this test.
No large HTTP speedup is claimed from this approximately 1ms copy optimization.

Artifacts in workspace `optimization/vision-copy/artifacts`:
- results.json: unprofiled paired timings and equality results.
- vit-copy.nsys-rep / .sqlite / .analysis.json: before/after vision trace.
- service/full-chain.nsys-rep / .sqlite / .analysis.json: complete service trace.
- tests.log and tests_rope.log: regression results.

Only isolated test processes were launched on GPU 5; tests used free GPU 6.
The server process group stopped, port 18488 released, GPU 5/6 returned to 0MiB,
and the dedicated SSH master and authentication helper were removed. Changes
were subsequently included in the November 18 deployment; see [the integrated bundle](qwen3-verified-optimizations.md).
