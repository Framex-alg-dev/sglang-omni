# Reuse CPU visual grid metadata (2026-11-17)

`build_thinker_inputs` previously preferred the image encoder's CUDA grid. The
request builder later calls `.cpu()` to compute M-RoPE, synchronizing with queued
vision work. The original preprocessing grid is still available and has identical
values and row order. Reuse that CPU tensor when the encoder grid is CUDA and its
shape matches. Missing CPU metadata, mismatched shapes, and CPU encoder outputs
retain the existing fallback. Image/video embeddings and DeepStack features are
not copied or recomputed.

The current encoder returns the input grid unchanged, including batch splitting.
This invariant is required by the optimization. The normal colocated path uses
ordered CUDA work on the default stream; CUDA IPC has its own readiness-event
handling. This change does not replace producer/consumer stream synchronization.

Validation on RTX PRO 5000 72GB, public Qwen3-Omni FP8:

- 7 tests pass: fallback handling; image/video, one/four grid rows; bitwise equal
  M-RoPE positions and deltas for one/four images. Feature objects are unchanged.
- Three serial image requests, 32 output tokens each, return identical text before
  and after. The chat endpoint ignores the requested OpenAI logprob fields; the
  rollout endpoint does not accept this raw-image input shape. No successful
  full-model logits/logprob comparison was obtained. This is not a comprehensive
  task-accuracy evaluation or a claim that batch=4 matches batch=1.
- Same 300ms encoder batching window and request fixture as the preceding test,
  eight groups per concurrency, two output tokens. Encoder batch sizes remain
  1/2/2 for request concurrency 1/2/4.

| Request concurrency | Before P50 | After P50 | Observed reduction |
|---|---:|---:|---:|
| 1 | 1523.30ms | 1508.91ms | 0.94% |
| 2 | 2575.86ms | 2554.43ms | 0.83% |
| 4 | 4422.66ms | 4390.75ms | 0.72% |

These small differences are sequential-run observations, not statistically proven
speedups. Single-request builder wall time changed from 587.76ms to 1.07ms;
concurrency-4 maximum builder time changed from 1218.32ms to 1.78ms. Most of that
former time was waiting for vision kernels, which still have to run. Do not claim
a 587ms end-to-end saving. Decode scheduling and ViT compute remain bottlenecks.

Raw artifacts: workspace `optimization/host-grid/artifacts`, compared with
`optimization/batch-size/artifacts/window300-*`. The isolated test server was
stopped; no production deployment or push was performed.
