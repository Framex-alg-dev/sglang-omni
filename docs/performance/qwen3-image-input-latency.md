# Image input latency optimizations (2026-11-17)

## Changes

Image Base64 decoding now uses the project's existing `pybase64` dependency with
strict validation. If validation fails (including whitespace, non-ASCII input or
nonstandard padding), decoding falls back to Python's original `base64.b64decode`.
This preserves permissive/error behavior rather than trusting the accelerated
library's different handling of malformed inputs. An unavailable optional import
also falls back. Decoded compressed bytes, image decoding, pixels and model
computation are unchanged.

`create_image_encoder_executor` now accepts a validated `max_batch_size` factory
argument. Its default remains 32, with the existing 50ms default batching window.
A deployment can explicitly choose serial visual encoding:

```yaml
runtime_overrides:
  image_encoder:
    max_batch_size: 1
    max_batch_wait_ms: 0
```

With maximum batch size 1 the scheduler immediately returns the first request,
without consuming queued requests or waiting to fill a batch. This is a latency
policy, not a universal throughput improvement. The repository's existing single
GPU deployment templates already set wait=0; do not claim a new 50ms saving for
those deployments. The default profiles were not changed.

## Measurement

Single RTX PRO 5000 72GB, GPU 5 on 121.4.54.133. Public Qwen3-Omni FP8, prior local
optimizations including host grid reuse. Same 1830x3394 image workload as the
previous tests, PNG with one distinct corner pixel per request to avoid media
cache hits. Serial requests, 32 output tokens, 16 measured requests per variant;
all actual encoder batches were 1. First request warmup excluded. Baseline and
candidates were separate sequential server runs, not a randomized repeated A/B.

| Configuration | P50 total | P95 total | P50 TTFT |
|---|---:|---:|---:|
| Default 50ms batching window | 1437.74ms | 1442.91ms | 1233.70ms |
| Serial encoding, zero wait | 1393.33ms | 1402.41ms | 1190.76ms |
| Zero wait + SIMD Base64 | 1371.95ms | 1377.83ms | 1171.78ms |

The already-zero-wait comparison shows a 21.38ms / 1.53% P50 reduction. This is a
small-sample observation, not a guaranteed gain or proof that all 21ms comes from
Base64. Direct interleaved Base64-only measurement (50 samples after warmup):

| Compressed bytes | stdlib | accelerated strict decoder |
|---|---:|---:|
| JPEG, 406017 bytes | 0.499ms | 0.0196ms |
| PNG, 3101366 bytes | 3.986ms | 0.146ms |

The direct measured decoding saving is about 0.48ms for this JPEG and 3.84ms for
this PNG. Larger end-to-end differences may include run-to-run variation. This
work does not reduce ViT arithmetic or resolve concurrent Thinker decode stalls.

## Validation and limits

14 unit tests pass: configurable limits/default compatibility, invalid limits
rejected before loading weights, no waiting/consuming another message at limit=1,
valid payloads and whitespace, missing-library fallback, and 1000 deterministic
random malformed-input comparisons against stdlib including exception messages.
The permissive fast decoder initially failed differential tests; only the strict
fast path with stdlib fallback is retained. Its interrupted trial is archived as
`invalid-simd-*` and excluded from results.

All 16 corresponding returned texts (32 generated tokens per request) matched
across all three configurations. This is not a logits-level or broad task-accuracy
evaluation. Base64 byte equality is the direct correctness property for the new
fast path. Concurrency/batching policy changes still need workload-specific
validation; no batch=4 precision claim is made.

Artifacts: workspace `optimization/no-batch-wait/artifacts`, including configs,
server logs, request events, raw timings, comparisons and tests.log. Dedicated
test server stopped, port 18486 released, GPU 5 returned to 0MiB. Task SSH master
and temporary authentication helper cleaned up. At measurement time changes were local; no push or
production deployment.
