# Qwen3-Omni FP8 MoE Prefill tuning on RTX PRO 5000

## Change

The TP=1 expert configuration (`E=128,N=768`, FP8 block shape `[128,128]`)
now uses `BLOCK_SIZE_M=64` and `num_stages=3` for 4096–8192 input tokens.
Previously these requests selected the nearest entry, the 32-token config
(`BLOCK_SIZE_M=32`, `num_stages=4`). Other parameters, weights, quantization,
routing, activation, and attention are unchanged.

SGLang selects the nearest numeric key, rather than a floor bucket. Paired
keys 4095/4096 and 8192/8193 make the interval explicit. The 1-token and
32-token configurations remain unchanged; requests outside this interval,
including Decode, retain their previous settings. TP=2's N=384 config is
unchanged. The existing CLI installer installs this asset on the exact named
GPU. Revert the four added JSON entries to restore the previous behavior.

## Measurement (2026-11-17)

GPU 5 on 121.4.54.133, RTX PRO 5000 72GB Blackwell, public
Qwen3-Omni-30B-A3B-FP8, SGLang 0.5.16, PyTorch 2.11+cu130, Triton 3.6.0.
ViT used FA4 in both full-service variants. The tested prompt has 6058 tokens,
including 6042 visual tokens. Requests are sequential, with two output tokens,
image batch size 1, and zero batch wait. Prefix caching was disabled. Each
request used a distinct PNG (one corner pixel changed) to avoid encoder cache
hits. Both variants used the same server and resident model, switching private
MoE config directories and clearing the config cache before Prefill.

Two verification requests and six warmup requests were excluded. The 32 measured
requests used repeated baseline/tuned/tuned/baseline ordering (16 per variant).
HTTP time starts after client image encoding and includes upload/response.
Prefill time is a CUDA-event span around the custom multimodal forward, including
any GPU idle gaps between submissions; it is not a sum of kernel durations.
The isolated timing hook synchronizes the end event in both variants.

| Median | Previous | Tuned | Reduction |
| --- | ---: | ---: | ---: |
| Thinker Prefill | 289.619 ms | 226.580 ms | 21.8% |
| HTTP end to end | 1114.646 ms | 1051.443 ms | 5.7% |

These are single-request results, not a concurrency or throughput claim.
The roughly 63 ms saving comes from MoE; the visual encoder remains the largest
remaining stage. No production service was modified or restarted for this test.

## Numerical verification

- Captured actual inputs, FP8 weights/scales, and routing from layers 0, 23, and
  47 of a 6058-token image request. Compared 18 launch configurations per layer.
- Chosen configuration: whole MoE sequence median about 3.53 → 2.12 ms across
  those layers (about 40%). Timings include input restoration, token alignment,
  quantization, activation, both expert GEMMs, and reduction.
- Complete FA4 service: two distinct image requests × 48 MoE layers, running
  baseline and tuned on identical layer inputs. All 96 results satisfied
  `torch.equal`; maximum absolute difference was zero.
- Replayed the three captured layers at 4096 tokens (sliced) and 8192 tokens
  (repeated then sliced). Both boundaries were bitwise equal in all three
  layers. The 8192-token case is a replay, not an additional full-model request.
- Verified nearest-key selection for every length 1–20000; all lengths outside
  the tuned interval preserve the old configuration.
- Existing kernel-config installation/idempotence tests: 2 passed.

This change preserves observed outputs in these tests. It does not make the
separate FA4 attention change bitwise equivalent to SDPA, and is not a proof
for every possible input, model, or software version. Retune/revalidate after
changing the GPU, model shape, SGLang, Triton, or CUDA.

The benchmark hooks are isolated experiment code, not runtime changes. Raw
results and replay/client scripts are retained in the workspace's
`optimization/moe-prefill/` directory. The repository change is the JSON asset
and its documentation. These measurements predate the November 18 deployment;
see [the integrated bundle](qwen3-verified-optimizations.md).
