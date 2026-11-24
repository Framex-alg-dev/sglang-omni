# Default FA4 for the supported Qwen3 vision inference path

The user explicitly requested switching ViT Attention to FA4 after reviewing its
measured numerical differences. On the tested CUDA SM120, BF16, head-dimension-72,
eval/no-grad SDPA-compatible path, Qwen3OmniMoeVisionEncoderCompat now selects FA4
by default. This changes vision attention only, not Thinker attention. Existing
fused RoPE and single-segment copy avoidance remain enabled.

Set before constructing the encoder:

```sh
export SGLANG_OMNI_VISION_ATTENTION_BACKEND=fa4  # default
# Roll back to the prior attention backend:
export SGLANG_OMNI_VISION_ATTENTION_BACKEND=sdpa
```

Invalid values raise ValueError. CPU, training/autograd, other GPU architectures,
other dtypes/head dimensions and explicitly configured non-SDPA HF backends retain
their upstream behavior. No global Transformers attention function is patched.

FA4 is loaded lazily and cached per encoder. This integration requires
`flash-attn-4 >= 4.0.0b38`; older installed beta builds failed on the tested shape.
Missing dependencies/old versions produce a warning and use SDPA. The repository's
base dependency floor and lockfile may still install an older beta; changing code
alone does not upgrade an existing runtime. Verify the startup log reports
`Qwen3 vision attention backend: FA4`. Do not silently treat fallback as FA4.
The validated isolated environment uses flash-attn-4 4.0.0b38, CUTLASS DSL 4.8.1,
PyTorch 2.11.0+cu130. No production dependency installation was changed.

Execution/compilation errors from the FA4 kernel are surfaced; CUDA OOM and illegal
access are not swallowed. The existing fused-RoPE compiler fallback still applies:
FA4 can continue with upstream RoPE, but loses the fusion's speed benefit.

## Validation

35 selection, fallback, RoPE and compatibility tests passed. Existing bitwise RoPE
regression tests explicitly select SDPA. Real public Qwen3-Omni weights were tested
with the integrated FA4 path on RTX PRO 5000 72GB. All five visual outputs exactly
match the earlier isolated FA4 prototype on large and small images. This is NOT
bitwise equality against SDPA: the preceding experiment measured approximately
1.94% relative L2 difference in the large-image pooled features. No business
accuracy equivalence is claimed.

One model, alternating implementations, four warmup pairs and twelve measured
pairs; GPU-resident preprocessed inputs, synchronization around each full ViT:

| Input | SDPA | Integrated FA4 |
|---|---:|---:|
| Original cars.jpg, grid 1x114x212 | 610.33ms | 549.01ms |
| Small 448x256 image | 5.93ms | 7.83ms |

Large-image ViT latency drops about 10%; small images are slower. These timings do
not include HTTP, preprocessing, transfer or Thinker. No automatic size threshold
was introduced: the requested FA4 default is explicit, with SDPA rollback.

Workspace evidence: `optimization/fa4-integration/artifacts/results.json` and
`tests.log`. These measurements predate the November 18 deployment; see [the integrated bundle](qwen3-verified-optimizations.md) for scope and validation limits.
