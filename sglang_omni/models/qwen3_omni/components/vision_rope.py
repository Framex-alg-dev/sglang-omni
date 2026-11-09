# SPDX-License-Identifier: Apache-2.0
"""Fused inference RoPE with the HF FP32 multiply/add rounding order.

Loaded lazily by vision_compat so CPU-only environments do not require Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    CS: tl.constexpr,
    CD: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N * H * D
    d = i % D
    h = (i // D) % H
    s = i // (H * D)
    other = tl.where(d < D // 2, d + D // 2, d - D // 2)
    c = tl.load(C + s * CS + d * CD, mask, 0).to(tl.float32)
    sn = tl.load(S + s * SS + d * SD, mask, 0).to(tl.float32)
    q = tl.load(Q + s * QS + h * QH + d * QD, mask, 0).to(tl.float32)
    qr = tl.load(Q + s * QS + h * QH + other * QD, mask, 0).to(tl.float32)
    k = tl.load(K + s * KS + h * KH + d * KD, mask, 0).to(tl.float32)
    kr = tl.load(K + s * KS + h * KH + other * KD, mask, 0).to(tl.float32)
    qr = tl.where(d < D // 2, -qr, qr)
    kr = tl.where(d < D // 2, -kr, kr)
    # Explicit round-to-nearest instructions prevent algebraic contraction or
    # reassociation, including with FP32 (rather than BF16) position tensors.
    qc = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [q, c], tl.float32, True, 1
    )
    qs = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [qr, sn], tl.float32, True, 1
    )
    kc = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [k, c], tl.float32, True, 1
    )
    ks = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [kr, sn], tl.float32, True, 1
    )
    qo = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [qc, qs], tl.float32, True, 1
    )
    ko = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [kc, ks], tl.float32, True, 1
    )
    tl.store(OQ + i, qo, mask)
    tl.store(OK + i, ko, mask)


def fused_vision_rope(q, k, cos, sin):
    assert (
        q.shape == k.shape
        and q.ndim == 3
        and cos.shape == sin.shape == (q.shape[0], q.shape[2])
    )
    assert q.shape[-1] % 2 == 0 and q.dtype == k.dtype == torch.bfloat16
    assert not torch.is_grad_enabled()
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    n, h, d = q.shape
    # Keep multiplication and addition separate: FMA changes BF16 results.
    # Guard the device for callers whose current CUDA device differs from q.device.
    with torch.cuda.device(q.device):
        _rope[(triton.cdiv(q.numel(), 1024),)](
            q,
            k,
            cos,
            sin,
            oq,
            ok,
            n,
            h,
            d,
            *q.stride(),
            *k.stride(),
            *cos.stride(),
            *sin.stride(),
            1024,
            enable_fp_fusion=False,
        )
    return oq, ok
