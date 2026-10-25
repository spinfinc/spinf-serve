"""W4A16 GEMM for a few rows (the extraction decoder's steps, qwen_decode.py): the int4 weights read as stored, no dequantized copy.

A decode step runs every linear on M = slots rows (tens): the weights (13 GB of int4) are the traffic, so the kernel streams them
once and dequantizes in registers; dequantizing to a bf16 / int8 copy first, as the scoring path does for its 16k-row GEMMs, writes
and reads twice the bytes (QWEN.md §10). Layout as Int4Linear keeps it: q [N, K/2] uint8 (low nibble = even k, high = odd k, value
q - 8), s [N, K/32] fp16 group scales. Each program owns BN output channels and a K slice (split-K for the narrow linears: the
partials land in a [splits, M, N] fp32 buffer and are summed in a fixed order: deterministic); the x tile's even and odd columns
meet the low and high nibbles in two fp16 tensor-core dots with fp32 accumulation. Numerics: W4A16 with fp32 accumulation -- the
dequantized weight times fp16 activations, closer to exact than the scoring path's int8 or fp16-accumulate schemes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w4_small(X, Q, S, Y, M, N, K, k_per, sxm, sqn, ssn, sys_, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # program (n block, k split): y[split, :M, n block] = x[:, k slice] . w[n block, k slice]^T
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    nmask = rn < N
    mmask = rm < M
    j = tl.arange(0, BK // 2)  # byte j of a BK-wide k tile holds k = 2j (low nibble) and 2j + 1 (high nibble)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    k_lo = pid_k * k_per
    for k0 in range(k_lo, k_lo + k_per, BK):
        qb = tl.load(Q + rn[:, None] * sqn + (k0 // 2 + j)[None, :], mask=nmask[:, None], other=0)  # [BN, BK/2] uint8
        sg = tl.load(S + rn[:, None] * ssn + (k0 // 32 + tl.arange(0, BK // 32))[None, :], mask=nmask[:, None], other=0.0)
        sg3 = tl.reshape(sg, [BN, BK // 32, 1])  # a group of 32 k is 16 bytes
        lo = tl.reshape(tl.reshape((qb & 15).to(tl.float16) - 8.0, [BN, BK // 32, 16]) * sg3, [BN, BK // 2])
        hi = tl.reshape(tl.reshape((qb >> 4).to(tl.float16) - 8.0, [BN, BK // 32, 16]) * sg3, [BN, BK // 2])
        xe = tl.load(X + rm[:, None] * sxm + (k0 + 2 * j)[None, :], mask=mmask[:, None], other=0.0)  # [BM, BK/2] fp16
        xo = tl.load(X + rm[:, None] * sxm + (k0 + 2 * j + 1)[None, :], mask=mmask[:, None], other=0.0)
        acc = tl.dot(xe, tl.trans(lo), acc)
        acc = tl.dot(xo, tl.trans(hi), acc)
    tl.store(Y + pid_k * sys_ + rm[:, None] * sym + rn[None, :], acc, mask=mmask[:, None] & nmask[None, :])


SMALL = {"BN": 64, "BK": 256, "warps": 4, "stages": 3, "programs": 512}  # tuned in research/qwen/check_w4.py (29 ms per step)


def linear(x, q, s, extra=None):
    """x [M, K] (bf16 / fp16, M up to 128), q [N, K/2] uint8, s [N, K/32] fp16 -> [M, N (+ E)] bf16; extra [E, K] bf16 rows
    appended to the output (the GDN's b / a projections) run as a plain matmul"""
    M, K = x.shape
    N = q.shape[0]
    BN, BK = SMALL["BN"], SMALL["BK"]
    assert K % BK == 0 and M <= 128
    BM = max(16, triton.next_power_of_2(M))
    n_blocks = triton.cdiv(N, BN)
    splits = max(1, min(K // BK, SMALL["programs"] // n_blocks))
    while (K // BK) % splits:
        splits -= 1
    k_per = K // splits
    xh = x if x.dtype == torch.float16 else x.to(torch.float16)
    part = torch.empty((splits, M, N), dtype=torch.float32, device=x.device)
    _w4_small[(n_blocks, splits)](
        xh, q, s, part, M, N, K, k_per, xh.stride(0), q.stride(0), s.stride(0), part.stride(0), part.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=SMALL["warps"], num_stages=SMALL["stages"],
    )  # fmt: skip
    y = (part[0] if splits == 1 else part.sum(0)).to(torch.bfloat16)
    if extra is not None:
        y = torch.cat([y, (x.to(extra.dtype) @ extra.t()).to(torch.bfloat16)], -1)
    return y
