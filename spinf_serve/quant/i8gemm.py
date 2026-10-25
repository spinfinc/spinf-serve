"""int8 GEMM for the W8A8 linears (qwen.py mode i8): y = (x8 * sx) @ (w8 * sw)^T in bf16, the activation scale per row and block
of BK input channels, the weight scale per output channel, both applied in the epilogue (no int32 round trip through memory).

On a 4090 the int8 tensor cores run at twice the fp16-accumulate rate (torch._int_mm 545-620 TOPS vs 270-297 TF, same power
cap), and a block scale per 128 input channels keeps int8 activations at bf16 accuracy where a per-token scale lost 3.7 points
(QWEN.md). Numerics: every K block's dot product is exact in int32 (|x|, |w| <= 127, 128 terms), the blocks are summed in fp32 in
ascending order, so a row's result does not depend on the tile configuration or on how many rows run with it: batch-invariant
for any M, no GEMM floor. The configuration per (N, K) is chosen by timing at load (tune), the candidates below.
"""

import time

import torch
import triton
import triton.language as tl

BLK = 128  # input channels per activation scale = the kernel's K tile
# (BM, BN, num_stages, num_warps)
CONFIGS = [
    (128, 128, 3, 4),
    (128, 128, 4, 4),
    (128, 64, 4, 4),
    (128, 64, 3, 4),
    (64, 128, 4, 4),
    (64, 64, 4, 4),
    (64, 64, 4, 2),
    (128, 128, 3, 8),
    (64, 256, 3, 4),
    (256, 64, 3, 8),
    (128, 64, 5, 4),
    (128, 128, 2, 4),
    (64, 128, 3, 2),
]
CHOSEN: dict[tuple[int, int], tuple[int, int, int, int]] = {}  # (N, K) -> config


@triton.jit
def _gemm(
    a,
    b,
    c,
    sa,
    sb,
    M,
    N,
    K,
    sam,
    sbn,
    scm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    # a [M, K] int8 (row stride sam), b [N, K] int8 (row stride sbn; read as b^T), c [M, N] bf16 (row stride scm),
    # sa [M, K / BK] fp32 contiguous, sb [N] fp32; grouped tile order along M for L2 reuse of b
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP * num_n
    gid = pid // group
    first_m = gid * GROUP
    gsize = min(num_m - first_m, GROUP)
    pid_m = first_m + ((pid % group) % gsize)
    pid_n = (pid % group) // gsize
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    m_m = om < M
    m_n = on < N
    pa = a + om[:, None].to(tl.int64) * sam + ok[None, :]
    pb = b + on[None, :].to(tl.int64) * sbn + ok[:, None]
    nkb = tl.cdiv(K, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, nkb):
        part = tl.dot(tl.load(pa, mask=m_m[:, None], other=0), tl.load(pb, mask=m_n[None, :], other=0))
        acc += part.to(tl.float32) * tl.load(sa + om * nkb + kb, mask=m_m, other=0.0)[:, None]
        pa += BK
        pb += BK
    y = acc * tl.load(sb + on, mask=m_n, other=0.0)[None, :]
    tl.store(c + om[:, None].to(tl.int64) * scm + on[None, :], y.to(tl.bfloat16), mask=m_m[:, None] & m_n[None, :])


@triton.jit
def _round(x):  # half away from zero
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quant(x, x8, sx, M, K, sxm, BM: tl.constexpr, BK: tl.constexpr):
    # x [M, K] (row stride sxm) -> x8 [M, K] int8, sx [M, K / BK] fp32: per row and block of BK columns, absmax / 127
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    ok = pid_k * BK + tl.arange(0, BK)
    m = om < M
    v = tl.load(x + om[:, None].to(tl.int64) * sxm + ok[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(v), 1), 1e-12) / 127.0
    q = _round(v / sc[:, None])
    tl.store(x8 + om[:, None].to(tl.int64) * K + ok[None, :], q.to(tl.int8), mask=m[:, None])
    tl.store(sx + om * (K // BK) + pid_k, sc, mask=m)


def quantize(x):
    """x [M, K] (bf16 / fp16 / fp32, K a multiple of BLK) -> (int8 [M, K], scales [M, K / BLK] fp32), one pass"""
    M, K = x.shape
    assert K % BLK == 0 and x.stride(1) == 1
    x8 = torch.empty(M, K, dtype=torch.int8, device=x.device)
    sx = torch.empty(M, K // BLK, dtype=torch.float32, device=x.device)
    BM = 32
    _quant[(triton.cdiv(M, BM), K // BLK)](x, x8, sx, M, K, x.stride(0), BM=BM, BK=BLK, num_warps=4)
    return x8, sx


@triton.jit
def _dequant(q, s, inv, out, N, K, G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # q [N, K / 2] uint8 (low nibble = the even input), s [N, K / G] fp16 group scales, inv [N] fp32 -> out [N, K] int8:
    # round(s_g (q - 8) inv_n), clamped to +-127
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    on = pid_n * BN + tl.arange(0, BN)
    oj = pid_k * (BK // 2) + tl.arange(0, BK // 2)  # byte columns
    mn = on < N
    b = tl.load(q + on[:, None].to(tl.int64) * (K // 2) + oj[None, :], mask=mn[:, None], other=0)
    grp = (2 * oj) // G  # columns 2j and 2j + 1 share a group (G even)
    sg = tl.load(s + on[:, None].to(tl.int64) * (K // G) + grp[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
    f = sg * tl.load(inv + on, mask=mn, other=0.0)[:, None]
    lo = tl.clamp(_round(((b & 15).to(tl.float32) - 8.0) * f), -127.0, 127.0).to(tl.int8)
    hi = tl.clamp(_round(((b >> 4).to(tl.float32) - 8.0) * f), -127.0, 127.0).to(tl.int8)
    w = tl.interleave(lo, hi)  # [BN, BK]: even columns from the low nibbles
    ok = pid_k * BK + tl.arange(0, BK)
    tl.store(out + on[:, None].to(tl.int64) * K + ok[None, :], w, mask=mn[:, None])


def dequant(q, s, inv, G=32, extra=None):
    """GPTQ int4 rows q [N, K / 2] uint8 (two inputs per byte), s [N, K / G] fp16, inv [N] fp32 -> int8 [N (+ E), K], one pass;
    extra [E, K] int8 rows are copied below them"""
    N, K2 = q.shape
    K = 2 * K2
    E = 0 if extra is None else extra.shape[0]
    out = torch.empty(N + E, K, dtype=torch.int8, device=q.device)
    BN, BK = 64, 256
    _dequant[(triton.cdiv(N, BN), K // BK)](q, s, inv, out, N, K, G=G, BN=BN, BK=BK, num_warps=4)
    if E:
        out[N:].copy_(extra)
    return out


def linear(x, q, s, inv_sw, sw, extra8=None, sw_ext=None):
    """the W8A8 linear: x [M, K] -> [M, N (+ E)] bf16; extra8 [E, K] int8 rows appended to the weight (their scales in sw_ext)"""
    x8, sx = quantize(x)
    return gemm(x8, sx, dequant(q, s, inv_sw, extra=extra8), sw if extra8 is None else sw_ext)


# ------------------------------------------------------------------------------------------------- the split scheme (cuBLASLt)
# For a linear whose activation outliers live in a fixed set of input channels (the norm-fed in_proj_qkvz, qkv_proj, gate_up_proj:
# 99-100 % of the tokens peak in the same 16 channels, research/qwen/act_outliers.py), a per-token scale over the other channels
# is as accurate as the block scale (WinoGrande 85.0 with 64 channels exact), and a per-token scale lets cuBLASLt's int8 GEMM
# (torch._int_mm, 560-630 TOPS vs the Triton kernel's 350-380) do the work: the outlier channels are zero in the int8 activations,
# their exact contribution (x_out [M, D] fp16 times the dequantized int4 columns [N, D]) is added in the epilogue that scales the
# int32 result into bf16. The int32 accumulation is exact, so cuBLASLt's kernel choice by M cannot change a result: invariant.


@triton.jit
def _quant_tok(x, keep, x8, sx, xo, oi, M, K, sxm, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    # per row: scale = absmax over the kept channels / 127; the other channels are written as 0; D > 0: the columns oi [D] of x
    # are also written to xo [M, D] fp16 (the split epilogues' exact term), in the same pass over x
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    mm = om < M
    if D > 0:
        od = tl.arange(0, D)
        col = tl.load(oi + od)
        vo = tl.load(x + om[:, None].to(tl.int64) * sxm + col[None, :], mask=mm[:, None], other=0.0)
        tl.store(xo + om[:, None].to(tl.int64) * D + od[None, :], vo.to(tl.float16), mask=mm[:, None])
    amax = tl.zeros([BM], dtype=tl.float32)
    for kb in range(0, K // BK):
        ok = kb * BK + tl.arange(0, BK)
        kp = tl.load(keep + ok)
        v = tl.load(x + om[:, None].to(tl.int64) * sxm + ok[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        v = tl.where(kp[None, :] != 0, v, 0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(v), 1))
    sc = tl.maximum(amax, 1e-12) / 127.0
    tl.store(sx + om, sc, mask=mm)
    for kb in range(0, K // BK):
        ok = kb * BK + tl.arange(0, BK)
        kp = tl.load(keep + ok)
        v = tl.load(x + om[:, None].to(tl.int64) * sxm + ok[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        v = tl.where(kp[None, :] != 0, v, 0.0)
        tl.store(x8 + om[:, None].to(tl.int64) * K + ok[None, :], _round(v / sc[:, None]).to(tl.int8), mask=mm[:, None])


def quantize_tok(x, keep, out_idx=None):
    """x [M, K], keep [K] uint8 (1 = quantized channel) -> (int8 [M, K] with the other channels 0, scales [M] fp32); with
    out_idx [D] int64 also the columns out_idx of x in fp16 [M, D] (what the split epilogues' outlier dot reads), one pass"""
    M, K = x.shape
    assert K % BLK == 0 and x.stride(1) == 1 and keep.dtype == torch.uint8
    x8 = torch.empty(M, K, dtype=torch.int8, device=x.device)
    sx = torch.empty(M, dtype=torch.float32, device=x.device)
    D = 0 if out_idx is None else out_idx.shape[0]
    xo = torch.empty(M, D, dtype=torch.float16, device=x.device) if D else sx
    BM = 32
    _quant_tok[(triton.cdiv(M, BM),)](x, keep, x8, sx, xo, out_idx if D else keep, M, K, x.stride(0), D=D, BM=BM, BK=BLK, num_warps=4)
    return (x8, sx, xo) if D else (x8, sx)


@triton.jit
def _epilogue(acc, sx, sw, xo, wo, out, M, N, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # out[m, n] = acc[m, n] * sx[m] * sw[n] + sum_d xo[m, d] * wo[n, d]; acc int32 [M, N], xo [M, D] fp16, wo [N, D] fp16
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    od = tl.arange(0, D)
    mm = om < M
    mn = on < N
    m2 = mm[:, None] & mn[None, :]
    a = tl.load(acc + om[:, None].to(tl.int64) * N + on[None, :], mask=m2, other=0).to(tl.float32)
    y = a * tl.load(sx + om, mask=mm, other=0.0)[:, None] * tl.load(sw + on, mask=mn, other=0.0)[None, :]
    xt = tl.load(xo + om[:, None].to(tl.int64) * D + od[None, :], mask=mm[:, None], other=0.0)
    wt = tl.load(wo + on[None, :].to(tl.int64) * D + od[:, None], mask=mn[None, :], other=0.0)
    y += tl.dot(xt, wt)
    tl.store(out + om[:, None].to(tl.int64) * N + on[None, :], y.to(tl.bfloat16), mask=m2)


# the split GEMMs' int32 accumulator is produced in row chunks of this size, into one reused buffer: at 16k rows the gate_up
# accumulator alone is 2.3 GB (the scoring forward's peak, QWEN.md §11); the epilogues are row-wise and the chunk is a multiple of
# their tiles, so the rows are bit-identical to one call. A trailing chunk of 16 rows or fewer joins the one before it (torch._int_mm
# wants more than 16)
ROW_CHUNK = 8192


def row_chunks(M: int):
    bounds = []
    m0 = 0
    while m0 < M:
        m1 = M if M - (m0 + ROW_CHUNK) <= 16 else m0 + ROW_CHUNK
        bounds.append((m0, m1))
        m0 = m1
    return bounds


def split_linear(x, q, s, inv_sw, sw, keep, out_idx, w_out, extra8=None, sw_ext=None, w_out_ext=None):
    """x [M, K] -> [M, N (+ E)] bf16: cuBLASLt int8 over the kept channels (per-token scale), the outlier channels out_idx [D]
    exact through w_out [N (+ E), D] fp16 in the epilogue; inv_sw / sw are the weights' per-channel scales over the kept columns"""
    x8, sx, xo = quantize_tok(x, keep, out_idx)
    w8t = dequant(q, s, inv_sw, extra=extra8).t()
    M, N = x8.shape[0], w8t.shape[1]
    wo = w_out if extra8 is None else w_out_ext
    swn = sw if extra8 is None else sw_ext
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    BM, BN = 64, 128
    chunks = row_chunks(M)
    acc_buf = torch.empty(chunks[0][1], N, dtype=torch.int32, device=x.device)  # one accumulator, reused by the chunks
    for m0, m1 in chunks:
        acc = torch._int_mm(x8[m0:m1], w8t, out=acc_buf[: m1 - m0])
        _epilogue[(triton.cdiv(m1 - m0, BM), triton.cdiv(N, BN))](
            acc, sx[m0:], swn, xo[m0:], wo, out[m0:], m1 - m0, N, D=wo.shape[1], BM=BM, BN=BN, num_warps=4
        )
    return out


@triton.jit
def _epilogue_glu_q(acc, sx, sw, xo, wo, out8, so, M, I, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # the gate_up epilogue fused with the MLP's silu(gate) * up and the block quantization of the result (the down_proj's input):
    # g = acc[m, n] sx[m] sw[n] + xo[m] . wo[n], u = the same at column I + n (n < I); h = silu(g) u in fp32;
    # out8[m, n] = round(h / s), so[m, n // BN] = s = absmax of the tile's BN (= BLK) columns / 127. I is a multiple of BN.
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    od = tl.arange(0, D)
    mm = om < M
    rows = om[:, None].to(tl.int64) * (2 * I)
    s_m = tl.load(sx + om, mask=mm, other=0.0)[:, None]
    xt = tl.load(xo + om[:, None].to(tl.int64) * D + od[None, :], mask=mm[:, None], other=0.0)
    g = tl.load(acc + rows + on[None, :], mask=mm[:, None], other=0).to(tl.float32) * s_m * tl.load(sw + on)[None, :]
    g += tl.dot(xt, tl.load(wo + on[None, :].to(tl.int64) * D + od[:, None]))
    u = tl.load(acc + rows + (I + on)[None, :], mask=mm[:, None], other=0).to(tl.float32) * s_m * tl.load(sw + I + on)[None, :]
    u += tl.dot(xt, tl.load(wo + (I + on)[None, :].to(tl.int64) * D + od[:, None]))
    h = g * tl.sigmoid(g) * u
    sc = tl.maximum(tl.max(tl.abs(h), 1), 1e-12) / 127.0
    tl.store(so + om * (I // BN) + pid_n, sc, mask=mm)
    tl.store(out8 + om[:, None].to(tl.int64) * I + on[None, :], _round(h / sc[:, None]).to(tl.int8), mask=mm[:, None])


GLU_TILE = {"BM": 32, "BN": BLK, "warps": 4}  # BN = BLK: one program's columns are one activation block of the down_proj


def split_glu_q(x, q, s, inv_sw, sw, keep, out_idx, w_out):
    """the MLP's gate_up_proj (split scheme, [N = 2 I] outputs: gate then up) with silu(gate) * up and the block quantization of
    the result fused into the epilogue: x [M, K] -> (int8 [M, I], scales [M, I / BLK]) for down_proj (linear_q). Saves the
    [M, 2 I] bf16 round trip and the quantizer pass; row-wise, so batch-invariant like split_linear."""
    x8, sx, xo = quantize_tok(x, keep, out_idx)
    w8t = dequant(q, s, inv_sw).t()
    M, N = x8.shape[0], w8t.shape[1]
    I = N // 2
    assert N % 2 == 0 and I % BLK == 0
    out8 = torch.empty(M, I, dtype=torch.int8, device=x.device)
    so = torch.empty(M, I // BLK, dtype=torch.float32, device=x.device)
    BM, BN, warps = GLU_TILE["BM"], GLU_TILE["BN"], GLU_TILE["warps"]
    chunks = row_chunks(M)
    acc_buf = torch.empty(chunks[0][1], N, dtype=torch.int32, device=x.device)
    for m0, m1 in chunks:
        acc = torch._int_mm(x8[m0:m1], w8t, out=acc_buf[: m1 - m0])
        _epilogue_glu_q[(triton.cdiv(m1 - m0, BM), I // BN)](
            acc, sx[m0:], sw, xo[m0:], w_out, out8[m0:], so[m0:], m1 - m0, I, D=w_out.shape[1], BM=BM, BN=BN, num_warps=warps
        )
    return out8, so


def linear_q(x8, sx, q, s, inv_sw, sw):
    """the W8A8 linear on an already quantized activation (quantize's / split_glu_q's int8 + block scales) -> [M, N] bf16"""
    return gemm(x8, sx, dequant(q, s, inv_sw), sw)


@triton.jit
def _epilogue_gdn(acc, sx, sw, xo, wo, conv, cw, out, M, N, NC, NN, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # the Gated DeltaNet in_proj epilogue: the columns < NC (q k v) get the causal conv along the token's path (conv [M, 4] int64:
    # the rows of its taps, M = the zero row; cw [NC, 4] fp32 tap weights) and silu, the columns < NN (q, k; BN = one head) the
    # l2 normalisation per head row; the other columns (z, b, a) the plain scaled int32 + outlier dot. All in fp32, bf16 out.
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    od = tl.arange(0, D)
    mm = om < M
    mn = on < N
    swn = tl.load(sw + on, mask=mn, other=0.0)[None, :]
    wt = tl.load(wo + on[None, :].to(tl.int64) * D + od[:, None], mask=mn[None, :], other=0.0)
    if pid_n * BN < NC:
        y = tl.zeros([BM, BN], dtype=tl.float32)
        for j in tl.static_range(4):
            r = tl.load(conv + om * 4 + j, mask=mm, other=M)
            mr = r < M
            a = tl.load(acc + r[:, None] * N + on[None, :], mask=mr[:, None] & mn[None, :], other=0).to(tl.float32)
            v = a * tl.load(sx + r, mask=mr, other=0.0)[:, None] * swn
            v += tl.dot(tl.load(xo + r[:, None] * D + od[None, :], mask=mr[:, None], other=0.0), wt)
            y += v * tl.load(cw + on * 4 + j, mask=mn, other=0.0)[None, :]
        y = y * tl.sigmoid(y)
        if pid_n * BN < NN:
            y = y * tl.rsqrt(tl.sum(y * y, 1) + 1e-6)[:, None]
    else:
        a = tl.load(acc + om[:, None].to(tl.int64) * N + on[None, :], mask=mm[:, None] & mn[None, :], other=0).to(tl.float32)
        y = a * tl.load(sx + om, mask=mm, other=0.0)[:, None] * swn
        y += tl.dot(tl.load(xo + om[:, None].to(tl.int64) * D + od[None, :], mask=mm[:, None], other=0.0), wt)
    tl.store(out + om[:, None].to(tl.int64) * N + on[None, :], y.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


GDN_TILE = {"BM": 32, "BN": 128, "warps": 4}  # BN = the GDN head dim (128): a q / k tile is one head row


def split_gdn_in(x, q, s, inv_sw, sw_ext, keep, out_idx, w_out_ext, extra8, conv, cw, n_conv, n_norm):
    """the Gated DeltaNet's in_proj_qkvz (split scheme with the b / a rows appended: [N = conv_dim + value dim + 2 hv]) with the
    short causal conv (conv [M, 4] int64 rows of the taps, M = none; cw [conv_dim, 4] fp32), silu and the q / k l2 norm (the first
    n_norm columns; 0 = none) fused into the epilogue: x [M, K] -> [M, N] bf16 with the conv'd q k v, then z, b, a as the GEMM
    gives them. Replaces the gather + conv pass and the norm pass; each row's result depends on its taps' rows only: batch-invariant."""
    x8, sx, xo = quantize_tok(x, keep, out_idx)
    w8 = dequant(q, s, inv_sw, extra=extra8)
    acc = torch._int_mm(x8, w8.t())
    M, N = acc.shape
    BM, BN, warps = GDN_TILE["BM"], GDN_TILE["BN"], GDN_TILE["warps"]
    assert n_conv % BN == 0 and n_norm % BN == 0 and conv.shape == (M, 4) and conv.is_contiguous() and cw.shape == (n_conv, 4)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _epilogue_gdn[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        acc, sx, sw_ext, xo, w_out_ext, conv, cw, out, M, N, n_conv, n_norm, D=w_out_ext.shape[1], BM=BM, BN=BN, num_warps=warps
    )
    return out


def gemm(x8, sx, w8, sw, config=None):
    """x8 [M, K] int8, sx [M, K / BLK] fp32, w8 [N, K] int8, sw [N] fp32 -> [M, N] bf16"""
    M, K = x8.shape
    N = w8.shape[0]
    assert K % BLK == 0 and sx.shape == (M, K // BLK) and sx.is_contiguous() and x8.stride(1) == 1 and w8.stride(1) == 1
    BM, BN, stages, warps = config or CHOSEN.get((N, K), CONFIGS[0])
    c = torch.empty(M, N, dtype=torch.bfloat16, device=x8.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    strides = (x8.stride(0), w8.stride(0), c.stride(0))
    _gemm[grid](x8, w8, c, sx, sw, M, N, K, *strides, BM=BM, BN=BN, BK=BLK, GROUP=8, num_stages=stages, num_warps=warps)
    return c


def tune(shapes, M: int, device) -> dict:
    """time every configuration per (N, K) at M rows (random int8 data) and keep the fastest in CHOSEN -> {(N, K): (config, ms)}"""
    out = {}
    for N, K in sorted(set(shapes)):
        x8 = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8)
        w8 = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)
        sx = torch.rand(M, K // BLK, device=device) + 0.5
        sw = torch.rand(N, device=device) + 0.5
        best = None
        for cfg in CONFIGS:
            try:
                gemm(x8, sx, w8, sw, cfg)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(3):
                    gemm(x8, sx, w8, sw, cfg)
                torch.cuda.synchronize(device)
                ms = (time.perf_counter() - t0) / 3 * 1000
            except (triton.runtime.errors.OutOfResources, RuntimeError):
                continue
            if best is None or ms < best[1]:
                best = (cfg, ms)
        if best is None:
            raise RuntimeError(f"i8gemm: no configuration runs for N {N} K {K}")
        CHOSEN[(N, K)] = best[0]
        out[(N, K)] = (best[0], round(best[1], 2))
        del x8, w8, sx, sw
    torch.cuda.empty_cache()
    return out
