"""Gated DeltaNet over trie segments: initial states read by index (no gathered copies), final states written only for the
segments that have children, and short segments run token by token (no per-chunk state snapshots).

A level of a TreePlan (qwen.py) is a set of segments; each starts from a row of its parents' state tensor (or zero) and may
have to leave its final state for its own children. Two kernels:

  chunked    fla's chunk pipeline (cumsum, WY intra-chunk, state recurrence, output) for long segments; the state recurrence
             is fla's chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (fla-core 0.5.2, MIT, (c) 2023-2026 Songlin Yang, Yu Zhang,
             Zhiyuan Li) with two changes: h0 is read at row h0_idx[n] and ht written at row ht_idx[n] (skipped when < 0)
  recurrent  one program per (segment, value head, 32 values): the 128 x 32 state in registers, one token at a time; for the
             short segments of a trie (a 64-token chunk per 5-token leaf would snapshot 1.5 MB of state per layer)

q and k arrive l2-normalised; g is the log decay per token and head, beta the write strength, scale 1 / sqrt(K).
"""

import torch
import triton
import triton.language as tl


@triton.heuristics({"USE_INITIAL_STATE": lambda args: args["h0"] is not None, "STORE_FINAL_STATE": lambda args: args["ht"] is not None})
@triton.jit(do_not_specialize=["T"])
def _fwd_h_indexed(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    h0_idx,
    ht,
    ht_idx,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # fla's kernel for K = 128 (two 64-row halves), state [K, V], USE_G, varlen, SAVE_NEW_VALUE; + h0_idx / ht_idx
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    if USE_INITIAL_STATE:
        src = tl.load(h0_idx + i_n).to(tl.int64)
        if src >= 0:
            p0 = h0 + (src * HV + i_h) * K * V
            b_h1 += tl.load(p0 + o_k1[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)
            b_h2 += tl.load(p0 + o_k2[:, None] * V + o_v[None, :], mask=m_v[None, :], other=0.0).to(tl.float32)
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_v[None, :])
        p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_v[None, :])
        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None], other=0.0)
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        p_vn = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_v = b_v * tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.math.exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_v = b_v.to(k.dtype.element_ty)
        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_h1 += tl.dot(tl.load(p_k, mask=m_t[None, :], other=0.0), b_v)
        p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
        b_h2 += tl.dot(tl.load(p_k, mask=m_t[None, :], other=0.0), b_v)
    if STORE_FINAL_STATE:
        dst = tl.load(ht_idx + i_n).to(tl.int64)
        if dst >= 0:
            pt = ht + (dst * HV + i_h) * K * V
            tl.store(pt + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht.dtype.element_ty), mask=m_v[None, :])
            tl.store(pt + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht.dtype.element_ty), mask=m_v[None, :])


def chunked(q, k, v, g, beta, scale, cu, cu_cpu, h0=None, h0_idx=None, ht=None, ht_idx=None, chunk_size=64):
    """q, k [1, T, H, K] (l2-normalised), v [1, T, HV, V], g (log decay) / beta [1, T, HV], cu [N + 1] -> o [1, T, HV, V];
    sequence n starts from h0[h0_idx[n]] (h0_idx < 0 or h0 None: zero) and, where ht_idx[n] >= 0, leaves its final state in
    ht[ht_idx[n]] (fp32 [*, HV, K, V])"""
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
    from fla.ops.utils.constant import RCP_LN2

    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    assert K == 128, "the indexed state kernel is written for K = 128"
    ci = prepare_chunk_indices(cu, chunk_size, cu_seqlens_cpu=cu_cpu)
    g = chunk_local_cumsum(g, chunk_size=chunk_size, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size)
    N = len(cu_cpu) - 1
    NT = len(ci)
    offs = prepare_chunk_offsets(cu, chunk_size)
    h = k.new_empty(1, NT, HV, K, V)
    v_new = torch.empty_like(u)
    BV = 64
    _fwd_h_indexed[(triton.cdiv(V, BV), N * HV)](
        k, u, w, v_new, g, h, h0, h0_idx, ht, ht_idx, cu, offs, T, H=H, HV=HV, K=K, V=V, BT=chunk_size, BV=BV, num_warps=4, num_stages=2
    )
    return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size)


@triton.jit
def _recurrent(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    h0_idx,
    ht,
    ht_idx,
    cu,
    idx,
    scale,
    sq,
    sv,
    sg,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE: tl.constexpr,
    USE_IDX: tl.constexpr,
):
    # token t of the call reads row t of q k v g beta (row strides sq, sv, sg) or, with USE_IDX, row idx[t] of the stream
    # tensors themselves, and writes o at the same row (o [rows, HV, V] contiguous)
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu + i_n).to(tl.int64), tl.load(cu + i_n + 1).to(tl.int64)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_H0:
        src = tl.load(h0_idx + i_n).to(tl.int64)
        if src >= 0:
            b_h += tl.load(h0 + (src * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]).to(tl.float32)
    for t in range(bos, eos):
        r = tl.load(idx + t).to(tl.int64) if USE_IDX else t
        b_q = tl.load(q + r * sq + i_h * K + o_k).to(tl.float32) * scale
        b_k = tl.load(k + r * sq + i_h * K + o_k).to(tl.float32)
        b_v = tl.load(v + r * sv + i_hv * V + o_v).to(tl.float32)
        b_g = tl.load(g + r * sg + i_hv).to(tl.float32)
        b_b = tl.load(beta + r * sg + i_hv).to(tl.float32)
        b_h *= tl.exp(b_g)
        b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_b
        b_h += b_k[:, None] * b_v[None, :]
        tl.store(o + (r * HV + i_hv) * V + o_v, tl.sum(b_h * b_q[:, None], 0).to(o.dtype.element_ty))
    if STORE:
        dst = tl.load(ht_idx + i_n).to(tl.int64)
        if dst >= 0:
            tl.store(ht + (dst * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :], b_h.to(ht.dtype.element_ty))


# values per program (V / bv programs per segment and head) and its warps, timed in research/qwen/bench_recurrent.py
RECURRENT = {"bv": 32, "warps": 2}


def recurrent(q, k, v, g, beta, scale, cu, n, h0=None, h0_idx=None, ht=None, ht_idx=None, idx=None, o=None):
    """chunked()'s contract, token by token: for short segments. idx [sum of lengths] (GPU int64): the call's tokens are these
    rows of q k v g beta (the stream tensors, any row stride; q / k [1, rows, H, K] with H * K contiguous per row, v [1, rows,
    HV, V] with HV * V contiguous, g / beta [1, rows, HV]) and the outputs go to the same rows of `o` [1, rows, HV, V]
    (contiguous), which is returned; without idx: the gathered tensors, rows 0.. in call order"""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if idx is None:
        o = torch.empty_like(v)
    assert o.is_contiguous() and q.stride(1) == k.stride(1) and g.stride(1) == beta.stride(1)
    BV = RECURRENT["bv"]
    _recurrent[(triton.cdiv(V, BV), n * HV)](
        q,
        k,
        v,
        g,
        beta,
        o,
        h0,
        h0_idx,
        ht,
        ht_idx,
        cu,
        idx,
        scale,
        q.stride(1),
        v.stride(1),
        g.stride(1),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        USE_H0=h0 is not None,
        STORE=ht is not None,
        USE_IDX=idx is not None,
        num_warps=RECURRENT["warps"],
    )
    return o
