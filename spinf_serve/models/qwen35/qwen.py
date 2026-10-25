"""Qwen3.5-architecture text stack (Qwen3.8-27B) over the packed tree stream, int4 GPTQ weights.

64 layers: 48 Gated DeltaNet ("linear attention", a recurrent state per head instead of K/V) and every 4th a full-attention layer
(24 query heads with an output gate, 4 KV heads, head_dim 256, rope on the first 64 dims). Full-attention layers read the packed
stream through the same tree mask as Gemma (FlexAttention, scorer._init_flex). A Gated DeltaNet layer cannot be masked: a branch
must start from the recurrent state at the end of its parent. TreePlan cuts the stream into segments (a job's prefix, each run of
a branch up to the next fork) and runs them level by level through fla's variable-length chunk kernel, each segment's initial
state the final state of its parent; the layer's short causal conv (4 taps) reads each token's path predecessors (TreePlan.conv)
instead of its stream neighbours. A flat job (no branches) is one segment: the plain varlen kernel.

Weights: GPTQ v1 int4, symmetric, group 32 along the input (w = s * (q - 8)), kept packed on the card (13 GB for the 64 layers);
each linear dequantizes its weight for the call (EngineConfig.quant, else QwenOptions.mode):
  f16   fp16 weights and activations, cuBLAS with fp16 accumulation (torch allow_fp16_accumulation): the 4090's fp16-accumulate
        rate is twice its fp32-accumulate rate; output error ~0.3 %, as bf16 rounding
  i8    int8 tensor cores (twice the fp16-accumulate rate again): weights re-rounded to int8 per output channel for the call,
        activations int8 per token and block of 128 input channels, i8gemm's Triton kernel with the scales in the epilogue.
        WinoGrande as bf16 (a per-token activation scale alone lost 3.7 points); its rows do not depend on M: no GEMM floor.
        With the checkpoint's outlier table (spinf_outliers.json next to the weights, quantizations/export_outliers.py; third-party
        checkpoints: quant/tables/<model>/<revision>-outliers.json) the three norm-fed linears run the split
        scheme instead: per-token int8 on cuBLASLt (torch._int_mm) with 64 calibrated channels exact in the epilogue, and the
        consumers fused into those epilogues (QwenOptions fuse_glu / fuse_gdn: the MLP's silu x up + down_proj quantizer, the
        GDN's conv + silu + q / k norm). The production configuration and the default: -30 % time vs f16 at f16 accuracy (QWEN.md §8.2-8.3)
  fp8   re-quantized to e4m3 with a per-tensor scale, dynamic per-tensor activation scale, the fp8 GEMM: ~10 % faster than f16,
        ~3 % output error (WinoGrande -1.7 points)
  bf16  exact weights, the bf16 GEMM (fp32 accumulation): the reference, 1.5x slower than f16
Projections of one input are fused (qkv+z, q+k+v, gate+up), and each GEMM shape runs in the faster orientation (tune_orientation).
The embedding table stays on the host (2.4 GB): rows are gathered per forward into pinned memory. With QwenOptions.compact the model runs on
the stream's real tokens only; the alignment pads exist in the flex layers alone (the GEMMs skip them).
Batch invariance (docs/QWEN.md §5): aligned jobs, f16 streams of at least MIN_STREAM rows, fixed lm-head rows.
Images: the vision tower's embeddings over the placeholder tokens and 3-D rope positions for the full-attention layers
(qwen_vision.py); the recurrent layers and the tree mask see image tokens as text. Extraction: qwen_decode.py (the decoder
re-runs the recurrence per slot from the prefix's captured state).
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields

import numpy as np

from spinf_serve.models.base import Stack
from spinf_serve.quant.tables import load_outliers_table, outliers_table_path

FP8_MAX = 448.0
MIN_STREAM = {"f16": 5248}  # per linear mode, see build_model (i8: none needed; fp8 / bf16: not measured)
# i8 with an outlier table (per linear, the input channels whose activation outliers sit in fixed positions): these linears run the
# cuBLASLt scheme (i8gemm.split_linear); the others the block-scaled Triton GEMM
SPLIT_LINEARS = ("in_proj_qkvz", "qkv_proj", "gate_up_proj")


@dataclass(frozen=True)
class QwenOptions:
    """the stack's run-time choices, one object per load (Qwen35Stack): the production configuration by default, the research
    settings through EngineConfig.stack_options ("key=value,key=value": parse) or built directly by a research script"""

    # the linears' scheme when EngineConfig.quant is empty: f16 | i8 | fp8 | bf16, or a sim_* research mode (Int4Linear.forward)
    mode: str = "i8"
    # i8 split: the MLP's gate_up epilogue computes silu(gate) * up and quantizes it for down_proj in the same pass (i8gemm.split_glu_q):
    # the [T, 2 I] bf16 round trip and the quantizer pass go; the numerics change at the rounding level (silu x mul in fp32, quantized
    # from fp32), row-wise as before. Off: the plain split_linear + the compiled silu x mul
    fuse_glu: bool = True
    # the Gated DeltaNet's in_proj epilogue runs the short conv, silu and the q / k l2 norm (i8gemm.split_gdn_in): the gather +
    # conv pass and the norm pass go
    fuse_gdn: bool = True
    # how the Gated DeltaNet layers run a stream's segments (TreePlan.path):
    #   levels   fla's chunk kernel per level, the parents' states gathered into per-segment copies (3 MiB each, calls of at most
    #            96); every segment chunked (64-token chunks, one state snapshot per chunk)
    #   indexed  gdn.py: states read and written by index (no copies, two calls per level), segments of at most `gdn_short` tokens
    #            token by token (no per-chunk snapshots), longer ones chunked; q / k l2-normalised in the compiled layer (fla's
    #            l2norm kernel is autotuned on the stream length). Equal on coarse tries (decider jobs 311 vs 312 s), +34 % on fine
    #            ones (news-oil: 2,340 segments in a 16k stream)
    # A segment's numerics depend only on itself (its length picks the kernel), so either path keeps batch invariance.
    gdn_path: str = "indexed"
    gdn_short: int = 64
    # rec_idx: the token-by-token kernel indexes the stream tensors. call_tokens: a level's long segments go to the chunked kernel
    # in calls of at most this many tokens (its per-64-token state snapshots are 3 MiB each: 0.77 GB for a 16k call, the scoring
    # forward's peak with the in_proj accumulator; segments are independent, so the split changes nothing in the rows)
    rec_idx: bool = True
    call_tokens: int = 8192
    # compact rows: the linears (and everything row-wise) run on the stream's real tokens only, packed in front of a few zero rows
    # (whole 128-blocks, at least the GEMM floor MIN_STREAM); the alignment pads (8 % of a production stream, 35-55 % on 100-token
    # jobs) exist only in the flex layers, where q / k / v are scattered into the stream layout the tree mask describes. Row
    # results do not change (row-independent kernels, one GEMM kernel per shape from the floor up): batch invariance holds as before
    compact: bool = True
    # the extraction decoder's linears (qwen_decode): "w4" = w4gemm (the int4 weights streamed once, fp32 accumulation: no
    # dequantized copy per step, the dominant cost of "model"); "model" = the Int4Linear of the scoring path (its numerics,
    # dequantized per call). decode_graphs: the step captured once in a CUDA graph and replayed
    decode_linear: str = "w4"
    decode_graphs: bool = True
    embed_host: bool = True  # the embedding table off the card (2.4 GB), rows gathered per forward into pinned memory
    n_out: int = 64  # i8 split: exact channels per linear when `outliers` holds per-channel maxima instead of the table's indices
    # research only, built directly (not from a string): the outlier channels per linear (per-channel absmax tensors, or the
    # table's index lists), the linears a sim_* mode applies to, the K block of the sim_*b modes, SmoothQuant's alpha
    outliers: object = None
    only: object = None
    blk: int = 128
    alpha: float = 0.5

    @classmethod
    def parse(cls, text: str) -> "QwenOptions":
        """ "key=value,key=value" over the fields above (bool: 1/0/true/false; empty text: the defaults)"""
        kw: dict = {}
        types = {f.name: f.type for f in fields(cls)}
        for item in filter(None, (x.strip() for x in text.split(","))):
            k, _, v = item.partition("=")
            k = k.strip()
            if k not in types or k in ("outliers", "only"):
                raise ValueError(
                    f"stack_options: unknown Qwen option {k!r} (one of {', '.join(n for n in types if n not in ('outliers', 'only'))})"
                )
            t = types[k]
            if t in (bool, "bool"):
                kw[k] = v.strip().lower() in ("1", "true", "yes")
            elif t in (int, "int"):
                kw[k] = int(v)
            elif t in (float, "float"):
                kw[k] = float(v)
            else:
                kw[k] = v.strip()
        return cls(**kw)


@dataclass
class Call:
    idx: object  # [n] int64 stream positions of the call's segments, back to back (GPU)
    cu: object  # [segs + 1] int64 cumulative lengths (GPU)
    cu_cpu: object  # the same on the host
    parent: object | None  # levels: [segs] rows of the previous level's kept states, or None (roots: zero state)
    keep: bool  # levels: output the final states (the segments with children)
    kind: str = "chunked"  # indexed: chunked | recurrent
    h0_idx: object | None = None  # indexed: [segs] row of the previous level's states (-1 = zero), None at level 0
    ht_idx: object | None = None  # indexed: [segs] row of this level's states to write (-1 = none), None when no segment keeps


@dataclass
class TreePlan:
    conv: object  # [T, 4] int64: each token's path predecessors 3, 2, 1 back and itself; T = a zero row (none)
    levels: list  # per depth: [Call]
    n_segments: int
    path: str = "levels"
    n_keep: list | None = None  # indexed: per level, the number of kept states
    # compact rows (QwenOptions.compact): the model runs on `rows` rows, the stream's real tokens first (sidx[r] = the stream position of
    # row r, GPU int64) then zeros; conv and the calls' idx index these rows (none = rows). None: the stream layout itself
    rows: int | None = None
    sidx: object | None = None
    # extraction (scorer, qwen_decode): the segment of each row (-1: none), each segment's level and its slot in that level's
    # state tensor (-1: its final state is not kept), and the batch's capture request (Scorer._execute): {"ranges": [(row, n,
    # key)] K/V rows of the attention layers, "tok": [(row, n, key)] q k v g beta rows of the recurrent layers, "xin": [(row,
    # key)] the layer input at a row's conv taps, "state": [(row, key)] the final state of a row's segment, "out": the results}
    seg_of_row: object | None = None
    seg_level: object | None = None
    seg_slot: object | None = None
    cap: dict | None = None


def tree_plan(
    prev: np.ndarray,
    real: np.ndarray,
    device,
    max_init: int = 96,
    path: str = "indexed",
    short: int = 64,
    rows: int | None = None,
    keep=None,
    call_tokens: int = 8192,
) -> TreePlan:
    """prev[i] = the token before i on its path (-1: none), real[i] = not a pad -> TreePlan. A segment starts at a token whose
    predecessor is not the previous stream token, or is a fork (several children); it runs to the next start or pad. levels:
    calls with initial states hold at most `max_init` segments (each state is 48 x 128 x 128 fp32 = 3 MiB). rows: the plan
    indexes compact rows (the real tokens in stream order, then zero rows up to `rows`) instead of stream positions. keep: stream
    positions whose segment must leave its final state even without children (extraction: the prefix state seeds the decoder).
    path, short, call_tokens: QwenOptions gdn_path, gdn_short, call_tokens."""
    import torch

    T = len(prev)
    real_idx = np.flatnonzero(real)
    c_of_s = None  # stream position -> compact row; T (the conv's "none") -> rows
    if rows is not None:
        if len(real_idx) > rows:
            raise ValueError(f"tree_plan: {len(real_idx)} real tokens do not fit {rows} compact rows")
        c_of_s = np.full(T + 1, rows, np.int64)
        c_of_s[real_idx] = np.arange(len(real_idx))
    nchild = np.bincount(prev[prev >= 0], minlength=T)
    ar = np.arange(T)
    start = real & ((prev != ar - 1) | (prev < 0) | (np.where(prev >= 0, nchild[np.maximum(prev, 0)], 0) > 1))
    # segments: a start, then real tokens that are not starts
    seg = np.full(T, -1, np.int64)
    sid = np.cumsum(start) - 1
    seg[real] = sid[real]
    starts = np.flatnonzero(start)
    n = len(starts)
    lens = np.bincount(seg[real], minlength=n)
    psg = np.where(prev[starts] >= 0, seg[np.maximum(prev[starts], 0)], -1)  # parent segment
    depth = np.zeros(n, np.int64)
    for s in range(n):  # parents come first in the stream (pre-order)
        if psg[s] >= 0:
            depth[s] = depth[psg[s]] + 1
    has_kids = np.zeros(n, bool)
    has_kids[psg[psg >= 0]] = True
    if keep is not None:
        for s in keep:
            if real[s]:
                has_kids[seg[s]] = True

    def gpu(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)

    def tokens(part):
        t = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in part])
        return t if c_of_s is None else c_of_s[t]

    levels = []
    slot = np.full(n, -1, np.int64)  # row of a kept segment in its level's state tensor
    n_keep = []
    for d in range(int(depth.max()) + 1 if n else 0):
        calls = []
        if path == "indexed":
            segs = np.flatnonzero(depth == d)
            kept = segs[has_kids[segs]]
            slot[kept] = np.arange(len(kept))
            n_keep.append(len(kept))
            for kind, whole in (("chunked", segs[lens[segs] > short]), ("recurrent", segs[lens[segs] <= short])):
                if not len(whole):
                    continue
                parts = [whole]
                if kind == "chunked" and lens[whole].sum() > call_tokens:  # the state snapshots of a call (memory)
                    parts, run, acc = [], [], 0
                    for sg in whole:
                        if run and acc + lens[sg] > call_tokens:
                            parts.append(np.array(run))
                            run, acc = [], 0
                        run.append(sg)
                        acc += lens[sg]
                    parts.append(np.array(run))
                for part in parts:
                    cu = np.concatenate([[0], np.cumsum(lens[part])])
                    cu_t = torch.from_numpy(cu)
                    h0 = None if d == 0 else gpu(np.where(psg[part] >= 0, slot[np.maximum(psg[part], 0)], -1))
                    ht = gpu(np.where(has_kids[part], slot[part], -1)) if has_kids[part].any() else None
                    calls.append(Call(gpu(tokens(part)), cu_t.to(device, non_blocking=True), cu_t, None, False, kind, h0, ht))
            levels.append(calls)
            continue
        for keep in (True, False):
            segs = np.flatnonzero((depth == d) & (has_kids == keep))
            if not len(segs):
                continue
            if keep:
                slot[segs] = np.arange(len(segs))
            step = len(segs) if d == 0 else max_init
            for a in range(0, len(segs), step):
                part = segs[a : a + step]
                cu = np.concatenate([[0], np.cumsum(lens[part])])
                cu_t = torch.from_numpy(cu)
                calls.append(
                    Call(gpu(tokens(part)), cu_t.to(device, non_blocking=True), cu_t, None if d == 0 else gpu(slot[psg[part]]), keep)
                )
        levels.append(calls)
    conv = np.empty((T, 4), np.int64)
    conv[:, 3] = ar
    p = np.where(real, prev, -1)
    for j in (2, 1, 0):
        conv[:, j] = np.where(p >= 0, p, T)
        p = np.where(p >= 0, prev[np.maximum(p, 0)], -1)
    sidx = None
    if c_of_s is not None:  # the real rows' taps in compact rows; the zero rows have none
        cc = np.full((rows, 4), rows, np.int64)
        cc[: len(real_idx)] = c_of_s[conv[real_idx]]
        conv = cc
        sidx = gpu(real_idx)
    seg_rows = seg
    if c_of_s is not None:
        seg_rows = np.full(rows, -1, np.int64)
        seg_rows[: len(real_idx)] = seg[real_idx]
    return TreePlan(gpu(conv), levels, n, path, n_keep if path == "indexed" else None, rows, sidx, seg_rows, depth, slot)


# ---------------------------------------------------------------------------------------------------------------- the model
def build_model(cfg, model_dir: str, device, options: QwenOptions | None = None):
    """-> (model, text config): QwenTextModel with the checkpoint's int4 linears, bf16 everything else; with cfg.media the vision
    tower (qwen_vision.QwenVision) as model._media. options: the stack's run-time choices (the defaults = production); the linears'
    mode is cfg.quant when set, else options.mode"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors import safe_open
    from torch.nn.attention.flex_attention import flex_attention
    from transformers import AutoConfig

    from spinf_serve.models.base import compile_options
    from spinf_serve.models.qwen35 import qwen_vision

    device = torch.device(device)
    hf = AutoConfig.from_pretrained(model_dir)
    tcfg = hf.text_config
    qc = getattr(hf, "quantization_config", None) or {}
    qc = qc if isinstance(qc, dict) else qc.to_dict()
    if qc.get("quant_method") != "gptq" or qc.get("bits") != 4 or not qc.get("sym") or qc.get("desc_act"):
        raise RuntimeError(f"qwen: expected a GPTQ int4 symmetric checkpoint without act-order, got {qc}")
    G = int(qc["group_size"])
    qopts = options or QwenOptions()
    if cfg.quant and cfg.quant not in ("f16", "i8", "fp8", "bf16"):
        raise RuntimeError(f"qwen: quant {cfg.quant!r} not available for int4 checkpoints (f16, i8, fp8, bf16)")
    mode = cfg.quant or qopts.mode
    if mode == "fp8" and torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("qwen: quant fp8 needs fp8 tensor cores (compute capability 8.9+: Ada, Hopper, Blackwell)")
    from spinf_serve.quant import i8gemm

    i8gemm.linear = torch.compiler.disable(i8gemm.linear)  # three Triton launches (quantize, dequant, gemm): outside the graphs
    i8gemm.split_linear = torch.compiler.disable(i8gemm.split_linear)
    i8gemm.split_glu_q = torch.compiler.disable(i8gemm.split_glu_q)
    i8gemm.linear_q = torch.compiler.disable(i8gemm.linear_q)
    i8gemm.split_gdn_in = torch.compiler.disable(i8gemm.split_gdn_in)
    outliers = qopts.outliers  # i8: the split scheme's channels, set by a research script or the revision's table
    if mode == "i8" and outliers is None:
        table = load_outliers_table(cfg.model_id, model_dir)
        if table is None:  # without it i8 would silently run another scheme (per-token scales alone: WinoGrande -3.7 points)
            raise RuntimeError(
                f"qwen: quant i8 needs the checkpoint's outlier table (spinf_outliers.json in {model_dir}, or "
                f"{outliers_table_path(cfg.model_id, model_dir)}); export it with quantizations/export_outliers.py, or set SPINF_QUANT=f16"
            )
        outliers = table["outliers"]
    opts = compile_options(cfg)

    @torch.compile(dynamic=True, options=opts)
    def dq_bf16(q, s):  # q [out, in/2] uint8 (low nibble = even input), s [out, in/G] -> [out, in] bf16
        o = q.shape[0]
        w = torch.stack([(q & 15), (q >> 4)], -1).view(o, -1, G).float() - 8.0
        return (w * s.float()[:, :, None]).view(o, -1).to(torch.bfloat16)

    @torch.compile(dynamic=True, options=opts)
    def dq_fp8(q, s, inv):  # inv = 1 / the per-tensor weight scale -> [out, in] e4m3
        o = q.shape[0]
        w = torch.stack([(q & 15), (q >> 4)], -1).view(o, -1, G).float() - 8.0
        return (w * (s.float() * inv)[:, :, None]).view(o, -1).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    @torch.compile(dynamic=True, options=opts)
    def dq_f16(q, s):  # -> [out, in] fp16 (q - 8) * s, rounded once
        o = q.shape[0]
        w = torch.stack([(q & 15), (q >> 4)], -1).view(o, -1, G).float() - 8.0
        return (w * s.float()[:, :, None]).view(o, -1).to(torch.float16)

    @torch.compile(dynamic=True, options=opts)
    def quant_act(x):
        xf = x.reshape(-1, x.shape[-1]).float()
        s = xf.abs().amax().clamp_(min=1e-12) / FP8_MAX
        return (xf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s

    @torch.compile(dynamic=True, options=opts)
    def dq_w8i(q, s):  # research: the int4 weight re-rounded to int8 per output channel (symmetric, absmax / 127) -> bf16
        o = q.shape[0]
        w = torch.stack([(q & 15), (q >> 4)], -1).view(o, -1, G).float() - 8.0
        w = (w * s.float()[:, :, None]).view(o, -1)
        sc = w.abs().amax(1, keepdim=True).clamp_(min=1e-12) / 127.0
        return (torch.round(w / sc) * sc).to(torch.bfloat16)

    @torch.compile(dynamic=True, options=opts)
    def dq_w8s(q, s, sc):  # research: the int4 weight times the smoothing factors sc [in], re-rounded to int8 per output channel
        o = q.shape[0]
        w = torch.stack([(q & 15), (q >> 4)], -1).view(o, -1, G).float() - 8.0
        w = (w * s.float()[:, :, None]).view(o, -1) * sc[None, :]
        st = w.abs().amax(1, keepdim=True).clamp_(min=1e-12) / 127.0
        return (torch.round(w / st) * st).to(torch.bfloat16)

    @torch.compile(dynamic=True, options=opts)
    def a8i(x):  # research: int8 per token (symmetric, absmax / 127), back to bf16
        xf = x.reshape(-1, x.shape[-1]).float()
        sc = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-12) / 127.0
        return (torch.round(xf / sc) * sc).to(torch.bfloat16)

    @torch.compile(dynamic=True, options=opts)
    def a8blk(x, blk: int):  # research: int8 per (token, block of blk input channels): a scale per K block in the GEMM
        xf = x.reshape(-1, x.shape[-1] // blk, blk).float()
        sc = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-12) / 127.0
        return (torch.round(xf / sc) * sc).reshape(-1, x.shape[-1]).to(torch.bfloat16)

    class Int4Linear(nn.Module):
        """GPTQ int4 weights (several projections of the same input concatenated along the output: one GEMM). swap: run the
        product transposed (W x^T, then a transposed view): cuBLAS' fp8 heuristics pick a slow tile for some shapes (K 5120 / 6144
        with N <= 12288 on a 4090: ~220 vs ~340 TFLOPS), set by tune_orientation"""

        def __init__(self, qweight, scales):
            super().__init__()
            self.swap = False
            sh = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
            q = ((qweight[:, None, :] >> sh[None, :, None]) & 15).reshape(-1, qweight.shape[1]).t()  # [out, in] 0..15
            self.out_features, self.in_features = q.shape
            self.register_buffer("q", (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous())
            s = scales.t().contiguous()  # [out, in/G] fp16, as stored
            self.register_buffer("s", s)
            ws = float(s.float().abs().amax()) * 8.0 / FP8_MAX  # |q - 8| <= 8
            self.register_buffer("ws", torch.tensor(ws, dtype=torch.float32, device=s.device))
            self.register_buffer("inv", torch.tensor(1.0 / ws, dtype=torch.float32, device=s.device))
            self.extra = None
            self.name = ""  # the projection's name (in_proj_qkvz, out_proj, qkv_proj, o_proj, gate_up_proj, down_proj)
            self.out16 = False  # f16 mode: return the GEMM's fp16 output as is (else bf16, the model dtype)
            self.split = False  # i8: the cuBLASLt scheme with the outlier channels exact (set_split)
            if mode == "i8":  # the int8 weight scale per output channel (int8 rows are made per call: i8gemm.dequant)
                if G != 32:
                    raise RuntimeError(f"qwen: mode i8 is written for group 32 (got {G})")
                amax = dq_bf16(self.q, self.s).float().abs().amax(1).clamp_(min=1e-12)
                self.register_buffer("sw", amax / 127.0)
                self.register_buffer("inv_sw", 127.0 / amax)

        def set_split(self, out_idx):
            """i8, the split scheme (i8gemm.split_linear): the input channels out_idx (the calibrated outliers, outliers_table /
            QwenOptions.outliers) stay exact through their dequantized columns in fp16, the rest run int8 with a per-token
            scale; the weights' int8 scale is over the kept columns"""
            out_idx = torch.as_tensor(out_idx, dtype=torch.int64, device=self.q.device).sort().values
            keep = torch.ones(self.in_features, dtype=torch.uint8, device=out_idx.device)
            keep[out_idx] = 0
            w = dq_bf16(self.q, self.s).float()
            amax = (w.abs() * keep[None, :]).amax(1).clamp_(min=1e-12)
            self.sw, self.inv_sw = amax / 127.0, 127.0 / amax
            self.register_buffer("w_out", w[:, out_idx].to(torch.float16).contiguous())
            self.register_buffer("keep", keep)
            self.register_buffer("out_idx", out_idx)
            self.split = True

        def set_extra(self, w):
            """bf16 rows [E, in] appended to the output (the Gated DeltaNet's b / a projections): in f16 / bf16 / i8 they ride in
            the same GEMM (a separate N = 96 GEMM changes kernel with the stream length: not batch-invariant)"""
            del self.extra
            self.register_buffer("extra", w.to(torch.bfloat16).contiguous())
            self.register_buffer("extra16", w.to(torch.float16).contiguous())
            if mode == "i8":
                wf = w.float()
                amax = (wf.abs() * (self.keep[None, :].float() if self.split else 1.0)).amax(1).clamp_(min=1e-12)
                self.register_buffer("extra8", torch.round(wf * (127.0 / amax)[:, None]).clamp_(-127.0, 127.0).to(torch.int8).contiguous())
                self.register_buffer("sw_ext", torch.cat([self.sw, amax / 127.0]))
                if self.split:
                    self.register_buffer("w_out_ext", torch.cat([self.w_out, wf[:, self.out_idx].to(torch.float16)]).contiguous())

        def forward(self, x):
            x2 = x.reshape(-1, x.shape[-1])
            ex = self.extra is not None
            if mode == "i8":  # int8 tensor cores: weights per output channel, activations per token and 128-block
                if self.split:  # (or per token with the outlier channels exact, cuBLASLt: i8gemm.split_linear)
                    ext = (self.extra8, self.sw_ext, self.w_out_ext) if ex else ()
                    y = i8gemm.split_linear(x2, self.q, self.s, self.inv_sw, self.sw, self.keep, self.out_idx, self.w_out, *ext)
                elif ex:
                    y = i8gemm.linear(x2, self.q, self.s, self.inv_sw, self.sw, self.extra8, self.sw_ext)
                else:
                    y = i8gemm.linear(x2, self.q, self.s, self.inv_sw, self.sw)
                return y.reshape(*x.shape[:-1], y.shape[-1])
            if mode == "f16":  # fp16 operands, fp16 accumulation in cuBLAS (allow_fp16_accumulation)
                xh, w = x2.to(torch.float16), dq_f16(self.q, self.s)
                if ex:
                    w = torch.cat([w, self.extra16])
                y = torch.mm(w, xh.t()).t() if self.swap else torch.mm(xh, w.t())
                if not self.out16:  # consumers that take fp16 skip a 1.1 GB cast pass per 16k rows (the GDN's q k v z b a)
                    y = y.to(torch.bfloat16)
                return y.reshape(*x.shape[:-1], y.shape[-1])
            if mode == "bf16":
                w = dq_bf16(self.q, self.s)
                y = x2 @ (torch.cat([w, self.extra]) if ex else w).t()
                return y.reshape(*x.shape[:-1], y.shape[-1])
            if mode == "fp8":
                x8, xs = quant_act(x2)
                w8 = dq_fp8(self.q, self.s, self.inv)
                if self.swap:
                    y = torch._scaled_mm(w8, x8.t(), scale_a=self.ws, scale_b=xs, out_dtype=torch.bfloat16).t()
                else:
                    y = torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)
            elif mode == "sim_w8":  # research: fp8 weights, bf16 activations (the weights' share of the error)
                y = x2 @ (dq_fp8(self.q, self.s, self.inv).to(torch.bfloat16) * self.ws.to(torch.bfloat16)).t()
            elif mode == "sim_a8":  # research: exact weights, fp8 activations (the activations' share)
                x8, xs = quant_act(x2)
                y = (x8.to(torch.bfloat16) * xs.to(torch.bfloat16)) @ dq_bf16(self.q, self.s).t()
            elif mode in (
                "sim_w8i", "sim_a8i", "sim_w8a8i", "sim_a8b", "sim_w8a8b", "sim_w8a8o", "sim_w8a8ob", "sim_w8a8s", "sim_w8a8os"
            ):  # fmt: skip
                m = mode  # research, int8 numerics: weights per output channel; activations per token (i), per
                only = qopts.only  # token and K block (b, QwenOptions.blk), per token with the outlier channels
                # exact (o: QwenOptions.outliers[full name], n_out of them), or o for the norm-fed linears and b for the rest (ob);
                # s: SmoothQuant (x / s_c per token, W s_c per output channel, self.smooth), os: o for the norm-fed, s for the rest
                smooth = m.endswith("s") and (m == "sim_w8a8s" or self.name not in SPLIT_LINEARS)
                if m == "sim_w8i" or (only and self.name not in only):
                    xa = x2
                elif smooth:
                    xa = a8i(x2 * (1.0 / self.smooth).to(x2.dtype))
                elif m.endswith("o") or (m.endswith(("ob", "os")) and self.name in SPLIT_LINEARS):
                    keep = self.keep_idx
                    xa = x2.clone()
                    xa[:, keep] = a8i(x2[:, keep])
                else:
                    xa = a8blk(x2, qopts.blk) if m.endswith("b") else a8i(x2)
                if m in ("sim_a8i", "sim_a8b"):
                    w = dq_bf16(self.q, self.s)
                elif smooth:
                    w = dq_w8s(self.q, self.s, self.smooth)
                else:
                    w = dq_w8i(self.q, self.s)
                y = xa @ w.t()
            else:
                raise ValueError(f"qwen: linear mode {mode!r}")
            if ex:
                y = torch.cat([y, x2 @ self.extra.t()], -1)
            return y.reshape(*x.shape[:-1], y.shape[-1])

    class RMSNorm(nn.Module):  # Qwen3_5RMSNorm: (1 + w), in fp32
        def __init__(self, n, eps):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(n), requires_grad=False)
            self.eps = eps

        def forward(self, x):
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight.float())).type_as(x)

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    @torch.compiler.disable
    def gdn_tree(q, k, v, g, beta, plan: TreePlan, mod=None):
        """q, k [T, Hk, Dk], v [T, Hv, Dv], g, beta [T, Hv] -> [T, Hv, Dv]: every segment from its parent's final state. mod: the
        layer's mixer (its .li keys the capture of extraction rows, plan.cap: the templates' q k v g beta and the prefixes' final
        states). The per-layer index and the capture dict are read here, outside the compiled graphs: a guard on either would
        recompile the layer forward per layer (the compile cache is per code object) or per batch."""
        cap = plan.cap
        li = mod.li if mod is not None else -1
        if cap is not None:
            if plan.path != "indexed" and cap["state"]:
                raise RuntimeError("qwen: extraction needs the indexed recurrent path (QwenOptions gdn_path)")
            for r0, n, key in cap["tok"]:
                cap["out"]["gdn"].setdefault(key, {})[li] = tuple(t[r0 : r0 + n].clone() for t in (q, k, v, g, beta))
        if plan.path == "indexed":
            return gdn_indexed(q, k, v, g, beta, plan, li)
        o = torch.zeros_like(v)
        kept = None
        for calls in plan.levels:
            new = []
            for c in calls:
                a = [x.index_select(0, c.idx)[None] for x in (q, k, v, g, beta)]
                init = None if c.parent is None else kept.index_select(0, c.parent)
                out, st = chunk_gated_delta_rule(
                    *a,
                    initial_state=init,
                    output_final_state=c.keep,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=c.cu,
                    cu_seqlens_cpu=c.cu_cpu,
                )
                o.index_copy_(0, c.idx, out[0])
                if c.keep:
                    new.append(st)
            kept = new[0] if len(new) == 1 else (torch.cat(new) if new else None)
        return o

    def gdn_indexed(q, k, v, g, beta, plan: TreePlan, li: int = -1):
        from spinf_serve.models.qwen35 import gdn

        # q, k arrive l2-normalised (GatedDeltaNet.forward: fla's l2norm kernel is autotuned on the stream length, so its
        # rows would change with the packing)
        scale = q.shape[-1] ** -0.5
        o = torch.zeros_like(v)
        prev_states = None
        want = []  # extraction: (level, slot, key) of the segments whose final state the decoder needs
        if plan.cap is not None:
            for row, key in plan.cap["state"]:
                sg = int(plan.seg_of_row[row])
                lvl, slot = int(plan.seg_level[sg]), int(plan.seg_slot[sg])
                if sg < 0 or slot < 0:
                    raise RuntimeError(f"qwen: no kept state for row {row} (segment {sg}, slot {slot})")
                want.append((lvl, slot, key))
        for d, (calls, nk) in enumerate(zip(plan.levels, plan.n_keep, strict=True)):
            states = q.new_empty((nk, v.shape[1], q.shape[-1], v.shape[-1]), dtype=torch.float32) if nk else None
            for c in calls:
                kw = dict(
                    h0=prev_states if c.h0_idx is not None else None,
                    h0_idx=c.h0_idx,
                    ht=states if c.ht_idx is not None else None,
                    ht_idx=c.ht_idx,
                )
                if c.kind == "recurrent" and qopts.rec_idx:  # the kernel reads / writes the stream rows itself
                    gdn.recurrent(
                        q[None], k[None], v[None], g[None], beta[None], scale, c.cu, len(c.cu_cpu) - 1, idx=c.idx, o=o[None], **kw
                    )
                    continue
                a = [x.index_select(0, c.idx)[None] for x in (q, k, v, g, beta)]
                if c.kind == "chunked":
                    out = gdn.chunked(*a, scale, c.cu, c.cu_cpu, **kw)
                else:
                    out = gdn.recurrent(*a, scale, c.cu, len(c.cu_cpu) - 1, **kw)
                o.index_copy_(0, c.idx, out[0])
            for lvl, slot, key in want:
                if lvl == d:
                    plan.cap["out"]["state"].setdefault(key, {})[li] = states[slot].clone()
            prev_states = states
        return o

    @torch.compiler.disable
    def capture_xin(plan: TreePlan, mod, x):
        """extraction: the recurrent layer's input x [rows, hidden] at the 4 conv taps (3 path predecessors and itself) of each
        requested row -> plan.cap["out"]["xin"][key][layer] [4, hidden] (the decoder recomputes their pre-conv projections). A
        no-op without a capture; called unconditionally so the compiled graph never reads plan.cap"""
        cap = plan.cap
        if cap is not None and cap["xin"]:
            li = mod.li
            ext = torch.cat([x, x.new_zeros(1, x.shape[1])])  # the conv's "none" index is a zero row
            for row, key in cap["xin"]:
                cap["out"]["xin"].setdefault(key, {})[li] = ext[plan.conv[row]].clone()

    @torch.compiler.disable
    def tree_attention(q, k, v, mask, scale, sidx=None, plan=None, mod=None):
        """q, k, v [1, heads, rows, d]; sidx (compact rows): the stream position of each real row: the real rows are scattered into
        the stream layout the mask describes (pads: zero rows, attending to themselves), the output gathered back. plan.cap: keep
        the K / V rows of the extraction ranges for the decoder, as [1, kv heads, n, d], under the layer's index mod.li"""
        cap = plan.cap if plan is not None else None
        if cap is not None:
            for r0, n, key in cap["ranges"]:
                cap["out"]["kv"].setdefault(key, {})[mod.li] = (k[:, :, r0 : r0 + n].clone(), v[:, :, r0 : r0 + n].clone())
        if sidx is None:
            return flex_attn(q, k, v, block_mask=mask, scale=scale, enable_gqa=True, kernel_options=FLEX_TILE)
        T, R = mask.seq_lengths[0], sidx.shape[0]

        def up(t):
            return t.new_zeros(t.shape[0], t.shape[1], T, t.shape[3]).index_copy_(2, sidx, t[:, :, :R])

        o = flex_attn(up(q), up(k), up(v), block_mask=mask, scale=scale, enable_gqa=True, kernel_options=FLEX_TILE)
        out = q.new_zeros(q.shape)
        out[:, :, :R] = o.index_select(2, sidx)
        return out

    flex_attn = torch.compile(flex_attention, dynamic=True, options=opts)
    FLEX_TILE = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2, "num_warps": 4}  # head_dim 256 on sm89 shared memory

    class GatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hk, self.hv = tcfg.linear_num_key_heads, tcfg.linear_num_value_heads
            self.dk, self.dv = tcfg.linear_key_head_dim, tcfg.linear_value_head_dim
            self.conv_dim = 2 * self.hk * self.dk + self.hv * self.dv
            self.conv_w = nn.Parameter(torch.empty(self.conv_dim, 4), requires_grad=False)
            self.A_log = nn.Parameter(torch.empty(self.hv), requires_grad=False)
            self.dt_bias = nn.Parameter(torch.empty(self.hv), requires_grad=False)
            self.norm_w = nn.Parameter(torch.empty(self.dv), requires_grad=False)
            self.eps = tcfg.rms_norm_eps

        def forward(self, x, plan: TreePlan):
            T = x.shape[0]
            kd = self.hk * self.dk
            ip = self.in_proj_qkvz
            capture_xin(plan, self, x)
            fused = mode == "i8" and ip.split and qopts.fuse_gdn
            if fused:  # the conv, silu and (indexed) q / k norm below run in the GEMM's epilogue
                y = i8gemm.split_gdn_in(
                    x.reshape(-1, x.shape[-1]), ip.q, ip.s, ip.inv_sw, ip.sw_ext, ip.keep, ip.out_idx, ip.w_out_ext, ip.extra8,
                    plan.conv, self.conv_wf, self.conv_dim, 2 * kd if plan.path == "indexed" else 0,
                )  # fmt: skip
            else:
                y = ip(x)  # [T, conv_dim + value dim + 2 hv]: q k v z, then b and a (Int4Linear.extra)
            vd = self.hv * self.dv
            qkv, z = y[:, : self.conv_dim], y[:, self.conv_dim : self.conv_dim + vd]
            b, a = y[:, self.conv_dim + vd : self.conv_dim + vd + self.hv], y[:, self.conv_dim + vd + self.hv :]
            if not fused:
                # causal conv along each token's path: taps 0..3 = 3 back .. itself; index T = a zero row
                ext = torch.cat([qkv, qkv.new_zeros(1, self.conv_dim)])
                y = (ext[plan.conv].float() * self.conv_w.float()[None].transpose(1, 2)).sum(1)
                qkv = F.silu(y).to(x.dtype)
            q = qkv[:, :kd].reshape(T, self.hk, self.dk)
            k = qkv[:, kd : 2 * kd].reshape(T, self.hk, self.dk)
            v = qkv[:, 2 * kd :].reshape(T, self.hv, self.dv)
            beta = torch.sigmoid(b.float()).to(x.dtype)
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
            if plan.path == "indexed" and not fused:  # x / sqrt(sum x^2 + 1e-6) per head row, as fla's l2norm, compiled with the layer
                q = (q.float() * torch.rsqrt(q.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
                k = (k.float() * torch.rsqrt(k.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
            o = gdn_tree(q, k, v, g, beta, plan, self)  # [T, hv, dv]
            of = o.float()
            of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
            of = (self.norm_w * of.to(x.dtype)).float() * F.silu(z.reshape(T, self.hv, self.dv).float())
            return self.out_proj(of.to(x.dtype).reshape(T, -1))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.hkv, self.d = tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim
            self.q_norm = RMSNorm(self.d, tcfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.d, tcfg.rms_norm_eps)
            self.scale = self.d**-0.5

        def forward(self, x, cos, sin, mask, plan):
            T = x.shape[0]
            nq, nk = self.h * 2 * self.d, self.hkv * self.d
            y = self.qkv_proj(x)
            qg = y[:, :nq].reshape(T, self.h, 2 * self.d)
            q, gate = qg[..., : self.d], qg[..., self.d :].reshape(T, -1)
            q = self.q_norm(q)
            k = self.k_norm(y[:, nq : nq + nk].reshape(T, self.hkv, self.d))
            v = y[:, nq + nk :].reshape(T, self.hkv, self.d)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            o = tree_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], mask, self.scale, plan.sidx, plan, self
            )
            o = o[0].transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate)
            return self.o_proj(o)

    def rope(x, cos, sin):  # x [T, H, D]; cos, sin [T, R] on the first R dims (rotate-half within them)
        R = cos.shape[-1]
        xr, xp = x[..., :R], x[..., R:]
        x1, x2 = xr[..., : R // 2], xr[..., R // 2 :]
        rot = torch.cat([-x2, x1], -1)
        return torch.cat([xr * cos[:, None] + rot * sin[:, None], xp], -1)

    class MLP(nn.Module):
        def forward(self, x):
            gp, dp = self.gate_up_proj, self.down_proj
            if mode == "i8" and gp.split and qopts.fuse_glu:  # silu x mul and the quantizer in the epilogue
                x2 = x.reshape(-1, x.shape[-1])
                h8, sh = i8gemm.split_glu_q(x2, gp.q, gp.s, gp.inv_sw, gp.sw, gp.keep, gp.out_idx, gp.w_out)
                y = i8gemm.linear_q(h8, sh, dp.q, dp.s, dp.inv_sw, dp.sw)
                return y.reshape(*x.shape[:-1], y.shape[-1])
            gu = gp(x)
            n = gu.shape[-1] // 2
            return dp(F.silu(gu[:, :n]) * gu[:, n:])

    class Layer(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.input_layernorm = RMSNorm(tcfg.hidden_size, tcfg.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(tcfg.hidden_size, tcfg.rms_norm_eps)
            self.mlp = MLP()
            if kind == "linear_attention":
                self.linear_attn = GatedDeltaNet()
            else:
                self.self_attn = Attention()

        def forward(self, x, cos, sin, mask, plan):
            h = self.input_layernorm(x)
            h = self.linear_attn(h, plan) if self.kind == "linear_attention" else self.self_attn(h, cos, sin, mask, plan)
            x = x + h
            return x + self.mlp(self.post_attention_layernorm(x))

    class QwenTextModel(nn.Module):
        hybrid = True

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(t) for t in tcfg.layer_types[: cfg.layers or None]])
            self.norm = RMSNorm(tcfg.hidden_size, tcfg.rms_norm_eps)
            rp = tcfg.rope_parameters
            R = int(tcfg.head_dim * rp.get("partial_rotary_factor", 1.0))
            self.register_buffer("inv_freq", 1.0 / (rp["rope_theta"] ** (torch.arange(0, R, 2, dtype=torch.float32) / R)), persistent=False)
            self.register_buffer("mrope_axis", torch.zeros(R // 2, dtype=torch.int64), persistent=False)  # qwen_vision.mrope_axis

        def embed(self, ids_host, rows=None):
            """token ids (host int64 [n]) -> [rows or n, hidden] bf16 on the card: rows gathered from the host table (pinned),
            zero rows after the n tokens"""
            n = len(ids_host)
            rows = rows or n
            if self.embed_host is None:
                x = self.embed_tokens(torch.from_numpy(ids_host).to(device))
                return torch.cat([x, x.new_zeros(rows - n, x.shape[1])]) if rows > n else x
            if self._emb_buf is None or self._emb_buf.shape[0] < rows:
                self._emb_buf = torch.empty((max(rows, 16384), self.embed_host.shape[1]), dtype=torch.bfloat16).pin_memory()
            buf = self._emb_buf[:rows]
            torch.index_select(self.embed_host, 0, torch.from_numpy(ids_host), out=buf[:n])
            if rows > n:
                buf[n:].zero_()
            return buf.to(device, non_blocking=True)

        def forward(self, ids, pos, mask, plan: TreePlan, embeds=None):
            """ids [T], pos [T] (or [3, T]: t, h, w positions of a batch with images, qwen_vision.mrope_positions) -> final hidden
            states [T, hidden] (after the final norm)"""
            x = self.embed_tokens(ids) if embeds is None else embeds
            # [3, T] (mrope): each rotary frequency reads its axis; [T] (text): the same position on the three axes
            pf = pos.t().float()[:, self.mrope_axis] if pos.dim() == 2 else pos.float()[:, None]
            f = pf * self.inv_freq[None]
            e = torch.cat([f, f], -1)
            cos, sin = e.cos().to(x.dtype), e.sin().to(x.dtype)
            for layer in self.layers:
                x = layer(x, cos, sin, mask, plan)
            return self.norm(x)

    with torch.device("meta"):
        model = QwenTextModel()
    model.embed_host, model._emb_buf = None, None
    for i, layer in enumerate(model.layers):  # the mixers know their layer index (extraction captures are keyed by it)
        (layer.linear_attn if layer.kind == "linear_attention" else layer.self_attn).li = i
    rp = tcfg.rope_parameters
    R = int(tcfg.head_dim * rp.get("partial_rotary_factor", 1.0))
    model.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, R, 2, dtype=torch.float32, device=device) / R))
    model.mrope_axis = torch.from_numpy(qwen_vision.mrope_axis(rp, R // 2)).to(device)

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    prefix = "model.language_model."
    keys = sorted(k for k in idx if k.startswith(prefix))
    n_layers = len(model.layers)
    t0 = time.perf_counter()
    groups: dict[str, dict] = {}  # int4 linears: module path -> {qweight, scales}
    pending_ab: dict[int, dict] = {}
    ab_rows: dict[int, object] = {}
    files = {}
    for k in keys:
        name = k[len(prefix) :]
        parts = name.split(".")
        if parts[0] == "layers" and int(parts[1]) >= n_layers:
            continue
        files.setdefault(idx[k], []).append((k, name))
    model._lm_weight = model._readout = None
    readout = load_readout(model_dir)
    if readout is not None:  # a decision model: the readout rows score the answer codes (scorer.readout_logprobs)
        model._readout = readout["weight"].to(device, torch.bfloat16)
        model._readout_ids = torch.tensor(readout["token_ids"], dtype=torch.long, device=device)
        model._readout_temperature = float(readout["temperature"])
    elif "lm_head.weight" not in idx:
        raise RuntimeError("qwen: no lm_head.weight (tied heads are not supported)")
    else:
        files.setdefault(idx["lm_head.weight"], []).append(("lm_head.weight", "lm_head.weight"))

    # projections of one input run as one GEMM (larger N: also the fast fp8 tiles, see Int4Linear)
    FUSE = {
        "in_proj_qkv": ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
        "in_proj_z": ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
        **{p: ("qkv_proj", ("q_proj", "k_proj", "v_proj")) for p in ("q_proj", "k_proj", "v_proj")},
        **{p: ("gate_up_proj", ("gate_proj", "up_proj")) for p in ("gate_proj", "up_proj")},
    }

    def setmod(path, mod):
        parent, _, child = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, mod)

    def param(t):
        return nn.Parameter(t.to(device), requires_grad=False)

    for shard in sorted(files):
        with safe_open(os.path.join(model_dir, shard), "pt", device="cpu") as f:
            for k, name in files[shard]:
                mod_path, _, pname = name.rpartition(".")
                if pname in ("qweight", "scales", "qzeros", "g_idx"):
                    if pname in ("qweight", "scales"):
                        head, _, leaf = mod_path.rpartition(".")
                        fused, parts = FUSE.get(leaf, (leaf, (leaf,)))
                        g = groups.setdefault(head + "." + fused, {})
                        g[(leaf, pname)] = f.get_tensor(k).to(device)
                        if len(g) == 2 * len(parts):
                            qw = torch.cat([g[(p, "qweight")] for p in parts], 1)
                            sc = torch.cat([g[(p, "scales")] for p in parts], 1)
                            del groups[head + "." + fused], g
                            lin = Int4Linear(qw, sc)
                            lin.name = fused
                            if mode == "i8" and outliers and fused in SPLIT_LINEARS:
                                o = outliers[head + "." + fused]  # the table's indices, or act_outliers.py's per-channel absmax
                                lin.set_split(o if isinstance(o, list) else o.topk(qopts.n_out).indices)
                            if mode in (
                                "sim_w8a8o",
                                "sim_w8a8ob",
                                "sim_w8a8os",
                            ):  # per-token channels: all but the top n_out
                                ch = qopts.outliers[head + "." + fused].to(device)
                                keep = torch.ones(lin.in_features, dtype=torch.bool, device=device)
                                keep[ch.topk(qopts.n_out).indices] = False
                                lin.register_buffer("keep_idx", keep.nonzero().squeeze(1))
                            if mode in ("sim_w8a8s", "sim_w8a8os"):  # SmoothQuant factors per input channel:
                                ch = qopts.outliers[head + "." + fused].to(device).float().clamp(min=1e-5)
                                al = qopts.alpha  # s_c = amax_c(x)^a / amax_c(W)^(1 - a); x / s_c, W s_c
                                wmax = dq_bf16(lin.q, lin.s).float().abs().amax(0).clamp_(min=1e-5)
                                lin.register_buffer("smooth", (ch**al / wmax ** (1 - al)).contiguous())
                            setmod(head + "." + fused, lin)
                            del qw, sc, lin
                    elif pname == "qzeros":
                        z = f.get_tensor(k)
                        if not bool((z == 0x77777777).all()):  # v1 format, symmetric: every stored zero is 7 (true zero 8)
                            raise RuntimeError(f"qwen: {k}: unexpected zeros")
                    continue
                t = f.get_tensor(k)
                if name == "lm_head.weight":
                    model._lm_weight = t.to(device, torch.bfloat16)
                    continue
                if name == "embed_tokens.weight":
                    if qopts.embed_host:  # 2.4 GB kept off the card
                        model.embed_host = t.to(torch.bfloat16).contiguous()
                    else:
                        model.embed_tokens = nn.Embedding.from_pretrained(t.to(device, torch.bfloat16), freeze=True)
                    continue
                if ".linear_attn." in name:
                    li = int(name.split(".")[1])
                    la = model.layers[li].linear_attn
                    if pname in ("weight",) and mod_path.endswith(("in_proj_a", "in_proj_b")):
                        pending_ab.setdefault(li, {})[mod_path.rsplit(".", 1)[1]] = t
                        if len(pending_ab[li]) == 2:
                            ab_rows[li] = torch.cat([pending_ab[li]["in_proj_b"], pending_ab[li]["in_proj_a"]]).to(device)
                            del pending_ab[li]
                        continue
                    if mod_path.endswith("conv1d"):
                        la.conv_w = param(t[:, 0, :])
                        continue
                    if mod_path.endswith(".norm"):
                        la.norm_w = param(t)
                        continue
                    if pname in ("A_log", "dt_bias"):
                        setattr(la, pname, param(t))
                        continue
                mod = model.get_submodule(mod_path) if mod_path else model
                setattr(mod, pname, param(t))
    if groups or pending_ab:
        raise RuntimeError(f"qwen: incomplete linears {list(groups)[:3]} {list(pending_ab)[:3]}")
    for li, w in ab_rows.items():
        la = model.layers[li].linear_attn
        la.in_proj_qkvz.set_extra(w)
        la.in_proj_qkvz.out16 = True  # GatedDeltaNet.forward reads q k v z b a in any dtype
        la.register_buffer("conv_wf", la.conv_w.float().contiguous(), persistent=False)  # the fused epilogue's tap weights
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.device.type == "meta":
            raise RuntimeError(f"still on meta: {n}")
    model.eval()
    model._quant = "int4-" + mode
    if any(getattr(m, "split", False) for m in model.modules()):
        model._quant += "-split"
    model._scale_table = None
    model._options = qopts  # what qwen_decode reads (decode_linear, decode_graphs)
    model._media = qwen_vision.QwenVision.load(hf, model_dir, device) if cfg.media else None  # the vision tower (0.92 GB)
    # extraction: the captures held for one decode (K/V, prefix states, template rows) on a 24 GB card next to the 16k forward's
    # 19.9 GiB peak and the decoder's own working set (about the captures again): a smaller budget than Gemma's 4 GiB (QWEN.md §11)
    model._decode_budget_gib = 1.0
    # the decoder's buffers: the recurrent row caches alone take ~1 MB per generated row (48 layers x (q, k, v, g, beta)), so 10
    # slots x 6 sequences x 65 steps are 4 GB; next to the idle 16.7 GiB and the captures, a decode may hold 3 GiB (measured
    # on a 4090: a 15k-token prefix with 10 x 64-token slots ran out of memory in one decode; Scorer._x_flush chunks it)
    model._decoder_budget_gib = 2.5
    # what one 16k forward may capture for its extraction branches (a 16k stream of 8 extraction inputs with 10 slots each would
    # capture ~2.6 GB of template rows, K/V and states while the forward itself peaks 2.5 GiB over idle: out of memory on 24 GB)
    model._capture_budget_gib = 1.0
    model._load_s = time.perf_counter() - t0
    model._n_quant_linears = sum(isinstance(m, Int4Linear) for m in model.modules())
    # batch invariance (with EngineConfig.align_jobs): below M = 5248 rows cuBLAS' fp16 GEMM of the output projections
    # (6144 -> 5120) changes kernel and a row's result with it; from 5248 up every Qwen3.8 shape gives bit-identical rows
    # (4090, torch 2.14 cu130, research/qwen/gemm_rows_sweep.py): streams are padded to at least this
    model._min_stream = MIN_STREAM.get(mode, 0)
    if mode == "f16":
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
    if mode == "i8":  # the int8 GEMM's tile configuration per shape (its rows do not depend on it: i8gemm)
        lins = [m for m in model.modules() if isinstance(m, Int4Linear)]
        shapes = [(m.sw_ext.shape[0] if m.extra is not None else m.out_features, m.in_features) for m in lins]
        model._orientation = i8gemm.tune(shapes, cfg.token_budget, device)
    else:
        tune_orientation(model, Int4Linear, cfg.token_budget, device, mode)
    torch.cuda.synchronize(device)
    return model, tcfg


def tune_orientation(model, cls, M: int, device, mode: str = "fp8") -> dict:
    """per distinct linear shape: time the GEMM of `mode` (fp8 or f16) both ways at M rows (random data) and set .swap where
    the transposed product is faster by > 3 % -> {(out, in): (ms, ms swapped)}"""
    import torch

    one = torch.tensor(1.0, device=device)
    dt = torch.float16 if mode == "f16" else torch.float8_e4m3fn

    def mm(a, b):
        if mode == "f16":
            return torch.mm(a, b)
        return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    res = {}
    lins = [m for m in model.modules() if isinstance(m, cls)]
    for shape in sorted({(m.out_features, m.in_features) for m in lins}):
        N, K = shape
        x = torch.randn(M, K, device=device).to(dt)
        w = torch.randn(N, K, device=device).to(dt)
        t = []
        for fn in (lambda x=x, w=w: mm(x, w.t()), lambda x=x, w=w: mm(w, x.t()).t().contiguous()):
            fn()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            torch.cuda.synchronize(device)
            t.append((time.perf_counter() - t0) / 3 * 1000)
        res[shape] = tuple(round(v, 2) for v in t)
        del x, w
    for m in lins:
        a, b = res[(m.out_features, m.in_features)]
        m.swap = b < a * 0.97
    torch.cuda.empty_cache()
    model._orientation = res
    return res


def load_readout(model_dir: str) -> dict | None:
    """a decision readout in place of the LM head (perplexity-ai/pplx-decider-v1-27b and our quantized copy: readout.safetensors
    [codes, hidden] over the single-token answer codes of decision_config.json, read at its fitted temperature) -> {"weight",
    "token_ids", "temperature"} or None when the checkpoint has an LM head"""
    path = os.path.join(model_dir, "readout.safetensors")
    if not os.path.exists(path):
        return None
    from safetensors.torch import load_file

    with open(os.path.join(model_dir, "decision_config.json")) as f:
        cfg = json.load(f)
    w = load_file(path)["weight"]
    if w.shape[0] != len(cfg["token_ids"]) or cfg.get("format_version") != 1:
        raise RuntimeError(f"qwen: readout {tuple(w.shape)} does not match decision_config ({len(cfg['token_ids'])} codes)")
    return {"weight": w, "token_ids": list(cfg["token_ids"]), "temperature": float(cfg["temperature"])}


def is_qwen(model_dir: str) -> bool:
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f).get("model_type") in ("qwen3_5", "qwen3_5_text")
    except OSError:
        return False


class Qwen35Stack(Stack):
    """the Qwen3.5 family (hybrid: Gated DeltaNet + full attention): a TreePlan per batch on the host, compact rows, the captures of
    extraction ranges on the plan and the recurrent decoder (qwen_decode.Decoder)"""

    family = "qwen35"
    image_scheme = "qwen"
    hybrid = True
    bidirectional_images = False  # Qwen's image tokens attend causally, as text (qwen_vision.py)

    def __init__(self, options: QwenOptions | None = None):
        """options: built by a research script (tensors in `outliers`); None = EngineConfig.stack_options at build"""
        self._options = options
        self.options = options or QwenOptions()

    @staticmethod
    def detect(model_dir: str) -> bool:
        return is_qwen(model_dir)

    def build(self, cfg, model_dir: str, device):
        self.options = self._options or QwenOptions.parse(cfg.stack_options)
        return build_model(cfg, model_dir, device, self.options)

    def capture_bytes(self, model) -> tuple[int, int, int]:
        """on the hybrid also the recurrent rows per template token and the prefix state per group (qwen_decode); the K/V of the
        attention layers only"""
        gdn = [la.linear_attn for la in model.layers if la.kind == "linear_attention"]
        la0 = gdn[0]
        row = len(gdn) * ((2 * la0.hk * la0.dk + la0.hv * la0.dv) * 2 + la0.hv * 6)
        state = len(gdn) * la0.hv * la0.dk * la0.dv * 4
        tok = sum(2 * la.self_attn.hkv * la.self_attn.d * 2 for la in model.layers if la.kind != "linear_attention")
        return tok, row, state

    def compact_rows(self, sc, b) -> int | None:
        """with options.compact: the rows the model runs on, the stream's real tokens in whole blocks and at least the GEMM floor
        (min_stream, aligned jobs only: the floor serves invariance); None = the stream layout"""
        from spinf_serve.packer import BLOCK

        if not self.options.compact:
            return None
        R = int((b.art >= 0).sum())
        floor = sc.min_stream if sc.cfg.align_jobs else 0
        return min(b.length, max((R + BLOCK - 1) // BLOCK * BLOCK, floor))

    @staticmethod
    def x_keep(b) -> list[int] | None:
        """stream positions whose recurrent state the extraction decoder needs (the prefix ends of jobs with extraction slots)"""
        rows = [o.start + o.prefix_len - 1 for o in b.owners if o.xbranches]
        return rows or None

    def plan(self, sc, b) -> None:
        """the recurrent layers' plan, on the host ahead of the GPU"""
        qopts = self.options
        b.plan = tree_plan(
            b.prev, b.art >= 0, sc.device, path=qopts.gdn_path, short=qopts.gdn_short, rows=self.compact_rows(sc, b), keep=self.x_keep(b),
            call_tokens=qopts.call_tokens
        )  # fmt: skip

    def stage(self, sc, b, st: dict) -> dict:
        torch = sc.torch
        plan = getattr(b, "plan", None)
        if plan is None:
            self.plan(sc, b)
            plan = b.plan
        if plan.rows is not None:  # compact rows: the real tokens' positions, then zeros
            real = b.art >= 0
            pc = np.zeros(plan.rows, np.int64)
            pc[: int(real.sum())] = b.pos[real]
            st["pos"][: plan.rows].copy_(torch.from_numpy(pc))
            pos = st["pos"][: plan.rows].to(sc.device, non_blocking=True)[None]
        else:
            n = b.length
            st["pos"][:n].copy_(torch.from_numpy(b.pos.astype(np.int64)))
            pos = st["pos"][:n].to(sc.device, non_blocking=True)[None]
        return {"plan": plan, "pos": pos}

    def forward(self, sc, b, inputs: dict, masks: dict):
        torch = sc.torch
        dev = sc.device
        plan, pos = inputs["plan"], inputs["pos"]
        with torch.no_grad():
            real = b.art >= 0
            compact = plan.rows is not None
            x = sc.model.embed(b.ids[real], plan.rows) if compact else sc.model.embed(b.ids)
            ph = pos[0]
            if b.media:  # the images' rows over their placeholders, and (t, h, w) positions (qwen_vision.py)
                from spinf_serve.models.qwen35.qwen_vision import mrope_positions

                for start, obj in b.media:
                    c = int(np.count_nonzero(real[:start])) if compact else start
                    x[c : c + obj.n_tokens] = sc._media_embeds(obj)
                p3 = mrope_positions(b.pos, b.art, b.media)
                if compact:
                    pc3 = np.zeros((3, plan.rows), np.int64)
                    pc3[:, : int(real.sum())] = p3[:, real]
                    p3 = pc3
                ph = torch.from_numpy(p3).pin_memory().to(dev, non_blocking=True)
            H = sc.model(None, ph, masks["full_attention"], plan, embeds=x)
            if not compact:
                return H[None]
            # back to the stream layout (the read rows b.sel are stream positions)
            R = plan.sidx.shape[0]
            return H.new_zeros(b.length, H.shape[-1]).index_copy_(0, plan.sidx, H[:R])[None]

    @contextmanager
    def capture(self, sc, b, xo: list):
        """the capture request of a batch's extraction ranges, on its TreePlan (TreePlan.cap, compact rows): the K/V rows of the
        prefixes and templates, the templates' recurrent rows, the layer inputs at each slot's conv taps and the prefixes' final
        states; pos0 = per prompt key the position of the first generated token"""
        from spinf_serve.models.qwen35.qwen_vision import mrope_positions

        plan = b.plan
        real = b.art >= 0
        c_of = np.cumsum(real) - 1 if plan.rows is not None else np.arange(b.length)
        pos1 = mrope_positions(b.pos, b.art, b.media)[0] if b.media else b.pos
        ranges, tok, xin, state, pos0 = [], [], [], [], {}
        for o in xo:
            pk = ("p", id(o))
            ranges.append((int(c_of[o.start]), o.prefix_len, pk))
            state.append((int(c_of[o.start + o.prefix_len - 1]), pk))
            for bs, bl, pr in o.xbranches:
                tk = ("t", pr.key)
                r0 = int(c_of[bs]) if bl else 0
                ranges.append((r0, bl, tk))
                tok.append((r0, bl, tk))
                xin.append((int(c_of[bs + bl - 1]), tk))
                pos0[pr.key] = int(pos1[bs + bl - 1]) + 1
        out: dict = {"kv": {}, "gdn": {}, "xin": {}, "state": {}}
        plan.cap = {"ranges": ranges, "tok": tok, "xin": xin, "state": state, "out": out}
        yield {"kv": out["kv"], "extra": out, "pos0": pos0}

    def group_extra(self, sc, g, o, cap: dict, S: int) -> None:
        """the owner's recurrent captures into its decode.Group: the prefix's recurrent states (48 x 3 MiB fp32 per group) wait on
        the host: a 16k batch of extraction owners would otherwise hold GiBs of them on the card next to the next forward's peak
        (the decoder copies them back)"""
        from spinf_serve.models.qwen35 import qwen_decode

        extra, pos0 = cap["extra"], cap["pos0"]
        states = extra["state"].pop(("p", id(o)))
        g.extra = {
            "state": {li: sc._to_host(t) for li, t in states.items()},
            "tmpl": [{"tok": extra["gdn"].pop(("t", pr.key), {}), "xin": extra["xin"].pop(("t", pr.key))} for _, _, pr in o.xbranches],
            "pos0": [pos0[pr.key] for _, _, pr in o.xbranches],
        }
        g.extra_bytes = qwen_decode.group_extra_bytes(sc.model, g, S)

    def chunk_extra_bytes(self, model, groups: list) -> int:
        from spinf_serve.models.qwen35 import qwen_decode

        return qwen_decode.chunk_extra_bytes(model, groups)

    def sub_extra_bytes(self, model, sub) -> int:
        from spinf_serve.models.qwen35 import qwen_decode

        return qwen_decode.group_extra_bytes(model, sub, max(p.max_tokens for p, *_ in sub.prompts) + 1)

    def decoder(self, sc, chunk: list):
        from spinf_serve.models.qwen35 import qwen_decode

        return qwen_decode.Decoder(sc, chunk)
