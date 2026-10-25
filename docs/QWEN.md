# The Qwen3.5-architecture stack: a 27B hybrid model on one 24 GB card

This document records how spinf-serve runs a Qwen3.5-architecture model (48 Gated DeltaNet layers, 16 full-attention layers,
int4 GPTQ weights) over the same packed tree stream as the Gemma stack, and the measurements behind each choice: the tree
plan that carries recurrent states across branches; the int4 weights run first through fp16-accumulate GEMMs, then through
int8 tensor cores with block scales and an outlier-channel split, batch-invariant at every step; images through the
checkpoint's vision tower and 3-D rotary positions; extraction decoding on a model that has recurrent state; and the memory
bounds that keep a 27B model with 16k-token streams, captures and decodes inside 24 GB. The code is `models/qwen35/`
(`qwen.py`, `gdn.py`, `qwen_decode.py`, `qwen_vision.py`) and `quant/i8gemm.py`, `quant/w4gemm.py`.

Setup: one RTX 4090, torch 2.14.0+cu130, transformers 5.16.1, fla 0.5.2 (flash-linear-attention), October 2026. The
checkpoint at the time was the third-party GPTQ copy `btbtyler09/Qwen3.8-27B-GPTQ-4bit` (gptqmodel 6.0.3, 4 bits, group 32,
symmetric, no act-order); the published `spinf/qwen3.8-27b-int4` is our re-quantization with the same recipe
(`quantizations/qwen3.8-27b-int4`), and its parity references are of those weights. Why this model: on the 11 decision
benchmarks of the pplx-decider model card it scored a 78-80 mean against 66.9 for Gemma 4 12B in the maintainers' benchmark
runs. The run-time knobs named below are fields of `QwenOptions` (`models/qwen35/qwen.py`; `--stack-options key=value` on the
command line); the defaults are the production configuration. `flash-linear-attention==0.5.2` is pinned by the `engine` extra: `gdn.py` imports its
chunk helpers lazily, so the Gemma path never loads it.

## 1. The architecture and the tree plan

64 layers: 48 **Gated DeltaNet** layers ("linear attention": per head a 128 x 128 recurrent state, a 4-tap causal conv before
it) and every 4th a **full-attention** layer (24 query heads with a sigmoid output gate, 4 KV heads, head_dim 256, rope on the
first 64 dims). Hidden 5120, MLP 17408, vocabulary 248,320, untied lm head; a vision tower (§8) and an MTP head (unused).

The engine runs a call as packed streams: each job's shared prefix once, its prompts as a token trie of branches, a tree mask
in FlexAttention (`ARCHITECTURE.md`). The full-attention layers take that mask unchanged. A recurrent layer cannot be masked:
a branch must start from the state at the end of its parent. `TreePlan` (`qwen.py`):

- the packer records each token's path predecessor; a **segment** starts where the predecessor is not the previous stream
  token or is a fork, and runs to the next start;
- per layer, segments run **level by level** (prefixes, then their children, ...), each starting from its parent's final state;
- the conv reads each token's 3 path predecessors (a gather), not its stream neighbours.

Two ways to run the levels (`gdn_path`, both kept):

| path | how | decision-benchmark jobs (coarse tries) | news call (fine tries) |
|---|---|---|---|
| `levels` | fla's `chunk_gated_delta_rule` per level, the parents' states gathered into per-segment copies (3 MiB each, calls of <= 96) | 312.2 s | 2,757 tok/s |
| `indexed`, all chunked | `gdn.py`: fla's state kernel vendored with states read / written by row index (no copies, 2 calls per level); bit-identical to `levels` | | 3,333 tok/s |
| **`indexed`** (default) | + segments of <= 64 tokens (`gdn_short`) token by token (`gdn.py` recurrent kernel: state in registers, no 1.5 MB snapshot per 64-token chunk) | **311.5 s** | **~3,700 tok/s** |

Why it matters: a 16k stream of news articles with fine question tries has ~2,300 segments over 6-7 levels, a decision-benchmark
stream 30-100. With `levels` the recurrent layers took ~30 % of the forward on the news workload (state copies 8 %, chunk-state
snapshots 11 %) against 6 % on the benchmark jobs. The short-segment threshold is flat between 16 and 256 tokens (3,680-3,740
tok/s). `indexed` vs `levels` scores: mean |d logprob| 0.02 (the token-by-token kernel's rounding), both 0.061 from exact bf16.
fla's own recurrent kernel was not usable (NaN, then an illegal address with varlen + initial states in fla 0.5.2), and its l2norm
kernel is autotuned on the stream length (its rows change with the packing): `indexed` l2-normalises q / k in the compiled layer.

Checked: a CPU test on random packed tries (`tests/test_qwen_plan.py`: the level-by-level run equals a sequential recurrence
along every path; the conv taps); on the GPU a branch run from its parent's state equals the prefix + branch run as one sequence
(4.9e-4 max abs, bf16); our layers against transformers' `Qwen3_5DecoderLayer` with the same dequantized weights: 0.7-0.8 %
relative on a prefix + 3 branches, MLP exact, the mixer 0.35 % (fla's kernel vs transformers' fp32 fallback).

## 2. Memory on 24 GB

| | GiB |
|---|---|
| 64 layers, int4 packed + fp16 group scales | 13.2 |
| lm head (bf16, untied) | 2.4 |
| embedding table | on the host (`embed_host`: a pinned gather per forward) |
| vision tower (`EngineConfig.media`, on by default; §8) | 0.86 |
| MTP head | not loaded |
| loaded | 16.1 |
| peak with 16k-token streams | 18.4 (f16); the i8 configuration 18.2, 19.0 with the tower (§11) |

## 3. The linears: what runs the int4 weights

Per projection at M = 16,384 rows (maintainers' micro-benchmarks):

| option | TFLOPS | output error | note |
|---|---|---|---|
| bf16 GEMM, int4 dequantized per call | 160-170 | exact (0.3 % bf16 rounding) | dequant costs ~3 % |
| fp8 (int4 -> e4m3 per-tensor, dynamic per-tensor activation scale) | 190-340 | ~3.8 % | cuBLAS picks a slow tile for K 5120/6144 with N <= 12288 |
| torch `_weight_int4pack_mm` (tinygemm) | 18 | | a decode kernel |
| Triton W4A16, in-register unpack, fp16-accumulate tiles | 205-223 | 0.18 % | Triton's own fp16 ceiling here 255-261 |
| **cuBLAS fp16 with fp16 accumulation** (`allow_fp16_accumulation`) | **265-303** | **0.3-0.37 %** | the 4090's fp16-accumulate rate is 2x its fp32-accumulate rate |

Fusions: q k v z (+ the b/a gates, §5) of the recurrent layers, q k v of the attention layers, gate + up of the MLP run as one
GEMM each (larger N also reaches cuBLAS' fast tiles). Each shape's orientation (x W^T or (W x^T)^T) is timed at load.

**fp8 costs accuracy here.** WinoGrande (1,267 items, very sensitive): vLLM 84.8, engine bf16 84.6, fp8 82.9. Simulated apart,
fp8 weights alone 84.1 and fp8 activations alone 84.1: each ~0.5 point, together ~1.7. Not outliers: a linear's stream max is
at most 27x (usually < 10x) a typical token's, and per-token scales barely lower the error (2.2 % -> 2.1 %): it is e4m3's 3-bit
mantissa. **f16 became the first default** (`mode`): 84.45 on WinoGrande, log-probs 4.6x closer to bf16 than fp8 (mean |d logprob|
0.059 vs 0.269), no overflow on 3.2M tokens. §7 replaces it with i8.

## 4. Speed against vLLM

Workload: the 11 decision benchmarks of the pplx-decider card, 100 items each: 2,211 jobs, 4,786 prompts, 3,199,106 tokens as a
plain server runs them, 1,340,225 after prefix + trie sharing. The same jobs everywhere; GPU-busy time.

| engine | time | logical tok/s | vs vLLM |
|---|---|---|---|
| vLLM 0.26 (GPTQ Marlin, prefix caching, 1 token + top-20 per prompt, 8 concurrent) | 685 s | 4,671 | 1.0x |
| engine, every prompt in full (no sharing), f16 | 744 s | 4,299 | 0.92x |
| engine bf16 | 456 s | 7,011 | 1.5x |
| engine fp8 | 285 s | 11,225 | 2.4x |
| **engine f16** | **312 s** | **10,257** | **2.2x** |
| engine f16, batch-invariant (§5) | 340 s | 9,415 | 2.0x |

Raw speed of a full 16,384-token stream (RAGTruth, f16): 3.71 s, 4.4k tok/s; GEMMs 77 % of it at ~278 TFLOPS, the recurrent
kernel 5.5 %, MLP activation 3 %, norms / flex attention / casts / dequant ~1.5 % each. The sharing (trie + prefix, with the
recurrent states carried across branches) is the 2.4x between the flat and the tree rows. Along the way: fusions + orientation
took an FPB stream from 3.2k to 4.6k real tok/s (fp8).

The maintainers' test bed suites (in process, protocol path `ScoreService.score_request`, one call at a time; the Gemma numbers
are `BATCH_INVARIANCE.md` §4):

| suite | Qwen f16, `levels` | **Qwen f16, `indexed`** | Gemma 4 12B fp8s |
|---|---|---|---|
| news (1,024 articles, 64 per call) | 2,713 tok/s | **3,627 tok/s** | 9,874 |
| moderation (1,024 texts x 106 prompts, 32 per call) | 2,359 tok/s | **3,646 tok/s** | 9,832 |

The tokenizers differ, so wall time on the same requests is the honest comparison: news 91.6 s (Gemma) vs 228.9 s (Qwen, 2.50x),
moderation 118.2 vs 313.4 s (2.65x), the decision sets 128.2 vs 340 s (2.65x); token counts after sharing within 0.7 %. Qwen has
2.06x the non-embedding parameters; the rest (~20-30 %) is fp16-accumulate GEMMs instead of fp8 and the recurrent layers. On
the decision sets the extra time buys 79.5 vs 66.9.

## 5. Batch invariance

f16 has no activation scale (the source of drift in Gemma's dynamic fp8), and the int4 group scales are fixed by the checkpoint.
What still moved a job's scores with its packing (mean 0.044 |d logprob|, max 0.39, alone vs packed) was **cuBLAS picking another
GEMM kernel by M**:

- the separate b/a gate projection (N = 96, bf16) at every M: now extra rows of the fused q k v z GEMM (N = 16,480: invariant);
- the output projections (6144 -> 5120) for M < 5,248; every other shape from 1,280 or less (a sweep over every multiple of 128
  up to 16,384).

With aligned jobs (as Gemma), streams of at least **5,248** rows (`MIN_STREAM["f16"]`, taken by `Scorer.min_stream`) and the fixed
lm-head rows: **1,944 reads alone vs packed vs reversed, all bit-identical**. fla's varlen kernels are per-sequence: no packing
effect. Cost +9 % (340 s vs 312 s). Measured on one 4090 and this torch / cuBLAS: other cards need the sweep again. The i8 scheme
of §7 needs no floor at all (its GEMM rows do not depend on M).

## 6. Accuracy on the 11 decision benchmarks

Engine f16 against an earlier vLLM run of the same GPTQ copy (full test sets, each benchmark's chosen reading): mean 78.37 vs
78.76. Seven sets agree to < 0.005 mean |dp| and 99-100 % of predictions (ContractNLI 82.26 = 82.26, RAGTruth 77.63 = 77.63,
FPB, Belebele, BBH, JevBench, WinoGrande). Three differ more: Circa (80.0 vs 81.2, mean |dp| 0.10), TabFact (82.6 vs 85.0, 0.07),
TruthfulQA (87.0 vs 89.6, 0.027). **The engine is right there:** the vLLM harness read each option from the top-20 next-token
list and gave a token outside it the 20th log-prob, so rare multi-token options got too much mass (vLLM minus engine on the least
likely label: +0.08 Circa, +0.06 TabFact, +0.02 TruthfulQA, +0.001 ContractNLI, whose letters are always in the top 20). The
engine in exact bf16 agrees with f16 (0.002 / 0.007) and not with vLLM (0.10 / 0.07). The full run (11 test sets + dev reads) took
69 minutes on the engine; vLLM needed ~2.5 h for RAGTruth alone.

## 7. The squeeze: compact rows, int8 tensor cores, the outlier split, fused epilogues (2026-10-05)

Fresh profiles (f16) of a news call (13.3 s GPU, 4 streams) and a 16k RAGTruth stream: GEMMs 73-76 %, the recurrent kernel
5.7 %, MLP silu x mul 3.0 %, casts and residual adds at the graph breaks 3.2 %, norms 2.2 %, conv gather 1.8 %, dequant 1.7 %,
GDN gathers / scatters 1.4 %, chunked GDN kernels 1.5 %, flex 1.2 %, lm head 0.9 %. Three facts framed the work:

1. **The card is power-bound.** During the GEMM loop the 4090 sits at 478 of 480 W, SM clock 2.2 GHz, where the fp16-accumulate
   peak is ~289 TF: cuBLAS' 267-297 TF is 93-100 % of it. cuBLASLt, a larger workspace and N-splits are all slower.
2. **Padding is 8 % of a production stream** (news 8.1 %, moderation 7.7 %), 12 % of the decision set, 34-57 % on 100-token sets;
   every pad row went through every GEMM.
3. **Cross-job header sharing is worthless here**: the jobs of a call share 0 tokens at their start on news / moderation (the
   content comes first) and <= 2 % on 9 of 11 decision sets. Dropped.

### 7.1 Numerics-preserving changes (f16; all on by default)

| change | knob | decision set (aligned f16) | note |
|---|---|---|---|
| baseline (§5) | | 341.9 s | |
| **compact rows**: the model runs on the real tokens (whole 128-blocks, >= the GEMM floor); pads exist in the 16 flex layers only, where q k v are scattered into the stream layout the mask describes | `compact` | 317.3 s (-7.2 %) | 13,010 / 13,010 reads bit-identical to the stream layout; peak memory unchanged |
| the GDN in_proj returns fp16 (no 1.1 GB cast pass per 16k rows); the recurrent kernel at 2 warps (-13..-27 % on realistic levels); the token-by-token kernel reads / writes the stream rows through the call's index instead of 5 gathers + 1 scatter (-6..-20 %) | `rec_idx`, `gdn.RECURRENT` | **312.8 s (-8.5 %)** | numerics move 0.058 mean \|d logprob\| vs the compact run, the same distance every f16 variant has from exact bf16 (0.0597-0.0603): a rounding noise floor |

Batch invariance after all of it: 1,944 / 1,944 reads alone vs packed vs reversed bit-identical. Test bed suites: news 3,881
tok/s (was 3,627, +7.0 %), moderation 3,958 (was 3,646, +8.6 %). The fine-trie suites gain more than the decision set because
the recurrent kernel and its gathers weighed more there.

### 7.2 int8 tensor cores: the 2x lever and what it costs

`torch._int_mm` on the Qwen shapes under the same power cap: **545-628 TOPS, 2.0-2.2x the fp16-accumulate GEMM** (int8 MMA does
twice the work per joule). Accuracy, simulated on WinoGrande (bf16 84.6, f16 84.45):

| activations | weights | WinoGrande | |
|---|---|---|---|
| bf16 | int8 per output channel (re-rounded from the int4) | 84.53 | free |
| int8 per token | bf16 | 80.9 | dead; each linear group alone loses ~1 point: the loss is spread, not one culprit |
| int8 per token | int8 | 81.9 | |
| **int8 per token and block of 128 input channels** | int8 | **84.77** | the block scale restores bf16 accuracy (64: 85.0, 256: 84.37) |

The kernel side: an unfused int32 epilogue (int32 out, then scale) makes `_int_mm` slower than f16 for 4 of 5 shapes (the int32
round trip through memory), and a block scale per 128 channels needs the scale applied per K block inside the GEMM, which
cuBLASLt cannot do. So `quant/i8gemm.py`: a Triton int8 GEMM, int32 per K block (exact: |x|, |w| <= 127, 128 terms), scaled into
an fp32 accumulator in ascending block order, bf16 out, M / N tails masked.

| kernel | TOPS | vs f16acc |
|---|---|---|
| Triton int8, per-tensor scales (the ceiling of Triton's int8 codegen here) | 383-405 | 1.45x |
| **Triton int8, block-128 activation scales** (`i8gemm`) | **354-377** | **1.30-1.37x** |
| `_int_mm` (cuBLASLt, int32 out, no scales) | 558-625 | 2.1x |

The block epilogue costs ~6 %; the rest of the gap to cuBLASLt is Triton's int8 MMA efficiency on Ada. Properties of the kernel:
every configuration matches the per-block `_int_mm` reference within one bf16 ulp, and **a row's result is bit-identical across M
and tile configurations** (each K block exact, fixed summation order): no GEMM floor, the configuration per shape is timed at load
(`i8gemm.tune`). Mode `i8` (`--quant i8`, `EngineConfig.quant`): int8 weights per output channel made per call from the int4
(`i8gemm.dequant`, half the bytes of the fp16 dequant), the b / a rows int8 too, activations block-quantized, the three launches
together outside the compiled graphs; inductor's versions of the two small kernels cost the whole GEMM gain, so they are one-pass
Triton kernels (0.29 / 0.38 ms per 16k-row gate_up).

| | f16 (all of §7.1) | **i8** |
|---|---|---|
| decision set, aligned | 312.8 s | **275.3 s (-12 %)**; 11,620 logical tok/s, 2.49x vLLM |
| test bed news / moderation | 3,881 / 3,958 tok/s | **4,466 / 4,481 tok/s** |
| 16k RAGTruth stream | ~3.7 s | **3.30 s**; the GEMM 72 % at ~335 TOPS in situ, quantize 3.8 %, dequant 1.5 % |
| WinoGrande | 84.45 | **84.45** (bf16 84.6) |
| batch invariance (1,944 reads) | bit-identical, floor 5,248 | **bit-identical, floor 1,024** |
| vs exact bf16, 13,010 decision reads | mean \|d logprob\| 0.060, top option as bf16 99.1-99.5 % | mean 0.137, top option as bf16 98.3 % |
| all 11 decision sets, full test sets (headline readings) | mean 78.37 | **mean 78.39** |

So i8 is a speed tier at twice the f16 rounding noise and the same accuracy; what it changes is which close calls flip (1.7 % of
the top options vs 0.5-0.9 % for f16), not how many are right.

### 7.3 The outlier split: cuBLASLt's 600 TOPS on the norm-fed linears

cuBLASLt's int8 kernel needs a per-token activation scale. On one 8k RAGTruth stream (bf16), in the three norm-fed linears
(in_proj_qkvz, qkv_proj, gate_up_proj: 69 % of the GEMM FLOPs) 99-100 % of the tokens have their largest |x| in the same 16
channels; in down_proj / out_proj / o_proj the maxima are spread (the top 128 channels hold 28-57 %). Per-token int8 with the
top-n channels kept exact on every linear (LLM.int8()'s decomposition with a **fixed** set per linear, so still batch-invariant):
n = 64 82.16, n = 128 83.98 on WinoGrande. The mixed scheme, the split on the norm-fed linears and block scales on the other
three: **n = 64 85.0**, n = 128 84.69.

Built as `i8gemm.split_linear` for `SPLIT_LINEARS` (`qwen.py`): a one-pass per-token quantizer over the kept channels (the
outlier channels written as 0 and copied out in fp16 in the same pass), `torch._int_mm` (int32 out; the integer accumulation is
exact, so cuBLASLt's kernel choice by M cannot change a result: invariant), then a fused epilogue that reads each int32 tile once,
scales it, adds the 64 outlier channels' exact contribution (x_out [M, 64] fp16 times the dequantized int4 columns [N, 64], a
small in-register dot) and writes bf16. Within a bf16 ulp of the torch reference, rows bit-identical across M, 6x less error than
the block scheme on outlier-heavy data. The int32 round trip (4 B written, 4 B read per output) bounds the gain.

The outlier channels are a calibration artifact of the revision: 128 linears x 64 sorted input channels, exported from the
per-channel activation maxima of a calibration stream (`quantizations/export_outliers.py`). The published checkpoints carry the
table inside the snapshot as `spinf_outliers.json`; the third-party copy run by its id reads
`quant/tables/btbtyler09__Qwen3.8-27B-GPTQ-4bit/<revision>-outliers.json` (`quant/tables.py load_outliers_table`). `i8` runs
the split scheme whenever the revision has a table and refuses to run without one (without it the split would silently degrade to
per-token scales). Ties: 89 of the 128 linears have several channels tied at the 64th-largest absmax (bf16-derived values), and
CPU and GPU `topk` break ties differently, so a set chosen on the GPU at load differed from the table on 39 linears; both are
valid splits (a tie channel is a small one), but the table is the reference and the loader selects on the CPU exactly as the
export did.

| | i8 (block) | **i8 split** (64 channels exact on the norm-fed linears) |
|---|---|---|
| decision set, aligned | 275.3 s | **245.5 s (-10.8 %; -21.5 % vs f16)**; 13,032 logical tok/s, 2.79x vLLM |
| WinoGrande | 84.45 | **85.0** (bf16 84.6) |
| vs exact bf16, 13,010 decision reads | 0.137 mean \|d logprob\|, top option as bf16 98.3 % | **0.119, 98.8 %** |
| peak memory | 18.4 GiB | 20.0 GiB (the int32 accumulators; §11 brings it back) |
| batch invariance (1,944 reads) | bit-identical | **bit-identical** |
| 16k RAGTruth stream | 3.30 s | **2.94 s**: `_int_mm` 30.7 % (~580 TOPS), the block GEMM 25.5 % (down / out / o_proj), the epilogue 12.1 % (the int32 round trip, at bandwidth) |
| test bed news / moderation | 4,466 / 4,481 tok/s | **4,882 / 4,987 tok/s** (f16 3,881 / 3,958: +26 %) |
| drift from f16 on news (512 items, 78,894 reads) | | mean \|d logprob\| 0.046, p99 0.22; two-option log-odds 0.055 mean, p95 0.15 (the Gemma engine's own packing noise under dynamic fp8 is 0.135, `PRECISION.md` §3) |
| all 11 decision sets, full test sets | mean 78.39 | **mean 78.54** (f16 78.37): every set within 0.6 of f16, none more than 0.1 below; 55 min for the run (f16 69) |

**Why not SmoothQuant.** Moving the three block-scaled linears to cuBLASLt too would need a per-token scale there as well.
SmoothQuant-style static per-channel factors (x / s_c per token, W s_c re-rounded to int8 per output channel, s_c =
amax_c(x)^a / amax_c(W)^(1-a)) do not get there: WinoGrande 82.24 (a = 0.5, the split on the norm-fed linears + smoothing on the
rest), 82.08 (a = 0.75), 83.11 (smoothing on all six) vs 85.0. Their outliers are token-dependent (the top 128 channels hold
28-57 % of the maxima), which a static factor cannot flatten.

### 7.4 The consumers move into the split epilogues (`fuse_glu`, `fuse_gdn`, on by default)

The int32 round trip itself cannot go (torch exposes no cuBLASLt epilogue for int8), but the passes after it fold into the
epilogue that already reads every int32 tile once, and the per-token quantizer writes what the epilogue needs:

- `glu`: the MLP's gate_up epilogue computes silu(gate) x up and block-quantizes the result for down_proj in the same pass
  (`i8gemm.split_glu_q`; a 128-column tile is one activation block, `GLU_TILE`). The [T, 2I] bf16 write and read, the silu x mul
  kernel (4 %) and down_proj's quantizer pass go.
- `gdn`: the Gated DeltaNet's in_proj epilogue runs the 4-tap causal conv along the token's path (the taps' rows read from the
  int32 accumulator, with their outlier dots), silu, and the q / k l2 norm (a 128-column tile is one head, `GDN_TILE`;
  `i8gemm.split_gdn_in`). The gather + conv pass (2.4 %) and the norm pass go; the fused epilogue costs 0.1 ms more than the
  plain one per 16k in_proj (7.44 vs 7.32 ms).

Numerics: everything in fp32 from the unrounded accumulator where the unfused path rounded gate / up and the conv input to bf16
first; rms error vs an fp32 reference 1.00e-2 fused vs 1.04e-2 unfused (glu), 1.7e-3 vs 2.4-2.9e-3 (gdn). Row-wise, so
batch-invariant. One caveat: the gdn kernel's norm reduction order follows its tile configuration (fixed: 4 warps at any BM give
identical results, 8 warps differ in the last bit), so the configuration is part of the numerics, as the fp16 GEMM kernel is.

| | split, plain epilogues (§7.3) | + glu | **+ glu, gdn, quantizer writes the outlier columns** |
|---|---|---|---|
| 16k RAGTruth stream | 2.95 s | 2.72 s (-7.8 %) | **2.63 s (-10.8 %)**: `_int_mm` 34.5 %, the block GEMM 28.8 %, the glu epilogue 7.1 %, the gdn epilogue 3.6 %, RMSNorm 2.6 %, the GDN kernels ~5 % |
| decision set, aligned | 245.5 s | 224.9 s (-8.4 %) | **220.4 s (-10.2 %; -29.5 % vs f16 312.8, -35.5 % vs the day's 341.9)**; 14,517 logical tok/s, 3.11x vLLM |
| peak memory | 20.0 GiB | 19.2 GiB | 19.2 GiB |
| vs exact bf16, 13,010 reads | 0.1105, 98.81 % (the table's channels) | 0.131, 98.6 % | **0.112, 98.7 %** (f16 0.060, 99.5 %) |
| WinoGrande | 85.08 | | 84.37 (f16 84.45, bf16 84.6: +-1 point is the standard error on 1,267 items) |
| batch invariance (1,944 reads, floor 1,024) | bit-identical | | **bit-identical** |
| all 11 decision sets, full test sets | mean 78.54 | | **mean 78.53** (f16 78.37); per set within 0.8 of f16; 49 min for the run |
| test bed news / moderation | 4,882 / 4,987 tok/s | | **5,378 / 5,473 tok/s** (f16 3,881 / 3,958; the day's start 3,627 / 3,646: +48 % / +50 %) |

Left on this track: the block GEMM (down / out / o_proj, Triton at ~335 TOPS vs cuBLASLt's ~580) is 29 % of the forward and the
one lever of size left on the GEMM side; `_int_mm` at 34.5 % is the hardware.

## 8. Images: the vision tower on the hybrid model (2026-10-06)

`qwen_vision.py`; `EngineConfig.media` (on by default) loads it. The checkpoint's tower is transformers' `Qwen3_5VisionModel`:
16x16-pixel patches over two identical "frames" (temporal patch 2), 27 blocks at width 1152, a 2x2 merge into one token of the
text width (5120); 460M parameters, 0.86 GiB bf16. It runs as transformers' module (sdpa), one image at a time, cached for the
call (a prefix re-sent in several streams embeds its images once). One token per 32x32 pixels: the preprocessing (`decode_image`,
numpy / PIL) follows the checkpoint's `Qwen2VLImageProcessor` (`smart_resize` to multiples of 32 within [65,536 pixels, the
budget], bicubic, (x / 255 - 0.5) / 0.5, patches in spatial-merge-block order) with the budget per `detail` set here: low 256,
auto 1,024, high 4,096 tokens (`DETAIL_TOKENS`; Gemma's are 70 / 280 / 1,120 on 48-px blocks; the processor's own default allows
16,384). The compiler lays an image out as the chat template does, `<|vision_start|>` + n x `<|image_pad|>` + `<|vision_end|>`,
images first in the content; the Scorer writes the tower's rows over the placeholders' embeddings (compact rows: at the
placeholders' compact index). No audio in this checkpoint (`input_audio` is refused); video frames are images after their
timestamps.

Positions. The text stack's rope is 3-D (`mrope_interleaved`, sections [11, 11, 10]: of the 32 rotary frequencies, every third
reads the t, h, w axis in turn, the last two the t axis). Text tokens have one position on the three axes, so the text path's 1-D
rope is exact and unchanged; an image's tokens share the t position of the image's start and take their row / column on h / w,
and the text after the image continues from start + max(rows, cols), not start + tokens (transformers `get_rope_index`).
`mrope_positions` computes the [3, T] table on the host from the packer's path positions and the batch's media, the shrink applied
to every token of the job after the image, on every branch. Qwen3.5 attends to image tokens causally (no Gemma-style bidirectional
block), so the tree mask, the block tables and the Gated DeltaNet plan are untouched: the images are text to them.

Verification: preprocessing vs transformers' `Qwen2VLImageProcessor` (PIL backend): the same grids, max |pixel difference|
1.2e-7 over seven cases (300x200 to 2000x1500, a photo at the three details); positions vs `Qwen3_5Model.get_rope_index`: exact on
a packed tree (a prefix with two images, three branches); layers 0-3 on the tower's embeddings with 3-D positions vs transformers'
decoder layers with the same dequantized weights: rel err 0.57 % (transformers with 1-D positions differs from its 3-D by 1.88 %);
end to end on the production configuration: a red square -> red 0.9996, a printed 3 at detail low -> 3 0.999, a photo of a car ->
car 0.988 (auto, 960 tokens) / 0.989 (low, 240 tokens), a text control in the same call unchanged (bf16: 0.9997 / 0.9991 /
0.9890); vs vLLM 0.26.0 on the same rendered prompts (images sized to multiples of 32 so neither side resizes): the same token
counts, the top token the same on all four image prompts with its log-prob within 0.03, the mean |difference| over the top-20
0.22-0.45 (i8) / 0.18-0.34 (bf16), and a text-only control shows the same spread: the two stacks' numerics, not the image path.

Cost. The tower: 13 / 59 / 438 ms per image of 234 / 972 / 4,015 tokens, so the text stack dominates (1,024 image tokens run as
text, ~0.19 s), and `detail` is the lever. Memory: +0.86 GiB resident. Text requests are untouched: the decision set 219.9 s with
the tower resident vs 220.4 s in §7.4, its 13,010 reads bit-identical.

## 9. Extraction: decoding on the hybrid model (2026-10-06)

`qwen_decode.py`, the capture hooks in `qwen.py` (`TreePlan.cap`), the stack's `capture` and `decoder` (`models/base.py`). The
extraction API (`{*} {n} {$}` slots, the search of `extract.py`, the batched beams of `decode.py` over the scoring pass's captures:
`EXTRACTION.md`) runs on the Qwen stack; the compiler, the protocol and the search are unchanged.

What the scoring pass keeps for a slot (per batch, in compact rows): the K / V rows of the job's prefix and of the slot's template
in the 16 full-attention layers (as the Gemma stack's capture), and for the 48 recurrent layers three things the Gemma path has no
counterpart for: the prefix's final state S_P (kept even for a prefix without branches; 48 x 3 MiB per job), the template's
q k v g beta rows as the layer computed them (post conv, silu, q / k norm), and the layer input at the 4 conv taps of the slot's
start (the decoder recomputes their pre-conv projections, so the fused epilogue needs no change). A slot with no template text (the
template merges into the content) is a zero-length branch at the prefix's end: all jobs of a call are packed and decoded together.

The decode step (`qwen_decode.Decoder`, with `decode.Decoder`'s search and attention buffers): one token per slot, the 64 layers
in order. The linears run at M = slots through `quant/w4gemm.py` (`decode_linear = "w4"`): the int4 weights streamed once and
dequantized in registers, fp32 accumulation. The scoring path's `Int4Linear` (`"model"`) dequantizes every weight into a bf16 /
int8 copy per call, which at M = 16k rows is amortised and at M = 12 is the whole step (42 ms of dequant + 36 ms of int8 GEMMs per
step; w4gemm's 6 shapes take 29 ms at ~430 GB/s). A recurrent layer does not keep a state per slot (144 MiB each, copied whenever
a beam fans out): it re-runs the recurrence from S_P over the slot's rows, its template's and then the tokens it generated, with
the token-by-token kernel (`gdn.recurrent` indexing the row caches); a few dozen tokens per slot and layer, 3 MiB of state read
per (slot, layer), nothing stored. The 4-tap conv reads a per-slot ring of the 3 previous pre-conv rows. A full-attention layer
writes the token's K / V into the decoder's buffers and attends over (prefix | own template | own generated) through
`decode._x_attention`, positions continuing from the template's end (after an image: the shrunk position, §8). When beams fan
out, the row caches and the rings follow them as the K / V do. The step is replayed from a CUDA graph (`decode_graphs`; ~6,600
kernels): the row index of the recurrence is built on the GPU from the step counter, the embeddings are copied into a static
input before the replay, the ring advances after it, and every write inside is idempotent.

Capture and torch.compile: the captures read the layer index and the capture dict only inside `torch.compiler.disable`d
functions. A first version read the layer index in the compiled forward: dynamo guards on integer module attributes, one recompile
per layer, the cache limit in the first forward and the layers fall back to eager for good, for text batches too.

Verification: teacher-forced, two jobs (a template of 8 tokens; no template, the zero-length branch) decoded for 5 / 10 forced
tokens, the decoder's next-token log-probs vs the packed forward on the whole sequence at the same positions: step 0 is the
scoring pass's own distribution (identical); in i8 the forced token is within 0.000-0.104 nats, the top token the same at every
step, the mean |d| over the reference's 20 likeliest tokens 0.02-0.10 (at most 0.43), the i8 path's drift between two runs of the
same tokens through different kernels (§7.4); bf16 the same picture. End to end through protocol -> compiler -> scoring +
extraction, x_batched and the merged-template path: "The company named in the text is "{*}"" -> Apple Inc. 0.39 (Apple 0.11),
"Quarterly revenue: {$}" -> 94.9 billion 0.60, "Growth in percent: {n}" -> 6 0.50, the merged slot (content ending "the lower
bound is ") -> 5.25 0.64; 7 slots, 210 generated tokens, 2.4 s with the scoring passes. The text path with the hooks in place:
the decision set 220.0 s (220.4 in §7.4), 13,010 / 13,010 reads bit-identical, no recompiles. The w4 step linears vs the "model"
ones: the forced token within 0.000-0.086, the answers above unchanged to the third digit.

Cost (2 groups x 6 slots, i8 weights): the step 38.7 ms wall / 33.5 ms GPU from the graph (eager 48.2 / 42.9; the "model" linears
116 ms), of which the w4 GEMMs 22 ms and ~6,600 small kernels 12 ms; the first step 0.2 s. Per step the search adds the lm head
over the live slots (~4 ms) and the beam bookkeeping; a decode of 7 slots (209 tokens) with its two scoring passes took 3.4 s.

## 10. Brand extraction from video frames (2026-10-06)

The maintainers' earlier extraction study's benchmark: 207 TV commercials with the advertised brand as gold, 32 / 4 / 1 frames at
detail low in time order, an intro line, the slot `The brand featured in this commercial is "{*}"`, on the Qwen stack through the
production path (protocol -> compiler -> scoring + extraction, 4 ads per request). A frame is 528 x 288 pixels: 144 tokens on
Qwen (Gemma's processor: 70), so a 32-frame ad is 4,640 content tokens. Graded as there: the top answer names the brand
(sub-brands count); "merged" sums the probability over answers naming the same brand. The videos are not redistributed.

| | brand32 | brand4 | brand1 |
|---|---|---|---|
| Qwen3.8-27B int4, i8 engine: top answer | **0.986** | **0.647** | **0.222** |
| brand-merged | 0.986 | 0.710 | 0.275 |
| mean P(top); AUC of P(top) right vs wrong | 0.88; - | 0.49; 0.93 | 0.15; 0.92 |
| Gemma 4 12B (the earlier study's harness) | 0.97 | 0.57 | 0.19 |
| Gemma 4 31B | 0.99 | 0.74 | 0.29 |
| content tokens per ad | 4,640 | 604 | 172 |
| seconds per ad (scoring + decode, 4 ads per request) | 1.30 | 0.27 | 0.18 |

Between the two Gemmas, at the 31B's level on the full video. The three brand32 misses are Bud Light <-> Budweiser twice (one
company, two brands) and a Pepsi ad whose product is bubly (a PepsiCo brand: the model's answer names what is on screen). The
probability is far better behaved than Gemma's raw P (which averaged about half the accuracy): 0.88 at 98.6 % right on 32 frames,
and it ranks right vs wrong at AUC 0.92-0.93 when the task is hard.

Memory on this run: the first full pass (6 ads per request, the 16k budget) ran out of memory after 90 ads at 22.4 GiB: the i8
scoring peak on a 16k stream (19.9 GiB with the tower) plus the captures every 32-frame ad of the call leaves for its decode
(4,640 tokens of K/V in 16 layers, 300 MB, plus 144 MiB of prefix states), plus the media cache. Three changes since: the split
GEMMs produce their int32 accumulator in row chunks of 8,192 into one reused buffer (`i8gemm.ROW_CHUNK`; the MLP's transient
2.51 -> 1.56 GB); a level's long segments go to the chunked kernel in calls of at most 8,192 tokens (`call_tokens`; the
recurrent layer's transient 2.81 -> 1.86 GB); captures are bounded before the decode (§11). The 16k forward then peaked at
18.99 GiB with the tower (18.18 without), 13,010 / 13,010 reads bit-identical; the 32-frame ads at the 16k budget with 6 or 8 per
request run at 21.74 GiB, 1.45 s per ad.

## 11. Memory bounds on 24 GB (2026-10-10)

A production request ran the engine out of memory (`i8gemm.dequant`, 21.9 GiB allocated of 23.5) and the engine never recovered:
every later call failed in 0.4 s until the process was restarted. Two faults, both fixed, plus the bounds measured with a
maintainers' sweep (the production path in process, synthetic requests by shape, peak / reserved / after per call).

**Why it never recovered.** A failed call kept its GPU memory: the producer thread had prefetched up to 6 batches (their recurrent
plans on the card) and blocked forever on the full queue; the decode groups held for the call's decode, an open capture and the
media embeddings stayed referenced; and the exception's traceback kept the failed frames' tensors alive while the handler ran.
`Scorer._execute` now stops the producer, drains the queue and drops everything on any failure (`Scorer.abandon`);
`ScoreService.score_request` catches an out-of-memory error outside the `except` block, releases the allocator's cache, checks the
card is back at its idle level, and answers `request_too_large` (a 413). If memory stayed held (> 1 GiB over idle) the engine marks
itself degraded (`/readyz` and `/livez` 503) and exits for a restart (`SPINF_EXIT_ON_LEAK`). Every response carries
`memory: {peak_gib, reserved_gib}` and the engine logs one line per call (shape, peak, decode chunks).

**A compiler bug found on the way.** When a template does not tokenize cleanly after the content (content ending in whitespace,
template starting with a newline: common), the compiler cuts the content a few tokens before its end and re-tokenizes the tail
with each template. The cut was tried at 32 / 64 / 128 / 256 characters from the end; a cut inside a token never re-tokenizes the
same, so in about a third of such contents no cut was found and the fallback re-sent the WHOLE content once per slot (a 15k-token
content with 10 slots: 157k tokens processed instead of 19k) with an empty prefix, which the hybrid plan cannot decode. `_cut`
(`compiler.py`) now tries the last 64 token boundaries (fast tokenizers' offsets) and, when none is clean, reads the content once
and joins the templates at the token boundary (warning `join_not_merged`).

**Where the memory goes** (Qwen3.8-27B int4-i8, 4090, idle 16.7 GiB after warm-up):

| | GiB | bound |
|---|---|---|
| 16k-token forward, transient | +2.5 (peak 19.2) | `token_budget` |
| captures of a batch's extraction branches: prefix K/V 64 KB/token, template rows 1 MB/token, prefix states 144 MiB/group | up to 2.6 on a 16k stream of 8 extraction inputs | `capture_budget_gib` (the Qwen stack's default 1.0; the packer puts fewer extraction inputs per stream above it); the states go to pinned host memory right after the forward |
| captures held across forwards until the decode | | `decode_budget_gib` (1.0 here), and a group over it alone is decoded at once |
| the decoder: row caches ~1 MB per generated row (q, k, v, g, beta over 48 layers) x 6 sequences x (max_tokens + 1) per slot, K/V, states | 10 slots x 64 tokens = 4 GB in one decode | `decoder_budget_gib` (2.5 here): `Scorer._x_flush` decodes in chunks, a group with too many slots in runs of its prompts; a chunk is costed as the decoder pads it (below) |
| media embeddings cached for the call | 10 MB per auto image: 96 images = 1 GB | `MEDIA_CACHE_GIB` 0.5 (`scorer.py`) |
| allocator fragmentation | 2-3 GiB "reserved but unallocated" at the failures | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**The chunk is costed padded (2026-10-11).** The decoder pads every group of a chunk to the chunk's largest prompt count (slots x
beams), max_tokens, template total and prefix; the chunker summed each group's own size. A request with 16 frames and a
transcript, 9 scored reads, 5 searched slots at max_tokens 12-16 and 2 generated at 64 arrives as two groups whose own sizes sum
to 2.46-2.50 GiB: under the 2.5 GiB budget they were decoded together, and the decoder allocated 4.6-4.9 GiB (every searched slot
got 65 rows instead of 17, the generated group 30 slots instead of 12) for a 22.3 GiB peak and an out-of-memory error on 45 of 60
such requests; slightly longer transcripts pushed the sum over 2.5 GiB, decoded in two chunks and peaked at 19.4 GiB, which looked
random. `decode.chunk_bytes` + the stack's `chunk_extra_bytes` (`models/base.py`) now mirror the decoder's allocations for the
whole chunk; the two groups decode in two chunks of ~1.0 GiB each. Replayed in process on the real bodies: 11 of 24 calls failed
before, 0 of 36 after (peak 19.4 GiB).

**The sweep after the changes** (expandable segments, budget 16k; content within the hosted limits: 10 slots x 64 tokens per
content, images at their detail budgets; "generate" slots sample n = 6 sequences):

| shape | inputs | content tokens | images | slots | peak GiB | wall s | decode chunks |
|---|---|---|---|---|---|---|---|
| text 2k .. 15k, 1 scored query | 1 | 2,030 .. 15,318 | - | - | 18.6 .. 19.2 | 0.4 .. 3.0 | - |
| text 15k, 10 searched slots x 64 | 1 | 15,318 | - | 10 | 20.6 | 11.9 | 4 |
| text 15k, 10 generated slots x 64 | 1 | 15,318 | - | 10 | 20.4 | 12.5 | 4 |
| 3 high images, 10 generated slots | 1 | 14,324 | 3 high | 10 | 20.4 | 11.9 | 4 |
| 48 low images, 10 generated slots | 1 | 14,414 | 48 low | 10 | 20.5 | 8.1 | 4 |
| 16 inputs x 900, 10 generated slots each | 16 | 14,886 | - | 160 | 20.4 | 74.3 | 32 |
| 4 inputs x 15k, 10 generated slots each | 4 | 61,272 | - | 40 | 20.5 | 47.9 | 16 |
| 8 inputs x (12 auto images + 1k), 10 slots each | 8 | 106,640 | 96 auto | 80 | 20.8 | 111.0 | 32 |

No shape runs out of memory; the worst peak is 20.8 GiB on 23.5 (before the changes the same shapes failed at 21-22 GiB, and the
10-slot x 15k shape failed alone). Memory is per call (calls are serialised on the GPU), so the server's concurrency does not
change it. The idle level creeps from 16.63 to 16.86 GiB over a sweep (compiled kernel variants and workspaces for new decoder
shapes), not per call. The 12B (fp8s, idle 12.1 GiB) on the same shapes: peaks 15.5-18.0 GiB, one decode chunk for 10 slots (its
decoder has no recurrent rows), 3.7 s for the 15k-token 10-slot request, 25 s for 16 inputs x 10 slots.

**What is not bounded: time.** A Qwen decode step streams all the int4 weights (13 GB) whatever the slot count, so a 64-token
decode is ~3 s per chunk and the chunks multiply with the slots: 10 slots x 64 tokens on a 15k content = 12 s (3 s without the
slots), 16 inputs x 10 slots = 74 s, 8 inputs with 12 images and 10 slots each = 111 s. A front with a deadline (the hosted
service's is 30 s per engine call) caps the slots per request accordingly, or gives extraction requests a longer one; the
per-content limits (10 slots, 64 tokens, 16k content) and images need no change on 24 GB.

## 12. Open

- The block GEMM of down / out / o_proj (Triton, ~335 TOPS) against cuBLASLt's ~580: the remaining GEMM-side lever (§7.4).
- The decode step: the w4 kernel at half the card's bandwidth, the ~6,600 small kernels (a compiled step), the lm head read per
  step (§9); fewer, larger decode chunks (rows cached without q, a smaller beam for generated slots) would move the time bound of
  §11, not the memory bound.
- `max_tokens` is left at the 16k budget on 24 GB until a longer solo stream is measured.
- The `levels` path's fla l2norm runs per call, autotuned on the call's length; it was invariant on the decision-benchmark jobs,
  not checked on other workloads (the default `indexed` is).
- The Gemma stack pays the same 8 % of alignment pads in every GEMM; compact rows there would need the same scatter around its
  flex layers inside the transformers forward, not done.
- Other cards: the GEMM-floor sweep of §5 (f16) and the invariance checks are of one 4090 and this torch / cuBLAS; i8's
  invariance is by construction (exact integer blocks, fixed summation order) but the kernel configurations picked at load are
  part of the numerics, so scores differ across card architectures (the fingerprint includes the capability).
- Decision models on this stack (a checkpoint with `readout.safetensors` + `decision_config.json` instead of `lm_head.weight`,
  such as `spinf/pplx-decider-v1-27b-int4`) load with the readout as the head: `scorer.readout_logprobs` puts the codes' logits /
  temperature on the 255 answer-code tokens and -inf elsewhere, so an option set renormalised over its codes is exactly the
  decider's masked softmax; extraction slots are refused. Its own outlier table is exported like the base model's.
