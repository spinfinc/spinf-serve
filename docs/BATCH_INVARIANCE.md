# Batch invariance: an input's scores do not depend on what it is packed with

spinf-serve packs every input of a call, and the inputs of several calls, into shared 16k-token streams. This document
establishes that with the `fp8s` scheme (fixed, calibrated activation scales) and aligned jobs, an input's log-probabilities
are **bit-identical** whether it is scored alone, packed with fifteen others or packed in another order, and that two fresh
processes of the same engine on the same card model produce the same bits. It records what moved the scores before, what each
cause cost to remove, and what the design costs in throughput. The Qwen stack reaches the same property by another route (its
int8 GEMMs are exact per block of input channels, so they have no row-count floor): `QWEN.md` §5 and §8.2.

Setup: one RTX 4090, Gemma 4 12B, torch 2.14.0+cu130 (2026-10-01/02). Test data: the maintainers' test bed (news articles,
moderation texts, multilingual texts, long documents up to ~55k tokens, odd inputs, images, audio, video; not redistributed).

## 1. The problem

With the dynamic `fp8` scheme, a text's scores moved with the rest of the call: `spinf/moderation` pack category scores up to
0.14 apart for the same text, and micro-layer decisions flipped. Measured on 64 moderation texts (pack 0.1.2): alone vs in a
call of 64, mean |d category score| 0.015, max 0.18-0.24; a few decision flips per 64.

## 2. Causes (each measured separately)

1. **The fp8 activation scale.** `fp8` quantizes each linear's input with the max over the whole packed stream. Alone vs in a
   batch, that max differed in 218 of 328 linears. `_scaled_mm` itself, with a fixed scale, gives bit-identical rows whatever
   the number of rows (tested M = 128 .. 16,384 on every 12B shape, with one exception, see 4).
2. **The attention block layout.** A job's position relative to the 128-token flex blocks moved with its neighbours, so its
   attention summed in a different order. Aligning every job to a block made "+1 query" bit-identical in every scheme.
   Related: sliding layers used the dense tables (with full blocks) when no job in the stream reached the window and sliding
   tables otherwise, so a long neighbour changed a short job's sliding layers.
3. **The lm head GEMM.** bf16 GEMMs (and row-wise fp8, the `fp8row` scheme) are not row-invariant: a row's result depends on
   M. The read rows of a forward went through the lm head in chunks whose size depended on the call.
4. **Short streams.** The fp8 GEMM of the full-attention layers' k/v projection (3840 -> 512) takes another kernel at M = 128
   and 256, so a tiny job alone (a 128-token stream) scored differently from the same job in a longer stream.

Not the cause: per-token activation scales (`fp8row`) remove 1 but not 3, and run at 35 % of fp8's speed (cuBLAS row-wise
scaling on Ada). Pack side, separately: the `spinf/moderation` evaluator read `p` (rounded to 6 decimals) and took logs of rare
options, which amplified any noise by up to ~4.8 z (fixed in the pack: 0.1.3 reads `logprob`).

## 3. The design: `fp8s` + aligned jobs

- `fp8s` (`quant/linears.py Fp8StaticLinear`): fp8 weights as `fp8`, and a **fixed activation scale per linear**. The
  published `spinf/gemma-4-*-fp8` checkpoints carry the scales inside the checkpoint as `input_scale` per linear
  (`quant/fp8.py`), the one path the engine reads them from; a bf16 checkpoint run by its path takes them from a scale table
  of its revision (`quant/tables/<org>__<model>/<revision>.json`). Same kernel as `fp8`, minus the per-forward max.
- `EngineConfig.align_jobs` (default on): every job starts on a 128-token block; sliding tables are always built
  (`packer.py block_tables(always_window=True)`); streams are at least `min_stream` = 1024 tokens.
- `EngineConfig.lm_fixed_rows` (default on): the lm head always multiplies `lm_rows` rows (a short chunk is padded).
- `lm_head` default `f32out` (no cost, ~10 % less noise than bf16, `PRECISION.md` §3).

`fp8s` is the automatic scheme on fp8 cards when the model has its scales; `fp8` stays available (`--quant fp8` or
`SPINF_QUANT=fp8`).

### Calibration

A scale must cover the largest input its linear will see: past 448 x scale the input is clipped. `spinf-serve calibrate`
(`quant/calibrate.py`) runs the engine with dynamic `fp8` (eager) over a corpus and records each linear's largest |input|;
scale = 2^ceil(log2(2 x max / 448)) (a power of two, at least 2x headroom). The corpus used for the published tables is a
stratified sample of the maintainers' test bed (news, moderation, multilingual, long documents up to ~55k tokens, odd inputs,
images, audio, video), 944k tokens. Long documents set the max of 159 of the 328 linears, ESC-50 audio 72: a corpus of short
texts alone would clip them (an early table from 32 short texts was up to 2.1x too small on some linears). A calibration corpus
must therefore include every kind of input the model will serve, and the engine's `max_tokens` must be set as it will be in
production: inputs a model refuses (audio on the 31B, documents over its maximum) are skipped one by one and never calibrate
anything.

Monitor (diagnostics only): with `fp8s_monitor_every` = N, one forward in N also records each linear's largest input / its clip
level, and the engine logs a warning above 1. It is **off in live inference**: the monitored forwards compile to another graph,
whose fusion changes the numerics (pass 1 vs pass 2 of the parity fixture differed with it on at 1/64; identical with it off),
and on every forward it costs ~18 %. Release checks run the test bed with it on: the full test bed (7,841 news articles, 2,200
moderation texts, ... 13M tokens) peaked at 0.94 of the clip level (layers.1.mlp.down_proj): nothing clipped.

## 4. Results (test bed v1, one 4090)

Invariance, fp8s + alignment: alone vs packed (16 per call) vs reversed (7 per call), log-probs **bit-identical** on moderation
(29k values), the news suite (8k), long documents, images.

Speed: the test bed's `speed` mode, in process, real (unpadded) tokens per second of wall time, compile excluded, best of two
warm passes; the news suite = 1,024 articles at 64 per call, moderation = 1,024 texts with the pack's 106 prompts at 32 per call.
One 4090; single runs, about ±1-2 % noise.

| engine | news tok/s | moderation tok/s |
|---|---|---|
| fp8, per-stream activation scale, dense packing, bf16 lm head (the original scheme) | 9,489 | 9,621 |
| fp8s, diagnostic peak monitor on every forward | 7,878 | 7,886 |
| fp8s, monitor off | 9,522-9,629 | 9,643-9,674 |
| **fp8s + deterministic compile options + query groups (the shipped configuration)** | **9,874** | **9,832** |

What each piece costs (fp8s, news, monitor off unless said): no alignment 10,344 tok/s; + 128-token alignment 9,771 (-5.5 %);
+ fixed 1,024-row lm head 9,727 (-0.5 %); the monitor on every forward 7,881 (-19 %, a second pass over every linear's input).
The fixed scales gain about what alignment costs: they drop the per-forward max over the stream (in the first experiment,
dynamic fp8 aligned 8.8k vs fixed scales aligned 9.4k). Deterministic compile: no measurable cost. f32out vs bf16 lm head: ~1 %.
On the maintainers' news fixture (18 articles x 1,328 two-option cells): fp8 ~10.4k, fp8s 10.5-10.8k. Rejected on speed: fp8row
(per-token scales) 3.6k (cuBLAS row-wise scaling on Ada), w8a16 f32out 6.2k (5.7k aligned). Not measured: a single tiny input's
latency (a stream is now at least 1,024 tokens: a few ms), images / video / long documents alone, other cards. 31B on an L40S: §7.

Accuracy: fp8s = fp8 within noise on every labelled suite (imagenette 0.94/0.94, imagewoof 0.63/0.64, multilingual, GTZAN,
UCF101), and as close or closer to w8a16 f32out (the quieter scheme) than fp8: news raw 0.140 vs 0.153, long documents 0.141 vs
0.177. Drift fp8s vs fp8: ~0.15 raw on news (the same size as fp8's own packing noise): the scores change once.

## 5. The same numbers in every process (2026-10-02)

Two fresh containers of the same image on the same card model answered with different scores (mean |d log-prob| 0.12-0.19 on
400 moderation texts), each stable and batch-invariant on its own: at least four discrete variants over nine fresh starts, while
every local run landed on the same one. The compile caches of the variants were identical except for the autotuner's
`.best_config`: the RMSNorm reduction (`triton_red_fused__to_copy_mean_mul_pow`) was R0_BLOCK 4096 in one process and 2048 in
another, picked by on-device timing at its first launch, and the block size sets the summation order.

Inductor's deterministic mode filters reduction configs to one choice, but it was never on: every generated kernel said
`"deterministic": False`. `EngineConfig.deterministic_compile` set `torch._inductor.config.deterministic = True` (and
`TORCHINDUCTOR_DETERMINISTIC=1` in the environment does the same), but Dynamo re-applies `torch.use_deterministic_algorithms(
<saved state>)` around every frame it traces, and in torch 2.14 that call writes inductor's `deterministic` (False here) before
the kernels are generated. The fix: `{"deterministic": True}` as a **compile option** of every `torch.compile` in the engine
(`models/base.py compile_options`: the decoder layers, flex attention, the activation quantizers), which inductor applies inside
the compile. After it, all 59 kernels are deterministic and fresh processes agree bit for bit. This predates fp8s:
`PRECISION.md` §2's "identical run to run under the flag" held on one quiet host by the autotuner's luck, not by the flag.

## 6. Query groups: a pack scores the same with or without the caller's queries

Batch invariance is across inputs. Within one input, the prompts of all its queries share one tree: a caller query added to a
pack call shifted where the pack's branches sat relative to the blocks, and could share trie nodes with them (pack 0.1.2
category scores moved up to 0.06 with one extra caller query). `protocol.py Query.group` (optional): per input, the ungrouped
queries come first, laid out exactly as without groups; each named group follows in name order with its own trie, starting on a
block. A group only attends to the content and itself, and the content's last block is partial in both layouts (masked entries
add exact zeros), so its numerics do not depend on the other groups. The shared prefix is now the content alone (the templates'
common start is a trie node, the same tokens), so the ungrouped queries' layout does not depend on whether groups follow. Cost:
up to one block of padding per group (~64 tokens; a pack call has ~1,700 tokens of pack prompts). The server puts a pack's
queries in the group `"<pack>@<version>"` and reserves it in calls using the pack (`server/api.py`).

Measured (one 4090, 48 moderation inputs, pack 0.1.2 queries in a group): with one caller query, the pack's log-probs are
bit-identical with or without it, alone or 16 per call, caller query before or after the pack in the request; the caller's own
scores are identical with or without the pack. With a 3-combination caller query, 47 of 48 inputs are bit-identical and one
(496 content tokens) moves (max 1.1 log-prob on its pack queries) although its pack layout, tables and read rows are the same:
open, see §7.

Not covered: a caller template that merges with the content's last token (the content is then cut a few tokens early for every
prompt of the input, packs included); the empty floor (one job per request over all queries); splits of an input over the
budget.

## 7. Open

- The one input of §6 whose pack scores still move with a caller query (stream 1,664 vs 1,792 tokens, identical pack layout and
  attention tables; the lm head is row-position invariant): to bisect layer by layer.
- 31B: table calibrated 2026-10-02 on an L40S (the test bed corpus, 810k tokens; audio and the 3 long documents over its 32k
  limit skipped; long documents set 183 of 410 maxima, images 151). Gate on the L40S: bit-identical alone / packed / reversed on
  6 suites, 3.76k tok/s vs 3.34k for fp8 (+12 %), accuracy equal within a few items, clip peak 0.53 (layers.17.mlp.down_proj).
- 31B on Ampere (A6000 / A40): no fp8. FP8 tensor cores (e4m3 / e5m2) start with Ada (sm_89) and Hopper (Blackwell has them
  too); Ampere (sm_80 / 86) has FP16 / BF16 / TF32 and INT8 tensor cores only, and `torch._scaled_mm` refuses fp8 there, so the
  automatic scheme picks w8a16 (int8 weights expanded to bf16 in the matmul). A bf16 GEMM's rows change by one bf16 ulp with M
  (cuBLAS picks another kernel per shape, §2.3), so w8a16 is deterministic but not batch-invariant, and its scores differ from
  the Ada cards' (the fingerprint includes the capability). The alternative, every linear at a fixed row chunk as the lm head
  does, is untested.
- 31B on Blackwell (RTX PRO 6000 Server MIG 2g.48gb, sm_120), gate 2026-10-02 with the same table: bit-identical alone / packed /
  reversed on the same 6 suites; 4,168 tok/s news, 4,028 moderation (L40S 3,756 / 3,743; fp8 on the same MIG 3,192 / 3,159, so
  fp8s is +27-31 % there); accuracy as on the L40S within an item or two; clip peak 0.52. Its scores differ from Ada's by
  ~0.10-0.14 raw (another architecture: other kernels), so it is a second engine with its own fingerprint. A fleet that must be
  one engine picks one architecture.
- The extraction decode (batched across trees) is not covered by this; the extraction answers' scores were equal alone vs packed
  on the odd-inputs suite, not proven in general (`EXTRACTION.md`).
- Other cards (L40S, RTX PRO 6000 Blackwell): the same design; invariance per card must be re-checked (cuBLAS picks kernels per
  architecture), and scores differ across architectures as before (the fingerprint includes the capability).

One operational consequence: the fingerprint hashes what changes the scores (weights revision, scheme, lm head, engine build),
so a pack fitted on one fingerprint should be re-run against its golden fixture whenever the engine behind it changes
(`ARCHITECTURE.md`).
