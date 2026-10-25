# Precision: what moves a score, and by how much

This document establishes which parts of the numerics move a log-probability and by how much, measured on the Gemma 4 12B
engine. Three things do: the kernel variant a fresh `torch.compile` lands on (two variants, ~0.12 logit apart on fp8, fixed by
inductor's deterministic mode applied as a compile option); the packing noise of a read scored next to other work (fp8 ~0.13
logit mean, w8a16 with fp32 logits ~0.025: the fp8 matmuls are the source, not the lm head and not the activation scale, and the
calibrated `fp8s` scheme of `BATCH_INVARIANCE.md` removes it altogether); and the lm head's output precision (`f32out` trims
~10 % of the noise for free). It also records how an answer word is read: every case and spacing variant of it summed, because
a model spreads an answer over its spellings; and that the question template, not the content, sets most of the `residual`
and moves `p` between templates by up to 20 points.

**Setup** (2026-09-24): one RTX 4090, torch 2.14.0+cu130, Gemma 4 12B, a fresh inductor/triton cache per run unless stated.
Workloads: the maintainers' news fixture (18 articles x 1,328 two-option cells, `"{content}\n\nAs a result, {F} is likely to"`
+ " rise" / " fall") and a product-style yes/no set (the same 18 articles x 12 statements, content + floor = 432 reads, through
the interim yes/no reading of an early version of the engine). The scripts are the maintainers' research scripts; the parity run
itself is `spinf-serve parity` (`parity.py`).

## 1. Parity with the maintainers' original scorer

The engine reproduces the maintainers' original scorer bit for bit when both land on the same compiled kernel variant: fp8
1,328/1,328 cells identical (first run), w8a16 1,328/1,328 identical against the original's second run. The packing is identical
by construction (`packer.py` keeps the original's stream layout; the CPU tests check the trees and streams, `tests/test_packer.py`).

## 2. Compile variants: a fresh compile picks one of two

The original scorer, same source, fresh cache per run:

| scheme | runs | result |
|---|---|---|
| fp8 | 4 | 2 on variant A, 2 on variant B; A vs B: mean abs raw diff 0.122, corr 0.9964, sign agreement 97 % |
| w8a16 | 3 | 2 on one variant, 1 on the other; mean abs raw diff 0.041 |

So a fresh compile (every new serverless worker, every cleared cache) scored with a random variant. The "fp8 variant flips on
incidental source changes" of an earlier port's notes is most likely this randomness: inductor's autotuner times the candidate
kernels on the device at their first launch, and the winner, which sets the summation order of a reduction, depends on what else
the card was doing.

**The fix: inductor's deterministic mode as a compile option.** `torch._inductor.config.deterministic = True` (or
`TORCHINDUCTOR_DETERMINISTIC=1` in the environment) makes inductor skip on-device benchmarking where it changes numerics. Under
it the original scorer landed on variant A three times out of three; the engine is identical run to run, but on variant B,
whether the flag is set in code or in the environment; the cause of the A/B split between the two codebases under the flag was
not found. No measurable speed cost (fp8 10.26-10.30k tok/s with, ~10.36k without; 4090, the fixture).
`EngineConfig.deterministic_compile` is on by default (decided 2026-09-24).

**Correction (2026-10-02, `BATCH_INVARIANCE.md` §5): the flag never reached the kernels.** Dynamo re-applies
`torch.use_deterministic_algorithms(<saved state>)` around every frame it traces, and in torch 2.14 that call writes inductor's
`deterministic` (False) before the kernels are generated: every generated kernel said `"deterministic": False`, and the
run-to-run identity above held on one quiet host by the autotuner's luck. The setting now goes in as a **compile option** of every
`torch.compile` in the engine (`models/base.py compile_options`: the decoder layers, flex attention, the activation quantizers),
which inductor applies inside the compile. After it all kernels are deterministic, and fresh processes on the same card model
agree bit for bit.

**A persistent compile cache keeps an older variant (found 2026-09-25).** A deployment that kept its inductor/triton cache in a
volume across engine builds gave, on the same GPU and image, p(rise) 0.638 from the kept cache and 0.695 from a fresh compile on
the same request (both deterministic). Start every process with a fresh compile: the serverless workers do, the long-running
servers now do too, and they score alike.

Consequence: parity with the original scorer is "identical up to the compile variant" (zero bias, ~0.12 logit per read on fp8).
A comparison between two engines must be run under deterministic mode on both, and settle which variant is the reference.

## 3. Packing noise: the same work scored alone vs packed with other work

Mean abs difference against each job scored alone (its own stream). "raw" = the log-odds between the two options; p in
percentage points. Scoring the same job alone twice is bit-identical in every scheme.

| scheme, lm head | tok/s (packed) | fixture raw | fixture p mean / max | yes/no raw | yes/no p mean / max | yes/no flips /432 |
|---|---|---|---|---|---|---|
| fp8, bf16 (the original production setting) | ~10.4k | 0.135 | 1.75 / 20.6 | 0.141 | 3.19 / 21.3 | 45 |
| fp8, f32out | ~10.4k | 0.122 | 1.55 / 18.2 | 0.128 | 2.88 / 20.3 | 27 |
| fp8, fp32 weights | | 0.122 | 1.55 / 18.2 | 0.128 | 2.88 / 20.3 | 27 |
| fp8row (per-token act., per-channel weight scales), f32out | 3.7k | 0.112 | 1.42 / 12.3 | 0.125 | 2.81 / 13.8 | 31 |
| w8a16, bf16 | ~6.6k | 0.044 | 0.58 / 6.2 | 0.059 | 1.31 / 9.3 | 22 |
| **w8a16, f32out** | ~6.6k | **0.024** | **0.31 / 3.1** | **0.025** | **0.54 / 7.4** | 21 |
| fp8, f32out, `TORCHINDUCTOR_BATCH_INVARIANT=1` | 10.1-11.3k | 0.119 | 1.51 / 16.7 | 0.131 | 2.97 / 17.1 | 44 |
| w8a16, f32out, batch-invariant | ~6.6k | 0.025 | 0.32 / 6.6 | 0.027 | 0.60 / 5.1 | 5 |

Reading:

- **The bf16 lm head is not the main source** (the first hypothesis, wrong): fp32 logits remove ~10 % of the fp8 noise.
  `f32out` (the bf16 matmul written to fp32 logits, `torch.mm(..., out_dtype=float32)`) equals full fp32 weights exactly, at
  no cost.
- **The per-tensor fp8 activation scale is not the main source either**: per-token scales (fp8row) barely help and run at 35 %
  of the speed (cuBLAS row-wise scaling on Ada).
- **The fp8 matmuls are.** w8a16 (bf16 matmuls, fp32 accumulation) is ~5x quieter; there the lm head matters (0.044 -> 0.024).
  The residual noise follows the stream shape (the GEMM algorithm per M, the attention block layout), which inductor's
  batch-invariant mode does not touch (cuBLAS and `_scaled_mm` are extern kernels). Likely the limited accumulation precision of
  the fp8 tensor-core MMA on Ada, with the split and tiling varying per M (an inference, not measured directly).
- The yes/no reading is noisier in p than the news cells because its p values sit nearer 0.5 (the interim template, §6), where
  dp/draw is largest.

The packing noise of dynamic fp8 was later taken apart cause by cause and removed: the fixed activation scales of `fp8s`, the
128-token job alignment and the fixed lm-head row count of `BATCH_INVARIANCE.md` make a read bit-identical whatever it is packed
with, at the speed of fp8.

## 4. What it means for a user of the engine

- **`fp8s` with aligned jobs** (the default where the card has fp8 tensor cores and the checkpoint its scales) is batch-invariant:
  a read does not move with its neighbours, so the noise of this table does not apply. The one-time drift from dynamic fp8 to
  fp8s is ~0.15 raw on news, the size of fp8's own packing noise.
- **Dynamic `fp8`** (`--quant fp8`): the first row of the table, with `f32out`: ~1.5-3 pp mean movement of p with the packing,
  up to ~15-20 pp on rare reads, zero bias.
- **`w8a16` + `f32out`** is ~5x quieter than fp8 (~0.3-0.6 pp mean) at ~0.64x its throughput, and is what the engine picks on
  cards without fp8 tensor cores (Ampere). Scored alone (one request per stream) it is bit-identical run to run under
  deterministic compile; it is not batch-invariant (a bf16 GEMM's rows change by an ulp with M, `BATCH_INVARIANCE.md` §7).
- **`lm_head = f32out`** is the default: free, ~10 % less noise than bf16 logits. `bf16` remains for parity with references
  recorded with bf16 logits (log-probs then land on 1/8 steps above |16|).

## 5. Answer case: an answer word is read as the sum of its spellings

Setup: fp8, one RTX 4090, the 18 fixture articles x 12 statements, the full next-token distribution at the answer position with
fp32 logits. Every case (lower, Capitalized, UPPER) x spacing (leading space or not) variant of yes, no, true and false is **one
token** in Gemma 4 (24/24). Mean mass after the template's "Answer:":

| yes | mass | no | mass |
|---|---|---|---|
| " Yes" | 0.229 | " No" | 0.146 |
| " yes" | 0.018 | " no" | 0.010 |
| " YES" | 0.006 | " NO" | 0.008 |
| "Yes" | 0.001 | "No" | 0.0005 |
| "yes", "YES" | < 0.0001 | "no", "NO" | < 0.0001 |

On empty content (the floor) " NO" (0.016) outweighs " no" (0.006). Summing all six variants of each answer instead of
" Yes"/" yes" and " No"/" no" moves p by 0.6 pp on average (max 2.1 pp, 3 flips in 216) on content, 1.1 pp on the floor.

**Adopted:** an option is read as the log-sum-exp of every case and spacing variant of it that is a single token
(`options.py case_variants`; the compiler applies it to every option, `compiler.py`), which makes the reading case-insensitive
and stops a score from depending on the spelling the model happens to prefer. Single-token variants share the option's read row,
so the only cost is lm-head columns. A request sets `"case_insensitive": false` to read exactly the spellings it lists (needed
when two options differ by case only); the `logprob` field reports the summed value, `p` the normalisation over the options.

## 6. The template decides the residual, and moves p

Same setup, the 12 statements read as yes/no after the content:

| template | per-query overhead (tokens) | mass on yes+no | residual | p corr. with T0 | mean abs p diff vs T0 |
|---|---|---|---|---|---|
| T0: `…Statement: {q}\nQuestion: Is the statement true according to the text above?\nAnswer:` | 20 | 0.42 (+0.20 on true/false) | **0.58** | 1 | 0 |
| T1: T0 + ` Answer yes or no.` | 25 | 0.63 | 0.37 | 0.85 | 8.0 pp |
| T2: `Does the text above support the statement "{q}"? Answer Yes or No.\nAnswer:` | 18 | 0.52 | 0.48 | 0.33 | 20.3 pp |
| T3: `{q} Yes or no?\nAnswer:` | **8** | **0.75** | **0.25** | 0.34 | 18.3 pp |
| T4: `Statement: {q}\nTrue or false?\nAnswer:`, read as true/false | 12 | 0.72 (true/false) | 0.28 | 0.69 | 9.1 pp |

- **T0 leaks half its answer mass to " True"/" False"** because it asks "Is the statement true…". Its residual (0.58) is mostly
  the template, not the question.
- **The residual barely varies by statement** (T0: 0.53-0.65 across the 12 statements, including "The article is about a sports
  event." on oil news). A reading of `residual` as "high = the question does not apply to the content" is not supported across
  questions; it holds only within one question, as a content whose residual is far above that question's usual level
  (`calibration.md`).
- **Templates disagree on p** (correlation 0.33-0.85 with T0). Which one is right needs labels, not agreement: residual and cost
  only rank them. T3 is the cheapest (8 tokens against 20) with the lowest residual, but it is yes-biased (mean p 0.74).

The user-facing advice that follows from this (name the answers in the question, keep the template short, compare `p` to
`floor_p` rather than reading the residual's size, check a template on a few labelled examples) is in `writing-questions.md`
and `calibration.md`.

## 7. Open

- The A/B variant split between the original scorer and the engine under deterministic mode.
- The fp8 noise mechanism: the accumulation precision of the fp8 MMA per tile and split (inferred from the pattern, not
  measured in the kernel).
- w8a16 as a batch-invariant scheme: every linear at a fixed row chunk, as the lm head does (`BATCH_INVARIANCE.md` §7), is
  untested.
