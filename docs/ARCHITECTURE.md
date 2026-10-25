# spinf-serve: architecture

spinf-serve is one process on one GPU that reads a piece of content once and answers many questions about it as
probabilities. This document is the map of the code: the primitive the engine computes, the path a request takes through the
package, how prompts are packed into one forward pass, what a model stack is, the quantization schemes and their tables, and
the rules that keep the scores reproducible. The measurements behind the design are in the sibling documents:
[BATCH_INVARIANCE.md](BATCH_INVARIANCE.md), [PRECISION.md](PRECISION.md), [QWEN.md](QWEN.md), [EXTRACTION.md](EXTRACTION.md).
The user-facing API is in [scoring-api.md](scoring-api.md) and [configuration.md](configuration.md).

## 1. The primitive

One **job** is one piece of content asked several **prompts**. A prompt is the full text up to the scored slot; an **option** is
the exact string that may follow it (leading space included when it follows a word). The engine returns `log P(option |
prompt)` for every option: the sum of the option's per-token log-probabilities, multi-token options by the chain rule. Nothing
else happens at this level: the normalisation over the options (`p`), the residual, the floor on empty content and the usage
accounting belong to the layer above (the compiler), and the engine never sees a template.

The primitive is exposed as is for research and the parity runs: `ScoreService.score` (service.py) and `POST /v0/score` take
`{"jobs": [{"id", "prompts": [...], "options": [[...], ...]}]}` with the content repeated inside every prompt. It is wasteful
over a network, so the public API is the protocol path below.

## 2. The request path

| step | module | what happens |
|---|---|---|
| parse | `protocol.py` | the request shape (`messages` or `inputs`, `scoring.queries`, placeholders and combinations, extraction slots, media parts, `limits`); every check that needs no tokenizer. A front end may run the same checks first; the engine repeats them |
| prepare | `server/api.py` | the public body becomes the protocol request: `messages` normalised to one input, the packs expanded (`packs/refs.py`: the content wrapped in the pack's context, the pack's queries added in their own group), the server's features and limits applied, a trusted caller's per-call limits capped |
| compile | `compiler.py` | the content is tokenized once and each filled template on its own, then joined; the join is checked against a joint tokenization and a template that merges with the content's end is tokenized whole. Options become their case variants (`options.py`). Token limits are enforced here, before any GPU work. Each input is one grouped job (content = shared prefix, prompts = branches); the floors (the filled templates without content) are one job of the request's distinct floor prompts; extraction slots become `ExtractJob`s whose templates are branches of the same tree |
| pack | `packer.py` | jobs become trees and tries, trees are packed into streams of at most `token_budget` tokens (§3) |
| score | `scorer.py` | one forward per stream through the model stack (§4), the option log-probabilities read at the branch ends through the lm head or the readout, the K/V of extraction branches captured |
| decode | `decode.py`, `extract.py` | the extraction slots of the whole call decoded together in batched beam steps (EXTRACTION.md) |
| respond | `compiler.py`, `server/api.py`, `packs/pipeline.py` | per option `logprob`, `p`, `residual`, `floor_p`; the packs' category scores and decision computed from their queries' answers; the usage and timing blocks; the public envelope |

`service.py` is the one entry point the adapters call (`ScoreService.score_request`): the HTTP server (`server/app.py`) and
any serverless handler wrap it. GPU work is serialised per process (one request on the card at a time, a bounded queue in front);
the batches of a request are tokenized and packed in a thread ahead of the GPU (`Scorer._execute`, `EngineConfig.prefetch`).

## 3. Packing and tree attention

The prompts of a job share their longest common token prefix, normally the content, which is sent once; every prompt continues
as its own **branch** that attends to the prefix and to itself. An option of several tokens is read by the chain rule on a branch
extended by the option's leading tokens, shared by options with the same lead-in. Grouped jobs go further: after the content, the
filled templates and their lead-ins form a token **trie** (`packer.trie_branches`), so a template's stem shared by its
combinations, an instruction repeated by several queries and a prompt under each multi-token lead-in run once. In a news
workload with 74 questions per article this takes the tokens actually run below the tokens the caller is billed for (0.66-0.80x).

The stream lays each trie out in pre-order, so a token's descendants are the contiguous run up to its subtree's end (`Batch.end`)
and the attention mask needs one rule: q sees k when k <= q, same job, q < end[k]. The mask is a FlexAttention block mask over
128-token blocks (`packer.BLOCK`, `packer.block_tables`); sliding-window layers get their own tables, always built. Jobs are
packed back to back into streams of up to `token_budget` tokens (16,384 by default: the fastest for news-length content on a
24 GB card). With `align_jobs` (the default) every job starts on a block, streams are at least `min_stream` tokens and the lm head
always multiplies `lm_rows` rows: the layout of a job then does not depend on its neighbours, which is half of batch invariance
(BATCH_INVARIANCE.md §2-3). A job over the budget is split into contiguous runs of prompts that each re-send the prefix
(`split_trees`); an input longer than the budget, up to `max_tokens`, runs alone in its own stream.

**Query groups.** A pack's queries carry a group name; per input, the ungrouped queries come first, laid out exactly as without
groups, and each group follows with its own trie, starting on a block and attending to the content and itself only. A pack
therefore scores the same whether or not the caller added questions of their own (BATCH_INVARIANCE.md §6).

## 4. The model stacks

`models/base.py Stack` is the interface, one implementation per architecture family, picked by the manifest's `family` or, for a
checkpoint run by its path, by the family whose `detect()` recognises its `config.json` (`models/registry.py`). A stack builds its
model from (`EngineConfig`, snapshot directory, device) and gives the Scorer what differs between families: image preprocessing,
the host-side plan of a packed batch, the staging and forward of a stream, the capture of extraction ranges, and the decoder that
consumes them. Everything above `models/` works on the packer's arrays and hidden rows; it knows the head is an LM head or a
readout and nothing else about the family.

- **`models/gemma4.py`** (Gemma 4 12B and 31B): the text model built from transformers' Gemma 4 modules, its linears replaced by
  the scheme's quantized linears (`quant/linears.py`), the decoder layers compiled with torch.compile under inductor's deterministic
  option (`compile_options`). Attention over the packed stream is FlexAttention with the tree mask; the 12B's "unified"
  checkpoint reads images, audio and video frames through its own embedders, inline in the token sequence between marker tokens
  (`media.py`); the hook around flex attention captures K/V ranges for extraction (`CAPTURE`). The lm head reads the branch
  ends only, in fixed row chunks, in `f32out` (bf16 matmul written to fp32 logits).
- **`models/qwen35/`** (Qwen3.8-27B and the decision models built on it): 48 Gated DeltaNet layers that cannot be masked, so
  `TreePlan` cuts the stream into segments and runs them level by level, each from its parent's final state (`gdn.py`, with one
  vendored and modified fla kernel, `NOTICE`); the 16 full-attention layers take the tree mask unchanged. The GPTQ int4 weights
  stay packed on the card and run through fp16-accumulate GEMMs (`f16`) or int8 tensor cores with block scales and an
  outlier-channel split (`i8`, the default; `quant/i8gemm.py`). Images go through the checkpoint's vision tower with 3-D rotary
  positions (`qwen_vision.py`); extraction decodes by re-running the recurrence from the captured prefix state
  (`qwen_decode.py`, `quant/w4gemm.py`). QWEN.md has the design and the numbers.
- **Readout heads.** A decision model (`pplx-decider`) ships `readout.safetensors` and `decision_config.json` instead of an lm
  head. The stack loads the readout as the head and `scorer.readout_logprobs` puts the codes' logits over their temperature on the
  answer-code tokens and -inf elsewhere, so an option set renormalised over its codes is exactly the model's masked softmax.
  Extraction is refused on these models.

## 5. Quantization schemes, tables, checkpoints

| scheme | stack | weights | activations | note |
|---|---|---|---|---|
| `fp8s` | Gemma | e4m3, per-tensor scale | e4m3, a **fixed calibrated scale per linear** | the default on cards with fp8 tensor cores; batch-invariant; same speed as fp8 |
| `fp8` | Gemma | e4m3 | e4m3, the stream's max per forward | vLLM's `--quantization fp8` numerics; a job's scores move with its neighbours |
| `fp8row` | Gemma | e4m3 per channel | e4m3 per token | 0.35x speed on Ada, not quieter: research only |
| `w8a16` | Gemma | int8 | bf16 | the quiet scheme (~5x less packing noise than fp8), ~0.64x throughput; the choice on Ampere |
| `w8a8` | Gemma | int8 | int8 per token | research |
| `bf16` | both | bf16 | bf16 | the reference numerics; needs the memory |
| `f16` | Qwen | int4 GPTQ, dequantized per call | fp16, fp16 accumulation | the first Qwen default; batch-invariant with a 5,248-row stream floor |
| `i8` | Qwen | int4 GPTQ re-rounded to int8 per channel | int8 per token and 128-channel block, 64 outlier channels exact on the norm-fed linears | the Qwen default: 1.4x `f16`, exact per K block so invariant for any stream length |

`EngineConfig.quant` empty means auto: `fp8s` where the card has fp8 and the model its scales, `i8` for an int4 checkpoint with
its outlier table, else `w8a16` or `bf16`. The lm head is `f32out` by default (PRECISION.md §3).

**Where the scales live.** The published `spinf/...` checkpoints carry everything their scheme needs inside the snapshot: the
fp8-static ones hold fp8 weights with `weight_scale` and `input_scale` per linear in the compressed-tensors layout (`quant/fp8.py`),
the int4 ones hold `spinf_outliers.json`. The engine reads them from the checkpoint and nowhere else. For an original bf16 Gemma
checkpoint run by its Hub id, `fp8s` takes a scale table from `quant/tables/<org>__<model>/<revision>.json`, made by
`spinf-serve calibrate` over a corpus of everything the model will see (long documents set most maxima: BATCH_INVARIANCE.md); a
third-party int4 Qwen copy run by its id takes its outlier table from the same folder. `quantizations/` holds the recipe, the
calibration output and the model card of every published checkpoint, and the maintainers' scripts that made them.

**Manifests** (`manifests/*.toml`) pin each served model to a Hub repository and revision, so the weights, the scales they carry
and the parity references always name the same tensors; they also list the model's aliases, family, allowed schemes, media, image
scheme and default limits. `EngineConfig.from_manifest` applies one; a checkpoint without a manifest runs by its path with the
family detected.

## 6. Determinism, batch invariance, the fingerprint

Three properties hold on the default configuration, each measured in its document:

- **Run to run:** a request scored alone is bit-identical across runs and across fresh processes (PRECISION.md §2: inductor's
  deterministic mode as a compile option; without it a fresh compile picks one of two kernel variants ~0.12 logit apart).
- **Alone or packed:** with `fp8s` (or `i8`) and aligned jobs, an input's log-probabilities are bit-identical whatever else is in
  the call, and a pack's scores do not move when the caller adds questions (BATCH_INVARIANCE.md).
- **Across machines:** cards of one architecture score alike; across architectures (Ada vs Blackwell vs Ampere) they do not,
  because cuBLAS and Triton pick other kernels.

Every response carries `system_fingerprint`: `fp_` + a hash of what changes the scores (model, weights revision, scheme, scale
table, lm head, engine version and build, GPU compute capability). Two responses with the same fingerprint are comparable; the
engine logs what the hash stands for at start. The dependency pins of the `engine` extra are exact and change only together
with a parity run (§9).

## 7. Extraction and generation

An extraction slot (`{*}`, `{n}`, `{$}`) is a template branch like any query: the scoring pass captures the K/V of the prefix
and the template and the first-step log-probabilities, and all slots of a call are then decoded together, one step for all, each
slot's beams attending to its group's prefix, its own template and what it generated, nothing copied per beam. The search is
best-first over continuations in which every live beam may also end, so `P(answer)` is the exact probability of that string
followed by an end; a `generate` slot writes plain text instead. Memory is bounded by three budgets and the decode runs in chunks
costed as the decoder pads them. EXTRACTION.md has the design and the cost measurements; QWEN.md §9 the hybrid model's decoder.

## 8. Media

`media.py` decodes and preprocesses images and audio on the CPU; only `data:` URLs are accepted (the engine never fetches), audio
comes as base64 with its container format and is decoded by ffmpeg. The Gemma 12B reads images and audio inline through its
embedders, one placeholder token per soft token (image budgets by `detail`: 70 / 280 / 1,120 tokens; audio one token per 40 ms,
30 s per segment); the 31B reads images; the Qwen stack reads images through its vision tower at its own sizes (256 / 1,024 /
4,096 tokens per detail). Video is frames: the client samples them, and the engine renders each frame as an image after its
timestamp, in place in the content. A video token costs about 1.1-1.2x a text token on the 12B (EXTRACTION.md §5).

## 9. Parity and release rules

- **CPU tests** (`tests/`, run in CI): the protocol and compiler, the tree and stream construction, the block tables against a
  dense mask, the extraction grammar and search bookkeeping, the Qwen tree plan against a sequential recurrence, the pack
  evaluator against the official packs' golden fixtures, the server with a fake engine, the public hygiene scan.
- **GPU parity** (by hand, before anything that touches a kernel, a compile boundary, a scheme or a pin ships):
  `spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz --reference fixtures/<reference>` scores the public fixture
  (24 articles x 56 two-option cells, `fixtures/README.md`) alone and compares every cell with the recorded reference of that
  checkpoint and scheme. Expected on the same card architecture: `lp_exact_share` 1.0. A new scheme or checkpoint records its
  reference with `--record`, two passes agreeing.
- **Pins.** The `engine` extra pins torch, triton, transformers, tokenizers, safetensors and fla exactly; the scores come from
  these kernels, so a pin changes only together with a parity run, and a change to the recorded references is a visible event
  (and a new fingerprint for every server that upgrades).
- **The quantized linears' source is frozen** (`quant/linears.py`): editing a class body changes the compiled kernel variant and
  with it the fp8 numerics.
- **Packs:** a pack is fitted on one fingerprint; when a model's fingerprint changes, its packs' golden fixtures are re-run and the
  pack re-fitted or re-versioned if they no longer reproduce.

## 10. Layout

```
spinf_serve/
  protocol.py            the request as the engine receives it; checks without a tokenizer
  compiler.py            templates -> token tries, token checks, floors, extraction specs; the result -> per-option scores
  options.py             case-insensitive options (an answer summed over its case and spacing variants)
  packer.py              trees, tries, block tables, streams
  scorer.py              one GPU, one model: the forward loop, the reads, K/V capture, the extraction decode driver
  decode.py, extract.py  the batched decoder and the slot search / grammar / merging
  media.py               image and audio decoding (CPU)
  service.py             ScoreService: load, warm up, score_request / score, the fingerprint, the memory guard
  config.py              EngineConfig (hardware and numerics)
  weights.py             where snapshots are found (cache roots, the Hub, pinned revisions)
  manifests/             the served models (TOML)
  models/                base.py (the stack interface), registry.py, gemma4.py, qwen35/ (qwen, gdn, qwen_decode, qwen_vision)
  quant/                 linears.py (the Gemma schemes), fp8.py (the fp8-static checkpoint layout), i8gemm.py, w4gemm.py,
                         tables.py + tables/ (per-revision scales and outliers for checkpoints run by their id), calibrate.py
  packs/                 the pack file, its validation, the evaluator (pipeline.py), a request's packs (refs.py), sources, catalog
  server/                api.py (the public API around the engine), config.py (ServerConfig), app.py (FastAPI endpoints)
  cli.py                 spinf-serve serve | score | packs | parity | calibrate
quantizations/           recipe, calibration output and model card per published checkpoint; the maintainers' quantize scripts
fixtures/                the public parity fixture and its references
benchmarks/              the throughput comparison against a completions server, the fixture builder
tests/                   the CPU suite
docs/                    the user guides and these engineering documents
```
