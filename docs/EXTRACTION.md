# Extraction: slots decoded from the same read as the questions

Extraction slots (`{*}` text, `{n}` numbers, `{$}` amounts, and `generate` on a text slot) ride the pass that scores an input's
questions. A slot's template is one more branch of the input's scoring tree; the packed forward captures the branch's K/V and the
log-probabilities at its end, and when the pass is over the slots of every input in it decode together, one batched beam step at a
time, each query attending to its own prefix, template and generated tokens without copying any of them. A searched slot costs
about 200 to 600 input tokens of GPU time on news-length content (less with more slots per content, since a call's decode is a
fixed ~17 steps shared by all its slots), and a video frame at low detail costs about 1.1-1.2 text tokens of its size on the 12B
model, 1.8-1.9 on the 31B. The request fields and the result shape are in [scoring-api.md](scoring-api.md#extraction-and-generation-slots);
this document is how the engine does it and what it measured.

Setup: Gemma 4 12B in fp8 on one RTX 4090 (24 GB) unless said; the 31B on an RTX PRO 6000 (a MIG slice of 48 GB). Measured
2026-09-28 to 2026-10-10 on the maintainers' news fixture (18 articles of ~350 tokens, 13-198 two-option cells each, 1,328 cells)
and a set of TV commercials with the advertised brand as gold.

## 1. The slots

A slot replaces `{?}` in a template: the model writes the answer instead of scoring given options. `{*}` takes any tokens but the
special ones and ends at a token that starts with the template's text after the slot (a closing quote and its typographic twin),
at a newline or at `<eos>`; `{n}` follows a number grammar (`[space] [- | − | (] digits [,digits]* [.digits] [thousand | million |
billion | %] [)]`) and ends when the grammar is complete and the next token cannot extend it; `{$}` is the same grammar with one
currency symbol (`$ € £ ¥`) after the sign. Only `{*}` may have text after it. A text slot with `generate` is written by plain
generation. Per slot the result is the most likely answers with their probabilities (`text`, `value`, `currency`), `mass` (the
probability on all answers found), `p_negative` and `p_calibrated`; a generated slot gives text with a finish reason and no
probability. Query fields: `max_tokens` (default 16 for every kind: `extract.DEFAULT_MAX_TOKENS`), `top_k` (default 5), a
`calibration` `{a, b}`; limits in `protocol.Limits` (`max_extractions_per_input`, 10 by default).

## 2. The search

The search (`scorer.Scorer._search` for one slot, `decode.Decoder` for all slots of a pass) is a best-first beam over the
continuations after `content + template` in which **every live beam may also end**, with probability P(beam) × P(end | beam). So
"Bud" and "Bud Light" are different answers, and P(answer) is the exact probability of that string followed by an end. `BEAM` 6
live beams, `TOPK` 6 continuations per beam per step; beams and answers below `MIN_P` 0.002 are dropped, and beams below `REL_P`
0.02 of the slot's best answer so far too (they cannot change the answer). For number slots the candidates are the ~376 tokens made
only of number characters (`extract.NUMBER_ALPHABET`); per grammar state the valid extensions are memoised, and P(end) = 1 −
P(valid extensions). Answers are merged over surface forms: text by `extract.norm_text` (lower case, punctuation as spaces, no
English articles, no spaces: "Coca-Cola" = "COCA COLA"), numbers by (value, currency), each with its summed probability. `mass` is
the probability on everything found; `p_negative` the share of the top value's magnitude that sits on its negative (the sign is
often ambiguous in accounting tables); `p_calibrated` = sigmoid(a·logit(p) + b) of the top answer when the caller sends a
calibration.

**Generated slots** (`protocol.Generate`, `extract.gen_next`). The search cannot write a sentence: a 20-word path has a
probability far below `MIN_P` after 4-7 words and is pruned. `generate: true` is greedy: per step the arg-max over the tokens and
the end event (the log-sum-exp of the stop set). An object `{temperature 0-2, top_p, top_k 1-100 (default 40), seed, n}` samples
the next token from the top_k candidates (end included) inside the nucleus with a generator seeded per sequence
(`numpy.random.default_rng([seed, sequence])`), so a request reproduces. The n sequences take the slot's `BEAM` rows of the
decoder (`GEN_MAX_N` = `BEAM`); a sequence ends at the stop set or is cut at `max_tokens` (`finish: "length"`). Not allowed on
`{n}` / `{$}`, with `top_k` or `calibration`, or on `{?}`; a decision model with a readout head refuses every slot.

## 3. The engine path

**Compiler** (`compiler.compile_request`). The extraction templates of an input are branches of its scoring tree, marked in
`GroupedTree.xbranches`, so one packed pass reads the content, the scored cells and the templates and captures what the decode
needs. When a template merges with the content's end (content ending in whitespace, template starting with a newline: the
tokenizer joins them), the content is cut at a clean token boundary near its end (`compiler._cut`: the last `CUT_TOKENS` = 64
boundaries are tried, from at least `CUT_CHARS` = 32 characters; a cut inside a token never re-tokenises the same) and the tail is
re-tokenised with each template, so the content is still read once. When no boundary is clean the content is read once and the
templates join at the token boundary, with the warning `join_not_merged`; an earlier version re-sent the whole content once per
slot in that case (a 15k-token content with 10 slots processed 157k tokens instead of 19k). A slot with no template text is a
zero-length branch at the prefix's end (`ExtractJob`, `ExtractPrompt`).

**Capture** (`models/gemma4.py CAPTURE`). The flex attention hook, outside the compiled graph, keeps each layer's keys and values
of the stream ranges it is asked for (sliding layers their last window − 1 tokens, as transformers' `DynamicCache` does), plus the
next-token log-probabilities at each template's end: step 0 of the search is the scoring pass's own distribution.

**Decoder** (`decode.Decoder`, attention `x_decode`). All slots of all trees in a pass decode together: one batch row per tree
("group"), its slots × beams as the queries of that row. A query attends to the group's prefix (stored once per group), to its
own template and to the tokens its own beam generated; nothing is copied per slot or per beam, and when beams fan out only the
small per-beam state follows them. The step runs the decoder layers eagerly with sdpa over the cache (`Scorer._eager_sdpa`
switches the attention implementation and bypasses the compiled layers; flex is restored after), one token per live beam. The
reference path (`EngineConfig.x_batched = false`): `Scorer.prefill` + `_step` + `_search`, one slot at a time from a copy of the
prefix's cache; it stays for templates that merge with their content and for checks. The Qwen stack has its own capture and
decoder ([QWEN.md](QWEN.md) §9).

**Budgets.** Three settings in `config.py` bound the memory: `capture_budget_gib`, what one forward may capture for the extraction
branches packed in its stream (the packer puts fewer extraction inputs in a stream above it, since the captures exist next to the
forward's own peak); `decode_budget_gib`, the captures held across forwards until a decode; `decoder_budget_gib`, the decoder's own
working set per decode. `Scorer._x_flush` decodes the held groups in chunks under that last budget, and a group with too many slots
in runs of its prompts. A chunk is costed as the decoder allocates it (`decode.chunk_bytes`): the decoder pads every group of a
chunk to the chunk's largest prefix, template total, prompt count and `max_tokens`, so a chunk costs more than the sum of its
groups. Found the hard way: 5 searched slots at `max_tokens` 12-16 next to 2 generated ones at 64 took 65 rows each instead of
17, 4.6 GiB against 2.5 estimated, and ran a 24 GB card out of memory on 45 of 60 clips; costed padded, the two groups decode in
two chunks of ~1 GiB. The server warms the extraction path at start (`warmup_extract_s`, ~26 s on the 12B): the compiler
specialises a one-tree batch, and the first decode compiles.

## 4. Measured: time and throughput

First cases (12B, 4090, fp8):

| case | time | notes |
|---|---|---|
| 32-frame commercial (~2.2k tokens) + 1 scored query + brand `{*}` | 0.57-0.64 s | brand 19/20 right, frames as images or as a video part |
| TAT-QA document (200-1,500 tokens), 2-4 `{$}` slots | 0.4-1.6 s | ~0.3-0.4 s per slot (eager decode steps ~40 ms) |
| 8k / 16k / 32k / 58k tokens + scored query + `{n}` | 2.0 / 2.9 / 8.2 / 23.8 s | peak 13.5 / 14.5 / 16.5 / 19.9 GB; the code at the end found every time |

**Next to the regular cells.** The fixture's articles ×3 = 54 articles in calls of ~24k processed tokens, each article's cells as
scored queries (no floor) plus k extraction slots (company, country, a price `{$}`, a percentage `{n}`, ...). Input tokens =
content + query tokens.

| slots per article | one slot at a time, after its own read | batched, beam 6 | batched, beam 3 | batched, beam 2 |
|---|---|---|---|---|
| 0 | 9,357 tok/s | 9,007 tok/s (7.7 articles/s) | | |
| 1 | 2,118 | 6,287 (5.3 art/s) | 6,186 | 6,619 |
| 2 | 1,429 | 6,245 (5.2 art/s) | | |
| 10 | 383 | 3,863 (3.0 art/s) | 4,882 | 5,470 |

Top answer vs beam 6 (54 / 540 slots): beam 3 51/54 and 454/540 (84 %), beam 2 52/54 and 423/540 (78 %); the changes are
near-ties, mostly on slots without a clear answer. Batched vs the one-by-one reference: 62/72 same top answer, |Δp| 0.018 (the
packing noise of [PRECISION.md](PRECISION.md); the differences are near-ties). A CUDA-graph decode step (`x_cuda_graphs`) gained
nothing (3,722 vs 3,890 tok/s at 10 slots): the step is GPU-bound at these batch sizes, so it is off.

**The cost of a slot** (beam 6, the extraction path warmed; ±20 % run to run):

| slots / article | 0 | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| 12B / 4090: input tok/s | 8,754 | 5,813 | 6,001 | 5,600 | 4,271 | 3,904 | 3,766 |
| 12B tokens of GPU time per extraction | - | 611 | 285 | 184 | 229 | 207 | 183 |
| 31B / MIG 48 GB (36 articles): input tok/s | 3,149 | 2,446 | 2,235 | 1,903 | 1,474 | 1,267 | 1,169 |
| 31B tokens of GPU time per extraction | - | 352 | 255 | 212 | 247 | 246 | 230 |

Cost model: a call pays a fixed decode of ~16-17 steps (the longest-living slot; ~0.7-0.9 s on the 12B / 4090) shared by all its
slots, plus ~0.012 s (~100 tokens) per slot; with 4 or more slots per article in 24k-token calls one extraction costs ~180-230 input
tokens of GPU time. Per extraction the 31B costs about the same number of its own tokens as the 12B does of its (~230-350; a 31B
token is ~2.8× the 12B's GPU time on these cards). A decode step is ~45-70 ms and CPU-bound at these sizes (74 ms CPU vs 32 ms GPU
in the profile: 48 layer calls, dynamo lookups).

## 5. Video frames against text of the same size

Commercials as video parts (low detail, frames after their timestamps), 10 / 20 / 30 frames (~750 / 1,485 / 2,220 tokens), 7
attribute questions scored, then plus the brand `{*}`; the control is news text cut to each video's token count with the same
queries.

| | 10 frames | 20 frames | 30 frames |
|---|---|---|---|
| 12B / 4090: video time / text time (scored; + brand) | 1.16; 1.23 | 1.13; 1.20 | 1.13; 1.19 |
| 12B brand accuracy | 0.80 | 0.95 | 0.95 |
| 31B / MIG 48 GB: video time / text time (scored; + brand) | 1.79; 1.88 | 1.78; 1.92 | 1.79; 1.86 |
| 31B brand accuracy | 0.925 | 0.975 | 0.975 |

- The first run measured 6.5-8×: the content ended with "\n" and the templates start with "\n", the tokenizer merged them, and
  the compiler put every query in a tree of its own, re-sending the content per query. The clean cut of §3 fixed it, for scored
  and extraction queries alike; text and image requests stayed byte-identical.
- 12B: a video token costs ~1.13-1.23× a text token, about half of it CPU-side frame decoding, which overlaps other calls' GPU
  work under load. 31B: ~1.8-1.9×, its 27-layer vision tower running over 16 px patches (9 per image token) in bf16. Batching
  frames through the embedders by size group gained ~1 % on the 12B, nothing on the 31B, and ran out of memory at 64 frames: not kept.
- The 31B hit torch.compile's recompile limit (64) during plain scoring (transformers' Gemma 4 layer guards on the shared K/V
  state: one variant per layer of 60); the limits are raised at load.

## 6. Frames resized before they travel or by the engine

A front may resize images and frames larger than their detail's target and re-encode them as JPEG q90 before sending them on
(the hosted service does). Check: 60 commercials × 20 frames at the videos' native 854×480, (A) sent lossless as PNG, the engine
resizing, (B) resized and re-encoded first; 12B, the 7 attribute questions plus the brand `{*}`.

- Request size: 338 KB per frame (A) vs 65 KB (B).
- Attribute yes/no log-odds: mean |A − B| 0.13-0.28 (correlation 0.95-0.985), the size of the fp8 packing noise (~0.13 on the
  fixture); AUC against the labels, A vs B: 0.921 / 0.902, 0.529 / 0.561, 0.851 / 0.853, 0.849 / 0.825, 0.804 / 0.820,
  0.893 / 0.889, 0.729 / 0.756: no direction.
- Brand: 58 / 60 right both ways, the same top answer 57 / 60 (the misses "Toyota" vs "Toyota Highlander"), |p_A − p_B| 0.023.

## 7. Long contents

`EngineConfig.max_tokens` (default 65,536, `SPINF_MAX_TOKENS`) is the largest input the engine runs. A tree over the packing
budget (16k) runs alone in its own stream ("solo" batch); a content over half the budget is not split and re-sent per run.
`content_too_long` names the engine maximum when an input exceeds it. The hosted service caps the content per account tier up to
that maximum. The flex and staging buffers are sized to `max_tokens`; normal batches are unchanged (parity identical).

Cost grows faster than length: 8k / 16k / 32k / 58k tokens with a scored query and a `{n}` slot took 2.0 / 2.9 / 8.2 / 23.8 s on
the 12B / 4090 (32k → 58k: 1.8× the tokens, 2.9× the time), peaking at 13.5 / 14.5 / 16.5 / 19.9 GB. 58k is near the hosted
service's 30 s engine deadline.

## 8. Open

- Decode speed: the step is eager (~40-70 ms) and CPU-bound at small batch sizes; compiling the step would cut extraction time
  roughly in half. The Qwen stack replays its step from a CUDA graph ([QWEN.md](QWEN.md) §9).
- The 31B's extraction and video paths were measured through the same code but less widely than the 12B's.
- Memory is bounded by the three budgets; time is not: many slots on long contents multiply the decode chunks (the Qwen numbers
  in [QWEN.md](QWEN.md) §10). A server with a request deadline should also cap the slots per request.
- The request body (`max_body_bytes`, `server/config.py`, 2 MB by default) against long videos: 32 low-detail frames are ~1 MB.
