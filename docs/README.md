# spinf-serve documentation

spinf-serve answers many questions about a piece of content in a single pass: bulk classification, extraction and the
decisions built from them. A question is a template that ends where
the answer goes, with the answers to score there; the engine returns the probability of each answer for the content.
The content is read once and every question is answered from that read, so adding questions costs only the tokens of
the questions and answers themselves. The same request format runs against the hosted spinf API at `api.spinf.com`:
switching between the two is a base URL and a key.

## Pages

- [Quickstart](quickstart.md): install, serve a model, make the first call, read the scores.
- [Examples](examples.md): curl calls for one question, two classifications in parallel, classification plus extraction,
  many contents in one call, and a prompt pack next to your own questions.
- [Scoring API](scoring-api.md): templates, answers, extraction and generation slots, media, the request and response
  reference, errors.
- [Configuration](configuration.md): models, hardware, server options, features and limits, trusted callers.
- [Writing questions](writing-questions.md): content labels, answer lists, a few labelled examples, iterating on your data.
- [Calibration](calibration.md): how to use `p`, `residual` and `floor_p`.
- [Prompt packs](prompt-packs.md): ready-made question sets from this and other repositories, in the same call as yours.
- [FAQ](faq.md).

## Guides

One concept or job per page, with the measurements behind the advice ([guides/](guides/index.md)):

- Concepts: [decision models](guides/decision-models.md), [probabilities from a language model](guides/llm-probabilities.md),
  [probabilistic if statements](guides/probabilistic-if.md), [zero-shot and few-shot classification](guides/zero-shot-few-shot-classification.md),
  [scoring without output tokens](guides/no-output-tokens.md).
- Jobs: [entity-level sentiment for news](guides/entity-sentiment-analysis.md), [aspect-based sentiment analysis](guides/aspect-based-sentiment-analysis.md),
  [coding open-ended survey responses](guides/open-ended-survey-coding.md), [custom content moderation policies](guides/custom-content-moderation.md),
  [intent classification and routing](guides/intent-classification.md).

## Engineering documents

How the engine works and the measurements behind its design, for people evaluating or contributing to it:

- [Architecture](ARCHITECTURE.md): the primitive, the request path through the package, packing and tree attention, the model
  stacks, the quantization schemes and their tables, the parity and release rules.
- [Batch invariance](BATCH_INVARIANCE.md): why an input's scores moved with its neighbours, and how `fp8s` with aligned jobs makes
  them bit-identical alone or packed, and the same in every fresh process.
- [Precision](PRECISION.md): compile variants, packing noise per scheme, the lm head, how answers are read case-insensitively, and
  how much the question template moves a score.
- [The Qwen stack](QWEN.md): a 27B hybrid model (Gated DeltaNet + attention, int4 weights) on one 24 GB card: the tree plan, int8
  tensor cores with the outlier split, images, extraction, memory bounds.
- [Extraction](EXTRACTION.md): slots decoded from the same read as the questions: the search, the batched decoder, what a slot costs,
  video frames against text.

## Status

These pages describe the public API of spinf-serve 0.1, which is the public API of the hosted spinf service. The engine
(scoring, extraction, generation, media, batch invariance) is complete and parity-tested; the server implements the
specification: the `spinf-serve` command line (`serve`, `score`, `packs`, `parity`, `calibrate`), `/v1/score` with
`messages` or `inputs`, `/v1/chat/completions`, `/v1/models`, `/v1/packs`, the response and error envelopes, the detailed
`usage` and `timing` blocks, `scoring.packs` with `--packs` sources. Not yet in the code at the time of writing:

- `spinf-serve quantize`, the maintainers' command that produces the published checkpoints (the scripts are in
  `quantizations/`);
- the conformance suite against the hosted API runs by hand (`tests/conformance.py`), not in CI; the GPU parity gate too (the
  CPU suite and lint run in CI).

Each item is removed from this list as it lands.
