# spinf-serve

**The open-source inference engine for decision models.**

spinf-serve reads a piece of content once and answers hundreds of questions about it in parallel, as probabilities
rather than text. It turns a standard open model (Gemma 4, Qwen3.8) into a decision model through **decision mode**:
one input, many questions, scores instead of generation, and the same score whatever else is in the batch. It runs on
one 24 GB GPU and serves an OpenAI-compatible API that is also the API of the hosted spinf service.

## What it does

- **Scoring.** A question is a template that ends where the answer goes, with the answers to score there. The engine
  returns the probability of each answer, its log-probability, and the same question's probability on empty content
  (the floor), so a score can be read as a lift over the question's own bias.
- **Many questions, one read.** The content is read once; every question is a branch of the same pass. Adding a
  question costs its own tokens only. Placeholders turn one template into a grid of questions.
- **Extraction and generation** from the same read: `{*}` text, `{n}` numbers and `{$}` amounts decoded with their
  probabilities, or free text generated into a slot.
- **Batch invariance.** With the calibrated fp8 scheme, an input's scores are bit-identical whether it is scored
  alone or packed with thousands of others. Scores compare across calls, days and servers; a `system_fingerprint`
  says when they do not.
- **Media.** Images, audio and video frames in the content, read by the models that support them.
- **Prompt packs.** Ready-made, calibrated question sets preloaded into the server and named in a request
  ([below](#prompt-packs)).
- **Throughput.** Inputs are packed into 16k-token passes on one GPU: about 10,000 tokens per second on an RTX 4090
  with the 12B model.
- **One API, hosted or self-hosted.** The request and response are those of `api.spinf.com`; a client switches by
  changing the base URL and the key.
- **Clients and agents.** [spinf-client](https://github.com/spinfinc/spinf-client) (`pip install spinf-client`) talks to
  either server; the [agent skills](https://github.com/spinfinc/skills) teach a coding agent the API, this server and the pack
  format; [spinf-prompts](https://github.com/spinfinc/spinf-prompts) holds the official packs, and community packs such as
  [thinkingtext/spinf-news-pack](https://github.com/thinkingtext/spinf-news-pack) load with `--packs`.

## Speed and scale

spinf-serve exists for one workload: **many questions about one content**. Our own first use, a daily oil-market
index read from 15 million news articles, taught us that asking thousands of narrow, calibrated questions of every
article gives a far more accurate read than any single prompt, and that no engine built for chat could do it at a
price that made sense. A chat engine re-reads the content for every question; even with prefix caching it generates
one token at a time per prompt and treats every question as a separate request. spinf-serve reads the content once
and lays every question out as a branch of that read, packed with other inputs into 16k-token passes on one GPU.

Measured on one RTX 4090 with Gemma 4 12B in fp8, the same 1,344 questions (24 news articles of about 470 tokens, 56
questions each, every answer a probability over two options: the public job set in `fixtures/`) sent over HTTP to
spinf-serve and to **vLLM 0.26 with prefix caching**, which is what most people would reach for (`benchmarks/`):

| one RTX 4090, warm, per second | spinf-serve | vLLM 0.26, prefix caching, 128 requests in flight |
|---|---|---|
| questions answered | **491** | 200 |
| tokens of work processed (content once, then each question's own tokens) | 11,200 | 4,600 |
| the same result for an engine that re-reads the content per question | 236,000 tokens | 96,000 tokens |
| the 1,344 questions | **2.7 s** | 6.7 s |

2.5 times faster on the same card for the same answers, and vLLM is already doing the right thing here: its prefix
cache reuses each article across its 56 questions. What it cannot do is answer the questions in one forward pass, so it
runs 1,344 single-token requests against the cache, while spinf-serve lays the 56 questions out as branches of one read
and packs the 24 articles into two 16k-token passes. The gap grows with the number of questions per content: at 56
questions the shared read saves 21 times the tokens; at a thousand it saves hundreds of times, and a thousand questions
on one content still cost about what the questions themselves weigh. Processed-token throughput stays at the engine's
rate, about 11,000 per second per 4090 for the 12B model (on an earlier set of 18 shorter articles with 74 questions each
the ratio was 2.9). The int4 Qwen3.8 27B model showed the same ratio range (2.2 to 2.8 times vLLM with prefix caching) on
2,211 decision-benchmark jobs in our internal measurements. Anyone can rerun the comparison: `benchmarks/README.md`.

What this means for a job:

- **Content moderation**: the `spinf/moderation` pack asks 106 questions of every message (22 categories, three
  decision modes). One 4090 moderates about 20,000 messages per hour through the full pack, each message scored for
  every category in one read.
- **News and document analysis**: 74 calibrated questions per article, aggregated into indexes, at nearly 700
  questions per second per GPU.
- **Triage, routing, tagging**: intent in two levels, urgency, sentiment, risk flags, personal-data checks and a
  policy check in one pass, each a question in the same list, each costing its own tokens only.

**Extraction and generation** ride the same read but add decoding steps: a searched slot (`{*}` `{n}` `{$}`) costs
about as much GPU time as 200 to 600 tokens of scoring, a generated slot up to about 2,000. They still benefit from
the shared read and the packing: the content and the questions are not re-read, the slots of many inputs are decoded
together, and a few slots next to hundreds of scored questions change the pass's cost only a little.

The example packs (triage, product listings, entertainment metadata, reviews, brand analysis) are in
[spinf-prompts](https://github.com/spinfinc/spinf-prompts).

## Prompt packs

A specialised use case is rarely one question: content moderation is 106 of them, each calibrated against normal
traffic and combined into category scores and a decision. spinf-serve standardises a format for this: a **prompt pack**
is one file holding a set of questions, how to present the content to the model, how the answers turn into scores,
and how the scores turn into a decision. Packs are preloaded into the server, so a request names one (`"packs": [{"id":
"spinf/moderation"}]`) instead of carrying hundreds of questions over and over, and gets the pack's categories and
decision back next to its own questions, from the same read of the content.

Because a pack is a file, it is also a **ready-to-go, shareable set**: the server loads packs from
[spinf-prompts](https://github.com/spinfinc/spinf-prompts), the official repository (content moderation first), and
from any GitHub repository or local folder you point it at, enabling the ones written for the model it serves. Publish
your own with a manifest at the root of your repository. Details, the file format and the request fields:
[docs/prompt-packs.md](docs/prompt-packs.md).

## Install and run

Linux, an NVIDIA GPU with 24 GB (fp8 tensor cores for the batch-invariant scheme: RTX 4090 / 5090, L4, L40S, RTX 6000
Ada, H100), a driver supporting CUDA 13, Python 3.12.

```sh
pip install "spinf-serve[server]"
spinf-serve serve spinf/gemma-4-12b-fp8          # downloads the checkpoint on first start; serves on port 8000
```

Or the image, with the NVIDIA Container Toolkit installed (the volume keeps the model and the compile caches between runs;
arguments after the image name are those of `spinf-serve serve`):

```sh
docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve
```

Then ask. Two classifications and one extraction, answered from a single read of the content:

```sh
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spinf/gemma-4-12b-fp8",
    "messages": [
      { "role": "user",
        "content": "Customer support message:\nI was charged twice for September, please refund one of the charges." }
    ],
    "scoring": {
      "queries": [
        { "id": "topic",
          "template": "\n\nTopic (billing, technical, account, shipping, or cancellation):{?}",
          "options": [" billing", " technical", " account", " shipping", " cancellation"] },
        { "id": "refund",
          "template": "\n\nIs the customer asking for a refund?\nAnswer:{?}",
          "options": [" yes", " no"] },
        { "id": "month",
          "template": "\n\nQ: Which month was charged twice?\nA: \"{*}\"" }
      ]
    }
  }'
```

The response gives, for `topic` and `refund`, the probability of each answer and the same probability on empty
content (the lift between the two is what the message says), and for `month` the most likely answers with their
probabilities. The same body, with an API key, runs against the hosted API at `api.spinf.com`; the OpenAI SDKs work
through `/v1/chat/completions` with the `scoring` block in `extra_body` ([quickstart](docs/quickstart.md)).

As a library: `from spinf_serve import Engine; Engine("spinf/gemma-4-12b-fp8").score(request)`.

## Models

| name | based on | GPU | reads |
|---|---|---|---|
| `spinf/gemma-4-12b-fp8` (default) | Google Gemma 4 12B, FP8 with static activation scales | one 24 GB | text, images, audio, video |
| `spinf/gemma-4-31b-fp8` | Google Gemma 4 31B, FP8 static | one 48 GB | text, images, video |
| `spinf/qwen3.8-27b-int4` | Qwen3.8-27B, GPTQ int4 | one 24 GB | text, images, video |
| `spinf/pplx-decider-v1-27b-int4` | Perplexity pplx-decider-v1-27b, GPTQ int4: a decision model with a 255-code readout | one 24 GB | text; scores only, with its own prompts and packs |

Each is a checkpoint we quantized and calibrated, published on the Hugging Face Hub with its recipe
(`quantizations/`); the engine pins the revision it was parity-tested on. Any other Hub checkpoint of a supported
architecture family runs by its path, quantized at load time.

## Documentation

[docs/](docs/README.md): [quickstart](docs/quickstart.md), [examples](docs/examples.md) (curl calls for one question,
parallel classifications, classification plus extraction, many contents, a prompt pack),
[scoring API](docs/scoring-api.md), [configuration](docs/configuration.md), [writing questions](docs/writing-questions.md),
[calibration](docs/calibration.md), [prompt packs](docs/prompt-packs.md), [FAQ](docs/faq.md); the [guides](docs/guides/index.md)
(decision models, probabilities you can act on, few-shot classification, sentiment, survey coding, moderation, intent routing)
and the engineering documents ([docs/README.md](docs/README.md#engineering-documents)).

## Status

The engine (scoring, extraction, generation, media, batch invariance, the two model stacks) is complete and
parity-tested against recorded references. The server speaks the public API of the hosted service: `spinf-serve serve`,
`/v1/score` and its chat-completions alias, `/v1/models`, `/v1/packs`, pack sources, the response and error envelopes.
The `spinf/...` checkpoints are being published; [docs/README.md](docs/README.md#status) keeps the list of what is
left. How the engine works and the measurements behind it: the engineering documents in
[docs/](docs/README.md#engineering-documents) (architecture, batch invariance, precision, the Qwen stack, extraction).

## Layout

```
spinf_serve/
  protocol.py, compiler.py, packer.py, scorer.py, decode.py, extract.py, media.py, service.py
                         the engine: request shape, templates to token tries, packing, one packed forward, extraction
  manifests/             the served models (TOML): pinned checkpoint, family, schemes, capabilities, media, limits
  packs/                 prompt packs (pure Python): the file and its validation, the evaluator, a request's packs, the sources
                         (GitHub, folders, the official repository) and the catalog; `python -m spinf_serve.packs validate | list`
  models/                one stack per architecture family behind one interface (base.py, registry.py): gemma4.py, qwen35/
  quant/                 the quantization schemes (linears, kernels, the fp8-static layout), the per-revision tables, the maintainer
                         commands that produce our checkpoints
  server/                the HTTP server (FastAPI, one process per GPU): the public API around the engine (api.py), the settings
                         (config.py: flags > environment > config file > manifest), the app
  cli.py                 the `spinf-serve` command: serve, score, packs, parity, calibrate
quantizations/           one folder per published checkpoint: recipe, calibration output, model card
fixtures/                the public parity fixture (24 Wikinews articles x 56 cells) and one recorded reference per checkpoint
benchmarks/              the throughput comparison script (spinf-serve against a completions server), its results, the fixture builder
tests/                   CPU tests (run in CI); the GPU parity gate runs by hand (below)
docs/                    the user guides and the engineering documents
```

## Develop

```sh
uv sync --extra server
uv run pytest                                   # CPU tests (the official packs' golden fixtures need ../spinf-prompts or SPINF_PROMPTS_DIR)
uv run ruff check . && uv run ruff format --check .
CUDA_VISIBLE_DEVICES=0 uv run spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz \
    --reference fixtures/wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz --passes 2
```

Every change that touches a kernel, a compile boundary or a quantization scheme ships with the GPU parity run:
bit-identical log-probabilities on the recorded reference of each published checkpoint ([fixtures/](fixtures/README.md):
24 Wikinews articles x 56 two-option cells, a reference per model recorded and verified on its card). Scores come from
exact dependency pins; they change only together with a parity run.

## Licence and acknowledgements

Apache 2.0 (`LICENSE`, `NOTICE`). The models we run and re-quantize, the projects we build on, the techniques we
implement and the data we checked on are credited in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).
