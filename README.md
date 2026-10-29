# spinf-serve

**Bulk classification and decision inference on one GPU.**

spinf-serve reads a piece of content once and answers many questions about it in the same pass. Each answer is a
probability over the options you give, not generated text. It runs on one 24 GB NVIDIA GPU and serves an
OpenAI-compatible API, the same API as the hosted spinf service. Apache 2.0.

## Why it exists

We built this engine for our own work: a daily market index read from millions of news articles, each article asked
dozens of narrow, calibrated questions. A serving engine built for chat re-reads the content for every question and
generates one token at a time, which made that job slow and expensive. spinf-serve lays every question out as a
branch of one read of the content and packs many contents into one forward pass. Bulk classification is where it
started and where it is strongest.

The same mechanism is what is now called a **decision model**: a model that returns a probability over a fixed set
of answers (yes or no, a label, a level) instead of a paragraph. Any open instruction model works this way in
spinf-serve, and dedicated decision models run as they are. The hosted spinf API grew out of this engine; we are
publishing it because people asked to run it on their own hardware.

It is not a chat server. Nothing is generated unless you ask for it, and it is built for throughput rather than for
the latency of a single request.

## Install and run

Linux, an NVIDIA GPU with 24 GB and fp8 tensor cores (Ada, Hopper or Blackwell: RTX 4090 / 5090, L4, L40S, RTX 6000
Ada, H100), a driver supporting CUDA 13, Python 3.12. Ampere cards run with int8 weights instead
([configuration](docs/configuration.md#models)).

```sh
pip install "spinf-serve[server]"
spinf-serve serve spinf/gemma-4-12b-fp8          # downloads the checkpoint on first start; serves on port 8000
```

Or the image, with the NVIDIA Container Toolkit installed (the volume keeps the model and the compile caches;
arguments after the image name are those of `spinf-serve serve`):

```sh
docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve
```

Then ask. Two classifications and one extraction from a single read of the content:

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

For each scored question the response gives every option's probability, and the same probability on empty content
(real numbers from the 12B model):

```json
{ "id": "refund", "combinations": [ { "values": {}, "options": [
    { "text": " yes", "logprob": -0.90, "p": 0.72, "floor_p": 0.60 },
    { "text": " no",  "logprob": -1.86, "p": 0.28, "floor_p": 0.40 } ],
    "residual": 0.44 } ] }
```

`p` is the probability among your options. `floor_p` is the same question with no content, so `p - floor_p` is what
the content added (here `topic` moves " billing" from 0.47 to 0.91). `residual` is the probability that went to words
outside your options. For `month` the response lists the most likely answers with their probabilities ("September" first).
The same body, with an API key, runs against `api.spinf.com`; the OpenAI SDKs work through `/v1/chat/completions` with
the `scoring` block in `extra_body` ([quickstart](docs/quickstart.md)).

## What you get

- **One read, many questions.** The content is read once; every question is a branch of the same pass and costs its
  own tokens only. Placeholders turn one template into a grid of questions.
- **Scores you can threshold.** A probability per option with its floor and log-probability; a decision is a
  threshold you choose on your own data ([calibration](docs/calibration.md)).
- **Extraction** from the same read: `{*}` text, `{n}` numbers and `{$}` amounts decoded with their probabilities,
  or free text generated into a slot.
- **The same score whatever else is in the batch.** With the fp8 checkpoints, a content's scores are bit-identical
  alone or packed with thousands of others, across calls, days and servers. The `system_fingerprint` says when two
  responses are not comparable.
- **Images, audio and video frames** in the content, for the models that read them.
- **Prompt packs.** Ready-made, calibrated question sets for a job (content moderation, triage, product listings,
  reviews, brand, entertainment), preloaded and named in a request ([below](#prompt-packs)).
- **Hosted or self-hosted.** A client switches between this server and `api.spinf.com` by changing the base URL and
  the key. [spinf-client](https://github.com/spinfinc/spinf-client) talks to both; the
  [agent skills](https://github.com/spinfinc/skills) teach a coding agent the API, this server and the pack format.

## Speed

Measured on one RTX 4090 with Gemma 4 12B in fp8: 24 Wikinews articles of about 470 tokens, 56 two-option questions
each, 1,344 questions in all (the public job set in `fixtures/`), sent over HTTP to spinf-serve and to vLLM 0.26 with
prefix caching. The script and the full results are in [benchmarks/](benchmarks/README.md).

| one RTX 4090, warm | spinf-serve | vLLM 0.26, prefix caching, 128 requests in flight |
|---|---|---|
| questions answered per second | **491** | 200 |
| tokens of work per second (content once, then each question's own tokens) | 11,200 | 4,600 |
| the 1,344 questions | **2.7 s** | 6.7 s |

vLLM already reuses each article across its 56 questions through the prefix cache. What it cannot do is answer them
in one forward pass: it runs 1,344 single-token requests, while spinf-serve lays the 56 questions out as branches of one
read and packs the 24 articles into two 16k-token passes. The gap grows with the number of questions per content.

In practice, per 4090 with the 12B model: the `spinf/moderation` pack (106 questions per message, 22 categories)
scores about 20,000 messages per hour; a news article asked 74 questions is read at nearly 700 questions per second.
Extraction slots add decoding work (a searched slot costs about as much as 200 to 600 tokens of scoring, a generated
one up to about 2,000) but share the read.

## Prompt packs

A job is rarely one question: content moderation is 106 of them, each calibrated against normal traffic and combined
into category scores and a decision. A **prompt pack** is one file holding the questions, how the content is presented
to the model, how the answers turn into scores and how the scores turn into a decision. The server preloads packs and
a request names one (`"packs": [{"id": "spinf/moderation"}]`), getting the pack's categories and decision back next
to its own questions, from the same read.

The official packs are in [spinf-prompts](https://github.com/spinfinc/spinf-prompts), loaded by default. The server
loads packs from any GitHub repository or local folder given with `--packs`; **[PACKS.md](PACKS.md) lists every
pack we know of**, ours and others', and an [issue](https://github.com/spinfinc/spinf-serve/issues/new?template=pack.yml)
adds yours. Format, request fields and sources: [docs/prompt-packs.md](docs/prompt-packs.md).

## Models

| name | based on | GPU | reads |
|---|---|---|---|
| `spinf/gemma-4-12b-fp8` (default) | Google Gemma 4 12B, FP8 with static activation scales | one 24 GB | text, images, audio, video |
| `spinf/gemma-4-31b-fp8` | Google Gemma 4 31B, FP8 static | one 48 GB | text, images, video |
| `spinf/qwen3.8-27b-int4` | Qwen3.8-27B, GPTQ int4 | one 24 GB | text, images, video |
| `spinf/pplx-decider-v1-27b-int4` | Perplexity pplx-decider-v1-27b, GPTQ int4: a decision model with a 255-code readout | one 24 GB | text; scores only, with its own prompts and packs |

Each is a checkpoint we quantized and calibrated, published on the [Hugging Face Hub](https://huggingface.co/spinf)
with its recipe (`quantizations/`); the engine pins the revision it was parity-tested on. Any other Hub checkpoint of
a supported architecture family runs by its path, quantized at load time.

## Documentation

[docs/](docs/README.md): [quickstart](docs/quickstart.md), [examples](docs/examples.md), [scoring API](docs/scoring-api.md),
[configuration](docs/configuration.md), [writing questions](docs/writing-questions.md), [calibration](docs/calibration.md),
[prompt packs](docs/prompt-packs.md), [FAQ](docs/faq.md); the [guides](docs/guides/index.md) (decision models,
probabilities you can act on, few-shot classification, sentiment, survey coding, moderation, intent routing); and the
engineering documents on how the engine works ([architecture](docs/ARCHITECTURE.md),
[batch invariance](docs/BATCH_INVARIANCE.md), [precision](docs/PRECISION.md), [the Qwen stack](docs/QWEN.md),
[extraction](docs/EXTRACTION.md)).

## Status

Version 0.1. The engine (scoring, extraction, generation, media, batch invariance, the two model stacks) is complete
and parity-tested against recorded references; the server speaks the public API of the hosted service; the four
checkpoints are public. What is still missing is listed in [docs/README.md](docs/README.md#status).

Questions, ideas and bugs go to the [issues](https://github.com/spinfinc/spinf-serve/issues) of this repository.
There is no forum or chat, and nothing is too small.

## Layout

`spinf_serve/` holds the engine (`protocol`, `compiler`, `packer`, `scorer`, `extract`, `service`), the served models'
manifests, the `packs` module (pure Python, also used by the client and the hosted API), one model stack per architecture
family in `models/`, the quantization schemes in `quant/`, the HTTP server in `server/` and the `spinf-serve` command.
`quantizations/` has one folder per published checkpoint (recipe, calibration output, model card), `fixtures/` the public
parity fixture and references, `benchmarks/` the throughput comparison. The map of the code is
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Develop

```sh
uv sync --extra server
uv run pytest                                   # CPU tests (the official packs' golden fixtures need ../spinf-prompts or SPINF_PROMPTS_DIR)
uv run ruff check . && uv run ruff format --check .
CUDA_VISIBLE_DEVICES=0 uv run spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz \
    --reference fixtures/wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz --passes 2
```

Every change that touches a kernel, a compile boundary or a quantization scheme ships with the GPU parity run:
bit-identical log-probabilities on the recorded reference of each published checkpoint ([fixtures/](fixtures/README.md)).
Dependencies are pinned exactly; they change only together with a parity run. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence and acknowledgements

Apache 2.0 (`LICENSE`, `NOTICE`). The models we run and re-quantize, the projects we build on, the techniques we
implement and the data we checked on are credited in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).
