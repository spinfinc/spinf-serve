# Configuration

## Models

| name | based on | card | reads | capabilities |
|---|---|---|---|---|
| `spinf/gemma-4-12b-fp8` (default) | Google Gemma 4 12B, FP8 with static activation scales | one 24 GB GPU with fp8 tensor cores (RTX 4090 / 5090, L4, L40S, RTX 6000 Ada, H100, RTX PRO 6000); Ampere (3090, A5000, A6000, A40) with int8 weights instead | text, images, audio, video | score, extract, generate |
| `spinf/gemma-4-31b-fp8` | Google Gemma 4 31B, FP8 static | one 48 GB GPU (RTX 6000 Ada, L40S, L40, RTX PRO 6000) | text, images, video | score, extract, generate |
| `spinf/qwen3.8-27b-int4` | Qwen3.8-27B, GPTQ int4 | one 24 GB GPU | text, images, video | score, extract, generate |
| `spinf/pplx-decider-v1-27b-int4` | Perplexity pplx-decider-v1-27b, GPTQ int4: a decision model with a 255-code readout | one 24 GB GPU | text | score only, with its own prompt protocol and packs |

Each model is a checkpoint we quantized and calibrated, published under `spinf/` on the Hugging Face Hub with its
recipe and model card; the quantization parameters travel with the weights. The engine's `manifests/` directory pins
each model's revision. A model name is also the Hub repo, so `spinf-serve serve spinf/gemma-4-12b-fp8` downloads what
it needs on first start (Gemma 4 and Qwen are Apache 2.0, no access request).

Any other Hub checkpoint of a supported architecture family (Gemma 4, Qwen3.5) can be served by its path
(`spinf-serve serve google/gemma-4-12B`): it is quantized at load time with the dynamic scheme, served under that
name, and is not batch-invariant (no calibrated scales). Your own manifests go in a directory named by
`SPINF_MANIFEST_DIRS`; a file with a built-in id overrides it.

## Hardware and software

- Linux, an NVIDIA driver supporting CUDA 13.0, Python 3.12. The exact `torch`, `triton` and `transformers` versions are
  pinned by the package: the scores come from these kernels, and the pins change only together with a parity run.
- One GPU per server process. Several GPUs: one process each, behind your own load balancer; a process never shares its
  GPU with another engine (the Qwen model needs the whole 24 GB card).
- `gcc` at run time (the kernels are compiled on the host at first start, then cached) and `ffmpeg` for audio.
- fp8 tensor cores (compute capability 8.9 and up) select the `fp8s` scheme: the calibrated static scales that make a
  score bit-identical whatever else is in the batch. On Ampere the engine runs `w8a16` (int8 weights expanded to bf16):
  deterministic run to run, but a score then depends slightly on how the batch is packed, and differs from the Ada
  workers'. Scores also differ between GPU architectures; `system_fingerprint` tells them apart.

## The image

`ghcr.io/spinfinc/spinf-serve` is spinf-serve from PyPI on `python:3.12-slim`, nothing else baked in: the model is downloaded
from the Hub at first start into `/root/.cache` (mount a volume there), the official packs are fetched there too, and the
kernels compile on your GPU at first start and are cached there. The image's `ENTRYPOINT` is `spinf-serve serve`, so
everything in this page applies: flags after the image name, `SPINF_*` variables with `-e`, a config file mounted and named
with `--config`. A local snapshot is served with `-v /path/to/snapshot:/model:ro` and `--model-dir /model`; a shared model
volume with `-e SPINF_CACHE_DIRS=/models -v /path/to/models:/models:ro`. The container's `HEALTHCHECK` is `/readyz` with a
20-minute start period, which covers a first-start download; set your orchestrator's readiness probe on `/readyz` as well.

```sh
docker run --gpus '"device=1"' -p 8010:8000 -v spinf-cache:/root/.cache \
  ghcr.io/spinfinc/spinf-serve:0.1.2 spinf/gemma-4-12b-fp8 --token s3cret --media none --max-content-tokens 8192
```

## Server options

Settings are layered: command-line flags over `SPINF_*` environment variables over a config file (`--config
server.toml`, or `SPINF_CONFIG`) over the model's manifest defaults. The config file has the same settings as the flags,
with underscores, in the tables `[engine]`, `[server]`, `[features]` and `[limits]`:

```toml
[engine]
model = "spinf/gemma-4-12b-fp8"
quant = "fp8s"
[server]
port = 8010
token = "change-me"
[features]
media_kinds = ["image", "video"]
packs = ["github:acme/prompt-packs@v1.2", "./my-packs"]
[limits]
max_content_tokens = 8192
```

Three groups:

**Engine** (hardware and numerics)

| flag | env | default | meaning |
|---|---|---|---|
| `--device` | `SPINF_DEVICE` | `cuda:0` | the GPU |
| `--quant` | `SPINF_QUANT` | auto | the scheme: `fp8s` (fp8 cards, the model's scales), `fp8`, `fp8row`, `w8a16`, `w8a8`, `bf16`; int4 models: `i8`, `f16`, `fp8`, `bf16` |
| `--token-budget` | `SPINF_TOKEN_BUDGET` | 16,384 | packed tokens per forward pass; 16k is the fastest for news-length content on a 4090 |
| `--max-tokens` | `SPINF_MAX_TOKENS` | model | the largest single input (content and its questions) the engine runs; longer inputs are refused |
| `--model-dir` | `SPINF_MODEL_DIR` | the Hub | a local snapshot directory |
| `--no-media` | `SPINF_MEDIA=0` | on | skip loading the image and audio embedders (and disable the media kinds) |
| `--no-compile` | `SPINF_COMPILE=0` | on | debugging only: 2 to 3 times slower |
| `--warmup` | `SPINF_WARMUP` | `full` | what runs before the server reports ready: `full` = every compile a request could pay (short streams, a stream at the token budget, the extraction path: about 1.5 to 2 minutes after the load on a 4090 with the 12B, the extraction path being most of it), `short` = the short streams and the extraction path, `none` = no warm-up, the first requests compile |
| `--lm-head` | `SPINF_LM_HEAD` | `f32out` | the lm head's output precision: `f32out` (fp32 logits, free, the quietest), `bf16` (parity with references recorded with bf16 logits), `fp32` (fp32 weights, research) |
| `--host`, `--port` | `SPINF_HOST`, `SPINF_PORT` | `0.0.0.0`, 8000 | |
| `--token` | `SPINF_TOKEN` | none | require `Authorization: Bearer <token>` on every call; without any token the server is open |
| `--trusted-token` | `SPINF_ENGINE_TOKEN` | none | the trusted callers' token ([below](#trusted-callers)); it is an access token too |
| `--config` | `SPINF_CONFIG` | none | a TOML config file |
| | `SPINF_CACHE_DIRS` | | extra directories of Hugging Face snapshots searched before the local cache (a shared model volume, say) |
| | `SPINF_MANIFEST_DIRS` | | directories of your own model manifests |
| | `SPINF_PACKS_OFFICIAL` | the pinned spinf-prompts tag | where the official pack slot reads from: a GitHub source or a checkout ([Prompt packs](prompt-packs.md#sources)) |
| | `SPINF_PACKS_CACHE` | `~/.cache/spinf-serve/packs` | where GitHub pack sources are fetched to |
| | `GITHUB_TOKEN` | | for pack sources in private GitHub repositories |
| `--stack-options` | `SPINF_STACK_OPTIONS` | | research knobs of the model family as `key=value,key=value` (the Qwen stack's `QwenOptions`; none for Gemma); empty is the production configuration |
| | `HF_TOKEN` | | a Hugging Face token: optional (the spinf model repositories are public); raises the Hub's download rate limits, needed for a private copy of your own |

**Features**

| flag | default | meaning |
|---|---|---|
| `--media image,audio,video` (`SPINF_MEDIA_KINDS`) | all the model reads | which media kinds requests may send; `--media none` for text only |
| `--extraction / --no-extraction` | on | the `{*}` `{n}` `{$}` slots |
| `--generation / --no-generation` | on | `generate` on `{*}` slots (off whenever extraction is off) |
| `--packs <source>` (`SPINF_PACKS`, comma-separated) | the official repo | prompt-pack sources, repeatable ([Prompt packs](prompt-packs.md#sources)); `--no-official-packs` leaves the official repository out; `--packs-allow-namespace spinf` serves the reserved namespace from another source; `--packs-refresh` resolves GitHub branches again |
| `--debug` | off | adds `packed_tokens` and the `engine` block to responses (engine internals, normally reserved to trusted callers) |

**Limits** (the defaults are those of the hosted service)

| flag | default | meaning |
|---|---|---|
| `--max-content-tokens` | 4,096 | per input, media included |
| `--max-query-tokens` | 256 | the filled template |
| `--max-prompts-per-input` | 10,000 | templates × combinations |
| `--max-extractions-per-input` | 10 | extraction slots (queries × combinations) |
| `--max-processed-tokens` | 2,000,000 | per request: content, questions, floors, decode |
| `--max-body-bytes` | 2 MB | |
| `--max-options`, `--max-option-tokens` | 1,000; 5 | answers per question, and tokens per answer (a cost guard: scoring long strings tells little, extraction is the tool for anything longer than a label) |
| `--max-frames`, `--max-audio-seconds`, `--max-image-pixels` | 1,024; 600; 50 M | per video, per audio part, per image |
| `--queue` | 4 | requests accepted while one is on the GPU; beyond it `429 concurrency_limit` with `Retry-After` |

Every limit is also an environment variable (`SPINF_MAX_CONTENT_TOKENS`, ...) and a key of the `[limits]` table.

The engine has been tested with content up to 64k tokens: raise `--max-content-tokens` and `--max-tokens` together.
Long inputs run alone in their own pass, and the current models become less effective on them.

`GET /v1/models` reports the effective features and limits, so a client can size a request before sending.

## Trusted callers

A front end (the hosted API is one) may send `limits` and `features` blocks in the request body to set them per call,
each value capped by the server's own: the account-level limits of a service. Only a trusted caller may do so: a
request carrying `--trusted-token`, or `--token` when no trusted token is set (a single-token setup is your own front
end). From anyone else the blocks are ignored and the server values apply. Trusted callers also receive the engine's
internal fields (`packed_tokens`, the flat engine counters in `usage`, `engine`), the compact `rejected` form next to the
error envelope, and may call the engine-internal endpoints (`/v0/score`, `/v0/info`, `/v0/drain`).

```json
{ "model": "spinf/gemma-4-12b-fp8", "inputs": [ "…" ], "scoring": { "…": "…" },
  "limits": { "max_content_tokens": 2048, "max_extractions_per_input": 4 },
  "features": { "media": ["image"], "extraction": true, "generation": false } }
```

## Throughput

The engine is built for throughput: one 16k-token forward at a time, inputs packed together, the content read once for
all its questions. A single 4090 scores about 10,000 tokens per second on news-length content with the 12B model. Send
many inputs per call rather than many small calls. The server reports ready (`/readyz`) only once its warm-up has run every
compile a request could pay (`--warmup full`, the default), so no call waits behind a compile; the compile cache on disk makes
the warm-up itself faster on the next start.
