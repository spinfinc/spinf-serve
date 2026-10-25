# FAQ

## Why do I write a template with answers instead of asking a free-form question?

We tested several approaches, and a question written as a template that ends where the answer goes, with the answers
you choose (`" rise"`, `" fall"`), proved much more controllable and predictable than free-form questions or generated
text. You decide exactly which words are scored and where the model reads them, which gives the level of control that
fine-grained analysis needs. When the answer is open, an extraction slot reads it from the same pass
([Scoring API](scoring-api.md#how-a-question-works)).

## Are the scores deterministic?

On the fp8s scheme (fp8 tensor cores, the `spinf/...` checkpoints with their calibrated scales), yes, and more than
that: a content's scores are **bit-identical** whether it is scored alone, with other contents, or with other
questions in the same call, and identical across fresh starts of the engine on the same GPU architecture. Two things
change them: another scheme (Ampere's `w8a16`: deterministic run to run, but a score then moves slightly with how the
batch is packed) and another GPU architecture (Ada and Blackwell run different kernels, so their scores differ by a few
hundredths). The `system_fingerprint` of each response identifies the engine build, scheme, weights revision and GPU
architecture that produced it: compare it to know whether two responses are directly comparable.

## Are images, audio or video supported?

Yes, for the models that read them: the Gemma 4 12B model reads images, audio and video (as frames you sample), the
31B and the Qwen models images and video. The server enables all the kinds a model reads unless `--media` restricts
them ([Scoring API](scoring-api.md#images-video-and-audio)).

## Which models can I run?

The four `spinf/...` checkpoints ([Configuration](configuration.md#models)): quantized and calibrated copies of Google's
Gemma 4 12B and 31B, of Qwen3.8-27B, and of Perplexity's pplx-decider (a decision model). Any other Hub checkpoint of
a supported architecture family runs too, quantized at load time, without the batch-invariance guarantee. Supporting a
new family means a stack in `models/`, a calibration and a parity reference; see [ARCHITECTURE.md](ARCHITECTURE.md) §4 and §9.

## How is this related to the hosted spinf API?

Same engine, same request and response. The hosted service at `api.spinf.com` adds accounts, keys, billing, several
models behind one host and reserved capacity. A client switches between a self-hosted server and the hosted API by
changing the base URL and the key ([Scoring API](scoring-api.md#differences-from-the-hosted-api)).

## Can spinf-serve make decisions in real time?

It is built for throughput: one packed forward pass at a time, so a single small request waits for the pass in front
of it. For live decisions, run a dedicated server (or several, behind your load balancer) sized to your traffic and
let its warm-up finish before routing traffic to it (`/readyz` turns 200 only then, with the default `--warmup full`); a
16k-token pass on a 4090 takes about two seconds.

## Do you publish benchmarks?

Yes. [spinf-benchmarks](https://github.com/spinfinc/spinf-benchmarks) on GitHub has the test sets (synthetic and
public), the exact questions and example sets, the results per model, and the code to re-run everything against a
server or the hosted API.

## Where do I report a problem?

Open an issue on [spinf-serve](https://github.com/spinfinc/spinf-serve). Include the `system_fingerprint` and, for a
request, the `X-Request-Id` of the response or error.
