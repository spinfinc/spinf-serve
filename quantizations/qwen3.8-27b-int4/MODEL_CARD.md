---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-classification
tags:
  - spinf-serve
  - decision-model
  - batch-invariant
  - gptq
  - int4
  - qwen3_5
---
# spinf Qwen3.8 27B (GPTQ int4)

A quantized copy of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) prepared by [spinf](https://github.com/spinfinc/spinf-serve) for
**decision inference**: one input read once, hundreds of questions scored in parallel, scores that do not change with what else
is in the batch. Weights in GPTQ int4 (group 32, symmetric, no act-order, FOEM) with GPTQModel 6.0.3. About 19 GB on disk against 55 GB for the bf16 original.

**Run it with [spinf-serve](https://github.com/spinfinc/spinf-serve)**, the open-source inference engine for decision models:

```sh
pip install spinf-serve[server]
spinf-serve serve spinf/qwen3.8-27b-int4            # an OpenAI-compatible server on port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="spinf/qwen3.8-27b-int4",
    messages=[{"role": "user", "content": "Apple reported record iPhone revenue while Mac margins slipped."}],
    extra_body={"scoring": {"queries": [
        {"id": "direction", "template": "\n\nWill Apple's revenue rise or fall next quarter?\nAnswer:{?}", "options": [" rise", " fall"]},
    ]}},
)
```

The same request, with the same results, runs against the hosted API at `api.spinf.com` by changing the base URL and the key.

## Why this copy

- **Fits on one 24 GB GPU (RTX 4090 / 5090 class).** int4 weights kept packed on the card, int8 tensor-core GEMMs for the activations; about 19 GB in use at a 16k-token packed batch.
- **Batch-invariant scores.** An input's scores are bit-identical whether it is scored alone or packed with other inputs
  (spinf-serve `docs/BATCH_INVARIANCE.md`). The quantization parameters that make this possible are in the checkpoint.
- **Pinned and parity-tested.** spinf-serve names this repo at an exact revision and ships a parity reference recorded on it;
  a new revision is never served untested.
- **Media:** images and video (the vision tower, kept in bf16); no audio in this architecture.

## What spinf adds to this repo

- `spinf_outliers.json`: for each norm-fed linear, the 64 input channels whose activations carry fixed-position outliers.
  spinf-serve keeps those channels exact in fp16 and runs the rest in int8 per token (the mixed-precision idea of LLM.int8()
  with the channel set fixed in advance), which is what makes int8 activations accurate and batch-invariant here. Without the
  file the engine runs fp16 activations: the standard behaviour, about 20 % slower.

Nothing in this repo is required by other engines: the checkpoint is a standard quantized checkpoint and loads as one elsewhere.

## What is different from the original

The GPTQ int4 error against the bf16 original: on 13k decision reads, mean |Δ log-prob| 0.06 with the int8 activation scheme,
the top option agreeing with bf16 on more than 99 % of reads (spinf-serve `docs/QWEN.md`, measured on the copy below with
the same recipe). On eleven public decision benchmarks (20k test items, spinf's label-free prompts) this copy scores a mean of 79.0
with the int8 activation scheme, against 78.4 for that earlier copy; the int8 scheme and fp16 activations agree on the sign of
99.3 % of 1,328 reference reads. The recipe is the one of btbtyler09/Qwen3.8-27B-GPTQ-4bit, which we served before publishing our
own copy.
Not a fine-tune; tokenizer and chat template unchanged.

## Provenance

The recipe and this card are kept in spinf-serve under `quantizations/qwen3.8-27b-int4/` (`recipe.yaml`); the script that
produced the weights is `quantizations/gptq_quantize.py` (GPTQModel 6.0.3, 256 calibration samples, seed 20261006) and the one that
produced `spinf_outliers.json` is `quantizations/export_outliers.py`. Source: Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, quantized on 2026-10-14.
`spinf_quantize.json` in this repo records the source revision, the seed, the calibration digest and the tool versions.

## Licence and attribution

Qwen3.8-27B is released by its authors under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0); this quantized copy is distributed under the
same licence, with the original notices retained and the modified files marked as such. "Qwen3.8-27B" and the model
names are their owners' trademarks, used only to describe the origin of this copy; no endorsement is implied. See spinf-serve's
`ACKNOWLEDGEMENTS.md` for everything this work builds on.
