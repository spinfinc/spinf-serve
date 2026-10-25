---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-12B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-classification
tags:
  - spinf-serve
  - decision-model
  - batch-invariant
  - fp8
  - compressed-tensors
  - gemma4
---
# spinf Gemma 4 12B (FP8, static scales)

A quantized copy of [Google Gemma 4 12B](https://huggingface.co/google/gemma-4-12B) prepared by [spinf](https://github.com/spinfinc/spinf-serve) for
**decision inference**: one input read once, hundreds of questions scored in parallel, scores that do not change with what else
is in the batch. Weights in FP8 (e4m3, per-tensor scales) with **static per-tensor activation scales** calibrated by spinf, in the compressed-tensors FP8 format. About 12 GB on disk against 22 GB for the bf16 original.

**Run it with [spinf-serve](https://github.com/spinfinc/spinf-serve)**, the open-source inference engine for decision models:

```sh
pip install spinf-serve[server]
spinf-serve serve spinf/gemma-4-12b-fp8            # an OpenAI-compatible server on port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="spinf/gemma-4-12b-fp8",
    messages=[{"role": "user", "content": "Apple reported record iPhone revenue while Mac margins slipped."}],
    extra_body={"scoring": {"queries": [
        {"id": "direction", "template": "\n\nWill Apple's revenue rise or fall next quarter?\nAnswer:{?}", "options": [" rise", " fall"]},
    ]}},
)
```

The same request, with the same results, runs against the hosted API at `api.spinf.com` by changing the base URL and the key.

## Why this copy

- **Fits on one 24 GB GPU (RTX 4090 / 5090, L4, L40S, A5000 class with fp8 tensor cores: Ada, Hopper, Blackwell).** FP8 weights and the fp8 tensor cores; about 14 GB of card memory in use at a 16k-token packed batch, so the rest holds the batch, images and extraction state.
- **Batch-invariant scores.** An input's scores are bit-identical whether it is scored alone or packed with other inputs
  (spinf-serve `docs/findings/BATCH_INVARIANCE.md`). The quantization parameters that make this possible are in the checkpoint.
- **Pinned and parity-tested.** spinf-serve names this repo at an exact revision and ships a parity reference recorded on it;
  a new revision is never served untested.
- **Media:** images, audio and video (the checkpoint's own embedders, kept in bf16, never quantized).

## What spinf adds to this repo

- The static activation scales (`input_scale` per linear): calibrated as the largest input each linear sees over a broad corpus
  (news, moderation, multilingual, long documents, odd inputs, images, audio, video; ~944k tokens) with 2x headroom. They are what
  removes the dependence of a score on the rest of the batch; an engine that ignores them runs dynamic scales and is not batch-invariant.
- `calibration.json`: the calibration record (per-linear maxima, where they came from, the corpus composition), for provenance.

Nothing in this repo is required by other engines: the checkpoint is a standard quantized checkpoint and loads as one elsewhere.

## What is different from the original

The usual FP8 quantization error against the bf16 original. On our labelled suites (image, audio, video and multilingual
classification) the static-scale scheme equals dynamic FP8 within noise; on news scoring it drifts from dynamic FP8 by about as
much as dynamic FP8's own batch noise (details in spinf-serve `docs/findings/BATCH_INVARIANCE.md` and `PRECISION.md`).
Not a fine-tune: the weights are the original's, quantized; the tokenizer and the chat template are unchanged.

## Provenance

The recipe, the calibration output and this card are kept in spinf-serve under `quantizations/gemma-4-12b-fp8/`
(`recipe.yaml`). Source: `google/gemma-4-12B` at revision `023679ed352de9bb66cc873c9009ce3482585c08`. The activation scales were
calibrated on 2026-10-01 (`calibration.json`); the weights were written on 2026-10-15 with spinf-serve's `quantizations/fp8_quantize.py`
(fp8 per-tensor weight scales, the table's scales as `input_scale`). spinf-serve's parity check scores this checkpoint bit-identically
to the bf16 original quantized at load time with the same scales (1,328 reference reads, RTX 4090). `spinf_quantize.json` in this
repo records the source revision, the calibration digest and the tool versions.

## Licence and attribution

Google Gemma 4 12B is released by its authors under the Apache License 2.0 (https://ai.google.dev/gemma/docs/gemma_4_license); this quantized copy is distributed under the
same licence, with the original notices retained and the modified files marked as such. "Google Gemma 4 12B" and the model
names are their owners' trademarks, used only to describe the origin of this copy; no endorsement is implied. See spinf-serve's
`ACKNOWLEDGEMENTS.md` for everything this work builds on.
