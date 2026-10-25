---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: perplexity-ai/pplx-decider-v1-27b
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
  - readout
---
# spinf pplx-decider 27B (GPTQ int4)

A quantized copy of [Perplexity pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) prepared by [spinf](https://github.com/spinfinc/spinf-serve) for
**decision inference**: one input read once, hundreds of questions scored in parallel, scores that do not change with what else
is in the batch. Weights in GPTQ int4 (group 32, symmetric, no act-order, FOEM) with GPTQModel 6.0.3; the 255-code decision readout and its temperature unchanged. About 17 GB on disk.

**Run it with [spinf-serve](https://github.com/spinfinc/spinf-serve)**, the open-source inference engine for decision models:

```sh
pip install spinf-serve[server]
spinf-serve serve spinf/pplx-decider-v1-27b-int4            # an OpenAI-compatible server on port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="spinf/pplx-decider-v1-27b-int4",
    messages=[{"role": "user", "content": "Apple reported record iPhone revenue while Mac margins slipped."}],
    extra_body={"scoring": {"queries": [
        {"id": "direction", "template": "\n\nWill Apple's revenue rise or fall next quarter?\nAnswer:{?}", "options": [" rise", " fall"]},
    ]}},
)
```

The same request, with the same results, runs against the hosted API at `api.spinf.com` by changing the base URL and the key.

## Why this copy

- **Fits on one 24 GB GPU (RTX 4090 / 5090 class).** int4 weights kept packed on the card, int8 tensor-core GEMMs for the activations, as for spinf/qwen3.8-27b-int4.
- **Batch-invariant scores.** An input's scores are bit-identical whether it is scored alone or packed with other inputs
  (spinf-serve `docs/findings/BATCH_INVARIANCE.md`). The quantization parameters that make this possible are in the checkpoint.
- **Pinned and parity-tested.** spinf-serve names this repo at an exact revision and ships a parity reference recorded on it;
  a new revision is never served untested.
- **Media:** text at launch (the vision tower is in the checkpoint and not yet validated on spinf-serve for this model).

## What spinf adds to this repo

- `spinf_outliers.json`: the outlier channels kept exact in the int8 scheme, as for spinf/qwen3.8-27b-int4 (exported on these weights).
- Nothing else: `readout.safetensors` and `decision_config.json` are the original's.

Nothing in this repo is required by other engines: the checkpoint is a standard quantized checkpoint and loads as one elsewhere.

## What is different from the original

A decision model: no LM head, so no text generation or extraction; questions are written in the decider's own protocol
(State / Question / Options, answer codes as the scored options) and spinf-serve's packs for it do that. On the 11 decision
sets of the original's card (public test splits, 20,156 items), this int4 copy scores a mean of 83.0 through the decider's protocol
against 84.3 reported for the original (plain mean of the card's rows; above it on Belebele, WinoGrande, ContractNLI and JevBench,
below on Circa, FinancialPhraseBank and BBH). The int8 activation scheme and fp16 activations give the same answer on 98.9 % of a
440-item sample. Not a fine-tune of the decider; its readout is unchanged.

## Provenance

The recipe and this card are kept in spinf-serve under `quantizations/pplx-decider-v1-27b-int4/` (`recipe.yaml`); the script that
produced the weights is `quantizations/gptq_quantize.py` (GPTQModel 6.0.3, 256 calibration samples, seed 20261006) and the one that
produced `spinf_outliers.json` is `quantizations/export_outliers.py`. Source: perplexity-ai/pplx-decider-v1-27b @ d8c3bd18e31dc1804433d38528b3e770019bf00c, quantized on 2026-10-14.
`spinf_quantize.json` in this repo records the source revision, the seed, the calibration digest and the tool versions.

## Licence and attribution

Perplexity pplx-decider-v1-27b is released by its authors under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0); this quantized copy is distributed under the
same licence, with the original notices retained and the modified files marked as such. "Perplexity pplx-decider-v1-27b" and the model
names are their owners' trademarks, used only to describe the origin of this copy; no endorsement is implied. See spinf-serve's
`ACKNOWLEDGEMENTS.md` for everything this work builds on.

## Quantization details (as recorded on 2026-10-06)

GPTQ + FOEM 4-bit quantization of [perplexity-ai/pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b)
(Apache 2.0, a Qwen3.8-27B fine-tune with a 255-code decision readout instead of the LM head), with the exact recipe of
[btbtyler09/Qwen3.8-27B-GPTQ-4bit](https://huggingface.co/btbtyler09/Qwen3.8-27B-GPTQ-4bit): GPTQModel 6.0.3, 4 bits, group 32,
symmetric, no act-order, FOEM alpha 0.25 beta 0.2, mse 2.0, act_group_aware, 256 calibration samples from evol-codealpaca-v1 and
C4 binned over 256-2048 tokens (`quantizations/gptq_quantize.py` in spinf-serve, one GPU, a fixed seed). Quantized on one RTX 6000 Ada in 2 h 10 min.

- Text decoder linears int4 (400 tensors); `linear_attn.in_proj_a/b`, norms, embeddings and the vision tower (333 tensors) bf16;
  `readout.safetensors` ([255, 5120]) and `decision_config.json` (codes, token ids, temperature 2.2076) unchanged from the decider.
- No LM head: the decider has none (the random one GPTQModel's loader created was dropped from shard 2). Loading needs a class
  without a head (`Qwen3_5Model`) or tolerates a missing `lm_head.weight`.
- Inference = the decider's own protocol (its `source/src/autojev/model.py`): chat template with thinking off, prompt
  "State / Question / Options / Return only the letter code", last token's hidden state x readout / temperature, softmax over the
  question's options.
- 17.6 GB. Smoke check against the bf16 decider: option probabilities within 0.01 on the same rows.
