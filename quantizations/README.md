# Quantizations: the spinf checkpoints on the Hub

One folder per checkpoint we publish under the `spinf` organisation, named as the Hub repo (= the served model id, plan §3): its recipe (`recipe.yaml`), the calibration output the
recipe produced and the model card to upload. The weights themselves are not in git: `spinf-serve quantize <folder> --out DIR`
produces them from the source checkpoint, locally in the maintainers' model store (not in git), and the upload pushes that directory
with these files in it. The engine reads quantization parameters from the checkpoint only (plan §2.4), so everything a model
needs to run is in its Hub repo; this folder is the record of how it was made.

| folder | source | scheme | status (2026-10-16) | missing |
|---|---|---|---|---|
| `gemma-4-12b-fp8` | google/gemma-4-12B @ 023679ed | fp8 weights, per-tensor weight scales, static per-tensor activation scales (compressed-tensors FP8 static) | scales calibrated (2026-10-01, testbed v1, 944k tokens, 4090); **checkpoint written 2026-10-15** (`fp8_quantize.py` on a 4090, 13.0 GB, 3 shards, 328 linears, in the maintainers' model store), parity bit-identical to the load-time fp8s path (1,328 reads); card done; **uploaded 2026-10-16** as a private repo at `a689df2c`, the manifest pins it | make the repo public at launch |
| `gemma-4-31b-fp8` | google/gemma-4-31B @ 5bbc2fb1 | same | scales calibrated (2026-10-02, testbed v1 minus audio and 3 long docs, 810k tokens, L40S); **checkpoint written 2026-10-15** (`fp8_quantize.py` on an RTX PRO 6000 pod and, byte-identically, on a 4090; 33.3 GB, 7 shards, 410 linears), parity bit-identical to the load-time fp8s path (1,328 reads); card done; **uploaded 2026-10-16** as a private repo at `8c54e985`, the manifest pins it | make the repo public at launch |
| `qwen3.8-27b-int4` | Qwen/Qwen3.8-27B | GPTQ int4, group 32, symmetric, no act-order, FOEM (GPTQModel 6.0.3) | **quantized 2026-10-14** (`gptq_quantize.py`, A40 pod, 115 min, 19 GB, in the maintainers' model store); `spinf_outliers.json` exported 2026-10-14 (`export_outliers.py`); f16 and i8 parity references recorded (i8 vs f16: 99.25 % sign agreement); **11-set run done** (79.0 mean with spinf's prompts, i8; the third-party copy 78.4); **uploaded 2026-10-15** as a private repo at `22de4859`, the manifest pins it | make the repo public at launch |
| `pplx-decider-v1-27b-int4` | perplexity-ai/pplx-decider-v1-27b | same recipe | **quantized 2026-10-14** (`gptq_quantize.py`, A40 pod, 126 min, 17 GB, in the maintainers' model store; replaces the 2026-10-06 copy made with a derived script); smoke-tested on the engine (440 decider questions: 81.6 % vs 80.5 % for the 2026-10-06 copy, 97.7 % same answers); card drafted; `spinf_outliers.json` exported 2026-10-14; **11-set run done** (83.0 mean through the decider's protocol, i8; the original's card 84.3); **uploaded 2026-10-15** as a private repo at `5579a6ec`, the manifest pins it (status available) | make the repo public at launch |

Files per folder:

- `recipe.yaml`: source repo and revision, scheme, tool versions, calibration corpus, what the command writes.
- `calibration.json` (FP8 static): the calibrate output as it is (`spinf_serve/quant/calibrate.py` on the bf16 original): `scales`
  per linear (powers of two, headroom 2 over the largest input seen), `amax` and where it came from, the corpus composition. The
  checkpoint is written by `fp8_quantize.py` at this level: each text-decoder linear as fp8 `weight`, `weight_scale` and the table's
  scale as `input_scale` (compressed-tensors float-quantized, `quantization_config` in config.json, `spinf_serve/quant/fp8.py`),
  everything else bf16; `calibration.json` and `spinf_quantize.json` ship inside the repo. Written on a GPU: the CPU's division and
  fp8 cast differ in the last bit from CUDA's, and a CPU-written copy scored 0.08-0.10 mean |Δ log-prob| from the load-time scheme
  (2026-10-15), where the GPU-written one reproduces the recorded fp8s reference bit for bit.
- `spinf_outliers.json` (int4): the 64 input channels per norm-fed linear kept exact in the i8 split scheme (plan §2.4), written by
  `export_outliers.py` at this level (the engine in bf16 mode, one packed calibration stream, per-channel absmax). Ships inside the
  model repo under the same name, where the engine reads it; optional for any other engine. The third-party copy's table stays in
  the package (`quant/tables/`) for anyone still running that copy by its Hub id; no manifest points at it since 2026-10-15.
- `quantize_config.json` (GPTQ): what GPTQModel wrote; the script is `gptq_quantize.py` at this level, shared by the two
  Qwen-architecture checkpoints (the recipe of btbtyler09/Qwen3.8-27B-GPTQ-4bit, our own code; it also writes `spinf_quantize.json`
  with the source revision, the seed and the calibration digest into the checkpoint).
- `MODEL_CARD.md`: the README of the Hub repo (front matter: `license`, `license_link`, `base_model`, `base_model_relation: quantized`,
  tags; sections: how to run with spinf-serve, why this copy, what spinf adds, what differs from the original, provenance, licence).

Rules: a recipe names an exact source revision; a new source revision is a new quantization (new folder entry or a new
`recipe.yaml` revision), never an edit of the calibration in place; every upload gets a parity reference recorded by the engine
on the uploaded revision before a manifest points at it.
