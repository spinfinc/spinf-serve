# Acknowledgements

spinf-serve stands on other people's open work. This file names it: the models we run and re-quantize, the projects we
build on, the techniques we implement, and the data we checked our calibrations and prompts on. The legal attributions
for code included in this repository are in `NOTICE`; each checkpoint we publish credits its source in its model card
(`quantizations/*/MODEL_CARD.md`). The prompt packs have their own acknowledgements in the spinf-prompts repository.

Rule for contributors: the change that adds a dependency, vendors code, adopts a recipe, adds a model or checks results
on a dataset also adds the line here (and in `NOTICE` when code is included).

## Models and checkpoints

| what | by | terms | how we use it |
|---|---|---|---|
| [Gemma 4](https://huggingface.co/google) 12B and 31B | Google DeepMind | Apache 2.0 (Gemma 4; Google's licence page: ai.google.dev/gemma/docs/gemma_4_license) | the `spinf/gemma-4-12b-fp8` / `spinf/gemma-4-31b-fp8` models: quantized copies with the original notices retained |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Alibaba Qwen team | Apache 2.0 | the `spinf/qwen3.8-27b-int4` model (our GPTQ int4 copy) |
| [pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) | Perplexity AI | Apache 2.0 (derived from Qwen3.8-27B, revision 1d4bf0f2) | the decision model with a 255-code readout; our GPTQ int4 copy |
| [btbtyler09/Qwen3.8-27B-GPTQ-4bit](https://huggingface.co/btbtyler09/Qwen3.8-27B-GPTQ-4bit) | btbtyler09 | see the repo | the GPTQ recipe we adopted (GPTQModel 6.0.3, group 32, FOEM) and the checkpoint we served before publishing our own |

## Projects

- [PyTorch](https://pytorch.org) (inductor, FlexAttention, `torch.compile`), [Triton](https://github.com/triton-lang/triton): the kernels.
- [Hugging Face](https://huggingface.co) transformers, tokenizers, safetensors, huggingface_hub: model definitions (the Gemma 4 and
  Qwen3.5 modules, the vision towers), tokenization, weights.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (Songlin Yang, Yu Zhang and contributors): the Gated
  DeltaNet chunk kernels; one state kernel is vendored with changes (`NOTICE`).
- [GPTQModel](https://github.com/modelcloud/gptqmodel) (ModelCloud): the int4 quantization of our Qwen-architecture checkpoints.
- [vLLM](https://github.com/vllm-project/vllm): the reference for the fp8 scheme's numerics (`--quantization fp8`), the engine we
  compare throughput against, and the OpenAI-compatible server shape we follow.
- [FastAPI](https://fastapi.tiangolo.com), [uvicorn](https://www.uvicorn.org), [pydantic](https://docs.pydantic.dev): the server and the protocol models.

## Techniques

- GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, 2022) and FOEM (First-Order Error Matters, AAAI 2026): the int4 weights.
- LLM.int8() (Dettmers, Lewis, Belkada, Zettlemoyer, 2022): the mixed-precision decomposition our i8 split scheme fixes in advance
  (the outlier channels kept exact, the rest int8 per token).
- SmoothQuant (Xiao et al., 2022): evaluated as an alternative to the outlier split (docs/findings/QWEN.md).
- Gated DeltaNet (Yang, Kautz, Hatamizadeh, 2024): the recurrent layers of the Qwen3.5 architecture.
- FlexAttention (PyTorch): the tree masks that let many prompts share one read of the content.

## Data

Used to calibrate the fp8 activation scales and the outlier channels and to check accuracy; never bundled, never used to
train anything that ships in this repository.

| dataset | used for | terms |
|---|---|---|
| [Wikinews](https://en.wikinews.org) articles | the public parity fixture and benchmark job set (`fixtures/`, 24 articles; each one's title, URL and revision in the fixture's `sources`) | CC BY 2.5 |
| Imagenette and Imagewoof (fast.ai) | image calibration and accuracy suites | Apache 2.0 |
| UCF101 | video suites | see the dataset's terms |
| ESC-50 | audio calibration and suite | CC BY-NC 3.0 (calibration and checks only) |
| GTZAN | audio suite | see the dataset's terms |
| RAGTruth | the outlier-channel calibration stream; a decision benchmark | see the dataset's terms |
| evol-codealpaca-v1 and C4 | GPTQ calibration samples (the adopted recipe) | see each dataset's terms (C4: ODC-BY) |
| the decision benchmarks of the pplx-decider card (Belebele, TruthfulQA, WinoGrande, Circa, FinancialPhraseBank, TabFact, RAGTruth, ContractNLI, JudgeBench, BBH, JevBench) | accuracy checks of the Qwen-architecture models | each set's own terms |

The news and moderation texts of our internal test bed are not redistributed; their sources are listed in that repository.
