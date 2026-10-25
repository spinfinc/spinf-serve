"""Engine settings. The library never reads the environment on import: adapters build an EngineConfig (from_env or directly)."""

import os
from dataclasses import dataclass, fields

QUANTS = ("fp8s", "fp8", "fp8row", "w8a16", "w8a8", "bf16", "f16", "i8")  # f16, i8: int4 checkpoints (qwen.py) only
# lm head (read rows only): bf16 = bf16 logits as vLLM (the original production numerics; log-probs land on 1/8 steps above |16|),
# f32out = the same bf16 matmul written to fp32 logits (no rounding of the logits), fp32 = fp32 weights (research, +4 GB transient)
LM_HEADS = ("bf16", "f32out", "fp32")
WARMUPS = ("full", "short", "none")


@dataclass(frozen=True)
class EngineConfig:
    # the manifest this engine runs (manifests/: id or alias; empty = no manifest, the fields below stand alone). With a manifest,
    # requests may name the model by any of its aliases
    model: str = ""
    # the public model name this engine serves (requests for another model are rejected)
    served_model: str = "spinf/gemma-4-12b-fp8"
    model_id: str = "google/gemma-4-12B"
    # the checkpoint revision (Hub commit hash) to load; empty = whatever snapshot the cache holds or the Hub serves (unpinned,
    # logged as a warning: the quantization parameters and the parity references name a revision)
    revision: str = ""
    # local snapshot directory; empty = look it up (weights.find_weights)
    model_dir: str = ""
    # one of QUANTS; empty = auto for the card (models/gemma4.py auto_quant: fp8s where the card has fp8 and the model its scales)
    quant: str = ""
    # packed tokens per forward: 16k is the fastest for news-length content on a 4090
    token_budget: int = 16384
    # the largest single input (content + its prompts) the engine runs: one over token_budget runs alone in its own stream (a
    # "solo" batch) instead of being rejected; the "max" content tier of an account is this, measured per model and card
    max_tokens: int = 65536
    # extraction slots decoded after a scoring pass: the decoder's K/V (prefixes, templates, generated tokens) per chunk of
    # trees; more trees than fit are decoded in several chunks
    decode_budget_gib: float = 4.0
    # the decoder's own working set per decode (decode.group_bytes: the K/V and, on the hybrid, the recurrent rows of every
    # slot's BEAM sequences over max_tokens steps): more slots than fit are decoded in several chunks (Scorer._x_flush), a group
    # with too many slots in runs of its prompts. Qwen sets its own (qwen.py: 3 GiB next to its 16k forward on 24 GB)
    decoder_budget_gib: float = 4.0
    # what one forward may capture for the extraction branches packed in its stream (prefix K/V, template rows, states:
    # Scorer._x_tree_bytes): the packer puts fewer extraction inputs in a stream above it, since the captures exist next to the
    # forward's own peak. Qwen sets its own (qwen.py: 1 GiB)
    capture_budget_gib: float = 2.0
    # extraction slots as branches of the scoring trees, decoded in batches (decode.py); false = one by one after their own read
    # (the reference path, for checks)
    x_batched: bool = True
    # replay the extraction decode step from a CUDA graph (captured once per decode). Off: measured no gain on a 4090 (the
    # step is GPU-bound at these batch sizes: small fp8 matmuls over all weights + the x_decode attention), kept as an option
    x_cuda_graphs: bool = False
    # batches packed ahead of the GPU
    prefetch: int = 6
    # read rows per lm-head chunk: bounds the [rows, vocab] fp32 temporaries (~3 GiB at 1024 rows, 262k vocab)
    lm_rows: int = 1024
    # batch invariance (docs/BATCH_INVARIANCE.md): every job starts on a 128-token block, and the lm head always
    # multiplies lm_rows rows (a short last chunk is padded): with fp8s, a job scores bit-identically whatever it is packed with
    align_jobs: bool = True
    lm_fixed_rows: bool = True
    # with align_jobs, the shortest stream: below 384 rows the fp8 GEMM of the full layers' k/v projection (3840 -> 512) takes
    # another kernel, so a short job alone would not score as in a longer stream (BATCH_INVARIANCE.md)
    min_stream: int = 1024
    # fp8s diagnostics only (testbed / release checks, never live): one forward in N also keeps each linear's largest input / its
    # clip level (Scorer.fp8s_peak; above 1 = clipped, logged). The monitored forwards compile to another graph whose fusion
    # changes the numerics, so with it on scores are neither reproducible nor batch-invariant (BATCH_INVARIANCE.md); 0 = off
    fp8s_monitor_every: int = 0
    # torch.compile the decoder layers (off only for debugging: 2-3x slower)
    compile: bool = True
    # what runs before the engine reports ready (service.warmup), one of WARMUPS: "full" = short streams, one stream at the token
    # budget and the extraction decode path (the compiles a request would otherwise pay: ~1.5-2 min on a 4090 for the 12B);
    # "short" = the short streams and the extraction path; "none" = nothing (the first requests compile)
    warmup: str = "full"
    # inductor's deterministic mode: no on-device benchmarking where it changes numerics. Without it a fresh compile picks one
    # of two fp8 (or w8a16) kernel variants at random, sd ~0.15 logit per read (docs/PRECISION.md §2); no speed cost
    deterministic_compile: bool = True
    # load the image / audio embedders (the 12B "unified" checkpoint: ~53M bf16 parameters, ~0.1 GB) so image_url and
    # input_audio parts can be read; the API decides who may send them
    media: bool = True
    # debug: keep only the first n decoder layers (0 = all)
    layers: int = 0
    # lm head precision, one of LM_HEADS (f32out: no cost, ~10 % less packing noise than bf16, PRECISION.md §3)
    lm_head: str = "f32out"
    # the stack's research knobs as "key=value,key=value" (models/qwen35/qwen.py QwenOptions; the Gemma stack has none); empty =
    # the production configuration
    stack_options: str = ""

    def __post_init__(self) -> None:
        if self.quant and self.quant not in QUANTS:
            raise ValueError(f"quant {self.quant!r}: one of {', '.join(QUANTS)} or empty for auto")
        if self.lm_head not in LM_HEADS:
            raise ValueError(f"lm_head {self.lm_head!r}: one of {', '.join(LM_HEADS)}")
        if self.warmup not in WARMUPS:
            raise ValueError(f"warmup {self.warmup!r}: one of {', '.join(WARMUPS)}")

    @classmethod
    def from_manifest(cls, name: str, **overrides) -> "EngineConfig":
        """the config of a manifest (manifests.resolve: id or alias), its defaults overridden by the given fields"""
        from spinf_serve import manifests

        return cls(**{**manifests.resolve(name).engine_defaults(), **overrides})

    @classmethod
    def from_env(cls, prefix: str = "SPINF_") -> "EngineConfig":
        """SPINF_MODEL (a manifest: its defaults first), then SPINF_MODEL_ID, SPINF_MODEL_DIR, SPINF_QUANT, SPINF_TOKEN_BUDGET, ...
        (field names upper-cased) on top"""
        kw: dict[str, object] = {}
        for f in fields(cls):
            v = os.environ.get(prefix + f.name.upper())
            if v is None or v == "":
                continue
            if f.type in (bool, "bool"):
                kw[f.name] = v.lower() in ("1", "true", "yes")
            elif f.type in (int, "int"):
                kw[f.name] = int(v)
            elif f.type in (float, "float"):
                kw[f.name] = float(v)
            else:
                kw[f.name] = v
        if kw.get("model"):
            return cls.from_manifest(str(kw.pop("model")), **kw)
        return cls(**kw)
