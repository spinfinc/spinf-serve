"""The stack interface (plan §4): one Stack per architecture family (models/registry.py), picked by the manifest's `family` or, for a
checkpoint run by its path, by its config.json. A stack never reads the environment or module globals: it builds its model from
(EngineConfig, model_dir, device) and gives the Scorer what differs between families: how images are preprocessed, the host-side plan
of a packed batch, the staging and forward of a stream, the capture of extraction ranges and the decoder that consumes them. Everything
above models/ works on Batch arrays and hidden rows; it knows the head is an LM head or a readout and nothing else about the family."""

from contextlib import contextmanager

import numpy as np


def compile_options(cfg) -> dict | None:
    """inductor options of every torch.compile in the engine. {"deterministic": True} goes in as a compile option, not as
    torch._inductor.config.deterministic, because Dynamo re-applies torch.use_deterministic_algorithms(<saved>) around every frame
    it traces, and that call writes inductor's `deterministic` (torch 2.14): set globally, it was back to False before any kernel
    was generated, so the RMSNorm reduction's block size was picked by on-device timing and each process landed on one of several
    score variants (BATCH_INVARIANCE.md §5)"""
    return {"deterministic": True} if cfg.deterministic_compile else None


class Stack:
    """the per-family behaviour; `sc` is the Scorer (its model, device, config, staging buffers and media cache)"""

    family = ""
    image_scheme = "gemma"  # media.SCHEME["image"]: how the protocol decodes images for this model
    # recurrent + attention layers (Qwen): a TreePlan per batch on the host, extraction prompts read through the packed path
    hybrid = False
    bidirectional_images = True  # an image's tokens attend to each other both ways in the packed mask; False: causal, as text

    @staticmethod
    def detect(model_dir: str) -> bool:
        """whether the checkpoint in model_dir is this family's (its config.json model_type)"""
        raise NotImplementedError

    def build(self, cfg, model_dir: str, device):
        """-> (model, text config); model._quant, model._lm_weight or model._readout, model._media, model._scale_table set"""
        raise NotImplementedError

    def capture_bytes(self, model) -> tuple[int, int, int]:
        """what a forward captures for an extraction branch (Scorer._x_tree_bytes): (K/V bytes per token over the layers that store
        it, recurrent bytes per template row, state bytes per group). Default: transformers' attention layers, the K and V
        projections' widths in bf16 (a layer sharing another's K/V has none, stores none)"""
        tok = 0
        for layer in model.layers:
            at = getattr(layer, "self_attn", None)
            if at is not None and getattr(at, "k_proj", None) is not None and getattr(at, "v_proj", None) is not None:
                tok += (at.k_proj.out_features + at.v_proj.out_features) * 2
        return tok, 0, 0

    def plan(self, sc, b) -> None:
        """host-side preparation of a packed batch, ahead of the GPU (the producer thread); nothing by default"""

    def stage(self, sc, b, st: dict) -> dict:
        """the batch's inputs into the pinned staging buffers `st` and on to the device -> {"ids": [1, n], "pos": [1, n]} (what
        forward reads); a stack with its own layout (compact rows) returns its own"""
        torch = sc.torch
        n = b.length
        st["ids"][:n].copy_(torch.from_numpy(b.ids))
        st["pos"][:n].copy_(torch.from_numpy(b.pos.astype(np.int64)))
        return {"ids": st["ids"][:n].to(sc.device, non_blocking=True)[None], "pos": st["pos"][:n].to(sc.device, non_blocking=True)[None]}

    def forward(self, sc, b, inputs: dict, masks: dict):
        """the packed forward -> final hidden states [1, n, hidden] in the stream layout"""
        raise NotImplementedError

    @contextmanager
    def capture(self, sc, b, xo: list):
        """around the forward of a batch whose owners `xo` have extraction branches: yields the capture dict the forward fills,
        {"kv": {key: {layer: (k, v)}}, "extra": per-family state or None, "pos0": per prompt key the first generated position or None}"""
        raise NotImplementedError

    def group_extra(self, sc, g, o, cap: dict, S: int) -> None:
        """the owner o's per-family captures into its decode.Group g (g.extra, g.extra_bytes); nothing for attention-only models"""

    def chunk_extra_bytes(self, model, groups: list) -> int:
        """the decoder's per-family buffers for groups decoded together (next to decode.chunk_bytes)"""
        return 0

    def sub_extra_bytes(self, model, sub) -> int:
        return 0

    def decoder(self, sc, chunk: list):
        from spinf_serve import decode

        return decode.Decoder(sc, chunk)

    def abandon(self, sc) -> None:
        """drop a capture left open by a failed call"""
