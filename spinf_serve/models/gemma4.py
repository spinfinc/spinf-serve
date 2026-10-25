"""The Gemma 4 stack (Gemma4Stack, models/base.py): the text model from a safetensors checkpoint, linear weights quantized at load
time (quant/linears.py) or read pre-quantized, the media embedders, and the packed forward with K/V capture for extraction.

Origin: spinf's internal scorer (torchscore.py), single-device path only. The numerics are kept identical to the recorded parity
references: fp8 scores change with incidental kernel choices (docs/PRECISION.md).

Schemes (quant):
  fp8s   fp8 with fixed (calibrated) per-layer activation scales: from the checkpoint when it is an fp8-static one (the spinf/...
         uploads: fp8 weights, weight_scale and input_scale per linear, quant/fp8.py), else from a scale table of the revision
         (quant/tables/, quant/calibrate.py). No scale shared across the packed stream, so with aligned jobs (EngineConfig.align_jobs)
         a job scores bit-identically whatever it is packed with (docs/BATCH_INVARIANCE.md). Same speed as fp8. The default where
         the card has fp8 and the model its scales.
An fp8-static checkpoint runs any scheme: fp8s and fp8 straight from its tensors, the others (fp8row, w8a16, w8a8, bf16) from the
weights dequantized to bf16 (weight * weight_scale), so they differ slightly from the same scheme on the bf16 original.
  fp8    e4m3 weights with per-tensor scales, dynamic per-tensor activation scales, torch._scaled_mm (as vLLM --quantization fp8);
         needs compute capability >= 8.9 (4090, 5090, L4/L40S, RTX PRO 6000, H100). The fastest. The activation scale is the
         max over the whole packed stream, so a score depends on what it is packed with.
  fp8row e4m3 weights with per-output-channel scales and per-token activation scales (row-wise _scaled_mm): no scale shared
         across the stream (research, opt-in).
  w8a16  int8 weights, per-output-channel scales, dequantized to bf16 in the matmul: Ampere (3090 / A5000), closer to bf16 than fp8.
  w8a8   int8 weights and per-token int8 activations (torch._int_mm): 1.5x fp8 speed, 2.4x its error. Opt-in only.
  bf16   no quantization: needs ~1.1x the checkpoint + 6 GiB.
"""

import glob
import json
import os
import time
from contextlib import contextmanager

from spinf_serve.config import EngineConfig
from spinf_serve.models.base import Stack, compile_options
from spinf_serve.quant.fp8 import is_fp8_static
from spinf_serve.quant.linears import quant_linears
from spinf_serve.quant.tables import load_scale_table, scale_table_path
from spinf_serve.weights import checkpoint_bytes

BF16_HEADROOM = 6 << 30  # activations + allocator (A6000 bf16: 24.2 GB peak on 22.3 GB of weights at 8k)
# head_dim -> BLOCK_M, BLOCK_N, num_stages, num_warps of the flex kernel: inductor's default tiles overflow the 4090's 99 KB shared
# memory (sliding layers have head_dim 256, the full-attention layers 512 with one KV head)
FLEX_TILES = {256: (64, 64, 2, 4), 512: (32, 32, 2, 4)}


def auto_quant(capability: tuple[int, int], vram_bytes: int, weight_bytes: int) -> str:
    """fp8 where the card has fp8 tensor cores; below that bf16 when the weights fit with room for the activations, else w8a16"""
    if capability >= (8, 9):
        return "fp8"
    return "bf16" if weight_bytes * 1.1 + BF16_HEADROOM <= vram_bytes else "w8a16"


def quant_scheme(cfg: EngineConfig, model_dir: str, device, prequant: bool = False) -> str:
    import torch

    q = cfg.quant or auto_quant(
        torch.cuda.get_device_capability(device),
        torch.cuda.get_device_properties(device).total_memory,
        checkpoint_bytes(model_dir) * (2 if prequant else 1),  # the bf16 size decides between bf16 and w8a16 on Ampere
    )
    if q == "fp8" and not cfg.quant and (prequant or load_scale_table(cfg.model_id, model_dir) is not None):
        q = "fp8s"
    if q in ("fp8s", "fp8", "fp8row") and torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError(f"quant {q} needs compute capability >= 8.9, {torch.cuda.get_device_name(device)} has no fp8 tensor cores")
    return q


# K/V capture for extraction (scorer.Scorer): while CAPTURE[device] = {"window", "ranges": [(start, n, key)], "kv": {}} is set,
# the flex attention hook keeps each layer's keys and values of stream positions start .. start + n - 1 under kv[key][layer]
# (sliding layers only their last window - 1, as transformers' DynamicCache keeps them), so a prefix read through the packed
# flex path (its own prefill, or the scoring pass of the same content) can seed a KV cache for decoding
CAPTURE: dict = {}


def register_flex(opts: dict | None) -> None:
    """attention interface "tree_flex": FlexAttention over the packed stream with the BlockMask passed as attention_mask (GQA handled
    by the kernel), compiled on its own and excluded from the decoder-layer compile"""
    import torch
    import torch._dynamo
    from torch.nn.attention.flex_attention import flex_attention
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import flex_attention_mask

    if "tree_flex" in AttentionInterface._global_mapping:
        return
    # compiled variants per code object: the decoder-layer forward compiles per layer instance (the 31B's layers also differ
    # in a guard on transformers' shared_kv_states: 60 layers hit a limit of 64 during plain scoring, and the layers past it ran
    # uncompiled), times the flex / extraction-decode paths and their shapes
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 1024))
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 8192))
    # per device: streams vary in length (multiples of BLOCK), a static kernel would recompile per length
    compiled = {}
    kopts = {d: dict(zip(("BLOCK_M", "BLOCK_N", "num_stages", "num_warps"), t, strict=True)) for d, t in FLEX_TILES.items()}

    # graph break: keeps inductor from fusing the norms / rope into the flex template, which overflows sm89 shared memory
    @torch.compiler.disable
    def tree_flex_forward(module, query, key, value, attention_mask, scaling=None, sliding_window=None, **kw):
        cap = CAPTURE.get(query.device)
        if cap is not None:
            keep = cap["window"] - 1
            sliding = module.config.layer_types[module.layer_idx] == "sliding_attention"
            for start, n, k in cap["ranges"]:
                a = start + (max(0, n - keep) if sliding else 0)
                cap["kv"].setdefault(k, {})[module.layer_idx] = (key[:, :, a : start + n].clone(), value[:, :, a : start + n].clone())
        fn = compiled.get(query.device)
        if fn is None:
            fn = compiled[query.device] = torch.compile(flex_attention, dynamic=True, options=opts)
        out = fn(
            query,
            key,
            value,
            block_mask=attention_mask,
            scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
            kernel_options=kopts.get(query.shape[-1]),
        )
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("tree_flex", tree_flex_forward)
    AttentionMaskInterface.register("tree_flex", flex_attention_mask)  # plain causal calls (no mask given) still work


def build_model(cfg: EngineConfig, model_dir: str, device):
    """-> (model, text config). model._quant = the scheme, model._lm_weight = the tied lm head (Gemma4Stack.build)."""
    import torch
    import torch.nn as nn
    from safetensors import safe_open
    from transformers import AutoConfig

    device = torch.device(device)
    hf_cfg = AutoConfig.from_pretrained(model_dir)
    if hf_cfg.model_type == "gemma4_unified":  # dense 12B / 31B
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextModel as TextModel,
        )
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextRotaryEmbedding as Rotary,
        )
    else:  # gemma4: E2B / E4B (per-layer input embeddings, KV-shared layers)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as TextModel
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding as Rotary

    opts = compile_options(cfg)
    L = quant_linears(opts, device)  # the quantized linears, compiled with this load's options

    tcfg = hf_cfg.text_config
    tcfg._attn_implementation = "tree_flex"
    register_flex(opts)
    prequant = is_fp8_static(model_dir)
    quant = quant_scheme(cfg, model_dir, device, prequant)
    table = None
    if quant == "fp8s" and not prequant:
        table = load_scale_table(cfg.model_id, model_dir)
        if table is None:
            raise RuntimeError(
                f"quant fp8s: no scale table for this revision ({scale_table_path(cfg.model_id, model_dir)}); "
                "make one with python -m spinf_serve.calibrate"
            )
    if cfg.layers:
        tcfg.num_hidden_layers = cfg.layers
        tcfg.layer_types = tcfg.layer_types[: cfg.layers]
    with torch.device("meta"):
        model = TextModel(tcfg)
    model.rotary_emb = Rotary(tcfg).to(device)  # its inv_freq buffers are not in the checkpoint
    mods = dict(model.named_modules())
    prefix = "model.language_model."
    t0 = time.perf_counter()
    n_lin = n_fp8 = 0

    def tensor(f, key):  # a scale next to its weight: in the open shard, else in the shard the index names
        if key in f.keys():  # noqa: SIM118
            return f.get_tensor(key)
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as idx:
            other = json.load(idx)["weight_map"][key]
        with safe_open(os.path.join(model_dir, other), "pt", device="cpu") as g:
            return g.get_tensor(key)

    def evict(path):  # drop the file's pages from the page cache (no root needed)
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    # the safetensors mmap leaves the file (23-58 GB) in the page cache: evict as we go so the host's free memory is not exhausted
    # (pinned allocations in the driver failed at 1 GB free, and low-memory watchdogs kill the process)
    for shard in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        nbytes = 0
        with safe_open(shard, "pt", device="cpu") as f:
            for k in [k for k in f.keys() if k.startswith(prefix)]:  # noqa: SIM118 (safe_open is not a dict)
                name = k[len(prefix) :]
                mod_name, _, pname = name.rpartition(".")
                if mod_name not in mods:  # a layer beyond cfg.layers
                    continue
                mod = mods[mod_name]
                if prequant and pname in ("weight_scale", "input_scale"):
                    continue  # read with the weight
                t = f.get_tensor(k)
                nbytes += t.numel() * t.element_size()
                if nbytes > 2 << 30:
                    torch.cuda.synchronize()
                    evict(shard)
                    nbytes = 0
                if isinstance(mod, nn.Linear) and pname == "weight" and t.dtype == torch.float8_e4m3fn:  # fp8-static checkpoint
                    n_fp8 += 1
                    parent_name, _, child = mod_name.rpartition(".")
                    ws, xs = tensor(f, k + "_scale"), tensor(f, prefix + mod_name + ".input_scale")
                    if quant == "fp8s":
                        lin = L.prequantized(L.Fp8StaticLinear, t, ws, xs)
                    elif quant == "fp8":
                        lin = L.prequantized(L.Fp8Linear, t, ws)
                    else:  # dequantized, then the scheme as on a bf16 checkpoint
                        w = (t.to(device).float() * ws.to(device)).to(torch.bfloat16)
                        if quant == "bf16":
                            lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta")
                            lin.weight = nn.Parameter(w, requires_grad=False)
                        elif quant == "fp8row":
                            lin = L.Fp8RowLinear(w)
                        else:
                            lin = L.Int8Linear(w, quant == "w8a8")
                    setattr(mods[parent_name] if parent_name else model, child, lin)
                    n_lin += 1
                elif isinstance(mod, nn.Linear) and pname == "weight" and quant != "bf16":
                    parent_name, _, child = mod_name.rpartition(".")
                    w = t.to(device, non_blocking=True)
                    if quant == "fp8s":
                        if mod_name not in table["scales"]:
                            raise KeyError(f"scale table {table['sha']}: no scale for {mod_name}")
                        lin = L.Fp8StaticLinear(w, float(table["scales"][mod_name]))
                    elif quant == "fp8":
                        lin = L.Fp8Linear(w)
                    elif quant == "fp8row":
                        lin = L.Fp8RowLinear(w)
                    else:
                        lin = L.Int8Linear(w, quant == "w8a8")
                    setattr(mods[parent_name] if parent_name else model, child, lin)
                    n_lin += 1
                elif pname in mod._parameters:
                    mod._parameters[pname] = nn.Parameter(t.to(device), requires_grad=False)
                elif pname in mod._buffers:
                    mod._buffers[pname] = t.to(device)
                else:
                    raise KeyError(name)
        torch.cuda.synchronize()
        evict(shard)
    for m in model.modules():  # non-persistent embed_scale buffers (embed_tokens, embed_tokens_per_layer)
        if hasattr(m, "scalar_embed_scale"):
            m._buffers["embed_scale"] = torch.tensor(m.scalar_embed_scale, device=device)
    model._lm_weight = model.embed_tokens.weight
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.device.type == "meta":
            raise RuntimeError(f"still on meta: {n}")
    model._media = None
    if cfg.media and hf_cfg.model_type == "gemma4_unified":
        model._media = MediaEmbedder.load(hf_cfg, model_dir, device)
    elif cfg.media and getattr(hf_cfg, "vision_config", None) is not None:  # 31B: a vision encoder, no audio
        model._media = VisionTowerEmbedder.load(hf_cfg, model_dir, device)
    # images attend both ways on every layer (12B unified) or on the sliding layers only (31B): scorer._init_flex
    model._vision_bidir_all = hf_cfg.model_type == "gemma4_unified"
    model.eval()
    model._quant = quant
    model._scale_table = "checkpoint" if prequant and quant in ("fp8s", "fp8") else table["sha"] if table else None
    if prequant and quant in ("fp8s", "fp8") and n_lin != n_fp8:
        raise RuntimeError(f"fp8-static checkpoint: {n_fp8} fp8 linears in the shards, {n_lin} loaded")
    model._load_s = time.perf_counter() - t0
    model._n_quant_linears = n_lin
    torch.cuda.synchronize(device)
    return model, tcfg


class MediaEmbedder:
    """image and audio -> input embeddings of the text stack (docs/ARCHITECTURE.md, media). Kept in bf16, never quantized; the
    vectors replace the placeholder tokens' embeddings and are not multiplied by the text embed_scale."""

    def __init__(self, torch, audio_w, vision, ids: dict, eps: float, device):
        self.torch, self.audio_w, self.vision, self.ids, self.eps, self.device = torch, audio_w, vision, ids, eps, device

    @classmethod
    def load(cls, hf_cfg, model_dir: str, device) -> "MediaEmbedder":
        import torch
        from safetensors import safe_open
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

        vision = Gemma4UnifiedVisionEmbedder(hf_cfg.vision_config, hf_cfg.text_config)
        sd, audio_w = {}, None
        for shard in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
            with safe_open(shard, "pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    if k.startswith("model.vision_embedder."):  # renames as transformers conversion_mapping.py
                        sd[k[len("model.vision_embedder.") :]] = f.get_tensor(k)
                    elif k == "model.embed_vision.embedding_projection.weight":
                        sd["multimodal_embedder.embedding_projection.weight"] = f.get_tensor(k)
                    elif k == "model.embed_audio.embedding_projection.weight":
                        audio_w = f.get_tensor(k)
        vision.load_state_dict(sd, strict=True)
        ids = {
            "boi": hf_cfg.boi_token_id,
            "image": hf_cfg.image_token_id,
            "eoi": hf_cfg.eoi_token_id,
            "boa": hf_cfg.boa_token_id,
            "audio": hf_cfg.audio_token_id,
            "eoa": hf_cfg.eoa_token_index,
        }
        return cls(
            torch,
            audio_w.to(device, torch.bfloat16),
            vision.to(device, torch.bfloat16).eval(),
            ids,
            hf_cfg.audio_config.rms_norm_eps,
            device,
        )

    def image(self, img) -> "torch.Tensor":  # noqa: F821
        """media.Image -> [n, hidden] bf16"""
        torch = self.torch
        with torch.no_grad():
            out = self.vision(
                torch.from_numpy(img.patches)[None].to(self.device), torch.from_numpy(img.positions)[None].to(self.device), return_dict=True
            )
        return out.pooler_output[0].to(torch.bfloat16)

    def audio(self, aud) -> "torch.Tensor":  # noqa: F821
        """media.Audio -> [n, hidden] bf16: frames cast to bf16, RMSNorm without scale in fp32, projected"""
        torch = self.torch
        x = torch.from_numpy(aud.frames).to(self.device).to(torch.bfloat16).float()
        x = (x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)).to(torch.bfloat16)
        return x @ self.audio_w.t()


class VisionTowerEmbedder:
    """image -> input embeddings for the checkpoints with a vision encoder (the 31B, model_type gemma4; no audio): the 48x48
    blocks of media.Image re-cut into the processor's 16x16 patches (Gemma4ImageProcessor: raster order, HWC inside a patch,
    (x, y) positions), the 27-layer encoder with 3x3 pooling back to one vector per block (Gemma4VisionModel), then
    Gemma4MultimodalEmbedder (RMSNorm without scale + projection). The transformers modules as they are, in bf16, never quantized;
    one image per call (no padding patches)."""

    def __init__(self, torch, tower, proj, ids: dict, device):
        self.torch, self.tower, self.proj, self.ids, self.device = torch, tower, proj, ids, device

    @classmethod
    def load(cls, hf_cfg, model_dir: str, device) -> "VisionTowerEmbedder":
        import torch
        from safetensors import safe_open
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

        vcfg = hf_cfg.vision_config
        vcfg._attn_implementation = "sdpa"
        tower = Gemma4VisionModel(vcfg)
        proj = Gemma4MultimodalEmbedder(vcfg, hf_cfg.text_config)
        sd_t, sd_p = {}, {}
        for shard in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
            with safe_open(shard, "pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    if k.startswith("model.vision_tower."):
                        sd_t[k[len("model.vision_tower.") :]] = f.get_tensor(k)
                    elif k.startswith("model.embed_vision."):
                        sd_p[k[len("model.embed_vision.") :]] = f.get_tensor(k)
        tower.load_state_dict(sd_t, strict=True)
        proj.load_state_dict(sd_p, strict=True)
        ids = {"boi": hf_cfg.boi_token_id, "image": hf_cfg.image_token_id, "eoi": hf_cfg.eoi_token_id}
        ids.update(boa=None, audio=None, eoa=None)
        return cls(torch, tower.to(device, torch.bfloat16).eval(), proj.to(device, torch.bfloat16).eval(), ids, device)

    @staticmethod
    def patches(img) -> tuple["np.ndarray", "np.ndarray"]:  # noqa: F821
        """media.Image (48x48 blocks, raster) -> ([n*9, 16*16*3] patches, [n*9, 2] (x, y)) in Gemma4ImageProcessor's layout"""
        from spinf_serve.media import PATCH, SIDE

        nx, ny = int(img.positions[:, 0].max()) + 1, int(img.positions[:, 1].max()) + 1
        x = img.patches.reshape(ny, nx, SIDE, SIDE, 3).transpose(0, 2, 1, 3, 4).reshape(ny * SIDE, nx * SIDE, 3)  # [H, W, 3]
        py, px = ny * SIDE // PATCH, nx * SIDE // PATCH
        p = x.reshape(py, PATCH, px, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(py * px, PATCH * PATCH * 3)
        import numpy as np

        pos = np.stack(np.meshgrid(np.arange(px), np.arange(py), indexing="xy"), -1).reshape(-1, 2)
        return np.ascontiguousarray(p), pos.astype(np.int64)

    def image(self, img) -> "torch.Tensor":  # noqa: F821
        """media.Image -> [n, hidden] bf16"""
        torch = self.torch
        p, pos = self.patches(img)
        with torch.no_grad():
            h = self.tower(torch.from_numpy(p)[None].to(self.device), torch.from_numpy(pos)[None].to(self.device)).last_hidden_state
            return self.proj(h).to(torch.bfloat16)

    def audio(self, aud):
        raise RuntimeError("this model has no audio input")


class Gemma4Stack(Stack):
    """the Gemma 4 family: transformers' text model with the tree_flex attention and the K/V capture of CAPTURE for extraction"""

    family = "gemma4"
    image_scheme = "gemma"

    @staticmethod
    def detect(model_dir: str) -> bool:
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                return json.load(f).get("model_type") in ("gemma4", "gemma4_unified")
        except (OSError, ValueError):
            return False

    def build(self, cfg: EngineConfig, model_dir: str, device):
        return build_model(cfg, model_dir, device)

    def forward(self, sc, b, inputs: dict, masks: dict):
        torch = sc.torch
        with torch.no_grad():
            if b.media:  # text embeddings, then the media rows over their placeholders (not scaled by embed_scale)
                x = sc.model.embed_tokens(inputs["ids"])
                for start, obj in b.media:
                    x[0, start : start + obj.n_tokens] = sc._media_embeds(obj)
                return sc.model(inputs_embeds=x, position_ids=inputs["pos"], use_cache=False, attention_mask=masks).last_hidden_state
            return sc.model(input_ids=inputs["ids"], position_ids=inputs["pos"], use_cache=False, attention_mask=masks).last_hidden_state

    @contextmanager
    def capture_ranges(self, sc, ranges: list):
        """the K/V of the stream ranges [(start, n, key)] kept by the forward inside -> the dict the flex hook fills under "kv"
        ({key: {layer: (k, v)}}; sliding layers their last window - 1 tokens only)"""
        cap = {"window": sc.tcfg.sliding_window, "ranges": ranges, "kv": {}}
        CAPTURE[sc.device] = cap
        try:
            yield cap
        finally:
            CAPTURE.pop(sc.device, None)

    @contextmanager
    def capture(self, sc, b, xo: list):
        rs = [(o.start, o.prefix_len, ("p", id(o))) for o in xo] + [(bs, bl, ("t", pr.key)) for o in xo for bs, bl, pr in o.xbranches]
        with self.capture_ranges(sc, rs) as cap:
            yield {"kv": cap["kv"], "extra": None, "pos0": None}

    def abandon(self, sc) -> None:
        CAPTURE.pop(sc.device, None)
