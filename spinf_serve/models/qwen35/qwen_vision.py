"""Images for the Qwen3.5-architecture stack (qwen.py): the vision tower and the 3-D rotary positions (mrope).

The checkpoint's vision model (transformers Qwen3_5VisionModel, 0.92 GB bf16, run as is): 16x16-pixel patches over two identical
"frames" (temporal patch 2), 27 transformer blocks at width 1152, then 2x2 patches merged into one token of the text width
(5120). One token per 32x32 pixel block: an image resized to h x w pixels (multiples of 32) is (h/32) x (w/32) tokens, in raster
order, between <|vision_start|> and <|vision_end|> (compiler._content; the chat template renders an image the same way).

Positions: the text stack's rope is 3-D (mrope_interleaved: the 32 rotary frequencies cycle through the axes t, h, w, sections
[11, 11, 10]). Text tokens have the same position on the three axes (the 1-D rope the text path computes); an image's tokens share
the t position of the image's start and take their row / column on h / w; the text after the image continues from start +
max(rows, cols), not start + tokens (transformers Qwen3_5Model.get_rope_index). Image tokens attend causally, as text (no
bidirectional block as Gemma's): the tree mask and the Gated DeltaNet plan are unchanged.

Preprocessing follows the checkpoint's Qwen2VLImageProcessor (PIL backend): smart_resize to multiples of 32 within a pixel
budget, bicubic, (x / 255 - 0.5) / 0.5, patches in spatial-merge-block order. The pixel budget per `detail` is DETAIL_TOKENS x
1024 pixels; the processor's own floor (64 tokens) stays. Only images; the checkpoint reads no audio.
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from spinf_serve.media import Image, MediaError, open_image

PATCH, MERGE, TEMPORAL = 16, 2, 2  # vision_config: patch_size, spatial_merge_size, temporal_patch_size
FACTOR = PATCH * MERGE  # 32: one token = a 32 x 32 pixel block
PATCH_DIM = 3 * TEMPORAL * PATCH * PATCH  # 1536 values per patch (channel, frame, row, column)
MIN_PIXELS = 65536  # preprocessor_config size.shortest_edge: at least 64 tokens
DETAIL_TOKENS = {"low": 256, "auto": 1024, "high": 4096}  # the token budget per detail (max_pixels = tokens x 32 x 32)
MAX_ASPECT = 200


@dataclass(eq=False)
class QwenImage(Image):
    """patches [grid_h x grid_w, PATCH_DIM] float32 normalised, in spatial-merge-block order (the 4 patches of a token together);
    positions [n_tokens, 2] (x, y) of each token on the merged grid, raster order; grid = (1, grid_h, grid_w) in patches"""

    grid: tuple = (1, 0, 0)

    @property
    def n_tokens(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2] // (MERGE * MERGE)


def smart_resize(height: int, width: int, max_pixels: int, min_pixels: int = MIN_PIXELS, factor: int = FACTOR) -> tuple[int, int]:
    """the resized (height, width): multiples of factor, within [min_pixels, max_pixels], the aspect ratio kept (transformers
    image_processing_qwen2_vl.smart_resize)"""
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise MediaError(f"the image's aspect ratio must be below {MAX_ASPECT}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(x: np.ndarray) -> np.ndarray:
    """a normalised image [3, H, W] (H, W multiples of 32) -> patches [grid_h x grid_w, PATCH_DIM] as the processor flattens them:
    (block row, block column, row in block, column in block) then (channel, frame, pixel row, pixel column)"""
    _, H, W = x.shape
    gh, gw = H // PATCH, W // PATCH
    p = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).transpose(1, 4, 2, 5, 0, 3, 6)
    p = np.broadcast_to(p[:, :, :, :, :, None], (*p.shape[:5], TEMPORAL, PATCH, PATCH))
    return np.ascontiguousarray(p.reshape(gh * gw, PATCH_DIM))


def decode_image(url: str, detail: str = "auto", max_pixels: int | None = None) -> QwenImage:
    """a data:image/...;base64 URL -> QwenImage (the checkpoint's image processor semantics, numpy / PIL only)"""
    from PIL import Image as PILImage

    budget = DETAIL_TOKENS.get(detail)
    if budget is None:
        raise MediaError(f"detail {detail!r}: use low, auto or high")
    im = open_image(url, *([max_pixels] if max_pixels else []))
    w, h = im.size
    th, tw = smart_resize(h, w, budget * FACTOR * FACTOR)
    if (th, tw) != (h, w):
        im = im.resize((tw, th), PILImage.BICUBIC)
    x = np.asarray(im, np.float32).transpose(2, 0, 1)  # [3, H, W]
    x = (x * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)  # rescale, then normalise with mean = std = 0.5
    gh, gw = th // PATCH, tw // PATCH
    hl, wl = gh // MERGE, gw // MERGE
    pos = np.stack(np.meshgrid(np.arange(wl), np.arange(hl), indexing="xy"), -1).reshape(-1, 2)
    return QwenImage(patchify(x), pos.astype(np.int64), grid=(1, gh, gw))


def mrope_positions(pos: np.ndarray, art: np.ndarray, media: list) -> np.ndarray:
    """the packed stream's 1-D path positions [n] and job ids [n] (packer.Batch), its media [(stream offset, QwenImage)] ->
    (t, h, w) positions [3, n] int64 as transformers' get_rope_index: an image's tokens at (start, start + row, start + column),
    the job's tokens after it (on every branch) moved back by tokens - max(rows, columns)"""
    p3 = np.tile(pos.astype(np.int64), (3, 1))
    for off, img in sorted(media, key=lambda m: m[0]):
        if not isinstance(img, QwenImage):
            raise TypeError(f"mrope_positions: {type(img).__name__} is not a QwenImage")
        n = img.n_tokens
        start = int(p3[0, off])
        later = (art == art[off]) & (p3[0] >= start + n)  # after the image on the job's paths (by position, not stream order)
        p3[0, off : off + n] = start
        p3[1, off : off + n] = start + img.positions[:, 1]
        p3[2, off : off + n] = start + img.positions[:, 0]
        p3[:, later] -= n - max(img.grid[1], img.grid[2]) // MERGE
    return p3


def mrope_axis(rope_parameters: dict, n_freq: int):
    """the axis (0 t, 1 h, 2 w) of each of the n_freq rotary frequencies: interleaved (mrope_interleaved, transformers
    apply_interleaved_mrope: t h w t h w ... then t for the frequencies past the h / w sections), or chunked (the sections
    back to back); all 0 without mrope_section (plain rope)"""
    sec = rope_parameters.get("mrope_section")
    axis = np.zeros(n_freq, np.int64)
    if not sec:
        return axis
    if rope_parameters.get("mrope_interleaved"):
        for a, length in ((1, sec[1]), (2, sec[2])):
            axis[a : 3 * length : 3] = a
    else:
        if sum(sec) != n_freq:
            raise ValueError(f"mrope_section {sec} does not cover {n_freq} frequencies")
        axis[:] = np.repeat(np.arange(3), sec)
    return axis


class QwenVision:
    """media.Image -> input embeddings of the text stack: the checkpoint's vision tower (transformers Qwen3_5VisionModel) in
    bf16. `ids`: the marker and placeholder tokens the compiler lays out (no audio: those entries are None)."""

    def __init__(self, torch, vision, ids: dict, device):
        self.torch, self.vision, self.ids, self.device = torch, vision, ids, device

    @classmethod
    def load(cls, hf_cfg, model_dir: str, device) -> "QwenVision":
        import torch
        from safetensors import safe_open
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        vc = hf_cfg.vision_config
        vc._attn_implementation = "sdpa"
        with torch.device("meta"):
            vision = Qwen3_5VisionModel(vc)
        prefix = "model.visual."
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            idx = json.load(f)["weight_map"]
        files: dict[str, list[str]] = {}
        for k in idx:
            if k.startswith(prefix):
                files.setdefault(idx[k], []).append(k)
        sd = {}
        for shard in sorted(files):
            with safe_open(os.path.join(model_dir, shard), "pt", device="cpu") as f:
                for k in files[shard]:
                    sd[k[len(prefix) :]] = f.get_tensor(k).to(device, torch.bfloat16)
        res = vision.load_state_dict(sd, strict=False, assign=True)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"qwen vision: missing {res.missing_keys[:3]} unexpected {res.unexpected_keys[:3]}")
        rot = vision.rotary_pos_emb  # its non-persistent buffer was created on meta: recompute it on the card
        ar = torch.arange(0, rot.dim, 2, dtype=torch.float32, device=device)
        rot.inv_freq = torch.nn.Buffer(1.0 / (rot.theta ** (ar / rot.dim)), persistent=False)
        for n, p in list(vision.named_parameters()) + list(vision.named_buffers()):
            if p.device.type == "meta":
                raise RuntimeError(f"qwen vision: still on meta: {n}")
        vision.eval()
        ids = {
            "boi": hf_cfg.vision_start_token_id,
            "image": hf_cfg.image_token_id,
            "eoi": hf_cfg.vision_end_token_id,
            "boa": None,
            "audio": None,
            "eoa": None,
        }
        return cls(torch, vision, ids, device)

    @property
    def bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.vision.parameters())

    def image(self, img: QwenImage) -> "torch.Tensor":  # noqa: F821
        """QwenImage -> [n_tokens, hidden] bf16 (the merged tokens, raster order)"""
        torch = self.torch
        with torch.no_grad():
            hs = torch.from_numpy(img.patches).to(self.device, torch.bfloat16)  # as get_image_features: cast to the tower's dtype
            grid = torch.tensor([img.grid], dtype=torch.int64, device=self.device)
            out = self.vision(hs, grid_thw=grid, return_dict=True)
        return out.pooler_output.to(torch.bfloat16)

    def audio(self, aud):
        raise MediaError("this model does not read audio")
