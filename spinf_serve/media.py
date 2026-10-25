"""Image and audio input: decoding and preprocessing on the CPU (docs/ARCHITECTURE.md, media).

The 12B checkpoint (Gemma 4 "unified") has no encoder towers. Media go inline in the token sequence between marker tokens, one
placeholder token per soft token; the placeholders' embeddings are replaced by the media embeddings (model.MediaEmbedder):
  image  <|image> + N x <|image|> + <image|>   N = one per 48x48 block after the resize rule; soft-token budget by detail:
                                                low 70, auto 280, high 1120; the N tokens attend to each other both ways
  audio  <|audio> + N x <|audio|> + <audio|>   N = one per 40 ms (640 samples at 16 kHz); at most 750 (30 s) per segment,
                                                longer audio is several segments
Only data: URLs for images (the engine never fetches a URL). Audio comes as base64 with its container format, decoded by ffmpeg.
The Qwen stack (qwen.py) reads images through its own preprocessing and vision tower (qwen_vision.py): the Scorer sets
SCHEME["image"] at load, and decode_image dispatches on it (one model per process).
"""

import base64
import binascii
import math
import re
import subprocess
from dataclasses import dataclass

import numpy as np

PATCH, POOL = 16, 3  # processor_config.json: patch_size, pooling_kernel_size; one soft token = a 48x48 block
SIDE = PATCH * POOL
DETAIL_TOKENS = {"low": 70, "auto": 280, "high": 1120}
SAMPLE_RATE = 16000
FRAME = 640  # samples per audio token (40 ms)
AUDIO_SEGMENT = 750  # audio tokens per <|audio> ... <audio|> segment (30 s)
AUDIO_FORMATS = {"wav", "mp3", "flac", "ogg", "m4a", "webm"}
IMAGE_TYPES = {"png", "jpeg", "jpg", "webp", "gif"}
MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard
MAX_AUDIO_SECONDS = 600
_DATA_URL = re.compile(r"data:image/([a-z]+);base64,(.*)", re.S)
SCHEME = {"image": "gemma"}  # the loaded model's image preprocessing: gemma (this module) | qwen (qwen_vision.decode_image)


class MediaError(ValueError):
    pass


@dataclass(eq=False)
class Image:
    patches: np.ndarray  # [n, 48*48*3] float32 in [0, 1], HWC order, blocks in raster order (qwen_vision.QwenImage differs)
    positions: np.ndarray  # [n, 2] (x, y)
    inline: bool = False  # stays where it is in the content (a video frame after its timestamp), not moved to the front

    @property
    def n_tokens(self) -> int:
        return len(self.patches)


@dataclass(eq=False)
class Audio:
    frames: np.ndarray  # [n, 640] float32 mono 16 kHz in [-1, 1]

    @property
    def n_tokens(self) -> int:
        return len(self.frames)


def _b64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("the data is not valid base64") from None


def open_image(url: str, max_pixels: int = MAX_IMAGE_PIXELS):
    """a data:image/...;base64 URL -> a PIL RGB image (the first frame of an animation)"""
    from io import BytesIO

    from PIL import Image as PILImage

    m = _DATA_URL.fullmatch(url.strip())
    if not m:
        raise MediaError("send the image as a data URL: data:image/<png|jpeg|webp|gif>;base64,<data>")
    if m.group(1) not in IMAGE_TYPES:
        raise MediaError(f"image type {m.group(1)!r} is not supported (png, jpeg, webp, gif)")
    PILImage.MAX_IMAGE_PIXELS = max_pixels
    try:
        im = PILImage.open(BytesIO(_b64(m.group(2))))
        im.seek(0)  # first frame of an animation
        return im.convert("RGB")
    except MediaError:
        raise
    except Exception as e:  # PIL raises many types for bad data
        raise MediaError(f"the image could not be decoded: {type(e).__name__}") from None


def decode_image(url: str, detail: str = "auto", max_pixels: int = MAX_IMAGE_PIXELS) -> Image:
    """a data:image/...;base64 URL -> the loaded model's image object (SCHEME): Gemma's patches and positions
    (Gemma4UnifiedImageProcessor semantics, numpy/PIL only) or qwen_vision.decode_image"""
    if SCHEME["image"] == "qwen":
        from spinf_serve.models.qwen35.qwen_vision import decode_image as qwen_decode

        return qwen_decode(url, detail, max_pixels)
    from PIL import Image as PILImage

    soft = DETAIL_TOKENS.get(detail)
    if soft is None:
        raise MediaError(f"detail {detail!r}: use low, auto or high")
    im = open_image(url, max_pixels)
    w, h = im.size
    f = math.sqrt(soft * POOL * POOL * PATCH * PATCH / (h * w))
    th = int(math.floor(f * h / SIDE)) * SIDE
    tw = int(math.floor(f * w / SIDE)) * SIDE
    if th == 0:
        th, tw = SIDE, min(int(w / h) * SIDE, soft // (POOL * POOL) * SIDE)
    if tw == 0:
        tw, th = SIDE, min(int(h / w) * SIDE, soft // (POOL * POOL) * SIDE)
    if (th, tw) != (h, w):
        im = im.resize((tw, th), PILImage.BICUBIC)
    x = np.asarray(im, np.float32) / 255.0  # [H, W, 3]
    ny, nx = th // SIDE, tw // SIDE
    p = x.reshape(ny, SIDE, nx, SIDE, 3).transpose(0, 2, 1, 3, 4).reshape(ny * nx, SIDE * SIDE * 3)
    pos = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy"), -1).reshape(-1, 2)
    return Image(np.ascontiguousarray(p), pos.astype(np.int64))


def decode_audio(data_b64: str, fmt: str, max_seconds: int = MAX_AUDIO_SECONDS) -> Audio:
    """base64 audio in a container format -> [n, 640] frames of mono 16 kHz float32 (ffmpeg), zero-padded to a whole frame"""
    if fmt not in AUDIO_FORMATS:
        raise MediaError(f"audio format {fmt!r} is not supported ({', '.join(sorted(AUDIO_FORMATS))})")
    raw = _b64(data_b64)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "mp4" if fmt == "m4a" else fmt,
        "-i",
        "pipe:0",
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
    except FileNotFoundError:
        raise MediaError("audio decoding is not available on this engine (no ffmpeg)") from None
    except subprocess.TimeoutExpired:
        raise MediaError("the audio took too long to decode") from None
    if r.returncode != 0 or not r.stdout:
        raise MediaError(f"the audio could not be decoded as {fmt}")
    wave = np.frombuffer(r.stdout, np.float32)
    wave = np.pad(wave, (0, (-len(wave)) % FRAME))
    return Audio(wave.reshape(-1, FRAME).copy())
