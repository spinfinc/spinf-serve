import base64
import io
import shutil
import wave

import numpy as np
import pytest

from spinf_serve import media
from spinf_serve.compiler import _content
from spinf_serve.packer import BLOCK, GroupedTree, Packer, ReadPlan, block_tables
from spinf_serve.protocol import Input, Rejected, content_segments

IDS = {"boi": 900, "image": 901, "eoi": 902, "boa": 910, "audio": 911, "eoa": 912}


def png_url(w, h):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def wav_b64(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        t = np.arange(int(seconds * rate)) / rate
        w.writeframes((np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def test_image_tokens_follow_the_detail_budget():
    for detail, budget in media.DETAIL_TOKENS.items():
        img = media.decode_image(png_url(640, 480), detail)
        assert 0 < img.n_tokens <= budget
        assert img.patches.shape == (img.n_tokens, 48 * 48 * 3) and img.positions.shape == (img.n_tokens, 2)
    assert media.decode_image(png_url(320, 240)).n_tokens == 266  # docs/ARCHITECTURE.md: 266 for 320x240 at the default budget


@pytest.mark.parametrize("url", ["https://example.com/x.png", "data:image/png;base64,@@@", "data:image/bmp;base64,AAAA"])
def test_bad_images(url):
    with pytest.raises(media.MediaError):
        media.decode_image(url)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_audio_frames():
    a = media.decode_audio(wav_b64(2.0), "wav")
    assert a.n_tokens == 50 and a.frames.shape == (50, 640)  # 40 ms per token
    with pytest.raises(media.MediaError):
        media.decode_audio(base64.b64encode(b"not audio at all").decode(), "mp3")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_segments_images_first_audio_in_place():
    inp = Input.model_validate(
        {
            "id": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Listen to this clip."},
                        {"type": "input_audio", "input_audio": {"data": wav_b64(0.5), "format": "wav"}},
                        {"type": "text", "text": " And look:"},
                        {"type": "image_url", "image_url": {"url": png_url(96, 96), "detail": "low"}},
                    ],
                }
            ],
        }
    )
    segs = content_segments(inp, "inputs.0")
    assert isinstance(segs[0], media.Image)  # the image moved to the front
    assert segs[1] == "Listen to this clip." and isinstance(segs[2], media.Audio) and segs[3] == " And look:"
    bad = Input.model_validate(
        {"id": "x", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    )
    with pytest.raises(Rejected) as e:
        content_segments(bad, "inputs.0")
    assert e.value.code == "invalid_media" and e.value.param == "inputs.0.messages.0.content.0"


class Tok:
    def __call__(self, text, add_special_tokens=True, split_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self(t, add_special_tokens)["input_ids"] for t in text]}
        return {"input_ids": ([1] if add_special_tokens else []) + [2 + (ord(c) % 50) for c in text]}


def test_content_layout_markers_and_long_audio_split():
    img = media.Image(np.zeros((4, 6912), np.float32), np.zeros((4, 2), np.int64))
    aud = media.Audio(np.zeros((800, 640), np.float32))  # 32 s -> 750 + 50
    c = _content(Tok(), [img, "ab", aud], [1], IDS, "inputs.0")
    ids = c.ids
    assert ids[:7] == [1, 900, 901, 901, 901, 901, 902]
    assert [o for o, _ in c.media] == [2, 10, 10 + 750 + 2]
    assert [m.n_tokens for _, m in c.media] == [4, 750, 50]
    assert ids.count(911) == 800 and ids.count(910) == 2 and ids[-1] == 912 and c.tail is None
    with pytest.raises(Rejected) as e:
        _content(Tok(), [img], [1], None, "inputs.0")
    assert e.value.code == "unsupported_content"


def test_image_only_model_refuses_audio():
    ids = {**IDS, "boa": None, "audio": None, "eoa": None}  # the 31B: a vision encoder, no audio
    img = media.Image(np.zeros((4, 6912), np.float32), np.zeros((4, 2), np.int64))
    assert _content(Tok(), [img, "ab"], [1], ids, "inputs.0").ids[:3] == [1, 900, 901]
    with pytest.raises(Rejected) as e:
        _content(Tok(), [media.Audio(np.zeros((5, 640), np.float32))], [1], ids, "inputs.0")
    assert e.value.code == "unsupported_content" and "audio" in e.value.message


def test_tower_patches_follow_the_processor_layout():
    """48x48 blocks -> the 16x16 patches of Gemma4ImageProcessor (raster, HWC inside a patch) with (x, y) positions"""
    from spinf_serve.models.gemma4 import VisionTowerEmbedder

    img = media.decode_image(png_url(640, 480), "low")
    ny, nx = img.positions[:, 1].max() + 1, img.positions[:, 0].max() + 1
    full = img.patches.reshape(ny, nx, 48, 48, 3).transpose(0, 2, 1, 3, 4).reshape(ny * 48, nx * 48, 3)
    chw = full.transpose(2, 0, 1)
    ref = chw.reshape(3, ny * 3, 16, nx * 3, 16).transpose(1, 3, 2, 4, 0).reshape(-1, 768)  # convert_image_to_patches
    p, pos = VisionTowerEmbedder.patches(img)
    assert np.array_equal(p, ref) and len(pos) == 9 * img.n_tokens
    assert pos[1].tolist() == [1, 0] and pos[nx * 3].tolist() == [0, 1]


def dense(art, end, grp):
    n = len(art)
    q = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    m = (k <= q) & (art[q] >= 0) & (art[k] == art[q]) & (q < end[k])
    m |= (grp[q] == grp[k]) & (grp[q] >= 0)
    return m | (q == k)


def test_image_groups_in_the_stream_and_bidirectional_blocks():
    img = media.Image(np.zeros((300, 6912), np.float32), np.zeros((300, 2), np.int64))  # spans 3+ blocks
    prefix = [1, 900] + [901] * 300 + [902] + [5, 6]
    plan = ReadPlan.build([[7], [8]])
    t = GroupedTree(prefix, [[9], [10, 11]], [[(0, 0)], [(1, 1)]], [plan, plan], media=[(2, img)])
    packer = Packer(4096)
    other = GroupedTree([1, 5, 6, 7], [[9]], [[(0, 0)]], [plan])
    list(packer.push("other", other))
    list(packer.push("img", t))
    (b,) = list(packer.flush())
    ((start, obj),) = b.media
    assert obj is img and b.ids[start] == 901 and b.ids[start - 1] == 900
    assert (b.grp[start : start + 300] > 0).all() and (b.grp >= 0).sum() == 300
    (pn, pi, fn, fi), _ = block_tables(b.art, b.end, b.pos, None, b.grp)
    m = dense(b.art, b.end, b.grp)
    nb = b.length // BLOCK
    for i in range(nb):
        listed = set(pi[i, : pn[i]].tolist()) | set(fi[i, : fn[i]].tolist())
        for j in range(nb):
            if m[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK].any():
                assert j in listed, (i, j)  # every block pair that attends is visited, including later image blocks


@pytest.mark.parametrize("bidir_all", [True, False])
def test_sliding_tables_keep_image_blocks_past_the_window(bidir_all):
    img = media.Image(np.zeros((600, 6912), np.float32), np.zeros((600, 2), np.int64))
    prefix = [1, 900] + [901] * 600 + [902] + [5] * 400
    plan = ReadPlan.build([[7], [8]])
    t = GroupedTree(prefix, [[9], [10, 11]], [[(0, 0)], [(1, 1)]], [plan, plan], media=[(2, img)])
    packer = Packer(4096)
    list(packer.push("img", t))
    (b,) = list(packer.flush())
    window = 200
    _, sliding = block_tables(b.art, b.end, b.pos, window, b.grp)
    q = np.arange(b.length)[:, None]
    k = np.arange(b.length)[None, :]
    near = b.pos[q] - b.pos[k] < window
    tree = (k <= q) & (b.art[q] >= 0) & (b.art[k] == b.art[q]) & (q < b.end[k])
    same = (b.grp[q] == b.grp[k]) & (b.grp[q] >= 0)
    # as Scorer's sliding_img mask functions: the 12B's image pairs attend past the window, the 31B's within it
    m = (tree & near) | (same if bidir_all else same & near) | (q == k)
    pn, pi, fn, fi = sliding
    for i in range(b.length // BLOCK):
        listed = set(pi[i, : pn[i]].tolist()) | set(fi[i, : fn[i]].tolist())
        for j in range(b.length // BLOCK):
            blk = m[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK]
            if j in fi[i, : fn[i]]:
                assert blk.all(), (i, j)
            elif j not in listed:
                assert not blk.any(), (i, j)


def test_media_counts_for_billing():
    img = media.Image(np.zeros((4, 6912), np.float32), np.zeros((4, 2), np.int64))
    frame = media.Image(np.zeros((6, 6912), np.float32), np.zeros((6, 2), np.int64), inline=True)
    aud = media.Audio(np.zeros((800, 640), np.float32))  # 32 s: two segments
    c = _content(Tok(), [img, "ab", " 00:00 ", frame, " 00:01 ", frame, aud], [1], IDS, "inputs.0")
    assert c.mcount == {
        "images": 1,
        "image_tokens": 4 + 2,
        "video_frames": 2,
        "video_tokens": 2 * (6 + 2),
        "audio_ms": 32000,
        "audio_tokens": 800 + 4,
    }
