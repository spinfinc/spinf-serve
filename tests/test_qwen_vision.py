"""qwen_vision on the CPU: the processor's resize rule and patch layout, the token budgets, the 3-D (mrope) positions over a packed
batch and the frequency-to-axis table"""

import base64
import io

import numpy as np
import pytest

from spinf_serve import media
from spinf_serve.models.qwen35 import qwen_vision as qv


def png_url(w, h, color=(120, 30, 200)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_smart_resize_is_the_processors():
    assert qv.smart_resize(480, 640, 1024 * 1024) == (480, 640)  # within the budget: rounded to 32 only
    assert qv.smart_resize(1500, 2000, 1024 * 1024) == (864, 1152)  # over it: floor, aspect kept (27 x 36 = 972 tokens)
    assert qv.smart_resize(100, 150, 1024 * 1024) == (224, 320)  # under the 64-token floor: ceil up (7 x 10 = 70 tokens)
    with pytest.raises(media.MediaError):
        qv.smart_resize(10, 3000, 1024 * 1024)


def test_patchify_layout():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 64, 96)).astype(np.float32)  # grid 4 x 6 patches, 2 x 3 tokens
    p = qv.patchify(x)
    assert p.shape == (24, qv.PATCH_DIM)
    gw = 6
    for br in range(2):
        for bc in range(3):
            for ir in range(2):
                for ic in range(2):
                    k = (br * (gw // 2) + bc) * 4 + ir * 2 + ic
                    v = p[k].reshape(3, qv.TEMPORAL, qv.PATCH, qv.PATCH)
                    want = x[:, (br * 2 + ir) * 16 : (br * 2 + ir + 1) * 16, (bc * 2 + ic) * 16 : (bc * 2 + ic + 1) * 16]
                    assert np.array_equal(v[:, 0], want) and np.array_equal(v[:, 1], want)


def test_decode_image_tokens_and_values():
    img = qv.decode_image(png_url(640, 480))
    assert isinstance(img, qv.QwenImage) and img.grid == (1, 30, 40) and img.n_tokens == 300
    assert img.patches.shape == (1200, qv.PATCH_DIM) and img.positions.shape == (300, 2)
    assert img.positions[21].tolist() == [1, 1] and img.positions[299].tolist() == [19, 14]  # (x, y) raster over 15 x 20
    want = (np.array([120, 30, 200], np.float32) / 255 - 0.5) / 0.5
    v = img.patches.reshape(1200, 3, -1)
    assert np.allclose(v, want[None, :, None], atol=1e-6)
    for detail, budget in qv.DETAIL_TOKENS.items():
        big = qv.decode_image(png_url(4000, 3000), detail)
        assert budget * 0.9 < big.n_tokens <= budget
    assert qv.decode_image(png_url(40, 30)).n_tokens >= 64  # the processor's floor
    with pytest.raises(media.MediaError):
        qv.decode_image(png_url(64, 64), "huge")


def test_media_scheme_dispatch():
    media.SCHEME["image"] = "qwen"
    try:
        assert isinstance(media.decode_image(png_url(64, 64)), qv.QwenImage)
    finally:
        media.SCHEME["image"] = "gemma"
    assert not isinstance(media.decode_image(png_url(64, 64)), qv.QwenImage)


def _image(gh, gw):
    hl, wl = gh // 2, gw // 2
    pos = np.stack(np.meshgrid(np.arange(wl), np.arange(hl), indexing="xy"), -1).reshape(-1, 2)
    return qv.QwenImage(np.zeros((gh * gw, qv.PATCH_DIM), np.float32), pos.astype(np.int64), grid=(1, gh, gw))


def test_mrope_positions_follow_get_rope_index():
    # job 0: t0 boi [6 image tokens, 2 x 3] eoi t1 t2 | branch b0 b1; job 1 (text only) packed after it, with a pad between
    img = _image(4, 6)
    pos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 1, 2], np.int32)
    art = np.array([0] * 13 + [-1] + [1] * 3, np.int32)
    p3 = qv.mrope_positions(pos, art, [(2, img)])
    assert p3[:, :2].tolist() == [[0, 1]] * 3
    assert p3[0, 2:8].tolist() == [2] * 6
    assert p3[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3] and p3[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]
    assert p3[:, 8:13].tolist() == [[5, 6, 7, 8, 9]] * 3  # after the image: start + max(2, 3) = 5, then +1 per token
    assert p3[:, 13:].tolist() == [[0, 0, 1, 2]] * 3  # the pad and the other job untouched
    # two images: the second starts where the first's shrink put it
    img2 = _image(2, 2)
    pos = np.arange(14, dtype=np.int32)
    art = np.zeros(14, np.int32)
    p3 = qv.mrope_positions(pos, art, [(2, img), (10, img2)])  # t boi img(6) eoi boi img2(1) eoi t t
    assert p3[:, 8:10].tolist() == [[5, 6]] * 3 and p3[:, 10].tolist() == [7, 7, 7] and p3[:, 11:].tolist() == [[8, 9, 10]] * 3


def test_mrope_axis_tables():
    inter = qv.mrope_axis({"mrope_interleaved": True, "mrope_section": [11, 11, 10]}, 32)
    assert [int((inter == a).sum()) for a in range(3)] == [11, 11, 10]
    assert inter[:9].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2] and inter[29:].tolist() == [2, 0, 1]
    chunk = qv.mrope_axis({"mrope_section": [16, 24, 24]}, 64)
    assert chunk[:16].tolist() == [0] * 16 and chunk[16:40].tolist() == [1] * 24 and chunk[40:].tolist() == [2] * 24
    assert qv.mrope_axis({}, 32).tolist() == [0] * 32
    with pytest.raises(ValueError):
        qv.mrope_axis({"mrope_section": [1, 2, 3]}, 32)
