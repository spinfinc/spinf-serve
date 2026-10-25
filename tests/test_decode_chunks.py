"""Scorer._x_flush's chunking of the held decode groups under the decoder budget (decode.group_bytes): groups that fit decode
together, a group with too many slots is split into runs of its prompts sharing the captured prefix."""

import torch

from spinf_serve import decode
from spinf_serve import extract as ex
from spinf_serve.models.base import Stack
from spinf_serve.scorer import Scorer

MIB = 2**20


class Stub:
    """the Scorer fields _x_chunks / _x_split / _x_sub read, without a model"""

    stack = Stack()  # attention-only: no per-family decoder buffers
    model = object()  # no _decoder_budget_gib: the config's applies

    class cfg:
        decoder_budget_gib = 4.0

    _x_per_tok = {0: 1024}  # one K/V layer, 1 KiB per token

    _x_group_bytes = Scorer._x_group_bytes
    _x_chunk_bytes = Scorer._x_chunk_bytes
    _x_chunks = Scorer._x_chunks
    _x_split = Scorer._x_split
    _x_sub = Scorer._x_sub


def group(P: int, n_prompts: int, max_tokens: int, T: int = 8) -> decode.Group:
    kv = {0: (torch.zeros(1, 1, P, 1), torch.zeros(1, 1, P, 1))}
    prompts = [(ex.ExtractPrompt(("x", k), [1] * T, "text", '"', max_tokens, None), {}, T, torch.zeros(4)) for k in range(n_prompts)]
    return decode.Group(kv, P, prompts)


def test_groups_under_the_budget_decode_together():
    s = Stub()
    gs = [group(1000, 2, 16) for _ in range(3)]  # (1000 + 16 + 2 x 6 x 17) KiB each ~ 1.2 MiB
    s.cfg.decoder_budget_gib = 4 * MIB / 2**30
    chunks = list(s._x_chunks(list(gs)))
    assert [len(c) for c in chunks] == [3]
    s.cfg.decoder_budget_gib = 2.5 * MIB / 2**30
    chunks = list(s._x_chunks(list(gs)))
    assert [len(c) for c in chunks] == [2, 1]


def test_a_chunk_costs_what_the_padded_decoder_allocates():
    """the h-tech video request (2026-10-11): 5 searched slots (max_tokens 12-16) and 2 generated ones (64) came as two groups
    whose own sizes summed under the budget, but decoded together every slot takes 65 rows and both groups get 30 slots:
    the decoder allocated 4.6 GiB against 2.5 estimated and the card ran out of memory"""
    s = Stub()
    searched, generated = group(4000, 5, 16, T=42), group(4000, 2, 64, T=42)
    own = s._x_group_bytes(searched) + s._x_group_bytes(generated)
    both = s._x_chunk_bytes([searched, generated])
    # padded: G=2 x (prefix 4000 + templates 210 + Q 30 x S 65) KiB; alone each equals its group_bytes
    assert both == 2 * (4000 + 210 + 30 * 65) * 1024 and both > own
    assert s._x_chunk_bytes([searched]) == s._x_group_bytes(searched)
    assert s._x_chunk_bytes([generated]) == s._x_group_bytes(generated)
    s.cfg.decoder_budget_gib = (own + 1024) / 2**30  # the sum fits, the padded chunk does not
    chunks = list(s._x_chunks([searched, generated]))
    assert [[len(g.prompts) for g in c] for c in chunks] == [[5], [2]]


def test_hybrid_chunk_extra_bytes_match_the_decoder_buffers():
    """qwen_decode.chunk_extra_bytes against the Qwen3.8-27B shapes measured in-process (h-tech clip: R = 4,195 rows)"""
    from types import SimpleNamespace as NS

    from spinf_serve.models.qwen35 import qwen_decode

    la = NS(hk=16, dk=128, hv=48, dv=128, conv_dim=10240)
    model = NS(layers=[NS(kind="linear_attention", linear_attn=la)] * 48 + [NS(kind="full_attention")] * 16)
    searched, generated = group(4000, 5, 16, T=42), group(4000, 2, 64, T=42)
    n_g, G, n_slots, S = 48, 2, 60, 65
    R = 7 * 42 + n_slots * S  # 294 template rows (the stub's templates are 42 each; the clip's were 295) + 3,900 generated
    assert R == 4194
    want = n_g * G * 48 * 128 * 128 * 4  # h0
    want += R * (n_g * ((2 * 16 * 128 + 48 * 128) * 2 + 48 * 6) + 48 * 128 * 2)  # cq ck cv cg cb per layer, o_buf
    want += n_g * n_slots * 4 * 10240 * 2  # ring + pre_new
    assert qwen_decode.chunk_extra_bytes(model, [searched, generated]) == want
    assert want > 4 * 2**30  # the measured 4.6 GiB chunk (with the K/V) against the 2.5 GiB budget


def test_a_group_over_the_budget_is_split_into_runs_of_prompts():
    s = Stub()
    g = group(1000, 10, 64)  # 10 prompts x 6 x 65 generated rows = 3,900 KiB + prefix 1,000 KiB
    s.cfg.decoder_budget_gib = 2 * MIB / 2**30  # ~2 prompts per run next to the prefix
    chunks = list(s._x_chunks([g]))
    runs = [r for c in chunks for r in c]
    assert [len(r.prompts) for r in runs] == [2, 2, 2, 2, 2]
    assert [p[0].key for r in runs for p in r.prompts] == [("x", k) for k in range(10)]
    # every run reads the same captured prefix; the original group let go of it
    assert all(r.prefix_kv[0][0] is runs[0].prefix_kv[0][0] for r in runs) and g.prefix_kv == {} and g.prompts == []
    assert all(s._x_group_bytes(r) <= 2 * MIB for r in runs)
    # one prompt alone over the budget still decodes (alone)
    s.cfg.decoder_budget_gib = 0.5 * MIB / 2**30
    runs = [r for c in s._x_chunks([group(1000, 3, 64)]) for r in c]
    assert [len(r.prompts) for r in runs] == [1, 1, 1]


def test_hybrid_extras_are_sliced_per_run(monkeypatch):
    from spinf_serve.models.qwen35 import qwen_decode

    monkeypatch.setattr(qwen_decode, "group_extra_bytes", lambda model, g, S: 0)  # needs the model's recurrent layers
    s = Stub()
    g = group(100, 4, 8)
    g.extra = {"state": {0: torch.zeros(2)}, "tmpl": [f"t{k}" for k in range(4)], "pos0": [10, 20, 30, 40]}
    sub = s._x_sub(g, 1, 3)
    assert [p[0].key for p in sub.prompts] == [("x", 1), ("x", 2)]
    assert sub.extra["tmpl"] == ["t1", "t2"] and sub.extra["pos0"] == [20, 30]
    assert sub.extra["state"] is not g.extra["state"] and sub.extra["state"][0] is g.extra["state"][0]
