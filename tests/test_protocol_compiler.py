import math
import re

import numpy as np
import pytest

from spinf_serve import compiler
from spinf_serve.packer import Packer, ReadPlan
from spinf_serve.protocol import Rejected, ScoreRequest, Template, check_options, combinations, content_text
from spinf_serve.scorer import Scorer, read_grouped


class WordTok:
    """one token per newline and per word with its leading space; BOS = 1"""

    PAT = re.compile(r"\n| ?[^ \n]+| +")

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(w, len(self.vocab) + 2) for w in self.PAT.findall(text)]

    def __call__(self, text, add_special_tokens=True, split_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self(t, add_special_tokens)["input_ids"] for t in text]}
        return {"input_ids": ([1] if add_special_tokens else []) + self.ids(text)}


def req(**kw):
    base = {
        "model": "spinf/gemma-4-12b-fp8",
        "inputs": [
            {"id": "a1", "messages": [{"role": "user", "content": "Apple sold more phones."}]},
            {
                "id": "a2",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Mac margins fell."}]}],
                "queries": [{"id": "own", "template": "\nIs it good?{?}", "options": [" yes", " no"]}],
            },
        ],
        "scoring": {
            "queries": [
                {
                    "id": "dir",
                    "template": "\nWill {company} {metric} go{?}",
                    "options": [" up", " down", " sideways today"],
                    "combinations": [{"company": "Apple", "metric": "revenue"}, {"company": "Apple", "metric": "margin"}],
                }
            ]
        },
    }
    base.update(kw)
    return ScoreRequest.model_validate(base)


# ---------------------------------------------------------------- protocol
def test_template_parsing():
    t = Template("\nWill {company}'s {metric} {{really}} go{?}", "p")
    assert t.names == ["company", "metric"]
    assert t.fill({"company": "Apple", "metric": "revenue"}) == "\nWill Apple's revenue {really} go"
    for bad in ("no slot", "{?} then text", "{a}{b}{c}{d}{e}{f}{?}", "bad {name!}{?}", "stray } brace{?}"):
        with pytest.raises(Rejected) as e:
            Template(bad, "p")
        assert e.value.code == "invalid_template"


def test_combinations_and_options():
    r = req().scoring.queries[0]
    t = Template(r.template, "p")
    assert len(combinations(r, t, "p")) == 2
    all_ = r.model_copy(update={"combinations": "all", "values": {"company": ["A", "B"], "metric": ["x", "y", "z"]}})
    assert len(combinations(all_, t, "p")) == 6
    for upd, code in (
        ({"combinations": [{"company": "A"}]}, "missing_value"),
        ({"combinations": [{"company": "A", "metric": "x", "extra": "e"}]}, "unknown_placeholder"),
        ({"combinations": "all", "values": {"company": ["a"] * 40, "metric": ["b"] * 30}}, "too_many_combinations"),
        ({"combinations": None}, "missing_value"),
    ):
        with pytest.raises(Rejected) as e:
            combinations(r.model_copy(update=upd), t, "p")
        assert e.value.code == code
    with pytest.raises(Rejected) as e:
        check_options(r.model_copy(update={"options": [" up", "Up"]}), True, "p")
    assert e.value.code == "duplicate_option"
    check_options(r.model_copy(update={"options": [" up", "Up"]}), False, "p")


def test_content_text_and_unsupported_parts():
    inp = req().inputs[1]
    assert content_text(inp, "inputs.1") == "Mac margins fell."
    img = inp.model_copy(update={"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    img = ScoreRequest.model_validate({**req().model_dump(), "inputs": [img.model_dump()]}).inputs[0]
    with pytest.raises(Rejected) as e:
        content_text(img, "inputs.0")
    assert e.value.code == "unsupported_content"


# ---------------------------------------------------------------- read plans
def test_read_plan_rows_and_entries():
    plan = ReadPlan.build([[5], [6, 7], [6, 8], [6, 7, 9]])
    # lead-ins (6,) and (6, 7); rows: 0 base, 1 for (6,), 2-3 for (6, 7)
    assert plan.leadins == [(6,), (6, 7)] and plan.n_rows == 4
    got = sorted(zip(plan.entry_opt.tolist(), plan.entry_row.tolist(), plan.entry_tok.tolist(), strict=True))
    assert got == [(0, 0, 5), (1, 0, 6), (1, 1, 7), (2, 0, 6), (2, 1, 8), (3, 0, 6), (3, 2, 7), (3, 3, 9)]


# ---------------------------------------------------------------- compiler
def test_compile_usage_layout_and_floor_dedup():
    tok = WordTok()
    r = req(inputs=req().inputs + [req().inputs[0].model_copy(update={"id": "a3"})])
    c = compiler.compile_request(r, tok, 16384)
    # a1 and a3 take the shared query, a2 the shared one and its own
    assert [[s.q.id for s, _, _ in e] for e in c.layout] == [["dir"], ["dir", "own"], ["dir"]]
    content = sum(len(tok.ids(content_text(i, ""))) for i in r.inputs)
    filled = ["\nWill Apple revenue go", "\nWill Apple margin go"]
    dir_prompt = sum(len(tok.ids(f)) for f in filled) + 2 * 1  # " sideways today" = 2 tokens: 1 extra per prompt
    own_prompt = len(tok.ids("\nIs it good?"))
    assert c.usage == {
        "content_tokens": content,
        "text_tokens": content,
        "query_tokens": 3 * dir_prompt + own_prompt,
        "floor_tokens": dir_prompt + own_prompt,
        **dict.fromkeys(compiler.MEDIA_USAGE, 0),  # text only
    }  # the shared query's floors once for three inputs
    assert len(c.floor_index) == 3
    keys = {k for k, _, _ in c.items}
    assert keys == {("in", 0), ("in", 1), ("in", 2), ("floor",)}


def test_compile_rejections():
    tok = WordTok()
    cases = [
        (req(inputs=[{**req().inputs[0].model_dump(), "query_ids": ["nope"]}]), "unknown_query_id"),
        (
            req(
                scoring={
                    **req().scoring.model_dump(),
                    "queries": [{**req().scoring.queries[0].model_dump(), "options": [" one two three four five six"]}],
                }
            ),
            "option_too_long",
        ),
        (
            req(
                scoring={
                    **req().scoring.model_dump(),
                    "queries": [
                        {**req().scoring.queries[0].model_dump(), "template": "\nWill {company} {metric} go{?}", "options": ["up"]}
                    ],
                }
            ),
            "option_boundary",
        ),
        (req(limits={"max_content_tokens": 2}), "content_too_long"),
        (req(limits={"max_query_tokens": 3}), "template_too_long"),
        (req(limits={"max_processed_tokens": 10}), "request_too_large"),
        (req(inputs=[req().inputs[0].model_dump(), req().inputs[0].model_dump()]), "duplicate_id"),
    ]
    for r, code in cases:
        with pytest.raises(Rejected) as e:
            compiler.compile_request(r, tok, 16384)
        assert e.value.code == code, (code, e.value.message)


class NoBosTok(WordTok):
    """a tokenizer without a beginning-of-sequence token (Qwen)"""

    def __call__(self, text, add_special_tokens=True, split_special_tokens=False):
        return super().__call__(text, False, split_special_tokens)


def test_empty_floor_needs_text_without_bos():
    bare = req(scoring={"queries": [{"id": "dir", "template": "{?}", "options": [" up", " down"]}]})
    compiler.compile_request(bare, WordTok(), 16384)  # with <bos> the floor read is "<bos> option": fine
    with pytest.raises(Rejected) as e:
        compiler.compile_request(bare, NoBosTok(), 16384)
    assert e.value.code == "invalid_field" and e.value.param == "scoring.queries.0.template"
    off = req(scoring={"queries": [{"id": "dir", "template": "{?}", "options": [" up", " down"], "empty_floor": False}]})
    compiler.compile_request(off, NoBosTok(), 16384)
    texted = req(scoring={"queries": [{"id": "dir", "template": "\nGo{?}", "options": [" up", " down"]}]})
    compiler.compile_request(texted, NoBosTok(), 16384)


def test_junction_that_merges_is_retokenized_in_one_tree():
    tok = WordTok()
    # content ends without a separator and the template starts with a word: "done" + "Will" -> "doneWill" (one token)
    r = req(
        inputs=[{"id": "x", "messages": [{"role": "user", "content": "done"}]}],
        scoring={"queries": [{"id": "q", "template": "Will it{?}", "options": [" yes", " no"]}]},
    )
    c = compiler.compile_request(r, tok, 16384)
    ((spec, key, first),) = c.layout[0]
    assert key == ("in", 0) and first == 0
    tree = next(t for k, _, t in c.items if k == ("in", 0))
    # the content's tail is re-tokenized with the template: "doneWill" is one token, the content is read once
    assert tree.prefix + tree.branches[0] == [1] + tok.ids("doneWill it")
    # a long content: only its last characters are re-tokenized, the rest is the content's own tokens
    long = " ".join(f"w{k}" for k in range(40)) + " done"
    r = req(
        inputs=[{"id": "x", "messages": [{"role": "user", "content": long}]}],
        scoring={
            "queries": [
                {"id": "q", "template": "Will it{?}", "options": [" yes", " no"]},
                {"id": "p", "template": "Will they{?}", "options": [" yes", " no"]},
            ]
        },
    )
    c = compiler.compile_request(r, tok, 16384)
    trees = [t for k, _, t in c.items if k == ("in", 0)]
    assert len(trees) == 1 and len(trees[0].branches) >= 2
    # the content up to the cut is the prefix; the re-tokenized tail with the templates' common start is the trie's first node
    run = trees[0].prefix + trees[0].branches[0]
    assert run[: len(tok.ids(long)) - 8] == ([1] + tok.ids(long))[: len(tok.ids(long)) - 8]


def fake_lp(prev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """a deterministic 'model': log P(target | previous token)"""
    return -((prev * 31 + target * 17) % 97) / 10.0 - 0.01


def fake_score_grouped(items, budget=16384):
    """the real packer and read_grouped, with fake_lp in place of the GPU forward"""
    out = {}
    for key, first, tree in items:
        slots = out.setdefault(key, [])
        slots.extend([None] * max(0, first + len(tree.plans) - len(slots)))
    packer = Packer(budget)
    batches = [b for key, first, tree in items for b in packer.push((key, first), tree)] + list(packer.flush())
    for b in batches:
        cols = Scorer._cols(b)
        prev = b.ids[b.sel]
        lp = fake_lp(prev[:, None], np.asarray(cols)[None, :])
        read_grouped(lp, cols, b, out)
    return out


def test_end_to_end_read_geometry_and_assembly():
    tok = WordTok()
    r = req()
    c = compiler.compile_request(r, tok, 256)  # a small budget: jobs split over several trees
    scores = fake_score_grouped(c.items, 256)
    res = compiler.assemble(c, r, scores)
    # expected: for each prompt and option variant, sum of fake_lp(previous token, token) along the option
    for i, entries in enumerate(c.layout):
        content_ids = tok(content_text(r.inputs[i], ""), True)["input_ids"]
        for s, key, first in entries:
            for k in range(len(s.combos)):
                prompt = content_ids + s.filled_ids[k]
                exp_var = []
                for v in s.variants:
                    ids, prev, tot = tok.ids(v), prompt[-1], 0.0
                    for t in ids:
                        tot += float(fake_lp(np.array(prev), np.array(t)))
                        prev = t
                    exp_var.append(tot)
                np.testing.assert_allclose(scores[key][first + k], exp_var, rtol=0, atol=1e-6)
                exp_opt = np.logaddexp.reduceat(np.array(exp_var), s.starts)
                got = res["results"][i]["queries"][[e[0] for e in entries].index(s)]["combinations"][k]
                assert [o["text"] for o in got["options"]] == s.q.options
                np.testing.assert_allclose([o["logprob"] for o in got["options"]], exp_opt, atol=1e-5)
                p = np.exp(exp_opt - exp_opt.max())
                np.testing.assert_allclose([o["p"] for o in got["options"]], p / p.sum(), atol=1e-5)
                assert math.isclose(got["residual"], max(0.0, 1 - float(np.exp(exp_opt).sum())), abs_tol=1e-5)
                assert got["floor_residual"] is not None and all(o["floor_p"] is not None for o in got["options"])


def test_case_variants_are_read_and_summed():
    tok = WordTok()
    r = req(
        inputs=[req().inputs[0].model_dump()],
        scoring={"queries": [{"id": "q", "template": "\nIs it good?{?}", "options": [" yes", " no"]}]},
    )
    c = compiler.compile_request(r, tok, 16384)
    s = c.layout[0][0][0]
    # the no-space variants merge with "?" in this tokenizer ("good?yes"), so the boundary check drops them
    assert s.variants[: s.starts[1]] == [" yes", " Yes", " YES"]
    r_off = r.model_copy(update={"scoring": r.scoring.model_copy(update={"case_insensitive": False})})
    s_off = compiler.compile_request(r_off, tok, 16384).layout[0][0][0]
    assert s_off.variants == [" yes", " no"]


def group_layout(c, tok_ids, group_first_branch=None):
    """the input's job packed alone (aligned): (ids, pos, end relative to the region start, block tables of its q blocks
    relative to the region's first block) for the region from `group_first_branch` (a branch index) to the job's end, or
    from the prefix's end to the first aligned branch when None"""
    from spinf_serve.packer import BLOCK, block_tables

    (tree,) = [t for k, _, t in c.items if k == ("in", 0)]
    p = Packer(16384, align=True)
    list(p.push("x", tree))
    (b,) = list(p.flush())
    pad_start = [k for k in range(b.length) if b.art[k] == -1]
    starts = sorted({int(k) for k in np.flatnonzero(b.seg > 0)})
    if group_first_branch is None:
        lo = len(tree.prefix)
        hi = b.length if not tree.aligned else min(k for k in starts if b.seg[k] == tree.aligned[0] + 1)
    else:
        lo = min(k for k in starts if b.seg[k] == group_first_branch + 1)
        hi = max(k for k in range(b.length) if b.art[k] == 0) + 1
    tabs = block_tables(b.art, b.end, b.pos, None, always_window=True)[0]
    pn, pi, fn, fi = tabs
    qb = range(lo // BLOCK, (hi - 1) // BLOCK + 1)
    rel = [(sorted(int(j) for j in pi[i, : pn[i]]), sorted(int(j) for j in fi[i, : fn[i]])) for i in qb]
    region = (b.ids[lo:hi].tolist(), b.pos[lo:hi].tolist(), [int(e) - lo if e >= lo else int(e) for e in b.end[lo:hi]])
    return lo % BLOCK, region, [(len(rel), [len(x) for x, _ in rel])], pad_start


def test_query_group_layout_does_not_depend_on_other_queries():
    tok = WordTok()
    text = "Oil prices rose after the cartel cut output again this week " * 6
    caller = [{"id": "c", "template": "\nIs it about energy?{?}", "options": [" yes", " no"]}]
    pack = [
        {
            "id": "p1",
            "template": "\nQuestion: Does it mention {F}?\nAnswer:{?}",
            "options": [" Yes", " No"],
            "combinations": [{"F": f} for f in ("oil", "a cartel", "prices", "the weather")],
            "group": "pk@1",
        },
        {"id": "p2", "template": "\nThe tone is{?}", "options": [" positive", " negative", " neutral"], "group": "pk@1"},
    ]

    def compile_with(queries):
        r = ScoreRequest.model_validate(
            {
                "model": "spinf/gemma-4-12b-fp8",
                "inputs": [{"id": "x", "messages": [{"role": "user", "content": text}]}],
                "scoring": {"queries": queries, "empty_floor": False},
            }
        )
        return compiler.compile_request(r, tok, 16384)

    both, pack_only, caller_only = compile_with(caller + pack), compile_with(pack), compile_with(caller)
    tb = next(t for k, _, t in both.items if k == ("in", 0))
    tp = next(t for k, _, t in pack_only.items if k == ("in", 0))
    assert tb.aligned and tp.aligned
    # the pack group: same tokens, positions, subtree ends and attention tables relative to its block, with or without the caller
    off_b, region_b, tabs_b, _ = group_layout(both, tok, tb.aligned[0])
    off_p, region_p, tabs_p, _ = group_layout(pack_only, tok, tp.aligned[0])
    assert off_b == off_p == 0 and region_b == region_p and tabs_b == tabs_p
    # the caller's queries: laid out as without the pack (the pack starts after them, on a new block)
    _, region_c, _, _ = group_layout(caller_only, tok)
    _, region_bc, _, _ = group_layout(both, tok)
    assert region_c[0] == region_bc[0][: len(region_c[0])] and region_c[1] == region_bc[1][: len(region_c[1])]
    # and the output keeps the request's query order
    assert [s.q.id for s, _, _ in both.layout[0]] == ["c", "p1", "p2"]


def test_junction_without_a_clean_cut_reads_the_content_once():
    """a long trailing run with no token boundary near its end (one 400-character word): no cut re-tokenizes to the content's own
    tokens, so the templates join at the token boundary (a warning) instead of each branch re-sending the whole content"""
    tok = WordTok()
    long = " ".join(f"w{k}" for k in range(300)) + " " + "x" * 400
    r = req(
        inputs=[{"id": "x", "messages": [{"role": "user", "content": long}]}],
        scoring={
            "queries": [
                {"id": "q", "template": "Will it{?}", "options": [" yes", " no"]},
                {"id": "s", "template": 'The item is "{*}"', "max_tokens": 4},
                {"id": "t", "template": 'The place is "{*}"', "max_tokens": 4},
            ]
        },
    )
    c = compiler.compile_request(r, tok, 16384)
    (tree,) = [t for k, _, t in c.items if k == ("in", 0)]
    assert tree.prefix == [1] + tok.ids(long)  # the content's own tokens, once
    assert len(tree.xbranches) == 2 and all(len(tree.branches[b]) < 8 for b, _ in tree.xbranches)
    assert c.processed_tokens < len(tree.prefix) + 120  # the branches, the floor job and the slots' decode allowance
    assert [w["code"] for w in c.warnings] == ["join_not_merged"]
    out = compiler.assemble(c, r, fake_score_grouped(c.items), {})
    assert [w["code"] for w in out["warnings"]] == ["join_not_merged"]


def test_extraction_needs_content_on_a_model_without_bos():
    r = req(
        inputs=[{"id": "x", "messages": [{"role": "user", "content": ""}]}],
        scoring={"queries": [{"id": "s", "template": 'The item is "{*}"', "max_tokens": 4}]},
    )
    with pytest.raises(Rejected) as e:
        compiler.compile_request(r, NoBosTok(), 16384)
    assert e.value.code == "unsupported_content"


def test_max_tokens_cap_is_64_searched_and_100_generated():
    from spinf_serve.protocol import MAX_EXTRACT_TOKENS, MAX_GENERATE_TOKENS, Query

    assert (MAX_EXTRACT_TOKENS, MAX_GENERATE_TOKENS) == (64, 100)
    Query(id="a", template='\nA: "{*}"', max_tokens=64)
    Query(id="b", template='\nA: "{*}"', max_tokens=100, generate=True)
    Query(id="c", template='\nA: "{*}"', max_tokens=100, generate={"temperature": 0.7})
    with pytest.raises(ValueError, match="searched slot is at most 64"):
        Query(id="d", template='\nA: "{*}"', max_tokens=65)
    with pytest.raises(ValueError, match="searched slot is at most 64"):
        Query(id="e", template='\nA: "{*}"', max_tokens=65, generate=False)
    with pytest.raises(ValueError):
        Query(id="f", template='\nA: "{*}"', max_tokens=101, generate=True)
