"""Extraction slots ({*} {n} {$}), video parts, long contents: the CPU side (protocol, compiler, grammar, merging)."""

import math

import numpy as np
import pytest
from test_media import png_url
from test_protocol_compiler import WordTok

from spinf_serve import compiler, media
from spinf_serve import extract as ex
from spinf_serve.packer import BLOCK, GroupedTree, Packer
from spinf_serve.protocol import Calibration, Input, Rejected, ScoreRequest, Template, content_segments, timestamp


# ---------------------------------------------------------------- templates
def test_extraction_slots_in_templates():
    t = Template('\nThe brand featured is "{*}"', "p")
    assert (t.kind, t.stop, t.fill({})) == ("text", '"', '\nThe brand featured is "')
    t = Template("\nQ: How many {what} were sold?\nA:{n}", "p")
    assert (t.kind, t.stop, t.names) == ("number", "", ["what"])
    assert Template("\nA:{$}", "p").kind == "currency"
    assert Template("\nIt is {{n}} or{?}", "p").kind == "score"  # an escaped mark is text
    assert Template("\nA: {*}", "p").stop == ""  # no text after {*}: the answer ends at a newline
    for bad in ("\nA:{n} units", "\nA:{$}.", '\nA: "{*}" and {x}', "\nA: {*}\n", "\nA:{?}{n}"):
        with pytest.raises(Rejected) as e:
            Template(bad, "p")
        assert e.value.code == "invalid_template", bad


def _req(queries, content="Apple sold 3.5 million phones.", case_insensitive=True, **kw):
    return ScoreRequest.model_validate(
        {
            "model": "spinf/gemma-4-12b-fp8",
            "inputs": [{"id": "a", "messages": [{"role": "user", "content": content}]}],
            "scoring": {"queries": queries, "case_insensitive": case_insensitive},
            **kw,
        }
    )


def test_compile_extraction_jobs_and_usage():
    tok = WordTok()
    r = _req(
        [
            {"id": "units", "template": "\nQ: How many phones?\nA:{n}"},
            {"id": "brand", "template": '\nThe brand is "{*}"', "top_k": 3, "max_tokens": 8},
            {"id": "dir", "template": "\nDid sales go{?}", "options": [" up", " down"]},
        ],
        case_insensitive=False,
    )
    c = compiler.compile_request(r, tok, 16384)
    assert c.extract_jobs == []  # the slots are branches of the input's scoring tree, read in its pass
    tree = next(t for key, _, t in c.items if key == ("in", 0))
    xb = [(tree.branches[bi], pr) for bi, pr in tree.xbranches]
    assert [(pr.key, pr.kind, pr.stop, pr.max_tokens) for _, pr in xb] == [
        (("x", 0, 0, 0), "number", "", ex.DEFAULT_MAX_TOKENS["number"]),
        (("x", 0, 1, 0), "text", '"', 8),
    ]
    common = len(tree.prefix) - len([1] + tok.ids("Apple sold 3.5 million phones."))  # the templates' shared start ("\n")
    assert tree.prefix[: -common or None] == [1] + tok.ids("Apple sold 3.5 million phones.")
    assert xb[1][1].ids == tok.ids('\nThe brand is "') and xb[1][0] == xb[1][1].ids[common:]
    assert len(tree.prompt_rows) == 1 and len(c.items) == 2  # one scored prompt; its floor is the other item
    assert c.usage["query_tokens"] == len(tok.ids("\nQ: How many phones?\nA:")) + len(tok.ids('\nThe brand is "')) + len(
        tok.ids("\nDid sales go")
    )
    out = compiler.assemble(
        c,
        r,
        {("in", 0): [np.log([0.7, 0.3])], ("floor",): [np.log([0.5, 0.5])]},
        {("x", 0, 0, 0): ex.Found({"3.5 million": 0.4, "3,500,000": 0.2, "3": 0.1}), ("x", 0, 1, 0): ex.Found({"Apple": 0.6})},
    )
    q = out["results"][0]["queries"]
    assert q[0]["kind"] == "number" and q[0]["combinations"][0]["answers"][0] == {"text": "3.5 million", "p": 0.6, "value": 3.5e6}
    assert q[0]["combinations"][0]["p_negative"] == 0.0
    assert q[1]["combinations"][0]["answers"] == [{"text": "Apple", "p": 0.6}]
    assert q[2]["combinations"][0]["options"][0]["text"] == " up"


def test_extraction_rejections():
    tok = WordTok()
    for queries, code in (
        ([{"id": "a", "template": "\nA:{n}", "options": [" 1"]}], "invalid_field"),
        ([{"id": "a", "template": "\nA:{?}", "options": [" 1"], "top_k": 3}], "invalid_field"),
        ([{"id": "a", "template": "\nA:{?}"}], "missing_value"),
    ):
        with pytest.raises(Rejected) as e:
            compiler.compile_request(_req(queries), tok, 16384)
        assert e.value.code == code, queries


def test_long_content_runs_alone_and_the_engine_maximum():
    tok = WordTok()
    long_text = " ".join(f"w{i}" for i in range(12000))  # 12k one-token words: over half of an 16k budget
    queries = [{"id": f"q{k}", "template": f"\nIs {k} true?{{?}}", "options": [" yes", " no"], "empty_floor": False} for k in range(40)]
    lim = {"max_content_tokens": 100000}
    c = compiler.compile_request(_req(queries, long_text, limits=lim), tok, 16384, max_tokens=65536)
    assert len(c.items) == 1  # one solo tree, the content sent once
    assert c.items[0][2].n_tokens > 12000
    with pytest.raises(Rejected) as e:
        compiler.compile_request(_req(queries, long_text, limits=lim), tok, 8192, max_tokens=8192)
    assert e.value.code == "content_too_long"
    # the packer runs a tree over its budget alone
    tree = GroupedTree(list(range(20000)), [], [], [])
    batches = list(Packer(16384, 65536).push("solo", tree))
    assert len(batches) == 1 and batches[0].length == math.ceil(20000 / BLOCK) * BLOCK
    with pytest.raises(ValueError):
        list(Packer(16384, 16384).push("solo", tree))


# ---------------------------------------------------------------- video
def _video_input(ts):
    return Input.model_validate(
        {
            "id": "v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_url(96, 96), "detail": "low"}},
                        {"type": "text", "text": "A commercial:\n"},
                        {"type": "video", "video": {"frames": [{"t": t, "image_url": {"url": png_url(160, 90)}} for t in ts]}},
                        {"type": "text", "text": "\n"},
                    ],
                }
            ],
        }
    )


def test_video_frames_stay_in_place_after_their_timestamps():
    segs = content_segments(_video_input([0.0, 1.9, 75.2]), "inputs.0")
    assert isinstance(segs[0], media.Image) and not segs[0].inline  # a plain image still goes first
    assert segs[1] == "A commercial:\n00:00 "
    assert isinstance(segs[2], media.Image) and segs[2].inline and segs[3] == " 00:01 " and segs[5] == " 01:15 " and segs[7] == "\n"
    assert segs[2].n_tokens <= 70  # detail low by default
    assert timestamp(3599.9) == "59:59" and timestamp(3600) == "60:00"
    with pytest.raises(Rejected) as e:
        content_segments(_video_input([2.0, 1.0]), "inputs.0")
    assert e.value.code == "invalid_field" and e.value.param.endswith("frames.1.t")


# ---------------------------------------------------------------- grammar, parsing, merging
def _run(g, toks):
    st = g.start
    for t in toks:
        st = g.step(st, t)
        if st is None:
            return "reject"
    return "complete" if g.complete(st) else "incomplete"


@pytest.mark.parametrize(
    "toks,number,currency",
    [
        ([" ", "1", ",", "4", "9", "6", ".", "5"], "complete", "complete"),
        ([" (", "1", "6", "8", ")"], "complete", "complete"),
        ([" −", "2", "9", "8"], "complete", "complete"),
        ([" ", "1", "2", ".", "5", "%"], "complete", "complete"),
        ([" ", "3", ".", "5", " million"], "complete", "complete"),
        ([" $", "1", ",", "4"], "reject", "complete"),
        ([" $", "(", "1", ")"], "reject", "complete"),
        ([" ", "1", ",", " ", "2"], "reject", "reject"),
        ([" (", "1", "6"], "incomplete", "incomplete"),
        ([" ", "3", "."], "complete", "complete"),
    ],
)
def test_number_grammar(toks, number, currency):
    assert _run(ex.NumberGrammar(False), toks) == number
    assert _run(ex.NumberGrammar(True), toks) == currency


def test_parse_merge_sign_and_calibration():
    assert ex.parse_value("$1,496.5") == (1496.5, "$")
    assert ex.parse_value("(168)") == (-168.0, None) and ex.parse_value("$(1.6)") == (-1.6, "$")
    assert ex.parse_value("3.5 million") == (3.5e6, None) and ex.parse_value("12.5%") == (12.5, None)
    assert ex.norm_text("The Coca-Cola") == ex.norm_text("COCA COLA") == ex.norm_text("CocaCola")
    r = ex.result("text", ex.Found({"Coca-Cola": 0.3, "Coca Cola": 0.2, "Pepsi": 0.1}), 5, Calibration(a=1.0, b=1.0))
    assert r["answers"][0] == {"text": "Coca-Cola", "p": 0.5} and r["mass"] == 0.6
    assert r["p_calibrated"] == round(1 / (1 + math.exp(-1.0)), 6)  # logit(0.5) = 0
    r = ex.result("number", ex.Found({"(168)": 0.2, "168": 0.3, "-168": 0.1}), 5)
    assert r["answers"][0]["value"] == -168.0 and r["p_negative"] == 0.5
    r = ex.result("currency", ex.Found({"$168": 0.3, "168": 0.2}), 5)
    assert [a.get("currency") for a in r["answers"]] == ["$", None]


def test_stop_sets_and_candidates():
    texts = ['"', '",', ' "', "”", "\n", "a\nb", "Bud", " Light", "<|image|>", "1", " ", "$", " million", "x"]
    stop = ex.text_stop_ids(texts, '"', {99})
    assert stop == sorted([0, 1, 3, 4, 5, 99])
    cand = ex.number_candidates(texts, special={8})
    assert [texts[i] for i in cand] == ["1", " ", "$", " million"]


def test_usage_text_and_media_add_up_to_content():
    r = ScoreRequest.model_validate(
        {
            "model": "spinf/gemma-4-12b-fp8",
            "inputs": [{"id": "v", "messages": [{"role": "user", "content": _video_input([0.0, 1.0]).messages[0].content}]}],
            "scoring": {"queries": [{"id": "q", "template": "\nIs it?{?}", "options": [" yes", " no"]}]},
        }
    )
    ids = {"boi": 900, "image": 901, "eoi": 902, "boa": 910, "audio": 911, "eoa": 912}
    u = compiler.compile_request(r, WordTok(), 16384, ids).usage
    assert u["images"] == 1 and u["video_frames"] == 2 and u["image_tokens"] > 0 and u["video_tokens"] > 0
    assert u["content_tokens"] == u["text_tokens"] + u["image_tokens"] + u["video_tokens"] + u["audio_tokens"]
    assert u["text_tokens"] == len(WordTok().ids("A commercial:\n00:00 ")) + len(WordTok().ids(" 00:01 ")) + len(WordTok().ids("\n"))


def test_extraction_slots_per_input_are_limited():
    tok = WordTok()
    qs = [{"id": "a", "template": "\nQ: {what}?\nA:{n}", "combinations": [{"what": f"x{k}"} for k in range(4)]}]
    compiler.compile_request(_req(qs, limits={"max_extractions_per_input": 4}), tok, 16384)  # 4 slots: fine
    with pytest.raises(Rejected) as e:
        compiler.compile_request(_req(qs, limits={"max_extractions_per_input": 3}), tok, 16384)
    assert e.value.code == "too_many_extractions"


def test_zero_length_extraction_branch_at_the_prefix_end():
    """a hybrid model's Scorer.extract: a prompt without template tokens is an xbranch -1, packed as a zero-length branch whose
    end is the prefix's last token"""
    pr = ex.ExtractPrompt("k", [], "number", "", 8)
    tree = GroupedTree(
        [1, 2, 3, 4, 5], [[7, 8]], [], [], xbranches=[(0, ex.ExtractPrompt("t", [7, 8], "text", '"', 4)), (-1, pr)], parents=[-1]
    )
    b = Packer(1024, 1024)._pack([(("x",), tree)], BLOCK)
    (o,) = b.owners
    assert o.prefix_len == 5 and o.start == 0
    assert [(bs, bl, p.key) for bs, bl, p in o.xbranches] == [(5, 2, "t"), (5, 0, "k")]
    # the slot's read position bs + bl - 1 is the prefix end for the zero-length branch, the template end otherwise
    assert [bs + bl - 1 for bs, bl, _ in o.xbranches] == [6, 4]


# ---------------------------------------------------------------- generation (a text slot written instead of searched)
def test_generate_field_parses_and_is_plumbed():
    tok = WordTok()
    r = _req(
        [
            {"id": "sum", "template": '\nIn one sentence, this ad: "{*}"', "max_tokens": 48, "generate": True},
            {"id": "alt", "template": '\nAlso: "{*}"', "generate": {"temperature": 0.7, "top_p": 0.8, "seed": 3, "n": 2}},
            {"id": "brand", "template": '\nThe brand is "{*}"', "top_k": 3},
        ]
    )
    assert r.scoring.queries[0].gen_spec() == ex.Generate()  # true = greedy, one sequence
    assert r.scoring.queries[1].gen_spec() == ex.Generate(0.7, 0.8, ex.GEN_TOP_K_DEFAULT, 3, 2)  # top_k defaults when sampling
    assert r.scoring.queries[2].gen_spec() is None
    c = compiler.compile_request(r, tok, 16384)
    tree = next(t for key, _, t in c.items if key == ("in", 0))
    gens = [pr.gen for _, pr in tree.xbranches]
    assert gens[0] == ex.Generate() and gens[1].n == 2 and gens[2] is None
    # the response: generated slots answer with text + finish and no probability; searched slots as before
    found = {("x", 0, 0, 0): ex.Found(gen=[("A car ad", "stop")]), ("x", 0, 1, 0): ex.Found(gen=[("x", "length"), ("y", "stop")])}
    found[("x", 0, 2, 0)] = ex.Found(answers={"Apple": 0.4})
    out = compiler.assemble(c, r, {}, found)["results"][0]["queries"]
    assert out[0] == {
        "id": "sum",
        "kind": "text",
        "generate": True,
        "combinations": [{"values": {}, "answers": [{"text": "A car ad", "finish": "stop"}]}],
    }
    assert [a["finish"] for a in out[1]["combinations"][0]["answers"]] == ["length", "stop"]
    assert "generate" not in out[2] and out[2]["combinations"][0]["mass"] == 0.4


def test_generate_rejections():
    def code(q, **kw):
        with pytest.raises(Rejected) as e:
            compiler.compile_request(_req([q], **kw), WordTok(), 16384)
        return e.value.code

    assert code({"id": "n", "template": "\nA:{n}", "generate": True}) == "invalid_field"  # numbers are read, not written
    assert code({"id": "t", "template": '\nA: "{*}"', "generate": True, "top_k": 2}) == "invalid_field"  # no probabilities to rank
    assert code({"id": "t", "template": '\nA: "{*}"', "generate": True, "calibration": {"a": 1, "b": 0}}) == "invalid_field"
    assert code({"id": "s", "template": "\nIs it?{?}", "options": [" yes", " no"], "generate": True}) == "invalid_field"
    with pytest.raises(ValueError):  # pydantic: n over the beam rows, temperature out of range
        _req([{"id": "t", "template": '\nA: "{*}"', "generate": {"n": ex.GEN_MAX_N + 1}}])
    with pytest.raises(ValueError):
        _req([{"id": "t", "template": '\nA: "{*}"', "generate": {"temperature": 3}}])
    # generate: false is no generation
    r = _req([{"id": "t", "template": '\nA: "{*}"', "generate": False, "top_k": 2}])
    assert r.scoring.queries[0].gen_spec() is None


def test_gen_next_greedy_and_sampling():
    top_lp = np.log(np.array([0.5, 0.3, 0.1, 0.05, 0.05]))
    top_ids = np.array([11, 12, 13, 14, 15])
    greedy = ex.Generate()
    assert ex.gen_next(greedy, None, math.log(0.4), top_lp, top_ids) == 11  # the best token beats the end
    assert ex.gen_next(greedy, None, math.log(0.6), top_lp, top_ids) is None  # the end wins: the sequence stops
    # sampling: reproducible from the seed, within top_k / top_p, and the end event is one of the candidates
    gen = ex.Generate(temperature=1.0, top_p=1.0, top_k=5, seed=7, n=1)
    draws = [ex.gen_next(gen, ex.gen_rngs(gen)[0], math.log(0.4), top_lp, top_ids) for _ in range(3)]
    assert draws[0] == draws[1] == draws[2]
    rng = ex.gen_rngs(gen)[0]
    picks = {ex.gen_next(gen, rng, math.log(0.4), top_lp, top_ids) for _ in range(400)}
    assert None in picks and picks - {None} <= set(top_ids.tolist()) and len(picks) >= 4
    # a tight nucleus keeps only the most likely candidates: token 11 (0.5 of 1.4) and the end (0.4) cover 0.64 of the mass
    tight = ex.Generate(temperature=1.0, top_p=0.6, top_k=5, seed=1)
    rng = ex.gen_rngs(tight)[0]
    assert {ex.gen_next(tight, rng, math.log(0.4), top_lp, top_ids) for _ in range(200)} == {11, None}
    tighter = ex.Generate(temperature=1.0, top_p=0.3, top_k=5, seed=1)
    rng = ex.gen_rngs(tighter)[0]
    assert {ex.gen_next(tighter, rng, math.log(0.4), top_lp, top_ids) for _ in range(200)} == {11}
    # top_k 1 at temperature > 0 is greedy over {end, best token}
    one = ex.Generate(temperature=1.0, top_p=1.0, top_k=1, seed=1)
    rng = ex.gen_rngs(one)[0]
    assert {ex.gen_next(one, rng, math.log(0.05), top_lp, top_ids) for _ in range(50)} <= {11, None}
    # a low temperature sharpens towards the arg-max
    cold = ex.Generate(temperature=0.05, top_p=1.0, top_k=5, seed=2)
    rng = ex.gen_rngs(cold)[0]
    assert {ex.gen_next(cold, rng, math.log(0.05), top_lp, top_ids) for _ in range(200)} == {11}
    assert len(ex.gen_rngs(ex.Generate(n=3))) == 3
    # a generated sequence is stripped of the stop text the model wrote into its last token (`."`, `”`)
    f = ex.Found()
    f.add_gen(' Five Star offers deals."', "stop", '"')
    f.add_gen("Ends here.”", "stop", '"')
    f.add_gen("no stop text", "length", "")
    assert f.gen == [("Five Star offers deals.", "stop"), ("Ends here.", "stop"), ("no stop text", "length")]
