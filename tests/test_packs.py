"""spinf_serve.packs on a small synthetic pack: the file validation, the request side (contexts, parameters, grouped queries),
the evaluator in its three modes with requires / decides / critical and the lock fallback, the request refs and their error
codes, the sources (a fake GitHub), the catalog rules and the commands. The official files are checked in test_pack_fixtures.py."""

import base64
import copy
import io
import json
import math
import os
import struct
import subprocess
import sys
import tarfile
import urllib.error

import pytest

from spinf_serve import packs
from spinf_serve.packs import PackError, catalog, pipeline, refs, sources
from spinf_serve.protocol import Rejected

MODEL = "spinf/gemma-4-12b-fp8"


def make_pack(pack_id="acme/demo", version="0.1.0", model=MODEL, micro=True, **over):
    """two queries (a yes/no pair with combinations, a 4-way choice), reads of every kind, three categories: `fraud` (decides),
    `money` (a gate: thresholded, decides: false), `urgent` (requires money), plus a linear micro-layer when `micro`"""
    data = {
        "pack": pack_id,
        "version": version,
        "model": model,
        "media": ["text"],
        "parameters": {"brand": {"label": "Brand", "required": False, "default": "Acme"}},
        "contexts": {"mail": "Brand: {brand}\n\nAn email:\n\n{X}\n\n(end; the brand is {brand})", "chat": "A chat message:\n\n{X}"},
        "scoring": {
            "empty_floor": False,
            "case_insensitive": False,
            "queries": [
                {
                    "id": "q.is",
                    "template": "\n\nIs this {F}?\nAnswer:{?}",
                    "options": [" yes", " no"],
                    "combinations": [{"F": "a scam"}, {"F": "urgent"}],
                },
                {"id": "q.topic", "template": "\n\nTopic:{?}", "options": [" none", " money", " fraud", " other"]},
            ],
        },
        "reads": [
            {"id": "scam:pair", "query": "q.is", "F": "a scam", "kind": "pair", "category": "fraud", "mean": -2.0, "sd": 1.5},
            {
                "id": "fraud:opt",
                "query": "q.topic",
                "kind": "option",
                "option": 2,
                "category": "fraud",
                "mu": -3.0,
                "mean": 0.0,
                "sd": 1.0,
                "view": "a",
            },
            {
                "id": "any:topic",
                "query": "q.topic",
                "kind": "any",
                "category": "fraud",
                "mu": [0, -1, -1, -1],
                "mean": 0.5,
                "sd": 2.0,
                "view": "b",
            },
            {
                "id": "money:set",
                "query": "q.topic",
                "kind": "set",
                "unsafe": [1, 2],
                "category": "money",
                "mu": [0, -1, -1, -1],
                "mean": 0.0,
                "sd": 1.0,
            },
            {"id": "urgent:pair", "query": "q.is", "F": "urgent", "kind": "pair", "sign": -1, "category": "urgent", "mean": 1.0, "sd": 1.0},
        ],
        "categories": {
            "fraud": {
                "label": "Fraud",
                "platt": [1.2, -0.5],
                "threshold": 0.6,
                "aggregate": "views",
                "normal_quantiles": {"levels": [0, 0.5, 1], "values": [0.1, 0.3, 0.9]},
            },
            "money": {"label": "About money", "platt": [1.0, 0.0], "threshold": 0.5, "decides": False},
            "urgent": {"label": "Urgent", "platt": [1.0, 0.0], "threshold": 0.5, "requires": ["money"]},
        },
        "decision": {
            "mode": "per_category",
            "fallback": "per_category",
            "critical": ["urgent"],
            "max": {"threshold": 0.7},
            "per_category": {},
        },
        "answers": "logprob",
    }
    if micro:
        w = [1.0, 0.5, -0.25, 0.1, 0.4]  # w over [c:fraud, c:money, r:scam:pair], b, threshold
        data["decision"]["micro_layer"] = {
            "inputs": ["c:fraud", "c:money", "r:scam:pair"],
            "hidden": 0,
            "activation": "tanh",
            "dtype": "float32",
            "weights": base64.b64encode(struct.pack("<5f", *w)).decode(),
        }
        data["decision"]["micro_layer"]["lock"] = pipeline.lock_of(data)
        data["decision"]["mode"] = "micro_layer"
    data.update(over)
    return data


def result_for(p_scam: float, p_urgent: float, topic: list[float]) -> dict:
    """one input's /v1/score result for the demo pack (logprobs), with a caller's own query next to it"""

    def opts(ps, texts):
        return [{"text": t, "logprob": math.log(p), "p": round(p, 6)} for t, p in zip(texts, ps, strict=True)]

    return {
        "input_id": "0",
        "queries": [
            {"id": "own", "combinations": [{"values": {}, "options": opts([0.9, 0.1], [" en", " it"])}]},
            {
                "id": "q.is",
                "combinations": [
                    {"values": {"F": "a scam"}, "options": opts([p_scam, 1 - p_scam], [" yes", " no"])},
                    {"values": {"F": "urgent"}, "options": opts([p_urgent, 1 - p_urgent], [" yes", " no"])},
                ],
            },
            {"id": "q.topic", "combinations": [{"values": {}, "options": opts(topic, [" none", " money", " fraud", " other"])}]},
        ],
    }


SCAM = result_for(0.95, 0.9, [0.05, 0.3, 0.6, 0.05])
PLAIN = result_for(0.02, 0.1, [0.8, 0.1, 0.05, 0.05])


# ---------------------------------------------------------------- the file
def test_a_valid_pack_loads_and_an_invalid_one_lists_its_problems(tmp_path):
    p = packs.from_data(make_pack())
    assert (p.id, p.namespace, p.version, p.model, p.group) == ("acme/demo", "acme", "0.1.0", MODEL, "acme/demo@0.1.0")
    assert p.query_ids == {"q.is", "q.topic"} and p.prompts == 3 and p.lock_ok and p.layer is not None
    assert p.modes == ["max", "per_category", "micro_layer"] and p.contexts == ["mail", "chat"] and p.media == ("text",)
    f = tmp_path / "demo.json"
    f.write_text(json.dumps(make_pack()))
    assert packs.load(f, source="local:x").source == "local:x"
    f.write_text("{not json")
    with pytest.raises(PackError, match="not JSON"):
        packs.load(f)

    def problems(**over):
        d = make_pack()
        for k, v in over.items():
            d[k] = v
        return packs.validate(d)

    assert problems() == []
    assert problems(pack="Demo") == ["pack: an id of the form <namespace>/<name> (lower case, digits, . _ -)"]
    assert any("exactly one {X}" in s for s in problems(contexts={"a": "no slot"}))
    assert any("unknown placeholders ['who']" in s for s in problems(contexts={"a": "{who}: {X}"}))
    assert any("media" in s for s in problems(media=["image"]))
    assert any("answers" in s for s in problems(answers="logit"))
    bad = make_pack()
    bad["reads"][0]["F"] = "a joke"
    bad["reads"][1]["option"] = 7
    bad["reads"][3]["unsafe"] = [0, 1, 2, 3]
    bad["reads"][4]["sd"] = 0
    bad["categories"]["ghost"] = {"label": "no reads", "platt": [1, 0], "threshold": 2}
    bad["categories"]["urgent"]["requires"] = ["nope"]
    bad["decision"]["critical"] = ["x"]
    bad["scoring"]["queries"][1]["group"] = "g"
    out = packs.validate(bad)
    for needle in (
        "F 'a joke'",
        "option index below 4",
        "not all of them",
        "sd (> 0)",
        "no read feeds it",
        "threshold is null or between",
        "requires lists",
        "critical",
        "no group",
    ):
        assert any(needle in s for s in out), needle
    with pytest.raises(PackError) as e:
        packs.from_data(bad, "bad.json")
    assert e.value.where == "bad.json" and len(e.value.problems) == len(out)
    assert any("micro_layer" in s for s in packs.validate(make_pack(micro=False, decision={"mode": "micro_layer"})))
    # a draft (a workbench's pack before calibration): no mu / mean / sd / platt / thresholds / micro-layer; the request side works
    draft = make_pack(micro=False, decision={"mode": "micro_layer", "fallback": "max"})
    for r in draft["reads"]:
        for k in ("mu", "mean", "sd"):
            r.pop(k, None)
    for c in draft["categories"].values():
        c.pop("platt", None)
        c.pop("threshold", None)
    assert len(packs.validate(draft)) > 5 and packs.validate(draft, draft=True) == []
    d = packs.from_data(draft, draft=True)
    assert d.draft and d.layer is None and not d.lock_ok and d.prompts == 3
    assert d.request(["hi"])["scoring"]["queries"][0]["group"] == "acme/demo@0.1.0"
    with pytest.raises(ValueError, match="draft"):
        d.evaluate({})
    draft["reads"][0]["F"] = "a joke"
    assert any("F 'a joke'" in s for s in packs.validate(draft, draft=True))  # the structure is still checked
    ml = make_pack()
    ml["decision"]["micro_layer"]["weights"] = base64.b64encode(struct.pack("<3f", 1, 2, 3)).decode()
    assert any("3 floats for 3 inputs" in s for s in packs.validate(ml))
    assert packs.validate([]) == ["a pack file is a JSON object"] and packs.validate({}) == [
        f"missing field {k!r}" for k in ("pack", "version", "model", "contexts", "scoring", "reads", "categories", "decision")
    ]


def test_contexts_parameters_and_the_request_body():
    p = packs.from_data(make_pack())
    assert p.fill("hi", "chat") == "A chat message:\n\nhi"
    assert p.fill("hi") == "Brand: Acme\n\nAn email:\n\nhi\n\n(end; the brand is Acme)"  # the first context, the default value
    assert p.fill("hi", "mail", {"brand": "Kestrel"}).count("Kestrel") == 2
    assert p.wrap("hi", "chat") == [{"type": "text", "text": "A chat message:\n\n"}, {"type": "text", "text": "hi"}]
    assert p.fill("{X} {brand}", "chat") == "A chat message:\n\n{X} {brand}"  # the text is never a template
    with pytest.raises(ValueError, match="no parameter 'who'"):
        p.fill("hi", params={"who": "x"})
    with pytest.raises(ValueError, match="no context 'sms'"):
        p.fill("hi", "sms")
    req = make_pack()
    req["parameters"]["brand"]["required"] = True
    del req["parameters"]["brand"]["default"]
    with pytest.raises(ValueError, match="'brand' is required"):
        packs.from_data(req).fill("hi")
    assert packs.from_data(req).parameter_values({"brand": 7}) == {"brand": "7"}
    qs = p.queries()
    assert [q["group"] for q in qs] == ["acme/demo@0.1.0"] * 2 and all(q["empty_floor"] is False for q in qs)
    body = p.request(["a", "b"], "chat", ids=["x", "y"], params=[None, {"brand": "K"}])
    assert body["model"] == MODEL and [i["id"] for i in body["inputs"]] == ["x", "y"]
    assert body["inputs"][1]["messages"][0]["content"][1] == {"type": "text", "text": "b"}
    assert body["scoring"] == {"queries": qs, "case_insensitive": False}
    assert p.request(["a"], model="other")["model"] == "other"
    pub = p.public(["0.1.0", "0.0.9"])
    assert pub["versions"] == ["0.1.0", "0.0.9"] and pub["queries"] == 2 and pub["prompts"] == 3 and pub["name"] == "acme/demo"
    assert pub["decision"] == {
        "mode": "micro_layer",
        "modes": ["max", "per_category", "micro_layer"],
        "fallback": "per_category",
        "max_threshold": 0.7,
        "critical": ["urgent"],
    }
    assert pub["parameters"] == [{"id": "brand", "label": "Brand", "required": False, "default": "Acme"}]
    assert [c["id"] for c in pub["categories"]] == ["fraud", "money", "urgent"] and pub["contexts"][0]["id"] == "mail"


# ---------------------------------------------------------------- the evaluator
def test_reads_and_answers():
    p = packs.from_data(make_pack())
    ans = p.answers(SCAM)
    assert set(ans) == {("q.is", "a scam"), ("q.is", "urgent"), ("q.topic", None)}  # the caller's own query is not in
    assert abs(sum(ans[("q.topic", None)]) - 1) < 1e-9
    # the same answers from the rounded p when the pack says so
    by_p = pipeline.answers_from_result(SCAM, {"q.is"}, "p")
    assert set(by_p) == {("q.is", "a scam"), ("q.is", "urgent")} and by_p[("q.is", "a scam")] == [0.95, 0.05]
    r = {x["id"]: x for x in p.data["reads"]}
    assert pipeline.read_value(r["scam:pair"], ans) == pytest.approx(math.log(0.95 / 0.05))
    assert pipeline.read_value(r["urgent:pair"], ans) == pytest.approx(-math.log(0.9 / 0.1))  # sign -1
    assert pipeline.read_value(r["fraud:opt"], ans) == pytest.approx(math.log(0.6) + 3.0)
    a = [math.log(x) - mu for x, mu in zip([0.05, 0.3, 0.6, 0.05], [0, -1, -1, -1], strict=True)]
    assert pipeline.read_value(r["any:topic"], ans) == pytest.approx(pipeline._logsumexp(a[1:]) - a[0])
    assert pipeline.read_value(r["money:set"], ans) == pytest.approx(pipeline._logsumexp(a[1:3]) - pipeline._logsumexp([a[0], a[3]]))
    with pytest.raises(ValueError, match="unknown read kind"):
        pipeline.read_value({**r["any:topic"], "kind": "sum"}, ans)
    z = pipeline.read_z(p.data, ans)
    macros = pipeline.category_macros(p.data, z)
    assert macros["fraud"] == pytest.approx(
        (z["fraud:opt"] + z["any:topic"]) / 2
    )  # views: the pair read has no view, the two views average
    assert macros["money"] == z["money:set"]
    assert pipeline.interp(0.2, [0.1, 0.3, 0.9], [0, 0.5, 1]) == pytest.approx(0.25)
    assert pipeline.interp(0.0, [0.1, 0.3], [0, 1]) == 0 and pipeline.interp(5, [0.1, 0.3], [0, 1]) == 1


def test_the_three_modes_requires_decides_and_critical():
    p = packs.from_data(make_pack())
    scam = p.evaluate(SCAM)
    assert scam["id"] == "acme/demo" and scam["version"] == "0.1.0" and scam["context"] == "mail"
    cats = scam["categories"]
    assert cats["fraud"]["flagged"] is True and 0 <= cats["fraud"]["percentile"] <= 1
    assert cats["money"]["flagged"] is True and cats["money"]["percentile"] is None
    assert cats["urgent"]["flagged"] is False  # P(urgent) = 0.9 with sign -1: a low value
    assert scam["decision"]["mode"] == "micro_layer" and scam["decision"]["unsafe"] is True
    v = scam["decision"]["score"]
    z = pipeline.read_z(p.data, p.answers(SCAM))
    m = pipeline.category_macros(p.data, z)
    assert v == pytest.approx(1.0 * m["fraud"] + 0.5 * m["money"] - 0.25 * z["scam:pair"] + 0.1, abs=1e-6) and scam["decision"][
        "threshold"
    ] == pytest.approx(0.4)
    assert "critical" not in scam["decision"]

    plain = p.evaluate(PLAIN, "per_category")
    assert plain["decision"] == {"unsafe": False, "mode": "per_category", "categories": []}
    # urgent requires money: P(urgent) = 0.1 with sign -1 gives a high value, but money is not flagged, so urgent is not either
    assert plain["categories"]["urgent"]["flagged"] is False
    # give money a lower threshold: urgent flags, is critical, and decides even though money itself never does
    low = p.evaluate(PLAIN, "per_category", thresholds={"money": 0.0})
    assert low["categories"]["money"]["flagged"] is True and low["categories"]["urgent"]["flagged"] is True
    assert low["decision"]["categories"] == ["urgent"] and low["decision"]["critical"] == ["urgent"] and "by" not in low["decision"]
    # in max mode the gate is not a deciding category: the top is fraud or urgent
    mx = p.evaluate(PLAIN, "max", thresholds={"money": 0.0})
    assert mx["decision"]["category"] == "urgent" and mx["decision"]["threshold"] == 0.7
    assert mx["decision"]["unsafe"] is True and mx["decision"]["critical"] == ["urgent"] and "by" not in mx["decision"]
    hi = p.evaluate(PLAIN, "max", thresholds={"money": 0.0}, threshold=0.9)  # under the max threshold: critical forces unsafe
    assert hi["decision"]["unsafe"] is True and hi["decision"]["by"] == "critical"
    assert p.evaluate(PLAIN, "max", thresholds={"money": 0.0}, threshold=0.9, critical=[])["decision"]["unsafe"] is False
    assert p.evaluate(PLAIN, "max", threshold=0.0)["decision"]["unsafe"] is True
    assert p.evaluate(SCAM, "micro_layer", threshold=100.0)["decision"]["unsafe"] is False
    none = p.evaluate(PLAIN, "per_category", thresholds={"fraud": None, "urgent": None})
    assert none["categories"]["fraud"]["flagged"] is None and none["decision"]["unsafe"] is False
    with pytest.raises(KeyError):
        p.evaluate(PLAIN, thresholds={"nope": 0.1})
    with pytest.raises(ValueError, match="mode must be"):
        pipeline.evaluate(p.data, p.answers(PLAIN), "vote")
    # the answers can be passed directly, and the raw evaluator also returns the macros
    assert p.evaluate(p.answers(SCAM))["decision"] == scam["decision"]
    assert set(pipeline.evaluate(p.data, p.answers(SCAM))) == {"categories", "decision", "macros"}


def test_lock_fallback_and_an_edited_pack():
    data = make_pack()
    p = packs.from_data(data)
    assert p.lock_ok and pipeline.lock_of(data) == data["decision"]["micro_layer"]["lock"]
    edited = copy.deepcopy(data)
    edited["categories"]["fraud"]["platt"][1] += 0.1
    e = packs.from_data(edited)
    assert not e.lock_ok
    out = e.evaluate(SCAM)
    assert out["decision"]["mode"] == "per_category" and out["decision"]["warning"] == pipeline.LOCK_WARNING
    assert e.evaluate(SCAM, "max")["decision"]["mode"] == "max" and "warning" not in e.evaluate(SCAM, "max")["decision"]
    # thresholds, critical and requires are not in the lock
    edited = copy.deepcopy(data)
    edited["categories"]["fraud"]["threshold"] = 0.9
    edited["decision"]["critical"] = []
    edited["categories"]["urgent"]["requires"] = []
    assert packs.from_data(edited).lock_ok
    # a hidden layer decodes and runs
    hidden = copy.deepcopy(data)
    n, h = 3, 2
    w = [0.1] * (n * h) + [0.0] * h + [1.0, -1.0] + [0.05, 0.2]
    hidden["decision"]["micro_layer"].update({"hidden": h, "weights": base64.b64encode(struct.pack(f"<{len(w)}f", *w)).decode()})
    hidden["decision"]["micro_layer"]["lock"] = pipeline.lock_of(hidden)
    f, t = pipeline.micro_layer(hidden)
    assert t == pytest.approx(0.2) and f([1.0, 1.0, 1.0]) == pytest.approx(math.tanh(0.3) - math.tanh(0.3) + 0.05)
    assert pipeline.micro_layer_size(3, 0) == 5 and pipeline.micro_layer_size(3, 2) == 12
    with pytest.raises(ValueError, match="only float32"):
        pipeline.micro_layer({"decision": {"micro_layer": {**data["decision"]["micro_layer"], "dtype": "float64"}}})


# ---------------------------------------------------------------- a request's packs
def two_source_catalog(tmp_path, model=MODEL):
    c = catalog.Catalog(model)
    s = catalog.Scan("acme/packs@v1", str(tmp_path), None)
    s.packs = [packs.from_data(make_pack(), source="acme/packs@v1"), packs.from_data(make_pack(version="0.2.0"), source="acme/packs@v1")]
    c.add(s)
    return c


def test_parse_refs_and_their_errors(tmp_path):
    c = two_source_catalog(tmp_path)
    assert c.versions("acme/demo") == ["0.2.0", "0.1.0"]  # discovered: newest first, the default
    [r] = refs.parse_refs([{"id": "acme/demo"}], c)
    assert r.pack.version == "0.2.0" and r.context == "mail" and r.mode == "micro_layer" and not r.include_queries
    [r] = refs.parse_refs(
        [
            {
                "id": "acme/demo",
                "version": "0.1.0",
                "context": "chat",
                "include_queries": True,
                "decision": {
                    "mode": "max",
                    "threshold": 0.5,
                    "thresholds": {"fraud": None, "money": 0.2},
                    "critical": ["urgent", "urgent"],
                },
            }
        ],
        c,
    )
    assert (r.pack.version, r.context, r.mode, r.threshold, r.thresholds, r.critical, r.include_queries) == (
        "0.1.0",
        "chat",
        "max",
        0.5,
        {"fraud": None, "money": 0.2},
        ["urgent"],
        True,
    )

    def code(value):
        with pytest.raises(Rejected) as e:
            refs.parse_refs(value, c)
        return e.value.code, e.value.param

    assert code({}) == ("invalid_field", "scoring.packs")
    assert code([1]) == ("invalid_field", "scoring.packs.0")
    assert code([{"id": "acme/demo", "x": 1}]) == ("invalid_field", "scoring.packs.0.x")
    assert code([{"id": ""}]) == ("invalid_field", "scoring.packs.0.id")
    assert code([{"id": "acme/nope"}]) == ("unknown_pack", "scoring.packs.0.id")
    assert code([{"id": "acme/demo"}, {"id": "acme/demo"}]) == ("duplicate_id", "scoring.packs.1.id")
    assert code([{"id": "acme/demo", "version": "9"}]) == ("unknown_pack_version", "scoring.packs.0.version")
    assert code([{"id": "acme/demo", "context": "sms"}]) == ("invalid_field", "scoring.packs.0.context")
    assert code([{"id": "acme/demo", "include_queries": 1}]) == ("invalid_field", "scoring.packs.0.include_queries")
    assert code([{"id": "acme/demo", "decision": 1}]) == ("invalid_field", "scoring.packs.0.decision")
    assert code([{"id": "acme/demo", "decision": {"vote": 1}}]) == ("invalid_field", "scoring.packs.0.decision.vote")
    assert code([{"id": "acme/demo", "decision": {"mode": "vote"}}]) == ("invalid_field", "scoring.packs.0.decision.mode")
    assert code([{"id": "acme/demo", "decision": {"mode": "per_category", "threshold": 0.5}}]) == (
        "invalid_field",
        "scoring.packs.0.decision.threshold",
    )
    assert code([{"id": "acme/demo", "decision": {"mode": "max", "threshold": 1.5}}]) == (
        "invalid_field",
        "scoring.packs.0.decision.threshold",
    )
    assert code([{"id": "acme/demo", "decision": {"threshold": "x"}}]) == ("invalid_field", "scoring.packs.0.decision.threshold")
    assert code([{"id": "acme/demo", "decision": {"thresholds": {"nope": 0.1}}}]) == (
        "unknown_category",
        "scoring.packs.0.decision.thresholds.nope",
    )
    assert code([{"id": "acme/demo", "decision": {"thresholds": {"fraud": 2}}}]) == (
        "invalid_field",
        "scoring.packs.0.decision.thresholds.fraud",
    )
    assert code([{"id": "acme/demo", "decision": {"critical": "urgent"}}]) == ("invalid_field", "scoring.packs.0.decision.critical")
    assert code([{"id": "acme/demo", "decision": {"critical": ["nope"]}}]) == ("unknown_category", "scoring.packs.0.decision.critical.0")
    # a micro-layer threshold is unbounded; a pack without a micro-layer does not offer the mode
    [r] = refs.parse_refs([{"id": "acme/demo", "decision": {"threshold": -3}}], c)
    assert r.threshold == -3.0
    lite = catalog.Catalog(MODEL)
    lite.add(catalog.Scan("l", "", None, [packs.from_data(make_pack(micro=False, decision={"mode": "max", "max": {"threshold": 0.5}}))]))
    with pytest.raises(Rejected) as e:
        refs.parse_refs([{"id": "acme/demo", "decision": {"mode": "micro_layer"}}], lite)
    assert "max, per_category" in e.value.message
    # listed but coming soon
    soon = catalog.Catalog(MODEL)
    soon.listed["acme/soon"] = catalog.RepoPack(
        id="acme/soon", name="Soon", versions=[catalog.RepoVersion(version="1", file="x.json", model=MODEL, status="coming_soon")]
    )
    with pytest.raises(Rejected) as e:
        refs.parse_refs([{"id": "acme/soon"}], soon)
    assert e.value.code == "pack_not_available"


def test_the_call_with_packs_wrap_media_queries_and_apply(tmp_path):
    c = two_source_catalog(tmp_path)
    [r] = refs.parse_refs([{"id": "acme/demo", "context": "chat"}], c)
    assert r.wrap([{"type": "text", "text": "hi"}]) == [{"type": "text", "text": "A chat message:\n\n"}, {"type": "text", "text": "hi"}]
    [m] = refs.parse_refs([{"id": "acme/demo"}], c)
    parts = m.wrap([{"type": "text", "text": "hi"}], {"brand": "K"})
    assert parts[0]["text"].startswith("Brand: K") and parts[-1]["text"].endswith("the brand is K)")
    with pytest.raises(Rejected) as e:
        m.wrap([], {"who": "x"})
    assert e.value.code == "invalid_field"
    r.check_media({"text"})
    with pytest.raises(Rejected) as e:
        r.check_media({"text", "image"}, "inputs.0")
    assert (e.value.code, e.value.param) == ("pack_media_not_supported", "inputs.0")
    refs.check_call([r], ["own"], None)
    refs.check_call([r], [], False)
    with pytest.raises(Rejected) as e:
        refs.check_call([r], ["q.is"], None)
    assert e.value.code == "duplicate_id"
    with pytest.raises(Rejected) as e:
        refs.check_call([r], [], True)
    assert e.value.code == "pack_conflict"
    assert [q["id"] for q in refs.queries([r])] == ["q.is", "q.topic"] and refs.queries([r])[0]["group"] == "acme/demo@0.2.0"
    out, warnings = refs.apply([r], [SCAM, PLAIN], [{"query_id": "q.is", "code": "x"}, {"query_id": "own", "code": "y"}])
    assert [q["id"] for q in out[0]["queries"]] == ["own"] and warnings == [{"query_id": "own", "code": "y"}]
    assert (
        out[0]["packs"][0]["id"] == "acme/demo"
        and out[0]["packs"][0]["context"] == "chat"
        and out[1]["packs"][0]["decision"]["unsafe"] is False
    )
    r.include_queries = True
    out, _ = refs.apply([r], [SCAM], [])
    assert len(out[0]["queries"]) == 3


# ---------------------------------------------------------------- sources
def test_source_specs_and_labels(monkeypatch):
    s = sources.parse("github:acme/prompt-packs@v1.2#packs/")
    assert (s.kind, s.owner, s.repo, s.ref, s.path) == ("github", "acme", "prompt-packs", "v1.2", "packs")
    assert s.label() == "acme/prompt-packs@v1.2" and not s.official
    assert (
        sources.parse("github:acme/x").label("a" * 40) == "acme/x@aaaaaaaaaaaa" and sources.parse("github:acme/x").label() == "acme/x@HEAD"
    )
    with pytest.raises(sources.SourceError):
        sources.parse("github:acme")
    local = sources.parse("./my-packs")
    assert (local.kind, local.folder, local.label()) == ("local", "./my-packs", "local:./my-packs")
    monkeypatch.delenv("SPINF_PACKS_OFFICIAL", raising=False)
    o = sources.parse("official")
    assert (
        o.official
        and o.spec == sources.OFFICIAL == f"github:spinfinc/spinf-prompts@{sources.OFFICIAL_TAG}"
        and o.ref == sources.OFFICIAL_TAG
    )
    monkeypatch.setenv("SPINF_PACKS_OFFICIAL", "/srv/spinf-prompts")
    assert sources.parse("official") == sources.parse("/srv/spinf-prompts")
    monkeypatch.setenv("SPINF_PACKS_OFFICIAL", "github:me/spinf-prompts-fork@main")
    assert not sources.parse("official").official
    monkeypatch.setenv("SPINF_PACKS_CACHE", "/x/cache")
    assert sources.cache_dir() == "/x/cache"
    monkeypatch.delenv("SPINF_PACKS_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", "/y")
    assert sources.cache_dir() == "/y/spinf-serve/packs"


class FakeGitHub:
    """the GitHub API of one repository: tag v1 and branch main at one commit, a tarball with a manifest and one pack"""

    SHA = "0123456789abcdef0123456789abcdef01234567"

    def __init__(self, files: dict[str, str]):
        self.files, self.calls, self.headers = files, [], []

    def __call__(self, url: str, headers: dict[str, str]) -> bytes:
        self.calls.append(url)
        self.headers.append(headers)
        tail = url.split("/repos/acme/packs/")[1]
        if tail in ("git/ref/tags/v1", "git/ref/heads/main"):
            return b"{}"
        if tail.startswith("git/ref/"):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        if tail in ("commits/v1", "commits/main", "commits/HEAD"):
            return self.SHA.encode()
        if tail == f"tarball/{self.SHA}":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                for name, text in self.files.items():
                    data = text.encode()
                    info = tarfile.TarInfo(f"acme-packs-{self.SHA[:7]}/{name}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
            return buf.getvalue()
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)


def repo_files(pack_id="acme/demo", namespace="acme", **over) -> dict[str, str]:
    data = make_pack(pack_id, **over)
    manifest = {
        "format": 1,
        "namespace": namespace,
        "name": "Acme packs",
        "license": "Apache-2.0",
        "packs": [
            {
                "id": pack_id,
                "name": "Demo",
                "description": "A demo pack.",
                "tags": ["demo"],
                "media": ["text"],
                "parameters": ["brand"],
                "readme": "packs/demo/README.md",
                "versions": [{"version": data["version"], "file": "packs/demo/0.1.0.json", "model": data["model"], "status": "available"}],
            }
        ],
    }
    return {"prompt-packs.json": json.dumps(manifest), "packs/demo/0.1.0.json": json.dumps(data), "packs/demo/README.md": "# demo"}


def refs_file(cache: str) -> dict:
    with open(os.path.join(cache, "acme", "packs", "refs.json")) as f:
        return json.load(f)


def test_fetch_from_github_caches_by_commit_and_warns_on_a_branch(tmp_path, caplog):
    gh = FakeGitHub(repo_files())
    api = sources.GitHub(gh, token="tok")
    cache = str(tmp_path / "cache")
    folder, label = sources.fetch(sources.parse("github:acme/packs@v1"), cache, github=api)
    assert label == "acme/packs@v1" and folder == os.path.join(cache, "acme", "packs", FakeGitHub.SHA)
    assert os.path.exists(os.path.join(folder, "prompt-packs.json")) and gh.headers[0]["Authorization"] == "Bearer tok"
    assert [c.split("/repos/acme/packs/")[1] for c in gh.calls] == ["git/ref/tags/v1", "commits/v1", f"tarball/{FakeGitHub.SHA}"]
    assert refs_file(cache) == {"v1": FakeGitHub.SHA}
    gh.calls.clear()
    sources.fetch(sources.parse("github:acme/packs@v1"), cache, github=api)  # the tag is cached: no network
    assert gh.calls == []
    sources.fetch(sources.parse("github:acme/packs@v1"), cache, refresh=True, github=api)  # refresh resolves again, the tarball is there
    assert [c.split("/repos/acme/packs/")[1] for c in gh.calls] == ["git/ref/tags/v1", "commits/v1"]
    gh.calls.clear()
    with caplog.at_level("WARNING", logger="spinf_serve.packs"):
        _, label = sources.fetch(sources.parse("github:acme/packs@main"), cache, github=api)
    assert label == "acme/packs@main" and "not reproducible" in caplog.text
    assert [c.split("/repos/acme/packs/")[1] for c in gh.calls] == ["git/ref/tags/main", "git/ref/heads/main", "commits/main"]
    assert "main" not in refs_file(cache)
    gh.calls.clear()
    _, label = sources.fetch(sources.parse(f"github:acme/packs@{FakeGitHub.SHA}#packs"), cache, github=api)
    assert gh.calls == [] and label == f"acme/packs@{FakeGitHub.SHA}"
    with pytest.raises(sources.SourceError, match="no such tag, branch or commit"):
        sources.fetch(sources.parse("github:acme/packs@v9"), cache, github=api)
    with pytest.raises(sources.SourceError, match="no folder 'nope'"):
        sources.fetch(sources.parse("github:acme/packs@v1#nope"), cache, github=api)
    with pytest.raises(sources.SourceError, match="not a folder"):
        sources.fetch(sources.parse(str(tmp_path / "missing")))
    with caplog.at_level("WARNING", logger="spinf_serve.packs"):
        sources.fetch(sources.parse("github:acme/packs"), cache, github=api)  # no ref: the default branch
    assert "the default branch" in caplog.text


# ---------------------------------------------------------------- the catalog
def write_repo(folder, files: dict[str, str]):
    for name, text in files.items():
        p = folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return str(folder)


def test_scan_through_the_manifest_and_by_discovery(tmp_path):
    repo = write_repo(tmp_path / "repo", repo_files())
    s = catalog.scan(repo, "acme/packs@v1")
    assert s.ok and s.manifest.namespace == "acme" and [p.id for p in s.packs] == ["acme/demo"] and set(s.listed) == {"acme/demo"}
    p = s.packs[0]
    assert (p.source, p.status, p.name, p.description, p.tags, p.listing) == (
        "acme/packs@v1",
        "available",
        "Demo",
        "A demo pack.",
        ("demo",),
        {"readme": "packs/demo/README.md"},
    )
    # discovery: a folder of files, fixtures and samples left alone, problems collected
    bare = tmp_path / "bare"
    write_repo(
        bare,
        {
            "a.json": json.dumps(make_pack()),
            "b.json": json.dumps(make_pack(version="0.2.0")),
            "fixtures/x.json": json.dumps({"pack": "acme/demo", "cases": []}),
            "samples.json": "[1]",
            "bad.json": json.dumps({**make_pack(), "pack": "BAD"}),
            "junk.json": "{",
        },
    )
    s = catalog.scan(str(bare))
    assert s.manifest is None and sorted(p.version for p in s.packs) == ["0.1.0", "0.2.0"] and s.listed == {}
    assert len(s.problems) == 1 and "bad.json" in s.problems[0]
    # manifest problems
    files = repo_files()
    m = json.loads(files["prompt-packs.json"])
    m["packs"][0]["versions"].append(
        {"version": "0.2.0", "file": "packs/demo/0.2.0.json", "model": MODEL, "fixture": "packs/demo/fixtures/0.2.0.json"}
    )
    m["packs"][0]["versions"].append({"version": "0.3.0", "file": "packs/demo/0.3.0.json", "model": MODEL, "status": "coming_soon"})
    m["packs"][0]["samples"] = "packs/demo/samples/"
    m["packs"].append({"id": "other/x", "name": "x", "versions": [{"version": "1", "file": "x.json", "model": MODEL}]})
    files["prompt-packs.json"] = json.dumps(m)
    files["packs/demo/0.2.0.json"] = json.dumps(make_pack(version="0.2.0", model="spinf/gemma-4-31b-fp8"))
    s = catalog.scan(write_repo(tmp_path / "repo2", files))
    assert [p.version for p in s.packs] == ["0.1.0"]
    assert sorted(s.problems) == sorted(
        [
            "prompt-packs.json: acme/demo 0.2.0 (spinf/gemma-4-12b-fp8): the file says acme/demo 0.2.0 (spinf/gemma-4-31b-fp8)",
            "prompt-packs.json: acme/demo 0.2.0: fixture packs/demo/fixtures/0.2.0.json is missing",
            "prompt-packs.json: acme/demo: packs/demo/samples/ is missing",
            "prompt-packs.json: other/x is not in the namespace 'acme'",
        ]
    )
    (tmp_path / "repo2" / "prompt-packs.json").write_text(json.dumps({"format": 2, "namespace": "acme", "packs": []}))
    s = catalog.scan(str(tmp_path / "repo2"))
    assert not s.ok and "format" in s.problems[0] and s.packs == []
    with pytest.raises(PackError):
        catalog.read_manifest(str(tmp_path / "repo2"))
    assert catalog.read_manifest(str(bare)) is None


def test_the_catalog_rules(tmp_path):
    acme = write_repo(tmp_path / "acme", repo_files())
    other = write_repo(tmp_path / "other", repo_files(model="spinf/gemma-4-31b-fp8"))
    spinf = write_repo(tmp_path / "spinf", repo_files("spinf/demo", "spinf"))
    c = catalog.build([acme, other], MODEL, official=False)
    assert c.errors == [] and c.versions("acme/demo") == ["0.1.0"] and c.get("acme/demo").source == f"local:{acme}"
    assert [(s.id, s.source, s.reason) for s in c.skipped] == [
        ("acme/demo", f"local:{other}", f"written for spinf/gemma-4-31b-fp8, this server serves {MODEL}")
    ]
    assert c.sources == [f"local:{acme}", f"local:{other}"]
    # no model: every version, and the same id from two sources is an error
    c = catalog.build([acme, other], None, official=False)
    assert len(c.errors) == 1 and "comes from both" in c.errors[0] and c.versions("acme/demo") == ["0.1.0"]
    # the spinf/ namespace from a non-official source is skipped unless allowed
    c = catalog.build([spinf], MODEL, official=False)
    assert c.packs == {} and "reserved for the official source" in c.skipped[0].reason
    c = catalog.build([spinf], MODEL, official=False, allow_namespaces=("spinf",))
    assert c.versions("spinf/demo") == ["0.1.0"]
    # the official source (overridden to a local folder) may carry spinf/
    os.environ["SPINF_PACKS_OFFICIAL"] = spinf
    try:
        c = catalog.build([acme], MODEL)
        assert sorted(c.packs) == ["acme/demo", "spinf/demo"] and c.sources[0] == f"local:{spinf}"
        assert catalog.build([spinf], MODEL).sources == [f"local:{spinf}"]  # the same source once
    finally:
        del os.environ["SPINF_PACKS_OFFICIAL"]
    # a source that cannot be read is an error, the others load
    c = catalog.build([str(tmp_path / "missing"), acme], MODEL, official=False)
    assert len(c.errors) == 1 and "not a folder" in c.errors[0] and c.versions("acme/demo") == ["0.1.0"]
    # the official source unavailable (offline, the tag not published): not an error, the server starts without it
    os.environ["SPINF_PACKS_OFFICIAL"] = str(tmp_path / "missing")
    try:
        c2 = catalog.build([acme], MODEL)
        assert c2.errors == [] and "not a folder" in c2.official_error and c2.versions("acme/demo") == ["0.1.0"]
        from spinf_serve.server.app import build_catalog
        from spinf_serve.server.config import ServerConfig

        cfg = ServerConfig.build({"model": MODEL, "packs": acme}, env={})
        assert build_catalog(cfg, MODEL).official_error == c2.official_error
    finally:
        del os.environ["SPINF_PACKS_OFFICIAL"]
    # aliases: the served name and the pack's name may differ in case or be an alias of one manifest
    assert catalog.same_model("SPINF/Gemma-4-12B-FP8", MODEL) and not catalog.same_model("spinf/gemma-4-31b-fp8", MODEL)
    assert catalog.same_model("google/gemma-4-12B", MODEL) and not catalog.same_model("google/gemma-4-31B", MODEL)  # an alias
    # the public listing: enabled packs with their versions, coming-soon ids from the manifest
    pub = c.public()
    assert [p["id"] for p in pub] == ["acme/demo"] and pub[0]["source"] == f"local:{acme}" and pub[0]["status"] == "available"
    c.listed["acme/soon"] = catalog.RepoPack(
        id="acme/soon",
        name="Soon",
        description="later",
        versions=[catalog.RepoVersion(version="1", file="x", model=MODEL, status="coming_soon")],
    )
    assert c.public()[1] == {
        "id": "acme/soon",
        "name": "Soon",
        "status": "coming_soon",
        "version": None,
        "versions": [],
        "description": "later",
        "tags": [],
        "model": MODEL,
        "media": ["text"],
    }
    # a deprecated default is skipped over when an available version exists
    d = catalog.Catalog(MODEL)
    old, new = packs.from_data(make_pack(), status="deprecated", source="s"), packs.from_data(make_pack(version="0.2.0"), source="s")
    entry = catalog.RepoPack(id="acme/demo", name="Demo", versions=[catalog.RepoVersion(version="0.1.0", file="a", model=MODEL)])
    d.add(catalog.Scan("s", "", catalog.RepoManifest(format=1, namespace="acme", packs=[entry]), [old, new], {"acme/demo": entry}))
    assert d.versions("acme/demo") == ["0.1.0", "0.2.0"]  # the manifest's order is kept
    assert d.get("acme/demo").version == "0.2.0"  # the default is the first available version
    assert d.get("acme/demo", "0.1.0").status == "deprecated"
    with pytest.raises(KeyError):
        d.get("acme/none")


def test_the_commands(tmp_path, capsys):
    from spinf_serve.packs.__main__ import main

    repo = write_repo(tmp_path / "repo", repo_files())
    assert main(["validate", repo]) == 0
    out = capsys.readouterr().out
    assert "namespace acme" in out and "acme/demo  0.1.0" in out and "ok: 1 pack version(s)" in out
    assert main(["list", "--no-official", "--packs", repo, "--model", "spinf/gemma-4-31b-fp8"]) == 0
    out = capsys.readouterr().out
    assert "no pack enabled for spinf/gemma-4-31b-fp8" in out and "skipped:" in out
    assert main(["list", "--no-official", "--packs", repo, "--packs", str(tmp_path / "nope")]) == 1
    assert main(["list", "--no-official", "--packs", repo]) == 0
    assert "0.1.0 (default)" in capsys.readouterr().out
    (tmp_path / "repo" / "packs" / "demo" / "README.md").unlink()
    assert main(["validate", repo]) == 1
    assert "README.md is missing" in capsys.readouterr().err
    assert main(["list", "--no-official", "--packs", repo]) == 1  # a source with problems is an error for a server too


def test_the_module_imports_without_torch():
    """the hosted API and spinf-client import spinf_serve.packs on machines without a GPU stack"""
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['numpy'] = None; "
        "import spinf_serve.packs, spinf_serve.packs.refs, spinf_serve.packs.catalog; print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=os.path.dirname(os.path.dirname(__file__))
    )
    assert out.stdout.strip() == "ok"
