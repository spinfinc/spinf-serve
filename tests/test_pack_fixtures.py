"""The official packs against their golden fixtures: every version listed in spinf-prompts' manifest that has a fixture (the
shipped samples scored on a server, the raw results and the expected categories and decision per mode) is reproduced by this
evaluator within the fixture's rounding. The repository is a sibling checkout (SPINF_PROMPTS_DIR, default ../spinf-prompts next
to this repository); without it these tests are skipped."""

import json
import os

import pytest

from spinf_serve import packs
from spinf_serve.packs import catalog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.environ.get("SPINF_PROMPTS_DIR") or os.path.join(os.path.dirname(ROOT), "spinf-prompts")
TOL = 2e-6  # the fixtures are rounded to 6 decimals

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(PROMPTS, catalog.MANIFEST)), reason=f"no spinf-prompts checkout at {PROMPTS}"
)


def versions_with_fixtures():
    m = catalog.read_manifest(PROMPTS)
    return [(e.id, v.version, v.file, v.fixture) for e in m.packs for v in e.versions if v.fixture]


def test_the_official_repository_validates():
    s = catalog.scan(PROMPTS, "spinfinc/spinf-prompts@local")
    assert s.ok, s.problems
    assert s.manifest.namespace == "spinf" and len(s.packs) >= 9
    for p in s.packs:
        assert p.layer is None or p.lock_ok, p.id
    assert {p.model for p in s.packs} <= {"spinf/gemma-4-12b-fp8", "spinf/qwen3.8-27b-int4", "spinf/pplx-decider-v1-27b-int4"}


@pytest.mark.parametrize(
    "pack_id,version,file,fixture", versions_with_fixtures() if os.path.exists(os.path.join(PROMPTS, catalog.MANIFEST)) else []
)
def test_the_golden_fixture_is_reproduced(pack_id, version, file, fixture):
    pack = packs.load(os.path.join(PROMPTS, file))
    assert (pack.id, pack.version) == (pack_id, version)
    with open(os.path.join(PROMPTS, fixture)) as f:
        fx = json.load(f)
    assert (fx["pack"], fx["version"]) == (pack_id, version) and fx["cases"]
    for case in fx["cases"]:
        answers = pack.answers(case["result"])
        assert len(answers) == pack.prompts
        for mode, want in case["expected"].items():
            got = pack.evaluate(answers, mode, context=case.get("context"))
            assert got["categories"].keys() == want["categories"].keys()
            for c, w in want["categories"].items():
                g = got["categories"][c]
                assert g["flagged"] == w["flagged"], (case.get("id"), mode, c)
                assert abs(g["score"] - w["score"]) < TOL, (case.get("id"), mode, c)
                assert (g["percentile"] is None) == (w["percentile"] is None) and (
                    g["percentile"] is None or abs(g["percentile"] - w["percentile"]) < TOL
                )
            for k, w in want["decision"].items():
                g = got["decision"][k]
                assert (abs(g - w) < TOL) if isinstance(w, float) else g == w, (case.get("id"), mode, k)
        # the fixture's context and parameters fill the pack's context
        filled = pack.fill(case["text"], case.get("context"), case.get("params"))
        assert case["text"] in filled
        if pack.parameters:
            assert case.get("params"), "a pack with parameters records them in its fixture"


def test_the_catalog_of_a_12b_server_from_the_official_folder(monkeypatch):
    monkeypatch.setenv("SPINF_PACKS_OFFICIAL", PROMPTS)
    c = catalog.build([], "spinf/gemma-4-12b-fp8")
    assert c.errors == [] and c.sources == [f"local:{PROMPTS}"]
    assert c.get("spinf/moderation").version == "0.1.3" and c.versions("spinf/moderation") == ["0.1.3", "0.1.1"]
    assert {"spinf/triage", "spinf/entertainment", "spinf/products", "spinf/reviews", "spinf/brand"} <= set(c.packs)
    assert {s.id for s in c.skipped} == {"spinf/moderation-qwen", "spinf/moderation-qwen-lite", "spinf/moderation-decider"}
    pub = {p["id"]: p for p in c.public()}
    assert pub["spinf/moderation"]["versions"] == ["0.1.3", "0.1.1"] and pub["spinf/moderation"]["decision"]["critical"] == [
        "selfharm",
        "child",
    ]
    assert pub["spinf/brand"]["parameters"][0]["id"] == "brand"
    q = catalog.build([], "spinf/qwen3.8-27b-int4")
    assert sorted(q.packs) == ["spinf/moderation-qwen", "spinf/moderation-qwen-lite"]
