"""Parity against a recorded reference: a fixture of articles x two-option cells, each cell a template read after the article
("{X}\\n\\nAs a result, {F} is likely to" + " rise" / " fall", ...), scored alone and compared cell by cell with a recorded run.

The public fixture is fixtures/wikinews_20261024.json.gz (24 Wikinews articles x 56 cells = 1,344 cells, fixtures/README.md); its
own lp_pos / lp_neg are the 12B's fp8s run, and one reference file per published checkpoint sits next to it. A reference is a run
of the fixture alone with the same engine and scheme, recorded with `--record`; `reference=` compares against it. The expected
result on the same card architecture and the pinned dependencies is bit-identical (lp_exact_share 1.0, docs/BATCH_INVARIANCE.md).
"""

import gzip
import json

import numpy as np

from spinf_serve.scorer import Job


def load_fixture(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def fixture_jobs(fx: dict) -> list[Job]:
    return [
        Job(
            j["id"],
            [c["template"].format(X=j["text"], F=c["phrase"], CTX="") for c in j["cells"]],
            [[" " + c["pair"][0], " " + c["pair"][1]] for c in j["cells"]],
        )
        for j in fx["jobs"]
        if j["cells"]
    ]


def fixture_reference(fx: dict) -> dict[str, list[list[float]]]:
    return {j["id"]: [[c["lp_pos"], c["lp_neg"]] for c in j["cells"]] for j in fx["jobs"] if j["cells"]}


def compare(ref: dict[str, list[list[float]]], got: dict[str, list[list[float]]]) -> dict[str, float]:
    """log-prob and raw (lp_pos - lp_neg) differences over every cell"""
    r = np.array([x for k in ref for x in ref[k]])
    g = np.array([x for k in ref for x in got[k]])
    d = np.abs(g - r)
    rr, gr = r[:, 0] - r[:, 1], g[:, 0] - g[:, 1]
    dr = np.abs(gr - rr)
    return {
        "n": len(r),
        "lp_max_abs": round(float(d.max()), 4),
        "lp_mean_abs": round(float(d.mean()), 5),
        "lp_exact_share": round(float((d == 0).mean()), 4),
        "raw_max_abs": round(float(dr.max()), 4),
        "raw_mean_abs": round(float(dr.mean()), 5),
        "raw_corr": round(float(np.corrcoef(rr, gr)[0, 1]), 6),
        "raw_sign_agree": round(float((np.sign(rr) == np.sign(gr)).mean()), 4),
    }


def run(scorer, fixture_path: str, reference_path: str | None = None, passes: int = 1, record: str | None = None) -> dict:
    """score the fixture `passes` times (the first pays the compile) -> timings + parity per pass; record: write the last pass's
    scores as a reference fixture (the fixture with lp_pos / lp_neg replaced)"""
    fx = load_fixture(fixture_path)
    ref = fixture_reference(load_fixture(reference_path)) if reference_path else fixture_reference(fx)
    jobs = fixture_jobs(fx)
    out = {"fixture": fx.get("day"), "reference": reference_path or fx.get("reference"), "passes": []}
    for _ in range(passes):
        st = scorer.stats
        tok0, busy0, wall0 = st.tokens, st.busy_s, st.wall_s
        res = scorer.score(jobs)
        wall = st.wall_s - wall0
        out["passes"].append(
            {
                "wall_s": round(wall, 2),
                "gpu_busy_s": round(st.busy_s - busy0, 2),
                "tokens": st.tokens - tok0,
                "tok_s": round((st.tokens - tok0) / wall),
                "parity": compare(ref, {k: v.logprobs for k, v in res.items()}),
            }
        )
    if record:
        rec = json.loads(json.dumps(fx))
        for j in rec["jobs"]:
            for c, (lp, ln) in zip(j["cells"], res[j["id"]].logprobs if j["cells"] else [], strict=True):
                c["lp_pos"], c["lp_neg"] = float(lp), float(ln)
        rec["reference"] = f"spinf-serve quant {scorer.quant}, lm head {scorer.lm_head}, aligned {scorer.cfg.align_jobs}"
        with gzip.open(record, "wt") as f:
            json.dump(rec, f)
        out["recorded"] = record
    return out
