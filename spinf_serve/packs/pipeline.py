"""A pack's answers -> reads -> category scores -> decision (standard library only, no GPU). The one evaluator: the hosted
API, spinf-client and the pack workbench import this module, so every evaluator gives the same answer, and the golden fixtures of
spinf-prompts (packs/<name>/fixtures/<version>.json) pin it within float rounding. Format: spinf-prompts docs/pack-format.md.

Per content:
  answers     the option probabilities per (query id, combination value F), renormalised over the query's options; from the
              result's `p` or, when the pack says "answers": "logprob", from the full-precision logprobs
  reads       pair: sign * logit(p[0]); option: ln p[j] - mu; any: logsumexp(a[1:]) - a[0];
              set: logsumexp(a[unsafe]) - logsumexp(a[rest]), with a = ln p - mu; then z = (value - mean) / sd
  categories  macro = the mean z of the category's reads ("views": the mean of the per-view means);
              score = sigmoid(a * macro + b); percentile = the score in the normal-traffic quantiles; flagged = score >= threshold
              (threshold null = reported only); a category with `requires` is flagged only while the categories it lists are
  decision    max: the highest score of the deciding categories >= one threshold; per_category: any deciding category flagged;
              micro_layer: w . x + b >= threshold over category macros ("c:") and read z's ("r:"), only while the lock matches,
              else the fallback mode; a deciding category has a threshold and not `decides: false`; then, in every mode, a
              flagged category listed in decision.critical forces unsafe (not part of the lock: editable like the thresholds)
"""

import base64
import bisect
import hashlib
import json
import math
import struct
from collections.abc import Callable
from typing import Any

CLIP = 1e-9
PAIR_CLIP = 1e-6
MODES = ("max", "per_category", "micro_layer")
LOCK_WARNING = "micro-layer lock mismatch: the pack was edited; using the fallback mode"

Answers = dict[tuple[str, str | None], list[float]]
Layer = tuple[Callable[[list[float]], float], float]


def answers_from_result(result: dict[str, Any] | list[dict[str, Any]], query_ids: set[str] | None = None, source: str = "p") -> Answers:
    """One input's scoring result (or its `queries` list) -> {(query id, F value or None): [p per option]} for the pack's queries
    (the caller's own queries can share the result). source = the pack's "answers": "p" (the API's p, rounded to 6 decimals, so a
    rare option's p sits on a 1e-6 grid) or "logprob" (exp(logprob - logsumexp over the query's options), full precision)."""
    out: Answers = {}
    for q in result["queries"] if isinstance(result, dict) else result:
        if query_ids is not None and q["id"] not in query_ids:
            continue
        for c in q["combinations"]:
            if source == "logprob":
                lp = [o["logprob"] for o in c["options"]]
                lse = _logsumexp(lp)
                ps = [math.exp(x - lse) for x in lp]
            else:
                ps = [o["p"] for o in c["options"]]
            out[(q["id"], (c.get("values") or {}).get("F"))] = ps
    return out


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def read_value(read: dict[str, Any], answers: Answers) -> float:
    p = [max(x, CLIP) for x in answers[(read["query"], read.get("F"))]]
    kind = read["kind"]
    if kind == "pair":
        pa = min(max(p[0] / (p[0] + p[1]), PAIR_CLIP), 1 - PAIR_CLIP)
        return read.get("sign", 1) * math.log(pa / (1 - pa))
    total = sum(p)
    logs = [math.log(x / total) for x in p]
    if kind == "option":
        return logs[read["option"]] - read["mu"]
    a = [x - mu for x, mu in zip(logs, read["mu"], strict=True)]
    if kind == "any":
        return _logsumexp(a[1:]) - a[0]
    if kind == "set":
        unsafe = set(read["unsafe"])
        return _logsumexp([a[i] for i in sorted(unsafe)]) - _logsumexp([x for i, x in enumerate(a) if i not in unsafe])
    raise ValueError(f"unknown read kind {kind!r}")


def read_z(pack: dict[str, Any], answers: Answers) -> dict[str, float]:
    return {r["id"]: (read_value(r, answers) - r["mean"]) / r["sd"] for r in pack["reads"]}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def category_macros(pack: dict[str, Any], z: dict[str, float]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    views: dict[str, dict[str, list[float]]] = {}
    for r in pack["reads"]:
        by.setdefault(r["category"], []).append(z[r["id"]])
        if r.get("view"):
            views.setdefault(r["category"], {}).setdefault(r["view"], []).append(z[r["id"]])
    return {
        c: _mean([_mean(v) for v in views[c].values()]) if spec.get("aggregate") == "views" else _mean(by[c])
        for c, spec in pack["categories"].items()
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def interp(x: float, xp: list[float], fp: list[float]) -> float:
    """numpy.interp: piecewise linear, clamped at both ends (xp increasing)."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = bisect.bisect_right(xp, x) - 1
    return fp[i] + (x - xp[i]) * (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])


def percentile(spec: dict[str, Any], score: float) -> float | None:
    q = spec.get("normal_quantiles")
    return interp(score, q["values"], q["levels"]) if q else None


def lock_of(pack: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of what the micro-layer depends on (queries, reads, Platt maps, inputs)."""
    ml = pack["decision"]["micro_layer"]
    obj = {
        "queries": pack["scoring"]["queries"],
        "reads": pack["reads"],
        "platt": {c: s["platt"] for c, s in pack["categories"].items()},
        "inputs": ml["inputs"],
    }
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def micro_layer(pack: dict[str, Any]) -> Layer:
    """The decoded layer: (x -> value, threshold). Weights (float32, base64): linear [w..., b, threshold]; with hidden units
    [W1 (inputs x hidden), b1, w2, b2, threshold], tanh."""
    ml = pack["decision"]["micro_layer"]
    if ml.get("dtype", "float32") != "float32":
        raise ValueError(f"micro-layer dtype {ml['dtype']!r}: only float32")
    raw = base64.b64decode(ml["weights"])
    w = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    n, h = len(ml["inputs"]), ml.get("hidden", 0)
    if len(w) != micro_layer_size(n, h):
        raise ValueError(f"micro-layer weights: {len(w)} floats for {n} inputs and {h} hidden units ({micro_layer_size(n, h)} expected)")
    if h:
        w1 = [w[i * h : (i + 1) * h] for i in range(n)]
        b1, w2 = w[n * h : n * h + h], w[n * h + h : n * h + 2 * h]
        b2, t = w[n * h + 2 * h], w[n * h + 2 * h + 1]

        def hidden(x: list[float]) -> float:
            units = [math.tanh(sum(x[i] * w1[i][k] for i in range(n)) + b1[k]) for k in range(h)]
            return sum(u * v for u, v in zip(units, w2, strict=True)) + b2

        return hidden, t
    ww, b, t = w[:n], w[n], w[n + 1]
    return (lambda x: sum(a * c for a, c in zip(x, ww, strict=True)) + b), t


def micro_layer_size(n_inputs: int, hidden: int) -> int:
    """how many float32 the weights of a layer hold, the threshold included"""
    return n_inputs * hidden + 2 * hidden + 2 if hidden else n_inputs + 2


def deciding(pack: dict[str, Any], limits: dict[str, float | None]) -> list[str]:
    """the categories that can decide: a threshold (the pack's or the call's) and not `decides: false`"""
    return [c for c, spec in pack["categories"].items() if limits[c] is not None and spec.get("decides", True)]


def evaluate(
    pack: dict[str, Any],
    answers: Answers,
    mode: str | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float | None] | None = None,
    critical: list[str] | None = None,
    lock_ok: bool | None = None,
    layer: Layer | None = None,
) -> dict[str, Any]:
    """One content's answers -> {"categories", "decision", "macros"}.

    mode: the decision mode (default: the pack's). threshold: the one threshold of max or micro_layer (default: the pack's).
    thresholds: per-category overrides (None = reported only). critical: the categories that force unsafe when flagged (default:
    the pack's decision.critical; [] = none). lock_ok / layer: precomputed by a loaded Pack, else computed here."""
    z = read_z(pack, answers)
    macros = category_macros(pack, z)
    limits = {c: spec.get("threshold") for c, spec in pack["categories"].items()}
    for c, t in (thresholds or {}).items():
        if c not in limits:
            raise KeyError(f"no category {c!r}")
        limits[c] = t
    cats: dict[str, dict[str, Any]] = {}
    for c, spec in pack["categories"].items():
        a, b = spec["platt"]
        s = _sigmoid(a * macros[c] + b)
        t = limits[c]
        cats[c] = {"score": s, "percentile": percentile(spec, s), "flagged": None if t is None else s >= t}
    for c, spec in pack["categories"].items():  # requires: flagged only while the required categories are
        if cats[c]["flagged"] and spec.get("requires"):
            cats[c]["flagged"] = all(cats[r]["flagged"] for r in spec["requires"])

    dec = pack["decision"]
    mode = mode or dec["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    used = mode
    if mode == "micro_layer":
        ml = dec.get("micro_layer")
        if lock_ok is None:
            lock_ok = bool(ml) and ml.get("lock") == lock_of(pack)
        if not lock_ok:
            used = dec.get("fallback", "per_category")
    decide = deciding(pack, limits)
    out: dict[str, Any]
    if used == "micro_layer":
        f, t = layer or micro_layer(pack)
        t = t if threshold is None else threshold
        x = [macros[i[2:]] if i.startswith("c:") else z[i[2:]] for i in dec["micro_layer"]["inputs"]]
        v = f(x)
        out = {"unsafe": v >= t, "mode": used, "score": v, "threshold": t}
    elif used == "max":
        t = dec["max"]["threshold"] if threshold is None or used != mode else threshold
        top = max(decide, key=lambda c: cats[c]["score"]) if decide else None
        v = cats[top]["score"] if top else 0.0
        out = {"unsafe": top is not None and v >= t, "mode": used, "score": v, "threshold": t, "category": top}
    else:
        hit = [c for c in decide if cats[c]["flagged"]]
        out = {"unsafe": bool(hit), "mode": "per_category", "categories": hit}
    crit = [c for c in (dec.get("critical", []) if critical is None else critical) if cats.get(c, {}).get("flagged")]
    if crit:
        out["critical"] = crit
        if not out["unsafe"]:
            out["unsafe"] = True
            out["by"] = "critical"
    if used != mode:
        out["warning"] = LOCK_WARNING
    return {"categories": cats, "decision": out, "macros": macros}
