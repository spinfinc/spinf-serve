"""The conformance suite (plan §3.3): the same request fixtures sent to a spinf-serve server and to the hosted API, the responses
compared minus the hosted-only fields. The test that the "change the base URL and the token" promise holds; run before a release
of either side. Not collected by pytest (it needs two live endpoints):

    uv run python tests/conformance.py --a http://localhost:8010 --a-token T --a-model spinf/gemma-4-12b-fp8 \\
                                       --b https://api.spinf.com --b-token $SPINF_API_KEY --b-model spinf-12b [--values]

Fixtures: tests/conformance/*.json, each {"request": <body with model "$MODEL">, "expect": {"status": 200 | 400 ..., "code": ...}},
plus an optional "expect_b" for a known difference of side B (the hosted API catching up with a new code). Shapes are compared
always (keys, lengths, ids, option texts, error codes and params); `usage` on the chat/completions fields only (the detailed blocks
are additive); the numbers (`p`, `logprob`, pack scores) only with --values, when both sides report the same system_fingerprint
(otherwise the hardware or the engine build differ), or with --force-values regardless.
"""

import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request

HOSTED_ONLY = {"billed_tokens", "media", "extraction"}  # under usage: the hosted billing fields (extraction = its billed slots)
IGNORED = {"id", "created", "request_id", "timing", "engine", "system_fingerprint", "endpoint", "usage"}
NUMERIC = {"p", "logprob", "floor_p", "residual", "floor_residual", "score", "percentile", "mass", "p_negative", "p_calibrated"}


def call(base: str, token: str, path: str, body: dict | None) -> tuple[int, dict]:
    req = urllib.request.Request(base.rstrip("/") + path, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def shape(x, values: bool):
    """a response with its ignorable fields removed and (without `values`) its numbers replaced by their type"""
    if isinstance(x, dict):
        return {k: shape(v, values) for k, v in x.items() if k not in IGNORED}
    if isinstance(x, list):
        return [shape(v, values) for v in x]
    if isinstance(x, float) and not values:
        return "number"
    return x


def diff(a, b, path="") -> list[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only on B")
            elif k not in b:
                out.append(f"{path}.{k}: only on A")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} vs {len(b)} items")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        if isinstance(a, float) and isinstance(b, float) and abs(a - b) <= 2e-6 * max(1.0, abs(a), abs(b)):
            return out
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def strip_hosted(resp: dict) -> dict:
    """the hosted API's own fields: billing in usage, the hosted-only usage blocks"""
    out = dict(resp)
    if isinstance(out.get("usage"), dict):
        u = {k: v for k, v in out["usage"].items() if k not in HOSTED_ONLY}
        if isinstance(u.get("prompt_tokens_details"), dict):
            u["prompt_tokens_details"] = {k: v for k, v in u["prompt_tokens_details"].items() if k not in HOSTED_ONLY}
        out["usage"] = u
    return out


def compare_usage(a: dict, b: dict) -> list[str]:
    """the token counts both sides must agree on"""
    out = []
    ua, ub = a.get("usage", {}), b.get("usage", {})
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if ua.get(k) != ub.get(k):
            out.append(f"usage.{k}: {ua.get(k)} vs {ub.get(k)}")
    da, db = ua.get("prompt_tokens_details", {}), ub.get("prompt_tokens_details", {})
    for k in ("content_tokens", "query_tokens", "floor_tokens"):
        if da.get(k) != db.get(k):
            out.append(f"usage.prompt_tokens_details.{k}: {da.get(k)} vs {db.get(k)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="base URL of side A (a spinf-serve server)")
    ap.add_argument("--a-token", default="")
    ap.add_argument("--a-model", required=True)
    ap.add_argument("--b", required=True, help="base URL of side B (the hosted API, or another server)")
    ap.add_argument("--b-token", default="")
    ap.add_argument("--b-model", required=True)
    ap.add_argument("--fixtures", default=os.path.join(os.path.dirname(__file__), "conformance"))
    ap.add_argument("--only", default=None, help="a fixture name")
    ap.add_argument("--values", action="store_true", help="compare the numbers too (same system_fingerprint expected)")
    ap.add_argument("--force-values", action="store_true", help="compare the numbers whatever the fingerprints")
    ap.add_argument("--path", default="/v1/score")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.fixtures, "*.json")))
    failed = 0
    for f in files:
        name = os.path.basename(f)[:-5]
        if a.only and a.only != name:
            continue
        with open(f) as fh:
            fx = json.load(fh)
        body = json.loads(json.dumps(fx["request"]))
        path = fx.get("path", a.path)
        ra = call(a.a, a.a_token, path, json.loads(json.dumps(body).replace("$MODEL", a.a_model)) if body is not None else None)
        rb = call(a.b, a.b_token, path, json.loads(json.dumps(body).replace("$MODEL", a.b_model)) if body is not None else None)
        problems = []
        exp = fx.get("expect", {})
        exp_b = {**exp, **fx.get("expect_b", {})}
        for side, (st, resp), e in (("A", ra, exp), ("B", rb, exp_b)):
            if e.get("status") and st != e["status"]:
                problems.append(f"{side}: status {st}, expected {e['status']}: {json.dumps(resp)[:200]}")
            if e.get("code") and (resp.get("error") or {}).get("code") != e["code"]:
                problems.append(f"{side}: error code {(resp.get('error') or {}).get('code')!r}, expected {e['code']!r}")
        if ra[0] != rb[0]:
            problems.append(f"status {ra[0]} vs {rb[0]}")
        if ra[0] == 200 and rb[0] == 200:
            sa, sb = strip_hosted(ra[1]), strip_hosted(rb[1])
            same_fp = ra[1].get("system_fingerprint") == rb[1].get("system_fingerprint")
            values = a.force_values or (a.values and same_fp)
            sa["model"], sb["model"] = "$MODEL", "$MODEL"
            problems += diff(shape(sa, values), shape(sb, values))
            problems += compare_usage(sa, sb)
            if a.values and not same_fp and not a.force_values:
                fa, fb = ra[1].get("system_fingerprint"), rb[1].get("system_fingerprint")
                problems.append(f"fingerprints differ ({fa} vs {fb}): numbers not compared")
        elif ra[0] >= 400 and rb[0] >= 400 and "expect_b" not in fx:
            ea, eb = ra[1].get("error") or {}, rb[1].get("error") or {}
            if ea.get("code") != eb.get("code"):
                problems.append(f"error code {ea.get('code')!r} vs {eb.get('code')!r}")
            if ea.get("param") != eb.get("param"):
                problems.append(f"error param {ea.get('param')!r} vs {eb.get('param')!r}")
        status = "ok" if not problems else "DIFF"
        print(f"{name:32s} {ra[0]} / {rb[0]}  {status}")
        for p in problems:
            print("    " + p)
        failed += bool(problems)
    print(f"\n{len(files) - failed}/{len(files)} fixtures agree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
