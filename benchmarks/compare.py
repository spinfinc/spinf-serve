"""Throughput of many questions about one content: spinf-serve against an OpenAI-compatible completion server (vLLM with
prefix caching, the usual alternative), the same jobs over HTTP on the same GPU.

A job = one content + N questions, each a template read after the content with single-token answers to score. The two
servers get exactly the same text:

  spinf-serve   POST /v0/score, one request per pass: every prompt and its options, scored in packed passes
  completions   POST /v1/completions per (content, question): prompt = content + template, max_tokens 1, logprobs 20,
                `workers` requests in flight (prefix caching shares the content across its questions). The option
                log-probs are read from the top-20 by token id; an option outside the top 20 gets the 20th (a bound).

Jobs come from a JSONL file ({"id", "text", "questions": [{"template": "...{?}", "options": [" a", " b"]}]}) or, for
maintainers, the parity fixture. Two passes each; the second (warm) pass is the number. Prints the tokens of the work
(content once + the questions' own tokens: what both servers are asked to read) and what a per-question re-read would be.

    python benchmarks/compare.py --jobs news.jsonl --spinf http://localhost:8000 --vllm http://localhost:8033 --model google/gemma-4-12B
"""

import argparse
import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url: str, body: dict, timeout: float = 900, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_jobs(path: str) -> list[dict]:
    if path.endswith(".json.gz") or path.endswith(".json"):  # the parity fixture (maintainers)
        import gzip

        with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
            fx = json.load(f)
        return [
            {
                "id": j["id"],
                "text": j["text"],
                "questions": [
                    {
                        "template": c["template"].replace("{X}", "{content}").format(content="{content}", F=c["phrase"], CTX=""),
                        "options": [" " + c["pair"][0], " " + c["pair"][1]],
                    }
                    for c in j["cells"]
                ],
            }
            for j in fx["jobs"]
            if j["cells"]
        ]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prompts_of(job: dict) -> list[tuple[str, list[str]]]:
    """(prompt text up to the slot, options) per question; the template holds {content} (or none: it is appended)"""
    out = []
    for q in job["questions"]:
        t = q["template"].split("{?}")[0]
        p = t.replace("{content}", job["text"]) if "{content}" in t else job["text"] + t
        out.append((p, q["options"]))
    return out


# ---------------------------------------------------------------- spinf-serve
def run_spinf(url: str, jobs: list[dict], token: str | None) -> tuple[float, dict]:
    body = {"jobs": [{"id": j["id"], "prompts": [p for p, _ in prompts_of(j)], "options": [o for _, o in prompts_of(j)]} for j in jobs]}
    t0 = time.perf_counter()
    out = post(url.rstrip("/") + "/v0/score", body, token=token)
    wall = time.perf_counter() - t0
    scores = {(r["id"], k): lp for r in out["results"] for k, lp in enumerate(r["logprobs"])}
    return wall, scores


# ---------------------------------------------------------------- completions server
def run_completions(url: str, model: str, jobs: list[dict], workers: int, tok) -> tuple[float, dict]:
    reqs = []
    for j in jobs:
        for k, (p, opts) in enumerate(prompts_of(j)):
            ids = [tok(o, add_special_tokens=False)["input_ids"] for o in opts]
            if any(len(i) != 1 for i in ids):
                raise SystemExit(f"{j['id']}: multi-token option in {opts}: this comparison scores single-token answers")
            reqs.append(((j["id"], k), p, [i[0] for i in ids]))

    def one(r):
        key, p, oids = r
        out = post(
            url.rstrip("/") + "/v1/completions",
            {"model": model, "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": True},
        )
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        top = {int(k.split(":")[1]): v for k, v in top.items()}
        floor = min(top.values())
        return key, [top.get(i, floor) for i in oids]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        scores = dict(ex.map(one, reqs))
    return time.perf_counter() - t0, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--spinf", default=None, help="spinf-serve base URL")
    ap.add_argument("--spinf-token", default=None)
    ap.add_argument("--vllm", default=None, help="completions server base URL")
    ap.add_argument("--model", default="google/gemma-4-12B", help="the completions server's model name, and the tokenizer")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    jobs = load_jobs(a.jobs)
    n_q = sum(len(j["questions"]) for j in jobs)
    content_tokens = [len(tok(j["text"], add_special_tokens=False)["input_ids"]) for j in jobs]

    def tail(j, p):  # the question's own tokens: the prompt without the content
        return len(tok(p[len(j["text"]) :] if p.startswith(j["text"]) else p.replace(j["text"], ""), add_special_tokens=False)["input_ids"])

    tail_tokens = sum(tail(j, p) for j in jobs for p, _ in prompts_of(j))
    logical = sum(content_tokens) + tail_tokens
    reread = sum(c * len(j["questions"]) for c, j in zip(content_tokens, jobs, strict=True)) + tail_tokens
    mean_c = statistics.mean(content_tokens)
    print(f"jobs {len(jobs)}, questions {n_q} ({n_q / len(jobs):.0f} per content), content tokens mean {mean_c:.0f}")
    print(f"work: {logical:,} tokens (content once + questions); per-question re-read: {reread:,} tokens ({reread / logical:.0f}x)")
    results = {}
    if a.spinf:
        for i in range(a.passes):
            wall, scores = run_spinf(a.spinf, jobs, a.spinf_token)
            print(
                f"spinf-serve pass {i + 1}: {wall:.2f} s, {n_q / wall:,.0f} questions/s, {logical / wall:,.0f} tok/s of work, "
                f"{reread / wall:,.0f} tok/s re-read equivalent"
            )
        results["spinf"] = (wall, scores)
    if a.vllm:
        for i in range(a.passes):
            wall, scores = run_completions(a.vllm, a.model, jobs, a.workers, tok)
            print(
                f"completions pass {i + 1}: {wall:.2f} s, {n_q / wall:,.0f} questions/s, {logical / wall:,.0f} tok/s of work, "
                f"{reread / wall:,.0f} tok/s re-read equivalent ({a.workers} in flight)"
            )
        results["vllm"] = (wall, scores)
    if len(results) == 2:
        ws, ss = results["spinf"]
        wv, sv = results["vllm"]
        print(f"\nspinf-serve is {wv / ws:.1f}x faster on the same questions")
        # agreement of the scores: P(first option) over the two options, both servers
        d = []
        for key, lp in ss.items():
            a_, b_ = lp[0], lp[1]
            pa = 1 / (1 + math.exp(b_ - a_))
            va, vb = sv[key][0], sv[key][1]
            pv = 1 / (1 + math.exp(vb - va))
            d.append(abs(pa - pv))
        print(f"P(first answer) agreement: mean |diff| {statistics.mean(d):.4f}, max {max(d):.4f} over {len(d)} questions")


if __name__ == "__main__":
    main()
