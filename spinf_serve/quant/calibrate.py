"""Make an fp8s scale table: python -m spinf_serve.calibrate [--images DIR] [--audio DIR] [--headroom 2] [--out FILE]

fp8s quantizes each linear's input with a fixed scale (quant/linears.py Fp8StaticLinear) instead of the per-forward max over the packed
stream, so a job's scores do not depend on its neighbours (docs/BATCH_INVARIANCE.md). The scale of a linear must cover the
largest input it will ever see: an input past 448 x scale is clipped. This runs the engine with the dynamic fp8 scheme (eager, no
compile) over a broad corpus and records the largest |input| of every linear:
  --corpus                       a JSON(.gz) file of scoring requests covering what the model will see: text of every length
                                 (long documents set most maxima), images, audio, video; its composition is recorded in the table
  the parity fixture             the news articles and cells of the parity fixture
  --images / --audio             image files (jpg / png, read at high and low detail) and wav clips, when the model reads them
scale = 2^ceil(log2(headroom x amax / 448)): a power of two (exact in the kernel) at least `headroom` times the largest input seen.
The table goes to quant/tables.py scale_table_path (spinf_serve/quant/tables/<org>__<model>/<revision>.json), where fp8s loads it.

    CUDA_VISIBLE_DEVICES=1 python -m spinf_serve.quant.calibrate --corpus corpus.json.gz --images ./images --audio ./audio
"""

import argparse
import base64
import dataclasses
import datetime
import glob
import gzip
import hashlib
import json
import math
import os
import random
import time

from spinf_serve.config import EngineConfig


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


QUESTION = {
    "id": "q",
    "template": "\n\nQuestion: {F}?\nAnswer:{?}",
    "options": [" Yes", " No"],
    "combinations": [{"F": "Is this safe for children"}, {"F": "Is there a person in it"}, {"F": "Is it outdoors"}],
}


def media_requests(served: str, images: str | None, audio: str | None, n_img: int, n_audio: int, audio_ok: bool) -> list[dict]:
    rng = random.Random(7)
    reqs = []
    if images:
        files = sorted(glob.glob(os.path.join(images, "**", "*.*"), recursive=True))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = rng.sample(files, min(n_img, len(files)))
        for detail in ("high", "low"):
            inputs = []
            for k, f in enumerate(files):
                mime = "image/png" if f.lower().endswith(".png") else "image/jpeg"
                url = f"data:{mime};base64," + b64(f)
                parts = [{"type": "image_url", "image_url": {"url": url, "detail": detail}}]
                if k % 2:
                    parts.insert(0, {"type": "text", "text": "Describe what you see.\n"})
                inputs.append({"id": f"i{k}", "messages": [{"role": "user", "content": parts}]})
            for i in range(0, len(inputs), 8):
                reqs.append(
                    {"kind": f"image/{detail}", "body": {"model": served, "inputs": inputs[i : i + 8], "scoring": {"queries": [QUESTION]}}}
                )
    if audio and audio_ok:
        files = sorted(glob.glob(os.path.join(audio, "**", "*.wav"), recursive=True))
        files = rng.sample(files, min(n_audio, len(files)))
        inputs = [
            {
                "id": f"a{k}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Listen to this clip.\n"},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": b64(f), "format": "wav"},
                            },
                        ],
                    }
                ],
            }
            for k, f in enumerate(files)
        ]
        for i in range(0, len(inputs), 4):
            reqs.append({"kind": "audio", "body": {"model": served, "inputs": inputs[i : i + 4], "scoring": {"queries": [QUESTION]}}})
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser(prog="spinf_serve.calibrate")
    ap.add_argument("--corpus", required=True, help="scoring requests to calibrate on (JSON or JSON.gz)")
    ap.add_argument("--fixture", default="fixtures/wikinews_20261024.json.gz")
    ap.add_argument("--images", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--n-audio", type=int, default=8)
    ap.add_argument("--headroom", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="default: the table path fp8s reads for this model revision")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch

    from spinf_serve import parity, protocol
    from spinf_serve.quant import tables
    from spinf_serve.quant.fp8 import FP8_MAX
    from spinf_serve.service import ScoreService

    cfg = dataclasses.replace(EngineConfig.from_env(), quant="fp8", compile=False)
    svc = ScoreService(cfg, a.device)
    svc.load()
    sc = svc.scorer
    lins = {n: m for n, m in sc.model.named_modules() if type(m).__name__ == "Fp8Linear"}
    amax = {n: 0.0 for n in lins}
    where = {n: "" for n in lins}
    cur = {"kind": ""}

    def hook(name):
        def f(_m, args):
            v = args[0].abs().amax().item()
            if v > amax[name]:
                amax[name], where[name] = v, cur["kind"]

        return f

    for n, m in lins.items():
        m.register_forward_pre_hook(hook(n))

    with gzip.open(a.corpus, "rt") as f:
        corpus = json.load(f)
    reqs = corpus["requests"] + media_requests(
        cfg.served_model, a.images, a.audio, a.n_images, a.n_audio, sc.media_ok and getattr(sc.model._media, "audio_w", None) is not None
    )
    t0 = time.time()
    tokens0 = sc.stats.tokens
    counts: dict[str, int] = {}
    for r in reqs:
        cur["kind"] = r["kind"]
        body = {**r["body"], "model": cfg.served_model}
        body.setdefault("limits", {"max_content_tokens": cfg.max_tokens, "max_processed_tokens": 50_000_000})
        try:
            svc.score_request(protocol.ScoreRequest.model_validate(body))
        except protocol.Rejected:  # an input the engine refuses (empty, over max_tokens, a media kind the model has no
            # embedder for): score the request's inputs one by one, skipping only those
            kept = []
            for inp in body["inputs"]:
                try:
                    svc.score_request(protocol.ScoreRequest.model_validate({**body, "inputs": [inp]}))
                    kept.append(inp)
                except protocol.Rejected as e:
                    print(f"skipped a {r['kind']} input: {e.body()['code']}: {e.body()['message'][:120]}", flush=True)
            body["inputs"] = kept
        counts[r["kind"]] = counts.get(r["kind"], 0) + len(body["inputs"])
        print(f"{r['kind']:12s} done, {sc.stats.tokens - tokens0:,} tokens, {time.time() - t0:.0f} s", flush=True)
    cur["kind"] = "news"
    sc.score(parity.fixture_jobs(parity.load_fixture(a.fixture)))
    counts["news"] = 18
    tokens = sc.stats.tokens - tokens0

    def scale(v):
        return 2.0 ** math.ceil(math.log2(max(v, 1e-6) * a.headroom / FP8_MAX))

    with open(a.corpus, "rb") as f:
        corpus_sha = hashlib.sha256(f.read()).hexdigest()[:12]
    table = {
        "model_id": cfg.model_id,
        "revision": os.path.basename(sc.model_dir.rstrip("/")),
        "scheme": "fp8s",
        "headroom": a.headroom,
        "created": datetime.date.today().isoformat(),
        "calibrated_with": {
            "quant": "fp8",
            "gpu": torch.cuda.get_device_name(a.device),
            "torch": torch.__version__,
            "image": os.environ.get("GIT_SHA", ""),
        },
        "corpus": {
            "file": os.path.basename(a.corpus),
            "testbed": corpus.get("testbed"),
            "sha256_12": corpus_sha,
            "inputs": counts,
            "tokens": tokens,
            "images": bool(a.images and counts.get("image/high")),
            "audio": bool(counts.get("audio")),
        },
        "amax": {n: round(v, 4) for n, v in amax.items()},
        "amax_from": where,
        "scales": {n: scale(v) for n, v in amax.items()},
    }
    out = a.out or tables.scale_table_path(cfg.model_id, sc.model_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    by_kind: dict[str, int] = {}
    for k in where.values():
        by_kind[k] = by_kind.get(k, 0) + 1
    print(
        json.dumps(
            {
                "out": out,
                "linears": len(amax),
                "tokens": tokens,
                "inputs": counts,
                "max_from": by_kind,
                "amax_range": [min(amax.values()), max(amax.values())],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
