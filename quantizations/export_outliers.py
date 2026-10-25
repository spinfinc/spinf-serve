"""Export a Qwen-architecture int4 checkpoint's outlier table (spinf_outliers.json) for spinf-serve's int8 activation scheme.

Why a table: the i8 scheme (quant/i8gemm.py, models/qwen35/qwen.py) runs the three norm-fed linears of every layer
(in_proj_qkvz, qkv_proj, gate_up_proj) as a split GEMM. A fixed set of input channels, the ones whose activations carry the
model's outliers, is computed exactly from fp16; the other channels run int8 with a per-token scale. Fixing the set in
advance keeps the scheme fast and batch-invariant (LLM.int8()'s mixed-precision idea with a static channel set), and the set is
a property of the checkpoint revision, hence this file next to the weights. Without it the engine runs f16 activations.

How: the model loads in bf16 mode (GPTQ weights dequantized on the fly, eager, no compile) and runs one packed stream of
calibration jobs, the largest stream the packer builds from JOBS under --budget tokens. Forward pre-hooks on the linears
record each input channel's absmax over the stream's real tokens; the table keeps the top --n-out channels per linear.
The report prints how concentrated the outliers are (the share of tokens whose largest |x| falls in the top-n channels) as the
check that a fixed set suits the checkpoint.

    python quantizations/export_outliers.py MODEL_DIR JOBS.jsonl OUT_DIR [--n-out 64] [--budget 8192] [--device cuda:0]

JOBS.jsonl: one {"id", "prompts": [...], "options": [[...], ...]} per line (the engine's job shape; the maintainers use 300
RAGTruth items, about 8k tokens in one stream). OUT_DIR/spinf_outliers.json: {"n_out", "source", "model", "stream_tokens",
"checkpoint" (the spinf_quantize.json provenance when present), "outliers": {module name: sorted input channels}}.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from spinf_serve.config import EngineConfig  # noqa: E402
from spinf_serve.models.qwen35 import qwen  # noqa: E402
from spinf_serve.packer import Packer  # noqa: E402
from spinf_serve.scorer import Job, Scorer  # noqa: E402

KINDS = ("in_proj_qkvz", "qkv_proj", "gate_up_proj", "down_proj", "out_proj", "o_proj")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--model", default="", help="the model id recorded in the table (default: spinf/<directory name>)")
    ap.add_argument("--n-out", type=int, default=64, help="exact channels per linear")
    ap.add_argument("--budget", type=int, default=8192, help="tokens of the calibration stream")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    model_dir = a.model_dir.rstrip("/")
    model_id = a.model or "spinf/" + os.path.basename(model_dir)
    if not qwen.is_qwen(model_dir):
        sys.exit(f"{model_dir}: not a Qwen-architecture checkpoint")
    with open(a.jobs) as f:
        jobs = [Job(j["id"], j["prompts"], j["options"]) for j in map(json.loads, f) if j.get("prompts")]
    if not jobs:
        sys.exit(f"{a.jobs}: no jobs")
    os.makedirs(a.out, exist_ok=True)

    cfg = EngineConfig(
        model_id=model_id,
        model_dir=model_dir,
        quant="bf16",
        stack_options="compact=0",  # the stream layout as packed: the hooks index the real tokens by the batch's mask
        media=False,
        align_jobs=False,
        compile=False,
        token_budget=a.budget,
        max_tokens=a.budget,
    )
    t0 = time.time()
    sc = Scorer(cfg, a.device)
    print(f"loaded {model_dir} in bf16 mode, {time.time() - t0:.0f} s", flush=True)

    pk = Packer(sc.budget, sc.max_tokens)
    batches = []
    for j in jobs:
        for first, tree in sc.trees(j):
            batches += list(pk.push((j.id, first), tree))
    batches += list(pk.flush())
    b = max(batches, key=lambda x: x.n_tokens)
    real = torch.from_numpy(b.art >= 0).to(sc.device)
    n_real = int(real.sum())
    print(f"{len(jobs)} jobs -> {len(batches)} streams; running the largest: {n_real} real tokens of {b.length}", flush=True)

    chmax: dict[str, torch.Tensor] = {}
    shares: dict[tuple, list] = {}
    tops = sorted({16, a.n_out, 128})

    def hook(name):
        def f(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1])[real].abs()
            ch = x.amax(0).float()
            chmax[name] = ch
            top = ch.topk(max(tops)).indices
            tokmax_ch = x.argmax(-1)
            del x
            kind = name.rsplit(".", 1)[-1]
            for n in tops:
                shares.setdefault((kind, n), []).append(torch.isin(tokmax_ch, top[:n]).float().mean().item())
            shares.setdefault((kind, "ratio"), []).append((ch.topk(a.n_out).values[-1] / ch.median()).item())

        return f

    hooks = []
    for name, m in sc.model.named_modules():
        if hasattr(m, "swap") and getattr(m, "name", "") in KINDS:  # the int4 linears (qwen.Int4Linear)
            hooks.append(m.register_forward_pre_hook(hook(name)))
    with torch.no_grad():
        sc._run(b)
    torch.cuda.synchronize()
    for h in hooks:
        h.remove()

    out = {
        name: v.topk(a.n_out).indices.sort().values.tolist() for name, v in chmax.items() if name.rsplit(".", 1)[-1] in qwen.SPLIT_LINEARS
    }
    table = {
        "n_out": a.n_out,
        "source": f"quantizations/export_outliers.py: per-channel absmax over one {n_real}-token stream of {len(jobs)} "
        f"calibration jobs ({os.path.basename(a.jobs)}), bf16 activations",
        "model": model_id,
        "stream_tokens": n_real,
        "outliers": out,
    }
    prov = os.path.join(model_dir, "spinf_quantize.json")
    if os.path.exists(prov):
        with open(prov) as f:
            table["checkpoint"] = json.load(f)
    path = os.path.join(a.out, "spinf_outliers.json")
    with open(path, "w") as f:
        json.dump(table, f)
    print(f"{len(out)} linears x {a.n_out} channels -> {path} ({os.path.getsize(path) // 1024} KB)")
    print("share of tokens whose largest |x| lies in the top-n channels (by stream absmax); n-th channel max / median channel max")
    for kind in KINDS:
        row = [f"top {n}: {sum(shares[(kind, n)]) / len(shares[(kind, n)]):.2f}" for n in tops if (kind, n) in shares]
        r = shares.get((kind, "ratio"))
        if r:
            split = "  (split)" if kind in qwen.SPLIT_LINEARS else ""
            print(f"  {kind:14s} " + "  ".join(row) + f"  ratio {sum(r) / len(r):.1f}" + split)


if __name__ == "__main__":
    main()
