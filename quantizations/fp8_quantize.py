#!/usr/bin/env python3
"""Write an fp8-static checkpoint of a Gemma 4 model: the FP8 weights the engine's fp8s scheme computes at load time, with the
calibrated activation scales as tensors, in the compressed-tensors layout (spinf_serve/quant/fp8.py).

What it does: every nn.Linear of the text decoder (q / k / v / o, gate / up / down) becomes `weight` (float8_e4m3fn [out, in],
the bf16 weight divided by its per-tensor scale), `weight_scale` (float32 [1], |w|max / 448) and `input_scale` (float32 [1], the
linear's activation scale from the scale table that quant/calibrate.py made on this source revision). Everything else is copied
as it is in bf16: embeddings, norms, the media embedders (vision tower, vision and audio projections), the tied LM head. The
config gains `quantization_config` (every Linear quantized, the unquantized modules listed in `ignore`); the tokenizer, processor
and generation configs are copied; `calibration.json` is the scale table verbatim and `spinf_quantize.json` records the source,
its revision, the table's digest and the tool versions. The arithmetic is quant/fp8.py's quantize_weight, the one the load-time
path uses: the engine scores the written checkpoint bit-identically to the bf16 source quantized at load with the same table
(the parity run of quantizations/README.md checks it).

    uv run python quantizations/fp8_quantize.py SRC OUT [--table TABLE.json] [--revision HASH] [--shard-gb 4.5] [--device cuda:0]

SRC is a local checkpoint directory (a Hub snapshot or a plain download); TABLE defaults to the table of SRC's revision in
quant/tables/. The weights are quantized on a GPU, one tensor at a time (a few hundred MB of card memory), the shards streamed:
a minute for the 12B. The GPU is not optional for a checkpoint that must reproduce the recorded references: the engine quantizes
on the GPU at load time, and CUDA's division by a Python scalar (a reciprocal multiply) and its float -> e4m3 cast differ from the
CPU's in the last bit, which moves the scale of most tensors by one ulp and the rounding of ~0.2 % of the elements (2026-10-15:
a CPU-written 31B scored 0.08 mean |delta log-prob| from the load-time scheme; written on the GPU, zero). --device cpu is for a
checkpoint that will not be compared bit for bit. OUT holds the checkpoint in shards of about --shard-gb, ready to upload.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PREFIX = "model.language_model."
COPY = (
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "LICENSE",
    "NOTICE",
)


def text_linears(hf_cfg) -> set[str]:
    """the module names (under PREFIX) of the text decoder's nn.Linear modules, from the architecture on the meta device"""
    import torch
    import torch.nn as nn

    if hf_cfg.model_type == "gemma4_unified":
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel as TextModel
    else:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as TextModel
    with torch.device("meta"):
        model = TextModel(hf_cfg.text_config)
    return {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}


class Shards:
    """tensors accumulated into model-0000i-of-0000N.safetensors files of about `limit` bytes (N known at the end)"""

    def __init__(self, out: str, limit: int):
        self.out, self.limit = out, limit
        self.cur: dict = {}
        self.cur_bytes = 0
        self.files: list[str] = []
        self.weight_map: dict[str, str] = {}
        self.total = 0

    def add(self, *named) -> None:
        """(name, tensor) pairs that stay in one shard (a weight with its scales)"""
        n = sum(t.numel() * t.element_size() for _, t in named)
        if self.cur and self.cur_bytes + n > self.limit:
            self.flush()
        for name, t in named:
            self.cur[name] = t.contiguous()
        self.cur_bytes += n
        self.total += n

    def flush(self) -> None:
        from safetensors.torch import save_file

        if not self.cur:
            return
        path = os.path.join(self.out, f"shard-{len(self.files):05d}.safetensors")
        save_file(self.cur, path, metadata={"format": "pt"})
        self.files.append(path)
        for k in self.cur:
            self.weight_map[k] = path
        self.cur, self.cur_bytes = {}, 0

    def finish(self) -> list[str]:
        self.flush()
        n = len(self.files)
        names = []
        for i, path in enumerate(self.files):
            name = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.rename(path, os.path.join(self.out, name))
            names.append(name)
            for k, v in self.weight_map.items():
                if v == path:
                    self.weight_map[k] = name
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}, f, indent=2)
        return names


def main() -> None:
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig

    from spinf_serve.quant import fp8
    from spinf_serve.weights import revision as snapshot_revision

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="local checkpoint directory (bf16 safetensors)")
    ap.add_argument("out", help="output directory for the fp8-static checkpoint")
    ap.add_argument("--table", help="the fp8s scale table (default: quant/tables/<org>__<name>/<revision>.json of SRC's revision)")
    ap.add_argument("--revision", help="the source's Hub revision (default: read from SRC's snapshot path or download metadata)")
    ap.add_argument("--shard-gb", type=float, default=4.5)
    ap.add_argument("--device", default="cuda:0", help="where the weights are quantized (see above)")
    a = ap.parse_args()

    if a.device == "cpu":
        print("warning: quantizing on the CPU, the checkpoint will not match the GPU's load-time quantization bit for bit", flush=True)
    rev = a.revision or snapshot_revision(a.src)
    hf_cfg = AutoConfig.from_pretrained(a.src)
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("quantization_config"):
        sys.exit(f"{a.src} is already quantized ({cfg['quantization_config'].get('quant_method')})")
    table_path = a.table
    if table_path is None:
        from spinf_serve.quant import TABLES_DIR

        cand = glob.glob(os.path.join(TABLES_DIR, "*", rev + ".json"))
        if len(cand) != 1:
            sys.exit(f"no scale table for revision {rev} in {TABLES_DIR} (pass --table)")
        table_path = cand[0]
    with open(table_path, "rb") as f:
        table_raw = f.read()
    table = json.loads(table_raw)
    if table.get("scheme") != "fp8s" or table.get("revision") not in (rev, None):
        sys.exit(f"{table_path}: scheme {table.get('scheme')} for revision {table.get('revision')}, source is {rev}")
    scales = table["scales"]
    linears = text_linears(hf_cfg)
    missing = sorted(linears - set(scales))
    if missing:
        sys.exit(f"{table_path}: no activation scale for {len(missing)} linears: {missing[:5]} ...")

    os.makedirs(a.out, exist_ok=True)
    if glob.glob(os.path.join(a.out, "*.safetensors")):
        sys.exit(f"{a.out} already holds shards")
    shards = Shards(a.out, int(a.shard_gb * (1 << 30)))
    ignore: list[str] = []
    n_q = 0
    t0 = time.perf_counter()
    for shard in sorted(glob.glob(os.path.join(a.src, "model*.safetensors"))):
        with safe_open(shard, "pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                t = f.get_tensor(k)
                mod_name, _, pname = k.rpartition(".")
                if k.startswith(PREFIX) and mod_name[len(PREFIX) :] in linears and pname == "weight":
                    w8, s = fp8.quantize_weight(t.to(a.device))
                    xs = float(scales[mod_name[len(PREFIX) :]])
                    shards.add((k, w8.cpu()), (k + "_scale", s.cpu().reshape(1)), (mod_name + ".input_scale", torch.tensor([xs])))
                    n_q += 1
                else:
                    if pname == "weight" and t.ndim == 2 and not k.startswith(PREFIX + "layers."):
                        ignore.append(mod_name)  # a linear or an embedding outside the quantized set
                    shards.add((k, t))
                if n_q and n_q % 50 == 0 and pname == "weight":
                    print(f"  {n_q} linears quantized, {shards.total / 1e9:.1f} GB written, {time.perf_counter() - t0:.0f} s", flush=True)
    names = shards.finish()
    if n_q != len(linears):
        sys.exit(f"{n_q} linears quantized, the architecture has {len(linears)}")
    ignore = sorted(set(ignore) | {"lm_head"})
    cfg["quantization_config"] = fp8.quantization_config(ignore)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(a.out, "calibration.json"), "wb") as f:
        f.write(table_raw)
    copied = []
    for name in COPY:
        if os.path.exists(os.path.join(a.src, name)):
            shutil.copy(os.path.join(a.src, name), os.path.join(a.out, name))
            copied.append(name)
    with open(os.path.join(a.out, "spinf_quantize.json"), "w") as f:
        json.dump(
            {
                "scheme": "fp8-static",
                "source": table.get("model_id") or os.path.basename(a.src.rstrip("/")),
                "revision": rev,
                "calibration": {
                    "file": "calibration.json",
                    "sha256": hashlib.sha256(table_raw).hexdigest(),
                    "created": table.get("created"),
                    "headroom": table.get("headroom"),
                    "tokens": (table.get("corpus") or {}).get("tokens"),
                },
                "quantized_linears": n_q,
                "ignore": ignore,
                "shards": names,
                "copied": copied,
                "date": datetime.date.today().isoformat(),
                "device": torch.cuda.get_device_name(a.device) if a.device != "cpu" else "cpu",
                "tool": "spinf-serve quantizations/fp8_quantize.py",
                "tools": {"torch": torch.__version__, "transformers": transformers.__version__},
            },
            f,
            indent=2,
        )
    print(
        f"done: {n_q} linears quantized, {len(ignore)} modules kept in bf16, {len(names)} shards, {shards.total / 1e9:.1f} GB, "
        f"{time.perf_counter() - t0:.0f} s -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
