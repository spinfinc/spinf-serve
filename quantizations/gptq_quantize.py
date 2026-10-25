#!/usr/bin/env python3
"""GPTQ int4 quantization of a Qwen3.5-architecture checkpoint (Qwen3.8-27B, pplx-decider-v1-27b) for spinf-serve.

Recipe (GPTQModel 6.0.3): 4 bits, group 32 along the input, symmetric, no act-order, true-sequential, FOEM (alpha 0.25,
beta 0.2), activation-weighted MSE 2.0, act-group-aware; 256 calibration samples from evol-codealpaca-v1 and C4, spread
evenly over four token-length bins between 256 and 2048 tokens. The recipe is the one of btbtyler09/Qwen3.8-27B-GPTQ-4bit
(ACKNOWLEDGEMENTS.md); this script is spinf's own.

Quantized: every linear of the text decoder that GPTQModel's Qwen3.5 module tree lists (q / k / v / o, the Gated DeltaNet
in_proj_qkv / in_proj_z / out_proj, gate / up / down). Left as they are: in_proj_a / in_proj_b, norms, embeddings, the LM
head, the vision tower, the MTP head, and for the decider its readout and decision config.

A decision model (readout.safetensors + decision_config.json in place of lm_head.weight, architectures ["Qwen3_5Model"]) is
loaded through a prepared view of the source: the same shards, the config patched to Qwen3_5ForConditionalGeneration so
GPTQModel's loader accepts it (the missing LM head is randomly initialised and never quantized). After the save the random
lm_head.weight is dropped from the shards, the architectures entry is restored and the readout files are copied over.

    python gptq_quantize.py SRC OUT [--revision HASH] [--seed 20261006] [--samples 256] \
        [--keep readout.safetensors decision_config.json chat_template.jinja processor_config.json]

Environment (run 2026-10-14 locally and on a RunPod A40; the engine's own pins do not apply here): an isolated venv with
torch 2.14.0 + torchvision 0.29.1 (PyPI, cu130: a 580+ driver), transformers 5.10.0 (GPTQModel 6.0.3 breaks on 5.12+),
tokenizers, safetensors, huggingface_hub<2, datasets, accelerate, pillow, numpy==2.2.6 (GPTQModel's pin), and the build
dependencies pypcre + ninja; then `BUILD_CUDA_EXT=0 pip install --no-build-isolation gptqmodel==6.0.3` (the CUDA extension
is not needed to quantize and does not build against a system CUDA older than torch's), then `pip uninstall kernels`
(GPTQModel pulls it in, and transformers cannot import with it present).

SRC is a local checkpoint directory (a Hub snapshot, or the decider prepared as Qwen3_5ForConditionalGeneration). One GPU is
enough (inactive layers are offloaded to disk); a 27B takes about two hours on an RTX 6000 Ada. The run writes
OUT/spinf_quantize.json with the source, revision, seed, sample count and tool versions, next to GPTQModel's own
quantize_config.json.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time

CODE_SET = ("theblackcat102/evol-codealpaca-v1", "train", None)
TEXT_SET = ("allenai/c4", "train", "en/c4-train.00001-of-01024.json.gz")
BINS = [(256, 512), (512, 1024), (1024, 1536), (1536, 2048)]
KEEP_IF_PRESENT = (
    "readout.safetensors",
    "decision_config.json",
    "LICENSE",
    "NOTICE",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
DECIDER_ARCH = "Qwen3_5Model"
LOADER_ARCH = "Qwen3_5ForConditionalGeneration"
MAX_CHARS = 16_000  # a document above this is longer than the top bin in any tokenizer: skipped before tokenizing

log = logging.getLogger("gptq_quantize")


def documents(rng, tokenizer, max_docs):
    """Yield (text, tokens, source) for a random interleaving of the two corpora, at most max_docs of each."""
    from datasets import load_dataset

    pools = []
    for name, split, files in (CODE_SET, TEXT_SET):
        ds = load_dataset(name, split=split, data_files=files) if files else load_dataset(name, split=split)
        order = list(range(len(ds)))
        rng.shuffle(order)
        pools.append((name, ds, order[:max_docs]))
        log.info("%s: %d documents, using up to %d", name, len(ds), max_docs)
    cursors = [0, 0]
    while any(c < len(p[2]) for c, p in zip(cursors, pools, strict=True)):
        k = rng.randrange(2)
        name, ds, order = pools[k]
        if cursors[k] >= len(order):
            k = 1 - k
            name, ds, order = pools[k]
        row = ds[order[cursors[k]]]
        cursors[k] += 1
        if name == CODE_SET[0]:
            text = f"### Instruction:\n{row.get('instruction', '')}\n\n### Response:\n{row.get('output', '')}"
        else:
            text = row.get("text", "")
        if len(text) < 50 or len(text) > MAX_CHARS:
            continue
        yield text, len(tokenizer(text, add_special_tokens=False)["input_ids"]), name


def calibration_set(rng, tokenizer, n_samples, max_docs=40_000):
    """n_samples texts, the same number in each token-length bin: single documents where the corpora have them, and
    documents joined with a separator where a bin runs short."""
    per_bin = [n_samples // len(BINS) + (1 if i < n_samples % len(BINS) else 0) for i in range(len(BINS))]
    binned = [[] for _ in BINS]
    short = []  # documents below the lowest bin, the material for joined samples
    for text, n, _ in documents(rng, tokenizer, max_docs):
        if all(len(b) >= k for b, k in zip(binned, per_bin, strict=True)):
            break
        for i, (lo, hi) in enumerate(BINS):
            if lo <= n < hi:
                if len(binned[i]) < per_bin[i]:
                    binned[i].append(text)
                break
        else:
            if n < BINS[0][0]:
                short.append((text, n))
    log.info("single documents per bin: %s (wanted %s); %d short documents to join", [len(b) for b in binned], per_bin, len(short))
    for i, (lo, hi) in enumerate(BINS):
        tries = 0
        while len(binned[i]) < per_bin[i] and tries < 200 * per_bin[i]:
            tries += 1
            parts, n = [], 0
            while n < lo:
                text, k = rng.choice(short)
                if n + k >= hi:
                    continue
                parts.append(text)
                n += k + 2  # the separator
            joined = "\n\n---\n\n".join(parts)
            if lo <= len(tokenizer(joined, add_special_tokens=False)["input_ids"]) < hi:
                binned[i].append(joined)
        if len(binned[i]) < per_bin[i]:
            raise SystemExit(f"bin {lo}-{hi}: only {len(binned[i])} of {per_bin[i]} samples; raise --max-docs")
    samples = [t for b in binned for t in b]
    rng.shuffle(samples)
    return samples


def is_decider(src):
    return os.path.exists(os.path.join(src, "readout.safetensors"))


def prepared_view(src, workdir):
    """A directory that loads as a Qwen3_5ForConditionalGeneration: symlinks to every file of src, config.json patched."""
    view = os.path.join(workdir, "prepared-" + os.path.basename(os.path.normpath(src)))
    shutil.rmtree(view, ignore_errors=True)
    os.makedirs(view)
    for name in os.listdir(src):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(src, name)), os.path.join(view, name))
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("architectures") != [DECIDER_ARCH]:
        raise SystemExit(f"a decision model with architectures {cfg.get('architectures')}: expected [{DECIDER_ARCH!r}]")
    cfg["architectures"] = [LOADER_ARCH]
    with open(os.path.join(view, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return view


def drop_lm_head(out):
    """Remove lm_head.weight (random for a decision model) from the saved checkpoint and restore the architectures entry."""
    from safetensors.torch import load_file, save_file

    index_path = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard = index["weight_map"].pop("lm_head.weight", None)
        files = [shard] if shard else []
    else:
        index, files = None, ["model.safetensors"]
    for name in files:
        path = os.path.join(out, name)
        tensors = load_file(path)
        if "lm_head.weight" not in tensors:
            continue
        removed = tensors.pop("lm_head.weight")
        save_file(tensors, path, metadata={"format": "pt"})
        log.info("dropped lm_head.weight %s from %s", tuple(removed.shape), name)
        if index is not None:
            index["metadata"]["total_size"] = index.get("metadata", {}).get("total_size", 0) - removed.numel() * removed.element_size()
    if index is not None:
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["architectures"] = [DECIDER_ARCH]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="local checkpoint directory")
    ap.add_argument("out", help="output directory for the quantized checkpoint")
    ap.add_argument("--revision", help="the source's Hub revision, recorded in spinf_quantize.json (read from a snapshot path if omitted)")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-docs", type=int, default=40_000, help="documents drawn from each corpus at most")
    ap.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help=f"files SRC must have, copied to OUT as they are; copied anyway when present: {' '.join(KEEP_IF_PRESENT)}",
    )
    ap.add_argument("--workdir", default=".", help="where the prepared view of a decision model and GPTQModel's offload go")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    import gptqmodel
    import torch
    import transformers
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.config import FOEMConfig
    from transformers import AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    revision = a.revision or (re.search(r"/snapshots/([0-9a-f]{40})", a.src) or [None, None])[1]
    log.info("source %s (revision %s) -> %s; seed %d, %d samples", a.src, revision, a.out, a.seed, a.samples)

    missing = [n for n in a.keep if not os.path.exists(os.path.join(a.src, n))]
    if missing:
        raise SystemExit(f"--keep files not in {a.src}: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(a.src, use_fast=True, trust_remote_code=True)
    t0 = time.time()
    samples = calibration_set(random.Random(a.seed), tokenizer, a.samples, a.max_docs)
    digest = hashlib.sha256("\n\x00".join(samples).encode()).hexdigest()
    log.info("%d calibration samples in %.0f s (sha256 %s)", len(samples), time.time() - t0, digest[:16])

    config = QuantizeConfig(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        true_sequential=True,
        lm_head=False,
        offload_to_disk=True,  # only the active layer on the GPU
        auto_forward_data_parallel=False,
        device=a.device,
        foem=FOEMConfig(alpha=0.25, beta=0.2),
        mse=2.0,
    )
    decider = is_decider(a.src)
    load_from = prepared_view(a.src, a.workdir) if decider else a.src
    log.info("loading %s%s", load_from, " (decision model: LM head randomly initialised, dropped after the save)" if decider else "")
    model = GPTQModel.load(load_from, config, trust_remote_code=True)
    t0 = time.time()
    model.quantize(samples, batch_size=1)
    log.info("quantized in %.0f min", (time.time() - t0) / 60)
    model.save(a.out)
    tokenizer.save_pretrained(a.out)
    if decider:
        drop_lm_head(a.out)
    kept = []
    for name in list(a.keep) + [n for n in KEEP_IF_PRESENT if n not in a.keep]:
        if os.path.exists(os.path.join(a.src, name)) and (name in a.keep or not os.path.exists(os.path.join(a.out, name))):
            shutil.copy(os.path.join(a.src, name), os.path.join(a.out, name))
            kept.append(name)
    with open(os.path.join(a.out, "spinf_quantize.json"), "w") as f:
        json.dump(
            {
                "source": a.src,
                "revision": revision,
                "seed": a.seed,
                "samples": len(samples),
                "samples_sha256": digest,
                "bins": BINS,
                "corpora": [CODE_SET[0], f"{TEXT_SET[0]} {TEXT_SET[2]}"],
                "kept": kept,
                "decision_model": decider,
                "tools": {"gptqmodel": gptqmodel.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
            },
            f,
            indent=1,
        )
    log.info("saved %s", a.out)


if __name__ == "__main__":
    main()
