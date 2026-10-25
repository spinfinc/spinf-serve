"""Where the weights come from: the snapshot roots in SPINF_CACHE_DIRS (colon-separated; a serverless host's model cache, say),
then the local Hugging Face cache, then a download. A snapshot counts only when it is complete: a host can hold a partly populated
cache (a worker once loaded a snapshot without its vision-tower tensors from such a cache)."""

import glob
import json
import logging
import os
import struct

log = logging.getLogger(__name__)


def cache_roots() -> list[str]:
    """the snapshot roots searched in order: SPINF_CACHE_DIRS, then the Hugging Face cache"""
    return [d for d in os.environ.get("SPINF_CACHE_DIRS", "").split(":") if d] + [_hf_hub()]


def _hf_hub() -> str:
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def incomplete(d: str) -> str | None:
    """why snapshot directory d cannot be loaded (config, a shard of its index missing or shorter than its header says), or None"""
    if not os.path.isfile(os.path.join(d, "config.json")):
        return "no config.json"
    idx = os.path.join(d, "model.safetensors.index.json")
    try:
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    except FileNotFoundError:
        shards = ["model.safetensors"]
    except (OSError, ValueError, KeyError) as e:
        return f"unreadable index: {e}"
    for name in shards:
        p = os.path.join(d, name)
        if not os.path.isfile(p):  # a dangling symlink (blob not downloaded) counts as missing
            return f"missing {name}"
        try:
            with open(p, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            want = 8 + n + max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
            size = os.path.getsize(p)
        except (OSError, ValueError, KeyError, struct.error) as e:
            return f"unreadable {name}: {e}"
        if size < want:
            return f"truncated {name} ({size} of {want} bytes)"
    return None


def revision(model_dir: str) -> str:
    """the weights revision of a snapshot directory: its name (the commit hash in a Hugging Face cache); for a plain directory
    downloaded with huggingface_hub (hf download --local-dir) the commit hash recorded in its .cache/huggingface metadata, so
    that the fingerprint and the scale / outlier tables name the same revision wherever the checkpoint sits"""
    d = model_dir.rstrip("/")
    for meta in sorted(glob.glob(os.path.join(d, ".cache", "huggingface", "download", "*.metadata"))):
        try:
            with open(meta) as f:
                first = f.readline().strip()
        except OSError:
            continue
        if len(first) == 40 and all(c in "0123456789abcdef" for c in first):
            return first
    return os.path.basename(d)


def find_weights(model_id: str, revision: str = "", download: bool = True) -> tuple[str, str]:
    """-> (snapshot directory, source: the cache root it was found in or "downloaded"). With a revision (a Hub commit hash) only a
    snapshot of that revision counts, and the download asks for it; without one, the first complete snapshot found, then the
    Hub's current revision (unpinned: logged)"""
    sub = "models--" + model_id.replace("/", "--") + "/snapshots/*"
    for root in cache_roots():
        for d in sorted(glob.glob(os.path.join(root, sub))):
            why = incomplete(d)
            if why is not None:
                log.warning("weights: skipping incomplete snapshot %s (%s)", d, why)
            elif revision and _revision(d) != revision:
                log.info("weights: skipping snapshot %s (revision %s, want %s)", d, _revision(d), revision)
            else:
                return d, root
    if not download:
        want = f" at revision {revision}" if revision else ""
        raise FileNotFoundError(f"no complete local snapshot of {model_id}{want}")
    if not revision:
        log.warning("weights: %s is not pinned to a revision: downloading the Hub's current one", model_id)
    from huggingface_hub import snapshot_download

    d = snapshot_download(
        model_id, revision=revision or None, token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "*.safetensors", "tokenizer*"]
    )
    why = incomplete(d)
    if why is not None:
        raise RuntimeError(f"downloaded snapshot of {model_id} is incomplete: {why}")
    if revision and _revision(d) != revision:
        raise RuntimeError(f"downloaded snapshot of {model_id} is revision {_revision(d)}, not {revision}")
    return d, "downloaded"


def _revision(d: str) -> str:
    return revision(d)


def checkpoint_bytes(model_dir: str) -> int:
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(model_dir, "model*.safetensors")))
