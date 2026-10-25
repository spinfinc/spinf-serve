import json
import os
import struct

import pytest

from spinf_serve import weights
from spinf_serve.weights import find_weights, incomplete


def shard(path, n_bytes=16, truncate=0):
    header = json.dumps({"w": {"dtype": "U8", "shape": [n_bytes], "data_offsets": [0, n_bytes]}}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + b"\0" * (n_bytes - truncate))


def snapshot(root, model_id, rev, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"), skip=()):
    d = os.path.join(root, "models--" + model_id.replace("/", "--"), "snapshots", rev)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write("{}")
    if len(shards) > 1:
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}, f)
    for s in shards:
        if s not in skip:
            shard(os.path.join(d, s))
    return d


def test_incomplete_snapshots_are_named(tmp_path):
    d = snapshot(str(tmp_path), "g/m", "a")
    assert incomplete(d) is None
    os.remove(os.path.join(d, "model-00001-of-00002.safetensors"))
    assert incomplete(d) == "missing model-00001-of-00002.safetensors"
    os.symlink(os.path.join(str(tmp_path), "no-blob"), os.path.join(d, "model-00001-of-00002.safetensors"))
    assert incomplete(d) == "missing model-00001-of-00002.safetensors"  # dangling symlink: blob never downloaded
    os.remove(os.path.join(d, "model-00001-of-00002.safetensors"))
    shard(os.path.join(d, "model-00001-of-00002.safetensors"), truncate=4)
    assert incomplete(d).startswith("truncated model-00001-of-00002.safetensors")
    single = snapshot(str(tmp_path), "g/one", "a", shards=("model.safetensors",))
    assert incomplete(single) is None
    os.remove(os.path.join(single, "config.json"))
    assert incomplete(single) == "no config.json"


def test_find_weights_skips_a_partial_cache(tmp_path, monkeypatch):
    runpod, hf = str(tmp_path / "runpod"), str(tmp_path / "hf")
    monkeypatch.setenv("SPINF_CACHE_DIRS", runpod)
    monkeypatch.setenv("HF_HOME", hf)
    snapshot(runpod, "g/m", "a", skip=("model-00001-of-00002.safetensors",))
    with pytest.raises(FileNotFoundError):
        find_weights("g/m", download=False)
    good = snapshot(os.path.join(hf, "hub"), "g/m", "a")
    assert find_weights("g/m", download=False) == (good, os.path.join(hf, "hub"))
    full = snapshot(runpod, "g/m", "b")  # a complete snapshot in an extra cache root still comes first
    assert find_weights("g/m", download=False) == (full, runpod)


def test_local_checkpoints_are_complete():
    for model_id in ("google/gemma-4-12B", "google/gemma-4-31B"):
        try:
            d, _ = find_weights(model_id, download=False)
        except FileNotFoundError:
            continue
        assert incomplete(d) is None


def test_revision_reads_the_commit_hash_of_a_plain_download(tmp_path):
    """a directory downloaded with huggingface_hub (local-dir layout) names its revision by the commit hash in .cache metadata,
    as a snapshot directory does by its name; anything else by its name"""

    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / "b64f44dfedb71b73287219e0ec348b0a20c447e2"
    snap.mkdir(parents=True)
    assert weights.revision(str(snap)) == "b64f44dfedb71b73287219e0ec348b0a20c447e2"
    plain = tmp_path / "Qwen3.8-27B-GPTQ-4bit"
    meta = plain / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(
        "b64f44dfedb71b73287219e0ec348b0a20c447e2\n1f94a151d57ae5c5fc24bfe27544e26140fb74f5\n1791175530.8\n"
    )
    assert weights.revision(str(plain) + "/") == "b64f44dfedb71b73287219e0ec348b0a20c447e2"
    other = tmp_path / "some-dir"
    other.mkdir()
    assert weights.revision(str(other)) == "some-dir"


def test_the_qwen_outlier_table_is_named_by_the_checkpoint_revision(tmp_path):
    from spinf_serve.quant import tables

    plain = tmp_path / "Qwen3.8-27B-GPTQ-4bit"
    meta = plain / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text("b64f44dfedb71b73287219e0ec348b0a20c447e2\netag\n1.0\n")
    path = tables.outliers_table_path("btbtyler09/Qwen3.8-27B-GPTQ-4bit", str(plain))
    assert path.endswith("quant/tables/btbtyler09__Qwen3.8-27B-GPTQ-4bit/b64f44dfedb71b73287219e0ec348b0a20c447e2-outliers.json")
    assert tables.load_outliers_table("btbtyler09/Qwen3.8-27B-GPTQ-4bit", str(plain))["n_out"] > 0


def test_a_checkpoint_ships_its_own_outlier_table(tmp_path):
    from spinf_serve.quant import tables

    own = tmp_path / "qwen3.8-27b-int4"
    own.mkdir()
    assert tables.load_outliers_table("spinf/qwen3.8-27b-int4", str(own)) is None
    (own / "spinf_outliers.json").write_text('{"n_out": 64, "outliers": {"layers.0.mlp.gate_up_proj": [1, 2]}}')
    assert tables.load_outliers_table("spinf/qwen3.8-27b-int4", str(own))["outliers"]["layers.0.mlp.gate_up_proj"] == [1, 2]


def test_find_weights_pins_the_revision(tmp_path, monkeypatch):
    runpod, hf = str(tmp_path / "runpod"), str(tmp_path / "hf")
    monkeypatch.setenv("SPINF_CACHE_DIRS", runpod)
    monkeypatch.setenv("HF_HOME", hf)
    a, b = "a" * 40, "b" * 40
    snapshot(runpod, "g/m", a)
    good = snapshot(os.path.join(hf, "hub"), "g/m", b)
    assert find_weights("g/m", b, download=False) == (good, os.path.join(hf, "hub"))  # the pinned one, not the first found
    with pytest.raises(FileNotFoundError, match="revision"):
        find_weights("g/m", "c" * 40, download=False)
    assert find_weights("g/m", download=False)[0].endswith(a)  # unpinned: the first complete snapshot


def test_service_accepts_the_manifest_aliases():
    from spinf_serve.config import EngineConfig
    from spinf_serve.service import ScoreService

    s = ScoreService(EngineConfig.from_manifest("spinf/gemma-4-12b-fp8"))
    assert {"spinf/gemma-4-12b-fp8", "spinf-12b", "google/gemma-4-12b"} <= s.model_names  # the manifest aliases, lower-cased
    assert ScoreService(EngineConfig(served_model="x/y")).model_names == {"x/y"}
