"""The per-revision tables under quant/tables/<org>__<name>/: the fp8s activation scales of a bf16 Gemma checkpoint run by its path
(<revision>.json, quant/calibrate.py) and the i8 outlier channels of a third-party Qwen int4 copy (<revision>-outliers.json,
quantizations/export_outliers.py). The spinf uploads carry both inside the checkpoint (the fp8-static tensors, spinf_outliers.json:
plan §2.4) and never read these; they stay for anyone running an original by its id."""

import hashlib
import json
import os

from spinf_serve.quant import TABLES_DIR


def _table_dir(model_id: str) -> str:
    return os.path.join(TABLES_DIR, model_id.replace("/", "__"))


def scale_table_path(model_id: str, model_dir: str) -> str:
    """the fp8s scale table of a model revision: quant/tables/<org>__<name>/<revision>.json (the revision = the snapshot directory name)"""
    from spinf_serve.weights import revision

    return os.path.join(_table_dir(model_id), revision(model_dir) + ".json")


def load_scale_table(model_id: str, model_dir: str) -> dict | None:
    """-> {"scales": {linear module name: activation scale}, "sha": short hash of the file, ...} or None when the revision has none"""
    path = scale_table_path(model_id, model_dir)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    t = json.loads(raw)
    t["sha"] = hashlib.sha256(raw).hexdigest()[:10]
    return t


def outliers_table_path(model_id: str, model_dir: str) -> str:
    """the i8 split scheme's outlier channels of a model revision, next to the fp8s scale tables:
    quant/tables/<org>__<name>/<revision>-outliers.json = {"n_out", "source", "outliers": {linear module name: [input channels]}}"""
    from spinf_serve.weights import revision

    return os.path.join(_table_dir(model_id), revision(model_dir) + "-outliers.json")


def load_outliers_table(model_id: str, model_dir: str) -> dict | None:
    """the checkpoint's own spinf_outliers.json (what our uploads ship, written by quantizations/export_outliers.py), else the
    per-revision table under quant/tables/ (third-party checkpoints); None without either"""
    for path in (os.path.join(model_dir, "spinf_outliers.json"), outliers_table_path(model_id, model_dir)):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None
