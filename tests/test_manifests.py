import os

import pytest

from spinf_serve import manifests
from spinf_serve.config import EngineConfig
from spinf_serve.manifests import Manifest, catalog, load_file, resolve


def test_builtin_catalog_loads_and_is_consistent():
    cat = catalog()
    assert {"spinf/gemma-4-12b-fp8", "spinf/gemma-4-31b-fp8", "spinf/qwen3.8-27b-int4", "spinf/pplx-decider-v1-27b-int4"} <= set(cat)
    for m in cat.values():
        assert m.path.endswith(".toml")
        if m.status == "available":
            assert len(m.source.revision) == 40
    names = [n for m in cat.values() for n in m.names]
    assert len(names) == len(set(names))  # no alias names two models


def test_resolve_by_id_and_alias_case_insensitive():
    assert resolve("spinf/gemma-4-12b-fp8").id == "spinf/gemma-4-12b-fp8"
    assert resolve("SPINF/Gemma-4-12B-FP8").id == "spinf/gemma-4-12b-fp8"
    assert resolve("spinf/pplx-decider-v1-27b-int4").id == "spinf/pplx-decider-v1-27b-int4"
    # the hosted service's names are aliases, so a client written for it works against a server as is
    assert catalog()["spinf/gemma-4-12b-fp8"].aliases[0] == "spinf-12b" and "spinf-12b" in catalog()["spinf/gemma-4-12b-fp8"].names
    assert resolve("QWEN3.8-27B-S").id == "spinf/qwen3.8-27b-int4" and catalog()["spinf/pplx-decider-v1-27b-int4"].aliases == []
    with pytest.raises(KeyError):
        resolve("gpt-4")


def test_decider_is_an_available_readout_model():
    m = resolve("spinf/pplx-decider-v1-27b-int4")
    assert m.status == "available" and m.head == "readout" and m.capabilities == ["score"] and m.prompt_style == "decider"
    assert m.media == [] and len(m.source.revision) == 40


def test_the_spinf_checkpoints_are_our_own_hub_repos():
    for mid in ("spinf/gemma-4-12b-fp8", "spinf/gemma-4-31b-fp8", "spinf/qwen3.8-27b-int4", "spinf/pplx-decider-v1-27b-int4"):
        assert resolve(mid).source.repo == mid  # no third-party repo is a dependency
    assert {resolve(m).source.format for m in ("spinf/gemma-4-12b-fp8", "spinf/gemma-4-31b-fp8")} == {"fp8-static"}


def test_validation_rules():
    base = {
        "id": "x",
        "name": "x",
        "family": "gemma4",
        "source": {"repo": "o/m", "revision": "a" * 40, "format": "safetensors-bf16"},
        "quant": {"allowed": ["fp8s"]},
    }
    Manifest.model_validate(base)
    with pytest.raises(ValueError, match="commit hash"):
        Manifest.model_validate({**base, "source": {**base["source"], "revision": "main"}})
    with pytest.raises(ValueError, match="pins source.revision"):
        Manifest.model_validate({**base, "source": {**base["source"], "revision": ""}})
    with pytest.raises(ValueError, match="readout"):
        Manifest.model_validate({**base, "head": "readout"})
    with pytest.raises(ValueError, match="image"):
        Manifest.model_validate({**base, "media": ["image"]})
    with pytest.raises(ValueError, match="quant.default"):
        Manifest.model_validate({**base, "quant": {"default": "bf16", "allowed": ["fp8s"]}})


def test_user_manifest_dir_overrides_a_builtin(tmp_path):
    (tmp_path / "gemma-4-12b-fp8.toml").write_text(
        'id = "spinf/gemma-4-12b-fp8"\nname = "mine"\naliases = ["my-12b"]\nfamily = "gemma4"\n'
        '[source]\nrepo = "me/gemma"\nrevision = "' + "b" * 40 + '"\nformat = "safetensors-bf16"\n'
        '[quant]\nallowed = ["bf16"]\n'
    )
    m = resolve("my-12b", [manifests.BUILTIN_DIR, str(tmp_path)])
    assert (m.name, m.source.repo) == ("mine", "me/gemma")
    assert resolve("spinf/gemma-4-12b-fp8", [manifests.BUILTIN_DIR, str(tmp_path)]).source.repo == "me/gemma"
    assert resolve("spinf/gemma-4-12b-fp8").source.repo == "spinf/gemma-4-12b-fp8"  # the default catalog is untouched


def test_engine_config_from_manifest_and_env(monkeypatch):
    cfg = EngineConfig.from_manifest("spinf/gemma-4-12b-fp8")
    assert (cfg.model, cfg.served_model, cfg.model_id) == ("spinf/gemma-4-12b-fp8", "spinf/gemma-4-12b-fp8", "spinf/gemma-4-12b-fp8")
    assert cfg.revision == resolve("spinf/gemma-4-12b-fp8").source.revision and cfg.quant == "" and cfg.media is True
    q = EngineConfig.from_manifest("spinf/qwen3.8-27b-int4")
    assert (q.max_tokens, q.token_budget, q.model_id) == (16384, 16384, "spinf/qwen3.8-27b-int4")
    monkeypatch.setenv("SPINF_MODEL", "spinf/gemma-4-31b-fp8")
    monkeypatch.setenv("SPINF_MAX_TOKENS", "8192")  # env wins over the manifest's default
    monkeypatch.setenv("SPINF_QUANT", "w8a16")
    e = EngineConfig.from_env()
    assert (e.model, e.model_id, e.max_tokens, e.quant, e.media) == ("spinf/gemma-4-31b-fp8", "spinf/gemma-4-31b-fp8", 8192, "w8a16", True)
    monkeypatch.delenv("SPINF_MODEL")
    assert EngineConfig.from_env().model == ""  # no manifest: the plain fields, as before


def test_manifest_files_are_the_catalog():
    for m in catalog().values():
        assert load_file(m.path).id == m.id
        assert os.path.dirname(m.path) == manifests.BUILTIN_DIR
