"""models/registry.py and models/base.py: the family of a checkpoint, the stack a manifest names, the Qwen options object"""

import json

import pytest

from spinf_serve.config import EngineConfig
from spinf_serve.models import registry
from spinf_serve.models.base import Stack, compile_options
from spinf_serve.models.gemma4 import Gemma4Stack
from spinf_serve.models.qwen35.qwen import Qwen35Stack, QwenOptions, tree_plan


def checkpoint(tmp_path, name: str, model_type: str) -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def test_the_family_is_read_from_the_checkpoint_or_the_manifest(tmp_path):
    gemma = checkpoint(tmp_path, "g", "gemma4_unified")
    qwen = checkpoint(tmp_path, "q", "qwen3_5")
    assert isinstance(registry.stack_for(gemma), Gemma4Stack)
    assert isinstance(registry.stack_for(checkpoint(tmp_path, "g31", "gemma4")), Gemma4Stack)
    assert isinstance(registry.stack_for(qwen), Qwen35Stack)
    assert isinstance(registry.stack_for(qwen, "qwen35"), Qwen35Stack)
    with pytest.raises(RuntimeError, match="not a gemma4 checkpoint"):
        registry.stack_for(qwen, "gemma4")  # the manifest and the weights disagree
    with pytest.raises(ValueError, match="unknown model family"):
        registry.stack_for(gemma, "llama")
    with pytest.raises(RuntimeError, match="no supported architecture"):
        registry.stack_for(checkpoint(tmp_path, "x", "llama"))
    assert not Gemma4Stack.detect(str(tmp_path / "missing"))


def test_every_builtin_manifest_names_a_registered_family():
    from spinf_serve import manifests

    for m in manifests.catalog().values():
        assert m.family in registry.stacks()


def test_stack_flags_and_defaults():
    g, q = Gemma4Stack(), Qwen35Stack()
    assert (g.hybrid, g.image_scheme, g.bidirectional_images) == (False, "gemma", True)
    assert (q.hybrid, q.image_scheme, q.bidirectional_images) == (True, "qwen", False)
    assert Stack().chunk_extra_bytes(None, []) == 0 and Stack().group_extra(None, None, None, {}, 1) is None
    assert compile_options(EngineConfig()) == {"deterministic": True}
    assert compile_options(EngineConfig(deterministic_compile=False)) is None


def test_qwen_options_parse():
    o = QwenOptions.parse("")
    assert o == QwenOptions() and (o.mode, o.compact, o.gdn_path, o.gdn_short, o.decode_linear) == ("i8", True, "indexed", 64, "w4")
    o = QwenOptions.parse("compact=0, mode=f16, gdn_short=32, decode_graphs=false, alpha=0.3")
    assert (o.compact, o.mode, o.gdn_short, o.decode_graphs, o.alpha) == (False, "f16", 32, False, 0.3)
    with pytest.raises(ValueError, match="unknown Qwen option"):
        QwenOptions.parse("fuse=1")
    with pytest.raises(ValueError, match="unknown Qwen option"):
        QwenOptions.parse("outliers=x")  # tensors come from a research script, not a string
    assert EngineConfig(stack_options="compact=0").stack_options == "compact=0"
    s = Qwen35Stack(QwenOptions(compact=False))
    assert s.options.compact is False and s._options is not None


def test_tree_plan_takes_its_knobs_as_arguments():
    import numpy as np

    prev = np.array([-1, 0, 1, 1, 3], dtype=np.int64)
    real = np.ones(5, bool)
    p = tree_plan(prev, real, "cpu", path="levels")
    assert p.path == "levels" and p.n_keep is None
    p = tree_plan(prev, real, "cpu", short=1, call_tokens=2)
    assert p.path == "indexed" and p.n_segments == 3
