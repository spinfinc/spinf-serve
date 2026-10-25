"""quant/fp8.py: the fp8-static checkpoint layout and its numerics (the writer is quantizations/fp8_quantize.py)"""

import json
import os

import pytest

from spinf_serve.quant import fp8


def test_quantize_weight_is_the_fp8_linears_arithmetic():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    w = (torch.randn(64, 48) * 3).to(torch.bfloat16)
    w8, s = fp8.quantize_weight(w)
    assert w8.dtype == torch.float8_e4m3fn and w8.shape == (64, 48) and s.dtype == torch.float32 and s.shape == ()
    wf = w.float()  # the formula of models/gemma4.py Fp8StaticLinear.__init__, written out
    ref_s = wf.abs().amax().clamp(min=1e-12) / 448.0
    ref = (wf / ref_s).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert float(s) == float(ref_s)
    assert torch.equal(w8.view(torch.uint8), ref.view(torch.uint8))
    assert float(w8.float().abs().amax()) == 448.0  # the largest weight lands on the e4m3 maximum
    assert float((w8.float() * s - wf).abs().amax()) <= float(wf.abs().amax()) / 16  # e4m3: 3 mantissa bits


def write_config(tmp_path, q):
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump({"model_type": "gemma4_unified", **({"quantization_config": q} if q else {})}, f)
    return str(tmp_path)


def test_fp8_static_checkpoints_are_recognised(tmp_path):
    q = fp8.quantization_config(["lm_head", "model.embed_vision.embedding_projection"])
    assert q["quant_method"] == "compressed-tensors" and q["format"] == "float-quantized"
    g = q["config_groups"]["group_0"]
    assert g["targets"] == ["Linear"] and g["weights"]["strategy"] == "tensor" and g["input_activations"]["dynamic"] is False
    assert fp8.is_fp8_static(write_config(tmp_path, q)) is True


def test_other_checkpoints_are_not(tmp_path):
    assert fp8.is_fp8_static(write_config(tmp_path, None)) is False
    assert fp8.is_fp8_static(write_config(tmp_path, {"quant_method": "gptq", "bits": 4})) is False  # the int4 checkpoints
    assert fp8.is_fp8_static(str(tmp_path / "nowhere")) is False
    dynamic = fp8.quantization_config([])
    dynamic["config_groups"]["group_0"]["input_activations"]["dynamic"] = True  # RedHat's FP8-dynamic: no input scales
    with pytest.raises(RuntimeError, match="not the fp8-static layout"):
        fp8.is_fp8_static(write_config(tmp_path, dynamic))
