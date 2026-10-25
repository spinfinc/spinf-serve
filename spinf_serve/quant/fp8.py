"""The FP8 static scheme (fp8s) as a checkpoint: e4m3 weights with a per-tensor scale and the calibrated per-tensor activation scale
of every quantized linear, in the compressed-tensors "float-quantized" layout (the RedHat / Neural Magic FP8 checkpoints, read by
vLLM): `<linear>.weight` (float8_e4m3fn [out, in]), `<linear>.weight_scale` (float32 [1]), `<linear>.input_scale` (float32 [1]),
`quantization_config` in config.json. The numerics are those of the load-time fp8s path (models/gemma4.py Fp8StaticLinear): a
checkpoint written by quantizations/fp8_quantize.py scores bit-identically to the bf16 original quantized at load with the same
scale table, and the engine takes its scales from the checkpoint and nowhere else (plan §2.4, one path only)."""

import json
import os

FP8_MAX = 448.0  # the largest finite e4m3 value


def quantize_weight(w, FP8_MAX=FP8_MAX):
    """a bf16 [out, in] weight -> (float8_e4m3fn [out, in], float32 scale [] with w ~= w8 * scale): the exact arithmetic of the
    engine's fp8 linears (float32 amax / 448, the quotient clamped to the e4m3 range, round to nearest even)"""
    import torch

    wf = w.float()
    s = wf.abs().amax().clamp(min=1e-12) / FP8_MAX
    return (wf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s.reshape(())


def quantization_config(ignore: list[str]) -> dict:
    """the config.json entry of an fp8-static checkpoint: every nn.Linear quantized except the modules in `ignore` (compressed-tensors
    names: a module path, or re:<regex>)"""
    scheme = {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True, "dynamic": False}
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": dict(scheme),
                "input_activations": dict(scheme),
                "output_activations": None,
            }
        },
        "ignore": list(ignore),
    }


def is_fp8_static(model_dir: str) -> bool:
    """whether the checkpoint carries fp8 weights with static per-tensor activation scales (config.json quantization_config as
    quantization_config() writes it); any other quantization_config is refused (the engine quantizes bf16 checkpoints only)"""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            q = json.load(f).get("quantization_config")
    except (OSError, ValueError):
        return False
    if not q or q.get("quant_method") != "compressed-tensors":  # none, or another format (GPTQ int4: models/qwen35)
        return False
    groups = list((q.get("config_groups") or {}).values())
    static_tensor = (
        q.get("format") == "float-quantized"
        and len(groups) == 1
        and all(
            (groups[0].get(part) or {}).get("strategy") == "tensor"
            and (groups[0].get(part) or {}).get("type") == "float"
            and (groups[0].get(part) or {}).get("num_bits") == 8
            and not (groups[0].get(part) or {}).get("dynamic")
            for part in ("weights", "input_activations")
        )
    )
    if not static_tensor:
        raise RuntimeError(
            f"{model_dir}: quantization_config {q.get('quant_method')} / {q.get('format')} is not the fp8-static layout the engine "
            "reads (compressed-tensors float-quantized, per-tensor static weights and input activations)"
        )
    return True
