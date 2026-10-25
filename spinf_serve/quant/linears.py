"""The quantized linears of the Gemma stack (models/gemma4.py build_model), one set per load: e4m3 weights with a per-tensor scale and
dynamic (Fp8Linear) or calibrated static (Fp8StaticLinear) activation scales, per-row scales both sides (Fp8RowLinear), int8 weights
with bf16 (w8a16) or int8 per-token (w8a8) activations (Int8Linear). The activation quantizers are compiled with the load's compile
options (models/base.py compile_options), and the classes are made inside quant_linears so they call that load's compiled functions.

The class bodies are kept exactly as they were in the Gemma module: editing Fp8Linear's source flips the compiled fp8 kernel variant
(docs/PRECISION.md §2), and Fp8Linear must stay bit-identical to the recorded references."""

from types import SimpleNamespace

from spinf_serve.quant.fp8 import FP8_MAX

# fp8s: Scorer._run turns this on for one forward in EngineConfig.fp8s_monitor_every (the peak check costs one reduction per linear)
MONITOR = {"on": False}


def quant_linears(opts: dict | None, device) -> SimpleNamespace:
    """-> Int8Linear, Fp8Linear, Fp8RowLinear, Fp8StaticLinear (bf16 [out, in] weights on `device` in, quantized buffers out) and
    prequantized (the fp8 classes from an fp8-static checkpoint's tensors)"""
    import torch
    import torch.nn as nn

    @torch.compile(dynamic=True, options=opts)
    def quant_act(x):
        xf = x.reshape(-1, x.shape[-1]).float()
        s = xf.abs().amax().clamp_(min=1e-12) / FP8_MAX
        return (xf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s

    @torch.compile(dynamic=True, options=opts)
    def quant_act_row(x):  # dynamic per-token (row) scales, fp8
        xf = x.reshape(-1, x.shape[-1]).float()
        s = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-12) / FP8_MAX
        return (xf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s

    @torch.compile(dynamic=True, options=opts)
    def quant_act_int8(x):  # dynamic per-token (row) scales
        xf = x.reshape(-1, x.shape[-1]).float()
        s = xf.abs().amax(-1, keepdim=True).clamp_(min=1e-12) / 127.0
        return (xf / s).round_().clamp_(-127, 127).to(torch.int8), s

    class Int8Linear(nn.Module):
        def __init__(self, w, act8):
            super().__init__()
            wf = w.float()
            s = wf.abs().amax(-1).clamp(min=1e-12) / 127.0
            w8 = (wf / s[:, None]).round_().clamp_(-127, 127).to(torch.int8)
            if act8:
                self.register_buffer("w8t", w8.t())  # [in, out] col-major
                self.register_buffer("ws", s)  # [out] fp32
            else:
                self.register_buffer("w8", w8)  # [out, in]
                self.register_buffer("ws", s.to(torch.bfloat16)[:, None])  # [out, 1]
            self.act8 = act8
            self.out_features, self.in_features = w.shape

        def forward(self, x):
            x2 = x.reshape(-1, x.shape[-1])
            m = x2.shape[0]
            if self.act8:
                x8, xs = quant_act_int8(x2)
                if m <= 16:  # _int_mm needs M > 16
                    x8 = torch.nn.functional.pad(x8, (0, 0, 0, 17 - m))
                y = (torch._int_mm(x8, self.w8t)[:m].float() * xs * self.ws).to(torch.bfloat16)
            else:
                y = torch.nn.functional.linear(x2, self.w8.to(torch.bfloat16) * self.ws)  # scale folded into the weight
            return y.reshape(*x.shape[:-1], self.out_features)

    class Fp8Linear(nn.Module):
        def __init__(self, w):  # w: bf16 [out, in] on device
            super().__init__()
            wf = w.float()
            s = wf.abs().amax().clamp(min=1e-12) / FP8_MAX
            self.register_buffer("w8t", (wf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).t())  # [in, out] col-major
            self.register_buffer("ws", s.reshape(()))
            self.out_features, self.in_features = w.shape

        def forward(self, x):
            x8, xs = quant_act(x)
            y = torch._scaled_mm(x8, self.w8t, scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)
            return y.reshape(*x.shape[:-1], self.out_features)

    # kept apart from Fp8Linear on purpose: editing Fp8Linear's source flips the compiled fp8 kernel variant (AMPERE_PORT.md §8,
    # seen again 2026-09-24), and Fp8Linear must stay bit-identical to the recorded references
    class Fp8RowLinear(nn.Module):
        def __init__(self, w):  # w: bf16 [out, in] on device
            super().__init__()
            wf = w.float()
            s = wf.abs().amax(-1, keepdim=True).clamp(min=1e-12) / FP8_MAX  # [out, 1]
            self.register_buffer("w8t", (wf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).t())  # [in, out] col-major
            self.register_buffer("ws", s.t().contiguous())  # [1, out]
            self.out_features, self.in_features = w.shape

        def forward(self, x):
            x8, xs = quant_act_row(x)
            y = torch._scaled_mm(x8, self.w8t, scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)
            return y.reshape(*x.shape[:-1], self.out_features)

    class Fp8StaticLinear(nn.Module):
        """fp8 with a fixed activation scale (fp8s): the same per-tensor _scaled_mm as Fp8Linear (its rows do not depend on the
        stream length), without the per-forward amax over the stream. `peak` keeps the largest |input| / (448 * scale) seen: above 1
        the input was clipped (the scale table's calibration did not cover it)"""

        def __init__(self, w, xs: float):  # w: bf16 [out, in] on device
            super().__init__()
            wf = w.float()
            s = wf.abs().amax().clamp(min=1e-12) / FP8_MAX
            self.register_buffer("w8t", (wf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).t())  # [in, out] col-major
            self.register_buffer("ws", s.reshape(()))
            self.register_buffer("xs", torch.tensor(xs, dtype=torch.float32, device=w.device))
            self.register_buffer("peak", torch.zeros((), dtype=torch.float32, device=w.device))
            self.out_features, self.in_features = w.shape

        def forward(self, x):
            x2 = x.reshape(-1, x.shape[-1]).float()
            if MONITOR["on"]:  # sampled forwards only (Scorer._run): a second graph, compiled once
                torch.maximum(self.peak, x2.abs().amax() / (self.xs * FP8_MAX), out=self.peak)
            x8 = (x2 / self.xs).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            y = torch._scaled_mm(x8, self.w8t, scale_a=self.xs, scale_b=self.ws, out_dtype=torch.bfloat16)
            return y.reshape(*x.shape[:-1], self.out_features)

    def prequantized(cls, w8, ws, xs=None):
        """Fp8StaticLinear / Fp8Linear from an fp8-static checkpoint's tensors (the same buffers the constructors compute from bf16)"""
        lin = cls.__new__(cls)
        nn.Module.__init__(lin)
        lin.register_buffer("w8t", w8.to(device, non_blocking=True).t())  # [in, out] col-major
        lin.register_buffer("ws", ws.to(device, torch.float32).reshape(()))
        if xs is not None:
            lin.register_buffer("xs", xs.to(device, torch.float32).reshape(()))
            lin.register_buffer("peak", torch.zeros((), dtype=torch.float32, device=device))
        lin.out_features, lin.in_features = w8.shape
        return lin

    return SimpleNamespace(
        Int8Linear=Int8Linear, Fp8Linear=Fp8Linear, Fp8RowLinear=Fp8RowLinear, Fp8StaticLinear=Fp8StaticLinear, prequantized=prequantized
    )
