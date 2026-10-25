"""The one entry point the adapters call (the HTTP server in spinf_serve.server, serverless handlers).

score_request: the scoring protocol (protocol.py, compiler.py).
score: the engine-internal v0 format (fully formed prompts and option strings), kept for research and the parity fixture.
"""

import logging
import os
import platform
import threading
import time
from importlib import metadata

from pydantic import BaseModel, Field, model_validator

from spinf_serve import protocol, weights
from spinf_serve.config import EngineConfig

log = logging.getLogger(__name__)

MAX_JOBS = 1000
# memory still allocated after a failed call's cleanup, above the idle level, that marks the engine degraded (service._recover)
LEAK_GIB = 1.0
EXIT_ON_LEAK = os.environ.get("SPINF_EXIT_ON_LEAK", "1").lower() in ("1", "true", "yes")


def is_oom(e: BaseException) -> bool:
    """a CUDA out-of-memory error, from torch's allocator or raised by a kernel (Triton, cuBLAS) as a RuntimeError"""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    s = str(e).lower()
    return "out of memory" in s or "cuda_error_out_of_memory" in s or "cublas_status_alloc_failed" in s


class JobIn(BaseModel):
    id: str = Field(max_length=128)
    prompts: list[str] = Field(min_length=1)
    options: list[list[str]] = Field(description="per prompt: the option strings, leading space included where one belongs")

    @model_validator(mode="after")
    def _shape(self) -> "JobIn":
        if len(self.options) != len(self.prompts):
            raise ValueError("options must have one list per prompt")
        if any(not o or any(not s for s in o) for o in self.options):
            raise ValueError("every prompt needs at least one non-empty option")
        return self


class ScoreRequest(BaseModel):
    jobs: list[JobIn] = Field(min_length=1, max_length=MAX_JOBS)


class JobOut(BaseModel):
    id: str
    logprobs: list[list[float]]
    packed_tokens: int
    trees: int


class ScoreResponse(BaseModel):
    results: list[JobOut]
    usage: dict[str, int]
    timing: dict[str, float]
    engine: dict[str, str | int]


class ScoreService:
    """owns the model; score() calls are serialised on the one GPU (jobs are packed within a call, never across calls, so a caller's
    scores do not depend on other callers' traffic; see ARCHITECTURE.md, determinism)"""

    def __init__(self, cfg: EngineConfig | None = None, device: str = "cuda:0"):
        self.cfg = cfg or EngineConfig()
        self.device = device
        self.scorer = None
        self.boot: dict[str, float] = {}
        self._lock = threading.Lock()
        self.base_gib: float | None = None  # GiB allocated on the card when idle (after warm-up): the leak check's reference
        # the names a request may call the model: the served name, plus the manifest's id and aliases when the config names one
        self.model_names = {self.cfg.served_model.lower()}
        if self.cfg.model:
            from spinf_serve import manifests

            self.model_names |= manifests.resolve(self.cfg.model).names
        self.degraded: str | None = None  # set when memory leaked after a failure: /readyz and /livez answer 503

    @property
    def ready(self) -> bool:
        return self.scorer is not None

    @property
    def manifest(self):
        """the served model's manifest (manifests.Manifest), None for a checkpoint served by its path"""
        if not self.cfg.model:
            return None
        from spinf_serve import manifests

        return manifests.resolve(self.cfg.model)

    def load(self) -> None:
        t0 = time.perf_counter()
        import torch  # noqa: F401  (import time is part of the cold start)

        from spinf_serve.scorer import Scorer

        self.boot["import_s"] = round(time.perf_counter() - t0, 1)
        t0 = time.perf_counter()
        self.scorer = Scorer(self.cfg, self.device)
        self.boot["load_s"] = round(time.perf_counter() - t0, 1)
        self.boot["weights"] = self.scorer.weights_source

    def warmup(self) -> None:
        """Compile the kernels before the first real request (EngineConfig.warmup; `boot["warm"]` marks the end: the server reports
        ready only then, so no request ever waits behind a compile). Scoring: several stream lengths, because torch.compile
        specialises on the first shape it sees and compiles the dynamic-shape graph on the second (without it the first real
        request took 3.5 s), and with "full" one stream at the token budget (the largest shape, the flex mask over every block).
        Then the extraction decode path, whose first compile is the long one (~1-1.5 min on the 12B / 4090)."""
        from spinf_serve.scorer import Job

        if self.cfg.warmup == "none":
            self.boot["warm"] = True
            self._idle_level()
            return
        t0 = time.perf_counter()
        text = "The quick brown fox jumps over the lazy dog. "
        reps = [1, 40, 400]
        if self.cfg.warmup == "full":  # one job filling the budget: repetitions of the sentence up to the budget less a block
            per = len(self.scorer.tok(text, add_special_tokens=False)["input_ids"])
            reps.append(max(401, (self.scorer.budget - 256) // per))
        # fp8s with the (diagnostic) peak monitor on: its forwards are a second graph (quant/linears.py MONITOR), compile both
        for force in (True, False) if self.scorer.quant == "fp8s" and self.cfg.fp8s_monitor_every else (None,):
            self.scorer.monitor_force = force
            for n in reps:
                prompts = [text * n + "It is", text * n + "The fox was"]
                self.scorer.score([Job(f"warmup{n}", prompts, [[" fast", " slow"], [" quick", " lazy"]])])
        self.scorer.monitor_force = None
        self.boot["warmup_s"] = round(time.perf_counter() - t0, 1)
        if self.scorer.readout:  # a decision model reads options only: no extraction (score_request refuses the slots)
            self.boot["warm"] = True
            self._idle_level()
            return
        # the extraction decode path: one tree and two (the compiler specialises a batch of one), text and number slots,
        # and the vocabulary tables; without it the first extraction request waits ~20 s
        t0 = time.perf_counter()
        from spinf_serve import protocol

        for n in (1, 2):
            self.score_request(
                protocol.ScoreRequest.model_validate(
                    {
                        "model": self.cfg.served_model,
                        "inputs": [
                            {"id": f"w{k}", "messages": [{"role": "user", "content": f"The fox ran {k + 3} miles to Paris. " * 3}]}
                            for k in range(n)
                        ],
                        "scoring": {
                            "queries": [
                                {"id": "city", "template": '\nThe city named is "{*}"'},
                                {"id": "miles", "template": "\nQ: How many miles?\nA:{n}"},
                            ]
                        },
                    }
                )
            )
        self.boot["warmup_extract_s"] = round(time.perf_counter() - t0, 1)
        self.boot["warm"] = True
        self._idle_level()

    def _idle_level(self) -> None:
        """record the memory the loaded, warm engine holds between calls (weights, tables, compiled kernels' workspaces)"""
        torch = self.scorer.torch
        torch.cuda.synchronize(self.scorer.device)
        torch.cuda.empty_cache()
        self.base_gib = torch.cuda.memory_allocated(self.scorer.device) / 2**30
        self.boot["idle_gib"] = round(self.base_gib, 2)

    def fingerprint(self) -> str:
        """the public system_fingerprint: opaque (no hardware, precision or build details), and it changes whenever anything that
        changes the scores does (weights revision, scheme, lm head, engine build, GPU architecture). The engine logs what it stands
        for at start."""
        import hashlib

        if getattr(self, "_fingerprint", None) is None:
            i = self.info()
            # the GPU architecture too: cards of one architecture score alike, across architectures they do not (hashed, so the
            # card is not disclosed)
            key = "|".join(str(i[k]) for k in ("model", "revision", "quant", "image", "engine", "capability")) + f"|{self.scorer.lm_head}"
            if i.get("scale_table"):  # fp8s: the activation scales are part of the numerics
                key += f"|{i['scale_table']}"
            self._fingerprint = "fp_" + hashlib.sha256(key.encode()).hexdigest()[:10]
            log.info("system_fingerprint %s = %s", self._fingerprint, key)
        return self._fingerprint

    def info(self) -> dict[str, str | int]:
        """what produced a score: enough to reproduce it (weights revision, scheme, card, stack, image)"""
        import torch
        import transformers

        s = self.scorer
        props = torch.cuda.get_device_properties(s.device)
        return {
            "engine": metadata.version("spinf-serve"),
            "image": os.environ.get("GIT_SHA", "unknown"),
            "model": self.cfg.model_id,
            "revision": weights.revision(s.model_dir),
            "quant": s.quant,
            **({"scale_table": s.model._scale_table} if s.model._scale_table else {}),
            "token_budget": s.budget,
            "max_tokens": s.max_tokens,
            "gpu": props.name,
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "",
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        }

    def score_request(self, req: "protocol.ScoreRequest", pack_groups: frozenset[str] = frozenset()) -> dict:
        """a protocol request (protocol.ScoreRequest) -> {results, warnings, usage, detail, system_fingerprint, timing, engine}: the
        engine-internal shape (the server builds the public envelope around it). pack_groups: the query groups of the call's packs
        (compiler.compile_request). Raises protocol.Rejected before any GPU work when a check fails"""
        from spinf_serve import compiler, protocol

        if self.scorer is None:
            raise RuntimeError("model not loaded")
        if req.model.lower() not in self.model_names:
            raise protocol.Rejected("model_not_served", f"This engine serves {self.cfg.served_model}, not {req.model}.", "model")
        media = self.scorer.model._media
        if self.scorer.readout and req.limits.max_extractions_per_input:
            req = req.model_copy(update={"limits": req.limits.model_copy(update={"max_extractions_per_input": 0})})
        tc = time.perf_counter()
        c = compiler.compile_request(
            req,
            self.scorer.tok,
            self.scorer.budget,
            media.ids if media is not None else None,
            self.scorer.max_tokens,
            self.cfg.x_batched,
            pack_groups,
        )
        compile_s = time.perf_counter() - tc  # parsing, media decoding, tokenization: on the CPU, before the GPU lock
        torch = self.scorer.torch
        shape = (
            f"{len(req.inputs)} inputs, {c.usage['content_tokens']} content tokens ({c.usage['image_tokens'] + c.usage['video_tokens']} "
            f"in {c.usage['images'] + c.usage['video_frames']} images), {c.processed_tokens} processed, {self._n_slots(c)} slots"
        )
        tq = time.perf_counter()
        # the device is thread-local in torch: a server thread that did not load the model still has cuda:0 current, and the Triton
        # kernels launch on the current device ("Pointer argument cannot be accessed from Triton" on a second GPU) -> set it here
        with self._lock, torch.cuda.device(self.scorer.device):
            queue_s = time.perf_counter() - tq
            st = self.scorer.stats
            tok0, busy0 = st.tokens, st.busy_s
            t0 = time.perf_counter()
            self.scorer.x_found, self.scorer.x_busy_s, self.scorer.x_chunks = {}, 0.0, 0
            torch.cuda.reset_peak_memory_stats(self.scorer.device)
            oom = False
            try:
                scores = self.scorer.score_grouped(c.items) if c.items else {}
                t1 = time.perf_counter()
                extracted = self.scorer.x_found
                if c.extract_jobs:  # templates that merge with their content's end: read and decoded one by one
                    extracted.update(self.scorer.extract(c.extract_jobs))
            except Exception as e:
                if not is_oom(e):
                    raise
                peak = torch.cuda.max_memory_allocated(self.scorer.device) / 2**30
                log.warning("score: out of memory on %s at %.2f GiB:", shape, peak, exc_info=True)
                oom = True
            if oom:  # outside the except block: the exception's traceback kept the failed frames' tensors alive
                self._recover(shape)  # frees the call's memory, or marks the engine degraded; raises Rejected
            wall = time.perf_counter() - t0
            x_wall = (time.perf_counter() - t1) + self.scorer.x_busy_s
            packed, busy = st.tokens - tok0, st.busy_s - busy0
            peak = torch.cuda.max_memory_allocated(self.scorer.device) / 2**30
            reserved = torch.cuda.memory_reserved(self.scorer.device) / 2**30
            self._check_peak()
        out = compiler.assemble(c, req, scores, extracted)
        info = self.info()
        gen_keys = {k for k, v in extracted.items() if v.gen}
        generated = sum(f.generated for k, f in extracted.items() if k in gen_keys)
        decode = sum(f.generated for k, f in extracted.items() if k not in gen_keys)
        out["usage"] = {**c.usage, "generated_tokens": generated, "packed_tokens": packed}
        detail = dict(c.detail)
        detail["extraction"] = {**detail["extraction"], "answers": sum(len(f.answers) for f in extracted.values()), "decode_tokens": decode}
        detail["generation"] = {**detail["generation"], "generated_tokens": generated}
        detail["processed_tokens"] = c.processed_tokens
        out["detail"] = detail
        out["system_fingerprint"] = self.fingerprint()
        out["timing"] = {
            "wall_s": round(wall, 3),
            "gpu_busy_s": round(busy, 3),
            "compile_s": round(compile_s, 3),
            "queue_s": round(queue_s, 3),
            "score_s": round(wall - x_wall, 3) if extracted else round(wall, 3),
        }
        if extracted:
            out["timing"]["extract_wall_s"] = round(x_wall, 3)
            out["timing"]["decode_chunks"] = self.scorer.x_chunks
        out["memory"] = {"peak_gib": round(peak, 2), "reserved_gib": round(reserved, 2)}
        out["engine"] = info
        log.info("score: %s: %.2f GiB peak (%.2f reserved), %.2f s, %d decode chunks", shape, peak, reserved, wall, self.scorer.x_chunks)
        return out

    @staticmethod
    def _n_slots(c) -> int:
        """the extraction slots of a compiled request (decoded branches plus the jobs read one by one)"""
        n = sum(len(t.xbranches) for _, _, t in c.items if getattr(t, "xbranches", None))
        return n + sum(len(j.prompts) for j in c.extract_jobs)

    def _recover(self, shape: str) -> None:
        """after an out-of-memory error: drop what the call held (Scorer.abandon ran in the GPU loop), release the allocator's
        cached blocks and check the card is back at its idle level. When it is, the call is refused as too large for this engine
        (the API answers 413 request_too_large, not charged) and the engine keeps serving. When memory stays held (a leak the
        cleanup does not reach), the engine marks itself degraded: /readyz and /livez answer 503 so routers stop sending calls,
        and the process exits so the container restarts with a clean card (SPINF_EXIT_ON_LEAK=0 keeps it up, degraded)"""
        import gc

        torch = self.scorer.torch
        self.scorer.abandon()
        gc.collect()
        torch.cuda.synchronize(self.scorer.device)
        torch.cuda.empty_cache()
        held = torch.cuda.memory_allocated(self.scorer.device) / 2**30
        base = self.base_gib or 0.0
        leaked = held - base
        log.warning("score: %.2f GiB held after the cleanup (idle %.2f): %s", held, base, shape)
        if leaked > LEAK_GIB:
            self.degraded = f"{leaked:.2f} GiB not freed after an out-of-memory error"
            log.error("engine degraded: %s; %s", self.degraded, "exiting for a restart" if EXIT_ON_LEAK else "staying up")
            if EXIT_ON_LEAK:
                threading.Timer(2.0, lambda: os._exit(70)).start()
        raise protocol.Rejected(
            "request_too_large",
            "The request did not fit this engine's GPU memory. Send less content, fewer or smaller images, or fewer extraction slots.",
        )

    def _check_peak(self) -> None:
        """fp8s: warn when a linear's input went past its clip level for the first time (and at each new high above 1)"""
        p = self.scorer.fp8s_peak()
        if p is None or p[0] <= max(1.0, getattr(self, "_peak_warned", 1.0)):
            return
        self._peak_warned = p[0]
        log.warning(
            "fp8s: inputs clipped, %s reached %.2fx its scale table's range (recalibrate: python -m spinf_serve.calibrate)", p[1], p[0]
        )

    def score(self, req: ScoreRequest) -> ScoreResponse:
        from spinf_serve.scorer import Job

        if self.scorer is None:
            raise RuntimeError("model not loaded")
        jobs = [Job(j.id, j.prompts, j.options) for j in req.jobs]
        with self._lock, self.scorer.torch.cuda.device(self.scorer.device):
            st = self.scorer.stats
            tok0, fwd0, busy0 = st.tokens, st.forwards, st.busy_s
            t0 = time.perf_counter()
            res = self.scorer.score(jobs)
            wall = time.perf_counter() - t0
            usage = {"packed_tokens": st.tokens - tok0, "forwards": st.forwards - fwd0}
            timing = {"wall_s": round(wall, 3), "gpu_busy_s": round(st.busy_s - busy0, 3)}
        return ScoreResponse(
            results=[
                JobOut(id=j.id, logprobs=res[j.id].logprobs, packed_tokens=res[j.id].packed_tokens, trees=res[j.id].trees) for j in jobs
            ],
            usage=usage,
            timing=timing,
            engine=self.info(),
        )
