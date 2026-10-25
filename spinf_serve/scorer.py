"""One GPU, one model: jobs in, option log-probabilities out.

A job is one piece of content asked several prompts: each prompt is the full text up to the scored slot, each option the exact string
that may follow it (leading space included when the option follows a word). The read of an option is log P(option | prompt), the sum
of its per-token log-probs; everything else (null-body correction, normalisation over the options, templates) belongs to the caller.

Origin: spinf's internal scorer (torchscore.py: Scorer, flex stream path, single device).
"""

import copy
import math
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from spinf_serve.config import EngineConfig
from spinf_serve.extract import ExtractJob
from spinf_serve.media import Image
from spinf_serve.models.base import compile_options
from spinf_serve.packer import BLOCK, Batch, GroupedTree, Packer, Tree, block_tables, split_trees
from spinf_serve.weights import find_weights

MEDIA_CACHE_GIB = 0.5  # media embeddings kept for a call (Scorer._media_embeds)


@dataclass
class Job:
    id: str
    prompts: list[str]
    options: list[list[str]]  # per prompt: the option strings

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.options):
            raise ValueError(f"job {self.id}: {len(self.prompts)} prompts but {len(self.options)} option lists")


@dataclass
class JobResult:
    logprobs: list[list[float]]  # per prompt, per option: log P(option | prompt)
    packed_tokens: int = 0  # tokens actually run (shared prefixes once per tree)
    trees: int = 0  # >1 when the prompts did not fit one budget and the prefix was sent again


@dataclass
class Stats:
    tokens: int = 0
    forwards: int = 0
    busy_s: float = 0.0  # CUDA-event time of the forwards
    wall_s: float = 0.0
    _events: list = field(default_factory=list, repr=False)


def read_grouped(lp: np.ndarray, cols: list[int], b: Batch, out: dict[object, list]) -> None:
    """one batch's option log-probs lp [read rows, len(cols)] -> out[key][prompt] = per-option log-probs of each grouped job's
    prompts (the sum over each option's tokens, as its ReadPlan says)"""
    cols_arr = np.asarray(cols, np.int64)
    colidx: dict[int, np.ndarray] = {}
    k = 0
    for o in b.owners:
        key, first = o.job_id
        for c, plan in enumerate(o.plans or ()):
            ci = colidx.get(id(plan))
            if ci is None:
                ci = colidx[id(plan)] = np.searchsorted(cols_arr, plan.entry_tok)
            vals = lp[k + o.row_start[c] + plan.entry_row, ci].astype(np.float64)
            out[key][first + c] = np.bincount(plan.entry_opt, weights=vals, minlength=plan.n_opts)
        k += o.n_rows


def readout_logprobs(torch, h, R, ids, temperature: float, vocab: int):
    """a decision model's log-probs: the hidden rows [S, hidden] times the readout [codes, hidden] over its answer-code tokens
    `ids`, divided by the fitted temperature, log-softmax over the codes (every other token -inf): the decider's own softmax over
    a question's options falls out when the options are renormalised (pplx-decider: logits / T, masked softmax)"""
    # the lm head's f32out GEMM on the card; the plain float product off it (tests)
    logits = torch.mm(h, R.t(), out_dtype=torch.float32) if h.is_cuda else h.float() @ R.float().t()
    logits = logits / temperature
    full = torch.full((h.shape[0], vocab), float("-inf"), dtype=torch.float32, device=h.device)
    full[:, ids] = logits
    return torch.log_softmax(full, dim=-1)


class Scorer:
    def __init__(self, cfg: EngineConfig | None = None, device: str = "cuda:0", stack=None):
        """stack: a models.base.Stack built by the caller (research: options with tensors), else the family's from the manifest or
        the checkpoint (models/registry.py)"""
        import torch
        from transformers import AutoTokenizer

        self.cfg = cfg = cfg or EngineConfig()
        self.torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.model_dir, self.weights_source = (cfg.model_dir, "config") if cfg.model_dir else find_weights(cfg.model_id, cfg.revision)
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        from spinf_serve import media
        from spinf_serve.models.registry import stack_for

        family = ""
        if cfg.model:
            from spinf_serve import manifests

            family = manifests.resolve(cfg.model).family
        self.stack = stack or stack_for(
            self.model_dir, family
        )  # the family's behaviour (models/base.py): build, forward, captures, decoder
        self.hybrid = self.stack.hybrid  # Gated DeltaNet + full attention (qwen.py): run along TreePlans
        media.SCHEME["image"] = self.stack.image_scheme  # the image preprocessing the protocol applies (one model per process)
        self.model, self.tcfg = self.stack.build(cfg, self.model_dir, self.device)
        if cfg.compile:
            for layer in self.model.layers:
                layer.compile(dynamic=True, options=compile_options(cfg))
        # the shortest aligned stream: the model's own floor where its GEMMs change kernel below some M (qwen.py)
        self.min_stream = max(cfg.min_stream, getattr(self.model, "_min_stream", 0))
        pk = Packer(cfg.token_budget, cfg.max_tokens)
        self.budget, self.max_tokens = pk.budget, pk.max_tokens
        self._init_flex(self.max_tokens)
        # pinned host staging for the per-forward inputs, double-buffered: asynchronous copies keep the host from waiting on the GPU
        # queue, and a per-forward pin_memory() failed in the driver when the page cache had eaten the host's free memory
        n = self.max_tokens
        dts = (("ids", torch.int64), ("pos", torch.int64), ("art", torch.int32), ("end", torch.int32), ("pos32", torch.int32))
        self._stage = [{k: torch.empty(n, dtype=dt).pin_memory() for k, dt in dts} for _ in range(2)]
        self._stage_ev = [None, None]
        self._stage_i = 0
        self.softcap = getattr(self.tcfg, "final_logit_softcapping", None)
        self.lm_head = cfg.lm_head
        self._opt_ids: dict[str, list[int]] = {}
        self.stats = Stats()
        self._vocab = None  # extraction: token texts, special ids, stop sets, number candidates (built on first use)
        self.x_found: dict[object, object] = {}  # extraction answers decoded after scoring passes: prompt key -> extract.Found
        self.x_busy_s = 0.0  # the time of those decodes (inside the scoring passes)
        self.x_chunks = 0  # decodes run for the call (_x_flush: one per chunk of groups under the decoder budget)
        self._x_pending: list = []  # decode.Group waiting for the call's decode
        # what a forward captures for its extraction branches (_x_tree_bytes, the packer's second dimension): K/V bytes per
        # token over the layers that store it (upper bounds: sliding layers store less); on the hybrid also the recurrent rows
        # per template token and the prefix state per group (Stack.capture_bytes)
        self._x_tok_bytes, self._x_row_bytes, self._x_state_bytes = self.stack.capture_bytes(self.model)
        self._x_used = 0  # their bytes

    @property
    def quant(self) -> str:
        return self.model._quant

    # ------------------------------------------------------------ tokenization
    def option_ids(self, option: str) -> list[int]:
        ids = self._opt_ids.get(option)
        if ids is None:
            ids = self._opt_ids[option] = self.tok(option, add_special_tokens=False)["input_ids"]
        return ids

    def trees(self, job: Job) -> list[tuple[int, Tree]]:
        prompt_ids = self.tok(job.prompts, add_special_tokens=True)["input_ids"]
        option_ids = [[self.option_ids(o) for o in opts] for opts in job.options]
        try:
            return split_trees(prompt_ids, option_ids, self.budget)
        except ValueError as e:
            raise ValueError(f"job {job.id}: {e}") from None

    # ------------------------------------------------------------ attention masks
    def _init_flex(self, n: int) -> None:
        """stream metadata buffers (sized to the budget) and the mask functions over them: token i attends to j when j <= i, same job,
        and i lies in j's subtree (i < end[j]: j is the prefix, an ancestor branch or i's own branch); pads attend to themselves.
        Sliding layers add pos[i] - pos[j] < window."""
        torch = self.torch
        W = getattr(self.tcfg, "sliding_window", None)
        art = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        end = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        pos = torch.zeros(n, dtype=torch.int32, device=self.device)

        def full(b, h, q, k):
            return ((k <= q) & (art[q] >= 0) & (art[k] == art[q]) & (q < end[k])) | (q == k)

        def sliding(b, h, q, k):
            return full(b, h, q, k) & (pos[q] - pos[k] < W)

        # batches with images: an image's placeholders also attend to each other both ways (use_bidirectional_attention =
        # "vision"); separate mask functions, so text-only batches keep the kernels above unchanged. Which layers, as the
        # checkpoint's own forward: the 12B "unified" on every layer (Gemma4UnifiedModel.forward: the block overlay on both
        # masks, and past the window); the 31B (gemma4) on the sliding layers only and within the window, the global layers
        # stay causal (Gemma4Model.forward -> create_masks_for_vision_model)
        grp = torch.full((n,), -1, dtype=torch.int32, device=self.device)

        if getattr(self.model, "_vision_bidir_all", True):

            def full_img(b, h, q, k):
                return full(b, h, q, k) | ((grp[q] == grp[k]) & (grp[q] >= 0))

            def sliding_img(b, h, q, k):
                return sliding(b, h, q, k) | ((grp[q] == grp[k]) & (grp[q] >= 0))

        else:
            full_img = full

            def sliding_img(b, h, q, k):
                return sliding(b, h, q, k) | ((grp[q] == grp[k]) & (grp[q] >= 0) & (pos[q] - pos[k] < W))

        self._bufs = (art, end, pos)
        self._grp = grp
        self._mods = {"full_attention": full, "sliding_attention": sliding}
        self._mods_img = {"full_attention": full_img, "sliding_attention": sliding_img}
        self._media_cache: dict[int, object] = {}
        self._media_bytes = 0

    def _block_masks(self, b: Batch, st: dict):
        torch = self.torch
        from torch.nn.attention.flex_attention import BlockMask

        n = b.length
        grp = b.grp if self.stack.bidirectional_images else None  # Qwen's image tokens attend causally, as text
        tabs = block_tables(b.art, b.end, b.pos, getattr(self.tcfg, "sliding_window", None), grp, self.cfg.align_jobs)
        mods = self._mods if grp is None else self._mods_img
        if grp is not None:
            self._grp[:n].copy_(torch.from_numpy(grp).pin_memory(), non_blocking=True)
        for key, arr in (("art", b.art), ("end", b.end), ("pos32", b.pos)):
            st[key][:n].copy_(torch.from_numpy(arr))
        for buf, key in zip(self._bufs, ("art", "end", "pos32"), strict=True):
            buf[:n].copy_(st[key][:n], non_blocking=True)
        pinned = {
            "full_attention": [torch.from_numpy(t).pin_memory() for t in tabs[0]],
            "sliding_attention": [torch.from_numpy(t).pin_memory() for t in (tabs[1] or tabs[0])],
        }
        out = {}
        for name, ts in pinned.items():
            pn, pi, fn, fi = [t.to(self.device, non_blocking=True) for t in ts]
            out[name] = BlockMask.from_kv_blocks(
                pn[None, None],
                pi[None, None],
                fn[None, None],
                fi[None, None],
                BLOCK_SIZE=BLOCK,
                mask_mod=mods[name],
                seq_lengths=(n, n),
                compute_q_blocks=False,
            )
        return out

    # ------------------------------------------------------------ forward
    def _forward(self, b: Batch, cols: list[int], xpos: list[int] | None = None):
        """batch -> (log-probs [n_sel, len(cols)] fp32 on the GPU, full log-probs [len(xpos), vocab] at the extraction branches'
        ends or None); the lm head runs over lm_rows read rows at a time"""
        torch = self.torch
        dev = self.device
        H = self._run(b)
        with torch.no_grad():
            rows = self.cfg.lm_rows
            Hf = H.reshape(-1, H.shape[-1])
            if len(b.sel):
                sel = torch.from_numpy(b.sel).pin_memory().to(dev, non_blocking=True)
                h = Hf[sel]
                ci = torch.tensor(cols, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
                lp = torch.cat([self._logprobs_rows(h[s : s + rows])[:, ci] for s in range(0, h.shape[0], rows)])
            else:
                lp = torch.zeros((0, len(cols)), dtype=torch.float32, device=dev)
            xlp = None
            if xpos:
                hx = Hf[torch.tensor(xpos, device=dev)]
                xlp = torch.cat([self._logprobs_rows(hx[s : s + rows]) for s in range(0, hx.shape[0], rows)])
            return lp, xlp

    def _run(self, b: Batch):
        """batch -> final hidden states [1, N, hidden] (the packed flex forward)"""
        torch = self.torch
        st = self._stage[self._stage_i]
        ev = self._stage_ev[self._stage_i]
        if ev is not None:
            ev.synchronize()  # the copies issued from this buffer two forwards ago are done
        if self.quant == "fp8s":
            from spinf_serve.quant.linears import MONITOR

            k = self.cfg.fp8s_monitor_every
            force = getattr(self, "monitor_force", None)  # ScoreService.warmup compiles both variants
            MONITOR["on"] = force if force is not None else bool(k) and self.stats.forwards % k == 0
        inputs = self.stack.stage(self, b, st)
        masks = self._block_masks(b, st)
        self._stage_ev[self._stage_i] = torch.cuda.Event()
        self._stage_ev[self._stage_i].record()
        self._stage_i ^= 1
        return self.stack.forward(self, b, inputs, masks)

    def _media_embeds(self, obj):
        """media.Image | media.Audio -> [n, hidden] bf16 on the GPU, cached for the call (a prefix re-sent in several streams) up
        to MEDIA_CACHE_GIB: a call with many images (96 auto-detail images are 1 GB of embeddings) would otherwise hold them all
        next to the forwards and decodes; past the bound the cache starts over (a re-sent prefix's images are encoded again)"""
        e = self._media_cache.get(id(obj))
        if e is None:
            m = self.model._media
            if m is None:
                raise RuntimeError("this engine was loaded without media support")
            e = m.image(obj) if isinstance(obj, Image) else m.audio(obj)
            n = e.numel() * e.element_size()
            if self._media_cache and self._media_bytes + n > MEDIA_CACHE_GIB * 2**30:
                self._media_clear()
            self._media_cache[id(obj)] = e
            self._media_bytes += n
        return e

    def _media_clear(self) -> None:
        self._media_cache.clear()
        self._media_bytes = 0

    @property
    def media_ok(self) -> bool:
        return self.model._media is not None

    def fp8s_peak(self) -> tuple[float, str] | None:
        """fp8s with the monitor on: (the largest input / clip level over the linears since load, that linear's name), else None"""
        if self.quant != "fp8s" or not self.cfg.fp8s_monitor_every:
            return None
        if getattr(self, "_peak_mods", None) is None:
            self._peak_mods = [(n, m) for n, m in self.model.named_modules() if hasattr(m, "peak")]
        v = self.torch.stack([m.peak for _, m in self._peak_mods])
        k = int(v.argmax())
        return float(v[k]), self._peak_mods[k][0]

    @property
    def vocab_n(self) -> int:
        """the vocabulary the log-probs span (the LM head's rows; a readout model: the embedding's rows)"""
        W = getattr(self.model, "_lm_weight", None)
        if W is not None:
            return W.shape[0]
        emb = getattr(self.model, "embed_host", None)
        return emb.shape[0] if emb is not None else self.model.embed_tokens.weight.shape[0]

    def _logprobs_rows(self, h):
        """_logprobs over one lm_rows chunk of a forward's read rows; with lm_fixed_rows a short chunk is padded to lm_rows, so the
        lm head is always the same GEMM shape (a bf16 GEMM's rows change with its M: batch invariance, BATCH_INVARIANCE.md)"""
        n, rows = h.shape[0], self.cfg.lm_rows
        if not self.cfg.lm_fixed_rows or n == rows:
            return self._logprobs(h)
        return self._logprobs(self.torch.nn.functional.pad(h, (0, 0, 0, rows - n)))[:n]

    @property
    def readout(self) -> bool:
        """a decision model: options are scored by its readout over the answer codes, not by an LM head (qwen.load_readout)"""
        return getattr(self.model, "_readout", None) is not None

    def _logprobs(self, h):
        """read rows [S, hidden] -> log-probs [S, vocab] fp32, final soft-capping included; self.lm_head (EngineConfig.lm_head,
        switchable at run time for comparisons) sets the precision of the logits"""
        torch = self.torch
        if self.readout:
            return readout_logprobs(torch, h, self.model._readout, self.model._readout_ids, self.model._readout_temperature, self.vocab_n)
        W = self.model._lm_weight
        if self.lm_head == "f32out":
            logits = torch.mm(h, W.t(), out_dtype=torch.float32)
        elif self.lm_head == "fp32":
            logits = h.float() @ W.float().t()
        else:
            logits = h @ W.t()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.log_softmax(logits.float(), dim=-1)

    def _readback(self, small):
        torch = self.torch
        host = torch.empty(small.shape, dtype=small.dtype).pin_memory()
        host.copy_(small, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(small.device))
        return host, ev

    # ------------------------------------------------------------ public
    def score(self, jobs: list[Job]) -> dict[str, JobResult]:
        """score the jobs (packed together) -> {job id: JobResult}. Tokenization and packing run in a thread ahead of the GPU."""
        results = {j.id: JobResult([[0.0] * len(o) for o in j.options]) for j in jobs}
        if len(results) != len(jobs):
            raise ValueError("job ids must be unique within a call")

        def batches():
            packer = Packer(self.cfg.token_budget, self.cfg.max_tokens, self.cfg.align_jobs, self.min_stream)
            for j in jobs:
                for first, tree in self.trees(j):
                    results[j.id].trees += 1
                    yield from packer.push((j.id, first), tree)
            yield from packer.flush()

        self._execute(batches, lambda item: self._drain(item, results))
        return results

    def score_grouped(self, items: list[tuple[object, int, "GroupedTree"]]) -> dict[object, list[np.ndarray | None]]:
        """grouped jobs (spinf_serve.compiler): items = (key, first prompt index, tree), several items may share a key (a job
        split over the budget) -> {key: per prompt, the log-probs of its plan's options (float64 array)}. Trees with extraction
        branches are decoded right after the forward that read them (decode.py): their answers land in self.x_found."""
        out: dict[object, list[np.ndarray | None]] = {}
        for key, first, tree in items:
            slots = out.setdefault(key, [])
            need = first + len(tree.plans)
            if len(slots) < need:
                slots.extend([None] * (need - len(slots)))

        cap = getattr(self.model, "_capture_budget_gib", None) or self.cfg.capture_budget_gib

        def batches():
            packer = Packer(
                self.cfg.token_budget,
                self.cfg.max_tokens,
                self.cfg.align_jobs,
                self.min_stream,
                cost=self._x_tree_bytes,
                cost_budget=int(cap * 2**30),
            )
            for key, first, tree in items:
                yield from packer.push((key, first), tree)
            yield from packer.flush()

        def drain(item):
            (host, ev), cols, b = item
            ev.synchronize()
            read_grouped(host.numpy(), cols, b, out)

        try:
            self._execute(batches, drain)
        finally:
            self._media_clear()
        return out

    @staticmethod
    def _cols(b: Batch) -> list[int]:
        """the option columns a batch reads (only these leave the lm head)"""
        cols = {t for o in b.owners for _, t, _, _ in o.plan}
        seen = set()
        for o in b.owners:
            for plan in o.plans or ():
                if id(plan) not in seen:
                    seen.add(id(plan))
                    cols.update(plan.entry_tok.tolist())
        return sorted(cols)

    def _execute(self, batches, drain) -> None:
        """the GPU loop: batches() runs in a thread ahead of the GPU (tokenized, packed); each forward's option log-probs are
        read back one forward later and handed to drain((host, event), cols, batch). A batch with extraction branches also
        captures their K/V (Stack.capture); they are decoded together at the end (_x_collect / _x_flush)."""
        from contextlib import nullcontext

        torch = self.torch
        q: queue.Queue = queue.Queue(self.cfg.prefetch)
        stop = threading.Event()  # set when the GPU loop fails: the producer drops what it has instead of blocking on a full queue

        def put(x) -> bool:
            while not stop.is_set():
                try:
                    q.put(x, timeout=0.2)
                    return True
                except queue.Full:
                    pass
            return False

        def producer():
            try:
                for b in batches():
                    if stop.is_set():
                        return
                    self.stack.plan(self, b)  # the hybrid's recurrent plan, on the host ahead of the GPU
                    if not put(b):
                        return
            except BaseException as e:
                put(e)
            put(None)

        threading.Thread(target=producer, daemon=True).start()
        t0 = time.perf_counter()
        pending = []
        try:
            while True:
                b = q.get()
                if b is None:
                    break
                if isinstance(b, BaseException):
                    raise b
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                cols = self._cols(b)
                e0.record()
                xo = [o for o in b.owners if o.xbranches]
                xpos = [bs + bl - 1 for o in xo for bs, bl, _ in o.xbranches]
                with self.stack.capture(self, b, xo) if xo else nullcontext() as cap:
                    lp, xlp = self._forward(b, cols, xpos if xo else None)
                if xo:
                    self._x_collect(xo, cap, xlp)
                e1.record()
                self.stats._events.append((e0, e1))
                pending.append((self._readback(lp), cols, b))
                self.stats.tokens += b.n_tokens
                self.stats.forwards += 1
                if len(pending) > 1:
                    drain(pending.pop(0))
            for item in pending:
                drain(item)
            self._x_flush()
        except BaseException:
            # a failed call (an out-of-memory error in a forward or a decode) must not keep its GPU memory: the batches the
            # producer prefetched (their plans live on the card), the decode groups held for the call's decode and the
            # media embeddings would otherwise stay referenced until the process ends, and every later call would fail
            stop.set()
            pending.clear()
            b = cap = xlp = lp = None  # noqa: F841
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            self.abandon()
            raise
        torch.cuda.synchronize(self.device)
        self.stats.busy_s += sum(a.elapsed_time(c) for a, c in self.stats._events) / 1000
        self.stats._events = []
        self.stats.wall_s += time.perf_counter() - t0

    def abandon(self) -> None:
        """drop everything a call holds on the card between its forwards (used when a call fails): the decode groups waiting for
        the call's decode, a capture left open, the media embeddings and the timing events"""
        self._x_pending, self._x_used = [], 0
        self.stack.abandon(self)
        self._media_clear()
        self.stats._events = []
        self.x_found = {}

    @staticmethod
    def _drain(item, results: dict[str, JobResult]) -> None:
        (host, ev), cols, b = item
        ev.synchronize()
        lp = host.numpy()
        col = {c: i for i, c in enumerate(cols)}
        k = 0
        for o in b.owners:
            jid, first = o.job_id
            res = results[jid]
            res.packed_tokens += o.n_tokens
            acc = res.logprobs
            for r, t, c, side in o.plan:
                acc[first + c][side] += float(lp[k + r, col[t]])
            k += o.n_rows

    # ------------------------------------------------------------ extraction (spinf_serve.extract)
    @contextmanager
    def _eager_sdpa(self, impl: str = "sdpa", compiled: bool = False):
        """decode steps: the decoder layers uncompiled and attention through `impl` (sdpa over a DynamicCache, transformers
        building the masks; or decode.py's x_decode); the packed flex path is restored after"""
        layers = list(self.model.layers)
        saved = [getattr(m, "_compiled_call_impl", None) for m in layers]
        before = self.tcfg._attn_implementation
        try:
            if not compiled:
                for m in layers:
                    m._compiled_call_impl = None
            self.tcfg._attn_implementation = impl
            yield
        finally:
            for m, c in zip(layers, saved, strict=True):
                m._compiled_call_impl = c
            self.tcfg._attn_implementation = before

    def _vocab_tables(self):
        """token texts and the sets the searches need, built once (decoding the vocabulary takes a few seconds)"""
        if self._vocab is None:
            from spinf_serve import extract as ex

            torch = self.torch
            V = self.vocab_n
            texts = [self.tok.decode([i]) for i in range(V)]
            eos = {self.tok.eos_token_id}
            for t in ("<end_of_turn>", "<turn|>", "<|im_end|>", "<|endoftext|>"):
                i = self.tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i != self.tok.unk_token_id:
                    eos.add(i)
            eos.discard(None)
            special = set(self.tok.all_special_ids) | {i for i, s in enumerate(texts) if s.startswith("<") and s.endswith(">") and "|" in s}
            special -= eos
            cand = ex.number_candidates(texts, special)
            self._vocab = {
                "texts": texts,
                "eos": eos,
                "special": torch.tensor(sorted(special), device=self.device),
                "cand": cand,
                "cand_t": torch.tensor(cand, device=self.device),
                "cand_txt": [texts[i] for i in cand],
                "stops": {},
                "memo": {},
            }
        return self._vocab

    def prefill(self, prefix: list[int], media: list):
        """the prefix read once through the packed flex path (media, bidirectional images and the sliding window as scoring does),
        its keys and values captured into a transformers DynamicCache -> (cache, log-probs of the token after the prefix [1, V])"""
        from spinf_serve.packer import GroupedTree

        torch = self.torch
        P = len(prefix)
        tree = GroupedTree(list(prefix), [], [], [], media=list(media))
        b = Packer(self.cfg.token_budget, self.cfg.max_tokens)._pack([(("prefill",), tree)], (P + BLOCK - 1) // BLOCK * BLOCK)
        try:
            with self.stack.capture_ranges(self, [(0, P, "prefill")]) as cap:
                H = self._run(b)
        finally:
            self._media_clear()
        kv = cap["kv"].get("prefill", {})
        with torch.no_grad():
            lp = self._logprobs(H[0, P - 1 : P])
        return self._cache(kv, P), lp

    def _cache(self, kv: dict, P: int):
        """captured K/V by layer -> a transformers DynamicCache of a P-token prefix (the dict is emptied)"""
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.tcfg)
        for li in range(len(self.model.layers)):
            k, v = kv.pop(li)
            cache.update(k, v, li)
            layer = cache.layers[li]
            if hasattr(layer, "cumulative_length"):  # a sliding layer got its last window - 1 only: the true length is P
                layer.cumulative_length = P
        return cache

    def _step(self, cache, ids: list[list[int]], pos0: int):
        """tokens [B, L] after the cache (all rows at positions pos0..pos0+L-1) -> log-probs of the next token [B, V]"""
        torch = self.torch
        B, L = len(ids), len(ids[0])
        with torch.no_grad(), self._eager_sdpa():
            out = self.model(
                input_ids=torch.tensor(ids, device=self.device),
                position_ids=torch.arange(pos0, pos0 + L, device=self.device)[None].expand(B, L),
                past_key_values=cache,
                use_cache=True,
            )
            return self._logprobs(out.last_hidden_state[:, -1])

    def _stop_ids(self, stop: str):
        from spinf_serve import extract as ex

        v = self._vocab_tables()
        t = v["stops"].get(stop)
        if t is None:
            t = v["stops"][stop] = self.torch.tensor(ex.text_stop_ids(v["texts"], stop, v["eos"]), device=self.device)
        return t

    def _valid(self, g, st):
        """the number candidates that continue grammar state st: [(index into cand, new state)]"""
        v = self._vocab_tables()
        key = (g.currency, st)
        out = v["memo"].get(key)
        if out is None:
            out = v["memo"][key] = [(j, ns) for j, s in enumerate(v["cand_txt"]) if (ns := g.step(st, s)) is not None]
        return out

    def _search(self, cache, lp, pos: int, kind: str, stop: str, max_tokens: int, gen=None):
        """best-first beam from the cache (extract.py): every live beam also ends with P(beam) x P(end | beam) -> extract.Found;
        gen (extract.Generate): plain generation of n sequences instead (greedy or sampled, no pruning)"""
        from spinf_serve import extract as ex

        torch = self.torch
        v = self._vocab_tables()
        found = ex.Found()
        g = ex.NumberGrammar(kind == "currency") if kind != "text" else None
        live = [(0.0, [], r) for r in ex.gen_rngs(gen)] if gen else [(0.0, [], g.start if g else None)]
        if gen and gen.n > 1:  # the n sequences start from the same cache
            cache.reorder_cache(torch.tensor([0] * gen.n, device=self.device))
            lp = lp.expand(gen.n, -1).clone()
        stop_t = self._stop_ids(stop) if g is None else None
        for step in range(max_tokens + 1):
            cands = []
            if gen is not None:
                end = torch.logsumexp(lp[:, stop_t], -1).cpu().numpy()
                lp[:, stop_t] = -float("inf")
                lp[:, v["special"]] = -float("inf")
                top_v, top_i = (x.cpu().numpy() for x in lp.topk(max(ex.TOPK, gen.top_k), -1))
                for b, (s, toks, rng) in enumerate(live):
                    if step == max_tokens:
                        found.add_gen(self.tok.decode(toks), "length", stop)
                        continue
                    nxt = ex.gen_next(gen, rng, float(end[b]), top_v[b], top_i[b])
                    if nxt is None:
                        found.add_gen(self.tok.decode(toks), "stop", stop)
                        continue
                    cands.append((s + float(top_v[b][list(top_i[b]).index(nxt)]), b, nxt, rng))
                if step == max_tokens or not cands:
                    break
                cache.reorder_cache(torch.tensor([c[1] for c in cands], device=self.device))
                live = [(c[0], live[c[1]][1] + [c[2]], c[3]) for c in cands]
                lp = self._step(cache, [[c[2]] for c in cands], pos + step)
                found.steps += 1
                found.generated += len(cands)
                continue
            if g is None:
                end = torch.logsumexp(lp[:, stop_t], -1).cpu().numpy()
                lp[:, stop_t] = -float("inf")
                lp[:, v["special"]] = -float("inf")
                top_v, top_i = (x.cpu().numpy() for x in lp.topk(ex.TOPK, -1))
                for b, (s, toks, _) in enumerate(live):
                    if toks:
                        found.add(self.tok.decode(toks), s + float(end[b]))
                    cands += [(s + float(top_v[b, j]), b, int(top_i[b, j]), None) for j in range(ex.TOPK)]
            else:
                lpc = lp[:, v["cand_t"]].cpu().numpy()
                for b, (s, toks, st) in enumerate(live):
                    ext = self._valid(g, st)
                    if ext:
                        ev = lpc[b, [j for j, _ in ext]]
                        if toks and g.complete(st):
                            m = float(np.logaddexp.reduce(ev))
                            found.add(self.tok.decode(toks), s + math.log(max(1e-12, 1.0 - math.exp(min(m, 0.0)))))
                        for k in np.argsort(-ev)[: ex.TOPK]:
                            cands.append((s + float(ev[k]), b, v["cand"][ext[k][0]], ext[k][1]))
                    elif toks and g.complete(st):
                        found.add(self.tok.decode(toks), s)  # nothing can extend it: it ends here
            if step == max_tokens:
                break
            cands = [c for c in sorted(cands, key=lambda c: -c[0]) if c[0] >= found.floor()][: ex.BEAM]
            if not cands:
                break
            cache.reorder_cache(torch.tensor([c[1] for c in cands], device=self.device))  # repeated indices fan the cache out
            live = [(c[0], live[c[1]][1] + [c[2]], c[3]) for c in cands]
            lp = self._step(cache, [[c[2]] for c in cands], pos + step)
            found.steps += 1
            found.generated += len(cands)
        return found

    def extract(self, jobs: list[ExtractJob]) -> dict[object, object]:
        """extraction jobs -> {prompt key: extract.Found}. Per job the prefix is read once; each prompt decodes from a copy of its
        cache (the last one from the cache itself)."""
        if self.stack.hybrid:
            return self._extract_hybrid(jobs)
        torch = self.torch
        out: dict[object, object] = {}
        t0 = time.perf_counter()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for job in jobs:
            cache0, lp0 = self.prefill(job.prefix, job.media)
            self.stats.tokens += len(job.prefix)
            for k, pr in enumerate(job.prompts):
                cache = cache0 if k == len(job.prompts) - 1 else copy.deepcopy(cache0)
                lp = self._step(cache, [pr.ids], len(job.prefix)) if pr.ids else lp0.clone()
                self.stats.tokens += len(pr.ids)
                found = self._search(cache, lp, len(job.prefix) + len(pr.ids), pr.kind, pr.stop, pr.max_tokens, pr.gen)
                self.stats.tokens += found.generated
                out[pr.key] = found
            del cache0
        e1.record()
        torch.cuda.synchronize(self.device)
        self.stats.busy_s += e0.elapsed_time(e1) / 1000
        self.stats.wall_s += time.perf_counter() - t0
        return out

    def _extract_hybrid(self, jobs: list[ExtractJob]) -> dict[object, object]:
        """hybrid models: the jobs' prefixes read through the packed path with their prompts as extraction branches (a prompt
        without template tokens: a zero-length branch at the prefix's end), all decoded together (qwen_decode.Decoder)"""
        from spinf_serve.packer import GroupedTree

        items = []
        for n, job in enumerate(jobs):
            branches: list[list[int]] = []
            xb = []
            for pr in job.prompts:
                if pr.ids:
                    xb.append((len(branches), pr))
                    branches.append(list(pr.ids))
                else:
                    xb.append((-1, pr))
            tree = GroupedTree(list(job.prefix), branches, [], [], media=list(job.media), xbranches=xb, parents=[-1] * len(branches))
            items.append((("x", n), 0, tree))
        self.score_grouped(items)
        return {pr.key: self.x_found.pop(pr.key) for job in jobs for pr in job.prompts}

    def _x_collect(self, owners, cap: dict, xlp) -> None:
        """one packed forward's extraction branches -> decode.Group per tree run, held until _x_flush (the end of the call, or
        earlier when the held K/V would pass EngineConfig.decode_budget_gib): one decode for all slots of a call. cap: what the
        forward captured (Stack.capture: the K/V, and on the hybrid the recurrent captures and first generated positions)"""
        from spinf_serve import decode

        budget = (getattr(self.model, "_decode_budget_gib", None) or self.cfg.decode_budget_gib) * 2**30
        kv = cap["kv"]
        k = 0
        for o in owners:
            g = decode.Group(kv.pop(("p", id(o))), o.prefix_len)
            for _, bl, pr in o.xbranches:
                g.prompts.append((pr, kv.pop(("t", pr.key), {}), bl, xlp[k]))
                k += 1
            S = max(p.max_tokens for p, *_ in g.prompts) + 1
            if cap["extra"] is not None:
                self.stack.group_extra(self, g, o, cap, S)
            if not hasattr(self, "_x_per_tok"):
                self._x_per_tok = {
                    li: (kk.shape[1] * kk.shape[3] + vv.shape[1] * vv.shape[3]) * 2 for li, (kk, vv) in sorted(g.prefix_kv.items())
                }
            need = decode.group_bytes(g, self._x_per_tok, S)
            if self._x_pending and self._x_used + need > budget:
                self._x_flush()
            self._x_pending.append(g)
            self._x_used += need
        if self._x_used > budget:  # over on its own (a long prefix): decode now rather than next to the next forward's peak
            self._x_flush()

    def _x_flush(self) -> None:
        """decode the held groups (decode.Decoder) -> self.x_found, in chunks whose decoder working set (the K/V and, on the hybrid,
        the recurrent rows of every slot's BEAM sequences over max_tokens steps: decode.group_bytes) stays under
        EngineConfig.decoder_budget_gib; a group with more slots than fit is decoded in runs of its prompts (_x_split)"""
        import time as _t

        from spinf_serve import decode

        if not self._x_pending:
            return
        t0 = _t.perf_counter()
        decode.register()
        groups, self._x_pending, self._x_used = self._x_pending, [], 0
        for chunk in self._x_chunks(groups):
            dec = self.stack.decoder(self, chunk)
            for key, f in dec.run().items():
                self.x_found[key] = f
                self.stats.tokens += f.generated
            del dec, chunk
            self.x_chunks += 1
        self.torch.cuda.synchronize(self.device)
        self.x_busy_s += _t.perf_counter() - t0

    def _x_tree_bytes(self, tree) -> int:
        """the bytes a forward captures for a tree's extraction branches (its prefix K/V, the templates' K/V and recurrent rows,
        the prefix state): the packer keeps a stream's sum under EngineConfig.capture_budget_gib, next to the forward's own peak"""
        xb = getattr(tree, "xbranches", None)
        if not xb:
            return 0
        rows = sum(len(tree.branches[b]) for b, _ in xb if b >= 0)
        return len(tree.prefix) * self._x_tok_bytes + rows * (self._x_tok_bytes + self._x_row_bytes) + self._x_state_bytes

    def _to_host(self, t):
        """a GPU tensor -> its copy in pinned host memory, queued on the current stream (a later copy back on the same stream
        reads it after the transfer)"""
        h = self.torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        h.copy_(t, non_blocking=True)
        return h

    def _x_group_bytes(self, g) -> int:
        from spinf_serve import decode

        return decode.group_bytes(g, self._x_per_tok, max(p.max_tokens for p, *_ in g.prompts) + 1)

    def _x_chunk_bytes(self, groups: list) -> int:
        """the decoder's memory for groups decoded together: decode.Decoder pads them to the chunk's largest prompt count,
        max_tokens, template total and prefix (decode.chunk_bytes), so this is more than the sum of their group_bytes"""
        from spinf_serve import decode

        return decode.chunk_bytes(groups, self._x_per_tok, self.stack.chunk_extra_bytes(self.model, groups))

    def _x_chunks(self, groups: list):
        """the held groups -> chunks decoded together, each under the decoder budget as the decoder will allocate it
        (_x_chunk_bytes: the padded chunk, not the sum of the groups); a group over the budget alone is split into runs of its
        prompts. The groups' own references to the captures are dropped as they are split, so a prefix's K/V is freed once the
        last run that reads it has copied it."""
        budget = (getattr(self.model, "_decoder_budget_gib", None) or self.cfg.decoder_budget_gib) * 2**30
        chunk: list = []
        while groups:
            g = groups.pop(0)
            for part in self._x_split(g, budget):
                if chunk and self._x_chunk_bytes(chunk + [part]) > budget:
                    yield chunk
                    chunk = []
                chunk.append(part)
            del g, part
        if chunk:
            yield chunk

    def _x_split(self, g, budget: int) -> list:
        """a group whose decoder bytes pass the budget -> runs of its prompts that fit (at least one prompt each), sharing the
        captured prefix K/V and recurrent states (each run's decoder copies them into its own buffers)"""
        if self._x_chunk_bytes([g]) <= budget or len(g.prompts) == 1:
            return [g]
        parts, a = [], 0
        while a < len(g.prompts):
            b = a + 1
            while b < len(g.prompts) and self._x_chunk_bytes([self._x_sub(g, a, b + 1)]) <= budget:
                b += 1
            parts.append(self._x_sub(g, a, b))
            a = b
        g.prefix_kv, g.extra, g.prompts = {}, None, []  # the runs hold the captures now
        return parts

    def _x_sub(self, g, a: int, b: int):
        """the run of prompts [a, b) of a group as its own decode.Group (the captures by reference, the per-prompt extras sliced)"""
        from spinf_serve import decode

        extra = None
        if g.extra is not None:
            extra = {"state": dict(g.extra["state"]), "tmpl": g.extra["tmpl"][a:b], "pos0": g.extra["pos0"][a:b]}
        sub = decode.Group(dict(g.prefix_kv), g.P, list(g.prompts[a:b]), extra)
        if extra is not None:
            sub.extra_bytes = self.stack.sub_extra_bytes(self.model, sub)
        return sub
