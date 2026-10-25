"""Batched decoding of extraction slots right after the scoring pass that read them (compiler: templates as tree branches).

For each tree with extraction branches ("group"), the scoring pass captured (model.CAPTURE) the K/V of the tree's shared
prefix and of every template branch, and the next-token log-probs at each template's end. The answers of all slots of all
groups in a pass are then decoded together, one step for all: one batch row per group, its slots x beams as the queries of
that row. A query attends to the group's prefix (stored once per group), to its own template, and to the tokens its own beam
generated: nothing is copied per slot or per beam (attention "x_decode" over GroupKV below). Beams, ends and answers follow
extract.py (the same search as Scorer._search, run for every slot at once).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from spinf_serve import extract as ex

PAD_TOKEN = 0
_CURRENT: list = [None]  # the decoder whose step is running (the attention interface reads its tensors)


@dataclass
class Group:
    """one tree run's captured state: prefix K/V by layer ([1, kv heads, stored length, head dim]; sliding layers the last
    window - 1 tokens), its length P, and its slots (prompt, template K/V by layer, template length, log-probs after it)"""

    prefix_kv: dict
    P: int
    prompts: list = field(default_factory=list)  # [(ExtractPrompt, {layer: (k, v)}, T, lp0 [V] fp32)]
    # hybrid models (qwen_decode): the prefix's recurrent states, the templates' recurrent-layer rows and conv inputs, the
    # position of each prompt's first generated token; and the bytes those take in the decoder (group_bytes)
    extra: dict | None = None
    extra_bytes: int = 0


def register() -> None:
    """the attention interface "x_decode" (decoding steps only: Scorer switches to it inside _eager_sdpa(impl="x_decode"))"""
    from transformers import AttentionInterface

    if "x_decode" not in AttentionInterface._global_mapping:
        AttentionInterface.register("x_decode", _x_attention)


def _disable(fn):
    """keep a function out of the compiled decoder-layer graphs (it reads the decoder's per-step state)"""
    import torch

    return torch.compiler.disable(fn)


class GroupKV:
    """the cache transformers' layers write the new tokens' K/V into (update) during a decoding step: it stores them at the
    current step and hands the layer index to the attention (the K/V themselves live in the decoder's tensors)"""

    is_compileable = False

    def __init__(self, dec: "Decoder"):
        self.dec = dec

    @_disable
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        d = self.dec  # key_states [groups, kv heads, slots, head dim] -> the slots' generated K/V at the current step
        d.gk[layer_idx].index_copy_(3, d.step_t, key_states.unsqueeze(3))
        d.gv[layer_idx].index_copy_(3, d.step_t, value_states.unsqueeze(3))
        return ("x", layer_idx), ("x", layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0


@_disable
def _x_attention(module, query, key, value, attention_mask, scaling=None, sliding_window=None, softcap=None, **kw):
    """query [G, H, Q, d] -> [G, Q, H, d]: softmax over (the group's prefix | the slot's own template | the beam's own
    generated tokens), masked by ownership and, on sliding layers, the window"""
    import torch

    d = _CURRENT[0]
    li = key[1]
    G, H, Q, hd = query.shape
    kp, vp = d.kp[li], d.vp[li]  # [G, h, Lp, d]
    h = kp.shape[1]
    r = H // h
    q = query.reshape(G, h, r, Q, hd).to(kp.dtype)
    sc = scaling if scaling is not None else hd**-0.5
    # scores in bf16 matmuls (fp32 accumulation), softmax in fp32; the generated range is always all S slots, masked to the
    # tokens written so far (static shapes: the step is replayed from a CUDA graph)
    sp = torch.einsum("ghrqd,ghkd->ghrqk", q, kp).float() * sc
    st = torch.einsum("ghrqd,ghkd->ghrqk", q, d.kt[li]).float() * sc
    sg = torch.einsum("ghrqd,ghqsd->ghrqs", q, d.gk[li]).float() * sc
    if softcap:
        sp, st, sg = (torch.tanh(x / softcap) * softcap for x in (sp, st, sg))
    sliding = sliding_window is not None
    mp = d.mask_p[li] if not sliding else d.mask_p[li] & ((d.qpos[:, :, None] - d.pos_p[li][:, None, :]) < sliding_window)
    mt = d.mask_t if not sliding else d.mask_t & ((d.qpos[:, :, None] - d.pos_t[:, None, :]) < sliding_window)
    ninf = torch.finfo(torch.float32).min
    sp = sp.masked_fill(~mp[:, None, None], ninf)
    st = st.masked_fill(~mt[:, None, None], ninf)
    sg = sg.masked_fill(d.gen_idx > d.step_t, ninf)
    a = torch.softmax(torch.cat([sp, st, sg], -1), -1).to(vp.dtype)
    Lp, Lt = sp.shape[-1], st.shape[-1]
    out = torch.einsum("ghrqk,ghkd->ghrqd", a[..., :Lp], vp).float()
    out += torch.einsum("ghrqk,ghkd->ghrqd", a[..., Lp : Lp + Lt], d.vt[li]).float()
    out += torch.einsum("ghrqs,ghqsd->ghrqd", a[..., Lp + Lt :], d.gv[li]).float()
    return out.reshape(G, H, Q, hd).transpose(1, 2).contiguous().to(query.dtype), None


class Decoder:
    """one chunk of groups decoded together (Scorer.decode_groups builds it and runs its steps)"""

    def __init__(self, scorer, groups: list[Group]):
        import torch

        self.sc, self.groups = scorer, groups
        dev = scorer.device
        self.kv_layers = sorted(groups[0].prefix_kv)  # the layers with K/V (all of them on Gemma, the full-attention ones on Qwen)
        G = len(groups)
        J = max(len(g.prompts) for g in groups)
        self.Q = Q = J * ex.BEAM
        self.S = S = max(p.max_tokens for g in groups for p, *_ in g.prompts) + 1
        self.step = 0
        # prefix, per layer (sliding layers store fewer tokens): padded [G, h, Lp, d] + true positions (-1 = padding)
        self.kp, self.vp, self.pos_p, self.mask_p = {}, {}, {}, {}
        self.kt, self.vt, self.gk, self.gv = {}, {}, {}, {}
        Tt = max(sum(T for _, _, T, _ in g.prompts) for g in groups)
        self.pos_t = torch.full((G, Tt), -1, dtype=torch.int64, device=dev)
        own_t = torch.full((G, Tt), -1, dtype=torch.int64, device=dev)
        for gi, g in enumerate(groups):
            o = 0
            for j, (_, _, T, _) in enumerate(g.prompts):
                self.pos_t[gi, o : o + T] = torch.arange(g.P, g.P + T, device=dev)
                own_t[gi, o : o + T] = j
                o += T
        for li in self.kv_layers:
            k0, v0 = groups[0].prefix_kv[li]
            h, hd = k0.shape[1], k0.shape[3]
            Lp = max(g.prefix_kv[li][0].shape[2] for g in groups)
            kp = torch.zeros((G, h, Lp, hd), dtype=k0.dtype, device=dev)
            vp = torch.zeros((G, h, Lp, v0.shape[3]), dtype=v0.dtype, device=dev)
            pos = torch.full((G, Lp), -1, dtype=torch.int64, device=dev)
            kt = torch.zeros((G, h, Tt, hd), dtype=k0.dtype, device=dev)
            vt = torch.zeros((G, h, Tt, v0.shape[3]), dtype=v0.dtype, device=dev)
            for gi, g in enumerate(groups):
                k, v = g.prefix_kv[li]
                n = k.shape[2]
                kp[gi, :, :n], vp[gi, :, :n] = k[0], v[0]
                pos[gi, :n] = torch.arange(g.P - n, g.P, device=dev)  # a sliding layer keeps the last n of the prefix
                o = 0
                for _, tkv, T, _ in g.prompts:
                    kt[gi, :, o : o + T], vt[gi, :, o : o + T] = tkv[li][0][0], tkv[li][1][0]
                    o += T
            self.kp[li], self.vp[li], self.pos_p[li], self.mask_p[li] = kp, vp, pos, (pos >= 0)[:, None, :].expand(G, Q, Lp)
            self.kt[li], self.vt[li] = kt, vt
            self.gk[li] = torch.zeros((G, h, Q, S, hd), dtype=k0.dtype, device=dev)
            self.gv[li] = torch.zeros((G, h, Q, S, v0.shape[3]), dtype=v0.dtype, device=dev)
            for g in groups:  # the captured tensors are now copied: free them
                g.prefix_kv[li] = None
        for g in groups:
            for p in g.prompts:
                p[1].clear()
        qj = torch.arange(Q, device=dev) // ex.BEAM  # slot -> prompt index within its group
        self.mask_t = own_t[:, None, :] == qj[None, :, None]  # [G, Q, Tt]
        self.base = torch.zeros((G, Q), dtype=torch.int64, device=dev)  # the position of each slot's first generated token
        for gi, g in enumerate(groups):
            pos0 = g.extra.get("pos0") if g.extra else None  # hybrid: the packer's position after the template (images shrink it)
            for j, (_, _, T, _) in enumerate(g.prompts):
                self.base[gi, j * ex.BEAM : (j + 1) * ex.BEAM] = pos0[j] if pos0 else g.P + T
        self.qpos = self.base.clone()  # static buffers of a step (a CUDA graph replays on them)
        self.tok_buf = torch.full((G, Q), PAD_TOKEN, dtype=torch.int64, device=dev)
        self.step_t = torch.zeros(1, dtype=torch.int64, device=dev)
        self.gen_idx = torch.arange(S, device=dev)
        self.graph, self.h_out = None, None

    def run(self) -> dict:
        """-> {prompt key: extract.Found}"""
        import torch

        sc = self.sc
        v = sc._vocab_tables()
        G, Q, B = len(self.groups), self.Q, ex.BEAM
        prompts = [(gi, j, p) for gi, g in enumerate(self.groups) for j, (p, *_) in enumerate(g.prompts)]
        found = {p.key: ex.Found() for _, _, p in prompts}
        gram = {p.key: ex.NumberGrammar(p.kind == "currency") for _, _, p in prompts if p.kind != "text"}
        # a live beam: (log-prob, tokens, state): the grammar state of a number, a generated sequence's random generator, else None
        live = {
            p.key: [(0.0, [], r) for r in ex.gen_rngs(p.gen)] if p.gen else [(0.0, [], gram[p.key].start if p.key in gram else None)]
            for _, _, p in prompts
        }
        # step 0: the log-probs at the end of each template (from the scoring pass), one row per prompt
        rows = {p.key: [gi * Q + j * B] for gi, j, p in prompts}
        lp = torch.stack([self.groups[gi].prompts[j][3] for gi, j, _ in prompts])
        row_of = {gi * Q + j * B: k for k, (gi, j, _) in enumerate(prompts)}
        import os
        import time as _t

        prof = os.environ.get("SPINF_X_PROFILE")
        t_fw = t_cpu = 0.0
        for step in range(self.S):
            self.step = step
            t0 = _t.perf_counter()
            # ends and candidates of every live beam (on the GPU per stop set / candidate set, then per prompt on the CPU)
            order = [(gi, j, p) for gi, j, p in prompts if live[p.key]]
            if not order:
                break
            text_rows = [(p, [row_of[s] for s in rows[p.key]]) for _, _, p in order if p.kind == "text"]
            num_rows = [(p, [row_of[s] for s in rows[p.key]]) for _, _, p in order if p.kind != "text"]
            ends, tops = {}, {}
            by_stop: dict[str, list] = {}
            for p, rr in text_rows:
                by_stop.setdefault(p.stop, []).append((p, rr))
            for stop, items in by_stop.items():
                idx = torch.tensor([r for _, rr in items for r in rr], device=sc.device)
                x = lp[idx]
                st = sc._stop_ids(stop)
                e = torch.logsumexp(x[:, st], -1).cpu().numpy()
                x[:, st] = -float("inf")
                x[:, v["special"]] = -float("inf")
                K = max([ex.TOPK] + [p.gen.top_k for p, _ in items if p.gen])  # sampling draws from more candidates than the search
                tv, ti = (t.cpu().numpy() for t in x.topk(K, -1))
                o = 0
                for p, rr in items:
                    ends[p.key] = e[o : o + len(rr)]
                    tops[p.key] = (tv[o : o + len(rr)], ti[o : o + len(rr)])
                    o += len(rr)
            lpc = {}
            if num_rows:
                idx = torch.tensor([r for _, rr in num_rows for r in rr], device=sc.device)
                xc = lp[idx][:, v["cand_t"]].cpu().numpy()
                o = 0
                for p, rr in num_rows:
                    lpc[p.key] = xc[o : o + len(rr)]
                    o += len(rr)
            # new beams per prompt -> slot tokens and the parents of the slots (for the generated K/V)
            tokens = torch.full((G, Q), PAD_TOKEN, dtype=torch.int64)
            parent = torch.arange(G * Q)
            new_rows = {}
            for gi, j, p in order:
                f = found[p.key]
                cands = []
                if p.gen is not None:  # generation: each sequence continues with its own next token or ends; no pruning
                    tv, ti = tops[p.key]
                    for b, (s, toks, rng) in enumerate(live[p.key]):
                        if step >= p.max_tokens:
                            f.add_gen(sc.tok.decode(toks), "length", p.stop)
                            continue
                        r = min(b, len(tv) - 1)  # step 0: the n sequences start from the template's one row
                        nxt = ex.gen_next(p.gen, rng, float(ends[p.key][r]), tv[r], ti[r])
                        if nxt is None:
                            f.add_gen(sc.tok.decode(toks), "stop", p.stop)
                            continue
                        cands.append((s + float(tv[r][list(ti[r]).index(nxt)]), b, nxt, rng))
                    old = live[p.key]
                    old_slots = rows[p.key]
                    live[p.key] = [(c[0], old[c[1]][1] + [c[2]], c[3]) for c in cands]
                    slots = [gi * Q + j * B + b for b in range(len(cands))]
                    for sl, c in zip(slots, cands, strict=True):
                        parent[sl] = old_slots[min(c[1], len(old_slots) - 1)]
                        tokens.view(-1)[sl] = c[2]
                    new_rows[p.key] = slots
                    f.steps += 1 if cands else 0
                    f.generated += len(cands)
                    continue
                for b, (s, toks, gst) in enumerate(live[p.key]):
                    if p.kind == "text":
                        if toks:
                            f.add(sc.tok.decode(toks), s + float(ends[p.key][b]))
                        tv, ti = tops[p.key]
                        cands += [(s + float(tv[b, t]), b, int(ti[b, t]), None) for t in range(ex.TOPK)]
                    else:
                        g = gram[p.key]
                        ext = sc._valid(g, gst)
                        if ext:
                            ev = lpc[p.key][b, [e for e, _ in ext]]
                            if toks and g.complete(gst):
                                m = float(np.logaddexp.reduce(ev))
                                f.add(sc.tok.decode(toks), s + math.log(max(1e-12, 1.0 - math.exp(min(m, 0.0)))))
                            for t in np.argsort(-ev)[: ex.TOPK]:
                                cands.append((s + float(ev[t]), b, v["cand"][ext[t][0]], ext[t][1]))
                        elif toks and g.complete(gst):
                            f.add(sc.tok.decode(toks), s)
                if step >= p.max_tokens:
                    live[p.key] = []
                    continue
                cands = [c for c in sorted(cands, key=lambda c: -c[0]) if c[0] >= f.floor()][:B]
                old = live[p.key]
                old_slots = rows[p.key]
                live[p.key] = [(c[0], old[c[1]][1] + [c[2]], c[3]) for c in cands]
                slots = [gi * Q + j * B + b for b in range(len(cands))]
                for sl, c in zip(slots, cands, strict=True):
                    parent[sl] = old_slots[c[1]]
                    tokens.view(-1)[sl] = c[2]
                new_rows[p.key] = slots
                f.steps += 1 if cands else 0
                f.generated += len(cands)
            rows = {k: new_rows.get(k, []) for k in rows}
            if not any(rows.values()):
                break
            # the generated K/V follow their beams (a child takes its parent's first `step` tokens)
            if step > 0 and not bool((parent == torch.arange(G * Q)).all()):
                self._reorder(parent.to(sc.device))
            self.qpos.copy_(self.base + step)
            self.step_t.fill_(step)
            if prof:
                torch.cuda.synchronize(sc.device)
            t1 = _t.perf_counter()
            h = self._forward(tokens.to(sc.device))  # [G, Q, hidden]
            alive = sorted(s for sl in rows.values() for s in sl)
            row_of = {s: k for k, s in enumerate(alive)}
            hh = h.reshape(G * Q, -1)[torch.tensor(alive, device=sc.device)]
            rr = sc.cfg.lm_rows
            with torch.no_grad():
                lp = torch.cat([sc._logprobs(hh[s : s + rr]) for s in range(0, hh.shape[0], rr)])
            if prof:
                torch.cuda.synchronize(sc.device)
                t_cpu += t1 - t0
                t_fw += _t.perf_counter() - t1
        if prof:
            print(
                f"decode: {len(self.groups)} groups, {len(prompts)} prompts, {step + 1} steps: beams/ends {t_cpu:.2f}s, "
                f"forward+lm head {t_fw:.2f}s ({t_fw / (step + 1) * 1000:.0f} ms/step)",
                flush=True,
            )
        return found

    def _reorder(self, pidx) -> None:
        """the per-slot state follows the beams: slot s takes slot pidx[s]'s generated K/V (in place: a CUDA graph holds these
        tensors)"""
        G, Q = len(self.groups), self.Q
        for li in self.kv_layers:
            for t in (self.gk[li], self.gv[li]):
                view = t.permute(0, 2, 1, 3, 4)  # [G, Q, h, S, d]
                view.copy_(view.reshape(G * Q, *view.shape[2:])[pidx].reshape(view.shape))

    def _forward(self, tokens):
        """one decoding step on the static buffers -> hidden states [G, Q, hidden]. With EngineConfig.x_cuda_graphs the step is
        captured once (after one warm-up run) and replayed: the ~1,500 kernel launches of a step cost no Python after that."""
        import torch

        sc = self.sc
        self.tok_buf.copy_(tokens)
        if self.graph is not None:
            self.graph.replay()
            return self.h_out
        if not sc.cfg.x_cuda_graphs:
            return self._step_once()
        s = torch.cuda.Stream(sc.device)
        s.wait_stream(torch.cuda.current_stream(sc.device))
        with torch.cuda.stream(s):
            self._step_once()  # warm-up (lazy initialisations must not happen inside the capture)
        torch.cuda.current_stream(sc.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.h_out = self._step_once()
        self.graph = g
        g.replay()
        return self.h_out

    def _step_once(self):
        import torch

        sc = self.sc
        _CURRENT[0] = self
        try:
            with torch.no_grad(), sc._eager_sdpa("x_decode", compiled=sc.cfg.compile):
                out = sc.model(
                    input_ids=self.tok_buf,
                    position_ids=self.qpos,
                    past_key_values=GroupKV(self),
                    use_cache=True,
                    attention_mask={"full_attention": None, "sliding_attention": None},
                )
            return out.last_hidden_state
        finally:
            _CURRENT[0] = None


def chunk_bytes(groups: list[Group], per_tok_kv: dict, extra: int = 0) -> int:
    """the decoder's memory for groups decoded together. Decoder pads every group to the chunk's largest prefix, template total,
    prompt count (Q = J x BEAM slots) and max_tokens (S): a chunk costs more than the sum of its groups' own group_bytes. The
    5 searched slots (max_tokens 16) of a video next to its 2 generated ones (64) take 65 rows each, not 17, and 30 slots
    instead of 12: 4.6 GiB against 2.5 estimated, the Qwen OOM of 2026-10-11. extra: a hybrid chunk's recurrent buffers
    (qwen_decode.chunk_extra_bytes). For one group this equals group_bytes."""
    G = len(groups)
    J = max(len(g.prompts) for g in groups)
    S = max(p.max_tokens for g in groups for p, *_ in g.prompts) + 1
    Tt = max(sum(T for _, _, T, _ in g.prompts) for g in groups)
    total = extra
    for li, per_tok in per_tok_kv.items():
        Lp = max((g.prefix_kv[li][0].shape[2] if g.prefix_kv.get(li) is not None else g.P) for g in groups)
        total += per_tok * G * (Lp + Tt + J * ex.BEAM * S)
    return total


def group_bytes(g: Group, per_tok_kv: dict, S: int) -> int:
    """the decoder's memory for one group: prefix + templates stored once, generated K/V per slot (bf16 K and V; per_tok_kv: the
    bytes per token of each layer with K/V), plus a hybrid group's recurrent extras (Group.extra_bytes)"""
    J = len(g.prompts)
    T = sum(t for _, _, t, _ in g.prompts)
    total = g.extra_bytes
    for li, per_tok in per_tok_kv.items():
        lp = g.prefix_kv[li][0].shape[2] if g.prefix_kv.get(li) is not None else g.P
        total += per_tok * (lp + T + J * ex.BEAM * S)
    return total
