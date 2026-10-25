"""Decoding steps on the Qwen stack (qwen.py) for extraction slots: decode.Decoder's search over the hybrid model's state.

What a slot has after the scoring pass (qwen.TreePlan.cap, filled by Qwen35Stack.capture): the K/V of its group's prefix and
of its own template in the 16 full-attention layers (as Gemma's decoder), and for the 48 recurrent layers the prefix's final
state S_P (3 MiB per layer, one per group), the template's q k v g beta rows and the layer input at the template end's conv taps.

A step runs the 64 layers on one token per slot. The linears run at M = slots through w4gemm (the int4 weights streamed once, no
dequantized copy: QwenOptions.decode_linear; "model" = the scoring path's Int4Linear, padded to 32 rows for torch._int_mm). A
recurrent layer re-runs the recurrence from S_P over the slot's rows -- its template's, then the tokens it generated
(gdn.recurrent indexing the row caches) -- a few dozen tokens per slot: no per-slot state is stored (144 MiB each) or copied when
beams fan out, only the small row caches follow the beams. The 4-tap conv reads a per-slot ring of the 3 previous pre-conv rows
plus the token's own. A full-attention layer attends over (prefix | own template | own generated) through decode._x_attention,
the new K/V written into the decoder's buffers as decode.GroupKV does.

The step is replayed from a CUDA graph (QwenOptions.decode_graphs): ~13,000 launches per step otherwise. Everything inside the graph
reads static buffers (the token embeddings copied in before, the step counter, the beam positions, the row index built on the
GPU) and is idempotent; the conv ring advances after the step, outside the graph.
"""

import numpy as np

from spinf_serve import decode
from spinf_serve import extract as ex

PAD_ROWS = 32  # linear "model": the step's row count is a multiple of this (zero rows after the slots: torch._int_mm wants > 16)


def group_extra_bytes(model, g: decode.Group, S: int) -> int:
    """a hybrid group's decoder memory beyond its K/V: the prefix states, the recurrent row caches (template rows + S generated
    rows per slot) and the conv rings"""
    gdn = [la.linear_attn for la in model.layers if la.kind == "linear_attention"]
    la = gdn[0]
    states = sum(t.numel() * t.element_size() for t in g.extra["state"].values())
    per_row = len(gdn) * ((2 * la.hk * la.dk + la.hv * la.dv) * 2 + la.hv * 6)
    J = len(g.prompts)
    rows = sum(T for _, _, T, _ in g.prompts) + J * ex.BEAM * S
    ring = J * ex.BEAM * len(gdn) * 4 * la.conv_dim * 2
    return states + rows * per_row + ring


def chunk_extra_bytes(model, groups: list[decode.Group]) -> int:
    """a hybrid chunk's decoder memory beyond its K/V, as Decoder.__init__ allocates it for groups decoded together: the prefix
    states per group (h0), the recurrent row caches and the output buffer over the padded rows R (every template row, then
    G x Q x S generated rows: Q and S are the chunk's largest), the conv rings per slot (decode.chunk_bytes takes the result)"""
    gdn = [la.linear_attn for la in model.layers if la.kind == "linear_attention"]
    la = gdn[0]
    n_g = len(gdn)
    G = len(groups)
    J = max(len(g.prompts) for g in groups)
    S = max(p.max_tokens for g in groups for p, *_ in g.prompts) + 1
    n_slots = G * J * ex.BEAM
    R = sum(T for g in groups for _, _, T, _ in g.prompts) + n_slots * S
    states = n_g * G * la.hv * la.dk * la.dv * 4
    per_row = n_g * ((2 * la.hk * la.dk + la.hv * la.dv) * 2 + la.hv * 6) + la.hv * la.dv * 2  # cq ck cv cg cb, o_buf
    ring = n_g * n_slots * 4 * la.conv_dim * 2  # ring (3 rows) + pre_new
    return states + R * per_row + ring


def rope(x, cos, sin):  # x [S, H, D]; cos, sin [S, R] on the first R dims (rotate-half within them), as qwen.rope
    import torch

    R = cos.shape[-1]
    xr, xp = x[..., :R], x[..., R:]
    x1, x2 = xr[..., : R // 2], xr[..., R // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos[:, None] + rot * sin[:, None], xp], -1)


class Decoder(decode.Decoder):
    """decode.Decoder (the search, the K/V buffers of the attention layers) plus the recurrent layers' state"""

    def __init__(self, scorer, groups: list[decode.Group]):
        import torch

        super().__init__(scorer, groups)
        m, dev = scorer.model, scorer.device
        self.m = m
        G, Q, S = len(groups), self.Q, self.S
        n_slots = G * Q
        self.gdn = [i for i, la in enumerate(m.layers) if la.kind == "linear_attention"]
        self.gdn_pos = {li: x for x, li in enumerate(self.gdn)}
        la0 = m.layers[self.gdn[0]].linear_attn
        hk, hv, dk, dv, cd = la0.hk, la0.hv, la0.dk, la0.dv, la0.conv_dim
        self.scale = dk**-0.5
        n_g = len(self.gdn)
        # the prefix states [layers, G, hv, dk, dv] fp32, read by slot -> group (the captures freed as they are copied: 144 MiB each)
        self.h0 = torch.empty((n_g, G, hv, dk, dv), dtype=torch.float32, device=dev)
        for x, li in enumerate(self.gdn):
            for gi, g in enumerate(groups):
                self.h0[x, gi].copy_(g.extra["state"].pop(li))
        self.h0_idx = torch.arange(n_slots, device=dev) // Q
        # row caches per recurrent layer: the templates' rows (per prompt), then S generated rows per slot
        t0: dict[tuple[int, int], int] = {}
        r = 0
        for gi, g in enumerate(groups):
            for j, (_, _, T, _) in enumerate(g.prompts):
                t0[(gi, j)] = r
                r += T
        self.gen0 = r
        R = r + n_slots * S
        dt = torch.bfloat16
        for g in groups:
            for tm in g.extra["tmpl"]:
                if tm["tok"]:
                    dt = next(iter(tm["tok"].values()))[0].dtype
                    break
        self.cq = torch.zeros((n_g, R, hk * dk), dtype=dt, device=dev)
        self.ck = torch.zeros((n_g, R, hk * dk), dtype=dt, device=dev)
        self.cv = torch.zeros((n_g, R, hv * dv), dtype=dt, device=dev)
        self.cg = torch.zeros((n_g, R, hv), dtype=torch.float32, device=dev)
        self.cb = torch.zeros((n_g, R, hv), dtype=dt, device=dev)
        for gi, g in enumerate(groups):
            for j, tm in enumerate(g.extra["tmpl"]):
                T = g.prompts[j][2]
                if not T:
                    continue
                a = t0[(gi, j)]
                for x, li in enumerate(self.gdn):
                    q, k, v, gg, bb = tm["tok"][li]
                    self.cq[x, a : a + T] = q.reshape(T, -1).to(dt)
                    self.ck[x, a : a + T] = k.reshape(T, -1).to(dt)
                    self.cv[x, a : a + T] = v.reshape(T, -1).to(dt)
                    self.cg[x, a : a + T] = gg.float()
                    self.cb[x, a : a + T] = bb.to(dt)
        # per slot: its template rows [t0, t0 + T) and its prompt (slots past a group's prompts have none)
        slot_T = np.zeros(n_slots, np.int64)
        slot_t0 = np.zeros(n_slots, np.int64)
        prompt_of_slot = np.zeros(n_slots, np.int64)
        p = 0
        for gi, g in enumerate(groups):
            for j, (_, _, T, _) in enumerate(g.prompts):
                sl = slice(gi * Q + j * ex.BEAM, gi * Q + (j + 1) * ex.BEAM)
                slot_T[sl], slot_t0[sl], prompt_of_slot[sl] = T, t0[(gi, j)], p
                p += 1
        self.slot_T = torch.from_numpy(slot_T).to(dev)
        self.slot_t0 = torch.from_numpy(slot_t0).to(dev)
        self.gen_rows = self.gen0 + torch.arange(n_slots, device=dev) * S
        self.flat = torch.arange(int(slot_T.sum()) + n_slots * S, device=dev)  # the recurrence's rows at most
        self.R = R
        # conv rings [layers, slots, 3, conv_dim]: the 3 pre-conv rows before the token (taps 3 back .. 1 back); the token's own
        # lands in pre_new during the step and the ring advances after it. Initially the template end's path predecessors: their
        # layer inputs were captured, the projection runs here (M = 4 x prompts)
        self.ring = torch.zeros((n_g, n_slots, 3, cd), dtype=dt, device=dev)
        self.pre_new = torch.zeros((n_g, n_slots, cd), dtype=dt, device=dev)
        pos = torch.from_numpy(prompt_of_slot).to(dev)
        with torch.no_grad():
            for x, li in enumerate(self.gdn):
                la = m.layers[li].linear_attn
                xin = torch.stack([tm["xin"][li] for g in groups for tm in g.extra["tmpl"]])  # [prompts, 4, hidden]
                flat = xin.reshape(-1, xin.shape[-1])
                pre = self._lin(la.in_proj_qkvz, flat)[: flat.shape[0], :cd].reshape(xin.shape[0], 4, cd)
                self.ring[x] = pre[:, 1:4].to(dt)[pos]
        self.o_buf = torch.empty((R, hv, dv), dtype=dt, device=dev)
        self.x_in = torch.zeros((n_slots, m.norm.weight.shape[0]), dtype=torch.bfloat16, device=dev)
        self.graph = self.h_out = None
        for g in groups:  # copied: free the captures
            g.extra = None

    def _rows(self):
        """the recurrence's rows this step, on the GPU from the step counter: per slot its template rows, then its generated rows
        0 .. step -> (idx [max rows] (entries past the end clamped into range, unread), cu [slots + 1])"""
        import torch
        import torch.nn.functional as F

        n, S = self.slot_T.shape[0], self.S
        L = self.slot_T + self.step_t + 1
        cu = F.pad(torch.cumsum(L, 0), (1, 0))
        slot = (torch.searchsorted(cu, self.flat, right=True) - 1).clamp(max=n - 1)
        within = self.flat - cu[slot]
        T = self.slot_T[slot]
        row = torch.where(within < T, self.slot_t0[slot] + within, self.gen0 + slot * S + within - T)
        return row.clamp(0, self.R - 1), cu

    def _reorder(self, pidx) -> None:
        super()._reorder(pidx)
        n, S = self.slot_T.shape[0], self.S
        for t in (self.cq, self.ck, self.cv, self.cg, self.cb):
            gen = t[:, self.gen0 :].reshape(t.shape[0], n, S, -1)
            for x in range(gen.shape[0]):  # layer by layer: the gather's temporary is one layer's rows, not all 48 layers' (GiBs)
                gen[x].copy_(gen[x][pidx])
        self.ring.copy_(self.ring[:, pidx])

    def _forward(self, tokens):
        """one decoding step -> hidden states [G, Q, hidden] after the final norm: the embeddings into the static input, the step
        (from the graph after the first), then the conv rings advance"""
        import torch

        m = self.m
        self.tok_buf.copy_(tokens)
        ids = tokens.reshape(-1).cpu().numpy()
        with torch.no_grad():
            self.x_in.copy_(m.embed(ids) if m.embed_host is not None else m.embed_tokens(tokens.reshape(-1)))
            if not m._options.decode_graphs:
                out = self._step_once()
            elif self.graph is not None:
                self.graph.replay()
                out = self.h_out
            else:
                # the stream, the kernels' first loads and the graph's private pool take device memory outside the allocator: the
                # scoring pass's cached blocks go back to the device first
                torch.cuda.synchronize(self.sc.device)
                torch.cuda.empty_cache()
                s = torch.cuda.Stream(self.sc.device)
                s.wait_stream(torch.cuda.current_stream(self.sc.device))
                with torch.cuda.stream(s):
                    self._step_once()  # warm-up (compilations, autotuning: not inside the capture); idempotent
                torch.cuda.current_stream(self.sc.device).wait_stream(s)
                torch.cuda.synchronize(self.sc.device)
                torch.cuda.empty_cache()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.h_out = self._step_once()
                self.graph = g
                g.replay()
                out = self.h_out
            self.ring[:, :, :2] = self.ring[:, :, 1:].clone()
            self.ring[:, :, 2] = self.pre_new
        return out

    def _step_once(self):
        """the step on the static buffers (x_in, qpos, step_t) -> [G, Q, hidden]"""
        import torch

        m = self.m
        G, Q = len(self.groups), self.Q
        n = G * Q
        M = n if m._options.decode_linear == "w4" else max(PAD_ROWS, -(-n // 16) * 16)
        x = self.x_in
        if n < M:
            x = torch.cat([x, x.new_zeros(M - n, x.shape[1])])
        pos = self.qpos.reshape(-1).float()
        f = pos[:, None] * m.inv_freq[None]
        e = torch.cat([f, f], -1)
        cos, sin = e.cos().to(x.dtype), e.sin().to(x.dtype)
        idx, cu = self._rows()
        rows_new = self.gen_rows + self.step_t
        decode._CURRENT[0] = self
        try:
            for li, layer in enumerate(m.layers):
                h = layer.input_layernorm(x)
                if layer.kind == "linear_attention":
                    h = self._gdn(layer.linear_attn, self.gdn_pos[li], h, n, idx, cu, rows_new)
                else:
                    h = self._attn(layer.self_attn, li, h, n, cos, sin)
                x = x + h
                x = x + self._mlp(layer.mlp, layer.post_attention_layernorm(x))
        finally:
            decode._CURRENT[0] = None
        return m.norm(x[:n]).reshape(G, Q, -1)

    def _lin(self, lin, x):
        """a linear of the model on x [M, in] -> [rows, out (+ extra)] bf16 (QwenOptions.decode_linear); "model" pads M to PAD_ROWS
        (the caller slices the rows it needs)"""
        import torch

        if self.m._options.decode_linear == "w4":
            from spinf_serve.quant import w4gemm

            return w4gemm.linear(x, lin.q, lin.s, lin.extra)
        M = max(PAD_ROWS, -(-x.shape[0] // 16) * 16)
        if x.shape[0] < M:
            x = torch.cat([x, x.new_zeros(M - x.shape[0], x.shape[1])])
        return lin(x)

    def _mlp(self, mlp, h):
        import torch.nn.functional as F

        if self.m._options.decode_linear != "w4":
            return mlp(h)
        gu = self._lin(mlp.gate_up_proj, h)
        k = gu.shape[-1] // 2
        return self._lin(mlp.down_proj, F.silu(gu[:, :k]) * gu[:, k:])

    def _gdn(self, la, x, h, n, idx, cu, rows_new):
        """one recurrent layer on h [M, hidden] (rows past n are zeros) -> [M, hidden]"""
        import torch
        import torch.nn.functional as F

        from spinf_serve.models.qwen35 import gdn

        y = self._lin(la.in_proj_qkvz, h)[:n]  # [n, conv_dim + value dim + 2 hv]: q k v (pre-conv) z, then b and a
        cd, hv, dv, hk, dk = la.conv_dim, la.hv, la.dv, la.hk, la.dk
        vd, kd = hv * dv, hk * dk
        pre = y[:, :cd].to(self.ring.dtype)
        self.pre_new[x].copy_(pre)
        taps = torch.cat([self.ring[x], pre[:, None]], 1)  # [n, 4, conv_dim]: 3 back .. itself
        conv = (taps.float() * la.conv_w.float()[None].transpose(1, 2)).sum(1)
        qkv = F.silu(conv).to(h.dtype)
        q = qkv[:, :kd].reshape(n, hk, dk)
        k = qkv[:, kd : 2 * kd].reshape(n, hk, dk)
        v = qkv[:, 2 * kd :].reshape(n, hv, dv)
        q = (q.float() * torch.rsqrt(q.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(h.dtype)
        k = (k.float() * torch.rsqrt(k.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(h.dtype)
        z = y[:, cd : cd + vd]
        b, a = y[:, cd + vd : cd + vd + hv], y[:, cd + vd + hv :]
        beta = torch.sigmoid(b.float()).to(h.dtype)
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias.float())
        cq, ck, cv, cg, cb = self.cq[x], self.ck[x], self.cv[x], self.cg[x], self.cb[x]
        cq.index_copy_(0, rows_new, q.reshape(n, -1).to(cq.dtype))
        ck.index_copy_(0, rows_new, k.reshape(n, -1).to(ck.dtype))
        cv.index_copy_(0, rows_new, v.reshape(n, -1).to(cv.dtype))
        cg.index_copy_(0, rows_new, g.float())
        cb.index_copy_(0, rows_new, beta.to(cb.dtype))
        R = self.R
        gdn.recurrent(
            cq.view(1, R, hk, dk),
            ck.view(1, R, hk, dk),
            cv.view(1, R, hv, dv),
            cg.view(1, R, hv),
            cb.view(1, R, hv),
            self.scale,
            cu,
            n,
            h0=self.h0[x],
            h0_idx=self.h0_idx,
            idx=idx,
            o=self.o_buf[None],
        )
        o = self.o_buf[rows_new]  # [n, hv, dv]: the new tokens' outputs
        of = o.float()
        of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + la.eps)
        of = (la.norm_w * of.to(h.dtype)).float() * F.silu(z.reshape(n, hv, dv).float())
        full = h.new_zeros(h.shape[0], vd)
        full[:n] = of.to(h.dtype).reshape(n, -1)
        return self._lin(la.out_proj, full)[: h.shape[0]]

    def _attn(self, at, li, h, n, cos, sin):
        """one full-attention layer on h [M, hidden] -> [M, hidden]: the new token's K/V into the decoder's buffers, attention over
        (prefix | own template | own generated) (decode._x_attention)"""
        import torch

        G, Q = len(self.groups), self.Q
        y = self._lin(at.qkv_proj, h)[:n]
        nq, nk = at.h * 2 * at.d, at.hkv * at.d
        qg = y[:, :nq].reshape(n, at.h, 2 * at.d)
        q, gate = qg[..., : at.d], qg[..., at.d :].reshape(n, -1)
        q = at.q_norm(q)
        k = at.k_norm(y[:, nq : nq + nk].reshape(n, at.hkv, at.d))
        v = y[:, nq + nk :].reshape(n, at.hkv, at.d)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        kk = k.reshape(G, Q, at.hkv, at.d).permute(0, 2, 1, 3).unsqueeze(3)
        vv = v.reshape(G, Q, at.hkv, at.d).permute(0, 2, 1, 3).unsqueeze(3)
        self.gk[li].index_copy_(3, self.step_t, kk.to(self.gk[li].dtype))
        self.gv[li].index_copy_(3, self.step_t, vv.to(self.gv[li].dtype))
        query = q.reshape(G, Q, at.h, at.d).permute(0, 2, 1, 3)  # [G, H, Q, d]
        o, _ = decode._x_attention(None, query, ("x", li), None, None, scaling=at.scale)
        full = h.new_zeros(h.shape[0], at.h * at.d)
        full[:n] = o.reshape(n, -1) * torch.sigmoid(gate)
        return self._lin(at.o_proj, full)[: h.shape[0]]
