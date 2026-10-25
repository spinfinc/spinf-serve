"""qwen.tree_plan on packed tries: the path predecessors spell each prompt, and running segments level by level from their
parents' final states equals a sequential recurrence along every path (a toy recurrence stands in for Gated DeltaNet)."""

import random

import numpy as np

from spinf_serve.models.qwen35.qwen import tree_plan
from spinf_serve.packer import Packer, ReadPlan, build_grouped_tree


def _jobs(rng, n):
    jobs = []
    for _ in range(n):
        content = [rng.randrange(5, 50) for _ in range(rng.randrange(1, 300))]
        stems = [[rng.randrange(5, 50) for _ in range(rng.randrange(1, 6))] for _ in range(rng.randrange(1, 4))]
        prompts = []
        for _ in range(rng.randrange(1, 12)):
            p = content + rng.choice(stems) + [rng.randrange(5, 50) for _ in range(rng.randrange(1, 8))]
            if p not in prompts:
                prompts.append(p)
        opts = [[[rng.randrange(5, 50) for _ in range(rng.randrange(1, 3))] for _ in range(rng.randrange(1, 4))] for _ in prompts]
        jobs.append((prompts, [ReadPlan.build(o) for o in opts]))
    return jobs


def _step(s, t):
    return (s * 1_000_003 + t + 1) % (1 << 61)


def _run_indexed(b, plan):
    """the toy recurrence through the indexed plan: states read at h0_idx of the previous level, written at ht_idx"""
    got = np.zeros(len(b.ids), np.int64)
    prev_states = None
    for calls, nk in zip(plan.levels, plan.n_keep, strict=True):
        states = [None] * nk
        for c in calls:
            idx, cu = c.idx.numpy(), c.cu.numpy()
            for s in range(len(cu) - 1):
                h = 0 if c.h0_idx is None or int(c.h0_idx[s]) < 0 else prev_states[int(c.h0_idx[s])]
                for i in idx[cu[s] : cu[s + 1]]:
                    h = _step(h, int(b.ids[i]))
                    got[i] = h
                if c.ht_idx is not None and int(c.ht_idx[s]) >= 0:
                    states[int(c.ht_idx[s])] = h
        assert all(x is not None for x in states)
        prev_states = states
    return got


def test_tree_plan_matches_paths():
    rng = random.Random(0)
    for trial in range(20):
        pk = Packer(4096, 8192, align=True)
        jobs = _jobs(rng, rng.randrange(1, 8))
        trees = [build_grouped_tree(p, pl) for p, pl in jobs]
        batches = []
        for k, t in enumerate(trees):
            batches += list(pk.push(k, t))
        batches += list(pk.flush())
        for b in batches:
            real = b.art >= 0
            plan = tree_plan(b.prev, real, "cpu", max_init=3, path="levels")
            # 1. path spelled by prev = the tokens of a prompt (each read row's path ends a prompt or a lead-in)
            for i in np.flatnonzero(real):
                path, j = [], i
                while j >= 0:
                    path.append(int(b.ids[j]))
                    j = b.prev[j]
                assert len(path) == b.pos[i] + 1
            # 2. a toy recurrence run level by level == along the path
            want = np.zeros(len(b.ids), np.int64)
            for i in range(len(b.ids)):
                if real[i]:
                    want[i] = _step(want[b.prev[i]] if b.prev[i] >= 0 else 0, int(b.ids[i]))
            got = _run_indexed(b, tree_plan(b.prev, real, "cpu", path="indexed", short=4))
            assert (got[real] == want[real]).all(), trial
            got = np.zeros(len(b.ids), np.int64)
            kept = None
            for calls in plan.levels:
                new = []
                for c in calls:
                    idx, cu = c.idx.numpy(), c.cu.numpy()
                    fin = []
                    for s in range(len(cu) - 1):
                        st = 0 if c.parent is None else int(kept[int(c.parent[s])])
                        for i in idx[cu[s] : cu[s + 1]]:
                            st = _step(st, int(b.ids[i]))
                            got[i] = st
                        fin.append(st)
                    if c.keep:
                        new += fin
                kept = new
            assert (got[real] == want[real]).all(), trial
            # 3. conv taps: path predecessors 3, 2, 1 back (or the zero row)
            T = len(b.ids)
            for i in np.flatnonzero(real):
                p = [i]
                for _ in range(3):
                    p.append(b.prev[p[-1]] if p[-1] < T and b.prev[p[-1]] >= 0 else T)
                assert list(plan.conv[i].numpy()) == [p[3], p[2], p[1], i]
            # 4. compact rows: the same plan over the real tokens packed in front of zero rows
            R = int(real.sum())
            rows = (R + 127) // 128 * 128
            cp = tree_plan(b.prev, real, "cpu", path="indexed", short=4, rows=rows)
            sidx = cp.sidx.numpy()
            assert (sidx == np.flatnonzero(real)).all()
            ip = tree_plan(b.prev, real, "cpu", path="indexed", short=4)
            for lc, ls in zip(cp.levels, ip.levels, strict=True):
                for c, s in zip(lc, ls, strict=True):
                    assert (sidx[c.idx.numpy()] == s.idx.numpy()).all() and (c.cu.numpy() == s.cu.numpy()).all()
            back = np.full(rows + 1, T, np.int64)  # compact row -> stream position; the zero row -> T
            back[:R] = sidx
            assert (back[cp.conv.numpy()[:R]] == ip.conv.numpy()[sidx]).all() and (cp.conv.numpy()[R:] == rows).all()
            ids_c = np.zeros(rows, np.int64)
            ids_c[:R] = b.ids[sidx]
            got = _run_indexed(type("B", (), {"ids": ids_c})(), cp)
            assert (got[:R] == want[sidx]).all(), trial


def test_plan_segments_and_kept_states_for_extraction():
    """seg_of_row / seg_level / seg_slot name each row's segment and where its final state lives; `keep` makes a leaf's segment
    (no children) keep its state too (the extraction decoder starts from the prefix's state)"""
    rng = random.Random(1)
    pk = Packer(4096, 8192, align=True)
    jobs = _jobs(rng, 3)
    batches = []
    for k, (p, pl) in enumerate(jobs):
        batches += list(pk.push(k, build_grouped_tree(p, pl)))
    batches += list(pk.flush())
    for b in batches:
        real = b.art >= 0
        R = int(real.sum())
        rows = (R + 127) // 128 * 128
        plan = tree_plan(b.prev, real, "cpu", path="indexed", short=4, rows=rows)
        c_of = np.cumsum(real) - 1
        assert plan.seg_of_row.shape == (rows,) and (plan.seg_of_row[R:] == -1).all() and (plan.seg_of_row[:R] >= 0).all()
        n_seg = plan.n_segments
        assert plan.seg_level.shape == (n_seg,) and plan.seg_slot.shape == (n_seg,)
        for o in b.owners:  # the prefix (a fork: it has children) keeps its state at level 0
            end = o.start + o.prefix_len - 1
            if o.prefix_len and b.prev[end + 1] == end and (b.prev == end).sum() > 1:
                sg = plan.seg_of_row[c_of[end]]
                assert plan.seg_level[sg] == 0 and plan.seg_slot[sg] >= 0
        # a leaf row's segment keeps no state ... unless asked to
        leaves = [i for i in np.flatnonzero(real) if not (b.prev == i).any()]
        leaf = leaves[0]
        sg = plan.seg_of_row[c_of[leaf]]
        assert plan.seg_slot[sg] == -1
        kept = tree_plan(b.prev, real, "cpu", path="indexed", short=4, rows=rows, keep=[leaf])
        sg2 = kept.seg_of_row[c_of[leaf]]
        assert sg2 == sg and kept.seg_slot[sg2] >= 0
        lvl = kept.seg_level[sg2]
        assert kept.n_keep[lvl] == plan.n_keep[lvl] + 1
        # the kept state is written by the call that runs the leaf's segment (ht_idx of that segment >= 0)
        row = c_of[leaf]
        hit = False
        for c in kept.levels[lvl]:
            idx, cu = c.idx.numpy(), c.cu.numpy()
            where = np.flatnonzero(idx == row)
            if len(where):
                seg_in_call = int(np.searchsorted(cu, where[0], side="right") - 1)
                hit = c.ht_idx is not None and int(c.ht_idx[seg_in_call]) >= 0
        assert hit


def test_readout_logprobs_are_the_deciders_masked_softmax():
    """pplx-decider: softmax over a question's options of (hidden x readout / T); the engine's log-probs over the vocabulary put
    the codes' logits / T on the code tokens and -inf elsewhere, so renormalising over the options gives the same probabilities"""
    import math

    import torch

    from spinf_serve.scorer import readout_logprobs

    torch.manual_seed(0)
    h = torch.randn(3, 16, dtype=torch.bfloat16)
    R = torch.randn(5, 16, dtype=torch.bfloat16)  # 5 answer codes
    ids = torch.tensor([7, 2, 9, 11, 4])
    T = 2.2
    lp = readout_logprobs(torch, h, R, ids, T, vocab=20)
    assert lp.shape == (3, 20) and lp.dtype == torch.float32
    assert torch.isinf(lp[:, 0]).all() and torch.isinf(lp[:, 19]).all()
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-5)
    # a question using codes 0 and 2 (tokens 7 and 9): the decider's probabilities
    logits = (h.float() @ R.float().t()) / T
    ref = torch.softmax(logits[:, [0, 2]], -1)
    ours = torch.softmax(lp[:, [7, 9]], -1)
    assert torch.allclose(ours, ref, atol=1e-3)
    assert math.isinf(float(lp[0, 1]))
