import os

import numpy as np
import pytest

from spinf_serve.packer import BLOCK, Packer, ReadPlan, block_tables, build_grouped_tree, build_tree, split_trees, trie_branches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tree_shares_prefix_and_reads_last_prompt_token():
    t = build_tree([[1, 5, 6, 7], [1, 5, 6, 8, 9]], [[[10], [11]], [[12], [13]]])
    assert t.prefix == [1, 5, 6]
    assert t.branches == [[7], [8, 9]]
    # single-token options are read at the last token of their prompt's branch
    assert t.reads == [(0, 0, 10, 0, 0), (0, 0, 11, 0, 1), (1, 1, 12, 1, 0), (1, 1, 13, 1, 1)]
    assert t.n_tokens == 3 + 1 + 2


def test_tree_keeps_one_token_per_prompt_when_prompts_are_equal():
    t = build_tree([[1, 2, 3], [1, 2, 3]], [[[4]], [[5]]])
    assert t.prefix == [1, 2]
    assert t.branches == [[3], [3]]


def test_multi_token_options_share_lead_in_branches():
    # three options: [20], [21, 22], [21, 23] -> the two options starting with 21 share the extended branch
    t = build_tree([[1, 2, 3]], [[[20], [21, 22], [21, 23]]])
    assert t.branches == [[3], [3, 21]]
    assert sorted(t.reads) == sorted(
        [
            (0, 0, 20, 0, 0),
            (0, 0, 21, 0, 1),
            (1, 1, 22, 0, 1),
            (0, 0, 21, 0, 2),
            (1, 1, 23, 0, 2),
        ]
    )
    assert t.n_options == [3]


def test_empty_option_rejected():
    with pytest.raises(ValueError):
        build_tree([[1, 2]], [[[]]])


def test_split_trees_halves_until_each_run_fits():
    prompts = [[1] * 50 + [i] * 30 for i in range(2, 10)]  # 50 shared + 30 own tokens each
    options = [[[100], [101]]] * len(prompts)
    whole = build_tree(prompts, options).n_tokens
    assert whole == 50 + 8 * 30
    runs = split_trees(prompts, options, 50 + 2 * 30)
    assert [a for a, _ in runs] == [0, 2, 4, 6]
    assert all(t.n_tokens <= 110 for _, t in runs)
    with pytest.raises(ValueError):
        split_trees(prompts, options, 60)


def test_packer_streams_respect_budget_and_rows():
    packer = Packer(256)
    trees = [build_tree([[1] + [i] * n, [1] + [i] * n + [7]], [[[9], [8]], [[9], [8]]]) for i, n in ((2, 100), (3, 90), (4, 80))]
    batches = [b for i, t in enumerate(trees) for b in packer.push(i, t, window=2)] + list(packer.flush())
    seen = sorted(o.job_id for b in batches for o in b.owners)
    assert seen == [0, 1, 2]
    for b in batches:
        assert b.length % BLOCK == 0 and b.length <= 256
        assert b.n_tokens == sum(o.n_tokens for o in b.owners)
        assert len(b.sel) == sum(o.n_rows for o in b.owners)
        # every read row is the last token of a branch of its job, and both options of a prompt share it
        for o in b.owners:
            assert len(o.plan) == 4 and o.n_rows == 2


def dense_tree_mask(art, end):
    n = len(art)
    q = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    m = (k <= q) & (art[q] >= 0) & (art[k] == art[q]) & (q < end[k])
    return m | (q == k)


def test_block_tables_cover_the_dense_mask():
    t1 = build_tree([[1] * 200 + [2] * 40, [1] * 200 + [3] * 60], [[[9]], [[9]]])
    t2 = build_tree([[1] * 150 + [4] * 10], [[[9]]])
    packer = Packer(1024)
    list(packer.push("a", t1))
    list(packer.push("b", t2))
    (b,) = list(packer.flush())
    (pn, pi, fn, fi), sliding = block_tables(b.art, b.end, b.pos, window=None)
    assert sliding is None
    m = dense_tree_mask(b.art, b.end)
    nb = b.length // BLOCK
    for i in range(nb):
        partial = set(pi[i, : pn[i]].tolist())
        full = set(fi[i, : fn[i]].tolist())
        assert not partial & full
        for j in range(nb):
            blk = m[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK]
            if j in full:
                assert blk.all()
            elif j not in partial:
                assert not blk.any()


def job_tables(b, owner, window):
    """a job's attention tables relative to its first block: (per q block: partial kv blocks, full kv blocks), dense and sliding"""
    s0 = owner.start // BLOCK
    nb = (owner.start + owner.n_tokens + BLOCK - 1) // BLOCK - s0
    out = []
    for pn, pi, fn, fi in block_tables(b.art, b.end, b.pos, window, always_window=True):
        out.append([(sorted(int(j) - s0 for j in pi[i, : pn[i]]), sorted(int(j) - s0 for j in fi[i, : fn[i]])) for i in range(s0, s0 + nb)])
    return out


def test_aligned_jobs_start_on_blocks_and_keep_their_tables():
    rng = np.random.default_rng(3)
    trees = [
        build_tree([list(rng.integers(1, 50, n)) + [7, k], list(rng.integers(1, 50, n)) + [8]], [[[9], [8]], [[9], [8]]])
        for k, n in enumerate((300, 77, 513, 129, 40))
    ]  # 513: past the window, the others not
    alone = {}
    for k, t in enumerate(trees):
        p = Packer(4096, align=True)
        list(p.push(k, t))
        (b,) = list(p.flush())
        alone[k] = job_tables(b, b.owners[0], window=256)
    p = Packer(4096, align=True)
    for k, t in enumerate(trees):
        list(p.push(k, t))
    (b,) = list(p.flush())
    assert b.n_tokens == sum(t.n_tokens for t in trees)  # padding is not counted
    for o in b.owners:
        assert o.start % BLOCK == 0
        assert job_tables(b, o, window=256) == alone[o.job_id]


def test_dense_packing_shares_blocks_between_jobs():
    t1 = build_tree([[1] * 100], [[[9]]])
    t2 = build_tree([[2] * 100], [[[9]]])
    p = Packer(4096)
    list(p.push("a", t1))
    list(p.push("b", t2))
    (b,) = list(p.flush())
    assert sorted(o.start for o in b.owners) == [0, t1.n_tokens]  # the second job starts inside the first one's last block


def check_tables(m, tables, nb):
    """full blocks attend everywhere, unlisted blocks nowhere"""
    pn, pi, fn, fi = tables
    for i in range(nb):
        partial = set(pi[i, : pn[i]].tolist())
        full = set(fi[i, : fn[i]].tolist())
        assert not partial & full
        for j in range(nb):
            blk = m[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK]
            if j in full:
                assert blk.all(), (i, j)
            elif j not in partial:
                assert not blk.any(), (i, j)


def test_sliding_tables_follow_the_window():
    rng = np.random.default_rng(1)
    for _trial in range(12):
        packer = Packer(4096)
        for a in range(int(rng.integers(1, 4))):
            stem = [1] * int(rng.integers(100, 1200))
            prompts = [stem + rng.integers(2, 6, int(rng.integers(1, 200))).tolist() for _ in range(int(rng.integers(1, 4)))]
            list(packer.push(str(a), build_grouped_tree(prompts, [ReadPlan.build([[9], [10, 11]])] * len(prompts))))
        (b,) = list(packer.flush())
        window = int(rng.integers(100, 700))
        if int(b.pos.max()) < window:
            continue
        full_tabs, sliding = block_tables(b.art, b.end, b.pos, window)
        m = dense_tree_mask(b.art, b.end)
        m &= (b.pos[:, None] - b.pos[None, :] < window) | np.eye(b.length, dtype=bool)
        nb = b.length // BLOCK
        check_tables(m, sliding, nb)
        assert sliding[2].sum() == 0 and sliding[0].sum() <= full_tabs[0].sum() + full_tabs[2].sum()


def test_sliding_tables_skip_blocks_past_the_window():
    t = build_tree([[1] * 2000 + [2]], [[[9]]])
    b = Packer(4096)._pack([("t", t)], 2048)
    _, (pn, pi, fn, fi) = block_tables(b.art, b.end, b.pos, 512)
    # q block 14 (1792-1919): kv blocks 10 (1280-1407, pairs 385..639 apart) to 14 in reach, 0-9 out of it; all partial, in order
    assert pi[14, : pn[14]].tolist() == [10, 11, 12, 13, 14]
    assert fn.sum() == 0


def test_trie_runs_shared_template_text_and_lead_ins_once():
    # two combinations of one template (stem 5 6 7) and a prompt under three options, two of them with the lead-in 21
    plans = [ReadPlan.build([[30], [31]]), ReadPlan.build([[30], [31]]), ReadPlan.build([[20], [21, 22], [21, 23]])]
    t = build_grouped_tree([[1, 2, 5, 6, 7, 8], [1, 2, 5, 6, 7, 9], [1, 2, 4]], plans)
    assert t.prefix == [1, 2]
    assert t.branches == [[5, 6, 7], [8], [9], [4, 21]]  # a chain without forks is one branch
    assert t.parents == [-1, 0, 0, -1]
    assert t.prompt_rows == [[(1, 0)], [(2, 0)], [(3, 0), (3, 1)]]
    assert t.n_tokens == 2 + 3 + 1 + 1 + 2  # flat: 2 + 4 + 4 + 1 + 2


def test_trie_merges_equal_prompts_and_nested_lead_ins():
    branches, parents, rows = trie_branches([[3], [3]], [ReadPlan.build([[9]]), ReadPlan.build([[4, 5, 6], [4, 5, 7], [4, 8]])])
    assert branches == [[3, 4, 5]] and parents == [-1]
    # prompt 1: its last token, then (plan order) lead-in (4, 5) at 4 and 5, lead-in (4,) at 4
    assert rows == [[(0, 0)], [(0, 0), (0, 1), (0, 2), (0, 1)]]


def visible(b, q):
    """what token q sees under the dense mask, in position order"""
    m = dense_tree_mask(b.art, b.end)[q]
    ks = np.nonzero(m)[0]
    return [int(b.ids[k]) for k in ks], [int(b.pos[k]) for k in ks]


def test_trie_reads_see_exactly_their_prompt():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n_p = int(rng.integers(1, 9))
        stem = [1] + rng.integers(2, 6, 3).tolist()
        prompts = [stem + rng.integers(2, 6, int(rng.integers(1, 6))).tolist() for _ in range(n_p)]
        n_o = [int(rng.integers(1, 6)) for _ in prompts]
        plans = [ReadPlan.build([rng.integers(6, 9, int(rng.integers(1, 4))).tolist() for _ in range(k)]) for k in n_o]
        t = build_grouped_tree(prompts, plans)
        other = build_grouped_tree([[1, 7, 7, 2], [1, 7, 7, 3]], [ReadPlan.build([[9]])] * 2)
        packer = Packer(1024)
        list(packer.push("o", other))
        list(packer.push("t", t))
        (b,) = list(packer.flush())
        o = next(o for o in b.owners if o.job_id == "t")
        k0 = sum(x.n_rows for x in b.owners[: b.owners.index(o)])
        for c, (x, plan) in enumerate(zip(prompts, plans, strict=True)):
            # row 0 reads the prompt itself; rows of a lead-in read the prompt + the lead-in's first j tokens
            want = [x] + [x + list(lead[:j]) for lead in plan.leadins for j in range(1, len(lead) + 1)]
            for r, w in enumerate(want):
                q = int(b.sel[k0 + o.row_start[c] + r])
                ids, pos = visible(b, q)
                assert ids == w, (trial, c, r)
                assert pos == list(range(len(w)))
        (pn, pi, fn, fi), _ = block_tables(b.art, b.end, b.pos, window=None)
        m = dense_tree_mask(b.art, b.end)
        for i in range(b.length // BLOCK):
            for j in range(b.length // BLOCK):
                blk = m[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK]
                if j in fi[i, : fn[i]]:
                    assert blk.all()
                elif j not in pi[i, : pn[i]]:
                    assert not blk.any()


def test_block_tables_skip_finished_sibling_branches():
    # a long prompt branch, then a sibling: the sibling's q blocks never visit the first branch's blocks
    t = build_grouped_tree([[1] * 200 + [2] * 300, [1] * 200 + [3] * 300], [ReadPlan.build([[9]])] * 2)
    b = Packer(1024)._pack([("t", t)], 1024)
    (pn, pi, fn, fi), _ = block_tables(b.art, b.end, b.pos, window=None)
    # prefix 0-199, first branch 200-499, second 500-799: q block 5 (640-767) skips block 2 (256-383, first branch only)
    listed = set(pi[5, : pn[5]].tolist()) | set(fi[5, : fn[5]].tolist())
    assert 2 not in listed and {1, 3, 4, 5} <= listed
    assert 0 in fi[5, : fn[5]]  # prefix blocks stay full


def test_packer_second_dimension_limits_costly_jobs_per_stream():
    """cost(tree) summed per stream under cost_budget (the bytes a forward captures for extraction branches): costly jobs spread
    over more streams than the token budget alone would make; free jobs pack as before; one over the budget still runs alone"""

    def tree(n, costly):
        t = build_grouped_tree([list(range(5, 5 + n)) + [3]], [ReadPlan.build([[4]])])
        if costly:
            t.xbranches = [(0, object())]
        return t

    costly = [tree(300, True) for _ in range(4)]
    cost = lambda t: 100 if t.xbranches else 0  # noqa: E731
    pk = Packer(4096, cost=cost, cost_budget=250)
    batches = [b for k, t in enumerate(costly) for b in pk.push(k, t)] + list(pk.flush())
    assert sorted(len(b.owners) for b in batches) == [2, 2]  # 1,200 tokens would fit one stream; the cost allows two per stream
    pk = Packer(4096, cost=cost, cost_budget=50)
    batches = [b for k, t in enumerate(costly) for b in pk.push(k, t)] + list(pk.flush())
    assert [len(b.owners) for b in batches] == [1, 1, 1, 1]  # each over the budget alone: still packed, alone
    pk = Packer(4096, cost=cost, cost_budget=250)
    jobs = costly + [tree(300, False) for _ in range(3)]
    batches = [b for k, t in enumerate(jobs) for b in pk.push(k, t)] + list(pk.flush())
    assert sum(len(b.owners) for b in batches) == 7 and len(batches) == 2  # the free jobs fill the two streams
