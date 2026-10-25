"""Tree packing: one job = the prompts of one piece of content, each followed by its options.

The prompts of a job share their longest common token prefix (normally the content), which is sent once; every prompt continues as
its own branch that attends to the prefix and to itself only (tree attention, per-branch position ids). An option of several tokens
is read by the chain rule: token j of the option is read at the position of token j-1, on a branch extended by the option's leading
tokens (shared by options with the same lead-in). Jobs are packed back to back into streams of up to `budget` tokens.

Grouped jobs (the protocol path, spinf_serve.compiler) go further: their branches form a token trie (trie_branches), so text that
prompts share after the content (a template's stem across its combinations, an instruction several queries repeat, a prompt under
its options' lead-ins) is run once. A branch attends to the prefix, its ancestors and itself; the stream lays the trie out in
pre-order, so a token's descendants are the contiguous run up to its subtree's end (Batch.end), which is all the mask needs.

Origin: spinf's internal scorer (torchscore.py: Packer, stream mode); answer pairs generalised to any number of options per prompt.
"""

from dataclasses import dataclass, field

import numpy as np

from spinf_serve.media import Image

BLOCK = 128  # the flex attention's block size: packed streams are multiples of it, jobs align to it (align_jobs)


@dataclass
class Tree:
    prefix: list[int]
    branches: list[list[int]]
    # (branch index, offset in the branch, target token id, prompt index, option index)
    reads: list[tuple[int, int, int, int, int]]
    n_options: list[int]  # per prompt

    @property
    def n_tokens(self) -> int:
        return len(self.prefix) + sum(map(len, self.branches))


def build_tree(prompt_ids: list[list[int]], option_ids: list[list[list[int]]]) -> Tree:
    """prompt_ids: the full tokenization of each prompt (with the BOS); option_ids[i]: the token ids of prompt i's options.
    The prefix is the longest common prefix of the prompts, and every prompt keeps at least one token of its own, so each branch is
    exactly what the model would see after the shared part."""
    if not prompt_ids:
        raise ValueError("a job needs at least one prompt")
    p = 0
    m = min(map(len, prompt_ids)) - 1
    while p < m and all(x[p] == prompt_ids[0][p] for x in prompt_ids):
        p += 1
    prefix = prompt_ids[0][:p]
    branches: list[list[int]] = []
    reads = []
    for c, (x, opts) in enumerate(zip(prompt_ids, option_ids, strict=True)):
        suf = x[p:]
        key = {(): len(branches)}
        branches.append(suf)
        for side, t in enumerate(opts):
            if not t:
                raise ValueError(f"prompt {c}: option {side} is empty")
            for j in range(len(t)):
                k = () if j == 0 else tuple(t[:-1])
                if k not in key:
                    key[k] = len(branches)
                    branches.append(suf + list(k))
                reads.append((key[k], len(suf) - 1 + j, t[j], c, side))
    return Tree(prefix, branches, reads, [len(o) for o in option_ids])


@dataclass(eq=False)
class ReadPlan:
    """How to read a set of options (up to thousands) after one prompt, vectorised: the prompt's read rows are its last token
    (row 0) and, for each distinct lead-in L = option[:-1] of the multi-token options, the positions of L's tokens on a branch
    extended by L; entry e adds log P(entry_tok[e]) at row entry_row[e] to option entry_opt[e]. The rows depend on the lead-ins
    only, not on the number of options: 1,000 one-token options cost one row and 1,000 lm-head columns."""

    leadins: list[tuple[int, ...]]
    entry_row: np.ndarray
    entry_tok: np.ndarray
    entry_opt: np.ndarray
    n_opts: int
    n_rows: int

    @classmethod
    def build(cls, option_ids: list[list[int]]) -> "ReadPlan":
        leadins: dict[tuple[int, ...], int] = {}
        start: list[int] = []
        rows, toks, opts = [], [], []
        n_rows = 1
        for o, t in enumerate(option_ids):
            if not t:
                raise ValueError(f"option {o} is empty")
            rows.append(0)
            toks.append(t[0])
            opts.append(o)
            if len(t) > 1:
                key = tuple(t[:-1])
                if key not in leadins:
                    leadins[key] = len(start)
                    start.append(n_rows)
                    n_rows += len(key)
                s0 = start[leadins[key]]
                for j in range(1, len(t)):
                    rows.append(s0 + j - 1)
                    toks.append(t[j])
                    opts.append(o)
        return cls(list(leadins), np.array(rows, np.int64), np.array(toks, np.int64), np.array(opts, np.int64), len(option_ids), n_rows)


@dataclass
class GroupedTree:
    """a job whose prompts are read with ReadPlans (spinf_serve.compiler); prompt_rows[c] = the (branch, offset) of each of
    prompt c's read rows, in its plan's row order. parents[b] = the branch that branch b continues (-1: the prefix), branches in
    pre-order (a parent before its subtree, each subtree contiguous); empty = every branch continues the prefix"""

    prefix: list[int]
    branches: list[list[int]]
    prompt_rows: list[list[tuple[int, int]]]
    plans: list[ReadPlan]
    # media in the shared prefix: (offset of the first placeholder, media.Image | media.Audio); the placeholders' embeddings are
    # replaced by the media's (scorer), and an image's placeholders attend to each other both ways
    media: list = field(default_factory=list)
    # extraction slots read in the same pass: (branch index, extract.ExtractPrompt); the branch is the slot's template text, the
    # scorer keeps its K/V and the next-token distribution at its end, and decodes the answer from there (decode.py). Branch
    # index -1: no template tokens, the slot starts right after the prefix (hybrid models' Scorer.extract)
    xbranches: list = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    # branches that start a query group on a BLOCK boundary (relative to the job's start; the packer pads before them): a group's
    # prompts then sit in the same blocks whatever groups come before it (protocol Query.group, compiler._tree)
    aligned: list[int] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        """stream tokens, the padding before aligned groups included"""
        n = len(self.prefix)
        al = set(self.aligned)
        for b, br in enumerate(self.branches):
            if b in al:
                n = (n + BLOCK - 1) // BLOCK * BLOCK
            n += len(br)
        return n


def trie_branches(suffixes: list[list[int]], plans: list[ReadPlan]) -> tuple[list[list[int]], list[int], list[list[tuple[int, int]]]]:
    """the prompts' suffixes (each non-empty) and, under each, its plan's lead-ins, as one token trie: every distinct token path
    is run once. -> (branches in pre-order, parents, prompt_rows): a branch is a chain of trie nodes that ends where the trie forks
    or a path ends; prompt_rows[c] = (branch, offset) of prompt c's read rows (its last token, then each lead-in's tokens), as
    build_grouped_tree lays them out"""
    kids: list[dict[int, int]] = [{}]  # node 0: the end of the shared prefix
    tok = [-1]

    def walk(node: int, seq) -> list[int]:
        path = []
        for t in seq:
            nxt = kids[node].get(t)
            if nxt is None:
                nxt = kids[node][t] = len(kids)
                kids.append({})
                tok.append(t)
            node = nxt
            path.append(node)
        return path

    reads = []
    for s, plan in zip(suffixes, plans, strict=True):
        end = walk(0, s)[-1]
        nodes = [end]
        for lead in plan.leadins:
            nodes += walk(end, lead)
        reads.append(nodes)
    loc: list[tuple[int, int]] = [(-1, -1)] * len(kids)
    branches: list[list[int]] = []
    parents: list[int] = []
    stack = [(c, -1) for c in reversed(kids[0].values())]
    while stack:
        node, parent = stack.pop()
        b = len(branches)
        toks: list[int] = []
        while True:
            loc[node] = (b, len(toks))
            toks.append(tok[node])
            if len(kids[node]) != 1:
                break
            (node,) = kids[node].values()
        branches.append(toks)
        parents.append(parent)
        stack += [(c, b) for c in reversed(kids[node].values())]
    return branches, parents, [[loc[n] for n in nodes] for nodes in reads]


def build_grouped_tree(prompt_ids: list[list[int]], plans: list[ReadPlan]) -> GroupedTree:
    """the prompts share their longest common prefix (the content); after it, the prompts and their plans' lead-ins form a token
    trie (trie_branches)"""
    if not prompt_ids:
        raise ValueError("a job needs at least one prompt")
    p = 0
    m = min(map(len, prompt_ids)) - 1
    while p < m and all(x[p] == prompt_ids[0][p] for x in prompt_ids):
        p += 1
    branches, parents, prompt_rows = trie_branches([x[p:] for x in prompt_ids], plans)
    return GroupedTree(prompt_ids[0][:p], branches, prompt_rows, list(plans), parents=parents)


def split_grouped_trees(prompt_ids: list[list[int]], plans: list[ReadPlan], budget: int) -> list[tuple[int, GroupedTree]]:
    """split_trees for grouped jobs: contiguous runs of prompts that each fit the budget (each run re-sends the prefix)"""
    out = []
    stack = [(0, len(prompt_ids))]
    while stack:
        a, b = stack.pop()
        t = build_grouped_tree(prompt_ids[a:b], plans[a:b])
        if t.n_tokens <= budget:
            out.append((a, t))
        elif b - a == 1:
            raise ValueError(f"prompt {a} alone is {t.n_tokens} tokens, over the budget {budget}")
        else:
            mid = (a + b) // 2
            stack += [(mid, b), (a, mid)]
    return sorted(out, key=lambda x: x[0])


def split_trees(prompt_ids: list[list[int]], option_ids: list[list[list[int]]], budget: int) -> list[tuple[int, Tree]]:
    """-> [(index of the first prompt, tree)]: one tree when the job fits the budget, else the prompts halved into contiguous runs
    until each fits (each run re-sends its shared prefix)"""
    out = []
    stack = [(0, len(prompt_ids))]
    while stack:
        a, b = stack.pop()
        t = build_tree(prompt_ids[a:b], option_ids[a:b])
        if t.n_tokens <= budget:
            out.append((a, t))
        elif b - a == 1:
            raise ValueError(f"prompt {a} alone is {t.n_tokens} tokens, over the budget {budget}")
        else:
            mid = (a + b) // 2
            stack += [(mid, b), (a, mid)]
    return sorted(out, key=lambda x: x[0])


@dataclass
class Owner:
    job_id: object
    n_rows: int  # read rows of this job in the batch's `sel`
    n_tokens: int  # packed tokens of this job
    plan: list[tuple[int, int, int, int]]  # (row, target id, prompt index, option index); empty for a grouped job
    n_options: list[int]
    # grouped jobs: the first read row of each prompt (relative to the job's rows) and its ReadPlan
    row_start: np.ndarray | None = None
    plans: list[ReadPlan] | None = None
    start: int = 0  # stream position of the job's first token (its shared prefix starts here)
    prefix_len: int = 0
    # extraction branches: (stream position of the branch's first token, its length, extract.ExtractPrompt)
    xbranches: list = field(default_factory=list)


@dataclass
class Batch:
    ids: np.ndarray  # [N] int64
    pos: np.ndarray  # [N] int32, restarts at the prefix length for every branch
    seg: np.ndarray  # [N] int32: 0 prefix, k branch k, -1 pad
    art: np.ndarray  # [N] int32: job slot in the stream, -1 pad
    sel: np.ndarray  # read positions, int64
    owners: list[Owner]
    n_tokens: int
    media: list = field(default_factory=list)  # (stream position of the first placeholder, media object)
    grp: np.ndarray | None = None  # [N] int32: image id per token (bidirectional block), -1 elsewhere; None without images
    # [N] int32: the end (exclusive) of the token's subtree: the job's end for the prefix, the branch's end plus its descendants
    # for a branch token; -1 pad. Token q sees k <= q of its job when q < end[k]
    end: np.ndarray | None = None
    # [N] int64: the token before each token on its path (prefix -> branch -> sub-branch), -1 for a job's first token and pads;
    # the recurrent layers of hybrid models (qwen.py) run along it
    prev: np.ndarray | None = None

    @property
    def length(self) -> int:
        return len(self.ids)


class Packer:
    """align: every job starts on a BLOCK boundary (pads between jobs), so a job's attention blocks, and with fixed activation scales
    its scores, do not depend on what it is packed with (docs/BATCH_INVARIANCE.md); off = the dense layout"""

    def __init__(
        self, budget: int, max_tokens: int | None = None, align: bool = False, min_stream: int = 0, cost=None, cost_budget: int = 0
    ):
        self.budget = (budget + BLOCK - 1) // BLOCK * BLOCK
        self.max_tokens = max(self.budget, (max_tokens or 0) + BLOCK - 1) // BLOCK * BLOCK
        self.align = align
        # aligned streams are padded to at least min_stream tokens (one GEMM kernel per shape, see EngineConfig.min_stream)
        self.min_stream = min(self.budget, (min_stream + BLOCK - 1) // BLOCK * BLOCK) if align else 0
        # a second packing dimension: cost(tree) units (e.g. the bytes its extraction branches make the forward capture) summed
        # per stream up to cost_budget; a tree over it alone still runs (alone). None: tokens only
        self.cost, self.cost_budget = cost, cost_budget
        self.buf: list[tuple[object, Tree | GroupedTree]] = []

    def push(self, job_id, tree: "Tree | GroupedTree", window: int = 96):
        """add one job; yields batches once `window` jobs are buffered. flush() yields what is left. A job over the budget (up to
        max_tokens) is not packed: it runs alone, at once (a long document)."""
        if tree.n_tokens > self.max_tokens:
            raise ValueError(f"job {job_id}: {tree.n_tokens} tokens exceed the maximum {self.max_tokens}")
        if tree.n_tokens > self.budget:
            yield self._pack([(job_id, tree)], self._length(tree.n_tokens))
            return
        self.buf.append((job_id, tree))
        if len(self.buf) >= window:
            yield from self._flush(final=False)

    def flush(self):
        yield from self._flush(final=True)

    def _flush(self, final: bool):
        # first-fit-decreasing bin packing into streams of at most `budget` tokens; the least full bin waits for the next window
        buf = sorted(self.buf, key=lambda it: -it[1].n_tokens)
        bins: list[list] = []  # [tokens, items, cost]
        for it in buf:
            n = self._size(it[1])
            c = self.cost(it[1]) if self.cost is not None else 0
            for b in bins:
                if b[0] + n <= self.budget and (c == 0 or b[2] + c <= self.cost_budget):
                    b[0] += n
                    b[1].append(it)
                    b[2] += c
                    break
            else:
                bins.append([n, [it], c])
        if not final and len(bins) > 1:
            bins.sort(key=lambda b: -b[0])
            self.buf = bins.pop()[1]
        else:
            self.buf = []
        for n, items, _ in bins:
            yield self._pack(items, self._length(n))

    def _length(self, n: int) -> int:
        """the stream length for n tokens: whole blocks, at least min_stream"""
        return max((n + BLOCK - 1) // BLOCK * BLOCK, self.min_stream)

    def _size(self, tree) -> int:
        """the stream tokens a job takes: its tokens, rounded up to a block when aligned"""
        return (tree.n_tokens + BLOCK - 1) // BLOCK * BLOCK if self.align else tree.n_tokens

    @staticmethod
    def _rows(reads, start, sel):
        """one read row per distinct position (shared by the options of one lead-in), appended to sel"""
        rows: dict[int, int] = {}
        plan = []
        n0 = len(sel)
        for bi, off, tgt, c, side in reads:
            q = start[bi] + off
            if q not in rows:
                rows[q] = len(sel) - n0
                sel.append(q)
            plan.append((rows[q], tgt, c, side))
        return plan, len(sel) - n0

    def _pack(self, items, n: int) -> Batch:
        ids = np.zeros(n, np.int64)
        pos = np.zeros(n, np.int32)
        seg = np.full(n, -1, np.int32)
        art = np.full(n, -1, np.int32)
        end = np.full(n, -1, np.int32)
        prev = np.full(n, -1, np.int64)
        sel: list[int] = []
        owners = []
        media: list = []
        grp = None
        k = real = 0
        for a, (jid, t) in enumerate(items):
            grouped = isinstance(t, GroupedTree)
            P = len(t.prefix)
            for off, obj in getattr(t, "media", ()):
                if off + obj.n_tokens > P:
                    raise ValueError("media must lie in the shared prefix")
                media.append((k + off, obj))
                if isinstance(obj, Image):
                    if grp is None:
                        grp = np.full(n, -1, np.int32)
                    grp[k + off : k + off + obj.n_tokens] = len(media)
            s0 = k
            start = []
            ids[k : k + P] = t.prefix
            pos[k : k + P] = range(P)
            seg[k : k + P] = 0
            prev[k + 1 : k + P] = np.arange(k, k + P - 1)
            k += P
            parents = getattr(t, "parents", None) or [-1] * len(t.branches)
            aligned = set(getattr(t, "aligned", ()))
            depth: list[int] = []
            art[s0:k] = a
            for b, br in enumerate(t.branches, 1):
                pa = parents[b - 1]
                d = 0 if pa < 0 else depth[pa] + len(t.branches[pa])
                depth.append(d)
                if b - 1 in aligned:  # a query group starts on a block (pads before it, art -1)
                    k = s0 + (k - s0 + BLOCK - 1) // BLOCK * BLOCK
                start.append(k)
                if len(br):
                    prev[k] = (start[pa] + len(t.branches[pa]) - 1) if pa >= 0 else (s0 + P - 1 if P else -1)
                    prev[k + 1 : k + len(br)] = np.arange(k, k + len(br) - 1)
                art[k : k + len(br)] = a
                ids[k : k + len(br)] = br
                pos[k : k + len(br)] = range(P + d, P + d + len(br))
                seg[k : k + len(br)] = b
                k += len(br)
            real += P + sum(map(len, t.branches))
            end[s0 : s0 + P] = k
            size = list(map(len, t.branches))
            for b in range(len(size) - 1, -1, -1):
                if parents[b] >= 0:
                    size[parents[b]] += size[b]
            for b, br in enumerate(t.branches):
                end[start[b] : start[b] + len(br)] = start[b] + size[b]
            if grouped:
                n0 = len(sel)
                row_start = np.empty(len(t.prompt_rows), np.int64)
                for c, rows in enumerate(t.prompt_rows):
                    row_start[c] = len(sel) - n0
                    sel += [start[bi] + off for bi, off in rows]
                # an extraction branch (-1: no template tokens, the slot is read at the prefix's end: a zero-length branch there)
                xb = [(start[bi], len(t.branches[bi]), pr) if bi >= 0 else (s0 + P, 0, pr) for bi, pr in t.xbranches]
                owners.append(Owner(jid, len(sel) - n0, k - s0, [], [pl.n_opts for pl in t.plans], row_start, t.plans, s0, P, xb))
            else:
                plan, nr = self._rows(t.reads, start, sel)
                owners.append(Owner(jid, nr, k - s0, plan, t.n_options, start=s0))
            if self.align:
                k = (k + BLOCK - 1) // BLOCK * BLOCK
        return Batch(ids, pos, seg, art, np.array(sel, np.int64), owners, real, media, grp, end, prev)


def block_tables(
    art: np.ndarray, end: np.ndarray, pos: np.ndarray, window: int | None, grp: np.ndarray | None = None, always_window: bool = False
):
    """block sparsity of the tree mask from the packing metadata, on the CPU (no GPU sync). For q block i and kv block j:
    full (every pair attends: j < i, both blocks inside one job, every kv token's subtree covers block i) / partial (any pair may
    attend; mask_mod decides per element) / none (other jobs, or kv subtrees that end before block i: earlier sibling branches).
    -> (partial_num, partial_idx, full_num, full_idx) for the full-attention layers, and the same for sliding layers when the
    window can bind, else None. Sliding layers: a kv block with no token within the window of any q token is skipped (unless the
    two blocks share an image, whose placeholders attend to each other past the window), the rest are partial: a full block would
    change the kernel's accumulation order (scores drift ~0.1 logit on fp8, no faster), skipping leaves them bit-identical."""
    n = len(art)
    nb = (n + BLOCK - 1) // BLOCK
    if nb * BLOCK != n:
        pad = np.full(nb * BLOCK - n, -1, np.int32)
        art = np.concatenate([art, pad])
        end = np.concatenate([end, pad])
        pos = np.concatenate([pos, np.zeros(nb * BLOCK - n, pos.dtype)])
    arts = [set(art[k * BLOCK : (k + 1) * BLOCK].tolist()) for k in range(nb)]
    one_art = [(len(a) == 1 and -1 not in a) for a in arts]
    eb = end.reshape(nb, BLOCK)
    min_end = eb.min(1)  # pads are -1: only read for blocks without pads (one_art)
    max_end = eb.max(1)
    # always_window (aligned jobs): sliding tables even when no job in this stream reaches the window, so a job's sliding layers
    # use the same tables (no full blocks) whatever it is packed with
    binding = window is not None and (always_window or int(pos.max()) >= window)
    # position range of each block's tokens, pads left out (they only attend to themselves, on the diagonal)
    pb = pos.reshape(nb, BLOCK).astype(np.int64)
    real = art.reshape(nb, BLOCK) >= 0
    min_pos = np.where(real, pb, np.iinfo(np.int64).max).min(1)
    max_pos = np.where(real, pb, np.iinfo(np.int64).min).max(1)
    # an image's placeholders attend to each other both ways: a q block also needs the LATER kv blocks of its images
    later: dict[int, set[int]] = {}
    groups = [set() for _ in range(nb)]
    if grp is not None:
        g = np.concatenate([grp, np.full(nb * BLOCK - n, -1, np.int32)]) if nb * BLOCK != n else grp
        blocks_of: dict[int, list[int]] = {}
        for k in range(nb):
            groups[k] = set(g[k * BLOCK : (k + 1) * BLOCK].tolist()) - {-1}
            for gid in groups[k]:
                blocks_of.setdefault(gid, []).append(k)
        for bl in blocks_of.values():
            for i in bl:
                later.setdefault(i, set()).update(j for j in bl if j > i)
    out = []
    for windowed in (False, True) if binding else (False,):
        pn = np.zeros(nb, np.int32)
        pi = np.zeros((nb, nb), np.int32)
        fn = np.zeros(nb, np.int32)
        fi = np.zeros((nb, nb), np.int32)
        for i in range(nb):
            P = []
            F = []
            for j in range(i + 1):
                if windowed and j < i and min_pos[i] - max_pos[j] >= window and not groups[i] & groups[j]:
                    continue
                if j < i and one_art[i] and one_art[j] and arts[i] == arts[j] and min_end[j] >= (i + 1) * BLOCK and not windowed:
                    F.append(j)
                elif j == i or ((arts[i] & arts[j]) - {-1} and max_end[j] > i * BLOCK):
                    P.append(j)
            P += sorted(later.get(i, ()))
            pn[i] = len(P)
            pi[i, : len(P)] = P
            fn[i] = len(F)
            fi[i, : len(F)] = F
        out.append((pn, pi, fn, fi))
    return out[0], (out[1] if binding else None)
