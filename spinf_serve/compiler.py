"""The template compiler: a protocol request (protocol.ScoreRequest) -> grouped jobs for the scorer -> the response.

Per input, every prompt is `content + filled template` (the text before {?}); the options are read at its end. The content is
tokenized once and each filled template on its own, then joined; the join is checked once per (input, query) against a joint
tokenization, and a query whose join does not split cleanly is tokenized whole (its own job). Each input is one grouped job
(content = shared prefix, prompts = branches); the floors (the same filled templates without content) are one job of the
distinct floor prompts of the whole request.

Options are case-insensitive by default: each is read as the log-sum-exp of its single-token case/space variants
(options.case_variants). Token checks happen here, before any GPU work.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from spinf_serve import extract as ex
from spinf_serve.options import case_variants
from spinf_serve.packer import GroupedTree, ReadPlan, build_grouped_tree, trie_branches
from spinf_serve.protocol import (
    Limits,
    Query,
    Rejected,
    ScoreRequest,
    Template,
    check_options,
    combinations,
    content_segments,
)

BOUNDARY_TAIL_CHARS = 32
# media in usage (per request): counts and their tokens. content_tokens = text_tokens + image_tokens + video_tokens +
# audio_tokens (a video frame's "mm:ss " timestamp is text)
MEDIA_USAGE = ("images", "image_tokens", "video_frames", "video_tokens", "audio_ms", "audio_tokens")
CUT_CHARS = 32  # a template that merges with its content's end: the characters re-tokenized with it (compile_request, _cut)
CUT_TOKENS = 64  # token boundaries tried for that cut, from the content's end
MAX_BOUNDARY_TAILS = 16
LOW_OPTION_MASS = 0.9  # PLACEHOLDER: mean residual above which a query gets a low_option_mass warning


@dataclass(eq=False)
class QuerySpec:
    """one query, parsed and tokenized once (a shared query serves every input)"""

    q: Query
    param: str
    combos: list[dict[str, str]]
    filled: list[str]
    filled_ids: list[list[int]]
    option_extra: int  # billed option tokens per prompt: the tokens after each option's first
    variants: list[str]
    starts: np.ndarray  # first variant of each option (variants are grouped by option, in option order)
    plan: ReadPlan
    floor: bool


@dataclass(eq=False)
class ExtractSpec:
    """an extraction query ({*} {n} {$}): the filled templates' text before the slot, the slot kind and where a text answer ends;
    gen: a text slot generated instead of searched (extract.Generate)"""

    q: Query
    param: str
    combos: list[dict[str, str]]
    filled: list[str]
    filled_ids: list[list[int]]
    kind: str  # text | number | currency
    stop: str
    max_tokens: int
    top_k: int
    gen: ex.Generate | None = None


@dataclass
class Compiled:
    items: list[tuple[object, int, GroupedTree]]
    layout: list[list[tuple["QuerySpec | ExtractSpec", object, int]]]  # per input, per query: (spec, job key, first prompt index)
    floor_index: dict[tuple, int]  # floor key -> prompt index in the floor job
    usage: dict[str, int]
    processed_tokens: int = 0
    floor_key: object = ("floor",)
    warnings: list[dict] = field(default_factory=list)
    extract_jobs: list[ex.ExtractJob] = field(default_factory=list)
    # what the request asked for, per kind of work (the public usage blocks, plan §3.4): scoring {queries, combinations, options,
    # prompts, floor_prompts}, extraction {slots, text, number, currency}, generation {slots, sequences, max_tokens},
    # packs {count, query_tokens} (the share of query_tokens the pack groups added)
    detail: dict[str, dict[str, int]] = field(default_factory=dict)


def _tok(tok, texts: list[str], special: bool) -> list[list[int]]:
    """user text -> ids; special-token strings in the text (e.g. "<|image|>") stay text, so they can never pose as media"""
    return tok(texts, add_special_tokens=special, split_special_tokens=True)["input_ids"] if texts else []


@dataclass
class Content:
    ids: list[int]  # <bos> + the content's tokens (media as marker + placeholders + marker)
    media: list  # (offset of the first placeholder, media object)
    tail: str | None  # the trailing text run (None when the content ends with media)
    tail_len: int  # its tokens at the end of ids
    # media in the content, for billing by kind (usage): counts and their tokens (placeholders + markers). A video frame's
    # timestamp ("mm:ss ") is text: in content_tokens, not in video_tokens
    mcount: dict = field(default_factory=dict)


def _content(tok, segments: list, bos: list[int], media_ids: dict | None, param: str) -> Content:
    from spinf_serve import media as mm

    ids = list(bos)
    placed: list = []
    tail, tail_len = None, 0
    counts = dict.fromkeys(MEDIA_USAGE, 0)
    for seg in segments:
        if isinstance(seg, str):
            t = _tok(tok, [seg], False)[0]
            ids += t
            tail, tail_len = seg, len(t)
            continue
        if media_ids is None:
            raise Rejected("unsupported_content", "this engine does not read images or audio.", param)
        tail, tail_len = None, 0
        if not isinstance(seg, mm.Image) and media_ids.get("audio") is None:
            raise Rejected("unsupported_content", "this model does not read audio.", param)
        if isinstance(seg, mm.Image):
            chunks, (start, ph, end) = [seg], (media_ids["boi"], media_ids["image"], media_ids["eoi"])
        else:  # audio: segments of at most 30 s, each between its own markers
            step = mm.AUDIO_SEGMENT
            chunks = [mm.Audio(seg.frames[k : k + step]) for k in range(0, seg.n_tokens, step)]
            start, ph, end = media_ids["boa"], media_ids["audio"], media_ids["eoa"]
        for ch in chunks:
            ids.append(start)
            placed.append((len(ids), ch))
            ids += [ph] * ch.n_tokens
            ids.append(end)
        n = sum(ch.n_tokens + 2 for ch in chunks)
        if isinstance(seg, mm.Image) and seg.inline:
            counts["video_frames"] += 1
            counts["video_tokens"] += n
        elif isinstance(seg, mm.Image):
            counts["images"] += 1
            counts["image_tokens"] += n
        else:
            counts["audio_ms"] += seg.n_tokens * mm.FRAME * 1000 // mm.SAMPLE_RATE
            counts["audio_tokens"] += n
    return Content(ids, placed, tail, tail_len, counts)


def _xspec(tok, q: Query, tpl: Template, param: str, max_query_tokens: int | None) -> ExtractSpec:
    if q.options is not None:
        raise Rejected("invalid_field", "an extraction slot ({*} {n} {$}) takes no `options`.", f"{param}.options")
    gen = q.gen_spec()
    if gen is not None:
        if tpl.kind != "text":
            raise Rejected("invalid_field", "`generate` is for text slots ({*}); {n} and {$} are read as numbers.", f"{param}.generate")
        if q.top_k is not None or q.calibration is not None:
            msg = "a generated slot takes no top_k or calibration: its answers come without a probability (several: generate.n)."
            raise Rejected("invalid_field", msg, param)
    combos = combinations(q, tpl, param)
    filled = [tpl.fill(c) for c in combos]
    filled_ids = _tok(tok, filled, False)
    _check_length(filled_ids, max_query_tokens, param)
    return ExtractSpec(
        q,
        param,
        combos,
        filled,
        filled_ids,
        tpl.kind,
        tpl.stop,
        q.max_tokens or ex.DEFAULT_MAX_TOKENS[tpl.kind],
        q.top_k or ex.DEFAULT_TOP_K,
        gen,
    )


def _check_length(filled_ids: list[list[int]], max_query_tokens: int | None, param: str) -> None:
    """the filled templates against the query limit (None: a pack's templates, exempt from the caller's limit)"""
    if max_query_tokens is None:
        return
    for i, ids in enumerate(filled_ids):
        if len(ids) > max_query_tokens:
            raise Rejected(
                "template_too_long",
                f"the filled template is {len(ids):,} tokens; the limit is {max_query_tokens:,}.",
                f"{param}.combinations.{i}",
            )


def _spec(
    tok, q: Query, param: str, case_insensitive: bool, floor_default: bool, lim: Limits, unlimited: bool = False
) -> "QuerySpec | ExtractSpec":
    """unlimited: the query belongs to a pack of the call, whose filled templates are exempt from the caller's query limit (a pack
    is sized by its author; the decider frame lists every option and runs over the default 256 tokens)"""
    tpl = Template(q.template, f"{param}.template")
    max_query_tokens = None if unlimited else lim.max_query_tokens
    if tpl.kind != "score":
        return _xspec(tok, q, tpl, param, max_query_tokens)
    if q.max_tokens is not None or q.top_k is not None or q.calibration is not None or q.generate is not None:
        raise Rejected("invalid_field", "max_tokens, top_k, calibration and generate are for extraction slots ({*} {n} {$}).", param)
    combos = combinations(q, tpl, param)
    check_options(q, case_insensitive, param, lim.max_options)
    filled = [tpl.fill(c) for c in combos]
    filled_ids = _tok(tok, filled, False)
    _check_length(filled_ids, max_query_tokens, param)
    opt_ids = _tok(tok, q.options, False)
    for i, ids in enumerate(opt_ids):
        if not 1 <= len(ids) <= lim.max_option_tokens:
            raise Rejected(
                "option_too_long",
                f"option {q.options[i]!r} is {len(ids)} tokens; options are 1-{lim.max_option_tokens}.",
                f"{param}.options.{i}",
            )
    # the option must tokenize the same after the template as on its own (checked on the distinct template endings)
    tails = list(dict.fromkeys(f[-BOUNDARY_TAIL_CHARS:] for f in filled))[:MAX_BOUNDARY_TAILS]
    variants: list[str] = []
    var_ids: list[list[int]] = []
    starts = []
    per_opt = [case_variants(tok, o) if case_insensitive else [o] for o in q.options]
    cand = [v for vs in per_opt for v in vs]
    cand_ids = dict(zip(cand, _tok(tok, cand, False), strict=True))
    ok = {v: True for v in cand}
    for tail in tails:
        tail_ids = _tok(tok, [tail], False)[0]
        joint = _tok(tok, [tail + v for v in cand], False)
        for v, j in zip(cand, joint, strict=True):
            if j != tail_ids + cand_ids[v]:
                ok[v] = False
    for i, (o, vs) in enumerate(zip(q.options, per_opt, strict=True)):
        if not ok[o]:
            raise Rejected(
                "option_boundary",
                f"option {o!r} merges with the end of the template when tokenized; add or remove a "
                "leading space, or end the template differently.",
                f"{param}.options.{i}",
            )
        starts.append(len(variants))
        for v in vs:
            if ok[v] and v not in variants:
                variants.append(v)
                var_ids.append(cand_ids[v])
    plan = ReadPlan.build(var_ids)
    floor = floor_default if q.empty_floor is None else q.empty_floor
    extra = sum(len(ids) - 1 for ids in opt_ids)
    return QuerySpec(q, param, combos, filled, filled_ids, extra, variants, np.array(starts, np.int64), plan, floor)


def compile_request(
    req: ScoreRequest,
    tok,
    budget: int,
    media_ids: dict | None = None,
    max_tokens: int | None = None,
    x_batched: bool = True,
    pack_groups: frozenset[str] = frozenset(),
) -> Compiled:
    """max_tokens: the engine's largest single input (EngineConfig.max_tokens); a content over half the budget runs alone in its
    own stream (packer: solo) instead of being split and re-sent. pack_groups: the query groups of the call's packs: their templates
    are exempt from the caller's query limit and their query tokens are counted apart (usage.packs)"""
    max_tokens = max(budget, max_tokens or budget)
    sc = req.scoring
    lim = req.limits
    shared: dict[str, tuple[int, Query]] = {}
    for i, q in enumerate(sc.queries):
        if q.id in shared:
            raise Rejected("duplicate_id", f"query id {q.id!r} is used twice.", f"scoring.queries.{i}.id")
        shared[q.id] = (i, q)
    specs: dict[int, QuerySpec | ExtractSpec] = {}  # id(Query) -> spec

    def spec_of(q: Query, param: str) -> "QuerySpec | ExtractSpec":
        s = specs.get(id(q))
        if s is None:
            s = specs[id(q)] = _spec(tok, q, param, sc.case_insensitive, sc.empty_floor, lim, q.group in pack_groups)
        return s

    bos = _tok(tok, [""], True)[0]
    ids_seen = set()
    contents: list[Content] = []
    for i, inp in enumerate(req.inputs):
        if inp.id in ids_seen:
            raise Rejected("duplicate_id", f"input id {inp.id!r} is used twice.", f"inputs.{i}.id")
        ids_seen.add(inp.id)
        contents.append(_content(tok, content_segments(inp, f"inputs.{i}", lim), bos, media_ids, f"inputs.{i}"))

    items: list[tuple[object, int, GroupedTree]] = []
    layout: list[list[tuple[QuerySpec, object, int]]] = []
    floor_index: dict[tuple, int] = {}
    floor_suffix: list[list[int]] = []
    floor_plans: list[ReadPlan] = []
    usage = {"content_tokens": 0, "text_tokens": 0, "query_tokens": 0, "floor_tokens": 0, **dict.fromkeys(MEDIA_USAGE, 0)}
    detail = {
        "scoring": dict.fromkeys(("queries", "combinations", "options", "prompts", "floor_prompts"), 0),
        "extraction": dict.fromkeys(("slots", "text", "number", "currency"), 0),
        "generation": dict.fromkeys(("slots", "sequences", "max_tokens"), 0),
        "packs": {"count": len(pack_groups), "query_tokens": 0},
    }
    extract_jobs: list[ex.ExtractJob] = []
    warnings: list[dict] = []
    for i, inp in enumerate(req.inputs):
        param = f"inputs.{i}"
        ct = contents[i]
        c_ids = ct.ids
        n_content = len(c_ids) - len(bos)
        if n_content > lim.max_content_tokens:
            raise Rejected(
                "content_too_long", f"content is {n_content:,} tokens; the limit for your account is {lim.max_content_tokens:,}.", param
            )
        usage["content_tokens"] += n_content
        usage["text_tokens"] += n_content - ct.mcount["image_tokens"] - ct.mcount["video_tokens"] - ct.mcount["audio_tokens"]
        for k, v in ct.mcount.items():
            usage[k] += v
        chosen: list[tuple[Query, str]] = []
        if inp.query_ids is None:
            chosen = [(q, f"scoring.queries.{k}") for k, q in shared.values()]
        else:
            for k, qid in enumerate(inp.query_ids):
                if qid not in shared:
                    raise Rejected("unknown_query_id", f"no shared query {qid!r}.", f"{param}.query_ids.{k}")
                chosen.append((shared[qid][1], f"scoring.queries.{shared[qid][0]}"))
        own_ids = set()
        for k, q in enumerate(inp.queries or []):
            if q.id in own_ids or (q.id in shared and any(c[0].id == q.id for c in chosen)):
                raise Rejected("duplicate_id", f"query id {q.id!r} is used twice for this input.", f"{param}.queries.{k}.id")
            own_ids.add(q.id)
            chosen.append((q, f"{param}.queries.{k}"))
        if not chosen:
            raise Rejected("empty_queries", "this input has no queries to score.", param)
        entries: list[tuple[QuerySpec | ExtractSpec, object, int]] = []
        main_suffix: list[list[int]] = []
        main_plans: list[ReadPlan] = []
        x_branches: list[tuple[list[int], ex.ExtractPrompt]] = []
        n_prompts = 0
        specs_in: list[tuple[int, QuerySpec | ExtractSpec, bool]] = []
        n_extract = 0
        for qi, (q, qparam) in enumerate(chosen):
            s = spec_of(q, qparam)
            n_prompts += len(s.combos)
            if isinstance(s, ExtractSpec):
                n_extract += len(s.combos)
                if n_extract > lim.max_extractions_per_input:
                    raise Rejected("too_many_extractions", f"over {lim.max_extractions_per_input} extraction slots for one input.", param)
            if n_prompts > lim.max_prompts_per_input:
                raise Rejected("too_many_prompts", f"over {lim.max_prompts_per_input:,} prompts for one input.", param)
            longest = len(c_ids) + max(map(len, s.filled_ids)) + (s.max_tokens if isinstance(s, ExtractSpec) else lim.max_option_tokens)
            if longest > max_tokens:
                raise Rejected(
                    "content_too_long", f"content and prompt are {longest:,} tokens; the engine's maximum is {max_tokens:,}.", param
                )
            # the join content | template: clean after media (a marker token); after text, checked on the text's end
            tail = (ct.tail or "")[-256:]
            clean = ct.tail is None or _tok(tok, [tail + s.filled[0]], False)[0] == _tok(tok, [tail], False)[0] + s.filled_ids[0]
            clean = clean and not (isinstance(s, ExtractSpec) and not all(s.filled_ids))  # an empty template needs a branch
            specs_in.append((qi, s, clean))
        # a template that merges with the content's end: the content is cut at a clean point a few tokens before its end, and
        # every prompt's branch starts with that short tail re-tokenized with its template (one tree; the rest read once)
        prefix, tail_text = c_ids, None
        if not all(c for _, _, c in specs_in) and ct.tail:
            prefix, tail_text = _cut(tok, c_ids, ct)
            if tail_text is None:  # no clean cut: the templates join the content at a token boundary, read once
                warnings.append(
                    {
                        "code": "join_not_merged",
                        "message": "the content's end and a template tokenize differently together than apart; the template was "
                        "read after the content's own tokens.",
                        "param": f"inputs.{i}",
                    }
                )

        def branch(s, k: int, tail_text=tail_text) -> list[int]:
            return s.filled_ids[k] if tail_text is None else _tok(tok, [tail_text + s.filled[k]], False)[0]

        # query groups (Query.group): the ungrouped queries first, then each named group in name order; the prompts are laid out
        # group by group (entries keep the request's order and point at their prompts)
        gname = {qi: (s.q.group or "") for qi, s, _ in specs_in}
        order = sorted(specs_in, key=lambda e: (gname[e[0]] != "", gname[e[0]]))
        groups: list[list] = []  # [name, first prompt index, n prompts, x_branches]
        firsts: dict[int, int] = {}
        for qi, s, _clean in order:
            if not groups or groups[-1][0] != gname[qi]:
                groups.append([gname[qi], len(main_suffix), 0, []])
            g = groups[-1]
            if isinstance(s, ExtractSpec):
                for k in range(len(s.combos)):
                    ids = branch(s, k)
                    if ids and x_batched:
                        g[3].append((ids, ex.ExtractPrompt(("x", i, qi, k), ids, s.kind, s.stop, s.max_tokens, s.gen)))
                continue
            firsts[qi] = len(main_suffix)
            main_suffix += [branch(s, k) for k in range(len(s.combos))]
            main_plans += [s.plan] * len(s.combos)
            g[2] += len(s.combos)
        x_branches = [x for g in groups for x in g[3]]
        if x_branches and not prefix:  # no <bos> and no content: a branch would start the stream, with no prefix state to decode from
            raise Rejected("unsupported_content", "an extraction slot needs some content before its template.", f"inputs.{i}")
        for qi, s, _clean in specs_in:
            if isinstance(s, ExtractSpec):
                usage["query_tokens"] += sum(map(len, s.filled_ids))
                dx = detail["extraction"]
                dx["slots"] += len(s.combos)
                dx[s.kind] += len(s.combos)
                if s.gen is not None:
                    dg = detail["generation"]
                    dg["slots"] += len(s.combos)
                    dg["sequences"] += len(s.combos) * s.gen.n
                    dg["max_tokens"] = max(dg["max_tokens"], s.max_tokens)
                entries.append((s, ("x", i, qi), 0))
                for k in range(len(s.combos)):
                    key = ("x", i, qi, k)
                    ids = branch(s, k)
                    if not (ids and x_batched):  # the reference path (x_batched false), or no tokens to branch on: its own read
                        # of the whole prompt; otherwise a branch of the input's tree (above): read in its scoring pass, decoded after
                        extract_jobs.append(
                            ex.ExtractJob(prefix + ids, list(ct.media), [ex.ExtractPrompt(key, [], s.kind, s.stop, s.max_tokens, s.gen)])
                        )
                continue
            qt = sum(map(len, s.filled_ids)) + s.option_extra * len(s.combos)
            usage["query_tokens"] += qt
            if s.q.group in pack_groups:
                detail["packs"]["query_tokens"] += qt
            ds = detail["scoring"]
            ds["prompts"] += len(s.combos)
            if not ds.get(("seen", id(s))):  # the distinct queries of the call: a shared one counts once
                ds[("seen", id(s))] = 1
                ds["queries"] += 1
                ds["combinations"] += len(s.combos)
                ds["options"] += len(s.q.options)
            entries.append((s, ("in", i), firsts[qi]))
            if s.floor:
                opts = tuple(s.q.options)
                for f, fids in zip(s.filled, s.filled_ids, strict=True):
                    if not bos and not fids:  # a model without <bos> cannot read an option after nothing at all
                        raise Rejected(
                            "invalid_field",
                            "the empty-floor read needs text before the slot on this model: add text to the template or set "
                            "`empty_floor` false.",
                            f"{qparam}.template",
                        )
                    fk = (f, opts, sc.case_insensitive)
                    if fk not in floor_index:
                        floor_index[fk] = len(floor_suffix)
                        floor_suffix.append(fids)
                        floor_plans.append(s.plan)
                        usage["floor_tokens"] += len(fids) + s.option_extra
                        detail["scoring"]["floor_prompts"] += 1
        if main_suffix or x_branches:
            gs = [(g[1], g[2], g[3], g[0] != "") for g in groups]
            items += [
                (("in", i), first, t) for first, t in _split_groups(prefix, main_suffix, main_plans, budget, ct.media, max_tokens, gs)
            ]
        layout.append(entries)
    if floor_suffix:
        items += [(("floor",), first, t) for first, t in _split(bos, floor_suffix, floor_plans, budget)]
    processed = (
        sum(t.n_tokens for _, _, t in items)
        + sum(len(j.prefix) + sum(len(p.ids) for p in j.prompts) for j in extract_jobs)
        + sum(p.max_tokens * ex.BEAM for _, _, t in items for _, p in t.xbranches)
        + sum(p.max_tokens * ex.BEAM for j in extract_jobs for p in j.prompts)
    )
    if processed > lim.max_processed_tokens:
        raise Rejected(
            "request_too_large",
            f"the request needs {processed:,} processed tokens; the limit is {lim.max_processed_tokens:,}. Split it.",
            None,
        )
    detail["scoring"] = {k: v for k, v in detail["scoring"].items() if isinstance(k, str)}
    return Compiled(items, layout, floor_index, usage, processed, warnings=warnings, extract_jobs=extract_jobs, detail=detail)


def _offsets(tok, text: str) -> list[int] | None:
    """the character offset where each token of `text` starts (fast tokenizers); None when the tokenizer cannot say"""
    if not getattr(tok, "is_fast", False):
        return None
    enc = tok(text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
    return [s for s, _ in enc["offset_mapping"]]


def _cut(tok, c_ids: list[int], ct: Content) -> tuple[list[int], str | None]:
    """content ending in text -> (its tokens up to a clean cut near the end, the text after the cut): the shortest tail of at
    least CUT_CHARS characters whose two sides tokenize to exactly the content's own tokens, so the prefix is the content's tokens
    and the tail re-tokenized with each template is the content's end. Candidates are the last CUT_TOKENS token boundaries (a cut
    inside a token never re-tokenizes the same; a tokenizer without offsets tries every CUT_CHARS characters). None found (a run
    with no boundary, e.g. one long word): (all the content's tokens, None), and the templates join at the token boundary"""
    t = ct.tail
    tail_ids = c_ids[len(c_ids) - ct.tail_len :]
    starts = _offsets(tok, t)
    if starts is not None:
        cands = [len(t) - s for s in reversed(starts) if len(t) - s >= CUT_CHARS][:CUT_TOKENS]
    else:
        cands = [CUT_CHARS, 2 * CUT_CHARS, 4 * CUT_CHARS, 8 * CUT_CHARS]
    for n in cands:
        if n >= len(t):
            break
        b = _tok(tok, [t[-n:]], False)[0]
        if b != tail_ids[len(tail_ids) - len(b) :]:  # the tail alone must already be the content's last tokens (cheap check)
            continue
        a = _tok(tok, [t[:-n]], False)[0]
        if a + b == tail_ids:
            return c_ids[: len(c_ids) - len(b)], t[-n:]
    if len(t) <= 8 * CUT_CHARS and len(c_ids) > ct.tail_len:  # a short run: the whole of it, after <bos> or earlier content
        return c_ids[: len(c_ids) - ct.tail_len], t
    return c_ids, None


def _split(
    prefix: list[int],
    suffixes: list[list[int]],
    plans: list[ReadPlan],
    budget: int,
    media: list = (),
    max_tokens: int | None = None,
    xs: list = (),
    aligned: bool = False,
) -> list[tuple[int, GroupedTree]]:
    """prompts = prefix + suffix[k]; contiguous runs that fit the budget, each run one grouped tree (each re-sends the prefix and
    its media). A long prefix (over half the budget) is not re-sent: its runs may grow to max_tokens and run alone (solo).
    xs = [(template ids, ExtractPrompt)]: extraction branches, indexed after the scored prompts (a run of them has no rows)."""
    max_tokens = max(budget, max_tokens or budget)
    limit = max_tokens if len(prefix) > budget // 2 else budget
    n = len(suffixes)
    out = []
    stack = [(0, n + len(xs))]
    while stack:
        a, b = stack.pop()
        t = _tree(prefix, suffixes[a : min(b, n)], plans[a : min(b, n)], list(xs[max(a, n) - n : max(b, n) - n]), aligned=aligned)
        t.media = list(media)
        if t.n_tokens <= limit or (b - a == 1 and t.n_tokens <= max_tokens):
            out.append((a, t))
        elif b - a == 1:
            raise Rejected("request_too_large", f"one prompt is {t.n_tokens:,} tokens, over the engine's maximum {max_tokens:,}.", None)
        else:
            mid = (a + b) // 2
            stack += [(mid, b), (a, mid)]
    return sorted(out, key=lambda x: x[0])


def _split_groups(prefix, suffixes, plans, budget, media, max_tokens, groups) -> list[tuple[int, GroupedTree]]:
    """groups = [(first prompt, n prompts, x branches, aligned)]: one tree with every group when it fits; else each group on its
    own (a group keeps its block-aligned layout), each split further by _split when it alone does not fit"""
    whole = _tree(prefix, suffixes, plans, groups=groups)
    max_tokens = max(budget, max_tokens or budget)
    limit = max_tokens if len(prefix) > budget // 2 else budget
    if whole.n_tokens <= limit or len(groups) == 1:
        if whole.n_tokens <= limit:
            whole.media = list(media)
            return [(0, whole)]
        a, n, xs, al = groups[0]
        return _split(prefix, suffixes, plans, budget, media, max_tokens, xs, al)
    out = []
    for a, n, xs, al in groups:
        out += [(a + f, t) for f, t in _split(prefix, suffixes[a : a + n], plans[a : a + n], budget, media, max_tokens, xs, al)]
    return out


def _tree(
    prefix: list[int], suffixes: list[list[int]], plans: list[ReadPlan], xs: list = (), groups=None, aligned: bool = False
) -> GroupedTree:
    """build_grouped_tree without materialising prefix + suffix for every prompt: the shared part is the prefix plus the
    suffixes' common start (every prompt keeps at least one token of its own); after it, the scored prompts and their lead-ins
    form a token trie (packer.trie_branches). xs = [(template ids, ExtractPrompt)]: extraction templates as extra branches on
    the shared part after the trie, never inside it (their K/V seed a decode cache as prefix + branch; no read rows;
    tree.xbranches)"""
    if groups is None:
        groups = [(0, len(suffixes), list(xs), aligned)]
    if any(len(s) == 0 for s in suffixes):  # a prompt is the prefix alone: its last token must be its own
        if any(g[2] for g in groups):
            raise ValueError("an extraction branch needs template tokens")
        return build_grouped_tree([prefix + s for s in suffixes], plans)
    # the shared part is the content alone: the templates' common start is a trie node, so a query's layout does not depend on
    # which other queries the input has
    branches: list[list[int]] = []
    parents: list[int] = []
    prompt_rows: list[list[tuple[int, int]]] = []
    xbranches = []
    al = []
    for a, n, gxs, g_aligned in groups:
        if g_aligned and (n or gxs):
            al.append(len(branches))
        off = len(branches)
        br, pa, rows = trie_branches(suffixes[a : a + n], plans[a : a + n])
        branches += br
        parents += [p + off if p >= 0 else -1 for p in pa]
        prompt_rows += [[(b + off, o) for b, o in r] for r in rows]
        for x, pr in gxs:
            xbranches.append((len(branches), pr))
            branches.append(list(x))
            parents.append(-1)
    return GroupedTree(list(prefix), branches, prompt_rows, list(plans), xbranches=xbranches, parents=parents, aligned=al)


def assemble(c: Compiled, req: ScoreRequest, scores: dict[object, list[np.ndarray | None]], extracted: dict | None = None) -> dict:
    """grouped scores (per prompt, per variant log-probs) and extraction answers (per prompt key, extract.Found) -> the public
    results and warnings"""
    floor_scores = scores.get(("floor",), [])
    results = []
    warnings = list(c.warnings)
    for i, entries in enumerate(c.layout):
        qres = []
        for s, key, first in entries:
            if isinstance(s, ExtractSpec):
                combos_out = []
                for k, combo in enumerate(s.combos):
                    found = (extracted or {}).get(key + (k,)) or ex.Found()
                    combos_out.append({"values": combo, **ex.result(s.kind, found, s.top_k, s.q.calibration, s.gen)})
                qres.append({"id": s.q.id, "kind": s.kind, **({"generate": True} if s.gen else {}), "combinations": combos_out})
                continue
            rows = scores[key]
            combos_out = []
            residuals = []
            for k, combo in enumerate(s.combos):
                lp, p, res = _read(rows[first + k], s)
                fp = fres = None
                if s.floor:
                    fk = (s.filled[k], tuple(s.q.options), req.scoring.case_insensitive)
                    _, fp, fres = _read(floor_scores[c.floor_index[fk]], s)
                residuals.append(res)
                combos_out.append(
                    {
                        "values": combo,
                        "options": [
                            {
                                "text": o,
                                "logprob": round(float(lp[j]), 6),
                                "p": round(float(p[j]), 6),
                                "floor_p": None if fp is None else round(float(fp[j]), 6),
                            }
                            for j, o in enumerate(s.q.options)
                        ],
                        "residual": round(res, 6),
                        "floor_residual": None if fres is None else round(fres, 6),
                    }
                )
            # only for one-token options: a multi-token option's exact-string probability is small even when it fits well
            # (TODO: judge those on the mass of their first tokens)
            if s.plan.n_rows == 1 and residuals and sum(residuals) / len(residuals) > LOW_OPTION_MASS:
                warnings.append(
                    {
                        "code": "low_option_mass",
                        "input_id": req.inputs[i].id,
                        "query_id": s.q.id,
                        "message": f"the options take {1 - sum(residuals) / len(residuals):.1%} of the next-token mass on "
                        "average: the template or the options probably do not fit.",
                    }
                )
            qres.append({"id": s.q.id, "combinations": combos_out})
        results.append({"input_id": req.inputs[i].id, "queries": qres})
    return {"results": results, "warnings": warnings}


def _read(variant_lp: np.ndarray, s: QuerySpec) -> tuple[np.ndarray, np.ndarray, float]:
    """per-variant log-probs -> (per option log P summed over its variants, p normalised over the options, residual)"""
    lp = np.logaddexp.reduceat(variant_lp, s.starts)
    top = lp.max()
    w = np.exp(lp - top)
    p = w / w.sum()
    mass = math.exp(top) * float(w.sum())
    return lp, p, max(0.0, 1.0 - mass)
