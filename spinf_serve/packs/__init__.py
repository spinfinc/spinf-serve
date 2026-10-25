"""Prompt packs (plan §5): one self-contained JSON file per pack version holding the scoring request as the API takes it, how the
content is presented to the model, how the answers turn into category scores and how the scores turn into a decision. The format
is spinf-prompts docs/pack-format.md; the official files are the spinf-prompts repository, loaded by the server with any other
source (sources.py, catalog.py). This package is pure Python: the hosted API, spinf-client and the pack workbench import it.

    from spinf_serve import packs
    pack = packs.load("packs/moderation/0.1.3.json")
    body = pack.request(["some text"], context="forum")       # the /v1/score body: the text in the pack's context, its queries grouped
    out = pack.evaluate(result)                                 # one input's result -> {"id", "version", "categories", "decision"}

A pack is data, nothing executes: load() validates the file (validate()) and refuses one with problems (PackError)."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from spinf_serve import protocol
from spinf_serve.packs import pipeline
from spinf_serve.packs.pipeline import MODES, Answers, Layer

X = "{X}"
MEDIA = ("text", "image", "audio", "video")
ANSWERS = ("p", "logprob")
STATUSES = ("available", "deprecated", "coming_soon")
MAX_PACK_BYTES = 16 << 20  # a pack file is a few hundred KB; a larger one is not a pack
ID = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")  # <namespace>/<name>
VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]*$")
PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
OFFICIAL_NAMESPACE = "spinf"


class PackError(ValueError):
    """a pack file the loader refuses; .problems lists what is wrong, one line each"""

    def __init__(self, where: str, problems: list[str]):
        super().__init__(f"{where}: " + "; ".join(problems) if len(problems) == 1 else f"{where}:\n  " + "\n  ".join(problems))
        self.where, self.problems = where, problems


def _number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def validate(data: Any, draft: bool = False) -> list[str]:
    """every problem of a pack file, as the problems of a pack author (empty = a valid pack). draft: a pack before its calibration
    (no mu, mean, sd, platt, thresholds, no micro-layer yet): the request side and the structure are checked, the fitted constants
    are not"""
    p: list[str] = []
    if not isinstance(data, dict):
        return ["a pack file is a JSON object"]
    for k in ("pack", "version", "model", "contexts", "scoring", "reads", "categories", "decision"):
        if k not in data:
            p.append(f"missing field {k!r}")
    if p:
        return p
    if not isinstance(data["pack"], str) or not ID.match(data["pack"]):
        p.append("pack: an id of the form <namespace>/<name> (lower case, digits, . _ -)")
    if not isinstance(data["version"], str) or not VERSION.match(data["version"]):
        p.append("version: a version string (digits, letters, . + -)")
    if not isinstance(data["model"], str) or not data["model"]:
        p.append("model: the id of the model the pack is calibrated on")
    media = data.get("media", ["text"])
    if not isinstance(media, list) or not media or not set(media) <= set(MEDIA) or "text" not in media:
        p.append(f"media: text plus any of {', '.join(MEDIA[1:])}")
    if data.get("answers", "p") not in ANSWERS:
        p.append(f"answers: one of {', '.join(ANSWERS)}")
    params = data.get("parameters") or {}
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and PARAM.fullmatch("{" + k + "}") and isinstance(v, dict) for k, v in params.items()
    ):
        p.append("parameters: {name: {label, description?, required?, default?}}")
        params = {}
    for name, spec in params.items():
        if name == "X":
            p.append("parameters: X is the content placeholder")
        if not isinstance(spec.get("label", ""), str) or not isinstance(spec.get("required", True), bool):
            p.append(f"parameters.{name}: label is a string, required a boolean")
    contexts = data["contexts"]
    if not isinstance(contexts, dict) or not contexts or not all(isinstance(k, str) and isinstance(v, str) for k, v in contexts.items()):
        p.append("contexts: {name: template} with at least one context")
    else:
        for name, tpl in contexts.items():
            if tpl.count(X) != 1:
                p.append(f"contexts.{name}: exactly one {X}")
            if unknown := sorted({m for m in PARAM.findall(tpl) if m != "X" and m not in params}):
                p.append(f"contexts.{name}: unknown placeholders {unknown} (a parameter, or {X})")
    # the scoring object: the API's, validated as the engine validates a request; scored queries only (reads need options)
    sc = data["scoring"]
    queries: dict[str, dict] = {}
    if not isinstance(sc, dict) or not isinstance(sc.get("queries"), list) or not sc["queries"]:
        p.append("scoring: {queries: [...], empty_floor?, case_insensitive?}")
    else:
        for k in ("empty_floor", "case_insensitive"):
            if k in sc and not isinstance(sc[k], bool):
                p.append(f"scoring.{k}: true or false")
        if unknown := sorted(set(sc) - {"queries", "empty_floor", "case_insensitive"}):
            p.append(f"scoring: unknown fields {unknown}")
        for i, q in enumerate(sc["queries"]):
            where = f"scoring.queries[{i}]"
            if not isinstance(q, dict):
                p.append(f"{where}: an object")
                continue
            try:
                m = protocol.Query.model_validate(q)
            except ValidationError as e:
                p.append(f"{where}: " + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()))
                continue
            if not m.options or len(m.options) < 2 or "{?}" not in m.template:
                p.append(f"{where} ({m.id}): a scored query, a template with {{?}} and at least two options")
            if m.group is not None:
                p.append(f"{where} ({m.id}): no group (the loader sets the pack's)")
            if m.id in queries:
                p.append(f"{where}: duplicate id {m.id!r}")
            queries[m.id] = q
    # reads
    reads = data["reads"]
    read_ids: set[str] = set()
    categories = data["categories"] if isinstance(data["categories"], dict) else {}
    if not isinstance(reads, list) or not reads:
        p.append("reads: a non-empty list")
        reads = []
    for i, r in enumerate(reads):
        where = f"reads[{i}]"
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"]:
            p.append(f"{where}: an object with an id")
            continue
        where += f" ({r['id']})"
        if r["id"] in read_ids:
            p.append(f"{where}: duplicate id")
        read_ids.add(r["id"])
        q = queries.get(r.get("query"))
        if q is None:
            if queries:
                p.append(f"{where}: query {r.get('query')!r} is not in scoring.queries")
            continue
        combos = q.get("combinations")
        if isinstance(combos, list):
            if r.get("F") not in {c.get("F") for c in combos if isinstance(c, dict)}:
                p.append(f"{where}: F {r.get('F')!r} is not a combination of query {r['query']}")
        elif combos == "all":
            if r.get("F") not in (q.get("values") or {}).get("F", []):
                p.append(f"{where}: F {r.get('F')!r} is not a value of query {r['query']}")
        elif "F" in r:
            p.append(f"{where}: query {r['query']} has no combinations, the read no F")
        n = len(q["options"])
        kind = r.get("kind")
        if kind == "pair":
            if n != 2 or r.get("sign", 1) not in (1, -1):
                p.append(f"{where}: a pair read needs two options and sign 1 or -1")
        elif kind == "option":
            if not isinstance(r.get("option"), int) or not 0 <= r["option"] < n or not (draft or _number(r.get("mu"))):
                p.append(f"{where}: an option read needs an option index below {n} and a scalar mu")
        elif kind in ("any", "set"):
            mu = r.get("mu")
            if not draft and (not isinstance(mu, list) or len(mu) != n or not all(_number(x) for x in mu)):
                p.append(f"{where}: a {kind} read needs a mu per option ({n})")
            if kind == "set":
                u = r.get("unsafe")
                if not isinstance(u, list) or not u or not all(isinstance(x, int) and 0 <= x < n for x in u) or len(set(u)) == n:
                    p.append(f"{where}: a set read needs unsafe option indices below {n}, not all of them")
        else:
            p.append(f"{where}: kind is pair, option, any or set")
        if not draft and (not _number(r.get("mean")) or not _number(r.get("sd")) or r.get("sd", 0) <= 0):
            p.append(f"{where}: mean and sd (> 0) are numbers")
        if r.get("category") not in categories:
            p.append(f"{where}: category {r.get('category')!r} is not in categories")
    # categories
    if not isinstance(data["categories"], dict) or not categories:
        p.append("categories: {id: {label, platt, threshold, ...}} with at least one category")
    fed = {r["category"] for r in reads if isinstance(r, dict)}
    for c, spec in categories.items():
        where = f"categories.{c}"
        if not isinstance(spec, dict):
            p.append(f"{where}: an object")
            continue
        platt = spec.get("platt")
        if not draft and (not isinstance(platt, list) or len(platt) != 2 or not all(_number(x) for x in platt)):
            p.append(f"{where}: platt is [a, b]")
        t = spec.get("threshold")
        if t is not None and not (_number(t) and 0 <= t <= 1):
            p.append(f"{where}: threshold is null or between 0 and 1")
        if c not in fed:
            p.append(f"{where}: no read feeds it")
        q = spec.get("normal_quantiles")
        if q is not None:
            lv, vs = (q.get("levels"), q.get("values")) if isinstance(q, dict) else (None, None)
            ok = isinstance(lv, list) and isinstance(vs, list) and len(lv) == len(vs) >= 2 and all(_number(x) for x in lv + vs)
            if not ok or vs != sorted(vs) or lv != sorted(lv):
                p.append(f"{where}: normal_quantiles is {{levels, values}}, two increasing lists of the same length")
        if spec.get("aggregate") not in (None, "views"):
            p.append(f'{where}: aggregate is "views" or absent')
        req = spec.get("requires")
        if req is not None and (not isinstance(req, list) or not all(x in categories and x != c for x in req)):
            p.append(f"{where}: requires lists other categories")
        if not isinstance(spec.get("decides", True), bool):
            p.append(f"{where}: decides is true or false")
    # decision
    dec = data["decision"]
    if not isinstance(dec, dict) or dec.get("mode") not in MODES:
        p.append(f"decision: {{mode: {' | '.join(MODES)}, fallback?, max?, per_category?, micro_layer?, critical?}}")
        return p
    if dec.get("fallback", "per_category") not in ("max", "per_category"):
        p.append("decision.fallback: max or per_category")
    needs_max = dec.get("mode") == "max" or dec.get("fallback") == "max" or "max" in dec
    if needs_max and not draft and not _number((dec.get("max") or {}).get("threshold")):
        p.append("decision.max: {threshold}")
    crit = dec.get("critical") or []
    if not isinstance(crit, list) or not all(x in categories for x in crit):
        p.append("decision.critical: a list of category ids")
    ml = dec.get("micro_layer")
    if dec["mode"] == "micro_layer" and not ml and not draft:
        p.append("decision.micro_layer: needed by mode micro_layer")
    if ml:
        inputs = ml.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            p.append("decision.micro_layer.inputs: a list of c:<category> and r:<read id>")
        else:
            for x in inputs:
                ok = isinstance(x, str) and ((x.startswith("c:") and x[2:] in categories) or (x.startswith("r:") and x[2:] in read_ids))
                if not ok:
                    p.append(f"decision.micro_layer.inputs: {x!r} is not c:<category> or r:<read id>")
            try:
                pipeline.micro_layer(data)
            except Exception as e:  # noqa: BLE001 (the decoding error is the message)
                p.append(f"decision.micro_layer.weights: {e}")
        if not isinstance(ml.get("lock"), str) or not ml["lock"].startswith("sha256:"):
            p.append("decision.micro_layer.lock: sha256:<hex>")
    return p


@dataclass(frozen=True)
class Pack:
    """A loaded pack version: the file as it is plus what every request reuses. `source` says where it came from (the catalog
    sets it: "spinfinc/spinf-prompts@0.1.0", "local:./my-packs"); `status`, `name`, `description`, `tags` come from the repo
    manifest when there is one."""

    data: dict[str, Any]
    query_ids: frozenset[str]
    prompts: int  # the filled templates one input costs: queries x combinations
    lock_ok: bool
    layer: Layer | None  # the decoded micro-layer, None when the pack has none
    source: str = ""
    status: str = "available"
    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    listing: dict[str, Any] = field(default_factory=dict)  # the repo manifest's entry (readme, samples, fixture, note)
    draft: bool = False  # loaded with draft=True: the request side works, evaluate() does not (no fitted constants yet)

    @property
    def id(self) -> str:
        return self.data["pack"]

    @property
    def namespace(self) -> str:
        return self.id.partition("/")[0]

    @property
    def version(self) -> str:
        return self.data["version"]

    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def media(self) -> tuple[str, ...]:
        return tuple(self.data.get("media") or ("text",))

    @property
    def contexts(self) -> list[str]:
        return list(self.data["contexts"])

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return self.data.get("parameters") or {}

    @property
    def group(self) -> str:
        """the query group the pack's queries run in: their own trie, so the pack scores the same whatever else is in the call"""
        return f"{self.id}@{self.version}"

    @property
    def modes(self) -> list[str]:
        """the decision modes a call can pick"""
        dec = self.data["decision"]
        return [m for m in MODES if m == "per_category" or dec.get(m)]

    @property
    def case_insensitive(self) -> bool:
        return bool(self.data["scoring"].get("case_insensitive", False))

    def parameter_values(self, params: dict[str, Any] | None = None) -> dict[str, str]:
        """the values substituted into a context: the caller's, a parameter's default, "" for an optional one; a missing required
        parameter or an unknown name is a ValueError"""
        given = dict(params or {})
        out: dict[str, str] = {}
        for name, spec in self.parameters.items():
            v = given.pop(name, spec.get("default"))
            if v is None:
                if spec.get("required", True):
                    raise ValueError(f"pack parameter {name!r} is required")
                v = ""
            out[name] = str(v)
        if given:
            raise ValueError(f"{self.id} has no parameter {sorted(given)[0]!r}")
        return out

    def context_text(self, context: str | None = None, params: dict[str, Any] | None = None) -> tuple[str, str]:
        """the text before and after the content in the named context (default the first), the parameters filled in"""
        name = context or self.contexts[0]
        if name not in self.data["contexts"]:
            raise ValueError(f"{self.id} has no context {name!r}; contexts: {', '.join(self.contexts)}")
        before, _, after = self.data["contexts"][name].partition(X)
        for k, v in self.parameter_values(params).items():
            before, after = before.replace("{" + k + "}", v), after.replace("{" + k + "}", v)
        return before, after

    def fill(self, text: str, context: str | None = None, params: dict[str, Any] | None = None) -> str:
        before, after = self.context_text(context, params)
        return before + text + after

    def wrap(self, text: str, context: str | None = None, params: dict[str, Any] | None = None) -> list[dict[str, str]]:
        """a text in the context, as content parts: the context's text before and after the content"""
        before, after = self.context_text(context, params)
        return [{"type": "text", "text": s} for s in (before, text, after) if s]

    def queries(self) -> list[dict[str, Any]]:
        """the pack's queries as the engine takes them: the pack's floor setting on each (the request's own default does not change
        the pack's prompts) and the pack as their query group"""
        floor = self.data["scoring"].get("empty_floor", True)
        return [{"empty_floor": floor, **q, "group": self.group} for q in self.data["scoring"]["queries"]]

    def request(
        self,
        texts: list[str],
        context: str | None = None,
        ids: list[str] | None = None,
        params: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """the /v1/score body for a batch of texts: each in the context, the pack's queries grouped; `params` one dict for the
        batch or one per text; `model` when the server's name of the pack's model differs"""
        ids = ids or [str(i) for i in range(len(texts))]
        per = params if isinstance(params, list) else [params] * len(texts)
        inputs = [
            {"id": i, "messages": [{"role": "user", "content": self.wrap(t, context, p)}]} for i, t, p in zip(ids, texts, per, strict=True)
        ]
        return {
            "model": model or self.model,
            "inputs": inputs,
            "scoring": {"queries": self.queries(), "case_insensitive": self.case_insensitive},
        }

    def answers(self, result: dict[str, Any] | list[dict[str, Any]]) -> Answers:
        """one input's result of /v1/score (or its `queries` list) -> the pack's answers"""
        return pipeline.answers_from_result(result, set(self.query_ids), self.data.get("answers", "p"))

    def evaluate(
        self,
        result: dict[str, Any] | list[dict[str, Any]] | Answers,
        mode: str | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float | None] | None = None,
        critical: list[str] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """one input's result (or its answers) -> the `packs[]` entry the API returns: id, version, context, categories (score,
        percentile, flagged; rounded to 6 decimals) and the decision"""
        if self.draft:
            raise ValueError(f"{self.id} {self.version} is a draft: no fitted constants to evaluate with")
        answers = result if isinstance(result, dict) and all(isinstance(k, tuple) for k in result) else self.answers(result)
        out = pipeline.evaluate(self.data, answers, mode, threshold, thresholds, critical, self.lock_ok, self.layer)
        cats = {
            c: {"score": _r(v["score"]), "percentile": _r(v["percentile"]), "flagged": v["flagged"]} for c, v in out["categories"].items()
        }
        decision = {k: _r(v) if isinstance(v, float) else v for k, v in out["decision"].items()}
        return {"id": self.id, "version": self.version, "context": context or self.contexts[0], "categories": cats, "decision": decision}

    def public(self, versions: list[str] | None = None) -> dict[str, Any]:
        """the GET /v1/packs entry of this version"""
        d = self.data
        dec = d["decision"]
        return {
            "id": self.id,
            "name": self.name or self.id,
            "status": self.status,
            "version": self.version,
            "versions": versions or [self.version],
            "description": self.description,
            "tags": list(self.tags),
            "model": self.model,
            "media": list(self.media),
            "source": self.source,
            "parameters": [
                {"id": k, **{f: v[f] for f in ("label", "description", "required", "default") if f in v}}
                for k, v in self.parameters.items()
            ],
            "contexts": [{"id": k, "template": v} for k, v in d["contexts"].items()],
            "categories": [{"id": c, "label": s.get("label", c), "threshold": s.get("threshold")} for c, s in d["categories"].items()],
            "decision": {
                "mode": dec["mode"],
                "modes": self.modes,
                "fallback": dec.get("fallback", "per_category"),
                "max_threshold": (dec.get("max") or {}).get("threshold"),
                "critical": list(dec.get("critical") or []),
            },
            "queries": len(self.query_ids),
            "prompts": self.prompts,
        }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def from_data(data: dict[str, Any], where: str = "pack", draft: bool = False, **meta: Any) -> Pack:
    """a validated Pack from the file's data; `meta`: source, status, name, description, tags, listing. draft: a pack before its
    calibration (a workbench scores its questions with the request side; evaluate() refuses it)"""
    problems = validate(data, draft)
    if problems:
        raise PackError(where, problems)
    queries = data["scoring"]["queries"]
    prompts = sum(len(q["combinations"]) if isinstance(q.get("combinations"), list) else 1 for q in queries)
    ml = None if draft else data["decision"].get("micro_layer")
    return Pack(
        data,
        frozenset(q["id"] for q in queries),
        prompts,
        lock_ok=bool(ml) and ml.get("lock") == pipeline.lock_of(data),
        layer=pipeline.micro_layer(data) if ml else None,
        draft=draft,
        **meta,
    )


def load(path: str | os.PathLike, draft: bool = False, **meta: Any) -> Pack:
    """a pack file -> Pack, validated (PackError lists what is wrong); draft: see from_data"""
    path = os.fspath(path)
    if os.path.getsize(path) > MAX_PACK_BYTES:
        raise PackError(path, [f"larger than {MAX_PACK_BYTES >> 20} MB"])
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except ValueError as e:
            raise PackError(path, [f"not JSON: {e}"]) from None
    return from_data(data, path, draft, **meta)


__all__ = ["MEDIA", "MODES", "OFFICIAL_NAMESPACE", "Pack", "PackError", "from_data", "load", "pipeline", "validate"]
