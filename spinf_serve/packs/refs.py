"""A request's packs: the `scoring.packs` entries of a /v1/score body, validated against the catalog (the error codes of the
public API, docs/prompt-packs.md), and what the server does with them: wrap each input in the pack's context, add the pack's
queries to the call, and turn each input's result into its `packs[]` entry, hiding the pack's raw queries unless asked for."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from spinf_serve.packs import Pack
from spinf_serve.packs.catalog import Catalog
from spinf_serve.protocol import Rejected

FIELDS = ("id", "version", "context", "decision", "include_queries")
DECISION_FIELDS = ("mode", "threshold", "thresholds", "critical")


@dataclass
class PackRef:
    """One `scoring.packs` entry, validated."""

    pack: Pack
    context: str
    mode: str
    threshold: float | None = None
    thresholds: dict[str, float | None] = field(default_factory=dict)
    include_queries: bool = False
    critical: list[str] | None = None  # None = the pack's decision.critical

    def wrap(self, parts: list[dict[str, Any]], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """an input's content parts in the pack's context: the context's text before and after them"""
        try:
            before, after = self.pack.context_text(self.context, params)
        except ValueError as e:
            raise Rejected("invalid_field", str(e), "inputs.parameters") from None
        return [{"type": "text", "text": before}] * bool(before) + list(parts) + [{"type": "text", "text": after}] * bool(after)

    def check_media(self, kinds: Iterable[str], param: str = "inputs") -> None:
        """the content kinds of an input (text, image, audio, video) against the pack's `media`"""
        bad = sorted(set(kinds) - set(self.pack.media))
        if bad:
            raise Rejected(
                "pack_media_not_supported", f"{self.pack.id} scores {', '.join(self.pack.media)} content; {bad[0]} is not supported.", param
            )

    def evaluate(self, result: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """one input's result -> its `packs[]` entry"""
        return self.pack.evaluate(result, self.mode, self.threshold, self.thresholds, self.critical, self.context)


def _number(value: Any, param: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise Rejected("invalid_field", f"{param} must be a number.", param)
    if not lo <= value <= hi:
        raise Rejected("invalid_field", f"{param} must be between {lo:g} and {hi:g}.", param)
    return float(value)


def parse_refs(value: Any, catalog: Catalog, param: str = "scoring.packs") -> list[PackRef]:
    """scoring.packs: [{id, version?, context?, decision?: {mode?, threshold?, thresholds?, critical?}, include_queries?}]"""
    if not isinstance(value, list):
        raise Rejected("invalid_field", f"{param} must be a list.", param)
    refs: list[PackRef] = []
    for i, raw in enumerate(value):
        where = f"{param}.{i}"
        if not isinstance(raw, dict):
            raise Rejected("invalid_field", f"{where} must be an object.", where)
        unknown = sorted(set(raw) - set(FIELDS))
        if unknown:
            raise Rejected("invalid_field", f"{where}: unknown field {unknown[0]!r}.", f"{where}.{unknown[0]}")
        pack_id = raw.get("id")
        if not isinstance(pack_id, str) or not pack_id:
            raise Rejected("invalid_field", f"{where}.id must be a non-empty string.", f"{where}.id")
        versions = catalog.versions(pack_id)
        if not versions:
            if pack_id in catalog.listed:
                raise Rejected("pack_not_available", f"{where}.id: the {pack_id!r} pack is not available yet.", f"{where}.id")
            raise Rejected("unknown_pack", f"{where}.id: no pack {pack_id!r}. See GET /v1/packs.", f"{where}.id")
        if any(r.pack.id == pack_id for r in refs):
            raise Rejected("duplicate_id", f"{where}.id: {pack_id!r} is listed twice.", f"{where}.id")
        version = raw.get("version")
        if version is not None and version not in versions:
            raise Rejected(
                "unknown_pack_version",
                f"{where}.version: {pack_id} has no version {version!r}; versions: {', '.join(versions)}.",
                f"{where}.version",
            )
        pack = catalog.get(pack_id, version)
        if pack.status == "coming_soon":
            raise Rejected("pack_not_available", f"{where}: {pack_id} {pack.version} is not available yet.", where)
        context = raw.get("context", pack.contexts[0])
        if context not in pack.contexts:
            raise Rejected("invalid_field", f"{where}.context must be one of {', '.join(pack.contexts)}.", f"{where}.context")
        include = raw.get("include_queries", False)
        if not isinstance(include, bool):
            raise Rejected("invalid_field", f"{where}.include_queries must be true or false.", f"{where}.include_queries")
        refs.append(PackRef(pack, context, pack.data["decision"]["mode"], include_queries=include))
        _parse_decision(refs[-1], raw.get("decision"), f"{where}.decision")
    return refs


def _parse_decision(ref: PackRef, value: Any, param: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise Rejected("invalid_field", f"{param} must be an object.", param)
    unknown = sorted(set(value) - set(DECISION_FIELDS))
    if unknown:
        raise Rejected("invalid_field", f"{param}: unknown field {unknown[0]!r}.", f"{param}.{unknown[0]}")
    modes = ref.pack.modes
    mode = value.get("mode", ref.mode)
    if mode not in modes:
        raise Rejected("invalid_field", f"{param}.mode must be one of {', '.join(modes)}.", f"{param}.mode")
    ref.mode = mode
    if value.get("threshold") is not None:
        if mode == "per_category":
            raise Rejected(
                "invalid_field",
                f"{param}.threshold is the one threshold of max or micro_layer; per_category uses thresholds.",
                f"{param}.threshold",
            )
        # max compares a 0-1 category score; the micro-layer's value is unbounded
        ref.threshold = _number(
            value["threshold"], f"{param}.threshold", 0.0 if mode == "max" else -math.inf, 1.0 if mode == "max" else math.inf
        )
    thresholds = value.get("thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise Rejected("invalid_field", f"{param}.thresholds must be an object.", f"{param}.thresholds")
        cats = ref.pack.data["categories"]
        for c, t in thresholds.items():
            where = f"{param}.thresholds.{c}"
            if c not in cats:
                raise Rejected("unknown_category", f"{where}: {ref.pack.id} has no category {c!r}.", where)
            ref.thresholds[c] = None if t is None else _number(t, where, 0.0, 1.0)
    critical = value.get("critical")
    if critical is not None:
        where = f"{param}.critical"
        if not isinstance(critical, list) or not all(isinstance(c, str) for c in critical):
            raise Rejected("invalid_field", f"{where} must be a list of category ids ([] = none).", where)
        for i, c in enumerate(critical):
            if c not in ref.pack.data["categories"]:
                raise Rejected("unknown_category", f"{where}.{i}: {ref.pack.id} has no category {c!r}.", f"{where}.{i}")
        ref.critical = list(dict.fromkeys(critical))


def check_call(refs: list[PackRef], own_query_ids: Iterable[str], case_insensitive: bool | None, param: str = "scoring") -> None:
    """what a call with packs must respect: no own query id of a pack's, one case rule (the packs', or unset)"""
    own = set(own_query_ids)
    for r in refs:
        if clash := sorted(own & r.pack.query_ids):
            raise Rejected("duplicate_id", f"query id {clash[0]!r} belongs to the {r.pack.id} pack.", f"{param}.queries")
        if case_insensitive is not None and case_insensitive != r.pack.case_insensitive:
            raise Rejected(
                "pack_conflict",
                f"{param}.case_insensitive differs from the {r.pack.id} pack's ({str(r.pack.case_insensitive).lower()}); leave it out.",
                f"{param}.case_insensitive",
            )
    if len({r.pack.case_insensitive for r in refs}) > 1:
        raise Rejected("pack_conflict", "the packs of a call must share one case rule.", f"{param}.packs")


def queries(refs: list[PackRef]) -> list[dict[str, Any]]:
    """the packs' queries, grouped per pack, to add after the caller's own"""
    return [q for r in refs for q in r.pack.queries()]


def apply(
    refs: list[PackRef], results: list[dict[str, Any]], warnings: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """each input's `packs` added to its result; the packs' raw query results (and their warnings) left out unless the entry asked
    for them (`include_queries`)"""
    hidden = {qid for r in refs if not r.include_queries for qid in r.pack.query_ids}
    out = []
    for res in results:
        packs = [r.evaluate(res) for r in refs]
        out.append({**res, "queries": [q for q in res["queries"] if q["id"] not in hidden], "packs": packs})
    return out, [w for w in warnings if w.get("query_id") not in hidden]
