"""The public API around the engine (plan §3.3, docs/scoring-api.md): what the hosted service's front did and the server now does
itself. A request body -> the engine's protocol request (`messages` normalised to `inputs`, the packs expanded, the features and
limits applied); the engine's result -> the public response (envelope, `packs`, the `usage` blocks of §3.4, `timing`); the error
envelope. Pure Python: tested without a GPU.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from spinf_serve import protocol
from spinf_serve.packs import refs as pack_refs
from spinf_serve.packs.catalog import Catalog
from spinf_serve.protocol import Limits, Rejected, Template
from spinf_serve.server.config import Features, ServerConfig

TYPES = {400: "invalid_request", 401: "authentication_error", 404: "not_found", 413: "invalid_request", 429: "rate_limited"}
REJECTED_STATUS = {"request_too_large": 413, "body_too_large": 413}
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class ApiError(Exception):
    """an error answered in the public envelope {"error": {type, code, message, param}, "request_id"}"""

    def __init__(self, status: int, code: str, message: str, param: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.headers = status, code, message, param, headers or {}

    @classmethod
    def of(cls, e: Rejected) -> "ApiError":
        return cls(REJECTED_STATUS.get(e.code, 400), e.code, e.message, e.param)

    def body(self, request_id: str, rejected: bool = False) -> dict[str, Any]:
        """the envelope; `rejected` adds the engine's compact form for a trusted caller (the hosted front and the RunPod handler read
        `rejected`)"""
        out: dict[str, Any] = {
            "error": {
                "type": TYPES.get(self.status, "api_error" if self.status >= 500 else "invalid_request"),
                "code": self.code,
                "message": self.message,
                "param": self.param,
            },
            "request_id": request_id,
        }
        if self.status == 503:
            out["error"]["type"] = "unavailable"
        if rejected and self.status in (400, 413):
            out["rejected"] = {"code": self.code, "message": self.message, "param": self.param}
        return out


def ulid(prefix: str = "scr_") -> str:
    """a ULID (48-bit millisecond time + 80 random bits, Crockford base32): request ids sort by time"""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_B32[n & 31])
        n >>= 5
    return prefix + "".join(reversed(out))


def _loc(e: ValidationError) -> tuple[str, str]:
    err = e.errors(include_url=False)[0]
    param = ".".join(str(x) for x in err["loc"]) or None
    return param, err["msg"]


@dataclass
class Prepared:
    """a request ready for the engine"""

    req: protocol.ScoreRequest
    requested_model: str
    endpoint: str | None = None
    refs: list[pack_refs.PackRef] = field(default_factory=list)
    chat: bool = False

    @property
    def pack_groups(self) -> frozenset[str]:
        return frozenset(r.pack.group for r in self.refs)


def prepare(
    body: Any,
    cfg: ServerConfig,
    served_model: str,
    served_names: set[str],
    catalog: Catalog | None,
    trusted: bool = False,
    chat: bool = False,
) -> Prepared:
    """a decoded request body -> Prepared, or ApiError. served_names: every lower-cased name the model answers to."""
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_field", "The request body must be a JSON object.", "body")
    body = dict(body)
    has_messages, has_inputs = body.get("messages") is not None, body.get("inputs") is not None
    if has_messages == has_inputs:
        raise ApiError(400, "invalid_field", "Send exactly one of messages (one content) or inputs (many).", "messages")
    if chat and body.get("scoring") is None:
        raise ApiError(
            400,
            "scoring_required",
            "This server scores and extracts, it does not chat: send a `scoring` block with your questions (docs/scoring-api.md).",
            "scoring",
        )
    model = body.get("model", served_model)
    if not isinstance(model, str) or not model:
        raise ApiError(400, "invalid_field", "model must be a non-empty string.", "model")
    if model.lower() not in served_names:
        raise ApiError(400, "model_not_served", f"This server serves {served_model}, not {model}. See GET /v1/models.", "model")
    endpoint = body.get("endpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint or len(endpoint) > 128):
        raise ApiError(400, "invalid_field", "endpoint must be a string of up to 128 characters.", "endpoint")

    # per-call limits and features: a trusted caller's, capped by the server's; anyone else's are ignored
    limits, features = cfg.limits, cfg.features
    if trusted and body.get("limits") is not None:
        try:
            limits = Limits.model_validate(body["limits"]).capped(cfg.limits)
        except ValidationError as e:
            param, msg = _loc(e)
            raise ApiError(400, "invalid_field", msg, f"limits.{param}" if param else "limits") from None
    if trusted and body.get("features") is not None:
        features = _features(body["features"]).capped(cfg.features, cfg.model_media())

    scoring = body.get("scoring") if body.get("scoring") is not None else {}
    if not isinstance(scoring, dict):
        raise ApiError(400, "invalid_field", "scoring must be an object.", "scoring")
    scoring = dict(scoring)
    own = scoring.get("queries") if scoring.get("queries") is not None else []
    if not isinstance(own, list):
        raise ApiError(400, "invalid_field", "scoring.queries must be a list.", "scoring.queries")
    inputs = [{"id": "0", "messages": body["messages"]}] if has_messages else body["inputs"]
    if not isinstance(inputs, list):
        raise ApiError(400, "invalid_field", "inputs must be a list.", "inputs")

    refs: list[pack_refs.PackRef] = []
    try:
        if scoring.get("packs") is not None:
            refs = pack_refs.parse_refs(scoring["packs"], catalog or Catalog())
            scoring.pop("packs")
        if refs:
            own_ids = [q.get("id") for q in own if isinstance(q, dict)] + [
                q.get("id") for inp in inputs if isinstance(inp, dict) for q in (inp.get("queries") or []) if isinstance(q, dict)
            ]
            pack_refs.check_call(refs, [i for i in own_ids if isinstance(i, str)], scoring.get("case_insensitive"))
            groups = {r.pack.group for r in refs}
            for i, q in enumerate(own):
                if isinstance(q, dict) and q.get("group") in groups:
                    raise Rejected(
                        "reserved_group", f"{q['group']!r} is the query group of a pack of this call.", f"scoring.queries.{i}.group"
                    )
            scoring["case_insensitive"] = refs[0].pack.case_insensitive
            scoring["queries"] = list(own) + pack_refs.queries(refs)
            inputs = [_wrap_input(inp, i, refs, "messages" if has_messages else f"inputs.{i}.messages") for i, inp in enumerate(inputs)]
    except Rejected as e:
        raise ApiError.of(e) from None

    try:
        req = protocol.ScoreRequest.model_validate({"model": model, "inputs": inputs, "scoring": scoring, "limits": limits.model_dump()})
    except ValidationError as e:
        param, msg = _loc(e)
        raise ApiError(400, "invalid_field", msg, param) from None

    _check_features(req, cfg, features, refs)
    return Prepared(req, model, endpoint, refs, chat)


def _features(value: Any) -> Features:
    if not isinstance(value, dict):
        raise ApiError(400, "invalid_field", "features must be an object.", "features")
    media = value.get("media")
    if media is not None and (not isinstance(media, list) or not all(isinstance(k, str) for k in media)):
        raise ApiError(400, "invalid_field", "features.media must be a list of media kinds.", "features.media")
    for k in ("extraction", "generation"):
        if value.get(k) is not None and not isinstance(value[k], bool):
            raise ApiError(400, "invalid_field", f"features.{k} must be true or false.", f"features.{k}")
    return Features(
        tuple(media) if media is not None else None,
        value.get("extraction", True) if value.get("extraction") is not None else True,
        value.get("generation", True) if value.get("generation") is not None else True,
    )


def _parts(content: Any) -> list[dict[str, Any]]:
    """a message's content as parts (a string is one text part)"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content) if isinstance(content, list) else content


def _wrap_input(inp: Any, i: int, refs: list[pack_refs.PackRef], param: str) -> Any:
    """an input in the packs' context: the context's text before the first message's content and after the last's; the pack's
    query ids added to an input's `query_ids`; the content kinds checked against the packs' media"""
    if not isinstance(inp, dict) or not isinstance(inp.get("messages"), list) or not inp["messages"]:
        return inp  # the protocol's validation reports it
    inp = dict(inp)
    msgs = [dict(m) if isinstance(m, dict) else m for m in inp["messages"]]
    if not all(isinstance(m, dict) for m in msgs):
        return inp
    for m, msg in enumerate(msgs):
        parts = _parts(msg.get("content")) if isinstance(msg.get("content"), str | list) else []
        for k, p in enumerate(parts):
            if isinstance(p, dict):
                kind = protocol.MEDIA_PARTS.get(p.get("type"), "text")
                for r in refs:
                    r.check_media([kind], f"{param}.{m}.content.{k}.type")
    params = inp.get("parameters")
    texts = set()
    for r in refs:
        try:
            texts.add(r.pack.context_text(r.context, params))
        except ValueError as e:
            raise Rejected("invalid_field", str(e), f"{param.rsplit('.', 1)[0]}.parameters") from None
    if len(texts) > 1:
        raise Rejected(
            "pack_conflict", "the packs of this call wrap the content in different contexts; send one pack per call.", "scoring.packs"
        )
    ((before, after),) = texts
    first, last = _parts(msgs[0].get("content")), _parts(msgs[-1].get("content"))
    if not isinstance(first, list) or not isinstance(last, list):
        return inp
    if len(msgs) == 1:
        msgs[0]["content"] = (
            ([{"type": "text", "text": before}] if before else []) + first + ([{"type": "text", "text": after}] if after else [])
        )
    else:
        msgs[0]["content"] = ([{"type": "text", "text": before}] if before else []) + first
        msgs[-1]["content"] = last + ([{"type": "text", "text": after}] if after else [])
    inp["messages"] = msgs
    if isinstance(inp.get("query_ids"), list):
        inp["query_ids"] = list(dict.fromkeys([*inp["query_ids"], *(q["id"] for r in refs for q in r.pack.queries())]))
    return inp


def _check_features(req: protocol.ScoreRequest, cfg: ServerConfig, features: Features, refs: list[pack_refs.PackRef]) -> None:
    """media kinds the model reads and the server enables; extraction and generation on, and a model that can"""
    model_media = cfg.model_media()
    enabled = features.media_for(model_media)
    for i, inp in enumerate(req.inputs):
        for k in sorted(protocol.media_kinds(inp)):
            if k not in model_media:
                raise ApiError(400, "media_not_supported", f"{cfg.engine.served_model} does not read {k} input.", f"inputs.{i}.messages")
            if k not in enabled:
                raise ApiError(400, "media_disabled", f"{k} input is disabled on this server.", f"inputs.{i}.messages")
    caps = cfg.capabilities()
    queries = [(q, f"scoring.queries.{k}") for k, q in enumerate(req.scoring.queries)] + [
        (q, f"inputs.{i}.queries.{k}") for i, inp in enumerate(req.inputs) for k, q in enumerate(inp.queries or [])
    ]
    for q, param in queries:
        try:
            kind = Template(q.template, f"{param}.template").kind
        except Rejected as e:
            raise ApiError.of(e) from None
        if kind == "score":
            continue
        if not features.extraction or "extract" not in caps:
            m = cfg.manifest()
            why = (
                "a decision model scores options only" if m and "extract" not in m.capabilities else "extraction is disabled on this server"
            )
            raise ApiError(400, "extraction_disabled", f"{param}: extraction slots are not available: {why}.", f"{param}.template")
        if q.generate and (not features.generation or "generate" not in caps):
            raise ApiError(400, "generation_disabled", f"{param}: generation is not available on this server.", f"{param}.generate")


def respond(prep: Prepared, out: dict[str, Any], cfg: ServerConfig, trusted: bool, request_id: str) -> dict[str, Any]:
    """the engine's result (service.score_request) -> the public response"""
    results, warnings = out["results"], list(out.get("warnings") or [])
    if prep.refs:
        results, warnings = pack_refs.apply(prep.refs, results, warnings)
    u = out["usage"]
    d = out.get("detail") or {}
    prompt = u["content_tokens"] + u["query_tokens"] + u["floor_tokens"]
    generated = u.get("generated_tokens", 0)
    sc = dict(d.get("scoring") or {})
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": generated,
        "total_tokens": prompt + generated,
        "prompt_tokens_details": {
            "content_tokens": u["content_tokens"],
            "text_tokens": u.get("text_tokens", u["content_tokens"]),
            "query_tokens": u["query_tokens"],
            "floor_tokens": u["floor_tokens"],
        },
        "inputs": len(prep.req.inputs),
        "content": {
            "tokens": u["content_tokens"],
            "text_tokens": u.get("text_tokens", u["content_tokens"]),
            "images": u.get("images", 0),
            "image_tokens": u.get("image_tokens", 0),
            "video_frames": u.get("video_frames", 0),
            "video_tokens": u.get("video_tokens", 0),
            "audio_seconds": round(u.get("audio_ms", 0) / 1000, 3),
            "audio_tokens": u.get("audio_tokens", 0),
        },
        "scoring": {**sc, "query_tokens": u["query_tokens"], "floor_tokens": u["floor_tokens"]},
    }
    x = d.get("extraction") or {}
    if x.get("slots"):
        usage["extraction"] = x
    g = d.get("generation") or {}
    if g.get("slots"):
        usage["generation"] = g
    if prep.refs:
        usage["packs"] = {"count": len(prep.refs), "query_tokens": (d.get("packs") or {}).get("query_tokens", 0)}
    usage["processed_tokens"] = d.get("processed_tokens", prompt + generated)
    internal = trusted or cfg.debug
    if internal:
        usage["packed_tokens"] = u.get("packed_tokens", 0)
        # the flat engine counters the hosted front reads (its engine contract): additive, next to the public blocks
        for k in (
            "content_tokens",
            "text_tokens",
            "query_tokens",
            "floor_tokens",
            "generated_tokens",
            "images",
            "image_tokens",
            "video_frames",
            "video_tokens",
            "audio_ms",
            "audio_tokens",
        ):
            if k in u:
                usage[k] = u[k]
    t = out.get("timing") or {}
    timing = {
        "wall_ms": _ms(t.get("wall_s")),
        "queue_ms": _ms(t.get("queue_s")),
        "compile_ms": _ms(t.get("compile_s")),
        "score_ms": _ms(t.get("score_s", t.get("wall_s"))),
        "gpu_busy_ms": _ms(t.get("gpu_busy_s")),
    }
    if "extract_wall_s" in t:
        timing["decode_ms"] = _ms(t["extract_wall_s"])
    resp: dict[str, Any] = {
        "id": request_id,
        "object": "score",
        "created": int(time.time()),
        "model": prep.requested_model,
        **({"endpoint": prep.endpoint} if prep.endpoint is not None else {}),
        "system_fingerprint": out.get("system_fingerprint", ""),
        "results": results,
        "warnings": warnings,
        "usage": usage,
        "timing": timing,
    }
    if internal and out.get("engine"):
        resp["engine"] = out["engine"]
    if prep.chat:
        resp["choices"] = []
    return resp


def handle(
    body: Any,
    cfg: ServerConfig,
    service,
    catalog: Catalog | None,
    trusted: bool = False,
    chat: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    """a decoded request body -> the public response: prepare, score (service.score_request), respond. Raises ApiError (the engine's
    Rejected mapped) and lets the engine's own failures through. The one path of the server and of the RunPod handler."""
    served = service.cfg.served_model
    prep = prepare(body, cfg, served, set(getattr(service, "model_names", {served.lower()})), catalog, trusted=trusted, chat=chat)
    try:
        out = service.score_request(prep.req, prep.pack_groups)
    except Rejected as e:
        raise ApiError.of(e) from None
    rid = request_id or ulid()
    try:
        return respond(prep, out, cfg, trusted, rid)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ApiError(500, "engine_error", f"Pack evaluation failed: {type(e).__name__}.") from e


def _ms(s: float | None) -> int:
    return 0 if s is None else int(round(s * 1000))


def model_entry(cfg: ServerConfig, served_model: str, ready: bool, error: str | None = None) -> dict[str, Any]:
    """the GET /v1/models entry of the served model: names, capabilities, the media enabled, the effective limits (no prices)"""
    m = cfg.manifest()
    media = cfg.features.media_for(cfg.model_media())
    entry: dict[str, Any] = {
        "id": served_model,
        "name": m.name if m else served_model,
        "status": "available" if ready else ("error" if error else "loading"),
        "base_model": m.source.repo if m else cfg.engine.model_id,
        "aliases": list(m.aliases) if m else [],
        "family": m.family if m else None,
        "capabilities": cfg.capabilities(),
        "media": list(media),
        "limits": cfg.limits.model_dump(),
        "features": {
            "media": list(media),
            "extraction": cfg.features.extraction and "extract" in cfg.capabilities(),
            "generation": cfg.features.generation and "generate" in cfg.capabilities(),
        },
    }
    if m and m.image:
        entry["image"] = {"detail_tokens": dict(m.image.detail_tokens)}
    return entry
