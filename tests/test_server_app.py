"""The HTTP server with a fake engine: the public envelope and errors, the one-content form, the chat alias, tokens and trusted
callers, the queue, the models and packs endpoints, packs in a call (wrapping, grouped queries, evaluation, parameters), the
feature gates, and the engine-internal v0 endpoint."""

import importlib.util
import json
import math
import os
import threading

import pytest
from fastapi.testclient import TestClient

from spinf_serve.packs import catalog, from_data
from spinf_serve.protocol import Rejected
from spinf_serve.server.app import create_app
from spinf_serve.server.config import Features, ServerConfig
from spinf_serve.service import JobOut, ScoreRequest, ScoreResponse

_spec = importlib.util.spec_from_file_location("test_packs_mod", os.path.join(os.path.dirname(__file__), "test_packs.py"))
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
make_pack = _tp.make_pack

MODEL = "spinf/gemma-4-12b-fp8"


class FakeService:
    """answers every scored query with uniform probabilities; records the last request"""

    def __init__(self, ready=True, model=MODEL):
        from spinf_serve.config import EngineConfig

        self.cfg = EngineConfig(served_model=model)
        self.model_names = {model.lower(), "alias-12b"}
        self.ready = ready
        self.boot = {"load_s": 1.0, "warmup_s": 1.0, "warmup_extract_s": 1.0, "warm": True} if ready else {}
        self.last = None
        self.last_groups = None
        self.block: threading.Event | None = None
        self.started = threading.Event()

    def info(self):
        return {"engine": "test", "quant": "fp8s", "gpu": "fake"}

    def fingerprint(self):
        return "fp_test"

    def load(self):
        pass

    def warmup(self):
        pass

    def score_request(self, req, pack_groups=frozenset()):
        self.last, self.last_groups = req, pack_groups
        if self.block is not None:
            self.started.set()
            self.block.wait(5)
        if req.inputs[0].id == "too-long":
            raise Rejected("content_too_long", "content is too long", "inputs.0")
        if req.inputs[0].id == "huge":
            raise Rejected("request_too_large", "too much", None)
        shared = {q.id: q for q in req.scoring.queries}
        results = []
        n_prompts = 0
        for inp in req.inputs:
            qs = list(shared.values()) if inp.query_ids is None else [shared[i] for i in inp.query_ids]
            qs += list(inp.queries or [])
            out = []
            for q in qs:
                combos = q.combinations if isinstance(q.combinations, list) else [{}]
                n_prompts += len(combos)
                if q.options is None:
                    out.append(
                        {
                            "id": q.id,
                            "kind": "text",
                            "combinations": [{"values": c, "answers": [{"text": "x", "p": 0.5}], "mass": 0.5} for c in combos],
                        }
                    )
                    continue
                p = 1 / len(q.options)
                out.append(
                    {
                        "id": q.id,
                        "combinations": [
                            {
                                "values": c,
                                "options": [{"text": o, "logprob": math.log(p), "p": p, "floor_p": None} for o in q.options],
                                "residual": 0.0,
                                "floor_residual": None,
                            }
                            for c in combos
                        ],
                    }
                )
            results.append({"input_id": inp.id, "queries": out})
        return {
            "results": results,
            "warnings": [{"code": "low_option_mass", "input_id": req.inputs[0].id, "query_id": qs[0].id, "message": "m"}] if qs else [],
            "usage": {
                "content_tokens": 20,
                "text_tokens": 20,
                "query_tokens": 30,
                "floor_tokens": 10,
                "packed_tokens": 99,
                "generated_tokens": 0,
                "images": 0,
                "image_tokens": 0,
                "video_frames": 0,
                "video_tokens": 0,
                "audio_ms": 0,
                "audio_tokens": 0,
            },
            "detail": {
                "scoring": {"queries": len(shared), "combinations": 3, "options": 6, "prompts": n_prompts, "floor_prompts": 2},
                "extraction": {"slots": 0, "text": 0, "number": 0, "currency": 0, "answers": 0, "decode_tokens": 0},
                "generation": {"slots": 0, "sequences": 0, "max_tokens": 0, "generated_tokens": 0},
                "packs": {"count": len(pack_groups), "query_tokens": 12 if pack_groups else 0},
                "processed_tokens": 60,
            },
            "system_fingerprint": "fp_test",
            "timing": {"wall_s": 0.1, "gpu_busy_s": 0.05, "compile_s": 0.01, "queue_s": 0.0, "score_s": 0.1},
            "engine": self.info(),
        }

    def score(self, req: ScoreRequest) -> ScoreResponse:
        if req.jobs[0].id == "too-long":
            raise ValueError("over the budget")
        return ScoreResponse(
            results=[JobOut(id=j.id, logprobs=[[-1.0] * len(o) for o in j.options], packed_tokens=10, trees=1) for j in req.jobs],
            usage={"packed_tokens": 10, "forwards": 1},
            timing={"wall_s": 0.1, "gpu_busy_s": 0.05},
            engine=self.info(),
        )


def client(svc=None, cfg=None, cat=None):
    return TestClient(create_app(svc or FakeService(), cfg or ServerConfig.build({}, env={}), cat, load=False))


Q = {"id": "q", "template": "\n\nIs it?{?}", "options": [" yes", " no"]}
ONE = {"model": MODEL, "messages": [{"role": "user", "content": "some text"}], "scoring": {"queries": [Q]}}
MANY = {"model": MODEL, "inputs": [{"id": "a", "messages": [{"role": "user", "content": "some text"}]}], "scoring": {"queries": [Q]}}
AUTH = {"Authorization": "Bearer s3cret"}


def test_health_ready_and_warming_up():
    c = client()
    assert c.get("/healthz").json() == {"ok": True}
    assert c.get("/readyz").status_code == 200 and c.get("/livez").status_code == 200
    c = client(FakeService(ready=False))
    assert c.get("/readyz").status_code == 503 and c.get("/livez").status_code == 503
    r = c.post("/v1/score", json=ONE)
    assert r.status_code == 503 and r.json()["error"] == {
        "type": "unavailable",
        "code": "warming_up",
        "message": r.json()["error"]["message"],
        "param": None,
    }
    assert r.headers["Retry-After"] == "10" and r.headers["X-Request-Id"].startswith("scr_")


def test_one_content_and_the_public_envelope():
    svc = FakeService()
    c = client(svc)
    r = c.post("/v1/score", json=ONE)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["object"] == "score" and b["id"].startswith("scr_") and len(b["id"]) == 30 and b["id"] == r.headers["X-Request-Id"]
    assert b["model"] == MODEL and b["system_fingerprint"] == "fp_test" and "choices" not in b and "endpoint" not in b
    assert b["results"][0]["input_id"] == "0" and b["results"][0]["queries"][0]["id"] == "q"
    assert b["warnings"][0]["code"] == "low_option_mass"
    u = b["usage"]
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (60, 0, 60)
    assert u["prompt_tokens_details"] == {"content_tokens": 20, "text_tokens": 20, "query_tokens": 30, "floor_tokens": 10}
    assert u["inputs"] == 1 and u["content"]["tokens"] == 20 and u["scoring"]["prompts"] == 1 and u["scoring"]["floor_prompts"] == 2
    assert "extraction" not in u and "generation" not in u and "packs" not in u and "packed_tokens" not in u and "engine" not in b
    assert u["processed_tokens"] == 60
    assert b["timing"] == {"wall_ms": 100, "queue_ms": 0, "compile_ms": 10, "score_ms": 100, "gpu_busy_ms": 50}
    assert svc.last.inputs[0].id == "0" and svc.last.model == MODEL
    # the model's other names, and the model defaulting to the served one
    assert c.post("/v1/score", json={**ONE, "model": "ALIAS-12b"}).json()["model"] == "ALIAS-12b"
    body = dict(ONE)
    del body["model"]
    assert c.post("/v1/score", json=body).json()["model"] == MODEL
    # many contents, endpoint echoed
    b = c.post("/v1/score", json={**MANY, "endpoint": "gpu-a"}).json()
    assert b["results"][0]["input_id"] == "a" and b["endpoint"] == "gpu-a"


def test_errors_in_the_envelope():
    c = client()
    r = c.post("/v1/score", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json" and r.json()["request_id"].startswith("scr_")
    r = c.post("/v1/score", json={**ONE, "inputs": MANY["inputs"]})
    assert r.json()["error"]["code"] == "invalid_field" and r.json()["error"]["param"] == "messages"
    r = c.post("/v1/score", json={**ONE, "model": "other"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_served" and r.json()["error"]["param"] == "model"
    r = c.post("/v1/score", json={**ONE, "scoring": {"queries": [{"id": "q", "template": 3}]}})
    e = r.json()["error"]
    assert r.status_code == 400 and e["code"] == "invalid_field" and e["param"] == "scoring.queries.0.template"
    r = c.post("/v1/score", json={**MANY, "inputs": [{"id": "too-long", "messages": [{"role": "user", "content": "x"}]}]})
    assert r.status_code == 400 and r.json()["error"] == {
        "type": "invalid_request",
        "code": "content_too_long",
        "message": "content is too long",
        "param": "inputs.0",
    }
    assert "rejected" not in r.json()
    r = c.post("/v1/score", json={**MANY, "inputs": [{"id": "huge", "messages": [{"role": "user", "content": "x"}]}]})
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
    assert c.get("/v1/account").status_code == 404
    cfg = ServerConfig.build({"max_body_bytes": 100}, env={})
    r = client(cfg=cfg).post("/v1/score", json={**ONE, "messages": [{"role": "user", "content": "x" * 200}]})
    assert r.status_code == 413 and r.json()["error"]["code"] == "body_too_large"


def test_chat_completions_alias():
    c = client()
    r = c.post("/v1/chat/completions", json=ONE)
    assert r.status_code == 200 and r.json()["choices"] == [] and r.json()["results"][0]["input_id"] == "0"
    body = dict(ONE)
    del body["scoring"]
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "scoring_required"


def test_tokens_and_trusted_callers():
    svc = FakeService()
    cfg = ServerConfig.build({"token": "s3cret"}, env={})
    c = client(svc, cfg)
    r = c.post("/v1/score", json=ONE)
    assert r.status_code == 401 and r.json()["error"] == {
        "type": "authentication_error",
        "code": "invalid_token",
        "message": r.json()["error"]["message"],
        "param": None,
    }
    assert c.get("/healthz").status_code == 200 and c.get("/v1/models").status_code == 401
    # the access token's holders are trusted when no trusted token is set: per-call limits (capped), the engine fields
    body = {**ONE, "limits": {"max_content_tokens": 8192, "max_query_tokens": 100}}
    r = c.post("/v1/score", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    assert svc.last.limits.max_content_tokens == 4096 and svc.last.limits.max_query_tokens == 100
    b = r.json()
    assert b["usage"]["packed_tokens"] == 99 and b["usage"]["content_tokens"] == 20 and b["engine"]["quant"] == "fp8s"
    r = c.post("/v1/score", json={**MANY, "inputs": [{"id": "too-long", "messages": [{"role": "user", "content": "x"}]}]}, headers=AUTH)
    assert r.json()["rejected"] == {"code": "content_too_long", "message": "content is too long", "param": "inputs.0"}
    # a separate trusted token: the access token's holders are plain callers, their limits ignored
    cfg = ServerConfig.build({"token": "s3cret", "trusted_token": "fr0nt"}, env={})
    c = client(svc, cfg)
    r = c.post("/v1/score", json=body, headers=AUTH)
    assert r.status_code == 200 and svc.last.limits.max_query_tokens == 256 and "packed_tokens" not in r.json()["usage"]
    r = c.post("/v1/score", json=body, headers={"Authorization": "Bearer fr0nt"})
    assert r.status_code == 200 and svc.last.limits.max_query_tokens == 100 and "engine" in r.json()
    assert (
        c.get("/v0/info", headers=AUTH).status_code == 401
        and c.get("/v0/info", headers={"Authorization": "Bearer fr0nt"}).status_code == 200
    )
    # --debug: the engine fields for everyone
    r = client(svc, ServerConfig.build({"debug": True}, env={})).post("/v1/score", json=ONE)
    assert "engine" in r.json() and r.json()["usage"]["packed_tokens"] == 99


def test_queue_full_is_429():
    svc = FakeService()
    svc.block = threading.Event()
    c = client(svc, ServerConfig.build({"queue": 0}, env={}))
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", c.post("/v1/score", json=ONE)))
    t.start()
    assert svc.started.wait(5)
    r = c.post("/v1/score", json=ONE)
    assert r.status_code == 429 and r.json()["error"]["code"] == "concurrency_limit" and r.headers["Retry-After"]
    svc.block.set()
    t.join(5)
    assert out["r"].status_code == 200


def test_models_endpoint():
    c = client()
    m = c.get("/v1/models").json()["models"][0]
    assert m["id"] == MODEL and m["status"] == "available" and m["capabilities"] == ["score", "extract", "generate"]
    assert m["media"] == ["image", "audio", "video"] and m["limits"]["max_content_tokens"] == 4096 and "price" not in json.dumps(m)
    cfg = ServerConfig.build({"model": "spinf/pplx-decider-v1-27b-int4", "max_content_tokens": 2048}, env={})
    svc = FakeService(model="spinf/pplx-decider-v1-27b-int4")
    m = client(svc, cfg).get("/v1/models").json()["models"][0]
    assert m["capabilities"] == ["score"] and m["media"] == [] and m["limits"]["max_content_tokens"] == 2048 and m["name"]
    assert m["features"] == {"media": [], "extraction": False, "generation": False}


def _catalog(tmp_path, **over):
    data = make_pack(**over)
    (tmp_path / "demo.json").write_text(json.dumps(data))
    cat = catalog.Catalog(MODEL)
    cat.add(catalog.scan(str(tmp_path), "local:demo"), allow_namespaces=("acme",))
    assert not cat.errors, cat.errors
    return cat, data


def test_packs_endpoints(tmp_path):
    cat, data = _catalog(tmp_path)
    c = client(cat=cat)
    r = c.get("/v1/packs").json()
    assert r["packs"][0]["id"] == "acme/demo" and r["packs"][0]["versions"] == ["0.1.0"] and r["sources"] == ["local:demo"]
    assert c.get("/v1/packs/acme/demo").json() == data
    assert c.get("/v1/packs/acme/demo?version=0.1.0").json()["version"] == "0.1.0"
    r = c.get("/v1/packs/acme/nope")
    assert r.status_code == 404 and r.json()["error"] == {
        "type": "not_found",
        "code": "unknown_pack",
        "message": r.json()["error"]["message"],
        "param": "id",
    }
    assert c.get("/v1/packs/acme/demo?version=9").json()["error"]["code"] == "unknown_pack_version"
    # no catalog: an empty listing, unknown packs
    c = client()
    assert c.get("/v1/packs").json() == {"packs": [], "sources": []}
    r = c.post("/v1/score", json={**ONE, "scoring": {"packs": [{"id": "acme/demo"}]}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_pack"


def test_a_call_with_a_pack(tmp_path):
    cat, data = _catalog(tmp_path)
    svc = FakeService()
    c = client(svc, cat=cat)
    own = {"id": "own", "template": "\n\nLanguage:{?}", "options": [" en", " it"]}
    body = {
        "model": MODEL,
        "inputs": [
            {"id": "a", "messages": [{"role": "user", "content": "Send money now"}], "parameters": {"brand": "Kestrel"}},
            {"id": "b", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "query_ids": []},
        ],
        "scoring": {"queries": [own], "packs": [{"id": "acme/demo", "context": "mail", "decision": {"mode": "per_category"}}]},
    }
    r = c.post("/v1/score", json=body)
    assert r.status_code == 200, r.text
    req = svc.last
    # the content wrapped in the context with the input's parameters; the pack's queries grouped and appended; the case rule
    parts = req.inputs[0].messages[0].content
    assert (
        parts[0].text == "Brand: Kestrel\n\nAn email:\n\n"
        and parts[1].text == "Send money now"
        and parts[2].text.startswith("\n\n(end; the brand is Kestrel")
    )
    assert req.inputs[1].messages[0].content[0].text == "Brand: Acme\n\nAn email:\n\n"  # the parameter's default
    assert [q.id for q in req.scoring.queries] == ["own", "q.is", "q.topic"] and req.scoring.queries[1].group == "acme/demo@0.1.0"
    assert req.scoring.queries[1].empty_floor is False and req.scoring.case_insensitive is False
    assert req.inputs[1].query_ids == ["q.is", "q.topic"]  # an input with query_ids still gets the pack
    assert svc.last_groups == frozenset({"acme/demo@0.1.0"})
    b = r.json()
    res = b["results"][0]
    assert [q["id"] for q in res["queries"]] == ["own"]  # the pack's raw queries hidden
    p = res["packs"][0]
    assert (
        p["id"] == "acme/demo"
        and p["version"] == "0.1.0"
        and p["context"] == "mail"
        and set(p["categories"]) == {"fraud", "money", "urgent"}
    )
    assert p["decision"]["mode"] == "per_category" and isinstance(p["decision"]["unsafe"], bool)
    assert b["usage"]["packs"] == {"count": 1, "query_tokens": 12} and b["warnings"] == []  # the pack query's warning hidden too
    # include_queries keeps them; the micro-layer default mode
    body["scoring"]["packs"] = [{"id": "acme/demo", "include_queries": True}]
    res = c.post("/v1/score", json=body).json()["results"][0]
    assert [q["id"] for q in res["queries"]] == ["own", "q.is", "q.topic"] and res["packs"][0]["decision"]["mode"] == "micro_layer"

    # the errors
    def err(scoring, inputs=None):
        r = c.post("/v1/score", json={"model": MODEL, "inputs": inputs or body["inputs"], "scoring": scoring})
        assert r.status_code == 400, r.text
        return r.json()["error"]["code"]

    assert err({"queries": [own], "packs": [{"id": "acme/demo"}], "case_insensitive": True}) == "pack_conflict"
    assert err({"queries": [{**own, "id": "q.is"}], "packs": [{"id": "acme/demo"}]}) == "duplicate_id"
    assert err({"queries": [{**own, "group": "acme/demo@0.1.0"}], "packs": [{"id": "acme/demo"}]}) == "reserved_group"
    assert err({"packs": [{"id": "acme/demo", "decision": {"thresholds": {"nope": 0.5}}}]}) == "unknown_category"
    assert err({"packs": [{"id": "acme/demo", "version": "0.0.9"}]}) == "unknown_pack_version"
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert (
        err({"packs": [{"id": "acme/demo"}]}, [{"id": "a", "messages": [{"role": "user", "content": [img]}]}]) == "pack_media_not_supported"
    )
    r = c.post("/v1/score", json={**body, "inputs": [{**body["inputs"][0], "parameters": {"nope": 1}}]})
    assert r.json()["error"]["code"] == "invalid_field" and r.json()["error"]["param"] == "inputs.0.parameters"
    # the one-content form with a pack: the message is wrapped
    r = c.post(
        "/v1/score", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "scoring": {"packs": [{"id": "acme/demo"}]}}
    )
    assert r.status_code == 200 and r.json()["results"][0]["packs"][0]["context"] == "mail"
    assert svc.last.inputs[0].messages[0].content[1].text == "hi"


def test_feature_gates():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    with_img = {**ONE, "messages": [{"role": "user", "content": [{"type": "text", "text": "t"}, img]}]}
    xq = {"id": "x", "template": '\n\nThe city is "{*}"'}
    gq = {**xq, "generate": True}
    # a decision model: no media, no extraction
    cfg = ServerConfig.build({"model": "spinf/pplx-decider-v1-27b-int4"}, env={})
    c = client(FakeService(model="spinf/pplx-decider-v1-27b-int4"), cfg)
    dec = lambda body: {**body, "model": "spinf/pplx-decider-v1-27b-int4"}  # noqa: E731
    r = c.post("/v1/score", json=dec(with_img))
    assert r.json()["error"]["code"] == "media_not_supported" and r.json()["error"]["param"] == "inputs.0.messages"
    r = c.post("/v1/score", json=dec({**ONE, "scoring": {"queries": [xq]}}))
    assert r.json()["error"]["code"] == "extraction_disabled" and "decision model" in r.json()["error"]["message"]
    # the server's switches
    cfg = ServerConfig.build({"model": MODEL, "media_kinds": "image", "extraction": False}, env={})
    c = client(FakeService(), cfg)
    assert c.post("/v1/score", json=with_img).status_code == 200
    audio = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    r = c.post("/v1/score", json={**ONE, "messages": [{"role": "user", "content": [audio]}]})
    assert r.json()["error"]["code"] == "media_disabled"
    assert c.post("/v1/score", json={**ONE, "scoring": {"queries": [xq]}}).json()["error"]["code"] == "extraction_disabled"
    cfg = ServerConfig.build({"model": MODEL, "generation": False}, env={})
    c = client(FakeService(), cfg)
    assert c.post("/v1/score", json={**ONE, "scoring": {"queries": [xq]}}).status_code == 200
    r = c.post("/v1/score", json={**ONE, "scoring": {"queries": [gq]}})
    assert r.json()["error"]["code"] == "generation_disabled" and r.json()["error"]["param"] == "scoring.queries.0.generate"
    # a trusted caller's features, capped by the server's
    cfg = ServerConfig.build({"model": MODEL, "token": "t", "media_kinds": "image"}, env={})
    c = client(FakeService(), cfg)
    h = {"Authorization": "Bearer t"}
    assert c.post("/v1/score", json={**with_img, "features": {"media": ["image", "audio"]}}, headers=h).status_code == 200
    assert c.post("/v1/score", json={**with_img, "features": {"media": []}}, headers=h).json()["error"]["code"] == "media_disabled"
    r = c.post("/v1/score", json={**ONE, "scoring": {"queries": [xq]}, "features": {"extraction": False}}, headers=h)
    assert r.json()["error"]["code"] == "extraction_disabled"
    assert Features(("image",)).capped(Features(("audio", "image")), ("image", "audio", "video")).media == ("image",)


def test_v0_and_drain(monkeypatch, tmp_path):
    monkeypatch.setenv("SPINF_DRAIN_FILE", str(tmp_path / "drain"))
    svc = FakeService()
    cfg = ServerConfig.build({"trusted_token": "fr0nt"}, env={})
    c = client(svc, cfg)
    h = {"Authorization": "Bearer fr0nt"}
    body = {"jobs": [{"id": "a", "prompts": ["x is likely to"], "options": [[" rise", " fall"]]}]}
    assert c.post("/v0/score", json=body).status_code == 401  # trusted callers only
    r = c.post("/v0/score", json=body, headers=h)
    assert r.status_code == 200 and r.json()["results"][0]["logprobs"] == [[-1.0, -1.0]]
    r = c.post("/v0/score", json={"jobs": [{"id": "a", "prompts": ["p"], "options": []}]}, headers=h)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_field"
    r = c.post("/v0/score", json={"jobs": [{"id": "too-long", "prompts": ["p"], "options": [[" a"]]}]}, headers=h)
    assert r.status_code == 400
    # a trusted token alone closes the server too: callers need one of the tokens set
    assert c.post("/v1/score", json=ONE).status_code == 401 and c.post("/v1/score", json=ONE, headers=h).status_code == 200
    assert c.get("/readyz").status_code == 200
    assert c.post("/v0/drain", headers=h).json() == {"draining": True}
    r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["draining"] is True
    assert c.post("/v1/score", json=ONE, headers=h).status_code == 200  # still scores while draining
    assert c.get("/livez").status_code == 200  # and stays live for the local load balancer
    assert c.delete("/v0/drain", headers=h).json() == {"draining": False}
    assert c.get("/readyz").status_code == 200


def test_the_pack_module_evaluates_what_the_server_returns(tmp_path):
    """the server's packs[] entry is Pack.evaluate on the raw results: the same numbers a client computes from include_queries"""
    cat, data = _catalog(tmp_path)
    c = client(cat=cat)
    body = {**ONE, "scoring": {"packs": [{"id": "acme/demo", "include_queries": True}]}}
    res = c.post("/v1/score", json=body).json()["results"][0]
    pack = from_data(data)
    assert pack.evaluate(res) == res["packs"][0]


@pytest.mark.parametrize("path", ["/v1/score", "/v1/chat/completions"])
def test_invalid_body_shape(path):
    r = client().post(path, json=[1, 2])
    assert r.status_code == 400 and r.json()["error"]["param"] == "body"
