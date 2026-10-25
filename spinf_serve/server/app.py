"""spinf-serve over HTTP: one process per GPU (never uvicorn --workers N: each worker would load its own model). The public API of
the hosted service (docs/scoring-api.md), so a client switches between the two by changing the base URL and the token:

    POST /v1/score             the scoring request: `messages` (one content) or `inputs`, `scoring` with queries and packs
    POST /v1/chat/completions  the same body (the OpenAI SDKs pass `scoring` as extra_body); the response adds "choices": []
    GET  /v1/models            the served model: names, capabilities, media, the effective limits
    GET  /v1/packs             the prompt packs enabled for the served model; GET /v1/packs/{id}?version= the pack file
    GET  /healthz              the process is up (always 200 once uvicorn serves)
    GET  /readyz               200 once the model is loaded and warm (every warm-up compile done), else 503 (load balancers route
                               on this); 503 {"draining": true}
                               while the drain file exists (SPINF_DRAIN_FILE, default /tmp/spinf-drain)
    GET  /livez                200 once the model is loaded and warm, drain or not (a local proxy health-checks this, so a draining
                               engine keeps scoring while routers move calls away)
    POST / DELETE /v0/drain    start / stop draining (trusted callers)
    GET  /v0/info              what produces the scores (engine version, weights revision, scheme, card, stack): trusted callers
    POST /v0/score             the engine-internal v0 request (spinf_serve.service.ScoreRequest), trusted callers

Errors come in the public envelope {"error": {type, code, message, param}, "request_id"}; a trusted caller (server.config) also gets
the engine's compact `rejected` form, may send `limits` and `features` per call, and receives the engine-internal fields. Every
response carries X-Request-Id. The model loads in a background thread at start-up so the health check answers during the boot.
Requests are served in FastAPI's thread pool and serialised on the GPU by the service; `--queue` bounds how many wait.
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from spinf_serve.server import api
from spinf_serve.server.api import ApiError
from spinf_serve.server.config import ServerConfig

log = logging.getLogger("spinf_serve.server")


def build_catalog(cfg: ServerConfig, model: str | None):
    """the pack catalog of this server (packs.catalog.build), or None when packs are off; an error of a given source refuses the
    start, the official source being unavailable is logged and the server starts without it"""
    if not cfg.packs and not cfg.official_packs:
        return None
    from spinf_serve.packs import catalog

    cat = catalog.build(
        list(cfg.packs), model, official=cfg.official_packs, allow_namespaces=cfg.packs_allow_namespace, refresh=cfg.packs_refresh
    )
    cat.log_summary()
    if cat.errors:
        raise SystemExit("packs: " + "; ".join(cat.errors))
    return cat


def create_app(service=None, cfg: ServerConfig | None = None, catalog=None, load: bool = True) -> FastAPI:
    """service: a ScoreService (built from cfg when None); catalog: the pack catalog (built from cfg when None and packs are on)"""
    cfg = cfg or ServerConfig.build()
    if service is None:
        from spinf_serve.service import ScoreService

        service = ScoreService(cfg.engine, cfg.device)
    svc = service
    served_model = svc.cfg.served_model
    if catalog is None and load:
        catalog = build_catalog(cfg, served_model)
    state: dict[str, Any] = {"error": None}
    drain_file = os.environ.get("SPINF_DRAIN_FILE", "/tmp/spinf-drain")
    slots = threading.BoundedSemaphore(cfg.queue + 1)  # one on the GPU, `queue` waiting

    def boot() -> None:
        try:
            svc.load()
            svc.warmup()
            log.info("engine ready: %s %s, system_fingerprint %s", svc.info(), svc.boot, svc.fingerprint())
        except Exception as e:  # keep serving /healthz so the failure is visible, never ready
            log.exception("engine failed to load")
            state["error"] = f"{type(e).__name__}: {e}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load and not svc.ready:
            threading.Thread(target=boot, name="engine-boot", daemon=True).start()
        yield

    app = FastAPI(title="spinf-serve", version="0.1", lifespan=lifespan, docs_url=None, redoc_url=None)

    def request_id(request: Request) -> str:
        rid = getattr(request.state, "request_id", None)
        if rid is None:
            rid = request.state.request_id = api.ulid()
        return rid

    def bearer(request: Request) -> str:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        return token.strip() if scheme.lower() == "bearer" else ""

    def is_trusted(request: Request) -> bool:
        return cfg.trusted(bearer(request))

    def authorize(request: Request) -> None:
        if not cfg.authorized(bearer(request)):
            raise ApiError(401, "invalid_token", "This server requires a bearer token (Authorization: Bearer <token>).")

    def error_response(request: Request, e: ApiError) -> JSONResponse:
        rid = request_id(request)
        body = e.body(rid, rejected=is_trusted(request))
        return JSONResponse(body, status_code=e.status, headers={**e.headers, "X-Request-Id": rid})

    @app.middleware("http")
    async def with_request_id(request: Request, call_next):
        rid = request_id(request)
        response = await call_next(request)
        response.headers.setdefault("X-Request-Id", rid)
        return response

    @app.exception_handler(ApiError)
    async def on_api_error(request: Request, e: ApiError):
        return error_response(request, e)

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, e: RequestValidationError):
        err = e.errors()[0]
        param = ".".join(str(x) for x in err.get("loc", ()) if x != "body") or None
        return error_response(request, ApiError(400, "invalid_field", err.get("msg", "invalid request"), param))

    @app.exception_handler(Exception)
    async def on_error(request: Request, e: Exception):
        log.exception("request %s failed", request_id(request))
        return error_response(request, ApiError(500, "engine_error", "Scoring failed."))

    async def read_json(request: Request) -> Any:
        body = await request.body()
        if len(body) > cfg.max_body_bytes:
            raise ApiError(413, "body_too_large", f"The request body is over {cfg.max_body_bytes // (1024 * 1024)} MB.")
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "invalid_json", "The request body is not valid JSON.") from None

    def ready_or_503() -> None:
        if state["error"]:
            raise ApiError(503, "model_unavailable", f"The model failed to load: {state['error']}", headers={"Retry-After": "60"})
        if not svc.ready or not svc.boot.get("warm"):
            raise ApiError(503, "warming_up", "The model is loading. Retry shortly.", headers={"Retry-After": "10"})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> Response:
        if state["error"]:
            return JSONResponse({"ready": False, "error": state["error"]}, status_code=503)
        if not svc.ready or not svc.boot.get("warm"):
            return JSONResponse({"ready": False}, status_code=503)
        if getattr(svc, "degraded", None):
            return JSONResponse({"ready": False, "degraded": svc.degraded}, status_code=503)
        if os.path.exists(drain_file):
            return JSONResponse({"ready": False, "draining": True}, status_code=503)
        return JSONResponse({"ready": True, "boot": svc.boot})

    @app.get("/livez")
    def livez() -> Response:
        if state["error"] or not svc.ready or not svc.boot.get("warm"):
            return JSONResponse({"live": False}, status_code=503)
        if getattr(svc, "degraded", None):  # memory leaked after a failure: the load balancer must stop sending calls
            return JSONResponse({"live": False, "degraded": svc.degraded}, status_code=503)
        return JSONResponse({"live": True})

    def trusted_only(request: Request) -> None:
        authorize(request)
        if not is_trusted(request):
            raise ApiError(401, "invalid_token", "This endpoint is for trusted callers (the server's trusted token).")

    @app.post("/v0/drain")
    def drain(request: Request) -> dict:
        trusted_only(request)
        with open(drain_file, "w"):
            pass
        return {"draining": True}

    @app.delete("/v0/drain")
    def undrain(request: Request) -> dict:
        trusted_only(request)
        if os.path.exists(drain_file):
            os.remove(drain_file)
        return {"draining": False}

    @app.get("/v0/info")
    def info(request: Request) -> dict:
        trusted_only(request)
        ready_or_503()
        return {"engine": svc.info(), "boot": svc.boot}

    @app.get("/v1/models")
    def models(request: Request) -> dict:
        authorize(request)
        return {"models": [api.model_entry(cfg, served_model, svc.ready and bool(svc.boot.get("warm")), state["error"])]}

    @app.get("/v1/packs")
    def packs(request: Request) -> dict:
        authorize(request)
        return {"packs": catalog.public() if catalog is not None else [], "sources": list(catalog.sources) if catalog is not None else []}

    @app.get("/v1/packs/{pack_id:path}")
    def pack_file(pack_id: str, request: Request, version: str | None = None) -> dict:
        authorize(request)
        versions = catalog.versions(pack_id) if catalog is not None else []
        if not versions:
            raise ApiError(404, "unknown_pack", f"No pack {pack_id!r} on this server. See GET /v1/packs.", "id")
        if version is not None and version not in versions:
            raise ApiError(
                404, "unknown_pack_version", f"{pack_id} has no version {version!r}; versions: {', '.join(versions)}.", "version"
            )
        return catalog.get(pack_id, version).data

    def score_public(request: Request, body: Any, chat: bool) -> JSONResponse:
        authorize(request)
        trusted = is_trusted(request)
        ready_or_503()
        if not slots.acquire(blocking=False):
            raise ApiError(429, "concurrency_limit", "The server's queue is full. Retry shortly.", headers={"Retry-After": "2"})
        rid = request_id(request)
        try:
            resp = api.handle(body, cfg, svc, catalog, trusted=trusted, chat=chat, request_id=rid)
        except ApiError as e:
            if e.status == 500:
                log.exception("request %s: %s", rid, e.message)
            raise
        finally:
            slots.release()
        return JSONResponse(resp, headers={"X-Request-Id": rid})

    @app.post("/v1/score")
    async def score(request: Request) -> Response:
        body = await read_json(request)
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(score_public, request, body, False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await read_json(request)
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(score_public, request, body, True)

    @app.post("/v0/score")
    async def score_v0(request: Request) -> Response:
        trusted_only(request)
        ready_or_503()
        from pydantic import ValidationError
        from starlette.concurrency import run_in_threadpool

        from spinf_serve.service import ScoreRequest

        body = await read_json(request)
        try:
            req = ScoreRequest.model_validate(body)
        except ValidationError as e:
            err = e.errors(include_url=False)[0]
            raise ApiError(400, "invalid_field", err["msg"], ".".join(str(x) for x in err["loc"]) or None) from None
        try:
            res = await run_in_threadpool(svc.score, req)
        except ValueError as e:  # a job over the token budget, duplicate ids
            raise ApiError(400, "invalid_field", str(e)) from None
        return JSONResponse(res.model_dump(), headers={"X-Request-Id": request_id(request)})

    return app
