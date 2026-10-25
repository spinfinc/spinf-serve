"""The `spinf-serve` command line.

    spinf-serve serve [MODEL] [--port 8000] [--token T] [--packs SOURCE]... [--max-content-tokens N] ...   the server
    spinf-serve score REQUEST.json [MODEL] [--device cuda:0]      load the engine, score one request file, print the response
    spinf-serve packs validate FOLDER | list [--packs ...]        the pack commands (spinf_serve.packs)
    spinf-serve parity [...]                                      the GPU parity run (spinf_serve.__main__)
    spinf-serve calibrate [...]                                   the fp8s scale calibration (spinf_serve.quant.calibrate)

Settings are layered: flags > SPINF_* environment > --config server.toml > the model's manifest defaults (server/config.py).
"""

import argparse
import json
import logging
import sys
from typing import Any

from spinf_serve.config import LM_HEADS, QUANTS, WARMUPS
from spinf_serve.protocol import Limits

LIMIT_FLAGS = tuple(Limits.model_fields)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_server_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", nargs="?", default=None, help="a manifest id (spinf/gemma-4-12b-fp8), an alias, or a Hub checkpoint path")
    p.add_argument(
        "--config", dest="config_file", default=None, metavar="FILE", help="a TOML config file (tables engine, server, features, limits)"
    )
    e = p.add_argument_group("engine")
    e.add_argument("--device", default=None, help="the GPU (default cuda:0)")
    e.add_argument("--quant", default=None, choices=QUANTS, help="the scheme; default: auto for the card")
    e.add_argument("--model-dir", default=None, help="a local snapshot directory instead of the Hub")
    e.add_argument("--token-budget", type=int, default=None, help="packed tokens per forward (default 16384)")
    e.add_argument("--max-tokens", type=int, default=None, help="the largest single input the engine runs")
    e.add_argument("--lm-head", default=None, choices=LM_HEADS)
    e.add_argument("--no-media", dest="load_media", action="store_false", default=None, help="skip loading the image and audio embedders")
    e.add_argument("--no-compile", dest="compile", action="store_false", default=None, help="debugging only: 2-3x slower")
    e.add_argument(
        "--warmup",
        default=None,
        choices=WARMUPS,
        help="before the server is ready: full = every compile a request could pay (default), short = short streams only, none",
    )
    e.add_argument("--stack-options", default=None, help="research knobs of the model family as key=value,key=value")
    s = p.add_argument_group("server")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--token", default=None, help="require Authorization: Bearer <token> on every call")
    s.add_argument("--trusted-token", default=None, help="the trusted callers' token (per-call limits and features, engine fields)")
    s.add_argument("--queue", type=int, default=None, help="requests accepted while one is on the GPU (default 4)")
    s.add_argument("--max-body-bytes", type=int, default=None)
    s.add_argument("--debug", action="store_true", default=None, help="engine internals (packed_tokens, engine) in every response")
    f = p.add_argument_group("features")
    f.add_argument(
        "--media",
        dest="media_kinds",
        default=None,
        metavar="KINDS",
        help="image,audio,video (default: all the model reads); none for text only",
    )
    f.add_argument("--extraction", dest="extraction", action="store_true", default=None)
    f.add_argument("--no-extraction", dest="extraction", action="store_false")
    f.add_argument("--generation", dest="generation", action="store_true", default=None)
    f.add_argument("--no-generation", dest="generation", action="store_false")
    f.add_argument(
        "--packs",
        action="append",
        default=None,
        metavar="SOURCE",
        help="a pack source: github:<owner>/<repo>[@ref][#path] or a folder; repeatable",
    )
    f.add_argument(
        "--no-official-packs", dest="official_packs", action="store_false", default=None, help="leave the official repository out"
    )
    f.add_argument(
        "--packs-allow-namespace", action="append", default=None, metavar="NS", help="serve this reserved namespace from any source"
    )
    f.add_argument("--packs-refresh", action="store_true", default=None, help="resolve GitHub refs again")
    lim = p.add_argument_group("limits", "the request limits, server-wide (a trusted caller may lower them per call)")
    for name in LIMIT_FLAGS:
        lim.add_argument(_flag(name), dest=name, type=int, default=None)


def server_config(a: argparse.Namespace):
    from spinf_serve.server.config import ServerConfig

    cli: dict[str, Any] = {k: v for k, v in vars(a).items() if k not in ("cmd", "func", "request") and v is not None}
    if cli.get("load_media") is False:  # the embedders not loaded: no media kind can be read either
        cli["media_kinds"] = "none"
    if str(cli.get("media_kinds", "")).strip().lower() == "none":
        cli["load_media"] = False
    for k in ("packs", "packs_allow_namespace"):
        if k in cli:
            cli[k] = ",".join(cli[k])
    return ServerConfig.build(cli)


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from spinf_serve.server.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = server_config(a)
    log = logging.getLogger("spinf_serve.cli")
    log.info(
        "serving %s on %s:%d (device %s, packs %s)",
        cfg.engine.served_model,
        cfg.host,
        cfg.port,
        cfg.device,
        "off" if not (cfg.packs or cfg.official_packs) else ", ".join(("official",) * cfg.official_packs + cfg.packs),
    )
    uvicorn.run(
        create_app(cfg=cfg),
        host=cfg.host,
        port=cfg.port,
        workers=1,
        # longer than a proxy's idle upstream keep-alive, so the proxy always closes first
        timeout_keep_alive=cfg.keepalive_s,
        log_level="info",
    )
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    """one request file through the engine, without a server: the public response on stdout"""
    from spinf_serve.server import api
    from spinf_serve.server.app import build_catalog
    from spinf_serve.service import ScoreService

    logging.basicConfig(level=logging.WARNING)
    cfg = server_config(a)
    with open(a.request) as f:
        body = json.load(f)
    svc = ScoreService(cfg.engine, cfg.device)
    svc.load()
    svc.warmup()
    catalog = build_catalog(cfg, svc.cfg.served_model) if body.get("scoring", {}).get("packs") else None
    try:
        prep = api.prepare(body, cfg, svc.cfg.served_model, svc.model_names, catalog, trusted=True)
        out = svc.score_request(prep.req, prep.pack_groups)
        resp = api.respond(prep, out, cfg, trusted=a.debug or False, request_id=api.ulid())
    except api.ApiError as e:
        print(json.dumps(e.body(api.ulid()), indent=1), file=sys.stderr)
        return 1
    print(json.dumps(resp, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="spinf-serve", description="the open-source inference engine for decision models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="serve a model over HTTP (the public scoring API)")
    add_server_flags(serve)
    serve.set_defaults(func=cmd_serve)
    score = sub.add_parser("score", help="score one request file through the engine, no server")
    score.add_argument("request", help="a /v1/score request body (JSON file)")
    add_server_flags(score)
    score.set_defaults(func=cmd_score)
    sub.add_parser("packs", help="prompt packs: validate a repository, list what a server enables", add_help=False)
    sub.add_parser("parity", help="the GPU parity run against a recorded reference", add_help=False)
    sub.add_parser("calibrate", help="calibrate the fp8s activation scales of a checkpoint", add_help=False)
    if argv and argv[0] in ("packs", "parity", "calibrate"):
        rest = argv[1:]
        if argv[0] == "packs":
            from spinf_serve.packs.__main__ import main as packs_main

            return packs_main(rest)
        if argv[0] == "parity":
            from spinf_serve.__main__ import main as parity_main

            sys.argv = [sys.argv[0], "--parity", *rest]
            parity_main()
            return 0
        from spinf_serve.quant.calibrate import main as calibrate_main

        sys.argv = [sys.argv[0], *rest]
        calibrate_main()
        return 0
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
