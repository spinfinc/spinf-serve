"""python -m spinf_serve --parity [--reference FILE] [--passes 2]: load the engine on one GPU and score the parity fixture."""

import argparse
import json
import os

from spinf_serve.config import EngineConfig

FIXTURE = os.environ.get("SPINF_FIXTURE", "fixtures/wikinews_20261024.json.gz")


def main() -> None:
    ap = argparse.ArgumentParser(prog="spinf_serve")
    ap.add_argument("--parity", action="store_true", help="score the fixture and compare")
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--reference", default=None, help="a recorded reference (a --record output; default: the fixture's own scores)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--record", default=None, help="write this run's scores as a reference fixture (a new scheme's parity reference)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not a.parity:
        ap.error("nothing to do (--parity)")
    from spinf_serve import parity
    from spinf_serve.service import ScoreService

    svc = ScoreService(EngineConfig.from_env(), a.device)
    svc.load()
    out = parity.run(svc.scorer, a.fixture, a.reference, a.passes, a.record)
    print(json.dumps({"boot": svc.boot, "engine": svc.info(), **out}, indent=1))


if __name__ == "__main__":
    main()
