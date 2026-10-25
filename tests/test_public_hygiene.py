"""The public repository names no private repository, host, deployment or local path. The list is the one agreed for the launch
(plan §6 step 7); add to it rather than around it."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = (
    "spinf_serve",
    "tests",
    "docs",
    "quantizations",
    "benchmarks",
    "fixtures",
    ".github",
    "README.md",
    "NOTICE",
    "ACKNOWLEDGEMENTS.md",
    "pyproject.toml",
)
FORBIDDEN = [
    "gtrade",
    "gtgpu",
    "Bigdata",
    "spinf-inference-api",
    "spinf_api",
    "API.md",
    "spinf-console",
    "spinf-testbed",
    "benchmarks-lab",
    "runpod-volume",
    "~/Projects",
    "/home/",
    "workers.spinf.com",
    "api.runpod",
    "Valkey",
    "Terraform",
    "-private",  # the development mirrors of the public repositories (plan §6 step 7): the environment points at them, never the code
]
# the capitalised word: Python's lambda is fine
FORBIDDEN_RE = [re.compile(r"\bLambda\b")]
SKIP = {os.path.abspath(__file__)}


def files():
    for top in SCAN:
        p = os.path.join(ROOT, top)
        if os.path.isfile(p):
            yield p
            continue
        for d, _, names in os.walk(p):
            if "__pycache__" in d:
                continue
            for n in names:
                if n.endswith((".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", "NOTICE")):
                    yield os.path.join(d, n)


def test_no_internal_references():
    hits = []
    for f in files():
        if f in SKIP or f.endswith(".json"):  # calibration tables carry no prose
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                for tok in FORBIDDEN:
                    if tok in line:
                        hits.append(f"{os.path.relpath(f, ROOT)}:{i}: {tok}")
                for rx in FORBIDDEN_RE:
                    if rx.search(line):
                        hits.append(f"{os.path.relpath(f, ROOT)}:{i}: {rx.pattern}")
    assert not hits, "internal references in the public repo:\n" + "\n".join(hits)
