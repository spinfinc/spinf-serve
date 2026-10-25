"""The pack commands (`spinf-serve packs ...`, until the command line lands: `python -m spinf_serve.packs ...`):

validate <folder>                 a pack author's check: the repo manifest and every pack file (schema, lock, files present);
                                  prints the listing; exit 1 on any problem. The command a community repo runs in CI.
list [--packs <source>]... [--model <id>] [--no-official] [--allow-namespace <ns>] [--refresh]
                                  what a server with these options would enable, and what it skips and why; no model is loaded
"""

import argparse
import sys

from spinf_serve.packs import catalog as cat


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def validate(folder: str) -> int:
    s = cat.scan(folder)
    rows = [["pack", "version", "model", "status", "queries", "prompts", "lock"]]
    for p in s.packs:
        lock = "-" if p.layer is None else ("ok" if p.lock_ok else "MISMATCH")
        rows.append([p.id, p.version, p.model, p.status, str(len(p.query_ids)), str(p.prompts), lock])
        if p.layer is not None and not p.lock_ok:
            s.problems.append(f"{p.id} {p.version}: the micro-layer lock does not match the file (edited after it was fitted)")
    how = f"manifest {cat.MANIFEST}, namespace {s.manifest.namespace}" if s.manifest else f"no {cat.MANIFEST}, discovered files"
    print(f"{folder}: {how}")
    if s.packs:
        print(_table(rows))
    else:
        print("no pack files")
    for e in s.listed.values():
        for v in e.versions:
            if v.status == "coming_soon":
                print(f"{e.id} {v.version}: coming soon")
    if s.problems:
        print(f"\n{len(s.problems)} problem(s):", file=sys.stderr)
        for pr in s.problems:
            print("  " + pr, file=sys.stderr)
        return 1
    print(f"\nok: {len(s.packs)} pack version(s)")
    return 0


def list_packs(a: argparse.Namespace) -> int:
    c = cat.build(a.packs, a.model, official=not a.no_official, allow_namespaces=tuple(a.allow_namespace), refresh=a.refresh)
    print("sources: " + (", ".join(c.sources) or "none"))
    rows = [["pack", "version", "model", "status", "source"]]
    for pack_id, vs in c.packs.items():
        for v, p in vs.items():
            rows.append([pack_id, v + (" (default)" if c.get(pack_id).version == v else ""), p.model, p.status, p.source])
    print(_table(rows) if len(rows) > 1 else "no pack enabled" + (f" for {a.model}" if a.model else ""))
    if c.skipped:
        print("\nskipped:")
        for sk in c.skipped:
            print(f"  {sk.id} {sk.version} ({sk.source}): {sk.reason}")
    if c.official_error:
        print(f"\nofficial repository not available: {c.official_error}", file=sys.stderr)
    if c.errors:
        print(f"\n{len(c.errors)} error(s):", file=sys.stderr)
        for e in c.errors:
            print("  " + e, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spinf-serve packs", description="prompt packs: validate a repository, list what a server enables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check a pack repository or folder (exit 1 on any problem)")
    v.add_argument("folder")
    ls = sub.add_parser("list", help="the packs a server with these sources enables, and the skipped ones")
    ls.add_argument(
        "--packs", action="append", default=[], metavar="SOURCE", help="github:<owner>/<repo>[@<ref>][#<path>] or a folder; repeatable"
    )
    ls.add_argument("--model", default=None, help="the served model; without it every version is listed")
    ls.add_argument("--no-official", action="store_true", help="leave the official repository out")
    ls.add_argument("--allow-namespace", action="append", default=[], metavar="NS", help="serve this reserved namespace from any source")
    ls.add_argument("--refresh", action="store_true", help="resolve GitHub refs again")
    a = ap.parse_args(argv)
    return validate(a.folder) if a.cmd == "validate" else list_packs(a)


if __name__ == "__main__":
    sys.exit(main())
