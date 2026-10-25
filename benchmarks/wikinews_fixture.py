"""Build the public parity fixture and benchmark job set from Wikinews articles (CC BY 2.5).

    python benchmarks/wikinews_fixture.py --out fixtures/wikinews_20261024.json.gz [--articles 24] [--seed 20261024]

Wikinews articles are published under the Creative Commons Attribution 2.5 licence, so their text can ship in this repository
with attribution: the fixture records every article's title, URL and revision (`sources`), and fixtures/README.md credits
Wikinews. Articles are drawn at random (`list=random`, the seed only names the day: the API's random order is not seeded)
and kept when they are plain prose of 250-450 words after cleaning (the dateline, the "Sources" / "Related news" / "Sister
links" sections and the Wikinews boilerplate removed).

Cells: every article gets the same 56 two-option questions, in five template shapes, so a parity run exercises long shared
stems, different option pairs and tries of several depths; the shapes follow the engine's own protocol (a template read after the
content, two single-token answers). The fixture format is spinf_serve.parity's: {"day", "model", "reference", "license",
"sources", "jobs": [{"id", "text", "cells": [{"cell", "template", "phrase", "pair", "lp_pos", "lp_neg"}]}]}. The lp_pos / lp_neg
written here are placeholders (0.0): `spinf-serve parity --record` fills them with a run of the fixture alone.
`benchmarks/compare.py` reads the same file as its job set.
"""

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request

API = "https://en.wikinews.org/w/api.php"
UA = "spinf-serve fixture builder (https://github.com/spinfinc/spinf-serve)"

TEMPLATES = {
    # key: (template, pair, phrases)
    "effect": (
        "{X}\n\nAs a result, {F} is likely to",
        ["rise", "fall"],
        [
            "the price of oil",
            "inflation",
            "the stock market",
            "unemployment",
            "interest rates",
            "the euro",
            "the dollar",
            "consumer confidence",
            "government spending",
            "public trust in the authorities",
            "the number of people affected",
            "tension in the region",
        ],
    ),
    "topic": (
        "{X}\n\nQuestion: Is this article mainly about {F}?\nAnswer:",
        ["Yes", "No"],
        [
            "politics",
            "business",
            "sport",
            "science",
            "health",
            "crime",
            "the weather",
            "technology",
            "culture",
            "a war",
            "an election",
            "the environment",
            "a natural disaster",
            "a court case",
            "transport",
            "education",
        ],
    ),
    "mentions": (
        "{X}\n\nDoes the article mention {F}? Answer yes or no.\nAnswer:",
        ["yes", "no"],
        [
            "a government",
            "a company",
            "a death",
            "an amount of money",
            "a date",
            "a quotation",
            "a court",
            "a scientist",
            "a protest",
            "children",
            "the police",
            "a vote",
        ],
    ),
    "tone": (
        "{X}\n\nThe tone of this article towards {F} is",
        ["positive", "negative"],
        [
            "its main subject",
            "the government",
            "the people involved",
            "the outcome",
            "the organisation named first",
            "the future",
            "the person quoted",
            "the decision described",
        ],
    ),
    "statement": (
        "{X}\n\nStatement: {F}\nIs the statement true according to the article above?\nAnswer:",
        ["True", "False"],
        [
            "The events took place in Europe.",
            "Someone was injured or killed.",
            "The article reports on an election.",
            "The article quotes an official.",
            "The article is about a sporting event.",
            "A company is named in the article.",
            "The article reports scientific findings.",
            "The article concerns the United States.",
        ],
    ),
}

CUT_SECTIONS = ("== Sources ==", "== Source ==", "== Related news ==", "== Sister links ==", "== External links ==", "== See also ==")
BOILERPLATE = re.compile(
    r"^(Have an opinion on this story\?.*|Share it!.*|This article (features|is based on|mentions).*|Wikinews .*|"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), [A-Z][a-z]+ \d{1,2}, \d{4}\s*)$"
)


def api(**params) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def clean(extract: str) -> str:
    text = extract
    for marker in CUT_SECTIONS:
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or BOILERPLATE.match(line) or line.startswith("=="):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def plain_prose(text: str) -> bool:
    words = text.split()
    if not 250 <= len(words) <= 450:
        return False
    if "{{" in text or "[[" in text or "|" in text or "File:" in text:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text)) > 0.75


def fetch_candidates(n: int) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        rnd = api(action="query", list="random", rnnamespace=0, rnlimit=20)["query"]["random"]
        ids = [str(p["id"]) for p in rnd if p["id"] not in seen]
        seen.update(p["id"] for p in rnd)
        if not ids:
            continue
        pages = api(
            action="query",
            prop="extracts|revisions|info",
            explaintext=1,
            exsectionformat="wiki",
            rvprop="ids",
            inprop="url",
            pageids="|".join(ids),
        )["query"]["pages"]
        for p in pages:
            text = clean(p.get("extract", ""))
            if not plain_prose(text) or "/" in p["title"]:
                continue
            out.append(
                {
                    "title": p["title"],
                    "url": p["fullurl"],
                    "revision": p["revisions"][0]["revid"],
                    "text": text,
                }
            )
            print(f"  kept {len(out):2d}: {p['title']} ({len(text.split())} words)")
            if len(out) == n:
                break
        time.sleep(0.5)
    return out


def cells() -> list[dict]:
    out = []
    for key, (template, pair, phrases) in TEMPLATES.items():
        for phrase in phrases:
            slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
            out.append({"cell": f"{key}:{slug}", "template": template, "phrase": phrase, "pair": pair, "lp_pos": 0.0, "lp_neg": 0.0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=24)
    ap.add_argument("--day", default=time.strftime("%Y%m%d"))
    a = ap.parse_args()
    arts = fetch_candidates(a.articles)
    arts.sort(key=lambda x: x["title"])
    jobs = []
    for art in arts:
        jid = hashlib.sha1(art["url"].encode()).hexdigest()[:16]
        jobs.append({"id": jid, "text": art["text"], "cells": cells()})
    fx = {
        "day": a.day,
        "model": "spinf/gemma-4-12b-fp8",
        "reference": "placeholders: record with spinf-serve parity --record",
        "license": "Text: Wikinews contributors, CC BY 2.5 (https://creativecommons.org/licenses/by/2.5/); questions: Apache 2.0",
        "sources": [
            {"id": j["id"], "title": s["title"], "url": s["url"], "revision": s["revision"]} for j, s in zip(jobs, arts, strict=True)
        ],
        "jobs": jobs,
    }
    with gzip.open(a.out, "wt") as f:
        json.dump(fx, f, ensure_ascii=False)
    n_cells = sum(len(j["cells"]) for j in jobs)
    words = sum(len(j["text"].split()) for j in jobs)
    print(f"wrote {a.out}: {len(jobs)} articles ({words} words), {n_cells} cells")


if __name__ == "__main__":
    main()
