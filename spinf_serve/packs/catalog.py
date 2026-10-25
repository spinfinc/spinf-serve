"""The packs a server serves (plan §5.1, §5.2): the pack files of every source, read through the repo manifest `prompt-packs.json`
(format 1: namespace, licence, and per pack its name, description, tags, media, readme, samples and versions with file, model,
status and fixture) or, without one, discovered as the JSON files that carry a `pack` and a `scoring` field. Then, per server:
keep the versions whose `model` is the served model (skip the others with one log line each: a 31B pack on a 12B server is not
an error), refuse an id two sources both provide (no silent shadowing), and keep the `spinf/` namespace for the official source
unless a namespace is explicitly allowed (a locally edited copy of an official pack)."""

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from spinf_serve import manifests
from spinf_serve.packs import ID, OFFICIAL_NAMESPACE, Pack, PackError, load
from spinf_serve.packs.sources import Source, SourceError, fetch, parse

log = logging.getLogger("spinf_serve.packs")
MANIFEST = "prompt-packs.json"
NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RepoVersion(BaseModel):
    version: str
    file: str
    model: str
    status: Literal["available", "deprecated", "coming_soon"] = "available"
    fixture: str | None = None
    note: str | None = None


class RepoPack(BaseModel):
    id: str = Field(pattern=ID.pattern)
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    media: list[Literal["text", "image", "audio", "video"]] = Field(default_factory=lambda: ["text"])
    parameters: list[str] = Field(default_factory=list)
    readme: str | None = None
    samples: str | None = None
    versions: list[RepoVersion] = Field(min_length=1)


class RepoManifest(BaseModel):
    format: Literal[1]
    namespace: str = Field(pattern=NAMESPACE.pattern)
    name: str = ""
    homepage: str = ""
    license: str = ""
    packs: list[RepoPack]


@dataclass
class Scan:
    """one source read: its packs (every version, whatever the model), its listing entries and the problems found"""

    label: str
    folder: str
    manifest: RepoManifest | None
    packs: list[Pack] = field(default_factory=list)
    listed: dict[str, RepoPack] = field(default_factory=dict)  # from the manifest: every id, served or not
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _version_key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v))


def read_manifest(folder: str) -> RepoManifest | None:
    """the repo manifest of a folder, None when there is none; a bad one raises PackError"""
    path = os.path.join(folder, MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return RepoManifest.model_validate(data)
    except ValueError as e:
        msgs = (
            [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()] if isinstance(e, ValidationError) else [str(e)]
        )
        raise PackError(path, msgs) from None


def scan(folder: str, label: str = "") -> Scan:
    """every pack file of a folder, through its manifest or by discovery; problems are collected, not raised"""
    label = label or f"local:{folder}"
    try:
        manifest = read_manifest(folder)
    except PackError as e:
        return Scan(label, folder, None, problems=[str(e)])
    out = Scan(label, folder, manifest)
    if manifest is None:
        for path in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
            try:
                with open(path, encoding="utf-8") as f:
                    head = f.read(4096)
            except OSError:
                continue
            if '"pack"' not in head:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except ValueError:
                continue
            if not isinstance(data, dict) or "pack" not in data or "scoring" not in data:
                continue  # a fixture or a sample file
            try:
                out.packs.append(load(path, source=label))
            except PackError as e:
                out.problems.append(str(e))
        return out
    seen: set[tuple[str, str]] = set()
    for entry in manifest.packs:
        if not entry.id.startswith(manifest.namespace + "/"):
            out.problems.append(f"{MANIFEST}: {entry.id} is not in the namespace {manifest.namespace!r}")
            continue
        if entry.id in out.listed:
            out.problems.append(f"{MANIFEST}: {entry.id} is listed twice")
            continue
        out.listed[entry.id] = entry
        for rel in (entry.readme, entry.samples):
            if rel and not os.path.exists(os.path.join(folder, rel)):
                out.problems.append(f"{MANIFEST}: {entry.id}: {rel} is missing")
        for v in entry.versions:
            if (entry.id, v.version) in seen:
                out.problems.append(f"{MANIFEST}: {entry.id} {v.version} is listed twice")
                continue
            seen.add((entry.id, v.version))
            path = os.path.join(folder, v.file)
            if not os.path.exists(path):
                if v.status != "coming_soon":
                    out.problems.append(f"{MANIFEST}: {entry.id} {v.version}: {v.file} is missing")
                continue
            if v.fixture and not os.path.exists(os.path.join(folder, v.fixture)):
                out.problems.append(f"{MANIFEST}: {entry.id} {v.version}: fixture {v.fixture} is missing")
            try:
                p = load(
                    path,
                    source=label,
                    status=v.status,
                    name=entry.name,
                    description=entry.description,
                    tags=tuple(entry.tags),
                    listing={
                        k: val
                        for k, val in (("readme", entry.readme), ("samples", entry.samples), ("fixture", v.fixture), ("note", v.note))
                        if val
                    },
                )
            except PackError as e:
                out.problems.append(str(e))
                continue
            if (p.id, p.version, p.model) != (entry.id, v.version, v.model):
                out.problems.append(f"{MANIFEST}: {entry.id} {v.version} ({v.model}): the file says {p.id} {p.version} ({p.model})")
                continue
            if set(p.media) != set(entry.media):
                out.problems.append(f"{MANIFEST}: {entry.id}: media {entry.media} but {v.file} says {list(p.media)}")
            if sorted(p.parameters) != sorted(entry.parameters):
                out.problems.append(f"{MANIFEST}: {entry.id}: parameters {entry.parameters} but {v.file} says {sorted(p.parameters)}")
            out.packs.append(p)
    return out


def same_model(a: str, b: str) -> bool:
    """two model names for one model: equal, or the same manifest (an alias)"""
    if a.lower() == b.lower():
        return True
    try:
        return manifests.resolve(a).id == manifests.resolve(b).id
    except KeyError:
        return False


@dataclass(frozen=True)
class Skipped:
    id: str
    version: str
    source: str
    reason: str


@dataclass
class Catalog:
    """the packs enabled for one server: id -> version -> Pack, the default version first"""

    model: str | None = None
    packs: dict[str, dict[str, Pack]] = field(default_factory=dict)
    listed: dict[str, RepoPack] = field(default_factory=dict)  # every id a manifest lists, enabled or not
    skipped: list[Skipped] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # the labels, in order
    # the official source could not be read (offline, a tag not published yet): the server starts without it (build)
    official_error: str | None = None

    def versions(self, pack_id: str) -> list[str]:
        return list(self.packs.get(pack_id, {}))

    def get(self, pack_id: str, version: str | None = None) -> Pack:
        """a pack by id and version (default: the first available version listed)"""
        vs = self.packs.get(pack_id)
        if not vs:
            raise KeyError(pack_id)
        if version is None:
            version = next((v for v, p in vs.items() if p.status == "available"), next(iter(vs)))
        return vs[version]

    def public(self) -> list[dict[str, Any]]:
        """the GET /v1/packs listing: one entry per enabled pack (its default version) and per pack listed as coming soon"""
        out = []
        for pack_id, vs in self.packs.items():
            out.append(self.get(pack_id).public(list(vs)))
        for pack_id, entry in self.listed.items():
            if pack_id not in self.packs and any(v.status == "coming_soon" for v in entry.versions):
                models = sorted({v.model for v in entry.versions})
                out.append(
                    {
                        "id": pack_id,
                        "name": entry.name,
                        "status": "coming_soon",
                        "version": None,
                        "versions": [],
                        "description": entry.description,
                        "tags": entry.tags,
                        "model": models[0] if len(models) == 1 else models,
                        "media": entry.media,
                    }
                )
        return out

    def add(self, s: Scan, allow_namespaces: tuple[str, ...] = (), official: bool = False) -> None:
        """a source's scan filtered for this server"""
        self.sources.append(s.label)
        self.errors.extend(s.problems)
        self.listed.update(s.listed)
        for p in s.packs:
            if p.namespace == OFFICIAL_NAMESPACE and not official and OFFICIAL_NAMESPACE not in allow_namespaces:
                self.skipped.append(
                    Skipped(
                        p.id,
                        p.version,
                        s.label,
                        f"the {OFFICIAL_NAMESPACE}/ namespace is reserved for the official source "
                        f"(--packs-allow-namespace {OFFICIAL_NAMESPACE} to serve it)",
                    )
                )
                continue
            if self.model is not None and not same_model(p.model, self.model):
                self.skipped.append(Skipped(p.id, p.version, s.label, f"written for {p.model}, this server serves {self.model}"))
                continue
            have = self.packs.get(p.id)
            if have and next(iter(have.values())).source != s.label:
                self.errors.append(
                    f"{p.id} comes from both {next(iter(have.values())).source} and {s.label}: packs of one id come from one source"
                )
                continue
            if have and p.version in have:
                self.errors.append(f"{p.id} {p.version} appears twice in {s.label}")
                continue
            self.packs.setdefault(p.id, {})[p.version] = p
        for pack_id, vs in self.packs.items():
            if pack_id not in s.listed and len(vs) > 1:  # discovered without a manifest: newest first
                self.packs[pack_id] = dict(sorted(vs.items(), key=lambda kv: _version_key(kv[0]), reverse=True))

    def log_summary(self) -> None:
        for sk in self.skipped:
            log.info("pack %s %s (%s) skipped: %s", sk.id, sk.version, sk.source, sk.reason)
        for e in self.errors:
            log.error("packs: %s", e)
        if self.official_error:
            log.error("packs: the official repository is not available, serving without it: %s", self.official_error)
        n = sum(len(v) for v in self.packs.values())
        log.info("packs: %d enabled (%d ids) from %s", n, len(self.packs), ", ".join(self.sources) or "no source")


def build(
    specs: list[str | Source],
    model: str | None = None,
    official: bool = True,
    allow_namespaces: tuple[str, ...] = (),
    cache: str | None = None,
    refresh: bool = False,
    github=None,
) -> Catalog:
    """the catalog of a server: the official source first (unless `official` is off), then the given sources in order; `model`
    None keeps every version (`packs list` without a model). Source errors (a repository or folder that cannot be read) are
    catalog errors, not exceptions: the server refuses to start on an error of a source it was given, `packs list` prints them.
    The official source failing (offline, the pinned tag not published) is `official_error`: logged, the server starts without
    the official packs (a user who wants them anyway points SPINF_PACKS_OFFICIAL or --packs at a checkout)."""
    cat = Catalog(model)
    srcs = [(parse("official"), True)] * official + [(s if isinstance(s, Source) else parse(s), False) for s in specs]
    seen: set[str] = set()
    for src, is_official in srcs:  # the official slot may carry spinf/ wherever SPINF_PACKS_OFFICIAL points it
        if src.spec in seen:
            continue
        seen.add(src.spec)
        try:
            folder, label = fetch(src, cache, refresh, github)
        except SourceError as e:
            if is_official:
                cat.official_error = str(e)
            else:
                cat.errors.append(str(e))
                cat.sources.append(src.label())
            continue
        cat.add(scan(folder, label), allow_namespaces, official=is_official or src.official)
    return cat
