"""Pack sources (plan §5.1): where the server's pack files come from. A source is one of

  github:<owner>/<repo>[@<ref>][#<path>]   a GitHub repository at a tag or commit (a branch works but is not reproducible: logged
                                           as a warning, resolved again at every start), fetched as the tarball of the commit
                                           (no git binary needed) and cached under $SPINF_PACKS_CACHE (default
                                           ~/.cache/spinf-serve/packs/<owner>/<repo>/<commit>/); a private repository uses
                                           GITHUB_TOKEN; `refresh` resolves the ref again
  <folder>                                 a local folder: a checkout, or a bare folder of pack files
  official                                 the official repository, spinfinc/spinf-prompts at the tag this release pins (OFFICIAL);
                                           SPINF_PACKS_OFFICIAL overrides the spec (a fork, a private mirror, a checkout)

The code never names a private repository: the override is the environment's."""

import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("spinf_serve.packs")

OFFICIAL_REPO = "spinfinc/spinf-prompts"
OFFICIAL_TAG = "0.1.0"  # the spinf-prompts release this spinf-serve bundles; moves with a release of either
OFFICIAL = f"github:{OFFICIAL_REPO}@{OFFICIAL_TAG}"
GITHUB_API = "https://api.github.com"
_GITHUB = re.compile(r"^github:(?P<owner>[A-Za-z0-9][A-Za-z0-9-]*)/(?P<repo>[A-Za-z0-9._-]+)(?:@(?P<ref>[^#]+))?(?:#(?P<path>.+))?$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
Http = Callable[[str, dict[str, str]], bytes]


class SourceError(RuntimeError):
    """a source that cannot be read: a bad spec, a repository or ref that does not exist, a folder that is missing"""


@dataclass(frozen=True)
class Source:
    spec: str  # as given
    kind: str  # github | local
    owner: str = ""
    repo: str = ""
    ref: str = ""  # github: a tag, a commit or a branch; "" = the default branch
    path: str = ""  # github: the folder inside the repository (#path)
    folder: str = ""  # local

    @property
    def official(self) -> bool:
        return self.kind == "github" and f"{self.owner}/{self.repo}".lower() == OFFICIAL_REPO

    def label(self, commit: str = "") -> str:
        """how listings name the source: owner/repo@ref, local:<folder>"""
        if self.kind == "local":
            return f"local:{self.folder}"
        ref = self.ref or commit[:12] or "HEAD"
        return f"{self.owner}/{self.repo}@{ref}"


def official_spec() -> str:
    return os.environ.get("SPINF_PACKS_OFFICIAL") or OFFICIAL


def parse(spec: str) -> Source:
    spec = spec.strip()
    if spec == "official":
        spec = official_spec()
    if m := _GITHUB.match(spec):
        return Source(spec, "github", m["owner"], m["repo"], m["ref"] or "", (m["path"] or "").strip("/"))
    if spec.startswith("github:"):
        raise SourceError(f"{spec!r}: a GitHub source is github:<owner>/<repo>[@<ref>][#<path>]")
    return Source(spec, "local", folder=os.path.expanduser(spec))


def cache_dir() -> str:
    if d := os.environ.get("SPINF_PACKS_CACHE"):
        return d
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "spinf-serve", "packs")


def _urlopen(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class GitHub:
    """the few GitHub API calls a source needs; `http(url, headers) -> bytes` is replaceable (tests)"""

    def __init__(self, http: Http | None = None, token: str | None = None):
        self.http = http or _urlopen
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")

    def _get(self, url: str, accept: str = "application/vnd.github+json") -> bytes:
        headers = {"Accept": accept, "User-Agent": "spinf-serve", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            return self.http(url, headers)
        except urllib.error.HTTPError as e:
            raise SourceError(
                f"GitHub {url}: HTTP {e.code}" + (" (a private repository needs GITHUB_TOKEN)" if e.code in (401, 403, 404) else "")
            ) from None
        except OSError as e:
            raise SourceError(f"GitHub {url}: {e}") from None

    def kind_of_ref(self, owner: str, repo: str, ref: str) -> str:
        """tag | branch | commit"""
        if _SHA.match(ref):
            return "commit"
        for kind, path in (("tag", "tags"), ("branch", "heads")):
            try:
                self._get(f"{GITHUB_API}/repos/{owner}/{repo}/git/ref/{path}/{ref}")
                return kind
            except SourceError as e:
                if "HTTP 404" not in str(e):
                    raise
        raise SourceError(f"github:{owner}/{repo}@{ref}: no such tag, branch or commit")

    def commit_of(self, owner: str, repo: str, ref: str) -> str:
        """the commit a ref points at (an annotated tag is followed to its commit)"""
        if _SHA.match(ref):
            return ref
        sha = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/commits/{ref or 'HEAD'}", "application/vnd.github.sha").decode().strip()
        if not _SHA.match(sha):
            raise SourceError(f"github:{owner}/{repo}@{ref}: the API did not return a commit")
        return sha

    def tarball(self, owner: str, repo: str, commit: str) -> bytes:
        # the default Accept: GitHub answers 415 to application/octet-stream here and 302 to codeload otherwise (urllib follows)
        return self._get(f"{GITHUB_API}/repos/{owner}/{repo}/tarball/{commit}")


def _extract(tar_bytes: bytes, dest: str) -> None:
    """the tarball's single top-level directory extracted as `dest` (atomically: a temp dir renamed into place)"""
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".fetch-", dir=parent)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            tf.extractall(tmp, filter="data")  # no links, no absolute paths, no escapes
        tops = os.listdir(tmp)
        root = os.path.join(tmp, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp, tops[0])) else tmp
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(root, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(src: Source, cache: str | None = None, refresh: bool = False, github: GitHub | None = None) -> tuple[str, str]:
    """the folder of a source on disk -> (folder, label). A GitHub source is resolved to a commit (cached per ref under
    <cache>/<owner>/<repo>/refs.json unless `refresh`, never for a branch) and fetched once per commit."""
    if src.kind == "local":
        if not os.path.isdir(src.folder):
            raise SourceError(f"{src.spec!r}: not a folder")
        return src.folder, src.label()
    cache = cache or cache_dir()
    gh = github or GitHub()
    repo_dir = os.path.join(cache, src.owner, src.repo)
    refs_file = os.path.join(repo_dir, "refs.json")
    refs: dict[str, str] = {}
    if os.path.exists(refs_file):
        with open(refs_file) as f:
            refs = json.load(f)
    ref = src.ref
    if _SHA.match(ref):
        commit = ref
    elif ref and ref in refs and not refresh:
        commit = refs[ref]
    else:
        kind = gh.kind_of_ref(src.owner, src.repo, ref) if ref else "branch"
        commit = gh.commit_of(src.owner, src.repo, ref)
        if kind == "branch":
            log.warning("%s: %s is a branch, not reproducible (commit %s); pin a tag", src.spec, ref or "the default branch", commit[:12])
        else:
            refs[ref] = commit
            os.makedirs(repo_dir, exist_ok=True)
            with open(refs_file, "w") as f:
                json.dump(refs, f, indent=1)
    dest = os.path.join(repo_dir, commit)
    if not os.path.isdir(dest):
        log.info("%s: fetching %s/%s@%s", src.spec, src.owner, src.repo, commit[:12])
        _extract(gh.tarball(src.owner, src.repo, commit), dest)
    folder = os.path.join(dest, src.path) if src.path else dest
    if not os.path.isdir(folder):
        raise SourceError(f"{src.spec!r}: no folder {src.path!r} in the repository")
    return folder, src.label(commit)
