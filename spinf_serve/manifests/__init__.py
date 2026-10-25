"""Model manifests: what a served model is (plan §3). One TOML file per model in this directory (the built-in catalog), plus any
in the directories of SPINF_MANIFEST_DIRS (colon-separated; a file with the same id overrides the built-in one).

A manifest pins the checkpoint (repo + revision) so the weights, the quantization parameters they carry and the engine's
parity references always name the same revision; it lists the model's aliases, family, allowed schemes, head, capabilities,
media and the default limits. Hardware and numerics stay in EngineConfig; EngineConfig.from_env applies a manifest when
SPINF_MODEL names one (config.py).
"""

import glob
import os
import re
import tomllib
from functools import cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator

BUILTIN_DIR = os.path.dirname(__file__)
_SHA = re.compile(r"^[0-9a-f]{40}$")


class Source(BaseModel):
    repo: str
    # the Hub commit hash; required for an available model (an unpinned download could fetch weights the engine was never checked on)
    revision: str = ""
    format: Literal["safetensors-bf16", "fp8-static", "gptq-int4-g32"]


class Quant(BaseModel):
    default: str = "auto"  # a scheme name, or auto: the engine picks for the card (models/gemma4.py quant_scheme / the Qwen module default)
    allowed: list[str]


class ImageSpec(BaseModel):
    scheme: Literal["gemma", "qwen"]
    side: int  # pixels per image token (one token = side x side pixels)
    detail_tokens: dict[str, int]  # token budget per detail: low / auto / high


class Limits(BaseModel):
    token_budget: int = 16384
    max_tokens: int = 65536


class Manifest(BaseModel):
    # the served name is the Hub repo of the published checkpoint (spinf/gemma-4-12b-fp8); an arbitrary checkpoint is served by its path
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
    name: str
    aliases: list[str] = Field(default_factory=list)
    status: Literal["available", "planned"] = "available"
    family: Literal["gemma4", "qwen35"]
    source: Source
    quant: Quant
    head: Literal["lm_head", "readout"] = "lm_head"
    capabilities: list[Literal["score", "extract", "generate"]] = Field(default_factory=lambda: ["score", "extract", "generate"])
    prompt_style: Literal["default", "decider"] = "default"
    media: list[Literal["image", "audio", "video"]] = Field(default_factory=list)
    image: ImageSpec | None = None
    limits: Limits = Field(default_factory=Limits)
    path: str = ""  # the file it was read from (set by load_file)

    @model_validator(mode="after")
    def _consistent(self) -> "Manifest":
        if self.status == "available" and not _SHA.match(self.source.revision):
            raise ValueError(f"{self.id}: an available model pins source.revision to a 40-hex commit hash")
        if self.source.revision and not _SHA.match(self.source.revision):
            raise ValueError(f"{self.id}: source.revision is not a commit hash")
        if self.head == "readout" and ("extract" in self.capabilities or "generate" in self.capabilities):
            raise ValueError(f"{self.id}: a readout model has no LM head: capabilities are [score] only")
        if ("image" in self.media or "video" in self.media) and self.image is None:
            raise ValueError(f"{self.id}: image or video media need an [image] spec")
        if self.quant.default != "auto" and self.quant.default not in self.quant.allowed:
            raise ValueError(f"{self.id}: quant.default {self.quant.default!r} is not in quant.allowed")
        return self

    @property
    def names(self) -> set[str]:
        """every name the model answers to, lower-cased: its id and its aliases"""
        return {self.id.lower(), *(a.lower() for a in self.aliases)}

    def engine_defaults(self) -> dict:
        """the EngineConfig fields a manifest sets (env and explicit values override them)"""
        return {
            "model": self.id,
            "served_model": self.id,
            "model_id": self.source.repo,
            "revision": self.source.revision,
            "quant": "" if self.quant.default == "auto" else self.quant.default,
            "media": bool(self.media),
            "token_budget": self.limits.token_budget,
            "max_tokens": self.limits.max_tokens,
        }


def load_file(path: str) -> Manifest:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Manifest.model_validate({**data, "path": path})


def manifest_dirs() -> list[str]:
    extra = [d for d in os.environ.get("SPINF_MANIFEST_DIRS", "").split(":") if d]
    return [BUILTIN_DIR, *extra]


@cache
def _catalog(dirs: tuple[str, ...]) -> dict[str, Manifest]:
    out: dict[str, Manifest] = {}
    for d in dirs:  # later directories override earlier ones (the built-ins come first)
        for path in sorted(glob.glob(os.path.join(d, "*.toml"))):
            m = load_file(path)
            out[m.id] = m
    seen: dict[str, str] = {}
    for m in out.values():
        for n in m.names:
            if n in seen and seen[n] != m.id:
                raise ValueError(f"manifests: {n!r} names both {seen[n]} and {m.id}")
            seen[n] = m.id
    return out


def catalog(dirs: list[str] | None = None) -> dict[str, Manifest]:
    """id -> Manifest over the built-in directory and SPINF_MANIFEST_DIRS (or the given directories)"""
    return _catalog(tuple(dirs if dirs is not None else manifest_dirs()))


def resolve(name: str, dirs: list[str] | None = None) -> Manifest:
    """the manifest a model name refers to (its id or an alias, case-insensitive), or KeyError"""
    cat = catalog(dirs)
    low = name.lower()
    for m in cat.values():
        if low in m.names:
            return m
    raise KeyError(f"no manifest for model {name!r}; known: {', '.join(sorted(cat))}")
