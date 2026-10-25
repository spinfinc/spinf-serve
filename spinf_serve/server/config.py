"""The server's settings (plan §3.2): one ServerConfig in three groups, layered command-line flags > SPINF_* environment > a config
file (--config server.toml) > the model's manifest defaults > the hosted service's defaults.

  engine    hardware and numerics: EngineConfig (config.py), the device
  features  what requests may ask for: media kinds, extraction, generation, the pack sources
  limits    the request `limits` block as server-wide defaults (protocol.Limits), the body size, the queue

Plus the listener (host, port), the tokens (access and trusted) and --debug. A trusted caller (the hosted API's front) may send
`limits` and `features` per call, each capped by the server's.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from spinf_serve.config import EngineConfig
from spinf_serve.protocol import Limits

MEDIA_KINDS = ("image", "audio", "video")
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Features:
    """what the server lets a request ask for; `media` None = every kind the model reads"""

    media: tuple[str, ...] | None = None
    extraction: bool = True
    generation: bool = True

    def media_for(self, model_media: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """the media kinds enabled: the server's setting within what the model reads"""
        kinds = tuple(model_media)
        return kinds if self.media is None else tuple(k for k in kinds if k in self.media)

    def capped(self, by: "Features", model_media: tuple[str, ...] | list[str]) -> "Features":
        """a trusted caller's features, none beyond the server's"""
        allowed = by.media_for(model_media)
        extraction = self.extraction and by.extraction
        return Features(
            tuple(k for k in (self.media if self.media is not None else allowed) if k in allowed),
            extraction,
            extraction and self.generation and by.generation,  # generation is a kind of extraction slot
        )


@dataclass(frozen=True)
class ServerConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    # the access token: when set, every request carries `Authorization: Bearer <token>`; without it the server is open
    token: str = ""
    # the trusted callers' token (the hosted API's front, the RunPod handler: today's SPINF_ENGINE_TOKEN): a request with it may send
    # `limits` and `features`, gets the engine-internal fields and the compact `rejected` error form. It is also an access token.
    # Without a trusted token, the holders of the access token are trusted (a single-token setup is one's own front end)
    trusted_token: str = ""
    features: Features = field(default_factory=Features)
    # pack sources (--packs, repeatable): github:<owner>/<repo>[@ref][#path] or a folder; the official repository is on by default
    packs: tuple[str, ...] = ()
    official_packs: bool = True
    packs_allow_namespace: tuple[str, ...] = ()
    packs_refresh: bool = False
    limits: Limits = field(default_factory=Limits)
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    # requests accepted while one is on the GPU; beyond it 429 concurrency_limit with Retry-After
    queue: int = 4
    keepalive_s: int = 30
    # adds packed_tokens and the engine block to every response (engine internals, else reserved to trusted callers)
    debug: bool = False
    config_file: str = ""

    def trusted(self, bearer: str) -> bool:
        """whether a request's bearer token marks a trusted caller"""
        import hmac

        if self.trusted_token:
            return hmac.compare_digest(bearer, self.trusted_token)
        return bool(self.token) and hmac.compare_digest(bearer, self.token)

    def authorized(self, bearer: str) -> bool:
        """whether a request may be served at all (an open server: always)"""
        import hmac

        if not self.token and not self.trusted_token:
            return True
        return any(hmac.compare_digest(bearer, t) for t in (self.token, self.trusted_token) if t)

    @classmethod
    def build(cls, cli: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None, prefix: str = "SPINF_") -> "ServerConfig":
        """the layered configuration: `cli` is the parsed flags as {setting: value} (None = not given), `env` the environment (default
        os.environ). The settings' names are the dataclass fields, with `limits.*`, `features.*` and the engine's fields flattened:
        {"port": 8010, "max_content_tokens": 8192, "media_kinds": "image,video", "quant": "fp8s"}; `load_media` is the engine's
        embedder switch (--no-media)."""
        cli = {k: v for k, v in (cli or {}).items() if v is not None}
        env = os.environ if env is None else env
        file_path = cli.get("config_file") or env.get(prefix + "CONFIG") or ""
        file = _read_file(file_path) if file_path else {}

        def pick(name: str, env_name: str | None = None, cast=None):
            """cli > env > file > None"""
            if name in cli:
                return cli[name]
            v = env.get(prefix + (env_name or name.upper()))
            if v not in (None, ""):
                return _cast(v, cast)
            return file.get(name)

        # engine: the manifest's defaults, then the file's engine table, then env, then flags (EngineConfig.from_env does env > manifest)
        eng_kw: dict[str, Any] = {}
        for f in fields(EngineConfig):
            # the engine's `media` (load the embedders) is the setting `load_media`; the feature `media_kinds` is what requests may send
            v = pick("load_media", "MEDIA", cast=bool) if f.name == "media" else pick(f.name, cast=f.type)
            if v is not None:
                eng_kw[f.name] = v
        model = eng_kw.pop("model", None)
        engine = EngineConfig.from_manifest(str(model), **eng_kw) if model else EngineConfig(**eng_kw)

        lim_kw: dict[str, Any] = {}
        for name in Limits.model_fields:
            v = pick(name, cast=int)
            if v is not None:
                lim_kw[name] = v
        media = pick("media_kinds", cast=str)
        features = Features(
            _kinds(media) if media is not None else None,
            _bool(pick("extraction", cast=bool), True),
            _bool(pick("generation", cast=bool), True),
        )
        packs = pick("packs", cast=str)
        allow = pick("packs_allow_namespace", cast=str)
        return cls(
            engine=engine,
            device=pick("device", cast=str) or "cuda:0",
            host=pick("host", cast=str) or "0.0.0.0",
            port=int(pick("port", cast=int) or 8000),
            token=pick("token", cast=str) or "",
            trusted_token=pick("trusted_token", "ENGINE_TOKEN", cast=str) or "",
            features=features,
            packs=_list(packs),
            official_packs=_bool(pick("official_packs", cast=bool), True),
            packs_allow_namespace=_list(allow),
            packs_refresh=_bool(pick("packs_refresh", cast=bool), False),
            limits=Limits(**lim_kw),
            max_body_bytes=int(pick("max_body_bytes", cast=int) or DEFAULT_MAX_BODY_BYTES),
            queue=int(pick("queue", cast=int) if pick("queue", cast=int) is not None else 4),
            keepalive_s=int(pick("keepalive_s", cast=int) or 30),
            debug=_bool(pick("debug", cast=bool), False),
            config_file=file_path,
        )

    def manifest(self):
        """the served model's manifest, None for a checkpoint served by its path"""
        if not self.engine.model:
            return None
        from spinf_serve import manifests

        return manifests.resolve(self.engine.model)

    def model_media(self) -> tuple[str, ...]:
        """the media kinds the served model reads"""
        m = self.manifest()
        if m is not None:
            return tuple(m.media)
        return MEDIA_KINDS if self.engine.media else ()

    def capabilities(self) -> list[str]:
        """the model's capabilities within the features: score, extract, generate"""
        m = self.manifest()
        caps = list(m.capabilities) if m is not None else ["score", "extract", "generate"]
        if not self.features.extraction:
            caps = [c for c in caps if c not in ("extract", "generate")]
        if not self.features.generation:
            caps = [c for c in caps if c != "generate"]
        return caps


def _read_file(path: str) -> dict[str, Any]:
    """a TOML config file: top-level keys, and the tables [engine] [server] [features] [limits] flattened"""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    if isinstance(out.get("media_kinds"), list):
        out["media_kinds"] = ",".join(out["media_kinds"])
    for k in ("packs", "packs_allow_namespace"):
        if isinstance(out.get(k), list):
            out[k] = ",".join(out[k])
    return out


def _cast(v: str, cast) -> Any:
    if cast in (bool, "bool"):
        return v.lower() in ("1", "true", "yes", "on")
    if cast in (int, "int"):
        return int(v)
    if cast in (float, "float"):
        return float(v)
    return v


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


def _list(v: Any) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        return tuple(x.strip() for x in v.split(",") if x.strip())
    return tuple(v)


def _kinds(v: Any) -> tuple[str, ...]:
    kinds = _list(v)
    if kinds == ("none",) or kinds == ("",):
        return ()
    bad = [k for k in kinds if k not in MEDIA_KINDS]
    if bad:
        raise ValueError(f"media kinds: {bad[0]!r} is not one of {', '.join(MEDIA_KINDS)} (or none)")
    return kinds
