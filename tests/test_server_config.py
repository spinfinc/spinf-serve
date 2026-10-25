"""ServerConfig: the layering (flags > environment > config file > manifest > defaults), the media kinds, the limits and features
capping, the tokens, the command line's flag parsing."""

import pytest

from spinf_serve.cli import add_server_flags, server_config
from spinf_serve.protocol import Limits
from spinf_serve.server.config import Features, ServerConfig


def test_defaults_and_manifest():
    cfg = ServerConfig.build({}, env={})
    assert cfg.port == 8000 and cfg.limits == Limits() and cfg.features == Features() and cfg.queue == 4 and cfg.official_packs
    assert (
        cfg.engine.served_model == "spinf/gemma-4-12b-fp8" and cfg.manifest() is None and cfg.model_media() == ("image", "audio", "video")
    )
    cfg = ServerConfig.build({"model": "spinf/gemma-4-31b-fp8"}, env={})
    assert cfg.engine.model_id == "spinf/gemma-4-31b-fp8" and cfg.engine.quant == "" and cfg.model_media() == ("image", "video")
    assert cfg.capabilities() == ["score", "extract", "generate"]
    cfg = ServerConfig.build({"model": "spinf/pplx-decider-v1-27b-int4"}, env={})
    assert cfg.capabilities() == ["score"] and cfg.model_media() == () and cfg.engine.max_tokens == 16384 and not cfg.engine.media


def test_layering(tmp_path):
    f = tmp_path / "server.toml"
    f.write_text(
        '[server]\nport = 9000\ntoken = "file"\n[engine]\nmodel = "spinf/gemma-4-12b-fp8"\nquant = "w8a16"\ntoken_budget = 8192\n'
        '[features]\nmedia_kinds = ["image", "video"]\nextraction = false\npacks = ["./a", "github:x/y@1"]\n'
        "[limits]\nmax_content_tokens = 8192\n"
    )
    env = {"SPINF_CONFIG": str(f), "SPINF_PORT": "9100", "SPINF_MAX_QUERY_TOKENS": "512", "SPINF_ENGINE_TOKEN": "fr0nt", "SPINF_MEDIA": "0"}
    cfg = ServerConfig.build({"port": 9200, "max_option_tokens": 8}, env=env)
    assert cfg.port == 9200  # flag over env over file
    assert cfg.token == "file" and cfg.trusted_token == "fr0nt" and cfg.config_file == str(f)
    assert cfg.engine.quant == "w8a16" and cfg.engine.token_budget == 8192 and cfg.engine.model == "spinf/gemma-4-12b-fp8"
    assert cfg.engine.media is False  # SPINF_MEDIA: the embedder switch
    assert cfg.features == Features(("image", "video"), False, True) and cfg.packs == ("./a", "github:x/y@1")
    assert cfg.limits.max_content_tokens == 8192 and cfg.limits.max_query_tokens == 512 and cfg.limits.max_option_tokens == 8
    assert cfg.limits.max_prompts_per_input == 10_000
    cfg = ServerConfig.build({"config_file": str(f), "media_kinds": "none", "official_packs": False}, env={})
    assert cfg.features.media == () and cfg.port == 9000 and not cfg.official_packs
    with pytest.raises(ValueError):
        ServerConfig.build({"media_kinds": "image,gif"}, env={})


def test_tokens():
    open_ = ServerConfig.build({}, env={})
    assert open_.authorized("") and not open_.trusted("") and not open_.trusted("x")
    one = ServerConfig.build({"token": "s"}, env={})
    assert one.authorized("s") and not one.authorized("") and one.trusted("s") and not one.trusted("t")
    two = ServerConfig.build({"token": "s", "trusted_token": "f"}, env={})
    assert two.authorized("s") and two.authorized("f") and not two.trusted("s") and two.trusted("f")
    only_trusted = ServerConfig.build({"trusted_token": "f"}, env={})
    assert only_trusted.authorized("f") and not only_trusted.authorized("") and only_trusted.trusted("f")


def test_capping():
    lim = Limits(max_content_tokens=100_000, max_query_tokens=64).capped(Limits())
    assert lim.max_content_tokens == 4096 and lim.max_query_tokens == 64 and lim.max_options == 1000
    f = Features(None, True, True).capped(Features(("image",), False, True), ("image", "audio"))
    assert f == Features(("image",), False, False)  # generation needs extraction
    assert Features(("video",)).media_for(("image", "video")) == ("video",) and Features().media_for(["audio"]) == ("audio",)


def test_command_line_flags():
    import argparse

    ap = argparse.ArgumentParser()
    add_server_flags(ap)
    a = ap.parse_args(
        [
            "spinf/gemma-4-12b-fp8",
            "--port",
            "8010",
            "--media",
            "image",
            "--no-extraction",
            "--packs",
            "./p",
            "--packs",
            "./q",
            "--max-content-tokens",
            "2048",
            "--quant",
            "fp8s",
            "--token",
            "t",
            "--queue",
            "2",
        ]
    )
    cfg = server_config(a)
    assert cfg.port == 8010 and cfg.features == Features(("image",), False, True) and cfg.packs == ("./p", "./q")
    assert cfg.limits.max_content_tokens == 2048 and cfg.engine.quant == "fp8s" and cfg.token == "t" and cfg.queue == 2
    assert cfg.engine.model == "spinf/gemma-4-12b-fp8" and cfg.engine.media is True
    a = ap.parse_args(["--no-media"])
    cfg = server_config(a)
    assert cfg.engine.media is False and cfg.features.media == ()
    a = ap.parse_args(["--media", "none"])
    assert server_config(a).engine.media is False
