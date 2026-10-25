import pytest

from spinf_serve.config import EngineConfig


def test_from_env(monkeypatch):
    monkeypatch.setenv("SPINF_QUANT", "w8a16")
    monkeypatch.setenv("SPINF_TOKEN_BUDGET", "8192")
    monkeypatch.setenv("SPINF_COMPILE", "0")
    cfg = EngineConfig.from_env()
    assert (cfg.quant, cfg.token_budget, cfg.compile) == ("w8a16", 8192, False)


def test_bad_quant():
    with pytest.raises(ValueError):
        EngineConfig(quant="int4")


def test_from_env_types(monkeypatch):
    from spinf_serve.config import EngineConfig

    monkeypatch.setenv("SPINF_DECODE_BUDGET_GIB", "6.5")
    monkeypatch.setenv("SPINF_MAX_TOKENS", "32768")
    monkeypatch.setenv("SPINF_X_CUDA_GRAPHS", "true")
    c = EngineConfig.from_env()
    assert c.decode_budget_gib == 6.5 and c.max_tokens == 32768 and c.x_cuda_graphs is True
