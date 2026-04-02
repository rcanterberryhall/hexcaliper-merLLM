"""Tests for config.py — apply_overrides and defaults."""
import importlib
import os
import sys


def _fresh_config(**env):
    """Reload config with a clean env."""
    for mod in list(sys.modules.keys()):
        if mod == "config" or mod.startswith("config."):
            del sys.modules[mod]
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        import config
        return config
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults():
    cfg = _fresh_config()
    assert cfg.OLLAMA_0_URL == "http://host.docker.internal:11434"
    assert cfg.OLLAMA_1_URL == "http://host.docker.internal:11435"
    assert cfg.DAY_MODEL_GPU0 == "qwen3:32b"
    assert cfg.DAY_MODEL_GPU1 == "qwen3:30b-a3b"
    assert cfg.NIGHT_MODEL == "qwen3:32b"
    assert cfg.NIGHT_NUM_CTX == 32768
    assert cfg.INACTIVITY_TIMEOUT_MIN == 90
    assert cfg.BASE_DAY_END_LOCAL == "22:00"
    assert cfg.OLLAMA_MANAGE_VIA == "none"
    assert cfg.DRAIN_TIMEOUT_SEC == 300
    assert cfg.METRICS_INTERVAL_SEC == 10


def test_env_override():
    cfg = _fresh_config(
        OLLAMA_0_URL="http://localhost:9999",
        DAY_MODEL_GPU0="qwen3:8b",
        INACTIVITY_TIMEOUT_MIN="60",
        NIGHT_NUM_CTX="16384",
    )
    assert cfg.OLLAMA_0_URL == "http://localhost:9999"
    assert cfg.DAY_MODEL_GPU0 == "qwen3:8b"
    assert cfg.INACTIVITY_TIMEOUT_MIN == 60
    assert cfg.NIGHT_NUM_CTX == 16384


def test_apply_overrides_str():
    cfg = _fresh_config()
    cfg.apply_overrides({"ollama_0_url": "http://new:1234", "day_model_gpu0": "llama3:8b"})
    assert cfg.OLLAMA_0_URL == "http://new:1234"
    assert cfg.DAY_MODEL_GPU0 == "llama3:8b"


def test_apply_overrides_int():
    cfg = _fresh_config()
    cfg.apply_overrides({"night_num_ctx": 8192, "inactivity_timeout_min": 45})
    assert cfg.NIGHT_NUM_CTX == 8192
    assert cfg.INACTIVITY_TIMEOUT_MIN == 45


def test_apply_overrides_ignores_unknown():
    cfg = _fresh_config()
    original_url = cfg.OLLAMA_0_URL
    cfg.apply_overrides({"unknown_key": "something", "another_key": 42})
    assert cfg.OLLAMA_0_URL == original_url


def test_apply_overrides_invalid_int():
    cfg = _fresh_config()
    original = cfg.NIGHT_NUM_CTX
    cfg.apply_overrides({"night_num_ctx": "not-a-number"})
    assert cfg.NIGHT_NUM_CTX == original


def test_apply_overrides_none_value():
    cfg = _fresh_config()
    original = cfg.OLLAMA_0_URL
    cfg.apply_overrides({"ollama_0_url": None})
    assert cfg.OLLAMA_0_URL == original
