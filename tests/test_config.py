"""Tests for config.py — apply_overrides and defaults."""
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
    assert cfg.DEFAULT_MODEL == "qwen3:32b"
    assert cfg.RECLAIM_TIMEOUT == 300
    assert cfg.HEALTH_BACKOFF_BASE == 10
    assert cfg.HEALTH_BACKOFF_CAP == 300
    assert cfg.HEALTH_FAULT_TIMEOUT == 1800
    assert cfg.METRICS_INTERVAL_SEC == 10


def test_env_override():
    cfg = _fresh_config(
        OLLAMA_0_URL="http://localhost:9999",
        DEFAULT_MODEL="qwen3:8b",
        RECLAIM_TIMEOUT="120",
        HEALTH_BACKOFF_BASE="5",
    )
    assert cfg.OLLAMA_0_URL == "http://localhost:9999"
    assert cfg.DEFAULT_MODEL == "qwen3:8b"
    assert cfg.RECLAIM_TIMEOUT == 120
    assert cfg.HEALTH_BACKOFF_BASE == 5


def test_apply_overrides_str():
    cfg = _fresh_config()
    cfg.apply_overrides({"ollama_0_url": "http://new:1234", "default_model": "llama3:8b"})
    assert cfg.OLLAMA_0_URL == "http://new:1234"
    assert cfg.DEFAULT_MODEL == "llama3:8b"


def test_apply_overrides_int():
    cfg = _fresh_config()
    cfg.apply_overrides({"reclaim_timeout": 120, "health_backoff_base": 5})
    assert cfg.RECLAIM_TIMEOUT == 120
    assert cfg.HEALTH_BACKOFF_BASE == 5


def test_apply_overrides_ignores_unknown():
    cfg = _fresh_config()
    original_url = cfg.OLLAMA_0_URL
    cfg.apply_overrides({"unknown_key": "something", "another_key": 42})
    assert cfg.OLLAMA_0_URL == original_url


def test_apply_overrides_invalid_int():
    cfg = _fresh_config()
    original = cfg.RECLAIM_TIMEOUT
    cfg.apply_overrides({"reclaim_timeout": "not-a-number"})
    assert cfg.RECLAIM_TIMEOUT == original


def test_apply_overrides_none_value():
    cfg = _fresh_config()
    original = cfg.OLLAMA_0_URL
    cfg.apply_overrides({"ollama_0_url": None})
    assert cfg.OLLAMA_0_URL == original
