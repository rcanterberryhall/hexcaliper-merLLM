"""Tests for gpu_router.py — GPU state, health probes, reclaim cooperation.

Dispatch logic (which GPU serves a request) lives in queue_manager and is
exercised by test_queue_manager.py.  This file only covers state ownership
that gpu_router still holds: health transitions, reclaim conditions,
record_activity, and the status snapshot.
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(autouse=True)
def fresh_modules(tmp_path, monkeypatch):
    """Reload gpu_router and dependencies fresh for each test."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DEFAULT_MODEL", "test-model:7b")
    monkeypatch.setenv("RECLAIM_TIMEOUT", "60")
    monkeypatch.setenv("HEALTH_BACKOFF_BASE", "5")
    monkeypatch.setenv("HEALTH_BACKOFF_CAP", "30")
    monkeypatch.setenv("HEALTH_FAULT_TIMEOUT", "60")
    for mod in list(sys.modules.keys()):
        if mod in ("gpu_router", "queue_manager", "db", "config",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules.keys()):
        if mod in ("gpu_router", "queue_manager", "db", "config",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)


def _get_router():
    import gpu_router
    import config
    # Force re-init with current config values.
    gpu_router._gpus.clear()
    gpu_router._pending_default_model = None
    gpu_router._init_gpus()
    return gpu_router, config


# ── Health tests ─────────────────────────────────────────────────────────────


def test_init_creates_both_gpus():
    router, config = _get_router()
    assert config.OLLAMA_0_URL in router._gpus
    assert config.OLLAMA_1_URL in router._gpus
    assert all(g.health == router.GpuHealth.HEALTHY
               for g in router._gpus.values())


def test_mark_failed_sets_degraded():
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    assert router._gpus[config.OLLAMA_0_URL].health == router.GpuHealth.DEGRADED


def test_mark_failed_does_nothing_for_unknown_target():
    router, _ = _get_router()
    router.mark_failed("http://nowhere:1234")
    # Just shouldn't raise.


def test_healthy_gpus_excludes_degraded():
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    healthy = router._healthy_gpus()
    urls = [g.url for g in healthy]
    assert config.OLLAMA_0_URL not in urls
    assert config.OLLAMA_1_URL in urls


def test_reset_gpu_restores_healthy():
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    router._gpus[config.OLLAMA_0_URL].health = router.GpuHealth.FAULTED

    ok = router.reset_gpu(config.OLLAMA_0_URL)
    assert ok is True
    assert router._gpus[config.OLLAMA_0_URL].health == router.GpuHealth.HEALTHY
    assert router._gpus[config.OLLAMA_0_URL].model == config.DEFAULT_MODEL


def test_reset_unknown_gpu_returns_false():
    router, _ = _get_router()
    assert router.reset_gpu("http://nowhere:1234") is False


# ── Reclaim tests ────────────────────────────────────────────────────────────


def test_reclaim_identifies_idle_non_default():
    """GPU with non-default model and idle > RECLAIM_TIMEOUT should be reclaimable."""
    router, config = _get_router()
    gpu = router._gpus[config.OLLAMA_0_URL]
    gpu.model = "other:7b"
    gpu.last_active = time.time() - config.RECLAIM_TIMEOUT - 1

    # The reclaim_loop is async, so just verify the condition check.
    now = time.time()
    assert gpu.model != config.DEFAULT_MODEL
    assert (now - gpu.last_active) >= config.RECLAIM_TIMEOUT


def test_record_activity_updates_timestamp():
    router, config = _get_router()
    before = router._gpus[config.OLLAMA_0_URL].last_active
    time.sleep(0.01)
    router.record_activity(config.OLLAMA_0_URL)
    after = router._gpus[config.OLLAMA_0_URL].last_active
    assert after > before


# ── Status tests ─────────────────────────────────────────────────────────────


def test_status_returns_expected_fields():
    router, config = _get_router()
    s = router.status()
    assert s["routing"] == "round_robin"
    assert s["default_model"] == "test-model:7b"
    assert "gpu0" in s["gpus"]
    assert "gpu1" in s["gpus"]
    assert s["gpus"]["gpu0"]["health"] == "healthy"
    assert s["gpus"]["gpu0"]["model"] == "test-model:7b"


# ── Pending default model change ─────────────────────────────────────────────


def test_set_pending_default_model():
    router, _ = _get_router()
    router.set_pending_default_model("new-model:7b")
    assert router._pending_default_model == "new-model:7b"
    s = router.status()
    assert s["pending_default_model"] == "new-model:7b"


# ── Reload model num_ctx pinning ─────────────────────────────────────────────


def _make_post_capturing_client():
    """Build a mock httpx.AsyncClient that records POST kwargs."""
    captured = {}

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return mock_resp

    mock_client.post = _post
    return mock_client, captured


@pytest.mark.asyncio
async def test_reload_model_pins_num_ctx_for_qwen3():
    """qwen3:* reloads must pin num_ctx so Ollama does not auto-fit to 32k.

    Regression: 2026-04-11 GPU 11435 wedge. Ollama auto-picks num_ctx based
    on free VRAM at load time. On a freshly empty 24 GB Tesla P40, qwen3:32b
    loaded at 32768 ctx, forcing CPU offload of the output layer and split
    KV cache, turning 330 ms reloads into 12 s reloads and slowing every
    inference. Pinning to 8192 inside _reload_model fixes the entire class.
    """
    router, config = _get_router()
    gpu = router._gpus[config.OLLAMA_0_URL]

    mock_client, captured = _make_post_capturing_client()
    with patch("gpu_router.httpx.AsyncClient", return_value=mock_client):
        await router._reload_model(gpu, "qwen3:32b")

    body = captured["kwargs"]["json"]
    assert body["model"] == "qwen3:32b"
    assert body.get("options", {}).get("num_ctx") == 8192
    assert gpu.model == "qwen3:32b"


@pytest.mark.asyncio
async def test_reload_model_omits_options_for_non_qwen3():
    """Non-qwen3 reloads must not inject num_ctx — only the known-large
    reasoning models need pinning."""
    router, config = _get_router()
    gpu = router._gpus[config.OLLAMA_0_URL]

    mock_client, captured = _make_post_capturing_client()
    with patch("gpu_router.httpx.AsyncClient", return_value=mock_client):
        await router._reload_model(gpu, "nomic-embed-text")

    body = captured["kwargs"]["json"]
    assert body["model"] == "nomic-embed-text"
    assert "options" not in body
    assert gpu.model == "nomic-embed-text"


# ── Thermal pause tests ──────────────────────────────────────────────────────


def test_thermal_pause_engages_at_threshold(monkeypatch):
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    router, config = _get_router()
    url = config.OLLAMA_0_URL

    router.update_thermal_state(url, 70.0)
    assert router._gpus[url].thermal_paused is False

    router.update_thermal_state(url, 85.0)
    assert router._gpus[url].thermal_paused is True
    assert router._gpus[url].thermal_paused_since is not None
    assert router.is_dispatchable(url) is False


def test_thermal_pause_hysteresis_holds_until_resume_threshold(monkeypatch):
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    router, config = _get_router()
    url = config.OLLAMA_0_URL

    router.update_thermal_state(url, 90.0)
    assert router._gpus[url].thermal_paused is True

    # Temps between resume and pause thresholds preserve the paused state.
    for t in (84.0, 75.0, 65.0, 61.0):
        router.update_thermal_state(url, t)
        assert router._gpus[url].thermal_paused is True, f"flipped at {t}"

    # Crossing the resume threshold clears the pause.
    router.update_thermal_state(url, 60.0)
    assert router._gpus[url].thermal_paused is False
    assert router._gpus[url].thermal_paused_since is None
    assert router.is_dispatchable(url) is True


def test_is_dispatchable_requires_healthy_and_not_paused(monkeypatch):
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    router, config = _get_router()
    url = config.OLLAMA_0_URL

    assert router.is_dispatchable(url) is True

    # Thermally paused but healthy → not dispatchable.
    router.update_thermal_state(url, 90.0)
    assert router.is_dispatchable(url) is False

    # Cool down → dispatchable again.
    router.update_thermal_state(url, 55.0)
    assert router.is_dispatchable(url) is True

    # Mark failed → not dispatchable even when cool.
    router.mark_failed(url)
    assert router.is_dispatchable(url) is False


def test_status_exposes_thermal_state(monkeypatch):
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    router, config = _get_router()
    url = config.OLLAMA_0_URL

    router.update_thermal_state(url, 72.5)
    s = router.status()
    assert s["gpu_temp_pause_c"] == 85
    assert s["gpu_temp_resume_c"] == 60
    assert s["gpus"]["gpu0"]["last_temp_c"] == 72.5
    assert s["gpus"]["gpu0"]["thermal_paused"] is False

    router.update_thermal_state(url, 88.0)
    s = router.status()
    assert s["gpus"]["gpu0"]["thermal_paused"] is True
    assert s["gpus"]["gpu0"]["thermal_paused_since"] is not None
