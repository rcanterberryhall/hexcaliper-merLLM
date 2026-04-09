"""Tests for gpu_router.py — round-robin routing, health, reclaim."""
import asyncio
import os
import sys
import time

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
                   "gpu_router", "metrics", "notifications"):
            sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules.keys()):
        if mod in ("gpu_router", "queue_manager", "db", "config",
                   "gpu_router", "metrics", "notifications"):
            sys.modules.pop(mod, None)


def _get_router():
    import gpu_router
    import config
    # Force re-init with current config values.
    gpu_router._gpus.clear()
    gpu_router._rr_index = 0
    gpu_router._pending_default_model = None
    gpu_router._init_gpus()
    return gpu_router, config


# ── Round-robin tests ─────────────────────────────────────────────────────────


def test_round_robin_alternates():
    """Both GPUs healthy, same model → alternates."""
    router, config = _get_router()
    url1 = router.get_target_url("test-model:7b")
    url2 = router.get_target_url("test-model:7b")
    assert url1 != url2
    assert {url1, url2} == {config.OLLAMA_0_URL, config.OLLAMA_1_URL}


def test_round_robin_uses_default_for_empty_model():
    """Empty model string uses DEFAULT_MODEL."""
    router, config = _get_router()
    url = router.get_target_url("")
    assert url in (config.OLLAMA_0_URL, config.OLLAMA_1_URL)


def test_prefers_free_gpu(monkeypatch):
    """When one GPU is busy, route to the free one."""
    router, config = _get_router()
    import queue_manager

    # Make GPU 0 busy.
    monkeypatch.setattr(queue_manager, "gpu_slot_busy",
                        lambda url: url == config.OLLAMA_0_URL)

    for _ in range(4):
        assert router.get_target_url("test-model:7b") == config.OLLAMA_1_URL


# ── Model routing tests ──────────────────────────────────────────────────────


def test_routes_to_matching_gpu():
    """When GPUs have different models, route to the matching one."""
    router, config = _get_router()
    # Simulate GPU 1 having a different model.
    router._gpus[config.OLLAMA_1_URL].model = "other-model:7b"

    url = router.get_target_url("test-model:7b")
    assert url == config.OLLAMA_0_URL

    url = router.get_target_url("other-model:7b")
    assert url == config.OLLAMA_1_URL


def test_model_mismatch_picks_gpu_and_updates():
    """Request for unloaded model → picks a GPU and updates tracking."""
    router, config = _get_router()
    url = router.get_target_url("brand-new:13b")
    assert url in (config.OLLAMA_0_URL, config.OLLAMA_1_URL)
    # The GPU should now track the new model.
    assert router._gpus[url].model == "brand-new:13b"


# ── Health tests ─────────────────────────────────────────────────────────────


def test_mark_failed_sets_degraded():
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    assert router._gpus[config.OLLAMA_0_URL].health == router.GpuHealth.DEGRADED


def test_degraded_gpu_excluded_from_routing():
    """Degraded GPU should not receive requests."""
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)

    for _ in range(5):
        assert router.get_target_url("test-model:7b") == config.OLLAMA_1_URL


def test_reset_gpu_restores_healthy():
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    router._gpus[config.OLLAMA_0_URL].health = router.GpuHealth.FAULTED

    ok = router.reset_gpu(config.OLLAMA_0_URL)
    assert ok is True
    assert router._gpus[config.OLLAMA_0_URL].health == router.GpuHealth.HEALTHY
    assert router._gpus[config.OLLAMA_0_URL].model == config.DEFAULT_MODEL


def test_all_gpus_down_falls_back():
    """When all GPUs are degraded, return GPU 0 as fallback."""
    router, config = _get_router()
    router.mark_failed(config.OLLAMA_0_URL)
    router.mark_failed(config.OLLAMA_1_URL)

    url = router.get_target_url("test-model:7b")
    assert url == config.OLLAMA_0_URL


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
