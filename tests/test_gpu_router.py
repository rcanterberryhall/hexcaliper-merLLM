"""Tests for gpu_router.py — thermal hysteresis, FSM event feeds, status.

In the FSM rewrite (merLLM#55) gpu_router is no longer the dispatch
authority. It only translates hardware-level signals (temperature,
ollama reachability) into Slot FSM events and projects current slot
state for the UI. These tests cover the translation layer; lifecycle
behaviour itself is covered by test_scheduler.py and test_tick.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(autouse=True)
def fresh_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DEFAULT_MODEL", "test-model:7b")
    monkeypatch.setenv("HEALTH_BACKOFF_BASE", "5")
    monkeypatch.setenv("HEALTH_BACKOFF_CAP", "30")
    monkeypatch.setenv("HEALTH_FAULT_TIMEOUT", "60")
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    for mod in list(sys.modules.keys()):
        if mod in ("gpu_router", "queue_manager", "scheduler", "db",
                   "config", "metrics", "notifications"):
            sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules.keys()):
        if mod in ("gpu_router", "queue_manager", "scheduler", "db",
                   "config", "metrics", "notifications"):
            sys.modules.pop(mod, None)


def _seed_slots(state="ready", model="test-model:7b"):
    """Build a minimal _slots list so gpu_router can find indices."""
    import config
    import queue_manager
    from scheduler import Slot, SlotState
    st = SlotState(state) if isinstance(state, str) else state
    queue_manager._slots = [
        Slot(url=config.OLLAMA_0_URL, state=st, model_loaded=model),
        Slot(url=config.OLLAMA_1_URL, state=st, model_loaded=model),
    ]
    return queue_manager, config


# ── Activity tracking ────────────────────────────────────────────────────────


def test_record_activity_populates_idle_seconds():
    import gpu_router
    qm, config = _seed_slots()

    gpu_router.record_activity(config.OLLAMA_0_URL)
    s = gpu_router.status()
    assert s["gpus"]["gpu0"]["idle_seconds"] is not None
    assert s["gpus"]["gpu0"]["idle_seconds"] >= 0
    # gpu1 was never touched — no last_active recorded.
    assert s["gpus"]["gpu1"]["idle_seconds"] is None


# ── Thermal hysteresis ──────────────────────────────────────────────────────


def test_thermal_pause_engages_at_threshold():
    import gpu_router
    qm, config = _seed_slots()
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 70.0)
    assert gpu_router._thermal_paused.get(url, False) is False

    gpu_router.update_thermal_state(url, 85.0)
    assert gpu_router._thermal_paused[url] is True
    assert gpu_router._thermal_paused_since[url] is not None


def test_thermal_pause_hysteresis_holds_until_resume_threshold():
    import gpu_router
    qm, config = _seed_slots()
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 90.0)
    assert gpu_router._thermal_paused[url] is True

    for t in (84.0, 75.0, 65.0, 61.0):
        gpu_router.update_thermal_state(url, t)
        assert gpu_router._thermal_paused[url] is True, f"flipped at {t}"

    gpu_router.update_thermal_state(url, 60.0)
    assert gpu_router._thermal_paused[url] is False
    assert gpu_router._thermal_paused_since[url] is None


def test_thermal_trip_drives_ready_slot_to_cooling():
    """READY + THERMAL_TRIP → COOLING (immediate)."""
    import gpu_router
    from scheduler import SlotState
    qm, config = _seed_slots(state="ready")
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 90.0)
    assert qm._slots[0].state is SlotState.COOLING


def test_thermal_trip_latches_on_busy_slot():
    """BUSY + THERMAL_TRIP latches; slot stays BUSY until WORK_END."""
    import gpu_router
    from scheduler import SlotState
    qm, config = _seed_slots(state="busy")
    qm._slots[0].current_job = {"tid": "x", "model": "test-model:7b"}
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 90.0)
    assert qm._slots[0].state is SlotState.BUSY
    assert qm._slots[0].thermal_pending is True


def test_thermal_clear_drives_cooling_slot_to_ready():
    import gpu_router
    from scheduler import SlotState
    qm, config = _seed_slots(state="ready")
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 90.0)        # READY → COOLING
    assert qm._slots[0].state is SlotState.COOLING
    gpu_router.update_thermal_state(url, 55.0)        # COOLING → READY
    assert qm._slots[0].state is SlotState.READY


def test_thermal_event_swallowed_on_invalid_state():
    """THERMAL_TRIP from UNREACHABLE has no FSM transition — must not raise."""
    import gpu_router
    from scheduler import SlotState
    qm, config = _seed_slots(state="unreachable", model=None)
    url = config.OLLAMA_0_URL

    # Should not raise InvalidTransition.
    gpu_router.update_thermal_state(url, 95.0)
    assert qm._slots[0].state is SlotState.UNREACHABLE


# ── Status projection ──────────────────────────────────────────────────────


def test_status_returns_expected_fields():
    import gpu_router
    qm, config = _seed_slots(state="ready")
    s = gpu_router.status()
    assert s["routing"] == "fsm"
    assert s["default_model"] == "test-model:7b"
    assert s["gpu_temp_pause_c"] == 85
    assert s["gpu_temp_resume_c"] == 60
    assert "gpu0" in s["gpus"]
    assert "gpu1" in s["gpus"]
    assert s["gpus"]["gpu0"]["state"] == "ready"
    assert s["gpus"]["gpu0"]["health"] == "healthy"
    assert s["gpus"]["gpu0"]["model"] == "test-model:7b"
    assert s["gpus"]["gpu0"]["in_flight"] is False


def test_status_marks_unreachable_as_degraded():
    import gpu_router
    _seed_slots(state="unreachable", model=None)
    s = gpu_router.status()
    assert s["gpus"]["gpu0"]["health"] == "degraded"
    assert s["gpus"]["gpu0"]["state"] == "unreachable"


def test_status_marks_busy_loading_as_in_flight():
    import gpu_router
    qm, config = _seed_slots(state="busy")
    s = gpu_router.status()
    assert s["gpus"]["gpu0"]["in_flight"] is True


def test_status_exposes_thermal_state():
    import gpu_router
    qm, config = _seed_slots()
    url = config.OLLAMA_0_URL

    gpu_router.update_thermal_state(url, 72.5)
    s = gpu_router.status()
    assert s["gpus"]["gpu0"]["last_temp_c"] == 72.5
    assert s["gpus"]["gpu0"]["thermal_paused"] is False

    gpu_router.update_thermal_state(url, 88.0)
    s = gpu_router.status()
    assert s["gpus"]["gpu0"]["thermal_paused"] is True
    assert s["gpus"]["gpu0"]["thermal_paused_since"] is not None


# ── Operator reset ──────────────────────────────────────────────────────────


def test_reset_slot_unknown_url_returns_false():
    import gpu_router
    _seed_slots()
    assert gpu_router.reset_slot("http://nowhere:1234") is False


@pytest.mark.asyncio
async def test_reset_slot_known_url_returns_true():
    import gpu_router
    qm, config = _seed_slots(state="unreachable", model=None)
    # ensure_future needs a running loop; the spawned probe task will fail
    # to connect, which is fine — we just need reset_slot to schedule it.
    assert gpu_router.reset_slot(config.OLLAMA_0_URL) is True
