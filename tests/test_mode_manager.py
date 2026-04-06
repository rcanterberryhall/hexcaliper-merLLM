"""Tests for mode_manager.py — honest mode transition failures."""
import os
import sys

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(autouse=True)
def fresh_modules(tmp_path, monkeypatch):
    """Reload mode_manager fresh for each test with an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OLLAMA_MANAGE_VIA", "systemctl")
    for mod in list(sys.modules.keys()):
        if mod in ("mode_manager", "db", "config", "geoip", "metrics"):
            sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules.keys()):
        if mod in ("mode_manager", "db", "config", "geoip", "metrics"):
            sys.modules.pop(mod, None)


@pytest.mark.asyncio
async def test_transition_to_night_rolls_back_on_failure(monkeypatch):
    """If systemctl calls fail, mode must NOT change to NIGHT."""
    import mode_manager
    assert mode_manager.current_mode().value == "day"

    # Make all _systemctl calls fail.
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: False)

    ok = await mode_manager.transition_to_night(trigger="test")

    assert ok is False
    # Mode must be rolled back — not NIGHT, not TRANSITIONING.
    assert mode_manager.current_mode().value == "day"


@pytest.mark.asyncio
async def test_transition_to_day_rolls_back_on_failure(monkeypatch, tmp_path):
    """If systemctl calls fail during to-day transition, mode must stay NIGHT."""
    import mode_manager

    # Force into night mode by patching _systemctl to succeed first.
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: True)
    await mode_manager.transition_to_night(trigger="test")
    assert mode_manager.current_mode().value == "night"

    # Now make systemctl fail.
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: False)
    ok = await mode_manager.transition_to_day(trigger="test")

    assert ok is False
    assert mode_manager.current_mode().value == "night"


@pytest.mark.asyncio
async def test_transition_succeeds_when_systemctl_ok(monkeypatch):
    """Successful systemctl calls should advance the mode to NIGHT."""
    import mode_manager
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: True)

    ok = await mode_manager.transition_to_night(trigger="test")

    assert ok is True
    assert mode_manager.current_mode().value == "night"


@pytest.mark.asyncio
async def test_transition_proxy_only_always_succeeds(monkeypatch):
    """In proxy-only mode (OLLAMA_MANAGE_VIA=none) transitions should succeed."""
    monkeypatch.setenv("OLLAMA_MANAGE_VIA", "none")
    for mod in list(sys.modules.keys()):
        if mod in ("mode_manager", "config"):
            sys.modules.pop(mod, None)
    import mode_manager

    ok = await mode_manager.transition_to_night(trigger="test")
    assert ok is True
    assert mode_manager.current_mode().value == "night"


@pytest.mark.asyncio
async def test_set_override_returns_false_on_failure(monkeypatch):
    """set_override returns False when the underlying transition fails."""
    import mode_manager
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: False)

    result = await mode_manager.set_override("night")

    assert result is False
    assert mode_manager.current_mode().value == "day"


@pytest.mark.asyncio
async def test_set_override_returns_true_on_success(monkeypatch):
    """set_override returns True when the transition succeeds."""
    import mode_manager
    monkeypatch.setattr(mode_manager, "_systemctl", lambda action, service: True)

    result = await mode_manager.set_override("night")

    assert result is True
    assert mode_manager.current_mode().value == "night"


@pytest.mark.asyncio
async def test_set_override_none_always_returns_true(monkeypatch):
    """Clearing the override (mode=None) always returns True."""
    import mode_manager
    result = await mode_manager.set_override(None)
    assert result is True


def test_set_mode_endpoint_returns_409_on_failure(tmp_path, monkeypatch):
    """POST /api/merllm/mode returns 409 when the transition fails."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OLLAMA_MANAGE_VIA", "systemctl")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)

    import mode_manager as mm
    import app as app_mod
    from fastapi.testclient import TestClient

    # Make all systemctl calls fail
    monkeypatch.setattr(mm, "_systemctl", lambda action, service: False)

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/merllm/mode", json={"mode": "night"})

    assert resp.status_code == 409
    body = resp.json()
    assert body["ok"] is False
    assert body["mode"] == "day"
    assert "failed" in body["error"]


def test_set_mode_endpoint_returns_200_on_success(tmp_path, monkeypatch):
    """POST /api/merllm/mode returns 200 with ok:True on successful transition."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OLLAMA_MANAGE_VIA", "systemctl")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)

    import mode_manager as mm
    import app as app_mod
    from fastapi.testclient import TestClient

    monkeypatch.setattr(mm, "_systemctl", lambda action, service: True)

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/merllm/mode", json={"mode": "night"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["mode"] == "night"
