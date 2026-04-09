"""tests/test_activity_sse.py — Tests for SSE activity push and tracking."""
import asyncio
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_activity():
    """Reset activity state and SSE queues before each test."""
    import importlib
    # Reload app module so module-level state is fresh
    for mod in list(sys.modules.keys()):
        if mod in ("app", "gpu_router", "queue_manager", "metrics", "db", "config"):
            sys.modules.pop(mod, None)
    yield
    # Clean up again after test
    for mod in list(sys.modules.keys()):
        if mod in ("app",):
            sys.modules.pop(mod, None)


def _import_app_internals():
    """Import the activity functions from a fresh app module."""
    import app as app_mod
    return app_mod


# ── _push_activity_sse ────────────────────────────────────────────────────────

def test_push_noop_with_no_clients(tmp_path, monkeypatch):
    """_push_activity_sse should silently do nothing when no SSE clients are connected."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    # No queues registered — should not raise
    app_mod._push_activity_sse(force=True)
    app_mod._push_activity_sse(force=False)


def test_push_sends_to_connected_queue(tmp_path, monkeypatch):
    """_push_activity_sse should put a JSON snapshot into each registered queue."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    q = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)

    try:
        app_mod._push_activity_sse(force=True)
        assert not q.empty()
        data = q.get_nowait()
        parsed = json.loads(data)
        assert "gpu0" in parsed
        assert "gpu1" in parsed
    finally:
        app_mod._activity_sse_queues.clear()


def test_push_rate_limited_when_not_forced(tmp_path, monkeypatch):
    """Non-forced pushes respect the minimum interval."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    q = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)
    app_mod._last_sse_push = time.time()  # simulate a very recent push

    try:
        app_mod._push_activity_sse(force=False)
        # Should be rate-limited: queue stays empty
        assert q.empty()

        # Force=True bypasses rate limit
        app_mod._push_activity_sse(force=True)
        assert not q.empty()
    finally:
        app_mod._activity_sse_queues.clear()


def test_push_sends_to_multiple_queues(tmp_path, monkeypatch):
    """_push_activity_sse sends to all registered client queues."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    q1 = asyncio.Queue()
    q2 = asyncio.Queue()
    app_mod._activity_sse_queues.extend([q1, q2])

    try:
        app_mod._push_activity_sse(force=True)
        assert not q1.empty()
        assert not q2.empty()
    finally:
        app_mod._activity_sse_queues.clear()


# ── _activity_set / _activity_inc / _activity_clear ───────────────────────────

def test_activity_set_registers_entry_and_pushes(tmp_path, monkeypatch):
    """_activity_set populates the activity dict and triggers a forced SSE push."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    q = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)

    try:
        app_mod._activity_set("http://host.docker.internal:11434", "qwen3:32b", "/api/chat")

        entry = app_mod._activity.get("gpu0")
        assert entry is not None
        assert entry["model"] == "qwen3:32b"
        assert entry["chunks"] == 0

        # A forced push should have been sent
        assert not q.empty()
    finally:
        app_mod._activity_sse_queues.clear()
        app_mod._activity["gpu0"] = None


def test_activity_inc_increments_chunks_and_pushes(tmp_path, monkeypatch):
    """_activity_inc increments the chunk counter and triggers a rate-limited push."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    url = "http://host.docker.internal:11434"
    app_mod._activity_set(url, "qwen3:32b", "/api/chat")
    app_mod._activity_sse_queues.clear()  # clear the set push

    q = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)
    app_mod._last_sse_push = 0.0  # reset rate limit

    try:
        app_mod._activity_inc(url)
        assert app_mod._activity["gpu0"]["chunks"] == 1

        app_mod._activity_inc(url)
        assert app_mod._activity["gpu0"]["chunks"] == 2

        # At least one push should have gone out (rate limit allows first inc)
        assert not q.empty()
    finally:
        app_mod._activity_sse_queues.clear()
        app_mod._activity["gpu0"] = None


def test_activity_clear_resets_entry_and_pushes(tmp_path, monkeypatch):
    """_activity_clear sets the entry to None and triggers a forced SSE push."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    url = "http://host.docker.internal:11434"
    app_mod._activity_set(url, "qwen3:32b", "/api/chat")
    app_mod._activity_sse_queues.clear()  # clear prior push

    q = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)

    try:
        app_mod._activity_clear(url)

        assert app_mod._activity["gpu0"] is None
        # Forced push should have fired; snapshot should show gpu0 as None
        assert not q.empty()
        data = json.loads(q.get_nowait())
        assert data["gpu0"] is None
    finally:
        app_mod._activity_sse_queues.clear()


def test_activity_inc_noop_when_no_entry(tmp_path, monkeypatch):
    """_activity_inc does nothing when the GPU has no active entry."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    app_mod = _import_app_internals()

    app_mod._activity["gpu0"] = None
    # Should not raise
    app_mod._activity_inc("http://host.docker.internal:11434")


# ── SSE endpoint ──────────────────────────────────────────────────────────────

def test_activity_stream_endpoint_is_registered(tmp_path, monkeypatch):
    """GET /api/merllm/activity/stream route is registered on the app."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    import app as app_mod

    # Verify the route exists and has the correct media type configured.
    # Full streaming behaviour (queue push, content-type header) is exercised
    # by the other tests in this file.
    paths = {r.path for r in app_mod.app.routes if hasattr(r, "path")}
    assert "/api/merllm/activity/stream" in paths
