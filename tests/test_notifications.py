"""tests/test_notifications.py — Tests for batch job notification dispatch."""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_notifications():
    """Reload notifications module so _sse_listeners list is fresh each test."""
    for mod in list(sys.modules.keys()):
        if mod == "notifications":
            sys.modules.pop(mod)
    yield
    for mod in list(sys.modules.keys()):
        if mod == "notifications":
            sys.modules.pop(mod)


def _import_notif():
    import notifications
    return notifications


def _make_job(status="completed", source_app="parsival", prompt="test prompt"):
    return {
        "id":           "job-uuid-1234",
        "source_app":   source_app,
        "model":        "qwen3:32b",
        "status":       status,
        "submitted_at": "2026-04-05T00:00:00",
        "started_at":   "2026-04-05T00:01:00",
        "completed_at": "2026-04-05T00:02:00",
        "prompt":       prompt,
        "result":       "some result",
        "error":        None,
        "retries":      0,
    }


# ── SSE listener management ───────────────────────────────────────────────────

def test_add_sse_listener_returns_queue():
    notif = _import_notif()
    q = notif.add_sse_listener()
    assert isinstance(q, asyncio.Queue)
    notif.remove_sse_listener(q)


def test_remove_sse_listener():
    notif = _import_notif()
    q = notif.add_sse_listener()
    assert q in notif._sse_listeners
    notif.remove_sse_listener(q)
    assert q not in notif._sse_listeners


def test_remove_nonexistent_listener_is_noop():
    notif = _import_notif()
    q = asyncio.Queue()
    # Should not raise
    notif.remove_sse_listener(q)


def test_add_multiple_listeners():
    notif = _import_notif()
    q1 = notif.add_sse_listener()
    q2 = notif.add_sse_listener()
    assert len(notif._sse_listeners) == 2
    notif.remove_sse_listener(q1)
    notif.remove_sse_listener(q2)
    assert len(notif._sse_listeners) == 0


# ── _broadcast_sse ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_broadcast_sends_to_all_listeners():
    notif = _import_notif()
    q1 = notif.add_sse_listener()
    q2 = notif.add_sse_listener()

    try:
        await notif._broadcast_sse({"type": "job_complete", "job_id": "abc"})

        assert not q1.empty()
        assert not q2.empty()
        data1 = json.loads(q1.get_nowait())
        data2 = json.loads(q2.get_nowait())
        assert data1["type"] == "job_complete"
        assert data2["job_id"] == "abc"
    finally:
        notif._sse_listeners.clear()


@pytest.mark.asyncio
async def test_broadcast_noop_with_no_listeners():
    notif = _import_notif()
    # Should not raise
    await notif._broadcast_sse({"type": "job_complete", "job_id": "abc"})


@pytest.mark.asyncio
async def test_broadcast_event_is_valid_json():
    notif = _import_notif()
    q = notif.add_sse_listener()
    try:
        await notif._broadcast_sse({"type": "job_complete", "status": "completed"})
        raw = q.get_nowait()
        parsed = json.loads(raw)
        assert parsed["status"] == "completed"
    finally:
        notif._sse_listeners.clear()


# ── dispatch: SSE channel ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_broadcasts_sse_to_listener():
    notif = _import_notif()
    q = notif.add_sse_listener()

    try:
        await notif.dispatch(_make_job(), webhook_url=None)

        assert not q.empty()
        data = json.loads(q.get_nowait())
        assert data["type"] == "job_complete"
        assert data["job_id"] == "job-uuid-1234"
        assert data["source_app"] == "parsival"
        assert data["status"] == "completed"
    finally:
        notif._sse_listeners.clear()


@pytest.mark.asyncio
async def test_dispatch_includes_prompt_preview():
    notif = _import_notif()
    q = notif.add_sse_listener()

    try:
        long_prompt = "A" * 200
        await notif.dispatch(_make_job(prompt=long_prompt), webhook_url=None)
        data = json.loads(q.get_nowait())
        assert len(data["prompt_preview"]) == 100
        assert data["prompt_preview"] == "A" * 100
    finally:
        notif._sse_listeners.clear()


@pytest.mark.asyncio
async def test_dispatch_failed_job_status():
    notif = _import_notif()
    q = notif.add_sse_listener()

    job = _make_job(status="failed")
    job["error"] = "Connection refused"

    try:
        await notif.dispatch(job, webhook_url=None)
        data = json.loads(q.get_nowait())
        assert data["status"] == "failed"
        assert data["error"] == "Connection refused"
    finally:
        notif._sse_listeners.clear()


@pytest.mark.asyncio
async def test_dispatch_empty_prompt_preview():
    notif = _import_notif()
    q = notif.add_sse_listener()

    job = _make_job(prompt="")

    try:
        await notif.dispatch(job, webhook_url=None)
        data = json.loads(q.get_nowait())
        assert data["prompt_preview"] == ""
    finally:
        notif._sse_listeners.clear()


# ── dispatch: webhook channel ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_calls_webhook_when_url_set():
    notif = _import_notif()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await notif.dispatch(_make_job(), webhook_url="https://hooks.example.com/test")

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://hooks.example.com/test"
    payload = call_args[1]["json"]
    assert payload["job_id"] == "job-uuid-1234"
    assert payload["status"] == "completed"


@pytest.mark.asyncio
async def test_dispatch_no_webhook_when_url_none():
    notif = _import_notif()

    with patch("httpx.AsyncClient") as mock_cls:
        await notif.dispatch(_make_job(), webhook_url=None)
        # httpx client should not have been instantiated for webhook
        mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_webhook_failure_is_swallowed():
    """A webhook HTTP error should not propagate — it's fire-and-forget."""
    notif = _import_notif()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=Exception("Network error"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Should not raise
        await notif.dispatch(_make_job(), webhook_url="https://hooks.example.com/test")


@pytest.mark.asyncio
async def test_dispatch_webhook_payload_structure():
    """Webhook payload contains all expected fields."""
    notif = _import_notif()
    captured = {}

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def capture_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        return mock_response

    mock_client.post = capture_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await notif.dispatch(_make_job(), webhook_url="https://hooks.example.com/test")

    assert "job_id" in captured
    assert "source_app" in captured
    assert "status" in captured
    assert "submitted_at" in captured
    assert "completed_at" in captured
    assert "prompt_preview" in captured
    assert "error" in captured


# ── /api/merllm/events SSE endpoint ──────────────────────────────────────────

def test_events_endpoint_is_registered(tmp_path, monkeypatch):
    """GET /api/merllm/events route is registered on the app."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    # Clear cached modules to get a fresh app
    for mod in list(sys.modules.keys()):
        if mod in ("app", "mode_manager", "queue_manager", "geoip", "metrics",
                   "db", "config", "notifications"):
            sys.modules.pop(mod, None)

    import app as app_mod
    paths = {r.path for r in app_mod.app.routes if hasattr(r, "path")}
    assert "/api/merllm/events" in paths
