"""tests/test_myday.py — Tests for the /api/merllm/myday aggregation endpoint."""
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


def _parsival_payload():
    return {
        "action_count":    120,
        "cold_start":      False,
        "cold_start_msg":  "",
        "attended_count":  30,
        "ignored_count":   15,
        "last_updated":    "2026-04-05T10:00:00",
        "active_situations":  3,
        "new_investigating":  2,
        "overdue_followups":  1,
    }


def _lancellmot_payload():
    return {
        "acquisition_pending": 2,
        "escalation_pending":  1,
        "total_pending":       3,
    }


def _mock_httpx(parsival=None, lancellmot=None,
                parsival_exc=None, lancellmot_exc=None):
    """Return a mock httpx.AsyncClient that returns configured responses."""

    call_count = [0]

    async def mock_get(url, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "attention" in url:
            if parsival_exc:
                raise parsival_exc
            resp.json = MagicMock(return_value=parsival or _parsival_payload())
        else:
            if lancellmot_exc:
                raise lancellmot_exc
            resp.json = MagicMock(return_value=lancellmot or _lancellmot_payload())
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = mock_get
    return mock_client


# ── Basic response structure ──────────────────────────────────────────────────

def test_myday_returns_three_sections(client):
    with patch("httpx.AsyncClient", return_value=_mock_httpx()):
        resp = client.get("/api/merllm/myday")
    assert resp.status_code == 200
    body = resp.json()
    assert "parsival" in body
    assert "lancellmot" in body
    assert "merllm" in body


def test_myday_parsival_section(client):
    with patch("httpx.AsyncClient", return_value=_mock_httpx()):
        body = client.get("/api/merllm/myday").json()
    p = body["parsival"]
    assert p["ok"] is True
    assert p["active_situations"] == 3
    assert p["new_investigating"] == 2
    assert p["overdue_followups"] == 1
    assert p["cold_start"] is False


def test_myday_lancellmot_section(client):
    with patch("httpx.AsyncClient", return_value=_mock_httpx()):
        body = client.get("/api/merllm/myday").json()
    l = body["lancellmot"]
    assert l["ok"] is True
    assert l["acquisition_pending"] == 2
    assert l["escalation_pending"] == 1
    assert l["total_pending"] == 3


def test_myday_merllm_section_counts_jobs(client, tmp_path):
    import db
    db.insert_batch_job("q1", "test", "model", "prompt", {})
    db.insert_batch_job("c1", "test", "model", "prompt", {})
    db.update_batch_job("c1", status="completed", completed_at=time.time())
    db.insert_batch_job("f1", "test", "model", "prompt", {})
    db.update_batch_job("f1", status="failed", error="err")

    with patch("httpx.AsyncClient", return_value=_mock_httpx()):
        body = client.get("/api/merllm/myday").json()
    m = body["merllm"]
    assert m["ok"] is True
    assert m["queued_jobs"] == 1
    assert m["completed_jobs"] == 1
    assert m["failed_jobs"] == 1


def test_myday_merllm_ok_always_true(client):
    with patch("httpx.AsyncClient", return_value=_mock_httpx()):
        body = client.get("/api/merllm/myday").json()
    assert body["merllm"]["ok"] is True


# ── Offline / error handling ──────────────────────────────────────────────────

def test_myday_parsival_offline(client):
    with patch("httpx.AsyncClient",
               return_value=_mock_httpx(parsival_exc=Exception("Connection refused"))):
        body = client.get("/api/merllm/myday").json()
    p = body["parsival"]
    assert p["ok"] is False
    assert p["active_situations"] == 0
    assert p["overdue_followups"] == 0


def test_myday_lancellmot_offline(client):
    with patch("httpx.AsyncClient",
               return_value=_mock_httpx(lancellmot_exc=Exception("timeout"))):
        body = client.get("/api/merllm/myday").json()
    l = body["lancellmot"]
    assert l["ok"] is False
    assert l["total_pending"] == 0


def test_myday_both_offline_still_200(client):
    with patch("httpx.AsyncClient",
               return_value=_mock_httpx(
                   parsival_exc=Exception("down"),
                   lancellmot_exc=Exception("down"),
               )):
        resp = client.get("/api/merllm/myday")
    assert resp.status_code == 200
    body = resp.json()
    assert body["parsival"]["ok"] is False
    assert body["lancellmot"]["ok"] is False
    assert body["merllm"]["ok"] is True


# ── Endpoint registration ─────────────────────────────────────────────────────

def test_myday_endpoint_registered(client):
    import app as app_mod
    paths = {r.path for r in app_mod.app.routes if hasattr(r, "path")}
    assert "/api/merllm/myday" in paths
