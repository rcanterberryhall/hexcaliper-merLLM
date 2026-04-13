"""Tests for batch queue management endpoints (merLLM#4)."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with an isolated DB and mocked queue_manager."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


@pytest.fixture
def client_with_jobs(client, tmp_path, monkeypatch):
    """client fixture with a few pre-seeded batch jobs in various states."""
    import db
    db.insert_batch_job("q1", "test", "model", "queued prompt 1", {})
    db.insert_batch_job("q2", "test", "model", "queued prompt 2", {})
    db.insert_batch_job("f1", "test", "model", "failed prompt", {})
    db.update_batch_job("f1", status="failed", error="timeout")
    db.insert_batch_job("c1", "test", "model", "completed prompt", {})
    db.update_batch_job("c1", status="completed", completed_at=time.time())
    db.insert_batch_job("x1", "test", "model", "cancelled prompt", {})
    db.update_batch_job("x1", status="cancelled")
    return client


# ── POST /api/batch/retry-failed ─────────────────────────────────────────────


def test_retry_failed_requeues_failed_jobs(client_with_jobs):
    r = client_with_jobs.post("/api/batch/retry-failed")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["requeued"] == 1


def test_retry_failed_with_none_failed(client):
    r = client.post("/api/batch/retry-failed")
    assert r.status_code == 200
    assert r.json()["requeued"] == 0


# ── POST /api/batch/drain ─────────────────────────────────────────────────────


def test_drain_cancels_queued_jobs(client_with_jobs):
    r = client_with_jobs.post("/api/batch/drain")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["cancelled"] == 2


def test_drain_with_no_queued_jobs(client):
    r = client.post("/api/batch/drain")
    assert r.status_code == 200
    assert r.json()["cancelled"] == 0


# ── DELETE /api/batch/completed ──────────────────────────────────────────────


def test_delete_completed_removes_terminal_jobs(client_with_jobs):
    r = client_with_jobs.delete("/api/batch/completed")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["deleted"] == 2  # c1 (completed) + x1 (cancelled)


def test_delete_completed_with_age_filter(client_with_jobs):
    # All jobs were just created — none older than 1 day
    r = client_with_jobs.delete("/api/batch/completed?older_than_days=1")
    assert r.status_code == 200
    assert r.json()["deleted"] == 0


def test_delete_completed_with_no_terminal_jobs(client):
    r = client.delete("/api/batch/completed")
    assert r.status_code == 200
    assert r.json()["deleted"] == 0


def test_delete_completed_preserves_failed_by_default(client_with_jobs):
    r = client_with_jobs.delete("/api/batch/completed")
    assert r.status_code == 200
    assert r.json()["deleted"] == 2  # failed row survives

    import db
    assert db.get_batch_job("f1") is not None


def test_delete_completed_include_failed_drops_everything_terminal(client_with_jobs):
    r = client_with_jobs.delete("/api/batch/completed?include_failed=true")
    assert r.status_code == 200
    assert r.json()["deleted"] == 3  # completed + cancelled + failed

    import db
    assert db.get_batch_job("f1") is None
    assert db.get_batch_job("q1") is not None  # queued untouched
