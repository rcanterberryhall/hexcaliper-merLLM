"""Tests for POST /api/merllm/backup."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


def _fresh_app(tmp_path, monkeypatch):
    """Return a TestClient with isolated DB and backup dir."""
    db_file = tmp_path / "merllm.db"
    bk_dir  = tmp_path / "backups"
    monkeypatch.setenv("DB_PATH",    str(db_file))
    monkeypatch.setenv("BACKUP_DIR", str(bk_dir))
    monkeypatch.setenv("BACKUP_KEEP_DAYS", "3")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    # seed a minimal DB so the file exists
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE test (v INTEGER)")
    conn.execute("INSERT INTO test VALUES (42)")
    conn.commit()
    conn.close()

    import app as app_mod
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app, raise_server_exceptions=True)


def test_backup_creates_file(tmp_path, monkeypatch):
    client = _fresh_app(tmp_path, monkeypatch)
    resp = client.post("/api/merllm/backup")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["size_bytes"] > 0
    import pathlib
    assert pathlib.Path(body["backup"]).exists()


def test_backup_file_is_valid_sqlite(tmp_path, monkeypatch):
    client = _fresh_app(tmp_path, monkeypatch)
    resp = client.post("/api/merllm/backup")
    backup_path = resp.json()["backup"]
    conn = sqlite3.connect(backup_path)
    row = conn.execute("SELECT v FROM test").fetchone()
    conn.close()
    assert row == (42,)


def test_backup_rotation(tmp_path, monkeypatch):
    """After more than BACKUP_KEEP_DAYS backups, only BACKUP_KEEP_DAYS files remain."""
    import pathlib, time as _time
    kept = 3
    client = _fresh_app(tmp_path, monkeypatch)
    # Make kept+2 backups; stub time so each gets a unique timestamp
    counter = [0]
    orig_strftime = _time.strftime

    def _fake_strftime(fmt, *args):
        counter[0] += 1
        return f"20260101-{counter[0]:06d}"

    monkeypatch.setattr(_time, "strftime", _fake_strftime)

    for _ in range(kept + 2):
        r = client.post("/api/merllm/backup")
        assert r.status_code == 200

    bk_dir = tmp_path / "backups"
    remaining = sorted(bk_dir.glob("merllm-*.db"))
    assert len(remaining) == kept


def test_backup_missing_db_returns_500(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH",    str(tmp_path / "nonexistent.db"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    import app as app_mod
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app, raise_server_exceptions=False)
    resp = client.post("/api/merllm/backup")
    assert resp.status_code == 500
    assert "not found" in resp.json()["detail"]
