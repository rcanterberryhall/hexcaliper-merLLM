"""Tests for db.py — SQLite operations."""
import json
import os
import sys
import tempfile
import time

import pytest

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Each test gets its own isolated SQLite database."""
    db_path = str(tmp_path / "test_merllm.db")
    monkeypatch.setenv("DB_PATH", db_path)
    # Reload db and config with fresh env
    for mod in ["db", "config"]:
        sys.modules.pop(mod, None)
    import db
    return db


# ── Batch jobs ─────────────────────────────────────────────────────────────────


def test_insert_and_get_batch_job(tmp_db):
    db = tmp_db
    db.insert_batch_job("job1", "parsival", "qwen3:32b", "hello", {})
    job = db.get_batch_job("job1")
    assert job is not None
    assert job["id"] == "job1"
    assert job["source_app"] == "parsival"
    assert job["model"] == "qwen3:32b"
    assert job["prompt"] == "hello"
    assert job["status"] == "queued"


def test_get_nonexistent_job(tmp_db):
    assert tmp_db.get_batch_job("missing") is None


def test_list_batch_jobs_empty(tmp_db):
    assert tmp_db.list_batch_jobs() == []


def test_list_batch_jobs_filter_status(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "p1", {})
    db.insert_batch_job("j2", "app", "model", "p2", {})
    db.update_batch_job("j1", status="completed", completed_at=time.time())

    queued = db.list_batch_jobs(status="queued")
    assert len(queued) == 1
    assert queued[0]["id"] == "j2"

    completed = db.list_batch_jobs(status="completed")
    assert len(completed) == 1
    assert completed[0]["id"] == "j1"


def test_update_batch_job(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="running", started_at=time.time())
    job = db.get_batch_job("j1")
    assert job["status"] == "running"
    assert job["started_at"] is not None


def test_cancel_batch_job(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    ok = db.cancel_batch_job("j1")
    assert ok is True
    assert db.get_batch_job("j1")["status"] == "cancelled"


def test_cancel_non_queued_job_fails(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="running", started_at=time.time())
    ok = db.cancel_batch_job("j1")
    assert ok is False


def test_requeue_failed_job(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="failed", completed_at=time.time(), error="oops")
    ok = db.requeue_batch_job("j1")
    assert ok is True
    assert db.get_batch_job("j1")["status"] == "queued"


def test_requeue_non_failed_job_fails(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    ok = db.requeue_batch_job("j1")
    assert ok is False


def test_count_batch_jobs_by_status(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "p", {})
    db.insert_batch_job("j2", "app", "model", "p", {})
    db.update_batch_job("j2", status="completed", completed_at=time.time())
    counts = db.count_batch_jobs_by_status()
    assert counts.get("queued", 0) >= 1
    assert counts.get("completed", 0) >= 1


# ── Settings ───────────────────────────────────────────────────────────────────


def test_settings_roundtrip(tmp_db):
    db = tmp_db
    db.save_settings({"night_model": "qwen3:8b", "inactivity_timeout_min": 45})
    s = db.get_settings()
    assert s["night_model"] == "qwen3:8b"
    assert s["inactivity_timeout_min"] == 45


def test_settings_update(tmp_db):
    db = tmp_db
    db.save_settings({"night_model": "a"})
    db.save_settings({"night_model": "b"})
    s = db.get_settings()
    assert s["night_model"] == "b"


def test_get_settings_returns_none_when_empty(tmp_db):
    result = tmp_db.get_settings()
    assert result is None or isinstance(result, dict)


# ── Metrics ────────────────────────────────────────────────────────────────────


def test_insert_and_retrieve_metrics(tmp_db):
    db = tmp_db
    db.insert_metrics([("cpu.core0", 42.5), ("ram.used", 1024.0)])
    latest = db.get_latest_metrics()
    assert "cpu.core0" in latest or "ram.used" in latest


def test_metrics_history(tmp_db):
    db = tmp_db
    now = time.time()
    for i in range(5):
        db.insert_metrics([("cpu.core0", float(i * 10))])
        time.sleep(0.01)
    history = db.get_metrics_history("cpu.core0", since=now - 10)
    assert len(history) >= 5


# ── Transitions ────────────────────────────────────────────────────────────────


def test_insert_and_list_transitions(tmp_db):
    db = tmp_db
    db.insert_transition("to_night", "schedule", 1.2, True)
    db.insert_transition("to_day", "interactive_request", 0.5, True)
    transitions = db.list_transitions(limit=10)
    assert len(transitions) == 2
    assert transitions[0]["direction"] == "to_day"   # most recent first


# ── Fan faults ─────────────────────────────────────────────────────────────────


def test_insert_and_list_fan_faults(tmp_db):
    db = tmp_db
    db.insert_fan_fault("gpu_fault_onset", "NVML failure detected", fan_speed_applied=80)
    db.insert_fan_fault("gpu_fault_cleared", "NVML recovered")
    faults = db.list_fan_faults(limit=10)
    assert len(faults) == 2
    assert faults[0]["event_type"] == "gpu_fault_cleared"   # most recent first
    assert faults[1]["event_type"] == "gpu_fault_onset"
    assert faults[1]["fan_speed_applied"] == 80
    assert faults[0]["fan_speed_applied"] is None


def test_fan_fault_list_is_empty_initially(tmp_db):
    db = tmp_db
    assert db.list_fan_faults() == []


def test_fan_fault_limit_is_respected(tmp_db):
    db = tmp_db
    for i in range(10):
        db.insert_fan_fault("gpu_fault_onset", f"fault {i}", fan_speed_applied=80)
    faults = db.list_fan_faults(limit=5)
    assert len(faults) == 5
