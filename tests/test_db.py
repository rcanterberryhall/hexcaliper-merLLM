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
    db.save_settings({"default_model": "qwen3:8b", "reclaim_timeout": 120})
    s = db.get_settings()
    assert s["default_model"] == "qwen3:8b"
    assert s["reclaim_timeout"] == 120


def test_settings_update(tmp_db):
    db = tmp_db
    db.save_settings({"default_model": "a"})
    db.save_settings({"default_model": "b"})
    s = db.get_settings()
    assert s["default_model"] == "b"


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


def test_requeue_orphaned_jobs(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.insert_batch_job("j3", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="running", started_at=time.time())
    db.update_batch_job("j2", status="running", started_at=time.time())
    db.update_batch_job("j3", status="completed", completed_at=time.time())

    recovered = db.requeue_orphaned_jobs()
    assert recovered == 2

    j1 = db.get_batch_job("j1")
    j2 = db.get_batch_job("j2")
    j3 = db.get_batch_job("j3")
    assert j1["status"] == "queued"
    assert j1["started_at"] is None
    assert j2["status"] == "queued"
    assert j3["status"] == "completed"   # untouched


def test_requeue_orphaned_jobs_none_running(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    assert db.requeue_orphaned_jobs() == 0


def test_requeue_orphaned_jobs_preserves_error_history(tmp_db):
    """Prior retry/failure history must not be clobbered on restart recovery.

    Before the fix, ``requeue_orphaned_jobs`` unconditionally set
    ``error = 'Recovered after restart'``. A job that restarted twice
    during its retry loop lost every earlier failure trace, making
    post-mortems on flaky upstream calls impossible.
    """
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    # j1 had a retry failure before the restart happened.
    db.update_batch_job(
        "j1", status="running", started_at=time.time(),
        error="[attempt 1 failed: upstream timeout]",
    )
    # j2 never errored.
    db.update_batch_job("j2", status="running", started_at=time.time())

    db.requeue_orphaned_jobs()

    j1 = db.get_batch_job("j1")
    assert "attempt 1 failed: upstream timeout" in j1["error"]
    assert "[recovered after restart]" in j1["error"]

    j2 = db.get_batch_job("j2")
    assert j2["error"] == "[recovered after restart]"

    # Second restart must not duplicate the marker.
    db.update_batch_job("j1", status="running")
    db.requeue_orphaned_jobs()
    j1 = db.get_batch_job("j1")
    assert j1["error"].count("[recovered after restart]") == 1


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


# ── Batch queue management helpers ────────────────────────────────────────────


def test_drain_queued_jobs(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.insert_batch_job("j3", "app", "model", "prompt", {})
    db.update_batch_job("j3", status="running", started_at=time.time())

    count = db.drain_queued_jobs()
    assert count == 2
    assert db.get_batch_job("j1")["status"] == "cancelled"
    assert db.get_batch_job("j2")["status"] == "cancelled"
    assert db.get_batch_job("j3")["status"] == "running"  # not touched


def test_drain_queued_jobs_none_queued(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="running", started_at=time.time())
    assert db.drain_queued_jobs() == 0


def test_requeue_all_failed_jobs(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="failed", error="timeout")
    db.update_batch_job("j2", status="failed", error="oom")

    count = db.requeue_all_failed_jobs()
    assert count == 2
    j1 = db.get_batch_job("j1")
    assert j1["status"] == "queued"
    assert j1["error"] is None


def test_requeue_all_failed_jobs_none_failed(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    assert db.requeue_all_failed_jobs() == 0


def test_delete_terminal_jobs_all(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.insert_batch_job("j3", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="completed", completed_at=time.time())
    db.update_batch_job("j2", status="cancelled")
    # j3 stays queued

    count = db.delete_terminal_jobs()
    assert count == 2
    assert db.get_batch_job("j1") is None
    assert db.get_batch_job("j2") is None
    assert db.get_batch_job("j3") is not None


def test_delete_terminal_jobs_age_filter(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="completed", completed_at=time.time())
    db.update_batch_job("j2", status="completed", completed_at=time.time())

    # Filter by age=1 day — both jobs were just submitted, none old enough
    count = db.delete_terminal_jobs(older_than_days=1)
    assert count == 0
    assert db.get_batch_job("j1") is not None


def test_delete_terminal_jobs_preserves_failed_by_default(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.insert_batch_job("j3", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="completed")
    db.update_batch_job("j2", status="failed", error="boom")
    db.update_batch_job("j3", status="cancelled")

    count = db.delete_terminal_jobs()
    assert count == 2
    assert db.get_batch_job("j1") is None
    assert db.get_batch_job("j2") is not None  # failed stays as evidence
    assert db.get_batch_job("j3") is None


def test_delete_terminal_jobs_include_failed_opts_in(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    db.insert_batch_job("j3", "app", "model", "prompt", {})
    db.update_batch_job("j1", status="completed")
    db.update_batch_job("j2", status="failed", error="boom")
    db.update_batch_job("j3", status="queued")

    count = db.delete_terminal_jobs(include_failed=True)
    assert count == 2
    assert db.get_batch_job("j1") is None
    assert db.get_batch_job("j2") is None
    assert db.get_batch_job("j3") is not None  # queued untouched


# ── Retry columns ─────────────────────────────────────────────────────────────


def test_new_job_has_zero_retries(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    j = db.get_batch_job("j1")
    assert j["retries"] == 0
    assert j["retry_after"] is None


def test_list_batch_jobs_ready_only_excludes_deferred(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt 1", {})  # ready (no retry_after)
    db.insert_batch_job("j2", "app", "model", "prompt 2", {})  # deferred
    db.update_batch_job("j2", retry_after=time.time() + 3600)  # 1h in the future

    ready = db.list_batch_jobs(status="queued", ready_only=True)
    ids = [j["id"] for j in ready]
    assert "j1" in ids
    assert "j2" not in ids


def test_list_batch_jobs_ready_only_includes_past_retry_after(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.update_batch_job("j1", retry_after=time.time() - 10)  # in the past

    ready = db.list_batch_jobs(status="queued", ready_only=True)
    assert any(j["id"] == "j1" for j in ready)


def test_get_earliest_retry_after_none_when_empty(tmp_db):
    db = tmp_db
    assert db.get_earliest_retry_after() is None


def test_get_earliest_retry_after_ignores_ready_jobs(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    # j1 has no retry_after — should not appear
    assert db.get_earliest_retry_after() is None


def test_get_earliest_retry_after_returns_minimum(tmp_db):
    db = tmp_db
    db.insert_batch_job("j1", "app", "model", "prompt", {})
    db.insert_batch_job("j2", "app", "model", "prompt", {})
    future1 = time.time() + 60
    future2 = time.time() + 120
    db.update_batch_job("j1", retry_after=future1)
    db.update_batch_job("j2", retry_after=future2)

    result = db.get_earliest_retry_after()
    assert result is not None
    assert abs(result - future1) < 1.0  # closest future timestamp
