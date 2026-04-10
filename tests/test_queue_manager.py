"""Tests for queue_manager.py — unified GPU queue with tracking."""
import asyncio
import os
import sys
import time

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(autouse=True)
def reset_queue():
    """Re-import queue_manager fresh for each test to reset module state."""
    for mod in list(sys.modules.keys()):
        if mod in ("queue_manager", "config", "db", "gpu_router"):
            del sys.modules[mod]
    yield


@pytest.fixture
def qm():
    import queue_manager
    return queue_manager


def test_priority_constants(qm):
    assert qm.PRIORITY_INTERACTIVE == 0
    assert qm.PRIORITY_BATCH == 1
    assert qm.PRIORITY_INTERACTIVE < qm.PRIORITY_BATCH


def test_queue_depth_empty(qm):
    d = qm.queue_depth()
    assert d["queued"] == 0
    assert d["running"] == 0
    assert d["total"] == 0


# ── Request tracking tests ───────────────────────────────────────────────────


def test_track_request_returns_id(qm):
    tid = qm.track_request("test", "generate", "qwen3:32b",
                            qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    assert len(tid) == 36  # UUID format


def test_track_request_appears_in_active_queue(qm):
    tid = qm.track_request("lancellmot", "chat", "qwen3:32b",
                            qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    queue = qm.active_queue()
    assert len(queue) == 1
    entry = queue[0]
    assert entry["id"] == tid
    assert entry["source"] == "lancellmot"
    assert entry["request_type"] == "chat"
    assert entry["model"] == "qwen3:32b"
    assert entry["priority"] == "interactive"
    assert entry["status"] == "queued"


def test_track_request_shows_in_queue_depth(qm):
    qm.track_request("test", "generate", "m", qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    d = qm.queue_depth()
    assert d["queued"] == 1
    assert d["total"] == 1


@pytest.mark.asyncio
async def test_acquire_transitions_to_running(qm):
    tid = qm.track_request("test", "generate", "m",
                            qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    acquired = await qm.acquire_gpu_slot("http://gpu0:11434",
                                          qm.PRIORITY_INTERACTIVE, tid)
    assert acquired is True
    queue = qm.active_queue()
    assert queue[0]["status"] == "running"
    assert queue[0]["started_at"] is not None
    d = qm.queue_depth()
    assert d["running"] == 1
    assert d["queued"] == 0
    qm.release_gpu_slot("http://gpu0:11434", tid)


@pytest.mark.asyncio
async def test_release_transitions_to_completed(qm):
    tid = qm.track_request("test", "chat", "m",
                            qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    await qm.acquire_gpu_slot("http://gpu0:11434", qm.PRIORITY_INTERACTIVE, tid)
    qm.release_gpu_slot("http://gpu0:11434", tid)
    queue = qm.active_queue()
    assert queue[0]["status"] == "completed"
    assert queue[0]["completed_at"] is not None


@pytest.mark.asyncio
async def test_fail_request(qm):
    tid = qm.track_request("test", "generate", "m",
                            qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    await qm.acquire_gpu_slot("http://gpu0:11434", qm.PRIORITY_INTERACTIVE, tid)
    qm.fail_request(tid, "Ollama OOM")
    queue = qm.active_queue()
    assert queue[0]["status"] == "failed"
    assert queue[0]["error"] == "Ollama OOM"


# ── GPU slot tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_and_release_slot(qm):
    """Acquiring and then releasing a slot should leave it free."""
    target = "http://gpu0:11434"
    acquired = await qm.acquire_gpu_slot(target, qm.PRIORITY_INTERACTIVE)
    assert acquired is True
    assert qm.gpu_slot_busy(target) is True
    qm.release_gpu_slot(target)
    assert qm.gpu_slot_busy(target) is False


@pytest.mark.asyncio
async def test_two_gpus_independent(qm):
    """Slots for different GPU targets are completely independent."""
    gpu0 = "http://gpu0:11434"
    gpu1 = "http://gpu1:11435"
    ok0 = await qm.acquire_gpu_slot(gpu0, qm.PRIORITY_INTERACTIVE)
    ok1 = await qm.acquire_gpu_slot(gpu1, qm.PRIORITY_INTERACTIVE)
    assert ok0 is True
    assert ok1 is True
    assert qm.gpu_slot_busy(gpu0) is True
    assert qm.gpu_slot_busy(gpu1) is True
    qm.release_gpu_slot(gpu0)
    qm.release_gpu_slot(gpu1)


@pytest.mark.asyncio
async def test_interactive_timeout_when_slot_busy(monkeypatch, qm):
    """Interactive request returns False when GPU slot is held and timeout elapses."""
    import os
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "0")
    # Re-import config so the patched env var takes effect
    for mod in list(sys.modules.keys()):
        if mod in ("config", "queue_manager"):
            del sys.modules[mod]
    import queue_manager as qm2
    target = "http://gpu0:11434"
    # Occupy the slot
    await qm2.acquire_gpu_slot(target, qm2.PRIORITY_INTERACTIVE)
    # Second interactive request should time out immediately
    result = await qm2.acquire_gpu_slot(target, qm2.PRIORITY_INTERACTIVE)
    assert result is False
    qm2.release_gpu_slot(target)


@pytest.mark.asyncio
async def test_batch_waits_for_slot(qm):
    """Batch request acquires the slot once it is released."""
    target = "http://gpu0:11434"
    await qm.acquire_gpu_slot(target, qm.PRIORITY_INTERACTIVE)

    batch_acquired = asyncio.get_event_loop().create_future()

    async def _batch_waiter():
        ok = await qm.acquire_gpu_slot(target, qm.PRIORITY_BATCH)
        batch_acquired.set_result(ok)

    asyncio.create_task(_batch_waiter())
    await asyncio.sleep(0)  # yield so _batch_waiter starts waiting

    # Slot is still held — batch should not have acquired it yet
    assert not batch_acquired.done()

    qm.release_gpu_slot(target)
    await asyncio.sleep(0)  # yield so _batch_waiter can acquire

    assert batch_acquired.done()
    assert batch_acquired.result() is True
    qm.release_gpu_slot(target)


@pytest.mark.asyncio
async def test_queue_depth_tracks_running(qm):
    """queue_depth reports per-GPU running counts from tracked requests."""
    target = "http://gpu0:11434"
    d0 = qm.queue_depth()
    assert d0["running"] == 0

    tid = qm.track_request("test", "generate", "m",
                            qm.PRIORITY_INTERACTIVE, target)
    await qm.acquire_gpu_slot(target, qm.PRIORITY_INTERACTIVE, tid)
    d1 = qm.queue_depth()
    assert d1["running"] == 1
    assert d1["gpus"][target]["running"] == 1

    qm.release_gpu_slot(target, tid)


# ── Active queue ordering ────────────────────────────────────────────────────


def test_active_queue_sorted_by_submitted_at(qm):
    """active_queue returns entries sorted by submission time."""
    t1 = qm.track_request("a", "chat", "m", qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    t2 = qm.track_request("b", "generate", "m", qm.PRIORITY_BATCH, "http://gpu1:11435")
    queue = qm.active_queue()
    assert queue[0]["id"] == t1
    assert queue[1]["id"] == t2


# ── Batch job tests ──────────────────────────────────────────────────────────


def _fresh_qm(tmp_path, monkeypatch, extra_env=None):
    """Re-import queue_manager with an isolated DB and optional env overrides."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    for mod in list(sys.modules.keys()):
        if mod in ("queue_manager", "db", "config", "gpu_router"):
            sys.modules.pop(mod, None)
    import queue_manager
    return queue_manager


def test_submit_batch_job_returns_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in ["db", "config", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager as qm
    job_id = qm.submit_batch_job("test_app", "qwen3:32b", "hello world")
    assert len(job_id) == 36   # UUID format


def test_get_job_status_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in ["db", "config", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager as qm
    assert qm.get_job_status("nonexistent-id") is None


def test_get_job_result_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in ["db", "config", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager as qm
    assert qm.get_job_result("nonexistent-id") is None


def test_submit_and_retrieve_job_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in ["db", "config", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager as qm
    job_id = qm.submit_batch_job("parsival", "qwen3:32b", "analyze this")
    status = qm.get_job_status(job_id)
    assert status is not None
    assert status["id"] == job_id
    assert status["status"] == "queued"
    assert status["source_app"] == "parsival"
    assert status["model"] == "qwen3:32b"


@pytest.mark.asyncio
async def test_submit_batch_job_creates_job(tmp_path, monkeypatch):
    """submit_batch_job creates a job in the DB."""
    qm = _fresh_qm(tmp_path, monkeypatch)
    import db
    job_id = qm.submit_batch_job("test", "model", "hello")
    job = db.get_batch_job(job_id)
    assert job is not None
    assert job["source_app"] == "test"
    assert job["model"] == "model"


def test_batch_submit_rejects_oversized_prompt(tmp_path, monkeypatch):
    """POST /api/batch/submit returns 422 when prompt exceeds BATCH_MAX_PROMPT_LEN."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("BATCH_MAX_PROMPT_LEN", "50")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    import app as app_mod
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/batch/submit", json={
        "source_app": "test",
        "prompt": "x" * 51,
    })
    assert resp.status_code == 422
    assert "maximum length" in resp.json()["detail"]


def test_batch_submit_accepts_prompt_within_limit(tmp_path, monkeypatch):
    """POST /api/batch/submit accepts a prompt that fits within BATCH_MAX_PROMPT_LEN."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("BATCH_MAX_PROMPT_LEN", "50")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    import app as app_mod
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/batch/submit", json={
        "source_app": "test",
        "prompt": "x" * 50,
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── Queue change callback ────────────────────────────────────────────────────


def test_queue_change_callback_fires(qm):
    """set_queue_change_callback is called on track/release."""
    calls = []
    qm.set_queue_change_callback(lambda: calls.append(1))
    qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE, "http://gpu0:11434")
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_timeout_marks_tracked_as_failed(monkeypatch, qm):
    """Interactive timeout should mark the tracked entry as failed."""
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "0")
    for mod in list(sys.modules.keys()):
        if mod in ("config", "queue_manager"):
            del sys.modules[mod]
    import queue_manager as qm2
    target = "http://gpu0:11434"
    # Occupy the slot
    await qm2.acquire_gpu_slot(target, qm2.PRIORITY_INTERACTIVE)
    # Track and try to acquire — will timeout
    tid = qm2.track_request("test", "generate", "m",
                             qm2.PRIORITY_INTERACTIVE, target)
    result = await qm2.acquire_gpu_slot(target, qm2.PRIORITY_INTERACTIVE, tid)
    assert result is False
    queue = qm2.active_queue()
    failed_entries = [e for e in queue if e["id"] == tid]
    assert len(failed_entries) == 1
    assert failed_entries[0]["status"] == "failed"
    qm2.release_gpu_slot(target)
