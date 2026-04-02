"""Tests for queue_manager.py — priority queue behaviour."""
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
        if mod in ("queue_manager", "config", "db", "mode_manager", "geoip"):
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
    assert d["total"] == 0


@pytest.mark.asyncio
async def test_enqueue_increases_depth(qm):
    f = await qm.enqueue("http://localhost:11434", {"model": "test"})
    assert qm.queue_depth()["total"] == 1
    f.cancel()


@pytest.mark.asyncio
async def test_priority_ordering(qm):
    """Batch requests should be dequeued after interactive ones."""
    loop = asyncio.get_event_loop()

    # Enqueue batch first, then interactive
    f_batch = await qm.enqueue("http://localhost:11434", {"model": "b"}, qm.PRIORITY_BATCH)
    f_inter = await qm.enqueue("http://localhost:11434", {"model": "i"}, qm.PRIORITY_INTERACTIVE)

    # Drain internal queue to inspect order
    queue = qm._queue
    first  = await queue.get()
    second = await queue.get()

    assert first.priority  == qm.PRIORITY_INTERACTIVE
    assert second.priority == qm.PRIORITY_BATCH

    f_batch.cancel()
    f_inter.cancel()


@pytest.mark.asyncio
async def test_enqueue_returns_future(qm):
    f = await qm.enqueue("http://localhost:11434", {"model": "x"})
    assert isinstance(f, asyncio.Future)
    f.cancel()


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


def test_signal_night_mode(qm):
    """signal_night_mode should not raise."""
    qm.signal_night_mode()
    assert qm._batch_event.is_set()
