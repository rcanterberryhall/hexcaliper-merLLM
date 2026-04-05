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


# ── GPU slot tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_and_release_slot(qm):
    """Acquiring and then releasing a slot should leave it free."""
    target = "http://gpu0:11434"
    acquired = await qm.acquire_gpu_slot(target, qm.PRIORITY_INTERACTIVE)
    assert acquired is True
    assert qm.queue_depth()["gpus"][target]["in_flight"] == 1
    qm.release_gpu_slot(target)
    assert qm.queue_depth()["gpus"][target]["in_flight"] == 0


@pytest.mark.asyncio
async def test_two_gpus_independent(qm):
    """Slots for different GPU targets are completely independent."""
    gpu0 = "http://gpu0:11434"
    gpu1 = "http://gpu1:11435"
    ok0 = await qm.acquire_gpu_slot(gpu0, qm.PRIORITY_INTERACTIVE)
    ok1 = await qm.acquire_gpu_slot(gpu1, qm.PRIORITY_INTERACTIVE)
    assert ok0 is True
    assert ok1 is True
    assert qm.queue_depth()["gpus"][gpu0]["in_flight"] == 1
    assert qm.queue_depth()["gpus"][gpu1]["in_flight"] == 1
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
async def test_queue_depth_reports_in_flight(qm):
    """queue_depth includes per-GPU in_flight counts."""
    target = "http://gpu0:11434"
    depth_before = qm.queue_depth()
    assert depth_before["gpus"].get(target, {}).get("in_flight", 0) == 0

    await qm.acquire_gpu_slot(target, qm.PRIORITY_INTERACTIVE)
    assert qm.queue_depth()["gpus"][target]["in_flight"] == 1
    qm.release_gpu_slot(target)
    assert qm.queue_depth()["gpus"][target]["in_flight"] == 0
