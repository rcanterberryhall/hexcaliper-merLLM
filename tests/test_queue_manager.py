"""Tests for queue_manager.py — late-binding dispatcher with priority pipes."""
import asyncio
import os
import sys
import time

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture(autouse=True)
def reset_queue(tmp_path, monkeypatch):
    """Re-import queue_manager fresh for each test to reset module state."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("queue_manager", "config", "db", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    yield
    # Cancel any background dispatcher task before tearing down the module.
    qm = sys.modules.get("queue_manager")
    if qm is not None and getattr(qm, "_dispatcher_task", None):
        task = qm._dispatcher_task
        try:
            if not task.done():
                task.cancel()
        except RuntimeError:
            # The test's event loop has already closed — nothing to clean up.
            pass
    for mod in list(sys.modules.keys()):
        if mod in ("queue_manager", "config", "db", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)


@pytest.fixture
def qm():
    import queue_manager
    return queue_manager


@pytest.fixture
def gpu_urls():
    """Return the configured GPU target URLs after env reset."""
    import config
    return config.OLLAMA_0_URL, config.OLLAMA_1_URL


@pytest.fixture
def patch_reload(monkeypatch):
    """Stub out the model swap so the dispatcher does not hit httpx."""
    import gpu_router

    async def _noop_reload(gpu, model):
        gpu.model = model

    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)


# ── Basic constants and empty state ──────────────────────────────────────────


def test_priority_constants(qm):
    assert qm.PRIORITY_INTERACTIVE == 0
    assert qm.PRIORITY_BATCH == 1
    assert qm.PRIORITY_INTERACTIVE < qm.PRIORITY_BATCH


def test_queue_depth_empty(qm):
    d = qm.queue_depth()
    assert d["queued"] == 0
    assert d["running"] == 0
    assert d["total"] == 0


def test_pipe_depth_empty(qm):
    p = qm.pipe_depth()
    assert p["interactive"] == 0
    assert p["batch"] == 0


# ── Request tracking tests ───────────────────────────────────────────────────


def test_track_request_returns_id(qm):
    tid = qm.track_request("test", "generate", "qwen3:32b",
                           qm.PRIORITY_INTERACTIVE)
    assert len(tid) == 36  # UUID format


def test_track_request_appears_in_active_queue(qm):
    tid = qm.track_request("lancellmot", "chat", "qwen3:32b",
                           qm.PRIORITY_INTERACTIVE)
    queue = qm.active_queue()
    assert len(queue) == 1
    entry = queue[0]
    assert entry["id"] == tid
    assert entry["source"] == "lancellmot"
    assert entry["request_type"] == "chat"
    assert entry["model"] == "qwen3:32b"
    assert entry["priority"] == "interactive"
    assert entry["status"] == "queued"
    # Late binding: target is unset until dispatch.
    assert entry["target"] is None


def test_track_request_shows_in_queue_depth(qm):
    qm.track_request("test", "generate", "m", qm.PRIORITY_INTERACTIVE)
    d = qm.queue_depth()
    assert d["queued"] == 1
    assert d["total"] == 1


# ── Late-binding dispatcher tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_dispatch_assigns_target_and_marks_running(
        qm, gpu_urls, patch_reload):
    tid = qm.track_request("test", "generate", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert target in gpu_urls
    queue = qm.active_queue()
    entry = next(e for e in queue if e["id"] == tid)
    assert entry["status"] == "running"
    assert entry["started_at"] is not None
    assert entry["target"] == target
    assert qm.gpu_slot_busy(target) is True
    qm.release(tid)


@pytest.mark.asyncio
async def test_release_transitions_to_completed_and_frees_gpu(
        qm, patch_reload):
    tid = qm.track_request("test", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    qm.release(tid)
    queue = qm.active_queue()
    entry = next(e for e in queue if e["id"] == tid)
    assert entry["status"] == "completed"
    assert entry["completed_at"] is not None
    assert qm.gpu_slot_busy(target) is False


@pytest.mark.asyncio
async def test_fail_request_frees_gpu(qm, patch_reload):
    tid = qm.track_request("test", "generate", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    qm.fail_request(tid, "Ollama OOM")
    queue = qm.active_queue()
    entry = next(e for e in queue if e["id"] == tid)
    assert entry["status"] == "failed"
    assert entry["error"] == "Ollama OOM"
    assert qm.gpu_slot_busy(target) is False


@pytest.mark.asyncio
async def test_two_gpus_dispatched_independently(qm, gpu_urls, patch_reload):
    """Two requests can run concurrently — one per GPU."""
    tid_a = qm.track_request("a", "chat", "m", qm.PRIORITY_INTERACTIVE)
    tid_b = qm.track_request("b", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target_a = await qm.wait_for_dispatch(tid_a)
    target_b = await qm.wait_for_dispatch(tid_b)
    assert {target_a, target_b} == set(gpu_urls)
    assert qm.gpu_slot_busy(target_a) is True
    assert qm.gpu_slot_busy(target_b) is True
    qm.release(tid_a)
    qm.release(tid_b)


@pytest.mark.asyncio
async def test_third_request_waits_until_release(qm, patch_reload):
    """With both GPUs busy, the third request waits for a release."""
    tid_a = qm.track_request("a", "chat", "m", qm.PRIORITY_INTERACTIVE)
    tid_b = qm.track_request("b", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target_a = await qm.wait_for_dispatch(tid_a)
    target_b = await qm.wait_for_dispatch(tid_b)

    tid_c = qm.track_request("c", "chat", "m", qm.PRIORITY_INTERACTIVE)
    waiter = asyncio.create_task(qm.wait_for_dispatch(tid_c))
    await asyncio.sleep(0.05)
    assert not waiter.done()
    assert qm.pipe_depth()["interactive"] == 1

    qm.release(tid_a)
    target_c = await asyncio.wait_for(waiter, timeout=2.0)
    assert target_c == target_a
    qm.release(tid_b)
    qm.release(tid_c)


@pytest.mark.asyncio
async def test_interactive_dispatch_timeout(monkeypatch):
    """Interactive request raises DispatchTimeout when no GPU frees up."""
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "1")
    for mod in list(sys.modules.keys()):
        if mod in ("config", "queue_manager"):
            sys.modules.pop(mod, None)
    import queue_manager as qm2
    import gpu_router

    async def _noop_reload(gpu, model):
        gpu.model = model
    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

    # Saturate both GPUs via the reclaim-loop reservation hook so the
    # dispatcher cannot pick either one.
    for url in qm2._gpu_targets():
        assert qm2.reserve_gpu(url) is True

    tid = qm2.track_request("c", "chat", "m", qm2.PRIORITY_INTERACTIVE)
    with pytest.raises(qm2.DispatchTimeout):
        await qm2.wait_for_dispatch(tid)

    queue = qm2.active_queue()
    failed = [e for e in queue if e["id"] == tid]
    assert len(failed) == 1
    assert failed[0]["status"] == "failed"

    for url in qm2._gpu_targets():
        qm2.unreserve_gpu(url)


@pytest.mark.asyncio
async def test_interactive_drains_before_batch(qm, patch_reload):
    """Strict priority: interactive head must dispatch before any batch."""
    # Saturate both GPUs.
    tid_a = qm.track_request("a", "chat", "m", qm.PRIORITY_INTERACTIVE)
    tid_b = qm.track_request("b", "chat", "m", qm.PRIORITY_INTERACTIVE)
    await qm.wait_for_dispatch(tid_a)
    await qm.wait_for_dispatch(tid_b)

    # Enqueue a batch and an interactive — order: batch first, interactive
    # second. The interactive must still be dispatched first when a slot
    # frees, because the high-priority pipe drains before the low-pri pipe.
    tid_batch = qm.track_request("bat", "generate", "m", qm.PRIORITY_BATCH)
    batch_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_batch))
    await asyncio.sleep(0)  # let the batch land in its pipe

    tid_int = qm.track_request("int", "chat", "m", qm.PRIORITY_INTERACTIVE)
    int_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_int))
    await asyncio.sleep(0)

    qm.release(tid_a)

    # Interactive must resolve first.
    done, pending = await asyncio.wait(
        [int_waiter, batch_waiter],
        timeout=2.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert int_waiter in done
    assert batch_waiter in pending

    qm.release(tid_int)
    # Now batch can dispatch.
    await asyncio.wait_for(batch_waiter, timeout=2.0)
    qm.release(tid_b)
    qm.release(tid_batch)


@pytest.mark.asyncio
async def test_dispatcher_prefers_affinity(qm, gpu_urls, patch_reload):
    """When one GPU already holds the requested model, that GPU is chosen."""
    import gpu_router
    gpu_router._init_gpus()
    url0, url1 = gpu_urls
    gpu_router._gpus[url0].model = "default-model"
    gpu_router._gpus[url1].model = "rare-model"

    tid = qm.track_request("test", "chat", "rare-model",
                           qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert target == url1
    qm.release(tid)


@pytest.mark.asyncio
async def test_dispatcher_swaps_when_no_affinity(qm, patch_reload):
    """No GPU holds the model → dispatcher picks one and swaps it in."""
    import gpu_router
    gpu_router._init_gpus()

    tid = qm.track_request("test", "chat", "fresh-model:13b",
                           qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert gpu_router._gpus[target].model == "fresh-model:13b"
    qm.release(tid)


# ── Reclaim cooperation ──────────────────────────────────────────────────────


def test_reserve_gpu_blocks_when_busy(qm, gpu_urls):
    url0, _ = gpu_urls
    assert qm.reserve_gpu(url0) is True
    assert qm.reserve_gpu(url0) is False
    qm.unreserve_gpu(url0)
    assert qm.gpu_slot_busy(url0) is False


@pytest.mark.asyncio
async def test_dispatch_skipped_while_reserved(qm, gpu_urls, patch_reload):
    """A GPU reserved by the reclaim loop is not picked by the dispatcher."""
    url0, url1 = gpu_urls
    assert qm.reserve_gpu(url0) is True

    tid = qm.track_request("test", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert target == url1
    qm.release(tid)
    qm.unreserve_gpu(url0)


# ── Active queue ordering ────────────────────────────────────────────────────


def test_active_queue_sorted_by_submitted_at(qm):
    t1 = qm.track_request("a", "chat", "m", qm.PRIORITY_INTERACTIVE)
    t2 = qm.track_request("b", "generate", "m", qm.PRIORITY_BATCH)
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
    qm = _fresh_qm(tmp_path, monkeypatch)
    job_id = qm.submit_batch_job("test_app", "qwen3:32b", "hello world")
    assert len(job_id) == 36   # UUID format


def test_get_job_status_missing(tmp_path, monkeypatch):
    qm = _fresh_qm(tmp_path, monkeypatch)
    assert qm.get_job_status("nonexistent-id") is None


def test_get_job_result_missing(tmp_path, monkeypatch):
    qm = _fresh_qm(tmp_path, monkeypatch)
    assert qm.get_job_result("nonexistent-id") is None


def test_submit_and_retrieve_job_status(tmp_path, monkeypatch):
    qm = _fresh_qm(tmp_path, monkeypatch)
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


@pytest.mark.asyncio
async def test_run_batch_job_marks_activity_during_dispatch(tmp_path, monkeypatch):
    """Batch jobs must populate ``_activity`` for the duration of the dispatch.

    Regression test for the bug where ``_run_batch_job_async`` made a raw
    httpx POST and skipped the per-instance activity tracker, leaving the
    Ollama Instances card and the SSE stream reporting "idle" while a GPU
    was being hammered by parsival re-analyze.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "queue_manager", "db", "config", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)

    import app as app_mod
    import queue_manager as qm
    import gpu_router

    # Stub the model swap so the dispatcher does not hit httpx itself.
    async def _noop_reload(gpu, model):
        gpu.model = model
    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

    # Fake httpx client that blocks inside post() until released, so the
    # test can observe ``_activity`` while the batch job is mid-flight.
    in_flight = asyncio.Event()
    release_post = asyncio.Event()

    class _FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "ok"}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, json=None):
            in_flight.set()
            await release_post.wait()
            return _FakeResp()

    monkeypatch.setattr(qm.httpx, "AsyncClient", _FakeClient)

    # Submit a batch job and let the background task run.
    qm.submit_batch_job("test", "qwen3:32b", "hello")

    # Wait until the fake post is in flight, then assert at least one GPU
    # has a non-None activity entry. Without the fix, both stay None forever.
    await asyncio.wait_for(in_flight.wait(), timeout=5.0)
    active_gpus = [k for k, v in app_mod._activity.items() if v is not None]
    assert active_gpus, "no GPU shows as active during batch dispatch"
    entry = app_mod._activity[active_gpus[0]]
    assert entry["model"] == "qwen3:32b"
    assert entry["endpoint"] == "/api/generate"

    # Let the post complete and the cleanup run.
    release_post.set()
    for _ in range(100):
        if all(v is None for v in app_mod._activity.values()):
            break
        await asyncio.sleep(0.02)
    assert all(v is None for v in app_mod._activity.values()), \
        "_activity should be cleared after batch job completes"


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
    qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
    assert len(calls) >= 1
