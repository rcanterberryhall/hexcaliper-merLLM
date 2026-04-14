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
    # Cancel any background dispatcher and watchdog tasks before tearing
    # down the module.
    qm = sys.modules.get("queue_manager")
    for attr in ("_dispatcher_task", "_watchdog_task"):
        if qm is None:
            break
        task = getattr(qm, attr, None)
        if task is None:
            continue
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
        return True

    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)


# ── Basic constants and empty state ──────────────────────────────────────────


def test_priority_enum_values(qm):
    """Five buckets, CHAT highest, BACKGROUND lowest, strict int ordering."""
    assert qm.Priority.CHAT == 0
    assert qm.Priority.EMBEDDINGS == 1
    assert qm.Priority.SHORT == 2
    assert qm.Priority.FEEDBACK == 3
    assert qm.Priority.BACKGROUND == 4
    assert qm.Priority.CHAT < qm.Priority.EMBEDDINGS < qm.Priority.SHORT \
        < qm.Priority.FEEDBACK < qm.Priority.BACKGROUND


def test_back_compat_priority_aliases(qm):
    """Old PRIORITY_INTERACTIVE / PRIORITY_BATCH names still point at the
    right buckets so parsival and lancellmot keep working during migration."""
    assert qm.PRIORITY_INTERACTIVE == qm.Priority.CHAT
    assert qm.PRIORITY_BATCH == qm.Priority.BACKGROUND


def test_priority_from_name_canonical(qm):
    assert qm.priority_from_name("chat") == qm.Priority.CHAT
    assert qm.priority_from_name("embeddings") == qm.Priority.EMBEDDINGS
    assert qm.priority_from_name("short") == qm.Priority.SHORT
    assert qm.priority_from_name("feedback") == qm.Priority.FEEDBACK
    assert qm.priority_from_name("background") == qm.Priority.BACKGROUND


def test_priority_from_name_legacy_aliases(qm):
    """Legacy header values accepted during the one-release migration."""
    assert qm.priority_from_name("interactive") == qm.Priority.CHAT
    assert qm.priority_from_name("batch") == qm.Priority.BACKGROUND
    assert qm.priority_from_name("BATCH") == qm.Priority.BACKGROUND


def test_priority_from_name_unknown_falls_back_to_background(qm):
    """Typos must not silently escalate; unknown non-empty values → BACKGROUND."""
    assert qm.priority_from_name("urgent") == qm.Priority.BACKGROUND
    assert qm.priority_from_name("high") == qm.Priority.BACKGROUND
    # None/blank → default parameter, which defaults to BACKGROUND here.
    assert qm.priority_from_name(None) == qm.Priority.BACKGROUND
    assert qm.priority_from_name("") == qm.Priority.BACKGROUND


def test_queue_depth_empty(qm):
    d = qm.queue_depth()
    assert d["queued"] == 0
    assert d["running"] == 0
    assert d["total"] == 0


def test_pipe_depth_empty(qm):
    p = qm.pipe_depth()
    # All five buckets, plus back-compat aliases.
    assert p["chat"] == 0
    assert p["embeddings"] == 0
    assert p["short"] == 0
    assert p["feedback"] == 0
    assert p["background"] == 0
    assert p["interactive"] == 0
    assert p["batch"] == 0


# ── Request tracking tests ───────────────────────────────────────────────────


def test_track_request_returns_id(qm):
    tid = qm.track_request("test", "generate", "qwen3:32b",
                           qm.PRIORITY_INTERACTIVE)
    assert len(tid) == 36  # UUID format


def test_track_request_appears_in_active_queue(qm):
    tid = qm.track_request("lancellmot", "chat", "qwen3:32b",
                           qm.Priority.CHAT)
    queue = qm.active_queue()
    assert len(queue) == 1
    entry = queue[0]
    assert entry["id"] == tid
    assert entry["source"] == "lancellmot"
    assert entry["request_type"] == "chat"
    assert entry["model"] == "qwen3:32b"
    assert entry["priority"] == "chat"
    assert entry["status"] == "queued"
    # Late binding: target is unset until dispatch.
    assert entry["target"] is None


def test_active_queue_reports_each_bucket_name(qm):
    """The priority field in active_queue() uses the canonical bucket name."""
    qm.track_request("a", "chat", "m", qm.Priority.CHAT)
    qm.track_request("b", "generate", "m", qm.Priority.SHORT)
    qm.track_request("c", "generate", "m", qm.Priority.FEEDBACK)
    qm.track_request("d", "generate", "m", qm.Priority.BACKGROUND)
    by_source = {e["source"]: e["priority"] for e in qm.active_queue()}
    assert by_source == {
        "a": "chat",
        "b": "short",
        "c": "feedback",
        "d": "background",
    }


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
async def test_watchdog_reclaims_wedged_slot(qm, monkeypatch, patch_reload):
    """A dispatched request that exceeds SLOT_MAX_WALL_SECONDS is force-failed
    by the watchdog so the GPU slot returns to the idle pool.

    Regression: 2026-04-10 incident where two parsival ``feedback`` entries
    pinned both GPUs for ~13 minutes — Ollama showed no models loaded and
    no actual work was running, but ``release()`` was never called and
    the strict-priority dispatcher refused to dispatch the 295 background
    batches queued behind them. The watchdog now reclaims such slots.
    """
    import config
    monkeypatch.setattr(config, "SLOT_MAX_WALL_SECONDS", 1)
    monkeypatch.setattr(config, "WATCHDOG_INTERVAL_SECONDS", 1)

    tid = qm.track_request("parsival", "chat", "qwen3:32b",
                           qm.Priority.FEEDBACK)
    target = await qm.wait_for_dispatch(tid)
    assert qm.gpu_slot_busy(target) is True

    # Backdate started_at so the watchdog sees the slot as exceeding budget.
    qm._tracked[tid].started_at = time.time() - 5

    # Wait long enough for one watchdog tick.
    for _ in range(50):
        await asyncio.sleep(0.1)
        if not qm.gpu_slot_busy(target):
            break

    entry = qm._tracked.get(tid) or next(
        (e for e in qm.active_queue() if e["id"] == tid), None
    )
    assert qm.gpu_slot_busy(target) is False, "watchdog must release the GPU"
    if entry is not None:
        # Entry may still be present briefly before _remove_tracked fires.
        status = entry["status"] if isinstance(entry, dict) else entry.status
        assert status == "failed"


@pytest.mark.asyncio
async def test_cancel_tracked_removes_pending_waiter(qm, patch_reload):
    """A ``queued`` entry that has not yet been dispatched is pulled out
    of its bucket, its future resolved with an exception, and the entry
    is marked ``failed`` — so the dispatcher can never later pop a dead
    waiter and wedge a GPU slot.

    Regression: 2026-04-11 feedback-slot wedge.  A parsival ``feedback``
    caller timed out at 60 s while its ``_PendingRequest`` was still in
    the bucket. Starlette closed the proxy's async generator, which
    threw ``GeneratorExit`` at ``await wait_for_dispatch`` — but the
    generator had no ``finally`` so the pending entry stayed in the
    bucket. The dispatcher eventually popped it, marked ``_gpu_busy``,
    and no ``release`` ever fired.  This test enforces the contract
    the new ``finally`` relies on.
    """
    # Saturate both GPUs so the feedback entry must wait.
    busy_a = qm.track_request("a", "chat", "m", qm.Priority.CHAT)
    busy_b = qm.track_request("b", "chat", "m", qm.Priority.CHAT)
    await qm.wait_for_dispatch(busy_a)
    await qm.wait_for_dispatch(busy_b)

    tid = qm.track_request("parsival", "generate", "qwen3:32b",
                           qm.Priority.FEEDBACK)
    waiter = asyncio.create_task(qm.wait_for_dispatch(tid))
    await asyncio.sleep(0.05)  # let the waiter register in its bucket
    assert not waiter.done()
    assert qm.pipe_depth()["feedback"] == 1

    qm.cancel_tracked(tid, reason="client_disconnect")

    # The waiter's future must raise, not just hang forever, so the
    # handler's finally runs on both ends of the dispatcher.
    with pytest.raises(RuntimeError, match="cancelled"):
        await asyncio.wait_for(waiter, timeout=1.0)

    assert qm.pipe_depth()["feedback"] == 0
    entry = next((e for e in qm.active_queue() if e["id"] == tid), None)
    if entry is not None:
        assert entry["status"] == "failed"
        assert entry["error"] == "client_disconnect"

    # Critical invariant: releasing one of the saturating slots must not
    # hand it off to the cancelled (dead) waiter.
    qm.release(busy_a)
    await asyncio.sleep(0.05)
    # The dispatcher has nothing feedback-priority to pick up now.
    assert qm.pipe_depth()["feedback"] == 0

    qm.release(busy_b)


@pytest.mark.asyncio
async def test_cancel_tracked_releases_running_slot(qm, patch_reload):
    """A ``running`` entry routed through ``cancel_tracked`` releases its
    GPU slot via ``fail_request``. Covers the ``_queued_stream`` finally's
    handling of the race where ``_dispatch`` fires after the generator
    is already being closed by the client."""
    tid = qm.track_request("parsival", "generate", "qwen3:32b",
                           qm.Priority.FEEDBACK)
    target = await qm.wait_for_dispatch(tid)
    assert qm.gpu_slot_busy(target) is True

    qm.cancel_tracked(tid, reason="client_disconnect_or_error")

    assert qm.gpu_slot_busy(target) is False
    entry = next((e for e in qm.active_queue() if e["id"] == tid), None)
    if entry is not None:
        assert entry["status"] == "failed"


@pytest.mark.asyncio
async def test_cancel_tracked_is_idempotent_on_terminal(qm, patch_reload):
    """Calling ``cancel_tracked`` on a completed or unknown tid is a
    no-op — finally blocks must be able to call it unconditionally."""
    tid = qm.track_request("parsival", "chat", "m", qm.Priority.CHAT)
    target = await qm.wait_for_dispatch(tid)
    qm.release(tid)
    # Completed entries are scheduled for removal but may still be present.
    qm.cancel_tracked(tid, reason="safe_noop")
    qm.cancel_tracked("never-existed-uuid", reason="safe_noop")
    # GPU must still be idle after these no-ops.
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
    assert qm.pipe_depth()["chat"] == 1
    # Back-compat alias still works.
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
        return True
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
    """Strict priority: CHAT head must dispatch before any BACKGROUND."""
    # Saturate both GPUs.
    tid_a = qm.track_request("a", "chat", "m", qm.Priority.CHAT)
    tid_b = qm.track_request("b", "chat", "m", qm.Priority.CHAT)
    await qm.wait_for_dispatch(tid_a)
    await qm.wait_for_dispatch(tid_b)

    # Enqueue a BACKGROUND then a CHAT. Order of submission: background
    # first, chat second. The chat must still be dispatched first when a
    # slot frees, because CHAT drains before BACKGROUND.
    tid_bg = qm.track_request("bg", "generate", "m", qm.Priority.BACKGROUND)
    bg_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_bg))
    await asyncio.sleep(0)  # let the background request land in its bucket

    tid_chat = qm.track_request("chat", "chat", "m", qm.Priority.CHAT)
    chat_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_chat))
    await asyncio.sleep(0)

    qm.release(tid_a)

    # CHAT must resolve first.
    done, pending = await asyncio.wait(
        [chat_waiter, bg_waiter],
        timeout=2.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert chat_waiter in done
    assert bg_waiter in pending

    qm.release(tid_chat)
    # Now BACKGROUND can dispatch.
    await asyncio.wait_for(bg_waiter, timeout=2.0)
    qm.release(tid_b)
    qm.release(tid_bg)


@pytest.mark.asyncio
async def test_strict_priority_drain_top_down(qm, patch_reload):
    """Every bucket drains before any lower bucket dispatches.

    Submit one request in CHAT, SHORT, FEEDBACK, BACKGROUND (EMBEDDINGS
    stays empty for this test — it has dedicated coverage via the
    proxy_embeddings auto-classification path). Saturate both GPUs with
    dummy CHAT requests first. Then release slots one at a time and assert
    the dispatch order is CHAT → SHORT → FEEDBACK → BACKGROUND regardless
    of the order the requests were submitted in.
    """
    # Saturate both GPUs with holder requests.
    tid_hold_a = qm.track_request("hold_a", "chat", "m", qm.Priority.CHAT)
    tid_hold_b = qm.track_request("hold_b", "chat", "m", qm.Priority.CHAT)
    await qm.wait_for_dispatch(tid_hold_a)
    await qm.wait_for_dispatch(tid_hold_b)

    # Submit one item per priority in *reverse* order — the point is to
    # prove dispatch order follows priority, not submission order.
    tid_bg = qm.track_request("bg", "generate", "m", qm.Priority.BACKGROUND)
    bg_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_bg))
    await asyncio.sleep(0)

    tid_fb = qm.track_request("fb", "generate", "m", qm.Priority.FEEDBACK)
    fb_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_fb))
    await asyncio.sleep(0)

    tid_sh = qm.track_request("sh", "generate", "m", qm.Priority.SHORT)
    sh_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_sh))
    await asyncio.sleep(0)

    tid_ch = qm.track_request("ch", "chat", "m", qm.Priority.CHAT)
    ch_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_ch))
    await asyncio.sleep(0)

    depths = qm.pipe_depth()
    assert depths["chat"] == 1
    assert depths["embeddings"] == 0
    assert depths["short"] == 1
    assert depths["feedback"] == 1
    assert depths["background"] == 1

    # Release one slot at a time and verify the next-highest-priority head
    # is the one that dispatches. After each step, assert the remaining
    # waiters are still pending — a BACKGROUND waiter resolving before its
    # turn would indicate a priority inversion.
    def _assert_pending(*waiters):
        for w in waiters:
            assert not w.done()

    qm.release(tid_hold_a)
    await asyncio.wait_for(ch_waiter, timeout=2.0)
    _assert_pending(sh_waiter, fb_waiter, bg_waiter)

    qm.release(tid_hold_b)
    await asyncio.wait_for(sh_waiter, timeout=2.0)
    _assert_pending(fb_waiter, bg_waiter)

    qm.release(tid_ch)
    await asyncio.wait_for(fb_waiter, timeout=2.0)
    _assert_pending(bg_waiter)

    qm.release(tid_sh)
    await asyncio.wait_for(bg_waiter, timeout=2.0)

    qm.release(tid_fb)
    qm.release(tid_bg)


@pytest.mark.asyncio
async def test_background_cannot_preempt_waiting_chat(qm, patch_reload):
    """Regression guard: a BACKGROUND item waiting in the queue must not
    run while a CHAT item is also waiting, even if the BACKGROUND was
    enqueued first and the CHAT arrives only milliseconds before a slot
    frees."""
    # Saturate both GPUs.
    tid_a = qm.track_request("a", "chat", "m", qm.Priority.CHAT)
    tid_b = qm.track_request("b", "chat", "m", qm.Priority.CHAT)
    await qm.wait_for_dispatch(tid_a)
    await qm.wait_for_dispatch(tid_b)

    # Background arrives first and waits.
    tid_bg = qm.track_request("bg", "generate", "m", qm.Priority.BACKGROUND)
    bg_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_bg))
    await asyncio.sleep(0.01)

    # Chat arrives slightly later but is higher priority.
    tid_chat = qm.track_request("chat", "chat", "m", qm.Priority.CHAT)
    chat_waiter = asyncio.create_task(qm.wait_for_dispatch(tid_chat))
    await asyncio.sleep(0.01)

    # Only release one GPU — exactly one slot is available.
    qm.release(tid_a)

    # Chat must win despite arriving second.
    done, pending = await asyncio.wait(
        [chat_waiter, bg_waiter],
        timeout=2.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert chat_waiter in done
    assert bg_waiter in pending

    qm.release(tid_chat)
    await asyncio.wait_for(bg_waiter, timeout=2.0)
    qm.release(tid_b)
    qm.release(tid_bg)


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


@pytest.mark.asyncio
async def test_dispatcher_fails_request_when_reload_fails(qm, monkeypatch):
    """If _reload_model returns False, the dispatcher must fail the request
    rather than hand it off to a GPU whose model did not actually load.

    Regression: 2026-04-14 upload 500 race. _reload_model used to silently
    claim success for embedding models (wrong endpoint), and the dispatcher
    would trust it and proxy to an unloaded GPU, where concurrent requests
    raced on the real load and Ollama 500'd.
    """
    import gpu_router
    gpu_router._init_gpus()

    async def _failing_reload(gpu, model):
        return False
    monkeypatch.setattr(gpu_router, "_reload_model", _failing_reload)

    tid = qm.track_request("test", "chat", "fresh-model:13b",
                           qm.PRIORITY_INTERACTIVE)
    with pytest.raises(RuntimeError, match="failed to load model"):
        await qm.wait_for_dispatch(tid)


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
        return True
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


@pytest.mark.asyncio
async def test_empty_response_fails_job_without_retry(tmp_path, monkeypatch):
    """A blank ``response`` from Ollama must mark the job ``failed``
    immediately — no retries, since the prompt is deterministic.

    Before the fix, an empty response string was stored as a successful
    result, leaving callers (hexcaliper extractor) unable to tell whether
    the model had nothing to say or the prompt was truncated past
    ``num_ctx``.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "queue_manager", "db", "config", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)

    import app as _  # noqa: F401 — imported for side effects
    import queue_manager as qm
    import db
    import gpu_router

    async def _noop_reload(gpu, model):
        gpu.model = model
        return True
    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

    class _FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "response": "",
                "done_reason": "stop",
                "prompt_eval_count": 17000,
            }

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, json=None):
            return _FakeResp()

    monkeypatch.setattr(qm.httpx, "AsyncClient", _FakeClient)

    job_id = qm.submit_batch_job("test", "qwen3:32b", "hello")

    for _ in range(200):
        job = db.get_batch_job(job_id)
        if job and job["status"] in ("failed", "completed"):
            break
        await asyncio.sleep(0.02)

    job = db.get_batch_job(job_id)
    assert job["status"] == "failed", \
        f"expected failed, got {job['status']} with result={job.get('result')!r}"
    assert job["retries"] == 0, "empty-response must not retry"
    assert "empty response" in (job["error"] or "").lower()
    assert "num_ctx" in (job["error"] or "")


@pytest.mark.asyncio
async def test_batch_runner_lifts_think_flag_to_top_level(tmp_path, monkeypatch):
    """``think`` must be sent as a top-level Ollama request field, not nested
    inside ``options``. Before the fix it was buried in options where Ollama
    silently ignored it, so qwen3:* kept reasoning, burned num_predict, and
    returned ``done_reason='length'`` with an empty response — the failure
    mode that killed the first 122 extractor jobs of the April re-ingest."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "queue_manager", "db", "config", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)

    import app as _  # noqa: F401
    import queue_manager as qm
    import db
    import gpu_router

    async def _noop_reload(gpu, model):
        gpu.model = model
        return True
    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

    captured: list[dict] = []

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
            captured.append(json)
            return _FakeResp()

    monkeypatch.setattr(qm.httpx, "AsyncClient", _FakeClient)

    # Caller nests ``think`` inside options — mimics lancellmot extractor.
    job_id = qm.submit_batch_job(
        "lancellmot", "qwen3:32b", "extract prompt",
        options={"think": False, "num_predict": 384, "num_ctx": 8192},
    )

    for _ in range(200):
        job = db.get_batch_job(job_id)
        if job and job["status"] in ("failed", "completed"):
            break
        await asyncio.sleep(0.02)

    assert captured, "no Ollama request was dispatched"
    body = captured[0]
    assert body.get("think") is False, \
        f"think must be top-level, got body={body!r}"
    assert "think" not in body["options"], \
        f"think must not be nested inside options, got {body['options']!r}"
    # Other option keys still survive.
    assert body["options"]["num_predict"] == 384
    assert body["options"]["num_ctx"]     == 8192


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


# ── Thermal pause integration ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thermally_paused_gpu_excluded_from_dispatch(
        qm, gpu_urls, patch_reload, monkeypatch):
    """A GPU marked thermally paused is not chosen by _best_candidate_for,
    so work routes to the other GPU even if the paused one is idle."""
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    import gpu_router
    gpu0, gpu1 = gpu_urls

    # Pause GPU 0 thermally.
    gpu_router.update_thermal_state(gpu0, 90.0)
    assert gpu_router.is_dispatchable(gpu0) is False
    assert gpu_router.is_dispatchable(gpu1) is True

    # A dispatched request must land on GPU 1.
    tid = qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert target == gpu1
    qm.release(tid)


@pytest.mark.asyncio
async def test_thermal_resume_wakes_dispatcher(
        qm, gpu_urls, patch_reload, monkeypatch):
    """When all GPUs are thermally paused, work queues. Clearing the pause
    on one GPU wakes the dispatcher and that request moves."""
    monkeypatch.setenv("GPU_TEMP_PAUSE_C", "85")
    monkeypatch.setenv("GPU_TEMP_RESUME_C", "60")
    import gpu_router
    gpu0, gpu1 = gpu_urls

    gpu_router.update_thermal_state(gpu0, 90.0)
    gpu_router.update_thermal_state(gpu1, 92.0)

    tid = qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
    # Request is queued, not dispatched.
    waiter = asyncio.create_task(qm.wait_for_dispatch(tid))
    await asyncio.sleep(0.05)
    assert not waiter.done()

    # Clear one GPU's thermal pause → dispatcher should wake and assign it.
    gpu_router.update_thermal_state(gpu0, 55.0)
    target = await asyncio.wait_for(waiter, timeout=2.0)
    assert target == gpu0
    qm.release(tid)


# ── Operator pause / resume ──────────────────────────────────────────────────


def test_set_paused_toggles_and_reports(qm):
    """is_paused / paused_since reflect the operator switch."""
    assert qm.is_paused() is False
    assert qm.paused_since() is None

    changed = qm.set_paused(True, persist=False)
    assert changed is True
    assert qm.is_paused() is True
    assert isinstance(qm.paused_since(), float)

    changed = qm.set_paused(True, persist=False)
    assert changed is False  # idempotent — second call is a no-op

    qm.set_paused(False, persist=False)
    assert qm.is_paused() is False
    assert qm.paused_since() is None


def test_set_paused_persists_to_settings(qm, tmp_path, monkeypatch):
    """Persisting the flag means a restart re-reads it from SQLite."""
    import db
    qm.set_paused(True, persist=True)
    assert db.get_settings().get("queue_paused") is True
    qm.set_paused(False, persist=True)
    assert db.get_settings().get("queue_paused") is False


@pytest.mark.asyncio
async def test_paused_queue_blocks_new_dispatch(qm, gpu_urls, patch_reload):
    """While paused, tracked requests queue but never dispatch until resumed."""
    gpu0, _ = gpu_urls
    qm.set_paused(True, persist=False)
    try:
        tid = qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
        waiter = asyncio.create_task(qm.wait_for_dispatch(tid))
        # Give the dispatcher plenty of chances to run — it should no-op.
        await asyncio.sleep(0.1)
        assert not waiter.done()
        assert qm._tracked[tid].status == "queued"
        # Cancel the waiter so the test doesn't hang on teardown.
        qm.cancel_tracked(tid, reason="test_cleanup")
        with pytest.raises(RuntimeError):
            await waiter
    finally:
        qm.set_paused(False, persist=False)


@pytest.mark.asyncio
async def test_resume_wakes_dispatcher_and_clears_backlog(
        qm, gpu_urls, patch_reload):
    """Unpausing wakes the dispatcher so queued heads dispatch immediately."""
    qm.set_paused(True, persist=False)
    tid = qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
    waiter = asyncio.create_task(qm.wait_for_dispatch(tid))
    await asyncio.sleep(0.05)
    assert not waiter.done()

    qm.set_paused(False, persist=False)
    target = await asyncio.wait_for(waiter, timeout=2.0)
    assert target in gpu_urls
    qm.release(tid)


@pytest.mark.asyncio
async def test_in_flight_request_unaffected_by_pause(
        qm, gpu_urls, patch_reload):
    """Pausing after dispatch does not abort the running slot — pause is
    between-call, per the operator contract. Release still frees the GPU."""
    tid = qm.track_request("t", "chat", "m", qm.PRIORITY_INTERACTIVE)
    target = await qm.wait_for_dispatch(tid)
    assert target in gpu_urls
    assert qm._tracked[tid].status == "running"

    qm.set_paused(True, persist=False)
    # The already-dispatched slot is still busy and tracked as running.
    assert qm._gpu_busy[target] is True
    assert qm._tracked[tid].status == "running"

    # Normal release path still works while paused.
    qm.release(tid)
    assert qm._gpu_busy[target] is False
    qm.set_paused(False, persist=False)
