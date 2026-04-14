"""Tests for the v2 pure-FSM path in queue_manager (merLLM#55 cutover).

Covers _boot_reconcile in this commit; the tick loop and public-API flips
land in subsequent commits and will extend this file.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def qm(tmp_path, monkeypatch):
    """Fresh queue_manager + db wired to a throwaway sqlite file.

    Reloads the db/config/queue_manager modules so the tmp DB_PATH takes
    effect. Module globals (_slots, _buckets_v2, _tid_to_job, _inflight)
    therefore start empty for each test.
    """
    db_path = str(tmp_path / "test_merllm.db")
    monkeypatch.setenv("DB_PATH", db_path)
    # Probe calls httpx — default to "all probes succeed" unless a test
    # overrides via the _set_probe helper returned below. Monkeypatching
    # the already-imported httpx.AsyncClient is cleaner than per-test
    # httpx mocks because _probe_url catches every exception.
    for mod in ["db", "config", "notifications", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager
    return queue_manager


def _install_probe(qm, results: dict[str, bool]):
    """Replace _probe_url with a dict-backed stub — no httpx round-trip."""
    async def fake(url: str, timeout: float = 5.0) -> bool:
        return results.get(url, False)
    qm._probe_url = fake


# ── boot reconcile ──────────────────────────────────────────────────────────

def test_reconcile_cold_start_no_rows_probes_ok(qm):
    """Empty DB + healthy probes → two READY slots, no buckets."""
    import config
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: True})

    asyncio.run(qm._boot_reconcile())

    assert len(qm._slots) == 2
    assert all(s.state is qm.SlotState.READY for s in qm._slots)
    assert all(s.model_loaded is None for s in qm._slots)
    assert all(len(b) == 0 for b in qm._buckets_v2)


def test_reconcile_rehydrates_model_loaded_hint(qm):
    """slot_state row → Slot.model_loaded seed; probe then confirms READY."""
    import config, db
    db.upsert_slot_state(config.OLLAMA_0_URL, "ready", "qwen3:32b")
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: True})

    asyncio.run(qm._boot_reconcile())

    slot0 = next(s for s in qm._slots if s.url == config.OLLAMA_0_URL)
    assert slot0.state is qm.SlotState.READY
    assert slot0.model_loaded == "qwen3:32b"


def test_reconcile_probe_fail_marks_unreachable(qm):
    """One GPU unreachable at boot → that slot goes UNREACHABLE, other READY."""
    import config
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: False})

    asyncio.run(qm._boot_reconcile())

    states = {s.url: s.state for s in qm._slots}
    assert states[config.OLLAMA_0_URL] is qm.SlotState.READY
    assert states[config.OLLAMA_1_URL] is qm.SlotState.UNREACHABLE


def test_reconcile_wipes_stale_pending_rows(qm):
    """pending_work is a live mirror: stale rows from a prior process are
    orphaned (HTTP waiters gone, batch recovery re-kicks via
    _run_batch_job_async). Boot wipes them so they don't double-dispatch."""
    import config, db
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: True})

    db.insert_pending("stale-1", "s", "chat",  "m", 0)
    db.insert_pending("stale-2", "s", "batch", "m", 4, batch_job_id="bjob-1")

    asyncio.run(qm._boot_reconcile())

    assert db.count_pending() == 0
    assert all(len(b) == 0 for b in qm._buckets_v2)
    assert qm._tid_to_job == {}


def test_reconcile_persists_slot_state_after_probe(qm):
    """PROBE_OK/FAIL must be reflected in slot_state (durable) post-reconcile."""
    import config, db
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: False})

    asyncio.run(qm._boot_reconcile())

    rows = {r["gpu_url"]: r for r in db.list_slot_states()}
    assert rows[config.OLLAMA_0_URL]["state"] == "ready"
    assert rows[config.OLLAMA_1_URL]["state"] == "unreachable"


# ── tick loop ───────────────────────────────────────────────────────────────

def _install_fake_load(qm, *, succeed: bool = True):
    """Replace _do_load with an in-process stub — no httpx round-trip.

    The stub feeds LOAD_DONE / LOAD_FAIL synchronously so the next
    _tick_once sees the slot in READY. Avoids having to schedule real
    asyncio tasks inside the tick test bodies.
    """
    from scheduler import Event

    async def fake(slot_idx, url, model, evict):
        qm._feed_event(slot_idx, Event.LOAD_DONE if succeed else Event.LOAD_FAIL)
    qm._do_load = fake


async def _reconcile_all_ready(qm):
    import config
    _install_probe(qm, {config.OLLAMA_0_URL: True, config.OLLAMA_1_URL: True})
    await qm._boot_reconcile()


def test_tick_stages_model_when_bucket_has_demand(qm):
    """Empty slots + a pending request for model M → stage_pass emits load."""
    _install_fake_load(qm, succeed=True)

    async def scenario():
        await _reconcile_all_ready(qm)
        qm._buckets_v2[0].append(qm._make_job(
            tid="t1", source="s", request_type="chat", model="qwen3:32b",
            priority=0, batch_job_id=None, submitted_at=0.0,
        ))
        # First tick: stage_pass fires load effects, fake _do_load
        # immediately feeds LOAD_DONE, slot is READY on return.
        await qm._tick_once()
        # Give any queued create_task callbacks a chance to run.
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert any(s.model_loaded == "qwen3:32b" for s in qm._slots)


def test_tick_dispatches_head_when_model_is_resident(qm):
    """Slot already holds the demanded model → dispatch_pass pops + starts work."""
    _install_fake_load(qm, succeed=True)

    async def scenario():
        await _reconcile_all_ready(qm)
        # Pre-load slot 0 with the required model.
        from scheduler import Event
        qm._feed_event(0, Event.LOAD_BEGIN, model="qwen3:32b")
        qm._feed_event(0, Event.LOAD_DONE)

        # Register a waiter future for the job.
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        qm._inflight["t1"] = fut
        job = qm._make_job(
            tid="t1", source="s", request_type="chat", model="qwen3:32b",
            priority=0, batch_job_id=None, submitted_at=0.0,
        )
        qm._buckets_v2[0].append(job)
        qm._tid_to_job["t1"] = job

        # Also persist the pending row so work_end has something to delete.
        import db
        db.insert_pending("t1", "s", "chat", "qwen3:32b", 0)

        await qm._tick_once()
        # Waiter should now be resolved with a URL.
        return await asyncio.wait_for(fut, timeout=1.0)

    target = asyncio.run(scenario())
    import config
    assert target in (config.OLLAMA_0_URL, config.OLLAMA_1_URL)
    # bucket drained, inflight popped, busy_since stamped for that slot.
    assert all(len(b) == 0 for b in qm._buckets_v2)
    assert "t1" not in qm._inflight
    assert any(idx in qm._busy_since for idx in range(len(qm._slots)))


def test_tick_work_end_deletes_pending_row(qm):
    """work_end effect must remove the pending_work row (durable drain)."""
    _install_fake_load(qm, succeed=True)

    async def scenario():
        await _reconcile_all_ready(qm)
        from scheduler import Event
        qm._feed_event(0, Event.LOAD_BEGIN, model="m")
        qm._feed_event(0, Event.LOAD_DONE)

        import db
        db.insert_pending("t1", "s", "chat", "m", 0)
        fut = asyncio.get_event_loop().create_future()
        qm._inflight["t1"] = fut
        job = qm._make_job(tid="t1", source="s", request_type="chat",
                           model="m", priority=0, batch_job_id=None,
                           submitted_at=0.0)
        qm._buckets_v2[0].append(job)
        qm._tid_to_job["t1"] = job
        await qm._tick_once()
        await fut

        # Row still present during BUSY — durability guarantee.
        assert db.count_pending() == 1

        # Drive work_end manually (simulates release()).
        for e in qm._feed_event(0, Event.WORK_END, outcome="ok"):
            qm._apply_effect(e)

        return db.count_pending()

    assert asyncio.run(scenario()) == 0


def test_tick_busy_timeout_drives_fail_to_known_state(qm, monkeypatch):
    """BUSY past SLOT_MAX_WALL_SECONDS → work_end(timeout) → LOADING (forced evict)."""
    _install_fake_load(qm, succeed=True)
    import config
    # Make the timeout tiny so the test does not have to wait 1800 s.
    monkeypatch.setattr(config, "SLOT_MAX_WALL_SECONDS", 0.01)

    async def scenario():
        await _reconcile_all_ready(qm)
        from scheduler import Event, SlotState
        qm._feed_event(0, Event.LOAD_BEGIN, model="m")
        qm._feed_event(0, Event.LOAD_DONE)

        job = qm._make_job(tid="t1", source="s", request_type="chat",
                           model="m", priority=0, batch_job_id=None,
                           submitted_at=0.0)
        qm._feed_event(0, Event.WORK_BEGIN, job=job)
        qm._busy_since[0] = 0.0   # "started long ago"

        await qm._tick_once()
        return qm._slots[0].state

    state = asyncio.run(scenario())
    # Timeout drives BUSY→LOADING (fail-to-known-state via forced reload).
    from scheduler import SlotState
    assert state is SlotState.LOADING


def test_tick_paused_skips_dispatch_and_stage(qm):
    """While paused, buckets pile up; no load or start_work effects fire."""
    _install_fake_load(qm, succeed=True)

    async def scenario():
        await _reconcile_all_ready(qm)
        qm._paused = True
        qm._buckets_v2[0].append(qm._make_job(
            tid="t1", source="s", request_type="chat", model="m",
            priority=0, batch_job_id=None, submitted_at=0.0,
        ))
        result = await qm._tick_once()
        return result, list(qm._slots)

    result, slots = asyncio.run(scenario())
    assert result["paused"] is True
    # Slots stayed READY with no model — stage_pass never ran.
    assert all(s.model_loaded is None for s in slots)
