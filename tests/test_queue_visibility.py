"""Tests for queue_manager.active_queue() — the data behind the dashboard's
Active Requests table.

The Elapsed column renders, and since merLLM#74 also sorts by, whichever
duration field the row carries: a running request has ``elapsed_sec``, a
queued one ``waiting_sec``, and no row shape carries both. The frontend
comparator resolves that split with a fallback (``web/app.js`` ``_SORT_VALUE``),
so these tests pin the split itself — renaming either field, or adding the
missing one to a shape, would strand that fallback silently.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

GPU0 = "http://gpu0:11434"


@pytest.fixture
def qm(tmp_path, monkeypatch):
    """Fresh queue_manager wired to a throwaway sqlite file.

    Same module-reload idiom as test_tick.py's fixture: the globals
    active_queue() reads (_buckets_v2, _slots, _busy_since) start empty, so
    each test builds exactly the snapshot it asserts on.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_merllm.db"))
    for mod in ["db", "config", "notifications", "queue_manager"]:
        sys.modules.pop(mod, None)
    import queue_manager
    return queue_manager


def _enqueue(qm, tid, *, submitted_at, priority=None, model="qwen3:32b"):
    """Append a queued job into its bucket directly, bypassing dispatch."""
    prio = int(qm.Priority.SHORT if priority is None else priority)
    job = qm._make_job(tid=tid, source="parsival", request_type="chat",
                       model=model, priority=prio, batch_job_id=None,
                       submitted_at=submitted_at)
    qm._buckets_v2[prio].append(job)
    return job


def _occupy_slot(qm, tid, *, submitted_at, started_at, model="qwen3:32b"):
    """Put one BUSY slot in place, as _apply_effect's start_work leaves it."""
    job = qm._make_job(tid=tid, source="lancellmot", request_type="generate",
                       model=model, priority=int(qm.Priority.BACKGROUND),
                       batch_job_id=None, submitted_at=submitted_at)
    qm._slots = [qm.Slot(url=GPU0, state=qm.SlotState.BUSY,
                         model_loaded=model, current_job=job)]
    qm._busy_since = {0: started_at}
    return job


def _by_id(rows):
    return {r["id"]: r for r in rows}


# ── Row shapes: exactly one duration field each ──────────────────────────────

def test_queued_entry_reports_waiting_sec_and_no_elapsed_sec(qm):
    """A request that has not been dispatched has waited, not run."""
    _enqueue(qm, "q1", submitted_at=time.time() - 90)

    row = _by_id(qm.active_queue())["q1"]

    assert row["status"] == "queued"
    assert row["waiting_sec"] == pytest.approx(90, abs=2)
    assert "elapsed_sec" not in row


def test_running_entry_reports_elapsed_sec_and_no_waiting_sec(qm):
    """A dispatched request reports time since dispatch, not since submit."""
    now = time.time()
    _occupy_slot(qm, "r1", submitted_at=now - 300, started_at=now - 12)

    row = _by_id(qm.active_queue())["r1"]

    assert row["status"] == "running"
    assert row["elapsed_sec"] == pytest.approx(12, abs=2)
    assert "waiting_sec" not in row


def test_every_entry_exposes_exactly_one_duration_field(qm):
    """The split the Elapsed column's fallback rests on, asserted directly.

    Neither shape carries both keys and neither carries none, so resolving
    ``elapsed_sec`` then ``waiting_sec`` yields a number for every row.
    """
    now = time.time()
    _enqueue(qm, "q1", submitted_at=now - 45)
    _enqueue(qm, "q2", submitted_at=now - 5, priority=qm.Priority.BACKGROUND)
    _occupy_slot(qm, "r1", submitted_at=now - 200, started_at=now - 30)

    rows = qm.active_queue()

    assert len(rows) == 3
    for row in rows:
        present = {"elapsed_sec", "waiting_sec"} & row.keys()
        assert len(present) == 1, f"{row['id']} carries {present or 'neither'}"
        assert isinstance(row[present.pop()], (int, float))


def test_elapsed_measures_dispatch_not_submit(qm):
    """Why one field cannot stand in for the other: they count from different
    instants, so a long-queued job that just started running is 'new' here."""
    now = time.time()
    _enqueue(qm, "q1", submitted_at=now - 60)
    _occupy_slot(qm, "r1", submitted_at=now - 600, started_at=now - 3)

    rows = _by_id(qm.active_queue())

    assert rows["r1"]["elapsed_sec"] < rows["q1"]["waiting_sec"]
