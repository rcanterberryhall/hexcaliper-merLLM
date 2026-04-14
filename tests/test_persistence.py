"""Tests for pending_work + slot_state persistence added in Phase 2 of the
merLLM#52 FSM refactor (merLLM#54). Additive — old batch_jobs paths untouched."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_merllm.db")
    monkeypatch.setenv("DB_PATH", db_path)
    for mod in ["db", "config"]:
        sys.modules.pop(mod, None)
    import db
    return db


# ── pending_work round-trips ────────────────────────────────────────────────

def test_insert_pending_and_list(tmp_db):
    db = tmp_db
    db.insert_pending(
        req_id="r1", source="parsival", request_type="chat",
        model="qwen3:32b", priority=0,
        payload={"messages": [{"role": "user", "content": "hi"}]},
    )
    rows = db.list_pending_by_priority()
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "r1"
    assert r["source"] == "parsival"
    assert r["request_type"] == "chat"
    assert r["model"] == "qwen3:32b"
    assert r["priority"] == 0
    assert json.loads(r["payload_json"])["messages"][0]["content"] == "hi"
    assert r["batch_job_id"] is None


def test_insert_pending_with_batch_job_id(tmp_db):
    db = tmp_db
    db.insert_pending(req_id="r1", source="merllm", request_type="batch",
                      model="m", priority=4, batch_job_id="bjob-1")
    rows = db.list_pending_by_priority()
    assert rows[0]["batch_job_id"] == "bjob-1"


def test_insert_pending_default_payload_is_empty_dict(tmp_db):
    db = tmp_db
    db.insert_pending(req_id="r1", source="direct", request_type="generate",
                      model="m", priority=2)
    rows = db.list_pending_by_priority()
    assert json.loads(rows[0]["payload_json"]) == {}


def test_delete_pending_returns_true_when_row_existed(tmp_db):
    db = tmp_db
    db.insert_pending("r1", "s", "chat", "m", 0)
    assert db.delete_pending("r1") is True
    assert db.list_pending_by_priority() == []


def test_delete_pending_returns_false_when_no_row(tmp_db):
    assert tmp_db.delete_pending("never-inserted") is False


def test_count_pending(tmp_db):
    db = tmp_db
    assert db.count_pending() == 0
    db.insert_pending("r1", "s", "chat", "m", 0)
    db.insert_pending("r2", "s", "chat", "m", 0)
    assert db.count_pending() == 2


def test_duplicate_id_raises(tmp_db):
    """id is PRIMARY KEY — a caller submitting the same id twice is a bug
    that should be visible, not silently swallowed."""
    import sqlite3
    db = tmp_db
    db.insert_pending("r1", "s", "chat", "m", 0)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_pending("r1", "s", "chat", "m", 0)


# ── priority + FIFO ordering (crash-rebuild simulation) ─────────────────────

def test_list_orders_chat_before_background(tmp_db):
    db = tmp_db
    db.insert_pending("bg-1",   "s", "batch", "m", 4)   # BACKGROUND
    db.insert_pending("chat-1", "s", "chat",  "m", 0)   # CHAT
    db.insert_pending("short-1","s", "chat",  "m", 2)   # SHORT
    rows = db.list_pending_by_priority()
    order = [r["id"] for r in rows]
    assert order == ["chat-1", "short-1", "bg-1"]


def test_list_is_fifo_within_same_priority(tmp_db):
    db = tmp_db
    db.insert_pending("chat-a", "s", "chat", "m", 0)
    time.sleep(0.002)   # ensure submitted_at strictly increases
    db.insert_pending("chat-b", "s", "chat", "m", 0)
    time.sleep(0.002)
    db.insert_pending("chat-c", "s", "chat", "m", 0)
    rows = db.list_pending_by_priority()
    order = [r["id"] for r in rows]
    assert order == ["chat-a", "chat-b", "chat-c"]


def test_crash_rebuild_simulation_five_buckets(tmp_db):
    """End-to-end: insert a mixed workload across all five priority buckets,
    call list_pending_by_priority, and verify the caller can reconstruct
    the five in-memory deques in strict-priority FIFO order — which is what
    the boot reconciliation path will do in Phase 3."""
    db = tmp_db
    # Insert out of priority order on purpose so the SQL, not the insert
    # order, is what drives the result.
    for pr_name, pr in [("bg", 4), ("feedback", 3), ("chat", 0),
                         ("embeds", 1), ("short", 2)]:
        for i in range(3):
            db.insert_pending(f"{pr_name}-{i}", "s",
                              "chat" if pr == 0 else "batch",
                              "m", pr)
            time.sleep(0.001)

    rows = db.list_pending_by_priority()

    # Bucket boundaries must be strict: every CHAT row precedes every
    # EMBEDDINGS row, and so on down to BACKGROUND.
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities)

    # Within a bucket, FIFO by submitted_at.
    from itertools import groupby
    for p, group in groupby(rows, key=lambda r: r["priority"]):
        ids = [r["id"] for r in group]
        # IDs were emitted i=0,1,2 so lexicographic order matches FIFO.
        assert ids == sorted(ids)


# ── slot_state upsert ──────────────────────────────────────────────────────

def test_upsert_slot_state_insert(tmp_db):
    db = tmp_db
    db.upsert_slot_state("http://gpu0:11434", "ready", "qwen3:32b")
    rows = db.list_slot_states()
    assert len(rows) == 1
    assert rows[0]["gpu_url"] == "http://gpu0:11434"
    assert rows[0]["state"] == "ready"
    assert rows[0]["model_loaded"] == "qwen3:32b"
    assert rows[0]["last_transition_at"] > 0


def test_upsert_slot_state_updates_existing_row(tmp_db):
    db = tmp_db
    db.upsert_slot_state("http://gpu0:11434", "ready", "qwen3:32b")
    t1 = db.list_slot_states()[0]["last_transition_at"]
    time.sleep(0.01)
    db.upsert_slot_state("http://gpu0:11434", "busy", "qwen3:32b")
    rows = db.list_slot_states()
    assert len(rows) == 1   # upsert, not insert
    assert rows[0]["state"] == "busy"
    assert rows[0]["last_transition_at"] > t1


def test_list_slot_states_multiple_gpus(tmp_db):
    db = tmp_db
    db.upsert_slot_state("http://gpu0:11434", "ready", "m")
    db.upsert_slot_state("http://gpu1:11435", "cooling", None)
    rows = db.list_slot_states()
    assert len(rows) == 2
    by_url = {r["gpu_url"]: r for r in rows}
    assert by_url["http://gpu0:11434"]["state"] == "ready"
    assert by_url["http://gpu1:11435"]["state"] == "cooling"
    assert by_url["http://gpu1:11435"]["model_loaded"] is None


def test_delete_slot_state(tmp_db):
    db = tmp_db
    db.upsert_slot_state("http://gpu0:11434", "ready", "m")
    assert db.delete_slot_state("http://gpu0:11434") is True
    assert db.list_slot_states() == []
    assert db.delete_slot_state("http://gpu0:11434") is False


# ── performance at scale (per feedback_systems_testing memory) ──────────────

def test_list_pending_by_priority_scales_to_thousand_rows(tmp_db):
    """Query must stay fast under realistic backlog. Index on
    (priority, submitted_at) is the whole reason this is snappy —
    if someone removes it, this test catches the regression."""
    db = tmp_db
    # Insert 1000 rows across all five priorities.
    for i in range(1000):
        db.insert_pending(f"r-{i:04d}", "s", "chat", "m", i % 5)

    start = time.perf_counter()
    rows = db.list_pending_by_priority()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(rows) == 1000
    # Generous cap — sub-ms on typical hardware, never >100ms even with
    # a cold cache on the slowest host that runs CI.
    assert elapsed_ms < 500, f"list took {elapsed_ms:.1f}ms; index missing?"

    # Priority ordering still strict at this scale.
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities)


# ── coexistence: old batch_jobs paths still work ───────────────────────────

def test_pending_work_and_batch_jobs_coexist(tmp_db):
    """Phase 2 is strictly additive — ensure batch_jobs insert + list
    still works alongside the new tables."""
    db = tmp_db
    db.insert_batch_job("bjob-1", "parsival", "qwen3:32b", "prompt", {})
    db.insert_pending("pend-1", "parsival", "batch", "qwen3:32b", 4,
                      batch_job_id="bjob-1")

    assert db.get_batch_job("bjob-1") is not None
    pending = db.list_pending_by_priority()
    assert len(pending) == 1
    assert pending[0]["batch_job_id"] == "bjob-1"
