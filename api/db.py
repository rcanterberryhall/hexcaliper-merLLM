"""
db.py — SQLite WAL persistence for merLLM.

Tables:
  batch_jobs   — work items dispatched at the background priority bucket
  pending_work — in-flight submitted requests awaiting GPU dispatch; rebuilds
                 the five priority buckets after a restart (FSM refactor
                 Phase 2, merLLM#54)
  slot_state   — last-known state of each GPU slot FSM; reconciled against
                 live probes on boot (FSM refactor Phase 2, merLLM#54)
  metrics      — time-series system and GPU metrics (auto-pruned)
  settings     — key/value configuration overrides
  transitions  — vestigial day/night transition log (kept for forensic
                 history; no code path writes to it after the round-robin
                 refactor on 2026-04-09)
  fan_faults   — fan-controller fault events
"""
import json
import logging
import sqlite3
import threading
import time
from typing import Optional

import config

log = logging.getLogger(__name__)

lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _create_tables(_conn)
        row_count = _conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        log.info("database opened: %s (%d metrics rows)", config.DB_PATH, row_count)
    return _conn


def _create_tables(c: sqlite3.Connection) -> None:
    c.executescript("""
        CREATE TABLE IF NOT EXISTS batch_jobs (
            id           TEXT PRIMARY KEY,
            source_app   TEXT NOT NULL,
            model        TEXT NOT NULL,
            prompt       TEXT NOT NULL,
            options      TEXT NOT NULL DEFAULT '{}',
            status       TEXT NOT NULL DEFAULT 'queued',
            submitted_at REAL NOT NULL,
            started_at   REAL,
            completed_at REAL,
            result       TEXT,
            error        TEXT,
            retries      INTEGER NOT NULL DEFAULT 0,
            retry_after  REAL
        );

        CREATE TABLE IF NOT EXISTS metrics (
            ts     REAL NOT NULL,
            name   TEXT NOT NULL,
            value  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_ts   ON metrics(ts);
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name, ts);

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transitions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         REAL NOT NULL,
            direction  TEXT NOT NULL,
            trigger    TEXT NOT NULL,
            duration_s REAL,
            success    INTEGER NOT NULL DEFAULT 1,
            error      TEXT
        );

        CREATE TABLE IF NOT EXISTS fan_faults (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               REAL NOT NULL,
            event_type       TEXT NOT NULL,
            message          TEXT NOT NULL,
            fan_speed_applied INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_fan_faults_ts ON fan_faults(ts);

        CREATE TABLE IF NOT EXISTS pending_work (
            id            TEXT PRIMARY KEY,
            source        TEXT NOT NULL,
            request_type  TEXT NOT NULL,
            model         TEXT NOT NULL,
            priority      INTEGER NOT NULL,
            submitted_at  REAL NOT NULL,
            payload_json  TEXT NOT NULL DEFAULT '{}',
            batch_job_id  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pending_priority_submitted
            ON pending_work(priority, submitted_at);

        CREATE TABLE IF NOT EXISTS slot_state (
            gpu_url            TEXT PRIMARY KEY,
            state              TEXT NOT NULL,
            model_loaded       TEXT,
            last_transition_at REAL NOT NULL
        );
    """)
    # Schema migration: add columns that may be absent in older databases.
    existing = {row[1] for row in c.execute("PRAGMA table_info(batch_jobs)").fetchall()}
    if "retries" not in existing:
        c.execute("ALTER TABLE batch_jobs ADD COLUMN retries INTEGER NOT NULL DEFAULT 0")
    if "retry_after" not in existing:
        c.execute("ALTER TABLE batch_jobs ADD COLUMN retry_after REAL")


# ── Batch jobs ────────────────────────────────────────────────────────────────

def insert_batch_job(job_id: str, source_app: str, model: str,
                     prompt: str, options: dict) -> None:
    with lock:
        conn().execute(
            "INSERT INTO batch_jobs (id, source_app, model, prompt, options, submitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, source_app, model, prompt, json.dumps(options), time.time())
        )


def get_batch_job(job_id: str) -> Optional[dict]:
    with lock:
        row = conn().execute(
            "SELECT * FROM batch_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def list_batch_jobs(status: Optional[str] = None, ready_only: bool = False) -> list[dict]:
    """
    List batch jobs, optionally filtered by status.

    ready_only=True (only meaningful with status="queued"): exclude jobs whose
    retry_after timestamp is in the future, i.e. return only jobs that are
    ready to run right now.
    """
    with lock:
        if ready_only and status == "queued":
            rows = conn().execute(
                "SELECT * FROM batch_jobs WHERE status = 'queued' "
                "AND (retry_after IS NULL OR retry_after <= ?) "
                "ORDER BY submitted_at",
                (time.time(),)
            ).fetchall()
        elif status:
            rows = conn().execute(
                "SELECT * FROM batch_jobs WHERE status = ? ORDER BY submitted_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn().execute(
                "SELECT * FROM batch_jobs ORDER BY submitted_at DESC LIMIT 200"
            ).fetchall()
    return [dict(r) for r in rows]


def get_earliest_retry_after() -> Optional[float]:
    """
    Return the smallest retry_after timestamp among deferred queued jobs
    (those whose retry_after is in the future), or None if there are none.
    """
    with lock:
        row = conn().execute(
            "SELECT MIN(retry_after) AS min_ts FROM batch_jobs "
            "WHERE status = 'queued' AND retry_after IS NOT NULL AND retry_after > ?",
            (time.time(),)
        ).fetchone()
    val = row["min_ts"] if row else None
    return float(val) if val is not None else None


def update_batch_job(job_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with lock:
        conn().execute(
            f"UPDATE batch_jobs SET {cols} WHERE id = ?",
            (*fields.values(), job_id)
        )


def drain_queued_jobs() -> int:
    """Cancel all jobs currently in 'queued' status. Returns count cancelled."""
    with lock:
        cur = conn().execute(
            "UPDATE batch_jobs SET status = 'cancelled' WHERE status = 'queued'"
        )
    return cur.rowcount


def requeue_all_failed_jobs() -> int:
    """Requeue all failed jobs. Returns count requeued."""
    with lock:
        cur = conn().execute(
            "UPDATE batch_jobs SET status = 'queued', error = NULL, "
            "started_at = NULL, completed_at = NULL, result = NULL "
            "WHERE status = 'failed'"
        )
    return cur.rowcount


def delete_terminal_jobs(
    older_than_days: Optional[int] = None,
    include_failed: bool = False,
) -> int:
    """
    Delete terminal-state jobs. By default drops ``completed`` and ``cancelled``
    rows only — ``failed`` jobs are preserved because they're usually worth
    investigating (stuck model, bad prompt, unreachable host). Callers that
    want a hard wipe can opt in with ``include_failed=True``.

    If ``older_than_days`` is set, only rows whose ``submitted_at`` is older
    than that are deleted. Returns the count deleted.
    """
    statuses = ("completed", "cancelled", "failed") if include_failed \
        else ("completed", "cancelled")
    placeholders = ",".join("?" * len(statuses))
    params: list = list(statuses)
    sql = f"DELETE FROM batch_jobs WHERE status IN ({placeholders})"
    if older_than_days is not None:
        sql += " AND submitted_at < ?"
        params.append(time.time() - older_than_days * 86400)
    with lock:
        cur = conn().execute(sql, params)
    return cur.rowcount


def requeue_orphaned_jobs() -> int:
    """
    Reset any jobs left in 'running' status back to 'queued'.

    Called at startup to recover jobs that were mid-flight when the
    process was killed or the container was restarted.  Returns the
    number of jobs recovered.

    Preserves prior ``error`` text (retry history, failure attempts) by
    appending a ``[recovered after restart]`` marker rather than clobbering
    the field. A retry-heavy job that gets restarted repeatedly will show
    its full lineage in the UI instead of only the last recovery note.
    """
    marker = "[recovered after restart]"
    with lock:
        cur = conn().execute(
            "UPDATE batch_jobs "
            "SET status = 'queued', "
            "    started_at = NULL, "
            "    error = CASE "
            "        WHEN error IS NULL OR TRIM(error) = '' THEN ? "
            "        WHEN INSTR(error, ?) > 0 THEN error "
            "        ELSE error || ' ' || ? "
            "    END "
            "WHERE status = 'running'",
            (marker, marker, marker),
        )
    return cur.rowcount


def cancel_batch_job(job_id: str) -> bool:
    with lock:
        cur = conn().execute(
            "UPDATE batch_jobs SET status = 'cancelled' WHERE id = ? AND status = 'queued'",
            (job_id,)
        )
    return cur.rowcount > 0


def requeue_batch_job(job_id: str) -> bool:
    with lock:
        cur = conn().execute(
            "UPDATE batch_jobs SET status = 'queued', error = NULL, "
            "started_at = NULL, completed_at = NULL, result = NULL "
            "WHERE id = ? AND status = 'failed'",
            (job_id,)
        )
    return cur.rowcount > 0


def count_batch_jobs_by_status() -> dict:
    with lock:
        rows = conn().execute(
            "SELECT status, COUNT(*) as n FROM batch_jobs GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


# ── Metrics ───────────────────────────────────────────────────────────────────

def insert_metrics(points: list[tuple[str, float]]) -> None:
    ts = time.time()
    with lock:
        conn().executemany(
            "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)",
            [(ts, name, value) for name, value in points]
        )


def get_metrics_history(name: str, since: float) -> list[dict]:
    with lock:
        rows = conn().execute(
            "SELECT ts, value FROM metrics WHERE name = ? AND ts >= ? ORDER BY ts",
            (name, since)
        ).fetchall()
    return [{"ts": r["ts"], "value": r["value"]} for r in rows]


def get_latest_metrics() -> dict:
    with lock:
        rows = conn().execute(
            "SELECT name, value, ts FROM metrics "
            "WHERE ts = (SELECT MAX(ts) FROM metrics)"
        ).fetchall()
    return {r["name"]: {"value": r["value"], "ts": r["ts"]} for r in rows}


def prune_old_metrics(retain_days: int) -> None:
    cutoff = time.time() - retain_days * 86400
    with lock:
        c = conn().execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
        if c.rowcount:
            log.info("pruned %d metrics rows older than %d days", c.rowcount, retain_days)


# ── Settings ──────────────────────────────────────────────────────────────────

def get_settings() -> dict:
    with lock:
        rows = conn().execute("SELECT key, value FROM settings").fetchall()
    out = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            out[r["key"]] = r["value"]
    return out


def save_settings(data: dict) -> None:
    with lock:
        for k, v in data.items():
            conn().execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, json.dumps(v))
            )


# ── Transitions ───────────────────────────────────────────────────────────────

def insert_transition(direction: str, trigger: str, duration_s: Optional[float],
                       success: bool, error: Optional[str] = None) -> None:
    with lock:
        conn().execute(
            "INSERT INTO transitions (ts, direction, trigger, duration_s, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), direction, trigger, duration_s, int(success), error)
        )


def list_transitions(limit: int = 20) -> list[dict]:
    with lock:
        rows = conn().execute(
            "SELECT * FROM transitions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Fan faults ────────────────────────────────────────────────────────────────

def insert_fan_fault(event_type: str, message: str,
                     fan_speed_applied: Optional[int] = None) -> int:
    with lock:
        cur = conn().execute(
            "INSERT INTO fan_faults (ts, event_type, message, fan_speed_applied) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), event_type, message, fan_speed_applied)
        )
    return cur.lastrowid


def list_fan_faults(limit: int = 100) -> list[dict]:
    with lock:
        rows = conn().execute(
            "SELECT * FROM fan_faults ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Pending work (FSM refactor: durable submit queue, merLLM#54) ─────────────

def insert_pending(
    req_id: str,
    source: str,
    request_type: str,
    model: str,
    priority: int,
    payload: Optional[dict] = None,
    batch_job_id: Optional[str] = None,
) -> None:
    """Record a submitted request before acknowledging the caller.

    Called on every submit so buckets rebuild exactly after a crash
    ("Option A" persistence per merLLM#52 design).
    """
    with lock:
        conn().execute(
            "INSERT INTO pending_work "
            "(id, source, request_type, model, priority, submitted_at, "
            " payload_json, batch_job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (req_id, source, request_type, model, int(priority), time.time(),
             json.dumps(payload or {}), batch_job_id),
        )
    log.info("[pending] +%s src=%s type=%s model=%s prio=%d",
             req_id, source, request_type, model, priority)


def delete_pending(req_id: str) -> bool:
    """Remove a pending row after dispatch commits. Returns True if a row
    was deleted (i.e. the id was still pending), False otherwise."""
    with lock:
        cur = conn().execute(
            "DELETE FROM pending_work WHERE id = ?", (req_id,)
        )
    deleted = cur.rowcount > 0
    if deleted:
        log.info("[pending] -%s", req_id)
    return deleted


def list_pending_by_priority() -> list[dict]:
    """Return all pending rows in dispatch order: priority ASC, then FIFO.

    Matches the five-bucket strict-priority walk: CHAT (0) first, within a
    priority the oldest submission first. Used on boot to rebuild the
    in-memory buckets.
    """
    with lock:
        rows = conn().execute(
            "SELECT * FROM pending_work "
            "ORDER BY priority ASC, submitted_at ASC"
        ).fetchall()
    out = [dict(r) for r in rows]
    if out:
        by_prio: dict[int, int] = {}
        for r in out:
            by_prio[r["priority"]] = by_prio.get(r["priority"], 0) + 1
        log.info("[pending] reconciling %d rows from disk: %s",
                 len(out), by_prio)
    return out


def count_pending() -> int:
    with lock:
        row = conn().execute("SELECT COUNT(*) AS n FROM pending_work").fetchone()
    return int(row["n"]) if row else 0


# ── Slot state (FSM refactor: per-GPU last-known FSM state, merLLM#54) ───────

def upsert_slot_state(
    gpu_url: str, state: str, model_loaded: Optional[str]
) -> None:
    """Record the latest observed FSM state for a GPU slot.

    Persisted state is reconciled against a live probe at boot — it is a
    hint, not a guarantee. Transient in-memory fields (load_attempts,
    thermal_pending, drain_pending) are deliberately not persisted; they
    reset to defaults on restart.
    """
    with lock:
        conn().execute(
            "INSERT INTO slot_state (gpu_url, state, model_loaded, last_transition_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(gpu_url) DO UPDATE SET "
            "    state = excluded.state, "
            "    model_loaded = excluded.model_loaded, "
            "    last_transition_at = excluded.last_transition_at",
            (gpu_url, state, model_loaded, time.time()),
        )
    log.info("[slot] %s state=%s model=%s", gpu_url, state, model_loaded)


def list_slot_states() -> list[dict]:
    with lock:
        rows = conn().execute("SELECT * FROM slot_state").fetchall()
    return [dict(r) for r in rows]


def delete_slot_state(gpu_url: str) -> bool:
    with lock:
        cur = conn().execute(
            "DELETE FROM slot_state WHERE gpu_url = ?", (gpu_url,)
        )
    return cur.rowcount > 0
