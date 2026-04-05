"""
db.py — SQLite WAL persistence for merLLM.

Three tables:
  batch_jobs  — work items queued for night-mode processing
  metrics     — time-series system and GPU metrics (auto-pruned)
  settings    — key/value configuration overrides
"""
import json
import sqlite3
import threading
import time
from typing import Optional

import config

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
            error        TEXT
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
    """)


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


def list_batch_jobs(status: Optional[str] = None) -> list[dict]:
    with lock:
        if status:
            rows = conn().execute(
                "SELECT * FROM batch_jobs WHERE status = ? ORDER BY submitted_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn().execute(
                "SELECT * FROM batch_jobs ORDER BY submitted_at DESC LIMIT 200"
            ).fetchall()
    return [dict(r) for r in rows]


def update_batch_job(job_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with lock:
        conn().execute(
            f"UPDATE batch_jobs SET {cols} WHERE id = ?",
            (*fields.values(), job_id)
        )


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
            "SELECT name, value, ts FROM metrics m1 WHERE ts = "
            "(SELECT MAX(ts) FROM metrics m2 WHERE m2.name = m1.name)"
        ).fetchall()
    return {r["name"]: {"value": r["value"], "ts": r["ts"]} for r in rows}


def prune_old_metrics(retain_days: int) -> None:
    cutoff = time.time() - retain_days * 86400
    with lock:
        conn().execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))


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
