"""Stress tests for merLLM — exercises DB queries, routing, and queue
behaviour at production-realistic scale.

These tests ensure that performance does not degrade catastrophically as
data accumulates.  Every test that asserts a wall-clock bound uses a
generous threshold (10x headroom over typical) so that they reliably pass
on CI/slow machines while still catching O(n^2) regressions.
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.stress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

# Number of metric names mirrors production (see metrics.collect())
METRIC_NAMES = [f"cpu.core{i}" for i in range(32)] + [
    "cpu.avg", "ram.total", "ram.used", "ram.available",
    "swap.total", "swap.used", "net.bytes_sent", "net.bytes_recv",
    "net.tx_bps", "net.rx_bps", "disk.root.total_gb", "disk.root.used_gb",
    "disk.root.pct",
] + [
    f"gpu{i}.{m}"
    for i in range(2)
    for m in ("util_pct", "mem_used_mb", "mem_total_mb", "temp_c", "power_w")
]

# 3 days of 10-second intervals = ~26k collection rounds
COLLECTION_ROUNDS = 26_000
TOTAL_ROWS = COLLECTION_ROUNDS * len(METRIC_NAMES)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_modules(tmp_path, monkeypatch):
    """Each test gets an isolated DB and fresh module state."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "stress.db"))
    monkeypatch.setenv("DEFAULT_MODEL", "test-model:7b")
    monkeypatch.setenv("RECLAIM_TIMEOUT", "60")
    monkeypatch.setenv("HEALTH_BACKOFF_BASE", "5")
    monkeypatch.setenv("HEALTH_BACKOFF_CAP", "30")
    monkeypatch.setenv("HEALTH_FAULT_TIMEOUT", "60")
    for mod in list(sys.modules):
        if mod in ("db", "config", "gpu_router", "queue_manager",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    yield
    for mod in list(sys.modules):
        if mod in ("db", "config", "gpu_router", "queue_manager",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)


@pytest.fixture
def big_metrics_db():
    """Populate the metrics table with production-scale data (~1.4M rows).

    Inserts in large batches for speed.  Returns (db_module, row_count).
    """
    import db

    c = db.conn()
    base_ts = time.time() - 3 * 86400  # 3 days ago

    BATCH = 5000
    rows = []
    for i in range(COLLECTION_ROUNDS):
        ts = base_ts + i * 10
        for name in METRIC_NAMES:
            rows.append((ts, name, float(i % 100)))
        if len(rows) >= BATCH * len(METRIC_NAMES):
            c.executemany(
                "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)", rows
            )
            rows.clear()
    if rows:
        c.executemany(
            "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)", rows
        )

    actual = c.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    return db, actual


# ── DB query performance ─────────────────────────────────────────────────────


class TestMetricsQueryPerformance:
    """Ensure DB queries stay fast as the metrics table grows."""

    def test_get_latest_metrics_under_100ms(self, big_metrics_db):
        """get_latest_metrics must not degrade with millions of rows.

        The old correlated-subquery version took 3.6s on 2.8M rows.
        The fixed version should be <10ms; we allow 100ms for slow CI.
        """
        db, row_count = big_metrics_db
        assert row_count > 1_000_000, f"expected >1M rows, got {row_count}"

        # Warm up (first call may include page cache misses)
        db.get_latest_metrics()

        start = time.monotonic()
        result = db.get_latest_metrics()
        elapsed = time.monotonic() - start

        assert len(result) == len(METRIC_NAMES)
        assert elapsed < 0.1, (
            f"get_latest_metrics took {elapsed:.3f}s on {row_count} rows "
            f"(limit 0.1s)"
        )

    def test_get_metrics_history_1h_under_500ms(self, big_metrics_db):
        """History query for a single metric over 1 hour should be fast."""
        db, row_count = big_metrics_db

        since = time.time() - 3600
        start = time.monotonic()
        history = db.get_metrics_history("cpu.core0", since)
        elapsed = time.monotonic() - start

        assert len(history) > 0
        assert elapsed < 0.5, (
            f"get_metrics_history (1h) took {elapsed:.3f}s on {row_count} "
            f"rows (limit 0.5s)"
        )

    def test_get_metrics_history_24h_under_1s(self, big_metrics_db):
        """24-hour history query — heavier but should still be bounded."""
        db, row_count = big_metrics_db

        since = time.time() - 86400
        start = time.monotonic()
        history = db.get_metrics_history("cpu.core0", since)
        elapsed = time.monotonic() - start

        assert len(history) > 0
        assert elapsed < 1.0, (
            f"get_metrics_history (24h) took {elapsed:.3f}s on {row_count} "
            f"rows (limit 1.0s)"
        )

    def test_prune_deletes_old_rows(self, big_metrics_db):
        """Prune should remove rows and leave the recent window intact."""
        db, row_count = big_metrics_db

        start = time.monotonic()
        db.prune_old_metrics(retain_days=1)
        elapsed = time.monotonic() - start

        remaining = db.conn().execute(
            "SELECT COUNT(*) FROM metrics"
        ).fetchone()[0]

        # ~1 day of data should remain (8640 rounds * N metrics)
        expected_remaining = 8640 * len(METRIC_NAMES)
        assert remaining < expected_remaining * 1.1
        assert remaining > 0
        assert elapsed < 5.0, (
            f"prune took {elapsed:.3f}s (limit 5.0s)"
        )

    def test_insert_metrics_under_10ms(self, big_metrics_db):
        """Single collection-round insert should stay fast regardless of
        table size."""
        db, _ = big_metrics_db

        points = [(name, 42.0) for name in METRIC_NAMES]
        start = time.monotonic()
        db.insert_metrics(points)
        elapsed = time.monotonic() - start

        assert elapsed < 0.01, (
            f"insert_metrics ({len(points)} points) took {elapsed:.3f}s "
            f"(limit 0.01s)"
        )


# ── GPU router status under load ────────────────────────────────────────────


class TestRouterStress:
    """Status snapshots are polled on every dashboard tick — keep them fast."""

    def test_status_under_load(self):
        """status() is called on every dashboard poll — must be fast."""
        import gpu_router

        gpu_router._gpus.clear()
        gpu_router._init_gpus()

        start = time.monotonic()
        for _ in range(1000):
            s = gpu_router.status()
        elapsed = time.monotonic() - start

        assert "gpus" in s
        assert elapsed < 0.5, (
            f"1000 status() calls took {elapsed:.3f}s (limit 0.5s)"
        )


# ── Queue manager under load ────────────────────────────────────────────────


class TestQueueStress:
    """Ensure the tracked queue handles burst traffic correctly."""

    async def test_burst_tracking(self):
        """Track a large batch of requests — queue_depth should reflect them."""
        import queue_manager

        queue_manager._tracked.clear()

        N = 500
        for i in range(N):
            queue_manager.track_request(
                "test", "generate", "m",
                queue_manager.PRIORITY_INTERACTIVE,
            )

        depth = queue_manager.queue_depth()
        assert depth["queued"] == N
        assert depth["total"] == N

    async def test_dispatcher_contention(self, monkeypatch):
        """Many concurrent dispatch waiters should serialise without deadlock."""
        import queue_manager
        import gpu_router

        queue_manager._tracked.clear()
        for _bucket in queue_manager._buckets:
            _bucket.clear()
        for url in queue_manager._gpu_targets():
            queue_manager._gpu_busy[url] = False

        async def _noop_reload(gpu, model):
            gpu.model = model
        monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

        completed = 0

        async def request_cycle():
            nonlocal completed
            tid = queue_manager.track_request(
                "stress", "chat", "m", queue_manager.PRIORITY_INTERACTIVE,
            )
            target = await queue_manager.wait_for_dispatch(tid)
            await asyncio.sleep(0.001)
            queue_manager.release(tid)
            assert target in queue_manager._gpu_targets()
            completed += 1

        # 100 concurrent requests competing for 2 GPU slots.
        tasks = [asyncio.create_task(request_cycle()) for _ in range(100)]
        done, pending = await asyncio.wait(tasks, timeout=30)

        assert len(pending) == 0, f"{len(pending)} tasks timed out"
        assert completed == 100

    async def test_queue_depth_reporting(self):
        """queue_depth() should accurately reflect tracked items."""
        import queue_manager

        queue_manager._tracked.clear()

        for i in range(200):
            queue_manager.track_request(
                "test", "generate", "m",
                queue_manager.PRIORITY_INTERACTIVE,
            )

        depth = queue_manager.queue_depth()
        assert depth["total"] == 200


# ── Batch job DB stress ──────────────────────────────────────────────────────


class TestBatchJobStress:
    """Exercise batch job DB operations at scale."""

    def test_many_jobs_list_performance(self):
        """list_batch_jobs should handle hundreds of jobs efficiently."""
        import db

        N = 500
        for i in range(N):
            db.insert_batch_job(f"job-{i}", "stress", "model", f"prompt {i}", {})

        # Mark half as completed
        for i in range(0, N, 2):
            db.update_batch_job(f"job-{i}", status="completed",
                                completed_at=time.time())

        start = time.monotonic()
        all_jobs = db.list_batch_jobs()
        elapsed = time.monotonic() - start
        # Default list is capped at 200
        assert len(all_jobs) == 200
        assert elapsed < 0.1, f"list_batch_jobs took {elapsed:.3f}s"

        start = time.monotonic()
        queued = db.list_batch_jobs(status="queued")
        elapsed = time.monotonic() - start
        assert len(queued) == N // 2
        assert elapsed < 0.1

    def test_count_by_status_with_many_jobs(self):
        import db

        for i in range(300):
            db.insert_batch_job(f"j-{i}", "app", "model", "p", {})
        for i in range(100):
            db.update_batch_job(f"j-{i}", status="completed",
                                completed_at=time.time())
        for i in range(100, 200):
            db.update_batch_job(f"j-{i}", status="failed",
                                completed_at=time.time(), error="err")

        start = time.monotonic()
        counts = db.count_batch_jobs_by_status()
        elapsed = time.monotonic() - start

        assert counts["completed"] == 100
        assert counts["failed"] == 100
        assert counts["queued"] == 100
        assert elapsed < 0.05

    def test_drain_and_delete_at_scale(self):
        import db

        N = 400
        for i in range(N):
            db.insert_batch_job(f"j-{i}", "app", "model", "p", {})
        for i in range(N // 2):
            db.update_batch_job(f"j-{i}", status="completed",
                                completed_at=time.time())

        start = time.monotonic()
        drained = db.drain_queued_jobs()
        elapsed_drain = time.monotonic() - start
        assert drained == N // 2

        start = time.monotonic()
        deleted = db.delete_terminal_jobs()
        elapsed_delete = time.monotonic() - start
        # completed + cancelled (drained) = all N
        assert deleted == N

        assert elapsed_drain < 0.1
        assert elapsed_delete < 0.1


# ── Concurrent async operations ──────────────────────────────────────────────


class TestConcurrentAsync:
    """Simulate concurrent request patterns that hit the event loop."""

    async def test_concurrent_status_calls(self):
        """Multiple simultaneous status() calls should not block each other."""
        import gpu_router

        gpu_router._gpus.clear()
        gpu_router._init_gpus()

        async def call_status():
            return gpu_router.status()

        start = time.monotonic()
        results = await asyncio.gather(*[call_status() for _ in range(500)])
        elapsed = time.monotonic() - start

        assert all(r["routing"] == "round_robin" for r in results)
        assert elapsed < 1.0

    async def test_concurrent_dispatch_decisions(self, monkeypatch):
        """Parallel dispatcher waiters should not corrupt shared state."""
        import gpu_router
        import queue_manager
        import config

        gpu_router._gpus.clear()
        gpu_router._init_gpus()
        queue_manager._tracked.clear()
        for _bucket in queue_manager._buckets:
            _bucket.clear()
        for url in queue_manager._gpu_targets():
            queue_manager._gpu_busy[url] = False

        async def _noop_reload(gpu, model):
            gpu.model = model
        monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

        async def cycle():
            tid = queue_manager.track_request(
                "stress", "chat", "test-model:7b",
                queue_manager.PRIORITY_INTERACTIVE,
            )
            target = await queue_manager.wait_for_dispatch(tid)
            queue_manager.release(tid)
            return target

        results = await asyncio.gather(*[cycle() for _ in range(500)])
        assert all(
            r in (config.OLLAMA_0_URL, config.OLLAMA_1_URL) for r in results
        )

    async def test_reclaim_loop_does_not_spin(self):
        """reclaim_loop should yield to the event loop (sleep 30s).

        We run it alongside a counter task for 0.5s — the counter should
        get plenty of execution time if reclaim_loop is properly yielding.
        """
        import gpu_router

        gpu_router._gpus.clear()
        gpu_router._init_gpus()

        counter = 0

        async def count_ticks():
            nonlocal counter
            while True:
                counter += 1
                await asyncio.sleep(0.001)

        # Patch asyncio.sleep inside reclaim_loop so we don't wait 30s
        original_sleep = asyncio.sleep

        async def short_sleep(seconds):
            # reclaim_loop sleeps 30s — make it 0.05s for the test
            if seconds >= 10:
                await original_sleep(0.05)
            else:
                await original_sleep(seconds)

        with patch("asyncio.sleep", side_effect=short_sleep):
            reclaim_task = asyncio.create_task(gpu_router.reclaim_loop())
            count_task = asyncio.create_task(count_ticks())

            await asyncio.sleep(0.5)

            reclaim_task.cancel()
            count_task.cancel()
            try:
                await reclaim_task
            except asyncio.CancelledError:
                pass
            try:
                await count_task
            except asyncio.CancelledError:
                pass

        # If reclaim_loop were spinning, counter would be near 0
        assert counter > 50, (
            f"counter only reached {counter} — reclaim_loop may be "
            f"starving the event loop"
        )
