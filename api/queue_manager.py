"""
queue_manager.py — Late-binding GPU dispatcher with five priority buckets.

Requests enter one of five FIFO buckets by priority.  A central dispatcher
assigns a GPU to the head of each bucket only when a GPU becomes idle.
Callers do not pick a GPU at submission time — the target is determined by
the dispatcher at the moment one can actually serve the request.  This is
"late binding": the analogue of a thread scheduler that commits a thread to
a CPU only at dispatch time.

Priority buckets
────────────────
1. CHAT       — real-time chat tokens to the user. Only this.
2. EMBEDDINGS — embedding requests (auto-routed by app.proxy_embeddings).
                Sub-second on nomic-embed-text, so they cannot meaningfully
                starve SHORT and they get out of the way of any interleaved
                bulk traffic in BACKGROUND. Repurposed from the original
                RESERVED slot on 2026-04-11 (merLLM#38) after observing
                lancellmot ingest pile up 20+ embeds behind 32b chats.
3. SHORT      — parsival short work: live scan analyze, situation synthesis
                during a scan, contacts parsing, on-demand single-item clicks.
4. FEEDBACK   — LLM work spawned because a background job produced something
                that needs timely downstream processing.
5. BACKGROUND — bulk: reanalyze items, end-of-run briefings, lancellmot
                extractor, document upload/indexing.

Strict priority: bucket N only dispatches when buckets 1..N-1 are all empty.

Design rules
────────────
1. Late binding.  ``track_request`` registers the request with target=None.
   ``wait_for_dispatch`` pushes it onto a bucket and awaits the assigned URL.
2. Strict priority.  Higher buckets drain completely before the dispatcher
   looks at lower buckets.  No preemption.
3. Static swap cost.  ``config.MODEL_SWAP_COST_SECONDS`` is the conservative
   upper bound on loading the largest model.  The decision rule treats a
   swap as always preferable to waiting behind a busy match, because the
   swap cost is deterministic while the wait is unbounded.
4. No preemption.  A running job is never interrupted; the dispatcher only
   acts at job-completion events (release, swap-completion, new arrival,
   health change).

Dispatch decision rule at the head of a bucket
──────────────────────────────────────────────
At dispatch time (some GPU is idle), ``_best_candidate_for(req)``:

  - If an idle GPU already holds the requested model → that GPU (zero cost).
  - Else if any GPU is idle → the least-recently-active idle GPU (will be
    swapped before dispatch, ≤ MODEL_SWAP_COST_SECONDS cost).
  - Else → None (request stays at the head of the bucket; dispatcher retries
    on the next idle event).

Batch job submission stays in this module (SQLite-persisted, retryable) but
executes through the same dispatcher as interactive requests, always at
``Priority.BACKGROUND``.
"""
import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Union

import httpx

log = logging.getLogger(__name__)

# Dedicated dispatcher-trace logger. INFO-level so it shows up in docker logs
# without having to bump the root logger. All dispatcher-trace lines start
# with "[dispatch]" so they're easy to grep out of the activity stream, or
# silenced via:
#   logging.getLogger("queue_manager.dispatch").setLevel(logging.WARNING)
dispatch_log = logging.getLogger("queue_manager.dispatch")
dispatch_log.setLevel(logging.INFO)

import config
import db
import notifications


class Priority(IntEnum):
    """Five priority buckets, drained strictly top-down.

    Lower integer value = higher priority. The dispatcher walks buckets in
    order and a bucket only gets a turn when every bucket above it is empty.
    """
    CHAT       = 0   # real-time chat from the user
    EMBEDDINGS = 1   # embedding requests (auto-routed at proxy_embeddings)
    SHORT      = 2   # parsival short foreground work
    FEEDBACK   = 3   # LLM work spawned by background jobs
    BACKGROUND = 4   # bulk: reanalyze, briefings, extractor


# Back-compat aliases — kept for one release while parsival and lancellmot migrate.
PRIORITY_INTERACTIVE = Priority.CHAT         # old "interactive" → CHAT
PRIORITY_BATCH       = Priority.BACKGROUND   # old "batch"       → BACKGROUND

# Canonical name → Priority mapping used by the header parser.
_PRIORITY_BY_NAME: dict[str, Priority] = {
    "chat":        Priority.CHAT,
    "embeddings":  Priority.EMBEDDINGS,
    "short":       Priority.SHORT,
    "feedback":    Priority.FEEDBACK,
    "background":  Priority.BACKGROUND,
    # Back-compat aliases.
    "interactive": Priority.CHAT,
    "batch":       Priority.BACKGROUND,
}


def priority_from_name(name: Optional[str], default: Priority = Priority.BACKGROUND) -> Priority:
    """Parse an ``X-Priority`` header value into a :class:`Priority`.

    Missing, blank, or unknown names fall back to ``default`` (BACKGROUND by
    default — safest: typos can't silently escalate work to a higher lane).
    """
    if not name:
        return default
    return _PRIORITY_BY_NAME.get(name.strip().lower(), default)


def priority_name(priority: Union[Priority, int]) -> str:
    """Return the canonical lowercase name for a priority value."""
    try:
        return Priority(int(priority)).name.lower()
    except (ValueError, TypeError):
        return "background"


class DispatchTimeout(Exception):
    """Raised when a CHAT request exceeds INTERACTIVE_QUEUE_TIMEOUT."""


class EmptyResponseError(Exception):
    """Raised when Ollama returns a blank ``response`` field on /api/generate.

    Not retried: the usual cause is the submitted prompt exceeding ``num_ctx``
    so Ollama silently left-truncates it — a retry with the same inputs would
    burn another full decode for the same empty result.
    """


# ── Request tracking ─────────────────────────────────────────────────────────


@dataclass
class TrackedRequest:
    id: str
    source: str            # "lancellmot", "parsival", "direct", ...
    request_type: str      # "chat", "generate", "embeddings", "batch"
    model: str
    priority: int
    target: Optional[str]  # GPU URL assigned by dispatcher (None until dispatched)
    status: str            # "queued", "running", "completed", "failed"
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    batch_job_id: Optional[str] = None
    # Dispatcher instrumentation: set when the waiter enters its bucket.
    queued_for_dispatch_at: Optional[float] = None


_tracked: dict[str, TrackedRequest] = {}


@dataclass
class _PendingRequest:
    tid: str
    model: str
    priority: int
    future: "asyncio.Future[str]"   # resolves to assigned target URL


# ── Dispatcher state ─────────────────────────────────────────────────────────

# One deque per Priority bucket, indexed by the IntEnum value. The dispatcher
# walks these in order (CHAT first, BACKGROUND last) and strictly drains each
# bucket before moving to the next one down.
_buckets: list[deque["_PendingRequest"]] = [deque() for _ in Priority]
_gpu_busy: dict[str, bool] = {}
_state_changed: Optional[asyncio.Event] = None
_dispatcher_task: Optional[asyncio.Task] = None
_watchdog_task: Optional[asyncio.Task] = None

# ── Pause state ──────────────────────────────────────────────────────────────
# Operator-controlled pause switch. When paused, the dispatcher stops handing
# out new GPU slots; in-flight work continues to completion, the reclaim loop
# still runs, and new requests pile up in their priority buckets. The paused
# flag is persisted to the ``settings`` table at set time so a power-outage
# mid-pause doesn't silently resume bulk work on reboot.
_paused: bool = False
_paused_since: Optional[float] = None

# Callback notified on every track/start/complete/fail so the SSE layer can
# push updates.  Set by app.py at startup.
_on_queue_change: Optional[callable] = None

# ── Dispatcher-trace state (for observability only) ──────────────────────────
# Monotonic counter and timestamps used to correlate wake events with dispatch
# outcomes. None of this is load-bearing: it exists purely so the logs can
# reveal where the dispatcher stalls and why.
_wake_counter: int = 0
_last_wake_ts: Optional[float] = None
_last_dispatch_ts: Optional[float] = None   # last time a head was actually popped


def _gpu_targets() -> list[str]:
    """Return the configured GPU target URLs."""
    return [config.OLLAMA_0_URL, config.OLLAMA_1_URL]


def _ensure_gpu_state() -> None:
    """Populate ``_gpu_busy`` for any configured GPU not yet seen."""
    for url in _gpu_targets():
        _gpu_busy.setdefault(url, False)


def _ensure_dispatcher() -> None:
    """
    Start the dispatcher and watchdog loops lazily on first use.

    Called from ``track_request`` so both the app lifespan and tests pick up
    the dispatcher without extra wiring.  Safe to call repeatedly.
    """
    global _state_changed, _dispatcher_task, _watchdog_task
    _ensure_gpu_state()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet (e.g. import-time in a sync test) — defer.
        return
    if _state_changed is None:
        _state_changed = asyncio.Event()
    if _dispatcher_task is None or _dispatcher_task.done():
        _dispatcher_task = loop.create_task(_dispatcher_loop())
    if _watchdog_task is None or _watchdog_task.done():
        _watchdog_task = loop.create_task(_watchdog_loop())


def gpu_slot_busy(target: str) -> bool:
    """Return True if ``target`` is currently running or swapping."""
    _ensure_gpu_state()
    return _gpu_busy.get(target, False)


def set_queue_change_callback(cb: callable) -> None:
    global _on_queue_change
    _on_queue_change = cb


def _notify_change() -> None:
    if _on_queue_change:
        try:
            _on_queue_change()
        except Exception:
            pass


def _wake_dispatcher(reason: str = "unspecified") -> None:
    if _state_changed is not None:
        if not _state_changed.is_set():
            dispatch_log.info("[dispatch] wake requested reason=%s", reason)
        _state_changed.set()


def is_paused() -> bool:
    return _paused


def paused_since() -> Optional[float]:
    return _paused_since


def set_paused(paused: bool, persist: bool = True) -> bool:
    """Flip the operator pause switch.

    While paused the dispatcher refuses to hand out new GPU slots. In-flight
    work keeps running, the reclaim loop still reloads idle GPUs, and queued
    waiters simply wait longer. Unpausing wakes the dispatcher so the queued
    heads get another chance immediately.

    The flag is persisted to the ``settings`` table (``queue_paused``) so
    restarts remember the paused state — a power outage mid-pause must not
    silently resume bulk work on reboot. Tests that don't want the side
    effect pass ``persist=False``.

    Returns True if the flag changed, False if the call was a no-op.
    """
    global _paused, _paused_since
    if _paused == paused:
        return False
    _paused = paused
    _paused_since = time.time() if paused else None
    if persist:
        try:
            # Local import: db imports config, and test harnesses frequently
            # stub or reset db between cases — keep the coupling lazy.
            import db
            db.save_settings({"queue_paused": bool(paused)})
        except Exception:
            log.exception("failed to persist queue_paused=%s to settings", paused)
    log.info("queue %s by operator", "paused" if paused else "resumed")
    _notify_change()
    if not paused:
        _wake_dispatcher(reason="resume")
    return True


def track_request(source: str, request_type: str, model: str,
                  priority: Union[Priority, int],
                  batch_job_id: Optional[str] = None) -> str:
    """
    Register a new request in the tracker.

    The request starts with ``target=None``; the dispatcher assigns a URL at
    dispatch time.  Returns a tracking ID.

    ``priority`` may be a :class:`Priority` enum member or its int value.
    """
    _ensure_dispatcher()
    req_id = str(uuid.uuid4())
    _tracked[req_id] = TrackedRequest(
        id=req_id,
        source=source,
        request_type=request_type,
        model=model,
        priority=int(priority),
        target=None,
        status="queued",
        submitted_at=time.time(),
        batch_job_id=batch_job_id,
    )
    _notify_change()
    return req_id


async def wait_for_dispatch(tracking_id: str) -> str:
    """
    Push the tracked request onto its priority bucket and await a GPU
    assignment from the dispatcher.

    Returns the target URL once the dispatcher has marked the GPU busy and
    completed any required model swap.  Raises :class:`DispatchTimeout` if
    a ``CHAT`` request exceeds ``config.INTERACTIVE_QUEUE_TIMEOUT``; all
    other buckets wait indefinitely.
    """
    entry = _tracked.get(tracking_id)
    if entry is None:
        raise RuntimeError(f"unknown tracking_id {tracking_id}")

    _ensure_dispatcher()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    pending = _PendingRequest(
        tid=tracking_id,
        model=entry.model or config.DEFAULT_MODEL,
        priority=entry.priority,
        future=future,
    )
    _buckets[entry.priority].append(pending)
    entry.queued_for_dispatch_at = time.time()
    dispatch_log.info(
        "[dispatch] enqueue tid=%s src=%s type=%s model=%s bucket=%s depth=%d",
        tracking_id[:8], entry.source, entry.request_type, entry.model,
        priority_name(entry.priority), len(_buckets[entry.priority]),
    )

    _wake_dispatcher(reason=f"enqueue:{priority_name(entry.priority)}")

    try:
        if entry.priority == Priority.CHAT:
            target = await asyncio.wait_for(future, timeout=config.INTERACTIVE_QUEUE_TIMEOUT)
        else:
            target = await future
    except asyncio.TimeoutError:
        _remove_pending(tracking_id)
        if tracking_id in _tracked:
            _tracked[tracking_id].status = "failed"
            _tracked[tracking_id].error = "Timed out waiting for GPU slot"
            _tracked[tracking_id].completed_at = time.time()
            _notify_change()
            loop.call_later(10, _remove_tracked, tracking_id)
        raise DispatchTimeout("chat request exceeded INTERACTIVE_QUEUE_TIMEOUT")

    return target


def _remove_pending(tracking_id: str) -> None:
    """Remove a pending request from whichever bucket it's in (if any)."""
    for bucket in _buckets:
        for i, p in enumerate(bucket):
            if p.tid == tracking_id:
                del bucket[i]
                return


def release(tracking_id: str) -> None:
    """
    Mark a dispatched request complete and return its GPU to the idle pool.

    Wakes the dispatcher so the next pipe head can be evaluated.
    """
    entry = _tracked.get(tracking_id)
    if entry is None:
        return

    if entry.target is not None:
        _gpu_busy[entry.target] = False

    entry.status = "completed"
    entry.completed_at = time.time()
    ran_for = (entry.completed_at - entry.started_at) if entry.started_at else None
    dispatch_log.info(
        "[dispatch] release tid=%s gpu=%s ran=%.1fs src=%s bucket=%s",
        tracking_id[:8],
        entry.target[-5:] if entry.target else "?",
        ran_for if ran_for is not None else -1.0,
        entry.source, priority_name(entry.priority),
    )
    _notify_change()
    _wake_dispatcher(reason="release")

    # Non-batch requests drop from the tracker shortly after completion.
    # Jobs backed by a batch_job_id stay visible (persisted in SQLite anyway).
    if entry.batch_job_id is None:
        try:
            asyncio.get_running_loop().call_later(5, _remove_tracked, tracking_id)
        except RuntimeError:
            _remove_tracked(tracking_id)


def fail_request(tracking_id: str, error: str = "") -> None:
    """
    Mark a tracked request as failed (e.g. Ollama error during generation)
    and return its GPU to the idle pool.
    """
    entry = _tracked.get(tracking_id)
    if entry is None:
        return

    if entry.target is not None:
        _gpu_busy[entry.target] = False

    entry.status = "failed"
    entry.error = error
    entry.completed_at = time.time()
    ran_for = (entry.completed_at - entry.started_at) if entry.started_at else None
    dispatch_log.info(
        "[dispatch] fail tid=%s gpu=%s ran=%.1fs err=%s",
        tracking_id[:8],
        entry.target[-5:] if entry.target else "?",
        ran_for if ran_for is not None else -1.0,
        (error or "")[:120],
    )
    _notify_change()
    _wake_dispatcher(reason="fail")

    if entry.batch_job_id is None:
        try:
            asyncio.get_running_loop().call_later(10, _remove_tracked, tracking_id)
        except RuntimeError:
            _remove_tracked(tracking_id)


def _remove_tracked(tracking_id: str) -> None:
    _tracked.pop(tracking_id, None)
    _notify_change()


def cancel_tracked(tracking_id: str, reason: str = "cancelled") -> None:
    """
    Idempotent cleanup for a tracked request that exited abnormally.

    Handles three cases safely, so a proxy-handler ``finally`` block can call
    this without knowing which state the request is in:

    1. **Still in a priority bucket** (``status == "queued"``): removes the
       ``_PendingRequest`` from the bucket, resolves its future with an
       exception so any concurrent waiter unblocks, and marks the tracked
       entry ``failed``. Prevents the dispatcher from later popping a dead
       waiter and wasting a GPU slot — the original bug behind the
       2026-04-11 feedback-slot wedge.
    2. **Already running** (``status == "running"``): delegates to
       ``fail_request`` which releases ``_gpu_busy[target]`` and wakes the
       dispatcher. The same path is safe against a normal ``release`` that
       happens to race.
    3. **Already terminal / already removed**: no-op.

    Call this from every proxy-handler ``finally`` block so that whether the
    exit path is a normal completion, an upstream error, a client disconnect
    (``GeneratorExit`` into a pending ``await``), or an unexpected exception,
    the slot always returns to the idle pool.
    """
    entry = _tracked.get(tracking_id)
    if entry is None:
        return

    if entry.status == "queued":
        # Still waiting in a bucket — pull the pending entry out and
        # resolve its future so any concurrent awaiter unwinds cleanly.
        for bucket in _buckets:
            for i, p in enumerate(bucket):
                if p.tid == tracking_id:
                    del bucket[i]
                    if not p.future.done():
                        p.future.set_exception(
                            RuntimeError(f"request cancelled: {reason}")
                        )
                    break
        entry.status = "failed"
        entry.error = reason
        entry.completed_at = time.time()
        dispatch_log.info(
            "[dispatch] cancel tid=%s state=queued reason=%s",
            tracking_id[:8], reason,
        )
        _notify_change()
        _wake_dispatcher(reason="cancel")
        if entry.batch_job_id is None:
            try:
                asyncio.get_running_loop().call_later(10, _remove_tracked, tracking_id)
            except RuntimeError:
                _remove_tracked(tracking_id)
        return

    if entry.status == "running":
        # Slot is already assigned — route through fail_request so the
        # GPU is released and the dispatcher wakes.
        dispatch_log.info(
            "[dispatch] cancel tid=%s state=running reason=%s",
            tracking_id[:8], reason,
        )
        fail_request(tracking_id, reason)
        return

    # terminal (completed/failed) — nothing to do.


def clear_tracker(force: bool = False) -> dict:
    """
    Drop entries from the in-memory tracker to restore a known baseline.

    Default behaviour (safe):
        Only entries with status in {"completed", "failed"} are removed.
        In-flight work (``queued`` / ``running``) is left untouched.
        Batch-job-backed entries that would normally stay visible forever
        (see ``release`` line 327) are cleared.

    ``force=True`` (unsafe — breaks in-flight requests):
        Also cancels every queued waiter (their ``_PendingRequest.future``
        is resolved with an exception so ``wait_for_dispatch`` raises and
        the client gets an error), and marks every running entry as
        ``failed``, releasing its GPU back to the idle pool. Any outbound
        HTTP request currently held by the dispatcher is *not* cancelled
        at the socket level — the httpx client keeps running until the
        upstream Ollama responds — but the tracker will no longer reflect
        it, so a subsequent ``release`` or ``fail_request`` becomes a
        no-op. Use only for admin reset or integration-test setup; a
        misfire during normal traffic will leak GPU accounting until the
        orphaned request completes.

    Returns a dict with per-status removal counts, e.g.::

        {"completed": 12, "failed": 0, "queued": 0, "running": 0}
    """
    removed = {"completed": 0, "failed": 0, "queued": 0, "running": 0}

    # Step 1: always drop terminal entries.
    for tid in [tid for tid, e in _tracked.items()
                if e.status in ("completed", "failed")]:
        removed[_tracked[tid].status] += 1
        _tracked.pop(tid, None)

    if not force:
        _notify_change()
        return removed

    # Step 2: force-clear queued entries. Resolve their pending futures
    # with a RuntimeError so wait_for_dispatch raises and the client path
    # unwinds cleanly.
    for bucket in _buckets:
        while bucket:
            pending = bucket.popleft()
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("queue cleared by admin force-reset")
                )
            entry = _tracked.pop(pending.tid, None)
            if entry is not None:
                removed["queued"] += 1

    # Step 3: force-clear running entries. Mark them failed, release
    # their GPU so the dispatcher can reuse it, drop from _tracked.
    # The in-flight httpx request still runs to completion upstream;
    # release()/fail_request() called by that path will no-op because
    # the entry is gone.
    for tid in [tid for tid, e in _tracked.items() if e.status == "running"]:
        entry = _tracked[tid]
        if entry.target is not None:
            _gpu_busy[entry.target] = False
        removed["running"] += 1
        _tracked.pop(tid, None)

    _notify_change()
    _wake_dispatcher()
    return removed


# ── Dispatcher core ──────────────────────────────────────────────────────────


def _best_candidate_for(model: str) -> Optional[str]:
    """
    Return the URL of the best idle GPU for ``model``, or None if no GPU
    is currently dispatchable.

    Decision rule:
      1. Prefer an idle GPU that already holds the requested model (affinity).
      2. Otherwise, any idle GPU is a candidate (will be swapped).
      3. Exclude unhealthy GPUs.
      4. Exclude thermally paused GPUs — their in-flight work is preserved
         but no new work is dispatched until they cool down.
    """
    # Deferred import to avoid module-load cycle.
    import gpu_router

    _ensure_gpu_state()
    dispatchable = {url for url in _gpu_targets()
                    if gpu_router.is_dispatchable(url)}
    if not dispatchable:
        return None

    idle = [url for url in _gpu_targets()
            if url in dispatchable and not _gpu_busy.get(url, False)]
    if not idle:
        return None

    # Prefer an idle GPU that already holds the requested model.
    for url in idle:
        gpu_state = gpu_router._gpus.get(url)
        if gpu_state and gpu_state.model == model:
            return url

    # No affinity match — any idle GPU will do. Pick the least-recently-active
    # one so load spreads evenly over time.
    idle.sort(key=lambda u: gpu_router._gpus[u].last_active if u in gpu_router._gpus else 0)
    return idle[0]


def _try_dispatch_heads() -> None:
    """
    Walk the priority buckets from CHAT down to BACKGROUND.  For each
    bucket, try to assign a GPU to the head.  Stop at the first head that
    cannot be dispatched — FIFO must be preserved within a bucket, so we
    never skip a head.  A lower-priority bucket only gets a turn when every
    higher-priority bucket is empty.
    """
    global _last_dispatch_ts

    # Paused by operator: do not hand out any new slots. In-flight work is
    # untouched; the walk simply no-ops until set_paused(False) wakes us.
    if _paused:
        dispatch_log.info(
            "[dispatch] walk skipped reason=paused paused_since=%s",
            _paused_since,
        )
        return

    # Snapshot of dispatch-time state so the log has enough context to
    # explain any stall we observe.
    import gpu_router
    depths = {priority_name(p): len(_buckets[p]) for p in Priority}
    busy_snapshot = {u[-5:]: _gpu_busy.get(u, False) for u in _gpu_targets()}
    try:
        dispatchable = {u for u in _gpu_targets() if gpu_router.is_dispatchable(u)}
    except Exception:
        dispatchable = set()
    idle_count = sum(
        1 for u in _gpu_targets()
        if u in dispatchable and not _gpu_busy.get(u, False)
    )
    dispatch_log.info(
        "[dispatch] walk start depths=%s gpus_busy=%s idle_gpus=%d",
        depths, busy_snapshot, idle_count,
    )

    dispatched = 0
    stopped_reason = "buckets_drained"
    stopped_bucket: Optional[str] = None
    for prio_idx, bucket in enumerate(_buckets):
        if not bucket:
            continue
        while bucket:
            req = bucket[0]
            target = _best_candidate_for(req.model)
            if target is None:
                # Head of this bucket cannot dispatch right now. We must not
                # fall through to a lower-priority bucket while a higher
                # bucket still has pending work — strict priority forbids
                # any inversion, even when the lower bucket's head happens
                # to be a better model fit.
                stopped_reason = "no_candidate"
                stopped_bucket = priority_name(Priority(prio_idx))
                dispatch_log.info(
                    "[dispatch] walk stop dispatched=%d reason=%s bucket=%s "
                    "head_tid=%s head_model=%s remaining_in_bucket=%d",
                    dispatched, stopped_reason, stopped_bucket,
                    req.tid[:8], req.model, len(bucket),
                )
                return
            bucket.popleft()
            _gpu_busy[target] = True
            if req.tid in _tracked:
                _tracked[req.tid].target = target
            entry = _tracked.get(req.tid)
            wait_ms = -1.0
            if entry and entry.queued_for_dispatch_at:
                wait_ms = (time.time() - entry.queued_for_dispatch_at) * 1000
            dispatch_log.info(
                "[dispatch] pop tid=%s bucket=%s gpu=%s model=%s wait_ms=%.0f",
                req.tid[:8], priority_name(Priority(prio_idx)),
                target[-5:], req.model, wait_ms,
            )
            asyncio.create_task(_dispatch(req, target))
            dispatched += 1
            _last_dispatch_ts = time.time()

    dispatch_log.info(
        "[dispatch] walk end dispatched=%d reason=%s bucket=%s",
        dispatched, stopped_reason, stopped_bucket,
    )


async def _dispatch(req: _PendingRequest, target: str) -> None:
    """
    Perform any required model swap for ``target`` and then resolve the
    waiter's future with the assigned URL.  Runs as a background task so
    the dispatcher loop itself never blocks on the swap.
    """
    import gpu_router
    t0 = time.time()
    try:
        gpu_state = gpu_router._gpus.get(target)
        swap_ms: float = 0.0
        if gpu_state is not None and gpu_state.model != req.model:
            # Swap cost is bounded by MODEL_SWAP_COST_SECONDS in practice.
            # The dispatcher has already marked the GPU busy so no other
            # request can land here until the swap completes.
            dispatch_log.info(
                "[dispatch] swap start tid=%s gpu=%s %s -> %s",
                req.tid[:8], target[-5:], gpu_state.model, req.model,
            )
            swap_start = time.time()
            await gpu_router._reload_model(gpu_state, req.model)
            swap_ms = (time.time() - swap_start) * 1000
            dispatch_log.info(
                "[dispatch] swap done tid=%s gpu=%s ms=%.0f",
                req.tid[:8], target[-5:], swap_ms,
            )

        entry = _tracked.get(req.tid)
        if entry is not None:
            entry.status = "running"
            entry.started_at = time.time()
            entry.target = target
            _notify_change()

        if not req.future.done():
            req.future.set_result(target)
        dispatch_log.info(
            "[dispatch] handoff tid=%s gpu=%s total_ms=%.0f swap_ms=%.0f",
            req.tid[:8], target[-5:], (time.time() - t0) * 1000, swap_ms,
        )
    except Exception as exc:
        log.exception("dispatch failed for %s on %s", req.tid, target)
        dispatch_log.warning(
            "[dispatch] handoff_fail tid=%s gpu=%s err=%s",
            req.tid[:8], target[-5:], exc,
        )
        _gpu_busy[target] = False
        if not req.future.done():
            req.future.set_exception(exc)
        _wake_dispatcher(reason="handoff_fail")


async def _watchdog_loop() -> None:
    """
    Reclaim wedged slots whose dispatched request has exceeded its
    wall-clock budget.

    Background: a downstream Ollama call can hang in ways the proxy
    layer cannot detect (half-open socket, model unloaded mid-stream,
    upstream silently dropping the connection). When that happens
    ``release()`` is never called and the slot stays "running" forever,
    blocking strict-priority dispatch — the queue deadlocks even though
    no actual GPU work is in flight. The 2026-04-10 parsival re-analysis
    incident wedged both GPUs this way for ~13 minutes with 295 batch
    jobs queued behind two zombie ``feedback`` entries.

    The watchdog scans tracked entries on a fixed interval and force-fails
    any that have been ``running`` longer than ``SLOT_MAX_WALL_SECONDS``.
    ``fail_request()`` releases the GPU slot and wakes the dispatcher so
    queued work can finally make progress. The slot is the unit reclaimed,
    not the upstream request — if Ollama eventually responds, the proxy
    coroutine will discover the entry is gone and quietly no-op.
    """
    while True:
        try:
            await asyncio.sleep(config.WATCHDOG_INTERVAL_SECONDS)
            now = time.time()
            wedged: list[tuple[str, float]] = []
            for tid, entry in list(_tracked.items()):
                if entry.status != "running" or entry.started_at is None:
                    continue
                age = now - entry.started_at
                if age > config.SLOT_MAX_WALL_SECONDS:
                    wedged.append((tid, age))
            for tid, age in wedged:
                log.warning(
                    "watchdog: reclaiming wedged slot %s (age=%.0fs > budget=%ds)",
                    tid[:8], age, config.SLOT_MAX_WALL_SECONDS,
                )
                fail_request(
                    tid,
                    f"Watchdog reclaimed slot after {age:.0f}s "
                    f"exceeded {config.SLOT_MAX_WALL_SECONDS}s budget",
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("watchdog loop iteration failed — continuing")
            await asyncio.sleep(1.0)


async def _dispatcher_loop() -> None:
    """
    Central dispatch loop: wakes on every relevant state change (new
    arrival, release, swap completion, health transition) and tries to
    assign work to any idle GPU.
    """
    global _state_changed, _wake_counter, _last_wake_ts
    if _state_changed is None:
        _state_changed = asyncio.Event()

    while True:
        try:
            await _state_changed.wait()
            _state_changed.clear()
            _wake_counter += 1
            now = time.time()
            gap_s = (now - _last_wake_ts) if _last_wake_ts is not None else 0.0
            gap_since_dispatch = (
                (now - _last_dispatch_ts) if _last_dispatch_ts is not None else -1.0
            )
            dispatch_log.info(
                "[dispatch] wake #%d gap_since_last_wake=%.2fs gap_since_last_pop=%.2fs",
                _wake_counter, gap_s, gap_since_dispatch,
            )
            _last_wake_ts = now
            _try_dispatch_heads()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("dispatcher loop iteration failed — continuing")
            # Don't hot-loop on repeated failures.
            await asyncio.sleep(0.1)


# ── Reclaim-loop cooperation ─────────────────────────────────────────────────


def reserve_gpu(target: str) -> bool:
    """
    Mark a GPU busy from outside the dispatcher (e.g. the reclaim loop in
    ``gpu_router`` before a scheduled model swap).  Returns False if the
    GPU was already busy.
    """
    _ensure_gpu_state()
    if _gpu_busy.get(target, False):
        return False
    _gpu_busy[target] = True
    return True


def unreserve_gpu(target: str) -> None:
    """Release a GPU previously reserved via ``reserve_gpu`` and wake the dispatcher."""
    _gpu_busy[target] = False
    _wake_dispatcher()


# ── Queue visibility ─────────────────────────────────────────────────────────


def active_queue() -> list[dict]:
    """
    Return all currently tracked requests (queued + running + recently
    completed), sorted by submitted_at.
    """
    now = time.time()
    result = []
    for entry in sorted(_tracked.values(), key=lambda e: e.submitted_at):
        d = {
            "id":            entry.id,
            "source":        entry.source,
            "request_type":  entry.request_type,
            "model":         entry.model,
            "priority":      priority_name(entry.priority),
            "target":        entry.target,
            "status":        entry.status,
            "submitted_at":  entry.submitted_at,
            "started_at":    entry.started_at,
            "completed_at":  entry.completed_at,
            "error":         entry.error,
            "batch_job_id":  entry.batch_job_id,
        }
        if entry.status == "running" and entry.started_at:
            d["elapsed_sec"] = round(now - entry.started_at, 1)
        elif entry.status == "queued":
            d["waiting_sec"] = round(now - entry.submitted_at, 1)
        result.append(d)
    return result


def queue_depth() -> dict:
    """Summary counts for the status endpoint."""
    queued = sum(1 for e in _tracked.values() if e.status == "queued")
    running = sum(1 for e in _tracked.values() if e.status == "running")

    gpus: dict = {}
    for entry in _tracked.values():
        if entry.status in ("queued", "running") and entry.target:
            gpu = gpus.setdefault(entry.target, {"queued": 0, "running": 0})
            gpu[entry.status] += 1

    return {"queued": queued, "running": running, "total": queued + running, "gpus": gpus}


def pipe_depth() -> dict:
    """Depth of each dispatcher bucket (pre-dispatch waiters).

    Returns one key per :class:`Priority` name plus legacy ``interactive`` /
    ``batch`` aliases (CHAT / BACKGROUND) so existing dashboards and tests
    that read the old keys keep working during the transition.
    """
    depths: dict = {
        priority_name(p): len(_buckets[p]) for p in Priority
    }
    # Back-compat aliases.
    depths["interactive"] = depths["chat"]
    depths["batch"]       = depths["background"]
    return depths


# ── Batch job submission ──────────────────────────────────────────────────────


def submit_batch_job(source_app: str, model: str, prompt: str,
                     options: Optional[dict] = None) -> str:
    """
    Persist a batch job and enqueue it for processing.

    Returns the job ID. The job runs at batch priority whenever a GPU is free.
    """
    job_id = str(uuid.uuid4())
    db.insert_batch_job(job_id, source_app, model or config.DEFAULT_MODEL,
                        prompt, options or {})
    asyncio.ensure_future(_run_batch_job_async(job_id))
    return job_id


async def _run_batch_job_async(job_id: str) -> None:
    """Execute a batch job through the late-binding dispatcher."""
    import gpu_router
    # Deferred import: app imports queue_manager at module load time, so we
    # cannot import app at the top of this file. Batch jobs must funnel
    # through the same activity tracker that ``app._proxy`` uses, otherwise
    # the Ollama Instances card and the SSE stream show the GPU as idle for
    # the entire duration of the batch run even though it's hammering away.
    import app as _app

    job = db.get_batch_job(job_id)
    if not job or job["status"] != "queued":
        return

    retries = job.get("retries", 0)
    db.update_batch_job(job_id, status="running", started_at=time.time())

    options: dict = {}
    try:
        options = json.loads(job.get("options", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    # ── Defensive option defaults ────────────────────────────────────────
    # Belt-and-suspenders for the "unbounded reasoning" failure mode:
    # if a caller submitted a batch job without the three keys that keep
    # qwen3:* from wedging a slot (think:false, num_predict, num_ctx),
    # fill them in here so old jobs, buggy callers, and third-party clients
    # still run with safe bounds. We only fill in MISSING keys — if the
    # caller specified a value we respect it.
    #
    # num_ctx also serves a second purpose: it forces every Ollama instance
    # to converge on the same KV-cache size. Without it the two GPUs can
    # end up loaded with different context lengths (one from an earlier
    # request, one from a later one) and the larger-ctx GPU becomes
    # ~2× slower on identical workloads because of the bigger KV cache.
    _defaults = {
        "think":       False,     # disable qwen3 reasoning emission
        "num_predict": 768,       # bounded decode cap
        "num_ctx":     8192,      # fixed KV cache so both GPUs match
        "temperature": 0.1,       # deterministic-ish for extraction tasks
    }
    for k, v in _defaults.items():
        options.setdefault(k, v)

    body = {
        "model":   job["model"],
        "prompt":  job["prompt"],
        "stream":  False,
        "options": options,
    }

    tracking_id = track_request(
        source=job["source_app"],
        request_type="batch",
        model=job["model"],
        priority=Priority.BACKGROUND,
        batch_job_id=job_id,
    )

    try:
        dispatch_wait_start = time.time()
        target = await wait_for_dispatch(tracking_id)
        dispatch_wait_s = time.time() - dispatch_wait_start
        dispatch_log.info(
            "[dispatch] batch wait_done job=%s tid=%s gpu=%s wait=%.2fs",
            job_id[:8], tracking_id[:8], target[-5:], dispatch_wait_s,
        )
        _app._activity_set(target, job["model"], "/api/generate")
        try:
            # Bounded read timeout: defence-in-depth behind the watchdog.
            # If the watchdog has already reclaimed the slot, this still
            # eventually unblocks the coroutine instead of leaking the
            # httpx connection forever.
            http_start = time.time()
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(config.PROXY_READ_TIMEOUT_SECONDS),
            ) as client:
                resp = await client.post(f"{target}/api/generate", json=body)
                resp.raise_for_status()
                payload = resp.json()
                result_text = payload.get("response", "")
            http_s = time.time() - http_start
            dispatch_log.info(
                "[dispatch] batch http_done job=%s tid=%s gpu=%s http=%.2fs "
                "resp_chars=%d",
                job_id[:8], tracking_id[:8], target[-5:], http_s,
                len(result_text),
            )
            gpu_router.record_activity(target)
            if not result_text.strip():
                prompt_tokens = payload.get("prompt_eval_count")
                done_reason   = payload.get("done_reason")
                num_ctx       = options.get("num_ctx")
                raise EmptyResponseError(
                    f"Ollama returned empty response "
                    f"(done_reason={done_reason!r}, "
                    f"prompt_tokens={prompt_tokens}, num_ctx={num_ctx}); "
                    f"likely prompt exceeds num_ctx and was truncated"
                )
            db.update_batch_job(job_id, status="completed",
                                completed_at=time.time(), result=result_text)
            log.info("batch job %s completed", job_id[:8])
            completed_job = db.get_batch_job(job_id)
            if completed_job:
                asyncio.ensure_future(notifications.dispatch(
                    completed_job,
                    webhook_url=config.NOTIFICATION_WEBHOOK_URL or None,
                ))
        finally:
            # Order matters: release() first so the dispatcher can hand the
            # GPU to the next waiter in the same event-loop tick. If we
            # clear activity before releasing, the UI briefly sees the GPU
            # as idle even though a new job is about to start on it — the
            # visible "stop/start" flicker the user reported during large
            # batches. The new request's _activity_set() will overwrite
            # ours, so the clear is only there to cover the "no work left"
            # case.
            release(tracking_id)
            _app._activity_clear(target)
    except Exception as exc:
        fail_request(tracking_id, str(exc))
        prev_error = (job.get("error") or "").strip()
        # Empty responses are deterministic given the same prompt — retrying
        # would just burn another full decode for the same blank result.
        is_retryable = not isinstance(exc, EmptyResponseError)
        if is_retryable and retries < config.BATCH_MAX_RETRIES:
            backoff    = 30 * (4 ** retries)
            retry_after = time.time() + backoff
            new_error  = f"{prev_error} [attempt {retries + 1} failed: {exc}]".lstrip()
            db.update_batch_job(
                job_id,
                status="queued",
                retries=retries + 1,
                retry_after=retry_after,
                started_at=None,
                error=new_error,
            )
            log.warning(
                "batch job %s failed (attempt %d/%d), retrying in %ds: %s",
                job_id[:8], retries + 1, config.BATCH_MAX_RETRIES + 1, backoff, exc,
            )
            await asyncio.sleep(backoff)
            await _run_batch_job_async(job_id)
        else:
            final_error = f"{prev_error} [final failure after {retries + 1} attempts: {exc}]".lstrip()
            db.update_batch_job(job_id, status="failed",
                                completed_at=time.time(), error=final_error)
            log.error("batch job %s permanently failed after %d attempt(s): %s",
                      job_id[:8], retries + 1, exc)
            failed_job = db.get_batch_job(job_id)
            if failed_job:
                asyncio.ensure_future(notifications.dispatch(
                    failed_job,
                    webhook_url=config.NOTIFICATION_WEBHOOK_URL or None,
                ))


def get_queue_position(job_id: str) -> Optional[int]:
    """
    Return the zero-based queue position of a queued job (number of jobs ahead).

    Returns None if the job is not in queued status.
    """
    jobs = db.list_batch_jobs(status="queued")
    queued_ids = [j["id"] for j in sorted(jobs, key=lambda j: j.get("submitted_at", ""))]
    try:
        return queued_ids.index(job_id)
    except ValueError:
        return None


def get_job_status(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job:
        return None
    result: dict = {
        "id":           job["id"],
        "source_app":   job["source_app"],
        "model":        job["model"],
        "status":       job["status"],
        "submitted_at": job["submitted_at"],
        "started_at":   job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error":        job.get("error"),
        "retries":      job.get("retries", 0),
        "retry_after":  job.get("retry_after"),
    }
    if job["status"] == "queued":
        result["queue_position"] = get_queue_position(job_id)
        result["estimated_start"] = None
    return result


def get_job_result(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job or job["status"] != "completed":
        return None
    return {
        "id":     job["id"],
        "result": job.get("result", ""),
    }
