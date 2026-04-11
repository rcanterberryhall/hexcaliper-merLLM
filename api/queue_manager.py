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
2. RESERVED   — empty. Reserved for future use.
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

import config
import db
import notifications


class Priority(IntEnum):
    """Five priority buckets, drained strictly top-down.

    Lower integer value = higher priority. The dispatcher walks buckets in
    order and a bucket only gets a turn when every bucket above it is empty.
    """
    CHAT       = 0   # real-time chat from the user
    RESERVED   = 1   # empty; reserved for future use
    SHORT      = 2   # parsival short foreground work
    FEEDBACK   = 3   # LLM work spawned by background jobs
    BACKGROUND = 4   # bulk: reanalyze, briefings, extractor


# Back-compat aliases — kept for one release while parsival and lancellmot migrate.
PRIORITY_INTERACTIVE = Priority.CHAT         # old "interactive" → CHAT
PRIORITY_BATCH       = Priority.BACKGROUND   # old "batch"       → BACKGROUND

# Canonical name → Priority mapping used by the header parser.
_PRIORITY_BY_NAME: dict[str, Priority] = {
    "chat":        Priority.CHAT,
    "reserved":    Priority.RESERVED,
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

# Callback notified on every track/start/complete/fail so the SSE layer can
# push updates.  Set by app.py at startup.
_on_queue_change: Optional[callable] = None


def _gpu_targets() -> list[str]:
    """Return the configured GPU target URLs."""
    return [config.OLLAMA_0_URL, config.OLLAMA_1_URL]


def _ensure_gpu_state() -> None:
    """Populate ``_gpu_busy`` for any configured GPU not yet seen."""
    for url in _gpu_targets():
        _gpu_busy.setdefault(url, False)


def _ensure_dispatcher() -> None:
    """
    Start the dispatcher loop lazily on first use.

    Called from ``track_request`` so both the app lifespan and tests pick up
    the dispatcher without extra wiring.  Safe to call repeatedly.
    """
    global _state_changed, _dispatcher_task
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


def _wake_dispatcher() -> None:
    if _state_changed is not None:
        _state_changed.set()


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

    _wake_dispatcher()

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
    _notify_change()
    _wake_dispatcher()

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
    _notify_change()
    _wake_dispatcher()

    if entry.batch_job_id is None:
        try:
            asyncio.get_running_loop().call_later(10, _remove_tracked, tracking_id)
        except RuntimeError:
            _remove_tracked(tracking_id)


def _remove_tracked(tracking_id: str) -> None:
    _tracked.pop(tracking_id, None)
    _notify_change()


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
    """
    # Deferred import to avoid module-load cycle.
    import gpu_router

    _ensure_gpu_state()
    healthy = {g.url for g in gpu_router._healthy_gpus()}
    if not healthy:
        return None

    idle = [url for url in _gpu_targets()
            if url in healthy and not _gpu_busy.get(url, False)]
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
    for bucket in _buckets:
        while bucket:
            req = bucket[0]
            target = _best_candidate_for(req.model)
            if target is None:
                # Head of this bucket cannot dispatch right now. We must not
                # fall through to a lower-priority bucket while a higher
                # bucket still has pending work — strict priority forbids
                # any inversion, even when the lower bucket's head happens
                # to be a better model fit.
                return
            bucket.popleft()
            _gpu_busy[target] = True
            if req.tid in _tracked:
                _tracked[req.tid].target = target
            asyncio.create_task(_dispatch(req, target))


async def _dispatch(req: _PendingRequest, target: str) -> None:
    """
    Perform any required model swap for ``target`` and then resolve the
    waiter's future with the assigned URL.  Runs as a background task so
    the dispatcher loop itself never blocks on the swap.
    """
    import gpu_router
    try:
        gpu_state = gpu_router._gpus.get(target)
        if gpu_state is not None and gpu_state.model != req.model:
            # Swap cost is bounded by MODEL_SWAP_COST_SECONDS in practice.
            # The dispatcher has already marked the GPU busy so no other
            # request can land here until the swap completes.
            await gpu_router._reload_model(gpu_state, req.model)

        entry = _tracked.get(req.tid)
        if entry is not None:
            entry.status = "running"
            entry.started_at = time.time()
            entry.target = target
            _notify_change()

        if not req.future.done():
            req.future.set_result(target)
    except Exception as exc:
        log.exception("dispatch failed for %s on %s", req.tid, target)
        _gpu_busy[target] = False
        if not req.future.done():
            req.future.set_exception(exc)
        _wake_dispatcher()


async def _dispatcher_loop() -> None:
    """
    Central dispatch loop: wakes on every relevant state change (new
    arrival, release, swap completion, health transition) and tries to
    assign work to any idle GPU.
    """
    global _state_changed
    if _state_changed is None:
        _state_changed = asyncio.Event()

    while True:
        try:
            await _state_changed.wait()
            _state_changed.clear()
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
        target = await wait_for_dispatch(tracking_id)
        _app._activity_set(target, job["model"], "/api/generate")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                resp = await client.post(f"{target}/api/generate", json=body)
                resp.raise_for_status()
                result_text = resp.json().get("response", "")
            gpu_router.record_activity(target)
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
            _app._activity_clear(target)
            release(tracking_id)
    except Exception as exc:
        fail_request(tracking_id, str(exc))
        prev_error = (job.get("error") or "").strip()
        if retries < config.BATCH_MAX_RETRIES:
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
