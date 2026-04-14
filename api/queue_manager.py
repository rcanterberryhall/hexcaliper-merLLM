"""
queue_manager.py — Pure-FSM scheduler over five priority buckets (merLLM#55).

Requests enter one of five FIFO buckets by priority. A single tick loop
drives a deterministic Slot FSM (see ``scheduler.py``) which decides when
to load models and when to start work. Callers do not pick a GPU at submit
time — the slot is chosen by ``dispatch_pass`` at the moment one is ready
to serve the request.

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

The orchestrator owns: the tick loop, effect application (load/start/end),
durable pending_work mirroring, and busy-timeout sweeping. The scheduler
itself is pure (state in → state+effects out) and lives in scheduler.py.

Batch job submission stays in this module (SQLite-persisted, retryable) but
executes through the same path as interactive requests, always at
``Priority.BACKGROUND``.
"""
import asyncio
import json
import logging
import time
import uuid
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


# ── Pause state ──────────────────────────────────────────────────────────────
# Operator-controlled pause switch. While paused the tick loop skips
# dispatch and stage; in-flight work continues to completion. Persisted to
# the ``settings`` table so a power-outage mid-pause does not silently
# resume bulk work on reboot.
_paused: bool = False
_paused_since: Optional[float] = None

# Callback notified on every track/start/complete/fail so the SSE layer can
# push updates. Set by app.py at startup.
_on_queue_change: Optional[callable] = None


def gpu_slot_busy(target: str) -> bool:
    """Return True if ``target`` is currently running or swapping.

    Reads v2 Slot state: BUSY means an inference is in flight, LOADING
    means a model swap is in flight. Either way, no new work should be
    handed to this GPU.
    """
    for s in _slots:
        if s.url == target:
            return s.state in (SlotState.BUSY, SlotState.LOADING)
    return False


def set_queue_change_callback(cb: callable) -> None:
    global _on_queue_change
    _on_queue_change = cb


def _notify_change() -> None:
    if _on_queue_change:
        try:
            _on_queue_change()
        except Exception:
            pass


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
        _wake_tick(reason="resume")
    return True


def track_request(source: str, request_type: str, model: str,
                  priority: Union[Priority, int],
                  batch_job_id: Optional[str] = None) -> str:
    """Register a new request and enqueue it into its priority bucket.

    Returns a tracking id. The caller then awaits ``wait_for_dispatch``
    with that id to block until the tick loop assigns a GPU.

    v2 contract (merLLM#55 cutover):
      * A durable ``pending_work`` row is inserted before we touch the
        in-memory bucket, so a crash between rows-written and bucket-pushed
        is benign (boot wipes pending_work anyway).
      * The job dict is appended to ``_buckets_v2[priority]`` and remembered
        in ``_tid_to_job`` so ``cancel_tracked`` can find it.
      * The tick loop is started lazily if an event loop is running.
    """
    tid = str(uuid.uuid4())
    submitted_at = time.time()
    prio = int(priority)

    try:
        db.insert_pending(
            req_id=tid, source=source, request_type=request_type,
            model=model, priority=prio, batch_job_id=batch_job_id,
        )
    except Exception:
        log.exception("[track] insert_pending failed tid=%s", tid)
        raise

    job = _make_job(
        tid=tid, source=source, request_type=request_type, model=model,
        priority=prio, batch_job_id=batch_job_id, submitted_at=submitted_at,
    )
    _buckets_v2[prio].append(job)
    _tid_to_job[tid] = job

    _ensure_tick()
    _wake_tick(reason=f"enqueue:{priority_name(prio)}")
    _notify_change()
    return tid


async def wait_for_dispatch(tracking_id: str) -> str:
    """Await a GPU assignment for ``tracking_id``. Returns the target URL.

    Raises ``DispatchTimeout`` if a CHAT request exceeds
    ``INTERACTIVE_QUEUE_TIMEOUT``; other buckets wait indefinitely.
    Raises ``RuntimeError`` if the tid was cancelled while waiting.
    """
    job = _tid_to_job.get(tracking_id)
    if job is None:
        raise RuntimeError(f"unknown tracking_id {tracking_id}")

    _ensure_tick()
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    _inflight[tracking_id] = fut
    # A fresh enqueue may have happened before _ensure_tick finished setting
    # up the event; kick again to cover that race.
    _wake_tick(reason=f"wait:{priority_name(job['priority'])}")

    try:
        if job["priority"] == Priority.CHAT:
            return await asyncio.wait_for(fut, timeout=config.INTERACTIVE_QUEUE_TIMEOUT)
        return await fut
    except asyncio.TimeoutError:
        # CHAT timed out in its bucket — pull it out and clean up.
        cancel_tracked(tracking_id, reason="chat_queue_timeout")
        raise DispatchTimeout("chat request exceeded INTERACTIVE_QUEUE_TIMEOUT")


def _slot_idx_owning(tid: str) -> Optional[int]:
    """Return the index of the slot currently running ``tid``, or None."""
    for i, s in enumerate(_slots):
        cj = s.current_job
        if cj is not None and cj.get("tid") == tid:
            return i
    return None


def release(tracking_id: str) -> None:
    """Mark a dispatched request complete. Feeds WORK_END(ok) on its slot."""
    idx = _slot_idx_owning(tracking_id)
    if idx is None:
        # Not on any slot — either already released, cancelled, or tid unknown.
        return
    for e in _feed_event(idx, Event.WORK_END, outcome="ok"):
        _apply_effect(e)
    _notify_change()
    _wake_tick(reason="release")


def fail_request(tracking_id: str, error: str = "") -> None:
    """Mark a dispatched request failed. Feeds WORK_END(fail) on its slot."""
    idx = _slot_idx_owning(tracking_id)
    if idx is None:
        return
    dispatch_log.info(
        "[dispatch] fail tid=%s err=%s",
        tracking_id[:8], (error or "")[:120],
    )
    for e in _feed_event(idx, Event.WORK_END, outcome="fail"):
        _apply_effect(e)
    _notify_change()
    _wake_tick(reason="fail")


def cancel_tracked(tracking_id: str, reason: str = "cancelled") -> None:
    """Idempotent cleanup for a request that exited abnormally.

    Three cases, safe in all of them:

    1. **Still in a priority bucket**: pop from bucket, delete pending_work,
       reject the waiter future. Prevents the scheduler from popping a dead
       waiter and locking a slot on a disconnected client.
    2. **Already on a slot (BUSY)**: route through ``fail_request`` so
       WORK_END fires and the slot recovers.
    3. **Terminal / unknown**: no-op.

    Called from every proxy-handler ``finally`` block so client disconnect,
    upstream error, and normal completion all share one cleanup path.
    """
    job = _tid_to_job.get(tracking_id)
    idx = _slot_idx_owning(tracking_id)

    if idx is not None:
        dispatch_log.info(
            "[dispatch] cancel tid=%s state=running reason=%s",
            tracking_id[:8], reason,
        )
        fail_request(tracking_id, reason)
        return

    if job is not None:
        # Still in a bucket — remove it.
        prio = job["priority"]
        try:
            _buckets_v2[prio].remove(job)
        except ValueError:
            pass
        _tid_to_job.pop(tracking_id, None)
        try:
            db.delete_pending(tracking_id)
        except Exception:
            log.exception("[cancel] delete_pending failed tid=%s", tracking_id)
        fut = _inflight.pop(tracking_id, None)
        if fut is not None and not fut.done():
            fut.set_exception(RuntimeError(f"request cancelled: {reason}"))
        dispatch_log.info(
            "[dispatch] cancel tid=%s state=queued reason=%s",
            tracking_id[:8], reason,
        )
        _notify_change()
        _wake_tick(reason="cancel")
        return

    # Unknown / already terminal — nothing to do.


# ── gpu_router shims (full removal in commit 5) ─────────────────────────────
# gpu_router.reclaim_loop still calls these to coordinate model swaps with
# the old dispatcher's busy bitmap. Commit 5 deletes reclaim_loop and pushes
# all model-swap decisions through the FSM, at which point these vanish.

def reserve_gpu(target: str) -> bool:
    """No-op shim. Returns True so reclaim_loop's existing branches work."""
    return True


def unreserve_gpu(target: str) -> None:
    """No-op shim; commit 5 removes the caller in gpu_router.reclaim_loop."""
    return None


def _wake_dispatcher(reason: str = "unspecified") -> None:
    """Compat shim: thermal_resume in gpu_router calls this. Commit 5 deletes it."""
    _wake_tick(reason=reason)


# ── Queue visibility ─────────────────────────────────────────────────────────


def active_queue() -> list[dict]:
    """Return the currently queued + running requests, sorted by submitted_at.

    Queued entries come from ``_buckets_v2``, running entries from slots in
    BUSY state. Post-completion tail is not retained — batch history lives
    in the ``batch_jobs`` table; interactive requests are done when the
    HTTP response returns to the client.
    """
    now = time.time()
    result: list[dict] = []

    for bucket in _buckets_v2:
        for job in bucket:
            result.append({
                "id":            job["tid"],
                "source":        job["source"],
                "request_type":  job["request_type"],
                "model":         job["model"],
                "priority":      priority_name(job["priority"]),
                "target":        None,
                "status":        "queued",
                "submitted_at":  job["submitted_at"],
                "started_at":    None,
                "completed_at":  None,
                "error":         None,
                "batch_job_id":  job["batch_job_id"],
                "waiting_sec":   round(now - job["submitted_at"], 1),
            })

    for i, s in enumerate(_slots):
        if s.state is not SlotState.BUSY or s.current_job is None:
            continue
        job = s.current_job
        started = _busy_since.get(i, now)
        result.append({
            "id":            job.get("tid", ""),
            "source":        job.get("source", ""),
            "request_type":  job.get("request_type", ""),
            "model":         job.get("model", s.model_loaded or ""),
            "priority":      priority_name(job.get("priority", Priority.BACKGROUND)),
            "target":        s.url,
            "status":        "running",
            "submitted_at":  job.get("submitted_at", started),
            "started_at":    started,
            "completed_at":  None,
            "error":         None,
            "batch_job_id":  job.get("batch_job_id"),
            "elapsed_sec":   round(now - started, 1),
        })

    result.sort(key=lambda d: d["submitted_at"])
    return result


def queue_depth() -> dict:
    """Summary counts for the status endpoint."""
    queued  = sum(len(b) for b in _buckets_v2)
    running = sum(1 for s in _slots if s.state is SlotState.BUSY)

    gpus: dict = {}
    for s in _slots:
        if s.state is SlotState.BUSY:
            gpus.setdefault(s.url, {"queued": 0, "running": 0})["running"] += 1
    # Per-GPU "queued" lane stays 0 — buckets are global, not per-GPU.
    # Field shape preserved so dashboards from before the cutover still parse.

    return {"queued": queued, "running": running, "total": queued + running, "gpus": gpus}


def pipe_depth() -> dict:
    """Depth of each priority bucket.

    Returns one key per :class:`Priority` name plus legacy ``interactive``
    / ``batch`` aliases so existing dashboards keep working.
    """
    depths: dict = {
        priority_name(p): len(_buckets_v2[p]) for p in Priority
    }
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

    # ``think`` is a top-level Ollama request parameter, not an entry in
    # ``options``. Callers (and our own defaults block above) routinely
    # tuck it into options because it sits alongside num_predict/num_ctx
    # conceptually — but Ollama silently ignores it there, so qwen3:* keeps
    # reasoning and burns the entire num_predict budget before emitting any
    # content, producing done_reason='length' with an empty response.
    # Lift it out so it actually takes effect.
    think_flag = options.pop("think", False)

    body = {
        "model":   job["model"],
        "prompt":  job["prompt"],
        "stream":  False,
        "think":   think_flag,
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
            # Bounded read timeout: defence-in-depth behind the tick's
            # busy-slot timeout sweep. If the slot has already been driven
            # through WORK_END(timeout), this still eventually unblocks the
            # coroutine instead of leaking the httpx connection forever.
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
                num_predict   = options.get("num_predict")
                # Distinguish the two empty-response failure modes so the
                # extractor logs point at the right knob. ``length`` with
                # the prompt fitting inside num_ctx means num_predict was
                # exhausted (almost always qwen3:* reasoning emission not
                # being suppressed); otherwise the prompt really did get
                # left-truncated past num_ctx.
                if (done_reason == "length"
                        and prompt_tokens is not None
                        and num_ctx is not None
                        and prompt_tokens < num_ctx):
                    hint = (f"decode hit num_predict={num_predict} before any "
                            f"content was emitted (check think=False for qwen3:*)")
                else:
                    hint = "likely prompt exceeds num_ctx and was truncated"
                raise EmptyResponseError(
                    f"Ollama returned empty response "
                    f"(done_reason={done_reason!r}, "
                    f"prompt_tokens={prompt_tokens}, num_ctx={num_ctx}); "
                    f"{hint}"
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


# ── Scheduler wiring ────────────────────────────────────────────────────────

from scheduler import (  # noqa: E402  — sibling module
    Slot, SlotState, Event, Effect,
    transition, dispatch_pass, stage_pass, project_status,
    log_transition, log_tick_summary, SchedulerStatus,
    tick_log,
)

# Canonical slot list — one Slot per configured GPU URL. Single source of
# truth for per-GPU state.
_slots: list[Slot] = []

# Five in-memory FIFO buckets. New arrivals are appended here and persisted
# via db.insert_pending. Elements are plain dicts (see _make_job).
_buckets_v2: list[list[dict]] = [[] for _ in range(5)]

# Waiters — tid → Future[url]. Resolved when a start_work effect fires
# for the owning job.
_inflight: dict[str, "asyncio.Future[str]"] = {}

# Reverse lookup for cancel_tracked — which job dict currently represents
# this tid? Cleared when the job leaves its bucket (dispatch or cancel).
_tid_to_job: dict[str, dict] = {}

# Tick loop wiring. _tick_wakeup is set whenever bucket or slot state
# changes; _tick_task owns the single background coroutine.
_tick_wakeup: Optional[asyncio.Event] = None
_tick_task:   Optional[asyncio.Task] = None


def _make_job(*, tid: str, source: str, request_type: str, model: str,
              priority: int, batch_job_id: Optional[str],
              submitted_at: float) -> dict:
    """Canonical shape of the dict stored in _buckets_v2.

    ``model`` is what the scheduler reads for dispatch decisions;
    everything else is orchestrator/observability metadata.
    """
    return {
        "tid":          tid,
        "source":       source,
        "request_type": request_type,
        "model":        model,
        "priority":     priority,
        "batch_job_id": batch_job_id,
        "submitted_at": submitted_at,
    }


def _feed_event(slot_idx: int, event: Event, **kw) -> list[Effect]:
    """Feed one event into slot[i]'s FSM.

    Applies the transition, persists the new slot_state row, logs via the
    ``[fsm]`` prefix, and returns the produced effects. Callers that drive
    I/O (``_apply_effect`` in the tick loop) consume the return value;
    purely-state-change callers (boot probe, thermal) can ignore it.
    """
    global _slots
    before = _slots[slot_idx]
    after, effects = transition(before, event, **kw)
    _slots = [*_slots[:slot_idx], after, *_slots[slot_idx + 1:]]
    try:
        db.upsert_slot_state(after.url, after.state.value, after.model_loaded)
    except Exception:
        # slot_state persistence is observability — a failed upsert must not
        # take down the FSM. Log and carry on; the next transition's upsert
        # will re-sync the durable row.
        log.exception("[slot] upsert_slot_state failed url=%s", after.url)
    log_transition(before, event, after, effects)
    return effects


async def _probe_url(url: str, timeout: float = 5.0) -> bool:
    """Single ``/api/tags`` probe. True on 2xx, False on any error.

    Used at boot to take each slot UNKNOWN→READY/UNREACHABLE, and later by
    health recovery. Bounded timeout: a slow GPU must not delay the other
    slot's READY at startup.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            r = await client.get(f"{url}/api/tags")
            r.raise_for_status()
        return True
    except Exception as exc:
        log.info("[boot] probe fail url=%s err=%s", url, exc)
        return False


async def _boot_reconcile() -> None:
    """Rehydrate in-memory state from durable SQLite state on app startup.

    Three steps:

    1. Seed ``_slots`` from configured GPU URLs. Enrich with last known
       ``model_loaded`` from ``slot_state`` so the first tick treats the
       slot as "probably holding this model" — the next probe confirms.
    2. Probe every GPU in parallel once. PROBE_OK drives UNKNOWN→READY;
       PROBE_FAIL drives UNKNOWN→UNREACHABLE.
    3. Wipe stale ``pending_work`` rows. pending_work is a live mirror of
       in-memory queue state, not a durable queue: interactive requests
       have no one to resume (their HTTP waiters are gone), and batch
       requests recover through the ``_run_batch_job_async`` re-kick in
       app lifespan — that path re-inserts fresh rows via
       ``track_request``. Leaving stale rows would double-enqueue every
       batch job whose prior row survived.

    Buckets start empty; callers (lifespan batch re-kick, proxy
    handlers) refill them via ``track_request``.
    """
    global _slots, _buckets_v2, _tid_to_job, _inflight, _busy_since

    rows_by_url = {r["gpu_url"]: r for r in db.list_slot_states()}
    _slots = []
    for url in [config.OLLAMA_0_URL, config.OLLAMA_1_URL]:
        row = rows_by_url.get(url)
        _slots.append(Slot(
            url=url, state=SlotState.UNKNOWN,
            model_loaded=(row["model_loaded"] if row else None),
        ))

    probes = await asyncio.gather(*[_probe_url(s.url) for s in _slots])
    for i, ok in enumerate(probes):
        _feed_event(i, Event.PROBE_OK if ok else Event.PROBE_FAIL)

    _buckets_v2 = [[] for _ in range(5)]
    _tid_to_job = {}
    _inflight = {}
    _busy_since = {}
    stale = db.clear_pending()

    slot_summary = [
        f"{s.url.rsplit(':', 1)[-1]}={s.state.value}/{s.model_loaded or '-'}"
        for s in _slots
    ]
    log.info("[boot] reconciled slots=%s stale_pending_cleared=%d",
             slot_summary, stale)


# ── v2: effect application ──────────────────────────────────────────────────

# Wall-clock stamp set when a slot enters BUSY so _sweep_busy_timeouts
# can drive recovery after SLOT_MAX_WALL_SECONDS. Keyed by slot index
# (matches _slots[i]). Cleared on every transition out of BUSY.
_busy_since: dict[int, float] = {}


async def _do_load(slot_idx: int, url: str, model: str,
                   evict: Optional[str]) -> None:
    """Warm ``model`` on ``url``; feed LOAD_DONE on success, LOAD_FAIL else.

    A ``load`` effect from the scheduler is the orchestrator's cue to
    post a warmup request. Eviction of any prior model happens
    implicitly — Ollama swaps on the first request carrying a different
    ``model`` so the explicit ``evict`` field is observability only.
    """
    body: dict = {"model": model, "prompt": "", "keep_alive": "10m"}
    if model.startswith("qwen3:"):
        # feedback_reasoning_model_caps: without num_ctx, qwen3 auto-fits
        # to training-max context on a cold GPU and reload takes ~12 s.
        body["options"] = {"num_ctx": 8192}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            r = await client.post(f"{url}/api/generate", json=body)
            r.raise_for_status()
        log.info("[load] ok url=%s model=%r evict=%r", url, model, evict)
        _feed_event(slot_idx, Event.LOAD_DONE)
    except Exception as exc:
        log.warning("[load] fail url=%s model=%r err=%s", url, model, exc)
        _feed_event(slot_idx, Event.LOAD_FAIL)
    _wake_tick(reason="load_settled")


def _slot_idx_for_url(url: str) -> Optional[int]:
    for i, s in enumerate(_slots):
        if s.url == url:
            return i
    return None


def _apply_effect(effect: Effect) -> None:
    """Apply one scheduler effect to the outside world.

    The four effect kinds that ``scheduler.transition`` emits:

    - ``load`` — kick an async ``_do_load`` task. Fire-and-forget; the
      task eventually feeds LOAD_DONE or LOAD_FAIL back into the FSM.
    - ``start_work`` — resolve the owning tid's future so
      ``wait_for_dispatch`` returns the target URL; stamp ``busy_since``
      so the tick loop can enforce the SLOT_MAX_WALL_SECONDS timeout.
      The pending_work row is NOT deleted here — durability demands it
      survive until work_end, so a crash mid-inference re-dispatches.
    - ``work_end`` — delete the pending_work row (durable queue drain)
      and clear ``busy_since``. Batch rows live in batch_jobs, untouched.
    - ``gpu_unreachable`` — observability only; the FSM has already
      cleared the slot so there is nothing further to change.
    """
    kind = effect.kind
    data = effect.data

    if kind == "load":
        url = data["url"]
        idx = _slot_idx_for_url(url)
        if idx is None:
            log.error("[effect] load for unknown url=%s", url)
            return
        asyncio.create_task(_do_load(idx, url, data["model"], data.get("evict")))
        return

    if kind == "start_work":
        url = data["url"]
        job = data["job"]
        tid = job.get("tid")
        idx = _slot_idx_for_url(url)
        if idx is not None:
            _busy_since[idx] = time.time()
        if tid:
            fut = _inflight.pop(tid, None)
            _tid_to_job.pop(tid, None)
            if fut is not None and not fut.done():
                fut.set_result(url)
        return

    if kind == "work_end":
        url = data.get("url")
        job = data.get("job") or {}
        tid = job.get("tid")
        idx = _slot_idx_for_url(url) if url else None
        if idx is not None:
            _busy_since.pop(idx, None)
        if tid:
            try:
                db.delete_pending(tid)
            except Exception:
                log.exception("[effect] delete_pending failed tid=%s", tid)
        return

    if kind == "gpu_unreachable":
        log.warning("[effect] gpu_unreachable url=%s", data.get("url"))
        return

    log.warning("[effect] unknown kind=%s data=%s", kind, data)


# ── v2: tick loop ───────────────────────────────────────────────────────────

def _wake_tick(reason: str = "unspecified") -> None:
    """Signal the tick loop to re-evaluate. No-op before _ensure_tick."""
    if _tick_wakeup is not None and not _tick_wakeup.is_set():
        tick_log.info("[tick] wake reason=%s", reason)
        _tick_wakeup.set()


def _ensure_tick() -> None:
    """Lazily start the tick loop on the running event loop, if any.

    Called from ``track_request`` so both the app lifespan and tests pick
    up the loop without extra wiring. Safe to call repeatedly.
    """
    global _tick_wakeup, _tick_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _tick_wakeup is None:
        _tick_wakeup = asyncio.Event()
    if _tick_task is None or _tick_task.done():
        _tick_task = loop.create_task(_tick_loop())


def _sweep_busy_timeouts() -> None:
    """Feed WORK_END(outcome='timeout') on any BUSY slot past its deadline.

    This collapses what used to be ``_watchdog_loop`` into the tick.
    Recovery goes through LOADING with a forced eviction — the scheduler
    models timeout as "fail to known state", so the recovery path is
    identical to a normal reload.
    """
    now = time.time()
    for i, s in enumerate(_slots):
        if s.state is not SlotState.BUSY:
            continue
        since = _busy_since.get(i)
        if since is None:
            continue
        if (now - since) >= config.SLOT_MAX_WALL_SECONDS:
            tick_log.warning(
                "[tick] busy timeout url=%s ran=%.0fs — driving work_end(timeout)",
                s.url, now - since,
            )
            for e in _feed_event(i, Event.WORK_END, outcome="timeout"):
                _apply_effect(e)


def _drive_pass(label: str, fn) -> int:
    """Run one pure scheduler pass, apply effects, update ``_slots``.

    ``fn`` is ``dispatch_pass`` or ``stage_pass``. Returns the effect
    count so the tick can decide whether to emit a summary line.
    """
    global _slots
    effects, new_slots = fn(_buckets_v2, _slots)
    _slots = new_slots
    for e in effects:
        _apply_effect(e)
    if effects:
        tick_log.info("[tick] %s produced %d effects", label, len(effects))
    return len(effects)


async def _tick_once() -> dict:
    """Run one iteration of the tick body. Test-friendly entry point.

    The main loop wraps this in an ``await _tick_wakeup.wait()`` /
    periodic-timeout harness. Callers in tests drive it directly to
    avoid racing a background task.
    """
    _sweep_busy_timeouts()
    if _paused:
        return {"dispatched": 0, "staged": 0, "paused": True}
    dispatched = _drive_pass("dispatch", dispatch_pass)
    staged     = _drive_pass("stage",    stage_pass)
    if dispatched or staged:
        status = project_status(
            paused=_paused, recovered=True,
            slots=_slots, buckets_nonempty=any(_buckets_v2),
        )
        log_tick_summary(
            status=status,
            dispatched=dispatched, staged=staged,
            bucket_depths=[len(b) for b in _buckets_v2],
            slot_summary=[(s.url, s.state.value, s.model_loaded)
                          for s in _slots],
        )
    return {"dispatched": dispatched, "staged": staged, "paused": False}


async def _tick_loop() -> None:
    """Single background coroutine driving the scheduler forward.

    Wakes on any of: enqueue, release, fail, cancel, probe result,
    thermal, resume, load settle — plus a 250 ms periodic tick so BUSY
    timeouts fire even during an otherwise silent period.

    While paused the body skips dispatch and stage; in-flight work still
    completes (its ``work_end`` effect drains pending_work) so pausing is
    "no new starts", never "no drains".
    """
    log.info("[tick] loop started")
    while True:
        try:
            await asyncio.wait_for(_tick_wakeup.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass
        _tick_wakeup.clear()
        try:
            await _tick_once()
        except Exception:
            log.exception("[tick] loop iteration failed")
