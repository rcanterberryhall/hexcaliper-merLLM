"""
gpu_router.py — GPU state, health tracking, and reclaim loop.

Both Ollama instances run continuously.  This module owns per-GPU metadata
(loaded model, health state, last-active timestamp) and the background
reclaim loop that returns idle non-default GPUs to the default model.

GPU *dispatch* (choosing which GPU serves a request) lives in
``queue_manager``'s late-binding dispatcher.  This module no longer picks
targets for incoming requests — it only exposes state that the dispatcher
reads and tools to reload models when the dispatcher decides to swap.

GPU health states:
  healthy   — accepting requests, eligible for dispatch
  degraded  — failed recently, being probed with exponential backoff
  faulted   — probes exhausted, requires manual reset
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Optional

import httpx

import config
import queue_manager

log = logging.getLogger(__name__)


class GpuHealth(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    FAULTED  = "faulted"


class GpuState:
    __slots__ = ("url", "model", "health", "last_active", "fail_since",
                 "next_probe", "probe_interval",
                 "thermal_paused", "last_temp_c", "thermal_paused_since")

    def __init__(self, url: str):
        self.url: str = url
        self.model: str = config.DEFAULT_MODEL
        self.health: GpuHealth = GpuHealth.HEALTHY
        self.last_active: float = time.time()
        self.fail_since: Optional[float] = None
        self.next_probe: float = 0.0
        self.probe_interval: float = config.HEALTH_BACKOFF_BASE
        # Thermal state is orthogonal to health: a GPU can be healthy but
        # thermally paused. In-flight work is not interrupted; only new
        # dispatch is blocked.
        self.thermal_paused: bool = False
        self.last_temp_c: Optional[float] = None
        self.thermal_paused_since: Optional[float] = None


# ── Module state ─────────────────────────────────────────────────────────────

_gpus: dict[str, GpuState] = {}
_pending_default_model: Optional[str] = None  # set when DEFAULT_MODEL change is pending


def _init_gpus() -> None:
    """Initialise GPU state if not already done."""
    if not _gpus:
        for url in [config.OLLAMA_0_URL, config.OLLAMA_1_URL]:
            _gpus[url] = GpuState(url)
        log.info("GPU states initialised: %s (default_model=%s)",
                 list(_gpus.keys()), config.DEFAULT_MODEL)


def _healthy_gpus() -> list[GpuState]:
    _init_gpus()
    return [g for g in _gpus.values() if g.health == GpuHealth.HEALTHY]


# ── Activity tracking ────────────────────────────────────────────────────────


def record_activity(target: str) -> None:
    """Mark a GPU as having just served a request."""
    _init_gpus()
    if target in _gpus:
        _gpus[target].last_active = time.time()


# ── Thermal management ───────────────────────────────────────────────────────


def update_thermal_state(target: str, temp_c: float) -> None:
    """
    Update a GPU's thermal state from a fresh temperature reading.

    Called from the metrics collection loop. Applies hysteresis:
      * temp >= GPU_TEMP_PAUSE_C  → mark thermally paused
      * temp <= GPU_TEMP_RESUME_C → clear thermal pause
      * in between → no transition (preserves current state)

    Wakes the dispatcher on resume so any queued work starts moving again.
    """
    _init_gpus()
    gpu = _gpus.get(target)
    if not gpu:
        return

    gpu.last_temp_c = temp_c
    was_paused = gpu.thermal_paused

    if temp_c >= config.GPU_TEMP_PAUSE_C and not was_paused:
        gpu.thermal_paused = True
        gpu.thermal_paused_since = time.time()
        log.warning(
            "GPU %s thermally paused at %.1f°C (threshold %d°C) — "
            "blocking new dispatch until it cools below %d°C",
            target, temp_c, config.GPU_TEMP_PAUSE_C, config.GPU_TEMP_RESUME_C,
        )
    elif temp_c <= config.GPU_TEMP_RESUME_C and was_paused:
        gpu.thermal_paused = False
        paused_for = (time.time() - gpu.thermal_paused_since) if gpu.thermal_paused_since else 0
        gpu.thermal_paused_since = None
        log.info(
            "GPU %s thermal pause cleared at %.1f°C after %.0fs — resuming dispatch",
            target, temp_c, paused_for,
        )
        # Wake the dispatcher so queued work can start flowing again.
        queue_manager._wake_dispatcher(reason="thermal_resume")


def is_dispatchable(target: str) -> bool:
    """
    True if the GPU is eligible to receive new dispatched work.

    A GPU is dispatchable only if it is both healthy and not thermally
    paused. In-flight work on a thermally paused GPU is not affected —
    only new dispatches are blocked.
    """
    _init_gpus()
    gpu = _gpus.get(target)
    if not gpu:
        return False
    return gpu.health == GpuHealth.HEALTHY and not gpu.thermal_paused


# ── Health management ────────────────────────────────────────────────────────


def mark_failed(target: str) -> None:
    """Transition a GPU to degraded state and start health probes."""
    _init_gpus()
    gpu = _gpus.get(target)
    if not gpu or gpu.health != GpuHealth.HEALTHY:
        return
    gpu.health = GpuHealth.DEGRADED
    gpu.fail_since = time.time()
    gpu.probe_interval = config.HEALTH_BACKOFF_BASE
    gpu.next_probe = time.time() + gpu.probe_interval
    log.warning("GPU %s marked degraded — starting health probes", target)


def reset_gpu(target: str) -> bool:
    """Manually reset a faulted GPU to healthy. Returns False if not found."""
    _init_gpus()
    gpu = _gpus.get(target)
    if not gpu:
        return False
    gpu.health = GpuHealth.HEALTHY
    gpu.fail_since = None
    gpu.next_probe = 0.0
    gpu.probe_interval = config.HEALTH_BACKOFF_BASE
    gpu.model = config.DEFAULT_MODEL
    log.info("GPU %s manually reset to healthy", target)
    return True


def set_pending_default_model(new_model: str) -> None:
    """Schedule a DEFAULT_MODEL change — GPUs reload when idle."""
    global _pending_default_model
    _pending_default_model = new_model
    log.info("DEFAULT_MODEL change pending → %r (will reload GPUs when idle)", new_model)


# ── Background loop ──────────────────────────────────────────────────────────


async def reclaim_loop() -> None:
    """
    Background task (runs every 30s):
    - Healthy GPUs with non-default model idle > RECLAIM_TIMEOUT → reload default.
    - Pending DEFAULT_MODEL change → reload idle GPUs.
    - Degraded GPUs → probe with exponential backoff.

    Reclaim model reloads cooperate with the dispatcher by calling
    ``queue_manager.reserve_gpu`` before the swap and ``unreserve_gpu``
    afterwards, so no request is dispatched onto a GPU mid-reload.
    """
    global _pending_default_model
    _init_gpus()

    while True:
        await asyncio.sleep(30)
        now = time.time()

        for gpu in _gpus.values():
            # ── Healthy GPU: reclaim or pending model change ─────────
            if gpu.health == GpuHealth.HEALTHY:
                # Pending DEFAULT_MODEL change — reload when idle.
                if _pending_default_model and queue_manager.reserve_gpu(gpu.url):
                    try:
                        await _reload_model(gpu, _pending_default_model)
                    finally:
                        queue_manager.unreserve_gpu(gpu.url)

                # Reclaim non-default model after idle timeout.
                elif (gpu.model != config.DEFAULT_MODEL
                      and (now - gpu.last_active) >= config.RECLAIM_TIMEOUT
                      and queue_manager.reserve_gpu(gpu.url)):
                    try:
                        log.info("GPU %s idle %.0fs with model %r — reclaiming to %r",
                                 gpu.url, now - gpu.last_active, gpu.model,
                                 config.DEFAULT_MODEL)
                        await _reload_model(gpu, config.DEFAULT_MODEL)
                    finally:
                        queue_manager.unreserve_gpu(gpu.url)

            # ── Degraded GPU: probe with backoff ─────────────────────
            elif gpu.health == GpuHealth.DEGRADED:
                if now >= gpu.next_probe:
                    await _probe_gpu(gpu, now)

        # If all healthy GPUs now have the pending model, commit the change.
        if _pending_default_model:
            healthy = _healthy_gpus()
            if healthy and all(g.model == _pending_default_model for g in healthy):
                config.DEFAULT_MODEL = _pending_default_model
                _pending_default_model = None
                log.info("DEFAULT_MODEL change complete")


async def _reload_model(gpu: GpuState, model: str) -> bool:
    """Send a warmup request to load a model on a GPU.

    Ollama's ``/api/generate`` endpoint rejects embedding-only models with
    HTTP 400 (``"<model>" does not support generate``). Warming those via
    ``/api/generate`` would silently no-op — Ollama returns an error body
    but does not load the model. The dispatcher would then hand off a
    request to a GPU whose real load happens on the first ``/api/embeddings``
    call, and any concurrent embedding traffic during that load races and
    gets 500s. So we branch on the model name and use ``/api/embed`` for
    anything with "embed" in it (nomic-embed-text, mxbai-embed-large, etc.).

    ``resp.raise_for_status()`` makes a non-2xx warmup a hard failure:
    ``gpu.model`` is only updated once the warmup actually succeeded, so
    the reclaim loop will retry on its next tick instead of believing a
    swap landed when it didn't. Returns True on success, False on failure.

    For large reasoning models we pin ``num_ctx`` so Ollama does not
    auto-fit the context window to the model's training maximum on a
    freshly empty GPU. On a 24 GB Tesla P40 the qwen3:32b model at the
    default 32 k training context forces CPU offload of the output
    layer and a split KV cache (~7 GB GPU + ~900 MB CPU), which makes
    every reload take ~12 s instead of ~330 ms and slows every
    inference. Pinning to 8192 here matches the batch-path defaults in
    ``queue_manager._run_batch_job_async`` and the
    ``feedback_reasoning_model_caps`` rule.
    """
    if "embed" in model:
        path = "/api/embed"
        body: dict = {"model": model, "input": "x", "keep_alive": "10m"}
    else:
        path = "/api/generate"
        body = {"model": model, "prompt": "", "keep_alive": "10m"}
        if model.startswith("qwen3:"):
            body["options"] = {"num_ctx": 8192}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(f"{gpu.url}{path}", json=body)
            resp.raise_for_status()
        gpu.model = model
        log.info("GPU %s reloaded with %r", gpu.url, model)
        return True
    except Exception as exc:
        log.warning("failed to reload %r on GPU %s: %s", model, gpu.url, exc)
        return False


async def _probe_gpu(gpu: GpuState, now: float) -> None:
    """Health-probe a degraded GPU."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(f"{gpu.url}/api/tags")
            r.raise_for_status()

        # Recovered — reload default model and mark healthy.
        gpu.health = GpuHealth.HEALTHY
        gpu.fail_since = None
        gpu.probe_interval = config.HEALTH_BACKOFF_BASE
        log.info("GPU %s recovered — marking healthy", gpu.url)
        await _reload_model(gpu, config.DEFAULT_MODEL)

    except Exception as exc:
        elapsed = now - (gpu.fail_since or now)
        if elapsed >= config.HEALTH_FAULT_TIMEOUT:
            gpu.health = GpuHealth.FAULTED
            log.error("GPU %s faulted after %.0fs of failures — manual reset required",
                      gpu.url, elapsed)
        else:
            gpu.probe_interval = min(gpu.probe_interval * 2,
                                     config.HEALTH_BACKOFF_CAP)
            gpu.next_probe = now + gpu.probe_interval
            log.info("GPU %s probe failed (%.0fs degraded, next in %.0fs): %s",
                     gpu.url, elapsed, gpu.probe_interval, exc)


# ── Status ───────────────────────────────────────────────────────────────────


def status() -> dict:
    """Full GPU state for the UI and status endpoint."""
    _init_gpus()
    now = time.time()
    gpus = {}
    for label, url in [("gpu0", config.OLLAMA_0_URL), ("gpu1", config.OLLAMA_1_URL)]:
        gpu = _gpus.get(url)
        if gpu:
            gpus[label] = {
                "url":            gpu.url,
                "model":          gpu.model,
                "health":         gpu.health.value,
                "idle_seconds":   round(now - gpu.last_active, 1),
                "fail_since":     gpu.fail_since,
                "in_flight":      queue_manager.gpu_slot_busy(gpu.url),
                "thermal_paused": gpu.thermal_paused,
                "last_temp_c":    gpu.last_temp_c,
                "thermal_paused_since": gpu.thermal_paused_since,
            }
    return {
        "routing":        "round_robin",
        "default_model":  config.DEFAULT_MODEL,
        "reclaim_timeout": config.RECLAIM_TIMEOUT,
        "pending_default_model": _pending_default_model,
        "gpu_temp_pause_c":  config.GPU_TEMP_PAUSE_C,
        "gpu_temp_resume_c": config.GPU_TEMP_RESUME_C,
        "gpus":           gpus,
    }
