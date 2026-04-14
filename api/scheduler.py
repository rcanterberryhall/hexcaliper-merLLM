"""
scheduler.py — Pure core for the merLLM FSM refactor (merLLM#52 / #53).

Two components, both pure:

1. Slot FSM — per-GPU state machine. ``transition(slot, event, **kw)`` returns
   a new :class:`Slot` and a list of :class:`Effect`; no I/O, no side channels.
2. Scheduler algorithm — ``dispatch_pass`` and ``stage_pass`` over
   ``(buckets, slots)`` return lists of effects. The scheduler itself carries
   no state; the tick loop is the only impure boundary (wired in Phase 3).

This module is not wired into the runtime. It exists alongside the existing
``queue_manager.py`` until the Phase 3 cutover (merLLM#55).

Design reference: the consolidated comment on merLLM#52.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional


# After this many consecutive LOAD_FAIL events on the same slot, give up
# and transition to UNREACHABLE. Empirical: three attempts is enough to
# ride out a transient ollama hiccup without pinning a dead GPU forever.
MAX_LOAD_ATTEMPTS = 3


# Dedicated loggers with short, greppable prefixes. Callers can silence
# either independently, e.g.
#     logging.getLogger("merllm.scheduler.fsm").setLevel(logging.WARNING)
# The transition function itself stays pure; only ``log_transition`` and
# ``log_tick_summary`` emit records. Phase 3's tick loop calls these after
# applying each effect list.
fsm_log  = logging.getLogger("merllm.scheduler.fsm")
tick_log = logging.getLogger("merllm.scheduler.tick")
fsm_log.setLevel(logging.INFO)
tick_log.setLevel(logging.INFO)


class SlotState(Enum):
    UNKNOWN     = "unknown"
    READY       = "ready"
    LOADING     = "loading"
    BUSY        = "busy"
    COOLING     = "cooling"
    UNREACHABLE = "unreachable"
    DRAINING    = "draining"


class Event(Enum):
    PROBE_OK      = "probe_ok"
    PROBE_FAIL    = "probe_fail"
    LOAD_BEGIN    = "load_begin"      # kwargs: model
    LOAD_DONE     = "load_done"
    LOAD_FAIL     = "load_fail"
    WORK_BEGIN    = "work_begin"      # kwargs: job (dict with "model")
    WORK_END      = "work_end"        # kwargs: outcome in {"ok","fail","timeout"}
    THERMAL_TRIP  = "thermal_trip"
    THERMAL_CLEAR = "thermal_clear"
    DRAIN         = "drain"
    UNDRAIN       = "undrain"


class InvalidTransition(Exception):
    """Raised when the FSM receives an event that has no defined transition
    from the current state. Always a bug — never catch silently."""


@dataclass
class Effect:
    kind: str
    data: dict = field(default_factory=dict)


@dataclass
class Slot:
    url: str
    state: SlotState = SlotState.UNKNOWN
    model_loaded: Optional[str] = None
    loading_model: Optional[str] = None
    current_job: Optional[dict] = None
    load_attempts: int = 0
    # Latched flags set while BUSY when we can't transition immediately.
    # Drained on the next WORK_END, at which point the slot transitions
    # to COOLING or DRAINING instead of READY.
    thermal_pending: bool = False
    drain_pending: bool = False


# ── Slot transition ─────────────────────────────────────────────────────────

def transition(slot: Slot, event: Event, **kw: Any) -> tuple[Slot, list[Effect]]:
    """Pure ``(slot, event) → (new_slot, effects)``.

    Raises :class:`InvalidTransition` for undefined (state, event) pairs.
    """
    s = slot.state

    # PROBE_FAIL from any idle-ish state → UNREACHABLE. BUSY ignores PROBE_FAIL
    # (in-flight work owns the verdict; BUSY timeout is the real recovery).
    if event is Event.PROBE_FAIL:
        if s is SlotState.BUSY:
            return slot, []
        if s in (SlotState.UNREACHABLE, SlotState.DRAINING):
            return slot, []
        return _enter_unreachable(slot)

    # DRAIN: from BUSY, latch and finish in-flight work first.
    if event is Event.DRAIN:
        if s is SlotState.DRAINING:
            return slot, []
        if s is SlotState.BUSY:
            return replace(slot, drain_pending=True), []
        return replace(slot, state=SlotState.DRAINING,
                       loading_model=None, load_attempts=0), []

    if s is SlotState.UNKNOWN:
        if event is Event.PROBE_OK:
            return replace(slot, state=SlotState.READY), []

    elif s is SlotState.READY:
        if event is Event.LOAD_BEGIN:
            model = kw["model"]
            return (
                replace(slot, state=SlotState.LOADING,
                        loading_model=model, load_attempts=0),
                [Effect("load", {"url": slot.url, "model": model,
                                 "evict": slot.model_loaded})],
            )
        if event is Event.WORK_BEGIN:
            job = kw["job"]
            if job.get("model") != slot.model_loaded:
                raise InvalidTransition(
                    f"work_begin model={job.get('model')!r} but slot holds "
                    f"{slot.model_loaded!r}"
                )
            return (
                replace(slot, state=SlotState.BUSY, current_job=job),
                [Effect("start_work", {"url": slot.url, "job": job})],
            )
        if event is Event.THERMAL_TRIP:
            return replace(slot, state=SlotState.COOLING), []
        if event is Event.PROBE_OK:
            return slot, []

    elif s is SlotState.LOADING:
        if event is Event.LOAD_DONE:
            return (
                replace(slot, state=SlotState.READY,
                        model_loaded=slot.loading_model,
                        loading_model=None, load_attempts=0),
                [],
            )
        if event is Event.LOAD_FAIL:
            attempts = slot.load_attempts + 1
            if attempts < MAX_LOAD_ATTEMPTS:
                return (
                    replace(slot, load_attempts=attempts),
                    [Effect("load", {"url": slot.url,
                                     "model": slot.loading_model,
                                     "evict": slot.model_loaded})],
                )
            return _enter_unreachable(
                replace(slot, loading_model=None, load_attempts=0)
            )
        if event is Event.THERMAL_TRIP:
            return (
                replace(slot, state=SlotState.COOLING,
                        loading_model=None, load_attempts=0),
                [],
            )

    elif s is SlotState.BUSY:
        if event is Event.WORK_END:
            outcome = kw.get("outcome", "ok")
            # Fail-to-known-state: timeout drives recovery through LOADING
            # with a forced eviction of whatever model is resident.
            if outcome == "timeout":
                model = slot.model_loaded
                next_slot = replace(slot, state=SlotState.LOADING,
                                    current_job=None, loading_model=model,
                                    load_attempts=0,
                                    thermal_pending=False, drain_pending=False)
                return next_slot, [
                    Effect("work_end", {"url": slot.url,
                                        "job": slot.current_job,
                                        "outcome": "timeout"}),
                    Effect("load", {"url": slot.url, "model": model,
                                    "evict": model}),
                ]
            next_state = SlotState.READY
            if slot.thermal_pending:
                next_state = SlotState.COOLING
            elif slot.drain_pending:
                next_state = SlotState.DRAINING
            return (
                replace(slot, state=next_state, current_job=None,
                        thermal_pending=False, drain_pending=False),
                [Effect("work_end", {"url": slot.url,
                                     "job": slot.current_job,
                                     "outcome": outcome})],
            )
        if event is Event.THERMAL_TRIP:
            return replace(slot, thermal_pending=True), []

    elif s is SlotState.COOLING:
        if event is Event.THERMAL_CLEAR:
            return replace(slot, state=SlotState.READY), []

    elif s is SlotState.UNREACHABLE:
        if event is Event.PROBE_OK:
            # On recovery, the loaded-model state of ollama is unknown.
            # Caller (scheduler/tick) will drive a LOAD once demand appears.
            return (
                replace(slot, state=SlotState.READY,
                        model_loaded=None, loading_model=None, load_attempts=0),
                [],
            )

    elif s is SlotState.DRAINING:
        if event is Event.UNDRAIN:
            return replace(slot, state=SlotState.READY), []

    raise InvalidTransition(
        f"event {event.value!r} not allowed in state {s.value!r}"
    )


def _enter_unreachable(slot: Slot) -> tuple[Slot, list[Effect]]:
    return (
        replace(slot, state=SlotState.UNREACHABLE,
                model_loaded=None, loading_model=None, load_attempts=0,
                current_job=None),
        [Effect("gpu_unreachable", {"url": slot.url})],
    )


# ── Scheduler algorithm ─────────────────────────────────────────────────────

# Priority ordering for the five-bucket walk. Integer values match the
# queue_manager.Priority IntEnum (CHAT=0 ... BACKGROUND=4). Duplicated here
# so the pure module does not depend on queue_manager.
PRIORITY_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)


def _ready_slot_holding(slots: list[Slot], model: str) -> Optional[int]:
    for i, s in enumerate(slots):
        if s.state is SlotState.READY and s.model_loaded == model:
            return i
    return None


def _idle_ready_slot(slots: list[Slot], exclude: set[int]) -> Optional[int]:
    for i, s in enumerate(slots):
        if i in exclude:
            continue
        if s.state is SlotState.READY:
            return i
    return None


def dispatch_pass(
    buckets: list[list[dict]], slots: list[Slot]
) -> tuple[list[Effect], list[Slot]]:
    """Walk buckets strict-priority, dispatch heads whose model is resident.

    Mutates ``buckets`` in place (pops). Returns ``(effects, updated_slots)``.
    Restarts from the top bucket after each pop so a higher-priority job
    arriving mid-dispatch cuts the line.
    """
    effects: list[Effect] = []
    progress = True
    while progress:
        progress = False
        for p in PRIORITY_ORDER:
            if p >= len(buckets) or not buckets[p]:
                continue
            head = buckets[p][0]
            slot_idx = _ready_slot_holding(slots, head["model"])
            if slot_idx is None:
                continue
            job = buckets[p].pop(0)
            new_slot, eff = transition(slots[slot_idx], Event.WORK_BEGIN, job=job)
            slots = [*slots[:slot_idx], new_slot, *slots[slot_idx + 1:]]
            effects.extend(eff)
            progress = True
            break
    return effects, slots


def stage_pass(
    buckets: list[list[dict]], slots: list[Slot]
) -> tuple[list[Effect], list[Slot]]:
    """Speculatively LOAD models on idle slots to match near-term demand.

    Demand is the unique bucket-head models in priority order, truncated to
    the number of slots. With one distinct demand and spare slots, idle
    slots mirror the demanded model (keeps swaps rare; most work uses the
    same model).
    """
    effects: list[Effect] = []

    demand: list[str] = []
    for p in PRIORITY_ORDER:
        if p >= len(buckets) or not buckets[p]:
            continue
        m = buckets[p][0]["model"]
        if m not in demand:
            demand.append(m)
    if not demand:
        return effects, slots

    # Mirror: spare slots inherit the top demand.
    while len(demand) < len(slots):
        demand.append(demand[0])

    used_slots: set[int] = set()
    satisfied_demands: set[int] = set()

    # First pass: any slot already holding / loading / running a demanded
    # model is treated as satisfying that demand — no speculative load.
    for di, model in enumerate(demand):
        for si, s in enumerate(slots):
            if si in used_slots:
                continue
            holds_it = (
                s.state in (SlotState.READY, SlotState.BUSY)
                and s.model_loaded == model
            ) or (
                s.state is SlotState.LOADING and s.loading_model == model
            )
            if holds_it:
                used_slots.add(si)
                satisfied_demands.add(di)
                break

    # Second pass: unmet demands seek an idle READY slot to load onto.
    for di, model in enumerate(demand):
        if di in satisfied_demands:
            continue
        si = _idle_ready_slot(slots, used_slots)
        if si is None:
            continue
        new_slot, eff = transition(slots[si], Event.LOAD_BEGIN, model=model)
        slots = [*slots[:si], new_slot, *slots[si + 1:]]
        effects.extend(eff)
        used_slots.add(si)
    return effects, slots


# ── Observable status projection (for UI / metrics only) ────────────────────

class SchedulerStatus(Enum):
    RECOVERING  = "recovering"
    PAUSED      = "paused"
    DEGRADED    = "degraded"
    DRAINING    = "draining"
    DISPATCHING = "dispatching"
    IDLE        = "idle"


def project_status(
    *, paused: bool, recovered: bool,
    slots: list[Slot], buckets_nonempty: bool,
) -> SchedulerStatus:
    """Pure observable-status projection. First match wins, top-down.

    Not consulted by the algorithm — UI/metrics only.
    """
    if not recovered:
        return SchedulerStatus.RECOVERING
    if paused:
        return SchedulerStatus.PAUSED
    if slots and all(s.state is SlotState.UNREACHABLE for s in slots):
        return SchedulerStatus.DEGRADED
    if any(s.state in (SlotState.BUSY, SlotState.LOADING) for s in slots):
        return SchedulerStatus.DRAINING
    if buckets_nonempty and any(s.state is SlotState.READY for s in slots):
        return SchedulerStatus.DISPATCHING
    return SchedulerStatus.IDLE


# ── Logging helpers (impure — call AFTER applying effects) ──────────────────

def log_transition(
    before: Slot, event: Event, after: Slot, effects: list[Effect]
) -> None:
    """Emit a single INFO line describing one slot transition.

    Greppable prefix ``[fsm]``. No-op if the state did not change and no
    effects were produced (pure kwargs-only updates like latched flags
    still log because they change the slot object).
    """
    if before == after and not effects:
        return
    kinds = ",".join(e.kind for e in effects) or "-"
    fsm_log.info(
        "[fsm] %s %s→%s event=%s model=%s→%s effects=%s",
        before.url,
        before.state.value, after.state.value,
        event.value,
        before.model_loaded, after.model_loaded,
        kinds,
    )


def log_tick_summary(
    *, status: SchedulerStatus,
    dispatched: int, staged: int,
    bucket_depths: list[int],
    slot_summary: list[tuple[str, str, Optional[str]]],
) -> None:
    """Emit one INFO line per tick that produced work.

    Greppable prefix ``[tick]``. Skip emission when the tick was a no-op
    so logs stay readable at rest; the tick loop should call this only
    when ``dispatched + staged > 0`` or status changed.
    """
    slots_fmt = " ".join(
        f"{url.rsplit(':', 1)[-1]}={state}/{model or '-'}"
        for url, state, model in slot_summary
    )
    tick_log.info(
        "[tick] status=%s dispatched=%d staged=%d buckets=%s slots=%s",
        status.value, dispatched, staged, bucket_depths, slots_fmt,
    )
