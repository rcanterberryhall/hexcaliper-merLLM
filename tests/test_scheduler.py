"""Tests for api/scheduler.py — pure Slot FSM, scheduler algorithm, and
status projection. Phase 1 of the merLLM#52 FSM refactor (merLLM#53)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from scheduler import (  # noqa: E402
    Effect,
    Event,
    InvalidTransition,
    MAX_LOAD_ATTEMPTS,
    SchedulerStatus,
    Slot,
    SlotState,
    dispatch_pass,
    project_status,
    stage_pass,
    transition,
)


GPU = "http://gpu0:11434"
GPU1 = "http://gpu1:11435"


def job(model: str, jid: str = "j1", priority: int = 0) -> dict:
    return {"id": jid, "model": model, "priority": priority}


# ─────────────────────────────────────────────────────────────────────────────
# Slot FSM — happy-path transitions
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_probe_ok_goes_ready():
    s = Slot(url=GPU)
    s2, eff = transition(s, Event.PROBE_OK)
    assert s2.state is SlotState.READY
    assert eff == []


def test_ready_load_begin_goes_loading_with_effect():
    s = Slot(url=GPU, state=SlotState.READY)
    s2, eff = transition(s, Event.LOAD_BEGIN, model="qwen3:32b")
    assert s2.state is SlotState.LOADING
    assert s2.loading_model == "qwen3:32b"
    assert s2.load_attempts == 0
    assert eff == [Effect("load", {"url": GPU, "model": "qwen3:32b", "evict": None})]


def test_ready_load_begin_records_eviction_of_current_model():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="qwen3:30b")
    _, eff = transition(s, Event.LOAD_BEGIN, model="qwen3:32b")
    assert eff[0].data["evict"] == "qwen3:30b"


def test_ready_work_begin_goes_busy_when_model_matches():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="qwen3:32b")
    j = job("qwen3:32b")
    s2, eff = transition(s, Event.WORK_BEGIN, job=j)
    assert s2.state is SlotState.BUSY
    assert s2.current_job == j
    assert eff == [Effect("start_work", {"url": GPU, "job": j})]


def test_ready_work_begin_wrong_model_raises():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="qwen3:32b")
    with pytest.raises(InvalidTransition):
        transition(s, Event.WORK_BEGIN, job=job("qwen3:30b"))


def test_ready_thermal_trip_goes_cooling():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="m")
    s2, _ = transition(s, Event.THERMAL_TRIP)
    assert s2.state is SlotState.COOLING
    assert s2.model_loaded == "m"   # preserved across COOLING


def test_ready_probe_ok_is_noop():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="m")
    s2, eff = transition(s, Event.PROBE_OK)
    assert s2 == s
    assert eff == []


def test_loading_load_done_goes_ready_with_model_set():
    s = Slot(url=GPU, state=SlotState.LOADING, loading_model="qwen3:32b",
             model_loaded="qwen3:30b", load_attempts=1)
    s2, eff = transition(s, Event.LOAD_DONE)
    assert s2.state is SlotState.READY
    assert s2.model_loaded == "qwen3:32b"
    assert s2.loading_model is None
    assert s2.load_attempts == 0
    assert eff == []


def test_loading_load_fail_retries_below_limit():
    s = Slot(url=GPU, state=SlotState.LOADING,
             loading_model="qwen3:32b", load_attempts=0)
    s2, eff = transition(s, Event.LOAD_FAIL)
    assert s2.state is SlotState.LOADING
    assert s2.load_attempts == 1
    assert len(eff) == 1 and eff[0].kind == "load"


def test_loading_load_fail_gives_up_at_max_attempts():
    s = Slot(url=GPU, state=SlotState.LOADING,
             loading_model="qwen3:32b", load_attempts=MAX_LOAD_ATTEMPTS - 1)
    s2, eff = transition(s, Event.LOAD_FAIL)
    assert s2.state is SlotState.UNREACHABLE
    assert s2.loading_model is None
    assert s2.load_attempts == 0
    assert eff and eff[0].kind == "gpu_unreachable"


def test_loading_thermal_trip_aborts_load_into_cooling():
    s = Slot(url=GPU, state=SlotState.LOADING,
             loading_model="qwen3:32b", load_attempts=1)
    s2, _ = transition(s, Event.THERMAL_TRIP)
    assert s2.state is SlotState.COOLING
    assert s2.loading_model is None
    assert s2.load_attempts == 0


# ─────────────────────────────────────────────────────────────────────────────
# Slot FSM — BUSY and its post-work fall-throughs
# ─────────────────────────────────────────────────────────────────────────────

def test_busy_work_end_ok_returns_ready_with_model_preserved():
    j = job("m")
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m", current_job=j)
    s2, eff = transition(s, Event.WORK_END, outcome="ok")
    assert s2.state is SlotState.READY
    assert s2.model_loaded == "m"
    assert s2.current_job is None
    assert eff == [Effect("work_end", {"url": GPU, "job": j, "outcome": "ok"})]


def test_busy_work_end_fail_still_returns_ready():
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m", current_job=job("m"))
    s2, eff = transition(s, Event.WORK_END, outcome="fail")
    assert s2.state is SlotState.READY
    assert eff[0].data["outcome"] == "fail"


def test_busy_work_end_timeout_drives_recovery_through_loading():
    j = job("m")
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m", current_job=j)
    s2, eff = transition(s, Event.WORK_END, outcome="timeout")
    assert s2.state is SlotState.LOADING
    assert s2.loading_model == "m"
    assert s2.current_job is None
    kinds = [e.kind for e in eff]
    assert kinds == ["work_end", "load"]
    assert eff[1].data["evict"] == "m"   # forced eviction of the resident model


def test_busy_thermal_trip_latches_and_flips_on_next_work_end():
    j = job("m")
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m", current_job=j)
    s2, eff = transition(s, Event.THERMAL_TRIP)
    assert s2.state is SlotState.BUSY
    assert s2.thermal_pending is True
    assert eff == []

    s3, _ = transition(s2, Event.WORK_END, outcome="ok")
    assert s3.state is SlotState.COOLING
    assert s3.thermal_pending is False


def test_busy_drain_latches_and_flips_on_next_work_end():
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m",
             current_job=job("m"))
    s2, eff = transition(s, Event.DRAIN)
    assert s2.state is SlotState.BUSY
    assert s2.drain_pending is True
    assert eff == []

    s3, _ = transition(s2, Event.WORK_END, outcome="ok")
    assert s3.state is SlotState.DRAINING
    assert s3.drain_pending is False


def test_busy_probe_fail_is_noop():
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m",
             current_job=job("m"))
    s2, eff = transition(s, Event.PROBE_FAIL)
    assert s2 == s
    assert eff == []


def test_busy_thermal_then_drain_prefers_cooling_over_draining():
    """If both flags latch, COOLING wins (thermal is the more urgent hazard)."""
    s = Slot(url=GPU, state=SlotState.BUSY, model_loaded="m",
             current_job=job("m"))
    s2, _ = transition(s, Event.THERMAL_TRIP)
    s3, _ = transition(s2, Event.DRAIN)
    assert s3.state is SlotState.BUSY
    assert s3.thermal_pending and s3.drain_pending

    s4, _ = transition(s3, Event.WORK_END, outcome="ok")
    assert s4.state is SlotState.COOLING


# ─────────────────────────────────────────────────────────────────────────────
# Slot FSM — COOLING / UNREACHABLE / DRAINING
# ─────────────────────────────────────────────────────────────────────────────

def test_cooling_thermal_clear_returns_ready():
    s = Slot(url=GPU, state=SlotState.COOLING, model_loaded="m")
    s2, _ = transition(s, Event.THERMAL_CLEAR)
    assert s2.state is SlotState.READY
    assert s2.model_loaded == "m"


def test_cooling_probe_fail_goes_unreachable():
    s = Slot(url=GPU, state=SlotState.COOLING, model_loaded="m")
    s2, eff = transition(s, Event.PROBE_FAIL)
    assert s2.state is SlotState.UNREACHABLE
    assert s2.model_loaded is None
    assert eff and eff[0].kind == "gpu_unreachable"


def test_unreachable_probe_ok_returns_ready_without_model():
    s = Slot(url=GPU, state=SlotState.UNREACHABLE)
    s2, eff = transition(s, Event.PROBE_OK)
    assert s2.state is SlotState.READY
    assert s2.model_loaded is None
    assert eff == []


def test_drain_from_ready_goes_draining_immediately():
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="m")
    s2, _ = transition(s, Event.DRAIN)
    assert s2.state is SlotState.DRAINING


def test_drain_from_loading_aborts():
    s = Slot(url=GPU, state=SlotState.LOADING, loading_model="m", load_attempts=2)
    s2, _ = transition(s, Event.DRAIN)
    assert s2.state is SlotState.DRAINING
    assert s2.loading_model is None
    assert s2.load_attempts == 0


def test_draining_undrain_returns_ready():
    s = Slot(url=GPU, state=SlotState.DRAINING)
    s2, _ = transition(s, Event.UNDRAIN)
    assert s2.state is SlotState.READY


# ─────────────────────────────────────────────────────────────────────────────
# Slot FSM — illegal transitions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state,event", [
    (SlotState.UNKNOWN,     Event.WORK_BEGIN),
    (SlotState.UNKNOWN,     Event.WORK_END),
    (SlotState.UNKNOWN,     Event.LOAD_DONE),
    (SlotState.READY,       Event.LOAD_DONE),
    (SlotState.READY,       Event.WORK_END),
    (SlotState.LOADING,     Event.WORK_BEGIN),
    (SlotState.LOADING,     Event.WORK_END),
    (SlotState.BUSY,        Event.LOAD_BEGIN),
    (SlotState.BUSY,        Event.LOAD_DONE),
    (SlotState.BUSY,        Event.WORK_BEGIN),
    (SlotState.COOLING,     Event.WORK_BEGIN),
    (SlotState.COOLING,     Event.LOAD_BEGIN),
    (SlotState.UNREACHABLE, Event.WORK_BEGIN),
    (SlotState.UNREACHABLE, Event.LOAD_BEGIN),
    (SlotState.DRAINING,    Event.WORK_BEGIN),
    (SlotState.DRAINING,    Event.LOAD_BEGIN),
])
def test_illegal_transition_raises(state, event):
    slot = Slot(url=GPU, state=state, model_loaded="m")
    with pytest.raises(InvalidTransition):
        # Provide common kwargs so we exercise the guard, not a KeyError.
        if event is Event.WORK_BEGIN:
            transition(slot, event, job=job("m"))
        elif event is Event.LOAD_BEGIN:
            transition(slot, event, model="m")
        elif event is Event.WORK_END:
            transition(slot, event, outcome="ok")
        else:
            transition(slot, event)


# ─────────────────────────────────────────────────────────────────────────────
# dispatch_pass
# ─────────────────────────────────────────────────────────────────────────────

def _buckets(**by_p: list[dict]) -> list[list[dict]]:
    """Build a 5-slot bucket list from kwargs keyed by priority integer name."""
    out: list[list[dict]] = [[] for _ in range(5)]
    for name, jobs in by_p.items():
        out[int(name[1:])] = list(jobs)
    return out


def test_dispatch_strict_priority_chat_before_background():
    buckets = _buckets(p0=[job("m", "chat-job")], p4=[job("m", "bg-job")])
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    eff, slots = dispatch_pass(buckets, slots)
    assert len(eff) == 1
    assert eff[0].data["job"]["id"] == "chat-job"
    assert slots[0].state is SlotState.BUSY
    # Background job still queued — only one READY slot.
    assert buckets[0] == []
    assert len(buckets[4]) == 1


def test_dispatch_restart_from_top_after_each_pop():
    """Two READY slots holding same model; two jobs in different priorities.
    Higher priority must be served first even though iteration lands on it
    second in the second inner-loop pass."""
    buckets = _buckets(p0=[job("m", "chat")], p4=[job("m", "bg")])
    slots = [
        Slot(url=GPU,  state=SlotState.READY, model_loaded="m"),
        Slot(url=GPU1, state=SlotState.READY, model_loaded="m"),
    ]
    eff, slots = dispatch_pass(buckets, slots)
    assigned = [e.data["job"]["id"] for e in eff]
    assert assigned[0] == "chat"    # CHAT served first
    assert "bg" in assigned
    assert all(s.state is SlotState.BUSY for s in slots)


def test_dispatch_skips_bucket_whose_head_has_no_matching_slot():
    """Priority inversion by design: if CHAT wants a model no slot holds, but
    BACKGROUND wants a model that IS loaded, BACKGROUND runs. stage_pass
    will separately kick off a load for the CHAT model."""
    buckets = _buckets(p0=[job("unloaded", "chat")], p4=[job("m", "bg")])
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    eff, slots = dispatch_pass(buckets, slots)
    assert len(eff) == 1
    assert eff[0].data["job"]["id"] == "bg"
    assert len(buckets[0]) == 1  # CHAT job preserved for next tick


def test_dispatch_requires_ready_state_not_busy_or_loading():
    buckets = _buckets(p0=[job("m")])
    slots = [
        Slot(url=GPU,  state=SlotState.BUSY,    model_loaded="m",
             current_job=job("m", "in-flight")),
        Slot(url=GPU1, state=SlotState.LOADING, loading_model="m"),
    ]
    eff, _ = dispatch_pass(buckets, slots)
    assert eff == []
    assert len(buckets[0]) == 1


def test_dispatch_noop_when_all_buckets_empty():
    buckets = _buckets()
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    eff, slots_after = dispatch_pass(buckets, slots)
    assert eff == []
    assert slots_after == slots


# ─────────────────────────────────────────────────────────────────────────────
# stage_pass
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_mirrors_single_demand_across_spare_slots():
    """One distinct demand, two idle slots, neither holds the model →
    both slots start loading it (idle-slot mirroring)."""
    buckets = _buckets(p0=[job("qwen3:32b")])
    slots = [
        Slot(url=GPU,  state=SlotState.READY, model_loaded=None),
        Slot(url=GPU1, state=SlotState.READY, model_loaded=None),
    ]
    eff, slots = stage_pass(buckets, slots)
    assert [e.kind for e in eff] == ["load", "load"]
    assert all(s.state is SlotState.LOADING and
               s.loading_model == "qwen3:32b" for s in slots)


def test_stage_skips_slots_already_holding_demand():
    buckets = _buckets(p0=[job("qwen3:32b")])
    slots = [
        Slot(url=GPU,  state=SlotState.READY, model_loaded="qwen3:32b"),
        Slot(url=GPU1, state=SlotState.READY, model_loaded=None),
    ]
    eff, slots = stage_pass(buckets, slots)
    # Slot 0 already holds the model; mirror loads onto slot 1.
    assert len(eff) == 1
    assert eff[0].data["url"] == GPU1
    assert slots[1].state is SlotState.LOADING


def test_stage_assigns_two_distinct_demands_to_two_slots():
    buckets = _buckets(p0=[job("a", "ja")], p2=[job("b", "jb")])
    slots = [
        Slot(url=GPU,  state=SlotState.READY, model_loaded=None),
        Slot(url=GPU1, state=SlotState.READY, model_loaded=None),
    ]
    eff, slots = stage_pass(buckets, slots)
    assert len(eff) == 2
    loaded_models = {s.loading_model for s in slots}
    assert loaded_models == {"a", "b"}


def test_stage_noop_when_demand_already_satisfied_everywhere():
    buckets = _buckets(p0=[job("m")])
    slots = [
        Slot(url=GPU,  state=SlotState.READY, model_loaded="m"),
        Slot(url=GPU1, state=SlotState.READY, model_loaded="m"),
    ]
    eff, slots_after = stage_pass(buckets, slots)
    assert eff == []
    assert slots_after == slots


def test_stage_noop_when_no_demand():
    buckets = _buckets()
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    eff, slots_after = stage_pass(buckets, slots)
    assert eff == []
    assert slots_after == slots


def test_stage_cannot_use_busy_slots():
    buckets = _buckets(p0=[job("m")])
    slots = [
        Slot(url=GPU,  state=SlotState.BUSY, model_loaded="other",
             current_job=job("other")),
        Slot(url=GPU1, state=SlotState.BUSY, model_loaded="other",
             current_job=job("other")),
    ]
    eff, _ = stage_pass(buckets, slots)
    assert eff == []


def test_stage_treats_in_progress_load_as_satisfying_demand():
    """A slot already LOADING the demanded model should not trigger another
    load — counts as satisfying that demand."""
    buckets = _buckets(p0=[job("qwen3:32b")])
    slots = [
        Slot(url=GPU,  state=SlotState.LOADING, loading_model="qwen3:32b"),
        Slot(url=GPU1, state=SlotState.READY,   model_loaded=None),
    ]
    eff, slots = stage_pass(buckets, slots)
    # The in-progress load on slot 0 satisfies demand 0; mirror fills slot 1.
    assert len(eff) == 1
    assert eff[0].data["url"] == GPU1


# ─────────────────────────────────────────────────────────────────────────────
# project_status (truth table)
# ─────────────────────────────────────────────────────────────────────────────

def test_status_not_recovered_wins_over_everything():
    slots = [Slot(url=GPU, state=SlotState.BUSY, model_loaded="m",
                  current_job=job("m"))]
    assert project_status(paused=True, recovered=False,
                          slots=slots, buckets_nonempty=True) \
        is SchedulerStatus.RECOVERING


def test_status_paused_when_recovered_and_paused():
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    assert project_status(paused=True, recovered=True,
                          slots=slots, buckets_nonempty=False) \
        is SchedulerStatus.PAUSED


def test_status_degraded_when_all_slots_unreachable():
    slots = [Slot(url=GPU,  state=SlotState.UNREACHABLE),
             Slot(url=GPU1, state=SlotState.UNREACHABLE)]
    assert project_status(paused=False, recovered=True,
                          slots=slots, buckets_nonempty=True) \
        is SchedulerStatus.DEGRADED


def test_status_draining_when_any_slot_busy():
    slots = [Slot(url=GPU,  state=SlotState.READY, model_loaded="m"),
             Slot(url=GPU1, state=SlotState.BUSY,  model_loaded="m",
                  current_job=job("m"))]
    assert project_status(paused=False, recovered=True,
                          slots=slots, buckets_nonempty=False) \
        is SchedulerStatus.DRAINING


def test_status_draining_when_any_slot_loading():
    slots = [Slot(url=GPU, state=SlotState.LOADING, loading_model="m")]
    assert project_status(paused=False, recovered=True,
                          slots=slots, buckets_nonempty=False) \
        is SchedulerStatus.DRAINING


def test_status_dispatching_when_ready_and_buckets_nonempty():
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    assert project_status(paused=False, recovered=True,
                          slots=slots, buckets_nonempty=True) \
        is SchedulerStatus.DISPATCHING


def test_status_idle_when_ready_and_buckets_empty():
    slots = [Slot(url=GPU, state=SlotState.READY, model_loaded="m")]
    assert project_status(paused=False, recovered=True,
                          slots=slots, buckets_nonempty=False) \
        is SchedulerStatus.IDLE


def test_status_idle_when_no_slots_at_all():
    assert project_status(paused=False, recovered=True,
                          slots=[], buckets_nonempty=False) \
        is SchedulerStatus.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (for Phase 3 wiring — must not break purity or raise)
# ─────────────────────────────────────────────────────────────────────────────

def test_log_transition_emits_on_state_change(caplog):
    from scheduler import log_transition
    import logging as _logging
    before = Slot(url=GPU, state=SlotState.READY, model_loaded="m")
    after, effects = transition(before, Event.WORK_BEGIN, job=job("m"))
    with caplog.at_level(_logging.INFO, logger="merllm.scheduler.fsm"):
        log_transition(before, Event.WORK_BEGIN, after, effects)
    assert any("[fsm]" in rec.message and "ready→busy" in rec.message
               for rec in caplog.records)


def test_log_transition_skips_noop(caplog):
    from scheduler import log_transition
    import logging as _logging
    s = Slot(url=GPU, state=SlotState.READY, model_loaded="m")
    after, effects = transition(s, Event.PROBE_OK)
    with caplog.at_level(_logging.INFO, logger="merllm.scheduler.fsm"):
        log_transition(s, Event.PROBE_OK, after, effects)
    assert not any("[fsm]" in rec.message for rec in caplog.records)


def test_log_tick_summary_emits_structured_line(caplog):
    from scheduler import log_tick_summary
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="merllm.scheduler.tick"):
        log_tick_summary(
            status=SchedulerStatus.DISPATCHING,
            dispatched=2, staged=1,
            bucket_depths=[0, 0, 1, 0, 3],
            slot_summary=[
                ("http://gpu0:11434", "busy", "qwen3:32b"),
                ("http://gpu1:11435", "loading", "qwen3:30b"),
            ],
        )
    assert any("[tick]" in rec.message and "dispatched=2" in rec.message
               and "staged=1" in rec.message for rec in caplog.records)
