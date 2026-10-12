"""tests/test_queue_load_failure_reporting.py — issue #91.

A failed ``GET /api/merllm/queue`` used to be swallowed at ``console.error``,
which left the shipped static markup asserting the reassuring state of every
element the load was meant to fill: an unpaused dispatcher, no rows in flight
and no bucket depths. Worse, ``renderQueuePauseState`` is the only writer of
``#queue-pause-btn``'s ``data-paused`` attribute and ``toggleQueuePause`` reads
that attribute to choose between pause and resume, so a failed load left the
Resume control unreachable.

Two groups of tests live here:

* **Backend premises** — the facts the frontend fix rests on. These must pass on
  ``origin/main`` as well, since they establish that the pause flag really is
  served by both endpoints and that a redundant pause really is a no-op. They
  are also the first coverage of the operator pause/resume API.
* **Shipped-asset assertions** — the frontend contract, read out of
  ``web/app.js`` and ``web/index.html``. Whole-line comments are stripped before
  matching, because this codebase's house style puts a *why* comment above each
  step and a needle would otherwise match the prose instead of the code.
"""
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications", "scheduler"):
            sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


@pytest.fixture
def qm(monkeypatch, tmp_path):
    """queue_manager with an isolated DB and the pause flag reset."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "qm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager"):
            sys.modules.pop(mod, None)
    import queue_manager as qm_mod
    qm_mod._paused = False
    qm_mod._paused_since = None
    yield qm_mod
    qm_mod._paused = False
    qm_mod._paused_since = None


# ── Source helpers ────────────────────────────────────────────────────────────

def _strip_line_comments(src: str) -> str:
    """Drop whole-line ``//`` comments.

    Only whole-line comments are removed, so a URL literal such as
    ``${proto}//${host}`` inside code is never touched. That is enough for the
    risk this guards against: the repo's convention of a prose comment above
    each step, which makes a bare text needle match the explanation rather than
    the statement it explains.
    """
    return "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("//")
    )


def _app_js() -> str:
    return _strip_line_comments((WEB / "app.js").read_text(encoding="utf-8"))


def _index_html() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


def _function_body(src: str, signature: str) -> str:
    """Return the brace-matched body of the function introduced by *signature*."""
    start = src.index(signature)
    depth, i, opened = 0, src.index("{", start), False
    while i < len(src):
        if src[i] == "{":
            depth += 1
            opened = True
        elif src[i] == "}":
            depth -= 1
            if opened and depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _catch_block(src: str, signature: str) -> str:
    body = _function_body(src, signature)
    return body[body.index("catch"):]


# ── Backend premises (must hold on origin/main too) ──────────────────────────

def test_status_endpoint_carries_the_pause_flag_and_its_timestamp(client):
    """``/api/merllm/status`` is a second, equally authoritative source.

    The fix paints the banner and the Pause/Resume button from this response as
    well as from ``/api/merllm/queue``, so the control survives either endpoint
    failing. That is only sound because both fields are actually here.
    """
    body = client.get("/api/merllm/status").json()
    assert "queue_paused" in body
    assert "queue_paused_since" in body


def test_queue_endpoint_carries_the_pause_flag_and_its_timestamp(client):
    body = client.get("/api/merllm/queue").json()
    assert "paused" in body
    assert "paused_since" in body


def test_both_endpoints_report_the_same_pause_state(client, monkeypatch):
    """The two fields are the same ``queue_manager.is_paused()`` call.

    This is what makes the badge disagreeing with the button a contradiction
    rather than two different facts.
    """
    import queue_manager

    monkeypatch.setattr(queue_manager, "is_paused", lambda: True)
    monkeypatch.setattr(queue_manager, "paused_since", lambda: 1234.0)

    status = client.get("/api/merllm/status").json()
    queue = client.get("/api/merllm/queue").json()

    assert status["queue_paused"] is True
    assert queue["paused"] is True
    assert status["queue_paused_since"] == queue["paused_since"] == 1234.0


def test_pause_response_carries_changed_and_the_authoritative_timestamp(client):
    """``toggleQueuePause`` renders the age from this response.

    It used to render ``Date.now() / 1000``, which reads ``(paused 0s ago)`` for
    a queue paused hours earlier.
    """
    first = client.post("/api/merllm/queue/pause").json()
    assert first["paused"] is True
    assert first["changed"] is True
    assert isinstance(first["paused_since"], (int, float))

    second = client.post("/api/merllm/queue/pause").json()
    assert second["changed"] is False, "a redundant pause must be a no-op"
    assert second["paused_since"] == first["paused_since"], (
        "a redundant pause must not restamp the pause age"
    )

    resumed = client.post("/api/merllm/queue/resume").json()
    assert resumed["paused"] is False


def test_redundant_pause_preserves_the_original_timestamp(qm):
    """``set_paused`` returns False unchanged when the flag already matches.

    This is why the mislabelled button's POST is silent: it does nothing and
    reports success, so nothing on screen changes and the operator cannot tell
    the control was wrong.
    """
    assert qm.set_paused(True, persist=False) is True
    first_since = qm.paused_since()
    assert first_since is not None

    assert qm.set_paused(True, persist=False) is False
    assert qm.paused_since() == first_since
    assert qm.is_paused() is True

    assert qm.set_paused(False, persist=False) is True
    assert qm.paused_since() is None


def test_scheduler_projects_paused_so_the_badge_has_a_second_source(qm):
    """``scheduler_status`` reads ``paused`` before every health state.

    ``renderSchedulerStatus`` paints it, which is the on-screen value the broken
    button could contradict.
    """
    import scheduler

    status = scheduler.project_status(
        paused=True, recovered=True, slots=[], buckets_nonempty=False,
    )
    assert status is scheduler.SchedulerStatus.PAUSED


# ── The static markup is the reason silence is not neutral ───────────────────

def test_static_pause_button_asserts_an_unpaused_queue(qm):
    """Premise, true on both refs: the shipped button text is *Pause Queue*.

    It also ships with no ``data-paused`` attribute, so until a render supplies
    one ``toggleQueuePause``'s ``dataset.paused === "1"`` is false and the
    button sends ``/pause``. A skipped render is therefore not a neutral
    no-op — it leaves an assertion on screen.
    """
    html = _index_html()
    button = re.search(r"<button[^>]*id=\"queue-pause-btn\"[^>]*>([^<]*)<", html)
    assert button, "#queue-pause-btn not found in web/index.html"
    assert button.group(1).strip() == "Pause Queue"
    assert "data-paused" not in button.group(0)


def test_static_pause_banner_ships_hidden(qm):
    """Premise: the banner explaining the pause is ``display:none`` in markup."""
    html = _index_html()
    banner = re.search(r"<div[^>]*id=\"queue-pause-banner\"[^>]*>", html)
    assert banner, "#queue-pause-banner not found"
    assert "display:none" in banner.group(0).replace(" ", "")


# ── The frontend contract (fails on origin/main) ─────────────────────────────

def test_failed_queue_load_paints_instead_of_only_logging():
    """``loadQueue``'s catch has to put the failure on screen.

    Its siblings two lines away in ``refresh()`` already do: ``loadFaultHistory``
    renders ``Could not load fault history: <reason>`` into its pane.
    """
    catch = _catch_block(_app_js(), "async function loadQueue()")
    assert "paintQueuePanes()" in catch, (
        "loadQueue's catch must repaint the panes it failed to fill, not just "
        "console.error"
    )
    assert "_queueLoad" in catch, "the failure reason must be recorded for the pane"


def test_failed_batch_load_paints_instead_of_only_logging():
    catch = _catch_block(_app_js(), "async function loadBatchJobs()")
    assert "paintBatchPanes()" in catch
    assert "_batchLoad" in catch


def test_queue_failure_reason_comes_from_the_error():
    """The reason shown is the endpoint's, not a generic string."""
    catch = _catch_block(_app_js(), "async function loadQueue()")
    assert "err.message" in catch


def test_empty_queue_message_requires_a_load_to_have_answered():
    """``No active requests.`` is a measurement, not a default.

    ``_lastQueueData`` is ``[]`` before any load succeeds, and
    ``_handleSortClick`` paints straight from it, so one click on a sortable
    header used to turn a never-answered query into a stated fact.
    """
    body = _function_body(_app_js(), "function renderQueueTable(")
    assert "No active requests." in body, "the empty state itself must survive"
    before = body.index("_queueLoad")
    assert before < body.index("No active requests."), (
        "the empty-state message must be guarded by the load state, not emitted "
        "whenever the array is empty"
    )


def test_empty_batch_message_requires_a_load_to_have_answered():
    body = _function_body(_app_js(), "function renderBatchTable(")
    assert "No jobs." in body
    assert body.index("_batchLoad") < body.index("No jobs.")


def test_bucket_lanes_distinguish_unread_from_zero_depth():
    """Five lanes reading 0 is a depth measurement nothing has taken."""
    body = _function_body(_app_js(), "function renderBucketLanes(")
    assert "if (!buckets)" in body, (
        "renderBucketLanes must separate 'never answered' from an answered "
        "response with no depths"
    )
    assert body.index("if (!buckets)") < body.index("_BUCKET_ORDER")


def test_pause_control_has_an_unknown_state():
    """While neither endpoint has reported the flag, the control says so.

    Showing the static ``Pause Queue`` instead would assert a dispatching queue
    and leave Resume unreachable.
    """
    body = _function_body(_app_js(), "function renderQueuePauseState(")
    assert "paused === null" in body, (
        "renderQueuePauseState needs a third state for 'not read yet'"
    )
    assert "Resume Queue" in body and "Pause Queue" in body


def test_unknown_pause_state_clears_the_stale_data_paused_attribute():
    """``toggleQueuePause`` picks its verb from ``data-paused``.

    Leaving a stale value there would make the disabled-state recovery send the
    wrong verb once the button is re-enabled.
    """
    body = _function_body(_app_js(), "function renderQueuePauseState(")
    assert "delete btn.dataset.paused" in body


def test_overview_paints_the_pause_control_from_the_status_response():
    """The second source. ``renderOverview`` already has both fields.

    ``/api/merllm/status`` is fetched by ``refresh()`` every 10s and its failure
    is already reported into ``#refresh-indicator``, so feeding the control from
    here means only a double failure can leave it unknown.
    """
    body = _function_body(_app_js(), "function renderOverview(")
    assert "renderQueuePauseState(" in body, (
        "renderOverview must paint the pause control it has the data for"
    )
    assert "s.queue_paused" in body
    assert "s.queue_paused_since" in body


def test_toggle_pause_uses_the_server_timestamp_not_the_local_clock():
    """A redundant pause returns ``changed: false`` and the original timestamp."""
    body = _function_body(_app_js(), "async function toggleQueuePause()")
    assert "r.paused_since" in body, "render the age the server reported"
    assert "Date.now()" not in body, (
        "inventing the pause age shows '(paused 0s ago)' for a queue paused "
        "hours ago"
    )


def test_failure_message_elements_exist_in_the_static_markup():
    """The panes need a home for the reason, beside the tables they describe."""
    html = _index_html()
    assert 'id="queue-load-msg"' in html
    assert 'id="batch-load-msg"' in html


def test_retry_control_reaches_the_loader():
    """A stated failure with no way to retry still strands the operator."""
    js = _app_js()
    body = _function_body(js, "function _renderLoadMsg(")
    assert "Retry" in body
    for call in ("loadQueue()", "loadBatchJobs()"):
        assert call in js


# ── Derived guard: a full-width row must span the whole table ────────────────

@pytest.mark.parametrize("table_id,tbody_id", [
    ("queue-table-wrap", "queue-tbody"),
    ("batch-table-wrap", "batch-tbody"),
])
def test_full_width_rows_span_every_column(table_id, tbody_id):
    """Every ``colspan`` emitted into these tbodies equals the header count.

    Derived from the shipped markup rather than hardcoded, so adding a column
    fails here instead of silently leaving a short row — the defect fixed in
    lanceLLMot #64, where an edit row spanned 6 of 7 columns and its background
    stopped short of the right edge.
    """
    html = _index_html()
    wrap = html[html.index(f'id="{table_id}"'):]
    wrap = wrap[:wrap.index("</table>")]
    columns = len(re.findall(r"<th[\s>]", wrap))
    assert columns, f"no <th> found for {table_id}"

    js = _app_js()
    render = "renderQueueTable(" if tbody_id == "queue-tbody" else "renderBatchTable("
    body = _function_body(js, f"function {render}")
    spans = [int(n) for n in re.findall(r'colspan="(\d+)"', body)]
    assert spans, f"no full-width row emitted into {tbody_id}"
    for span in spans:
        assert span == columns, (
            f"{tbody_id} has {columns} columns but a full-width row spans {span}"
        )
