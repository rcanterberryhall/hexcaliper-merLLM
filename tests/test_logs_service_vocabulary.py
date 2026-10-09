"""tests/test_logs_service_vocabulary.py — the Logs tab offers the services the
endpoint can actually serve, and the pane says whose lines it holds (issue #86).

Two files have to agree about one set of names: the ``#log-service`` dropdown in
``web/index.html`` and ``_ALL_LOG_SERVICES`` in ``api/app.py``. Nothing asserted
that before, and they had drifted — the dropdown offered ``ollama-gpu0`` and
``ollama-gpu1``, which are systemd units on the host rather than Docker
containers, so ``service_logs`` rejected both with a 422. ``ollama-gpu0`` being
the first option made it the default selection, so the failure was what an
operator met on the first Fetch of a freshly loaded tab.

The checks come in two kinds:

* behavioural, through ``TestClient`` — every name the dropdown offers gets past
  the endpoint's validation, and a name it does not offer is still rejected, so
  the first assertion cannot be satisfied by deleting the guard;
* structural, over the shipped ``web/`` assets — a Fetch overtaken by a later
  one does not render, and what does render names its own service and container
  (the dropdown stops naming it the moment the operator moves the selection).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


APP_JS = _read("web/app.js")
INDEX_HTML = _read("web/index.html")


def _function_body(src, name):
    """Return one top-level function body, declaration line to closing brace.

    Bodies are bracketed by a declaration at column 0 and the next ``}`` at
    column 0, which survives this file's template literals better than counting
    braces does.
    """
    lines = src.split("\n")
    start = next(
        i for i, line in enumerate(lines)
        if re.match(r"^(async )?function %s\(" % re.escape(name), line)
    )
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("}"))
    return "\n".join(lines[start:end + 1])


def _offered_services():
    """The option values of the Logs tab's service dropdown, in markup order."""
    select = re.search(
        r'<select id="log-service">(.*?)</select>', INDEX_HTML, re.S)
    assert select, "no #log-service dropdown found in web/index.html"
    return re.findall(r'<option value="([^"]+)"', select.group(1))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with an isolated DB so importing the app touches nothing live."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in ("app", "db", "config", "queue_manager", "gpu_router", "metrics"):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


# ── The dropdown and the endpoint agree about the set of services ────────────

def test_every_dropdown_option_is_accepted_by_the_endpoint(client):
    """A name the dropdown offers must get past ``service_logs``'s validation.

    Reaching Docker is not part of this: there is no daemon in CI, so a valid
    name ends in a 500 from the ``except Exception`` around
    ``containers.get``. What matters is that it is not the 422 that says the
    name itself is unknown, which is what both ollama options returned.
    """
    offered = _offered_services()
    assert offered, "the dropdown offers nothing at all"
    for service in offered:
        resp = client.get("/api/merllm/logs/%s?lines=10" % service)
        assert resp.status_code != 422, (service, resp.json())
        assert "Unknown service" not in resp.text, (service, resp.text)


def test_a_service_outside_the_dropdown_is_still_rejected(client):
    """The 422 guard stays, so the assertion above cannot be met by removing it.

    ``_LOG_DOCKER`` is an allowlist and ``{service}`` is a path parameter that
    becomes a Docker container name, so dropping the check would widen the
    endpoint to any container on the host.
    """
    resp = client.get("/api/merllm/logs/ollama-gpu0?lines=10")
    assert resp.status_code == 422
    assert "Unknown service" in resp.json()["detail"]


def test_the_dropdown_offers_every_service_the_endpoint_serves(client):
    """Set equality, in both directions, parsed from the two shipped files.

    An option the endpoint rejects is a dead entry in the UI; a service the
    endpoint serves but the dropdown omits is unreachable from the tab. Both
    are drift between these two files, which is what went unnoticed in #86.
    """
    import app as app_mod
    assert set(_offered_services()) == set(app_mod._ALL_LOG_SERVICES), (
        _offered_services(), sorted(app_mod._ALL_LOG_SERVICES))


def test_the_host_ollama_units_are_not_offered_as_log_services():
    """``service_logs`` reads container logs; these two are host systemd units.

    ``docker-compose.yml`` pins both as "GPU-pinned systemd services on host",
    so no mapping in ``_LOG_DOCKER`` could make them work — the endpoint has no
    mechanism that reads a journal. They are read on the host with
    ``journalctl -u ollama-gpu0``.
    """
    offered = _offered_services()
    assert "ollama-gpu0" not in offered
    assert "ollama-gpu1" not in offered


def test_every_mapped_service_resolves_to_a_configured_container_name(client):
    """A mapping whose config value is empty would ask Docker for "" and 500."""
    import app as app_mod
    for service, container in app_mod._LOG_DOCKER.items():
        assert isinstance(container, str) and container, (service, container)


# ── A superseded Fetch must not render ───────────────────────────────────────

def test_fetchlogs_takes_a_sequence_number_at_entry():
    body = _function_body(APP_JS, "fetchLogs")
    assert "const seq = ++_logsSeq;" in body
    assert "let _logsSeq = 0;" in APP_JS


def test_a_superseded_fetch_renders_neither_its_lines_nor_its_error():
    """Both exits need the check.

    Fetch a slow service, then a fast one. If only the success path is guarded,
    the slow request failing still overwrites the fast request's lines with an
    error naming a service the operator has moved away from.
    """
    body = _function_body(APP_JS, "fetchLogs")
    assert body.count("if (seq !== _logsSeq) return;") == 2, body


def test_the_supersession_check_sits_between_the_await_and_the_render():
    """Placement is the whole point — after the render it guards nothing."""
    body = _function_body(APP_JS, "fetchLogs")
    awaited = body.index("await api(")
    checked = body.index("if (seq !== _logsSeq) return;")
    rendered = body.index("el.textContent = `──")
    assert awaited < checked < rendered, body


# ── The pane identifies what it is showing ───────────────────────────────────

def test_the_pane_names_the_service_and_container_from_the_response():
    """The response carries ``service`` and ``container``; both were discarded.

    Reading them off the response rather than the dropdown is what makes the
    label survive the operator changing the selection mid-request.
    """
    body = _function_body(APP_JS, "fetchLogs")
    assert "${r.service}" in body
    assert "${r.container}" in body


def test_the_in_flight_and_failed_messages_both_name_their_service():
    body = _function_body(APP_JS, "fetchLogs")
    assert "`Fetching ${service}…`" in body
    assert "`${service}: ${err.message}`" in body


def test_the_endpoint_returns_the_service_and_container_the_pane_labels_with():
    """The label would read ``undefined (undefined)`` if these were dropped."""
    returned = re.search(
        r'return \{"service": service, "container": container_name, "lines": ',
        _read("api/app.py"))
    assert returned, "service_logs must return the names the pane renders"
