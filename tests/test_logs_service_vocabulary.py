"""tests/test_logs_service_vocabulary.py — the Logs tab offers the services the
endpoint can actually serve, and the pane says whose lines it holds (issues #86,
#88).

Two files have to agree about one set of names: the ``#log-service`` dropdown in
``web/index.html`` and ``_ALL_LOG_SERVICES`` in ``api/app.py``. Nothing asserted
that before, and they had drifted — the dropdown offered ``ollama-gpu0`` and
``ollama-gpu1``, which are systemd units on the host rather than Docker
containers, so ``service_logs`` rejected both with a 422. ``ollama-gpu0`` being
the first option made it the default selection, so the failure was what an
operator met on the first Fetch of a freshly loaded tab.

#86 restored the equality by removing the two options. #88 restores it the other
way round, by giving the endpoint a second reader: ``_LOG_SOURCES`` maps each
name to a ``(reader, target)`` pair, where "docker" reads a container through
the mounted socket and "journal" reads a host systemd unit over the SSH account
the terminal tab already uses. So the assertions here now have to hold across
two kinds of source, and the pane has to say which kind it is showing — calling
a systemd unit a container is the same class of untruth the equality prevents.

The checks come in two kinds:

* behavioural, through ``TestClient`` — every name the dropdown offers gets past
  the endpoint's validation, and a name it does not offer is still rejected, so
  the first assertion cannot be satisfied by deleting the guard;
* structural, over the shipped ``web/`` assets — a Fetch overtaken by a later
  one does not render, and what does render names its own service and source,
  and says which kind of source it is (the dropdown stops naming it the moment
  the operator moves the selection).
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


def _python_function_body(src, name):
    """Return one module-level Python function body, ``def`` line to dedent.

    ``_function_body`` above matches a JavaScript declaration and cannot be
    pointed at ``api/app.py``.
    """
    lines = src.split("\n")
    start = next(
        i for i, line in enumerate(lines)
        if re.match(r"^(async )?def %s\(" % re.escape(name), line)
    )
    end = next(
        i for i in range(start + 1, len(lines))
        if lines[i] and not lines[i][0].isspace()
    )
    return "\n".join(lines[start:end])


def _offered_services():
    """The option values of the Logs tab's service dropdown, in markup order."""
    select = re.search(
        r'<select id="log-service">(.*?)</select>', INDEX_HTML, re.S)
    assert select, "no #log-service dropdown found in web/index.html"
    return re.findall(r'<option value="([^"]+)"', select.group(1))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with an isolated DB so importing the app touches nothing live.

    ``SSH_USER`` is blanked as well, so the journal reader short-circuits with a
    503 instead of dialling a host. A test that wants the reader exercised
    replaces ``_read_unit_journal`` outright.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SSH_USER", "")
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

    ``_LOG_SOURCES`` is an allowlist and ``{service}`` is a path parameter that
    becomes either a Docker container name or a unit name on a remote command
    line, so dropping the check would widen the endpoint to any container on the
    host and any unit on it. The probe is a name no reader is mapped to;
    ``ollama-gpu0`` used to serve here and cannot now that #88 reads it.
    """
    resp = client.get("/api/merllm/logs/merllm-postgres?lines=10")
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


def test_the_host_ollama_units_are_offered_and_read_as_journals(client):
    """The two units are back in the dropdown, mapped to the journal reader.

    ``docker-compose.yml`` pins both as "GPU-pinned systemd services on host",
    so the Docker reader can never serve them — which is why #86 removed the
    options. They are offered again only because ``_LOG_SOURCES`` now routes
    them to ``_read_unit_journal``; offering them against the Docker reader is
    the dead entry #86 was about.
    """
    import app as app_mod
    offered = _offered_services()
    for unit in ("ollama-gpu0", "ollama-gpu1"):
        assert unit in offered
        reader, target = app_mod._LOG_SOURCES[unit]
        assert reader == "journal", (unit, reader)
        assert target == unit, (unit, target)


def test_merllm_nginx_is_offered_and_read_as_a_container(client):
    """The nginx in front of the API was in neither the dropdown nor the map.

    It is a live container with a fixed ``container_name`` in
    ``docker-compose.yml``, so unlike the ollama units it needed no new
    mechanism — only a setting, a map entry and an option.
    """
    import app as app_mod
    assert "merllm-nginx" in _offered_services()
    assert app_mod._LOG_SOURCES["merllm-nginx"] == (
        "docker", app_mod.config.LOG_CONTAINER_MERLLM_NGINX)


def test_every_mapped_service_resolves_to_a_known_reader_and_a_target(client):
    """An empty target would ask Docker for "" or journalctl for no unit.

    The reader has to be one the endpoint dispatches on as well: an unrecognised
    value falls to the Docker branch, which is how a unit would silently be
    looked up as a container again.
    """
    import app as app_mod
    for service, (reader, target) in app_mod._LOG_SOURCES.items():
        assert reader in ("docker", "journal"), (service, reader)
        assert isinstance(target, str) and target, (service, target)


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

def test_the_pane_names_the_service_and_source_from_the_response():
    """The response carries ``service`` and ``source``; both were discarded.

    Reading them off the response rather than the dropdown is what makes the
    label survive the operator changing the selection mid-request.
    """
    body = _function_body(APP_JS, "fetchLogs")
    assert "${r.service}" in body
    assert "${r.source}" in body


def test_the_pane_calls_a_host_unit_a_unit_and_a_container_a_container():
    """One label for two kinds of source would misname one of them.

    The old field was called ``container`` and the label printed it bare, so a
    journal would have been announced as a container named ``ollama-gpu0`` —
    a thing that does not exist. The pane branches on ``reader`` instead.
    """
    body = _function_body(APP_JS, "fetchLogs")
    assert 'r.reader === "journal"' in body
    assert "systemd unit ${r.source}" in body
    assert "container ${r.source}" in body


def test_the_in_flight_and_failed_messages_both_name_their_service():
    body = _function_body(APP_JS, "fetchLogs")
    assert "`Fetching ${service}…`" in body
    assert "`${service}: ${err.message}`" in body


def test_the_endpoint_returns_the_three_fields_the_pane_labels_with():
    """The label would read ``undefined (undefined undefined)`` without these."""
    returned = re.search(
        r'return \{"service": service, "reader": reader, "source": target,',
        _read("api/app.py"))
    assert returned, "service_logs must return the names the pane renders"


# ── The journal reader (issue #88) ───────────────────────────────────────────

def test_a_journal_service_is_served_by_the_journal_reader(client, monkeypatch):
    """The dispatch, end to end, without dialling a host.

    What this pins is that a journal name does not reach ``_read_container_logs``
    — the regression #86 described, where ``ollama-gpu0`` was looked up as a
    container that does not exist.
    """
    import app as app_mod

    async def fake_journal(unit, lines):
        return "Oct 15 01:44:52 hexcaliper ollama[3085]: starting runner\nsecond line"

    def refuse_docker(container, lines):
        raise AssertionError("a host unit was read as a container: %r" % container)

    monkeypatch.setattr(app_mod, "_read_unit_journal", fake_journal)
    monkeypatch.setattr(app_mod, "_read_container_logs", refuse_docker)

    body = client.get("/api/merllm/logs/ollama-gpu0?lines=10").json()
    assert body["reader"] == "journal"
    assert body["source"] == "ollama-gpu0"
    assert len(body["lines"]) == 2


def test_a_container_service_is_served_by_the_docker_reader(client, monkeypatch):
    """And the converse: adding a reader must not reroute the existing five."""
    import app as app_mod

    async def refuse_journal(unit, lines):
        raise AssertionError("a container was read as a unit: %r" % unit)

    monkeypatch.setattr(app_mod, "_read_container_logs", lambda c, n: "one line")
    monkeypatch.setattr(app_mod, "_read_unit_journal", refuse_journal)

    body = client.get("/api/merllm/logs/merllm-api?lines=10").json()
    assert body["reader"] == "docker"
    assert body["source"] == app_mod.config.LOG_CONTAINER_MERLLM_API
    assert body["lines"] == ["one line"]


def test_an_unconfigured_ssh_account_says_so_and_names_the_unit(client):
    """503 with the reason, not a 422 saying the name is unknown.

    The journal reader borrows the terminal tab's credentials, so a deployment
    that has not set ``SSH_USER`` cannot serve these two. That is a
    configuration answer the operator can act on; #86's 422 said the service did
    not exist, which sent the reader looking for a typo instead.
    """
    resp = client.get("/api/merllm/logs/ollama-gpu1?lines=10")
    assert resp.status_code == 503, resp.text
    detail = resp.json()["detail"]
    assert "ollama-gpu1" in detail
    assert "SSH_USER" in detail


def test_the_line_count_is_bounded_before_it_reaches_the_journal_reader(client, monkeypatch):
    """``lines`` arrives from the query string and ends on a command line.

    The dropdown's input caps at 500, but the query parameter is reachable
    directly, and ``journalctl -n`` with an unbounded count would return the
    whole journal through one response.
    """
    import app as app_mod
    seen = {}

    async def capture(unit, lines):
        seen["lines"] = lines
        return ""

    monkeypatch.setattr(app_mod, "_read_unit_journal", capture)
    client.get("/api/merllm/logs/ollama-gpu0?lines=100000")
    assert seen["lines"] == app_mod._LOG_LINES_MAX


def test_the_unit_name_is_quoted_into_the_journal_command():
    """``_LOG_UNIT_*`` is operator-set config interpolated into a shell command.

    The endpoint's allowlist means the path parameter cannot reach it, so this
    guards the other writer: a unit name configured with a space or a semicolon.
    Asserted on the source because the command string is built and consumed
    inside one ``async with``, with no seam to read it from.
    """
    body = _python_function_body(_read("api/app.py"), "_read_unit_journal")
    assert "shlex.quote(unit)" in body
    assert "journalctl -u" in body
