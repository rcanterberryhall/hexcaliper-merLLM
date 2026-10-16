"""tests/test_notif_permission_control.py — the Settings notification button is reachable, and says what it can do.

The dashboard raises a desktop notification on `job_complete` / `job_failed`,
gated on `Notification.permission === "granted"` (`web/app.js`). The Settings
tab's **Enable browser notifications** button is the only affordance for
granting that permission, and before #98 it could not work: the markup shipped
it `display:none` and nothing cleared that, while the handler meant to relabel
it looked up `notif-permission-btn` against markup that says `notif-perm-btn`.
Both of the handler's feedback branches were therefore dead, silently — a
`getElementById` miss plus a defensive `if (el)` raises no console error.

Three layers, because the defect spanned two languages and one platform rule:

* **markup and source** — the button carries no inline `display`, the dead id
  appears nowhere, and the handler holds no id of its own. Always run.
* **every id JS reads exists in the markup** — the set-difference scan that
  found #98, kept as a regression guard for the whole family rather than for
  this one button. Always run.
* **the state machine**, driven under `node` over the five states an operator
  can be in. `_notifPermissionState()` reads only `window`, so it lifts out of
  `web/app.js` without a DOM; there is no JS test runner in this repo.

The platform rule the third layer pins down: browsers keep `Notification` on
`window` off a secure context and pre-set `permission` to `"denied"` there, so
a `'Notification' in window` support guard never fires in Chrome and
`permission !== "granted"` cannot tell a blocked origin from an operator who
clicked Block. merLLM is served as plain `http://<host>:11400/web/`, so it *is*
an insecure context whenever it is opened from another machine, and
`window.isSecureContext` is the only signal that separates the two cases.
Measured on this host: `file://` and `http://127.0.0.1` report
`isSecureContext: true` with `permission: "default"`, while
`http://<lan-ip>` reports `false` and `"denied"`.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
APP_JS = WEB / "app.js"
INDEX_HTML = WEB / "index.html"

BUTTON_ID = "notif-perm-btn"
DEAD_ID = "notif-permission-btn"


# ── Layer 1: the control is reachable, and one id names it ───────────────────

def test_button_carries_no_inline_display():
    """The button ships visible.

    It lives inside `#tab-settings`, which `.tab { display: none }` already
    hides until the tab is opened, so an inline `display:none` on the button
    hid it a second time with nothing to ever clear it. The redundant
    declaration is the whole defect, so assert on the tag rather than on a
    JS writer that reveals it.
    """
    tag = re.search(r'<button[^>]*id="%s"[^>]*>' % BUTTON_ID, INDEX_HTML.read_text())
    assert tag, f"no <button id={BUTTON_ID!r}> in web/index.html"
    assert "display:none" not in tag.group(0).replace(" ", "")
    assert "style=" not in tag.group(0), (
        "the button's appearance is painted from the permission, so it carries "
        "no inline style: " + tag.group(0)
    )


def test_the_dead_id_is_gone_from_the_repo():
    """`notif-permission-btn` names nothing and must appear nowhere."""
    for path in (APP_JS, INDEX_HTML, WEB / "styles.css"):
        assert DEAD_ID not in path.read_text(), f"{DEAD_ID!r} still in {path.name}"


def test_the_handler_holds_no_id_of_its_own():
    """`requestNotifPermission()` requests, then delegates the repaint.

    The id it used to look up was wrong and nothing could have caught that.
    Keeping every write to the button in one painter means there is one place
    for the id to be wrong, and that place is covered by the scan below.
    """
    src = APP_JS.read_text()
    handler = re.search(
        r"async function requestNotifPermission\(\) \{.*?\n\}\n", src, re.S)
    assert handler, "requestNotifPermission() not found in web/app.js"
    body = handler.group(0)
    assert "getElementById" not in body, f"handler reads the DOM directly:\n{body}"
    assert "_renderNotifPermission()" in body, (
        f"handler does not repaint after the request resolves:\n{body}")


def test_the_painter_is_the_only_reader_of_the_button_id():
    src = APP_JS.read_text()
    reads = [src[:m.start()].count("\n") + 1
             for m in re.finditer(re.escape(f'getElementById("{BUTTON_ID}")'), src)]
    assert len(reads) == 1, f"{BUTTON_ID} is read at {len(reads)} sites: lines {reads}"
    painter = re.search(
        r"function _renderNotifPermission\(\) \{.*?\n\}\n", src, re.S)
    assert painter, "_renderNotifPermission() not found"
    assert f'getElementById("{BUTTON_ID}")' in painter.group(0)


def test_the_painter_runs_on_load_and_when_settings_opens():
    """A permission changed in the browser's own settings is picked up.

    On load alone the button would state the permission as it was when the page
    was opened; the operator granting it in browser settings and coming back to
    the tab is the case the second call site covers.
    """
    src = APP_JS.read_text()
    init = re.search(r"function startPolling\(\) \{.*?\n\}\n", src, re.S)
    assert init and "_renderNotifPermission()" in init.group(0), \
        "startPolling() does not paint the button on load"
    tab = re.search(r"function activateTab\(name, rest\) \{.*?\n\}\n", src, re.S)
    assert tab and "_renderNotifPermission()" in tab.group(0), \
        "activateTab() does not repaint the button when Settings is opened"


def test_the_painters_dependencies_are_hoisted():
    """Everything the painter needs survives being called from above its own definition.

    `activateTab` sits at the top of web/app.js and paints this button when
    Settings is opened; `initTabFromHash()` calls it during top-level execution,
    because the URL hash is the source of truth for the current tab. So opening
    or reloading the dashboard on `#settings` reaches the painter hundreds of
    lines before the notification region is evaluated.

    Function declarations are hoisted and initialised; `const` and `let` are
    hoisted into a temporal dead zone and throw. That throw is not contained:
    it propagates out of `initTabFromHash()` at the top level and aborts the
    rest of app.js, so `startPolling()` never runs and the whole dashboard
    stops polling. Measured while building this fix, with the table written as
    a `const`: "Uncaught ReferenceError: Cannot access '_NOTIF_PERMISSION_UI'
    before initialization", three times per load.
    """
    src = APP_JS.read_text()
    assert "_renderNotifPermission()" in src, (
        "web/app.js never calls _renderNotifPermission(), so nothing paints the "
        "button — see the call-site test above")
    first_call = src.index("_renderNotifPermission()")
    for name in ("_notifPermissionUi", "_notifPermissionState", "_renderNotifPermission"):
        decl = re.search(r"^(?:const|let|var|function)\s+%s\b" % name, src, re.M)
        assert decl, f"{name} is not declared at the top level of web/app.js"
        if decl.start() > first_call:
            assert decl.group(0).startswith("function"), (
                f"{name} is declared with `{decl.group(0).split()[0]}` at line "
                f"{src[:decl.start()].count(chr(10)) + 1}, after it is first called at "
                f"line {src[:first_call].count(chr(10)) + 1} — a reload on #settings "
                f"would throw out of top-level execution and stop startPolling()")


# ── Layer 2: no id read in JS is missing from the markup (family 17) ─────────

def test_every_element_id_read_in_js_exists():
    """The set-difference scan that found #98, as a standing guard.

    A `getElementById` miss is a silent no-op, so nothing in the running app
    reports it. Two definition forms have to count or this reports healthy code
    as broken: an id assigned in code (`el.id = "x"`) and one built in a
    template literal (`id="${...}"`), the latter excluded from the reference
    side and checked by reading the builder against its reader.
    """
    js, html = APP_JS.read_text(), INDEX_HTML.read_text()
    refs = {}
    patterns = (
        r"""getElementById\(\s*["']([^"'$]+)["']\s*\)""",
        r"""(?<![\w.])\$\(\s*["']([^"'$]+)["']\s*\)""",
        r"""querySelector(?:All)?\(\s*["']#([A-Za-z0-9_-]+)["']""",
    )
    for pattern in patterns:
        for m in re.finditer(pattern, js):
            refs.setdefault(m.group(1), []).append(js[:m.start()].count("\n") + 1)

    defined = set()
    for src in (html, js):
        defined |= set(re.findall(r"""\bid=["']([^"'${]+)["']""", src))
        defined |= set(re.findall(r"""\.id\s*=\s*["']([^"']+)["']""", src))
        defined |= set(re.findall(
            r"""setAttribute\(\s*["']id["']\s*,\s*["']([^"']+)["']""", src))

    missing = {k: v for k, v in refs.items() if k not in defined}
    assert not missing, "ids read in web/app.js that no element carries: " + "; ".join(
        f"{k!r} at line(s) {v}" for k, v in sorted(missing.items()))
    assert len(refs) > 30, f"scan only found {len(refs)} ids — patterns stopped matching"


# ── Layer 3: the five states, driven under node ─────────────────────────────

def _run_state_machine(scenarios):
    """Lift the DOM-free state machine out of web/app.js and run it under node."""
    src = APP_JS.read_text()
    start = src.index("function _notifPermissionState()")
    end = src.index("function _renderNotifPermission()")
    lifted = src[start:end]

    harness = """
const SCENARIOS = JSON.parse(process.env.VCHECK_SCENARIOS);
const out = [];
for (const s of SCENARIOS) {
  globalThis.window = globalThis;
  globalThis.isSecureContext = s.secure;
  if (s.supported) { globalThis.Notification = { permission: s.permission }; }
  else { delete globalThis.Notification; }
  const run = new Function(%s + "\\n; return {state: _notifPermissionState(), ui: _notifPermissionUi()};");
  const r = run();
  const entry = r.ui[r.state];
  out.push({name: s.name, state: r.state, label: entry.label,
            offer: !!entry.offer, title: entry.title,
            offering_states: Object.keys(r.ui).filter(k => r.ui[k].offer)});
}
console.log(JSON.stringify(out));
""" % json.dumps(lifted)

    proc = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60,
        env={**os.environ, "VCHECK_SCENARIOS": json.dumps(scenarios)})
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return {r["name"]: r for r in json.loads(proc.stdout)}


@pytest.fixture(scope="module")
def states():
    if not shutil.which("node"):
        pytest.skip("node is not installed; layers 1 and 2 still cover the ids")
    return _run_state_machine([
        {"name": "localhost",       "supported": True, "secure": True,  "permission": "default"},
        {"name": "lan_http",        "supported": True, "secure": False, "permission": "denied"},
        {"name": "secure_blocked",  "supported": True, "secure": True,  "permission": "denied"},
        {"name": "granted",         "supported": True, "secure": True,  "permission": "granted"},
        {"name": "granted_insecure","supported": True, "secure": False, "permission": "granted"},
        {"name": "unsupported",     "supported": False,"secure": True,  "permission": "default"},
    ])


def test_only_a_grantable_permission_offers_a_click(states):
    """Exactly one state invites a click, and it is the one a prompt can change."""
    assert states["localhost"]["offering_states"] == ["default"]
    assert states["localhost"]["state"] == "default"
    assert states["localhost"]["offer"] is True
    assert states["localhost"]["label"] == "Enable browser notifications"


def test_an_insecure_origin_says_so_rather_than_offering_a_dead_click(states):
    """The LAN case, which is how the dashboard is actually reached.

    `permission` already reads `"denied"` there with no prompt available, so a
    button keyed on `permission !== "granted"` would offer a control that
    cannot succeed. Keying on `isSecureContext` names the real obstacle.
    """
    lan = states["lan_http"]
    assert lan["state"] == "insecure"
    assert lan["offer"] is False
    assert "https" in lan["label"] and "localhost" in lan["label"]
    assert lan["label"] != states["secure_blocked"]["label"], (
        "a blocked origin and a refused prompt need different labels — the "
        "operator can fix one of them in browser settings and not the other")


def test_a_refused_prompt_points_at_browser_settings(states):
    blocked = states["secure_blocked"]
    assert blocked["state"] == "denied"
    assert blocked["offer"] is False
    assert "settings" in blocked["title"].lower()


def test_a_granted_permission_reports_itself_whatever_the_origin(states):
    """`granted` outranks every other test.

    A dashboard behind an https proxy, or opened on the host itself, can hold
    the permission; reporting it as blocked because of a later test would
    contradict the notifications the operator is already receiving.
    """
    for key in ("granted", "granted_insecure"):
        assert states[key]["state"] == "granted", key
        assert states[key]["offer"] is False, key
        assert states[key]["label"] == "Browser notifications enabled", key


def test_an_unsupported_browser_is_stated_on_the_button(states):
    """The old code alerted here, from a guard that cannot fire in Chrome."""
    assert states["unsupported"]["state"] == "unsupported"
    assert states["unsupported"]["offer"] is False
    assert "unsupported" in states["unsupported"]["label"].lower()
