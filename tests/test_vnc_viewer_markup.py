"""tests/test_vnc_viewer_markup.py — VNC tab: the viewer is the pane's only child.

Guards the fix for issue #82, which requires that ``connectVNC`` hand noVNC an
*empty* ``#vnc-container`` instead of a canvas of its own, and that the sizing
rule written for that hand-made canvas name an element noVNC actually uses.

noVNC v1.5.0 — pinned in ``nginx/Dockerfile`` and read off the copy the nginx
image serves — does not adopt a canvas supplied by the caller. Its constructor
builds both elements and appends them to the element it was given
(``core/rfb.js``: ``this._screen.appendChild(this._canvas)``, then
``this._target.appendChild(this._screen)`` while still constructing). So a
canvas placed in the container beforehand is never drawn into, and since
``#vnc-container`` is ``display: flex`` it survives as a second flex item: it
takes its intrinsic 300×150, pushes the viewer off centre, and shrinks the
``clientWidth`` that noVNC's ``autoscale()`` scales the remote desktop into.

The assignment still has one job — emptying the container is how a reconnect
drops the previous session's screen, since noVNC removes it only on a clean
``disconnect()`` — so these assertions check that the clear survived and only
the canvas went.

The assertions read the shipped ``web/`` assets as text. CI checks out the whole
repository and runs pytest from the root, so the assets are present here and in
CI.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


@pytest.fixture(scope="module")
def app_js():
    return (WEB / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def styles_css():
    return (WEB / "styles.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html():
    return (WEB / "index.html").read_text(encoding="utf-8")


def _block(source: str, opener: str) -> str:
    """Return the brace-balanced block beginning at ``opener``."""
    assert opener in source, f"{opener!r} not found"
    start = source.index(opener)
    depth = 0
    for pos in range(source.index("{", start), len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise AssertionError(f"unbalanced braces after {opener!r}")


def test_the_pane_lays_its_children_out_side_by_side(styles_css):
    """The premise: a second child of the pane is not hidden, it is laid out."""
    pane = _block(styles_css, "#vnc-container {")
    assert "display: flex" in pane, (
        "this file's other assertions exist because #vnc-container is a flex "
        "container, where anything left in it takes width from the viewer; if "
        "the layout changed, re-derive what a stray child now costs"
    )


def test_connect_hands_novnc_an_empty_container(app_js):
    """noVNC builds its own surface; a supplied one is never drawn into."""
    connect = _block(app_js, "async function connectVNC(")
    assert "<canvas" not in connect, (
        "connectVNC() must not put a canvas in #vnc-container: noVNC appends "
        "its own screen and canvas to the target, so this one is never drawn "
        "into and sits beside the real viewer as a second flex item"
    )
    assert 'container.innerHTML = ""' in connect, (
        "the container must still be emptied before the session is built — "
        "that is what drops a previous session's screen on reconnect, which "
        "noVNC itself only does on a clean disconnect()"
    )
    assert connect.index('container.innerHTML = ""') < connect.index("new RFB("), (
        "clear the pane before constructing the session, not after: noVNC "
        "appends its screen while still in the constructor"
    )


def test_the_hand_made_canvas_is_gone_from_every_shipped_asset(
    app_js, index_html, styles_css
):
    """An id nothing creates is a rule and a selector pointing at nothing."""
    for name, source in (
        ("web/app.js", app_js),
        ("web/index.html", index_html),
        ("web/styles.css", styles_css),
    ):
        assert "vnc-canvas" not in source, (
            f"{name} still references #vnc-canvas, an element noVNC does not "
            f"use; the element and the rule written for it go together"
        )


def test_the_viewer_sizing_rule_names_the_canvas_novnc_creates(styles_css):
    """The clamp has to land on the element that holds the picture."""
    rule = _block(styles_css, "#vnc-container canvas {")
    for prop in ("max-width: 100%", "max-height: 100%"):
        assert prop in rule, (
            f"the viewer canvas needs {prop} so the picture stays inside the "
            f"600px pane; the rule this replaces named #vnc-canvas, which "
            f"noVNC never creates, so it constrained nothing"
        )


def test_the_novnc_version_these_claims_were_read_from_is_still_pinned():
    """The fix rests on how this version of noVNC builds its DOM."""
    dockerfile = (ROOT / "nginx" / "Dockerfile").read_text(encoding="utf-8")
    assert "--branch v1.5.0" in dockerfile, (
        "noVNC moved off v1.5.0, whose constructor creates its own screen and "
        "canvas and appends them to the target element — re-read core/rfb.js "
        "and confirm the new version still does before changing this test"
    )
