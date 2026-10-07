"""tests/test_vnc_session_teardown.py — VNC tab: Disconnect closes the session.

Guards the fix for issue #80, which requires that the VNC tab hold at most one
live session and that **Disconnect** close it rather than only repaint the
viewer pane. Before the fix ``disconnectVNC`` assigned a message to
``#vnc-container`` and nothing else: the browser WebSocket, and the API's TCP
connection to the VNC server behind it, stayed open; each further **Connect**
stacked another session; and every stacked session kept a ``disconnect``
listener that would later repaint the pane over whichever session was live.

Clearing the pane cannot close the session, because noVNC appends its own
canvas to the target element and never checks whether that target is still
attached — so the teardown has to be an explicit ``RFB.disconnect()`` through a
retained handle. These assertions check that the handle exists, that both entry
points go through the teardown, and that the teardown happens before a
replacement session is constructed.

The assertions read the shipped ``web/`` assets as text. CI checks out the
whole repository and runs pytest from the root, so both files are present here
and in CI.
"""
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture(scope="module")
def app_js():
    return (WEB / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html():
    return (WEB / "index.html").read_text(encoding="utf-8")


def _block(source: str, opener: str) -> str:
    """Return the brace-balanced block beginning at ``opener``."""
    assert opener in source, f"{opener!r} not found in web/app.js"
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


def test_both_buttons_are_wired_to_the_two_entry_points(index_html):
    """The markup still reaches the pair of functions these tests constrain."""
    assert 'onclick="connectVNC()"' in index_html
    assert 'onclick="disconnectVNC()"' in index_html
    assert 'id="vnc-container"' in index_html


def test_the_live_session_is_held_in_a_module_level_handle(app_js):
    """A session built into a local const cannot be closed by anything later."""
    assert "let _vncRfb = null;" in app_js, (
        "the VNC tab must retain the live RFB, or Disconnect has nothing to "
        "close and Connect cannot tell that a session is already running"
    )
    for dead in ("_vncWs", "_vncSocket"):
        assert dead not in app_js, (
            f"{dead} was declared and never assigned or read; a handle that is "
            f"not the real one invites the same bug back"
        )


def test_teardown_closes_the_connection_after_clearing_the_handle(app_js):
    """The handle is cleared first so the resulting event reads as stale."""
    teardown = _block(app_js, "function _vncTeardown(")
    assert "rfb.disconnect()" in teardown, (
        "_vncTeardown() must call RFB.disconnect(); repainting #vnc-container "
        "leaves the WebSocket and the api→VNC connection open"
    )
    assert teardown.index("_vncRfb = null") < teardown.index("rfb.disconnect()"), (
        "_vncRfb must be cleared before disconnect() is called, or the "
        "'disconnect' event that call dispatches repaints the pane and "
        "overwrites the message Disconnect is about to write"
    )


def test_disconnect_button_delegates_to_the_teardown(app_js):
    """Disconnect must close the session, not only report that it is closed."""
    disconnect = _block(app_js, "function disconnectVNC(")
    assert "_vncTeardown()" in disconnect, (
        "disconnectVNC() only repaints the pane; the session stays live and "
        "the operator is told the remote desktop is disconnected when it is not"
    )
    assert disconnect.index("_vncTeardown()") < disconnect.index("innerHTML"), (
        "tear the session down before repainting, so noVNC removes its own "
        "canvas from a container that is still attached"
    )


def test_connect_tears_down_before_building_a_replacement(app_js):
    """Two Connects must not leave two sessions streaming at once."""
    connect = _block(app_js, "async function connectVNC(")
    teardowns = connect.count("_vncTeardown()")
    assert teardowns >= 2, (
        "connectVNC() must tear down on entry and again after awaiting the "
        f"noVNC import (found {teardowns}); a second click during the import "
        "otherwise installs a session the first click then overwrites"
    )
    assert connect.rindex("_vncTeardown()") < connect.index("new RFB("), (
        "the last teardown must precede the new RFB, or the replacement "
        "session is the one that gets closed"
    )


def test_a_replaced_session_cannot_repaint_the_pane(app_js):
    """A stale session's late drop must not detach a live viewer's canvas."""
    connect = _block(app_js, "async function connectVNC(")
    listener = _block(connect, 'rfb.addEventListener("disconnect"')
    assert "_vncRfb !== rfb" in listener, (
        "the disconnect listener must check that its own session is still the "
        "current one; otherwise an abandoned session dropping later paints "
        "'VNC disconnected.' over the canvas of the session that is connected"
    )
    assert listener.index("return;") < listener.index("innerHTML"), (
        "the staleness guard must return before the repaint, not after it"
    )
