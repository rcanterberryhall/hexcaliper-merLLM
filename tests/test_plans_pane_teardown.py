"""tests/test_plans_pane_teardown.py — Plans reader: one teardown for both paths.

Guards the fix for issue #78, which requires that a failed document open leave
no state behind from the document previously displayed: the reading pane kept
the old contents rail, step tally, scroll-spy and raw-mode flags because
``openDoc``'s failure path hand-rolled a partial copy of ``resetPlansView``.
Both paths now delegate their clearing to ``clearPlansPane``, so a pane element
cannot be cleared on one path and forgotten on the other.

These assertions read the shipped ``web/`` assets as text. CI checks out the
whole repository and runs pytest from the root, so both files are present here
and in CI.
"""
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"

# Elements of the reading pane holding state derived from the open document.
# The title and meta are deliberately excluded: each path writes its own text,
# which is why the teardown is a helper rather than a shared call to
# resetPlansView (that would replace the error message with "Select a
# document" and lose the report of what failed).
PANE_STATE_IDS = (
    "plans-progress",     # "N of M steps done" tally for the open document
    "plans-raw-toggle",   # reads "Rendered" only while raw source is displayed
    "plans-body",         # the rendered document, and its "raw" class
    "plans-toc",          # contents rail, whose links target headings in the body
)


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


def _open_doc_catch(app_js: str) -> str:
    """The failure path of openDoc: its ``catch`` block."""
    return _block(_block(app_js, "async function openDoc("), "} catch (err) {")


def test_pane_state_elements_exist_in_markup(index_html):
    """Every element the teardown clears is really in the shipped markup."""
    missing = [i for i in PANE_STATE_IDS if f'id="{i}"' not in index_html]
    assert not missing, f"web/index.html has no element for {missing}"


def test_clear_plans_pane_covers_every_pane_element(app_js):
    """The helper clears all document-derived pane state, including the spy."""
    helper = _block(app_js, "function clearPlansPane(")
    uncleared = [i for i in PANE_STATE_IDS if i not in helper]
    assert not uncleared, (
        f"clearPlansPane() never touches {uncleared}; a document's state would "
        f"survive into the next open"
    )
    assert "_plansSpy" in helper and "disconnect()" in helper, (
        "clearPlansPane() must disconnect the scroll-spy, or it keeps observing "
        "headings that were detached with the body"
    )
    assert "_plansOpen = null" in helper, (
        "clearPlansPane() must clear _plansOpen so a retry click on the same "
        "sidebar row fires openDoc again"
    )


def test_failed_open_delegates_its_clearing_to_the_helper(app_js):
    """openDoc's failure path must not hand-roll a partial teardown again."""
    catch = _open_doc_catch(app_js)
    assert "clearPlansPane()" in catch, (
        "openDoc's catch block does not call clearPlansPane(); this is the "
        "defect in issue #78, where it cleared the body but left the contents "
        "rail, step tally, scroll-spy and raw-mode state of the previous "
        "document on screen"
    )
    for element_id in PANE_STATE_IDS:
        assert element_id not in catch, (
            f"openDoc's catch block touches {element_id} directly; clearing "
            f"pane state in two places is what let the two paths drift apart"
        )


def test_failed_open_still_reports_what_failed(app_js):
    """Delegating the teardown must not cost the error message."""
    catch = _open_doc_catch(app_js)
    assert "plans-title" in catch and "plans-meta" in catch, (
        "openDoc's catch block must still name the document and the error; "
        "calling resetPlansView() instead would overwrite both with the "
        "'Select a document' empty state"
    )


def test_reset_plans_view_delegates_to_the_same_helper(app_js):
    """The deliberate reset shares the teardown, so the paths cannot drift."""
    reset = _block(app_js, "function resetPlansView(")
    assert "clearPlansPane()" in reset, (
        "resetPlansView() must clear the pane through clearPlansPane() so that "
        "any element added to the reading pane is cleared on both paths"
    )
