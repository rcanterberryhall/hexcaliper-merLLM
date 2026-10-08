"""tests/test_metrics_chart_range.py — the Metrics tab's charts cannot disagree
with the selected range (issue #84).

These assert structural properties of the three shipped web assets plus one
cross-file invariant against the history endpoint. The regressions they guard
are all shapes that reintroduce a chart holding data from a range the range bar
is no longer highlighting:

* a render reached without going through ``_renderChart``, which is the only
  place that resolves a card either way;
* ``_hist`` collapsing a failed request back into an empty list, which makes a
  fault indistinguishable from a range that genuinely holds no points;
* a load rendering without first checking it has not been superseded;
* a range offered in the markup that the endpoint does not know, which it would
  silently answer with 1 h of data.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


APP_JS = _read("web/app.js")
INDEX_HTML = _read("web/index.html")
STYLES_CSS = _read("web/styles.css")
APP_PY = _read("api/app.py")


def _function_body(src, name):
    """Return one top-level function body, declaration line to closing brace.

    Bodies are bracketed by a declaration at column 0 and the next ``}`` at
    column 0, which is more robust through this file's template literals than
    counting braces.
    """
    lines = src.split("\n")
    start = next(
        i for i, line in enumerate(lines)
        if re.match(r"^(async )?function %s\(" % re.escape(name), line)
    )
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("}"))
    return "\n".join(lines[start:end + 1])


# ── Every card is resolved through the one function that can blank it ─────────

def test_every_chart_canvas_in_the_markup_is_known_to_the_loader():
    """A canvas nothing in app.js names would stay blank with no explanation."""
    ids = re.findall(r'<canvas id="(chart-[a-z0-9]+)"></canvas>', INDEX_HTML)
    assert len(ids) == 6, ids
    for chart_id in ids:
        # chart-gpu0 / chart-gpu1 are built as `chart-gpu${idx}` in a loop.
        literal = '"%s"' % chart_id
        templated = '`chart-%s${idx}`' % re.sub(r"\d+$", "", chart_id[len("chart-"):])
        assert literal in APP_JS or templated in APP_JS, chart_id


def test_setchart_is_only_reached_through_renderchart():
    """``_renderChart`` is what guarantees a card is drawn *or* blanked.

    A ``_setChart`` call anywhere else can skip a card and leave the previous
    range's curve standing, which is the defect in #84.
    """
    call_sites = [
        line for line in APP_JS.split("\n")
        if "_setChart(" in line and not line.startswith("function _setChart(")
    ]
    assert len(call_sites) == 1, call_sites
    assert "_setChart(" in _function_body(APP_JS, "_renderChart")
    assert "_setChart(" not in _function_body(APP_JS, "loadMetricsCharts")


def test_renderchart_blanks_a_card_it_cannot_draw():
    body = _function_body(APP_JS, "_renderChart")
    assert body.count("_clearChart(") == 3, body
    # A failed series, an empty range, and a throw from the build must each end
    # in a blanked card rather than in a skipped render.
    assert "s.ok" in body
    assert "s.points.length" in body
    assert "catch" in body


def test_clearchart_destroys_the_instance_rather_than_hiding_it():
    """Leaving the Chart in ``_charts`` keeps the old curve on the canvas."""
    body = _function_body(APP_JS, "_clearChart")
    assert ".destroy()" in body
    assert "delete _charts[id]" in body


def test_setchart_removes_a_previous_blank_reason():
    assert '_setChartNote(canvas, "")' in _function_body(APP_JS, "_setChart")


# ── A failed request is not an empty range ───────────────────────────────────

def test_hist_distinguishes_a_failed_request_from_an_empty_range():
    body = _function_body(APP_JS, "_hist")
    assert "ok: true" in body
    assert "ok: false" in body
    assert "metric" in body, "the reported failure has to name its metric"
    assert not re.search(r"return \[\];", body), \
        "returning a bare [] on failure is what made a fault look like no data"


# ── A superseded load must not render ────────────────────────────────────────

def test_a_load_checks_it_has_not_been_overtaken_before_rendering():
    body = _function_body(APP_JS, "loadMetricsCharts")
    assert "++_metricsLoadSeq" in body
    assert re.search(r"if \(seq !== _metricsLoadSeq\) return;", body)
    # The check has to sit between the awaited fetches and the first render, or
    # a slow 7 d load still overwrites a later 1 h one.
    assert body.index("await Promise.all") < body.index("_metricsLoadSeq) return")
    assert body.index("_metricsLoadSeq) return") < body.index("_renderChart(")


def test_every_request_of_one_load_uses_the_range_captured_at_its_start():
    body = _function_body(APP_JS, "loadMetricsCharts")
    assert "const range = _metricsRange;" in body
    hist_calls = re.findall(r"_hist\((.*?)\)", body)
    assert len(hist_calls) == 12, hist_calls
    for call in hist_calls:
        assert call.endswith("range"), call
    assert "_metricsRange" not in _function_body(APP_JS, "_hist"), \
        "_hist must take the range, not re-read a global that may have moved"


# ── The blank reason has somewhere to render ──────────────────────────────────

def test_the_blank_reason_has_a_rule_and_a_positioning_context():
    assert ".chart-note" in STYLES_CSS
    note_rule = STYLES_CSS[STYLES_CSS.index(".chart-note"):]
    note_rule = note_rule[:note_rule.index("}")]
    assert "position: absolute" in note_rule
    wrap_rule = STYLES_CSS[STYLES_CSS.index(".chart-wrap"):]
    wrap_rule = wrap_rule[:wrap_rule.index("}")]
    assert "position: relative" in wrap_rule, \
        "an absolutely positioned note needs .chart-wrap as its containing block"
    assert 'className = "chart-note muted"' in _function_body(APP_JS, "_setChartNote")


# ── The ranges offered and the ranges understood are the same set ─────────────

def test_the_range_buttons_match_the_ranges_the_endpoint_knows():
    """``ranges.get(range, 3600)`` answers an unknown range with 1 h of data.

    So a button the endpoint does not recognise is the same defect by another
    route: a 1 h curve under another range's label, with no error anywhere.
    """
    offered = re.findall(r'class="range-btn[^"]*" data-range="([^"]+)"', INDEX_HTML)
    assert offered, "no range buttons found in the markup"
    declared = re.search(
        r'ranges = \{(.*?)\}', APP_PY, re.S).group(1)
    understood = re.findall(r'"([^"]+)":', declared)
    assert set(offered) == set(understood), (offered, understood)
