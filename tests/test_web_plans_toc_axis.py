"""tests/test_web_plans_toc_axis.py — the Plans contents rail is a <nav>, and the
stylesheet styles the tab bar with a bare `nav` element selector, so every
property that rule declares reaches the rail too. These tests pin the relation
between the two rules rather than the text of either: a property added to the
tab bar's rule tomorrow is lent to the rail tomorrow, and must be answered."""
import os
import re

WEB = os.path.join(os.path.dirname(__file__), "..", "web")


def _read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as fh:
        return fh.read()


def _blank_comments(css):
    """Drop comment bodies but keep their newlines, so line numbers survive."""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), css, flags=re.S)


def _media_spans(css):
    """(start, end) of every @media block body, for tagging the rules inside."""
    spans = []
    for m in re.finditer(r"@media[^{]*\{", css):
        depth, i = 1, m.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        spans.append((m.start(), i))
    return spans


def _declarations(block):
    out = {}
    for part in block.split(";"):
        if ":" not in part:
            continue
        prop, _, value = part.partition(":")
        out[prop.strip().lower()] = value.strip()
    return out


def _rules(css):
    """Flat list of (selector, declarations, line, inside_media).

    `[^{}]+` cannot cross a brace, so an `@media` prelude never matches and the
    rules nested inside it are yielded on their own — which is what we want.
    """
    css = _blank_comments(css)
    spans = _media_spans(css)
    rules = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css, re.S):
        selector = " ".join(m.group(1).split())
        if selector.startswith("@"):
            continue
        rules.append((
            selector,
            _declarations(m.group(2)),
            css.count("\n", 0, m.start(1)) + 1,
            any(a <= m.start() < b for a, b in spans),
        ))
    return rules


_SIDES = ("top", "right", "bottom", "left")
_EXPANSIONS = {
    "padding": [f"padding-{s}" for s in _SIDES],
    "margin": [f"margin-{s}" for s in _SIDES],
    "gap": ["row-gap", "column-gap"],
    "background": ["background-color", "background-image", "background-repeat",
                   "background-position", "background-size", "background-attachment",
                   "background-origin", "background-clip"],
    "border": [f"border-{s}" for s in _SIDES],
    "flex-flow": ["flex-direction", "flex-wrap"],
    **{f"border-{s}": [f"border-{s}-width", f"border-{s}-style", f"border-{s}-color"]
       for s in _SIDES},
}


def _neutralises(decls, prop):
    """Does this rule override `prop` — directly, or across all its longhands?"""
    if prop in decls:
        return True
    longhands = _EXPANSIONS.get(prop)
    if not longhands:
        return False
    return all(_neutralises(decls, lh) for lh in longhands)


def _only(rules, selector, inside_media=False):
    found = [r for r in rules if r[0] == selector and r[3] is inside_media]
    assert len(found) == 1, f"expected exactly one `{selector}` rule, found {len(found)}"
    return found[0]


# ── the premise ────────────────────────────────────────────────────────────

def test_the_rail_shares_its_tag_with_the_tab_bar():
    """If the rail stops being a <nav>, the rest of this file is moot — say so here."""
    html = _read("index.html")
    navs = re.findall(r"<nav\b[^>]*>", html)
    assert len(navs) == 2, f"expected the tab bar and the rail, found {len(navs)}: {navs}"
    tab_bar, rail = navs
    assert "class=" not in tab_bar, f"the tab bar is matched by tag alone: {tab_bar}"
    assert "plans-toc" in rail, f"the second <nav> is the contents rail: {rail}"


def test_the_tab_bar_is_styled_by_a_bare_element_selector():
    """The control for the test below: it is this rule the rail has to answer."""
    rules = _rules(_read("styles.css"))
    selectors = [sel for sel, _, _, _ in rules]
    assert "nav" in selectors, "the tab bar is no longer styled by a bare `nav` rule"
    _, decls, _, _ = _only(rules, "nav")
    assert decls.get("display") == "flex", decls
    for lent in ("gap", "padding", "border-bottom", "background"):
        assert lent in decls, f"`nav` no longer declares {lent}: {decls}"


# ── the defect ─────────────────────────────────────────────────────────────

def test_the_rail_neutralises_every_property_the_nav_rule_lends_it():
    """A row axis, the bar's 2px gap, its 8px/20px padding, its surface fill and
    its bottom border all arrive at the rail unopposed otherwise. `display` is
    excluded deliberately — see the test two below for why it must not be
    re-declared, and the one below for what is declared in its place."""
    rules = _rules(_read("styles.css"))
    _, nav, _, _ = _only(rules, "nav")
    _, rail, _, _ = _only(rules, ".plans-toc")
    borrowed = [p for p in nav if p != "display" and not _neutralises(rail, p)]
    assert not borrowed, (
        f"`.plans-toc` inherits {borrowed} from the tab bar's `nav` rule; "
        f"rail declares {sorted(rail)}"
    )


def test_the_rail_declares_a_column_axis_rather_than_a_display():
    """The rail is a flex container by inheritance, so its axis is what decides
    where the links go — `.plans-toc-link { display: block }` cannot overrule
    its container. Declaring the axis also leaves `display` uncontested, which
    is what keeps the narrow-screen hide working."""
    _, rail, _, _ = _only(_rules(_read("styles.css")), ".plans-toc")
    assert rail.get("flex-direction") == "column", (
        f"the rail must set its own axis, not inherit the tab bar's row: {rail}")
    assert "display" not in rail, (
        "`.plans-toc` must not declare `display` — see the narrow-screen test")


def test_the_narrow_screen_hide_is_not_out_ordered_by_a_later_display():
    """@media carries no specificity bonus. `.plans-toc { display: none }` in the
    max-width:1100px block and a `display` on the same selector outside it are
    both (0,1,0), so the later line wins at every width — and the rail would
    reappear below the breakpoint, stacked under the document."""
    rules = _rules(_read("styles.css"))
    hides = [(ln, d) for sel, d, ln, media in rules
             if sel == ".plans-toc" and media and d.get("display") == "none"]
    assert len(hides) == 1, "expected one narrow-screen hide for the rail"
    hide_line, _ = hides[0]
    later = [(sel, ln) for sel, d, ln, media in rules
             if sel == ".plans-toc" and not media and "display" in d and ln > hide_line]
    assert not later, (
        f"`display` declared on `.plans-toc` at line {later[0][1]}, after the "
        f"narrow-screen hide at line {hide_line}: it wins at every width")


# ── no regression in what the rail already did ─────────────────────────────

def test_the_rail_keeps_its_left_border_and_its_12px_indent():
    _, rail, _, _ = _only(_rules(_read("styles.css")), ".plans-toc")
    assert "var(--border)" in rail.get("border-left", ""), rail
    padding = rail.get("padding", "").split()
    left = padding[3] if len(padding) == 4 else rail.get("padding-left", "")
    assert left == "12px", f"the rail's text should stay 12px clear of its rule: {rail}"


def test_the_links_still_truncate_rather_than_wrap():
    """Vertical links make the ellipsis rare, not wrong — a heading genuinely
    wider than the 220px track must still be cut, not wrapped to three lines."""
    _, link, _, _ = _only(_rules(_read("styles.css")), ".plans-toc-link")
    assert link.get("white-space") == "nowrap", link
    assert link.get("text-overflow") == "ellipsis", link
    assert link.get("overflow") == "hidden", link
