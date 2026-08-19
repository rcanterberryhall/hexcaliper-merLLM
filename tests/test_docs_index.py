"""tests/test_docs_index.py — planning-docs index: parsing, discovery, cache, API."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


@pytest.fixture
def plans(tmp_path):
    """Build a fixture workspace mirroring the real *-plans layout."""
    def write(rel, text="# Fixture\n"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    write("havelock-plans/2026-07-06-plan-56-parser-extract-build.md",
          "# Parser Extract/Build Decoupling — Plan 56\n")
    write("havelock-plans/2026-07-05-landing-explainer-design.md",
          "# Landing Explainer\n")
    write("scrivener-plans/2026-08-10-concurrent-editing-requirements.md",
          "# Concurrent Editing\n")
    write("soonstone-plans/soonstone_roadmap.md", "# Soonstone Roadmap\n")
    write("parsival-plans/nested/deeper/2026-01-01-nested-design.md", "# Nested\n")
    # Must NOT be indexed:
    write("hexcaliper-havelock/README.md", "# Repo readme\n")   # not a *-plans dir
    (tmp_path / "revelmaster-plans").mkdir()
    (tmp_path / "revelmaster-plans" / "latency-baseline.py").write_text("x = 1\n")
    return tmp_path


def _mod(monkeypatch, root):
    """Import docs_index with PLANS_ROOT pointed at the fixture tree."""
    for name in ("docs_index", "config"):
        sys.modules.pop(name, None)
    monkeypatch.setenv("PLANS_ROOT", str(root))
    import docs_index
    docs_index.reset_cache()
    return docs_index


@pytest.mark.parametrize("stem,expected", [
    ("2026-07-06-plan-56-parser-extract-build",
     {"date": "2026-07-06", "type": "plan", "plan_no": 56}),
    ("2026-07-05-landing-explainer-design",
     {"date": "2026-07-05", "type": "design", "plan_no": None}),
    ("2026-08-10-concurrent-editing-requirements",
     {"date": "2026-08-10", "type": "requirements", "plan_no": None}),
    ("soonstone_roadmap",
     {"date": None, "type": "other", "plan_no": None}),
    # Precedence: plan is checked before requirements.
    ("2026-01-01-plan-3-requirements-sheet",
     {"date": "2026-01-01", "type": "plan", "plan_no": 3}),
])
def test_parse_filename(monkeypatch, plans, stem, expected):
    di = _mod(monkeypatch, plans)
    assert di.parse_filename(stem) == expected


def test_fallback_title_strips_date_plan_and_suffix(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    assert di.fallback_title("2026-07-06-plan-56-parser-extract-build") == "Parser extract build"
    assert di.fallback_title("2026-07-05-landing-explainer-design") == "Landing explainer"
    assert di.fallback_title("soonstone_roadmap") == "Soonstone roadmap"


def test_discover_only_plans_dirs_and_md(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    rels = sorted(r for r, _d, _p, _s in di.discover())
    assert "havelock-plans/2026-07-06-plan-56-parser-extract-build.md" in rels
    assert "parsival-plans/nested/deeper/2026-01-01-nested-design.md" in rels
    # A sibling repo directory is invisible.
    assert not any(r.startswith("hexcaliper-havelock") for r in rels)
    # Non-Markdown files are excluded.
    assert not any(r.endswith(".py") for r in rels)
    assert len(rels) == 5


def test_discover_excludes_oversized_files(monkeypatch, plans):
    (plans / "havelock-plans" / "huge.md").write_text("#" + "x" * 5000)
    monkeypatch.setenv("PLANS_MAX_BYTES", "1024")
    di = _mod(monkeypatch, plans)
    assert not any(r.endswith("huge.md") for r, _d, _p, _s in di.discover())


def test_discover_skips_escaping_symlink(monkeypatch, plans, tmp_path):
    outside = tmp_path.parent / "outside-secret.md"
    outside.write_text("# Secret\n")
    os.symlink(outside, plans / "havelock-plans" / "leak.md")
    di = _mod(monkeypatch, plans)
    assert not any(r.endswith("leak.md") for r, _d, _p, _s in di.discover())


def test_discover_raises_when_root_missing(monkeypatch, tmp_path):
    di = _mod(monkeypatch, tmp_path / "does-not-exist")
    with pytest.raises(di.PlansUnavailable):
        di.discover()
