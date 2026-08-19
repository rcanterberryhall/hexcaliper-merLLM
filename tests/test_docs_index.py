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


def test_title_comes_from_first_h1(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    index = di.build_index()
    meta = index["havelock-plans/2026-07-06-plan-56-parser-extract-build.md"]
    assert meta["title"] == "Parser Extract/Build Decoupling — Plan 56"
    assert meta["app"] == "havelock"
    assert meta["dir"] == "havelock-plans"
    assert meta["type"] == "plan"
    assert meta["plan_no"] == 56


def test_title_falls_back_when_no_h1(monkeypatch, plans):
    (plans / "havelock-plans" / "2026-02-02-no-heading-design.md").write_text(
        "Just prose, no heading at all.\n", encoding="utf-8")
    di = _mod(monkeypatch, plans)
    index = di.build_index()
    assert index["havelock-plans/2026-02-02-no-heading-design.md"]["title"] == "No heading"


def test_title_ignores_hash_inside_code_fence(monkeypatch, plans):
    (plans / "havelock-plans" / "2026-03-03-fenced-design.md").write_text(
        "```bash\n# not the title\n```\n\n# Real Title\n", encoding="utf-8")
    di = _mod(monkeypatch, plans)
    index = di.build_index()
    assert index["havelock-plans/2026-03-03-fenced-design.md"]["title"] == "Real Title"


def test_get_document_returns_markdown(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    doc = di.get_document("soonstone-plans/soonstone_roadmap.md")
    assert doc["markdown"] == "# Soonstone Roadmap\n"
    assert doc["app"] == "soonstone"


@pytest.mark.parametrize("rel", [
    "../../etc/passwd",
    "havelock-plans/../../etc/passwd",
    "havelock-plans/does-not-exist.md",
    "hexcaliper-havelock/README.md",
])
def test_get_document_rejects_unindexed_paths(monkeypatch, plans, rel):
    di = _mod(monkeypatch, plans)
    assert di.get_document(rel) is None


def test_cache_avoids_rereading_unchanged_files(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    di.build_index()
    calls = []
    real_read_title = di.read_title
    monkeypatch.setattr(di, "read_title",
                        lambda p, s: (calls.append(p), real_read_title(p, s))[1])
    di.build_index()
    assert calls == []


def test_cache_refreshes_changed_file(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    di.build_index()
    target = plans / "soonstone-plans" / "soonstone_roadmap.md"
    target.write_text("# Renamed Roadmap\n", encoding="utf-8")
    os.utime(target, (target.stat().st_atime + 10, target.stat().st_mtime + 10))
    index = di.build_index()
    assert index["soonstone-plans/soonstone_roadmap.md"]["title"] == "Renamed Roadmap"


def test_deleted_file_leaves_the_index(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    assert "soonstone-plans/soonstone_roadmap.md" in di.build_index()
    (plans / "soonstone-plans" / "soonstone_roadmap.md").unlink()
    assert "soonstone-plans/soonstone_roadmap.md" not in di.build_index()


def test_as_library_orders_apps_by_count_and_docs_newest_first(monkeypatch, plans):
    di = _mod(monkeypatch, plans)
    library = di.as_library(di.build_index())
    assert [a["app"] for a in library][0] == "havelock"      # 2 docs, the most
    assert library[0]["count"] == 2
    dates = [d["date"] for d in library[0]["docs"]]
    assert dates == ["2026-07-06", "2026-07-05"]             # newest first
    soonstone = next(a for a in library if a["app"] == "soonstone")
    assert soonstone["docs"][0]["date"] is None              # undated docs still listed


def test_as_library_sorts_undated_last(monkeypatch, plans):
    (plans / "soonstone-plans" / "2026-09-09-late-design.md").write_text(
        "# Late\n", encoding="utf-8")
    di = _mod(monkeypatch, plans)
    library = di.as_library(di.build_index())
    soonstone = next(a for a in library if a["app"] == "soonstone")
    assert [d["date"] for d in soonstone["docs"]] == ["2026-09-09", None]


def test_build_index_returns_copies_not_cached_objects(monkeypatch, plans):
    """Finding 1: mutation of returned dict must not poison the cache."""
    di = _mod(monkeypatch, plans)
    idx1 = di.build_index()
    key = "havelock-plans/2026-07-06-plan-56-parser-extract-build.md"
    original_title = idx1[key]["title"]

    # Mutate the returned dict.
    idx1[key]["title"] = "HACKED"

    # The cache should be unaffected.
    idx2 = di.build_index()
    assert idx2[key]["title"] == original_title


def test_cache_keys_by_root_not_just_path(monkeypatch, plans, tmp_path):
    """Finding 2: nested roots must not purge each other's cache entries."""
    # Create a nested structure: parent contains child-plans.
    child_root = plans / "child"
    child_root.mkdir()
    (child_root / "child-plans").mkdir()
    (child_root / "child-plans" / "2026-01-01-test-design.md").write_text(
        "# Child Doc\n", encoding="utf-8")

    di = _mod(monkeypatch, child_root)
    idx_child = di.build_index()
    assert "child-plans/2026-01-01-test-design.md" in idx_child

    # Now build with parent root.
    di2 = _mod(monkeypatch, plans)
    idx_parent = di2.build_index()

    # Child's cache should not be purged.
    di3 = _mod(monkeypatch, child_root)
    idx_child_again = di3.build_index()
    assert "child-plans/2026-01-01-test-design.md" in idx_child_again


def test_title_respects_tilde_fences(monkeypatch, plans):
    """Finding 3a: ~~~ fences are recognized like ``` fences."""
    (plans / "havelock-plans" / "2026-03-03-tilde-fenced-design.md").write_text(
        "~~~bash\n# not the title\n~~~\n\n# Real Title\n", encoding="utf-8")
    di = _mod(monkeypatch, plans)
    index = di.build_index()
    assert index["havelock-plans/2026-03-03-tilde-fenced-design.md"]["title"] == "Real Title"


def test_title_tilde_inside_backtick_fence_does_not_close(monkeypatch, plans):
    """Finding 3b: ~~~ inside ``` does not close the ``` fence."""
    (plans / "havelock-plans" / "2026-03-03-mixed-fences-design.md").write_text(
        "```\n~~~\n# not the title\n~~~\n```\n\n# Real Title\n", encoding="utf-8")
    di = _mod(monkeypatch, plans)
    index = di.build_index()
    assert index["havelock-plans/2026-03-03-mixed-fences-design.md"]["title"] == "Real Title"


def test_title_truncated_line_at_boundary_falls_back(monkeypatch, plans):
    """Finding 4: H1 straddling the 4096-byte boundary is discarded, not truncated."""
    # Create a file where line 1 fills most of 4096 bytes, and line 2 is an H1 that spans the cutoff.
    # We want the H1 to be truncated by the read boundary.
    di = _mod(monkeypatch, plans)
    # Create a file with ~4095 bytes of content before an H1 that will be cut off.
    prefix = "x" * 4090  # leaves ~6 bytes in the 4096-byte window
    h1_line = "# This is a very long heading that will be cut off at the boundary"
    content = prefix + "\n" + h1_line + "\n\n# Real Fallback Title\n"

    (plans / "havelock-plans" / "2026-03-03-boundary-design.md").write_text(
        content, encoding="utf-8")
    index = di.build_index()
    # The truncated H1 should not be used; should fall back.
    title = index["havelock-plans/2026-03-03-boundary-design.md"]["title"]
    # The fallback will be the de-slugged filename, not the truncated H1.
    assert title == "Boundary"
    assert not title.startswith("# This is a very long")
