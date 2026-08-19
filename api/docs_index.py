"""
docs_index.py — index of the workspace planning and design documents.

Walks the ``<app>-plans`` directories under ``config.PLANS_ROOT`` (a read-only
bind mount of the workspace) and derives per-document metadata from the
filename convention plus the document's first H1.

Deliberately free of FastAPI imports so it can be unit-tested directly against
a temporary directory tree.
"""
import fnmatch
import os
import re
from typing import Optional

import config

# 2026-07-06-plan-56-parser-extract-build.md
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
_PLAN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-plan-(\d+)-")

_PLANS_SUFFIX = "-plans"


class PlansUnavailable(RuntimeError):
    """PLANS_ROOT is missing or unreadable — the bind mount is not in place."""


def parse_filename(stem: str) -> dict:
    """Derive date, document type and plan number from a filename stem.

    Type precedence is deliberate: ``plan`` is tested first so that a name like
    ``2026-01-01-plan-3-requirements-sheet`` classifies as the plan it is.
    """
    match_date = _DATE_RE.match(stem)
    date = match_date.group(1) if match_date else None

    match_plan = _PLAN_RE.match(stem)
    if match_plan:
        return {"date": date, "type": "plan", "plan_no": int(match_plan.group(1))}
    if "requirements" in stem:
        return {"date": date, "type": "requirements", "plan_no": None}
    if stem.endswith("-design"):
        return {"date": date, "type": "design", "plan_no": None}
    return {"date": date, "type": "other", "plan_no": None}


def fallback_title(stem: str) -> str:
    """De-slug a filename, for documents that have no H1 to borrow."""
    trimmed = _PLAN_RE.sub("", stem) if _PLAN_RE.match(stem) else _DATE_RE.sub("", stem)
    for suffix in ("-design", "-requirements"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
    trimmed = trimmed.replace("-", " ").replace("_", " ").strip()
    return (trimmed[:1].upper() + trimmed[1:]) if trimmed else stem


def discover(root: Optional[str] = None) -> list:
    """Return ``(rel, dir_name, abs_path, stat)`` for every indexable document.

    Only top-level directories matching ``PLANS_DIR_GLOB`` are walked, so the
    rest of the mounted workspace is never enumerated.
    """
    root = os.path.realpath(root or config.PLANS_ROOT)
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError as exc:
        raise PlansUnavailable(root) from exc

    found = []
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        if not fnmatch.fnmatch(entry.name, config.PLANS_DIR_GLOB):
            continue
        found.extend(_walk(os.path.realpath(entry.path), entry.name))
    return found


def _walk(base: str, dir_name: str) -> list:
    """Collect indexable Markdown files under one plans directory."""
    found = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        if dirpath[len(base):].count(os.sep) >= config.PLANS_MAX_DEPTH:
            dirnames[:] = []
        for name in sorted(filenames):
            if not name.lower().endswith(".md"):
                continue
            abs_path = os.path.join(dirpath, name)
            # A symlink cannot be trusted to stay inside the plans directory.
            if os.path.islink(abs_path):
                continue
            if not os.path.realpath(abs_path).startswith(base + os.sep):
                continue
            try:
                stat = os.stat(abs_path)
            except OSError:
                continue
            if stat.st_size > config.PLANS_MAX_BYTES:
                continue
            rel = os.path.join(dir_name, os.path.relpath(abs_path, base))
            found.append((rel, dir_name, abs_path, stat))
    return found


# abs_path -> (mtime, size, metadata). Populated by build_index in Task 3.
_CACHE: dict = {}


def reset_cache() -> None:
    """Drop every cached entry. Used by tests to isolate fixture trees."""
    _CACHE.clear()


# Bytes read from the head of a document when hunting for its H1. Reading a
# prefix rather than the whole file keeps a full index build cheap even though
# the largest document in the corpus is 162 KB.
_TITLE_BYTES = 4096


def read_title(abs_path: str, stem: str) -> str:
    """Return the document's first H1, or a de-slugged filename if it has none.

    Fenced blocks are tracked because ``# comment`` is ordinary bash, and the
    corpus contains 2,645 bash fences — a naive regex would title documents
    after a shell comment. Recognizes both ``` and ~~~ as fence delimiters,
    tracking which marker opened the fence so they do not cross-close.
    """
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as handle:
            head = handle.read(_TITLE_BYTES)
    except OSError:
        return fallback_title(stem)

    # Discard any partial final line only when the read hit the 4096-byte cap.
    # A short file with no trailing newline is complete and should not be truncated.
    if len(head) == _TITLE_BYTES and head[-1] != '\n':
        head = head[:head.rfind('\n')]

    fence_marker = None
    for line in head.splitlines():
        stripped = line.lstrip()
        # Track both ``` and ~~~ fence markers, remembering which one opened the fence.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if fence_marker == marker:
                fence_marker = None
            elif fence_marker is None:
                fence_marker = marker
            continue
        if not fence_marker and line.startswith("# "):
            return line[2:].strip()
    return fallback_title(stem)


def build_index(root: Optional[str] = None) -> dict:
    """Return ``{rel: metadata}``, re-reading only files whose stat changed."""
    root = os.path.realpath(root or config.PLANS_ROOT)
    index, seen = {}, set()

    for rel, dir_name, abs_path, stat in discover(root):
        seen.add(abs_path)
        cache_key = (root, abs_path)
        cached = _CACHE.get(cache_key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            # Return a shallow copy to prevent caller mutation of the canonical cache.
            index[rel] = dict(cached[2])
            continue

        stem = os.path.basename(rel)[: -len(".md")]
        app = dir_name[: -len(_PLANS_SUFFIX)] if dir_name.endswith(_PLANS_SUFFIX) else dir_name
        meta = {
            "rel": rel,
            "app": app,
            "dir": dir_name,
            "title": read_title(abs_path, stem),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            **parse_filename(stem),
        }
        _CACHE[cache_key] = (stat.st_mtime, stat.st_size, meta)
        # Return a shallow copy to prevent caller mutation of the canonical cache.
        index[rel] = dict(meta)

    # Drop entries for files that have disappeared from this root, without touching
    # entries cached under a different root (which is what test fixtures use).
    for cache_key in [k for k in _CACHE if k[0] == root and k[1] not in seen]:
        _CACHE.pop(cache_key, None)
    return index


def get_document(rel: str, root: Optional[str] = None) -> Optional[dict]:
    """Return metadata plus raw Markdown, or None when ``rel`` is not indexed.

    ``rel`` is used to touch the filesystem only after it is confirmed to be a
    key in the freshly built index, so traversal is structurally impossible
    rather than filtered: ``../../etc/passwd`` is not a key, so it is a miss.
    """
    meta = build_index(root).get(rel)
    if meta is None:
        return None

    root = os.path.realpath(root or config.PLANS_ROOT)
    abs_path = os.path.join(root, rel)
    if not os.path.realpath(abs_path).startswith(root + os.sep):
        return None
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    return {**meta, "markdown": text}


def as_library(index: dict) -> list:
    """Group the index per app: apps by document count, documents newest first."""
    apps: dict = {}
    for meta in index.values():
        apps.setdefault(meta["app"], []).append(meta)

    library = []
    for app, docs in apps.items():
        docs.sort(key=lambda m: m["rel"])
        # Stable, so the rel ordering above survives as the final tiebreak.
        docs.sort(key=lambda m: (m["date"] or "", m["plan_no"] or 0), reverse=True)
        library.append({"app": app, "dir": docs[0]["dir"], "count": len(docs), "docs": docs})

    library.sort(key=lambda entry: (-entry["count"], entry["app"]))
    return library
