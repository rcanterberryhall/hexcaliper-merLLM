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
