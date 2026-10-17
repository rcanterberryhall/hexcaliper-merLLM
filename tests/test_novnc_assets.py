"""tests/test_novnc_assets.py — the nginx image ships a loadable noVNC graph.

Guards the fix for issue #102, which requires that every noVNC directory the
viewer's module graph reaches be copied into the nginx image, not just
``core/``. Before the fix ``nginx/Dockerfile`` copied ``core/`` alone, so the
dynamic ``import('/novnc/core/rfb.js')`` in ``web/app.js`` failed with
``TypeError: Failed to fetch dynamically imported module`` and the VNC tab's
**Connect** button could only ever render a load error.

The graph crosses out of ``core/`` exactly once, in a static import that the
browser must resolve before the module evaluates:

    core/rfb.js        -> ./inflator.js, ./deflator.js
    core/inflator.js   -> ../vendor/pako/lib/zlib/{inflate,zstream}.js
    core/deflator.js   -> ../vendor/pako/lib/zlib/{deflate,zstream}.js

so ``vendor/`` is as load-bearing as ``core/``, even though nothing in the
compression path runs until a Tight- or ZRLE-encoded framebuffer arrives.

The asset tree exists only inside the built image, which no test here can see,
so these assertions read ``nginx/Dockerfile`` instead: that it still pins a
noVNC release (the required directory set is release-specific), that it copies
both names, and that each lands under the shared ``/novnc/`` root keeping its
own name — which is what makes the relative ``../vendor/`` specifiers resolve.
"""
import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "nginx" / "Dockerfile"

# Top-level noVNC directories the viewer's import graph reaches, with what each
# supplies. A noVNC upgrade that moves compression out of vendor/pako should
# re-derive this set rather than delete the entry.
REQUIRED = {
    "core": "rfb.js and the decoders, display and input modules it imports",
    "vendor": "the pako zlib modules core/inflator.js and core/deflator.js import",
}

SERVED_ROOT = "/usr/share/nginx/html/novnc"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def copies(dockerfile: str) -> dict:
    """Map each copied noVNC subtree name to its destination path."""
    found = {}
    for src, dest in re.findall(
        r"^COPY\s+--from=novnc\s+(\S+)\s+(\S+)\s*$", dockerfile, re.MULTILINE
    ):
        found[src.rstrip("/").rsplit("/", 1)[-1]] = dest.rstrip("/")
    assert found, "no COPY --from=novnc lines found in nginx/Dockerfile"
    return found


def test_the_novnc_checkout_is_still_pinned_to_a_release(dockerfile: str):
    """A floating branch would make the required directory set unknowable."""
    assert re.search(
        r"git clone .*--branch\s+v\d+\.\d+(\.\d+)?\s", dockerfile
    ), "the noVNC clone must pin a release tag, so REQUIRED stays derivable"


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_every_directory_the_module_graph_reaches_is_copied(name, copies):
    assert name in copies, (
        f"nginx/Dockerfile does not copy noVNC's {name}/, which supplies "
        f"{REQUIRED[name]}; a browser resolves the whole graph before "
        f"core/rfb.js evaluates, so one missing file fails the import"
    )


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_each_copy_keeps_its_name_under_the_served_root(name, copies):
    """``../vendor/`` resolves only if the two sit side by side as served."""
    assert copies.get(name) == f"{SERVED_ROOT}/{name}", (
        f"noVNC's {name}/ is copied to {copies.get(name)!r}; it has to land at "
        f"{SERVED_ROOT}/{name} for core/'s relative imports to resolve"
    )
