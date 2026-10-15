"""tests/test_api_error_reason.py — the dashboard's rejection reasons stay short and free of markup.

`api()` in web/app.js is the single front door for every dashboard read, so the
string it puts on an Error is what every failure banner, header indicator and
alert in the page prints. It used to be the entire response body: against the
API that is short JSON, but when the API container is down `merllm-nginx`
answers instead with a full HTML error page (#95).

Three layers here, because the fix spans two languages:

* the API really does state its reasons under the two JSON keys the fix reads
  (`detail` from HTTPException, `error` from the hand-built JSONResponse paths)
  — these run always, and are what catches the API changing shape underneath;
* web/app.js no longer interpolates a raw body into the Error, and still logs
  the whole body for debugging — source assertions, also always run;
* the reason actually produced for real upstream bodies, driven through `node`
  because there is no JS test runner in this repo.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

APP_JS = Path(__file__).resolve().parent.parent / "web" / "app.js"

# Nginx 1.25.3's stock 502 body, which is what the browser receives whenever the
# API container is down or restarting behind merllm-nginx.
NGINX_502 = (
    "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n"
    "<center><h1>502 Bad Gateway</h1></center>\r\n"
    "<hr><center>nginx/1.25.3</center>\r\n</body>\r\n</html>\r\n"
)

# The cap web/app.js applies, kept in step with API_REASON_MAX there.
REASON_MAX = 160


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


# ── Layer 1: the API's error bodies are the shapes the fix reads ──────────────

def test_httpexception_states_its_reason_under_detail(client):
    """An unknown GPU name is refused with FastAPI's `detail` key."""
    r = client.post("/api/merllm/gpu/not-a-gpu/reset")
    assert r.status_code == 422
    assert r.json()["detail"] == "gpu must be 'gpu0' or 'gpu1'"


def test_settings_conflict_states_its_reason_under_error(client):
    """A default_model change without confirmation is refused under `error`, not `detail`.

    This is the body that made "prefer `detail`" the wrong fix on its own: the
    409's text is the operator's instruction for how to proceed, so a reader
    that only knows `detail` would replace it with a bare status line.
    """
    import config
    r = client.post(
        "/api/merllm/settings",
        json={"default_model": config.DEFAULT_MODEL + "-other"},
    )
    assert r.status_code == 409
    body = r.json()
    assert "detail" not in body
    assert "confirm_model_change" in body["error"]


def test_validation_error_states_its_reason_as_a_list_under_detail(client):
    """FastAPI's own 422 sends one object per rejected field, each with `msg`.

    `limit` is typed `int`, and web/app.js:1423 fetches this endpoint with a
    `limit` and prints the rejection into the fault-history pane — so the list
    shape reaches a printing surface and is not a framework curiosity.
    """
    r = client.get("/api/merllm/fans/faults?limit=not-a-number")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list) and detail
    assert all("msg" in d for d in detail)


# ── Layer 2: web/app.js does not put a raw body on the Error ─────────────────

def test_api_does_not_interpolate_the_response_body_into_the_error():
    """The throw that made the whole body the reason is gone."""
    src = APP_JS.read_text(encoding="utf-8")
    assert "${res.status} ${await res.text()}" not in src
    assert "_apiErrorReason(res, body)" in src


def test_api_still_logs_the_whole_body_for_debugging():
    """Clamping the on-screen reason must not cost the body a developer needs."""
    src = APP_JS.read_text(encoding="utf-8")
    body = _function_body(src, "api")
    assert "console.error" in body, "api() no longer logs the response body"
    assert "body" in body.split("console.error", 1)[1].split("\n", 1)[0]


def test_api_reads_both_json_reason_keys_the_backend_sends():
    """`detail` and `error` are both live producers — see layer 1."""
    reason_src = _function_body(APP_JS.read_text(encoding="utf-8"), "_jsonErrorReason")
    for key in ("detail", "error", "msg"):
        assert f".{key}" in reason_src, f"_jsonErrorReason ignores the `{key}` key"


def test_the_cap_in_app_js_matches_the_one_asserted_here():
    """Keep REASON_MAX above honest, so the length cases are testing the real cap."""
    src = APP_JS.read_text(encoding="utf-8")
    assert f"const API_REASON_MAX = {REASON_MAX};" in src


# ── Layer 3: the reason produced for real upstream bodies ────────────────────

CASES = [
    # (label, status, statusText, body, expected reason)
    ("nginx's 502 page carries nothing the status line does not",
     502, "Bad Gateway", NGINX_502, "502 Bad Gateway"),
    ("nginx's 504 page likewise",
     504, "Gateway Time-out",
     NGINX_502.replace("502 Bad Gateway", "504 Gateway Time-out"),
     "504 Gateway Time-out"),
    ("an HTTPException detail survives in full",
     404, "Not Found", json.dumps({"detail": "GPU not found"}),
     "404 GPU not found"),
    ("the settings 409 keeps the instruction telling the operator what to do",
     409, "Conflict",
     json.dumps({"ok": False, "error": "Changing default_model will affect new "
                                       "dispatches. Resend with confirm_model_change: "
                                       "true to proceed."}),
     "409 Changing default_model will affect new dispatches. Resend with "
     "confirm_model_change: true to proceed."),
    ("a validation 422 reports the rejected field's message",
     422, "Unprocessable Entity",
     json.dumps({"detail": [{"loc": ["body", "prompt"], "msg": "field required",
                             "type": "value_error.missing"}]}),
     "422 field required"),
    ("several rejected fields are joined",
     422, "Unprocessable Entity",
     json.dumps({"detail": [{"loc": ["body", "a"], "msg": "field required"},
                            {"loc": ["body", "b"], "msg": "not a valid integer"}]}),
     "422 field required; not a valid integer"),
    ("the fan controller's 503 reports the upstream exception text",
     503, "Service Unavailable",
     json.dumps({"error": "All connection attempts failed", "available": False}),
     "503 All connection attempts failed"),
    ("the /v1 routes' nested OpenAI error shape is unwrapped",
     503, "Service Unavailable",
     json.dumps({"error": {"message": "GPU busy — timed out waiting for slot",
                           "type": "server_error"}}),
     "503 GPU busy — timed out waiting for slot"),
    ("a plain-text body is usable as the reason",
     500, "Internal Server Error", "upstream connect error",
     "500 upstream connect error"),
    ("an empty body falls back to the status line",
     502, "Bad Gateway", "", "502 Bad Gateway"),
    ("with no statusText either, the bare status is all that is claimed",
     502, "", "", "502"),
    ("JSON carrying no reason key falls back to the status line",
     500, "Internal Server Error", json.dumps({"foo": "bar"}),
     "500 Internal Server Error"),
    ("a multi-line reason is collapsed to one line",
     500, "Internal Server Error",
     json.dumps({"detail": "Invalid JSON:\n  line 1\n  column 4"}),
     "500 Invalid JSON: line 1 column 4"),
]


@pytest.mark.parametrize("label,status,status_text,body,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_reason_for_real_upstream_body(label, status, status_text, body, expected):
    assert _reason(status, status_text, body) == expected


def test_an_over_long_reason_is_capped_and_marked_as_cut():
    long_detail = "Backup failed: " + "x" * 400
    got = _reason(500, "Internal Server Error", json.dumps({"detail": long_detail}))
    assert got.startswith("500 Backup failed: ")
    assert got.endswith("…")
    assert len(got) <= len(str(500)) + 1 + REASON_MAX


@pytest.mark.parametrize("label,status,status_text,body,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_no_reason_is_markup_long_or_multiline(label, status, status_text, body, expected):
    """The three properties the surfaces printing this string depend on."""
    got = _reason(status, status_text, body)
    assert "<" not in got
    assert len(got) <= len(str(status)) + 1 + REASON_MAX
    assert "\n" not in got and "\r" not in got


# ── Helpers ──────────────────────────────────────────────────────────────────

def _function_body(src: str, name: str) -> str:
    """Return the source of `function name(...)`, braces balanced.

    The parameter list is skipped before counting braces, because a default
    such as `opts = {}` would otherwise close the count on the first `}`.
    """
    start = src.index(f"function {name}(")
    paren = 0
    for i in range(start, len(src)):
        if src[i] == "(":
            paren += 1
        elif src[i] == ")":
            paren -= 1
            if paren == 0:
                body_start = src.index("{", i)
                break
    else:
        raise AssertionError(f"unbalanced parens reading {name}()")
    depth = 0
    for i in range(body_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"unbalanced braces reading {name}()")


def _reason(status: int, status_text: str, body: str) -> str:
    """Run web/app.js's own reason builder over a response, via node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to run web/app.js's reason builder")
    src = APP_JS.read_text(encoding="utf-8")
    cap = src.split("const API_REASON_MAX = ", 1)[1].split(";", 1)[0]
    script = textwrap.dedent("""\
        const API_REASON_MAX = %s;
        %s
        %s
        const arg = JSON.parse(process.argv[1]);
        process.stdout.write(_apiErrorReason(
          { status: arg.status, statusText: arg.statusText }, arg.body));
    """) % (cap, _function_body(src, "_jsonErrorReason"),
            _function_body(src, "_apiErrorReason"))
    out = subprocess.run(
        [node, "-e", script,
         json.dumps({"status": status, "statusText": status_text, "body": body})],
        capture_output=True, text=True, timeout=20,
    )
    assert out.returncode == 0, f"node failed: {out.stderr}"
    return out.stdout
