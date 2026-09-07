"""The page asks about a job once at a time, and once more if it changed.

Every SSE event triggered a GET for that job, and translation emits events far
faster than a request completes, so a long film produced hundreds of overlapping
GETs asking the same question. The fix is not a debounce - ordering is decided
by revision in the reducer and must stay decided there - but dropping requests
that are redundant by construction.

Like the reducer, it is a module with no DOM and no fetch, so the behaviour is
run rather than asserted about as source text.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "web" / "static" / "js" / "coalesce.js"
HARNESS = Path(__file__).resolve().parent / "js" / "request_coalescing.mjs"


def test_repeated_refreshes_collapse_into_one_request() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the coalescing harness needs it")

    completed = subprocess.run(
        [node, str(HARNESS), str(MODULE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok -" in completed.stdout


def test_the_page_routes_both_refreshes_through_it() -> None:
    # The wiring the harness cannot see: the module is only useful if the page
    # actually calls through it, per job id and once for the list.
    api = (ROOT / "src" / "web" / "static" / "js" / "jobsApi.js").read_text(
        encoding="utf-8"
    )

    assert "coalesce(requestJob)" in api
    assert "coalesce(requestAllJobs)" in api
