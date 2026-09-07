"""The page's job state is merged by revision, and that merge is executable.

Two GETs for one job can land newest-first - SSE progress triggers one per
event and the 3s poll issues another - and the older answer used to win by
arriving last: a `failed` card reappearing over the run that had replaced it.
Cancelling the earlier request does not help, because its response had already
arrived.

The rules now live in `jobsMerge.js`, which has no DOM and no fetch, so they can
be run rather than asserted about as source text.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "web" / "static" / "js" / "jobsMerge.js"
HARNESS = Path(__file__).resolve().parent / "js" / "jobs_merge_reducer.mjs"


def test_delivery_retry_clicks_do_not_depend_on_translation_settings() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    completed = subprocess.run(
        [node, str(HARNESS.with_name("jobs_delivery_retry.mjs")), str(MODULE.with_name("jobsRender.js"))],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_reducer_drops_what_arrived_out_of_order() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the reducer harness needs it")

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


def test_the_page_reads_the_list_revision_header() -> None:
    # The wiring the harness cannot see: the reducer is only as good as the
    # number handed to it, and that number arrives in a header.
    api = (ROOT / "src" / "web" / "static" / "js" / "jobsApi.js").read_text(
        encoding="utf-8"
    )

    assert "X-Jobs-Revision" in api
    assert "X-Jobs-Epoch" in api
    assert "reduceJobList" in api
    assert "state.jobsRevision = merged.revision;" in api
    assert "state.jobsEpoch = merged.epoch;" in api


def test_a_newer_epoch_makes_the_page_rebuild_from_the_list() -> None:
    """One job cannot restate a list.

    A single-job response carrying a newer epoch says the revisions the page
    holds were issued by a clock that no longer exists. The reducer refuses to
    merge it - correct, but on its own that would leave the page frozen on the
    old snapshot, so the fetch has to go and get the list.
    """
    api = (ROOT / "src" / "web" / "static" / "js" / "jobsApi.js").read_text(
        encoding="utf-8"
    )
    body = api[
        api.index("async function requestJob") : api.index(
            "async function requestAllJobs"
        )
    ]

    assert "compareEpoch" in body
    assert "await fetchAllJobs();" in body


def test_the_page_keeps_polling_once_every_job_is_finished() -> None:
    """A page with nothing running still has something to learn.

    Polling used to stop the moment no job was active, so a job deleted in
    another tab - or a service restart that emptied the list - stayed on screen
    until someone reloaded the page. There is no event to wait for either: SSE
    only speaks while a run is in progress.
    """
    api = (ROOT / "src" / "web" / "static" / "js" / "jobsApi.js").read_text(
        encoding="utf-8"
    )
    body = api[api.index("export function startJobPolling"):]

    assert "IDLE_POLL_MS" in body
    # Two calls: the active-job path, and the idle one that used to be a return.
    assert body.count("fetchAllJobs()") == 2
