"""What the single writer of `jobs.json` promises, and what it refuses to.

Persisting the job table used to happen on whichever thread had just changed a
job - including the event loop, once per progress tick, serializing every job of
every run. The writer that replaced it makes two different promises depending on
what changed, and both of them are worth a test: progress is coalesced and paced
because only the newest tick answers the question, while a terminal change is
waited for because "this job finished" surviving a crash is what the file is for.

The deletion half is the same subject from the client's side: a page holding a
card for a job that no longer exists needs to be told the difference between
"never existed" and "gone", which is what the tombstones answer.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import httpx
import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.app import create_app
from web import pipeline_manager as pm
from web.models import JobSpec, JobState


def _state(job_id: str, *, status: str = "translating", **progress) -> JobState:
    return JobState(
        id=job_id,
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-09-06T00:00:00.000+00:00",
        run_id="run-1",
        status=status,
        progress=dict(progress),
    )


async def _reset() -> None:
    async with pm._state_lock:
        pm._jobs.clear()
        pm._cancel_events.clear()
        pm._tombstones.clear()


@pytest.fixture(autouse=True)
def _isolate_tombstones():
    kept = dict(pm._tombstones)
    try:
        yield
    finally:
        pm._tombstones.clear()
        pm._tombstones.update(kept)


def test_a_burst_of_progress_becomes_one_write(monkeypatch):
    """Fifty ticks describe one state, and only the last one is true."""
    writer = pm._JobStoreWriter()
    writes: list[list[JobState]] = []
    monkeypatch.setattr(pm, "_write_jobs_snapshot", writes.append)

    for pct in range(50):
        writer.submit([_state("a", pct=pct)], confirm=False)
    writer.submit([_state("a", status="done")], confirm=True)

    # One write for the head of the burst, one for the terminal change that
    # cannot wait for the pace. The point is that it is not fifty-one.
    assert len(writes) <= 2
    assert writes[-1][0].status == "done"


def test_a_finished_job_is_on_disk_before_the_caller_moves_on(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    writer = pm._JobStoreWriter()

    writer.submit([_state("a", status="done")], confirm=True)

    payload = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
    assert [(item["id"], item["status"]) for item in payload] == [("a", "done")]


def test_later_progress_cannot_coalesce_away_a_terminal_barrier():
    from web.job_store import JobStoreWriter

    entered, release = threading.Event(), threading.Event()
    states = []

    def write(snapshot):
        if not states:
            entered.set()
            assert release.wait(3)
        states.append([(job.id, job.status) for job in snapshot])

    writer = JobStoreWriter(write)
    writer.enqueue([_state("a")], confirm=False)
    try:
        assert entered.wait(2)
        terminal = writer.enqueue([_state("a", status="done")], confirm=True)
        writer.enqueue([_state("a", status="done"), _state("b", pct=1)], confirm=False)
        writer.enqueue([_state("a", status="done"), _state("b", pct=2)], confirm=False)
    finally:
        release.set()
    writer.flush()
    assert terminal.result() == 2
    assert states[1] == [("a", "done")]
    assert states[2] == [("a", "done"), ("b", "translating")]


def test_a_wedged_writer_reports_timeout_without_starting_a_second_writer(monkeypatch):
    writer = pm._JobStoreWriter()
    writer._CONFIRM_TIMEOUT_S = 0.2
    release = threading.Event()
    inline: list[list[JobState]] = []

    def fake_write(snapshot: list[JobState]) -> None:
        if threading.current_thread().name == "job-store-writer":
            release.wait(5.0)
            return
        inline.append(snapshot)

    monkeypatch.setattr(pm, "_write_jobs_snapshot", fake_write)
    try:
        with pytest.raises(pm.JobPersistenceError, match="超时"):
            writer.submit([_state("a", status="failed")], confirm=True)
        assert inline == []
        assert writer._written == 0
    finally:
        release.set()
        writer.flush()


def test_only_progress_is_allowed_to_go_unconfirmed(tmp_path, monkeypatch):
    asyncio.run(_test_only_progress_is_allowed_to_go_unconfirmed(tmp_path, monkeypatch))


async def _test_only_progress_is_allowed_to_go_unconfirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset()
    confirmations: list[bool] = []
    monkeypatch.setattr(
        pm,
        "_persist",
        lambda snapshot, *, confirm=False: confirmations.append(confirm),
    )

    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        await pm.update_job_progress(
            job.id, {"stage": "asr", "pct": 10}, run_id=job.run_id
        )
        await pm._set_job(job, status="done", expected_run_id=job.run_id)
    finally:
        await _reset()

    # Created: waited for, because a job the user asked for and never sees again
    # is worse than a slow POST. Progress: not waited for. Finished: waited for.
    assert confirmations == [True, False, True]


def test_a_deletion_is_remembered_but_not_forever():
    pm._tombstones.clear()
    for index in range(pm._TOMBSTONE_LIMIT + 10):
        pm._entomb(f"job-{index}", index)

    assert len(pm._tombstones) == pm._TOMBSTONE_LIMIT
    # The oldest are dropped first: a client that far behind gets the same
    # answer it got before tombstones existed.
    assert pm.deleted_revision("job-0") is None
    assert pm.deleted_revision(f"job-{pm._TOMBSTONE_LIMIT + 9}") == pm._TOMBSTONE_LIMIT + 9


def test_a_deleted_job_answers_differently_from_one_that_never_existed(
    tmp_path, monkeypatch
):
    asyncio.run(
        _test_a_deleted_job_answers_differently_from_one_that_never_existed(
            tmp_path, monkeypatch
        )
    )


async def _test_a_deleted_job_answers_differently_from_one_that_never_existed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset()

    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        # DELETE on a job that is still queued cancels it instead of removing
        # it, so finish it first: this is about what a removal leaves behind.
        await pm._set_job(job, status="done", expected_run_id=job.run_id)
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await client.delete(f"/api/jobs/{job.id}")
            gone = await client.get(f"/api/jobs/{job.id}")
            never = await client.get("/api/jobs/no-such-job")

        assert gone.status_code == 410
        # Dated, so a list response taken before the deletion cannot be used to
        # argue the job is still there - and a job recreated with the same id
        # after it is recognisably newer.
        assert int(gone.headers["X-Jobs-Revision"]) >= job.revision
        assert int(gone.headers["X-Jobs-Epoch"]) == pm.state_epoch()
        assert never.status_code == 404
    finally:
        await _reset()
