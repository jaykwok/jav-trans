from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web import broadcaster
from web import pipeline_manager as pm
from web.models import JobSpec, JobState


async def _drain_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        queue.task_done()


async def _reset_pm_state() -> None:
    async with pm._state_lock:
        pm._jobs.clear()
        pm._cancel_events.clear()
    await _drain_queue(pm.gpu_queue)
    await _drain_queue(pm.trans_queue)
    # A module-level asyncio.Queue binds to the first loop that waits on it, and
    # every test here runs its own asyncio.run(): reuse would kill the next
    # test's workers with "bound to a different event loop" as soon as they
    # await an empty get(). Hand each test fresh queues instead.
    pm.gpu_queue = asyncio.Queue()
    pm.trans_queue = asyncio.Queue()


def test_translation_progress_promotes_counts_to_job_progress(tmp_path, monkeypatch):
    asyncio.run(_test_translation_progress_promotes_counts_to_job_progress(tmp_path, monkeypatch))


def test_previous_run_cannot_move_the_retried_job(tmp_path, monkeypatch):
    asyncio.run(_test_previous_run_cannot_move_the_retried_job(tmp_path, monkeypatch))


async def _test_previous_run_cannot_move_the_retried_job(tmp_path, monkeypatch):
    # A cancelled run's threads keep emitting until they reach their next
    # checkpoint. Those events arrive under the same job id as the retry that
    # replaced them, and used to overwrite the new run's progress with the old
    # run's counters - both in the stored job and on the page.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="retried-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-2",
        status="translating",
        current_stage="translation",
        progress={"stage": "translation", "translated": 5, "expected": 100},
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()
    subscriber = await broadcaster.subscribe()
    await _drain_queue(subscriber)

    stale = json.dumps(
        {
            "job_id": job.id,
            "run_id": "run-1",
            "video": "sample.mp4",
            "stage": "translation",
            "phase": "progress",
            "extra": {"translated": 90, "expected": 100},
        },
        ensure_ascii=False,
    )
    broadcaster.publish(stale)
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.progress["translated"] == 5
    assert subscriber.empty()  # and the page never saw it either

    live = json.dumps(
        {
            "job_id": job.id,
            "run_id": "run-2",
            "video": "sample.mp4",
            "stage": "translation",
            "phase": "progress",
            "extra": {"translated": 7, "expected": 100},
        },
        ensure_ascii=False,
    )
    broadcaster.publish(live)
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.progress["translated"] == 7
    assert not subscriber.empty()

    broadcaster.unsubscribe(subscriber)
    await _reset_pm_state()


def test_stale_progress_is_rejected_by_the_store_too(tmp_path, monkeypatch):
    asyncio.run(_test_stale_progress_is_rejected_by_the_store_too(tmp_path, monkeypatch))


async def _test_stale_progress_is_rejected_by_the_store_too(tmp_path, monkeypatch):
    # The fan-out filter above is a best-effort read outside the state lock, so
    # the stored job is checked again under the lock. This calls the writer
    # directly to keep that second defence pinned on its own.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="store-guard-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-2",
        status="translating",
        current_stage="translation",
        progress={"stage": "translation", "translated": 5, "expected": 100},
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()

    await pm.update_job_progress(
        job.id,
        {"stage": "translation", "translated": 90, "expected": 100},
        run_id="run-1",
    )
    current = await pm.get_job(job.id)
    assert current is not None
    assert current.progress["translated"] == 5

    await pm.update_job_progress(
        job.id,
        {"stage": "translation", "translated": 7, "expected": 100},
        run_id="run-2",
    )
    current = await pm.get_job(job.id)
    assert current is not None
    assert current.progress["translated"] == 7
    await _reset_pm_state()


def test_the_cancelled_runs_own_progress_cannot_undo_the_cancel(tmp_path, monkeypatch):
    asyncio.run(
        _test_the_cancelled_runs_own_progress_cannot_undo_the_cancel(tmp_path, monkeypatch)
    )


async def _test_the_cancelled_runs_own_progress_cannot_undo_the_cancel(
    tmp_path,
    monkeypatch,
):
    # Run-id filtering cannot help here: these events belong to the run that was
    # cancelled, so the id matches. A translation event mapping back to
    # "translating" put the card straight back to looking like a job that had
    # restarted itself - the exact symptom the user reported originally.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="cancelling-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-1",
        status="translating",
        current_stage="translation",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    assert await pm.cancel_job(job.id)
    assert (await pm.get_job(job.id)).status == "cancelling"

    await pm.update_job_progress(
        job.id,
        {"stage": "translation", "translated": 12, "expected": 100},
        run_id="run-1",
    )

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.status == "cancelling"
    assert current.current_stage == "cancelling"
    # Counters still move: that is what the run was actually doing.
    assert current.progress["translated"] == 12
    assert pm._cancel_events[job.id].is_set()
    await _reset_pm_state()


def test_events_without_a_run_id_are_still_delivered(tmp_path, monkeypatch):
    asyncio.run(_test_events_without_a_run_id_are_still_delivered(tmp_path, monkeypatch))


async def _test_events_without_a_run_id_are_still_delivered(tmp_path, monkeypatch):
    # The CLI emits under a job id with no run behind it, and so does anything
    # written before run ids existed. "Cannot tell" must not mean "drop".
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="no-run-id-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-9",
        status="translating",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()

    broadcaster.publish(
        json.dumps(
            {
                "job_id": job.id,
                "video": "sample.mp4",
                "stage": "translation",
                "phase": "progress",
                "extra": {"translated": 3, "expected": 10},
            },
            ensure_ascii=False,
        )
    )
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.progress["translated"] == 3
    await _reset_pm_state()


def test_timing_summary_does_not_replace_job_progress(tmp_path, monkeypatch):
    asyncio.run(_test_timing_summary_does_not_replace_job_progress(tmp_path, monkeypatch))


async def _test_timing_summary_does_not_replace_job_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="timing-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="writing",
        current_stage="write_output",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()

    broadcaster.publish(
        json.dumps(
            {
                "job_id": job.id,
                "video": "sample.mp4",
                "stage": "timing_summary",
                "phase": "done",
                "extra": {"rows": [{"label": "总计", "seconds": 12.3}]},
            },
            ensure_ascii=False,
        )
    )
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.current_stage == "write_output"
    await _reset_pm_state()


async def _test_translation_progress_promotes_counts_to_job_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    job = JobState(
        id="progress-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="writing",
        current_stage="write_output",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()

    broadcaster.publish(
        json.dumps(
            {
                "job_id": job.id,
                "video": "sample.mp4",
                "stage": "translation",
                "phase": "progress",
                "extra": {
                    "phase": "translating",
                    "translated": 40,
                    "expected": 100,
                    "content_chars": 1200,
                },
            },
            ensure_ascii=False,
        )
    )
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.status == "translating"
    assert current.current_stage == "translation"
    assert current.progress["stage"] == "translation"
    assert current.progress["translated"] == 40
    assert current.progress["expected"] == 100
    assert current.progress["content_chars"] == 1200
    assert current.progress["extra"]["translated"] == 40

    broadcaster.publish(
        json.dumps(
            {
                "job_id": job.id,
                "video": "sample.mp4",
                "stage": "write_output",
                "phase": "start",
                "extra": {},
            },
            ensure_ascii=False,
        )
    )
    await asyncio.sleep(0)

    current = await pm.get_job(job.id)
    assert current is not None
    assert current.status == "writing"
    assert current.current_stage == "write_output"
    await _reset_pm_state()
