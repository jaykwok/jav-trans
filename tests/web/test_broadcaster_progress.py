from __future__ import annotations

import asyncio
import json
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


def test_translation_progress_promotes_counts_to_job_progress(tmp_path, monkeypatch):
    asyncio.run(_test_translation_progress_promotes_counts_to_job_progress(tmp_path, monkeypatch))


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
