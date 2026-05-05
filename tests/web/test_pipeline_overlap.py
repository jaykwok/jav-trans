from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web import pipeline_manager as pm
from web.models import JobSpec


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


def test_pipeline_workers_overlap(tmp_path, monkeypatch):
    asyncio.run(_test_pipeline_workers_overlap(tmp_path, monkeypatch))


async def _test_pipeline_workers_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    asr_done_at: dict[str, float] = {}
    translation_started_at: dict[str, float] = {}

    def fake_asr(job, cancel_event=None):
        time.sleep(0.3)
        asr_done_at[job.id] = time.perf_counter()
        return {"job_id": job.id}

    def fake_translation(job, _asr_artifacts):
        translation_started_at[job.id] = time.perf_counter()
        time.sleep(1.0)
        return []

    monkeypatch.setattr(pm, "_run_asr_alignment_f0", fake_asr)
    monkeypatch.setattr(pm, "_run_translation_and_write", fake_translation)

    workers = [
        asyncio.create_task(pm.gpu_worker()),
        asyncio.create_task(pm.translation_worker()),
        asyncio.create_task(pm.translation_worker()),
    ]
    try:
        started = time.perf_counter()
        specs = [
            JobSpec(video_paths=[f"{name}.mp4"], skip_translation=False)
            for name in ("a", "b", "c")
        ]
        created = []
        for spec in specs:
            created.extend(await pm.create_job(spec))

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            jobs = [await pm.get_job(job.id) for job in created]
            if all(job is not None and job.status == "done" for job in jobs):
                break
            await asyncio.sleep(0.02)

        elapsed = time.perf_counter() - started
        jobs = [await pm.get_job(job.id) for job in created]
        assert all(job is not None and job.status == "done" for job in jobs)
        assert elapsed < 2.4

        first_job_id = created[0].id
        assert first_job_id in asr_done_at
        assert first_job_id in translation_started_at
        assert translation_started_at[first_job_id] - asr_done_at[first_job_id] < 0.2

        all_asr_done = max(asr_done_at.values())
        assert translation_started_at[first_job_id] < all_asr_done
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await _reset_pm_state()
