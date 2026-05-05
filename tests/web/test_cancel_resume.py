from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

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


def test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch):
    asyncio.run(_test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch))


async def _test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    started = threading.Event()
    observed_cancel = threading.Event()

    def fake_run_asr_alignment_f0(_video_path, *, cancel_event=None, **_kwargs):
        started.set()
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                observed_cancel.set()
                raise pm.pipeline_main.PipelineCancelledError("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel_event was not set")

    monkeypatch.setattr(pm, "run_asr_alignment_f0", fake_run_asr_alignment_f0)

    worker = asyncio.create_task(pm.gpu_worker())
    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        deadline = time.perf_counter() + 2.0
        while not started.is_set() and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        assert started.is_set()

        assert await pm.cancel_job(job.id)

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if observed_cancel.is_set():
                break
            await asyncio.sleep(0.01)

        assert observed_cancel.is_set()
        current = await pm.get_job(job.id)
        assert current is not None
        assert current.status == "cancelled"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_resume_cache_job_id_requires_valid_aligned_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / job_id))

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"sample")
    old_job_id = "old-job"
    new_job_id = "new-job"
    old_dir = tmp_path / old_job_id
    old_dir.mkdir()
    aligned_path = old_dir / "sample.aligned_segments.json"
    spec = JobSpec(
        video_paths=[str(video_path)],
        resume_from_job_id=old_job_id,
    )
    aligned_path.write_text(
        json.dumps(
            {
                "backend": spec.asr_backend,
                "audio_cache_key": "cache-key",
                "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pm.pipeline_main, "_get_audio_cache_key", lambda _path: "cache-key")

    job = JobState(
        id=new_job_id,
        spec=spec,
        created_at="2026-05-04T00:00:00.000+00:00",
        status="queued",
    )

    assert pm._resume_cache_job_id(job) == old_job_id

    aligned_path.write_text("{}", encoding="utf-8")
    assert pm._resume_cache_job_id(job) == new_job_id


def test_retry_cancelled_job_requeues_same_job_id(tmp_path, monkeypatch):
    asyncio.run(_test_retry_cancelled_job_requeues_same_job_id(tmp_path, monkeypatch))


async def _test_retry_cancelled_job_requeues_same_job_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    old_event = threading.Event()
    old_event.set()
    job = JobState(
        id="retry-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="cancelled",
        current_stage="cancelled",
        progress={"translated": 20, "expected": 100},
        artifacts=["old.srt"],
        error="cancelled",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = old_event
        pm._write_jobs_unlocked()

    retried = await pm.retry_job(job.id)

    assert retried is not None
    assert retried.id == job.id
    assert retried.status == "queued"
    assert retried.current_stage is None
    assert retried.progress == {}
    assert retried.artifacts == []
    assert retried.error is None
    assert pm._cancel_events[job.id] is not old_event
    assert not pm._cancel_events[job.id].is_set()

    queued = pm.gpu_queue.get_nowait()
    try:
        assert queued is retried
    finally:
        pm.gpu_queue.task_done()
        await _reset_pm_state()


def test_stale_cancelled_worker_cannot_overwrite_retried_job(tmp_path, monkeypatch):
    asyncio.run(_test_stale_cancelled_worker_cannot_overwrite_retried_job(tmp_path, monkeypatch))


async def _test_stale_cancelled_worker_cannot_overwrite_retried_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    old_event = threading.Event()
    new_event = threading.Event()
    stale_job = JobState(
        id="retry-race",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="cancelled",
    )
    retried_job = stale_job.model_copy(
        update={"status": "queued", "current_stage": None},
    )
    async with pm._state_lock:
        pm._jobs[retried_job.id] = retried_job
        pm._cancel_events[retried_job.id] = new_event
        pm._write_jobs_unlocked()

    await pm._set_job(
        stale_job,
        status="cancelled",
        current_stage="cancelled",
        expected_cancel_event=old_event,
    )

    current = await pm.get_job(retried_job.id)
    assert current is not None
    assert current.status == "queued"
    assert current.current_stage is None
    await _reset_pm_state()


def test_remove_finished_job_deletes_job_temp_dir(tmp_path, monkeypatch):
    asyncio.run(_test_remove_finished_job_deletes_job_temp_dir(tmp_path, monkeypatch))


async def _test_remove_finished_job_deletes_job_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / "jobs" / job_id))
    await _reset_pm_state()

    job = JobState(
        id="done-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="done",
        current_stage="done",
    )
    temp_dir = Path(pm._job_temp_dir(job.id))
    temp_dir.mkdir(parents=True)
    (temp_dir / "translation_cache.jsonl").write_text("{}", encoding="utf-8")
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    assert await pm.remove_job(job.id)
    assert not temp_dir.exists()
    await _reset_pm_state()


def test_cancel_active_job_keeps_job_temp_dir_for_retry(tmp_path, monkeypatch):
    asyncio.run(_test_cancel_active_job_keeps_job_temp_dir_for_retry(tmp_path, monkeypatch))


async def _test_cancel_active_job_keeps_job_temp_dir_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / "jobs" / job_id))
    await _reset_pm_state()

    job = JobState(
        id="active-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="translating",
        current_stage="translation",
    )
    temp_dir = Path(pm._job_temp_dir(job.id))
    temp_dir.mkdir(parents=True)
    (temp_dir / "translation_cache.jsonl").write_text("{}", encoding="utf-8")
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    assert await pm.remove_job(job.id)
    assert temp_dir.exists()
    current = await pm.get_job(job.id)
    assert current is not None
    assert current.status == "cancelled"
    await _reset_pm_state()
