from __future__ import annotations

import asyncio
import threading
from pathlib import Path


import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web import pipeline_manager as pm
from web.models import JobSpec


def test_pipeline_workers_overlap(tmp_path, monkeypatch):
    asyncio.run(_test_pipeline_workers_overlap(tmp_path, monkeypatch))


async def _test_pipeline_workers_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
    monkeypatch.setattr(pm, "_jobs", {})
    monkeypatch.setattr(pm, "_cancel_events", {})
    monkeypatch.setattr(pm, "_tombstones", {})
    monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
    monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())
    first_translation = threading.Event()
    two_translations = threading.Event()
    all_asr_done = threading.Event()
    release_translations = threading.Event()
    translation_lock = threading.Lock()
    asr_done: set[str] = set()
    translations_started: set[str] = set()
    created = []
    for name in ("a", "b", "c"):
        created.extend(await pm.create_job(JobSpec(
            video_paths=[str(tmp_path / f"sample-{name}.mp4")], skip_translation=False,
        )))

    def fake_asr(job, _cancel_event, _run_id):
        if job.id != created[0].id:
            # Later ASR can finish only while the first translation is running.
            assert first_translation.wait(10)
        asr_done.add(job.id)
        if len(asr_done) == len(created):
            all_asr_done.set()
        return {"job_id": job.id}

    def fake_translation(job, _asr_artifacts, _cancel_event, _run_id):
        with translation_lock:
            translations_started.add(job.id)
            if job.id == created[0].id:
                first_translation.set()
            if len(translations_started) == 2:
                two_translations.set()
        assert release_translations.wait(10)
        return []

    monkeypatch.setattr(pm, "_run_asr_alignment", fake_asr)
    monkeypatch.setattr(pm, "_run_translation_and_write", fake_translation)

    workers = [
        asyncio.create_task(pm.gpu_worker()),
        asyncio.create_task(pm.translation_worker()),
        asyncio.create_task(pm.translation_worker()),
    ]
    try:
        # Observe actual concurrency; elapsed time also measures CI disk speed.
        assert await asyncio.to_thread(two_translations.wait, 5)
        assert await asyncio.to_thread(all_asr_done.wait, 5)
        assert (await pm.get_job(created[0].id)).status == "translating"
        release_translations.set()
        await asyncio.wait_for(pm.gpu_queue.join(), 5)
        await asyncio.wait_for(pm.trans_queue.join(), 5)
        jobs = [await pm.get_job(job.id) for job in created]
        assert all(job is not None and job.status == "done" for job in jobs)
        assert translations_started == asr_done == {job.id for job in created}
    finally:
        release_translations.set()
        first_translation.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
