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
from pipeline.artifacts import AsrArtifacts, write_translation_artifacts_snapshot


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


def test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch):
    asyncio.run(_test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch))


async def _test_cancel_event_reaches_asr_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    started = threading.Event()
    observed_cancel = threading.Event()

    def fake_run_asr_alignment(_video_path, *, cancel_event=None, **_kwargs):
        started.set()
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                observed_cancel.set()
                raise pm.pipeline_main.PipelineCancelledError("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel_event was not set")

    monkeypatch.setattr(pm, "run_asr_alignment", fake_run_asr_alignment)

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
        # The final status is the worker's to write: 取消 only asks. Wait for the
        # run to actually stop rather than for the request to return.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            current = await pm.get_job(job.id)
            if current is not None and current.status == "cancelled":
                break
            await asyncio.sleep(0.01)
        current = await pm.get_job(job.id)
        assert current is not None
        assert current.status == "cancelled"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_resume_cache_job_id_requires_valid_aligned_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / job_id))
    monkeypatch.setattr(
        pm,
        "_translation_cache_path",
        lambda job_id: str(tmp_path / job_id / "translation_cache.jsonl"),
    )

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
    cache_signature = {"sig": "ok"}
    monkeypatch.setattr(
        pm.pipeline_main,
        "aligned_cache_expectations_for_ctx",
        lambda _ctx, **_kwargs: ("backend-label", cache_signature),
    )
    aligned_path.write_text(
        json.dumps(
            {
                "backend": "backend-label",
                "audio_cache_key": "cache-key",
                "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
                "cache_signature": cache_signature,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pm, "get_audio_cache_key", lambda _path: "cache-key")

    job = JobState(
        id=new_job_id,
        spec=spec,
        created_at="2026-05-04T00:00:00.000+00:00",
        status="queued",
    )

    assert pm._resume_cache_job_id(job) == old_job_id

    aligned_path.write_text("{}", encoding="utf-8")
    assert pm._resume_cache_job_id(job) == new_job_id


def test_gpu_worker_passes_captured_cancel_event_to_executor(tmp_path, monkeypatch):
    asyncio.run(_test_gpu_worker_passes_captured_cancel_event_to_executor(tmp_path, monkeypatch))


async def _test_gpu_worker_passes_captured_cancel_event_to_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    observed: list[threading.Event] = []
    entered = asyncio.Event()

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            observed.append(args[1])
            entered.set()
            pm._cancel_events[args[0].id] = threading.Event()
            raise pm.pipeline_main.PipelineCancelledError("cancelled")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: InlineLoop())

    worker = asyncio.create_task(pm.gpu_worker())
    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        old_event = pm._cancel_events[job.id]

        await asyncio.wait_for(entered.wait(), timeout=2.0)
        current = await pm.get_job(job.id)

        assert observed == [old_event]
        assert pm._cancel_events[job.id] is not old_event
        assert current is not None
        assert current.status == "asr"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_translation_worker_passes_captured_cancel_event_to_executor(tmp_path, monkeypatch):
    asyncio.run(
        _test_translation_worker_passes_captured_cancel_event_to_executor(
            tmp_path, monkeypatch
        )
    )


async def _test_translation_worker_passes_captured_cancel_event_to_executor(
    tmp_path,
    monkeypatch,
):
    # Mirror of the ASR guard above. The translation thread used to re-read
    # `_cancel_events[job.id]` after it had already been handed to the executor,
    # so a cancel-then-retry in that window gave the *old* translation the new
    # run's fresh event: its own cancel flag became unreachable and the batch
    # kept translating (and billing) under a job the UI had already retried.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    observed: list[threading.Event] = []
    entered = asyncio.Event()

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            del func
            observed.append(args[2])
            entered.set()
            pm._cancel_events[args[0].id] = threading.Event()
            raise pm.pipeline_main.PipelineCancelledError("cancelled")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: InlineLoop())

    worker = asyncio.create_task(pm.translation_worker())
    job = JobState(
        id="translation-event",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="translating",
    )
    old_event = threading.Event()
    try:
        async with pm._state_lock:
            pm._jobs[job.id] = job
            pm._cancel_events[job.id] = old_event
            pm._write_jobs_unlocked()
        await pm.trans_queue.put((job, object()))

        await asyncio.wait_for(entered.wait(), timeout=2.0)

        assert observed == [old_event]
        assert pm._cancel_events[job.id] is not old_event
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_running_translation_keeps_its_own_cancel_event_after_a_retry(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / job_id))
    seen: dict[str, object] = {}

    def fake_run_translation_and_write(_video_path, _artifacts, **kwargs):
        seen["cancel_event"] = kwargs.get("cancel_event")
        return []

    monkeypatch.setattr(pm, "run_translation_and_write", fake_run_translation_and_write)

    job = JobState(
        id="retry-race-translation",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="translating",
    )
    running_event = threading.Event()
    running_event.set()
    retry_event = threading.Event()
    monkeypatch.setitem(pm._cancel_events, job.id, retry_event)

    pm._run_translation_and_write(job, object(), running_event, job.run_id)

    # Its own event, not whatever the id currently maps to.
    assert seen["cancel_event"] is running_event
    assert seen["cancel_event"].is_set()


def test_retry_mints_a_new_run_id(tmp_path, monkeypatch):
    asyncio.run(_test_retry_mints_a_new_run_id(tmp_path, monkeypatch))


async def _test_retry_mints_a_new_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        assert job.run_id
        await pm.cancel_job(job.id)
        first_run = job.run_id

        retried = await pm.retry_job(job.id)

        assert retried is not None
        assert retried.id == job.id  # the job the user sees is the same one
        assert retried.run_id and retried.run_id != first_run
    finally:
        await _reset_pm_state()


def test_stale_run_cannot_finish_the_retried_job(tmp_path, monkeypatch):
    asyncio.run(_test_stale_run_cannot_finish_the_retried_job(tmp_path, monkeypatch))


async def _test_stale_run_cannot_finish_the_retried_job(tmp_path, monkeypatch):
    # The late arrival here is the *final* state, not progress: a previous run
    # reporting done/failed after the retry started would otherwise publish its
    # own outcome (and artifact list) over the run that is still working.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    stale_job = JobState(
        id="run-guard",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-1",
        status="translating",
    )
    async with pm._state_lock:
        pm._jobs[stale_job.id] = stale_job.model_copy(
            update={"run_id": "run-2", "status": "translating"},
        )
        pm._write_jobs_unlocked()

    await pm._set_job(
        stale_job,
        status="done",
        current_stage="done",
        artifacts=["stale.srt"],
        expected_run_id="run-1",
    )

    current = await pm.get_job(stale_job.id)
    assert current is not None
    assert current.status == "translating"
    assert current.artifacts == []

    await pm._set_job(
        stale_job,
        status="done",
        current_stage="done",
        artifacts=["fresh.srt"],
        expected_run_id="run-2",
    )
    current = await pm.get_job(stale_job.id)
    assert current is not None
    assert current.status == "done"
    assert current.artifacts == ["fresh.srt"]
    await _reset_pm_state()


def test_running_translation_emits_under_the_run_it_was_given(tmp_path, monkeypatch):
    # Producers stamp the run they captured. Reading the current run at send
    # time would let a stale thread impersonate the retry that replaced it.
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / job_id))
    seen: dict[str, str] = {}

    def fake_run_translation_and_write(_video_path, _artifacts, **kwargs):
        seen["emitted_run_id"] = pm.events._current_run_id()
        seen["passed_run_id"] = str(kwargs.get("run_id") or "")
        return []

    monkeypatch.setattr(pm, "run_translation_and_write", fake_run_translation_and_write)
    job = JobState(
        id="emitting-run",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-new",
        status="translating",
    )
    # The store already holds the retry, exactly as it would while the previous
    # run's thread is still winding down.
    monkeypatch.setitem(pm._jobs, job.id, job)

    pm._run_translation_and_write(job, object(), threading.Event(), "run-old")

    assert seen["emitted_run_id"] == "run-old"
    assert seen["passed_run_id"] == "run-old"


def test_cancelling_a_running_job_is_not_yet_cancelled(tmp_path, monkeypatch):
    asyncio.run(_test_cancelling_a_running_job_is_not_yet_cancelled(tmp_path, monkeypatch))


async def _test_cancelling_a_running_job_is_not_yet_cancelled(tmp_path, monkeypatch):
    # 已请求取消 and 已取消 are different facts. Reporting the second one while a
    # worker thread is still between checkpoints is what let the page offer 重试
    # next to a run that was still spending tokens.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    running = JobState(
        id="running-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-1",
        status="translating",
    )
    queued = JobState(
        id="queued-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        run_id="run-2",
        status="queued",
    )
    async with pm._state_lock:
        pm._jobs[running.id] = running
        pm._jobs[queued.id] = queued
        pm._write_jobs_unlocked()

    assert await pm.cancel_job(running.id)
    assert await pm.cancel_job(queued.id)

    assert (await pm.get_job(running.id)).status == "cancelling"
    assert await pm.retry_job(running.id) is None
    # Nothing was executing the queued one, so there is nothing to wind down.
    assert (await pm.get_job(queued.id)).status == "cancelled"
    assert await pm.retry_job(queued.id) is not None
    await _reset_pm_state()


def test_gpu_dispatch_pauses_while_a_child_still_owns_the_card(tmp_path, monkeypatch):
    asyncio.run(_test_gpu_dispatch_pauses_while_a_child_still_owns_the_card(tmp_path, monkeypatch))


async def _test_gpu_dispatch_pauses_while_a_child_still_owns_the_card(tmp_path, monkeypatch):
    # Refusing to start a second worker only produces a failed job per attempt.
    # Dispatch pauses instead, and resumes by itself once cleanup confirms.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_GPU_BLOCKED_POLL_S", 0.02)
    await _reset_pm_state()
    blocked = {"value": True}
    monkeypatch.setattr(pm.resources, "gpu_blocked", lambda: blocked["value"])
    started = threading.Event()

    def fake_asr(job, _cancel_event, _run_id):
        started.set()
        return {"job_id": job.id}

    monkeypatch.setattr(pm, "_run_asr_alignment", fake_asr)
    monkeypatch.setattr(
        pm,
        "_run_translation_and_write",
        lambda *_args, **_kwargs: [],
    )

    worker = asyncio.create_task(pm.gpu_worker())
    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            current = await pm.get_job(job.id)
            if current is not None and current.current_stage == "gpu_blocked":
                break
            await asyncio.sleep(0.01)

        current = await pm.get_job(job.id)
        assert current is not None
        assert current.status == "queued"
        assert current.current_stage == "gpu_blocked"
        assert not started.is_set()  # nothing was handed to the GPU

        blocked["value"] = False
        deadline = time.perf_counter() + 2.0
        while not started.is_set() and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        assert started.is_set()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_cancelling_while_the_gpu_is_blocked_does_not_wait_for_it(tmp_path, monkeypatch):
    asyncio.run(_test_cancelling_while_the_gpu_is_blocked_does_not_wait_for_it(tmp_path, monkeypatch))


async def _test_cancelling_while_the_gpu_is_blocked_does_not_wait_for_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_GPU_BLOCKED_POLL_S", 0.02)
    await _reset_pm_state()
    monkeypatch.setattr(pm.resources, "gpu_blocked", lambda: True)
    monkeypatch.setattr(
        pm,
        "_run_asr_alignment",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    worker = asyncio.create_task(pm.gpu_worker())
    try:
        job = (await pm.create_job(JobSpec(video_paths=["sample.mp4"])))[0]
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            current = await pm.get_job(job.id)
            if current is not None and current.current_stage == "gpu_blocked":
                break
            await asyncio.sleep(0.01)

        assert await pm.cancel_job(job.id)
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            current = await pm.get_job(job.id)
            if current is not None and current.status == "cancelled":
                break
            await asyncio.sleep(0.01)
        current = await pm.get_job(job.id)
        assert current is not None
        assert current.status == "cancelled"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await _reset_pm_state()


def test_load_jobs_marks_active_failed_but_keeps_stage(tmp_path, monkeypatch):
    asyncio.run(_test_load_jobs_marks_active_failed_but_keeps_stage(tmp_path, monkeypatch))


def test_create_job_recreates_missing_jobs_parent(tmp_path, monkeypatch):
    asyncio.run(_test_create_job_recreates_missing_jobs_parent(tmp_path, monkeypatch))


def test_unreadable_job_records_are_kept_before_the_file_is_rewritten(
    tmp_path, monkeypatch
):
    asyncio.run(_test_unreadable_job_records_are_kept(tmp_path, monkeypatch))


async def _test_unreadable_job_records_are_kept(tmp_path, monkeypatch):
    """A record the models can no longer parse must not be erased silently.

    `load_jobs` skips what it cannot validate, and the first write after that
    rewrites the file from what loaded - so the skipped record is gone for good.
    Retiring one enum value is enough to trigger it: every job spec written
    before the 2026-08-24 tier change says `medium`, which no longer passes the
    bare `Literal`. (`none` was the example here until that change made it a
    real tier again, which is exactly how this test stops testing anything.)
    """
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr(pm, "_jobs_path", jobs_path)
    await _reset_pm_state()
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    good = JobState(
        id="readable",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="translating",
    ).model_dump()
    stale = JobState(
        id="written-by-an-older-build",
        spec=JobSpec(video_paths=["older.mp4"]),
        created_at="2026-05-03T00:00:00.000+00:00",
        status="done",
    ).model_dump()
    stale["spec"]["llm_reasoning_effort"] = "medium"
    jobs_path.write_text(
        json.dumps([good, stale], ensure_ascii=False), encoding="utf-8"
    )

    await pm.load_jobs()

    # The active job was rewritten as failed, so the file no longer holds the
    # unreadable record...
    assert await pm.get_job("written-by-an-older-build") is None
    assert "written-by-an-older-build" not in jobs_path.read_text(encoding="utf-8")
    # ...which is exactly why a copy has to exist beside it.
    backups = list(tmp_path.glob("jobs.json.rejected-*"))
    assert len(backups) == 1
    assert "written-by-an-older-build" in backups[0].read_text(encoding="utf-8")
    await _reset_pm_state()


async def _test_load_jobs_marks_active_failed_but_keeps_stage(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr(pm, "_jobs_path", jobs_path)
    await _reset_pm_state()
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(
        json.dumps(
            [
                JobState(
                    id="interrupted",
                    spec=JobSpec(video_paths=["sample.mp4"]),
                    created_at="2026-05-04T00:00:00.000+00:00",
                    status="translating",
                    current_stage="translation",
                    progress={"stage": "translation"},
                ).model_dump()
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await pm.load_jobs()

    current = await pm.get_job("interrupted")
    assert current is not None
    assert current.status == "failed"
    assert current.current_stage == "translation"
    assert "上次运行被中断" in (current.error or "")
    await _reset_pm_state()


async def _test_create_job_recreates_missing_jobs_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "missing" / "jobs.json")
    await _reset_pm_state()

    try:
        jobs = await pm.create_job(JobSpec(video_paths=["sample.mp4"]))

        assert len(jobs) == 1
        assert pm._jobs_path.exists()
    finally:
        await _reset_pm_state()


def test_retry_cancelled_job_requeues_same_job_id(tmp_path, monkeypatch):
    asyncio.run(_test_retry_cancelled_job_requeues_same_job_id(tmp_path, monkeypatch))


def test_relative_artifacts_filters_deleted_temp_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    srt_path = output_dir / "sample.srt"
    srt_path.write_text("1\n", encoding="utf-8")
    temp_dir = tmp_path / "jobs" / "job-id"
    temp_dir.mkdir(parents=True)
    deleted_timings = temp_dir / "sample.timings.json"
    existing_report = output_dir / "sample.quality.md"
    existing_report.write_text("# report\n", encoding="utf-8")
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)

    artifacts = pm._relative_artifacts(
        [
            str(srt_path),
            str(deleted_timings),
            str(existing_report.relative_to(tmp_path)),
            str(output_dir),
        ],
        str(output_dir),
    )

    assert artifacts == ["sample.srt", "sample.quality.md"]


def _sample_asr_artifacts(job: JobState, temp_dir: Path, output_dir: Path):
    return AsrArtifacts(
        segments=[{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
        audio_path=str(temp_dir / "audio" / "sample.wav"),
        job_temp_dir=str(temp_dir),
        asr_details={"transcript_chunks": []},
        aligned_segments_path=str(temp_dir / "sample.aligned_segments.json"),
        transcript_path=str(temp_dir / "sample.transcript.json"),
        asr_manifest_path=str(temp_dir / "sample.asr_manifest.json"),
        pipeline_timings={"asr_alignment_total_s": 0.0},
        logger=None,
        run_log_path=None,
        audio_cache_key="cache-key",
        video_stem="sample",
        output_dir=str(output_dir),
        srt_path=str(output_dir / "sample.srt"),
        bilingual_json_path=str(temp_dir / "sample.bilingual.json"),
        quality_report_path="",
        bilingual=False,
        timings_path=str(temp_dir / "sample.timings.json"),
        translation_cache_path=str(temp_dir / "translation_cache.jsonl"),
        asr_log=["mock asr"],
        audio_cached=True,
        device="cpu",
        backend_label="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        video_duration_s=None,
        pipeline_started=0.0,
        job_id=job.id,
    )


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


def test_retry_translation_failed_job_requeues_translation_only(tmp_path, monkeypatch):
    asyncio.run(_test_retry_translation_failed_job_requeues_translation_only(tmp_path, monkeypatch))


async def _test_retry_translation_failed_job_requeues_translation_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / "jobs" / job_id))
    await _reset_pm_state()

    job = JobState(
        id="retry-translation",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="failed",
        current_stage="translation",
        progress={"stage": "translation"},
        error="Batch translation returned invalid or incomplete JSON",
    )
    temp_dir = Path(pm._job_temp_dir(job.id))
    output_dir = tmp_path / "out"
    temp_dir.mkdir(parents=True)
    output_dir.mkdir()
    write_translation_artifacts_snapshot(
        _sample_asr_artifacts(job, temp_dir, output_dir)
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    retried = await pm.retry_job(job.id)

    assert retried is not None
    assert retried.id == job.id
    assert retried.status == "queued"
    with pytest.raises(asyncio.QueueEmpty):
        pm.gpu_queue.get_nowait()

    queued_job, queued_artifacts = pm.trans_queue.get_nowait()
    try:
        assert queued_job is retried
        assert queued_artifacts.segments[0]["text"] == "こんにちは"
        assert queued_artifacts.translation_cache_path.endswith("translation_cache.jsonl")
    finally:
        pm.trans_queue.task_done()
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


def test_remove_finished_job_leaves_retired_boundary_cache_alone(tmp_path, monkeypatch):
    asyncio.run(
        _test_remove_finished_job_leaves_retired_boundary_cache_alone(tmp_path, monkeypatch)
    )


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


def test_remove_failed_job_signals_its_cancel_event(tmp_path, monkeypatch):
    asyncio.run(_test_remove_failed_job_signals_its_cancel_event(tmp_path, monkeypatch))


async def _test_remove_failed_job_signals_its_cancel_event(tmp_path, monkeypatch):
    # "failed" is the job's status, not the worker threads'. Removal used to
    # pop the cancel event without setting it, so batch workers still running
    # after a failed translation lost their only stop signal - on 2026-09-04
    # three of them kept billing against the API minutes after the job was
    # deleted from the UI, and even shutdown could no longer reach them because
    # it signals by walking `_cancel_events`.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / "jobs" / job_id))
    await _reset_pm_state()

    job = JobState(
        id="failed-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="failed",
        current_stage="translation",
    )
    orphan_event = threading.Event()
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = orphan_event
        pm._write_jobs_unlocked()

    assert await pm.remove_job(job.id)
    assert orphan_event.is_set()
    assert job.id not in pm._cancel_events
    await _reset_pm_state()


async def _test_remove_finished_job_leaves_retired_boundary_cache_alone(tmp_path, monkeypatch):
    # Job removal used to purge `tmp/cache/boundary/`. The chain that wrote that
    # cache was retired on 2026-07-31, so the purge deleted files nothing
    # produces. Re-introducing it would mean deleting a directory whose current
    # meaning is unknown, so pin that job removal no longer reaches in there.
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_temp_dir", lambda job_id: str(tmp_path / "jobs" / job_id))
    monkeypatch.setattr(pm, "get_audio_cache_key", lambda _path: "abcdef12")
    cache_root = tmp_path / "boundary-cache"
    cache_root.mkdir()
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(cache_root))
    stale = cache_root / "abcdef12.runtime.json"
    stale.write_text("{}", encoding="utf-8")
    await _reset_pm_state()

    job = JobState(
        id="done-cache-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="done",
        current_stage="done",
    )
    temp_dir = Path(pm._job_temp_dir(job.id))
    temp_dir.mkdir(parents=True)
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    assert await pm.remove_job(job.id)
    assert not temp_dir.exists()
    assert stale.exists()
    assert not hasattr(pm, "delete_for_audio_cache_key")
    await _reset_pm_state()


def test_cancel_active_job_keeps_job_temp_dir_for_retry(tmp_path, monkeypatch):
    asyncio.run(_test_cancel_active_job_keeps_job_temp_dir_for_retry(tmp_path, monkeypatch))


def test_remove_finished_jobs_deletes_temp_dirs_outside_state_lock(tmp_path, monkeypatch):
    asyncio.run(_test_remove_finished_jobs_deletes_temp_dirs_outside_state_lock(tmp_path, monkeypatch))


def test_evict_old_jobs_deletes_temp_dir_outside_state_lock(tmp_path, monkeypatch):
    asyncio.run(_test_evict_old_jobs_deletes_temp_dir_outside_state_lock(tmp_path, monkeypatch))


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
    # 已请求取消, not 已取消: the translation thread still owns this run, and the
    # card must not offer 重试 until it reports that it stopped.
    assert current.status == "cancelling"
    await _reset_pm_state()


async def _test_remove_finished_jobs_deletes_temp_dirs_outside_state_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    removed: list[str] = []

    def fake_remove_job_temp_dir(job_id: str) -> None:
        assert not pm._state_lock.locked()
        removed.append(job_id)

    monkeypatch.setattr(pm, "_remove_job_temp_dir", fake_remove_job_temp_dir)

    job = JobState(
        id="done-outside-lock",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2026-05-04T00:00:00.000+00:00",
        status="done",
        current_stage="done",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._cancel_events[job.id] = threading.Event()
        pm._write_jobs_unlocked()

    assert await pm.remove_finished_jobs() == 1
    assert removed == [job.id]
    assert await pm.get_job(job.id) is None
    await _reset_pm_state()


async def _test_evict_old_jobs_deletes_temp_dir_outside_state_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    removed: list[str] = []

    def fake_remove_job_temp_dir(job_id: str) -> None:
        assert not pm._state_lock.locked()
        removed.append(job_id)

    monkeypatch.setattr(pm, "_remove_job_temp_dir", fake_remove_job_temp_dir)
    job = JobState(
        id="expired-job",
        spec=JobSpec(video_paths=["sample.mp4"]),
        created_at="2020-01-01T00:00:00.000+00:00",
        status="done",
        current_stage="done",
    )
    async with pm._state_lock:
        pm._jobs[job.id] = job
        pm._write_jobs_unlocked()

    assert await pm.evict_old_jobs(max_age_hours=1) == 1
    assert removed == [job.id]
    assert await pm.get_job(job.id) is None
    await _reset_pm_state()
