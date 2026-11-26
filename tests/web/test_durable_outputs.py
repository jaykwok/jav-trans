"""Faults at the real state, publication and download boundaries."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

import main
from pipeline import artifact_store as store
from pipeline.artifacts import AsrArtifacts
from web import pipeline_manager as pm
from web.app import create_app
from web.models import JobSpec, JobState
from web.routes import files


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(files, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
    monkeypatch.setattr(pm, "_jobs", {})
    monkeypatch.setattr(pm, "_cancel_events", {})
    monkeypatch.setattr(pm, "_tombstones", {})
    monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
    monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())
    return tmp_path


def _outputs(root, run, *, job_id="sample-job"):
    lease = store.claim_output(root / "out" / "sample-a.srt", project_root=root, job_id=job_id, run_id=run)
    outputs = store.RunOutputs(lease=lease)
    outputs.stage(lease.target, name="srt", write=lambda p: p.write_text(run, "utf-8"))
    outputs.stage_json(root / "out" / "sample-a.json", {"run": run}, name="transcript")
    return outputs


def test_disk_failure_never_acknowledges_done_and_retry_only_saves(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)))[0]
        await pm._set_job(job, status="translating")
        await asyncio.to_thread(pm._job_store_writer.flush)
        old_sequence = pm._job_store_writer._written
        replace = Path.replace

        def fail_jobs(self, target):
            if Path(target) == pm._jobs_path:
                raise PermissionError("injected disk failure")
            return replace(self, target)

        with monkeypatch.context() as fault:
            fault.setattr(Path, "replace", fail_jobs)
            with pytest.raises(pm.JobPersistenceError):
                await pm._set_job(job, status="done", expected_run_id=job.run_id)
            assert pm._job_store_writer._written == old_sequence
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
                response = await client.get(f"/api/jobs/{job.id}")
                assert response.json()["status"] == "save_failed"
                retry = await client.post(f"/api/jobs/{job.id}/retry")
                assert retry.status_code == 503
            assert json.loads(pm._jobs_path.read_text("utf-8"))[0]["status"] == "translating"
        gpu_pending, translation_pending = pm.gpu_queue.qsize(), pm.trans_queue.qsize()
        saved = await pm.retry_job(job.id)
        assert saved.status == "done"
        assert saved.run_id == job.run_id
        assert (pm.gpu_queue.qsize(), pm.trans_queue.qsize()) == (gpu_pending, translation_pending)
        assert json.loads(pm._jobs_path.read_text("utf-8"))[0]["status"] == "done"

    asyncio.run(scenario())


def test_slow_terminal_write_keeps_reads_and_stop_signal_responsive(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")])))[0]
        writing, release = threading.Event(), threading.Event()
        original = pm._write_jobs_snapshot

        def slow(snapshot, **kwargs):
            writing.set()
            assert release.wait(3)
            original(snapshot, **kwargs)

        monkeypatch.setattr(pm, "_write_jobs_snapshot", slow)
        finished = asyncio.create_task(pm._set_job(job, status="done"))
        try:
            assert await asyncio.to_thread(writing.wait, 2)
            visible = await asyncio.wait_for(pm.get_job(job.id), 0.2)
            assert visible.status == "saving"
            _, pending_revision = await pm.jobs_snapshot()
            cancellation = asyncio.create_task(pm.cancel_job(job.id))
            await asyncio.sleep(0.02)
            assert pm._cancel_events[job.id].is_set()
            assert not finished.done()
        finally:
            release.set()
        await finished
        await cancellation
        _, committed_revision = await pm.jobs_snapshot()
        assert committed_revision > pending_revision

    asyncio.run(scenario())


def test_failed_commit_leaves_previous_generation_and_download_intact(workspace, monkeypatch):
    old = _outputs(workspace, "old")
    old_manifest = old.publish()
    store.export_output(old.lease)
    new = _outputs(workspace, "new")
    write = store.write_json_atomic

    def fail_commit(path, payload, **kwargs):
        if Path(path).name == "current.json":
            raise PermissionError("injected manifest failure")
        return write(path, payload, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(store, "write_json_atomic", fail_commit)
        with pytest.raises(store.OutputCommitError):
            new.publish()
    assert store.read_run_manifest(new.lease) is None
    assert store.read_run_manifest(old.lease) == old_manifest
    assert old.lease.target.read_text("utf-8") == "old"
    pm._jobs["sample-job"] = JobState(
        id="sample-job", run_id="old", created_at="2026-09-07", status="done",
        spec=JobSpec(video_paths=[str(workspace / "sample-a.mp4")], output_dir=str(workspace / "out")),
        output_target=str(old.lease.target), output_generation=old.lease.sequence,
    )

    async def download():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            response = await client.get("/api/output/sample-job/sample-a.srt")
            assert response.status_code == 200
            assert response.text == "old"
    asyncio.run(download())


def test_slow_old_publisher_cannot_overwrite_new_owner(workspace, monkeypatch):
    slow = _outputs(workspace, "old")
    paused, release = threading.Event(), threading.Event()
    digest = store._sha256
    errors = []

    def hold_old(path):
        if slow.staging_dir in path.parents and path.suffix == ".srt":
            paused.set()
            assert release.wait(3)
        return digest(path)

    monkeypatch.setattr(store, "_sha256", hold_old)

    def publish_old():
        try:
            slow.publish()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=publish_old)
    thread.start()
    try:
        assert paused.wait(2)
        fast = _outputs(workspace, "new", job_id="another-job")
        fast.publish()
        store.export_output(fast.lease)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], store.OutputsSupersededError)
    assert fast.lease.target.read_text("utf-8") == "new"


def test_export_uses_a_temp_on_destination_volume_and_can_retry(workspace, monkeypatch):
    outputs = _outputs(workspace, "new")
    outputs.publish()
    replace = store.replace_atomically
    moves = []

    def fail_export(source, target, **kwargs):
        if target == outputs.lease.target:
            assert source.parent == target.parent
            moves.append((source, target))
            raise PermissionError("injected sharing violation")
        return replace(source, target, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(store, "replace_atomically", fail_export)
        with pytest.raises(store.OutputExportError):
            store.export_output(outputs.lease)
    assert moves and store.read_run_manifest(outputs.lease, verify=True)
    store.export_output(outputs.lease)
    assert outputs.lease.target.read_text("utf-8") == "new"


@pytest.mark.parametrize("exported", [False, True])
def test_restart_recovers_committed_outputs_without_requeuing(workspace, exported):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], output_dir=str(workspace / "out"))))[0]
        outputs = store.RunOutputs(lease=pm.output_lease_for_job(job))
        outputs.stage(outputs.lease.target, name="srt", write=lambda p: p.write_text("paid result", "utf-8"))
        outputs.publish()
        if exported:
            store.export_output(outputs.lease)
        pm._jobs.clear()
        await pm.load_jobs()
        restored = await pm.get_job(job.id)
        assert restored.status == ("done" if exported else "export_failed")
        assert restored.artifacts
        if not exported:
            queued = pm.gpu_queue.qsize()
            restored = await pm.retry_job(job.id)
            assert restored.status == "done"
            assert pm.gpu_queue.qsize() == queued
    asyncio.run(scenario())


def _empty_artifacts(root, job):
    temp = root / "tmp" / "jobs" / job.id
    return AsrArtifacts(
        segments=[], audio_path=str(temp / "sample-a.wav"), job_temp_dir=str(temp),
        asr_details={"transcript_chunks": [], "pipeline_cuda_memory_skip_reason": "test"},
        aligned_segments_path=str(temp / "sample-a.aligned_segments.json"),
        transcript_path=str(temp / "sample-a.transcript.json"),
        asr_manifest_path=str(temp / "sample-a.asr_manifest.json"),
        pipeline_timings={}, logger=None, run_log_path=None, audio_cache_key="test",
        video_stem="sample-a", output_dir=str(root / "out"), srt_path=job.output_target,
        bilingual_json_path=str(temp / "sample-a.bilingual.json"),
        bilingual=True, timings_path=str(temp / "sample-a.timings.json"),
        translation_cache_path=str(temp / "translation_cache.jsonl"), asr_log=[],
        audio_cached=True, device="cpu", backend_label="test", video_duration_s=0.0,
        pipeline_started=time.perf_counter(), job_id=job.id, aligned_cache_signature={},
    )


def test_real_empty_main_recovers_prepared_generation_before_backend_or_compute(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(
            video_paths=[str(workspace / "sample-a.mp4")], output_dir=str(workspace / "out"),
        )))[0]
        artifacts = _empty_artifacts(workspace, job)
        ctx = pm._job_context(job)
        write = store.write_json_atomic

        def fail_commit(path, payload, **kwargs):
            if Path(path).name == "current.json" and payload.get("commits"):
                raise PermissionError("commit denied")
            return write(path, payload, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(store, "write_json_atomic", fail_commit)
            with pytest.raises(store.OutputCommitError):
                await asyncio.to_thread(main.run_translation_and_write, job.spec.video_paths[0], artifacts, ctx=ctx, job_id=job.id, run_id=job.run_id)
        lease = pm.output_lease_for_job(job)
        assert store.read_prepared_manifest(lease)
        assert store.read_run_manifest(lease) is None
        assert not lease.target.exists()

        def no_compute(*args, **kwargs):
            raise AssertionError("saved generation must not enter computation or acquire a backend")

        monkeypatch.setattr(main, "_run_translation_and_write_impl", no_compute)
        monkeypatch.setattr(main.llm_backends, "backend_lease", no_compute)
        paths = await asyncio.to_thread(main.run_translation_and_write, job.spec.video_paths[0], artifacts, ctx=ctx, job_id=job.id, run_id=job.run_id)
        assert all(Path(path).is_file() for path in paths)
        assert lease.target.is_file()
        await pm._set_job(job, status="done", artifacts=paths)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            quality = await client.get(f"/api/quality/{job.id}")
            assert quality.status_code == 404
    asyncio.run(scenario())


def test_restart_retries_uncommitted_generation_without_translation_preflight(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")])))[0]
        outputs = store.RunOutputs(lease=pm.output_lease_for_job(job))
        outputs.stage(outputs.lease.target, name="srt", write=lambda p: p.write_text("saved translation", "utf-8"))
        write = store.write_json_atomic
        with monkeypatch.context() as fault:
            def fail(path, payload, **kwargs):
                if Path(path).name == "current.json":
                    raise PermissionError("commit denied")
                return write(path, payload, **kwargs)
            fault.setattr(store, "write_json_atomic", fail)
            with pytest.raises(store.OutputCommitError):
                outputs.publish()
        pm._jobs.clear()
        await pm.load_jobs()
        assert (await pm.get_job(job.id)).status == "publish_failed"
        from web.routes import jobs as jobs_routes
        monkeypatch.setattr(jobs_routes, "translation_config_problems", lambda: ["no API configuration"])
        pending = (pm.gpu_queue.qsize(), pm.trans_queue.qsize())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            response = await client.post(f"/api/jobs/{job.id}/retry")
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "done"
            assert response.json()["run_id"] == job.run_id
        assert (pm.gpu_queue.qsize(), pm.trans_queue.qsize()) == pending
        assert outputs.lease.target.read_text("utf-8") == "saved translation"
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["bytes", "manifest", "owner"])
def test_corrupt_committed_outputs_refuse_download_and_recomputation(workspace, damage):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)))[0]
        lease = pm.output_lease_for_job(job)
        outputs = store.RunOutputs(lease=lease)
        outputs.stage(lease.target, name="srt", write=lambda p: p.write_text("good", "utf-8"))
        manifest = outputs.publish()
        store.export_output(lease)
        paths = store.manifest_paths(manifest, project_root=workspace)
        await pm._set_job(job, status="failed", artifacts=paths)
        if damage == "bytes":
            Path(paths[0]).write_text("evil", "utf-8")  # same size, different hash
        elif damage == "manifest":
            (Path(paths[0]).parent.parent / "manifest.json").write_text("{", "utf-8")
        else:
            (store.output_store_root(lease.target, project_root=workspace) / "current.json").write_text("{", "utf-8")
        pending = (pm.gpu_queue.qsize(), pm.trans_queue.qsize())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            downloaded = await client.get(f"/api/output/{job.id}/{lease.target.name}")
            retried = await client.post(f"/api/jobs/{job.id}/retry")
            assert downloaded.status_code == retried.status_code == 409
        assert (pm.gpu_queue.qsize(), pm.trans_queue.qsize()) == pending
    asyncio.run(scenario())


def test_failed_deletion_keeps_job_and_never_issues_a_tombstone(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)))[0]
        await pm._set_job(job, status="failed")
        _, old_revision = await pm.jobs_snapshot()
        with monkeypatch.context() as fault:
            fault.setattr(pm, "_write_jobs_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("disk full")))
            with pytest.raises(pm.JobPersistenceError):
                await pm.remove_job(job.id)
        assert await pm.get_job(job.id) is not None
        assert pm.deleted_revision(job.id) is None
        assert (await pm.jobs_snapshot())[1] == old_revision
        assert json.loads(pm._jobs_path.read_text("utf-8"))[0]["id"] == job.id
        assert await pm.remove_job(job.id)
        assert pm.deleted_revision(job.id) is not None
    asyncio.run(scenario())


def test_list_revision_only_advances_after_deletion_confirmation(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)))[0]
        await pm._set_job(job, status="failed")
        entered, release = threading.Event(), threading.Event()
        write = pm._write_jobs_snapshot

        def slow(snapshot, **kwargs):
            entered.set()
            assert release.wait(3)
            write(snapshot, **kwargs)

        monkeypatch.setattr(pm, "_write_jobs_snapshot", slow)
        removal = asyncio.create_task(pm.remove_job(job.id))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            before, before_revision = await pm.jobs_snapshot()
            assert any(item.id == job.id for item in before)
        finally:
            release.set()
        await removal
        after, after_revision = await pm.jobs_snapshot()
        assert not after
        assert after_revision > before_revision
    asyncio.run(scenario())


def test_cancellation_before_commit_revokes_publication(workspace):
    outputs = _outputs(workspace, "cancelled-run")
    store.revoke_output(outputs.lease)
    with pytest.raises(store.OutputsSupersededError):
        outputs.publish()
    assert store.read_run_manifest(outputs.lease) is None
    assert not outputs.lease.target.exists()


def test_cancellation_after_commit_cannot_invalidate_saved_work(workspace):
    outputs = _outputs(workspace, "committed-run")
    outputs.publish()
    assert store.revoke_output(outputs.lease) is False
    store.export_output(outputs.lease)
    assert outputs.lease.target.read_text("utf-8") == "committed-run"


def test_unreadable_job_file_does_not_start_with_an_empty_database(workspace):
    pm._jobs_path.write_text("{incomplete", "utf-8")
    with pytest.raises(pm.JobPersistenceError):
        asyncio.run(pm.load_jobs())
    assert pm._jobs_path.read_text("utf-8") == "{incomplete"


def test_worker_reports_commit_failure_and_retry_never_requeues(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)))[0]
        write = store.write_json_atomic
        calls = []

        def compute(job, _asr, _cancel, _run):
            calls.append(job.id)
            outputs = store.RunOutputs(lease=pm.output_lease_for_job(job))
            outputs.stage(outputs.lease.target, name="srt", write=lambda p: p.write_text("already calculated", "utf-8"))
            outputs.publish()

        monkeypatch.setattr(pm, "_run_translation_and_write", compute)
        with monkeypatch.context() as fault:
            def fail(path, payload, **kwargs):
                if Path(path).name == "current.json":
                    raise PermissionError("commit denied")
                return write(path, payload, **kwargs)
            fault.setattr(store, "write_json_atomic", fail)
            worker = asyncio.create_task(pm.translation_worker())
            await pm.trans_queue.put((job, object()))
            await asyncio.wait_for(pm.trans_queue.join(), 3)
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert (await pm.get_job(job.id)).status == "publish_failed"
        saved = await pm.retry_job(job.id)
        assert saved.status == "done"
        assert calls == [job.id]
        assert pm.trans_queue.empty()
    asyncio.run(scenario())


def test_job_submissions_share_active_work_and_reject_output_collisions(workspace):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            spec = {"video_paths": ["sample-a.mp4"], "output_dir": "out", "skip_translation": True}
            alias = {**spec, "video_paths": [str(workspace / "sample-a.mp4")], "output_dir": str(workspace / "out")}
            first, second = await asyncio.gather(client.post("/api/jobs", json=spec), client.post("/api/jobs", json=alias))
            assert first.status_code == second.status_code == 201
            assert first.json() == second.json()
            old_id = first.json()["ids"][0]
            repeated = await client.post("/api/jobs", json={**spec, "video_paths": ["sample-a.mp4"] * 2})
            assert repeated.json()["ids"] == [old_id, old_id]
            assert pm.gpu_queue.qsize() == 1
            old = await pm.get_job(old_id)
            lease = pm.output_lease_for_job(old)

            internal_collision = await client.post("/api/jobs", json={
                **spec, "video_paths": ["sample-b.mp4", "other/sample-b.mp4"],
            })
            assert internal_collision.status_code == 409
            assert internal_collision.json()["job_id"] is None

            rejected_batch = await client.post("/api/jobs", json={
                **spec, "video_paths": ["sample-b.mp4", "other/sample-a.mp4"],
            })
            assert rejected_batch.status_code == 409
            assert rejected_batch.json()["job_id"] == old_id
            assert [job.id for job in await pm.list_jobs()] == [old_id]
            assert pm.gpu_queue.qsize() == 1
            assert store.owns_output(lease)

            await pm._set_job(old, status="failed")
            new = await client.post("/api/jobs", json=spec)
            assert new.status_code == 201
            new_id = new.json()["ids"][0]
            assert new_id != old_id
            retry_conflict = await client.post(f"/api/jobs/{old_id}/retry")
            assert retry_conflict.status_code == 409
            assert retry_conflict.json()["job_id"] == new_id

    asyncio.run(scenario())


@pytest.mark.parametrize("cue_count", [0, 1])
def test_export_recovery_preserves_cue_count_and_cleans_temp(workspace, cue_count):
    async def scenario():
        spec = {
            "video_paths": ["sample-a.mp4"], "output_dir": "out", "skip_translation": True,
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            submitted = await client.post("/api/jobs", json=spec)
            assert submitted.status_code == 201
            job = pm.gpu_queue.get_nowait()
            pm.gpu_queue.task_done()
            assert submitted.json()["ids"] == [job.id]

            temp = Path(pm._job_temp_dir(job.id))
            temp.mkdir(parents=True)
            wav = temp / "sample-a.wav"
            wav.write_bytes(b"synthetic audio marker")
            artifacts = _empty_artifacts(workspace, job)
            artifacts.job_temp_dir = str(temp)
            artifacts.audio_path = str(wav)
            if cue_count:
                artifacts.segments = [{"start": 0.2, "end": 1.8, "text": "こんにちは。"}]
                artifacts.video_duration_s = 2.0
            target = Path(job.output_target)
            target.mkdir(parents=True)

            worker = asyncio.create_task(pm.translation_worker())
            try:
                await pm.trans_queue.put((job, artifacts))
                await asyncio.wait_for(pm.trans_queue.join(), 8)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

            state = (await client.get(f"/api/jobs/{job.id}")).json()
            assert state["status"] == "export_failed"
            assert state["subtitle_cue_count"] == cue_count
            assert wav.is_file()
            repeated = await client.post("/api/jobs", json=spec)
            assert repeated.json()["ids"] == [job.id]
            assert pm.gpu_queue.empty()
            competing = await client.post("/api/jobs", json={**spec, "video_paths": ["other/sample-a.mp4"]})
            assert competing.status_code == 409

            await pm.load_jobs()
            restored = (await client.get(f"/api/jobs/{job.id}")).json()
            assert restored["status"] == "export_failed"
            assert restored["subtitle_cue_count"] == cue_count
            target.rmdir()
            retry = await client.post(f"/api/jobs/{job.id}/retry")
            assert retry.status_code == 200
            assert retry.json()["status"] == "done"
            assert retry.json()["run_id"] == job.run_id
            assert retry.json()["subtitle_cue_count"] == cue_count
            assert target.read_text(encoding="utf-8-sig").count("-->") == cue_count
            assert not wav.exists()
            assert pm.gpu_queue.empty() and pm.trans_queue.empty()

    asyncio.run(scenario())


def test_load_jobs_strips_retired_diagnostic_spec_fields(workspace):
    async def scenario():
        legacy_record = {
            "id": "legacy-job-1",
            "status": "done",
            "created_at": "2026-09-01T00:00:00Z",
            "spec": {
                "video_paths": [str(workspace / "sample-a.mp4")],
                "output_dir": str(workspace / "out"),
                "subtitle_mode": "zh",
                "skip_translation": True,
                "keep_quality_report": True,
                "keep_temp_files": True,
                "advanced": {"RUN_LOG_ENABLED": "0"},
            },
        }
        pm._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        pm._jobs_path.write_text(json.dumps([legacy_record]), encoding="utf-8")
        pm._jobs.clear()
        await pm.load_jobs()
        loaded = await pm.get_job("legacy-job-1")
        assert loaded is not None
        assert not list(pm._jobs_path.parent.glob("*.rejected-*"))

    asyncio.run(scenario())


def test_rejected_job_creation_restores_previous_output_owner(workspace, monkeypatch):
    async def scenario():
        spec = JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)
        old = (await pm.create_job(spec))[0]
        outputs = store.RunOutputs(lease=pm.output_lease_for_job(old))
        outputs.stage(outputs.lease.target, name="srt", write=lambda path: path.write_text("old work", "utf-8"))
        await pm._set_job(old, status="failed")
        with monkeypatch.context() as fault:
            fault.setattr(pm, "_write_jobs_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("cannot save new job")))
            with pytest.raises(pm.JobPersistenceError):
                await pm.create_job(spec)
        assert [job.id for job in await pm.list_jobs()] == [old.id]
        outputs.publish()
        store.export_output(outputs.lease)
        assert outputs.lease.target.read_text("utf-8") == "old work"
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["target", "run", "sequence", "different_target"])
def test_broken_output_identity_never_falls_back_to_recomputation(workspace, damage):
    async def scenario():
        job = (await pm.create_job(JobSpec(
            video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True,
        )))[0]
        lease = pm.output_lease_for_job(job)
        outputs = store.RunOutputs(lease=lease)
        outputs.stage(lease.target, name="srt", write=lambda path: path.write_text("saved", "utf-8"))
        manifest = outputs.publish()
        await pm._set_job(job, status="failed", artifacts=store.manifest_paths(manifest, project_root=workspace))
        if damage == "target":
            job.output_target = ""
        elif damage == "run":
            job.run_id = ""
        elif damage == "sequence":
            job.output_generation = 0
        else:
            job.output_target = str(workspace / "another.srt")
        queued = (pm.gpu_queue.qsize(), pm.trans_queue.qsize())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            download = await client.get(f"/api/output/{job.id}/{lease.target.name}")
            retry = await client.post(f"/api/jobs/{job.id}/retry")
            assert download.status_code == retry.status_code == 409
        assert (pm.gpu_queue.qsize(), pm.trans_queue.qsize()) == queued
    asyncio.run(scenario())


def test_worker_keeps_committed_result_when_export_is_overtaken(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(
            video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True,
        )))[0]
        lease = pm.output_lease_for_job(job)

        def compute(*_args):
            old = store.RunOutputs(lease=lease)
            old.stage(lease.target, name="srt", write=lambda path: path.write_text("old result", "utf-8"))
            old.publish()
            new_lease = store.claim_output(lease.target, project_root=workspace, job_id="new-job", run_id="new-run")
            new = store.RunOutputs(lease=new_lease)
            new.stage(lease.target, name="srt", write=lambda path: path.write_text("new result", "utf-8"))
            new.publish()
            store.export_output(new_lease)
            store.export_output(lease)

        monkeypatch.setattr(pm, "_run_translation_and_write", compute)
        worker = asyncio.create_task(pm.translation_worker())
        try:
            await pm.trans_queue.put((job, object()))
            await asyncio.wait_for(pm.trans_queue.join(), 3)
            assert not worker.done()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        saved = await pm.get_job(job.id)
        assert saved.status == "export_failed"
        assert saved.artifacts and "接管" in saved.error
        queued = (pm.gpu_queue.qsize(), pm.trans_queue.qsize())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            download = await client.get(f"/api/output/{job.id}/{lease.target.name}")
            assert download.status_code == 200 and download.text == "old result"
            retry = await client.post(f"/api/jobs/{job.id}/retry")
            assert retry.status_code == 409
        assert lease.target.read_text("utf-8") == "new result"
        assert (pm.gpu_queue.qsize(), pm.trans_queue.qsize()) == queued
    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["create", "delete"])
def test_timed_out_state_write_cannot_commit_after_request_failed(workspace, monkeypatch, operation):
    async def scenario():
        spec = JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)
        job = (await pm.create_job(spec))[0]
        await pm._set_job(job, status="failed")
        before = pm._jobs_path.read_bytes()
        before_written = pm._job_store_writer._written
        entered, release = threading.Event(), threading.Event()
        write = pm._write_jobs_snapshot
        pm._job_store_writer._CONFIRM_TIMEOUT_S = 0.1

        def slow(snapshot, **kwargs):
            entered.set()
            assert release.wait(3)
            return write(snapshot, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(pm, "_write_jobs_snapshot", slow)
            change = asyncio.create_task(pm.create_job(spec) if operation == "create" else pm.remove_job(job.id))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                with pytest.raises(pm.JobPersistenceError, match="超时"):
                    await change
                assert [item.id for item in await pm.list_jobs()] == [job.id]
            finally:
                release.set()
            # The single writer is allowed to unwind, but the rejected mutation
            # must not become durable later or revive at the next startup.
            pm._job_store_writer._CONFIRM_TIMEOUT_S = 3
            try:
                await asyncio.to_thread(pm._job_store_writer.flush)
            except pm.JobPersistenceError:
                pass
        assert pm._jobs_path.read_bytes() == before
        assert pm._job_store_writer._written == before_written
        assert pm.deleted_revision(job.id) is None
    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_started_atomic_replace_is_settled_before_state_transaction_finishes(workspace, monkeypatch, interruption):
    async def scenario():
        job = (await pm.create_job(JobSpec(
            video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True,
        )))[0]
        entered, release = threading.Event(), threading.Event()
        confirming, settling = asyncio.Event(), asyncio.Event()
        replace = Path.replace
        writer = pm._job_store_writer
        confirm, settle_commit = writer.confirm, writer._settle_commit

        async def confirm_after_replace_started(receipt):
            # Disk preparation may exceed 50 ms on a hosted Windows runner.
            # Start the short deadline only once the atomic replace has begun.
            assert await asyncio.to_thread(entered.wait, 5)
            writer._CONFIRM_TIMEOUT_S = 0.05
            confirming.set()
            await confirm(receipt)

        async def observe_settling(wrapped):
            settling.set()
            await settle_commit(wrapped)

        def slow_replace(path, target):
            if Path(target) == pm._jobs_path:
                entered.set()
                assert release.wait(10)
            return replace(path, target)

        monkeypatch.setattr(writer, "confirm", confirm_after_replace_started)
        monkeypatch.setattr(writer, "_settle_commit", observe_settling)
        monkeypatch.setattr(Path, "replace", slow_replace)
        save = asyncio.create_task(pm._set_job(job, status="done"))
        try:
            await asyncio.wait_for(confirming.wait(), 5)
            if interruption == "cancel":
                save.cancel()
            await asyncio.wait_for(settling.wait(), 5)
            assert not save.done()
            assert (await pm.get_job(job.id)).status == "saving"
        finally:
            release.set()
            await asyncio.gather(save, return_exceptions=True)
        await save
        assert (await pm.get_job(job.id)).status == "done"
        assert json.loads(pm._jobs_path.read_text("utf-8"))[0]["status"] == "done"
    asyncio.run(scenario())


def test_upgrade_migration_relocates_legacy_outputs_keeps_completed_and_drops_unfinished(
    tmp_path, monkeypatch
):
    """1.8.x in-place upgrade moves tmp/outputs -> outputs, keeps completed and
    export-retryable job cards downloadable, and clears unfinished legacy jobs."""
    import shutil

    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(files, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "cache" / "web" / "jobs.json")
    monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
    monkeypatch.setattr(pm, "_jobs", {})
    monkeypatch.setattr(pm, "_cancel_events", {})
    monkeypatch.setattr(pm, "_tombstones", {})
    monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
    monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())

    def _create_legacy_committed(target_name: str, *, job_id: str, run_id: str, export: bool):
        target = tmp_path / "out" / target_name
        lease = store.claim_output(target, project_root=tmp_path, job_id=job_id, run_id=run_id)
        outputs = store.RunOutputs(lease=lease)
        outputs.stage(target, name="srt", write=lambda p: p.write_text(f"srt:{run_id}", "utf-8"))
        manifest = outputs.publish()
        if export:
            store.export_output(lease)
        # Relocate the freshly created store back to the 1.8.x tmp/outputs/ layout
        # and rewrite manifest.json back to tmp/outputs/ to simulate a real 1.8.4 install.
        new_store_dir = store.output_store_root(target, project_root=tmp_path)
        legacy_store_dir = tmp_path / "tmp" / "outputs" / new_store_dir.name
        legacy_store_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(new_store_dir), str(legacy_store_dir))
        for mpath in legacy_store_dir.glob("generations/*/manifest.json"):
            mdata = json.loads(mpath.read_text(encoding="utf-8"))
            mdata["files"] = {
                k: ("tmp/" + v if isinstance(v, str) and v.startswith("outputs/") else v)
                for k, v in mdata["files"].items()
            }
            mpath.write_text(json.dumps(mdata), encoding="utf-8")
        legacy_artifacts = [
            "tmp/" + v if isinstance(v, str) and v.startswith("outputs/") else v
            for v in manifest["files"].values()
        ]
        return target, lease, legacy_artifacts

    done_target, done_lease, done_artifacts = _create_legacy_committed(
        "sample-done.ja.srt", job_id="legacy-done", run_id="run-done", export=True
    )
    export_target, export_lease, export_artifacts = _create_legacy_committed(
        "sample-export.ja.srt", job_id="legacy-export", run_id="run-export", export=False
    )
    if (tmp_path / "outputs").exists():
        shutil.rmtree(tmp_path / "outputs")

    # Simulate legacy scratch dirs for unfinished jobs that should be cleared.
    for stale_id in ("legacy-active", "legacy-failed"):
        wav = tmp_path / "tmp" / "web" / "jobs" / stale_id / "audio" / "source.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"wav")

    legacy_records = [
        {
            "id": "legacy-done",
            "status": "done",
            "created_at": "2026-09-01T00:00:00Z",
            "run_id": done_lease.run_id,
            "output_target": str(done_target),
            "output_generation": done_lease.sequence,
            "artifacts": done_artifacts,
            "spec": {
                "video_paths": [str(tmp_path / "sample-done.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
                "keep_quality_report": True,
            },
        },
        {
            "id": "legacy-export",
            "status": "export_failed",
            "created_at": "2026-09-01T00:01:00Z",
            "run_id": export_lease.run_id,
            "output_target": str(export_target),
            "output_generation": export_lease.sequence,
            "artifacts": export_artifacts,
            "spec": {
                "video_paths": [str(tmp_path / "sample-export.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
            },
        },
        {
            "id": "legacy-active",
            "status": "translating",
            "created_at": "2026-09-01T00:02:00Z",
            "run_id": "run-active",
            "spec": {
                "video_paths": [str(tmp_path / "sample-active.mp4")],
                "skip_translation": True,
            },
        },
        {
            "id": "legacy-failed",
            "status": "failed",
            "created_at": "2026-09-01T00:03:00Z",
            "run_id": "run-failed",
            "spec": {
                "video_paths": [str(tmp_path / "sample-failed.mp4")],
                "skip_translation": True,
            },
        },
    ]
    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(json.dumps(legacy_records), encoding="utf-8")
    (tmp_path / "tmp" / "web" / pm._EPOCH_FILE_NAME).write_text('{"epoch": 123456}', encoding="utf-8")

    async def scenario():
        await pm.load_jobs()
        jobs_by_id = {job.id: job for job in await pm.list_jobs()}
        assert set(jobs_by_id) == {"legacy-done", "legacy-export"}
        assert jobs_by_id["legacy-done"].status == "done"
        assert jobs_by_id["legacy-export"].status == "export_failed"
        assert not (tmp_path / "tmp" / "outputs").exists()
        assert not legacy_jobs_path.exists()
        assert not (tmp_path / "tmp" / "web" / "jobs" / "legacy-active").exists()
        assert not (tmp_path / "tmp" / "web" / "jobs" / "legacy-failed").exists()
        assert (tmp_path / "cache" / "web" / pm._EPOCH_FILE_NAME).is_file()

        # Migrated completed job is downloadable via the live API with full hash verification.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            download = await client.get("/api/output/legacy-done/sample-done.ja.srt")
            assert download.status_code == 200
            assert download.text == "srt:run-done"

            # Migrated export_failed job can retry export cleanly without recomputing.
            retry = await client.post("/api/jobs/legacy-export/retry")
            assert retry.status_code == 200
            assert retry.json()["status"] == "done"
            assert export_target.read_text(encoding="utf-8") == "srt:run-export"

    asyncio.run(scenario())


def _setup_legacy_upgrade_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(files, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "cache" / "web" / "jobs.json")
    monkeypatch.setattr(pm, "_job_store_writer", pm._JobStoreWriter())
    monkeypatch.setattr(pm, "_jobs", {})
    monkeypatch.setattr(pm, "_cancel_events", {})
    monkeypatch.setattr(pm, "_tombstones", {})
    monkeypatch.setattr(pm, "gpu_queue", asyncio.Queue())
    monkeypatch.setattr(pm, "trans_queue", asyncio.Queue())


def _relocate_store_to_legacy(tmp_path: Path, target: Path) -> Path:
    import shutil

    new_store_dir = store.output_store_root(target, project_root=tmp_path)
    legacy_store_dir = tmp_path / "tmp" / "outputs" / new_store_dir.name
    legacy_store_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(new_store_dir), str(legacy_store_dir))
    for mpath in legacy_store_dir.glob("generations/*/manifest.json"):
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        mdata["files"] = {
            k: ("tmp/" + v if isinstance(v, str) and v.startswith("outputs/") else v)
            for k, v in mdata["files"].items()
        }
        mpath.write_text(json.dumps(mdata), encoding="utf-8")
    if (tmp_path / "outputs").exists() and not any((tmp_path / "outputs").iterdir()):
        (tmp_path / "outputs").rmdir()
    return legacy_store_dir


def test_migrate_legacy_outputs_survives_manifest_lock_and_repairs_already_moved_store(
    tmp_path, monkeypatch
):
    """Manifest rewrite happens before directory publish, and restart also repairs
    any store already under outputs/ whose manifest still points to tmp/outputs/."""
    import shutil

    # (1) Lock on manifest.json write leaves store in tmp/outputs/ so retry completes cleanly.
    target1 = tmp_path / "out" / "sample-lock.ja.srt"
    lease1 = store.claim_output(target1, project_root=tmp_path, job_id="j1", run_id="r1")
    outputs1 = store.RunOutputs(lease=lease1)
    outputs1.stage(target1, name="srt", write=lambda p: p.write_text("srt:r1", "utf-8"))
    outputs1.publish()
    legacy_dir1 = _relocate_store_to_legacy(tmp_path, target1)

    orig_write = store.write_json_atomic
    fail_once = {"armed": True}

    def flaky_write(path, payload, *, project_root):
        if fail_once["armed"] and Path(path).name == "manifest.json":
            fail_once["armed"] = False
            raise PermissionError("simulated manifest lock")
        return orig_write(path, payload, project_root=project_root)

    monkeypatch.setattr(store, "write_json_atomic", flaky_write)
    with pytest.raises(PermissionError):
        store.migrate_legacy_outputs(tmp_path)
    # Directory was not published half-baked with stale manifest paths.
    assert legacy_dir1.is_dir()
    assert not store.output_store_root(target1, project_root=tmp_path).exists()

    # Second attempt succeeds and passes strict hash verification.
    assert store.migrate_legacy_outputs(tmp_path) == 1
    assert not (tmp_path / "tmp" / "outputs").exists()
    verified1 = store.read_run_manifest(lease1, verify=True)
    assert verified1 is not None
    assert verified1["files"]["srt"].startswith("outputs/")

    # (2) Store already sitting under outputs/ with stale tmp/outputs/ manifest is repaired.
    target2 = tmp_path / "out" / "sample-moved.ja.srt"
    lease2 = store.claim_output(target2, project_root=tmp_path, job_id="j2", run_id="r2")
    outputs2 = store.RunOutputs(lease=lease2)
    outputs2.stage(target2, name="srt", write=lambda p: p.write_text("srt:r2", "utf-8"))
    outputs2.publish()
    legacy_dir2 = _relocate_store_to_legacy(tmp_path, target2)
    moved_dir2 = store.output_store_root(target2, project_root=tmp_path)
    shutil.move(str(legacy_dir2), str(moved_dir2))
    (tmp_path / "tmp" / "outputs").rmdir()

    with pytest.raises(store.OutputIntegrityError):
        store.read_run_manifest(lease2, verify=True)

    assert store.migrate_legacy_outputs(tmp_path) == 1
    verified2 = store.read_run_manifest(lease2, verify=True)
    assert verified2 is not None
    assert verified2["files"]["srt"].startswith("outputs/")


def test_upgrade_migration_pauses_and_keeps_legacy_index_when_output_move_fails(
    tmp_path, monkeypatch
):
    """If moving tmp/outputs/ fails, legacy jobs.json is preserved intact so a
    committed `saving` job is never dropped and recovers cleanly on restart."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    target = tmp_path / "out" / "sample-saving.ja.srt"
    lease = store.claim_output(target, project_root=tmp_path, job_id="legacy-saving", run_id="run-saving")
    outputs = store.RunOutputs(lease=lease)
    outputs.stage(target, name="srt", write=lambda p: p.write_text("srt:saving", "utf-8"))
    manifest = outputs.publish()
    store.export_output(lease)
    _relocate_store_to_legacy(tmp_path, target)

    legacy_records = [
        {
            "id": "legacy-saving",
            "status": "saving",
            "created_at": "2026-09-01T00:00:00Z",
            "run_id": lease.run_id,
            "output_target": str(target),
            "output_generation": lease.sequence,
            "artifacts": ["tmp/" + v for v in manifest["files"].values()],
            "spec": {
                "video_paths": [str(tmp_path / "sample-saving.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
            },
        }
    ]
    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(json.dumps(legacy_records), encoding="utf-8")

    orig_rename = store.os.rename
    block_move = {"armed": True}

    def flaky_rename(src, dst):
        if block_move["armed"] and "tmp" in str(src) and "outputs" in str(src):
            raise PermissionError("simulated folder lock")
        return orig_rename(src, dst)

    monkeypatch.setattr(store.os, "rename", flaky_rename)

    async def scenario():
        with pytest.raises(pm.JobPersistenceError):
            await pm.load_jobs()
        # Legacy index is preserved; new index was not written.
        assert legacy_jobs_path.is_file()
        assert not (tmp_path / "cache" / "web" / "jobs.json").exists()

        # Once the lock clears, restart migrates both outputs and the saving job.
        block_move["armed"] = False
        await pm.load_jobs()
        jobs = await pm.list_jobs()
        assert [j.id for j in jobs] == ["legacy-saving"]
        assert jobs[0].status == "done"
        assert not legacy_jobs_path.exists()
        verified = store.read_run_manifest(lease, verify=True)
        assert verified is not None

    asyncio.run(scenario())


def test_upgrade_migration_retries_locked_legacy_temp_dir_across_restarts(
    tmp_path, monkeypatch
):
    """If a dropped legacy job's temp WAV dir is locked during migration, its ID
    is persisted and retried on subsequent startups until removed."""
    import shutil

    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    wav = tmp_path / "tmp" / "web" / "jobs" / "legacy-wav" / "audio" / "source.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"wav")

    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-wav",
                    "status": "failed",
                    "created_at": "2026-09-01T00:00:00Z",
                    "run_id": "run-wav",
                    "spec": {
                        "video_paths": [str(tmp_path / "sample.mp4")],
                        "skip_translation": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    orig_rmtree = shutil.rmtree
    lock_state = {"locked": True}

    def flaky_rmtree(path, *args, **kwargs):
        if lock_state["locked"] and Path(path).name == "legacy-wav":
            raise PermissionError("simulated wav handle open")
        return orig_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pm.shutil, "rmtree", flaky_rmtree)

    async def scenario():
        await pm.load_jobs()
        assert (tmp_path / "cache" / "web" / "jobs.json").is_file()
        assert not legacy_jobs_path.exists()
        assert wav.is_file()
        pending_file = tmp_path / "cache" / "web" / pm._LEGACY_CLEANUP_FILE_NAME
        assert pending_file.is_file()
        assert json.loads(pending_file.read_text(encoding="utf-8")) == ["legacy-wav"]

        # Second startup after the file lock clears removes the WAV dir and marker.
        lock_state["locked"] = False
        await pm.load_jobs()
        assert not (tmp_path / "tmp" / "web" / "jobs" / "legacy-wav").exists()
        assert not pending_file.exists()

    asyncio.run(scenario())


def test_upgrade_migration_drops_unrecoverable_publish_and_export_failed_jobs(
    tmp_path, monkeypatch
):
    """publish_failed and export_failed cards with only an ownership record (no
    committed or prepared generation) are dropped; truly prepared publish_failed
    jobs remain recoverable."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    # (1) Ownership-only publish_failed (unrecoverable) -> must be dropped.
    pub_empty_target = tmp_path / "out" / "sample-pub-empty.ja.srt"
    pub_empty_lease = store.claim_output(
        pub_empty_target, project_root=tmp_path, job_id="legacy-pub-empty", run_id="run-pub-empty"
    )
    _relocate_store_to_legacy(tmp_path, pub_empty_target)

    # (2) Ownership-only export_failed (unrecoverable) -> must be dropped.
    exp_empty_target = tmp_path / "out" / "sample-exp-empty.ja.srt"
    exp_empty_lease = store.claim_output(
        exp_empty_target, project_root=tmp_path, job_id="legacy-exp-empty", run_id="run-exp-empty"
    )
    _relocate_store_to_legacy(tmp_path, exp_empty_target)

    # (3) Prepared-but-uncommitted publish_failed (recoverable) -> must be kept.
    pub_ready_target = tmp_path / "out" / "sample-pub-ready.ja.srt"
    pub_ready_lease = store.claim_output(
        pub_ready_target, project_root=tmp_path, job_id="legacy-pub-ready", run_id="run-pub-ready"
    )
    outputs = store.RunOutputs(lease=pub_ready_lease)
    outputs.stage(pub_ready_target, name="srt", write=lambda p: p.write_text("srt:prepared", "utf-8"))
    orig_commit = store.commit_output
    monkeypatch.setattr(
        store,
        "commit_output",
        lambda lease, **kwargs: (_ for _ in ()).throw(store.OutputCommitError(lease)),
    )
    with pytest.raises(store.OutputCommitError):
        outputs.publish()
    monkeypatch.setattr(store, "commit_output", orig_commit)
    _relocate_store_to_legacy(tmp_path, pub_ready_target)

    legacy_records = [
        {
            "id": "legacy-pub-empty",
            "status": "publish_failed",
            "created_at": "2026-09-01T00:00:00Z",
            "run_id": pub_empty_lease.run_id,
            "output_target": str(pub_empty_target),
            "output_generation": pub_empty_lease.sequence,
            "spec": {
                "video_paths": [str(tmp_path / "sample-pub-empty.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
            },
        },
        {
            "id": "legacy-exp-empty",
            "status": "export_failed",
            "created_at": "2026-09-01T00:01:00Z",
            "run_id": exp_empty_lease.run_id,
            "output_target": str(exp_empty_target),
            "output_generation": exp_empty_lease.sequence,
            "spec": {
                "video_paths": [str(tmp_path / "sample-exp-empty.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
            },
        },
        {
            "id": "legacy-pub-ready",
            "status": "publish_failed",
            "created_at": "2026-09-01T00:02:00Z",
            "run_id": pub_ready_lease.run_id,
            "output_target": str(pub_ready_target),
            "output_generation": pub_ready_lease.sequence,
            "spec": {
                "video_paths": [str(tmp_path / "sample-pub-ready.mp4")],
                "output_dir": str(tmp_path / "out"),
                "skip_translation": True,
            },
        },
    ]
    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(json.dumps(legacy_records), encoding="utf-8")

    async def scenario():
        await pm.load_jobs()
        jobs_by_id = {j.id: j for j in await pm.list_jobs()}
        assert set(jobs_by_id) == {"legacy-pub-ready"}
        assert jobs_by_id["legacy-pub-ready"].status == "publish_failed"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            retry = await client.post("/api/jobs/legacy-pub-ready/retry")
            assert retry.status_code == 200
            assert retry.json()["status"] == "done"
            assert pub_ready_target.read_text(encoding="utf-8") == "srt:prepared"

    asyncio.run(scenario())


def test_migrate_legacy_outputs_cross_device_copy_interruption_never_publishes_partial_store(
    tmp_path, monkeypatch
):
    """Cross-volume fallback copies into hidden staging, verifies completeness,
    and only publishes atomically; an interrupted copy leaves no partial target
    and succeeds cleanly on retry."""
    import errno
    import shutil

    target = tmp_path / "out" / "sample-xdev.ja.srt"
    lease = store.claim_output(target, project_root=tmp_path, job_id="j-xdev", run_id="r-xdev")
    outputs = store.RunOutputs(lease=lease)
    outputs.stage(target, name="srt", write=lambda p: p.write_text("srt:xdev", "utf-8"))
    outputs.publish()
    legacy_dir = _relocate_store_to_legacy(tmp_path, target)
    final_dir = store.output_store_root(target, project_root=tmp_path)

    orig_rename = store.os.rename
    orig_copy2 = shutil.copy2
    lock_current = {"armed": True}

    def xdev_rename(src, dst):
        if Path(src).resolve() == legacy_dir.resolve():
            raise OSError(errno.EXDEV, "simulated cross-device link")
        return orig_rename(src, dst)

    def flaky_copy2(src, dst, *args, **kwargs):
        if lock_current["armed"] and Path(src).name == "current.json":
            raise PermissionError("simulated current.json lock during copy")
        return orig_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(store.os, "rename", xdev_rename)
    monkeypatch.setattr(store.shutil, "copy2", flaky_copy2)

    with pytest.raises(OSError):
        store.migrate_legacy_outputs(tmp_path)
    # Neither final destination nor hidden staging directory was left behind.
    assert not final_dir.exists()
    assert legacy_dir.is_dir()
    assert not list((tmp_path / "outputs").glob(".migrate-*"))

    # Unlock current.json and retry -> verified staged copy publishes cleanly.
    lock_current["armed"] = False
    assert store.migrate_legacy_outputs(tmp_path) == 1
    assert final_dir.is_dir()
    assert not legacy_dir.exists()
    verified = store.read_run_manifest(lease, verify=True)
    assert verified is not None
    assert verified["files"]["srt"].startswith("outputs/")


def test_corrupt_new_manifest_does_not_block_web_startup_or_other_jobs(
    tmp_path, monkeypatch
):
    """A single non-UTF-8 manifest under outputs/ in a normal v2 install is
    isolated to its own job and never blocks load_jobs() or unrelated tasks."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    ok_target = tmp_path / "out" / "sample-ok.ja.srt"
    ok_lease = store.claim_output(ok_target, project_root=tmp_path, job_id="job-ok", run_id="run-ok")
    ok_outputs = store.RunOutputs(lease=ok_lease)
    ok_outputs.stage(ok_target, name="srt", write=lambda p: p.write_text("srt:ok", "utf-8"))
    ok_manifest = ok_outputs.publish()
    store.export_output(ok_lease)

    bad_target = tmp_path / "out" / "sample-bad.ja.srt"
    bad_lease = store.claim_output(bad_target, project_root=tmp_path, job_id="job-bad", run_id="run-bad")
    bad_outputs = store.RunOutputs(lease=bad_lease)
    bad_outputs.stage(bad_target, name="srt", write=lambda p: p.write_text("srt:bad", "utf-8"))
    bad_outputs.publish()
    bad_store = store.output_store_root(bad_target, project_root=tmp_path)
    bad_manifest = next(bad_store.glob("generations/*/manifest.json"))
    bad_manifest.write_bytes(b"\xff\xfe\x00not-utf8")

    jobs_path = tmp_path / "cache" / "web" / "jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "job-ok",
                    "status": "done",
                    "created_at": "2026-09-01T00:00:00Z",
                    "run_id": ok_lease.run_id,
                    "output_target": str(ok_target),
                    "output_generation": ok_lease.sequence,
                    "artifacts": list(ok_manifest["files"].values()),
                    "spec": {
                        "video_paths": [str(tmp_path / "sample-ok.mp4")],
                        "output_dir": str(tmp_path / "out"),
                        "skip_translation": True,
                    },
                },
                {
                    "id": "job-bad",
                    "status": "done",
                    "created_at": "2026-09-01T00:01:00Z",
                    "run_id": bad_lease.run_id,
                    "output_target": str(bad_target),
                    "output_generation": bad_lease.sequence,
                    "spec": {
                        "video_paths": [str(tmp_path / "sample-bad.mp4")],
                        "output_dir": str(tmp_path / "out"),
                        "skip_translation": True,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    async def scenario():
        await pm.load_jobs()
        jobs_by_id = {j.id: j for j in await pm.list_jobs()}
        assert jobs_by_id["job-ok"].status == "done"
        assert jobs_by_id["job-bad"].status == "publish_failed"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/api/output/job-ok/sample-ok.ja.srt")
            assert resp.status_code == 200
            assert resp.text == "srt:ok"

    asyncio.run(scenario())


def test_upgrade_migration_crash_after_new_index_and_read_error_on_cleanup_list(
    tmp_path, monkeypatch
):
    """Cleanup IDs are persisted *before* writing cache/web/jobs.json so a crash
    immediately after writing the new index still removes both tmp/web/jobs.json
    and dropped WAV dirs on restart, and a transient read failure on
    legacy-cleanup.json preserves the file for the next retry."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    wav = tmp_path / "tmp" / "web" / "jobs" / "legacy-crash" / "audio" / "source.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"wav")
    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-crash",
                    "status": "translating",
                    "created_at": "2026-09-01T00:00:00Z",
                    "run_id": "run-crash",
                    "spec": {
                        "video_paths": [str(tmp_path / "sample.mp4")],
                        "skip_translation": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    pending_file = tmp_path / "cache" / "web" / pm._LEGACY_CLEANUP_FILE_NAME
    orig_drain = pm._drain_pending_legacy_temp_dirs
    crash_after_index = {"armed": True}

    def crash_on_first_drain(*args, **kwargs):
        if crash_after_index["armed"]:
            crash_after_index["armed"] = False
            # Both legacy-cleanup.json and cache/web/jobs.json must already be on disk!
            assert pending_file.is_file()
            assert (tmp_path / "cache" / "web" / "jobs.json").is_file()
            raise SystemExit("simulated crash right after writing new jobs.json")
        return orig_drain(*args, **kwargs)

    monkeypatch.setattr(pm, "_drain_pending_legacy_temp_dirs", crash_on_first_drain)

    async def scenario():
        with pytest.raises(SystemExit):
            await pm.load_jobs()
        assert legacy_jobs_path.is_file()
        assert wav.is_file()
        assert pending_file.is_file()

        # Restart 1: simulate a transient OSError reading legacy-cleanup.json.
        orig_read_text = Path.read_text
        fail_cleanup_read = {"armed": True}

        def flaky_read_text(self, *args, **kwargs):
            if fail_cleanup_read["armed"] and self == pending_file:
                fail_cleanup_read["armed"] = False
                raise PermissionError("simulated transient lock on legacy-cleanup.json")
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)
        await pm.load_jobs()
        # Pending file must NOT be deleted when read fails.
        assert pending_file.is_file()
        assert wav.is_file()

        # Restart 2: lock cleared -> removes leftover legacy jobs.json, WAV dir, and marker.
        await pm.load_jobs()
        assert not legacy_jobs_path.exists()
        assert not (tmp_path / "tmp" / "web" / "jobs" / "legacy-crash").exists()
        assert not pending_file.exists()

    asyncio.run(scenario())


def test_upgrade_migration_records_integrity_review_for_corrupt_owner_without_generations(
    tmp_path, monkeypatch
):
    """A legacy publish_failed job whose store only has a corrupt current.json
    and no generations/ directory is not migrated as a retryable publish_failed
    card; it is recorded separately for integrity review."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    target = tmp_path / "out" / "sample-corrupt-owner.ja.srt"
    lease = store.claim_output(
        target, project_root=tmp_path, job_id="legacy-corrupt-pub", run_id="run-corrupt-pub"
    )
    legacy_dir = _relocate_store_to_legacy(tmp_path, target)
    (legacy_dir / "current.json").write_text("{broken json", encoding="utf-8")
    assert not (legacy_dir / "generations").exists()

    wav = tmp_path / "tmp" / "web" / "jobs" / "legacy-corrupt-pub" / "audio" / "source.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"wav")

    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-corrupt-pub",
                    "status": "publish_failed",
                    "created_at": "2026-09-01T00:00:00Z",
                    "run_id": lease.run_id,
                    "output_target": str(target),
                    "output_generation": lease.sequence,
                    "spec": {
                        "video_paths": [str(tmp_path / "sample-corrupt-owner.mp4")],
                        "output_dir": str(tmp_path / "out"),
                        "skip_translation": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    async def scenario():
        await pm.load_jobs()
        # Not kept as a retryable publish_failed card.
        assert await pm.list_jobs() == []
        assert not (tmp_path / "tmp" / "web" / "jobs" / "legacy-corrupt-pub").exists()
        # Separately recorded for integrity review.
        review_path = tmp_path / "cache" / "web" / pm._LEGACY_INTEGRITY_REVIEW_FILE_NAME
        assert review_path.is_file()
        entries = json.loads(review_path.read_text(encoding="utf-8"))
        assert len(entries) == 1
        assert entries[0]["status"] == "integrity_review_required"
        assert entries[0]["job_id"] == "legacy-corrupt-pub"
        assert entries[0]["has_generations"] is False

    asyncio.run(scenario())


def test_upgrade_migration_pauses_when_current_json_temporarily_locked(
    tmp_path, monkeypatch
):
    """A transient PermissionError reading current.json during legacy job
    classification pauses migration and preserves tmp/web/jobs.json rather than
    dropping a committed saving job as corrupt."""
    _setup_legacy_upgrade_env(tmp_path, monkeypatch)

    target = tmp_path / "out" / "sample-lock-owner.ja.srt"
    lease = store.claim_output(
        target, project_root=tmp_path, job_id="legacy-locked-saving", run_id="run-locked-saving"
    )
    outputs = store.RunOutputs(lease=lease)
    outputs.stage(target, name="srt", write=lambda p: p.write_text("srt:locked", "utf-8"))
    manifest = outputs.publish()
    store.export_output(lease)
    _relocate_store_to_legacy(tmp_path, target)

    legacy_jobs_path = tmp_path / "tmp" / "web" / "jobs.json"
    legacy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-locked-saving",
                    "status": "saving",
                    "created_at": "2026-09-01T00:00:00Z",
                    "run_id": lease.run_id,
                    "output_target": str(target),
                    "output_generation": lease.sequence,
                    "artifacts": ["tmp/" + v for v in manifest["files"].values()],
                    "spec": {
                        "video_paths": [str(tmp_path / "sample-lock-owner.mp4")],
                        "output_dir": str(tmp_path / "out"),
                        "skip_translation": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    orig_read_text = Path.read_text
    lock_owner = {"armed": True}

    def flaky_read_text(self, *args, **kwargs):
        if lock_owner["armed"] and self.name == "current.json":
            raise PermissionError("simulated transient lock on current.json")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    async def scenario():
        with pytest.raises(pm.JobPersistenceError):
            await pm.load_jobs()
        assert legacy_jobs_path.is_file()
        assert not (tmp_path / "cache" / "web" / "jobs.json").exists()
        assert not (tmp_path / "cache" / "web" / pm._LEGACY_INTEGRITY_REVIEW_FILE_NAME).exists()

        # Once the lock clears, restart migrates and keeps the committed job card.
        lock_owner["armed"] = False
        await pm.load_jobs()
        jobs = await pm.list_jobs()
        assert [j.id for j in jobs] == ["legacy-locked-saving"]
        assert jobs[0].status == "done"
        assert store.read_run_manifest(lease, verify=True) is not None

    asyncio.run(scenario())


def test_migrate_legacy_outputs_cross_device_equal_length_corruption_preserves_source(
    tmp_path, monkeypatch
):
    """Cross-volume staged copy verifies manifest SHA-256 before publishing or
    deleting the source directory, catching equal-length byte corruption."""
    import errno
    import shutil

    target = tmp_path / "out" / "sample-xdev-hash.ja.srt"
    lease = store.claim_output(target, project_root=tmp_path, job_id="j-xh", run_id="r-xh")
    outputs = store.RunOutputs(lease=lease)
    outputs.stage(target, name="srt", write=lambda p: p.write_text("srt:good", "utf-8"))
    outputs.publish()
    legacy_dir = _relocate_store_to_legacy(tmp_path, target)
    final_dir = store.output_store_root(target, project_root=tmp_path)

    orig_rename = store.os.rename
    orig_copy2 = shutil.copy2
    corrupt_once = {"armed": True}

    def xdev_rename(src, dst):
        if Path(src).resolve() == legacy_dir.resolve():
            raise OSError(errno.EXDEV, "simulated cross-device link")
        return orig_rename(src, dst)

    def corrupting_copy2(src, dst, *args, **kwargs):
        result = orig_copy2(src, dst, *args, **kwargs)
        if corrupt_once["armed"] and Path(src).suffix == ".srt":
            # Replace with same-length corrupted bytes.
            Path(dst).write_text("srt:evil", encoding="utf-8")
        return result

    monkeypatch.setattr(store.os, "rename", xdev_rename)
    monkeypatch.setattr(store.shutil, "copy2", corrupting_copy2)

    with pytest.raises(OSError, match="SHA-256"):
        store.migrate_legacy_outputs(tmp_path)
    # Destination was not published and intact source directory was preserved.
    assert not final_dir.exists()
    assert legacy_dir.is_dir()
    assert not list((tmp_path / "outputs").glob(".migrate-*"))

    corrupt_once["armed"] = False
    assert store.migrate_legacy_outputs(tmp_path) == 1
    assert final_dir.is_dir()
    assert not legacy_dir.exists()
    verified = store.read_run_manifest(lease, verify=True)
    assert verified is not None
    srt_path = store.artifact_path(verified["files"]["srt"], tmp_path)
    assert srt_path.read_text(encoding="utf-8") == "srt:good"
