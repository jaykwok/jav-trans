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
        bilingual_json_path=str(temp / "sample-a.bilingual.json"), quality_report_path="",
        bilingual=True, timings_path=str(temp / "sample-a.timings.json"),
        translation_cache_path=str(temp / "translation_cache.jsonl"), asr_log=[],
        audio_cached=True, device="cpu", backend_label="test", video_duration_s=0.0,
        pipeline_started=time.perf_counter(), job_id=job.id, aligned_cache_signature={},
    )


def test_real_empty_main_recovers_prepared_generation_before_backend_or_compute(workspace, monkeypatch):
    async def scenario():
        job = (await pm.create_job(JobSpec(
            video_paths=[str(workspace / "sample-a.mp4")], output_dir=str(workspace / "out"),
            advanced={"QUALITY_REPORT_ENABLED": "1", "RUN_LOG_ENABLED": "0"},
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
            assert quality.status_code == 200
            assert quality.json()["available"] is True
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


def test_rejected_job_creation_restores_previous_output_owner(workspace, monkeypatch):
    async def scenario():
        spec = JobSpec(video_paths=[str(workspace / "sample-a.mp4")], skip_translation=True)
        old = (await pm.create_job(spec))[0]
        outputs = store.RunOutputs(lease=pm.output_lease_for_job(old))
        outputs.stage(outputs.lease.target, name="srt", write=lambda path: path.write_text("old work", "utf-8"))
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
        replace = Path.replace
        pm._job_store_writer._CONFIRM_TIMEOUT_S = 0.05

        def slow_replace(path, target):
            if Path(target) == pm._jobs_path:
                entered.set()
                assert release.wait(3)
            return replace(path, target)

        monkeypatch.setattr(Path, "replace", slow_replace)
        save = asyncio.create_task(pm._set_job(job, status="done"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            if interruption == "cancel":
                save.cancel()
            await asyncio.sleep(0.1)
            assert not save.done()
            assert (await pm.get_job(job.id)).status == "saving"
        finally:
            release.set()
        await save
        assert (await pm.get_job(job.id)).status == "done"
        assert json.loads(pm._jobs_path.read_text("utf-8"))[0]["status"] == "done"
    asyncio.run(scenario())
