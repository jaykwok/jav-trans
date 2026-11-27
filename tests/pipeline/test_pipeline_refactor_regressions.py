"""Regression tests for Wave 2 Pipeline & Main architecture/reliability/perf refactor."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from core import resources
from helpers import make_job_context, run_pipeline, write_pcm_wav
from pipeline import (
    aligned_cache,
    artifact_store,
    artifacts as artifacts_module,
    audio as pipeline_audio,
    cleanup,
    gpu_oom_policy,
    gpu_worker,
    memory_safety,
    output as pipeline_output,
    run_logger,
    stage_log,
)
from pipeline.ids import sanitize_job_id


@pytest.fixture(autouse=True)
def _isolate_main_project_root(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# P0-14: Concurrent temp cleanup race & cache cleanup hygiene
# ---------------------------------------------------------------------------


def test_cleanup_empty_runtime_dir_tree_protects_recent_and_excluded_dirs(tmp_path):
    jobs_root = tmp_path / "tmp" / "jobs"
    fresh_audio = jobs_root / "job_fresh" / "audio"
    stale_audio = jobs_root / "job_stale" / "audio"
    excluded_audio = jobs_root / "job_active" / "audio"
    for path in (fresh_audio, stale_audio, excluded_audio):
        path.mkdir(parents=True)

    old_ts = time.time() - 300.0
    for path in (
        stale_audio,
        stale_audio.parent,
        excluded_audio,
        excluded_audio.parent,
    ):
        os.utime(path, (old_ts, old_ts))

    cleanup.cleanup_empty_runtime_dir_tree(
        jobs_root,
        tmp_path,
        min_age_s=60.0,
        exclude_names={"job_active"},
    )

    assert fresh_audio.is_dir()
    assert excluded_audio.is_dir()
    assert not stale_audio.exists()
    assert not stale_audio.parent.exists()


def test_cleanup_pipeline_temp_only_cleans_own_job_dir(monkeypatch, tmp_path):
    jobs_root = tmp_path / "tmp" / "jobs"
    chunks_root = tmp_path / "tmp" / "chunks"
    own_dir = jobs_root / "job_a"
    other_audio = jobs_root / "job_b" / "audio"
    other_chunk = chunks_root / "job_b"
    (own_dir / "audio").mkdir(parents=True)
    (own_dir / "audio" / "a.wav").write_bytes(b"wav")
    other_audio.mkdir(parents=True)
    other_chunk.mkdir(parents=True)

    monkeypatch.setenv("JOB_TEMP_DIR", str(jobs_root))
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(chunks_root))

    main._cleanup_pipeline_temp(str(own_dir), "", "")

    assert not own_dir.exists()
    assert other_audio.is_dir()
    assert other_chunk.is_dir()


def test_cleanup_translation_cache_distinguishes_empty_string_from_none(monkeypatch, tmp_path):
    env_cache = tmp_path / "shared" / "translation_cache.jsonl"
    env_cache.parent.mkdir(parents=True)
    env_cache.write_text("line\n", encoding="utf-8")
    monkeypatch.setenv("TRANSLATION_CACHE_PATH", str(env_cache))

    # Explicit empty string must NOT fall back to TRANSLATION_CACHE_PATH.
    cleanup.cleanup_translation_cache("")
    assert env_cache.is_file()

    # None explicitly opts into env fallback.
    cleanup.cleanup_translation_cache(None)
    assert not env_cache.exists()


def test_cleanup_translation_cache_glossary_only_inside_job_dir(tmp_path):
    job_dir = tmp_path / "jobs" / "job_1"
    shared_dir = tmp_path / "shared_cache"
    job_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    shared_cache = shared_dir / "cache.jsonl"
    shared_glossary = shared_dir / "translation_global_glossary.json"
    shared_cache.write_text("x\n", encoding="utf-8")
    shared_glossary.write_text("{}", encoding="utf-8")

    cleanup.cleanup_translation_cache(str(shared_cache), job_temp_dir=job_dir)
    assert not shared_cache.exists()
    # Shared external directory glossary must not be deleted by a single job.
    assert shared_glossary.is_file()

    job_cache = job_dir / "translation_cache.jsonl"
    job_glossary = job_dir / "translation_global_glossary_v1.json"
    job_cache.write_text("y\n", encoding="utf-8")
    job_glossary.write_text("{}", encoding="utf-8")

    cleanup.cleanup_translation_cache(str(job_cache), job_temp_dir=job_dir)
    assert not job_cache.exists()
    assert not job_glossary.exists()


# ---------------------------------------------------------------------------
# P0-17(1) + P1-7 + P1-14: gpu_worker split & hardening
# ---------------------------------------------------------------------------


def test_get_global_worker_closes_idle_exited_worker_before_replacing(monkeypatch):
    closed_handles: list[int] = []
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: (closed_handles.append(handle), True)[1],
    )
    client = gpu_worker._GpuWorkerClient()
    dead_child = SimpleNamespace(
        pid=9001,
        exitcode=0,
        is_alive=lambda: False,
        terminate=lambda: None,
        kill=lambda: None,
        join=lambda _t=None: None,
    )
    conn_closed = {"closed": False}
    fake_conn = SimpleNamespace(
        close=lambda: conn_closed.__setitem__("closed", True),
        send=lambda _msg: (_ for _ in ()).throw(AssertionError("dead child must not receive shutdown")),
    )
    client._process = dead_child
    client._conn = fake_conn
    client._job_handle = 555666
    client._child_generation = 1
    client._ensure_registered_locked(dead_child, 1)
    assert resources.is_registered(client._resource_id(), 1)

    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", client)
    replacement = gpu_worker._get_global_worker()

    assert replacement is not client
    assert closed_handles == [555666]
    assert conn_closed["closed"] is True
    assert not resources.is_registered(client._resource_id(), 1)


@pytest.mark.skipif(os.name != "nt", reason="Job Objects are Windows-only")
def test_start_worker_uses_subprocess_tools_and_closes_handle_when_attach_fails(monkeypatch):
    created_calls: list[int] = []
    bound_calls: list[tuple[int, object]] = []
    closed_handles: list[int] = []

    monkeypatch.setattr(
        gpu_worker,
        "create_kill_on_close_job",
        lambda: (created_calls.append(777), 777)[1],
    )
    monkeypatch.setattr(
        gpu_worker,
        "bind_process_to_kill_on_close_job",
        lambda job, proc: (bound_calls.append((job, proc)), job)[1],
    )
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: (closed_handles.append(handle), True)[1],
    )
    monkeypatch.setattr(resources, "attach_handle", lambda *_args: False)
    # Remove any cached local_backend to prove `_start_worker` never imports it.
    monkeypatch.delitem(sys.modules, "asr.local_backend", raising=False)

    class _ReadyConn:
        def __init__(self, msgs=()):
            self._msgs = list(msgs)

        def poll(self, _timeout=None):
            return bool(self._msgs)

        def recv(self):
            return self._msgs.pop(0)

        def close(self):
            pass

    proc = SimpleNamespace(
        pid=12345,
        exitcode=None,
        start=lambda: None,
        is_alive=lambda: True,
        terminate=lambda: None,
        kill=lambda: None,
        join=lambda _t=None: None,
    )
    client = gpu_worker._GpuWorkerClient()
    client._ctx = SimpleNamespace(
        Pipe=lambda duplex=True: (_ReadyConn([{"op": "ready"}]), _ReadyConn()),
        Process=lambda **_kwargs: proc,
    )

    client._start_worker()

    assert "asr.local_backend" not in sys.modules
    assert created_calls == [777]
    assert len(bound_calls) == 1 and bound_calls[0][0] == 777
    assert closed_handles == [777]
    assert client._job_handle is None


def test_idle_exit_message_restarts_without_consuming_disconnect_retry_budget(monkeypatch):
    client = gpu_worker._GpuWorkerClient()
    attempts = {"count": 0}

    def fake_once(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise gpu_worker._GpuWorkerIdleExitError("idle exit")
        if attempts["count"] == 2:
            raise gpu_worker._GpuWorkerDisconnectedError("real crash")
        return [{"start": 0.0, "end": 1.0, "text": "ok"}], [], {}

    monkeypatch.setattr(client, "_transcribe_and_align_once", fake_once)
    segments, _log, _details = client.transcribe_and_align("sample-a.wav")

    # 1st attempt: idle_exit (free restart); 2nd: real crash (uses the 1 disconnect retry); 3rd: succeeds!
    assert attempts["count"] == 3
    assert segments[0]["text"] == "ok"


def test_serve_worker_sends_idle_exit_after_rechecking_poll_zero(monkeypatch):
    sent: list[dict] = []
    polls: list[float] = []

    class _IdleConn:
        def poll(self, timeout=0.0):
            polls.append(timeout)
            return False

        def recv(self):  # pragma: no cover
            raise AssertionError("should not recv on idle exit")

        def send(self, msg):
            sent.append(msg)

    monkeypatch.setenv("ASR_STAGE_WORKER_MAX_IDLE_S", "0.05")
    monkeypatch.setattr(gpu_worker, "configure_ffmpeg_shared_runtime", lambda: ())
    monkeypatch.setattr(gpu_worker, "_clear_worker_cuda", lambda: None)

    gpu_worker._serve_worker(_IdleConn())

    assert [m["op"] for m in sent] == ["ready", "idle_exit"]
    assert polls == [0.05, 0]


def test_close_skips_shutdown_wait_when_busy(monkeypatch):
    client = gpu_worker._GpuWorkerClient()
    sent: list[dict] = []
    joins: list[float] = []
    killed = {"count": 0}
    client._conn = SimpleNamespace(send=sent.append, close=lambda: None)
    client._process = SimpleNamespace(
        is_alive=lambda: True,
        join=joins.append,
    )
    client._in_request = True
    monkeypatch.setattr(
        client,
        "_kill_child",
        lambda: (killed.__setitem__("count", killed["count"] + 1), True)[1],
    )

    assert client.close() is True
    assert sent == []
    assert joins == []
    assert killed["count"] == 1


def test_adaptive_runtime_tuning_checks_cuda_available_and_resolve_device_forbids_cpu(monkeypatch):
    torch_no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA 不可用"):
        gpu_worker._adaptive_runtime_tuning(torch_no_cuda, {}, enforce_minimum=True)

    with pytest.raises(RuntimeError, match="CPU"):
        gpu_worker._resolve_device("cpu")

    monkeypatch.setitem(sys.modules, "torch", torch_no_cuda)
    with pytest.raises(RuntimeError, match="CUDA 不可用"):
        gpu_worker._resolve_device("auto")


def test_stage_specific_oom_downshift_and_temporary_env_restores_all_four_keys():
    class _Props:
        total_memory = 8 * 1024 * 1024 * 1024

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(_idx):
            return _Props()

        @staticmethod
        def set_per_process_memory_fraction(_frac, _idx):
            pass

    torch_stub = SimpleNamespace(cuda=_Cuda())
    env = {"ASR_BACKEND": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"}
    prior = {
        key: os.environ.get(key)
        for key in (
            "ASR_BATCH_SIZE",
            "ASR_FEATURE_BATCH_SIZE",
            "ASR_ALIGN_BATCH_SIZE",
            "ASR_STAGE_WORKER_VRAM_BUDGET_MB",
        )
    }

    with gpu_worker._temporary_env(env):
        tuning = gpu_worker._adaptive_runtime_tuning(torch_stub, env)
        assert tuning["asr_feature_batch_size"] >= 2
        assert tuning["asr_align_batch_size"] >= 2
        assert os.environ["ASR_FEATURE_BATCH_SIZE"] == str(tuning["asr_feature_batch_size"])
        assert os.environ["ASR_ALIGN_BATCH_SIZE"] == str(tuning["asr_align_batch_size"])

    # All 4 tuning keys must be restored after leaving _temporary_env even though
    # none of them appeared in `env`.
    for key, expected in prior.items():
        assert os.environ.get(key) == expected

    # Per-stage OOM downshift targets the exact stage knob.
    chunk_exc = gpu_worker.GpuWorkerError(
        "oom",
        "CUDA out of memory",
        stage="audio_chunking",
        runtime_tuning={"asr_batch_size": 10, "asr_feature_batch_size": 8, "asr_align_batch_size": 8},
    )
    chunk_down = gpu_worker._oom_downshift({"ASR_BATCH_SIZE": "10"}, chunk_exc)
    assert chunk_down is not None
    assert chunk_down[1:] == ("ASR_FEATURE_BATCH_SIZE", 8.0, 4.0)
    assert chunk_down[0]["ASR_BATCH_SIZE"] == "10"

    align_exc = gpu_worker.GpuWorkerError(
        "oom",
        "CUDA out of memory",
        stage="subtitle_timing",
        runtime_tuning={"asr_batch_size": 10, "asr_feature_batch_size": 8, "asr_align_batch_size": 6},
    )
    align_down = gpu_worker._oom_downshift({"ASR_BATCH_SIZE": "10"}, align_exc)
    assert align_down is not None
    assert align_down[1:] == ("ASR_ALIGN_BATCH_SIZE", 6.0, 3.0)
    assert align_down[0]["ASR_BATCH_SIZE"] == "10"


def test_allocator_cap_resets_to_1_when_budget_disabled(monkeypatch):
    calls: list[tuple[float, int]] = []
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            set_per_process_memory_fraction=lambda frac, dev: calls.append((frac, dev)),
        )
    )
    monkeypatch.setattr(gpu_oom_policy, "_LAST_ALLOCATOR_FRACTION", None)

    assert gpu_oom_policy._apply_worker_vram_allocator_cap(
        torch_stub, total_mb=8192.0, budget_mb=4096.0
    ) == pytest.approx(0.5)
    assert gpu_oom_policy._apply_worker_vram_allocator_cap(
        torch_stub, total_mb=8192.0, budget_mb=0.0
    ) is None
    assert calls == [(0.5, 0), (1.0, 0)]


def test_gpu_model_manager_resets_shared_vram_baseline_and_no_redundant_empty_cache(monkeypatch):
    baseline_resets: list[bool] = []
    clear_calls: list[bool] = []

    monkeypatch.setattr(
        memory_safety,
        "reset_shared_vram_baseline",
        lambda pid=None: baseline_resets.append(True),
    )
    monkeypatch.setattr(gpu_worker, "configure_ffmpeg_shared_runtime", lambda: ())
    monkeypatch.setattr(gpu_worker, "_adaptive_runtime_tuning", lambda *_a, **_kw: {})
    monkeypatch.setattr(gpu_worker, "_clear_worker_cuda", lambda: clear_calls.append(True))

    class _OneShotConn:
        def __init__(self):
            self._msgs = [
                {
                    "op": "transcribe_and_align",
                    "job_id": "job-once",
                    "run_id": "run-1",
                    "audio_path": "sample.wav",
                    "device": "cpu",
                    "env": {"ASR_STAGE_WORKER_MOCK": "1"},
                },
                {"op": "shutdown"},
            ]
            self.sent: list[dict] = []

        def poll(self, _timeout=0.0):
            return bool(self._msgs)

        def recv(self):
            return self._msgs.pop(0)

        def send(self, msg):
            self.sent.append(msg)

    conn = _OneShotConn()
    gpu_worker._serve_worker(conn)

    assert baseline_resets == [True]
    # Only the explicit `shutdown` op calls `_clear_worker_cuda`; the request
    # path already executed `reset_cuda_state(reason="after_request")` once.
    assert clear_calls == [True]
    assert any(m.get("op") == "result" for m in conn.sent)


def test_prune_old_worker_logs_caps_count_and_age(tmp_path):
    now = time.time()
    for idx in range(5):
        log_file = tmp_path / f"asr-worker-{1000 + idx}.log"
        log_file.write_text("log", encoding="utf-8")
        ts = now - idx * 60
        os.utime(log_file, (ts, ts))
    ancient = tmp_path / "asr-worker-9999.log"
    ancient.write_text("old", encoding="utf-8")
    os.utime(ancient, (now - 10 * 86400, now - 10 * 86400))

    removed = gpu_worker._prune_old_worker_logs(
        log_dir=tmp_path,
        max_files=2,
        max_age_s=7 * 86400,
        exclude_pid=1004,
    )

    assert removed == 3
    remaining = sorted(p.name for p in tmp_path.glob("asr-worker-*.log"))
    assert remaining == ["asr-worker-1000.log", "asr-worker-1001.log", "asr-worker-1004.log"]


def test_client_missing_conn_raises_gpu_worker_error_and_env_overrides_heartbeat(monkeypatch, tmp_path):
    client = gpu_worker._GpuWorkerClient()
    monkeypatch.setattr(client, "_send_request", lambda *_a, **_kw: None)
    client._conn = None
    with pytest.raises(gpu_worker.GpuWorkerError) as exc_info:
        client._transcribe_and_align_once(str(tmp_path / "a.wav"), job_id="j1")
    assert exc_info.value.kind == "crash"

    # env_overrides heartbeat_s takes precedence over os.environ.
    monkeypatch.setenv("ASR_STAGE_WORKER_HEARTBEAT_S", "999")

    class _OneWaitConn:
        def __init__(self):
            self.n = 0

        def poll(self, _w):
            self.n += 1
            if self.n == 1:
                time.sleep(0.02)
                return False
            return True

        def recv(self):
            return {
                "op": "result",
                "job_id": "hb",
                "segments": [],
                "asr_log": [],
                "asr_details": {},
            }

    client._conn = _OneWaitConn()
    client._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    seen: list[str] = []
    client._transcribe_and_align_once(
        str(tmp_path / "a.wav"),
        job_id="hb",
        env_overrides={"ASR_STAGE_WORKER_HEARTBEAT_S": "0.01"},
        on_stage=seen.append,
    )
    assert any(msg.startswith("阶段心跳 ") for msg in seen)


# ---------------------------------------------------------------------------
# P0-15 + P1-12: artifact_store lock, retries, perf, and single-pass publish
# ---------------------------------------------------------------------------


def test_target_lock_is_reentrant_same_thread_and_does_not_write_lock_file(tmp_path):
    target = tmp_path / "out" / "sample-a.srt"
    started = time.perf_counter()
    with artifact_store._target_lock(target, project_root=tmp_path) as root1:
        with artifact_store._target_lock(target, project_root=tmp_path) as root2:
            assert root1 == root2
            lock_file = root1 / "owner.lock"
            assert lock_file.is_file()
            assert lock_file.stat().st_size == 0
    assert (time.perf_counter() - started) < 1.0


def test_read_owner_and_manifest_retry_transient_permission_error(monkeypatch, tmp_path):
    target = tmp_path / "out" / "sample-a.srt"
    lease = artifact_store.claim_output(
        target, project_root=tmp_path, job_id="j1", run_id="r1"
    )
    with artifact_store.RunOutputs(lease=lease) as outputs:
        outputs.stage(target, name="srt", write=lambda p: p.write_text("1", encoding="utf-8"))
        outputs.publish()

    real_read_text = Path.read_text
    counts = {"current.json": 0, "manifest.json": 0}

    def flaky_read_text(self, *args, **kwargs):
        if self.name in counts and counts[self.name] < 2:
            counts[self.name] += 1
            raise PermissionError("transient sharing violation")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    manifest = artifact_store.read_run_manifest(lease, verify=True)

    assert manifest is not None
    assert manifest["run_id"] == "r1"
    assert counts == {"current.json": 2, "manifest.json": 2}


def test_relativize_payload_paths_skips_base64_and_preserves_backslashes(tmp_path):
    inside = str(tmp_path / "jobs" / "clip" / "audio.wav")
    big_b64 = "A/B+C/D" * 10000
    payload = {
        "audio_path": inside,
        "text": r"¯\_(ツ)_/¯",
        "ja_text": r"あ\い",
        "zh_text": r"甲\乙",
        "frame_class_track": {"schema": "v2", "data_b64": big_b64},
        "words": [{"word": r"a\b", "start": 0.0, "end": 0.5}],
    }
    rel = artifact_store.relativize_payload_paths(payload, project_root=tmp_path)

    assert rel["audio_path"] == "jobs/clip/audio.wav"
    assert rel["text"] == r"¯\_(ツ)_/¯"
    assert rel["ja_text"] == r"あ\い"
    assert rel["zh_text"] == r"甲\乙"
    assert rel["frame_class_track"]["data_b64"] == big_b64
    assert rel["words"][0]["word"] == r"a\b"


def test_run_outputs_context_manager_discards_and_migrate_cleans_stale_staging(tmp_path):
    target = tmp_path / "out" / "sample-a.srt"
    lease = artifact_store.claim_output(
        target, project_root=tmp_path, job_id="j1", run_id="r1"
    )
    with pytest.raises(RuntimeError, match="boom"):
        with artifact_store.RunOutputs(lease=lease) as outputs:
            staging_dir = outputs.staging_dir
            outputs.stage(target, name="srt", write=lambda p: p.write_text("x", encoding="utf-8"))
            assert staging_dir.is_dir()
            raise RuntimeError("boom")
    assert not staging_dir.exists()

    store_root = artifact_store.output_store_root(target, project_root=tmp_path)
    stale_staging = store_root / ".s-stale00000000000"
    fresh_staging = store_root / ".s-fresh00000000000"
    stale_staging.mkdir(parents=True)
    fresh_staging.mkdir(parents=True)
    old_ts = time.time() - 7200.0
    os.utime(stale_staging, (old_ts, old_ts))

    artifact_store.migrate_legacy_outputs(tmp_path, stale_staging_max_age_s=3600.0)
    assert not stale_staging.exists()
    assert fresh_staging.is_dir()


def test_publish_and_export_avoid_redundant_sidecar_sha256_and_orphan_generations(
    monkeypatch, tmp_path
):
    target = tmp_path / "out" / "sample-a.srt"
    # 1. Cancelled publish must discard staging BEFORE rename -> zero generations/.
    lease_cancel = artifact_store.claim_output(
        target, project_root=tmp_path, job_id="j-cancel", run_id="r-cancel"
    )
    outputs_cancel = artifact_store.RunOutputs(lease=lease_cancel)
    outputs_cancel.stage(target, name="srt", write=lambda p: p.write_text("1", encoding="utf-8"))
    with pytest.raises(artifact_store.OutputsSupersededError):
        outputs_cancel.publish(still_current=lambda: False)
    store_root = artifact_store.output_store_root(target, project_root=tmp_path)
    assert not list((store_root / "generations").glob("*")) if (store_root / "generations").exists() else True
    assert not outputs_cancel.staging_dir.exists()

    # 2. Normal publish + export hashes the large JSON sidecar ONCE total.
    hashed_suffixes: list[str] = []
    real_sha = artifact_store._sha256

    def counting_sha(path: Path) -> str:
        hashed_suffixes.append(path.name)
        return real_sha(path)

    monkeypatch.setattr(artifact_store, "_sha256", counting_sha)
    lease = artifact_store.claim_output(
        target, project_root=tmp_path, job_id="j-ok", run_id="r-ok"
    )
    with artifact_store.RunOutputs(lease=lease) as outputs:
        outputs.stage(target, name="srt", write=lambda p: p.write_text("1", encoding="utf-8"))
        outputs.stage_json(
            tmp_path / "tmp" / "sample-a.bilingual.json",
            {"blocks": [{"text": "hello"}]},
            name="bilingual_json",
        )
        outputs.publish()
    artifact_store.export_output(lease)

    assert hashed_suffixes.count("sample-a.bilingual.json") == 1
    assert hashed_suffixes.count("sample-a.srt") == 2  # once in publish, once in export


def test_stage_truncates_overlong_internal_filename_under_max_path(tmp_path):
    long_stem = "very-long-media-title-" * 12
    target = tmp_path / "out" / f"{long_stem}.srt"
    bilingual_target = tmp_path / "tmp" / f"{long_stem}.bilingual.json"
    lease = artifact_store.claim_output(
        target, project_root=tmp_path, job_id="j-long", run_id="r-long"
    )
    with artifact_store.RunOutputs(lease=lease) as outputs:
        outputs.stage(target, name="srt", write=lambda p: p.write_text("1", encoding="utf-8"))
        outputs.stage_json(bilingual_target, {"blocks": []}, name="bilingual_json")
        manifest = outputs.publish()

    for name, rel_path in manifest["files"].items():
        full_path = artifact_store.artifact_path(rel_path, tmp_path)
        assert full_path.is_file()
        assert len(str(full_path)) <= 240
        if name == "srt":
            assert full_path.name.endswith(".srt")
        else:
            assert full_path.name.endswith(".bilingual.json")
    assert manifest["targets"]["srt"].endswith(f"{long_stem}.srt")


# ---------------------------------------------------------------------------
# P0-12 + P1-13: audio, artifacts, aligned_cache, ids, stage_log, memory_safety
# ---------------------------------------------------------------------------


def test_audio_extract_flags_duration_passthrough_and_empty_cache_rejected(monkeypatch, tmp_path):
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    assert pipeline_audio.audio_cache_is_usable(str(empty_file)) is False

    seen_commands: list[list[str]] = []
    seen_timeouts: list[float] = []

    def fake_run(command, *, timeout_s, **kwargs):
        del kwargs
        seen_commands.append(list(command))
        seen_timeouts.append(timeout_s)
        out = command[command.index("-loglevel") - 1]
        write_pcm_wav(out)

    monkeypatch.delenv("AUDIO_EXTRACT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    monkeypatch.setattr(
        pipeline_audio,
        "probe_video_duration_s",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not re-probe duration")),
    )

    pipeline_audio.extract_audio("clip.mp4", str(tmp_path / "out.wav"), duration_s=100.0)

    cmd = seen_commands[0]
    assert "-nostdin" in cmd
    assert "-sn" in cmd
    assert "-dn" in cmd
    assert cmd[cmd.index("-map_metadata") + 1] == "-1"
    assert seen_timeouts[0] == pytest.approx(520.0)


def test_artifacts_snapshot_uses_prior_elapsed_s_and_unknown_default_device(tmp_path):
    started = time.perf_counter() - 12.5
    art = artifacts_module.AsrArtifacts(
        segments=[],
        audio_path=str(tmp_path / "a.wav"),
        job_temp_dir=str(tmp_path),
        asr_details={},
        aligned_segments_path=str(tmp_path / "aligned.json"),
        transcript_path=str(tmp_path / "transcript.json"),
        asr_manifest_path=str(tmp_path / "manifest.json"),
        pipeline_timings={},
        logger=None,
        run_log_path=None,
        audio_cache_key="k",
        video_stem="clip",
        output_dir=str(tmp_path),
        srt_path=str(tmp_path / "clip.srt"),
        bilingual_json_path=str(tmp_path / "clip.bilingual.json"),
        bilingual=False,
        timings_path=str(tmp_path / "clip.timings.json"),
        translation_cache_path=str(tmp_path / "cache.jsonl"),
        asr_log=[],
        audio_cached=False,
        device="",
        backend_label="qwen",
        video_duration_s=10.0,
        pipeline_started=started,
        job_id="j",
    )
    snap_path = artifacts_module.write_translation_artifacts_snapshot(art)
    raw_text = Path(snap_path).read_text(encoding="utf-8")
    raw_json = json.loads(raw_text)
    assert "prior_elapsed_s" in raw_json
    assert "pipeline_started" not in raw_json
    assert raw_json["prior_elapsed_s"] >= 12.0
    # Compact JSON (indent=None) has no pretty-print newlines.
    assert "\n" not in raw_text

    restored = artifacts_module.load_translation_artifacts_snapshot(snap_path)
    assert restored is not None
    assert restored.device == "unknown"
    elapsed_restored = time.perf_counter() - restored.pipeline_started
    assert 12.0 <= elapsed_restored <= 20.0


def test_aligned_cache_logs_miss_reasons_with_diff_keys(tmp_path, caplog):
    cache_file = tmp_path / "clip.aligned_segments.json"
    cache_file.write_text(
        json.dumps(
            {
                "audio_cache_key": "k1",
                "backend": "qwen",
                "cache_signature": {"version": 11, "asr_stage_config": {"ASR_CHUNK_MAX_S": "30"}},
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger=aligned_cache.__name__):
        hit = aligned_cache.try_load_aligned_segments(
            str(cache_file),
            "k1",
            "qwen",
            {"version": 11, "asr_stage_config": {"ASR_CHUNK_MAX_S": "20"}},
        )
    assert hit is None
    assert "signature mismatch" in caplog.text
    assert "asr_stage_config.ASR_CHUNK_MAX_S" in caplog.text


def test_sanitize_job_id_handles_windows_reserved_names_and_non_ascii_collisions():
    assert sanitize_job_id("CON") == "job_CON"
    assert sanitize_job_id("nul.txt") == "job_nul.txt"
    assert sanitize_job_id("com9") == "job_com9"
    assert sanitize_job_id("") == "job"
    first = sanitize_job_id("日本語タイトル甲")
    second = sanitize_job_id("日本語タイトル乙")
    assert first.startswith("job_")
    assert second.startswith("job_")
    assert first != second
    assert sanitize_job_id("日本語タイトル甲") == first


def test_stage_log_parses_spaced_error_values_and_unifies_seconds_coercion():
    assert stage_log._coerce_event_value("12s") == 12.0
    assert stage_log._coerce_event_value("12.5s") == 12.5
    assert stage_log._coerce_event_value("42") == 42
    assert stage_log._coerce_event_value("segments") == "segments"

    parsed = stage_log._parse_stage_extra(
        "elapsed=12s error=RuntimeError('cuda oom on device 0') cached=False"
    )
    assert parsed["elapsed"] == 12.0
    assert parsed["error"] == "RuntimeError('cuda oom on device 0')"
    assert parsed["cached"] is False


def test_memory_safety_reset_shared_vram_baseline(monkeypatch):
    values = iter((50.0, 60.0, 60.0, 65.0))
    monkeypatch.setattr(
        memory_safety,
        "shared_vram_snapshot",
        lambda **_kw: {"shared_vram_mb": next(values), "shared_vram_monitor": "fake"},
    )
    monkeypatch.setattr(memory_safety, "physical_ram_snapshot", lambda _r: {})
    memory_safety.reset_shared_vram_baseline()

    assert memory_safety.runtime_memory_snapshot()["shared_vram_mb"] == 0.0
    assert memory_safety.runtime_memory_snapshot()["shared_vram_mb"] == 10.0

    memory_safety.reset_shared_vram_baseline()
    assert memory_safety.runtime_memory_snapshot()["shared_vram_mb"] == 0.0
    assert memory_safety.runtime_memory_snapshot()["shared_vram_mb"] == 5.0


def test_resolve_jobs_temp_root(monkeypatch, tmp_path):
    monkeypatch.delenv("JOB_TEMP_DIR", raising=False)
    assert pipeline_output.resolve_jobs_temp_root(tmp_path) == (tmp_path / "tmp" / "jobs").resolve()
    monkeypatch.setenv("JOB_TEMP_DIR", "custom/work")
    assert pipeline_output.resolve_jobs_temp_root(tmp_path) == (tmp_path / "custom" / "work").resolve()


# ---------------------------------------------------------------------------
# P0-12 + P0-17(2) + P1-12 + P1-13 + P2-14: main.py unification & isolation
# ---------------------------------------------------------------------------


def test_run_logger_isolated_instances_never_collide_across_same_stem(tmp_path):
    log_a = tmp_path / "job_a" / "20260101_000000_qwen.run.log"
    log_b = tmp_path / "job_b" / "20260101_000000_qwen.run.log"
    art_a = SimpleNamespace(logger=None, run_log_path=log_a)
    art_b = SimpleNamespace(logger=None, run_log_path=log_b)

    before_global_count = len(logging.Logger.manager.loggerDict)
    run_logger.reopen_snapshot_run_logger(art_a)
    run_logger.reopen_snapshot_run_logger(art_b)
    try:
        assert art_a.logger is not art_b.logger
        assert len(logging.Logger.manager.loggerDict) == before_global_count
        art_a.logger.info("line-from-a")
        art_b.logger.info("line-from-b")
    finally:
        run_logger.close_artifacts_logger(art_a)
        run_logger.close_artifacts_logger(art_b)

    assert "line-from-a" in log_a.read_text(encoding="utf-8")
    assert "line-from-b" not in log_a.read_text(encoding="utf-8")
    assert "line-from-b" in log_b.read_text(encoding="utf-8")
    assert "line-from-a" not in log_b.read_text(encoding="utf-8")


def test_unified_translation_outputs_preserve_cue_fields_and_strip_track_from_timings(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="zh",
        translation_max_workers=1,
    )
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )
    monkeypatch.setattr(
        pipeline_audio,
        "extract_audio",
        lambda _v, out, **_kw: write_pcm_wav(out),
    )
    fake_track = {
        "schema": "frame_class_track_v1",
        "classes": ["silence", "speech", "vocalisation"],
        "frames": 10,
        "frame_s": 0.04,
        "data_b64": "AAAA" * 1000,
    }
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        lambda *_a, **_kw: (
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "こんにちは",
                    "words": [{"word": "こんにちは", "start": 0.0, "end": 1.0}],
                    "chunk_acoustic_start": 0.0,
                    "chunk_acoustic_end": 1.0,
                    "chunk_acoustic_duration": 1.0,
                }
            ],
            ["mock"],
            {
                "transcript_chunks": [{"text": "こんにちは"}],
                "stage_timings": {},
                "frame_class_track": fake_track,
            },
        ),
    )
    monkeypatch.setattr(main.translator_module, "generate_global_context", lambda _items: "")
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda items, **_kw: (["你好" for _ in items], [], []),
    )

    real_prepare = main.subtitle_module.prepare_srt_blocks

    def _enrich_prepare(*args, **kwargs):
        cues = real_prepare(*args, **kwargs)
        cues[0] = {
            **cues[0],
            "continues_from_previous": True,
            "continues_into_next": True,
            "text_break_type": "punctuation",
            "text_break_gap_s": 0.25,
        }
        return cues

    monkeypatch.setattr(main.subtitle_module, "prepare_srt_blocks", _enrich_prepare)

    published = {Path(p).name: Path(p) for p in run_pipeline(video_path, ctx)}

    bilingual = json.loads(published["clip.bilingual.json"].read_text(encoding="utf-8"))
    block = bilingual["blocks"][0]
    assert block["continues_from_previous"] is True
    assert block["continues_into_next"] is True
    assert block["text_break_type"] == "punctuation"
    assert block["text_break_gap_s"] == 0.25
    assert block["chunk_acoustic_start"] == 0.0
    assert "postgate_flags" not in block

    timings = json.loads(published["clip.timings.json"].read_text(encoding="utf-8"))
    assert "frame_class_track" not in timings["asr_details"]
    assert timings["asr_details"]["frame_class_track_meta"]["frames"] == 10
    assert "data_b64" not in timings["asr_details"]["frame_class_track_meta"]

    aligned = json.loads(published["clip.aligned_segments.json"].read_text(encoding="utf-8"))
    assert aligned["asr_details"]["frame_class_track"]["data_b64"] == fake_track["data_b64"]


def test_zero_cues_after_cue_plan_skips_backend_lease_and_translation(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="zh",
        translation_max_workers=1,
    )
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )
    monkeypatch.setattr(pipeline_audio, "extract_audio", lambda _v, out, **_kw: write_pcm_wav(out))
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        lambda *_a, **_kw: (
            [
                {"start": 0.0, "end": 0.5, "text": "あっ"},
                {"start": 0.6, "end": 1.1, "text": "んっ"},
                {"start": 1.2, "end": 1.7, "text": "はぁ"},
            ],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        ),
    )
    monkeypatch.setattr(main.subtitle_module, "prepare_srt_blocks", lambda *_a, **_kw: [])
    lease_entered = {"count": 0}

    def fail_lease(*_args, **_kwargs):
        lease_entered["count"] += 1
        raise AssertionError("0-cue job must not acquire translation backend_lease")

    monkeypatch.setattr(main.llm_backends, "backend_lease", fail_lease)
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("0-cue job must not translate")),
    )

    output_paths = run_pipeline(video_path, ctx)
    assert lease_entered["count"] == 0
    assert (tmp_path / "out" / "clip.srt").is_file()
    assert output_paths


def test_aligned_cache_hit_across_jobs_skips_wav_extraction_even_without_old_audio(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        temp_root,
        job_id="new-job",
        skip_translation=True,
    )
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )
    monkeypatch.setattr(pipeline_audio, "probe_video_duration_s", lambda *_a, **_kw: 5.0)
    monkeypatch.setattr(
        main.aligned_cache_module,
        "try_load_aligned_segments",
        lambda *_a, **_kw: {
            "segments": [{"start": 0.0, "end": 1.0, "text": "キャッシュ済み"}],
            "asr_details": {"transcript_chunks": []},
            "asr_log": ["cached"],
        },
    )
    monkeypatch.setattr(
        pipeline_audio,
        "extract_audio",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not extract wav on aligned_cache hit")),
    )

    artifacts = main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id="new-job",
        cache_job_id="old-job-without-audio",
    )
    assert artifacts.audio_cached is True
    assert artifacts.segments[0]["text"] == "キャッシュ済み"


def test_materialize_usable_audio_replaces_corrupt_destination(tmp_path):
    source = tmp_path / "src.wav"
    dest = tmp_path / "dst.wav"
    write_pcm_wav(source, seconds=0.2)

    # Corrupt destination (truncated RIFF header claiming 0 frames).
    write_pcm_wav(dest, seconds=0.2)
    raw = bytearray(dest.read_bytes())
    raw[4:8] = (0).to_bytes(4, "little")
    raw[40:44] = (0).to_bytes(4, "little")
    dest.write_bytes(bytes(raw))
    assert pipeline_audio.audio_cache_is_usable(str(dest)) is False

    assert main._materialize_usable_audio(source, dest) is True
    assert pipeline_audio.audio_cache_is_usable(str(dest)) is True

    # Corrupt source must return False without touching a valid destination.
    bad_source = tmp_path / "bad_src.wav"
    bad_source.write_bytes(bytes(raw))
    valid_bytes = dest.read_bytes()
    assert main._materialize_usable_audio(bad_source, dest) is False
    assert dest.read_bytes() == valid_bytes


def test_empty_segments_and_skip_translation_share_unified_output_contract(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    log_dir = tmp_path / "run_logs"
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )
    monkeypatch.setattr(pipeline_audio, "extract_audio", lambda _v, out, **_kw: write_pcm_wav(out))

    # 1. Empty segments branch must emit `run_done` and include full bilingual/timings schema.
    ctx_empty = make_job_context(
        video_path,
        tmp_path / "out_empty",
        tmp_path / "jobs_empty",
        run_log_enabled=True,
        run_log_dir=log_dir,
    )
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        lambda *_a, **_kw: ([], [], {"transcript_chunks": [], "stage_timings": {}}),
    )
    published_empty = {Path(p).name: Path(p) for p in run_pipeline(video_path, ctx_empty)}
    timings_empty = json.loads(published_empty["clip.timings.json"].read_text(encoding="utf-8"))
    assert timings_empty["counts"] == {
        "transcript_chunks": 0,
        "segments": 0,
        "translation_cues": 0,
        "blocks": 0,
    }
    assert timings_empty["translation_api_retry_events"] == []
    bilingual_empty = json.loads(published_empty["clip.bilingual.json"].read_text(encoding="utf-8"))
    assert bilingual_empty["backend"] == "mock_asr"
    assert bilingual_empty["blocks"] == []
    empty_logs = list(log_dir.rglob("*.run.log"))
    assert empty_logs and "run_done " in empty_logs[0].read_text(encoding="utf-8")

    # 2. Skip-translation branch must write back asr_details["subtitle_cue_plan"] and log it.
    log_dir_ja = tmp_path / "run_logs_ja"
    ctx_ja = make_job_context(
        video_path,
        tmp_path / "out_ja",
        tmp_path / "jobs_ja",
        skip_translation=True,
        run_log_enabled=True,
        run_log_dir=log_dir_ja,
    )
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        lambda *_a, **_kw: (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        ),
    )
    artifacts_ja = main.run_asr_alignment(str(video_path), ctx=ctx_ja, job_id=ctx_ja.job_id)
    assert artifacts_ja.logger is None
    main.run_translation_and_write(str(video_path), artifacts_ja, ctx=ctx_ja, job_id=ctx_ja.job_id)
    assert "subtitle_cue_plan" in artifacts_ja.asr_details
    ja_logs = list(log_dir_ja.rglob("*.run.log"))
    assert ja_logs and "subtitle_cue_plan segments_before=1" in ja_logs[0].read_text(encoding="utf-8")


def test_run_logger_fallback_uses_job_temp_dir(monkeypatch, tmp_path):
    primary = tmp_path / "blocked_logs"
    job_temp = tmp_path / "job_temp"
    job_temp.mkdir()
    ctx = make_job_context(
        tmp_path / "clip.mp4",
        tmp_path / "out",
        tmp_path / "jobs",
        run_log_enabled=True,
        run_log_dir=primary,
    )
    ctx.job_temp_dir = str(job_temp)

    real_file_handler = logging.FileHandler

    def fail_primary(filename, *args, **kwargs):
        if Path(filename).is_relative_to(primary):
            raise PermissionError("primary log dir read-only")
        return real_file_handler(filename, *args, **kwargs)

    monkeypatch.setattr(logging, "FileHandler", fail_primary)
    logger, log_path = run_logger.setup_run_logger(
        "job-fb",
        "qwen",
        ctx,
        project_root=tmp_path,
    )
    try:
        assert log_path.is_relative_to(job_temp / "log")
        assert log_path.is_file()
    finally:
        run_logger.close_run_logger(logger)
