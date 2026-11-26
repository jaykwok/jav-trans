"""Tests for process-wide diagnostics, crash traceability, and audit regression guards."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# core.diagnostics
# ---------------------------------------------------------------------------


def test_format_exit_code_formats_windows_ntstatus():
    from core.diagnostics import format_exit_code

    assert format_exit_code(None) == "None"
    assert format_exit_code(0) == "0"
    assert format_exit_code(1) == "1"

    formatted_unsigned = format_exit_code(0xC0000005)
    formatted_signed = format_exit_code(-1073741819)
    if os.name == "nt":
        assert "0xC0000005" in formatted_unsigned
        assert "STATUS_ACCESS_VIOLATION" in formatted_unsigned
        assert "0xC0000005" in formatted_signed
        assert "STATUS_ACCESS_VIOLATION" in formatted_signed
        assert "0xC0000999" in format_exit_code(0xC0000999)
    else:
        assert formatted_unsigned == str(0xC0000005)


def test_format_exception_chain_walks_cause_and_context():
    from core.diagnostics import format_exception_brief, format_exception_chain

    assert format_exception_brief(RuntimeError()) == "RuntimeError"

    try:
        try:
            raise TimeoutError("socket read timed out")
        except TimeoutError as root:
            raise ConnectionError("request aborted") from root
    except ConnectionError as outer:
        chain = format_exception_chain(outer)
        assert "ConnectionError: request aborted" in chain
        assert "TimeoutError: socket read timed out" in chain
        assert " <- " in chain

    short = format_exception_chain(RuntimeError("x" * 200), max_chars=40)
    assert len(short) <= 40
    assert short.endswith("...")


def test_redact_url_credentials_scrubs_standalone_and_embedded_urls():
    from core.diagnostics import redact_url_credentials

    assert redact_url_credentials("http://127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert (
        redact_url_credentials("http://alice:s3cr3t@127.0.0.1:7890")
        == "http://***@127.0.0.1:7890"
    )
    assert (
        redact_url_credentials("socks5://user:p@ss@proxy.local:1080")
        == "socks5://***@proxy.local:1080"
    )
    embedded = redact_url_credentials(
        "ConnectError: proxy http://alice:s3cr3t@127.0.0.1:7890 refused"
    )
    assert "alice" not in embedded
    assert "s3cr3t" not in embedded
    assert "http://***@127.0.0.1:7890" in embedded


def test_configure_process_logging_uses_per_process_files_idempotently(tmp_path):
    from core import diagnostics

    log_path = diagnostics.configure_process_logging(
        "test_app.log",
        log_dir=tmp_path,
        enable_faulthandler=True,
        install_hooks=False,
    )
    assert log_path is not None
    assert log_path.name == f"test_app-{os.getpid()}.log"
    assert log_path.exists()
    fault_path = log_path.with_name(f"test_app-{os.getpid()}-fault.log")
    assert fault_path.exists()

    root = logging.getLogger()
    before_count = len(root.handlers)
    second_path = diagnostics.configure_process_logging(
        "test_app.log",
        log_dir=tmp_path,
        enable_faulthandler=True,
        install_hooks=False,
    )
    assert second_path == log_path
    assert len(root.handlers) == before_count

    logging.getLogger("test.diagnostics.check").info("hello diagnostic log")
    for h in root.handlers:
        h.flush()
    assert "hello diagnostic log" in log_path.read_text(encoding="utf-8")

    # Clean up test handler & fault stream so global logging state stays tidy.
    handler = diagnostics._INSTALLED_HANDLERS.pop(log_path, None)
    if handler is not None:
        root.removeHandler(handler)
        handler.close()
    stream = diagnostics._FAULT_STREAMS.pop(fault_path, None)
    if stream is not None:
        stream.close()


def test_two_concurrent_processes_rotate_app_logs_without_dropping_critical(tmp_path):
    """Two real processes sharing one log dir must rotate without PermissionError
    or losing CRITICAL records on Windows."""
    project_root = Path(__file__).resolve().parents[1]
    script = (
        "import logging, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(project_root / 'src')!r})\n"
        "from core.diagnostics import configure_process_logging\n"
        "log_dir = Path(sys.argv[1])\n"
        "marker = sys.argv[2]\n"
        "path = configure_process_logging('app.log', log_dir=log_dir, max_bytes=64*1024, backup_count=2, enable_faulthandler=False, install_hooks=False)\n"
        "log = logging.getLogger('multi_proc_test')\n"
        "payload = 'X' * 2048\n"
        "for i in range(60):\n"
        "    log.info('%s line %d %s', marker, i, payload)\n"
        "log.critical('CRITICAL_MARKER_%s pid=%d', marker, os.getpid())\n"
        "logging.shutdown()\n"
        "print(f'{os.getpid()}|{path}')\n"
    )
    p1 = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), "PROC1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), "PROC2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out1, err1 = p1.communicate(timeout=20)
    out2, err2 = p2.communicate(timeout=20)
    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    assert "PermissionError" not in err1
    assert "PermissionError" not in err2

    child_pid1, raw_path1 = out1.strip().split("|", 1)
    child_pid2, raw_path2 = out2.strip().split("|", 1)
    path1 = Path(raw_path1)
    path2 = Path(raw_path2)
    assert path1 != path2
    assert f"CRITICAL_MARKER_PROC1 pid={child_pid1}" in path1.read_text(encoding="utf-8")
    assert f"CRITICAL_MARKER_PROC2 pid={child_pid2}" in path2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# bootstrap.py
# ---------------------------------------------------------------------------


def test_bootstrap_format_exit_code_and_run_cli_crash_guard(monkeypatch, tmp_path):
    import bootstrap

    if os.name == "nt":
        assert "0xC0000005" in bootstrap._format_exit_code(-1073741819)

    restored: list[bool] = []
    logged: list[str] = []
    monkeypatch.setattr(bootstrap, "set_console_visible", lambda vis: restored.append(vis))
    monkeypatch.setattr(bootstrap, "log", lambda msg="": logged.append(str(msg)))
    monkeypatch.setattr(bootstrap, "_bootstrap_log_path", lambda: tmp_path / "bootstrap.log")

    def _boom(_argv=None):
        raise RuntimeError("boom in bootstrap")

    monkeypatch.setattr(bootstrap, "main", _boom)
    prev_enabled = bootstrap._BOOTSTRAP_LOG_ENABLED
    try:
        rc = bootstrap._run_cli([])
    finally:
        bootstrap._BOOTSTRAP_LOG_ENABLED = prev_enabled

    assert rc == 1
    assert restored == [True]
    combined = "\n".join(logged)
    assert "RuntimeError: boom in bootstrap" in combined
    assert "Traceback" in combined


# ---------------------------------------------------------------------------
# SystemExit hardening (issue #4 & #5 real-world sandbox shim failure mode)
# ---------------------------------------------------------------------------


def test_cleanup_job_temp_survives_system_exit_from_unlink(monkeypatch, tmp_path, caplog):
    from pipeline.cleanup import cleanup_job_temp

    job_dir = tmp_path / "job_1"
    job_dir.mkdir()
    target = job_dir / "asr_chunk_0001.wav"
    target.write_bytes(b"RIFF")

    def _exit_unlink(self, *args, **kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(Path, "unlink", _exit_unlink)
    with caplog.at_level(logging.WARNING):
        cleanup_job_temp(str(job_dir))

    assert "SystemExit" in caplog.text


def test_asr_transcribe_delete_path_survives_system_exit(monkeypatch, tmp_path, caplog):
    import shutil
    from asr.transcribe import _delete_path_for_cleanup

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_001.wav").write_bytes(b"data")

    def _exit_rmtree(*args, **kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(shutil, "rmtree", _exit_rmtree)
    with caplog.at_level(logging.WARNING):
        _delete_path_for_cleanup(chunk_dir)

    assert "SystemExit" in caplog.text


def test_launcher_sweep_and_cleanup_survive_system_exit(monkeypatch, tmp_path, caplog):
    import shutil

    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "launcher.py").read_text(encoding="utf-8")
    start = source.index("# Globs inside tmp/")
    end = source.index("def _version_tuple(", start)
    namespace: dict = {}
    preamble = "import json, os, shutil, time\nfrom pathlib import Path\n"
    exec(compile(preamble + source[start:end], "launcher.py", "exec"), namespace)  # noqa: S102

    chunk_dir = tmp_path / "tmp" / "chunks" / "job-a"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "chunk_0000.wav").write_bytes(b"audio")

    monkeypatch.setattr(shutil, "rmtree", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(1)))
    with caplog.at_level(logging.WARNING):
        namespace["_sweep_temp"](tmp_path)

    assert "SystemExit" in caplog.text


def test_gpu_worker_serve_catches_unexpected_system_exit_and_reports_error(monkeypatch):
    from pipeline import gpu_worker

    class _FakeConn:
        def __init__(self, incoming):
            self._incoming = list(incoming)
            self.sent: list[dict] = []

        def poll(self, timeout=0.0):
            return bool(self._incoming)

        def recv(self):
            return self._incoming.pop(0)

        def send(self, msg):
            self.sent.append(msg)

    conn = _FakeConn(
        [
            {
                "op": "transcribe_and_align",
                "job_id": "job-sysexit",
                "run_id": "run-1",
                "audio_path": "dummy.wav",
                "device": "cuda",
                "env": {"ASR_STAGE_WORKER_MOCK": "1"},
            }
        ]
    )

    monkeypatch.setattr(gpu_worker, "_adaptive_runtime_tuning", lambda *a, **kw: {})
    monkeypatch.setattr(gpu_worker, "_clear_worker_cuda", lambda: None)
    monkeypatch.setattr(
        gpu_worker.GpuModelManager,
        "run_transcribe_and_align",
        lambda self, **kw: (_ for _ in ()).throw(SystemExit(1)),
    )

    with pytest.raises(SystemExit) as excinfo:
        gpu_worker._serve_worker(conn)

    assert excinfo.value.code == 0  # error was successfully sent back over pipe
    error_msgs = [m for m in conn.sent if m.get("op") == "error"]
    assert len(error_msgs) == 1
    assert error_msgs[0]["job_id"] == "job-sysexit"
    assert "SystemExit(1)" in error_msgs[0]["detail"]


def test_gpu_worker_shutdown_returns_cleanly_without_system_exit(monkeypatch):
    from pipeline import gpu_worker

    class _FakeConn:
        def __init__(self):
            self._incoming = [{"op": "shutdown"}]
            self.sent: list[dict] = []

        def poll(self, timeout=0.0):
            return bool(self._incoming)

        def recv(self):
            return self._incoming.pop(0)

        def send(self, msg):
            self.sent.append(msg)

    conn = _FakeConn()
    cleared: list[bool] = []
    monkeypatch.setattr(gpu_worker, "_clear_worker_cuda", lambda: cleared.append(True))
    gpu_worker._serve_worker(conn)
    assert any(m.get("op") == "ready" for m in conn.sent)
    assert cleared == [True]


def test_pipeline_manager_converts_executor_system_exit_to_runtime_error(monkeypatch, tmp_path):
    from web import pipeline_manager
    from web.models import JobSpec, JobState

    job = JobState(
        id="job-sys-exit",
        status="queued",
        created_at="2026-01-01T00:00:00Z",
        spec=JobSpec(video_paths=[str(tmp_path / "sample.mp4")]),
    )
    cancel_event = threading.Event()

    monkeypatch.setattr(
        pipeline_manager,
        "run_asr_alignment",
        lambda *a, **kw: (_ for _ in ()).throw(SystemExit(1)),
    )
    with pytest.raises(RuntimeError, match=r"SystemExit\(1\)"):
        pipeline_manager._run_asr_alignment(job, cancel_event, "run-1")

    monkeypatch.setattr(
        pipeline_manager,
        "run_translation_and_write",
        lambda *a, **kw: (_ for _ in ()).throw(SystemExit(2)),
    )
    with pytest.raises(RuntimeError, match=r"SystemExit\(2\)"):
        pipeline_manager._run_translation_and_write(job, object(), cancel_event, "run-1")


# ---------------------------------------------------------------------------
# ASR zombie single-wait & fatal CUDA error propagation
# ---------------------------------------------------------------------------


def test_asr_finalize_waits_at_most_once_on_stuck_zombie_future(monkeypatch, tmp_path):
    from asr import local_backend

    backend = local_backend.LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    backend._alignment_head = SimpleNamespace(checkpoint_path="head.pt")

    release = threading.Event()
    wait_calls = 0

    class _StuckFuture:
        def done(self):
            return release.is_set()

        def result(self, timeout=None):
            nonlocal wait_calls
            wait_calls += 1
            if not release.wait(timeout=0.05):
                raise concurrent.futures.TimeoutError()
            return []

    backend._zombie_future = _StuckFuture()
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT_S", "1.0")
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 1.0)

    chunks = [
        {
            "normalized_path": str(tmp_path / f"chunk_{i}.wav"),
            "duration": 2.0,
            "language": "Japanese",
            "text": f"こんにちは{i}",
            "raw_text": f"こんにちは{i}",
            "log": [],
        }
        for i in range(4)
    ]
    started = time.monotonic()
    results = backend.finalize_text_results(chunks)
    elapsed = time.monotonic() - started

    assert len(results) == 4
    assert wait_calls == 1, f"expected single wait on stuck zombie, got {wait_calls}"
    assert elapsed < 0.5
    for item, _log in results:
        assert "zombie ASR generate worker" in item["align_error"]


def test_cuda_illegal_memory_access_propagates_out_of_alignment_and_local_backend():
    import numpy as np
    from asr.alignment import AlignmentHead, AlignmentVocab
    from asr.local_backend import LocalAsrBackend

    vocab = AlignmentVocab(("あ",))

    class _ExplodingHead(AlignmentHead):
        def log_probs(self, features):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    head = _ExplodingHead(module=None, vocab=vocab, upsample=2, device="cpu")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        head.align_extent(np.zeros((4, 8), dtype=np.float32), "あ")

    backend = LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    backend._alignment_head = head
    with pytest.raises(RuntimeError, match="illegal memory access"):
        backend._align_characters(
            "unused.wav",
            "あ",
            [],
            cached_features=np.zeros((4, 8), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# llama-server crash attribution & log scoping
# ---------------------------------------------------------------------------


def test_llamacpp_server_mid_translation_crash_raises_local_server_crashed_error(tmp_path):
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import LocalServerCrashedError

    backend = LlamaCppServerBackend()
    log_path = tmp_path / "llamacpp_server.log"
    old_entry = "=== previous launch ===\nOLD_NOISE_DO_NOT_INCLUDE\n"
    new_entry = "=== llama-server start ===\ncudaMalloc failed: out of memory\n"
    log_path.write_text(old_entry + new_entry, encoding="utf-8")

    class _DeadProc:
        pid = 4321

        def poll(self):
            return -1073741819

        def wait(self, timeout=None):
            return -1073741819

    class _FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionResetError("connection reset by peer")

    backend._port = 18080
    backend._proc = _DeadProc()
    backend._log_path = log_path
    backend._log_start_offset = len(old_entry.encode("utf-8"))
    backend._make_async_client = lambda port: _FailingClient()

    with pytest.raises(LocalServerCrashedError) as excinfo:
        backend._request_completion({"model": "local", "messages": []}, None)

    msg = str(excinfo.value)
    assert "4321" in msg
    assert "cudaMalloc failed: out of memory" in msg
    assert "OLD_NOISE_DO_NOT_INCLUDE" not in msg
    if os.name == "nt":
        assert "0xC0000005" in msg


def test_llamacpp_server_rejects_request_when_port_is_none():
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    backend._port = None
    with pytest.raises(RuntimeError, match="已停止或尚未就绪"):
        backend._make_async_client(None)


# ---------------------------------------------------------------------------
# OpenAI Responses SSE error (nested AND real SDK top-level) & refusal events
# ---------------------------------------------------------------------------


def test_openai_compat_handles_refusal_and_top_level_response_error_event(monkeypatch):
    from llm.backends import openai_compat
    from llm.errors import ContentPolicyRefusalError, RetryableTranslationFormatError

    class _FakeStream:
        def __init__(self, events):
            self._events = list(events)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return self._events.pop(0)

        async def close(self):
            return None

    def _run_with_events(events):
        class _FakeClient:
            class responses:
                @staticmethod
                async def create(**kwargs):
                    return _FakeStream(events)

            async def close(self):
                return None

        monkeypatch.setattr(openai_compat, "_make_async_client", lambda: _FakeClient())
        return openai_compat._chat_responses(
            messages=[{"role": "user", "content": "hi"}],
            expected_count=1,
            on_progress=None,
            reasoning_effort=None,
            on_usage=None,
            cancel_event=None,
            response_schema=None,
            temperature=0.6,
            top_p=0.9,
            max_tokens=1024,
        )

    with pytest.raises(ContentPolicyRefusalError, match="cannot assist"):
        _run_with_events([{"type": "response.refusal.done", "refusal": "cannot assist"}])

    # Real OpenAI SDK ResponseErrorEvent shape: code & message at top level.
    sdk_policy_event = SimpleNamespace(
        type="error",
        code="cyber_policy",
        message="blocked by safety policy",
        param=None,
        sequence_number=1,
    )
    with pytest.raises(ContentPolicyRefusalError, match="cyber_policy"):
        _run_with_events([sdk_policy_event])

    with pytest.raises(RetryableTranslationFormatError, match="upstream overloaded"):
        _run_with_events(
            [{"type": "error", "error": {"code": "server_error", "message": "upstream overloaded"}}]
        )


# ---------------------------------------------------------------------------
# main.py early run.log + failure timings preservation across publish failure
# ---------------------------------------------------------------------------


def test_run_asr_alignment_logs_preflight_failure_and_preserves_existing_timings_on_publish_failure(
    monkeypatch, tmp_path
):
    import main as pipeline_main
    from core.job_context import JobContext

    missing_video = tmp_path / "sample-a.mp4"
    temp_dir = tmp_path / "job_temp"
    temp_dir.mkdir()
    log_root = tmp_path / "run_logs"
    monkeypatch.setattr(pipeline_main, "_run_log_dir", lambda _ctx: log_root)

    ctx = JobContext.from_spec(
        SimpleNamespace(video_paths=[str(missing_video)], subtitle_mode="bilingual"),
        "job-missing",
        str(temp_dir),
        str(temp_dir / "cache.jsonl"),
    )

    with pytest.raises(RuntimeError):
        pipeline_main.run_asr_alignment(
            str(missing_video),
            ctx=ctx,
            job_id="job-missing",
        )

    run_logs = list(log_root.rglob("*.run.log"))
    assert len(run_logs) == 1
    log_text = run_logs[0].read_text(encoding="utf-8")
    assert "run_failed stage=asr_alignment" in log_text
    assert "Traceback" in log_text

    # Simulate pre-existing timings.json written during run-1 just before _publish_outputs fails.
    timings_path = temp_dir / "sample-a.timings.json"
    timings_path.write_text(
        json.dumps(
            {
                "video_path": str(missing_video),
                "run_id": "run-1",
                "stage_timings": {"translation_s": 12.3},
                "translation_request_timings": [{"batch": 0, "elapsed_s": 2.5}],
                "translation_api_retry_events": [{"attempt": 1, "error": "rate_limit"}],
            }
        ),
        encoding="utf-8",
    )
    fake_artifacts = SimpleNamespace(
        job_id="job-missing",
        job_temp_dir=str(temp_dir),
        audio_path=str(temp_dir / "audio.wav"),
        audio_cached=False,
        device="cuda",
        backend_label="qwen",
        segments=[{"start": 0.0, "end": 1.0, "ja_text": "こんにちは"}],
        pipeline_timings={"asr_s": 1.2},
        asr_details={"transcript_chunks": []},
        asr_log=[],
        srt_path=str(temp_dir / "sample-a.srt"),
        asr_manifest_path="",
        transcript_path="",
        aligned_segments_path="",
        bilingual_json_path="",
        timings_path=str(timings_path),
        run_log_path=str(run_logs[0]),
        logger=None,
    )
    publish_exc = OSError("disk full during export")
    pipeline_main._write_failure_timings_best_effort(
        video_path=str(missing_video),
        artifacts=fake_artifacts,
        stage="translation_and_write",
        exc=publish_exc,
        run_id="run-1",
    )

    payload = json.loads(timings_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["run_id"] == "run-1"
    assert payload["failed_stage"] == "translation_and_write"
    assert "disk full during export" in payload["error"]["message"]
    assert payload["translation_request_timings"] == [{"batch": 0, "elapsed_s": 2.5}]
    assert payload["translation_api_retry_events"] == [{"attempt": 1, "error": "rate_limit"}]
    assert payload["stage_timings"]["translation_s"] == 12.3

    # A subsequent retry with a new run_id must NOT inherit run-1's request/retry records.
    pipeline_main._write_failure_timings_best_effort(
        video_path=str(missing_video),
        artifacts=fake_artifacts,
        stage="translation_and_write",
        exc=RuntimeError("preflight failed before request"),
        run_id="run-2",
    )
    retry_payload = json.loads(timings_path.read_text(encoding="utf-8"))
    assert retry_payload["run_id"] == "run-2"
    assert retry_payload["translation_request_timings"] == []
    assert retry_payload["translation_api_retry_events"] == []
    assert "translation_s" not in retry_payload["stage_timings"]


def test_prune_old_process_logs_strictly_matches_pid_and_skips_locked_group(monkeypatch, tmp_path):
    from core.diagnostics import _prune_old_process_logs

    # Non-PID file that shares prefix `app-` must never be selected.
    unrelated = tmp_path / "app-investigation.log"
    unrelated.write_text("manual note", encoding="utf-8")

    # Three generated PID groups with distinct mtimes (1001 oldest, 1002 middle, 1003 newest).
    p1001 = tmp_path / "app-1001.log"
    p1001_bak = tmp_path / "app-1001.log.1"
    p1001_fault = tmp_path / "app-1001-fault.log"
    p1002 = tmp_path / "app-1002.log"
    p1002_bak = tmp_path / "app-1002.log.1"
    p1002_fault = tmp_path / "app-1002-fault.log"
    p1003 = tmp_path / "app-1003.log"
    for f in (p1001, p1001_bak, p1001_fault, p1002, p1002_bak, p1002_fault, p1003):
        f.write_text("log", encoding="utf-8")
    os.utime(p1001, (1000, 1000))
    os.utime(p1002, (2000, 2000))
    os.utime(p1003, (3000, 3000))

    real_unlink = Path.unlink

    def _locked_unlink(self, *args, **kwargs):
        if self.name == "app-1001.log":
            raise PermissionError("file held open by sibling window")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _locked_unlink)
    _prune_old_process_logs(tmp_path, stem="app", suffix=".log", keep=1, exclude=set())

    assert unrelated.exists()
    # Locked PID 1001 group must be skipped as a whole (main + backup + fault).
    assert p1001.exists()
    assert p1001_bak.exists()
    assert p1001_fault.exists()
    # Unlocked stale PID 1002 group is pruned completely.
    assert not p1002.exists()
    assert not p1002_bak.exists()
    assert not p1002_fault.exists()
    # Newest PID 1003 is kept.
    assert p1003.exists()


def test_alignment_model_and_head_load_propagate_fatal_cuda_errors(monkeypatch):
    from asr import transcribe as asr_transcribe
    from asr.alignment import AlignmentHead
    from asr.local_backend import LocalAsrBackend

    # 1. Text-cache-hit reload in transcribe._align_TRANSCRIPTION_results.
    class _ExplodingBackend:
        model = None

        def load(self, on_stage=None):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        def finalize_text_results(self, results, on_stage=None):
            return []

    monkeypatch.setattr(asr_transcribe, "alignment_head_configured", lambda: True)
    monkeypatch.setattr(
        asr_transcribe._result_cache_module, "finalize_lookup", lambda *a, **kw: None
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        asr_transcribe._align_TRANSCRIPTION_results(
            _ExplodingBackend(),
            [{"normalized_path": "a.wav", "text": "あ", "duration": 1.0, "language": "Japanese"}],
            on_stage=None,
        )

    # 2. Primary alignment head load in LocalAsrBackend._resolve_alignment_head.
    monkeypatch.setattr(
        AlignmentHead,
        "from_env",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                RuntimeError("CUDA error: an illegal memory access was encountered")
            )
        ),
    )
    backend = LocalAsrBackend("cpu")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        backend._resolve_alignment_head([])
    assert backend._alignment_head is False

    # 3. Audio chunking alignment head load in asr.pipeline._blank_runs_for_audio.
    from asr import pipeline as asr_pipeline

    monkeypatch.setattr(asr_pipeline, "_get_wav_duration", lambda _path: 5.0)
    monkeypatch.setattr(asr_pipeline, "AlignmentHead", AlignmentHead)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        asr_pipeline._blank_runs_for_audio("dummy.wav")


def test_pipeline_manager_snapshot_error_without_run_logger_and_identity_cleanup(
    monkeypatch, tmp_path
):
    from core import events
    from pipeline.artifacts import AsrArtifacts
    from utils import hf_progress
    from web import pipeline_manager
    from web.models import JobSpec, JobState

    events.set_current_run("", "")
    events._thread_local.video = ""
    hf_progress.set_current_job_id("", "")

    job = JobState(
        id="job-clean",
        status="queued",
        created_at="2026-01-01T00:00:00Z",
        spec=JobSpec(video_paths=[str(tmp_path / "sample-a.mp4")]),
    )
    cancel_event = threading.Event()
    fake_artifacts = AsrArtifacts(
        segments=[],
        audio_path=str(tmp_path / "a.wav"),
        audio_cache_key="k",
        audio_cached=False,
        asr_details={},
        asr_log=[],
        backend_label="qwen",
        device="cuda",
        job_id="job-clean",
        job_temp_dir=str(tmp_path),
        output_dir=str(tmp_path),
        pipeline_started=0.0,
        pipeline_timings={},
        video_duration_s=1.0,
        video_stem="sample-a",
        aligned_segments_path="",
        logger=None,
        run_log_path=None,
        srt_path=str(tmp_path / "sample-a.srt"),
        transcript_path="",
        asr_manifest_path="",
        bilingual_json_path="",
        timings_path="",
        translation_cache_path="",
        bilingual=True,
    )

    # Snapshot failure with logger=None must raise the original OSError, not AttributeError,
    # and must leave thread-local identity clean.
    monkeypatch.setattr(pipeline_manager, "run_asr_alignment", lambda *a, **kw: fake_artifacts)
    monkeypatch.setattr(
        pipeline_manager,
        "write_translation_artifacts_snapshot",
        lambda _art: (_ for _ in ()).throw(OSError("disk full on snapshot")),
    )
    with pytest.raises(OSError, match="disk full on snapshot"):
        pipeline_manager._run_asr_alignment(job, cancel_event, "run-snap-err")
    assert events._current_job_id() == ""
    assert events._current_run_id() == ""
    assert hf_progress.current_identity() == ("", "", "")

    # Happy path ASR & translation wrappers must also restore thread identity before returning.
    monkeypatch.setattr(pipeline_manager, "write_translation_artifacts_snapshot", lambda _art: None)
    pipeline_manager._run_asr_alignment(job, cancel_event, "run-ok")
    assert events._current_job_id() == ""
    assert events._current_run_id() == ""
    assert hf_progress.current_identity() == ("", "", "")

    monkeypatch.setattr(pipeline_manager, "run_translation_and_write", lambda *a, **kw: [])
    pipeline_manager._run_translation_and_write(job, fake_artifacts, cancel_event, "run-tr-ok")
    assert events._current_job_id() == ""
    assert events._current_run_id() == ""
    assert hf_progress.current_identity() == ("", "", "")


def test_bootstrap_launch_failure_hints_points_to_per_pid_app_log(monkeypatch):
    import bootstrap

    lines: list[str] = []
    monkeypatch.setattr(bootstrap, "log", lambda msg="": lines.append(str(msg)))
    bootstrap._log_launch_failure_hints()
    combined = "\n".join(lines)
    assert "app-<PID>.log" in combined
    assert "tmp\\log\\app.log" not in combined


def test_gpu_worker_final_cuda_cleanup_fatal_error_sends_error_not_result(monkeypatch):
    import types
    from pipeline import gpu_worker

    class _FakeConn:
        def __init__(self, incoming):
            self._incoming = list(incoming)
            self.sent: list[dict] = []

        def poll(self, timeout=0.0):
            return bool(self._incoming)

        def recv(self):
            return self._incoming.pop(0)

        def send(self, msg):
            self.sent.append(msg)

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: (_ for _ in ()).throw(
            RuntimeError("CUDA error: an illegal memory access was encountered")
        ),
        current_device=lambda: 0,
        max_memory_allocated=lambda _idx=0: 0,
        memory_allocated=lambda _idx=0: 0,
        reset_peak_memory_stats=lambda: None,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gpu_worker, "_adaptive_runtime_tuning", lambda *a, **kw: {})

    # 1. Fatal CUDA error during final CUDA cleanup on happy path must send error, never result.
    monkeypatch.setattr(
        gpu_worker.GpuModelManager,
        "run_transcribe_and_align",
        lambda self, **kw: ([], [], {"device": "cuda", "stage_worker": {}}),
    )
    conn = _FakeConn(
        [
            {
                "op": "transcribe_and_align",
                "job_id": "job-final-cuda",
                "run_id": "run-1",
                "audio_path": "dummy.wav",
                "device": "cuda",
                "env": {"ASR_STAGE_WORKER_MOCK": "1"},
            }
        ]
    )
    with pytest.raises(SystemExit) as excinfo:
        gpu_worker._serve_worker(conn)
    assert excinfo.value.code == 0
    assert not any(m.get("op") == "result" for m in conn.sent)
    error_msgs = [m for m in conn.sent if m.get("op") == "error"]
    assert len(error_msgs) == 1
    assert "illegal memory access" in error_msgs[0]["detail"]

    # 2. When handling an earlier failure, fatal CUDA error in error-path cleanup must keep first failure.
    monkeypatch.setattr(
        gpu_worker.GpuModelManager,
        "run_transcribe_and_align",
        lambda self, **kw: (_ for _ in ()).throw(ValueError("original decode failure")),
    )
    conn2 = _FakeConn(
        [
            {
                "op": "transcribe_and_align",
                "job_id": "job-first-err",
                "run_id": "run-2",
                "audio_path": "dummy.wav",
                "device": "cuda",
                "env": {"ASR_STAGE_WORKER_MOCK": "1"},
            }
        ]
    )
    with pytest.raises(SystemExit) as excinfo2:
        gpu_worker._serve_worker(conn2)
    assert excinfo2.value.code == 0
    error_msgs2 = [m for m in conn2.sent if m.get("op") == "error"]
    assert len(error_msgs2) == 1
    assert "original decode failure" in error_msgs2[0]["detail"]


def test_zombie_worker_fatal_cuda_error_escalates_on_next_join(monkeypatch):
    from asr import local_backend

    backend = local_backend.LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    release = threading.Event()

    def _slow_then_cuda_crash():
        release.wait(timeout=5.0)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_slow_then_cuda_crash)
    backend._zombie_future = future
    backend._zombie_executor = executor
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 1.0)

    # Release the zombie thread so it finishes with a fatal CUDA exception.
    release.set()
    with pytest.raises(RuntimeError, match="illegal memory access"):
        backend._join_zombie_worker()
    assert backend._zombie_future is None
    assert backend._zombie_executor is None
    assert backend.model is None
    assert backend.processor is None


def test_vram_budget_probe_and_cuda_memory_snapshot_propagate_fatal_cuda_error(monkeypatch):
    import types
    from asr import pipeline as asr_pipeline
    from asr import transcribe as asr_transcribe

    def _cuda_illegal(*_a, **_kw):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        current_device=lambda: 0,
        max_memory_reserved=_cuda_illegal,
        memory_reserved=lambda _i=0: 0,
        max_memory_allocated=lambda _i=0: 0,
        memory_allocated=lambda _i=0: 0,
        mem_get_info=_cuda_illegal,
        get_device_name=lambda _i=0: "Test GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("ASR_STAGE_WORKER_VRAM_BUDGET_MB", "4096")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        asr_transcribe._enforce_vram_budget("asr_text_transcribe", None)

    with pytest.raises(RuntimeError, match="illegal memory access"):
        asr_pipeline._cuda_memory_snapshot("after_chunking")

    # Non-fatal diagnostic probe errors must still degrade cleanly (isolate host PDH counters).
    monkeypatch.setattr(
        asr_pipeline.memory_safety,
        "runtime_memory_snapshot",
        lambda **_kw: {},
    )
    fake_torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        current_device=lambda: 0,
        mem_get_info=lambda _i=0: (_ for _ in ()).throw(RuntimeError("nvml busy")),
    )
    snap = asr_pipeline._cuda_memory_snapshot("after_chunking")
    assert "nvml busy" in str(snap.get("error", ""))


def test_completed_task_raising_timeout_error_does_not_poison_zombie_state(monkeypatch, tmp_path):
    from asr import local_backend

    # 1. If a zombie future finishes by raising TimeoutError itself, _join_zombie_worker
    #    must recognize it is done, reclaim it, and return True instead of skipping forever.
    backend = local_backend.LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: (_ for _ in ()).throw(TimeoutError("inner io timeout")))
    concurrent.futures.wait((future,), timeout=2.0)
    assert future.done()
    backend._zombie_future = future
    backend._zombie_executor = executor
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 1.0)

    assert backend._join_zombie_worker() is True
    assert backend._zombie_future is None
    assert backend._zombie_executor is None
    assert backend.model is None

    # 2. In transcribe_texts, a task-raised TimeoutError propagates cleanly and does not
    #    block the next batch from calling _transcribe_native again.
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF")
    calls = 0

    def _native(paths, lang):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("task internal timeout")
        return [local_backend.NativeAsrTranscription(language="Japanese", text="はい")]

    backend2 = local_backend.LocalAsrBackend("cpu")
    backend2.model = object()
    backend2.processor = object()
    monkeypatch.setattr(backend2, "_transcribe_native", _native)
    monkeypatch.setattr(local_backend, "_get_wav_duration", lambda _p: 1.0)

    with pytest.raises(TimeoutError, match="task internal timeout"):
        backend2.transcribe_texts([str(wav)])
    assert backend2._zombie_future is None

    second = backend2.transcribe_texts([str(wav)])
    assert calls == 2
    assert second[0]["text"] == "はい"


def test_cancelled_before_start_timeout_does_not_register_zombie_for_next_batch(
    monkeypatch, tmp_path
):
    """When worker thread startup is delayed so `future.cancel()` succeeds on
    timeout, batch 1 must NOT leave a cancelled Future in `_zombie_future` that
    would stall or skip batch 2."""
    from asr import local_backend

    wav = tmp_path / "delayed.wav"
    wav.write_bytes(b"RIFF")

    release_first_worker = threading.Event()
    real_executor_cls = concurrent.futures.ThreadPoolExecutor
    created_executors: list[concurrent.futures.ThreadPoolExecutor] = []

    class _DelayedFirstThreadExecutor(real_executor_cls):
        def __init__(self, *args, **kwargs):
            delay_this = len(created_executors) == 0
            super().__init__(*args, **kwargs)
            created_executors.append(self)
            self._delay_this = delay_this

        def _adjust_thread_count(self):
            if not self._delay_this:
                return super()._adjust_thread_count()
            # Delay starting the single worker thread so submit() queues the item
            # and wait() times out before the worker runs, allowing cancel() -> True.
            orig = super()._adjust_thread_count

            def _delayed_start():
                release_first_worker.wait(timeout=5.0)
                orig()

            threading.Thread(target=_delayed_start, daemon=True).start()

    native_calls: list[list[str]] = []

    def _native(paths, lang):
        native_calls.append(list(paths))
        return [local_backend.NativeAsrTranscription(language="Japanese", text="大丈夫")]

    backend = local_backend.LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    monkeypatch.setattr(backend, "_transcribe_native", _native)
    monkeypatch.setattr(local_backend, "_get_wav_duration", lambda _p: 1.0)
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 0.05)
    monkeypatch.setattr(
        local_backend.concurrent.futures,
        "ThreadPoolExecutor",
        _DelayedFirstThreadExecutor,
    )

    try:
        first = backend.transcribe_texts([str(wav)])
        assert first[0]["text"] == ""
        assert first[0]["asr_generation"]["error_kind"] == "timeout"
        assert backend._zombie_future is None
        assert backend._zombie_executor is None
    finally:
        release_first_worker.set()

    # Wait briefly for the first delayed executor to finish exiting cleanly.
    if created_executors:
        created_executors[0].shutdown(wait=True)

    # Batch 2 must execute `_transcribe_native` normally rather than skipping as zombie.
    second = backend.transcribe_texts([str(wav)])
    assert len(native_calls) == 1
    assert second[0]["text"] == "大丈夫"


def test_backend_close_during_oom_unwinding_preserves_primary_exception(monkeypatch):
    from asr import local_backend
    from pipeline import gpu_worker

    backend = local_backend.LocalAsrBackend("cpu")
    backend.model = object()
    backend.processor = object()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        lambda: (_ for _ in ()).throw(
            RuntimeError("CUDA error: an illegal memory access was encountered")
        )
    )
    concurrent.futures.wait((future,), timeout=2.0)
    backend._zombie_future = future
    backend._zombie_executor = executor
    monkeypatch.setattr(local_backend, "_transcription_timeout_s", lambda: 1.0)

    caught: BaseException | None = None
    try:
        try:
            raise RuntimeError("GPU VRAM budget exceeded: stage=asr peak_mb=9000 budget_mb=7000")
        finally:
            backend.close()
    except RuntimeError as exc:
        caught = exc

    assert caught is not None
    assert "GPU VRAM budget exceeded" in str(caught)
    assert gpu_worker._is_oom_error(caught, None) is True
    assert backend._zombie_future is None
    assert backend._zombie_executor is None
    assert backend.model is None


def test_gpu_worker_startup_probes_propagate_fatal_cuda_errors():
    from pipeline import gpu_worker

    def _cuda_illegal(*_a, **_kw):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    # 1. get_device_properties() fatal CUDA fault must not be misreported as 0 MiB VRAM.
    torch_bad_props = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=_cuda_illegal,
        )
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        gpu_worker._adaptive_runtime_tuning(torch_bad_props, {}, enforce_minimum=True)

    # 2. set_per_process_memory_fraction() fatal CUDA fault must not silently return None.
    torch_bad_cap = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            device_count=lambda: 1,
            get_device_name=lambda _i=0: "Test GPU",
            get_device_properties=lambda _i=0: SimpleNamespace(
                total_memory=16 * 1024 * 1024 * 1024, major=8, minor=9
            ),
            set_per_process_memory_fraction=_cuda_illegal,
        )
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        gpu_worker._adaptive_runtime_tuning(torch_bad_cap, {}, enforce_minimum=True)


# ---------------------------------------------------------------------------
# core.events TCP sink reconnect
# ---------------------------------------------------------------------------


def test_tcp_sink_reconnects_after_broken_socket(monkeypatch):
    from core import events

    created: list[object] = []

    class _FlakySocket:
        def __init__(self, fail_first_send: bool):
            self._fail_first = fail_first_send
            self.sent: list[bytes] = []
            self.closed = False

        def sendall(self, data: bytes):
            if self._fail_first:
                self._fail_first = False
                raise BrokenPipeError("pipe broken")
            self.sent.append(data)

        def close(self):
            self.closed = True

    def _fake_create_connection(addr, timeout=1.0):
        sock = _FlakySocket(fail_first_send=(len(created) == 0))
        created.append(sock)
        return sock

    monkeypatch.setattr(events.socket, "create_connection", _fake_create_connection)
    sink = events._TcpSink("127.0.0.1", 19999)
    sink.write(
        {
            "ts": "2026-01-01T00:00:00Z",
            "job_id": "j1",
            "run_id": "r1",
            "video": "sample-a.mp4",
            "stage": "asr",
            "phase": "progress",
            "extra": {"msg": "after reconnect"},
        }
    )

    assert len(created) == 2
    assert created[0].closed is True
    assert any(b"after reconnect" in chunk for chunk in created[1].sent)
    sink.close()


# ---------------------------------------------------------------------------
# web routes, proxy credential redaction, non-UTF-8 .env startup
# ---------------------------------------------------------------------------


def test_files_route_distinguishes_picker_error_from_user_cancel(monkeypatch):
    from fastapi import HTTPException
    from web.routes import files as files_route

    monkeypatch.setattr(files_route, "_pick_video_files_blocking", lambda: [])
    res = asyncio.run(files_route.pick_files())
    assert res == {"paths": []}

    def _picker_crash():
        raise files_route._PickerError("PowerShell failed (exit=1): STA error")

    monkeypatch.setattr(files_route, "_pick_video_files_blocking", _picker_crash)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(files_route.pick_files())
    assert excinfo.value.status_code == 500
    assert "STA error" in str(excinfo.value.detail)


def test_proxy_test_failure_redacts_credentials_from_logs_and_response(monkeypatch, caplog):
    import httpx
    from web.routes import config as config_routes

    monkeypatch.setenv("PROXY_PROTOCOL", "http")
    monkeypatch.setenv("PROXY_HOST", "secretuser:secretpass@127.0.0.1")
    monkeypatch.setenv("PROXY_PORT", "19998")

    class _RefusedClient:
        def __init__(self, **kwargs):
            self._proxy = kwargs.get("proxy", "")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            raise httpx.ConnectError(f"refused via {self._proxy}")

    monkeypatch.setattr(config_routes.httpx, "AsyncClient", _RefusedClient)
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(config_routes.test_proxy_connection())

    assert result["ok"] is False
    assert "secretuser" not in caplog.text
    assert "secretpass" not in caplog.text
    assert "secretuser" not in result["proxy_url"]
    assert "secretpass" not in result["proxy_url"]
    assert "secretuser" not in result["error"]
    assert "secretpass" not in result["error"]
    assert "http://***@127.0.0.1:19998" in result["proxy_url"]


def test_non_utf8_and_corrupt_config_does_not_block_startup_and_surfaces_warning(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from core import config as core_config
    from web.routes import config as config_route

    bad_config = tmp_path / "config" / "config.json"
    bad_config.parent.mkdir(parents=True, exist_ok=True)
    bad_config.write_bytes(b"\xff\xfeINVALID_UTF8")

    # Startup config load must succeed and record an actionable warning.
    monkeypatch.setattr(core_config, "CONFIG_PATH", bad_config)
    core_config.load_config(override_existing_env=False)
    assert "UTF-8" in core_config.config_load_error()

    # Settings read helper must not crash on non-UTF-8 config either.
    monkeypatch.setattr(config_route, "PROJECT_ROOT", tmp_path)
    assert config_route._read_config_entry("API_KEY") == (False, "")

    # Saving over non-UTF-8 config returns explicit HTTP 400.
    with pytest.raises(HTTPException) as excinfo:
        config_route._update_config_file({"API_KEY": "test"})
    assert excinfo.value.status_code == 400
    assert "UTF-8" in str(excinfo.value.detail)

    # Malformed JSON also records a warning on load and refuses to silently clobber on save.
    bad_config.write_text("{broken json", encoding="utf-8")
    core_config.load_config(override_existing_env=False)
    assert "JSON" in core_config.config_load_error()
    with pytest.raises(HTTPException) as json_exc:
        config_route._update_config_file({"API_KEY": "test"})
    assert json_exc.value.status_code == 400
    assert "JSON" in str(json_exc.value.detail)

    # Clear global config load error after test.
    monkeypatch.setattr(core_config, "CONFIG_PATH", tmp_path / "config" / "missing.json")
    core_config.load_config(override_existing_env=False)
    assert core_config.config_load_error() == ""


def test_config_route_non_sticky_cuda_probe(monkeypatch):
    from web.routes import config as config_route

    config_route._clear_cuda_environment_cache()
    calls = {"count": 0}

    def _probe(cmd, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("transient probe failure")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "ok", "ok": True, "device_name": "RTX Test"}),
            stderr="",
        )

    monkeypatch.setattr(config_route.subprocess, "run", _probe)
    first = config_route._cuda_environment_status()
    assert first["ok"] is False
    assert first["code"] == "probe_failed"
    second = config_route._cuda_environment_status()
    assert second["ok"] is True
    assert calls["count"] == 2
    config_route._clear_cuda_environment_cache()


# ---------------------------------------------------------------------------
# audio & subtitle edge guards
# ---------------------------------------------------------------------------


def test_audio_loading_rejects_empty_invalid_rate_and_handles_1_frame_stereo(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf
    from audio import loading

    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(sf, "read", lambda *a, **kw: (np.array([], dtype=np.float32), 16000))
    with pytest.raises(ValueError, match="no samples"):
        loading.load_audio_16k_mono(str(wav))

    monkeypatch.setattr(sf, "read", lambda *a, **kw: (np.zeros(16, dtype=np.float32), 0))
    with pytest.raises(ValueError, match="sample rate"):
        loading.load_audio_16k_mono(str(wav))

    # 1-frame stereo WAV has shape (1, 2); must average channels (axis=1) to 1 mono sample.
    monkeypatch.setattr(
        sf, "read", lambda *a, **kw: (np.array([[0.25, 0.75]], dtype=np.float32), 16000)
    )
    mono, rate = loading.load_audio_16k_mono(str(wav))
    assert rate == 16000
    assert mono.shape == (1,)
    assert float(mono[0]) == pytest.approx(0.5)


def test_subtitle_writer_and_vocalisation_edge_guards(tmp_path):
    from subtitles.writer import format_timestamp, write_bilingual_srt
    from subtitles.vocalisation import CueSplit, _apply_split

    assert format_timestamp(-1.25) == "00:00:00,000"
    assert format_timestamp(59.9999) == "00:00:59,999"

    nested_srt = tmp_path / "nested" / "dir" / "out.srt"
    write_bilingual_srt(
        [{"start": 0.0, "end": 1.0, "ja_text": "こんにちは", "zh_text": "你好"}],
        str(nested_srt),
    )
    assert nested_srt.is_file()

    # Vocalisation split updates both ja_text & text, clamps display_start >= 0.0,
    # and tolerates explicit display_start=None with display_end set.
    block = {
        "start": 0.30,
        "end": 1.0,
        "acoustic_start": 0.30,
        "acoustic_end": 1.0,
        "display_start": 0.05,  # lead_in = 0.25s
        "display_end": 1.10,
        "ja_text": "あっ、こんにちは",
        "text": "あっ、こんにちは",
        "words": [
            {"word": "あっ、", "start": 0.30, "end": 0.45},
            {"word": "こんにちは", "start": 0.50, "end": 1.0},
        ],
    }
    split = CueSplit(
        text="こんにちは",
        removed_prefix="あっ、",
        removed_suffix="",
        removed_seconds=0.20,
        start_s=0.10,
        end_s=1.0,
        prefix_span=(0.30, 0.45),
        suffix_span=None,
    )
    updated = _apply_split(block, split)
    assert updated["ja_text"] == "こんにちは"
    assert updated["text"] == "こんにちは"
    assert updated["display_start"] == 0.0

    null_display_start_block = {**block, "display_start": None, "display_end": 1.10}
    updated_null = _apply_split(null_display_start_block, split)
    assert updated_null["display_duration"] == pytest.approx(1.0)
