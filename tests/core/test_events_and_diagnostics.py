from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from pathlib import Path

from core import diagnostics, events
from utils import ffmpeg_runtime, runtime_paths


def test_tcp_sink_connects_lazily_and_configure_sink_does_not_block_on_stalled_send(monkeypatch):
    created: list[object] = []
    send_started = threading.Event()
    send_unblocked = threading.Event()

    class _BlockingSocket:
        def __init__(self) -> None:
            self.closed = False

        def sendall(self, _data: bytes) -> None:
            send_started.set()
            send_unblocked.wait(timeout=5.0)
            raise OSError("socket closed during shutdown")

        def close(self) -> None:
            self.closed = True
            send_unblocked.set()

    def _fake_connect(_addr, timeout=1.0):
        sock = _BlockingSocket()
        created.append(sock)
        return sock

    monkeypatch.setattr(events.socket, "create_connection", _fake_connect)
    try:
        events.configure_sink("tcp:127.0.0.1:19991")
        # Lazy connect: no socket created during configure_sink / __init__.
        assert created == []

        emitter = threading.Thread(
            target=events.emit,
            args=(
                {
                    "ts": "2026-01-01T00:00:00Z",
                    "job_id": "j1",
                    "run_id": "r1",
                    "video": "sample.mp4",
                    "stage": "model_download",
                    "phase": "error",
                    "extra": {"msg": "boom"},
                },
            ),
            daemon=True,
        )
        emitter.start()
        assert send_started.wait(timeout=3.0)
        assert len(created) == 1

        # Shutting down sink must not deadlock/block on the in-flight sendall.
        started = time.monotonic()
        events.configure_sink("")
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert created[0].closed is True

        emitter.join(timeout=3.0)
        assert not emitter.is_alive()
    finally:
        events.configure_sink("")


def test_set_current_job_id_clears_stale_run_id():
    events.set_current_run("job-1", "run-1")
    assert events._current_job_id() == "job-1"
    assert events._current_run_id() == "run-1"

    events.set_current_job_id("job-2")
    assert events._current_job_id() == "job-2"
    assert events._current_run_id() == ""

    events.set_current_job_id("")


def test_redact_url_credentials_handles_at_in_password_and_preserves_other_at_tokens():
    embedded = diagnostics.redact_url_credentials(
        "ConnectError: proxy socks5://alice:p@ss@word@127.0.0.1:1080 failed for contact@example.com"
    )
    assert "alice" not in embedded
    assert "p@ss" not in embedded
    assert "word" not in embedded
    assert "socks5://***@127.0.0.1:1080" in embedded
    assert "contact@example.com" in embedded

    # Path/query/fragment '@' without authority credentials must stay untouched.
    no_creds = "https://example.com/users/foo@bar?email=a@b.com#c@d"
    assert diagnostics.redact_url_credentials(no_creds) == no_creds


def test_diagnostics_atexit_disables_faulthandler_before_closing_streams(monkeypatch, tmp_path):
    order: list[str] = []

    class _FakeStream:
        def flush(self) -> None:
            order.append("flush")

        def close(self) -> None:
            order.append("close")

    monkeypatch.setattr(faulthandler, "is_enabled", lambda: True)
    monkeypatch.setattr(faulthandler, "disable", lambda: order.append("disable"))
    fault_path = tmp_path / "fake-fault.log"
    diagnostics._FAULT_STREAMS[fault_path] = _FakeStream()  # type: ignore[assignment]

    diagnostics._close_fault_streams()

    assert order == ["disable", "flush", "close"]
    assert fault_path not in diagnostics._FAULT_STREAMS
    assert not hasattr(diagnostics, "format_traceback_tail")


def test_configure_process_logging_closes_detached_old_handler(tmp_path):
    log_path = diagnostics.configure_process_logging(
        "reattach_test.log",
        log_dir=tmp_path,
        enable_faulthandler=False,
        install_hooks=False,
    )
    assert log_path is not None
    old_handler = diagnostics._INSTALLED_HANDLERS[log_path]
    root = logging.getLogger()
    root.removeHandler(old_handler)

    closed_called: list[bool] = []
    real_close = old_handler.close

    def tracked_close():
        closed_called.append(True)
        real_close()

    old_handler.close = tracked_close  # type: ignore[method-assign]
    try:
        second_path = diagnostics.configure_process_logging(
            "reattach_test.log",
            log_dir=tmp_path,
            enable_faulthandler=False,
            install_hooks=False,
        )
        assert second_path == log_path
        assert closed_called == [True]
    finally:
        new_handler = diagnostics._INSTALLED_HANDLERS.pop(log_path, None)
        if new_handler is not None:
            root.removeHandler(new_handler)
            new_handler.close()


def test_prune_old_process_logs_single_iterdir_and_tolerates_stat_error(monkeypatch, tmp_path):
    f1 = tmp_path / "app-1001.log"
    f1_bak = tmp_path / "app-1001.log.1"
    f2 = tmp_path / "app-1002.log"
    f3 = tmp_path / "app-1003.log"
    for p in (f1, f1_bak, f2, f3):
        p.write_text("x", encoding="utf-8")
    os.utime(f1, (1000, 1000))
    os.utime(f2, (2000, 2000))
    os.utime(f3, (3000, 3000))

    iterdir_calls = {"count": 0}
    real_iterdir = Path.iterdir
    real_stat = Path.stat

    def counted_iterdir(self):
        if self == tmp_path:
            iterdir_calls["count"] += 1
        return real_iterdir(self)

    def flaky_stat(self, *args, **kwargs):
        if self == f2:
            raise FileNotFoundError("vanished concurrently")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", counted_iterdir)
    monkeypatch.setattr(Path, "stat", flaky_stat)

    diagnostics._prune_old_process_logs(tmp_path, stem="app", suffix=".log", keep=1, exclude=set())

    assert iterdir_calls["count"] == 1
    assert not f1.exists()
    assert not f1_bak.exists()
    assert f3.exists()


def test_prepend_to_path_deduplicates_normalized_and_moves_to_front(monkeypatch, tmp_path):
    dir_a = tmp_path / "BinA"
    dir_b = tmp_path / "BinB"
    dir_a.mkdir()
    dir_b.mkdir()
    # Put dir_a at the end with trailing slash and quotes.
    monkeypatch.setenv("PATH", f'{dir_b}{os.pathsep}"{str(dir_a).lower()}{os.sep}"')

    runtime_paths.prepend_to_path(dir_a)
    entries = os.environ["PATH"].split(os.pathsep)
    assert os.path.normcase(os.path.normpath(entries[0])) == os.path.normcase(str(dir_a.resolve()))
    normed = [os.path.normcase(os.path.normpath(e.strip('"'))) for e in entries]
    assert normed.count(os.path.normcase(str(dir_a.resolve()))) == 1


def test_configure_ffmpeg_shared_runtime_preserves_priority_and_skips_failed_dll_dir(
    monkeypatch, tmp_path
):
    bundled = tmp_path / "bundled_bin"
    winget = tmp_path / "winget_bin"
    broken = tmp_path / "broken_bin"
    for d in (bundled, winget, broken):
        d.mkdir()
        for dll in ("avcodec-61.dll", "avformat-61.dll", "avutil-59.dll"):
            (d / dll).write_bytes(b"dll")

    monkeypatch.setattr(ffmpeg_runtime, "_CONFIGURED_DIRECTORIES", None)
    monkeypatch.setattr(ffmpeg_runtime, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(
        ffmpeg_runtime, "_candidate_directories", lambda: [bundled, broken, winget]
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    registered: list[str] = []

    def fake_add_dll_directory(path_str: str):
        if Path(path_str).resolve() == broken.resolve():
            raise OSError("cannot add dll dir")
        registered.append(path_str)
        return object()

    if hasattr(os, "add_dll_directory"):
        monkeypatch.setattr(os, "add_dll_directory", fake_add_dll_directory)

    result = ffmpeg_runtime.configure_ffmpeg_shared_runtime()
    path_entries = [
        os.path.normcase(os.path.normpath(p)) for p in os.environ["PATH"].split(os.pathsep)
    ]

    assert result == (str(bundled.resolve()), str(winget.resolve()))
    assert path_entries[0] == os.path.normcase(str(bundled.resolve()))
    assert path_entries[1] == os.path.normcase(str(winget.resolve()))
    assert os.path.normcase(str(broken.resolve())) not in path_entries
