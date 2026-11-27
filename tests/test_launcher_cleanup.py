"""Closing one window must not delete the other window's work, and every exit
path must shut down the uvicorn server cleanly before temp cleanup runs.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import launcher


def _hold_slot(directory: Path, pid: int):
    """A slot held by another live process."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / f"instance-{pid}.lock", "w", encoding="utf-8")
    handle.write("{}")
    handle.flush()
    if os.name != "nt":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _workspace(root: Path) -> tuple[Path, Path]:
    chunk = root / "tmp" / "chunks" / "job-a" / "chunk_0000.wav"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"audio")
    job_audio = root / "tmp" / "jobs" / "job-a" / "audio" / "source.wav"
    job_audio.parent.mkdir(parents=True, exist_ok=True)
    job_audio.write_bytes(b"audio")
    return chunk, job_audio


def test_importing_launcher_has_no_heavy_side_effects() -> None:
    """P1-7: importing launcher must not claim slots, register atexit, or load runtime."""
    assert launcher._instance_slot is None
    assert callable(launcher.main)


def test_the_last_window_out_still_sweeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunk, job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)

    try:
        launcher._cleanup_temp()
    finally:
        if not slot.closed:
            slot.close()

    assert not chunk.exists()
    assert not job_audio.exists()
    # Its own slot goes with it, so the next window starts from a clean list.
    assert not list((tmp_path / "tmp" / "launcher").glob("*.lock"))


def test_failed_job_audio_is_preserved_across_exit_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_audio = tmp_path / "tmp" / "jobs" / "job-failed" / "audio" / "source.wav"
    failed_audio.parent.mkdir(parents=True, exist_ok=True)
    failed_audio.write_bytes(b"failed-wav")
    done_audio = tmp_path / "tmp" / "jobs" / "job-done" / "audio" / "source.wav"
    done_audio.parent.mkdir(parents=True, exist_ok=True)
    done_audio.write_bytes(b"done-wav")
    jobs_json = tmp_path / "cache" / "web" / "jobs.json"
    jobs_json.parent.mkdir(parents=True, exist_ok=True)
    jobs_json.write_text(
        '[{"id": "job-failed", "status": "failed"}, {"id": "job-done", "status": "done"}]',
        encoding="utf-8",
    )

    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)
    try:
        launcher._cleanup_temp()
    finally:
        if not slot.closed:
            slot.close()

    assert failed_audio.is_file()
    assert not done_audio.exists()


def test_a_second_window_vetoes_the_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunk, job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)
    other = _hold_slot(tmp_path / "tmp" / "launcher", os.getpid() + 1)

    try:
        launcher._cleanup_temp()

        # The other window is mid-job: its chunks are not this process's to
        # delete, and neither is the audio of a job it may still be running.
        assert chunk.exists()
        assert job_audio.exists()
    finally:
        other.close()
        if not slot.closed:
            slot.close()


def test_a_crashed_window_does_not_veto_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk, _job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)
    stale = tmp_path / "tmp" / "launcher" / f"instance-{os.getpid() + 2}.lock"
    stale.write_text("{}", encoding="utf-8")

    try:
        launcher._cleanup_temp()
    finally:
        if not slot.closed:
            slot.close()

    assert not chunk.exists()
    assert not stale.exists()


def test_without_a_slot_nothing_is_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunk, job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_instance_slot", None)

    launcher._cleanup_temp()

    assert chunk.exists()
    assert job_audio.exists()


def test_slot_is_held_does_not_swallow_system_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = tmp_path / "instance-999.lock"
    slot.write_text("{}", encoding="utf-8")
    if os.name == "nt":
        monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(SystemExit(9)))
        with pytest.raises(SystemExit) as exc:
            launcher._slot_is_held(slot)
        assert exc.value.code == 9


def test_a_window_registering_mid_sweep_is_held_at_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk, _job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_GATE_CLAIM_WAIT_S", 0.2)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)
    original_scan = launcher._other_instances_running
    newcomer: list = []

    other_pid = os.getpid() + 100000

    def scan_then_start(root: Path, *, own_slot):
        result = original_scan(root, own_slot=own_slot)
        assert result is False
        with monkeypatch.context() as sub_mp:
            sub_mp.setattr(launcher.os, "getpid", lambda: other_pid)
            newcomer.append(launcher._claim_instance_slot(root))
        return result

    monkeypatch.setattr(launcher, "_other_instances_running", scan_then_start)

    try:
        launcher._cleanup_temp()
    finally:
        for handle in newcomer:
            if handle is not None and not handle.closed:
                handle.close()
        if not slot.closed:
            slot.close()

    assert newcomer == [None]
    assert not chunk.exists()
    fresh = launcher._claim_instance_slot(tmp_path)
    assert fresh is not None
    fresh.close()


def test_a_sweep_is_skipped_while_another_window_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk, job_audio = _workspace(tmp_path)
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_GATE_SWEEP_WAIT_S", 0.2)
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_instance_slot", slot)
    held = launcher._acquire_gate(tmp_path, 1.0)
    assert held is not None

    try:
        launcher._cleanup_temp()
        assert chunk.exists()
        assert job_audio.exists()
    finally:
        launcher._release_gate(held)
        if not slot.closed:
            slot.close()


def test_the_gate_outlives_the_window_that_took_it(tmp_path: Path) -> None:
    first = launcher._acquire_gate(tmp_path, 0.1)
    assert first is not None
    assert launcher._acquire_gate(tmp_path, 0.05) is None

    launcher._release_gate(first)

    second = launcher._acquire_gate(tmp_path, 0.1)
    assert second is not None
    launcher._release_gate(second)


def test_a_window_that_cannot_register_does_not_start_and_pops_dialog_when_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_FROZEN", True)
    monkeypatch.setattr(launcher, "_configure_runtime_environment", lambda: ())
    monkeypatch.setattr(launcher, "_claim_instance_slot", lambda _root: None)
    monkeypatch.setattr(
        launcher,
        "_bind_free_socket",
        lambda *_a, **_k: pytest.fail("must not bind port without an instance slot"),
    )
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        launcher,
        "_show_error_dialog",
        lambda title, msg: dialogs.append((title, msg)) or True,
    )

    assert launcher.main([]) == 1
    assert len(dialogs) == 1
    assert "工作目录" in dialogs[0][1]
    assert "logs/" in dialogs[0][1]


@pytest.mark.parametrize("exit_mode", ["normal", "update", "keyboard_interrupt", "ui_error"])
def test_graceful_server_shutdown_runs_in_finally_on_every_exit_path(
    exit_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-17(3): normal window close, update restart, Ctrl+C, and UI exceptions
    must all trigger `_shutdown.set()`, `server.should_exit = True`, and
    `server_thread.join()` before returning."""
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_configure_runtime_environment", lambda: ())
    slot = launcher._claim_instance_slot(tmp_path)
    assert slot is not None
    monkeypatch.setattr(launcher, "_claim_instance_slot", lambda _root: slot)
    monkeypatch.setattr(launcher, "_bind_free_socket", lambda port, **_k: (None, port))
    monkeypatch.setattr(launcher, "_resolve_events_port", lambda _p, e: e)
    monkeypatch.setattr(launcher, "_report_fatal_startup_error", lambda *_a, **_k: None)

    lifespan_flushed = threading.Event()
    server_started = threading.Event()

    class _FakeServer:
        def __init__(self) -> None:
            self.started = False
            self.should_exit = False

    fake_server = _FakeServer()

    def fake_run_server(_sock=None) -> None:
        launcher._uvicorn_server = fake_server
        fake_server.started = True
        server_started.set()
        while not fake_server.should_exit:
            launcher.time.sleep(0.01)
        # Simulate FastAPI _lifespan shutdown & _job_store_writer.flush().
        lifespan_flushed.set()

    monkeypatch.setattr(launcher, "_run_server", fake_run_server)

    fake_webview = types.ModuleType("webview")
    fake_dom = types.ModuleType("webview.dom")
    fake_dom.DOMEventHandler = lambda cb, *a, **k: cb

    class _Events:
        def __init__(self) -> None:
            self.dragover = 0
            self.drop = 0

    class _FakeWindow:
        def __init__(self) -> None:
            self.dom = SimpleNamespace(document=SimpleNamespace(events=_Events()))

    fake_webview.create_window = lambda *a, **k: _FakeWindow()

    def fake_start(_bind, _win, **_kw) -> None:
        assert server_started.is_set()
        if exit_mode == "normal":
            return
        if exit_mode == "update":
            launcher._update_restart_requested.set()
            return
        if exit_mode == "keyboard_interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("webview crashed")

    fake_webview.start = fake_start
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setitem(sys.modules, "webview.dom", fake_dom)

    try:
        code = launcher.main([])
    finally:
        launcher._cleanup_temp()

    assert launcher._shutdown.is_set()
    assert fake_server.should_exit is True
    assert lifespan_flushed.is_set()
    expected = {"normal": 0, "update": 42, "keyboard_interrupt": 130, "ui_error": 1}[exit_mode]
    assert code == expected


def test_smoke_imports_rejects_false_positives_and_passes_when_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-1: --smoke-imports must verify ffmpeg_major_version, core_library_path,
    and get_ffmpeg_library_versions()."""
    out_file = tmp_path / "smoke.json"
    monkeypatch.setenv("JAV_SUB_SMOKE_OUTPUT", str(out_file))

    dll = tmp_path / "libtorchcodec_core9.dll"
    dll.write_bytes(b"dll")

    def install_fake_torchcodec(*, major, core_path, versions_fn):
        pkg = types.ModuleType("torchcodec")
        pkg.__version__ = "0.17.0"
        core = types.ModuleType("torchcodec._core")
        core.ffmpeg_major_version = major
        core.core_library_path = core_path
        core.get_ffmpeg_library_versions = versions_fn
        pkg._core = core
        monkeypatch.setitem(sys.modules, "torchcodec", pkg)
        monkeypatch.setitem(sys.modules, "torchcodec._core", core)

    # Case 1: ffmpeg_major_version is None -> must fail
    install_fake_torchcodec(major=None, core_path=dll, versions_fn=lambda: {"libavcodec": [63, 0, 0]})
    assert launcher._run_smoke_imports() == 1
    assert json.loads(out_file.read_text(encoding="utf-8"))["ok"] is False

    # Case 2: core_library_path missing on disk -> must fail
    install_fake_torchcodec(
        major=9, core_path=tmp_path / "missing.dll", versions_fn=lambda: {"libavcodec": [63, 0, 0]}
    )
    assert launcher._run_smoke_imports() == 1
    assert json.loads(out_file.read_text(encoding="utf-8"))["ok"] is False

    # Case 3: get_ffmpeg_library_versions raises -> must fail
    install_fake_torchcodec(
        major=9,
        core_path=dll,
        versions_fn=lambda: (_ for _ in ()).throw(RuntimeError("DLL load failed")),
    )
    assert launcher._run_smoke_imports() == 1
    assert "DLL load failed" in json.loads(out_file.read_text(encoding="utf-8"))["error"]

    # Case 4: valid versions dict -> passes
    install_fake_torchcodec(major=9, core_path=dll, versions_fn=lambda: {"libavcodec": [63, 24, 100]})
    assert launcher._run_smoke_imports() == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["torchcodec_ffmpeg_major_version"] == 9
    assert payload["torchcodec_ffmpeg_library_versions"] == {"libavcodec": [63, 24, 100]}
