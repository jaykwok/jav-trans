"""Closing one window must not delete the other window's work.

The exit sweep removes `tmp/chunks/` outright and every `tmp/web/jobs/<id>/
audio/`. That is correct for the last window out and wrong for any other: ports
step forward when they collide, so a second launcher starts happily against the
*same* workspace, and the first one closed used to delete the chunk audio the
other was still transcribing.

Ownership is now a fact the launcher establishes - a slot file held open for the
life of the process - rather than something the sweep assumes.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _launcher_cleanup_namespace(root: Path, slot=None) -> dict:
    """The cleanup block, executed without running the launcher's module body.

    Importing launcher.py resolves ports, configures ffmpeg and pulls in webview
    at import time. The functions under test only need the standard library, so
    they are read out of the source - the same approach as test_launcher_port.
    """
    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")
    start = source.index("# Globs inside tmp/")
    end = source.index("def _version_tuple(", start)
    namespace: dict = {}
    preamble = "import json, os, shutil, time\nfrom pathlib import Path\n"
    exec(compile(preamble + source[start:end], "launcher.py", "exec"), namespace)  # noqa: S102
    namespace["_ROOT"] = root
    namespace["_instance_slot"] = slot
    return namespace


def _launcher_main_guard(slot) -> dict:
    """The `__main__` block up to the point where work would begin.

    Sliced rather than imported for the same reason as above, and cut before the
    server thread starts so running it cannot open a port or a window.
    """
    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")
    start = source.index('if __name__ == "__main__":')
    end = source.index("t = threading.Thread(target=_run_server", start)
    namespace: dict = {"__name__": "__main__", "_instance_slot": slot, "PORT": 2233}
    block = source[start:end] + "_reached_startup = True\n"
    exec(compile(block, "launcher.py", "exec"), namespace)  # noqa: S102
    return namespace


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
    job_audio = root / "tmp" / "web" / "jobs" / "job-a" / "audio" / "source.wav"
    job_audio.parent.mkdir(parents=True, exist_ok=True)
    job_audio.write_bytes(b"audio")
    return chunk, job_audio


def test_the_last_window_out_still_sweeps(tmp_path) -> None:
    chunk, job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)
    slot = namespace["_claim_instance_slot"](tmp_path)
    namespace["_instance_slot"] = slot

    try:
        namespace["_cleanup_temp"]()
    finally:
        if not slot.closed:
            slot.close()

    assert not chunk.exists()
    assert not job_audio.exists()
    # Its own slot goes with it, so the next window starts from a clean list.
    assert not list((tmp_path / "tmp" / "launcher").glob("*.lock"))


def test_a_second_window_vetoes_the_sweep(tmp_path) -> None:
    chunk, job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)
    slot = namespace["_claim_instance_slot"](tmp_path)
    namespace["_instance_slot"] = slot
    other = _hold_slot(tmp_path / "tmp" / "launcher", os.getpid() + 1)

    try:
        namespace["_cleanup_temp"]()

        # The other window is mid-job: its chunks are not this process's to
        # delete, and neither is the audio of a job it may still be running.
        assert chunk.exists()
        assert job_audio.exists()
    finally:
        other.close()
        if not slot.closed:
            slot.close()


def test_a_crashed_window_does_not_veto_forever(tmp_path) -> None:
    # A slot file with nobody behind it is debris, not an owner - otherwise one
    # crash would disable cleanup permanently.
    chunk, _job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)
    slot = namespace["_claim_instance_slot"](tmp_path)
    namespace["_instance_slot"] = slot
    stale = tmp_path / "tmp" / "launcher" / f"instance-{os.getpid() + 2}.lock"
    stale.write_text("{}", encoding="utf-8")

    try:
        namespace["_cleanup_temp"]()
    finally:
        if not slot.closed:
            slot.close()

    assert not chunk.exists()
    assert not stale.exists()


def test_without_a_slot_nothing_is_deleted(tmp_path) -> None:
    # Never established who owns this workspace, so there is no basis for
    # deleting anything in it.
    chunk, job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)

    namespace["_cleanup_temp"]()

    assert chunk.exists()
    assert job_audio.exists()


def test_a_window_registering_mid_sweep_is_held_at_the_gate(tmp_path) -> None:
    """The scan answers for the instant it ran, and the sweep runs after it.

    A second window that publishes its slot in that gap passes unseen, and the
    sweep then deletes the audio it has already started writing. Both sides take
    the workspace gate, so the newcomer waits until the sweep is over.
    """
    chunk, _job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)
    namespace["_GATE_CLAIM_WAIT_S"] = 0.2
    slot = namespace["_claim_instance_slot"](tmp_path)
    namespace["_instance_slot"] = slot
    original_scan = namespace["_other_instances_running"]
    newcomer: list = []

    def scan_then_start(root, *, own_slot):
        result = original_scan(root, own_slot=own_slot)
        assert result is False
        # The production claim path, at the worst possible moment: after the
        # scan has decided, before the sweep has started. Only the pid differs.
        real_os = namespace["os"]
        namespace["os"] = SimpleNamespace(
            name=os.name, getpid=lambda: os.getpid() + 100000
        )
        try:
            newcomer.append(namespace["_claim_instance_slot"](root))
        finally:
            namespace["os"] = real_os
        return result

    namespace["_other_instances_running"] = scan_then_start

    try:
        namespace["_cleanup_temp"]()
    finally:
        for handle in newcomer:
            if handle is not None and not handle.closed:
                handle.close()
        if not slot.closed:
            slot.close()

    assert newcomer == [None]
    # Held out, not locked out: the sweep it waited for still happened.
    assert not chunk.exists()
    assert namespace["_claim_instance_slot"](tmp_path) is not None


def test_a_sweep_is_skipped_while_another_window_registers(tmp_path) -> None:
    chunk, job_audio = _workspace(tmp_path)
    namespace = _launcher_cleanup_namespace(tmp_path)
    namespace["_GATE_SWEEP_WAIT_S"] = 0.2
    slot = namespace["_claim_instance_slot"](tmp_path)
    namespace["_instance_slot"] = slot
    # Another window is inside its claim: it does not exist for the scan yet, so
    # the only honest answer is to leave the temp files for the next window.
    held = namespace["_acquire_gate"](tmp_path, 1.0)
    assert held is not None

    try:
        namespace["_cleanup_temp"]()

        assert chunk.exists()
        assert job_audio.exists()
    finally:
        namespace["_release_gate"](held)
        if not slot.closed:
            slot.close()


def test_the_gate_outlives_the_window_that_took_it(tmp_path) -> None:
    # Releasing has to actually release, or the second window of the day would
    # wait out the full timeout and then refuse to register.
    namespace = _launcher_cleanup_namespace(tmp_path)
    first = namespace["_acquire_gate"](tmp_path, 0.1)
    assert first is not None
    assert namespace["_acquire_gate"](tmp_path, 0.05) is None

    namespace["_release_gate"](first)

    second = namespace["_acquire_gate"](tmp_path, 0.1)
    assert second is not None
    namespace["_release_gate"](second)


def test_a_window_that_cannot_register_does_not_start(tmp_path) -> None:
    """Not registering is not a soft landing.

    An unregistered window still writes chunk audio into the shared tmp/, where
    the window sweeping right now cannot see it - so "start anyway and skip the
    cleanup on exit" leaves exactly the deletion this slot exists to prevent.
    """
    with pytest.raises(SystemExit) as exit_info:
        _launcher_main_guard(None)

    assert exit_info.value.code == 1


def test_a_registered_window_starts_normally() -> None:
    namespace = _launcher_main_guard(object())

    assert namespace["_reached_startup"] is True


def test_the_slot_is_claimed_before_the_sweep_is_registered() -> None:
    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")

    claim = source.index("_instance_slot = _claim_instance_slot(_ROOT)")
    register = source.index("atexit.register(_cleanup_temp)")
    assert claim < register
