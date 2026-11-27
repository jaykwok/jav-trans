from __future__ import annotations

import json
import os
import threading
import time

import pytest

from core import atomic_io


def test_replace_atomically_retries_transient_permission_error(monkeypatch, tmp_path):
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.json"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")

    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(a, b):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("transient lock")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky_replace)
    atomic_io.replace_atomically(src, dst, retries=5, base_delay_s=0.001)

    assert attempts["count"] == 3
    assert dst.read_text(encoding="utf-8") == "new"
    assert not src.exists()


def test_replace_atomically_raises_after_exhausting_retries(monkeypatch, tmp_path):
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.json"
    src.write_text("new", encoding="utf-8")

    attempts = {"count": 0}

    def always_locked(_a, _b):
        attempts["count"] += 1
        raise PermissionError("permanently locked")

    monkeypatch.setattr(os, "replace", always_locked)
    with pytest.raises(PermissionError):
        atomic_io.replace_atomically(src, dst, retries=3, base_delay_s=0.001)
    assert attempts["count"] == 3


def test_write_text_and_json_atomic_replace_cleanly_and_cleanup_tmp_on_error(tmp_path):
    text_path = tmp_path / "nested" / "note.txt"
    atomic_io.write_text_atomic(text_path, "hello 世界\n")
    assert text_path.read_text(encoding="utf-8") == "hello 世界\n"
    assert list(text_path.parent.glob("*.tmp")) == []

    json_path = tmp_path / "nested" / "data.json"
    atomic_io.write_json_atomic(json_path, {"ok": True, "zh": "中文"})
    raw = json_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw) == {"ok": True, "zh": "中文"}
    assert list(json_path.parent.glob("*.tmp")) == []

    # Serialization failure must preserve the existing file and leave no .tmp residue.
    with pytest.raises(TypeError):
        atomic_io.write_json_atomic(json_path, {"bad": object()})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True, "zh": "中文"}
    assert list(json_path.parent.glob("*.tmp")) == []


def test_other_launcher_instances_running_handles_missing_stale_live_and_gate(tmp_path):
    # No tmp/launcher directory -> False without creating it.
    assert atomic_io.other_launcher_instances_running(tmp_path) is False
    assert not (tmp_path / "tmp" / "launcher").exists()

    launcher_dir = tmp_path / "tmp" / "launcher"
    launcher_dir.mkdir(parents=True)

    # Own slot is ignored, stale unlocked slot is pruned.
    own_slot = launcher_dir / f"instance-{os.getpid()}.lock"
    own_slot.write_text("own", encoding="utf-8")
    stale_slot = launcher_dir / "instance-999999.lock"
    stale_slot.write_text("stale", encoding="utf-8")

    assert atomic_io.other_launcher_instances_running(tmp_path) is False
    assert own_slot.exists()
    assert not stale_slot.exists()

    # Holding workspace_gate blocks concurrent slot scan/cleanup conservatively.
    another_stale = launcher_dir / "instance-888888.lock"
    another_stale.write_text("stale-during-gate", encoding="utf-8")
    with atomic_io.workspace_gate(tmp_path, wait_s=1.0) as acquired:
        assert acquired is True
        # While gate is held, a short-wait probe sees the workspace busy and
        # does NOT delete instance-*.lock files.
        assert atomic_io.other_launcher_instances_running(tmp_path, wait_s=0.05) is True
        assert another_stale.exists()

    # Once gate is released, stale slot is cleaned cleanly.
    assert atomic_io.other_launcher_instances_running(tmp_path, wait_s=1.0) is False
    assert not another_stale.exists()

    # An open slot file held by another process/handle reports True on Windows.
    if os.name == "nt":
        live_slot = launcher_dir / "instance-777777.lock"
        with live_slot.open("w", encoding="utf-8") as handle:
            handle.write('{"pid": 777777}')
            handle.flush()
            assert atomic_io.other_launcher_instances_running(tmp_path, wait_s=1.0) is True
            assert live_slot.exists()


def test_workspace_gate_serializes_concurrent_threads(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    second_acquired_at: list[float] = []

    def holder():
        with atomic_io.workspace_gate(tmp_path, wait_s=2.0) as ok:
            assert ok is True
            entered.set()
            release.wait(timeout=3.0)

    def waiter():
        assert entered.wait(timeout=3.0)
        with atomic_io.workspace_gate(tmp_path, wait_s=3.0, poll_s=0.01) as ok:
            assert ok is True
            second_acquired_at.append(time.monotonic())

    t1 = threading.Thread(target=holder, daemon=True)
    t2 = threading.Thread(target=waiter, daemon=True)
    t1.start()
    t2.start()
    assert entered.wait(timeout=3.0)
    time.sleep(0.08)
    assert second_acquired_at == []
    released_at = time.monotonic()
    release.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert len(second_acquired_at) == 1
    assert second_acquired_at[0] >= released_at - 0.01
