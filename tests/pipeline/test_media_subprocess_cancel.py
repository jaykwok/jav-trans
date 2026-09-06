"""ffmpeg/ffprobe must answer both the clock and the cancel button.

`subprocess.run(timeout=...)` answered only the first, and ffprobe had no
timeout at all: a cancelled job kept a two-hour extraction running to
completion (its computed timeout is ~8 hours) while the task bar already said
取消, and the worker thread stayed pinned behind it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from core.cancellation import PipelineCancelledError
from helpers import write_pcm_wav
from pipeline import audio as pipeline_audio
from utils import subprocess_tools
from utils.subprocess_tools import SubprocessCancelledError, run_cancellable


# Writes to a marker file until killed, so a test can tell whether the child
# actually died or merely stopped being waited on.
_BUSY_CHILD = (
    "import sys, time\n"
    "path = sys.argv[1]\n"
    "for _ in range(400):\n"
    "    with open(path, 'a') as handle:\n"
    "        handle.write('x')\n"
    "    time.sleep(0.05)\n"
)


def _stopped_writing(marker) -> bool:
    before = marker.stat().st_size
    time.sleep(0.5)
    return marker.stat().st_size == before


def test_run_cancellable_kills_the_child_on_cancel(tmp_path):
    marker = tmp_path / "alive.txt"
    marker.write_text("", encoding="utf-8")
    cancel = threading.Event()
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(SubprocessCancelledError):
            run_cancellable(
                [sys.executable, "-c", _BUSY_CHILD, str(marker)],
                timeout_s=30.0,
                cancel_requested=cancel.is_set,
                poll_interval_s=0.05,
            )
    finally:
        timer.cancel()

    assert time.monotonic() - started < 10.0
    assert _stopped_writing(marker)


def test_run_cancellable_kills_the_child_on_timeout(tmp_path):
    marker = tmp_path / "alive.txt"
    marker.write_text("", encoding="utf-8")

    with pytest.raises(subprocess.TimeoutExpired):
        run_cancellable(
            [sys.executable, "-c", _BUSY_CHILD, str(marker)],
            timeout_s=0.5,
            poll_interval_s=0.05,
        )

    assert _stopped_writing(marker)


def test_a_child_that_survives_the_kill_stays_tracked(monkeypatch, tmp_path):
    # `run_cancellable` returns and its locals go with it. A child it could not
    # confirm it killed used to stop existing as far as this process was
    # concerned - no record, no retry, nothing to show the user.
    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", lambda _h: True)
    monkeypatch.setattr(subprocess_tools, "child_state", lambda _proc: "unknown")
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(SubprocessCancelledError):
        run_cancellable(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_s=30.0,
            cancel_requested=cancel.is_set,
            poll_interval_s=0.05,
        )

    pending = subprocess_tools.unreleased_children()
    assert len(pending) == 1
    assert pending[0]["state"] == "unknown"
    assert pending[0]["reason"] == "cancel"
    # Only the program name is kept: the arguments carry media paths.
    assert pending[0]["name"] == os.path.basename(sys.executable)

    # ...and it is dropped only once the child is confirmed gone.
    monkeypatch.setattr(subprocess_tools, "child_state", lambda _proc: "gone")
    assert subprocess_tools.retry_unreleased_cleanup() == {"pending": 0, "released": 1}
    assert subprocess_tools.unreleased_children() == []


def test_a_job_handle_that_will_not_close_is_kept(monkeypatch):
    # CloseHandle failing releases nothing; forgetting the integer leaves a
    # kill-on-close job with no way to reach it.
    monkeypatch.setattr(subprocess_tools, "bind_kill_on_close_job", lambda _h: 4242)
    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", lambda _h: False)

    run_cancellable([sys.executable, "-c", "pass"], timeout_s=30.0)

    pending = subprocess_tools.unreleased_children()
    assert len(pending) == 1
    assert pending[0]["handle_open"] is True
    assert pending[0]["reason"] == "handle_close_failed"


def test_two_concurrent_retries_reap_a_child_once(monkeypatch):
    # 重试清理 walks this table and the ASR one from two threads. The list is
    # copied under the lock and the reap runs outside it (kill + wait blocks),
    # so without an exclusive claim both callers close the same job handle.
    closed: list[int] = []
    hold = threading.Event()

    def slow_close(handle):
        closed.append(handle)
        hold.wait(3.0)
        return True

    monkeypatch.setattr(subprocess_tools, "bind_kill_on_close_job", lambda _h: 4242)
    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", lambda _h: False)
    run_cancellable([sys.executable, "-c", "pass"], timeout_s=30.0)
    assert len(subprocess_tools.unreleased_children()) == 1

    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", slow_close)
    first = threading.Thread(target=subprocess_tools.retry_unreleased_cleanup)
    first.start()
    deadline = time.monotonic() + 3.0
    while not closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert closed == [4242]  # the first retry is inside CloseHandle right now

    second = threading.Thread(target=subprocess_tools.retry_unreleased_cleanup)
    second.start()
    second.join(3.0)
    assert second.is_alive() is False  # it skipped the claimed record

    hold.set()
    first.join(3.0)
    assert closed == [4242]
    assert subprocess_tools.unreleased_children() == []


def test_a_media_child_is_kept_until_it_is_confirmed_gone(monkeypatch):
    # The child is handed to the shared resource registry, which owns the claim,
    # the retry schedule and the "is this really released?" question. What must
    # stay true here: the record is only dropped when both the process and the
    # handle are confirmed.
    monkeypatch.setattr(subprocess_tools, "bind_kill_on_close_job", lambda _h: 303)
    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", lambda _h: False)
    run_cancellable([sys.executable, "-c", "pass"], timeout_s=30.0)

    pending = subprocess_tools.unreleased_children()
    assert len(pending) == 1
    assert pending[0]["handle_open"] is True
    assert subprocess_tools.retry_unreleased_cleanup() == {"pending": 1, "released": 0}

    monkeypatch.setattr(subprocess_tools, "close_kill_on_close_job", lambda _h: True)
    assert subprocess_tools.retry_unreleased_cleanup() == {"pending": 0, "released": 1}
    assert subprocess_tools.unreleased_children() == []


def test_run_cancellable_returns_captured_output():
    completed = run_cancellable(
        [sys.executable, "-c", "print('12.5')"],
        timeout_s=30.0,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "12.5"


def test_extract_audio_passes_a_live_cancel_channel(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        write_pcm_wav(command[command.index("-loglevel") - 1])

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    cancel_event = threading.Event()

    pipeline_audio.extract_audio(
        "input.mp4",
        str(tmp_path / "out.wav"),
        cancel_event=cancel_event,
    )

    assert seen["cancel_requested"]() is False
    cancel_event.set()
    assert seen["cancel_requested"]() is True


def test_cancelled_extraction_raises_cancelled_not_a_failure(monkeypatch, tmp_path):
    # What the cancelled extraction leaves on disk is covered by
    # test_audio_atomic_publish; here it only has to be reported as 取消 rather
    # than as a broken video.
    out_path = tmp_path / "sample.wav"

    def fake_run(command, **kwargs):
        del command, kwargs
        raise SubprocessCancelledError("cancelled")

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(PipelineCancelledError):
        pipeline_audio.extract_audio(
            "input.mp4",
            str(out_path),
            cancel_event=cancel_event,
        )


def test_probe_video_duration_is_bounded_and_cancellable(monkeypatch):
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "12.5\n", "")

    monkeypatch.delenv("FFPROBE_TIMEOUT_S", raising=False)
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)

    assert pipeline_audio.probe_video_duration_s("input.mp4") == 12.5
    assert 0.0 < seen["timeout_s"] <= 600.0
    assert seen["cancel_requested"] is not None


def test_probe_video_duration_reports_cancellation_instead_of_no_duration(monkeypatch):
    # Swallowing this as "unknown duration" would send the caller straight into
    # a multi-hour extraction for a job the user already cancelled.
    def fake_run(command, **kwargs):
        del command, kwargs
        raise SubprocessCancelledError("cancelled")

    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(PipelineCancelledError):
        pipeline_audio.probe_video_duration_s("input.mp4", cancel_event=cancel_event)


def test_probe_failure_still_reports_unknown_duration(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)

    assert pipeline_audio.probe_video_duration_s("input.mp4") is None


def test_extract_timeout_probe_inherits_the_cancel_event(monkeypatch):
    seen: dict = {}

    def fake_probe(video_path, *, cancel_event=None):
        del video_path
        seen["cancel_event"] = cancel_event
        return 60.0

    monkeypatch.delenv("AUDIO_EXTRACT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(pipeline_audio, "probe_video_duration_s", fake_probe)
    cancel_event = threading.Event()

    assert pipeline_audio._audio_extract_timeout_s(
        "input.mp4",
        cancel_event=cancel_event,
    ) == 360.0
    assert seen["cancel_event"] is cancel_event
