"""Audio extraction publishes a finished file, or nothing at all.

The cache reader downstream trusts the file under `audio_path`. As long as
ffmpeg wrote straight to that name, every way of dying mid-extraction -
cancellation, timeout, a crash, the process being hard-killed - left a
truncated wav there, and the next run replayed it as a complete extraction and
transcribed the film short. Cleanup on the error path cannot fix that: a hard
kill runs no cleanup. So extraction writes a `.partial` and publishes by an
atomic replace, and cleanup goes back to being about disk space.
"""

from __future__ import annotations

import subprocess
import threading
import wave
from pathlib import Path

import pytest

import main
from core.cancellation import PipelineCancelledError
from helpers import make_job_context, run_pipeline
from pipeline import audio as pipeline_audio
from utils.subprocess_tools import SubprocessCancelledError


def _write_wav(path: Path, *, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(16000 * seconds)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * frames)


def _write_truncated_wav(path: Path) -> None:
    """What a killed ffmpeg leaves: real audio bytes, unpatched header sizes."""
    _write_wav(path, seconds=0.5)
    raw = bytearray(path.read_bytes())
    raw[4:8] = (0).to_bytes(4, "little")  # RIFF size
    raw[40:44] = (0).to_bytes(4, "little")  # data size
    path.write_bytes(bytes(raw))


def _output_path(command: list[str]) -> str:
    return command[command.index("-loglevel") - 1]


def test_cancelled_extraction_publishes_nothing(monkeypatch, tmp_path):
    out_path = tmp_path / "sample.wav"
    partials: list[Path] = []

    def fake_run(command, **kwargs):
        del kwargs
        temp_path = Path(_output_path(command))
        _write_truncated_wav(temp_path)
        partials.append(temp_path)
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

    assert not out_path.exists()
    assert partials and partials[0].name.endswith(".partial")
    assert not partials[0].exists()


def test_hard_kill_during_write_leaves_the_previous_cache_intact(monkeypatch, tmp_path):
    # "Hard kill" = no cleanup runs at all: the partial survives on disk. What
    # must not happen is the cache name changing.
    out_path = tmp_path / "sample.wav"
    _write_wav(out_path, seconds=0.3)
    previous = out_path.read_bytes()

    def fake_run(command, **kwargs):
        del kwargs
        _write_truncated_wav(Path(_output_path(command)))
        raise KeyboardInterrupt("hard kill")

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    monkeypatch.setattr(pipeline_audio, "_discard_partial", lambda _path: None)

    with pytest.raises(KeyboardInterrupt):
        pipeline_audio.extract_audio("input.mp4", str(out_path))

    assert out_path.read_bytes() == previous
    assert list(tmp_path.glob("*.partial"))  # the residue is disk, not cache


@pytest.mark.parametrize("stage", ["validate", "publish"])
def test_failure_after_ffmpeg_never_half_publishes(monkeypatch, tmp_path, stage):
    out_path = tmp_path / "sample.wav"
    _write_wav(out_path, seconds=0.3)
    previous = out_path.read_bytes()

    def fake_run(command, **kwargs):
        del kwargs
        temp_path = Path(_output_path(command))
        if stage == "validate":
            _write_truncated_wav(temp_path)  # ffmpeg exits 0 with a bad file
        else:
            _write_wav(temp_path, seconds=0.2)

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)
    if stage == "publish":
        def fail_publish(_temp_path, _out_path):
            raise OSError("replace failed")

        monkeypatch.setattr(pipeline_audio, "publish_audio_file", fail_publish)

    with pytest.raises((RuntimeError, OSError)):
        pipeline_audio.extract_audio("input.mp4", str(out_path))

    assert out_path.read_bytes() == previous
    assert not list(tmp_path.glob("*.partial"))


def test_published_cache_is_the_complete_new_file(monkeypatch, tmp_path):
    out_path = tmp_path / "sample.wav"
    _write_wav(out_path, seconds=0.3)

    def fake_run(command, **kwargs):
        del kwargs
        _write_wav(Path(_output_path(command)), seconds=1.0)

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(out_path))

    assert pipeline_audio.audio_file_is_complete(str(out_path))
    with wave.open(str(out_path), "rb") as reader:
        assert reader.getnframes() == 16000
    assert not list(tmp_path.glob("*.partial"))


def test_extraction_writes_a_partial_not_the_cache_name(monkeypatch, tmp_path):
    out_path = tmp_path / "sample.wav"
    seen: dict[str, str] = {}

    def fake_run(command, **kwargs):
        del kwargs
        target = _output_path(command)
        seen["target"] = target
        # The muxer can no longer be inferred from the extension.
        assert command[command.index("-f") + 1] == "wav"
        _write_wav(Path(target))

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(out_path))

    assert seen["target"] != str(out_path)
    assert seen["target"].startswith(str(out_path))
    assert seen["target"].endswith(".partial")


def test_two_jobs_extracting_the_same_input_get_distinct_partials(tmp_path):
    out_path = str(tmp_path / "sample.wav")

    first = pipeline_audio._partial_audio_path(out_path)
    second = pipeline_audio._partial_audio_path(out_path)

    assert first != second
    assert first.startswith(out_path) and first.endswith(".partial")


def test_audio_cache_only_rejects_a_wav_ffmpeg_never_finished(tmp_path):
    complete = tmp_path / "complete.wav"
    truncated = tmp_path / "truncated.wav"
    foreign = tmp_path / "foreign.wav"
    _write_wav(complete)
    _write_truncated_wav(truncated)
    foreign.write_bytes(b"not a riff file")

    assert pipeline_audio.audio_cache_is_usable(str(complete))
    assert not pipeline_audio.audio_cache_is_usable(str(truncated))
    assert not pipeline_audio.audio_cache_is_usable(str(tmp_path / "missing.wav"))
    # Anything that is not a wav at all is left to the ASR stage to reject, so
    # this check can only ever heal the truncation case.
    assert pipeline_audio.audio_cache_is_usable(str(foreign))


def test_completeness_check_rejects_the_wrong_format(tmp_path):
    wrong_rate = tmp_path / "wrong.wav"
    wrong_rate.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wrong_rate), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\x00\x00" * 1000)

    assert not pipeline_audio.audio_file_is_complete(str(wrong_rate))


def test_reclaim_removes_only_stale_partials(tmp_path):
    import os
    import time

    jobs_root = tmp_path / "jobs" / "job-a" / "audio"
    jobs_root.mkdir(parents=True)
    stale = jobs_root / "sample.wav.deadbeef.partial"
    fresh = jobs_root / "sample.wav.cafebabe.partial"
    published = jobs_root / "sample.wav"
    for path in (stale, fresh, published):
        _write_wav(path)
    old = time.time() - 7200
    os.utime(stale, (old, old))

    removed = pipeline_audio.reclaim_stale_partials(tmp_path / "jobs")

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
    assert published.exists()


def test_truncated_cache_is_re_extracted_by_the_pipeline(monkeypatch, tmp_path):
    # End to end: an unfinished wav left by an older build under the cache name
    # must not be replayed as a cache hit.
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    temp_root = tmp_path / "jobs"
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )
    monkeypatch.setattr(main.audio_module, "get_audio_cache_key", lambda _path: "cachekey")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )
    stale_cache = temp_root / "clip" / "audio" / "clip.cachekey.wav"
    _write_truncated_wav(stale_cache)
    extracted: list[str] = []

    def fake_extract_audio(_video_path: str, out_path: str, **_kwargs) -> None:
        extracted.append(out_path)
        _write_wav(Path(out_path))

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        lambda *_args, **_kwargs: (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        ),
    )

    run_pipeline(video_path, ctx)

    assert extracted == [str(stale_cache)]
    assert pipeline_audio.audio_file_is_complete(str(stale_cache))


def test_timeout_still_reports_the_clear_error(monkeypatch, tmp_path):
    def fake_run(command, *, timeout_s, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(command, timeout_s)

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio, "run_cancellable", fake_run)

    with pytest.raises(TimeoutError, match="ffmpeg audio extraction timed out"):
        pipeline_audio.extract_audio("input.mp4", str(tmp_path / "sample.wav"))

    assert not (tmp_path / "sample.wav").exists()
