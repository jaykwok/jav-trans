from __future__ import annotations

import os
from datetime import datetime
from io import StringIO
from pathlib import Path

from rich.console import Console

from core import events
import main
from helpers import make_job_context, run_pipeline


def _configure_headless(monkeypatch) -> None:
    monkeypatch.setenv("STAGE_EVENT_SINK", "memory")
    monkeypatch.setattr(
        main,
        "console",
        Console(
            file=StringIO(),
            force_terminal=False,
            emoji=False,
        ),
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)


def _mock_minimal_pipeline(monkeypatch, segments: list[dict]) -> None:
    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00"
            b"\x02\x00\x10\x00data\x00\x00\x00\x00"
        )

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        if on_stage:
            on_stage("ASR 文本转写 1/1...")
        return (
            segments,
            ["mock asr"],
            {
                "transcript_chunks": [],
                "stage_timings": {},
            },
        )

    monkeypatch.setattr(main, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("translation should be skipped")
        ),
    )


def test_stage_events_memory_sink_records_pipeline_events(monkeypatch, tmp_path):
    events.configure_sink("memory")
    video_path = tmp_path / "nested" / "sample.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake-video")
    _configure_headless(monkeypatch)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="bilingual",
        skip_translation=True,
        keep_temp_files=True,
    )
    _mock_minimal_pipeline(
        monkeypatch,
        [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
    )

    run_pipeline(video_path, ctx)

    emitted = events.get_memory_events()
    observed = {(event["stage"], event["phase"]) for event in emitted}
    assert ("audio_prepare", "start") in observed
    assert ("audio_prepare", "done") in observed
    assert ("asr_text_transcribe", "start") in observed
    assert ("asr_text_transcribe", "done") in observed
    assert ("write_output", "done") in observed

    for event in emitted:
        datetime.fromisoformat(event["ts"])
        assert event["video"] == "sample.mp4"
        assert os.sep not in event["video"]
        assert "/" not in event["video"]


def test_empty_stage_event_sink_is_silent(monkeypatch, tmp_path, capsys):
    events.configure_sink("")
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    _configure_headless(monkeypatch)
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="bilingual",
        skip_translation=True,
        keep_temp_files=True,
    )
    monkeypatch.setenv("STAGE_EVENT_SINK", "")
    _mock_minimal_pipeline(
        monkeypatch,
        [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
    )

    run_pipeline(video_path, ctx)

    stdout = capsys.readouterr().out
    assert "StageEvent" not in stdout
    assert "stage_start" not in stdout
    assert events.get_memory_events() == []
