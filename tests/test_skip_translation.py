import re
from io import StringIO
from pathlib import Path

from rich.console import Console

import main
from pipeline import audio as pipeline_audio
from helpers import make_job_context, run_pipeline
_TIMELINE_RE = re.compile(
    r"(?P<start>\d\d:\d\d:\d\d,\d{3}) --> (?P<end>\d\d:\d\d:\d\d,\d{3})"
)


def _seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    secs, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(secs)
        + int(millis) / 1000.0
    )


def test_skip_translation_writes_japanese_srt(monkeypatch, tmp_path):
    job_temp_root = tmp_path / "jobs"
    monkeypatch.setattr(
        main,
        "console",
        Console(
            file=StringIO(),
            force_terminal=False,
            emoji=False,
        ),
    )
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        tmp_path,
        job_temp_root,
        subtitle_mode="bilingual",
        skip_translation=True,
        keep_temp_files=True,
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        if on_stage:
            on_stage("ASR mock")
        return (
            [
                {"start": 0.0, "end": 1.0, "text": "こんにちは"},
                {"start": 1.5, "end": 2.4, "text": "テストです"},
            ],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    def fail_translate_segments(*_args, **_kwargs):
        raise AssertionError("translate_segments must not be called")

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)
    monkeypatch.setattr(main.translator_module, "translate_segments", fail_translate_segments)

    run_pipeline(video_path, ctx)

    srt_path = tmp_path / "sample.ja.srt"
    assert srt_path.exists()
    assert not (tmp_path / "sample.srt").exists()

    content = srt_path.read_text(encoding="utf-8")
    assert "こんにちは" in content
    assert "テストです" in content

    windows = [_TIMELINE_RE.search(line) for line in content.splitlines()]
    windows = [match for match in windows if match]
    assert len(windows) == 2

    starts = [_seconds(match.group("start")) for match in windows]
    ends = [_seconds(match.group("end")) for match in windows]
    assert starts == sorted(starts)
    assert all(start < end for start, end in zip(starts, ends))

    timings_path = job_temp_root / "sample" / "sample.timings.json"
    timings = timings_path.read_text(encoding="utf-8")
    assert '"translation_skipped": true' in timings

