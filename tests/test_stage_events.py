from __future__ import annotations

import os
from datetime import datetime
from io import StringIO
from pathlib import Path

from rich.console import Console

from core import events
import main
from pipeline import audio as pipeline_audio
from pipeline.stage_log import _parse_asr_stage_event, _timing_summary_rows
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

    def fake_transcribe_and_align(
        _audio_path,
        *,
        device="auto",
        env_overrides=None,
        job_id="",
        on_stage=None,
        cancel_requested=None,
    ):
        assert device == "auto"
        assert env_overrides is not None
        assert job_id
        assert cancel_requested is not None
        if on_stage:
            for message in (
                "语音岛检测 1/1",
                "外边界精修 1/1",
                "语义切分判断 1/1",
                "内部切点精修 1/1",
                "Pre-ASR CueQC 1/1",
                "音频切块 1/1",
                "ASR 文本转写 1/1",
            ):
                on_stage(message)
        return (
            segments,
            ["mock asr"],
            {
                "transcript_chunks": [],
                "stage_timings": {},
            },
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        fake_transcribe_and_align,
    )
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
    assert ("speech_island_scorer", "done") in observed
    assert ("outer_edge_refiner", "done") in observed
    assert ("semantic_split_model", "done") in observed
    assert ("pre_asr_cueqc", "done") in observed
    assert ("audio_chunk_export", "done") in observed
    assert ("write_output", "done") in observed
    assert ("timing_summary", "done") in observed

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


def test_five_model_progress_labels_map_to_frontend_stages():
    assert _parse_asr_stage_event("语音岛检测 1/1") == (
        "speech_island_scorer",
        {"label": "语音岛检测", "current": 1, "total": 1},
    )
    assert _parse_asr_stage_event("外边界精修 1/1")[0] == "outer_edge_refiner"
    assert _parse_asr_stage_event("语义切分判断 3/8")[0] == "semantic_split_model"
    assert _parse_asr_stage_event("Pre-ASR CueQC 1/1")[0] == "pre_asr_cueqc"
    assert _parse_asr_stage_event("音频切块 2/5")[0] == "audio_chunk_export"


def test_timing_summary_matches_current_non_overlapping_pipeline_stages():
    rows = _timing_summary_rows(
        {
            "audio_prepare_s": 1.0,
            "asr_alignment_total_s": 20.0,
            "translation_handoff_snapshot_s": 0.2,
            "subtitle_cue_plan_s": 2.0,
            "translation_context_s": 3.0,
            "translation_s": 4.0,
            "write_output_s": 5.0,
            "pipeline_total_s": 35.0,
        },
        {
            "stage_timings": {
                "split_s": 6.0,
                "asr_model_load_s": 1.0,
                "asr_text_transcribe_s": 8.0,
                "asr_model_unload_s": 1.0,
                "alignment_s": 3.0,
                "subtitle_segment_s": 1.0,
            }
        },
    )

    keys = [row["key"] for row in rows]
    assert "split_s" in keys
    assert "asr_alignment_total_s" not in keys
    assert "translation_handoff_snapshot_s" not in keys
    assert keys[-1] == "pipeline_total_s"


def test_timing_summary_zeros_cached_asr_stages():
    rows = _timing_summary_rows(
        {"audio_prepare_s": 0.1, "asr_alignment_total_s": 0.0},
        {"stage_timings": {"split_s": 99.0, "asr_text_transcribe_s": 88.0}},
    )

    assert {row["seconds"] for row in rows if row["key"] in {"split_s", "asr_text_transcribe_s"}} == {0.0}
