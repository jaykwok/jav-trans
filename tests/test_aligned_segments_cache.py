import json
from pathlib import Path

import pytest

import main
from pipeline import audio as pipeline_audio
from pipeline.audio import get_audio_cache_key
from helpers import make_job_context, run_pipeline


def _assert_no_project_absolute_path(text: str) -> None:
    assert main.PROJECT_ROOT.resolve().as_posix() not in text.replace("\\", "/")


def _configure(monkeypatch) -> None:
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )


def _cache_signature(ctx) -> dict:
    return main.aligned_cache_expectations_for_ctx(ctx, backend_label="mock_asr")[1]


def test_aligned_cache_signature_uses_full_subtitle_options(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        skip_translation=True,
        keep_temp_files=True,
    )
    _configure(monkeypatch)
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setattr(pipeline_audio, "probe_video_fps", lambda _path: 25.0)

    expected = main.aligned_cache_expectations_for_ctx(
        ctx,
        backend_label="mock_asr",
        video_fps=pipeline_audio.probe_video_fps(str(video_path)),
    )[1]

    assert expected["asr_stage_config"]["BOUNDARY_FEATURE_FRAME_HOP_S"] == "0.02"
    assert "BOUNDARY_FRAME_HOP_S" not in expected["asr_stage_config"]
    assert expected["subtitle"]["timeline_mode"] == "alignment"
    assert expected["subtitle"]["video_fps"] == 25.0
    assert expected["subtitle"]["effective_video_fps"] == 25.0
    assert expected["subtitle"]["frame_gap_s"] == pytest.approx(2 / 25.0)
    assert expected["subtitle"]["display_policy"]["mode"] == "readability"
    assert expected["subtitle"]["display_policy"]["version"] == 1
    assert "dense_cue_merge_enabled" not in expected["subtitle"]


def test_aligned_segments_written_with_audio_cache_key(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    calls = {"asr": 0}

    _configure(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        calls["asr"] += 1
        assert include_details is True
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [{"text": "こんにちは"}], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    run_pipeline(video_path, ctx)

    aligned_path = temp_root / "clip" / "clip.aligned_segments.json"
    payload = json.loads(aligned_path.read_text(encoding="utf-8"))
    assert calls["asr"] == 1
    assert payload["backend"] == "mock_asr"
    assert payload["audio_cache_key"]
    assert payload["cache_stage"] == "ready"
    assert payload["cache_signature"]["version"] == 7
    assert payload["cache_signature"]["subtitle"]["timeline_mode"] == "alignment"
    assert payload["cache_signature"]["subtitle"]["display_policy"]["mode"] == "readability"
    assert payload["segments"] == [{"start": 0.0, "end": 1.0, "text": "こんにちは"}]
    assert payload["asr_log"] == ["mock asr"]
    _assert_no_project_absolute_path(aligned_path.read_text(encoding="utf-8"))


def test_aligned_cache_signature_changes_with_display_policy(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        skip_translation=True,
        keep_temp_files=True,
    )
    _configure(monkeypatch)
    monkeypatch.setattr(pipeline_audio, "probe_video_fps", lambda _path: 25.0)

    default_signature = main.aligned_cache_expectations_for_ctx(
        ctx,
        backend_label="mock_asr",
        video_fps=25.0,
    )[1]
    monkeypatch.setenv("SUBTITLE_DISPLAY_POLICY", "off")
    off_signature = main.aligned_cache_expectations_for_ctx(
        ctx,
        backend_label="mock_asr",
        video_fps=25.0,
    )[1]

    assert default_signature["subtitle"]["display_policy"]["mode"] == "readability"
    assert off_signature["subtitle"]["display_policy"]["mode"] == "off"
    assert default_signature != off_signature


def test_aligned_segments_cache_hit_skips_asr(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    job_dir = temp_root / "clip"
    job_dir.mkdir(parents=True)
    audio_cache_key = get_audio_cache_key(str(video_path))
    aligned_path = job_dir / "clip.aligned_segments.json"
    _configure(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )
    aligned_path.write_text(
        json.dumps(
            {
                "backend": "mock_asr",
                "audio_path": str(job_dir / "audio" / f"clip.{audio_cache_key}.wav"),
                "audio_cache_key": audio_cache_key,
                "segments": [{"start": 0.0, "end": 1.0, "text": "cached-ja"}],
                "asr_details": {"transcript_chunks": [], "stage_timings": {}},
                "asr_log": ["cached asr"],
                "cache_signature": _cache_signature(ctx),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_audio,
        "extract_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("extract_audio should be skipped on aligned cache hit")
        ),
    )
    monkeypatch.setattr(
        main.asr_module,
        "transcribe_and_align",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ASR should be skipped on aligned cache hit")
        ),
    )

    run_pipeline(video_path, ctx)

    assert "cached-ja" in (output_dir / "clip.ja.srt").read_text(encoding="utf-8")


def test_aligned_segments_cache_miss_when_signature_missing(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    job_dir = temp_root / "clip"
    job_dir.mkdir(parents=True)
    audio_cache_key = get_audio_cache_key(str(video_path))
    aligned_path = job_dir / "clip.aligned_segments.json"
    aligned_path.write_text(
        json.dumps(
            {
                "backend": "mock_asr",
                "audio_cache_key": audio_cache_key,
                "segments": [{"start": 0.0, "end": 1.0, "text": "stale"}],
                "asr_details": {},
                "asr_log": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls = {"asr": 0}

    _configure(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        calls["asr"] += 1
        return (
            [{"start": 0.0, "end": 1.0, "text": "fresh"}],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    run_pipeline(video_path, ctx)

    assert calls["asr"] == 1
    assert "fresh" in (output_dir / "clip.ja.srt").read_text(encoding="utf-8")


def test_aligned_segments_cache_miss_when_audio_key_changes(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    job_dir = temp_root / "clip"
    job_dir.mkdir(parents=True)
    aligned_path = job_dir / "clip.aligned_segments.json"
    aligned_path.write_text(
        json.dumps(
            {
                "backend": "mock_asr",
                "audio_cache_key": "stale-key",
                "segments": [{"start": 0.0, "end": 1.0, "text": "stale"}],
                "asr_details": {},
                "asr_log": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls = {"asr": 0}

    _configure(monkeypatch)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        calls["asr"] += 1
        return (
            [{"start": 0.0, "end": 1.0, "text": "fresh"}],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    run_pipeline(video_path, ctx)

    assert calls["asr"] == 1
    assert "fresh" in (output_dir / "clip.ja.srt").read_text(encoding="utf-8")

