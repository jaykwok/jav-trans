import json
from pathlib import Path

import main
from pipeline import audio as pipeline_audio
from helpers import make_job_context, run_pipeline


def _assert_no_project_absolute_path(text: str) -> None:
    assert main.PROJECT_ROOT.resolve().as_posix() not in text.replace("\\", "/")


def test_job_tempdir_groups_temp_outputs_and_keeps_srt_at_output_root(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    seen_audio_path = {}

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        seen_audio_path["path"] = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"")

    def fake_transcribe_and_align(audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        assert Path(audio_path).parent == temp_root / "sample" / "audio"
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("translation should be skipped")
        ),
    )

    run_pipeline(video_path, ctx)

    job_dir = temp_root / "sample"
    assert Path(seen_audio_path["path"]).is_file()
    assert (output_dir / "sample.ja.srt").is_file()
    assert not (job_dir / "sample.ja.srt").exists()

    for suffix in (
        "asr_manifest.json",
        "transcript.json",
        "aligned_segments.json",
        "bilingual.json",
        "timings.json",
    ):
        assert (job_dir / f"sample.{suffix}").is_file()
        assert not (output_dir / f"sample.{suffix}").exists()

    timings = json.loads((job_dir / "sample.timings.json").read_text(encoding="utf-8"))
    assert timings["job_id"] == "sample"
    assert Path(timings["job_temp_dir"]) == job_dir.relative_to(main.PROJECT_ROOT)
    assert Path(timings["outputs"]["srt"]) == (output_dir / "sample.ja.srt").relative_to(main.PROJECT_ROOT)
    assert timings["outputs"]["run_log"] is None
    _assert_no_project_absolute_path((job_dir / "sample.timings.json").read_text(encoding="utf-8"))
    _assert_no_project_absolute_path((job_dir / "sample.aligned_segments.json").read_text(encoding="utf-8"))
    _assert_no_project_absolute_path((job_dir / "sample.asr_manifest.json").read_text(encoding="utf-8"))


def test_run_log_is_written_only_when_enabled(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    log_dir = tmp_path / "logs"

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
        run_log_enabled=True,
        run_log_dir=log_dir,
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
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)
    monkeypatch.setattr(
        main.translator_module,
        "translate_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("translation should be skipped")
        ),
    )

    artifacts = main.run_asr_alignment_f0(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )
    logger = artifacts.logger
    assert logger is not None
    main.run_translation_and_write(
        str(video_path),
        artifacts,
        ctx=ctx,
        job_id=ctx.job_id,
    )
    assert artifacts.logger is None
    assert not logger.handlers
    assert getattr(main.events._thread_local, "run_logger", None) is not logger

    timings = json.loads(
        (temp_root / "sample" / "sample.timings.json").read_text(encoding="utf-8")
    )
    run_log = Path(timings["outputs"]["run_log"])
    assert run_log.parent == log_dir.relative_to(main.PROJECT_ROOT)
    run_log_path = main.PROJECT_ROOT / run_log
    assert run_log_path.is_file()
    content = run_log_path.read_text(encoding="utf-8")
    assert "run_start" in content
    assert "stage_start audio_prepare" in content
    assert "run_done" in content
    _assert_no_project_absolute_path(content)


def test_successful_run_cleans_job_temp_by_default(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=False,
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert include_details is True
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    run_pipeline(video_path, ctx)

    assert (output_dir / "sample.ja.srt").is_file()
    assert not (temp_root / "sample").exists()


def test_advanced_asr_stage_env_is_task_scoped(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setenv("WHISPERSEG_CHUNK_THRESHOLD_S", "1.0")
    original_f0_threshold = main.os.environ.get("F0_THRESHOLD_HZ")
    monkeypatch.delenv("WHISPERSEG_THRESHOLD", raising=False)
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
        vad_threshold=0.42,
        advanced={
            "WHISPERSEG_CHUNK_THRESHOLD_S": "1.5",
            "F0_THRESHOLD_HZ": "180",
        },
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        assert main.os.environ["WHISPERSEG_CHUNK_THRESHOLD_S"] == "1.5"
        assert main.os.environ["F0_THRESHOLD_HZ"] == "180"
        assert main.os.environ["WHISPERSEG_THRESHOLD"] == "0.42"
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            [],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    run_pipeline(video_path, ctx)

    assert main.os.environ["WHISPERSEG_CHUNK_THRESHOLD_S"] == "1.0"
    assert main.os.environ.get("F0_THRESHOLD_HZ") == original_f0_threshold
    assert "WHISPERSEG_THRESHOLD" not in main.os.environ

