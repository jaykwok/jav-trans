from pathlib import Path

import main
from helpers import ASR_17B_BACKEND, make_job_context
from pipeline import audio as pipeline_audio


def test_asr_stage_env_scope_reaches_cache_and_transcribe(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
    )
    # ASR_BACKEND is a deployment setting: the job carries no backend choice,
    # so the process value must reach every stage unchanged.
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.setenv("ASR_CONTEXT", "process actor")
    monkeypatch.setenv("ASR_CHUNK_TARGET_S", "20.0")
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    seen = {}

    def fake_get_backend_label():
        seen["backend_label_env"] = {
            "ASR_BACKEND": main.os.environ.get("ASR_BACKEND"),
            "ASR_CHUNK_TARGET_S": main.os.environ.get(
                "ASR_CHUNK_TARGET_S"
            ),
        }
        return f"backend:{main.os.environ['ASR_BACKEND']}"

    def fake_try_load_aligned_segments(
        _path,
        _audio_key,
        expected_backend,
        expected_signature=None,
    ):
        seen["cache_backend"] = expected_backend
        seen["cache_signature"] = expected_signature
        seen["cache_env"] = {
            "ASR_BACKEND": main.os.environ.get("ASR_BACKEND"),
            "ASR_CHUNK_TARGET_S": main.os.environ.get(
                "ASR_CHUNK_TARGET_S"
            ),
        }
        return None

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(
        _audio_path,
        *,
        device="auto",
        env_overrides=None,
        job_id="",
        on_stage=None,
        cancel_requested=None,
    ):
        seen["transcribe_env"] = {
            "ASR_BACKEND": env_overrides.get("ASR_BACKEND"),
            "ASR_CHUNK_TARGET_S": env_overrides.get(
                "ASR_CHUNK_TARGET_S"
            ),
        }
        assert device == "auto"
        assert job_id
        assert cancel_requested is not None
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(main.asr_module, "get_backend_label", fake_get_backend_label)
    monkeypatch.setattr(
        main.aligned_cache_module,
        "try_load_aligned_segments",
        fake_try_load_aligned_segments,
    )
    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        fake_transcribe_and_align,
    )

    artifacts = main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )

    expected_env = {
        "ASR_BACKEND": ASR_17B_BACKEND,
        "ASR_CHUNK_TARGET_S": "20.0",
    }
    assert seen["backend_label_env"] == expected_env
    assert seen["cache_env"] == expected_env
    assert seen["transcribe_env"] == expected_env
    assert seen["cache_backend"] == f"backend:{ASR_17B_BACKEND}"
    assert seen["cache_signature"]["backend_label"] == f"backend:{ASR_17B_BACKEND}"
    assert (
        seen["cache_signature"]["asr_stage_config"]["ASR_CHUNK_TARGET_S"]
        == "20.0"
    )
    assert "video_fps" not in seen["cache_signature"]["subtitle"]
    assert "effective_video_fps" not in seen["cache_signature"]["subtitle"]
    assert seen["cache_signature"]["subtitle"]["frame_gap_s"] == main.subtitle_module.SubtitleOptions().frame_gap_s
    assert "dense_cue_merge_enabled" not in seen["cache_signature"]["subtitle"]
    assert artifacts.backend_label == f"backend:{ASR_17B_BACKEND}"
    assert main.os.environ["ASR_BACKEND"] == ASR_17B_BACKEND
    assert main.os.environ["ASR_CONTEXT"] == "process actor"


def test_asr_stage_env_scope_passes_chunking_and_alignment_flags(monkeypatch, tmp_path):
    """Per-job advanced settings must survive the allowlist that feeds the worker.

    The allowlist listed only the retired chain's prefixes until 2026-07-31, so
    `ASR_ALIGNMENT_HEAD_PATH` set on a job was filtered out and the head simply
    never loaded - the failure looked like "alignment is off", not like a
    dropped setting.
    """
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
        advanced={
            "ASR_ALIGNMENT_HEAD_PATH": "src/checkpoints/ctc_aligner.pt",
            "ASR_CHUNK_MIN_PAUSE_S": "0.8",
        },
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    seen = {}

    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(main.aligned_cache_module, "try_load_aligned_segments", lambda *a, **k: None)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(
        _audio_path,
        *,
        device="auto",
        env_overrides=None,
        job_id="",
        on_stage=None,
        cancel_requested=None,
    ):
        seen["head_path"] = env_overrides.get("ASR_ALIGNMENT_HEAD_PATH")
        seen["min_pause"] = env_overrides.get("ASR_CHUNK_MIN_PAUSE_S")
        assert device == "auto"
        assert job_id
        assert cancel_requested is not None
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        fake_transcribe_and_align,
    )

    main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )

    assert seen == {
        "head_path": "src/checkpoints/ctc_aligner.pt",
        "min_pause": "0.8",
    }


def test_chunk_root_reaches_transcribe_but_not_aligned_signature(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        skip_translation=True,
        keep_temp_files=True,
        advanced={"ASR_CHUNK_ROOT": str(tmp_path / "chunks-a")},
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(
        main.asr_module,
        "_get_asr_runtime_signature",
        lambda last_boundary_signature=None: {"asr": "sig"},
    )

    seen = {}

    def fake_try_load_aligned_segments(
        _path,
        _audio_key,
        _expected_backend,
        expected_signature=None,
    ):
        seen["cache_signature"] = expected_signature
        return None

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(
        _audio_path,
        *,
        device="auto",
        env_overrides=None,
        job_id="",
        on_stage=None,
        cancel_requested=None,
    ):
        seen["transcribe_chunk_root"] = env_overrides.get("ASR_CHUNK_ROOT")
        assert device == "auto"
        assert job_id
        assert cancel_requested is not None
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(
        main.aligned_cache_module,
        "try_load_aligned_segments",
        fake_try_load_aligned_segments,
    )
    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        main.asr_stage_worker_module,
        "transcribe_and_align",
        fake_transcribe_and_align,
    )

    main.run_asr_alignment(str(video_path), ctx=ctx, job_id=ctx.job_id)

    assert seen["transcribe_chunk_root"] == str(tmp_path / "chunks-a")
    # Where the chunks are written does not change what they contain, so it
    # must not invalidate the aligned-segment cache.
    assert "ASR_CHUNK_ROOT" not in seen["cache_signature"]["asr_stage_config"]
