from pathlib import Path

import main
from helpers import ASR_06B_BACKEND, ASR_17B_BACKEND, make_job_context
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
        asr_backend=ASR_17B_BACKEND,
        asr_context="task actor",
        skip_translation=True,
        keep_temp_files=True,
    )
    monkeypatch.setenv("ASR_BACKEND", ASR_06B_BACKEND)
    monkeypatch.setenv("ASR_CONTEXT", "process actor")
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline_audio, "probe_video_fps", lambda _path: 60.0)

    seen = {}

    def fake_get_backend_label():
        seen["backend_label_env"] = {
            "ASR_BACKEND": main.os.environ.get("ASR_BACKEND"),
            "ASR_CONTEXT": main.os.environ.get("ASR_CONTEXT"),
            "BOUNDARY_FEATURE_FRAME_HOP_S": main.os.environ.get(
                "BOUNDARY_FEATURE_FRAME_HOP_S"
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
            "ASR_CONTEXT": main.os.environ.get("ASR_CONTEXT"),
            "BOUNDARY_FEATURE_FRAME_HOP_S": main.os.environ.get(
                "BOUNDARY_FEATURE_FRAME_HOP_S"
            ),
        }
        return None

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        seen["transcribe_env"] = {
            "ASR_BACKEND": main.os.environ.get("ASR_BACKEND"),
            "ASR_CONTEXT": main.os.environ.get("ASR_CONTEXT"),
            "BOUNDARY_FEATURE_FRAME_HOP_S": main.os.environ.get(
                "BOUNDARY_FEATURE_FRAME_HOP_S"
            ),
        }
        assert include_details is True
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
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    artifacts = main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )

    expected_env = {
        "ASR_BACKEND": ASR_17B_BACKEND,
        "ASR_CONTEXT": "task actor",
        "BOUNDARY_FEATURE_FRAME_HOP_S": "0.02",
    }
    assert seen["backend_label_env"] == expected_env
    assert seen["cache_env"] == expected_env
    assert seen["transcribe_env"] == expected_env
    assert seen["cache_backend"] == f"backend:{ASR_17B_BACKEND}"
    assert seen["cache_signature"]["backend_label"] == f"backend:{ASR_17B_BACKEND}"
    assert (
        seen["cache_signature"]["asr_stage_config"]["BOUNDARY_FEATURE_FRAME_HOP_S"]
        == "0.02"
    )
    assert seen["cache_signature"]["subtitle"]["video_fps"] == 60.0
    assert seen["cache_signature"]["subtitle"]["effective_video_fps"] == 60.0
    assert seen["cache_signature"]["subtitle"]["frame_gap_s"] == 2 / 60.0
    assert "dense_cue_merge_enabled" not in seen["cache_signature"]["subtitle"]
    assert artifacts.backend_label == f"backend:{ASR_17B_BACKEND}"
    assert main.os.environ["ASR_BACKEND"] == ASR_06B_BACKEND
    assert main.os.environ["ASR_CONTEXT"] == "process actor"


def test_asr_stage_env_scope_passes_boundary_refiner_flags(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    temp_root = tmp_path / "jobs"
    ctx = make_job_context(
        video_path,
        output_dir,
        temp_root,
        asr_backend=ASR_17B_BACKEND,
        skip_translation=True,
        keep_temp_files=True,
        advanced={
            "BOUNDARY_REFINER_ENABLED": "1",
            "BOUNDARY_REFINER_THRESHOLD": "0.61",
            "BOUNDARY_PLANNER_TARGET_CHUNK_S": "3.5",
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S": "5.5",
        },
    )
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline_audio, "probe_video_fps", lambda _path: 30.0)

    seen = {}

    monkeypatch.setattr(main.asr_module, "get_backend_label", lambda: "mock_asr")
    monkeypatch.setattr(main.aligned_cache_module, "try_load_aligned_segments", lambda *a, **k: None)

    def fake_extract_audio(_video_path: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"wav")

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        seen["enabled"] = main.os.environ.get("BOUNDARY_REFINER_ENABLED")
        seen["threshold"] = main.os.environ.get("BOUNDARY_REFINER_THRESHOLD")
        seen["target_s"] = main.os.environ.get("BOUNDARY_PLANNER_TARGET_CHUNK_S")
        seen["max_core_s"] = main.os.environ.get("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S")
        return (
            [{"start": 0.0, "end": 1.0, "text": "こんにちは"}],
            ["mock asr"],
            {"transcript_chunks": [], "stage_timings": {}},
        )

    monkeypatch.setattr(pipeline_audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
    )

    assert seen == {
        "enabled": "1",
        "threshold": "0.61",
        "target_s": "3.5",
        "max_core_s": "5.5",
    }


def test_boundary_cache_dir_reaches_transcribe_but_not_aligned_signature(
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
        asr_backend=ASR_06B_BACKEND,
        skip_translation=True,
        keep_temp_files=True,
        advanced={"BOUNDARY_CACHE_DIR": str(tmp_path / "boundary-cache-a")},
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

    def fake_transcribe_and_align(_audio_path, _device, on_stage=None, include_details=False):
        seen["transcribe_cache_dir"] = main.os.environ.get("BOUNDARY_CACHE_DIR")
        assert include_details is True
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
    monkeypatch.setattr(main.asr_module, "transcribe_and_align", fake_transcribe_and_align)

    main.run_asr_alignment(str(video_path), ctx=ctx, job_id=ctx.job_id)

    assert seen["transcribe_cache_dir"] == str(tmp_path / "boundary-cache-a")
    assert "BOUNDARY_CACHE_DIR" not in seen["cache_signature"]["asr_stage_config"]
    assert "BOUNDARY_CACHE_ENABLED" not in seen["cache_signature"]["asr_stage_config"]
