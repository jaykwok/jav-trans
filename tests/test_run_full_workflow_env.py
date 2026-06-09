from pathlib import Path

from tools.boundary.ja import run_full_workflow


def test_run_full_workflow_parse_args_uses_loaded_env(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame")
    monkeypatch.setenv("ASR_MODEL_PATH", "")
    monkeypatch.setenv("ALIGNER_MODEL_PATH", "")
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32",
    )
    monkeypatch.setenv("ALIGNER_BATCH_SIZE", "64")
    monkeypatch.setenv("ALIGN_LONG_CHUNK_BATCH_SIZE", "48")
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH", "src/boundary/checkpoints/boundary_refiner.pt")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "5.0")

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--task-name",
            "unit",
            "--label",
            "boundary",
        ]
    )

    assert args.asr_backend == "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
    assert args.asr_model_path == ""
    assert args.aligner_model_path == ""
    assert args.asr_batch_size == "auto"
    assert args.aligner_batch_size == 64
    assert args.align_long_chunk_batch_size == 48
    assert args.boundary_refiner_model_path == "src/boundary/checkpoints/boundary_refiner.pt"
    assert args.boundary_planner_target_chunk_s == 3.0
    assert args.boundary_planner_max_core_chunk_s == 5.0


def test_run_full_workflow_context_carries_boundary_env(monkeypatch, tmp_path):
    batch_table = (
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=32,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8"
    )
    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", batch_table)
    monkeypatch.setenv("BOUNDARY_REFINER_ENABLED", "1")
    monkeypatch.setenv("BOUNDARY_REFINER_THRESHOLD", "0.62")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.5")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "5.5")

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--task-name",
            "unit",
            "--label",
            "boundary",
        ]
    )
    paths = run_full_workflow.RunPaths(
        root=tmp_path,
        jobs=tmp_path / "jobs",
        generated=tmp_path / "generated",
        run_logs=tmp_path / "run-logs",
        archived=tmp_path / "archived",
        summary_json=tmp_path / "summary.json",
        summary_md=tmp_path / "summary.md",
    )
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")

    ctx = run_full_workflow.build_context(args=args, paths=paths, video=video)

    assert ctx.advanced["ASR_BATCH_SIZE"] == "auto"
    assert ctx.advanced["ASR_BATCH_SIZE_BY_REPO"] == batch_table
    assert ctx.advanced["BOUNDARY_REFINER_ENABLED"] == "1"
    assert ctx.advanced["BOUNDARY_REFINER_THRESHOLD"] == "0.62"
    assert ctx.advanced["BOUNDARY_PLANNER_TARGET_CHUNK_S"] == "3.5"
    assert ctx.advanced["BOUNDARY_PLANNER_MAX_CORE_CHUNK_S"] == "5.5"


def test_run_full_workflow_cli_batch_overrides_loaded_env(monkeypatch):
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32",
    )
    monkeypatch.setenv("ALIGNER_BATCH_SIZE", "64")

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--asr-backend",
            "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
            "--asr-batch-size",
            "12",
            "--aligner-batch-size",
            "48",
        ]
    )
    run_full_workflow.configure_env(args)

    assert run_full_workflow.os.environ["ASR_BACKEND"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
    assert run_full_workflow.os.environ["ASR_BATCH_SIZE"] == "12"
    assert run_full_workflow.os.environ["ALIGNER_BATCH_SIZE"] == "48"
