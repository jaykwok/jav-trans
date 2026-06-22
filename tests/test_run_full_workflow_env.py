from pathlib import Path

from tools.workflows import run_full_workflow


def test_run_full_workflow_paths_add_timestamp_to_unprefixed_task(monkeypatch):
    monkeypatch.setattr(run_full_workflow.time, "strftime", lambda fmt: "20260617_104500")

    paths = run_full_workflow.make_paths("unit workflow")

    assert paths.root.name == "20260617_104500_unit_workflow"


def test_run_full_workflow_paths_keep_existing_timestamp_prefix(monkeypatch):
    monkeypatch.setattr(run_full_workflow.time, "strftime", lambda fmt: "20260617_104500")

    paths = run_full_workflow.make_paths("20260615_094437_o10")

    assert paths.root.name == "20260615_094437_o10"


def test_run_full_workflow_operating_point_uses_opt_in_scorer_metadata():
    results = [
        {
            "boundary_signature": {
                "operating_point": "qwen-mamba2-frame-boundary-scorer-v4",
                "scorer_checkpoint": {
                    "schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_v4",
                    "metadata": {"operating_point": "qwen-mamba2-frame-boundary-scorer-synthetic-v3"},
                },
            }
        }
    ]

    assert (
        run_full_workflow.speech_boundary_operating_point(results)
        == "qwen-mamba2-frame-boundary-scorer-synthetic-v3"
    )


def test_run_full_workflow_operating_point_defaults_without_scorer():
    assert run_full_workflow.speech_boundary_operating_point([]) == "qwen-mamba2-frame-boundary-scorer-v4"
    assert (
        run_full_workflow.speech_boundary_operating_point(
            [{"boundary_signature": {"operating_point": "qwen-mamba2-frame-boundary-scorer-v4"}}]
        )
        == "qwen-mamba2-frame-boundary-scorer-v4"
    )


def test_run_full_workflow_parse_args_uses_loaded_env(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame")
    monkeypatch.setenv("ASR_MODEL_PATH", "")
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32",
    )
    boundary_mapping = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=src/boundary/checkpoints/boundary_edge_refiner_v6.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    cueqc_mapping = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=src/asr/checkpoints/cueqc_mamba_v4_binary.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    scorer_mapping = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=agents/temp/scorer.pt"
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", boundary_mapping)
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", cueqc_mapping)
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", scorer_mapping)
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD", "0.6")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_DROP_GAP_THRESHOLD", "0.8")

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
    assert args.asr_batch_size == "auto"
    assert args.boundary_refiner_model_path_by_repo == boundary_mapping
    assert args.cueqc_model_path_by_repo == cueqc_mapping
    assert args.speech_boundary_scorer_checkpoint_by_repo == scorer_mapping
    assert args.speech_boundary_split_threshold == 0.6
    assert args.speech_boundary_drop_gap_threshold == 0.8
    assert args.speech_boundary_speech_on_threshold == args.speech_boundary_threshold
    assert args.speech_boundary_speech_off_threshold == args.speech_boundary_threshold


def test_run_full_workflow_context_carries_boundary_env(monkeypatch, tmp_path):
    batch_table = (
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=32,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8"
    )
    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", batch_table)
    boundary_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=src/boundary/checkpoints/boundary_edge_refiner_v6.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    cueqc_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=src/asr/checkpoints/cueqc_mamba_v4_binary.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    scorer_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=agents/temp/scorer.pt"
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", boundary_mapping)
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", cueqc_mapping)
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", scorer_mapping)
    monkeypatch.setenv("BOUNDARY_REFINER_DEVICE", "cpu")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD", "0.65")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_DROP_GAP_THRESHOLD", "0.85")

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--task-name",
            "unit",
            "--label",
            "boundary",
            "--speech-boundary-speech-on-threshold",
            "0.7",
            "--speech-boundary-speech-off-threshold",
            "0.5",
            "--speech-boundary-split-threshold",
            "0.7",
            "--speech-boundary-drop-gap-threshold",
            "0.9",
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
    assert ctx.advanced["BOUNDARY_REFINER_MODEL_PATH_BY_REPO"] == boundary_mapping
    assert ctx.advanced["CUEQC_MODEL_PATH_BY_REPO"] == cueqc_mapping
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO"] == scorer_mapping
    assert ctx.advanced["BOUNDARY_REFINER_DEVICE"] == "cpu"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD"] == "0.7"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_DROP_GAP_THRESHOLD"] == "0.9"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"] == "0.7"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"] == "0.5"


def test_run_full_workflow_cli_batch_overrides_loaded_env(monkeypatch):
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32",
    )

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--asr-backend",
            "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
            "--asr-batch-size",
            "12",
            "--speech-boundary-speech-on-threshold",
            "0.7",
            "--speech-boundary-speech-off-threshold",
            "0.5",
        ]
    )
    run_full_workflow.configure_env(args)

    assert run_full_workflow.os.environ["ASR_BACKEND"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
    assert run_full_workflow.os.environ["ASR_BATCH_SIZE"] == "12"
    assert run_full_workflow.os.environ["SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"] == "0.7"
    assert run_full_workflow.os.environ["SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"] == "0.5"
