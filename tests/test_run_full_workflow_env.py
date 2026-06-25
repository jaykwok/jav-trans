import re
import json

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
                "operating_point": "qwen-mamba2-frame-boundary-scorer-v6",
                "scorer_checkpoint": {
                    "schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_v6",
                    "metadata": {"operating_point": "qwen-mamba2-frame-boundary-scorer-v6-native"},
                },
            }
        }
    ]

    assert (
        run_full_workflow.speech_boundary_operating_point(results)
        == "qwen-mamba2-frame-boundary-scorer-v6-native"
    )


def test_run_full_workflow_operating_point_defaults_without_scorer():
    assert run_full_workflow.speech_boundary_operating_point([]) == "qwen-mamba2-frame-boundary-scorer-v6"
    assert (
        run_full_workflow.speech_boundary_operating_point(
            [{"boundary_signature": {"operating_point": "qwen-mamba2-frame-boundary-scorer-v6"}}]
        )
        == "qwen-mamba2-frame-boundary-scorer-v6"
    )


def test_run_full_workflow_boundary_threshold_defaults_match_training_eval(monkeypatch):
    monkeypatch.setattr(run_full_workflow, "load_config", lambda: None)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD", raising=False)
    monkeypatch.delenv("CUEQC_SHADOW_ENABLED", raising=False)

    args = run_full_workflow.parse_args(["--video", "sample.mp4"])

    assert args.speech_boundary_threshold == 0.5
    assert args.speech_boundary_speech_on_threshold == 0.5
    assert args.speech_boundary_speech_off_threshold == 0.5
    assert args.cueqc_shadow_enabled is False


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
    monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "0")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", scorer_mapping)
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD", "0.6")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE", "0.4")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE", "0.6")
    monkeypatch.setenv("PRE_ASR_CUEQC_ENABLED", "1")
    monkeypatch.setenv("PRE_ASR_CUEQC_DROP_THRESHOLD", "0.88")

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
    assert args.cueqc_shadow_enabled is False
    assert args.speech_boundary_scorer_checkpoint_by_repo == scorer_mapping
    assert args.speech_boundary_split_score_quantile == 0.4
    assert args.speech_boundary_split_prominence_quantile == 0.6
    assert args.pre_asr_cueqc_enabled is True
    assert args.pre_asr_cueqc_drop_threshold == 0.88
    assert args.speech_boundary_speech_on_threshold == args.speech_boundary_threshold
    assert args.speech_boundary_speech_off_threshold == args.speech_boundary_threshold


def test_run_full_workflow_context_carries_boundary_env(monkeypatch, tmp_path):
    batch_table = (
        "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=32,"
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8"
    )
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", batch_table)
    boundary_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=src/boundary/checkpoints/boundary_edge_refiner_v6.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    cueqc_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=src/asr/checkpoints/cueqc_mamba_v4_binary.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    scorer_mapping = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=agents/temp/scorer.pt"
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH_BY_REPO", boundary_mapping)
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", cueqc_mapping)
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", scorer_mapping)
    monkeypatch.setenv("BOUNDARY_REFINER_DEVICE", "cpu")
    monkeypatch.setenv("PRE_ASR_CUEQC_ENABLED", "1")

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
            "--speech-boundary-split-score-quantile",
            "0.45",
            "--speech-boundary-split-prominence-quantile",
            "0.55",
            "--pre-asr-cueqc-drop-threshold",
            "0.9",
            "--no-cueqc-shadow-enabled",
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
    assert ctx.advanced["CUEQC_SHADOW_ENABLED"] == "0"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO"] == scorer_mapping
    assert ctx.advanced["BOUNDARY_REFINER_DEVICE"] == "cpu"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE"] == "0.45"
    assert ctx.advanced["SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE"] == "0.55"
    assert "SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE" not in ctx.advanced
    assert ctx.advanced["PRE_ASR_CUEQC_ENABLED"] == "1"
    assert ctx.advanced["PRE_ASR_CUEQC_DROP_THRESHOLD"] == "0.9"
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
            "--no-cueqc-shadow-enabled",
        ]
    )
    run_full_workflow.configure_env(args)

    assert run_full_workflow.os.environ["ASR_BACKEND"] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
    assert run_full_workflow.os.environ["ASR_BATCH_SIZE"] == "12"
    assert run_full_workflow.os.environ["CUEQC_SHADOW_ENABLED"] == "0"
    assert run_full_workflow.os.environ["SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"] == "0.7"
    assert run_full_workflow.os.environ["SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"] == "0.5"


def test_run_full_workflow_summary_uses_shadow_specific_cueqc_keys(tmp_path):
    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--pre-asr-cueqc-enabled",
            "--cueqc-shadow-enabled",
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
    paths.root.mkdir(parents=True, exist_ok=True)
    results = [{"boundary_signature": {"operating_point": "qwen-mamba2-frame-boundary-scorer-v6"}}]

    run_full_workflow.write_summary(paths, args, results)

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["pre_asr_cueqc_enabled"] is True
    assert payload["cueqc_shadow_enabled"] is True
    assert "cueqc_enabled" not in payload
    markdown = paths.summary_md.read_text(encoding="utf-8")
    assert "CueQC shadow" in markdown


def test_removed_split_env_is_absent_from_current_runtime_files():
    removed = (
        "SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD",
        "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE",
        "SPEECH_BOUNDARY_JA_SPLIT_TARGET_S",
    )
    checked_paths = [
        run_full_workflow.PROJECT_ROOT / ".env.example",
        run_full_workflow.PROJECT_ROOT / "src" / "boundary" / "cache.py",
        run_full_workflow.PROJECT_ROOT / "src" / "boundary" / "ja" / "backend.py",
        run_full_workflow.PROJECT_ROOT / "src" / "web" / "static" / "index.html",
        run_full_workflow.PROJECT_ROOT / "tools" / "boundary" / "build_refiner_frame_sequence_dataset.py",
        run_full_workflow.PROJECT_ROOT / "tools" / "boundary" / "ja" / "export_frame_scores.py",
        run_full_workflow.PROJECT_ROOT / "tools" / "workflows" / "run_full_workflow.py",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for key in removed:
            pattern = re.compile(rf"(?<![A-Z0-9_]){re.escape(key)}(?![A-Z0-9_])")
            assert not pattern.search(text), path
