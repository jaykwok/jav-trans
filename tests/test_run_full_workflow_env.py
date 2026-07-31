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


def test_run_full_workflow_reports_the_chunking_path_that_actually_ran():
    results = [
        {
            "boundary_signature": {
                "chunking": {
                    "schema": "blank_run_pregate_v1",
                    "source": "alignment_head_blank_runs",
                    "chunk_count": 42,
                }
            }
        }
    ]

    assert run_full_workflow.chunking_source(results) == "alignment_head_blank_runs"


def test_run_full_workflow_chunking_source_is_unknown_without_a_signature():
    # The fallbacks produce the same chunk shape as the head, so a report that
    # guessed would hide a head that silently failed to load.
    assert run_full_workflow.chunking_source([]) == "unknown"
    assert (
        run_full_workflow.chunking_source(
            [{"boundary_signature": {"chunking": {"source": "fixed_length_no_head"}}}]
        )
        == "fixed_length_no_head"
    )


def test_run_full_workflow_has_no_runtime_scorer_threshold_surface(monkeypatch):
    monkeypatch.setattr(run_full_workflow, "load_config", lambda: None)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_FRAME_DILATION_S", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", raising=False)

    args = run_full_workflow.parse_args(["--video", "sample.mp4"])

    assert not hasattr(args, "speech_boundary_threshold")
    assert not hasattr(args, "speech_boundary_speech_on_threshold")
    assert not hasattr(args, "speech_boundary_speech_off_threshold")
    assert not hasattr(args, "speech_boundary_frame_dilation_s")
    assert not hasattr(args, "speech_boundary_min_segment_s")
    assert not hasattr(args, "pre_asr_cueqc_drop_threshold")
    assert not hasattr(args, "cueqc_shadow_enabled")


def test_run_full_workflow_parse_args_uses_loaded_env(monkeypatch):
    monkeypatch.delenv("ASR_STAGE_WORKER_MODE", raising=False)
    monkeypatch.delenv("ASR_WORKER_MODE", raising=False)
    monkeypatch.setenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    monkeypatch.setenv("ASR_MODEL_PATH", "")
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4",
    )
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", "legacy=ignored")
    monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "1")
    monkeypatch.setenv("PRE_ASR_CUEQC_ENABLED", "1")

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

    # The single-model surface: no per-job backend choice survives on the CLI.
    assert not hasattr(args, "asr_backend")
    assert not hasattr(args, "asr_stage_worker_mode")
    assert not hasattr(args, "asr_worker_mode")
    assert args.asr_model_path == ""
    assert args.asr_batch_size == "auto"
    # Retired on 2026-07-31. Someone's shell may still export these; they must
    # not reappear as knobs for stages that no longer exist.
    for retired in (
        "outer_edge_refiner_model_path_by_repo",
        "semantic_split_model_path_by_repo",
        "cueqc_model_path_by_repo",
        "cueqc_shadow_enabled",
        "speech_boundary_scorer_checkpoint_by_repo",
        "speech_boundary_split_score_quantile",
        "speech_boundary_split_prominence_quantile",
        "pre_asr_cueqc_enabled",
        "boundary_feature_frame_hop_s",
        "boundary_cache",
    ):
        assert not hasattr(args, retired)
    assert not hasattr(args, "speech_boundary_threshold")
    assert not hasattr(args, "speech_boundary_speech_on_threshold")
    assert not hasattr(args, "speech_boundary_speech_off_threshold")


def test_run_full_workflow_context_drops_retired_boundary_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ASR_STAGE_WORKER_MODE", raising=False)
    monkeypatch.delenv("ASR_WORKER_MODE", raising=False)
    batch_table = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=8"
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv("ASR_BATCH_SIZE_BY_REPO", batch_table)
    monkeypatch.setenv("CUEQC_MODEL_PATH_BY_REPO", "legacy=ignored")
    monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "1")
    monkeypatch.setenv("OUTER_EDGE_REFINER_DEVICE", "cpu")
    monkeypatch.setenv("PRE_ASR_CUEQC_ENABLED", "1")

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
    assert "ASR_STAGE_WORKER_MODE" not in ctx.advanced
    assert "ASR_WORKER_MODE" not in ctx.advanced
    assert ctx.advanced["ASR_BATCH_SIZE_BY_REPO"] == batch_table
    # Even with the old vars exported in the shell, the run context must not
    # carry settings for stages that are gone.
    for retired in (
        "ASR_BOUNDARY_BACKEND",
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        "CUEQC_MODEL_PATH_BY_REPO",
        "CUEQC_SHADOW_ENABLED",
        "CUEQC_INFERENCE_BATCH_SIZE",
        "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        "OUTER_EDGE_REFINER_DEVICE",
        "PRE_ASR_CUEQC_ENABLED",
        "BOUNDARY_CACHE_ENABLED",
        "BOUNDARY_CACHE_DIR",
    ):
        assert retired not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_MIN_PRIMARY_SCORE" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_DENSE_CUT_GAP_S" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_THRESHOLD" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_FRAME_DILATION_S" not in ctx.advanced
    assert "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S" not in ctx.advanced


def test_run_full_workflow_cli_batch_overrides_loaded_env(monkeypatch):
    monkeypatch.delenv("ASR_STAGE_WORKER_MODE", raising=False)
    monkeypatch.delenv("ASR_WORKER_MODE", raising=False)
    monkeypatch.setenv("ASR_BATCH_SIZE", "auto")
    monkeypatch.setenv(
        "ASR_BATCH_SIZE_BY_REPO",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4",
    )
    monkeypatch.delenv("CUEQC_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("CUEQC_MODEL_PATH_BY_REPO", raising=False)
    monkeypatch.delenv("CUEQC_INFERENCE_BATCH_SIZE", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_FRAME_DILATION_S", raising=False)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", raising=False)
    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--asr-batch-size",
            "12",
        ]
    )
    run_full_workflow.configure_env(args)

    assert "ASR_STAGE_WORKER_MODE" not in run_full_workflow.os.environ
    assert "ASR_WORKER_MODE" not in run_full_workflow.os.environ
    assert run_full_workflow.os.environ["ASR_BATCH_SIZE"] == "12"
    assert not hasattr(args, "cueqc_shadow_enabled")
    assert not hasattr(args, "cueqc_model_path_by_repo")
    assert not hasattr(args, "cueqc_inference_batch_size")
    # `configure_env` no longer exports anything for the retired chain. It used
    # to also pop a list of legacy split controls; nothing reads them now, so
    # the guarantee that matters is that none are written.
    assert not any(
        name.startswith(("SPEECH_BOUNDARY_", "PRE_ASR_CUEQC_", "OUTER_EDGE_", "SEMANTIC_SPLIT_"))
        for name in run_full_workflow.configure_env.__code__.co_consts
        if isinstance(name, str)
    )
    assert "SPEECH_BOUNDARY_JA_THRESHOLD" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_FRAME_DILATION_S" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_SPLIT_MIN_PRIMARY_SCORE" not in run_full_workflow.os.environ
    assert "SPEECH_BOUNDARY_JA_DENSE_CUT_GAP_S" not in run_full_workflow.os.environ


def test_run_full_workflow_summary_reports_chunking_not_a_retired_chain(tmp_path):
    args = run_full_workflow.parse_args(["--video", "sample.mp4"])
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
    results = [
        {"boundary_signature": {"chunking": {"source": "alignment_head_blank_runs"}}}
    ]

    run_full_workflow.write_summary(paths, args, results)

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["chunking_source"] == "alignment_head_blank_runs"
    for retired in (
        "pre_asr_cueqc_enabled",
        "cueqc_enabled",
        "cueqc_shadow_enabled",
        "boundary_backend",
        "boundary_planner",
        "speech_boundary_operating_point",
        "speech_boundary_decision_mode",
        "speech_boundary_runtime_threshold",
        "speech_boundary_split_strategy",
        "speech_boundary_scorer_checkpoint_by_repo",
    ):
        assert retired not in payload
    markdown = paths.summary_md.read_text(encoding="utf-8")
    assert "alignment_head_blank_runs" in markdown
    # A report that still named the retired chain would be read as evidence
    # that it ran.
    assert "Pre-ASR CueQC" not in markdown
    assert "candidate-island" not in markdown


def test_removed_scorer_runtime_env_is_absent_from_active_surfaces():
    removed_scorer_runtime = (
        "SPEECH_BOUNDARY_JA_THRESHOLD",
        "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD",
        "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD",
        "SPEECH_BOUNDARY_JA_FRAME_DILATION_S",
        "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S",
    )
    active_paths = [
        run_full_workflow.PROJECT_ROOT / "src" / "asr" / "pregate.py",
        run_full_workflow.PROJECT_ROOT / "src" / "core" / "config.py",
        run_full_workflow.PROJECT_ROOT / "src" / "web" / "static" / "index.html",
        run_full_workflow.PROJECT_ROOT / "tools" / "workflows" / "run_full_workflow.py",
    ]
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        for key in removed_scorer_runtime:
            pattern = re.compile(rf"(?<![A-Z0-9_]){re.escape(key)}(?![A-Z0-9_])")
            assert not pattern.search(text), path


def test_removed_split_env_is_absent_from_current_and_legacy_runtime_files():
    removed_split = (
        "SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD",
        "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE",
        "SPEECH_BOUNDARY_JA_SPLIT_TARGET_S",
    )
    checked_paths = [
        run_full_workflow.PROJECT_ROOT / "src" / "asr" / "pregate.py",
        run_full_workflow.PROJECT_ROOT / "src" / "web" / "static" / "index.html",
        run_full_workflow.PROJECT_ROOT / "tools" / "workflows" / "run_full_workflow.py",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for key in removed_split:
            pattern = re.compile(rf"(?<![A-Z0-9_]){re.escape(key)}(?![A-Z0-9_])")
            assert not pattern.search(text), path
