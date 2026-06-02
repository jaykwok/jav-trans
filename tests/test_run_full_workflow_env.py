from pathlib import Path

from tools.fusionvad_ja import run_full_workflow


def test_run_full_workflow_context_carries_pre_asr_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_PRE_ASR_CUT_SPLIT_ENABLED", "1")
    monkeypatch.setenv("ASR_PRE_ASR_CUT_SPLIT_THRESHOLD", "0.95")
    monkeypatch.setenv("ASR_PRE_ASR_CUT_SPLIT_MIN_CORE_FRAMES", "420")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_ENABLED", "0")
    monkeypatch.setenv("ASR_PRE_ASR_RISK_SPLIT_ENABLED", "1")
    monkeypatch.setenv("ASR_PRE_ASR_RISK_SPLIT_THRESHOLD", "1.5")
    monkeypatch.setenv("ASR_PRE_ASR_RISK_SPLIT_CONTINUOUS_THRESHOLD", "2.5")
    monkeypatch.setenv("ASR_PRE_ASR_RISK_SPLIT_TARGET_CORE_FRAMES", "240")

    args = run_full_workflow.parse_args(
        [
            "--video",
            "sample.mp4",
            "--task-name",
            "unit",
            "--label",
            "r17",
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

    assert ctx.advanced["ASR_PRE_ASR_CUT_SPLIT_ENABLED"] == "1"
    assert ctx.advanced["ASR_PRE_ASR_CUT_SPLIT_THRESHOLD"] == "0.95"
    assert ctx.advanced["ASR_PRE_ASR_CUT_SPLIT_MIN_CORE_FRAMES"] == "420"
    assert ctx.advanced["ASR_PRE_ASR_VALLEY_SPLIT_ENABLED"] == "0"
    assert ctx.advanced["ASR_PRE_ASR_CUT_SPLIT_TARGET_CORE_FRAMES"] == "270"
    assert ctx.advanced["ASR_PRE_ASR_RISK_SPLIT_ENABLED"] == "1"
    assert ctx.advanced["ASR_PRE_ASR_RISK_SPLIT_THRESHOLD"] == "1.5"
    assert ctx.advanced["ASR_PRE_ASR_RISK_SPLIT_CONTINUOUS_THRESHOLD"] == "2.5"
    assert ctx.advanced["ASR_PRE_ASR_RISK_SPLIT_TARGET_CORE_FRAMES"] == "240"
