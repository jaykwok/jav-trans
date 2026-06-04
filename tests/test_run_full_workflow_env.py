from pathlib import Path

from tools.boundary.ja import run_full_workflow


def test_run_full_workflow_context_carries_boundary_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BOUNDARY_REFINER_ENABLED", "1")
    monkeypatch.setenv("BOUNDARY_REFINER_THRESHOLD", "0.62")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "8.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CHUNK_S", "28.0")

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

    assert ctx.advanced["BOUNDARY_REFINER_ENABLED"] == "1"
    assert ctx.advanced["BOUNDARY_REFINER_THRESHOLD"] == "0.62"
    assert ctx.advanced["BOUNDARY_PLANNER_TARGET_CHUNK_S"] == "8.0"
    assert ctx.advanced["BOUNDARY_PLANNER_MAX_CHUNK_S"] == "28.0"
