from pathlib import Path

import main
from helpers import make_job_context


def test_old_cli_runtime_helpers_removed():
    assert not hasattr(main, "_parse_args")
    assert not hasattr(main, "_apply_cli_env")
    assert not hasattr(main, "_build_runtime_context")
    assert not hasattr(main, "RuntimeContext")
    assert not hasattr(main, "_process_video")


def test_resolve_output_dir_defaults_to_video_folder(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "clip.mp4"
    video_path.write_bytes(b"fake")
    ctx = make_job_context(video_path, None, tmp_path / "jobs")

    assert Path(main._resolve_output_dir_for_ctx(str(video_path), ctx)) == video_dir


def test_resolve_output_dir_uses_job_context_output_dir(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "subs"
    ctx = make_job_context(video_path, output_dir, tmp_path / "jobs")

    assert Path(main._resolve_output_dir_for_ctx(str(video_path), ctx)) == output_dir
    assert output_dir.is_dir()


def test_subtitle_mode_comes_from_job_context(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    zh_ctx = make_job_context(video_path, tmp_path / "out", tmp_path / "jobs", subtitle_mode="zh")
    bilingual_ctx = make_job_context(
        video_path,
        tmp_path / "out",
        tmp_path / "jobs",
        subtitle_mode="bilingual",
    )

    assert main._resolve_subtitle_bilingual_for_ctx(zh_ctx) is False
    assert main._resolve_subtitle_bilingual_for_ctx(bilingual_ctx) is True
