from __future__ import annotations

from pathlib import Path

from core.job_context import JobContext


def resolve_output_dir_for_ctx(
    video_path: str,
    ctx: JobContext,
    *,
    project_root: Path,
) -> str:
    if ctx.output_dir:
        output_path = Path(ctx.output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = Path(video_path).expanduser()
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path = output_path.resolve().parent
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path.resolve())


def resolve_subtitle_bilingual_for_ctx(ctx: JobContext) -> bool:
    return ctx.subtitle_mode.strip().lower() in {
        "bilingual",
        "dual",
        "ja_zh",
        "ja-zh",
        "2",
    }
