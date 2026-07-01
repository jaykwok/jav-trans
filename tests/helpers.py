from __future__ import annotations

from pathlib import Path

import main
from core.job_context import JobContext
from pipeline.ids import sanitize_job_id

ASR_06B_BACKEND = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
ASR_17B_BACKEND = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


def make_job_context(
    video_path: Path | str,
    output_dir: Path | str | None,
    job_temp_root: Path | str,
    *,
    job_id: str | None = None,
    asr_backend: str = ASR_17B_BACKEND,
    subtitle_mode: str = "zh",
    skip_translation: bool = False,
    translation_max_workers: int = 4,
    translation_cache_path: Path | str | None = None,
    keep_temp_files: bool = True,
    keep_quality_report: bool = False,
    run_log_enabled: bool = False,
    run_log_dir: Path | str = "./tmp/log",
    advanced: dict[str, str] | None = None,
) -> JobContext:
    video = Path(video_path)
    effective_job_id = sanitize_job_id(job_id or video.stem)
    job_temp_dir = Path(job_temp_root) / effective_job_id
    return JobContext(
        asr_backend=asr_backend,
        subtitle_mode=subtitle_mode,
        skip_translation=skip_translation,
        target_lang="简体中文",
        translation_glossary="",
        translation_max_workers=translation_max_workers,
        translation_cache_path=str(translation_cache_path or ""),
        job_id=effective_job_id,
        job_temp_dir=str(job_temp_dir),
        output_dir=str(output_dir) if output_dir is not None else None,
        keep_quality_report=keep_quality_report,
        keep_temp_files=keep_temp_files,
        run_log_enabled=run_log_enabled,
        run_log_dir=str(run_log_dir),
        advanced=dict(advanced or {}),
    )


def run_pipeline(
    video_path: Path | str,
    ctx: JobContext,
    *,
    cache_job_id: str = "",
):
    artifacts = main.run_asr_alignment(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
        cache_job_id=cache_job_id,
    )
    return main.run_translation_and_write(
        str(video_path),
        artifacts,
        ctx=ctx,
        job_id=ctx.job_id,
    )
