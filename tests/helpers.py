from __future__ import annotations

from pathlib import Path

import main
from core.job_context import JobContext


def make_job_context(
    video_path: Path | str,
    output_dir: Path | str | None,
    job_temp_root: Path | str,
    *,
    job_id: str | None = None,
    asr_backend: str = "anime-whisper",
    asr_context: str = "",
    subtitle_mode: str = "zh",
    skip_translation: bool = False,
    vad_threshold: float = 0.35,
    translation_batch_size: int = 200,
    translation_max_workers: int = 4,
    translation_cache_path: Path | str | None = None,
    keep_temp_files: bool = True,
    keep_quality_report: bool = False,
    run_log_enabled: bool = False,
    run_log_dir: Path | str = "./temp/log",
    fail_on_qc_block: bool = True,
    advanced: dict[str, str] | None = None,
) -> JobContext:
    video = Path(video_path)
    effective_job_id = main._sanitize_job_id(job_id or video.stem)
    job_temp_dir = Path(job_temp_root) / effective_job_id
    return JobContext(
        asr_backend=asr_backend,
        asr_context=asr_context,
        subtitle_mode=subtitle_mode,
        show_gender=True,
        multi_cue_split=True,
        asr_recovery=False,
        vad_threshold=vad_threshold,
        skip_translation=skip_translation,
        target_lang="简体中文",
        translation_glossary="",
        translation_batch_size=translation_batch_size,
        translation_max_workers=translation_max_workers,
        translation_cache_path=str(translation_cache_path or ""),
        job_id=effective_job_id,
        job_temp_dir=str(job_temp_dir),
        output_dir=str(output_dir) if output_dir is not None else None,
        keep_quality_report=keep_quality_report,
        keep_temp_files=keep_temp_files,
        run_log_enabled=run_log_enabled,
        run_log_dir=str(run_log_dir),
        fail_on_qc_block=fail_on_qc_block,
        advanced=dict(advanced or {}),
    )


def run_pipeline(
    video_path: Path | str,
    ctx: JobContext,
    *,
    cache_job_id: str = "",
):
    artifacts = main.run_asr_alignment_f0(
        str(video_path),
        ctx=ctx,
        job_id=ctx.job_id,
        cache_job_id=cache_job_id,
    )
    try:
        return main.run_translation_and_write(
            str(video_path),
            artifacts,
            ctx=ctx,
            job_id=ctx.job_id,
        )
    finally:
        main._close_run_logger(artifacts.logger)
