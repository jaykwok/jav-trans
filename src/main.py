import os
import json
import logging
import time
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from core import events
from core.config import load_config
from core.job_context import JobContext
from asr import noise as asr_noise_module
from pipeline import aligned_cache as aligned_cache_module
from pipeline import audio as audio_module
from pipeline import cleanup as cleanup_module
from pipeline import output as output_module
from pipeline import output_writer as output_writer_module
from pipeline import quality as quality_module
from pipeline import stage_log as stage_log_module
from pipeline.artifacts import AsrArtifacts
from pipeline.ids import sanitize_job_id
from utils.model_paths import PROJECT_ROOT


_ENV_OVERRIDE_KEYS = (
    "JOB_TEMP_DIR",
    "ASR_BACKEND",
    "ASR_BOUNDARY_BACKEND",
    "ASR_WORKER_MODE",
    "ASR_MODEL_PATH",
    "ASR_MODEL_ID",
    "ALIGNER_MODEL_PATH",
    "ALIGNER_MODEL_ID",
    "API_KEY",
    "OPENAI_COMPATIBILITY_BASE_URL",
    "QUALITY_REPORT_ENABLED",
    "QUALITY_REPORT_DIR",
    "QC_HARD_FAIL",
    "_TEST_CRASH_TRANSLATION_BATCH",
)
_ENV_OVERRIDES = {
    key: os.environ[key]
    for key in _ENV_OVERRIDE_KEYS
    if os.getenv(key)
}
load_config()
os.environ.update(_ENV_OVERRIDES)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.path.abspath(os.getenv("HF_HOME"))
if os.getenv("TORCH_HOME"):
    os.environ["TORCH_HOME"] = os.path.abspath(os.getenv("TORCH_HOME"))
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

import torch

from asr import pipeline as asr_module
from subtitles import writer as subtitle_module
from llm import translator as translator_module
from asr.qc import asr_qc_gate, build_asr_manifest

from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

console = Console(force_terminal=False, emoji=False)
_ASR_PROGRESS_RE = stage_log_module._ASR_PROGRESS_RE


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _ctx_value(ctx: JobContext, name: str, default: str = "") -> str:
    raw = ctx.advanced.get(name)
    if raw is not None:
        return raw.strip()
    return os.getenv(name, default).strip()


def _ctx_float(ctx: JobContext, name: str, default: float) -> float:
    try:
        return float(_ctx_value(ctx, name, str(default)))
    except (TypeError, ValueError):
        return default


def _ctx_int(ctx: JobContext, name: str, default: int) -> int:
    try:
        return int(float(_ctx_value(ctx, name, str(default))))
    except (TypeError, ValueError):
        return default


def _ctx_flag(ctx: JobContext, name: str, default: bool = False) -> bool:
    raw = ctx.advanced.get(name)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if name == "SKIP_TRANSLATION":
        return ctx.skip_translation
    if name == "QUALITY_REPORT_ENABLED":
        return ctx.keep_quality_report
    return default


def _ctx_env_flag(ctx: JobContext, name: str, default: bool = False) -> bool:
    raw = ctx.advanced.get(name)
    if raw is None:
        raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_ASR_STAGE_ADVANCED_PREFIXES = (
    "ASR_QC_",
    "ASR_FRAGMENT_",
    "ASR_NATIVE_",
    "BOUNDARY_",
    "SPEECH_BOUNDARY_JA_",
    "ALIGNMENT_",
)
_ASR_STAGE_ADVANCED_KEYS = {
    "ALIGN_LONG_CHUNK_BATCH_SIZE",
    "ASR_BOUNDARY_BACKEND",
    "ASR_LANGUAGE",
    "ASR_FORCE_LANGUAGE",
    "ASR_MODEL_PATH",
    "ASR_MODEL_ID",
    "ASR_DTYPE",
    "ASR_ATTENTION",
    "ASR_HEAD_CONTEXT",
    "ASR_HEAD_CONTEXT_MAX_START_S",
    "ASR_MAX_NEW_TOKENS",
    "ASR_REPETITION_PENALTY",
    "ASR_WORKER_MODE",
    "ASR_CHUNK_MIN_DURATION_S",
    "ASR_CONTEXT_RESET_GAP_S",
    "ASR_SLIDING_CONTEXT_SEGS",
    "ASR_INITIAL_PROMPT_MAX_CHARS",
    "ASR_INITIAL_PROMPT_MAX_TOKENS",
    "ASR_MIN_EFFECTIVE_NEW_TOKENS",
    "TRANSCRIPTION_TIMEOUT_S",
    "TRANSCRIPTION_MAX_NEW_TOKENS",
    "ASR_BATCH_SIZE",
    "ASR_BATCH_SIZE_BY_REPO",
    "ALIGNER_BATCH_SIZE",
    "KEEP_ASR_CHUNKS",
    "BOUNDARY_CACHE_DIR",
    "BOUNDARY_CACHE_ENABLED",
}
_ASR_STAGE_CACHE_NEUTRAL_KEYS = {
    "BOUNDARY_CACHE_DIR",
    "BOUNDARY_CACHE_ENABLED",
}


def _is_asr_stage_advanced_key(name: str) -> bool:
    return (
        name in _ASR_STAGE_ADVANCED_KEYS
        or any(name.startswith(prefix) for prefix in _ASR_STAGE_ADVANCED_PREFIXES)
    )


def _asr_stage_env_overrides(ctx: JobContext) -> dict[str, str]:
    overrides = {
        str(key).strip(): str(value)
        for key, value in (ctx.advanced or {}).items()
        if str(key).strip() and _is_asr_stage_advanced_key(str(key).strip())
    }
    asr_backend = str(ctx.asr_backend or "").strip()
    if asr_backend:
        overrides["ASR_BACKEND"] = asr_backend
    overrides["ASR_CONTEXT"] = str(ctx.asr_context or "")
    return overrides


def _asr_stage_env_for_video(ctx: JobContext, video_fps: float | None) -> dict[str, str]:
    del video_fps
    overrides = _asr_stage_env_overrides(ctx)
    return overrides


_SUBTITLE_OPTION_KEYS = {
    "MAX_SUBTITLE_DURATION",
    "SUBTITLE_SOFT_MAX_S",
    "SUBTITLE_SOFT_SPLIT_ENABLED",
    "MIN_SUBTITLE_DURATION",
    "SUBTITLE_MIN_DURATION",
    "SUBTITLE_READING_CPS",
    "SUBTITLE_READING_BASE",
    "SUBTITLE_DURATION_RATIO_CAP",
    "SUBTITLE_DURATION_GRACE",
    "SUBTITLE_TIMELINE_MODE",
    "SUBTITLE_BILINGUAL_SECONDARY_WEIGHT",
    "SUBTITLE_ASCII_CHAR_WEIGHT",
    "SRT_LINE_MAX_CHARS",
    "SUBTITLE_MERGE_ADJACENT",
    "SUBTITLE_TIMING_POLISH_ENABLED",
    "SUBTITLE_SHORT_GAP_COLLAPSE_S",
    "SUBTITLE_LINGER_S",
    "SUBTITLE_DENSE_CUE_MERGE_ENABLED",
    "SUBTITLE_DENSE_CUE_MERGE_MAX_GAP_FRAMES",
    "SUBTITLE_DENSE_CUE_MERGE_MAX_SINGLE_FRAMES",
    "SUBTITLE_DENSE_CUE_MERGE_MAX_COMBINED_FRAMES",
    "SUBTITLE_DENSE_CUE_MERGE_MAX_TEXT_UNITS",
}


def _subtitle_env_overrides(ctx: JobContext) -> dict[str, str]:
    return {
        str(key).strip(): str(value)
        for key, value in (ctx.advanced or {}).items()
        if str(key).strip() in _SUBTITLE_OPTION_KEYS
    }


def _subtitle_options_for_ctx(ctx: JobContext):
    with _temporary_env(_subtitle_env_overrides(ctx)):
        return subtitle_module.SubtitleOptions.from_env()


def _subtitle_options_for_video(ctx: JobContext, video_fps: float | None):
    return _subtitle_options_for_ctx(ctx).with_video_fps(video_fps)


def _asr_runtime_signature_for_env() -> dict:
    try:
        return dict(asr_module._get_asr_runtime_signature(last_boundary_signature={}))
    except Exception:
        return {}


def _asr_stage_config_signature_for_env() -> dict:
    names = {
        name
        for name in os.environ
        if _is_asr_stage_advanced_key(name)
        or name
        in {
            "ASR_BACKEND",
            "ASR_CONTEXT",
            "ASR_WORKER_MODE",
            "BOUNDARY_REFINER_ENABLED",
        }
    }
    return {
        name: os.getenv(name, "")
        for name in sorted(names)
        if name not in _ASR_STAGE_CACHE_NEUTRAL_KEYS
    }


def _aligned_cache_signature_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str,
    subtitle_options=None,
) -> dict:
    options = subtitle_options or _subtitle_options_for_ctx(ctx)
    asr_signature = _asr_runtime_signature_for_env()
    subtitle_signature = dict(options.signature())
    subtitle_signature["effective_video_fps"] = options.effective_video_fps
    subtitle_signature["frame_gap_s"] = options.frame_gap_s
    return {
        "version": 5,
        "backend_label": backend_label,
        "asr": asr_signature,
        "asr_stage_config": _asr_stage_config_signature_for_env(),
        "subtitle": subtitle_signature,
    }


def aligned_cache_expectations_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str | None = None,
    video_fps: float | None = None,
) -> tuple[str, dict]:
    with _temporary_env(_asr_stage_env_for_video(ctx, video_fps)):
        resolved_backend_label = backend_label or asr_module.get_backend_label()
        subtitle_options = _subtitle_options_for_video(ctx, video_fps)
        signature = _aligned_cache_signature_for_ctx(
            ctx,
            backend_label=resolved_backend_label,
            subtitle_options=subtitle_options,
        )
    return resolved_backend_label, signature


def _aligned_segments_payload(
    *,
    backend_label: str,
    audio_path: str,
    audio_cache_key: str,
    segments: list[dict],
    asr_details: dict,
    asr_log: list[str],
    cache_signature: dict | None,
    subtitle_options,
    cache_stage: str = "ready",
) -> dict:
    payload = {
        "backend": backend_label,
        "audio_path": audio_path,
        "audio_cache_key": audio_cache_key,
        "segments": segments,
        "asr_details": asr_details,
        "asr_log": asr_log,
        "timeline_mode": subtitle_options.timeline_mode,
        "cache_stage": cache_stage,
    }
    if cache_signature is not None:
        payload["cache_signature"] = cache_signature
    return payload


def _quality_report_dir_for_ctx(ctx: JobContext) -> Path | None:
    report_dir_raw = (ctx.advanced or {}).get("QUALITY_REPORT_DIR")
    if report_dir_raw is None or not str(report_dir_raw).strip():
        return None
    return Path(str(report_dir_raw).strip())


def _quality_hard_fail_for_ctx(ctx: JobContext) -> bool | None:
    hard_fail_raw = (ctx.advanced or {}).get("QC_HARD_FAIL")
    if hard_fail_raw is None:
        return None
    return str(hard_fail_raw).strip() == "1"


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PipelineCancelledError(RuntimeError):
    pass


def _cancel_requested(cancel_event) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event) -> None:
    if _cancel_requested(cancel_event):
        raise PipelineCancelledError("任务已取消")


def _run_log_dir(ctx: JobContext) -> Path:
    raw = (ctx.run_log_dir or "./temp/log").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _setup_run_logger(
    job_id: str,
    backend_label: str,
    ctx: JobContext,
) -> tuple[logging.Logger, Path]:
    log_dir = _run_log_dir(ctx)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_backend = sanitize_job_id(backend_label.replace(" ", "_"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stamp}_{job_id}_{safe_backend}.run.log"
    logger_name = f"javtrans.run.{job_id}.{stamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger, log_path


def _close_run_logger(logger: logging.Logger | None) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _clear_thread_run_logger(logger: logging.Logger | None) -> None:
    if logger is not None and getattr(events._thread_local, "run_logger", None) is logger:
        try:
            delattr(events._thread_local, "run_logger")
        except AttributeError:
            pass


def _close_artifacts_logger(artifacts: "AsrArtifacts") -> None:
    logger = artifacts.logger if isinstance(artifacts.logger, logging.Logger) else None
    _close_run_logger(logger)
    _clear_thread_run_logger(logger)
    artifacts.logger = None


_event_ts = stage_log_module._event_ts
_coerce_event_value = stage_log_module._coerce_event_value
_parse_stage_extra = stage_log_module._parse_stage_extra
_emit_stage_event = stage_log_module._emit_stage_event
_emit_stage_log_event = stage_log_module._emit_stage_log_event
_parse_asr_stage_event = stage_log_module._parse_asr_stage_event
_log_stage = stage_log_module._log_stage


def _project_relative(path: str | Path | None) -> str | None:
    return output_writer_module.project_relative(path, project_root=PROJECT_ROOT)


def _project_relative_required(path: str | Path) -> str:
    return _project_relative(path) or ""


_log_timing_snapshot = stage_log_module._log_timing_snapshot


def _resolve_job_temp_dir(job_id: str) -> str:
    root = os.getenv("JOB_TEMP_DIR", "").strip()
    if not root:
        root = os.path.join("temp", "jobs")
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str((path / sanitize_job_id(job_id)).resolve())


def _asr_checkpoint_root() -> Path:
    try:
        return Path(
            getattr(asr_module, "_ASR_CHUNK_ROOT", Path("temp") / "chunks")
        ).resolve().parent
    except Exception:
        return Path("temp")


def _resolve_project_runtime_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _cleanup_pipeline_temp(job_temp_dir: str, audio_path: str, translation_cache_path: str = "") -> None:
    cleanup_module.cleanup_asr_checkpoint_for_audio(
        audio_path,
        asr_module._get_asr_checkpoint_path,
    )
    cleanup_module.cleanup_job_temp(
        job_temp_dir,
        translation_cache_path,
        checkpoint_root=_asr_checkpoint_root(),
    )
    cleanup_module.cleanup_runtime_ephemeral_temp(
        job_temp_root=os.getenv("JOB_TEMP_DIR", "temp/jobs") or "temp/jobs",
        asr_chunk_root=getattr(asr_module, "_ASR_CHUNK_ROOT", Path("temp") / "chunks"),
        project_root=PROJECT_ROOT,
    )


_format_asr_stage_label = stage_log_module._format_asr_stage_label


def _write_json(path: str, payload: dict) -> None:
    output_writer_module.write_json(path, payload, project_root=PROJECT_ROOT)


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    output_writer_module.write_json_atomic(path, payload, project_root=PROJECT_ROOT)


def _relativize_payload_paths(value):
    return output_writer_module.relativize_payload_paths(
        value,
        project_root=PROJECT_ROOT,
    )


def _timings_payload(
    *,
    video_path: str,
    audio_path: str,
    audio_cached: bool,
    job_id: str,
    job_temp_dir: str,
    device: str,
    backend: str,
    counts: dict,
    stage_timings: dict,
    asr_details: dict,
    translation_request_timings: list[dict],
    translation_api_retry_events: list[dict],
    outputs: dict,
    asr_log: list[str],
    asr_qc_blocked: bool | None = None,
) -> dict:
    return output_writer_module.timings_payload(
        video_path=video_path,
        audio_path=audio_path,
        audio_cached=audio_cached,
        job_id=job_id,
        job_temp_dir=job_temp_dir,
        device=device,
        backend=backend,
        counts=counts,
        stage_timings=stage_timings,
        asr_details=asr_details,
        translation_request_timings=translation_request_timings,
        translation_api_retry_events=translation_api_retry_events,
        outputs=outputs,
        asr_log=asr_log,
        asr_qc_blocked=asr_qc_blocked,
    )


def _write_quality_report_for_ctx(
    *,
    ctx: JobContext,
    video_stem: str,
    job_temp_dir: str,
    aligned_segments: list[dict],
    asr_details: dict,
    video_duration_s: float | None = None,
    enabled: bool | None = None,
    glossary: str | None = None,
) -> str | None:
    return quality_module.write_quality_report(
        video_stem=video_stem,
        job_temp_dir=job_temp_dir,
        aligned_segments=aligned_segments,
        asr_details=asr_details,
        project_root=PROJECT_ROOT,
        console=console,
        write_json_atomic=_write_json_atomic,
        env_flag=_env_flag,
        video_duration_s=video_duration_s,
        enabled=enabled,
        glossary=glossary,
        report_dir=_quality_report_dir_for_ctx(ctx),
        hard_fail=_quality_hard_fail_for_ctx(ctx),
    )


def _build_japanese_srt_blocks(segments: list[dict]) -> list[dict]:
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "ja_text": str(seg.get("text", "")),
            "zh_text": str(seg.get("text", "")),
            "words": list(seg.get("words") or []),
            "source_segment_ids": list(seg.get("source_segment_ids") or [idx]),
        }
        for idx, seg in enumerate(segments)
    ]


def _prepare_translation_cues(
    segments: list[dict],
    *,
    subtitle_options,
    bilingual: bool,
) -> list[dict]:
    source_blocks = _build_japanese_srt_blocks(segments)
    mode = "bilingual" if bilingual else "srt"
    cues = subtitle_module.prepare_srt_blocks(
        source_blocks,
        options=subtitle_options,
        mode=mode,
    )
    normalized: list[dict] = []
    for cue_id, cue in enumerate(cues):
        item = dict(cue)
        ja_text = str(item.get("ja_text") or item.get("text") or "").strip()
        item["cue_id"] = cue_id
        item["text"] = ja_text
        item["ja_text"] = ja_text
        item["zh_text"] = str(item.get("zh_text") or ja_text)
        item["words"] = list(item.get("words") or [])
        if "source_segment_ids" not in item:
            item["source_segment_ids"] = [cue_id]
        normalized.append(item)
    return normalized


def _print_timing_summary(stage_timings: dict, asr_details: dict) -> None:
    stage_log_module._print_timing_summary(console, stage_timings, asr_details)


def run_asr_alignment(
    video_path: str,
    *,
    ctx: JobContext,
    job_id: str = "",
    cache_job_id: str = "",
    cancel_event=None,
) -> AsrArtifacts:
    logger: logging.Logger | None = None
    run_log_path: Path | None = None
    pipeline_started = time.perf_counter()
    pipeline_timings: dict[str, float] = {}

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_filename = video_name
    job_id = sanitize_job_id(job_id or ctx.job_id or video_name)
    cache_job_id = sanitize_job_id(cache_job_id or job_id)
    effective_ctx = ctx
    events._thread_local.video = os.path.basename(video_path)
    events.set_current_job_id(job_id)
    _raise_if_cancelled(cancel_event)
    video_duration_s = audio_module.probe_video_duration_s(video_path)
    video_fps = audio_module.probe_video_fps(video_path)
    with _temporary_env(_asr_stage_env_for_video(effective_ctx, video_fps)):
        backend_label = asr_module.get_backend_label()
        if effective_ctx.run_log_enabled:
            logger, run_log_path = _setup_run_logger(job_id, backend_label, effective_ctx)
            events._thread_local.run_logger = logger

        _log_stage(logger, f"run_start video={_project_relative(video_path)}")
        _log_stage(logger, f"backend={backend_label}")
        _log_stage(
            logger,
            f"skip_translation={effective_ctx.skip_translation}",
        )
        _log_stage(
            logger,
            f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}",
        )
        if torch.cuda.is_available():
            _log_stage(logger, f"cuda_device={torch.cuda.get_device_name(0)}")
        if run_log_path is not None:
            console.print(f"[dim]运行日志：{_project_relative(run_log_path)}[/dim]")

        subtitle_options = _subtitle_options_for_video(effective_ctx, video_fps)
        _log_stage(
            logger,
            f"video_fps={subtitle_options.effective_video_fps:.6f}"
            f"{' fallback=29.97' if video_fps is None else ''}",
        )
        _log_stage(
            logger,
            "boundary_feature_frame_hop_s="
            f"{os.environ.get('BOUNDARY_FEATURE_FRAME_HOP_S', '')}",
        )
        _log_stage(
            logger,
            "subtitle_frame_gap_s="
            f"{subtitle_options.frame_gap_s:.6f}",
        )
        if cache_job_id != job_id:
            job_temp_dir = _resolve_job_temp_dir(cache_job_id)
        elif effective_ctx.job_temp_dir:
            job_temp_dir = str(Path(effective_ctx.job_temp_dir).expanduser().resolve())
        else:
            job_temp_dir = _resolve_job_temp_dir(job_id)
        os.makedirs(job_temp_dir, exist_ok=True)
        _log_stage(logger, f"job_temp_dir={_project_relative(job_temp_dir)}")
        if cache_job_id != job_id:
            _log_stage(logger, f"resume_from_job_id={cache_job_id}")
        aligned_cache_signature = _aligned_cache_signature_for_ctx(
            effective_ctx,
            backend_label=backend_label,
            subtitle_options=subtitle_options,
        )
        audio_dir = os.path.join(job_temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_cache_key = audio_module.get_audio_cache_key(video_path)
        audio_path = os.path.join(audio_dir, f"{video_name}.{audio_cache_key}.wav")
        aligned_segments_path = os.path.join(
            job_temp_dir, f"{video_filename}.aligned_segments.json"
        )
        aligned_cache = aligned_cache_module.try_load_aligned_segments(
            aligned_segments_path,
            audio_cache_key,
            backend_label,
            aligned_cache_signature,
        )
        segments: list[dict] = []
        asr_log: list[str] = []
        asr_details: dict = {}

        # 1. Extract audio (skipped if cached)
        audio_prepare_started = time.perf_counter()
        audio_cached = os.path.exists(audio_path)
        _log_stage(logger, "stage_start audio_prepare")
        _raise_if_cancelled(cancel_event)
        if aligned_cache is not None:
            segments = aligned_cache["segments"]
            asr_log = [str(item) for item in aligned_cache.get("asr_log", [])]
            asr_details = dict(aligned_cache.get("asr_details", {}))
            audio_cached = True
            _log_stage(
                logger,
                f"aligned_cache_hit path={_project_relative(aligned_segments_path)}",
            )
            console.print(
                f"[green]命中 aligned_segments cache，跳过音频提取与 ASR：{_project_relative(aligned_segments_path)}[/green]"
            )
        elif os.path.exists(audio_path):
            _log_stage(logger, f"audio_cache_hit path={_project_relative(audio_path)}")
            console.print(f"[dim]跳过提取：使用已缓存音频 {_project_relative(audio_path)}[/dim]")
        else:
            _log_stage(
                logger,
                f"extract_audio_start output={_project_relative(audio_path)}",
            )
            with console.status("[cyan]提取音频中...[/cyan]"):
                _raise_if_cancelled(cancel_event)
                audio_module.extract_audio(video_path, audio_path)
                _raise_if_cancelled(cancel_event)
            _log_stage(logger, f"extract_audio_done output={_project_relative(audio_path)}")
        pipeline_timings["audio_prepare_s"] = time.perf_counter() - audio_prepare_started
        pipeline_timings["audio_extract_s"] = (
            0.0 if audio_cached else pipeline_timings["audio_prepare_s"]
        )
        _log_stage(
            logger,
            f"stage_done audio_prepare elapsed={pipeline_timings['audio_prepare_s']:.2f}s cached={audio_cached}",
        )

        # 2. ASR text-only transcription, model unload, then alignment
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _log_stage(logger, f"device={device}")
        _raise_if_cancelled(cancel_event)

        if aligned_cache is None:
            asr_start = time.time()
            asr_pipeline_started = time.perf_counter()
            _log_stage(logger, "stage_start asr_alignment")
            asr_event_started: set[str] = set()
            asr_event_done: set[str] = set()
            asr_event_last: dict[str, dict] = {}

            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as asr_progress:
                asr_task_id = asr_progress.add_task(
                    "[magenta]ASR 初始化中...[/magenta]", total=None
                )
                asr_state = {"label": None, "total": None}

                def _on_stage(msg: str):
                    _raise_if_cancelled(cancel_event)
                    elapsed = time.time() - asr_start
                    _log_stage(logger, f"asr_stage elapsed={elapsed:.1f}s message={msg}")
                    parsed_event = _parse_asr_stage_event(msg)
                    if parsed_event is not None:
                        event_stage, event_extra = parsed_event
                        event_extra["elapsed_s"] = round(elapsed, 1)
                        asr_event_last[event_stage] = dict(event_extra)
                        if event_stage not in asr_event_started:
                            _emit_stage_event(
                                video_path,
                                event_stage,
                                "start",
                                event_extra,
                            )
                            asr_event_started.add(event_stage)
                        _emit_stage_event(
                            video_path,
                            event_stage,
                            "progress",
                            event_extra,
                        )
                    match = _ASR_PROGRESS_RE.search(msg)
                    if match:
                        raw_label = match.group("label")
                        label = _format_asr_stage_label(raw_label)
                        current = int(match.group("current"))
                        total = int(match.group("total"))
                        description = (
                            f"[magenta]{label}[/magenta] [dim]{elapsed:.0f}s[/dim]"
                        )
                        if asr_state["label"] != label or asr_state["total"] != total:
                            asr_progress.reset(
                                asr_task_id,
                                total=total,
                                completed=current,
                                description=description,
                            )
                            asr_state["label"] = label
                            asr_state["total"] = total
                        else:
                            asr_progress.update(
                                asr_task_id,
                                completed=current,
                                total=total,
                                description=description,
                            )
                        return

                    asr_progress.update(
                        asr_task_id,
                        description=f"[magenta]{msg}[/magenta] [dim]{elapsed:.0f}s[/dim]",
                    )

                segments, asr_log, asr_details = asr_module.transcribe_and_align(
                    audio_path,
                    device,
                    on_stage=_on_stage,
                    include_details=True,
                )
                _raise_if_cancelled(cancel_event)
                for event_stage in ("asr_text_transcribe", "alignment"):
                    if event_stage in asr_event_started and event_stage not in asr_event_done:
                        _emit_stage_event(
                            video_path,
                            event_stage,
                            "done",
                            asr_event_last.get(event_stage, {}),
                        )
                        asr_event_done.add(event_stage)
                if asr_state["total"]:
                    asr_progress.update(asr_task_id, completed=asr_state["total"])
            pipeline_timings["asr_alignment_total_s"] = (
                time.perf_counter() - asr_pipeline_started
            )
            _log_stage(
                logger,
                f"stage_done asr_alignment elapsed={pipeline_timings['asr_alignment_total_s']:.2f}s segments={len(segments)}",
            )
            _write_json(
                aligned_segments_path,
                _aligned_segments_payload(
                    backend_label=backend_label,
                    audio_path=audio_path,
                    audio_cache_key=audio_cache_key,
                    segments=segments,
                    asr_details=asr_details,
                    asr_log=asr_log,
                    cache_signature=None,
                    subtitle_options=subtitle_options,
                    cache_stage="asr_alignment",
                ),
            )
            _log_stage(
                logger,
                f"aligned_segments_written path={_project_relative(aligned_segments_path)}",
            )
        else:
            pipeline_timings["asr_alignment_total_s"] = 0.0
            _log_stage(logger, "stage_skip asr_alignment reason=aligned_cache")

    _raise_if_cancelled(cancel_event)
    console.print(f"[dim]ASR backend: {backend_label}[/dim]")
    for line in asr_log:
        console.print(f"[cyan]{line}[/cyan]")
        _log_stage(logger, f"asr_log {line}")
    console.print(f"[green]识别完成，共 {len(segments)} 个片段。[/green]")
    _log_stage(logger, f"segments_count={len(segments)}")

    skip_translation = effective_ctx.skip_translation
    if skip_translation:
        bilingual = False
    else:
        bilingual = output_module.resolve_subtitle_bilingual_for_ctx(effective_ctx)

    output_dir = output_module.resolve_output_dir_for_ctx(
        video_path,
        effective_ctx,
        project_root=PROJECT_ROOT,
    )
    srt_filename = f"{video_filename}.ja.srt" if skip_translation else f"{video_filename}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    transcript_path = os.path.join(job_temp_dir, f"{video_filename}.transcript.json")
    asr_manifest_path = os.path.join(job_temp_dir, f"{video_filename}.asr_manifest.json")
    bilingual_json_path = os.path.join(
        job_temp_dir, f"{video_filename}.bilingual.json"
    )
    timings_path = os.path.join(job_temp_dir, f"{video_filename}.timings.json")
    translation_cache_path = effective_ctx.translation_cache_path.strip()
    if not translation_cache_path:
        translation_cache_path = os.path.join(job_temp_dir, "translation_cache.jsonl")
    _log_stage(logger, f"output_dir={_project_relative(output_dir)}")
    _log_stage(logger, f"srt_path={_project_relative(srt_path)}")
    _log_stage(
        logger,
        f"translation_cache_path={_project_relative(translation_cache_path)}",
    )
    _raise_if_cancelled(cancel_event)

    return AsrArtifacts(
        segments=segments,
        audio_path=audio_path,
        job_temp_dir=job_temp_dir,
        asr_details=asr_details,
        aligned_segments_path=aligned_segments_path,
        transcript_path=transcript_path,
        asr_manifest_path=asr_manifest_path,
        pipeline_timings=pipeline_timings,
        logger=logger,
        run_log_path=run_log_path,
        audio_cache_key=audio_cache_key,
        video_stem=video_filename,
        output_dir=output_dir,
        srt_path=srt_path,
        bilingual_json_path=bilingual_json_path,
        quality_report_path="",
        bilingual=bilingual,
        timings_path=timings_path,
        translation_cache_path=translation_cache_path,
        asr_log=asr_log,
        audio_cached=audio_cached,
        device=device,
        backend_label=backend_label,
        video_duration_s=video_duration_s,
        video_fps=video_fps,
        pipeline_started=pipeline_started,
        job_id=job_id,
        aligned_cache_signature=aligned_cache_signature,
    )


def run_translation_and_write(
    video_path: str,
    artifacts: AsrArtifacts,
    *,
    ctx: JobContext,
    job_id: str = "",
    cancel_event=None,
) -> list[str]:
    try:
        return _run_translation_and_write_impl(
            video_path,
            artifacts,
            ctx=ctx,
            job_id=job_id,
            cancel_event=cancel_event,
        )
    finally:
        _close_artifacts_logger(artifacts)


def _run_translation_and_write_impl(
    video_path: str,
    artifacts: AsrArtifacts,
    *,
    ctx: JobContext,
    job_id: str = "",
    cancel_event=None,
) -> list[str]:
    output_paths: list[str] = []
    events._thread_local.video = os.path.basename(video_path)
    _raise_if_cancelled(cancel_event)

    segments = artifacts.segments
    audio_path = artifacts.audio_path
    audio_cache_key = artifacts.audio_cache_key
    audio_cached = artifacts.audio_cached
    asr_details = artifacts.asr_details
    asr_log = artifacts.asr_log
    backend_label = artifacts.backend_label or asr_module.get_backend_label()
    device = artifacts.device
    job_id = sanitize_job_id(job_id or ctx.job_id or artifacts.job_id)
    if job_id:
        events.set_current_job_id(job_id)
    job_temp_dir = artifacts.job_temp_dir
    os.makedirs(job_temp_dir, exist_ok=True)
    pipeline_started = artifacts.pipeline_started
    pipeline_timings = artifacts.pipeline_timings
    video_duration_s = artifacts.video_duration_s
    video_fps = artifacts.video_fps
    video_filename = artifacts.video_stem
    aligned_segments_path = artifacts.aligned_segments_path
    logger = artifacts.logger
    run_log_path = artifacts.run_log_path
    output_dir = artifacts.output_dir
    srt_path = artifacts.srt_path
    transcript_path = artifacts.transcript_path
    asr_manifest_path = artifacts.asr_manifest_path
    bilingual_json_path = artifacts.bilingual_json_path
    timings_path = artifacts.timings_path
    translation_cache_path = artifacts.translation_cache_path
    bilingual = artifacts.bilingual
    skip_translation = ctx.skip_translation
    subtitle_options = _subtitle_options_for_video(ctx, video_fps)
    aligned_cache_signature = artifacts.aligned_cache_signature
    if aligned_cache_signature is None:
        _, aligned_cache_signature = aligned_cache_expectations_for_ctx(
            ctx,
            backend_label=backend_label,
            video_fps=video_fps,
        )
    _raise_if_cancelled(cancel_event)

    _write_json(
        asr_manifest_path,
        {
            "backend": backend_label,
            "audio_path": audio_path,
            "job_id": job_id,
            "job_temp_dir": job_temp_dir,
            "manifest": build_asr_manifest(segments),
        },
    )
    output_paths.append(asr_manifest_path)
    artifacts.asr_manifest_path = asr_manifest_path
    _raise_if_cancelled(cancel_event)

    if not asr_qc_gate(segments, headless=True):
        _log_stage(logger, "stage_blocked asr_qc_gate")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        pipeline_timings["pipeline_total_s"] = time.perf_counter() - pipeline_started
        _log_timing_snapshot(logger, pipeline_timings, asr_details)
        _write_json(
            timings_path,
            {
                "video_path": video_path,
                "audio_path": audio_path,
                "audio_cached": audio_cached,
                "job_id": job_id,
                "job_temp_dir": job_temp_dir,
                "device": device,
                "backend": backend_label,
                "counts": {
                    "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                    "segments": len(segments),
                    "blocks": 0,
                },
                "stage_timings": pipeline_timings,
                "asr_details": asr_details,
                "translation_request_timings": [],
                "translation_api_retry_events": [],
                "asr_qc_blocked": True,
                "video_fps": {
                    "detected": video_fps,
                    "effective": subtitle_options.effective_video_fps,
                    "fallback": video_fps is None,
                },
                "outputs": {
                    "job_temp_dir": job_temp_dir,
                    "asr_manifest": asr_manifest_path,
                    "timings": timings_path,
                    "run_log": str(run_log_path) if run_log_path else None,
                },
                "asr_log": asr_log,
            },
        )
        output_paths.append(timings_path)
        _print_timing_summary(pipeline_timings, asr_details)
        console.print(
            "[bold red]ASR 空文本率过高，已停止翻译。[/bold red] "
            "设置 QC_IGNORE_EMPTY=1 可强制继续。"
        )
        if ctx.fail_on_qc_block:
            raise RuntimeError('ASR empty text rate too high; set QC_IGNORE_EMPTY=1 to skip')
        return output_paths

    if segments:
        asr_noise_items = asr_noise_module.find_asr_noise_segments(segments)
        if asr_noise_items:
            asr_details["asr_noise_diagnostics"] = {
                "count": len(asr_noise_items),
                "items": asr_noise_items,
                "policy": "diagnostic_only",
            }
            _log_stage(
                logger,
                f"asr_noise_diagnostics count={len(asr_noise_items)} policy=diagnostic_only",
            )
            artifacts.asr_details = asr_details

    if not segments:
        _log_stage(logger, "stage_skip translation reason=no_segments")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        write_started = time.perf_counter()
        _log_stage(logger, "stage_start write_output")
        _raise_if_cancelled(cancel_event)
        if bilingual:
            srt_blocks = subtitle_module.write_bilingual_srt(
                [],
                srt_path,
                options=subtitle_options,
            )
        else:
            srt_blocks = subtitle_module.write_srt(
                [],
                srt_path,
                options=subtitle_options,
            )
        _write_json(
            transcript_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "chunks": asr_details.get("transcript_chunks", []),
            },
        )
        _write_json(
            aligned_segments_path,
            _aligned_segments_payload(
                backend_label=backend_label,
                audio_path=audio_path,
                audio_cache_key=audio_cache_key,
                segments=[],
                asr_details=asr_details,
                asr_log=asr_log,
                cache_signature=aligned_cache_signature,
                subtitle_options=subtitle_options,
            ),
        )
        _write_json(
            bilingual_json_path,
            {
                "blocks": srt_blocks,
                "timeline_mode": subtitle_options.timeline_mode,
                "video_fps": {
                    "detected": video_fps,
                    "effective": subtitle_options.effective_video_fps,
                    "fallback": video_fps is None,
                },
            },
        )
        output_paths.extend(
            [srt_path, transcript_path, aligned_segments_path, bilingual_json_path]
        )
        pipeline_timings["write_output_s"] = time.perf_counter() - write_started
        pipeline_timings["pipeline_total_s"] = time.perf_counter() - pipeline_started
        _log_stage(
            logger,
            f"stage_done write_output elapsed={pipeline_timings['write_output_s']:.2f}s",
        )
        _log_timing_snapshot(logger, pipeline_timings, asr_details)
        quality_report_path = _write_quality_report_for_ctx(
            ctx=ctx,
            video_stem=video_filename,
            job_temp_dir=job_temp_dir,
            aligned_segments=[],
            asr_details=asr_details,
            video_duration_s=video_duration_s,
            enabled=_ctx_flag(ctx, "QUALITY_REPORT_ENABLED"),
            glossary=ctx.translation_glossary,
        )
        _write_json(
            timings_path,
            {
                "video_path": video_path,
                "audio_path": audio_path,
                "audio_cached": audio_cached,
                "job_id": job_id,
                "job_temp_dir": job_temp_dir,
                "device": device,
                "backend": backend_label,
                "counts": {"segments": 0, "blocks": 0},
                "stage_timings": pipeline_timings,
                "asr_details": asr_details,
                "translation_request_timings": [],
                "video_fps": {
                    "detected": video_fps,
                    "effective": subtitle_options.effective_video_fps,
                    "fallback": video_fps is None,
                },
                "outputs": {
                    "job_temp_dir": job_temp_dir,
                    "srt": srt_path,
                    "asr_manifest": asr_manifest_path,
                    "transcript_json": transcript_path,
                    "aligned_segments_json": aligned_segments_path,
                    "bilingual_json": bilingual_json_path,
                    "quality_report": quality_report_path,
                    "run_log": str(run_log_path) if run_log_path else None,
                },
                "asr_log": asr_log,
            },
        )
        output_paths.append(timings_path)
        if quality_report_path:
            output_paths.append(quality_report_path)
        artifacts.quality_report_path = quality_report_path or ""
        _print_timing_summary(pipeline_timings, asr_details)
        console.print(
            f"\n[bold yellow]未识别到可翻译字幕。[/bold yellow] 已生成空文件：{_project_relative(srt_path)}"
        )
        if not ctx.keep_temp_files:
            _cleanup_pipeline_temp(job_temp_dir, audio_path, translation_cache_path)
        return output_paths

    if skip_translation:
        console.print("[yellow]SKIP_TRANSLATION=1，跳过翻译并输出日文 SRT。[/yellow]")
        _log_stage(logger, "stage_skip translation reason=SKIP_TRANSLATION")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        translation_request_timings: list[dict] = []
        srt_blocks = _prepare_translation_cues(
            segments,
            subtitle_options=subtitle_options,
            bilingual=False,
        )
        asr_details["subtitle_cue_plan"] = {
            "segments_before": len(segments),
            "cues_after": len(srt_blocks),
            "mode": "srt",
            "stage": "pre_translation",
        }

        write_started = time.perf_counter()
        _log_stage(logger, "stage_start write_output")
        _raise_if_cancelled(cancel_event)
        srt_blocks = subtitle_module.write_srt(
            srt_blocks,
            srt_path,
            options=subtitle_options,
        )
        _write_json(
            transcript_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "chunks": asr_details.get("transcript_chunks", []),
            },
        )
        _write_json(
            aligned_segments_path,
            _aligned_segments_payload(
                backend_label=backend_label,
                audio_path=audio_path,
                audio_cache_key=audio_cache_key,
                segments=segments,
                asr_details=asr_details,
                asr_log=asr_log,
                cache_signature=aligned_cache_signature,
                subtitle_options=subtitle_options,
            ),
        )
        _write_json(
            bilingual_json_path,
            {
                "backend": backend_label,
                "timeline_mode": subtitle_options.timeline_mode,
                "blocks": srt_blocks,
                "translation_skipped": True,
                "video_fps": {
                    "detected": video_fps,
                    "effective": subtitle_options.effective_video_fps,
                    "fallback": video_fps is None,
                },
                "translation_request_timings": translation_request_timings,
                "translation_api_retry_events": [],
            },
        )
        output_paths.extend(
            [srt_path, transcript_path, aligned_segments_path, bilingual_json_path]
        )
        pipeline_timings["write_output_s"] = time.perf_counter() - write_started
        pipeline_timings["pipeline_total_s"] = time.perf_counter() - pipeline_started
        _log_stage(
            logger,
            f"stage_done write_output elapsed={pipeline_timings['write_output_s']:.2f}s blocks={len(srt_blocks)}",
        )
        quality_report_path = _write_quality_report_for_ctx(
            ctx=ctx,
            video_stem=video_filename,
            job_temp_dir=job_temp_dir,
            aligned_segments=quality_module.quality_segments_from_blocks(srt_blocks),
            asr_details=asr_details,
            video_duration_s=video_duration_s,
            enabled=_ctx_flag(ctx, "QUALITY_REPORT_ENABLED"),
            glossary=ctx.translation_glossary,
        )
        _write_json(
            timings_path,
            {
                "video_path": video_path,
                "audio_path": audio_path,
                "audio_cached": audio_cached,
                "job_id": job_id,
                "job_temp_dir": job_temp_dir,
                "device": device,
                "backend": backend_label,
                "counts": {
                    "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                    "segments": len(segments),
                    "translation_cues": len(srt_blocks),
                    "blocks": len(srt_blocks),
                },
                "stage_timings": pipeline_timings,
                "asr_details": asr_details,
                "translation_request_timings": translation_request_timings,
                "translation_api_retry_events": [],
                "translation_skipped": True,
                "video_fps": {
                    "detected": video_fps,
                    "effective": subtitle_options.effective_video_fps,
                    "fallback": video_fps is None,
                },
                "outputs": {
                    "job_temp_dir": job_temp_dir,
                    "srt": srt_path,
                    "asr_manifest": asr_manifest_path,
                    "transcript_json": transcript_path,
                    "aligned_segments_json": aligned_segments_path,
                    "bilingual_json": bilingual_json_path,
                    "quality_report": quality_report_path,
                    "run_log": str(run_log_path) if run_log_path else None,
                },
                "asr_log": asr_log,
            },
        )
        output_paths.append(timings_path)
        if quality_report_path:
            output_paths.append(quality_report_path)
        artifacts.quality_report_path = quality_report_path or ""
        _log_timing_snapshot(logger, pipeline_timings, asr_details)
        _log_stage(
            logger,
            f"run_done srt={_project_relative(srt_path)} timings={_project_relative(timings_path)}",
        )
        _print_timing_summary(pipeline_timings, asr_details)
        console.print(f"\n[bold green]完成！[/bold green] 已保存至：{_project_relative(srt_path)}")
        console.print(f"[dim]详细耗时：{_project_relative(timings_path)}[/dim]")
        if not ctx.keep_temp_files:
            _cleanup_pipeline_temp(job_temp_dir, audio_path, translation_cache_path)
        return output_paths

    # 4. Global context for LLM
    translation_segments = _prepare_translation_cues(
        segments,
        subtitle_options=subtitle_options,
        bilingual=bilingual,
    )
    asr_details["subtitle_cue_plan"] = {
        "segments_before": len(segments),
        "cues_after": len(translation_segments),
        "mode": "bilingual" if bilingual else "srt",
        "stage": "pre_translation",
    }
    _log_stage(
        logger,
        "subtitle_cue_plan "
        f"segments_before={len(segments)} "
        f"cues_after={len(translation_segments)} "
        f"mode={'bilingual' if bilingual else 'srt'}",
    )
    artifacts.asr_details = asr_details
    _raise_if_cancelled(cancel_event)

    context_started = time.perf_counter()
    _log_stage(logger, "stage_start translation_context")
    _raise_if_cancelled(cancel_event)
    global_context = translator_module.generate_global_context(translation_segments)
    _raise_if_cancelled(cancel_event)
    pipeline_timings["translation_context_s"] = time.perf_counter() - context_started
    _log_stage(
        logger,
        f"stage_done translation_context elapsed={pipeline_timings['translation_context_s']:.2f}s chars={len(global_context)}",
    )

    # 5. Batch-concurrent translation with full-film context
    translation_started = time.perf_counter()
    _log_stage(logger, f"stage_start translation cues={len(translation_segments)}")
    _raise_if_cancelled(cancel_event)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[yellow]翻译中...", total=len(translation_segments))
        completed_segments = 0

        def _on_translation_done(_timing: dict) -> None:
            nonlocal completed_segments
            _raise_if_cancelled(cancel_event)
            completed_segments = min(
                len(translation_segments),
                completed_segments + int(_timing.get("segment_count", 0) or 0),
            )
            progress.update(task_id, completed=completed_segments)
            _log_stage(logger, f"translation_batch_done timing={_timing}")

        def _on_translation_progress(evt: dict) -> None:
            _raise_if_cancelled(cancel_event)
            _log_stage(logger, f"translation_progress {evt}")
            _emit_stage_event(video_path, "translation", "progress", dict(evt))
            phase = evt.get("phase")
            if phase == "reset":
                progress.reset(task_id, completed=0)
                progress.update(
                    task_id,
                    description=f"[blue]翻译: 重试 #{int(evt.get('attempt', 0)) + 1} 思考中...",
                )
            elif phase == "thinking":
                progress.update(
                    task_id,
                    description=f"[blue]翻译: 思考中 ({int(evt.get('reasoning_chars', 0))} chars)...",
                )
            elif phase == "translating":
                progress.update(
                    task_id,
                    description="[yellow]翻译: 输出中",
                    completed=int(evt.get("translated", 0)),
                )
            elif phase == "done":
                progress.update(
                    task_id,
                    description="[green]翻译完成",
                    completed=int(evt.get("expected", len(translation_segments))),
                )

        (
            zh_texts,
            translation_request_timings,
            translation_api_retry_events,
        ) = translator_module.translate_segments(
            translation_segments,
            global_context=global_context,
            target_lang=ctx.target_lang,
            glossary=ctx.translation_glossary,
            character_reference=ctx.asr_context,
            max_workers=ctx.translation_max_workers,
            reasoning_effort=ctx.llm_reasoning_effort,
            api_format=ctx.llm_api_format,
            cache_path=translation_cache_path,
            on_batch_done=_on_translation_done,
            on_progress=_on_translation_progress,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(cancel_event)

    srt_blocks = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "zh_text": zh_text,
            "ja_text": seg.get("ja_text") or seg.get("text", ""),
            "words": list(seg.get("words") or []),
            "cue_id": seg.get("cue_id"),
            "source_segment_ids": list(seg.get("source_segment_ids") or []),
        }
        for seg, zh_text in zip(translation_segments, zh_texts)
    ]
    pipeline_timings["translation_s"] = time.perf_counter() - translation_started
    _log_stage(
        logger,
        f"stage_done translation elapsed={pipeline_timings['translation_s']:.2f}s blocks={len(srt_blocks)}",
    )

    # 6. Write SRT and detailed sidecar outputs
    write_started = time.perf_counter()
    _log_stage(logger, "stage_start write_output")
    _raise_if_cancelled(cancel_event)
    if bilingual:
        srt_blocks = subtitle_module.write_bilingual_srt(
            srt_blocks,
            srt_path,
            options=subtitle_options,
        )
    else:
        srt_blocks = subtitle_module.write_srt(
            srt_blocks,
            srt_path,
            options=subtitle_options,
        )

    _write_json(
        transcript_path,
        {
            "backend": backend_label,
            "audio_path": audio_path,
            "chunks": asr_details.get("transcript_chunks", []),
        },
    )
    _write_json(
        aligned_segments_path,
        _aligned_segments_payload(
            backend_label=backend_label,
            audio_path=audio_path,
            audio_cache_key=audio_cache_key,
            segments=segments,
            asr_details=asr_details,
            asr_log=asr_log,
            cache_signature=aligned_cache_signature,
            subtitle_options=subtitle_options,
        ),
    )
    _write_json(
        bilingual_json_path,
        {
            "backend": backend_label,
            "timeline_mode": subtitle_options.timeline_mode,
            "blocks": srt_blocks,
            "video_fps": {
                "detected": video_fps,
                "effective": subtitle_options.effective_video_fps,
                "fallback": video_fps is None,
            },
            "translation_request_timings": translation_request_timings,
            "translation_api_retry_events": translation_api_retry_events,
        },
    )
    output_paths.extend(
        [srt_path, transcript_path, aligned_segments_path, bilingual_json_path]
    )
    pipeline_timings["write_output_s"] = time.perf_counter() - write_started
    pipeline_timings["pipeline_total_s"] = time.perf_counter() - pipeline_started
    _log_stage(
        logger,
        f"stage_done write_output elapsed={pipeline_timings['write_output_s']:.2f}s",
    )
    quality_report_path = _write_quality_report_for_ctx(
        ctx=ctx,
        video_stem=video_filename,
        job_temp_dir=job_temp_dir,
        aligned_segments=quality_module.quality_segments_from_blocks(srt_blocks),
        asr_details=asr_details,
        video_duration_s=video_duration_s,
        enabled=_ctx_flag(ctx, "QUALITY_REPORT_ENABLED"),
        glossary=ctx.translation_glossary,
    )

    _write_json(
        timings_path,
        {
            "video_path": video_path,
            "audio_path": audio_path,
            "audio_cached": audio_cached,
            "job_id": job_id,
            "job_temp_dir": job_temp_dir,
            "device": device,
            "backend": backend_label,
            "counts": {
                "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                "segments": len(segments),
                "translation_cues": len(translation_segments),
                "blocks": len(srt_blocks),
            },
            "stage_timings": pipeline_timings,
            "asr_details": asr_details,
            "translation_request_timings": translation_request_timings,
            "translation_api_retry_events": translation_api_retry_events,
            "video_fps": {
                "detected": video_fps,
                "effective": subtitle_options.effective_video_fps,
                "fallback": video_fps is None,
            },
            "outputs": {
                "job_temp_dir": job_temp_dir,
                "srt": srt_path,
                "asr_manifest": asr_manifest_path,
                "transcript_json": transcript_path,
                "aligned_segments_json": aligned_segments_path,
                "bilingual_json": bilingual_json_path,
                "quality_report": quality_report_path,
                "run_log": str(run_log_path) if run_log_path else None,
            },
            "asr_log": asr_log,
        },
    )
    output_paths.append(timings_path)
    if quality_report_path:
        output_paths.append(quality_report_path)
    artifacts.quality_report_path = quality_report_path or ""
    _log_timing_snapshot(logger, pipeline_timings, asr_details)
    _log_stage(
        logger,
        f"run_done srt={_project_relative(srt_path)} timings={_project_relative(timings_path)}",
    )
    _print_timing_summary(pipeline_timings, asr_details)
    console.print(f"\n[bold green]完成！[/bold green] 已保存至：{_project_relative(srt_path)}")
    console.print(f"[dim]详细耗时：{_project_relative(timings_path)}[/dim]")
    if not ctx.keep_temp_files:
        _cleanup_pipeline_temp(job_temp_dir, audio_path, translation_cache_path)
    return output_paths
