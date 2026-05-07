import os
import re
import hashlib
import json
import logging
import time
import subprocess
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import xxhash

from core import events
from core.config import load_config
from core.job_context import JobContext
from utils.model_paths import PROJECT_ROOT


_ENV_OVERRIDE_KEYS = (
    "JOB_TEMP_DIR",
    "ASR_BACKEND",
    "ASR_VAD_BACKEND",
    "ASR_WORKER_MODE",
    "WHISPER_TIMESTAMP_MODE",
    "WHISPER_MODEL_PATH",
    "ASR_MODEL_PATH",
    "ASR_MODEL_ID",
    "ALIGNER_MODEL_PATH",
    "ALIGNER_MODEL_ID",
    "API_KEY",
    "OPENAI_COMPATIBILITY_BASE_URL",
    "F0_FILTER_NONE_SEGMENTS",
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

from whisper import pipeline as asr_module
from subtitles import writer as subtitle_module
from llm import translator as translator_module
from whisper.qc import asr_qc_gate, build_asr_manifest
from subtitles.qc import compute_quality_report

from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table

console = Console(force_terminal=False, emoji=False)
_ASR_PROGRESS_RE = re.compile(
    r"(?P<label>音频切块|ASR 文本转写|Alignment 对齐|Alignment 局部细化|Alignment 降级重试)\s+(?P<current>\d+)/(?P<total>\d+)"
)
_STAGE_LOG_RE = re.compile(
    r"^stage_(?P<phase>start|done|skip|blocked|degraded)\s+(?P<stage>[A-Za-z0-9_]+)(?:\s+(?P<extra>.*))?$"
)
_ASR_STAGE_MAP = {
    "ASR 文本转写": "asr_text_transcribe",
    "Alignment 对齐": "alignment",
    "Alignment 局部细化": "alignment",
    "Alignment 降级重试": "alignment",
}
_AUDIO_SAMPLE_RATE = max(8000, int(os.getenv("AUDIO_SAMPLE_RATE", "16000")))
_AUDIO_CHANNELS = max(1, int(os.getenv("AUDIO_CHANNELS", "1")))
_AUDIO_BASE_FILTER = os.getenv("AUDIO_FILTER", "highpass=f=70,lowpass=f=7600").strip()
_AUDIO_CACHE_SAMPLE_BYTES = 4 * 1024 * 1024
_AUDIO_DYNAUDNORM_FILTER = "agate=threshold=0.01,dynaudnorm=f=250:g=15"
_AUDIO_USE_LOUDNORM = os.getenv("AUDIO_USE_LOUDNORM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_AUDIO_DYNAUDNORM = os.getenv("AUDIO_DYNAUDNORM", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
    if name == "SUBTITLE_SHOW_GENDER":
        return ctx.show_gender
    if name == "MULTI_CUE_SPLIT_ENABLED":
        return ctx.multi_cue_split
    if name == "ASR_RECOVERY_ENABLED":
        return ctx.asr_recovery
    if name == "QUALITY_REPORT_ENABLED":
        return ctx.keep_quality_report
    return default


_ASR_STAGE_ADVANCED_PREFIXES = (
    "WHISPERSEG_",
    "SEGMENT_",
    "VAD_",
    "ASR_QC_",
    "ASR_RECOVERY_",
    "ASR_GRAY_",
    "ASR_CONTEXT_LEAK_",
    "ASR_FRAGMENT_",
    "ASR_NATIVE_",
    "ALIGNMENT_",
    "F0_",
    "TIMESTAMP_VAD_",
    "TEN_VAD_",
)
_ASR_STAGE_ADVANCED_KEYS = {
    "ASR_VAD_BACKEND",
    "WHISPER_TIMESTAMP_MODE",
    "TRANSCRIPTION_TIMEOUT_S",
    "TRANSCRIPTION_MAX_NEW_TOKENS",
    "ASR_BATCH_SIZE",
    "ALIGNER_BATCH_SIZE",
    "KEEP_ASR_CHUNKS",
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
    overrides.setdefault("WHISPERSEG_THRESHOLD", str(ctx.vad_threshold))
    return overrides


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
    safe_backend = _sanitize_job_id(backend_label.replace(" ", "_"))
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


def _event_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _coerce_event_value(value: str) -> object:
    stripped = value.strip().rstrip(",")
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        if any(marker in stripped for marker in (".", "e", "E")):
            return float(stripped.rstrip("s"))
        return int(stripped)
    except ValueError:
        return stripped


def _parse_stage_extra(raw_extra: str | None) -> dict:
    extra: dict[str, object] = {}
    if not raw_extra:
        return extra
    for token in raw_extra.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key:
            continue
        extra[key] = _coerce_event_value(value)
    return extra


def _emit_stage_event(
    video_path: str | None,
    stage: str,
    phase: str,
    extra: dict | None = None,
) -> None:
    if phase not in {"start", "done", "skip", "blocked", "degraded", "progress"}:
        return
    video = (
        os.path.basename(video_path)
        if video_path
        else str(getattr(events._thread_local, "video", "") or "")
    )
    events.emit(
        {
            "ts": _event_ts(),
            "job_id": events._current_job_id(),
            "video": video,
            "stage": stage,
            "phase": phase,
            "extra": dict(extra or {}),
        }
    )


def _emit_stage_log_event(video_path: str | None, message: str) -> None:
    match = _STAGE_LOG_RE.match(message)
    if not match:
        return
    phase = match.group("phase")
    if phase == "blocked":
        phase = "blocked"
    _emit_stage_event(
        video_path,
        match.group("stage"),
        phase,
        _parse_stage_extra(match.group("extra")),
    )


def _parse_asr_stage_event(message: str) -> tuple[str, dict] | None:
    match = _ASR_PROGRESS_RE.search(message)
    if not match:
        return None
    raw_label = match.group("label")
    stage = _ASR_STAGE_MAP.get(raw_label)
    if stage is None:
        return None
    current = int(match.group("current"))
    total = int(match.group("total"))
    extra = {
        "label": raw_label,
        "current": current,
        "total": total,
    }
    return stage, extra


def _log_stage(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.info(message)
    _emit_stage_log_event(None, message)


def _project_relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    project_root_text = PROJECT_ROOT.resolve().as_posix()
    normalized = raw.replace("\\", "/")
    root_pattern = re.compile(re.escape(project_root_text) + r"/?", re.IGNORECASE)
    normalized = root_pattern.sub("", normalized)
    if normalized != raw.replace("\\", "/"):
        return normalized or "."
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/")
    return raw.replace("\\", "/")


def _project_relative_required(path: str | Path) -> str:
    return _project_relative(path) or ""


def _log_timing_snapshot(
    logger: logging.Logger | None,
    stage_timings: dict,
    asr_details: dict,
) -> None:
    if logger is None:
        return
    asr_stage_timings = asr_details.get("stage_timings", {}) if asr_details else {}
    labels = (
        ("audio_prepare_s", "audio_prepare"),
        ("audio_extract_s", "audio_extract"),
        ("asr_model_load_s", "asr_model_load"),
        ("asr_text_transcribe_s", "asr_text_transcribe"),
        ("asr_model_unload_s", "asr_model_unload"),
        ("alignment_s", "alignment"),
        ("alignment_model_unload_s", "alignment_model_unload"),
        ("subtitle_merge_s", "subtitle_merge"),
        ("asr_alignment_total_s", "asr_alignment_total"),
        ("translation_context_s", "translation_context"),
        ("translation_s", "translation"),
        ("write_output_s", "write_output"),
        ("pipeline_total_s", "pipeline_total"),
    )
    for key, label in labels:
        value = stage_timings.get(key, asr_stage_timings.get(key))
        if value is not None:
            logger.info("timing %s=%.2fs", label, float(value))


def _build_audio_filter_chain() -> str:
    dynaudnorm_enabled = os.getenv(
        "AUDIO_DYNAUDNORM",
        "1" if _AUDIO_DYNAUDNORM else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    loudnorm_enabled = os.getenv(
        "AUDIO_USE_LOUDNORM",
        "1" if _AUDIO_USE_LOUDNORM else "0",
    ).strip().lower() in {"1", "true", "yes", "on"}

    filters = []
    if _AUDIO_BASE_FILTER:
        filters.append(_AUDIO_BASE_FILTER)
    if dynaudnorm_enabled:
        filters.append(_AUDIO_DYNAUDNORM_FILTER)
        if loudnorm_enabled:
            print("[WARN] AUDIO_USE_LOUDNORM ignored when AUDIO_DYNAUDNORM=1")
    elif loudnorm_enabled:
        filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    return ",".join(filters)


def _video_content_sample_hash(video_path: str) -> str:
    path = Path(video_path)
    size = path.stat().st_size
    hasher = xxhash.xxh3_64()
    with path.open("rb") as reader:
        if size <= _AUDIO_CACHE_SAMPLE_BYTES * 2:
            hasher.update(reader.read())
        else:
            hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
            reader.seek(size - _AUDIO_CACHE_SAMPLE_BYTES)
            hasher.update(reader.read(_AUDIO_CACHE_SAMPLE_BYTES))
    return hasher.hexdigest()


def _get_audio_cache_key(video_path: str) -> str:
    video_hash = _video_content_sample_hash(video_path)
    payload = (
        f"{_AUDIO_SAMPLE_RATE}|{_AUDIO_CHANNELS}|"
        f"{_build_audio_filter_chain()}|{video_hash}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _sanitize_job_id(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", (value or "").strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "job"


def _resolve_job_temp_dir(job_id: str) -> str:
    root = os.getenv("JOB_TEMP_DIR", "").strip()
    if not root:
        root = os.path.join("temp", "jobs")
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str((path / _sanitize_job_id(job_id)).resolve())


def _unlink_for_cleanup(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)


def _cleanup_translation_cache(cache_path: str = "") -> None:
    raw_path = cache_path or os.getenv("TRANSLATION_CACHE_PATH", "").strip()
    if not raw_path:
        return

    raw_cache_path = Path(raw_path)
    cache_paths = [raw_cache_path]
    if raw_cache_path.suffix.lower() == ".json":
        cache_paths.append(raw_cache_path.with_suffix(".jsonl"))
    elif raw_cache_path.suffix.lower() == ".jsonl":
        cache_paths.append(raw_cache_path.with_suffix(".json"))

    for cache_path in cache_paths:
        _unlink_for_cleanup(cache_path)
        for tmp_path in cache_path.parent.glob(f"{cache_path.name}.*.tmp"):
            _unlink_for_cleanup(tmp_path)


def _cleanup_asr_checkpoints(job_temp_dir: Path) -> None:
    try:
        checkpoint_root = Path(
            getattr(asr_module, "_ASR_CHUNK_ROOT", Path("temp") / "chunks")
        ).resolve().parent
    except Exception:
        checkpoint_root = Path("temp")

    job_marker = str(job_temp_dir.resolve()).replace("\\", "/").lower()
    for checkpoint_path in checkpoint_root.glob("asr_checkpoint_*.json"):
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_source = str(payload.get("audio_path", ""))
        except Exception:
            checkpoint_source = ""
        normalized_source = checkpoint_source.replace("\\", "/").lower()
        if job_marker and job_marker in normalized_source:
            _unlink_for_cleanup(checkpoint_path)


def _cleanup_asr_checkpoint_for_audio(audio_path: str) -> None:
    try:
        checkpoint_path = Path(asr_module._get_asr_checkpoint_path(audio_path))
    except Exception as exc:
        print(f"[WARN] cleanup failed to resolve ASR checkpoint: {exc}", file=sys.stderr)
        return
    _unlink_for_cleanup(checkpoint_path)


def _cleanup_job_temp(job_temp_dir: str, translation_cache_path: str = "") -> None:
    root = Path(job_temp_dir)
    _cleanup_translation_cache(translation_cache_path)
    _cleanup_asr_checkpoints(root)

    if not root.exists() or not root.is_dir():
        return

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {root}: {exc}", file=sys.stderr)


def _resolve_project_runtime_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _cleanup_empty_runtime_dir_tree(root: Path) -> None:
    try:
        resolved = root.resolve()
        resolved.relative_to(PROJECT_ROOT)
    except Exception as exc:
        print(f"[WARN] cleanup skipped for {root}: {exc}", file=sys.stderr)
        return

    if not resolved.exists() or not resolved.is_dir():
        return

    for path in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except Exception as exc:
            print(f"[WARN] cleanup failed for {path}: {exc}", file=sys.stderr)

    try:
        if resolved.exists() and not any(resolved.iterdir()):
            resolved.rmdir()
    except Exception as exc:
        print(f"[WARN] cleanup failed for {resolved}: {exc}", file=sys.stderr)


def _cleanup_runtime_ephemeral_temp() -> None:
    roots = [
        _resolve_project_runtime_path(os.getenv("JOB_TEMP_DIR", "temp/jobs") or "temp/jobs"),
        Path(getattr(asr_module, "_ASR_CHUNK_ROOT", Path("temp") / "chunks")).resolve(),
        _resolve_project_runtime_path(
            os.getenv("ASR_RECOVERY_OUTPUT_ROOT", Path("temp") / "recovery")
        ),
    ]
    for root in roots:
        _cleanup_empty_runtime_dir_tree(root)


def _cleanup_pipeline_temp(job_temp_dir: str, audio_path: str, translation_cache_path: str = "") -> None:
    _cleanup_asr_checkpoint_for_audio(audio_path)
    _cleanup_job_temp(job_temp_dir, translation_cache_path)
    _cleanup_runtime_ephemeral_temp()


def _format_asr_stage_label(raw_label: str) -> str:
    mapping = {
        "音频切块": "音频切块",
        "ASR 文本转写": "ASR 转写",
        "Alignment 对齐": "Alignment 对齐",
        "Alignment 局部细化": "Alignment 局部细化",
        "Alignment 降级重试": "Alignment 降级重试",
    }
    return mapping.get(raw_label, raw_label)


def _resolve_output_dir_for_ctx(video_path: str, ctx: JobContext) -> str:
    if ctx.output_dir:
        output_path = Path(ctx.output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = Path(video_path).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = output_path.resolve().parent
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path.resolve())


def _resolve_subtitle_bilingual_for_ctx(ctx: JobContext) -> bool:
    return ctx.subtitle_mode.strip().lower() in {"bilingual", "dual", "ja_zh", "ja-zh", "2"}


def extract_audio(video_path: str, out_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(_AUDIO_SAMPLE_RATE),
        "-ac",
        str(_AUDIO_CHANNELS),
    ]
    filter_chain = _build_audio_filter_chain()
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend([out_path, "-loglevel", "error"])
    subprocess.run(command, check=True)


def _probe_video_duration_s(video_path: str) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_relativize_payload_paths(payload), f, ensure_ascii=False, indent=2)


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(_relativize_payload_paths(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(target)


def _relativize_payload_paths(value):
    if isinstance(value, dict):
        return {key: _relativize_payload_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_payload_paths(item) for item in value]
    if isinstance(value, str):
        return _project_relative(value)
    return value


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
    payload = {
        "video_path": video_path,
        "audio_path": audio_path,
        "audio_cached": audio_cached,
        "job_id": job_id,
        "job_temp_dir": job_temp_dir,
        "device": device,
        "backend": backend,
        "counts": counts,
        "stage_timings": stage_timings,
        "asr_details": asr_details,
        "translation_request_timings": translation_request_timings,
        "translation_api_retry_events": translation_api_retry_events,
        "outputs": outputs,
        "asr_log": asr_log,
    }
    if asr_qc_blocked is not None:
        payload["asr_qc_blocked"] = asr_qc_blocked
    return payload


def _parse_glossary_pairs_from_text(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in re.split(r"[,，\n]+", text or ""):
        item = part.strip()
        if not item:
            continue
        if "→" in item:
            ja, zh = item.split("→", 1)
        elif "->" in item:
            ja, zh = item.split("->", 1)
        else:
            continue
        ja = ja.strip()
        zh = zh.strip()
        if ja and zh:
            pairs.append((ja, zh))
    return pairs


def _load_global_glossary_pairs(job_temp_dir: str, video_stem: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    root = Path(job_temp_dir)
    candidates = [
        root / f"{video_stem}.translation_global_glossary.json",
        root / "translation_global_glossary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            if not isinstance(terms, list):
                continue
            for item in terms:
                if not isinstance(item, dict):
                    continue
                ja = str(item.get("ja", "")).strip()
                zh = str(item.get("zh", "")).strip()
                if ja and zh:
                    pairs.append((ja, zh))
        except Exception as exc:
            console.print(f"[yellow]WARNING: 读取全局术语表失败 {path}: {exc}[/yellow]")
    return pairs


def _collect_glossary_pairs(
    job_temp_dir: str,
    video_stem: str,
    glossary: str | None = None,
) -> list[tuple[str, str]]:
    source_glossary = os.getenv("TRANSLATION_GLOSSARY", "") if glossary is None else glossary
    pairs = _parse_glossary_pairs_from_text(source_glossary)
    pairs.extend(_load_global_glossary_pairs(job_temp_dir, video_stem))

    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ja, zh in pairs:
        key = (ja, zh)
        if key in seen:
            continue
        seen.add(key)
        merged.append(key)
    return merged


def _quality_segments_from_blocks(blocks: list[dict]) -> list[dict]:
    quality_segments: list[dict] = []
    for block in blocks:
        segment = {
            "start": float(block.get("start", 0.0)),
            "end": float(block.get("end", 0.0)),
            "text": str(block.get("ja_text") or block.get("text") or block.get("ja") or ""),
            "ja": str(block.get("ja_text") or block.get("text") or block.get("ja") or ""),
            "zh": str(block.get("zh_text") or block.get("zh") or ""),
        }
        if "gender" in block:
            segment["gender"] = block.get("gender")
        quality_segments.append(segment)
    return quality_segments


def _write_quality_report(
    *,
    video_stem: str,
    job_temp_dir: str,
    aligned_segments: list[dict],
    asr_details: dict,
    video_duration_s: float | None = None,
    f0_filtered_count: int = 0,
    f0_failure: bool = False,
    enabled: bool | None = None,
    glossary: str | None = None,
) -> str | None:
    if enabled is None:
        enabled = _env_flag("QUALITY_REPORT_ENABLED")
    if not enabled:
        return None

    try:
        glossary_pairs = _collect_glossary_pairs(job_temp_dir, video_stem, glossary)
        fallback_count = int((asr_details or {}).get("fallback_count", 0))
        if video_duration_s is None:
            video_duration_s = float(
                (asr_details or {}).get("audio_duration_s", len(aligned_segments) * 2.0)
            )
        report = compute_quality_report(
            aligned_segments,
            video_duration_s,
            glossary_pairs,
            fallback_count,
            len(aligned_segments),
            f0_filtered_count=f0_filtered_count,
            f0_failure=f0_failure,
        )
        report_dir = Path(os.getenv("QUALITY_REPORT_DIR", "./reports")).expanduser()
        if not report_dir.is_absolute():
            report_dir = PROJECT_ROOT / report_dir
        report_path = report_dir.resolve() / f"{video_stem}.quality_report.json"
        _write_json_atomic(report_path, report)

        warnings_list = report.get("warnings") or []
        if warnings_list:
            console.print("[yellow]WARNING: 字幕质量报告发现以下问题：[/yellow]")
            for warning in warnings_list:
                console.print(f"[yellow]- {warning}[/yellow]")
        if os.getenv("QC_HARD_FAIL", "0").strip() == "1" and warnings_list:
            raise RuntimeError(
                f"QC_HARD_FAIL=1 and quality warnings present: {warnings_list}"
            )
        return str(report_path)
    except RuntimeError:
        raise
    except Exception as exc:
        console.print(f"[yellow]WARNING: 生成字幕质量报告失败：{exc}[/yellow]")
        return None


def _filter_f0_none_segments(
    segments: list[dict],
    *,
    f0_failed: bool = False,
    enabled: bool | None = None,
) -> tuple[list[dict], int]:
    if enabled is None:
        enabled = _env_flag("F0_FILTER_NONE_SEGMENTS")
    if not enabled:
        return segments, 0
    if f0_failed:
        console.print(
            "[yellow]F0 detection failed; skipping non-voice filter to preserve all segments[/yellow]"
        )
        return segments, 0
    filtered = [segment for segment in segments if segment.get("gender") is not None]
    return filtered, len(segments) - len(filtered)


def _word_gender_value(word: dict) -> str | None:
    gender = word.get("gender")
    return gender if gender in {"M", "F"} else None


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _gender_for_words(words: list[dict]) -> str | None:
    genders = [_word_gender_value(word) for word in words]
    genders = [gender for gender in genders if gender is not None]
    if not genders:
        return None
    return max(("M", "F"), key=genders.count)


def _segment_from_f0_words(segment: dict, words: list[dict]) -> dict | None:
    if not words:
        return None
    text = "".join(_word_text(word) for word in words).strip()
    if not text:
        return None
    try:
        start = min(float(word.get("start", segment.get("start", 0.0))) for word in words)
        end = max(float(word.get("end", segment.get("end", start))) for word in words)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    rebuilt = dict(segment)
    rebuilt.update(
        {
            "start": start,
            "end": end,
            "text": text,
            "words": [dict(word) for word in words],
            "gender": _gender_for_words(words),
            "source_chunk_index": words[0].get(
                "source_chunk_index",
                segment.get("source_chunk_index"),
            ),
        }
    )
    return rebuilt


def _split_segment_on_f0_gender_turns(segment: dict) -> list[dict]:
    words = [dict(word) for word in segment.get("words") or [] if isinstance(word, dict)]
    if len(words) < 2:
        return [segment]
    try:
        words.sort(
            key=lambda word: (
                float(word.get("start", segment.get("start", 0.0))),
                float(word.get("end", segment.get("end", 0.0))),
            )
        )
    except (TypeError, ValueError):
        return [segment]

    groups: list[list[dict]] = []
    current_words: list[dict] = []
    active_gender: str | None = None
    for word in words:
        gender = _word_gender_value(word)
        if (
            current_words
            and gender is not None
            and active_gender is not None
            and gender != active_gender
        ):
            groups.append(current_words)
            current_words = [word]
            active_gender = gender
            continue
        current_words.append(word)
        if gender is not None:
            active_gender = gender
    if current_words:
        groups.append(current_words)

    if len(groups) < 2:
        return [segment]

    rebuilt_segments = [
        rebuilt
        for group in groups
        for rebuilt in [_segment_from_f0_words(segment, group)]
        if rebuilt is not None
    ]
    return rebuilt_segments if len(rebuilt_segments) >= 2 else [segment]


def _split_segments_on_f0_gender_turns(segments: list[dict]) -> tuple[list[dict], int]:
    split_segments: list[dict] = []
    split_count = 0
    for segment in segments:
        pieces = _split_segment_on_f0_gender_turns(segment)
        split_count += max(0, len(pieces) - 1)
        split_segments.extend(pieces)
    return split_segments, split_count


def _try_load_aligned_segments(
    path: str,
    expected_audio_cache_key: str,
    expected_backend: str,
) -> dict | None:
    try:
        cache_path = Path(path)
        if not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("audio_cache_key") != expected_audio_cache_key:
            return None
        if payload.get("backend") != expected_backend:
            return None
        segments = payload.get("segments")
        if not isinstance(segments, list):
            return None
        payload["segments"] = [dict(segment) for segment in segments if isinstance(segment, dict)]
        payload["asr_details"] = payload.get("asr_details") if isinstance(payload.get("asr_details"), dict) else {}
        payload["asr_log"] = payload.get("asr_log") if isinstance(payload.get("asr_log"), list) else []
        return payload
    except Exception:
        return None


def _build_japanese_srt_blocks(segments: list[dict]) -> list[dict]:
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "ja_text": str(seg.get("text", "")),
            "zh_text": str(seg.get("text", "")),
        }
        for seg in segments
    ]


def _add_timing_row(table: Table, label: str, seconds: float | None) -> None:
    if seconds is None:
        return
    table.add_row(label, f"{seconds:.2f}s")


def _print_timing_summary(stage_timings: dict, asr_details: dict) -> None:
    table = Table(title="阶段耗时", show_lines=False)
    table.add_column("阶段")
    table.add_column("耗时", justify="right")

    asr_stage_timings = asr_details.get("stage_timings", {}) if asr_details else {}
    for label, key in (
        ("音频准备", "audio_prepare_s"),
        ("ASR 模型加载", "asr_model_load_s"),
        ("ASR 文本转写", "asr_text_transcribe_s"),
        ("ASR 模型卸载", "asr_model_unload_s"),
        ("Alignment 对齐", "alignment_s"),
        ("Alignment 模型卸载", "alignment_model_unload_s"),
        ("字幕合并", "subtitle_merge_s"),
        ("ASR+Alignment", "asr_alignment_total_s"),
        ("翻译上下文", "translation_context_s"),
        ("翻译", "translation_s"),
        ("输出写入", "write_output_s"),
        ("总计", "pipeline_total_s"),
    ):
        seconds = stage_timings.get(key, asr_stage_timings.get(key))
        _add_timing_row(table, label, seconds)

    console.print(table)


@dataclass
class AsrArtifacts:
    segments: list[dict]
    audio_path: str
    job_temp_dir: str
    asr_details: dict
    aligned_segments_path: str
    transcript_path: str
    asr_manifest_path: str
    pipeline_timings: dict
    logger: object  # logging.Logger | None
    run_log_path: object  # Path | None
    audio_cache_key: str
    video_stem: str
    output_dir: str
    srt_path: str
    bilingual_json_path: str
    quality_report_path: str
    bilingual: bool
    timings_path: str
    translation_cache_path: str
    asr_log: list[str]
    audio_cached: bool
    device: str
    backend_label: str
    video_duration_s: float | None
    pipeline_started: float
    f0_filtered_count: int
    f0_failed: bool
    job_id: str


def run_asr_alignment_f0(
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
    job_id = _sanitize_job_id(job_id or ctx.job_id or video_name)
    cache_job_id = _sanitize_job_id(cache_job_id or job_id)
    effective_ctx = ctx
    events._thread_local.video = os.path.basename(video_path)
    events.set_current_job_id(job_id)
    _raise_if_cancelled(cancel_event)
    video_duration_s = _probe_video_duration_s(video_path)
    previous_asr_backend = getattr(asr_module, "ASR_BACKEND", None)
    previous_asr_context = getattr(asr_module, "_ASR_CONTEXT", None)
    previous_asr_recovery = getattr(asr_module, "ASR_RECOVERY_ENABLED", None)
    if hasattr(asr_module, "ASR_BACKEND"):
        asr_module.ASR_BACKEND = effective_ctx.asr_backend.strip().lower()
    if hasattr(asr_module, "_ASR_CONTEXT"):
        asr_module._ASR_CONTEXT = effective_ctx.asr_context
    if hasattr(asr_module, "ASR_RECOVERY_ENABLED"):
        asr_module.ASR_RECOVERY_ENABLED = bool(effective_ctx.asr_recovery)
    backend_label = asr_module.get_backend_label()

    def _restore_asr_globals() -> None:
        if previous_asr_backend is not None and hasattr(asr_module, "ASR_BACKEND"):
            asr_module.ASR_BACKEND = previous_asr_backend
        if previous_asr_context is not None and hasattr(asr_module, "_ASR_CONTEXT"):
            asr_module._ASR_CONTEXT = previous_asr_context
        if previous_asr_recovery is not None and hasattr(asr_module, "ASR_RECOVERY_ENABLED"):
            asr_module.ASR_RECOVERY_ENABLED = previous_asr_recovery
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
    audio_dir = os.path.join(job_temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_cache_key = _get_audio_cache_key(video_path)
    audio_path = os.path.join(audio_dir, f"{video_name}.{audio_cache_key}.wav")
    aligned_segments_path = os.path.join(
        job_temp_dir, f"{video_filename}.aligned_segments.json"
    )
    aligned_cache = _try_load_aligned_segments(
        aligned_segments_path,
        audio_cache_key,
        backend_label,
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
            extract_audio(video_path, audio_path)
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

            try:
                with _temporary_env(_asr_stage_env_overrides(effective_ctx)):
                    segments, asr_log, asr_details = asr_module.transcribe_and_align(
                        audio_path,
                        device,
                        on_stage=_on_stage,
                        include_details=True,
                    )
            finally:
                _restore_asr_globals()
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
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "audio_cache_key": audio_cache_key,
                "segments": segments,
                "asr_details": asr_details,
                "asr_log": asr_log,
                "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
            },
        )
        _log_stage(
            logger,
            f"aligned_segments_written path={_project_relative(aligned_segments_path)}",
        )
    else:
        pipeline_timings["asr_alignment_total_s"] = 0.0
        _log_stage(logger, "stage_skip asr_alignment reason=aligned_cache")
        _restore_asr_globals()

    _raise_if_cancelled(cancel_event)
    console.print(f"[dim]ASR backend: {backend_label}[/dim]")
    for line in asr_log:
        console.print(f"[cyan]{line}[/cyan]")
        _log_stage(logger, f"asr_log {line}")
    console.print(f"[green]识别完成，共 {len(segments)} 个片段。[/green]")
    _log_stage(logger, f"segments_count={len(segments)}")

    f0_failed = False
    f0_filtered_count = 0
    f0_gender_split_count = 0
    if segments and not effective_ctx.skip_translation:
        try:
            from audio.f0_gender import detect_gender_f0_word_level

            console.print("[cyan]F0 gender detection...[/cyan]")
            f0_started = time.perf_counter()
            _log_stage(logger, "stage_start f0_gender_detection")
            _raise_if_cancelled(cancel_event)
            with _temporary_env(_asr_stage_env_overrides(effective_ctx)):
                segments = detect_gender_f0_word_level(
                    audio_path,
                    segments,
                    window_ms=_ctx_int(effective_ctx, "F0_WORD_WINDOW_MS", 300),
                    hop_ms=_ctx_int(effective_ctx, "F0_HOP_MS", 100),
                    median_filter_frames=_ctx_int(
                        effective_ctx,
                        "F0_MEDIAN_FILTER_FRAMES",
                        9,
                    ),
                    min_span_ms=_ctx_int(effective_ctx, "F0_WORD_MIN_SPAN_MS", 500),
                    f0_threshold_hz=_ctx_float(effective_ctx, "F0_THRESHOLD_HZ", 160.0),
                )
                if _ctx_flag(effective_ctx, "MULTI_CUE_SPLIT_ENABLED"):
                    f0_split_before = len(segments)
                    segments, f0_gender_split_count = _split_segments_on_f0_gender_turns(
                        segments
                    )
                    asr_details["f0_gender_split"] = {
                        "segments_before": f0_split_before,
                        "segments_after": len(segments),
                        "split_count": f0_gender_split_count,
                    }
                    _log_stage(
                        logger,
                        "f0_gender_split "
                        f"segments_before={f0_split_before} "
                        f"segments_after={len(segments)} "
                        f"split_count={f0_gender_split_count}",
                    )
                    if f0_gender_split_count:
                        console.print(
                            "[cyan]F0 gender split: "
                            f"{f0_split_before} -> {len(segments)} "
                            f"(split_count={f0_gender_split_count})[/cyan]"
                        )
            _raise_if_cancelled(cancel_event)
            _log_stage(
                logger,
                f"stage_done f0_gender_detection elapsed={time.perf_counter() - f0_started:.2f}s",
            )
        except PipelineCancelledError:
            raise
        except Exception:
            f0_failed = True
            _log_stage(logger, "stage_degraded f0_gender_detection")
            console.print(
                "[yellow]F0 detection degraded; translating without acoustic context[/yellow]"
            )
            segments = [dict(seg, gender=None) for seg in segments]
        segments, f0_filtered_count = _filter_f0_none_segments(
            segments,
            f0_failed=f0_failed,
            enabled=_ctx_flag(effective_ctx, "F0_FILTER_NONE_SEGMENTS"),
        )
        if f0_filtered_count:
            console.print(
                f"[yellow]F0 filter: removed {f0_filtered_count} non-voice segments[/yellow]"
            )

    skip_translation = effective_ctx.skip_translation
    if skip_translation:
        bilingual = False
    else:
        bilingual = _resolve_subtitle_bilingual_for_ctx(effective_ctx)

    output_dir = _resolve_output_dir_for_ctx(video_path, effective_ctx)
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
        pipeline_started=pipeline_started,
        f0_filtered_count=f0_filtered_count,
        f0_failed=f0_failed,
        job_id=job_id,
    )


def run_translation_and_write(
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
    backend_label = artifacts.backend_label
    device = artifacts.device
    f0_failed = artifacts.f0_failed
    f0_filtered_count = artifacts.f0_filtered_count
    job_id = _sanitize_job_id(job_id or ctx.job_id or artifacts.job_id)
    if job_id:
        events.set_current_job_id(job_id)
    job_temp_dir = artifacts.job_temp_dir
    os.makedirs(job_temp_dir, exist_ok=True)
    pipeline_started = artifacts.pipeline_started
    pipeline_timings = artifacts.pipeline_timings
    video_duration_s = artifacts.video_duration_s
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
    _raise_if_cancelled(cancel_event)

    _write_json(
        asr_manifest_path,
        {
            "backend": asr_module.get_backend_label(),
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
                "backend": asr_module.get_backend_label(),
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

    if not segments:
        _log_stage(logger, "stage_skip translation reason=no_segments")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        write_started = time.perf_counter()
        _log_stage(logger, "stage_start write_output")
        _raise_if_cancelled(cancel_event)
        if bilingual:
            subtitle_module.write_bilingual_srt([], srt_path, show_gender=ctx.show_gender)
        else:
            subtitle_module.write_srt([], srt_path, show_gender=ctx.show_gender)
        _write_json(
            transcript_path,
            {
                "backend": asr_module.get_backend_label(),
                "audio_path": audio_path,
                "chunks": asr_details.get("transcript_chunks", []),
            },
        )
        _write_json(
            aligned_segments_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "audio_cache_key": audio_cache_key,
                "segments": [],
                "asr_details": asr_details,
                "asr_log": asr_log,
            },
        )
        _write_json(
            bilingual_json_path,
            {
                "blocks": [],
                "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
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
        quality_report_path = _write_quality_report(
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
                "backend": asr_module.get_backend_label(),
                "counts": {"segments": 0, "blocks": 0},
                "stage_timings": pipeline_timings,
                "asr_details": asr_details,
                "translation_request_timings": [],
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
        srt_blocks = _build_japanese_srt_blocks(segments)

        write_started = time.perf_counter()
        _log_stage(logger, "stage_start write_output")
        _raise_if_cancelled(cancel_event)
        subtitle_module.write_srt(srt_blocks, srt_path, show_gender=ctx.show_gender)
        _write_json(
            transcript_path,
            {
                "backend": asr_module.get_backend_label(),
                "audio_path": audio_path,
                "chunks": asr_details.get("transcript_chunks", []),
            },
        )
        _write_json(
            aligned_segments_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "audio_cache_key": audio_cache_key,
                "segments": segments,
                "asr_details": asr_details,
                "asr_log": asr_log,
                "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
            },
        )
        _write_json(
            bilingual_json_path,
            {
                "backend": asr_module.get_backend_label(),
                "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
                "blocks": srt_blocks,
                "translation_skipped": True,
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
        quality_report_path = _write_quality_report(
            video_stem=video_filename,
            job_temp_dir=job_temp_dir,
            aligned_segments=_quality_segments_from_blocks(srt_blocks),
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
                "backend": asr_module.get_backend_label(),
                "counts": {
                    "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                    "segments": len(segments),
                    "blocks": len(srt_blocks),
                },
                "stage_timings": pipeline_timings,
                "asr_details": asr_details,
                "translation_request_timings": translation_request_timings,
                "translation_api_retry_events": [],
                "translation_skipped": True,
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
    context_started = time.perf_counter()
    _log_stage(logger, "stage_start translation_context")
    _raise_if_cancelled(cancel_event)
    global_context = translator_module.generate_global_context(segments)
    _raise_if_cancelled(cancel_event)
    pipeline_timings["translation_context_s"] = time.perf_counter() - context_started
    _log_stage(
        logger,
        f"stage_done translation_context elapsed={pipeline_timings['translation_context_s']:.2f}s chars={len(global_context)}",
    )

    # 5. One-shot full-context translate
    translation_started = time.perf_counter()
    _log_stage(logger, f"stage_start translation segments={len(segments)}")
    _raise_if_cancelled(cancel_event)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[yellow]翻译中...", total=len(segments))
        completed_segments = 0

        def _on_translation_done(_timing: dict) -> None:
            nonlocal completed_segments
            _raise_if_cancelled(cancel_event)
            completed_segments = min(
                len(segments),
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
                    completed=int(evt.get("expected", len(segments))),
                )

        (
            zh_texts,
            translation_request_timings,
            translation_api_retry_events,
        ) = translator_module.translate_segments(
            segments,
            global_context=global_context,
            target_lang=ctx.target_lang,
            glossary=ctx.translation_glossary,
            character_reference=ctx.asr_context,
            batch_size=ctx.translation_batch_size,
            max_workers=ctx.translation_max_workers,
            reasoning_effort=ctx.llm_reasoning_effort,
            api_format=ctx.llm_api_format,
            cache_path=translation_cache_path,
            on_batch_done=_on_translation_done,
            on_progress=_on_translation_progress,
        )
        _raise_if_cancelled(cancel_event)

    # --- Speaker Diarization (experimental, off by default) ---
    try:
        from audio.speaker_diarization import diarize_segments, build_speakers_report

        if os.getenv("EXPERIMENTAL_SPEAKER_DIARIZATION", "0") == "1":
            console.print("[cyan]Speaker diarization (experimental)...[/cyan]")
            segments = diarize_segments(segments, audio_path)
            _spk_report = build_speakers_report(segments)
            _spk_path = Path(output_dir) / f"{video_filename}.speakers.json"
            _spk_path.write_text(
                json.dumps(_spk_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_paths.append(str(_spk_path))
            console.print(f"[cyan]Speakers: {_spk_report['n_speakers']} detected[/cyan]")
    except Exception as _spk_exc:
        console.print(f"[yellow]Speaker diarization skipped: {_spk_exc}[/yellow]")

    srt_blocks = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "zh_text": zh_text,
            "ja_text": seg.get("text", ""),
            "speaker": seg.get("speaker"),
            "gender": seg.get("gender"),
        }
        for seg, zh_text in zip(segments, zh_texts)
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
        subtitle_module.write_bilingual_srt(srt_blocks, srt_path, show_gender=ctx.show_gender)
    else:
        subtitle_module.write_srt(srt_blocks, srt_path, show_gender=ctx.show_gender)

    _write_json(
        transcript_path,
        {
            "backend": asr_module.get_backend_label(),
            "audio_path": audio_path,
            "chunks": asr_details.get("transcript_chunks", []),
        },
    )
    _write_json(
        aligned_segments_path,
        {
            "backend": backend_label,
            "audio_path": audio_path,
            "audio_cache_key": audio_cache_key,
            "segments": segments,
            "asr_details": asr_details,
            "asr_log": asr_log,
            "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
            "f0_filtered_count": f0_filtered_count,
        },
    )
    _write_json(
        bilingual_json_path,
        {
            "backend": asr_module.get_backend_label(),
            "timeline_mode": subtitle_module.SUBTITLE_TIMELINE_MODE,
            "blocks": srt_blocks,
            "f0_filtered_count": f0_filtered_count,
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
    quality_report_path = _write_quality_report(
        video_stem=video_filename,
        job_temp_dir=job_temp_dir,
        aligned_segments=_quality_segments_from_blocks(srt_blocks),
        asr_details=asr_details,
        video_duration_s=video_duration_s,
        f0_filtered_count=f0_filtered_count,
        f0_failure=f0_failed,
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
            "backend": asr_module.get_backend_label(),
            "counts": {
                "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                "segments": len(segments),
                "blocks": len(srt_blocks),
                "f0_filtered": f0_filtered_count,
            },
            "stage_timings": pipeline_timings,
            "asr_details": asr_details,
            "translation_request_timings": translation_request_timings,
            "translation_api_retry_events": translation_api_retry_events,
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
