import contextlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from core.config import load_config  # noqa: E402

load_config()

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.path.abspath(os.getenv("HF_HOME"))
if os.getenv("TORCH_HOME"):
    os.environ["TORCH_HOME"] = os.path.abspath(os.getenv("TORCH_HOME"))

# Coverage note: atexit cleanup sweeps tmp/chunks, tmp/asr_timeouts, and
# non-failed tmp/jobs/<id>/audio directories. Caches (cache/), logs (logs/),
# outputs (outputs/), and private config (config/) live outside tmp/.

from asr import noise as asr_noise_module  # noqa: E402
from asr import pipeline as asr_module  # noqa: E402
from asr.manifest import build_asr_manifest  # noqa: E402
from core import events, stage_errors  # noqa: E402
from core.cancellation import PipelineCancelledError as _PipelineCancelledError  # noqa: E402
from core.cancellation import cancel_requested as _cancel_requested_impl  # noqa: E402
from core.cancellation import raise_if_cancelled as _raise_if_cancelled_impl  # noqa: E402
from core.diagnostics import format_exception_chain  # noqa: E402
from core.job_context import JobContext  # noqa: E402
from llm import backends as llm_backends  # noqa: E402
from llm import translator as translator_module  # noqa: E402
from pipeline import aligned_cache as aligned_cache_module  # noqa: E402
from pipeline import artifact_store  # noqa: E402
from pipeline import asr_env as asr_env_module  # noqa: E402
from pipeline import audio as audio_module  # noqa: E402
from pipeline import cleanup as cleanup_module  # noqa: E402
from pipeline import cue_plan as cue_plan_module  # noqa: E402
from pipeline import gpu_worker as asr_stage_worker_module  # noqa: E402
from pipeline import output as output_module  # noqa: E402
from pipeline import run_logger as run_logger_module  # noqa: E402
from pipeline import stage_log as stage_log_module  # noqa: E402
from pipeline.artifacts import AsrArtifacts  # noqa: E402
from pipeline.execution_plan import TRANSLATION_BACKEND, ExecutionPlan  # noqa: E402
from pipeline.ids import sanitize_job_id  # noqa: E402
from pipeline.output import resolve_jobs_temp_root  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from subtitles import writer as subtitle_module  # noqa: E402
from utils.model_paths import PROJECT_ROOT  # noqa: E402

# Lightweight compatibility stub so main/Web never imports real PyTorch or
# initializes CUDA while legacy test fixtures monkeypatching
# `main.torch.cuda.is_available` continue to work without loading torch.
torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

console = Console(force_terminal=False, emoji=False)
_ASR_PROGRESS_RE = stage_log_module._ASR_PROGRESS_RE

# Re-exported ASR env whitelist & cache-signature constants/helpers.
_ASR_STAGE_ADVANCED_PREFIXES = asr_env_module._ASR_STAGE_ADVANCED_PREFIXES
_ASR_STAGE_ADVANCED_KEYS = asr_env_module._ASR_STAGE_ADVANCED_KEYS
_ASR_STAGE_CACHE_NEUTRAL_KEYS = asr_env_module._ASR_STAGE_CACHE_NEUTRAL_KEYS
_ASR_STAGE_ENV_SNAPSHOT_KEYS = asr_env_module._ASR_STAGE_ENV_SNAPSHOT_KEYS
_ASR_DETAILS_LARGE_SIDECAR_KEYS = asr_env_module._ASR_DETAILS_LARGE_SIDECAR_KEYS
_is_asr_stage_advanced_key = asr_env_module._is_asr_stage_advanced_key
_asr_stage_env_snapshot_for_ctx = asr_env_module._asr_stage_env_snapshot_for_ctx
_asr_stage_config_signature_for_env = asr_env_module._asr_stage_config_signature_for_env
_compact_asr_details_for_sidecar = asr_env_module._compact_asr_details_for_sidecar
_aligned_segments_payload = asr_env_module._aligned_segments_payload


def _subtitle_options_for_ctx(_ctx: JobContext | None = None):
    return subtitle_module.SubtitleOptions.from_env()


def _asr_runtime_signature_for_env() -> dict:
    return asr_env_module._asr_runtime_signature_for_env(asr_module=asr_module)


def _aligned_cache_signature_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str,
    subtitle_options=None,
) -> dict:
    options = subtitle_options or _subtitle_options_for_ctx(ctx)
    return asr_env_module._aligned_cache_signature_for_ctx(
        ctx,
        backend_label=backend_label,
        subtitle_options=options,
        asr_module=asr_module,
        subtitle_module=subtitle_module,
    )


def aligned_cache_expectations_for_ctx(
    ctx: JobContext,
    *,
    backend_label: str | None = None,
) -> tuple[str, dict]:
    return asr_env_module.aligned_cache_expectations_for_ctx(
        ctx,
        backend_label=backend_label,
        asr_module=asr_module,
        subtitle_module=subtitle_module,
    )


# Re-exported from core so stage modules below main (pipeline.audio, and the
# media subprocess runner underneath it) can raise the same cancellation the
# Web layer already reports as 取消 rather than as a failure.
PipelineCancelledError = _PipelineCancelledError
_cancel_requested = _cancel_requested_impl
_raise_if_cancelled = _raise_if_cancelled_impl


# Run logger helpers (delegated to `pipeline.run_logger`, preserving `main.*`
# monkeypatch seams such as `_run_log_dir` and `_reopen_snapshot_run_logger`).
def _run_log_dir(ctx: JobContext) -> Path:
    return run_logger_module.run_log_dir(ctx, project_root=PROJECT_ROOT)


_run_log_component = run_logger_module.run_log_component


def _setup_run_logger(
    job_id: str,
    backend_label: str,
    ctx: JobContext,
) -> tuple[logging.Logger, Path]:
    return run_logger_module.setup_run_logger(
        job_id,
        backend_label,
        ctx,
        project_root=PROJECT_ROOT,
        run_log_root=_run_log_dir(ctx),
    )


_reopen_snapshot_run_logger = run_logger_module.reopen_snapshot_run_logger
_close_run_logger = run_logger_module.close_run_logger
_clear_thread_run_logger = run_logger_module.clear_thread_run_logger
_close_artifacts_logger = run_logger_module.close_artifacts_logger


def _persistent_timings_path(
    timings_path: str,
    run_log_path: Path | None,
) -> Path | None:
    if run_log_path is None:
        return None
    return Path(run_log_path).parent / Path(timings_path).name


def _diagnostic_output_paths(
    *,
    timings_path: str,
    run_log_path: Path | None,
) -> dict[str, str | None]:
    persistent_timings = _persistent_timings_path(timings_path, run_log_path)
    return {
        "run_log": str(run_log_path) if run_log_path else None,
        "timings_log": str(persistent_timings) if persistent_timings else None,
    }


def _append_unique_output_path(output_paths: list[str], path: str | Path | None) -> None:
    if path is None:
        return
    text = str(path)
    if text and text not in output_paths:
        output_paths.append(text)


def _write_failure_timings_best_effort(
    *,
    video_path: str,
    artifacts: AsrArtifacts,
    stage: str,
    exc: BaseException,
    run_id: str = "",
) -> None:
    """Persist diagnostic timings/retries even when translation or output fails.

    When an existing snapshot on disk belongs to the *same* ``run_id`` (e.g.
    output export/publish failed after translation wrote timings), its
    completed request timings and retry events are preserved; snapshots from an
    earlier retry's ``run_id`` are never inherited.
    """
    timings_path = getattr(artifacts, "timings_path", "")
    if not timings_path:
        return
    raw_logger = getattr(artifacts, "logger", None)
    logger = raw_logger if isinstance(raw_logger, logging.Logger) else None
    run_log_path = getattr(artifacts, "run_log_path", None)
    active_run_id = str(run_id or "").strip()
    existing_payload: dict | None = None
    if active_run_id:
        try:
            timings_file = Path(timings_path)
            if timings_file.is_file():
                loaded = json.loads(timings_file.read_text(encoding="utf-8"))
                if (
                    isinstance(loaded, dict)
                    and str(loaded.get("run_id") or "").strip() == active_run_id
                ):
                    existing_payload = loaded
        except Exception:  # noqa: BLE001 - best-effort preserve
            existing_payload = None

    diagnostics = getattr(exc, "_translation_diagnostics", None)
    request_timings = (
        list(diagnostics.get("translation_request_timings") or [])
        if isinstance(diagnostics, dict)
        else []
    )
    retry_events = (
        list(diagnostics.get("translation_api_retry_events") or [])
        if isinstance(diagnostics, dict)
        else []
    )
    if existing_payload is not None:
        if not request_timings and isinstance(
            existing_payload.get("translation_request_timings"), list
        ):
            request_timings = list(existing_payload["translation_request_timings"])
        if not retry_events and isinstance(
            existing_payload.get("translation_api_retry_events"), list
        ):
            retry_events = list(existing_payload["translation_api_retry_events"])

    stage_timings = dict(getattr(artifacts, "pipeline_timings", None) or {})
    if existing_payload is not None and isinstance(existing_payload.get("stage_timings"), dict):
        merged_stage_timings = dict(existing_payload["stage_timings"])
        merged_stage_timings.update(stage_timings)
        stage_timings = merged_stage_timings
    started_at = getattr(artifacts, "pipeline_started", None)
    if started_at:
        stage_timings["pipeline_total_s"] = time.perf_counter() - float(started_at)
    else:
        stage_timings.setdefault("pipeline_total_s", 0.0)
    asr_details = getattr(artifacts, "asr_details", None)
    asr_details_dict = asr_details if isinstance(asr_details, dict) else {}
    segments = getattr(artifacts, "segments", None) or []
    base_payload = dict(existing_payload) if existing_payload is not None else {}
    counts = dict(base_payload.get("counts")) if isinstance(base_payload.get("counts"), dict) else {}
    counts.setdefault("transcript_chunks", len(asr_details_dict.get("transcript_chunks", [])))
    counts.setdefault("segments", len(segments))
    outputs_map = (
        dict(base_payload.get("outputs"))
        if isinstance(base_payload.get("outputs"), dict)
        else {}
    )
    outputs_map.update(
        {
            "job_temp_dir": getattr(artifacts, "job_temp_dir", "") or outputs_map.get("job_temp_dir", ""),
            "srt": getattr(artifacts, "srt_path", "") or outputs_map.get("srt", ""),
            "asr_manifest": getattr(artifacts, "asr_manifest_path", "") or outputs_map.get("asr_manifest", ""),
            "transcript_json": getattr(artifacts, "transcript_path", "") or outputs_map.get("transcript_json", ""),
            "aligned_segments_json": getattr(artifacts, "aligned_segments_path", "") or outputs_map.get("aligned_segments_json", ""),
            "bilingual_json": getattr(artifacts, "bilingual_json_path", "") or outputs_map.get("bilingual_json", ""),
            **_diagnostic_output_paths(
                timings_path=timings_path,
                run_log_path=run_log_path,
            ),
        }
    )
    payload = {
        **base_payload,
        "status": "failed",
        "failed_stage": stage,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "chain": format_exception_chain(exc),
        },
        "video_path": video_path or base_payload.get("video_path", ""),
        "audio_path": getattr(artifacts, "audio_path", "") or base_payload.get("audio_path", ""),
        "audio_cached": bool(getattr(artifacts, "audio_cached", base_payload.get("audio_cached", False))),
        "job_id": getattr(artifacts, "job_id", "") or base_payload.get("job_id", ""),
        "run_id": active_run_id,
        "job_temp_dir": getattr(artifacts, "job_temp_dir", "") or base_payload.get("job_temp_dir", ""),
        "device": getattr(artifacts, "device", "") or base_payload.get("device", ""),
        "backend": getattr(artifacts, "backend_label", "") or base_payload.get("backend", ""),
        "counts": counts,
        "stage_timings": stage_timings,
        "asr_details": (
            _compact_asr_details_for_sidecar(
                asr_details_dict,
                strip_frame_class_track=True,
            )
            if asr_details_dict
            else base_payload.get("asr_details", {})
        ),
        "translation_request_timings": request_timings,
        "translation_api_retry_events": retry_events,
        "outputs": outputs_map,
        "asr_log": list(getattr(artifacts, "asr_log", None) or base_payload.get("asr_log", []) or []),
    }
    try:
        _write_timings_with_diagnostics(
            timings_path=timings_path,
            payload=payload,
            run_log_path=run_log_path,
            output_paths=[],
            logger=logger,
        )
        _log_stage(
            logger,
            f"failure_timings_written path={_project_relative(timings_path)} retries={len(retry_events)}",
        )
    except Exception as write_exc:  # noqa: BLE001 - diagnostics must never mask original error
        if logger is not None:
            logger.warning("Failed to write failure timings snapshot: %r", write_exc)


def _write_timings_with_diagnostics(
    *,
    timings_path: str,
    payload: dict,
    run_log_path: Path | None,
    output_paths: list[str],
    logger: logging.Logger | None,
) -> None:
    _write_json_atomic(timings_path, payload)
    if run_log_path is None:
        _append_unique_output_path(output_paths, timings_path)
        return
    _append_unique_output_path(output_paths, run_log_path)
    persistent_timings = _persistent_timings_path(timings_path, run_log_path)
    if persistent_timings is None:
        _append_unique_output_path(output_paths, timings_path)
        return
    try:
        if Path(timings_path).resolve() != persistent_timings.resolve():
            persistent_timings.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(timings_path, persistent_timings)
        _append_unique_output_path(output_paths, persistent_timings)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail jobs
        _log_stage(logger, f"diagnostics_timings_copy_failed error={exc!r}")
        _append_unique_output_path(output_paths, timings_path)


_event_ts = stage_log_module._event_ts
_coerce_event_value = stage_log_module._coerce_event_value
_parse_stage_extra = stage_log_module._parse_stage_extra
_emit_stage_event = stage_log_module._emit_stage_event
_emit_stage_log_event = stage_log_module._emit_stage_log_event
_parse_asr_stage_event = stage_log_module._parse_asr_stage_event
_log_stage = stage_log_module._log_stage
_log_timing_snapshot = stage_log_module._log_timing_snapshot
_format_asr_stage_label = stage_log_module._format_asr_stage_label


def _project_relative(path: str | Path | None) -> str | None:
    return artifact_store.project_relative(path, project_root=PROJECT_ROOT)


def _record_pipeline_cuda_memory(
    snapshots: list[dict],
    stage: str,
    *,
    logger: logging.Logger | None = None,
    elapsed_s: float | None = None,
    skip_reason: str = "asr_stage_worker_subprocess",
) -> None:
    # Main/Web never touches CUDA; CUDA memory snapshots live inside the GPU worker.
    reason = skip_reason or "asr_stage_worker_subprocess"
    snapshot: dict[str, object] = {
        "stage": stage,
        "cuda_available": False,
        "skipped": True,
        "reason": reason,
    }
    if elapsed_s is not None:
        snapshot["elapsed_s"] = round(float(elapsed_s), 3)
    snapshots.append(snapshot)
    _log_stage(logger, f"cuda_memory_skipped stage={stage} reason={reason}")


def _asr_details_cuda_skip_reason(asr_details: dict | None) -> str:
    # The ASR stage always runs in the unified GPU worker; the main/web process
    # must never initialize CUDA -- not even torch.cuda.is_available() -- or it
    # re-splits the CUDA context this refactor exists to consolidate.
    del asr_details
    return "asr_stage_worker_subprocess"


def _resolve_job_temp_dir(job_id: str) -> str:
    return str((resolve_jobs_temp_root(PROJECT_ROOT) / sanitize_job_id(job_id)).resolve())


def _materialize_cached_file(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Bring a resumed job's cached file into this job's directory.

    Published the same way an extraction is: hard link (already atomic) or copy
    to a partial and replace, so an interrupted copy cannot leave a short file
    under the name the cache reader trusts.
    """
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        return source_path.is_file()
    if not source_path.is_file():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        return destination_path.is_file()
    if not destination_path.exists():
        try:
            os.link(source_path, destination_path)
            return True
        except OSError:
            pass
    staged = destination_path.with_name(
        f"{destination_path.name}.{uuid.uuid4().hex[:8]}.partial"
    )
    try:
        shutil.copy2(source_path, staged)
        artifact_store.replace_atomically(staged, destination_path)
    finally:
        with contextlib.suppress(OSError):
            staged.unlink()
    return True


def _materialize_usable_audio(source: str | Path, destination: str | Path) -> bool:
    """Materialize a resumed job's audio, validating the source first and replacing corrupt targets."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not audio_module.audio_cache_is_usable(str(source_path)):
        return False
    if source_path == destination_path:
        return True
    if destination_path.exists() and audio_module.audio_cache_is_usable(str(destination_path)):
        return True
    # Destination is either absent or corrupt: materialize atomically from the verified source.
    if not _materialize_cached_file(source_path, destination_path, overwrite=True):
        return False
    return audio_module.audio_cache_is_usable(str(destination_path))


def _resolve_project_runtime_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _cleanup_pipeline_temp(
    job_temp_dir: str,
    audio_path: str = "",
    translation_cache_path: str = "",
) -> None:
    # Only clean this job's own temp directory. Never sweep sibling directories
    # under `tmp/jobs/` or `tmp/chunks/` here, as concurrent jobs may have just
    # created their empty `audio/` or chunk directories before writing files.
    # ASR results live in the cross-job content-addressed cache
    # (`asr.result_cache`) and deliberately survive job cleanup.
    del audio_path
    cleanup_module.cleanup_job_temp(
        job_temp_dir,
        translation_cache_path,
    )


def _write_json(path: str, payload: dict) -> None:
    artifact_store.write_json(path, payload, project_root=PROJECT_ROOT)


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    artifact_store.write_json_atomic(path, payload, project_root=PROJECT_ROOT)


def output_target_for_ctx(video_path: str, ctx: JobContext) -> Path:
    video = _resolve_project_runtime_path(video_path)
    directory = _resolve_project_runtime_path(ctx.output_dir) if ctx.output_dir else video.parent
    return directory / ExecutionPlan.for_asr(ctx).srt_filename(video.stem)


def _output_lease(
    video_path: str,
    ctx: JobContext,
    *,
    job_id: str,
    run_id: str,
    target: str | Path | None = None,
) -> artifact_store.OutputLease:
    destination = Path(target).resolve() if target else output_target_for_ctx(video_path, ctx)
    identity = run_id or ctx.output_run_id or uuid.uuid4().hex
    if ctx.output_generation and ctx.output_run_id == identity:
        if Path(ctx.output_target).resolve() != destination:
            raise ValueError("运行的输出目标与已登记的目标不一致")
        return artifact_store.OutputLease(
            destination,
            PROJECT_ROOT,
            job_id,
            identity,
            ctx.output_generation,
        )
    lease = artifact_store.claim_output(
        destination,
        project_root=PROJECT_ROOT,
        job_id=job_id,
        run_id=identity,
    )
    ctx.output_target = str(destination)
    ctx.output_generation = lease.sequence
    ctx.output_run_id = lease.run_id
    return lease


def _run_outputs(
    *,
    ctx: JobContext,
    run_id: str,
    job_id: str,
    lease: artifact_store.OutputLease | None = None,
    job_temp_dir: str = "",
    video_stem: str = "",
) -> artifact_store.RunOutputs:
    del job_temp_dir, video_stem
    resolved_lease = lease or artifact_store.OutputLease(
        Path(ctx.output_target),
        PROJECT_ROOT,
        job_id,
        run_id or ctx.output_run_id,
        ctx.output_generation,
    )
    return artifact_store.RunOutputs(lease=resolved_lease)


def _publish_outputs(
    outputs: artifact_store.RunOutputs,
    paths: list[str],
    artifacts: AsrArtifacts,
    cancel_event,
    *,
    subtitle_cue_count: int,
) -> list[str]:
    run_logger = getattr(artifacts, "logger", None)
    if isinstance(run_logger, logging.Logger):
        for handler in list(run_logger.handlers):
            try:
                handler.flush()
            except OSError:
                pass
    staged = {path.resolve() for path in outputs.targets()}
    run_log_raw = getattr(artifacts, "run_log_path", None)
    run_log_resolved = Path(run_log_raw).resolve() if run_log_raw else None
    if (
        run_log_resolved is not None
        and run_log_resolved not in staged
        and run_log_resolved.is_file()
    ):
        outputs.stage_file(run_log_resolved, name="run_log")
        staged.add(run_log_resolved)
    for index, raw in enumerate(paths):
        path = Path(raw).resolve()
        if path not in staged and path.is_file():
            outputs.stage_file(path, name=f"diagnostic_{index}")
            staged.add(path)
    manifest = outputs.publish(
        still_current=lambda: cancel_event is None or not cancel_event.is_set(),
        subtitle_cue_count=subtitle_cue_count,
    )
    resolved = {
        str(Path(raw).resolve() if Path(raw).is_absolute() else (PROJECT_ROOT / raw).resolve()): str(
            artifact_store.artifact_path(manifest["files"][name], PROJECT_ROOT)
        )
        for name, raw in manifest["targets"].items()
    }
    for name in ("transcript_path", "aligned_segments_path", "bilingual_json_path", "asr_manifest_path"):
        value = getattr(artifacts, name, "")
        if value and str(Path(value).resolve()) in resolved:
            setattr(artifacts, name, resolved[str(Path(value).resolve())])
    artifact_store.export_output(outputs.lease)
    return artifact_store.manifest_paths(manifest, project_root=PROJECT_ROOT)


# Re-exported cue planning helpers (preserving `main.*` monkeypatch seams).
_build_japanese_srt_blocks = cue_plan_module._build_japanese_srt_blocks
_frame_class_reader = cue_plan_module._frame_class_reader
_subtitle_cue_plan_summary = cue_plan_module._subtitle_cue_plan_summary
_log_subtitle_cue_diagnostics = cue_plan_module._log_subtitle_cue_diagnostics


def _prepare_translation_cues(
    segments: list[dict],
    *,
    subtitle_options,
    on_stage=None,
    asr_details: dict | None = None,
) -> tuple[list[dict], dict]:
    return cue_plan_module._prepare_translation_cues(
        segments,
        subtitle_options=subtitle_options,
        on_stage=on_stage,
        asr_details=asr_details,
        subtitle_module=subtitle_module,
    )


def _print_timing_summary(stage_timings: dict, asr_details: dict) -> None:
    stage_log_module._print_timing_summary(console, stage_timings, asr_details)


def _run_cue_plan_stage(
    *,
    segments: list[dict],
    subtitle_options,
    mode: str,
    asr_details: dict,
    pipeline_timings: dict,
    logger: logging.Logger | None,
) -> tuple[list[dict], dict]:
    cue_plan_started = time.perf_counter()
    _log_stage(logger, f"stage_start subtitle_cue_plan segments={len(segments)}")
    cues, cue_summary = _prepare_translation_cues(
        segments,
        subtitle_options=subtitle_options,
        on_stage=lambda stage, current, total: _log_stage(
            logger,
            f"subtitle_cue_plan stage={stage} {current}/{total}",
        ),
        asr_details=asr_details,
    )
    pipeline_timings["subtitle_cue_plan_s"] = time.perf_counter() - cue_plan_started
    _log_stage(
        logger,
        "stage_done subtitle_cue_plan "
        f"elapsed={pipeline_timings['subtitle_cue_plan_s']:.2f}s "
        f"cues={len(cues)}",
    )
    asr_details["subtitle_cue_plan"] = _subtitle_cue_plan_summary(
        segments_before=len(segments),
        mode=mode,
        cue_summary=cue_summary,
    )
    _log_subtitle_cue_diagnostics(logger, cue_summary)
    _log_stage(
        logger,
        "subtitle_cue_plan "
        f"segments_before={len(segments)} "
        f"cues_before={cue_summary.get('cues_before', 0)} "
        f"cues_after={len(cues)} "
        f"mode={mode}",
    )
    return cues, cue_summary


def run_asr_alignment(
    video_path: str,
    *,
    ctx: JobContext,
    job_id: str = "",
    run_id: str = "",
    cache_job_id: str = "",
    cancel_event=None,
) -> AsrArtifacts:
    # Always close this stage's run logger in `finally` (whether ASR succeeded,
    # failed, or was cancelled). The translation stage reopens `run_log_path`
    # in append mode and closes it in its own `finally`, preventing FileHandler
    # leaks if a job is cancelled between ASR and translation.
    events._thread_local.run_logger = None
    artifacts: AsrArtifacts | None = None
    try:
        artifacts = _run_asr_alignment_impl(
            video_path,
            ctx=ctx,
            job_id=job_id,
            run_id=run_id,
            cache_job_id=cache_job_id,
            cancel_event=cancel_event,
        )
        return artifacts
    except Exception as exc:
        run_logger = getattr(events._thread_local, "run_logger", None)
        if isinstance(run_logger, logging.Logger):
            if isinstance(exc, PipelineCancelledError):
                _log_stage(run_logger, "run_cancelled stage=asr_alignment")
            else:
                _log_stage(
                    run_logger,
                    f"run_failed stage=asr_alignment error={format_exception_chain(exc)!r}",
                )
                run_logger.exception("asr_alignment_stage_failed")
        raise
    finally:
        run_logger = getattr(events._thread_local, "run_logger", None)
        if isinstance(run_logger, logging.Logger):
            _close_run_logger(run_logger)
            _clear_thread_run_logger(run_logger)
        if artifacts is not None:
            _close_artifacts_logger(artifacts)


def _run_asr_alignment_impl(
    video_path: str,
    *,
    ctx: JobContext,
    job_id: str = "",
    run_id: str = "",
    cache_job_id: str = "",
    cancel_event=None,
) -> AsrArtifacts:
    logger: logging.Logger | None = None
    run_log_path: Path | None = None
    pipeline_started = time.perf_counter()
    pipeline_timings: dict[str, float] = {}
    pipeline_cuda_memory: list[dict] = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_filename = video_name
    job_id = sanitize_job_id(job_id or ctx.job_id or video_name)
    cache_job_id = sanitize_job_id(cache_job_id or job_id)
    effective_ctx = ctx

    with events.scoped_run_identity(job_id, run_id, video=os.path.basename(video_path)):
        backend_label = asr_module.get_backend_label()
        if effective_ctx.run_log_enabled:
            logger, run_log_path = _setup_run_logger(job_id, backend_label, effective_ctx)
            events._thread_local.run_logger = logger

        _log_stage(logger, f"run_start video={_project_relative(video_path)}")
        _log_stage(logger, f"backend={backend_label}")
        _raise_if_cancelled(cancel_event)
        # Ahead of ffprobe/ffmpeg, which would report a moved file as a non-zero
        # exit code and get described as a damaged or silent video.
        if not os.path.isfile(video_path):
            raise RuntimeError(f"{stage_errors.VIDEO_FILE_MISSING}\n{video_path}")
        lease = _output_lease(
            video_path,
            ctx,
            job_id=job_id,
            run_id=run_id or uuid.uuid4().hex,
        )
        run_id = lease.run_id
        events.set_current_run(job_id, run_id)
        video_duration_s = audio_module.probe_video_duration_s(
            video_path,
            cancel_event=cancel_event,
        )
        _log_stage(logger, "asr_stage_worker=unified_gpu_worker")
        _log_stage(logger, f"skip_translation={effective_ctx.skip_translation}")
        cuda_skip_reason = "asr_stage_worker_subprocess"
        _log_stage(logger, "cuda_diagnostics=worker_process")
        _record_pipeline_cuda_memory(
            pipeline_cuda_memory,
            "run_start",
            logger=logger,
            elapsed_s=time.perf_counter() - pipeline_started,
            skip_reason=cuda_skip_reason,
        )
        if run_log_path is not None:
            console.print(f"[dim]运行日志：{_project_relative(run_log_path)}[/dim]")

        subtitle_options = _subtitle_options_for_ctx(effective_ctx)
        _log_stage(logger, f"subtitle_frame_gap_s={subtitle_options.frame_gap_s:.6f}")
        if effective_ctx.job_temp_dir:
            job_temp_dir = str(Path(effective_ctx.job_temp_dir).expanduser().resolve())
        else:
            job_temp_dir = _resolve_job_temp_dir(job_id)
        cache_temp_dir = (
            str((Path(job_temp_dir).parent / cache_job_id).resolve())
            if cache_job_id != job_id
            else job_temp_dir
        )
        os.makedirs(job_temp_dir, exist_ok=True)
        _log_stage(logger, f"job_temp_dir={_project_relative(job_temp_dir)}")
        if cache_job_id != job_id:
            _log_stage(
                logger,
                "resume_from_job_id="
                f"{cache_job_id} cache_temp_dir={_project_relative(cache_temp_dir)}",
            )
        aligned_cache_signature = _aligned_cache_signature_for_ctx(
            effective_ctx,
            backend_label=backend_label,
            subtitle_options=subtitle_options,
        )
        audio_dir = os.path.join(job_temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_cache_key = audio_module.get_audio_cache_key(video_path)
        audio_path = os.path.join(audio_dir, f"{video_name}.{audio_cache_key}.wav")
        cache_audio_path = os.path.join(
            cache_temp_dir,
            "audio",
            f"{video_name}.{audio_cache_key}.wav",
        )
        aligned_segments_path = os.path.join(
            job_temp_dir,
            f"{video_filename}.aligned_segments.json",
        )
        cache_aligned_segments_path = os.path.join(
            cache_temp_dir,
            f"{video_filename}.aligned_segments.json",
        )
        aligned_cache = aligned_cache_module.try_load_aligned_segments(
            cache_aligned_segments_path,
            audio_cache_key,
            backend_label,
            aligned_cache_signature,
        )
        segments: list[dict] = []
        asr_log: list[str] = []
        asr_details: dict = {}

        # 1. Extract audio (skipped if cached or aligned_cache hit)
        audio_prepare_started = time.perf_counter()
        audio_cached = audio_module.audio_cache_is_usable(audio_path)
        _log_stage(logger, "stage_start audio_prepare")
        _raise_if_cancelled(cancel_event)
        if aligned_cache is not None:
            segments = aligned_cache["segments"]
            asr_log = [str(item) for item in aligned_cache.get("asr_log", [])]
            asr_details = dict(aligned_cache.get("asr_details", {}))
            if cache_job_id != job_id and not audio_cached:
                # Best-effort materialize cached audio if available, but never
                # extract a full-film WAV when ASR itself is skipped via cache.
                _materialize_usable_audio(cache_audio_path, audio_path)
            audio_cached = True
            if cache_aligned_segments_path != aligned_segments_path:
                _write_json_atomic(
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
                        cache_stage="resume_materialized",
                    ),
                )
            _log_stage(
                logger,
                "aligned_cache_hit source="
                f"{_project_relative(cache_aligned_segments_path)} "
                f"materialized={_project_relative(aligned_segments_path)}",
            )
            console.print(
                f"[green]命中 aligned_segments cache，跳过音频提取与 ASR：{_project_relative(aligned_segments_path)}[/green]"
            )
        elif audio_cached or _materialize_usable_audio(cache_audio_path, audio_path):
            audio_cached = True
            _log_stage(logger, f"audio_cache_hit path={_project_relative(audio_path)}")
            console.print(f"[dim]跳过提取：使用已缓存音频 {_project_relative(audio_path)}[/dim]")
        else:
            _log_stage(
                logger,
                f"extract_audio_start output={_project_relative(audio_path)}",
            )
            status_ctx = (
                console.status("[cyan]提取音频中...[/cyan]")
                if console.is_terminal
                else contextlib.nullcontext()
            )
            with status_ctx:
                _raise_if_cancelled(cancel_event)
                audio_module.extract_audio(
                    video_path,
                    audio_path,
                    cancel_event=cancel_event,
                    duration_s=video_duration_s,
                )
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
        _record_pipeline_cuda_memory(
            pipeline_cuda_memory,
            "audio_prepare_done",
            logger=logger,
            elapsed_s=time.perf_counter() - pipeline_started,
            skip_reason=cuda_skip_reason,
        )

        # 2. ASR text-only transcription, model unload, then subtitle timing.
        device = "auto"
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
                disable=not console.is_terminal,
            ) as asr_progress:
                asr_task_id = asr_progress.add_task(
                    "[magenta]ASR 初始化中...[/magenta]",
                    total=None,
                )
                asr_state = {"label": None, "total": None}

                def _on_stage(msg: str):
                    _raise_if_cancelled(cancel_event)
                    elapsed = time.time() - asr_start
                    _log_stage(logger, f"asr_stage elapsed={elapsed:.1f}s message={msg}")
                    if msg.startswith(stage_log_module.ASR_WORKER_DIAGNOSTIC_PREFIX):
                        return
                    is_heartbeat = msg.startswith(
                        stage_log_module.ASR_STAGE_HEARTBEAT_PREFIX
                    )
                    parsed_event = None if is_heartbeat else _parse_asr_stage_event(msg)
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
                        if (
                            int(event_extra.get("total") or 0) > 0
                            and int(event_extra.get("current") or 0)
                            >= int(event_extra["total"])
                            and event_stage not in asr_event_done
                        ):
                            _emit_stage_event(
                                video_path,
                                event_stage,
                                "done",
                                event_extra,
                            )
                            asr_event_done.add(event_stage)
                    match = None if is_heartbeat else _ASR_PROGRESS_RE.search(msg)
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
                    segments, asr_log, asr_details = (
                        asr_stage_worker_module.transcribe_and_align(
                            audio_path,
                            device=device,
                            env_overrides=_asr_stage_env_snapshot_for_ctx(
                                effective_ctx
                            ),
                            job_id=job_id,
                            run_id=run_id,
                            on_stage=_on_stage,
                            cancel_requested=lambda: _cancel_requested(cancel_event),
                        )
                    )
                    device = str(asr_details.get("device") or device)
                    _log_stage(logger, f"stage_worker_device={device}")
                except Exception as exc:
                    _log_stage(logger, f"stage_blocked asr_alignment error={exc!r}")
                    if logger is not None:
                        logger.exception("asr_alignment_failed")
                    raise
                _raise_if_cancelled(cancel_event)
                for event_stage in tuple(asr_event_started):
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
            _record_pipeline_cuda_memory(
                pipeline_cuda_memory,
                "asr_alignment_done",
                logger=logger,
                elapsed_s=time.perf_counter() - pipeline_started,
                skip_reason=cuda_skip_reason,
            )
            _write_json_atomic(
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
            _record_pipeline_cuda_memory(
                pipeline_cuda_memory,
                "asr_alignment_skipped",
                logger=logger,
                elapsed_s=time.perf_counter() - pipeline_started,
                skip_reason=cuda_skip_reason,
            )
            if device == "auto":
                device = str(asr_details.get("device") or "cache")

        _raise_if_cancelled(cancel_event)
        asr_details.setdefault("pipeline_cuda_memory", pipeline_cuda_memory)
        console.print(f"[dim]ASR backend: {backend_label}[/dim]")
        for line in asr_log:
            console.print(f"[cyan]{line}[/cyan]")
            _log_stage(logger, f"asr_log {line}")
        console.print(f"[green]识别完成，共 {len(segments)} 个片段。[/green]")
        _log_stage(logger, f"segments_count={len(segments)}")

        # One statement of what this run is: whether it translates, how its output
        # is spelled, and what each of its stages will take from the machine.
        plan = ExecutionPlan.for_asr(effective_ctx).after_asr(segment_count=len(segments))
        bilingual = plan.bilingual

        output_dir = output_module.resolve_output_dir_for_ctx(
            video_path,
            effective_ctx,
            project_root=PROJECT_ROOT,
        )
        srt_path = os.path.join(output_dir, plan.srt_filename(video_filename))
        transcript_path = os.path.join(job_temp_dir, f"{video_filename}.transcript.json")
        asr_manifest_path = os.path.join(job_temp_dir, f"{video_filename}.asr_manifest.json")
        bilingual_json_path = os.path.join(
            job_temp_dir,
            f"{video_filename}.bilingual.json",
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
            bilingual=bilingual,
            timings_path=timings_path,
            translation_cache_path=translation_cache_path,
            asr_log=asr_log,
            audio_cached=audio_cached,
            device=device,
            backend_label=backend_label,
            video_duration_s=video_duration_s,
            pipeline_started=pipeline_started,
            job_id=job_id,
            aligned_cache_signature=aligned_cache_signature,
        )


def _run_plan(artifacts: AsrArtifacts, ctx: JobContext) -> ExecutionPlan:
    """What this run will do, decided before it starts doing it.

    Built *before* the lease, because a lease is a wait: a Japanese-only run and
    a run with nothing to translate would otherwise queue behind a local server
    that is retiring, and fail on the timeout of a resource they never use.
    """
    return ExecutionPlan.for_run(ctx, has_segments=bool(artifacts.segments))


def run_translation_and_write(
    video_path: str,
    artifacts: AsrArtifacts,
    *,
    ctx: JobContext,
    job_id: str = "",
    run_id: str = "",
    cancel_event=None,
) -> list[str]:
    _reopen_snapshot_run_logger(artifacts)
    resolved_job_id = sanitize_job_id(job_id or ctx.job_id or artifacts.job_id)
    effective_run_id = run_id
    with events.scoped_run_identity(
        resolved_job_id,
        run_id,
        video=os.path.basename(video_path),
    ):
        try:
            _raise_if_cancelled(cancel_event)
            lease = _output_lease(
                video_path,
                ctx,
                job_id=resolved_job_id,
                run_id=run_id,
                target=artifacts.srt_path,
            )
            effective_run_id = lease.run_id
            events.set_current_run(resolved_job_id, effective_run_id)
            existing = artifact_store.resume_output(lease)
            if existing is not None:
                job_temp_dir = getattr(artifacts, "job_temp_dir", "")
                if job_temp_dir:
                    _cleanup_pipeline_temp(
                        job_temp_dir,
                        getattr(artifacts, "audio_path", ""),
                        getattr(artifacts, "translation_cache_path", ""),
                    )
                return artifact_store.manifest_paths(existing, project_root=PROJECT_ROOT)

            plan = _run_plan(artifacts, ctx)
            precomputed_cues: tuple[list[dict], dict] | None = None
            if plan.needs(TRANSLATION_BACKEND) and artifacts.segments:
                # Run CPU-only cue planning BEFORE acquiring the translation
                # backend lease so:
                #   1. local llama-server GPU slot is not held during CPU layout;
                #   2. if all cues are filtered out (0 cues remain), we skip the
                #      lease and empty translation request completely.
                _raise_if_cancelled(cancel_event)
                subtitle_options = _subtitle_options_for_ctx(ctx)
                asr_details_obj = getattr(artifacts, "asr_details", None)
                asr_details = asr_details_obj if isinstance(asr_details_obj, dict) else {}
                timings_obj = getattr(artifacts, "pipeline_timings", None)
                pipeline_timings = timings_obj if isinstance(timings_obj, dict) else {}
                raw_logger = getattr(artifacts, "logger", None)
                run_logger = raw_logger if isinstance(raw_logger, logging.Logger) else None
                planned_cues, cue_summary = _run_cue_plan_stage(
                    segments=artifacts.segments,
                    subtitle_options=subtitle_options,
                    mode="bilingual" if plan.bilingual else "srt",
                    asr_details=asr_details,
                    pipeline_timings=pipeline_timings,
                    logger=run_logger,
                )
                if hasattr(artifacts, "asr_details"):
                    artifacts.asr_details = asr_details
                precomputed_cues = (planned_cues, cue_summary)
                if not planned_cues:
                    plan = plan.after_asr(segment_count=0)
                _raise_if_cancelled(cancel_event)

            with ExitStack() as stack:
                if plan.needs(TRANSLATION_BACKEND):
                    # Resolved once here: the settings panel may change while a
                    # job runs, and the lease has to name the instance this task
                    # will actually use.
                    backend_name = llm_backends.selected_backend_name()
                    stack.enter_context(
                        llm_backends.backend_lease(backend_name, cancel_event)
                    )
                return _run_translation_and_write_impl(
                    video_path,
                    artifacts,
                    ctx=ctx,
                    plan=plan,
                    lease=lease,
                    precomputed_cues=precomputed_cues,
                    job_id=resolved_job_id,
                    run_id=lease.run_id,
                    cancel_event=cancel_event,
                )
        except Exception as exc:
            raw_logger = getattr(artifacts, "logger", None)
            run_logger = (
                raw_logger
                if isinstance(raw_logger, logging.Logger)
                else None
            )
            if isinstance(exc, (PipelineCancelledError, artifact_store.OutputsSupersededError)):
                _log_stage(run_logger, "run_cancelled stage=translation_and_write")
            else:
                _log_stage(
                    run_logger,
                    f"run_failed stage=translation_and_write error={format_exception_chain(exc)!r}",
                )
                if run_logger is not None:
                    run_logger.exception("translation_and_write_failed")
                _write_failure_timings_best_effort(
                    video_path=video_path,
                    artifacts=artifacts,
                    stage="translation_and_write",
                    exc=exc,
                    run_id=effective_run_id,
                )
            raise
        finally:
            _close_artifacts_logger(artifacts)


def _stage_and_publish_outputs(
    *,
    video_path: str,
    artifacts: AsrArtifacts,
    ctx: JobContext,
    plan: ExecutionPlan,
    lease: artifact_store.OutputLease,
    subtitle_options,
    aligned_cache_signature: dict,
    srt_blocks: list[dict],
    translation_cues_count: int,
    translation_request_timings: list[dict],
    translation_api_retry_events: list[dict],
    translation_skipped: bool,
    empty_notice: bool,
    cancel_event,
) -> list[str]:
    output_paths: list[str] = []
    segments = artifacts.segments
    audio_path = artifacts.audio_path
    audio_cache_key = artifacts.audio_cache_key
    audio_cached = artifacts.audio_cached
    asr_details = artifacts.asr_details
    asr_log = artifacts.asr_log
    backend_label = artifacts.backend_label or asr_module.get_backend_label()
    device = artifacts.device
    job_id = lease.job_id
    run_id = lease.run_id
    job_temp_dir = artifacts.job_temp_dir
    pipeline_started = artifacts.pipeline_started
    pipeline_timings = artifacts.pipeline_timings
    aligned_segments_path = artifacts.aligned_segments_path
    logger = artifacts.logger if isinstance(artifacts.logger, logging.Logger) else None
    run_log_path = artifacts.run_log_path
    srt_path = artifacts.srt_path
    transcript_path = artifacts.transcript_path
    asr_manifest_path = artifacts.asr_manifest_path
    bilingual_json_path = artifacts.bilingual_json_path
    timings_path = artifacts.timings_path
    translation_cache_path = artifacts.translation_cache_path
    bilingual = plan.bilingual

    write_started = time.perf_counter()
    _log_stage(logger, "stage_start write_output")
    _raise_if_cancelled(cancel_event)

    with _run_outputs(
        ctx=ctx,
        run_id=run_id,
        job_id=job_id,
        lease=lease,
    ) as outputs:
        if bilingual:
            srt_blocks = outputs.stage(
                srt_path,
                name="srt",
                write=lambda staged: subtitle_module.write_bilingual_srt(
                    srt_blocks,
                    str(staged),
                    options=subtitle_options,
                ),
            )
        else:
            srt_blocks = outputs.stage(
                srt_path,
                name="srt",
                write=lambda staged: subtitle_module.write_srt(
                    srt_blocks,
                    str(staged),
                    options=subtitle_options,
                    language=plan.subtitle_language,
                ),
            )

        outputs.stage_json(
            asr_manifest_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "job_id": job_id,
                "job_temp_dir": job_temp_dir,
                "manifest": build_asr_manifest(segments),
            },
            name="asr_manifest",
        )
        outputs.stage_json(
            transcript_path,
            {
                "backend": backend_label,
                "audio_path": audio_path,
                "chunks": asr_details.get("transcript_chunks", []),
            },
            name="transcript",
        )
        outputs.stage_json(
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
            name="aligned_segments",
        )
        bilingual_payload: dict = {
            "backend": backend_label,
            "timeline_mode": subtitle_options.timeline_mode,
            "blocks": srt_blocks,
            "translation_request_timings": translation_request_timings,
            "translation_api_retry_events": translation_api_retry_events,
        }
        if translation_skipped:
            bilingual_payload["translation_skipped"] = True
        outputs.stage_json(
            bilingual_json_path,
            bilingual_payload,
            name="bilingual_json",
        )
        _raise_if_cancelled(cancel_event)

        pipeline_timings["write_output_s"] = time.perf_counter() - write_started
        pipeline_timings["pipeline_total_s"] = time.perf_counter() - pipeline_started
        _log_stage(
            logger,
            f"stage_done write_output elapsed={pipeline_timings['write_output_s']:.2f}s blocks={len(srt_blocks)}",
        )
        _record_pipeline_cuda_memory(
            asr_details.setdefault("pipeline_cuda_memory", []),
            "write_output_done",
            logger=logger,
            elapsed_s=time.perf_counter() - pipeline_started,
            skip_reason=_asr_details_cuda_skip_reason(asr_details),
        )

        timings_payload: dict = {
            "video_path": video_path,
            "audio_path": audio_path,
            "audio_cached": audio_cached,
            "job_id": job_id,
            "run_id": run_id,
            "job_temp_dir": job_temp_dir,
            "device": device,
            "backend": backend_label,
            "counts": {
                "transcript_chunks": len(asr_details.get("transcript_chunks", [])),
                "segments": len(segments),
                "translation_cues": translation_cues_count,
                "blocks": len(srt_blocks),
            },
            "stage_timings": pipeline_timings,
            "asr_details": _compact_asr_details_for_sidecar(
                asr_details,
                strip_frame_class_track=True,
            ),
            "translation_request_timings": translation_request_timings,
            "translation_api_retry_events": translation_api_retry_events,
            "outputs": {
                "job_temp_dir": job_temp_dir,
                "srt": srt_path,
                "asr_manifest": asr_manifest_path,
                "transcript_json": transcript_path,
                "aligned_segments_json": aligned_segments_path,
                "bilingual_json": bilingual_json_path,
                **_diagnostic_output_paths(
                    timings_path=timings_path,
                    run_log_path=run_log_path,
                ),
            },
            "asr_log": asr_log,
        }
        if translation_skipped:
            timings_payload["translation_skipped"] = True

        _write_timings_with_diagnostics(
            timings_path=timings_path,
            payload=timings_payload,
            run_log_path=run_log_path,
            output_paths=output_paths,
            logger=logger,
        )
        _log_timing_snapshot(logger, pipeline_timings, asr_details)
        _log_stage(
            logger,
            f"run_done srt={_project_relative(srt_path)} timings={_project_relative(timings_path)}",
        )
        _print_timing_summary(pipeline_timings, asr_details)
        output_paths = _publish_outputs(
            outputs,
            output_paths,
            artifacts,
            cancel_event,
            subtitle_cue_count=len(srt_blocks),
        )

    if empty_notice:
        console.print(
            f"\n[bold yellow]未识别到可翻译字幕。[/bold yellow] 已生成空文件：{_project_relative(srt_path)}"
        )
    else:
        console.print(f"\n[bold green]完成！[/bold green] 已保存至：{_project_relative(srt_path)}")
        console.print(f"[dim]详细耗时：{_project_relative(timings_path)}[/dim]")
    _cleanup_pipeline_temp(job_temp_dir, audio_path, translation_cache_path)
    return output_paths


def _run_translation_and_write_impl(
    video_path: str,
    artifacts: AsrArtifacts,
    *,
    ctx: JobContext,
    plan: ExecutionPlan | None = None,
    lease: artifact_store.OutputLease | None = None,
    precomputed_cues: tuple[list[dict], dict] | None = None,
    job_id: str = "",
    run_id: str = "",
    cancel_event=None,
) -> list[str]:
    events._thread_local.video = os.path.basename(video_path)
    _raise_if_cancelled(cancel_event)

    segments = artifacts.segments
    asr_details = artifacts.asr_details
    backend_label = artifacts.backend_label or asr_module.get_backend_label()
    job_id = sanitize_job_id(job_id or ctx.job_id or artifacts.job_id)
    if lease is None:
        lease = _output_lease(
            video_path,
            ctx,
            job_id=job_id,
            run_id=run_id,
            target=artifacts.srt_path,
        )
    run_id = lease.run_id
    if job_id:
        events.set_current_run(job_id, run_id)
    job_temp_dir = artifacts.job_temp_dir
    os.makedirs(job_temp_dir, exist_ok=True)
    pipeline_timings = artifacts.pipeline_timings
    logger = artifacts.logger if isinstance(artifacts.logger, logging.Logger) else None
    translation_cache_path = artifacts.translation_cache_path
    if plan is None:
        plan = _run_plan(artifacts, ctx)
    skip_translation = plan.subtitle_language == "ja"
    subtitle_options = _subtitle_options_for_ctx(ctx)
    aligned_cache_signature = artifacts.aligned_cache_signature
    if aligned_cache_signature is None:
        _, aligned_cache_signature = aligned_cache_expectations_for_ctx(
            ctx,
            backend_label=backend_label,
        )
    _raise_if_cancelled(cancel_event)

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

    # Branch 1: No ASR segments at all.
    if not segments:
        _log_stage(logger, "stage_skip translation reason=no_segments")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        return _stage_and_publish_outputs(
            video_path=video_path,
            artifacts=artifacts,
            ctx=ctx,
            plan=plan,
            lease=lease,
            subtitle_options=subtitle_options,
            aligned_cache_signature=aligned_cache_signature,
            srt_blocks=[],
            translation_cues_count=0,
            translation_request_timings=[],
            translation_api_retry_events=[],
            translation_skipped=skip_translation,
            empty_notice=True,
            cancel_event=cancel_event,
        )

    # Cue planning (reuse precomputed cues when run_translation_and_write ran
    # CPU-only cue planning ahead of the translation backend lease).
    cue_mode = "srt" if skip_translation else ("bilingual" if plan.bilingual else "srt")
    if precomputed_cues is not None:
        translation_segments, _cue_summary = precomputed_cues
    else:
        translation_segments, _cue_summary = _run_cue_plan_stage(
            segments=segments,
            subtitle_options=subtitle_options,
            mode=cue_mode,
            asr_details=asr_details,
            pipeline_timings=pipeline_timings,
            logger=logger,
        )
        artifacts.asr_details = asr_details
    _raise_if_cancelled(cancel_event)

    # Branch 2: Japanese-only mode (`SKIP_TRANSLATION=1`).
    if skip_translation:
        console.print("[yellow]SKIP_TRANSLATION=1，跳过翻译并输出日文 SRT。[/yellow]")
        _log_stage(logger, "stage_skip translation reason=SKIP_TRANSLATION")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        return _stage_and_publish_outputs(
            video_path=video_path,
            artifacts=artifacts,
            ctx=ctx,
            plan=plan,
            lease=lease,
            subtitle_options=subtitle_options,
            aligned_cache_signature=aligned_cache_signature,
            srt_blocks=translation_segments,
            translation_cues_count=len(translation_segments),
            translation_request_timings=[],
            translation_api_retry_events=[],
            translation_skipped=True,
            empty_notice=False,
            cancel_event=cancel_event,
        )

    # Branch 2b: Segments existed, but pre-translation cue planning filtered all of them out.
    if not translation_segments:
        _log_stage(logger, "stage_skip translation reason=no_cues")
        pipeline_timings["translation_context_s"] = 0.0
        pipeline_timings["translation_s"] = 0.0
        return _stage_and_publish_outputs(
            video_path=video_path,
            artifacts=artifacts,
            ctx=ctx,
            plan=plan,
            lease=lease,
            subtitle_options=subtitle_options,
            aligned_cache_signature=aligned_cache_signature,
            srt_blocks=[],
            translation_cues_count=0,
            translation_request_timings=[],
            translation_api_retry_events=[],
            translation_skipped=False,
            empty_notice=True,
            cancel_event=cancel_event,
        )

    # Branch 3: Normal translation.
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

    translation_started = time.perf_counter()
    _log_stage(logger, f"stage_start translation cues={len(translation_segments)}")
    _log_stage(
        logger,
        "translation_settings "
        f"backend={llm_backends.selected_backend_name()} "
        f"model={os.getenv('LLM_MODEL_NAME', '').strip() or '-'} "
        f"reasoning_effort={ctx.llm_reasoning_effort} "
        f"target_lang={ctx.target_lang} "
        f"workers={ctx.translation_max_workers}",
    )
    _raise_if_cancelled(cancel_event)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=not console.is_terminal,
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
            character_reference="",
            max_workers=ctx.translation_max_workers,
            reasoning_effort=ctx.llm_reasoning_effort,
            cache_path=translation_cache_path,
            on_batch_done=_on_translation_done,
            on_progress=_on_translation_progress,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(cancel_event)

    # Lines that finished but should get a human look (for now: passed only
    # under the kana-free retry grammar). Carried per cue like postgate_flags.
    translation_review_by_id = {
        int(entry["id"]): {key: value for key, value in entry.items() if key != "id"}
        for timing in translation_request_timings
        for entry in (timing.get("translation_review") or [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    }

    # Preserve every cue field produced by layout/alignment (`continues_*`,
    # `text_break_*`, `chunk_acoustic_*`, `vocalisation_verdict`, etc.) while
    # updating `zh_text` and keeping empty `postgate_flags` absent on clean cues.
    srt_blocks: list[dict] = []
    for index, (seg, zh_text) in enumerate(zip(translation_segments, zh_texts)):
        ja_text = seg.get("ja_text") or seg.get("text", "")
        block = {
            **seg,
            "start": seg["start"],
            "end": seg["end"],
            "zh_text": zh_text,
            "ja_text": ja_text,
            "words": list(seg.get("words") or []),
            "source_segment_ids": list(seg.get("source_segment_ids") or []),
            "acoustic_start": seg.get("acoustic_start", seg.get("start")),
            "acoustic_end": seg.get("acoustic_end", seg.get("end")),
            "display_start": seg.get("display_start", seg.get("start")),
            "display_end": seg.get("display_end", seg.get("end")),
            "raw_text": seg.get("raw_text", ja_text),
            "display_decision": seg.get("display_decision", "keep"),
            "raw_texts": list(seg.get("raw_texts") or []),
            "primary_cut_candidates": list(seg.get("primary_cut_candidates") or []),
            "weak_cut_candidates": list(seg.get("weak_cut_candidates") or []),
        }
        if seg.get("postgate_flags"):
            block["postgate_flags"] = list(seg["postgate_flags"])
        else:
            block.pop("postgate_flags", None)
        if index in translation_review_by_id:
            block["translation_review"] = dict(translation_review_by_id[index])
        srt_blocks.append(block)

    pipeline_timings["translation_s"] = time.perf_counter() - translation_started
    _log_stage(
        logger,
        f"stage_done translation elapsed={pipeline_timings['translation_s']:.2f}s blocks={len(srt_blocks)}",
    )

    return _stage_and_publish_outputs(
        video_path=video_path,
        artifacts=artifacts,
        ctx=ctx,
        plan=plan,
        lease=lease,
        subtitle_options=subtitle_options,
        aligned_cache_signature=aligned_cache_signature,
        srt_blocks=srt_blocks,
        translation_cues_count=len(translation_segments),
        translation_request_timings=translation_request_timings,
        translation_api_retry_events=translation_api_retry_events,
        translation_skipped=False,
        empty_notice=False,
        cancel_event=cancel_event,
    )
