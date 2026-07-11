#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import (
    DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO,
    qwen_asr_default_model_path,
)
from core.config import load_config


DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV = ",".join(
    f"{repo_id}={batch_size}"
    for repo_id, batch_size in DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO.items()
)
DEFAULT_SPEECH_BOUNDARY_OPERATING_POINT = "qwen-mamba2-speech-island-scorer-v8"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    jobs: Path
    generated: Path
    run_logs: Path
    archived: Path
    summary_json: Path
    summary_md: Path


def safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "item"


def timestamped_label(value: str) -> str:
    label = safe_label(value)
    if re.match(r"^\d{8}_\d{6}_", label):
        return label
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"


def peak_cuda_reserved_mb(snapshots: list[dict[str, Any]]) -> float | None:
    values = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        value = snapshot.get("max_reserved_mb")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(max(values), 1) if values else None


def speech_boundary_operating_point(results: list[dict[str, Any]]) -> str:
    for result in results:
        signature = result.get("boundary_signature") or {}
        if not isinstance(signature, dict):
            continue
        scorer = signature.get("scorer_checkpoint")
        if isinstance(scorer, dict):
            metadata = scorer.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("operating_point"):
                return str(metadata["operating_point"])
            if scorer.get("schema"):
                return str(scorer["schema"])
        if signature.get("operating_point"):
            return str(signature["operating_point"])
    return DEFAULT_SPEECH_BOUNDARY_OPERATING_POINT


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_path_value(value: str | Path | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(project_path(raw))


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def make_paths(task_name: str) -> RunPaths:
    root = PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / timestamped_label(task_name)
    return RunPaths(
        root=root,
        jobs=root / "jobs",
        generated=root / "generated",
        run_logs=root / "run-logs",
        archived=root / "archived",
        summary_json=root / "summary.json",
        summary_md=root / "summary.md",
    )


def ensure_dirs(paths: RunPaths) -> None:
    for path in (paths.root, paths.jobs, paths.generated, paths.run_logs, paths.archived):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def copy_artifact(src: str | Path | None, dst: Path) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def resolve_run_log(paths: RunPaths, job_id: str, artifacts=None) -> Path | None:
    candidates: list[Path] = []
    if artifacts is not None and getattr(artifacts, "run_log_path", None):
        candidates.append(Path(artifacts.run_log_path))
    candidates.extend(sorted(paths.run_logs.glob(f"**/*_{job_id}_*.run.log")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda item: item.stat().st_mtime)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default)).strip()
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default)).strip()
    return float(raw) if raw else default


def _env_optional_float(key: str) -> float | None:
    raw = os.getenv(key, "").strip()
    return float(raw) if raw else None


def configure_env(args: argparse.Namespace) -> None:
    os.environ.pop("ASR_STAGE_WORKER_MODE", None)
    os.environ.pop("ASR_WORKER_MODE", None)
    os.environ.pop("ASR_WORKER_MODE_BY_REPO", None)
    os.environ["ASR_BACKEND"] = args.asr_backend
    os.environ["ASR_BOUNDARY_BACKEND"] = "speech_boundary_ja"
    os.environ["ASR_MODEL_PATH"] = project_path_value(args.asr_model_path)
    os.environ["ASR_MODEL_ID"] = ""
    os.environ["ASR_DTYPE"] = args.asr_dtype
    os.environ["ASR_ATTENTION"] = args.asr_attention
    os.environ["ASR_BATCH_SIZE"] = str(args.asr_batch_size)
    os.environ["ASR_BATCH_SIZE_BY_REPO"] = os.getenv(
        "ASR_BATCH_SIZE_BY_REPO",
        DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV,
    )
    os.environ["TRANSCRIPTION_TIMEOUT_S"] = str(args.transcription_timeout_s)
    os.environ["ASR_MAX_NEW_TOKENS"] = str(args.asr_max_new_tokens)
    if args.boundary_feature_frame_hop_s is not None:
        os.environ["BOUNDARY_FEATURE_FRAME_HOP_S"] = str(args.boundary_feature_frame_hop_s)
    if args.outer_edge_refiner_model_path_by_repo.strip():
        os.environ["OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO"] = (
            args.outer_edge_refiner_model_path_by_repo
        )
    else:
        os.environ.pop("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", None)
    if args.semantic_split_model_path_by_repo.strip():
        os.environ["SEMANTIC_SPLIT_MODEL_PATH_BY_REPO"] = (
            args.semantic_split_model_path_by_repo
        )
    else:
        os.environ.pop("SEMANTIC_SPLIT_MODEL_PATH_BY_REPO", None)
    if args.cut_edge_refiner_model_path_by_repo.strip():
        os.environ["CUT_EDGE_REFINER_MODEL_PATH_BY_REPO"] = (
            args.cut_edge_refiner_model_path_by_repo
        )
    else:
        os.environ.pop("CUT_EDGE_REFINER_MODEL_PATH_BY_REPO", None)
    os.environ["OUTER_EDGE_REFINER_DEVICE"] = args.outer_edge_refiner_device
    os.environ["SEMANTIC_SPLIT_DEVICE"] = args.semantic_split_device
    os.environ["CUT_EDGE_REFINER_DEVICE"] = args.cut_edge_refiner_device
    os.environ["PRE_ASR_CUEQC_ENABLED"] = "1" if args.pre_asr_cueqc_enabled else "0"
    if args.pre_asr_cueqc_model_path_by_repo.strip():
        os.environ["PRE_ASR_CUEQC_MODEL_PATH_BY_REPO"] = args.pre_asr_cueqc_model_path_by_repo
    else:
        os.environ.pop("PRE_ASR_CUEQC_MODEL_PATH_BY_REPO", None)
    os.environ["PRE_ASR_CUEQC_DEVICE"] = args.pre_asr_cueqc_device
    if args.pre_asr_cueqc_drop_threshold is None:
        os.environ.pop("PRE_ASR_CUEQC_DROP_THRESHOLD", None)
    else:
        os.environ["PRE_ASR_CUEQC_DROP_THRESHOLD"] = str(
            args.pre_asr_cueqc_drop_threshold
        )
    os.environ["KEEP_ASR_CHUNKS"] = "1" if args.keep_asr_chunks else "0"
    os.environ["BOUNDARY_CACHE_ENABLED"] = "1" if args.boundary_cache else "0"
    os.environ["SPEECH_BOUNDARY_JA_THRESHOLD"] = str(args.speech_boundary_threshold)
    os.environ["SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"] = str(args.speech_boundary_speech_on_threshold)
    os.environ["SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"] = str(args.speech_boundary_speech_off_threshold)
    os.environ["SPEECH_BOUNDARY_JA_FRAME_DILATION_S"] = str(args.speech_boundary_frame_dilation_s)
    os.environ["SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE"] = str(args.speech_boundary_split_score_quantile)
    os.environ["SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE"] = str(
        args.speech_boundary_split_prominence_quantile
    )
    os.environ["SPEECH_BOUNDARY_JA_SPLIT_SMOOTH_S"] = str(args.speech_boundary_split_smooth_s)
    os.environ["SPEECH_BOUNDARY_JA_SPLIT_NMS_S"] = str(args.speech_boundary_split_nms_s)
    os.environ["SPEECH_BOUNDARY_JA_SPLIT_SNAP_S"] = str(args.speech_boundary_split_snap_s)
    os.environ["SPEECH_BOUNDARY_JA_MIN_SPLIT_SEGMENT_S"] = str(args.speech_boundary_min_split_segment_s)
    os.environ["SPEECH_BOUNDARY_JA_PTM"] = args.speech_boundary_ptm
    if str(args.speech_boundary_model_path or "").strip():
        os.environ["SPEECH_BOUNDARY_JA_MODEL_PATH"] = project_path_value(args.speech_boundary_model_path)
    else:
        os.environ.pop("SPEECH_BOUNDARY_JA_MODEL_PATH", None)
    os.environ["SPEECH_BOUNDARY_JA_DEVICE"] = args.speech_boundary_device
    os.environ["SPEECH_BOUNDARY_JA_DTYPE"] = args.speech_boundary_dtype
    if args.speech_boundary_scorer_checkpoint_by_repo.strip():
        os.environ["SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO"] = (
            args.speech_boundary_scorer_checkpoint_by_repo
        )
    else:
        os.environ.pop("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", None)
    os.environ["SPEECH_BOUNDARY_JA_SCORER_DEVICE"] = args.speech_boundary_scorer_device
    os.environ["SPEECH_BOUNDARY_JA_WINDOW_S"] = str(args.speech_boundary_window_s)
    os.environ["SPEECH_BOUNDARY_JA_OVERLAP_S"] = str(args.speech_boundary_overlap_s)
    os.environ["SPEECH_BOUNDARY_JA_MIN_SEGMENT_S"] = str(args.speech_boundary_min_segment_s)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_context(*, args: argparse.Namespace, paths: RunPaths, video: Path):
    from core.job_context import JobContext
    from pipeline.ids import sanitize_job_id

    job_id = sanitize_job_id(f"{video.stem}_{safe_label(args.label)}")
    job_temp_dir = paths.jobs / job_id
    advanced = {
        "ASR_BACKEND": args.asr_backend,
        "ASR_BOUNDARY_BACKEND": "speech_boundary_ja",
        "ASR_MODEL_PATH": project_path_value(args.asr_model_path),
        "TRANSCRIPTION_TIMEOUT_S": str(args.transcription_timeout_s),
        "ASR_MAX_NEW_TOKENS": str(args.asr_max_new_tokens),
        "ASR_BATCH_SIZE": args.asr_batch_size,
        "ASR_BATCH_SIZE_BY_REPO": os.getenv(
            "ASR_BATCH_SIZE_BY_REPO",
            DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV,
        ),
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO": os.getenv(
            "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", ""
        ),
        "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO": os.getenv(
            "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO", ""
        ),
        "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO": os.getenv(
            "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO", ""
        ),
        "OUTER_EDGE_REFINER_DEVICE": os.getenv("OUTER_EDGE_REFINER_DEVICE", "auto"),
        "SEMANTIC_SPLIT_DEVICE": os.getenv("SEMANTIC_SPLIT_DEVICE", "auto"),
        "CUT_EDGE_REFINER_DEVICE": os.getenv("CUT_EDGE_REFINER_DEVICE", "auto"),
        "PRE_ASR_CUEQC_ENABLED": "1" if args.pre_asr_cueqc_enabled else "0",
        "PRE_ASR_CUEQC_MODEL_PATH_BY_REPO": os.getenv("PRE_ASR_CUEQC_MODEL_PATH_BY_REPO", ""),
        "PRE_ASR_CUEQC_DEVICE": os.getenv("PRE_ASR_CUEQC_DEVICE", "auto"),
        "PRE_ASR_CUEQC_DROP_THRESHOLD": (
            ""
            if args.pre_asr_cueqc_drop_threshold is None
            else str(args.pre_asr_cueqc_drop_threshold)
        ),
        "QUALITY_REPORT_ENABLED": "1",
        "QUALITY_REPORT_DIR": str(paths.root / "quality_reports"),
        "QC_HARD_FAIL": "0",
        "RUN_LOG_ENABLED": "1",
        "RUN_LOG_DIR": str(paths.run_logs),
        "BOUNDARY_CACHE_ENABLED": "1" if args.boundary_cache else "0",
        "BOUNDARY_CACHE_DIR": str(paths.root / "boundary-cache"),
        "SPEECH_BOUNDARY_JA_THRESHOLD": str(args.speech_boundary_threshold),
        "SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD": str(args.speech_boundary_speech_on_threshold),
        "SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD": str(args.speech_boundary_speech_off_threshold),
        "SPEECH_BOUNDARY_JA_FRAME_DILATION_S": str(args.speech_boundary_frame_dilation_s),
        "SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE": str(args.speech_boundary_split_score_quantile),
        "SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE": str(
            args.speech_boundary_split_prominence_quantile
        ),
        "SPEECH_BOUNDARY_JA_SPLIT_SMOOTH_S": str(args.speech_boundary_split_smooth_s),
        "SPEECH_BOUNDARY_JA_SPLIT_NMS_S": str(args.speech_boundary_split_nms_s),
        "SPEECH_BOUNDARY_JA_SPLIT_SNAP_S": str(args.speech_boundary_split_snap_s),
        "SPEECH_BOUNDARY_JA_MIN_SPLIT_SEGMENT_S": str(args.speech_boundary_min_split_segment_s),
        "SPEECH_BOUNDARY_JA_PTM": args.speech_boundary_ptm,
        "SPEECH_BOUNDARY_JA_MODEL_PATH": project_path_value(args.speech_boundary_model_path),
        "SPEECH_BOUNDARY_JA_DEVICE": args.speech_boundary_device,
        "SPEECH_BOUNDARY_JA_DTYPE": args.speech_boundary_dtype,
        "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO": os.getenv(
            "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
            "",
        ),
        "SPEECH_BOUNDARY_JA_SCORER_DEVICE": os.getenv(
            "SPEECH_BOUNDARY_JA_SCORER_DEVICE",
            "auto",
        ),
        "SPEECH_BOUNDARY_JA_WINDOW_S": str(args.speech_boundary_window_s),
        "SPEECH_BOUNDARY_JA_OVERLAP_S": str(args.speech_boundary_overlap_s),
        "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S": str(args.speech_boundary_min_segment_s),
        "KEEP_ASR_CHUNKS": "1" if args.keep_asr_chunks else "0",
    }
    spec = SimpleNamespace(
        asr_backend=args.asr_backend,
        subtitle_mode=args.subtitle_mode,
        skip_translation=not args.translate,
        output_dir=str(paths.generated / job_id),
        keep_quality_report=True,
        keep_temp_files=True,
        run_log_enabled=True,
        run_log_dir=str(paths.run_logs),
        translation_max_workers=args.translation_max_workers,
        target_lang=args.target_lang,
        translation_glossary=os.getenv("TRANSLATION_GLOSSARY", ""),
        llm_api_format=os.getenv("LLM_API_FORMAT", "chat"),
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "xhigh"),
        advanced=advanced,
    )
    if args.boundary_feature_frame_hop_s is not None:
        advanced["BOUNDARY_FEATURE_FRAME_HOP_S"] = str(args.boundary_feature_frame_hop_s)
    ctx = JobContext.from_spec(
        spec,
        job_id=job_id,
        job_temp_dir=str(job_temp_dir),
        cache_path=str(job_temp_dir / "translation_cache.jsonl"),
    )
    return ctx


def run_video(args: argparse.Namespace, paths: RunPaths, video: Path) -> dict[str, Any]:
    import main as pipeline_main

    started = time.perf_counter()
    ctx = build_context(args=args, paths=paths, video=video)
    print(
        f"=== START {video.name} label={args.label} "
        f"boundary=speech_boundary_ja speech_on={args.speech_boundary_speech_on_threshold:g} "
        f"speech_off={args.speech_boundary_speech_off_threshold:g} "
        f"asr={args.asr_backend} ===",
        flush=True,
    )
    artifacts = pipeline_main.run_asr_alignment(str(video), ctx=ctx, job_id=ctx.job_id)
    output_paths = pipeline_main.run_translation_and_write(
        str(video),
        artifacts,
        ctx=ctx,
        job_id=ctx.job_id,
    )
    elapsed = time.perf_counter() - started
    run_log = resolve_run_log(paths, ctx.job_id, artifacts)
    artifact_dir = paths.archived / video.stem
    copied = {
        "srt": copy_artifact(artifacts.srt_path, artifact_dir / Path(artifacts.srt_path).name),
        "timings": copy_artifact(artifacts.timings_path, artifact_dir / Path(artifacts.timings_path).name),
        "aligned_segments": copy_artifact(
            artifacts.aligned_segments_path,
            artifact_dir / Path(artifacts.aligned_segments_path).name,
        ),
        "transcript": copy_artifact(
            artifacts.transcript_path,
            artifact_dir / Path(artifacts.transcript_path).name,
        ),
        "asr_manifest": copy_artifact(
            artifacts.asr_manifest_path,
            artifact_dir / Path(artifacts.asr_manifest_path).name,
        ),
        "bilingual_json": copy_artifact(
            artifacts.bilingual_json_path,
            artifact_dir / Path(artifacts.bilingual_json_path).name,
        ),
        "quality_report": copy_artifact(
            artifacts.quality_report_path,
            artifact_dir / Path(artifacts.quality_report_path).name
            if artifacts.quality_report_path
            else artifact_dir / "quality_report.md",
        ),
        "run_log": str(run_log) if run_log else "",
    }
    timings = read_json(artifacts.timings_path)
    asr_details = timings.get("asr_details") if isinstance(timings.get("asr_details"), dict) else {}
    cuda_memory = asr_details.get("cuda_memory") if isinstance(asr_details.get("cuda_memory"), list) else []
    pipeline_cuda_memory = (
        asr_details.get("pipeline_cuda_memory")
        if isinstance(asr_details.get("pipeline_cuda_memory"), list)
        else []
    )
    result = {
        "video": video.name,
        "video_path": project_rel(video),
        "job_id": ctx.job_id,
        "status": "done",
        "elapsed_s": round(elapsed, 3),
        "paths": {key: value for key, value in copied.items() if value},
        "raw_output_paths": [project_rel(path) for path in output_paths],
        "counts": timings.get("counts", {}),
        "stage_timings": timings.get("stage_timings", {}),
        "cuda_memory_peak_reserved_mb": peak_cuda_reserved_mb(cuda_memory),
        "pipeline_cuda_memory_peak_reserved_mb": peak_cuda_reserved_mb(pipeline_cuda_memory),
        "cuda_memory": cuda_memory,
        "pipeline_cuda_memory": pipeline_cuda_memory,
        "boundary_signature": asr_details.get("boundary_signature", {}),
    }
    print(
        f"=== DONE {video.name} elapsed={elapsed:.1f}s "
        f"segments={result['counts'].get('segments')} "
        f"blocks={result['counts'].get('blocks')} ===",
        flush=True,
    )
    return result


def write_summary(paths: RunPaths, args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    payload = {
        "task": args.task_name,
        "label": args.label,
        "asr_backend": args.asr_backend,
        "asr_model_path": project_rel(project_path_value(args.asr_model_path)),
        "boundary_backend": "speech_boundary_ja",
        "speech_boundary_operating_point": speech_boundary_operating_point(results),
        "speech_boundary_threshold": args.speech_boundary_threshold,
        "speech_boundary_speech_on_threshold": args.speech_boundary_speech_on_threshold,
        "speech_boundary_speech_off_threshold": args.speech_boundary_speech_off_threshold,
        "speech_boundary_frame_dilation_s": args.speech_boundary_frame_dilation_s,
        "speech_boundary_split_strategy": "acoustic_proposal_then_semantic_split",
        "speech_boundary_split_score_quantile": args.speech_boundary_split_score_quantile,
        "speech_boundary_split_prominence_quantile": args.speech_boundary_split_prominence_quantile,
        "speech_boundary_split_smooth_s": args.speech_boundary_split_smooth_s,
        "speech_boundary_split_nms_s": args.speech_boundary_split_nms_s,
        "speech_boundary_split_snap_s": args.speech_boundary_split_snap_s,
        "speech_boundary_min_split_segment_s": args.speech_boundary_min_split_segment_s,
        "speech_boundary_scorer_checkpoint_by_repo": args.speech_boundary_scorer_checkpoint_by_repo,
        "asr_batch_size": args.asr_batch_size,
        "pre_asr_cueqc_enabled": bool(args.pre_asr_cueqc_enabled),
        "pre_asr_cueqc_drop_threshold": args.pre_asr_cueqc_drop_threshold,
        "boundary_planner": {
            "feature_frame_hop_s": args.boundary_feature_frame_hop_s,
            "order": [
                "speech_island_scorer",
                "outer_edge_refiner",
                "semantic_split_model",
                "cut_edge_refiner",
            ],
        },
        "translate": bool(args.translate),
        "results": results,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SpeechBoundary-JA Full Workflow",
        "",
        f"- ASR: `{args.asr_backend}`",
        "- Subtitle timing: `boundary chunk timeline`",
        f"- ASR model: `{project_rel(project_path_value(args.asr_model_path)) or 'auto'}`",
        (
            f"- Boundary: `speech_boundary_ja` speech on/off "
            f"`{args.speech_boundary_speech_on_threshold:g}` / "
            f"`{args.speech_boundary_speech_off_threshold:g}`, "
            f"frame dilation `{args.speech_boundary_frame_dilation_s:g}s`, "
            f"semantic split model"
        ),
        f"- Pre-ASR CueQC: `{'on' if args.pre_asr_cueqc_enabled else 'off'}`",
        f"- Translation: `{'on' if args.translate else 'off'}`",
        f"- Runtime root: `{project_rel(paths.root)}`",
        "",
        "| video | status | segments | blocks | asr_s | total_s | cuda_peak_mb | srt |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        timings = result.get("stage_timings") or {}
        counts = result.get("counts") or {}
        paths_payload = result.get("paths") or {}
        lines.append(
            f"| `{result.get('video')}` | `{result.get('status')}` | "
            f"{int(counts.get('segments') or 0)} | "
            f"{int(counts.get('blocks') or 0)} | "
            f"{float(timings.get('asr_alignment_total_s') or 0.0):.1f} | "
            f"{float(timings.get('pipeline_total_s') or result.get('elapsed_s') or 0.0):.1f} | "
            f"{float(result.get('cuda_memory_peak_reserved_mb') or 0.0):.1f} | "
            f"`{project_rel(paths_payload.get('srt'))}` |"
        )
    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_config()
    parser = argparse.ArgumentParser(
        description="Run project full workflow with SpeechBoundary-JA repo-id scorer, Boundary Refiner, and Qwen3-ASR."
    )
    parser.add_argument("--video", action="append", required=True, help="Video path or stem. Repeatable.")
    parser.add_argument("--task-name", default="full-workflow-qwen200k")
    parser.add_argument("--label", default="speech_boundary_ja_qwen200k")
    parser.add_argument(
        "--asr-backend",
        default=os.getenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"),
    )
    parser.add_argument(
        "--asr-model-path",
        default=os.getenv("ASR_MODEL_PATH", ""),
    )
    parser.add_argument("--asr-dtype", default=os.getenv("ASR_DTYPE", "bfloat16"))
    parser.add_argument("--asr-attention", default=os.getenv("ASR_ATTENTION", "sdpa"))
    parser.add_argument("--asr-batch-size", default=os.getenv("ASR_BATCH_SIZE", "auto"))
    parser.add_argument("--asr-max-new-tokens", type=int, default=_env_int("ASR_MAX_NEW_TOKENS", 128))
    parser.add_argument("--transcription-timeout-s", type=int, default=_env_int("TRANSCRIPTION_TIMEOUT_S", 300))
    parser.add_argument("--subtitle-mode", default="zh")
    parser.add_argument("--translate", action="store_true", help="Run LLM translation too. Default outputs Japanese SRT.")
    parser.add_argument("--target-lang", default=os.getenv("TARGET_LANG", "简体中文"))
    parser.add_argument("--translation-max-workers", type=int, default=1)
    parser.add_argument(
        "--boundary-feature-frame-hop-s",
        dest="boundary_feature_frame_hop_s",
        type=float,
        default=None,
        help=(
            "Override Boundary feature/score grid hop in seconds. This is not video FPS; "
            "subtitle frame timing uses a fixed 24000/1001 base, independent of source FPS."
        ),
    )
    parser.add_argument(
        "--outer-edge-refiner-model-path-by-repo",
        default=os.getenv("OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO", ""),
        help="Optional repo-id checkpoint map for outer_edge_refiner_v1.",
    )
    parser.add_argument(
        "--semantic-split-model-path-by-repo",
        default=os.getenv("SEMANTIC_SPLIT_MODEL_PATH_BY_REPO", ""),
        help="Optional repo-id checkpoint map for semantic_split_model_v2.",
    )
    parser.add_argument(
        "--cut-edge-refiner-model-path-by-repo",
        default=os.getenv("CUT_EDGE_REFINER_MODEL_PATH_BY_REPO", ""),
        help="Optional repo-id checkpoint map for cut_edge_refiner_v1.",
    )
    parser.add_argument(
        "--pre-asr-cueqc-enabled",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("PRE_ASR_CUEQC_ENABLED", False),
        help="Run Pre-ASR CueQC v12 after semantic boundary planning and before ASR export.",
    )
    parser.add_argument(
        "--pre-asr-cueqc-model-path-by-repo",
        default=os.getenv("PRE_ASR_CUEQC_MODEL_PATH_BY_REPO", ""),
        help="Optional repo-id checkpoint map for cueqc_pre_asr_semantic_chunk_v12_binary.",
    )
    parser.add_argument("--pre-asr-cueqc-device", default=os.getenv("PRE_ASR_CUEQC_DEVICE", "auto"))
    parser.add_argument(
        "--pre-asr-cueqc-drop-threshold",
        type=float,
        default=_env_optional_float("PRE_ASR_CUEQC_DROP_THRESHOLD"),
        help="Optional runtime override. Unset uses the active checkpoint decision_config.",
    )
    parser.add_argument(
        "--outer-edge-refiner-device",
        default=os.getenv("OUTER_EDGE_REFINER_DEVICE", "auto"),
    )
    parser.add_argument(
        "--semantic-split-device",
        default=os.getenv("SEMANTIC_SPLIT_DEVICE", "auto"),
    )
    parser.add_argument(
        "--cut-edge-refiner-device",
        default=os.getenv("CUT_EDGE_REFINER_DEVICE", "auto"),
    )
    parser.add_argument("--keep-asr-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-cache", action=argparse.BooleanOptionalAction, default=_env_bool("BOUNDARY_CACHE_ENABLED", True))
    parser.add_argument("--speech-boundary-threshold", dest="speech_boundary_threshold", type=float, default=_env_float("SPEECH_BOUNDARY_JA_THRESHOLD", 0.15))
    parser.add_argument(
        "--speech-boundary-speech-on-threshold",
        dest="speech_boundary_speech_on_threshold",
        type=float,
        default=_env_optional_float("SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD"),
        help="Speech activation threshold. Defaults to --speech-boundary-threshold.",
    )
    parser.add_argument(
        "--speech-boundary-speech-off-threshold",
        dest="speech_boundary_speech_off_threshold",
        type=float,
        default=_env_optional_float("SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD"),
        help="Speech deactivation threshold. Defaults to --speech-boundary-threshold.",
    )
    parser.add_argument("--speech-boundary-frame-dilation-s", dest="speech_boundary_frame_dilation_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_FRAME_DILATION_S", 0.2))
    parser.add_argument("--speech-boundary-split-score-quantile", dest="speech_boundary_split_score_quantile", type=float, default=_env_float("SPEECH_BOUNDARY_JA_SPLIT_SCORE_QUANTILE", 0.10))
    parser.add_argument("--speech-boundary-split-prominence-quantile", dest="speech_boundary_split_prominence_quantile", type=float, default=_env_float("SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE_QUANTILE", 0.10))
    parser.add_argument("--speech-boundary-split-smooth-s", dest="speech_boundary_split_smooth_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_SPLIT_SMOOTH_S", 0.08))
    parser.add_argument("--speech-boundary-split-nms-s", dest="speech_boundary_split_nms_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_SPLIT_NMS_S", 0.12))
    parser.add_argument("--speech-boundary-split-snap-s", dest="speech_boundary_split_snap_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_SPLIT_SNAP_S", 0.10))
    parser.add_argument("--speech-boundary-min-split-segment-s", dest="speech_boundary_min_split_segment_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_MIN_SPLIT_SEGMENT_S", 0.08))
    parser.add_argument(
        "--speech-boundary-ptm",
        dest="speech_boundary_ptm",
        default=os.getenv("SPEECH_BOUNDARY_JA_PTM", ""),
    )
    parser.add_argument(
        "--speech-boundary-model-path",
        dest="speech_boundary_model_path",
        default=os.getenv("SPEECH_BOUNDARY_JA_MODEL_PATH", ""),
    )
    parser.add_argument("--speech-boundary-device", dest="speech_boundary_device", default=os.getenv("SPEECH_BOUNDARY_JA_DEVICE", "auto"))
    parser.add_argument("--speech-boundary-dtype", dest="speech_boundary_dtype", default=os.getenv("SPEECH_BOUNDARY_JA_DTYPE", "bfloat16"))
    parser.add_argument(
        "--speech-boundary-scorer-checkpoint-by-repo",
        dest="speech_boundary_scorer_checkpoint_by_repo",
        default=os.getenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", ""),
        help=(
            "Optional repo-id scorer map: '<repo_id>=<speech_island_scorer_v8.pt>'"
            "[,<repo_id>=...]'. Empty uses the registered repo-id scorer when available."
        ),
    )
    parser.add_argument(
        "--speech-boundary-scorer-device",
        dest="speech_boundary_scorer_device",
        default=os.getenv("SPEECH_BOUNDARY_JA_SCORER_DEVICE", "auto"),
    )
    parser.add_argument("--speech-boundary-window-s", dest="speech_boundary_window_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_WINDOW_S", 20.0))
    parser.add_argument("--speech-boundary-overlap-s", dest="speech_boundary_overlap_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_OVERLAP_S", 4.0))
    parser.add_argument("--speech-boundary-min-segment-s", dest="speech_boundary_min_segment_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", 0.05))
    args = parser.parse_args(argv)
    if args.speech_boundary_speech_on_threshold is None:
        args.speech_boundary_speech_on_threshold = args.speech_boundary_threshold
    if args.speech_boundary_speech_off_threshold is None:
        args.speech_boundary_speech_off_threshold = args.speech_boundary_threshold
    if not str(args.speech_boundary_ptm or "").strip():
        args.speech_boundary_ptm = args.asr_backend
    if not str(args.speech_boundary_model_path or "").strip():
        args.speech_boundary_model_path = qwen_asr_default_model_path(args.speech_boundary_ptm)
    if args.speech_boundary_threshold < 0:
        parser.error("--speech-boundary-threshold must be non-negative")
    if args.speech_boundary_speech_on_threshold < 0:
        parser.error("--speech-boundary-speech-on-threshold must be non-negative")
    if args.speech_boundary_speech_off_threshold < 0:
        parser.error("--speech-boundary-speech-off-threshold must be non-negative")
    if args.speech_boundary_speech_on_threshold < args.speech_boundary_speech_off_threshold:
        parser.error("--speech-boundary-speech-on-threshold must be >= --speech-boundary-speech-off-threshold")
    if args.speech_boundary_frame_dilation_s < 0:
        parser.error("--speech-boundary-frame-dilation-s must be non-negative")
    for name in (
        "speech_boundary_split_smooth_s",
        "speech_boundary_split_nms_s",
        "speech_boundary_split_snap_s",
        "speech_boundary_min_split_segment_s",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("speech_boundary_split_score_quantile", "speech_boundary_split_prominence_quantile"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if (
        args.pre_asr_cueqc_drop_threshold is not None
        and not 0.0 <= args.pre_asr_cueqc_drop_threshold <= 1.0
    ):
        parser.error("--pre-asr-cueqc-drop-threshold must be between 0 and 1")
    if args.speech_boundary_window_s <= 0:
        parser.error("--speech-boundary-window-s must be positive")
    if args.speech_boundary_overlap_s < 0 or args.speech_boundary_overlap_s >= args.speech_boundary_window_s:
        parser.error("--speech-boundary-overlap-s must be non-negative and smaller than window")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_env(args)
    paths = make_paths(args.task_name)
    ensure_dirs(paths)
    results: list[dict[str, Any]] = []
    for item in args.video:
        video = project_path(item)
        if not video.exists() and not video.suffix:
            candidate = PROJECT_ROOT / "video" / f"{item}.mp4"
            if candidate.exists():
                video = candidate.resolve()
        if not video.exists():
            results.append({"video": item, "status": "failed", "error": "video not found"})
            continue
        try:
            results.append(run_video(args, paths, video))
        except Exception as exc:
            traceback_text = traceback.format_exc()
            print(f"=== FAIL {video.name}: {exc} ===", flush=True)
            print(traceback_text, flush=True)
            results.append(
                {
                    "video": video.name,
                    "video_path": project_rel(video),
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback_text,
                }
            )
    write_summary(paths, args, results)
    print(f"summary_json={project_rel(paths.summary_json)}", flush=True)
    print(f"summary_md={project_rel(paths.summary_md)}", flush=True)
    return 1 if any(result.get("status") == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
