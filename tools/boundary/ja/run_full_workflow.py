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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO
from core.config import load_config


DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV = ",".join(
    f"{repo_id}={batch_size}"
    for repo_id, batch_size in DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO.items()
)


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
    root = PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / safe_label(task_name)
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
    candidates.extend(sorted(paths.run_logs.glob(f"*_{job_id}_*.run.log")))
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


def configure_env(args: argparse.Namespace) -> None:
    os.environ["ASR_BACKEND"] = args.asr_backend
    os.environ["ASR_BOUNDARY_BACKEND"] = "speech_boundary_ja"
    os.environ["ASR_MODEL_PATH"] = project_path_value(args.asr_model_path)
    os.environ.setdefault("ASR_MODEL_ID", os.getenv("ASR_MODEL_ID", ""))
    os.environ.setdefault("ALIGNER_MODEL_PATH", project_path_value(args.aligner_model_path))
    os.environ.setdefault("ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
    os.environ.setdefault("ASR_WORKER_MODE", args.asr_worker_mode)
    os.environ.setdefault("ASR_DTYPE", args.asr_dtype)
    os.environ.setdefault("ASR_ATTENTION", args.asr_attention)
    os.environ.setdefault("ASR_BATCH_SIZE", args.asr_batch_size)
    os.environ.setdefault(
        "ASR_BATCH_SIZE_BY_REPO",
        os.getenv("ASR_BATCH_SIZE_BY_REPO", DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV),
    )
    os.environ.setdefault("ALIGNER_BATCH_SIZE", str(args.aligner_batch_size))
    os.environ.setdefault("ALIGN_LONG_CHUNK_BATCH_SIZE", str(args.align_long_chunk_batch_size))
    os.environ.setdefault("TRANSCRIPTION_TIMEOUT_S", str(args.transcription_timeout_s))
    os.environ.setdefault("TRANSCRIPTION_MAX_NEW_TOKENS", str(args.transcription_max_new_tokens))
    os.environ.setdefault("ASR_MAX_NEW_TOKENS", str(args.asr_max_new_tokens))
    if args.boundary_feature_frame_hop_s is not None:
        os.environ.setdefault(
            "BOUNDARY_FEATURE_FRAME_HOP_S",
            str(args.boundary_feature_frame_hop_s),
        )
    os.environ.setdefault("BOUNDARY_REFINER_ENABLED", "1")
    os.environ.setdefault("BOUNDARY_REFINER_MODEL_PATH", args.boundary_refiner_model_path)
    os.environ.setdefault("BOUNDARY_REFINER_BACKBONE", args.boundary_refiner_backbone)
    os.environ.setdefault("BOUNDARY_REFINER_DEVICE", args.boundary_refiner_device)
    os.environ.setdefault("BOUNDARY_REFINER_THRESHOLD", str(args.boundary_refiner_threshold))
    os.environ.setdefault(
        "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S",
        str(args.boundary_planner_max_core_chunk_s),
    )
    os.environ.setdefault(
        "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S",
        str(args.boundary_planner_max_padded_chunk_s),
    )
    os.environ.setdefault("BOUNDARY_PLANNER_TARGET_CHUNK_S", str(args.boundary_planner_target_chunk_s))
    os.environ.setdefault("BOUNDARY_PLANNER_MIN_CHUNK_S", str(args.boundary_planner_min_chunk_s))
    os.environ.setdefault("BOUNDARY_PLANNER_START_WEIGHT", str(args.boundary_planner_start_weight))
    os.environ.setdefault("BOUNDARY_CONTEXT_MAX_PADDING_S", str(args.boundary_context_max_padding_s))
    os.environ.setdefault(
        "BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S",
        str(args.boundary_context_max_speech_overlap_s),
    )
    os.environ.setdefault(
        "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT",
        str(args.boundary_planner_max_splits_per_segment),
    )
    os.environ.setdefault(
        "BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE",
        str(args.boundary_planner_sequence_batch_size),
    )
    os.environ.setdefault("KEEP_ASR_CHUNKS", "1" if args.keep_asr_chunks else "0")
    os.environ.setdefault("BOUNDARY_CACHE_ENABLED", "1" if args.boundary_cache else "0")
    os.environ["SPEECH_BOUNDARY_JA_THRESHOLD"] = str(args.speech_boundary_threshold)
    os.environ["SPEECH_BOUNDARY_JA_PAD_S"] = str(args.speech_boundary_pad_s)
    os.environ["SPEECH_BOUNDARY_JA_PTM"] = args.speech_boundary_ptm
    os.environ["SPEECH_BOUNDARY_JA_MODEL_PATH"] = project_path_value(args.speech_boundary_model_path)
    os.environ["SPEECH_BOUNDARY_JA_DEVICE"] = args.speech_boundary_device
    os.environ["SPEECH_BOUNDARY_JA_DTYPE"] = args.speech_boundary_dtype
    os.environ["SPEECH_BOUNDARY_JA_WINDOW_S"] = str(args.speech_boundary_window_s)
    os.environ["SPEECH_BOUNDARY_JA_OVERLAP_S"] = str(args.speech_boundary_overlap_s)
    os.environ["SPEECH_BOUNDARY_JA_MIN_SEGMENT_S"] = str(args.speech_boundary_min_segment_s)
    os.environ["SPEECH_BOUNDARY_JA_MERGE_GAP_S"] = str(args.speech_boundary_merge_gap_s)
    os.environ["SPEECH_BOUNDARY_JA_CUT_THRESHOLD"] = str(args.speech_boundary_cut_threshold)
    os.environ["SPEECH_BOUNDARY_JA_APPLY_CUT_TO_SPEECH"] = "1" if args.speech_boundary_apply_cut_to_speech else "0"
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
        "ALIGNER_MODEL_PATH": project_path_value(args.aligner_model_path),
        "ASR_WORKER_MODE": args.asr_worker_mode,
        "ASR_CONTEXT": args.asr_context,
        "TRANSCRIPTION_TIMEOUT_S": str(args.transcription_timeout_s),
        "TRANSCRIPTION_MAX_NEW_TOKENS": str(args.transcription_max_new_tokens),
        "ASR_MAX_NEW_TOKENS": str(args.asr_max_new_tokens),
        "ASR_BATCH_SIZE": args.asr_batch_size,
        "ASR_BATCH_SIZE_BY_REPO": os.getenv(
            "ASR_BATCH_SIZE_BY_REPO",
            DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV,
        ),
        "ALIGNER_BATCH_SIZE": str(args.aligner_batch_size),
        "ALIGN_LONG_CHUNK_BATCH_SIZE": str(args.align_long_chunk_batch_size),
        "BOUNDARY_REFINER_ENABLED": os.getenv("BOUNDARY_REFINER_ENABLED", "1"),
        "BOUNDARY_REFINER_MODEL_PATH": os.getenv("BOUNDARY_REFINER_MODEL_PATH", ""),
        "BOUNDARY_REFINER_BACKBONE": os.getenv(
            "BOUNDARY_REFINER_BACKBONE",
            "transformers.Mamba2Model",
        ),
        "BOUNDARY_REFINER_DEVICE": os.getenv("BOUNDARY_REFINER_DEVICE", "auto"),
        "BOUNDARY_REFINER_THRESHOLD": os.getenv("BOUNDARY_REFINER_THRESHOLD", "0.5"),
        "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S": os.getenv(
            "BOUNDARY_PLANNER_MAX_CORE_CHUNK_S",
            "5.0",
        ),
        "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S": os.getenv(
            "BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S",
            "6.5",
        ),
        "BOUNDARY_PLANNER_TARGET_CHUNK_S": os.getenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "3.0"),
        "BOUNDARY_PLANNER_MIN_CHUNK_S": os.getenv("BOUNDARY_PLANNER_MIN_CHUNK_S", "0.4"),
        "BOUNDARY_PLANNER_START_WEIGHT": os.getenv("BOUNDARY_PLANNER_START_WEIGHT", "1.5"),
        "BOUNDARY_CONTEXT_MAX_PADDING_S": os.getenv("BOUNDARY_CONTEXT_MAX_PADDING_S", "1.5"),
        "BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S": os.getenv("BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S", "0.25"),
        "BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT": os.getenv("BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", "16"),
        "BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE": os.getenv("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", "256"),
        "BOUNDARY_DP_CHUNK_BASE_COST": os.getenv("BOUNDARY_DP_CHUNK_BASE_COST", "0.04"),
        "BOUNDARY_DP_OVER_TARGET_WEIGHT": os.getenv("BOUNDARY_DP_OVER_TARGET_WEIGHT", "0.30"),
        "BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT": os.getenv("BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT", "1.50"),
        "BOUNDARY_DP_UNDER_MIN_WEIGHT": os.getenv("BOUNDARY_DP_UNDER_MIN_WEIGHT", "0.20"),
        "BOUNDARY_DP_LONG_GAP_WEIGHT": os.getenv("BOUNDARY_DP_LONG_GAP_WEIGHT", "0.35"),
        "BOUNDARY_DP_SPLIT_MERGE_WEIGHT": os.getenv("BOUNDARY_DP_SPLIT_MERGE_WEIGHT", "0.35"),
        "QUALITY_REPORT_ENABLED": "1",
        "QUALITY_REPORT_DIR": str(paths.root / "quality_reports"),
        "QC_HARD_FAIL": "0",
        "RUN_LOG_ENABLED": "1",
        "RUN_LOG_DIR": str(paths.run_logs),
        "BOUNDARY_CACHE_ENABLED": "1" if args.boundary_cache else "0",
        "BOUNDARY_CACHE_DIR": str(paths.root / "boundary-cache"),
        "SPEECH_BOUNDARY_JA_THRESHOLD": str(args.speech_boundary_threshold),
        "SPEECH_BOUNDARY_JA_PAD_S": str(args.speech_boundary_pad_s),
        "SPEECH_BOUNDARY_JA_PTM": args.speech_boundary_ptm,
        "SPEECH_BOUNDARY_JA_MODEL_PATH": project_path_value(args.speech_boundary_model_path),
        "SPEECH_BOUNDARY_JA_DEVICE": args.speech_boundary_device,
        "SPEECH_BOUNDARY_JA_DTYPE": args.speech_boundary_dtype,
        "SPEECH_BOUNDARY_JA_WINDOW_S": str(args.speech_boundary_window_s),
        "SPEECH_BOUNDARY_JA_OVERLAP_S": str(args.speech_boundary_overlap_s),
        "SPEECH_BOUNDARY_JA_MIN_SEGMENT_S": str(args.speech_boundary_min_segment_s),
        "SPEECH_BOUNDARY_JA_MERGE_GAP_S": str(args.speech_boundary_merge_gap_s),
        "SPEECH_BOUNDARY_JA_CUT_THRESHOLD": str(args.speech_boundary_cut_threshold),
        "SPEECH_BOUNDARY_JA_APPLY_CUT_TO_SPEECH": "1" if args.speech_boundary_apply_cut_to_speech else "0",
        "KEEP_ASR_CHUNKS": "1" if args.keep_asr_chunks else "0",
        "FAIL_ON_QC_BLOCK": "0",
    }
    spec = SimpleNamespace(
        asr_backend=args.asr_backend,
        asr_context=args.asr_context,
        subtitle_mode=args.subtitle_mode,
        skip_translation=not args.translate,
        output_dir=str(paths.generated / job_id),
        keep_quality_report=True,
        keep_temp_files=True,
        run_log_enabled=True,
        run_log_dir=str(paths.run_logs),
        fail_on_qc_block=False,
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
        f"boundary=speech_boundary_ja threshold={args.speech_boundary_threshold:g} "
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
        "asr_qc": (timings.get("asr_details") or {}).get("asr_qc", {}),
        "boundary_signature": (timings.get("asr_details") or {}).get("boundary_signature", {}),
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
        "speech_boundary_operating_point": "qwen-feature-energy-bootstrap-v1",
        "speech_boundary_threshold": args.speech_boundary_threshold,
        "speech_boundary_pad_s": args.speech_boundary_pad_s,
        "boundary_planner": {
            "feature_frame_hop_s": args.boundary_feature_frame_hop_s,
            "max_core_chunk_s": args.boundary_planner_max_core_chunk_s,
            "max_padded_chunk_s": args.boundary_planner_max_padded_chunk_s,
            "target_chunk_s": args.boundary_planner_target_chunk_s,
            "min_chunk_s": args.boundary_planner_min_chunk_s,
            "start_weight": args.boundary_planner_start_weight,
            "max_context_padding_s": args.boundary_context_max_padding_s,
            "max_speech_overlap_s": args.boundary_context_max_speech_overlap_s,
            "max_splits_per_segment": args.boundary_planner_max_splits_per_segment,
            "sequence_batch_size": args.boundary_planner_sequence_batch_size,
            "dp_chunk_base_cost": os.getenv("BOUNDARY_DP_CHUNK_BASE_COST", "0.04"),
            "dp_over_target_weight": os.getenv("BOUNDARY_DP_OVER_TARGET_WEIGHT", "0.30"),
            "dp_far_over_target_weight": os.getenv("BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT", "1.50"),
            "dp_under_min_weight": os.getenv("BOUNDARY_DP_UNDER_MIN_WEIGHT", "0.20"),
            "dp_long_gap_weight": os.getenv("BOUNDARY_DP_LONG_GAP_WEIGHT", "0.35"),
            "dp_split_merge_weight": os.getenv("BOUNDARY_DP_SPLIT_MERGE_WEIGHT", "0.35"),
        },
        "translate": bool(args.translate),
        "results": results,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SpeechBoundary-JA Full Workflow",
        "",
        f"- ASR: `{args.asr_backend}`",
        f"- ASR model: `{project_rel(project_path_value(args.asr_model_path)) or 'auto'}`",
        f"- Boundary: `speech_boundary_ja` threshold `{args.speech_boundary_threshold:g}`, pad `{args.speech_boundary_pad_s:g}s`",
        f"- Translation: `{'on' if args.translate else 'off'}`",
        f"- Runtime root: `{project_rel(paths.root)}`",
        "",
        "| video | status | segments | blocks | asr_s | total_s | srt |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
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
            f"`{project_rel(paths_payload.get('srt'))}` |"
        )
    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_config()
    parser = argparse.ArgumentParser(
        description="Run project full workflow with SpeechBoundary-JA bootstrap scores, Boundary Refiner, and Qwen3-ASR."
    )
    parser.add_argument("--video", action="append", required=True, help="Video path or stem. Repeatable.")
    parser.add_argument("--task-name", default="full-workflow-qwen200k")
    parser.add_argument("--label", default="speech_boundary_ja_qwen200k")
    parser.add_argument(
        "--asr-backend",
        default=os.getenv("ASR_BACKEND", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"),
    )
    parser.add_argument(
        "--asr-model-path",
        default=os.getenv("ASR_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--aligner-model-path",
        default=os.getenv("ALIGNER_MODEL_PATH", ""),
    )
    parser.add_argument("--asr-worker-mode", default=os.getenv("ASR_WORKER_MODE", "subprocess"))
    parser.add_argument("--asr-dtype", default=os.getenv("ASR_DTYPE", "bfloat16"))
    parser.add_argument("--asr-attention", default=os.getenv("ASR_ATTENTION", "sdpa"))
    parser.add_argument("--asr-batch-size", default=os.getenv("ASR_BATCH_SIZE", "auto"))
    parser.add_argument("--aligner-batch-size", type=int, default=_env_int("ALIGNER_BATCH_SIZE", 48))
    parser.add_argument("--align-long-chunk-batch-size", type=int, default=_env_int("ALIGN_LONG_CHUNK_BATCH_SIZE", 48))
    parser.add_argument("--asr-max-new-tokens", type=int, default=_env_int("ASR_MAX_NEW_TOKENS", 128))
    parser.add_argument("--transcription-max-new-tokens", type=int, default=_env_int("TRANSCRIPTION_MAX_NEW_TOKENS", 128))
    parser.add_argument("--transcription-timeout-s", type=int, default=_env_int("TRANSCRIPTION_TIMEOUT_S", 300))
    parser.add_argument("--asr-context", default=os.getenv("ASR_CONTEXT", ""))
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
            "subtitle frame timing is derived from each source video's real FPS."
        ),
    )
    parser.add_argument(
        "--boundary-refiner-model-path",
        default=os.getenv("BOUNDARY_REFINER_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--boundary-refiner-backbone",
        default="transformers.Mamba2Model",
        choices=("transformers.Mamba2Model",),
    )
    parser.add_argument("--boundary-refiner-threshold", type=float, default=_env_float("BOUNDARY_REFINER_THRESHOLD", 0.5))
    parser.add_argument("--boundary-refiner-device", default=os.getenv("BOUNDARY_REFINER_DEVICE", "auto"))
    parser.add_argument("--boundary-planner-max-core-chunk-s", type=float, default=_env_float("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", 5.0))
    parser.add_argument("--boundary-planner-max-padded-chunk-s", type=float, default=_env_float("BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S", 6.5))
    parser.add_argument("--boundary-planner-target-chunk-s", type=float, default=_env_float("BOUNDARY_PLANNER_TARGET_CHUNK_S", 3.0))
    parser.add_argument("--boundary-planner-min-chunk-s", type=float, default=_env_float("BOUNDARY_PLANNER_MIN_CHUNK_S", 0.4))
    parser.add_argument("--boundary-planner-start-weight", type=float, default=_env_float("BOUNDARY_PLANNER_START_WEIGHT", 1.5))
    parser.add_argument("--boundary-context-max-padding-s", type=float, default=_env_float("BOUNDARY_CONTEXT_MAX_PADDING_S", 1.5))
    parser.add_argument("--boundary-context-max-speech-overlap-s", type=float, default=_env_float("BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S", 0.25))
    parser.add_argument("--boundary-planner-max-splits-per-segment", type=int, default=_env_int("BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT", 16))
    parser.add_argument("--boundary-planner-sequence-batch-size", type=int, default=_env_int("BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE", 256))
    parser.add_argument("--keep-asr-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-cache", action=argparse.BooleanOptionalAction, default=_env_bool("BOUNDARY_CACHE_ENABLED", True))
    parser.add_argument("--speech-boundary-threshold", dest="speech_boundary_threshold", type=float, default=_env_float("SPEECH_BOUNDARY_JA_THRESHOLD", 0.200))
    parser.add_argument("--speech-boundary-pad-s", dest="speech_boundary_pad_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_PAD_S", 0.2))
    parser.add_argument(
        "--speech-boundary-ptm",
        dest="speech_boundary_ptm",
        default=os.getenv("SPEECH_BOUNDARY_JA_PTM", "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"),
    )
    parser.add_argument(
        "--speech-boundary-model-path",
        dest="speech_boundary_model_path",
        default=os.getenv("SPEECH_BOUNDARY_JA_MODEL_PATH", "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame"),
    )
    parser.add_argument("--speech-boundary-device", dest="speech_boundary_device", default=os.getenv("SPEECH_BOUNDARY_JA_DEVICE", "auto"))
    parser.add_argument("--speech-boundary-dtype", dest="speech_boundary_dtype", default=os.getenv("SPEECH_BOUNDARY_JA_DTYPE", "bfloat16"))
    parser.add_argument("--speech-boundary-window-s", dest="speech_boundary_window_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_WINDOW_S", 30.0))
    parser.add_argument("--speech-boundary-overlap-s", dest="speech_boundary_overlap_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_OVERLAP_S", 1.0))
    parser.add_argument("--speech-boundary-min-segment-s", dest="speech_boundary_min_segment_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_MIN_SEGMENT_S", 0.05))
    parser.add_argument("--speech-boundary-merge-gap-s", dest="speech_boundary_merge_gap_s", type=float, default=_env_float("SPEECH_BOUNDARY_JA_MERGE_GAP_S", 0.0))
    parser.add_argument("--speech-boundary-cut-threshold", dest="speech_boundary_cut_threshold", type=float, default=_env_float("SPEECH_BOUNDARY_JA_CUT_THRESHOLD", 0.500))
    parser.add_argument(
        "--speech-boundary-apply-cut-to-speech",
        dest="speech_boundary_apply_cut_to_speech",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SPEECH_BOUNDARY_JA_APPLY_CUT_TO_SPEECH", True),
    )
    args = parser.parse_args(argv)
    if args.speech_boundary_threshold < 0:
        parser.error("--speech-boundary-threshold must be non-negative")
    if args.speech_boundary_pad_s < 0:
        parser.error("--speech-boundary-pad-s must be non-negative")
    if args.speech_boundary_window_s <= 0:
        parser.error("--speech-boundary-window-s must be positive")
    if args.speech_boundary_overlap_s < 0 or args.speech_boundary_overlap_s >= args.speech_boundary_window_s:
        parser.error("--speech-boundary-overlap-s must be non-negative and smaller than window")
    if args.speech_boundary_cut_threshold < 0:
        parser.error("--speech-boundary-cut-threshold must be non-negative")
    if args.boundary_planner_sequence_batch_size <= 0:
        parser.error("--boundary-planner-sequence-batch-size must be positive")
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
