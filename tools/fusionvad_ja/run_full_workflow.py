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


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def make_paths(task_name: str) -> RunPaths:
    root = PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / safe_label(task_name)
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


def configure_env(args: argparse.Namespace) -> None:
    os.environ["ASR_BACKEND"] = args.asr_backend
    os.environ["ASR_VAD_BACKEND"] = "fusionvad_ja"
    os.environ["ASR_MODEL_PATH"] = str(project_path(args.asr_model_path))
    os.environ.setdefault("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
    os.environ.setdefault("ALIGNER_MODEL_PATH", str(project_path(args.aligner_model_path)))
    os.environ.setdefault("ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
    os.environ.setdefault("ASR_WORKER_MODE", args.asr_worker_mode)
    os.environ.setdefault("ASR_DTYPE", args.asr_dtype)
    os.environ.setdefault("ASR_ATTENTION", args.asr_attention)
    os.environ.setdefault("ASR_BATCH_SIZE", str(args.asr_batch_size))
    os.environ.setdefault("ALIGNER_BATCH_SIZE", str(args.aligner_batch_size))
    os.environ.setdefault("ALIGN_LONG_CHUNK_BATCH_SIZE", str(args.align_long_chunk_batch_size))
    os.environ.setdefault("TRANSCRIPTION_TIMEOUT_S", str(args.transcription_timeout_s))
    os.environ.setdefault("TRANSCRIPTION_MAX_NEW_TOKENS", str(args.transcription_max_new_tokens))
    os.environ.setdefault("ASR_MAX_NEW_TOKENS", str(args.asr_max_new_tokens))
    os.environ.setdefault("ASR_CHUNK_PACKING_ENABLED", "1")
    if args.chunk_pack_frame_hop_s is not None:
        os.environ.setdefault("ASR_CHUNK_PACK_FRAME_HOP_S", str(args.chunk_pack_frame_hop_s))
    os.environ.setdefault("ASR_CHUNK_PACK_WINDOW_FRAMES", str(args.chunk_pack_window_frames))
    os.environ.setdefault("ASR_CHUNK_PACK_RESERVE_FRAMES", str(args.chunk_pack_reserve_frames))
    os.environ.setdefault(
        "ASR_CHUNK_PACK_TARGET_PADDING_FRAMES",
        str(args.chunk_pack_target_padding_frames),
    )
    os.environ.setdefault("ASR_CHUNK_PACK_GAP_MERGE_FRAMES", str(args.chunk_pack_gap_merge_frames))
    os.environ.setdefault("KEEP_ASR_CHUNKS", "1" if args.keep_asr_chunks else "0")
    os.environ.setdefault("VAD_CHUNK_CACHE_ENABLED", "1" if args.vad_chunk_cache else "0")
    os.environ["FUSIONVAD_JA_CHECKPOINT"] = str(project_path(args.fusionvad_checkpoint))
    os.environ["FUSIONVAD_JA_THRESHOLD"] = str(args.fusionvad_threshold)
    os.environ["FUSIONVAD_JA_PAD_S"] = str(args.fusionvad_pad_s)
    os.environ["FUSIONVAD_JA_PTM"] = args.fusionvad_ptm
    os.environ["FUSIONVAD_JA_MODEL_PATH"] = str(project_path(args.fusionvad_model_path))
    os.environ["FUSIONVAD_JA_DEVICE"] = args.fusionvad_device
    os.environ["FUSIONVAD_JA_DTYPE"] = args.fusionvad_dtype
    os.environ["FUSIONVAD_JA_WINDOW_S"] = str(args.fusionvad_window_s)
    os.environ["FUSIONVAD_JA_OVERLAP_S"] = str(args.fusionvad_overlap_s)
    os.environ["FUSIONVAD_JA_MIN_SEGMENT_S"] = str(args.fusionvad_min_segment_s)
    os.environ["FUSIONVAD_JA_MERGE_GAP_S"] = str(args.fusionvad_merge_gap_s)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_context(*, args: argparse.Namespace, paths: RunPaths, video: Path):
    from core.job_context import JobContext
    from pipeline.ids import sanitize_job_id

    job_id = sanitize_job_id(f"{video.stem}_{safe_label(args.label)}")
    job_temp_dir = paths.jobs / job_id
    advanced = {
        "ASR_BACKEND": args.asr_backend,
        "ASR_VAD_BACKEND": "fusionvad_ja",
        "ASR_MODEL_PATH": str(project_path(args.asr_model_path)),
        "ALIGNER_MODEL_PATH": str(project_path(args.aligner_model_path)),
        "ASR_WORKER_MODE": args.asr_worker_mode,
        "ASR_CONTEXT": args.asr_context,
        "TRANSCRIPTION_TIMEOUT_S": str(args.transcription_timeout_s),
        "TRANSCRIPTION_MAX_NEW_TOKENS": str(args.transcription_max_new_tokens),
        "ASR_MAX_NEW_TOKENS": str(args.asr_max_new_tokens),
        "ASR_BATCH_SIZE": str(args.asr_batch_size),
        "ALIGNER_BATCH_SIZE": str(args.aligner_batch_size),
        "ALIGN_LONG_CHUNK_BATCH_SIZE": str(args.align_long_chunk_batch_size),
        "ASR_CHUNK_PACKING_ENABLED": "1",
        "ASR_CHUNK_PACK_WINDOW_FRAMES": str(args.chunk_pack_window_frames),
        "ASR_CHUNK_PACK_RESERVE_FRAMES": str(args.chunk_pack_reserve_frames),
        "ASR_CHUNK_PACK_TARGET_PADDING_FRAMES": str(args.chunk_pack_target_padding_frames),
        "ASR_CHUNK_PACK_GAP_MERGE_FRAMES": str(args.chunk_pack_gap_merge_frames),
        "QUALITY_REPORT_ENABLED": "1",
        "QUALITY_REPORT_DIR": str(paths.root / "quality_reports"),
        "QC_HARD_FAIL": "0",
        "RUN_LOG_ENABLED": "1",
        "RUN_LOG_DIR": str(paths.run_logs),
        "VAD_CHUNK_CACHE_ENABLED": "1" if args.vad_chunk_cache else "0",
        "VAD_CHUNK_CACHE_DIR": str(paths.root / "vad-cache"),
        "FUSIONVAD_JA_CHECKPOINT": str(project_path(args.fusionvad_checkpoint)),
        "FUSIONVAD_JA_THRESHOLD": str(args.fusionvad_threshold),
        "FUSIONVAD_JA_PAD_S": str(args.fusionvad_pad_s),
        "FUSIONVAD_JA_PTM": args.fusionvad_ptm,
        "FUSIONVAD_JA_MODEL_PATH": str(project_path(args.fusionvad_model_path)),
        "FUSIONVAD_JA_DEVICE": args.fusionvad_device,
        "FUSIONVAD_JA_DTYPE": args.fusionvad_dtype,
        "FUSIONVAD_JA_WINDOW_S": str(args.fusionvad_window_s),
        "FUSIONVAD_JA_OVERLAP_S": str(args.fusionvad_overlap_s),
        "FUSIONVAD_JA_MIN_SEGMENT_S": str(args.fusionvad_min_segment_s),
        "FUSIONVAD_JA_MERGE_GAP_S": str(args.fusionvad_merge_gap_s),
        "KEEP_ASR_CHUNKS": "1" if args.keep_asr_chunks else "0",
        "FAIL_ON_QC_BLOCK": "0",
    }
    spec = SimpleNamespace(
        asr_backend=args.asr_backend,
        asr_context=args.asr_context,
        subtitle_mode=args.subtitle_mode,
        skip_translation=not args.translate,
        multi_cue_split=True,
        vad_threshold=args.fusionvad_threshold,
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
    if args.chunk_pack_frame_hop_s is not None:
        advanced["ASR_CHUNK_PACK_FRAME_HOP_S"] = str(args.chunk_pack_frame_hop_s)
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
        f"vad=fusionvad_ja threshold={args.fusionvad_threshold:g} "
        f"asr={args.asr_backend} ===",
        flush=True,
    )
    artifacts = pipeline_main.run_asr_alignment_f0(str(video), ctx=ctx, job_id=ctx.job_id)
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
        "vad_signature": (timings.get("asr_details") or {}).get("vad_signature", {}),
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
        "asr_model_path": project_rel(project_path(args.asr_model_path)),
        "vad_backend": "fusionvad_ja",
        "fusionvad_checkpoint": project_rel(project_path(args.fusionvad_checkpoint)),
        "fusionvad_threshold": args.fusionvad_threshold,
        "fusionvad_pad_s": args.fusionvad_pad_s,
        "chunk_packing": {
            "frame_hop_s": args.chunk_pack_frame_hop_s,
            "window_frames": args.chunk_pack_window_frames,
            "reserve_frames": args.chunk_pack_reserve_frames,
            "target_padding_frames": args.chunk_pack_target_padding_frames,
            "gap_merge_frames": args.chunk_pack_gap_merge_frames,
        },
        "translate": bool(args.translate),
        "results": results,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# FusionVAD-JA Full Workflow",
        "",
        f"- ASR: `{args.asr_backend}`",
        f"- ASR model: `{project_rel(project_path(args.asr_model_path))}`",
        f"- VAD: `fusionvad_ja` threshold `{args.fusionvad_threshold:g}`, pad `{args.fusionvad_pad_s:g}s`",
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
    parser = argparse.ArgumentParser(
        description="Run project full workflow with FusionVAD-JA high-recall VAD and Qwen3-ASR."
    )
    parser.add_argument("--video", action="append", required=True, help="Video path or stem. Repeatable.")
    parser.add_argument("--task-name", default="full-workflow-qwen200k")
    parser.add_argument("--label", default="fusionvad_ja_qwen200k")
    parser.add_argument("--asr-backend", default="qwen3-asr-1.7b")
    parser.add_argument(
        "--asr-model-path",
        default="models/Qwen-Qwen3-ASR-1.7B-galgame-asr200k-bs4/checkpoint-6250",
    )
    parser.add_argument("--aligner-model-path", default="models/Qwen-Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--asr-worker-mode", default="subprocess")
    parser.add_argument("--asr-dtype", default="bfloat16")
    parser.add_argument("--asr-attention", default="sdpa")
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--aligner-batch-size", type=int, default=2)
    parser.add_argument("--align-long-chunk-batch-size", type=int, default=1)
    parser.add_argument("--asr-max-new-tokens", type=int, default=128)
    parser.add_argument("--transcription-max-new-tokens", type=int, default=128)
    parser.add_argument("--transcription-timeout-s", type=int, default=300)
    parser.add_argument("--asr-context", default=os.getenv("ASR_CONTEXT", ""))
    parser.add_argument("--subtitle-mode", default="zh")
    parser.add_argument("--translate", action="store_true", help="Run LLM translation too. Default outputs Japanese SRT.")
    parser.add_argument("--target-lang", default=os.getenv("TARGET_LANG", "简体中文"))
    parser.add_argument("--translation-max-workers", type=int, default=1)
    parser.add_argument(
        "--chunk-pack-frame-hop-s",
        type=float,
        default=None,
        help=(
            "Override ASR chunk frame duration in seconds. Default leaves it unset so "
            "the pipeline probes each video's FPS and uses 1/fps, with 29.97 fallback."
        ),
    )
    parser.add_argument("--chunk-pack-window-frames", type=int, default=899)
    parser.add_argument("--chunk-pack-reserve-frames", type=int, default=45)
    parser.add_argument("--chunk-pack-target-padding-frames", type=int, default=60)
    parser.add_argument("--chunk-pack-gap-merge-frames", type=int, default=45)
    parser.add_argument("--keep-asr-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad-chunk-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fusionvad-checkpoint",
        default=(
            "datasets/train/fusionvad-ja/v1-11/qwen3-asr-0.6b/"
            "addition-bilstm-ft-v1mini-galgame-synthv5-longgap-posw2-558-batch16-lr2e-4-steps1024/"
            "fusionvad_ja_addition_bilstm.pt"
        ),
    )
    parser.add_argument("--fusionvad-threshold", type=float, default=0.02)
    parser.add_argument("--fusionvad-pad-s", type=float, default=0.2)
    parser.add_argument("--fusionvad-ptm", default="qwen3-asr-0.6b")
    parser.add_argument("--fusionvad-model-path", default="models/Qwen-Qwen3-ASR-0.6B")
    parser.add_argument("--fusionvad-device", default="auto")
    parser.add_argument("--fusionvad-dtype", default="bfloat16")
    parser.add_argument("--fusionvad-window-s", type=float, default=30.0)
    parser.add_argument("--fusionvad-overlap-s", type=float, default=1.0)
    parser.add_argument("--fusionvad-min-segment-s", type=float, default=0.05)
    parser.add_argument("--fusionvad-merge-gap-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.fusionvad_threshold < 0:
        parser.error("--fusionvad-threshold must be non-negative")
    if args.fusionvad_pad_s < 0:
        parser.error("--fusionvad-pad-s must be non-negative")
    if args.fusionvad_window_s <= 0:
        parser.error("--fusionvad-window-s must be positive")
    if args.fusionvad_overlap_s < 0 or args.fusionvad_overlap_s >= args.fusionvad_window_s:
        parser.error("--fusionvad-overlap-s must be non-negative and smaller than window")
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
