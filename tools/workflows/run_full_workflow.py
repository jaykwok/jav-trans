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
    active_qwen_asr_model_id,
)
from core.config import load_config


DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV = ",".join(
    f"{repo_id}={batch_size}"
    for repo_id, batch_size in DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO.items()
)
# What produced the chunk boundaries. The five-model acoustic chain this used to
# name was retired on 2026-07-31; chunking now reads blank runs from the CTC
# alignment head, and falls back to fixed-length cuts when no head is bound.
DEFAULT_CHUNKING_SOURCE = "unknown"


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


def chunking_source(results: list[dict[str, Any]]) -> str:
    """Which path actually cut the audio, read back from the run rather than assumed.

    `alignment_head_blank_runs` and the two fixed-length fallbacks produce the
    same chunk shape, so a report that assumed one of them would hide a head
    that failed to load.
    """
    for result in results:
        signature = result.get("boundary_signature") or {}
        if not isinstance(signature, dict):
            continue
        chunking = signature.get("chunking")
        if isinstance(chunking, dict) and chunking.get("source"):
            return str(chunking["source"])
    return DEFAULT_CHUNKING_SOURCE


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
    root = PROJECT_ROOT / "agents" / "temp" / "full-workflow" / timestamped_label(task_name)
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


def configure_env(args: argparse.Namespace) -> None:
    os.environ.pop("ASR_STAGE_WORKER_MODE", None)
    os.environ.pop("ASR_WORKER_MODE", None)
    os.environ.pop("ASR_WORKER_MODE_BY_REPO", None)
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
    os.environ["KEEP_ASR_CHUNKS"] = "1" if args.keep_asr_chunks else "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_context(*, args: argparse.Namespace, paths: RunPaths, video: Path):
    from core.job_context import JobContext
    from pipeline.ids import sanitize_job_id

    job_id = sanitize_job_id(f"{video.stem}_{safe_label(args.label)}")
    job_temp_dir = paths.jobs / job_id
    advanced = {
        "TRANSCRIPTION_TIMEOUT_S": str(args.transcription_timeout_s),
        "ASR_MAX_NEW_TOKENS": str(args.asr_max_new_tokens),
        "ASR_BATCH_SIZE": args.asr_batch_size,
        "ASR_BATCH_SIZE_BY_REPO": os.getenv(
            "ASR_BATCH_SIZE_BY_REPO",
            DEFAULT_ASR_BATCH_SIZE_BY_REPO_ENV,
        ),
        "QUALITY_REPORT_ENABLED": "1",
        "QUALITY_REPORT_DIR": str(paths.root / "quality_reports"),
        "QC_HARD_FAIL": "0",
        "RUN_LOG_ENABLED": "1",
        "RUN_LOG_DIR": str(paths.run_logs),
        "KEEP_ASR_CHUNKS": "1" if args.keep_asr_chunks else "0",
    }
    spec = SimpleNamespace(
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
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
        advanced=advanced,
    )
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
        f"asr={active_qwen_asr_model_id()} ===",
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
        "asr_model_id": active_qwen_asr_model_id(),
        "asr_model_path": project_rel(project_path_value(args.asr_model_path)),
        "asr_batch_size": args.asr_batch_size,
        "chunking_source": chunking_source(results),
        "alignment_head_path": os.getenv("ASR_ALIGNMENT_HEAD_PATH", ""),
        "translate": bool(args.translate),
        "results": results,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Full Workflow",
        "",
        f"- ASR: `{active_qwen_asr_model_id()}`",
        "- Subtitle timing: `CTC forced alignment` (falls back to proportional "
        "when no alignment head is bound)",
        f"- ASR model: `{project_rel(project_path_value(args.asr_model_path)) or 'auto'}`",
        f"- Chunking: `{chunking_source(results)}`",
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
        description="Run the project's full workflow: chunking, Qwen3-ASR, CTC "
        "alignment, and optional translation."
    )
    parser.add_argument("--video", action="append", required=True, help="Video path or stem. Repeatable.")
    parser.add_argument("--task-name", default="full-workflow-qwen200k")
    parser.add_argument("--label", default="full_workflow")
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
    parser.add_argument("--keep-asr-chunks", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
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
