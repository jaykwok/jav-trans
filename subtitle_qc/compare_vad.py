#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_reference as ref_eval  # noqa: E402


DEFAULT_VADS: tuple[tuple[str, str], ...] = (
    ("whisperseg_adaptive", "whisperseg-adaptive"),
    ("fusion_lite", "fusion_lite"),
    ("fusion_lite_boost", "fusion_lite_boost"),
    ("fusion_lite_sigmoid", "fusion_lite_sigmoid"),
)
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")

_PIPELINE_MAIN = None
_JOB_CONTEXT = None
_SANITIZE_JOB_ID = None


@dataclass(frozen=True)
class VideoCase:
    video: Path
    reference: Path | None


@dataclass(frozen=True)
class TaskPaths:
    root: Path
    job_root: Path
    run_log_dir: Path
    generated_root: Path
    backup_root: Path
    summary_root: Path
    summary_json: Path
    summary_md: Path
    summary_metrics_csv: Path


def project_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def video_artifact_dir(video: Path) -> Path:
    return video.parent / video.stem


def video_artifact_path(video: Path, filename: str) -> Path:
    nested = video_artifact_dir(video) / filename
    if nested.exists():
        return nested
    return video.parent / filename


def video_artifact_output_path(video: Path, filename: str, *, group_by_video: bool) -> Path:
    if not group_by_video:
        return video.parent / filename
    return video_artifact_dir(video) / filename


def subtitle_qc_task_dir(video: Path, task_name: str) -> Path:
    return video_artifact_dir(video) / "subtitle_qc" / safe_label(task_name)


def safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.replace("-", "_").replace(".", "_")
    return clean.strip("_") or "item"


def default_asr_label(asr_backend: str) -> str:
    aliases = {
        "qwen3-asr-1.7b": "qwen3",
        "whisper-ja-anime-v0.3": "whisper_ja_anime_v0_3",
        "whisper-ja-1.5b": "whisper_ja_1_5b",
        "anime-whisper": "anime_whisper",
    }
    return aliases.get(asr_backend, safe_label(asr_backend))


def parse_vad_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, backend = spec.split("=", 1)
        label = safe_label(label)
        backend = backend.strip()
    else:
        backend = spec.strip()
        label = safe_label(backend)
    if not backend:
        raise SystemExit(f"invalid VAD spec: {spec!r}")
    return label, backend


def prefixed_label(prefix: str, vad_label: str, *, no_prefix: bool) -> str:
    label = safe_label(vad_label)
    if no_prefix:
        return label
    clean_prefix = safe_label(prefix)
    if not clean_prefix or label == clean_prefix or label.startswith(f"{clean_prefix}_"):
        return label
    return f"{clean_prefix}_{label}"


def selected_vads(args: argparse.Namespace) -> list[tuple[str, str]]:
    raw_specs = args.vad or [f"{label}={backend}" for label, backend in DEFAULT_VADS]
    prefix = args.label_prefix if args.label_prefix is not None else default_asr_label(args.asr_backend)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in raw_specs:
        vad_label, backend = parse_vad_spec(spec)
        label = prefixed_label(prefix, vad_label, no_prefix=args.no_label_prefix)
        if label in seen:
            raise SystemExit(f"duplicate output label after prefixing: {label}")
        seen.add(label)
        results.append((label, backend))
    return results


def resolve_video(value: str, video_dir: Path) -> Path:
    raw = project_path(value)
    if raw.exists():
        return raw
    candidates = []
    text = value.strip()
    for ext in VIDEO_EXTS:
        candidates.append(video_dir / f"{text}{ext}")
    for path in candidates:
        if path.exists():
            return path.resolve()
    available = ", ".join(path.name for path in sorted(video_dir.glob("*")) if path.suffix.lower() in VIDEO_EXTS)
    raise SystemExit(f"video not found: {value}; available videos: {available or 'none'}")


def discover_reference(video_stem: str, reference_dir: Path, *, required: bool) -> Path | None:
    try:
        return ref_eval.discover_reference(video_stem, reference_dir).resolve()
    except SystemExit:
        if required:
            raise
        return None


def discover_reference_cases(video_dir: Path, reference_dir: Path) -> list[VideoCase]:
    cases: list[VideoCase] = []
    for video in sorted(video_dir.iterdir()):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue
        reference = discover_reference(video.stem, reference_dir, required=False)
        if reference is not None:
            cases.append(VideoCase(video.resolve(), reference))
    if not cases:
        raise SystemExit(f"no videos with references found under {project_rel(video_dir)} and {project_rel(reference_dir)}")
    return cases


def selected_cases(args: argparse.Namespace) -> list[VideoCase]:
    video_dir = project_path(args.video_dir)
    reference_dir = project_path(args.reference_dir)
    require_reference = not args.no_evaluate
    cases: list[VideoCase] = []

    for item in args.case or []:
        if "=" not in item:
            raise SystemExit(f"--case must be video=reference: {item}")
        video_text, reference_text = item.split("=", 1)
        cases.append(
            VideoCase(
                resolve_video(video_text.strip(), video_dir),
                project_path(reference_text.strip()).resolve(),
            )
        )

    if args.reference and not args.video:
        raise SystemExit("--reference requires matching --video arguments")
    if args.video:
        references = args.reference or []
        if references and len(references) != len(args.video):
            raise SystemExit("--reference count must match --video count")
        for index, video_text in enumerate(args.video):
            video = resolve_video(video_text, video_dir)
            if references:
                reference = project_path(references[index]).resolve()
            else:
                reference = discover_reference(video.stem, reference_dir, required=require_reference)
            cases.append(VideoCase(video, reference))

    if not cases:
        cases = discover_reference_cases(video_dir, reference_dir)

    for case in cases:
        if not case.video.exists():
            raise SystemExit(f"video not found: {project_rel(case.video)}")
        if require_reference and (case.reference is None or not case.reference.exists()):
            raise SystemExit(f"reference not found for {project_rel(case.video)}")
        if case.reference is not None and not case.reference.exists():
            raise SystemExit(f"reference not found: {project_rel(case.reference)}")
    return cases


def make_task_paths(task_name: str) -> TaskPaths:
    root = PROJECT_ROOT / "agents" / "temp" / "subtitle_qc" / safe_label(task_name)
    summary_root = PROJECT_ROOT / "video" / "subtitle_qc" / safe_label(task_name)
    return TaskPaths(
        root=root,
        job_root=root / "jobs",
        run_log_dir=root / "run-logs",
        generated_root=root / "generated",
        backup_root=PROJECT_ROOT / "agents" / "rm" / "subtitle_qc" / safe_label(task_name),
        summary_root=summary_root,
        summary_json=summary_root / "summary.json",
        summary_md=summary_root / "summary.md",
        summary_metrics_csv=summary_root / "summary_metrics.csv",
    )


def ensure_dirs(paths: TaskPaths) -> None:
    for path in (
        paths.root,
        paths.job_root,
        paths.run_log_dir,
        paths.generated_root,
        paths.backup_root,
        paths.summary_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def archive_existing(paths: TaskPaths, path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = paths.backup_root / f"{stamp}.{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)


def copy_or_replace(paths: TaskPaths, src: str | Path | None, dst: Path, *, force: bool) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if force:
            archive_existing(paths, dst)
        else:
            return str(dst)
    shutil.copy2(src_path, dst)
    return str(dst)


def move_or_replace(paths: TaskPaths, src: str | Path | None, dst: Path, *, force: bool) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if force:
            archive_existing(paths, dst)
        else:
            return str(dst)
    src_path.replace(dst)
    return str(dst)


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["ASR_BACKEND"] = args.asr_backend
    os.environ.setdefault("ASR_WORKER_MODE", "subprocess")
    os.environ.setdefault("TRANSCRIPTION_TIMEOUT_S", str(args.transcription_timeout))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_pipeline(args: argparse.Namespace):
    global _PIPELINE_MAIN, _JOB_CONTEXT, _SANITIZE_JOB_ID
    if _PIPELINE_MAIN is None:
        configure_environment(args)
        import main as pipeline_main  # noqa: PLC0415
        from core.job_context import JobContext  # noqa: PLC0415
        from pipeline.ids import sanitize_job_id  # noqa: PLC0415

        _PIPELINE_MAIN = pipeline_main
        _JOB_CONTEXT = JobContext
        _SANITIZE_JOB_ID = sanitize_job_id
    return _PIPELINE_MAIN, _JOB_CONTEXT, _SANITIZE_JOB_ID


def require_whisperseg_cuda() -> None:
    from vad.whisperseg.whisperseg_core import WhisperSegSpeechSegmenter

    segmenter = WhisperSegSpeechSegmenter()
    segmenter._ensure_model()
    providers = list(segmenter._session.get_providers()) if segmenter._session else []
    actual_device = str(getattr(segmenter, "_actual_device", ""))
    try:
        del segmenter
        gc.collect()
    except Exception:
        pass
    if "CUDAExecutionProvider" not in providers or "GPU" not in actual_device:
        raise RuntimeError(
            "WhisperSeg ONNX is not using CUDA; "
            f"actual_device={actual_device!r} providers={providers!r}"
        )


def build_ctx(
    *,
    paths: TaskPaths,
    args: argparse.Namespace,
    video: Path,
    label: str,
    vad_backend: str,
):
    _pipeline_main, job_context_cls, sanitize_job_id = load_pipeline(args)
    job_id = sanitize_job_id(f"{video.stem}_{label}")
    job_temp_dir = paths.job_root / job_id
    output_dir = paths.generated_root / job_id
    quality_dir = subtitle_qc_task_dir(video, args.task_name) / "quality_reports"
    spec = SimpleNamespace(
        asr_backend=args.asr_backend,
        asr_context=args.asr_context,
        subtitle_mode=args.subtitle_mode,
        skip_translation=not args.translate,
        multi_cue_split=True,
        vad_threshold=float(os.getenv("WHISPERSEG_THRESHOLD", "0.35") or "0.35"),
        output_dir=str(output_dir),
        keep_quality_report=True,
        keep_temp_files=True,
        run_log_enabled=True,
        run_log_dir=str(paths.run_log_dir),
        fail_on_qc_block=False,
        translation_max_workers=1,
        target_lang=os.getenv("TARGET_LANG") or "ja",
        translation_glossary=os.getenv("TRANSLATION_GLOSSARY") or "",
        llm_api_format=os.getenv("LLM_API_FORMAT") or None,
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT") or None,
        advanced={
            "ASR_BACKEND": args.asr_backend,
            "ASR_VAD_BACKEND": vad_backend,
            "ASR_VAD_ADAPTIVE": os.getenv("ASR_VAD_ADAPTIVE", "1"),
            "ASR_WORKER_MODE": os.getenv("ASR_WORKER_MODE", "subprocess"),
            "ASR_CONTEXT": args.asr_context,
            "TRANSCRIPTION_TIMEOUT_S": str(args.transcription_timeout),
            "QUALITY_REPORT_ENABLED": "1",
            "QUALITY_REPORT_DIR": str(quality_dir),
            "QC_HARD_FAIL": "0",
            "RUN_LOG_ENABLED": "1",
            "RUN_LOG_DIR": str(paths.run_log_dir),
            "VAD_CHUNK_CACHE_ENABLED": os.getenv("VAD_CHUNK_CACHE_ENABLED", "1"),
            "VAD_CHUNK_CACHE_DIR": os.getenv("VAD_CHUNK_CACHE_DIR", "./temp/vad-cache"),
        },
    )
    return job_context_cls.from_spec(
        spec,
        job_id=job_id,
        job_temp_dir=str(job_temp_dir),
        cache_path=str(job_temp_dir / "translation_cache.jsonl"),
    )


def resolve_run_log_path(paths: TaskPaths, ctx, artifacts=None) -> Path | None:
    candidates: list[Path] = []
    if artifacts is not None and getattr(artifacts, "run_log_path", None):
        candidates.append(Path(artifacts.run_log_path))
    candidates.extend(sorted(paths.run_log_dir.glob(f"*_{ctx.job_id}_*.run.log")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def preserve_outputs(
    *,
    paths: TaskPaths,
    video: Path,
    label: str,
    artifacts,
    output_paths: list[str],
    run_log_path: Path | None,
    force: bool,
    group_by_video: bool,
) -> dict[str, str]:
    stem = video.stem
    preserved: dict[str, str] = {}
    preserved["srt"] = move_or_replace(
        paths,
        artifacts.srt_path,
        video_artifact_output_path(video, f"{stem}.{label}.srt", group_by_video=group_by_video),
        force=force,
    )
    preserved["bilingual_json"] = copy_or_replace(
        paths,
        artifacts.bilingual_json_path,
        video_artifact_output_path(video, f"{stem}.{label}.bilingual.json", group_by_video=group_by_video),
        force=force,
    )
    preserved["timings_json"] = copy_or_replace(
        paths,
        artifacts.timings_path,
        paths.root / f"{stem}.{label}.timings.json",
        force=force,
    )
    preserved["aligned_segments_json"] = copy_or_replace(
        paths,
        artifacts.aligned_segments_path,
        paths.root / f"{stem}.{label}.aligned_segments.json",
        force=force,
    )
    preserved["transcript_json"] = copy_or_replace(
        paths,
        artifacts.transcript_path,
        paths.root / f"{stem}.{label}.transcript.json",
        force=force,
    )
    preserved["asr_manifest_json"] = copy_or_replace(
        paths,
        artifacts.asr_manifest_path,
        paths.root / f"{stem}.{label}.asr_manifest.json",
        force=force,
    )
    if run_log_path:
        preserved["run_log"] = str(run_log_path)

    quality_md = artifacts.quality_report_path or ""
    for item in output_paths:
        candidate = Path(item)
        if candidate.name.endswith(".quality_report.md") and candidate.exists():
            quality_md = str(candidate)
            break
    preserved["quality_report_md"] = copy_or_replace(
        paths,
        quality_md,
        subtitle_qc_task_dir(video, paths.summary_root.name) / "quality_reports" / f"{stem}.{label}.quality_report.md",
        force=force,
    )
    quality_json = Path(quality_md).with_suffix(".json") if quality_md else None
    preserved["quality_report_json"] = copy_or_replace(
        paths,
        quality_json if quality_json and quality_json.exists() else "",
        subtitle_qc_task_dir(video, paths.summary_root.name) / "quality_reports" / f"{stem}.{label}.quality_report.json",
        force=force,
    )
    return {key: value for key, value in preserved.items() if value}


def existing_result(
    *,
    paths: TaskPaths,
    args: argparse.Namespace,
    video: Path,
    label: str,
    vad_backend: str,
) -> dict[str, Any] | None:
    srt_path = video_artifact_path(video, f"{video.stem}.{label}.srt")
    if not srt_path.exists() or srt_path.stat().st_size <= 0:
        return None
    timings_path = paths.root / f"{video.stem}.{label}.timings.json"
    quality_path = subtitle_qc_task_dir(video, args.task_name) / "quality_reports" / f"{video.stem}.{label}.quality_report.json"
    timings = read_json(timings_path)
    quality = read_json(quality_path)
    return {
        "video": video.name,
        "video_stem": video.stem,
        "label": label,
        "vad_backend": vad_backend,
        "asr_backend": args.asr_backend,
        "status": "existing",
        "elapsed_s": 0.0,
        "paths": {
            "srt": str(srt_path),
            "timings_json": str(timings_path) if timings_path.exists() else "",
            "quality_report_json": str(quality_path) if quality_path.exists() else "",
        },
        "counts": timings.get("counts", {}),
        "stage_timings": timings.get("stage_timings", {}),
        "asr_qc": (timings.get("asr_details") or {}).get("asr_qc", {}),
        "quality": quality,
    }


def run_one(
    *,
    paths: TaskPaths,
    args: argparse.Namespace,
    video: Path,
    label: str,
    vad_backend: str,
) -> dict[str, Any]:
    if not args.force:
        previous = existing_result(paths=paths, args=args, video=video, label=label, vad_backend=vad_backend)
        if previous is not None:
            print(f"=== SKIP existing {video.name} {label} ===", flush=True)
            return previous

    pipeline_main, _job_context_cls, _sanitize_job_id = load_pipeline(args)
    started = time.perf_counter()
    ctx = build_ctx(paths=paths, args=args, video=video, label=label, vad_backend=vad_backend)
    print(f"=== START {video.name} {label} ({vad_backend}, asr={args.asr_backend}) ===", flush=True)
    artifacts = pipeline_main.run_asr_alignment_f0(
        str(video),
        ctx=ctx,
        job_id=ctx.job_id,
    )
    output_paths = pipeline_main.run_translation_and_write(
        str(video),
        artifacts,
        ctx=ctx,
        job_id=ctx.job_id,
    )
    run_log_path = resolve_run_log_path(paths, ctx, artifacts)
    preserved = preserve_outputs(
        paths=paths,
        video=video,
        label=label,
        artifacts=artifacts,
        output_paths=output_paths,
        run_log_path=run_log_path,
        force=args.force,
        group_by_video=args.group_by_video,
    )
    timings = read_json(preserved.get("timings_json"))
    quality = read_json(preserved.get("quality_report_json"))
    elapsed = time.perf_counter() - started
    result = {
        "video": video.name,
        "video_stem": video.stem,
        "label": label,
        "vad_backend": vad_backend,
        "asr_backend": args.asr_backend,
        "job_id": ctx.job_id,
        "status": "done",
        "elapsed_s": round(elapsed, 3),
        "paths": preserved,
        "counts": timings.get("counts", {}),
        "stage_timings": timings.get("stage_timings", {}),
        "asr_qc": (timings.get("asr_details") or {}).get("asr_qc", {}),
        "quality": quality,
    }
    print(f"=== DONE {video.name} {label} elapsed={elapsed:.1f}s ===", flush=True)
    return result


def evaluate_video(
    *,
    paths: TaskPaths,
    case: VideoCase,
    run_results: list[dict[str, Any]],
    time_pad: float,
    low_threshold: float,
    good_threshold: float,
    review_limit: int,
) -> dict[str, Any]:
    if case.reference is None:
        raise RuntimeError(f"missing reference for {project_rel(case.video)}")
    output_dir = subtitle_qc_task_dir(case.video, paths.summary_root.name) / "reference_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_cues = ref_eval.load_subtitle(case.reference, "reference")
    reference_chars = ref_eval.japanese_char_count(reference_cues)
    if reference_chars < 20:
        raise RuntimeError(f"reference does not look Japanese: {project_rel(case.reference)}")

    all_metrics = []
    worst_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "video": project_rel(case.video),
        "reference": project_rel(case.reference),
        "reference_cue_count": len(reference_cues),
        "reference_japanese_chars": reference_chars,
        "time_pad": time_pad,
        "low_threshold": low_threshold,
        "good_threshold": good_threshold,
        "candidates": {},
    }

    for result in run_results:
        if result.get("video_stem") != case.video.stem or result.get("status") not in {"done", "existing"}:
            continue
        label = str(result.get("label") or "")
        srt_path = Path((result.get("paths") or {}).get("srt") or "")
        if not label or not srt_path.exists():
            continue
        spec = ref_eval.CandidateSpec(label=label, path=srt_path.resolve())
        candidate_cues = ref_eval.load_subtitle(spec.path, spec.label)
        metrics, cue_results = ref_eval.evaluate_candidate(
            spec=spec,
            reference_cues=reference_cues,
            candidate_cues=candidate_cues,
            time_pad=time_pad,
            low_threshold=low_threshold,
            good_threshold=good_threshold,
        )
        all_metrics.append(metrics)
        payload["candidates"][label] = {
            "metrics": asdict(metrics),
            "worst_reference_cues": [
                asdict(item)
                for item in sorted(cue_results, key=lambda value: (value.similarity, value.start))[
                    :review_limit
                ]
            ],
        }
        for item in sorted(cue_results, key=lambda value: (value.similarity, value.start))[:review_limit]:
            worst_rows.append(
                {
                    "candidate": spec.label,
                    "reference_index": item.index + 1,
                    "start": ref_eval.fmt_time(item.start),
                    "end": ref_eval.fmt_time(item.end),
                    "similarity": f"{item.similarity:.4f}",
                    "matched": str(item.matched).lower(),
                    "candidate_indexes": " ".join(str(index + 1) for index in item.candidate_indexes),
                    "reference_text": item.reference_text,
                    "candidate_text": item.candidate_text,
                }
            )

    if not all_metrics:
        raise RuntimeError(f"no candidates available for {case.video.stem}")

    metrics_rows = [
        {
            "candidate": item.label,
            "path": item.path,
            "cue_count": item.cue_count,
            "reference_cue_count": item.reference_cue_count,
            "matched_reference_count": item.matched_reference_count,
            "missing_reference_count": item.missing_reference_count,
            "low_similarity_count": item.low_similarity_count,
            "good_similarity_count": item.good_similarity_count,
            "extra_candidate_count": item.extra_candidate_count,
            "covered_reference_ratio": f"{item.covered_reference_ratio:.6f}",
            "good_reference_ratio": f"{item.good_reference_ratio:.6f}",
            "mean_similarity": f"{item.mean_similarity:.6f}",
            "median_similarity": f"{item.median_similarity:.6f}",
            "weighted_similarity": f"{item.weighted_similarity:.6f}",
            "p10_similarity": f"{item.p10_similarity:.6f}",
        }
        for item in sorted(all_metrics, key=lambda value: value.weighted_similarity, reverse=True)
    ]
    ref_eval.write_json(output_dir / "reference_eval.json", payload)
    ref_eval.write_csv(output_dir / "reference_eval_metrics.csv", metrics_rows, list(metrics_rows[0]))
    ref_eval.write_csv(
        output_dir / "reference_eval_worst_cues.csv",
        worst_rows,
        [
            "candidate",
            "reference_index",
            "start",
            "end",
            "similarity",
            "matched",
            "candidate_indexes",
            "reference_text",
            "candidate_text",
        ],
    )
    ref_eval.write_markdown_summary(
        path=output_dir / "reference_eval_summary.md",
        video_path=case.video,
        reference_path=case.reference,
        metrics=all_metrics,
    )
    ranked = sorted(all_metrics, key=lambda value: value.weighted_similarity, reverse=True)
    print(
        f"=== EVAL {case.video.name} best={ranked[0].label} "
        f"weighted={ranked[0].weighted_similarity:.3f} ===",
        flush=True,
    )
    return {
        "video": case.video.name,
        "video_stem": case.video.stem,
        "reference": project_rel(case.reference),
        "reference_cue_count": len(reference_cues),
        "reference_japanese_chars": reference_chars,
        "paths": {
            "json": str(output_dir / "reference_eval.json"),
            "metrics_csv": str(output_dir / "reference_eval_metrics.csv"),
            "worst_cues_csv": str(output_dir / "reference_eval_worst_cues.csv"),
            "summary_md": str(output_dir / "reference_eval_summary.md"),
        },
        "ranked": [asdict(item) for item in ranked],
    }


def write_summary(
    *,
    paths: TaskPaths,
    args: argparse.Namespace,
    cases: list[VideoCase],
    vads: list[tuple[str, str]],
    results: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> None:
    by_key = {
        (result.get("video_stem"), result.get("label")): result
        for result in results
        if result.get("status") in {"done", "existing"}
    }
    overall: dict[str, dict[str, float]] = {}
    for evaluation in evaluations:
        weight = float(evaluation.get("reference_japanese_chars") or evaluation.get("reference_cue_count") or 1)
        for item in evaluation["ranked"]:
            label = item["label"]
            bucket = overall.setdefault(label, {"weighted_sum": 0.0, "weight": 0.0, "video_count": 0.0})
            bucket["weighted_sum"] += float(item["weighted_similarity"]) * weight
            bucket["weight"] += weight
            bucket["video_count"] += 1

    overall_ranked = [
        {
            "label": label,
            "overall_weighted_similarity": values["weighted_sum"] / max(1.0, values["weight"]),
            "video_count": int(values["video_count"]),
        }
        for label, values in overall.items()
    ]
    overall_ranked.sort(key=lambda item: item["overall_weighted_similarity"], reverse=True)
    payload = {
        "task": safe_label(args.task_name),
        "asr_backend": args.asr_backend,
        "videos": [project_rel(case.video) for case in cases],
        "references": [project_rel(case.reference) for case in cases if case.reference is not None],
        "vad_modes": [{"label": label, "backend": backend} for label, backend in vads],
        "results": results,
        "evaluations": evaluations,
        "overall_ranked": overall_ranked,
    }
    paths.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VAD Compare",
        "",
        f"- ASR backend: `{args.asr_backend}`",
        f"- Summary: `{project_rel(paths.summary_root)}`",
        f"- Runtime temp: `{project_rel(paths.root)}`",
        f"- Translation: `{'on' if args.translate else 'off'}`",
        "",
        "## Overall",
        "",
    ]
    if overall_ranked:
        lines.extend(
            [
                "| rank | vad | weighted | videos |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for rank, item in enumerate(overall_ranked, start=1):
            lines.append(
                f"| {rank} | `{item['label']}` | "
                f"{item['overall_weighted_similarity']:.3f} | {item['video_count']} |"
            )
    else:
        lines.append("Reference evaluation was not run.")

    if evaluations:
        lines.extend(
            [
                "",
                "## By Video",
                "",
                "| video | rank | vad | weighted | mean | p10 | covered | good | low | extra | cues | asr_s | total_s |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for evaluation in evaluations:
            stem = evaluation["video_stem"]
            for rank, item in enumerate(evaluation["ranked"], start=1):
                result = by_key.get((stem, item["label"]), {})
                timings = result.get("stage_timings") or {}
                counts = result.get("counts") or {}
                lines.append(
                    f"| `{stem}` | {rank} | `{item['label']}` | "
                    f"{float(item['weighted_similarity']):.3f} | "
                    f"{float(item['mean_similarity']):.3f} | "
                    f"{float(item['p10_similarity']):.3f} | "
                    f"{float(item['covered_reference_ratio']):.1%} | "
                    f"{float(item['good_reference_ratio']):.1%} | "
                    f"{int(item['low_similarity_count'])} | "
                    f"{int(item['extra_candidate_count'])} | "
                    f"{int(counts.get('blocks') or item['cue_count'])} | "
                    f"{float(timings.get('asr_alignment_total_s') or 0.0):.1f} | "
                    f"{float(timings.get('pipeline_total_s') or result.get('elapsed_s') or 0.0):.1f} |"
                )
    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    flat_rows = []
    for evaluation in evaluations:
        for rank, item in enumerate(evaluation["ranked"], start=1):
            result = by_key.get((evaluation["video_stem"], item["label"]), {})
            timings = result.get("stage_timings") or {}
            flat_rows.append(
                {
                    "video": evaluation["video_stem"],
                    "rank": rank,
                    "candidate": item["label"],
                    "weighted_similarity": f"{float(item['weighted_similarity']):.6f}",
                    "mean_similarity": f"{float(item['mean_similarity']):.6f}",
                    "median_similarity": f"{float(item['median_similarity']):.6f}",
                    "p10_similarity": f"{float(item['p10_similarity']):.6f}",
                    "covered_reference_ratio": f"{float(item['covered_reference_ratio']):.6f}",
                    "good_reference_ratio": f"{float(item['good_reference_ratio']):.6f}",
                    "low_similarity_count": item["low_similarity_count"],
                    "extra_candidate_count": item["extra_candidate_count"],
                    "asr_alignment_total_s": f"{float(timings.get('asr_alignment_total_s') or 0.0):.3f}",
                    "pipeline_total_s": f"{float(timings.get('pipeline_total_s') or result.get('elapsed_s') or 0.0):.3f}",
                    "srt": project_rel((result.get("paths") or {}).get("srt")),
                }
            )
    if flat_rows:
        with paths.summary_metrics_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ASR backend across multiple VAD backends and optionally evaluate against Japanese references."
    )
    parser.add_argument("--asr-backend", default=os.getenv("ASR_BACKEND") or "whisper-ja-anime-v0.3")
    parser.add_argument("--asr-context", default=os.getenv("ASR_CONTEXT", ""))
    parser.add_argument("--label-prefix", default=None, help="Output label prefix. Defaults to a short ASR label.")
    parser.add_argument("--no-label-prefix", action="store_true", help="Use VAD labels directly for output filenames.")
    parser.add_argument(
        "--vad",
        "--mode",
        dest="vad",
        action="append",
        help="VAD backend or label=backend. Repeatable. Defaults to whisperseg-adaptive and fusion_lite variants.",
    )
    parser.add_argument("--video", action="append", help="Video path, name, or stem. Repeatable.")
    parser.add_argument("--reference", action="append", help="Reference path paired with --video. Repeatable.")
    parser.add_argument("--case", action="append", help="Explicit video=reference pair. Repeatable.")
    parser.add_argument("--video-dir", default="video")
    parser.add_argument("--reference-dir", default="video/reference")
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--flat-video-output",
        dest="group_by_video",
        action="store_false",
        help="Write SRT/JSON outputs directly under video/. Default groups outputs under video/<video-stem>/.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even when output SRT already exists.")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing output SRTs.")
    parser.add_argument("--no-evaluate", action="store_true", help="Run outputs but skip reference evaluation.")
    parser.add_argument("--allow-whisperseg-cpu", action="store_true", help="Do not fail when WhisperSeg ONNX CUDA is unavailable.")
    parser.add_argument("--translate", action="store_true", help="Run translation too. Default is Japanese-only SRT generation.")
    parser.add_argument("--subtitle-mode", default="zh")
    parser.add_argument("--transcription-timeout", type=int, default=int(os.getenv("TRANSCRIPTION_TIMEOUT_S", "300") or "300"))
    parser.add_argument("--time-pad", type=float, default=0.35)
    parser.add_argument("--low-threshold", type=float, default=0.55)
    parser.add_argument("--good-threshold", type=float, default=0.82)
    parser.add_argument("--review-limit", type=int, default=120)
    parser.set_defaults(group_by_video=True)
    args = parser.parse_args()
    if args.evaluate_only and args.no_evaluate:
        raise SystemExit("--evaluate-only and --no-evaluate cannot be used together")
    if args.task_name is None:
        args.task_name = f"vad-compare-{args.label_prefix or default_asr_label(args.asr_backend)}"
    return args


def main_cli() -> None:
    args = parse_args()
    configure_environment(args)
    paths = make_task_paths(args.task_name)
    ensure_dirs(paths)
    cases = selected_cases(args)
    vads = selected_vads(args)

    if not args.evaluate_only and not args.allow_whisperseg_cpu:
        require_whisperseg_cuda()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        for label, vad_backend in vads:
            try:
                if args.evaluate_only:
                    previous = existing_result(
                        paths=paths,
                        args=args,
                        video=case.video,
                        label=label,
                        vad_backend=vad_backend,
                    )
                    if previous is None:
                        raise RuntimeError(f"missing existing output for {case.video.stem} {label}")
                    results.append(previous)
                else:
                    results.append(
                        run_one(
                            paths=paths,
                            args=args,
                            video=case.video,
                            label=label,
                            vad_backend=vad_backend,
                        )
                    )
            except Exception as exc:
                failures.append(
                    {
                        "video": case.video.name,
                        "video_stem": case.video.stem,
                        "label": label,
                        "vad_backend": vad_backend,
                        "asr_backend": args.asr_backend,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"=== FAIL {case.video.name} {label}: {exc} ===", flush=True)

    evaluations: list[dict[str, Any]] = []
    if not args.no_evaluate:
        for case in cases:
            evaluations.append(
                evaluate_video(
                    paths=paths,
                    case=case,
                    run_results=results,
                    time_pad=args.time_pad,
                    low_threshold=args.low_threshold,
                    good_threshold=args.good_threshold,
                    review_limit=args.review_limit,
                )
            )

    all_results = results + failures
    write_summary(paths=paths, args=args, cases=cases, vads=vads, results=all_results, evaluations=evaluations)
    print(f"summary_json={project_rel(paths.summary_json)}", flush=True)
    print(f"summary_md={project_rel(paths.summary_md)}", flush=True)
    if paths.summary_metrics_csv.exists():
        print(f"summary_metrics={project_rel(paths.summary_metrics_csv)}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed; see {project_rel(paths.summary_json)}")


if __name__ == "__main__":
    main_cli()
