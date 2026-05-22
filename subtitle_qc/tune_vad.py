#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import compare_vad  # noqa: E402
import review_content  # noqa: E402


FUSION_BASE = {
    "FUSION_VAD_PRIMARY_WEIGHT": "0.45",
    "FUSION_VAD_GATE_WEIGHT": "0.25",
    "FUSION_VAD_RMS_WEIGHT": "0.15",
    "FUSION_VAD_SPECTRAL_FLUX_WEIGHT": "0.10",
    "FUSION_VAD_DURATION_WEIGHT": "0.05",
    "FUSION_VAD_MIN_SCORE": "0.45",
    "FUSION_VAD_MIN_GATE_OVERLAP_RATIO": "0.05",
}


BUILTIN_TRIALS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("adaptive_base", "whisperseg-adaptive", {}),
    ("adaptive_t32", "whisperseg-adaptive", {"WHISPERSEG_THRESHOLD": "0.32"}),
    ("adaptive_t38", "whisperseg-adaptive", {"WHISPERSEG_THRESHOLD": "0.38"}),
    ("lite_base", "fusion_lite", {}),
    ("lite_gate08", "fusion_lite", {**FUSION_BASE, "FUSION_VAD_MIN_GATE_OVERLAP_RATIO": "0.08"}),
    ("lite_strict47_gate08", "fusion_lite", {**FUSION_BASE, "FUSION_VAD_MIN_SCORE": "0.47", "FUSION_VAD_MIN_GATE_OVERLAP_RATIO": "0.08"}),
    ("lite_p50_g20_m47", "fusion_lite", {**FUSION_BASE, "FUSION_VAD_PRIMARY_WEIGHT": "0.50", "FUSION_VAD_GATE_WEIGHT": "0.20", "FUSION_VAD_MIN_SCORE": "0.47"}),
)


@dataclass(frozen=True)
class Trial:
    label: str
    backend: str
    env: dict[str, str]


def parse_env_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit(f"env override must be KEY=VALUE: {value}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"empty env key in override: {value}")
    return key, raw.strip()


def parse_trial(value: str) -> Trial:
    parts = value.split(":")
    if len(parts) < 2:
        raise SystemExit("--trial format is label:backend[:KEY=VALUE,...]")
    label = compare_vad.safe_label(parts[0])
    backend = parts[1].strip()
    if not label or not backend:
        raise SystemExit(f"invalid trial: {value}")
    env: dict[str, str] = {}
    if len(parts) > 2 and parts[2].strip():
        for item in parts[2].split(","):
            key, raw = parse_env_pair(item)
            env[key] = raw
    return Trial(label=label, backend=backend, env=env)


def selected_trials(args: argparse.Namespace) -> list[Trial]:
    if args.trial:
        return [parse_trial(value) for value in args.trial]
    return [Trial(label, backend, dict(env)) for label, backend, env in BUILTIN_TRIALS]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_content_metrics(paths: compare_vad.TaskPaths, cases: list[compare_vad.VideoCase], trials: list[Trial]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        candidates = []
        for trial in trials:
            path = compare_vad.video_artifact_path(case.video, f"{case.video.stem}.{trial.label}.srt")
            if path.exists():
                candidates.append(f"{trial.label}={compare_vad.project_rel(path)}")
        if not candidates:
            continue
        out_dir = compare_vad.subtitle_qc_task_dir(case.video, paths.summary_root.name) / "content_review"
        review_args = argparse.Namespace(
            candidate=candidates,
            all_candidates=False,
            mode=None,
            output_dir=str(out_dir),
            window_minutes=45.0,
            long_duration_s=6.5,
            long_text_chars=60,
            fast_cps=16.0,
            noise_pattern=None,
        )
        config = review_content.build_review_config(review_args)
        review_content.run_for_video(case.video, review_args, config, len(cases))
        metrics_path = out_dir / "content_review_metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["video"] = case.video.stem
                rows.append(row)
    return rows


def write_tuning_summary(
    *,
    paths: compare_vad.TaskPaths,
    args: argparse.Namespace,
    cases: list[compare_vad.VideoCase],
    trials: list[Trial],
    results: list[dict[str, Any]],
    content_rows: list[dict[str, Any]],
) -> None:
    paths.summary_root.mkdir(parents=True, exist_ok=True)
    trial_map = {trial.label: trial for trial in trials}
    by_label_video = {
        (row.get("video_stem"), row.get("label")): row
        for row in results
        if row.get("status") in {"done", "existing"}
    }
    content_by_label: dict[str, dict[str, float]] = {}
    for row in content_rows:
        label = str(row.get("label") or "")
        if not label:
            continue
        bucket = content_by_label.setdefault(label, {"issue_score": 0.0, "issue_cues": 0.0, "cue_count": 0.0, "videos": 0.0})
        bucket["issue_score"] += float(row.get("issue_score") or 0.0)
        bucket["issue_cues"] += float(row.get("issue_cues") or 0.0)
        bucket["cue_count"] += float(row.get("cue_count") or 0.0)
        bucket["videos"] += 1.0

    rows = []
    for trial in trials:
        content = content_by_label.get(trial.label, {})
        rows.append(
            {
                "label": trial.label,
                "backend": trial.backend,
                "videos": int(content.get("videos", 0.0)),
                "issue_score_total": f"{content.get('issue_score', 0.0):.2f}",
                "issue_cues_total": int(content.get("issue_cues", 0.0)),
                "cue_count_total": int(content.get("cue_count", 0.0)),
                "env_json": json.dumps(trial.env, ensure_ascii=False, sort_keys=True),
            }
        )
    rows.sort(key=lambda row: (float(row["issue_score_total"]), int(row["issue_cues_total"]), -int(row["cue_count_total"])))

    csv_path = paths.summary_root / "tuning_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "task": compare_vad.safe_label(args.task_name),
        "asr_backend": args.asr_backend,
        "videos": [compare_vad.project_rel(case.video) for case in cases],
        "trials": [asdict(trial) for trial in trials],
        "results": results,
        "content_metrics": content_rows,
        "ranked": rows,
    }
    (paths.summary_root / "tuning_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VAD Tuning",
        "",
        f"- ASR backend: `{args.asr_backend}`",
        f"- Summary: `{compare_vad.project_rel(paths.summary_root)}`",
        f"- Runtime temp: `{compare_vad.project_rel(paths.root)}`",
        "",
        "## Content Review Ranking",
        "",
        "| rank | trial | backend | issue_score | issue_cues | cues | env |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        env = row["env_json"]
        lines.append(
            f"| {rank} | `{row['label']}` | `{row['backend']}` | {row['issue_score_total']} | "
            f"{row['issue_cues_total']} | {row['cue_count_total']} | `{env}` |"
        )
    lines.extend(["", "## Per Video", ""])
    lines.extend([
        "| video | trial | issue_score | issue_cues | cues | path |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for row in sorted(content_rows, key=lambda item: (item.get("video", ""), float(item.get("issue_score") or 0.0))):
        label = str(row.get("label") or "")
        result = by_label_video.get((row.get("video"), label), {})
        srt = (result.get("paths") or {}).get("srt") or row.get("path") or ""
        lines.append(
            f"| `{row.get('video')}` | `{label}` | {float(row.get('issue_score') or 0.0):.2f} | "
            f"{int(float(row.get('issue_cues') or 0))} | {int(float(row.get('cue_count') or 0))} | `{compare_vad.project_rel(srt)}` |"
        )
    (paths.summary_root / "tuning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAD parameter tuning trials and summarize content-review metrics.")
    parser.add_argument("--asr-backend", default=os.getenv("ASR_BACKEND") or "whisper-ja-1.5b")
    parser.add_argument("--asr-context", default=os.getenv("ASR_CONTEXT", ""))
    parser.add_argument("--trial", action="append", help="Trial as label:backend[:KEY=VALUE,...]. Repeatable. Defaults to a small whisperseg-adaptive/fusion_lite grid.")
    parser.add_argument("--video", action="append", required=True, help="Video path, name, or stem. Repeatable.")
    parser.add_argument("--video-dir", default="video")
    parser.add_argument("--task-name", default="vad-tuning")
    parser.add_argument("--force", action="store_true", help="Rerun even when trial SRT already exists.")
    parser.add_argument("--allow-whisperseg-cpu", action="store_true", help="Do not fail when WhisperSeg ONNX CUDA is unavailable.")
    parser.add_argument("--transcription-timeout", type=int, default=int(os.getenv("TRANSCRIPTION_TIMEOUT_S", "300") or "300"))
    parser.add_argument("--subtitle-mode", default="zh")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--skip-reference-eval", action="store_true", help="Kept for clarity; tune_vad always uses content review metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trials = selected_trials(args)
    paths = compare_vad.make_task_paths(args.task_name)
    compare_vad.ensure_dirs(paths)
    cases = [
        compare_vad.VideoCase(compare_vad.resolve_video(value, compare_vad.project_path(args.video_dir)), None)
        for value in args.video
    ]
    if not args.allow_whisperseg_cpu:
        compare_vad.require_whisperseg_cuda()

    run_args = argparse.Namespace(
        asr_backend=args.asr_backend,
        asr_context=args.asr_context,
        task_name=args.task_name,
        translate=args.translate,
        subtitle_mode=args.subtitle_mode,
        transcription_timeout=args.transcription_timeout,
        force=args.force,
        group_by_video=True,
    )
    compare_vad.configure_environment(run_args)

    results: list[dict[str, Any]] = []
    for case in cases:
        for trial in trials:
            run_args.env_overrides = dict(trial.env)
            try:
                results.append(
                    compare_vad.run_one(
                        paths=paths,
                        args=run_args,
                        video=case.video,
                        label=trial.label,
                        vad_backend=trial.backend,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "video": case.video.name,
                        "video_stem": case.video.stem,
                        "label": trial.label,
                        "vad_backend": trial.backend,
                        "asr_backend": args.asr_backend,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"=== FAIL {case.video.name} {trial.label}: {exc} ===", flush=True)

    content_rows = collect_content_metrics(paths, cases, trials)
    write_tuning_summary(paths=paths, args=args, cases=cases, trials=trials, results=results, content_rows=content_rows)
    print(f"summary_md={compare_vad.project_rel(paths.summary_root / 'tuning_summary.md')}", flush=True)
    print(f"summary_csv={compare_vad.project_rel(paths.summary_root / 'tuning_summary.csv')}", flush=True)
    if any(result.get("status") == "failed" for result in results):
        raise SystemExit(f"some trials failed; see {compare_vad.project_rel(paths.summary_root / 'tuning_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())