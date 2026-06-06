#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.subtitles.analyze_subtitle_cue_merge_candidates import (
    build_summary as build_planner_summary,
)
from tools.subtitles.probe_speaker_sidecar import (
    build_adjacent_speaker_change_rows,
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("expected at least one threshold")
    return out


def parse_policy_list(value: str) -> list[str]:
    out: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in {"block", "penalize", "ignore"}:
            raise ValueError(f"unsupported speaker policy: {item}")
        out.append(item)
    if not out:
        raise ValueError("expected at least one speaker policy")
    return out


def pair_score_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(float(row.get("speaker_change_score") or 0.0) for row in pairs)
    if not scores:
        return {
            "pair_count": 0,
            "speaker_change_count": 0,
            "score_min": 0.0,
            "score_p50": 0.0,
            "score_p90": 0.0,
            "score_p95": 0.0,
            "score_max": 0.0,
        }

    def quantile(q: float) -> float:
        return round(scores[min(len(scores) - 1, int(q * (len(scores) - 1)))], 6)

    return {
        "pair_count": len(pairs),
        "speaker_change_count": sum(1 for row in pairs if row.get("speaker_change")),
        "score_min": quantile(0.0),
        "score_p50": quantile(0.5),
        "score_p90": quantile(0.9),
        "score_p95": quantile(0.95),
        "score_max": quantile(1.0),
    }


def run_case(
    *,
    embeddings: list[dict[str, Any]],
    bilingual_path: Path,
    timings_path: Path | None,
    diagnostics_path: Path | None,
    output_dir: Path,
    threshold: float,
    speaker_policy: str,
    fallback_risk_policy: str,
    video_fps: float,
    min_score: float,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
    max_reading_units_per_s: float,
) -> dict[str, Any]:
    case_name = f"th{int(round(threshold * 100)):02d}-{speaker_policy}"
    case_dir = output_dir / case_name
    pairs = build_adjacent_speaker_change_rows(embeddings, threshold=threshold)
    pairs_path = case_dir / "speaker_pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    planner = build_planner_summary(
        bilingual_path=bilingual_path,
        timings_path=timings_path,
        output_dir=case_dir,
        video_fps=video_fps,
        min_score=min_score,
        max_gap_s=max_gap_s,
        max_combined_s=max_combined_s,
        max_text_units=max_text_units,
        max_reading_units_per_s=max_reading_units_per_s,
        diagnostics_path=diagnostics_path,
        speaker_pairs_path=pairs_path,
        speaker_change_policy=speaker_policy,
        fallback_risk_policy=fallback_risk_policy,
    )
    return {
        "case": case_name,
        "threshold": threshold,
        "speaker_change_policy": speaker_policy,
        "fallback_risk_policy": fallback_risk_policy,
        "speaker_pairs": pair_score_summary(pairs),
        "planner_summary": project_rel(case_dir / "summary.json"),
        "planner_blocks": {
            "before": planner["before"]["block_count"],
            "after": planner["after"]["block_count"],
            "delta": planner["delta"]["block_count"],
            "merges": planner["after"]["planner_merge_count"],
        },
        "quality": {
            "short_segment_ratio": planner["after"]["quality"]["short_segment_ratio"],
            "per_min_subtitle_count": planner["after"]["quality"]["per_min_subtitle_count"],
            "kana_only_ratio": planner["after"]["quality"]["kana_only_ratio"],
            "repetition_ratio": planner["after"]["quality"]["repetition_ratio"],
            "subtitle_overlap_count": planner["after"]["quality"]["subtitle_overlap_count"],
        },
        "planner_blockers": planner["pair_analysis"]["planner_blocker_counts"],
        "constraints": planner["pair_analysis"]["constraint_counts"],
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Speaker Sidecar Cue Planner Sweep",
        "",
        f"- embeddings: `{summary['source_embeddings']}`",
        f"- bilingual: `{summary['source_bilingual']}`",
        f"- diagnostics: `{summary['source_diagnostics']}`",
        f"- thresholds: `{summary['thresholds']}`",
        f"- speaker_policies: `{summary['speaker_policies']}`",
        "",
        "| case | changes | merges | blocks | short | per_min | kana_only | overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        blocks = row["planner_blocks"]
        quality = row["quality"]
        lines.append(
            "| {case} | {changes} | {merges} | {before}->{after} | {short:.6f} | {per_min:.2f} | {kana:.6f} | {overlap} |".format(
                case=row["case"],
                changes=row["speaker_pairs"]["speaker_change_count"],
                merges=blocks["merges"],
                before=blocks["before"],
                after=blocks["after"],
                short=quality["short_segment_ratio"],
                per_min=quality["per_min_subtitle_count"],
                kana=quality["kana_only_ratio"],
                overlap=quality["subtitle_overlap_count"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_sweep(
    *,
    embeddings_path: Path,
    bilingual_path: Path,
    timings_path: Path | None,
    diagnostics_path: Path | None,
    output_dir: Path,
    thresholds: list[float],
    speaker_policies: list[str],
    fallback_risk_policy: str,
    video_fps: float,
    min_score: float,
    max_gap_s: float,
    max_combined_s: float,
    max_text_units: float,
    max_reading_units_per_s: float,
) -> dict[str, Any]:
    embeddings = read_jsonl(embeddings_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        for speaker_policy in speaker_policies:
            results.append(
                run_case(
                    embeddings=embeddings,
                    bilingual_path=bilingual_path,
                    timings_path=timings_path,
                    diagnostics_path=diagnostics_path,
                    output_dir=output_dir,
                    threshold=threshold,
                    speaker_policy=speaker_policy,
                    fallback_risk_policy=fallback_risk_policy,
                    video_fps=video_fps,
                    min_score=min_score,
                    max_gap_s=max_gap_s,
                    max_combined_s=max_combined_s,
                    max_text_units=max_text_units,
                    max_reading_units_per_s=max_reading_units_per_s,
                )
            )
    summary = {
        "source_embeddings": project_rel(embeddings_path),
        "source_bilingual": project_rel(bilingual_path),
        "source_timings": project_rel(timings_path),
        "source_diagnostics": project_rel(diagnostics_path),
        "output_dir": project_rel(output_dir),
        "thresholds": thresholds,
        "speaker_policies": speaker_policies,
        "fallback_risk_policy": fallback_risk_policy,
        "planner": {
            "video_fps": video_fps,
            "min_score": min_score,
            "max_gap_s": max_gap_s,
            "max_combined_s": max_combined_s,
            "max_text_units": max_text_units,
            "max_reading_units_per_s": max_reading_units_per_s,
        },
        "embedding_count": len(embeddings),
        "results": results,
    }
    write_json(output_dir / "sweep_summary.json", summary)
    (output_dir / "sweep_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep speaker sidecar thresholds and cue-planner policies. "
            "This is offline diagnostics only."
        )
    )
    parser.add_argument("--embeddings", required=True, help="speaker_embeddings.jsonl")
    parser.add_argument("--bilingual", required=True, help="bilingual.json")
    parser.add_argument("--timings", default="", help="timings.json")
    parser.add_argument("--diagnostics", default="", help="alignment diagnostics JSONL")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/speaker-sidecar-cue-planner-sweep",
    )
    parser.add_argument("--thresholds", default="0.75,0.85,0.95")
    parser.add_argument("--speaker-policies", default="block,penalize")
    parser.add_argument(
        "--fallback-risk-policy",
        choices=("block", "penalize", "ignore"),
        default="penalize",
    )
    parser.add_argument("--video-fps", type=float, default=30000 / 1001)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--max-gap-s", type=float, default=1.2)
    parser.add_argument("--max-combined-s", type=float, default=6.5)
    parser.add_argument("--max-text-units", type=float, default=56.0)
    parser.add_argument(
        "--max-reading-units-per-s",
        type=float,
        default=0.0,
        help="Optional reading-density gate passed to the cue planner; 0 disables.",
    )
    args = parser.parse_args(argv)

    summary = build_sweep(
        embeddings_path=project_path(args.embeddings),
        bilingual_path=project_path(args.bilingual),
        timings_path=project_path(args.timings) if args.timings else None,
        diagnostics_path=project_path(args.diagnostics) if args.diagnostics else None,
        output_dir=project_path(args.output_dir),
        thresholds=parse_float_list(args.thresholds),
        speaker_policies=parse_policy_list(args.speaker_policies),
        fallback_risk_policy=args.fallback_risk_policy,
        video_fps=float(args.video_fps),
        min_score=float(args.min_score),
        max_gap_s=float(args.max_gap_s),
        max_combined_s=float(args.max_combined_s),
        max_text_units=float(args.max_text_units),
        max_reading_units_per_s=float(args.max_reading_units_per_s),
    )
    print(
        "summary={path} cases={cases}".format(
            path=project_rel(project_path(args.output_dir) / "sweep_summary.json"),
            cases=len(summary["results"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
