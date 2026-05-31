#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.chunk_packer import pack_vad_segments  # noqa: E402
from vad.base import SpeechSegment  # noqa: E402


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def compact_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(min(values), 6),
        "p50": round(quantile(values, 0.5), 6),
        "p90": round(quantile(values, 0.9), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def normalize_segments(raw_segments: Any) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for item in raw_segments or []:
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        score = item.get("score")
        if end > start:
            segments.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    score=None if score is None else float(score),
                )
            )
    return sorted(segments, key=lambda item: (item.start, item.end))


def chunk_id(row: dict[str, Any]) -> int | None:
    value = row.get("chunk_index")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def risk_indices_from_diagnostics(diagnostics: list[dict[str, Any]]) -> set[int]:
    risky: set[int] = set()
    for row in diagnostics:
        if str(row.get("fallback_subtype") or "") != "vad_coarse_after_sentinel":
            continue
        index = chunk_id(row)
        if index is not None:
            risky.add(index)
    return risky


def analyze_valley_splits(
    *,
    vad_cache: Path,
    frame_scores_path: Path,
    output_dir: Path,
    diagnostics_path: Path | None,
    min_core_frames: int,
    target_core_frames: int,
    min_valley_frames: int,
    min_child_frames: int,
    max_children: int,
    valley_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = read_json(vad_cache)
    scores_payload = read_json(frame_scores_path)
    scores = [float(value) for value in (scores_payload.get("scores") or [])]
    frame_hop_s = float(
        scores_payload.get("frame_hop_s")
        or cache_payload.get("runtime_vad_signature", {}).get("frame_hop_s")
        or 0.02
    )
    diagnostics = read_jsonl(diagnostics_path) if diagnostics_path else []
    risky_indices = risk_indices_from_diagnostics(diagnostics)

    original_chunks = cache_payload.get("processing_spans") or []
    rows: list[dict[str, Any]] = []
    new_chunk_count = 0
    split_chunk_count = 0
    risk_split_count = 0
    child_durations: list[float] = []
    original_durations: list[float] = []

    for index, item in enumerate(original_chunks):
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        original_duration = max(0.0, end - start)
        original_durations.append(original_duration)
        segments = normalize_segments(item.get("vad_segments") or [])
        simulated = pack_vad_segments(
            segments,
            frame_hop_s=frame_hop_s,
            window_frames=899,
            reserve_frames=45,
            target_padding_frames=60,
            gap_merge_frames=45,
            pre_asr_valley_split_enabled=True,
            pre_asr_valley_split_min_core_frames=min_core_frames,
            pre_asr_valley_split_target_core_frames=target_core_frames,
            pre_asr_valley_split_min_valley_frames=min_valley_frames,
            pre_asr_valley_split_min_child_frames=min_child_frames,
            pre_asr_valley_split_max_children=max_children,
            pre_asr_valley_split_threshold=valley_threshold,
            frame_scores=scores,
        )
        child_count = max(0, len(simulated))
        split = child_count > 1 or any(chunk.valley_split_count for chunk in simulated)
        if split:
            split_chunk_count += 1
            if index in risky_indices:
                risk_split_count += 1
        new_chunk_count += child_count
        child_durations.extend(float(chunk.duration) for chunk in simulated)
        rows.append(
            {
                "chunk_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_s": round(original_duration, 3),
                "vad_seg_count": len(segments),
                "risk_vad_coarse_after_sentinel": index in risky_indices,
                "child_count": child_count,
                "valley_split": split,
                "children": [
                    {
                        "start": round(chunk.start, 3),
                        "end": round(chunk.end, 3),
                        "duration_s": round(chunk.duration, 3),
                        "core_start": None if chunk.core_start is None else round(chunk.core_start, 3),
                        "core_end": None if chunk.core_end is None else round(chunk.core_end, 3),
                        "split_reason": chunk.split_reason,
                        "split_policy": chunk.split_policy,
                        "valley_split_count": chunk.valley_split_count,
                        "valley_score_min": chunk.valley_score_min,
                    }
                    for chunk in simulated
                ],
            }
        )

    reason_counts = Counter()
    for row in rows:
        if row["risk_vad_coarse_after_sentinel"]:
            reason_counts["vad_coarse_after_sentinel"] += 1
        if row["valley_split"]:
            reason_counts["valley_split"] += 1
        if row["vad_seg_count"] <= 1 and row["duration_s"] >= min_core_frames * frame_hop_s:
            reason_counts["long_continuous_island"] += 1

    summary = {
        "vad_cache": project_rel(vad_cache),
        "frame_scores": project_rel(frame_scores_path),
        "diagnostics": project_rel(diagnostics_path),
        "frame_hop_s": frame_hop_s,
        "config": {
            "min_core_frames": min_core_frames,
            "target_core_frames": target_core_frames,
            "min_valley_frames": min_valley_frames,
            "min_child_frames": min_child_frames,
            "max_children": max_children,
            "valley_threshold": valley_threshold,
        },
        "original_chunk_count": len(rows),
        "new_chunk_count": new_chunk_count,
        "chunk_growth_ratio": (new_chunk_count / len(rows)) if rows else 0.0,
        "split_chunk_count": split_chunk_count,
        "risk_vad_coarse_after_sentinel_count": len(risky_indices),
        "risk_split_count": risk_split_count,
        "risk_split_ratio": (risk_split_count / len(risky_indices)) if risky_indices else 0.0,
        "original_duration_s": compact_stats(original_durations),
        "simulated_child_duration_s": compact_stats(child_durations),
        "reason_counts": dict(reason_counts.most_common()),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "valley_split_plan.jsonl", rows)
    (output_dir / "summary.md").write_text(
        build_markdown(summary),
        encoding="utf-8",
    )
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# R16 Valley Split Offline Analysis",
        "",
        f"- vad cache: `{summary['vad_cache']}`",
        f"- frame scores: `{summary['frame_scores']}`",
        f"- diagnostics: `{summary['diagnostics']}`",
        f"- original chunks: {summary['original_chunk_count']}",
        f"- simulated chunks: {summary['new_chunk_count']}",
        f"- chunk growth ratio: {summary['chunk_growth_ratio']:.3f}",
        f"- split chunks: {summary['split_chunk_count']}",
        f"- sentinel risk chunks: {summary['risk_vad_coarse_after_sentinel_count']}",
        f"- sentinel risk split: {summary['risk_split_count']} ({summary['risk_split_ratio']:.3f})",
        "",
        "## Duration",
        "",
        f"- original: `{summary['original_duration_s']}`",
        f"- simulated children: `{summary['simulated_child_duration_s']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in summary["reason_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    summary = analyze_valley_splits(
        vad_cache=project_path(args.vad_cache),
        frame_scores_path=project_path(args.frame_scores),
        output_dir=project_path(args.output_dir),
        diagnostics_path=project_path(args.diagnostics) if args.diagnostics else None,
        min_core_frames=args.min_core_frames,
        target_core_frames=args.target_core_frames,
        min_valley_frames=args.min_valley_frames,
        min_child_frames=args.min_child_frames,
        max_children=args.max_children,
        valley_threshold=args.valley_threshold,
    )
    print(f"summary={project_rel(Path(args.output_dir) / 'summary.json')}")
    print(
        "chunks={old}->{new} growth={growth:.3f} risk_split={risk}/{total}".format(
            old=summary["original_chunk_count"],
            new=summary["new_chunk_count"],
            growth=summary["chunk_growth_ratio"],
            risk=summary["risk_split_count"],
            total=summary["risk_vad_coarse_after_sentinel_count"],
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline R16 analysis: simulate frame-score valley splitting on packed VAD chunks."
    )
    parser.add_argument("--vad-cache", required=True)
    parser.add_argument("--frame-scores", required=True)
    parser.add_argument("--diagnostics", default="")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "r16-valley-analysis"),
    )
    parser.add_argument("--min-core-frames", type=int, default=420)
    parser.add_argument("--target-core-frames", type=int, default=270)
    parser.add_argument("--min-valley-frames", type=int, default=6)
    parser.add_argument("--min-child-frames", type=int, default=45)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--valley-threshold", type=float, default=0.20)
    args = parser.parse_args(argv)
    if args.min_core_frames < 0:
        parser.error("--min-core-frames must be non-negative")
    if args.target_core_frames < 0:
        parser.error("--target-core-frames must be non-negative")
    if args.min_valley_frames < 0:
        parser.error("--min-valley-frames must be non-negative")
    if args.min_child_frames < 0:
        parser.error("--min-child-frames must be non-negative")
    if args.max_children <= 0:
        parser.error("--max-children must be positive")
    if args.valley_threshold < 0.0:
        parser.error("--valley-threshold must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
