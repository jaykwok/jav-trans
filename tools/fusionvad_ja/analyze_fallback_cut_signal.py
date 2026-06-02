#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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


def row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(quantile(values, 0.50), 6),
        "p90": round(quantile(values, 0.90), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def scores_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    for key in keys:
        raw = payload.get(key)
        if raw is not None:
            return [float(value) for value in raw]
    return []


def contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(set(indices))
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous + 1))
        start = previous = index
    runs.append((start, previous + 1))
    return runs


def candidate_times_from_runs(
    runs: list[tuple[int, int]],
    *,
    frame_hop_s: float,
    core_start: float,
    core_end: float,
    min_child_s: float,
) -> list[float]:
    times: list[float] = []
    for start, end in runs:
        midpoint = (start + end) / 2.0 * frame_hop_s
        if midpoint - core_start < min_child_s:
            continue
        if core_end - midpoint < min_child_s:
            continue
        times.append(round(midpoint, 6))
    return sorted(set(times))


def greedy_split_max_child(
    *,
    core_start: float,
    core_end: float,
    candidates: list[float],
    target_child_s: float,
    min_child_s: float,
) -> tuple[bool, float, list[float]]:
    duration = max(0.0, core_end - core_start)
    if duration <= target_child_s:
        return True, duration, []
    selected: list[float] = []
    cursor = core_start
    remaining = sorted(time for time in candidates if core_start < time < core_end)
    while core_end - cursor > target_child_s:
        lower = cursor + min_child_s
        upper = min(cursor + target_child_s, core_end - min_child_s)
        eligible = [time for time in remaining if lower <= time <= upper]
        if not eligible:
            pieces = [*selected, core_end]
            starts = [core_start, *selected]
            max_child = max((end - start for start, end in zip(starts, pieces)), default=duration)
            return False, max(max_child, core_end - cursor), selected
        cut = max(eligible)
        selected.append(cut)
        remaining = [time for time in remaining if time > cut + 1e-6]
        cursor = cut
    pieces = [*selected, core_end]
    starts = [core_start, *selected]
    max_child = max((end - start for start, end in zip(starts, pieces)), default=duration)
    return max_child <= target_child_s + 1e-6, max_child, selected


def summarize_candidate_row(
    row: dict[str, Any],
    *,
    speech_scores: np.ndarray,
    cut_scores: np.ndarray,
    frame_hop_s: float,
    cut_threshold: float,
    valley_threshold: float,
    target_child_s: float,
    min_child_s: float,
) -> dict[str, Any]:
    core_start = row_float(row, "core_start", row_float(row, "start"))
    core_end = row_float(row, "core_end", row_float(row, "end", core_start))
    start_frame = max(0, int(np.floor(core_start / frame_hop_s)))
    end_frame = max(start_frame, int(np.ceil(core_end / frame_hop_s)))
    end_frame = min(end_frame, len(speech_scores))
    start_frame = min(start_frame, end_frame)
    speech_slice = speech_scores[start_frame:end_frame]
    cut_slice = cut_scores[start_frame:end_frame] if cut_scores.size else np.asarray([], dtype=np.float32)
    frame_indexes = list(range(start_frame, end_frame))
    valley_indexes = [
        index for local, index in enumerate(frame_indexes)
        if local < speech_slice.size and float(speech_slice[local]) <= valley_threshold
    ]
    cut_indexes = [
        index for local, index in enumerate(frame_indexes)
        if local < cut_slice.size and float(cut_slice[local]) >= cut_threshold
    ]
    valley_times = candidate_times_from_runs(
        contiguous_runs(valley_indexes),
        frame_hop_s=frame_hop_s,
        core_start=core_start,
        core_end=core_end,
        min_child_s=min_child_s,
    )
    cut_times = candidate_times_from_runs(
        contiguous_runs(cut_indexes),
        frame_hop_s=frame_hop_s,
        core_start=core_start,
        core_end=core_end,
        min_child_s=min_child_s,
    )
    combined_times = sorted(set([*valley_times, *cut_times]))
    feasible, max_child_s, selected = greedy_split_max_child(
        core_start=core_start,
        core_end=core_end,
        candidates=combined_times,
        target_child_s=target_child_s,
        min_child_s=min_child_s,
    )
    return {
        "chunk_index": row_int(row, "chunk_index", -1),
        "duration_s": round(row_float(row, "duration_s"), 6),
        "core_start": round(core_start, 6),
        "core_end": round(core_end, 6),
        "core_duration_s": round(max(0.0, core_end - core_start), 6),
        "split_reason": str(row.get("split_reason") or ""),
        "fallback_subtype": str(row.get("fallback_subtype") or ""),
        "speech_island_count": row_int(row, "speech_island_count"),
        "internal_gap_count": row_int(row, "internal_gap_count"),
        "internal_gap_max_s": round(row_float(row, "internal_gap_max_s"), 6),
        "longest_silence_s": round(row_float(row, "longest_silence_s"), 6),
        "text_chars": len(str(row.get("display_text") or "")),
        "speech_probability": stats([float(value) for value in speech_slice]),
        "cut_probability": stats([float(value) for value in cut_slice]),
        "valley_candidate_count": len(valley_times),
        "cut_candidate_count": len(cut_times),
        "combined_candidate_count": len(combined_times),
        "selected_cut_count": len(selected),
        "selected_cut_times_s": selected[:16],
        "can_reach_target_child": feasible,
        "max_child_s_after_greedy": round(max_child_s, 6),
    }


def analyze(
    *,
    unsafe_fallback_chunks: Path,
    frame_scores_path: Path,
    output_dir: Path,
    cut_threshold: float,
    valley_threshold: float,
    target_child_s: float,
    min_child_s: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(unsafe_fallback_chunks)
    if any("fallback_safe" in row or "fallback_reason" in row for row in raw_rows):
        rows = [
            row
            for row in raw_rows
            if str(row.get("fallback_reason") or row.get("fallback_subtype") or "")
            and not bool(row.get("fallback_safe"))
        ]
    else:
        rows = raw_rows
    score_payload = read_json(frame_scores_path)
    if not isinstance(score_payload, dict):
        raise ValueError(f"frame scores must be a JSON object: {frame_scores_path}")
    frame_hop_s = float(score_payload.get("frame_hop_s") or 0.02)
    speech_scores = np.asarray(
        scores_from_payload(score_payload, ("scores", "frame_scores")),
        dtype=np.float32,
    )
    cut_scores = np.asarray(
        scores_from_payload(score_payload, ("cut_scores", "cut_frame_scores")),
        dtype=np.float32,
    )
    if speech_scores.size <= 0:
        raise ValueError("frame scores do not contain speech probabilities")
    detail_rows = [
        summarize_candidate_row(
            row,
            speech_scores=speech_scores,
            cut_scores=cut_scores,
            frame_hop_s=frame_hop_s,
            cut_threshold=cut_threshold,
            valley_threshold=valley_threshold,
            target_child_s=target_child_s,
            min_child_s=min_child_s,
        )
        for row in rows
    ]
    feasible_rows = [row for row in detail_rows if row["can_reach_target_child"]]
    candidate_rows = [row for row in detail_rows if row["combined_candidate_count"] > 0]
    summary = {
        "unsafe_fallback_chunks": project_rel(unsafe_fallback_chunks),
        "frame_scores": project_rel(frame_scores_path),
        "output_dir": project_rel(output_dir),
        "frame_hop_s": frame_hop_s,
        "cut_threshold": cut_threshold,
        "valley_threshold": valley_threshold,
        "target_child_s": target_child_s,
        "min_child_s": min_child_s,
        "input_row_count": len(raw_rows),
        "row_count": len(detail_rows),
        "rows_with_any_candidate": len(candidate_rows),
        "rows_with_any_candidate_ratio": (len(candidate_rows) / len(detail_rows)) if detail_rows else 0.0,
        "rows_feasible_to_target": len(feasible_rows),
        "rows_feasible_to_target_ratio": (len(feasible_rows) / len(detail_rows)) if detail_rows else 0.0,
        "core_duration_s": stats([row["core_duration_s"] for row in detail_rows]),
        "max_child_s_after_greedy": stats([row["max_child_s_after_greedy"] for row in detail_rows]),
        "combined_candidate_count": stats([row["combined_candidate_count"] for row in detail_rows]),
        "selected_cut_count": stats([row["selected_cut_count"] for row in detail_rows]),
        "split_reason_counts": dict(Counter(str(row["split_reason"]) for row in detail_rows).most_common()),
        "fallback_subtype_counts": dict(
            Counter(str(row["fallback_subtype"]) for row in detail_rows).most_common()
        ),
        "note": (
            "Offline signal audit only. It does not rerun ASR or forced alignment, "
            "and it omits subtitle text from outputs except text length."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "cut_signal_details.jsonl", detail_rows)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fallback Cut Signal Analysis",
        "",
        f"- unsafe fallback chunks: `{summary['unsafe_fallback_chunks']}`",
        f"- frame scores: `{summary['frame_scores']}`",
        f"- rows: {summary['row_count']}",
        f"- rows with any candidate: {summary['rows_with_any_candidate']} ({summary['rows_with_any_candidate_ratio']:.3f})",
        f"- rows feasible to target: {summary['rows_feasible_to_target']} ({summary['rows_feasible_to_target_ratio']:.3f})",
        f"- target child: {summary['target_child_s']:.2f}s",
        f"- min child: {summary['min_child_s']:.2f}s",
        f"- cut threshold: {summary['cut_threshold']:.3f}",
        f"- valley threshold: {summary['valley_threshold']:.3f}",
        "",
        "## Distributions",
        "",
        f"- core duration: `{summary['core_duration_s']}`",
        f"- max child after greedy: `{summary['max_child_s_after_greedy']}`",
        f"- combined candidate count: `{summary['combined_candidate_count']}`",
        f"- selected cut count: `{summary['selected_cut_count']}`",
        f"- split reasons: `{summary['split_reason_counts']}`",
        "",
        "> Offline signal audit only. It does not rerun ASR or forced alignment, "
        "and detail output omits subtitle text except text length.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    summary = analyze(
        unsafe_fallback_chunks=project_path(args.unsafe_fallback_chunks),
        frame_scores_path=project_path(args.frame_scores),
        output_dir=project_path(args.output_dir),
        cut_threshold=args.cut_threshold,
        valley_threshold=args.valley_threshold,
        target_child_s=args.target_child_s,
        min_child_s=args.min_child_s,
    )
    print(f"summary={project_rel(project_path(args.output_dir) / 'summary.json')}")
    print(
        "rows={row_count} candidates={rows_with_any_candidate} "
        "feasible={rows_feasible_to_target} target={target_child_s:.2f}s".format(**summary)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether existing FusionVAD-JA speech/cut frame scores contain "
            "usable split candidates inside unsafe fallback chunks."
        )
    )
    parser.add_argument("--unsafe-fallback-chunks", required=True)
    parser.add_argument("--frame-scores", required=True)
    parser.add_argument("--cut-threshold", type=float, default=0.94)
    parser.add_argument("--valley-threshold", type=float, default=0.20)
    parser.add_argument("--target-child-s", type=float, default=9.0)
    parser.add_argument("--min-child-s", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "fallback-cut-signal-analysis"),
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.cut_threshold <= 1.0:
        parser.error("--cut-threshold must be in [0, 1]")
    if not 0.0 <= args.valley_threshold <= 1.0:
        parser.error("--valley-threshold must be in [0, 1]")
    if args.target_child_s <= 0.0:
        parser.error("--target-child-s must be positive")
    if args.min_child_s < 0.0:
        parser.error("--min-child-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
