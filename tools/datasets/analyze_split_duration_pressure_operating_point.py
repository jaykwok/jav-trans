#!/usr/bin/env python3
"""Read-only analysis for duration-pressure gated Split operating points."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.datasets.analyze_split_adaptive_operating_point import (  # noqa: E402
    _feature_dirs,
    _selected_adaptive_cuts,
    read_jsonl,
    safe_float,
)

SUMMARY_SCHEMA = "split_duration_pressure_operating_point_summary_v1"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _internal_cut_candidates(
    split_rows: Iterable[Mapping[str, Any]],
    *,
    start: float,
    end: float,
    min_chunk_after_split_s: float,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for row in split_rows:
        if str(row.get("label") or "") != "cut":
            continue
        time_s = safe_float(row.get("time_s"))
        if start + min_chunk_after_split_s <= time_s <= end - min_chunk_after_split_s:
            candidates.append({"time_s": time_s, "p_cut": safe_float(row.get("p_cut"))})
    return candidates


def _recursive_pressure_cuts(
    candidates: list[Mapping[str, float]],
    *,
    start: float,
    end: float,
    floor: float,
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
) -> list[float]:
    cuts: list[float] = []
    stack = [(start, end)]
    while stack:
        seg_start, seg_end = stack.pop()
        if seg_end - seg_start < long_chunk_min_s:
            continue
        eligible = [
            row
            for row in candidates
            if seg_start + min_chunk_after_split_s
            <= safe_float(row.get("time_s"))
            <= seg_end - min_chunk_after_split_s
            and safe_float(row.get("p_cut")) >= floor
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda row: safe_float(row.get("p_cut")))
        cut_time = safe_float(best.get("time_s"))
        cuts.append(cut_time)
        stack.append((seg_start, cut_time))
        stack.append((cut_time, seg_end))
    return sorted(cuts)


def _quantile_99(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 100:
        return round(max(values), 3)
    return round(statistics.quantiles(values, n=100)[98], 3)


def analyze_duration_pressure(
    *,
    reexport_dir: Path,
    floors: list[float],
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
    pure_adaptive_policy: Mapping[str, float],
) -> dict[str, Any]:
    long_chunks: list[dict[str, Any]] = []
    adaptive_new_in_long = 0
    adaptive_new_in_short = 0
    adaptive_new_unassigned = 0
    pressure: dict[str, dict[str, Any]] = {
        f"{floor:.2f}": {
            "new_cuts": 0,
            "chunks_hit": 0,
            "residual_long_segments": 0,
            "residual_max_s": 0.0,
            "_post_split_durations": [],
        }
        for floor in floors
    }

    for feature_dir in _feature_dirs(reexport_dir):
        candidate_path = feature_dir / "pre_asr_candidates.jsonl"
        split_path = feature_dir / "semantic_split_features.jsonl"
        if not candidate_path.exists() or not split_path.exists():
            continue
        chunks = read_jsonl(candidate_path)
        split_rows = read_jsonl(split_path)
        window_long: list[dict[str, Any]] = []
        for chunk in chunks:
            start = safe_float(chunk.get("start"))
            end = safe_float(chunk.get("end"), start)
            duration = safe_float(chunk.get("duration_s"), end - start)
            if duration < long_chunk_min_s:
                continue
            candidates = _internal_cut_candidates(
                split_rows,
                start=start,
                end=end,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            max_p = max((row["p_cut"] for row in candidates), default=0.0)
            window_long.append(
                {
                    "candidate_id": str(chunk.get("candidate_id") or chunk.get("sample_id") or ""),
                    "window": feature_dir.name,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "candidates": candidates,
                    "max_p": max_p,
                }
            )
        long_chunks.extend(window_long)

        adaptive_cuts = [
            cut
            for cut in _selected_adaptive_cuts(
                split_rows,
                abs_floor=float(pure_adaptive_policy["abs_floor"]),
                percentile_floor=float(pure_adaptive_policy["percentile_floor"]),
                z_floor=float(pure_adaptive_policy["z_floor"]),
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            if not bool(cut.get("accepted_current"))
        ]
        for cut in adaptive_cuts:
            time_s = safe_float(cut.get("time_s"))
            assigned = False
            for chunk in chunks:
                start = safe_float(chunk.get("start"))
                end = safe_float(chunk.get("end"), start)
                if start + min_chunk_after_split_s <= time_s <= end - min_chunk_after_split_s:
                    duration = safe_float(chunk.get("duration_s"), end - start)
                    if duration >= long_chunk_min_s:
                        adaptive_new_in_long += 1
                    else:
                        adaptive_new_in_short += 1
                    assigned = True
                    break
            if not assigned:
                adaptive_new_unassigned += 1

        for floor in floors:
            bucket = pressure[f"{floor:.2f}"]
            durations = bucket["_post_split_durations"]
            for chunk in window_long:
                cuts = _recursive_pressure_cuts(
                    chunk["candidates"],
                    start=float(chunk["start"]),
                    end=float(chunk["end"]),
                    floor=floor,
                    long_chunk_min_s=long_chunk_min_s,
                    min_chunk_after_split_s=min_chunk_after_split_s,
                )
                bucket["new_cuts"] += len(cuts)
                if cuts:
                    bucket["chunks_hit"] += 1
                bounds = sorted([float(chunk["start"]), *cuts, float(chunk["end"])])
                for seg_start, seg_end in zip(bounds, bounds[1:]):
                    duration = seg_end - seg_start
                    durations.append(duration)
                    if duration >= long_chunk_min_s:
                        bucket["residual_long_segments"] += 1
                        bucket["residual_max_s"] = max(bucket["residual_max_s"], duration)

    ceiling: dict[str, Any] = {}
    for floor in sorted({0.45, 0.50, 0.55, 0.60, 0.75, *floors}):
        below = [chunk for chunk in long_chunks if float(chunk["max_p"]) < floor]
        ceiling[f"max_p_below_{floor:.2f}"] = {
            "count": len(below),
            "durations": sorted(round(float(chunk["duration"]), 2) for chunk in below),
        }
    unfixable = [chunk for chunk in long_chunks if float(chunk["max_p"]) < 0.50]
    for bucket in pressure.values():
        durations = list(bucket.pop("_post_split_durations"))
        bucket["post_split_p99_s"] = _quantile_99(durations)
        bucket["post_split_max_s"] = round(max(durations), 3) if durations else None
        bucket["residual_max_s"] = round(float(bucket["residual_max_s"]), 6)

    return {
        "schema": SUMMARY_SCHEMA,
        "reexport_dir": repo_rel(reexport_dir),
        "long_chunk_min_s": float(long_chunk_min_s),
        "min_chunk_after_split_s": float(min_chunk_after_split_s),
        "long_chunk_count": len(long_chunks),
        "ceiling": ceiling,
        "unfixable_at_floor_0.50": [
            {
                "candidate_id": str(chunk["candidate_id"]),
                "window": str(chunk["window"]),
                "duration_s": round(float(chunk["duration"]), 2),
                "max_internal_p_cut": round(float(chunk["max_p"]), 4),
            }
            for chunk in sorted(unfixable, key=lambda item: -float(item["duration"]))
        ],
        "pure_adaptive_policy": dict(pure_adaptive_policy),
        "pure_adaptive_new_cut_placement": {
            "inside_long_chunks": adaptive_new_in_long,
            "inside_short_chunks": adaptive_new_in_short,
            "unassigned_edge_zone": adaptive_new_unassigned,
        },
        "duration_pressure_variant": pressure,
    }


def export_duration_pressure(
    *,
    reexport_dir: Path,
    output_dir: Path,
    floors: list[float],
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
    pure_adaptive_policy: Mapping[str, float],
) -> dict[str, Any]:
    summary = analyze_duration_pressure(
        reexport_dir=reexport_dir,
        floors=floors,
        long_chunk_min_s=long_chunk_min_s,
        min_chunk_after_split_s=min_chunk_after_split_s,
        pure_adaptive_policy=pure_adaptive_policy,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["output_dir"] = repo_rel(output_dir)
    write_json(output_dir / "summary.json", summary)
    return summary


def _parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one float")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--floors", type=_parse_floats, default=[0.45, 0.50, 0.55])
    parser.add_argument("--long-chunk-min-s", type=float, default=15.0)
    parser.add_argument("--min-chunk-after-split-s", type=float, default=1.2)
    parser.add_argument("--adaptive-abs-floor", type=float, default=0.50)
    parser.add_argument("--adaptive-percentile-floor", type=float, default=0.90)
    parser.add_argument("--adaptive-z-floor", type=float, default=1.00)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_duration_pressure(
        reexport_dir=project_path(args.reexport_dir),
        output_dir=project_path(args.output_dir),
        floors=list(args.floors),
        long_chunk_min_s=float(args.long_chunk_min_s),
        min_chunk_after_split_s=float(args.min_chunk_after_split_s),
        pure_adaptive_policy={
            "abs_floor": float(args.adaptive_abs_floor),
            "percentile_floor": float(args.adaptive_percentile_floor),
            "z_floor": float(args.adaptive_z_floor),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
