#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def frame_count_for_duration(duration_s: float, frame_hop_s: float) -> int:
    if duration_s <= 0.0 or frame_hop_s <= 0.0:
        return 0
    return max(1, int(round(duration_s / frame_hop_s)))


def frame_index(time_s: float, *, frame_hop_s: float, frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    return max(0, min(frame_count - 1, int(round(float(time_s) / frame_hop_s))))


def mark_radius(values: list[int], index: int, radius: int) -> None:
    start = max(0, index - radius)
    end = min(len(values), index + radius + 1)
    for offset in range(start, end):
        values[offset] = 1


def segment_key(segment: Mapping[str, Any]) -> tuple[float, float]:
    return (round(float(segment.get("start", 0.0)), 6), round(float(segment.get("end", 0.0)), 6))


def find_drop_gap_zones(
    *,
    baseline_segments: list[Mapping[str, Any]],
    planned_segments: list[Mapping[str, Any]],
    min_gap_s: float,
) -> list[dict[str, float]]:
    zones: list[dict[str, float]] = []
    for baseline in baseline_segments:
        base_start = float(baseline.get("start", 0.0))
        base_end = float(baseline.get("end", 0.0))
        inside = sorted(
            [
                item
                for item in planned_segments
                if base_start <= float(item.get("start", 0.0)) <= float(item.get("end", 0.0)) <= base_end
            ],
            key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
        )
        if len(inside) < 2:
            continue
        for left, right in zip(inside, inside[1:]):
            gap_start = float(left.get("end", 0.0))
            gap_end = float(right.get("start", 0.0))
            if gap_end - gap_start >= min_gap_s:
                zones.append({"start": gap_start, "end": gap_end})
    return zones


def split_points_from_plan(
    *,
    baseline_segments: list[Mapping[str, Any]],
    planned_segments: list[Mapping[str, Any]],
    drop_gap_zones: list[Mapping[str, float]],
) -> list[dict[str, Any]]:
    baseline_keys = {segment_key(item) for item in baseline_segments}
    planned = sorted(
        planned_segments,
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    points: list[dict[str, Any]] = []
    for left, right in zip(planned, planned[1:]):
        left_end = float(left.get("end", 0.0))
        right_start = float(right.get("start", 0.0))
        if any(abs(float(zone["start"]) - left_end) < 1e-5 and abs(float(zone["end"]) - right_start) < 1e-5 for zone in drop_gap_zones):
            continue
        point_time = (left_end + right_start) / 2.0
        if segment_key(left) in baseline_keys and segment_key(right) in baseline_keys:
            continue
        points.append(
            {
                "time_s": point_time,
                "left_end_s": left_end,
                "right_start_s": right_start,
                "gap_s": max(0.0, right_start - left_end),
            }
        )
    return points


def build_target_row(
    detail: Mapping[str, Any],
    *,
    frame_hop_s: float,
    split_radius_frames: int,
    min_drop_gap_s: float,
) -> dict[str, Any]:
    audio_id = str(detail.get("audio_id") or "")
    duration_s = float(detail.get("duration_s") or 0.0)
    frame_count = frame_count_for_duration(duration_s, frame_hop_s)
    split_frames = [0] * frame_count
    drop_gap_frames = [0] * frame_count
    baseline_segments = list(detail.get("baseline_segments") or [])
    planned_segments = list(detail.get("planned_segments") or [])
    drop_gap_zones = find_drop_gap_zones(
        baseline_segments=baseline_segments,
        planned_segments=planned_segments,
        min_gap_s=min_drop_gap_s,
    )
    split_points = split_points_from_plan(
        baseline_segments=baseline_segments,
        planned_segments=planned_segments,
        drop_gap_zones=drop_gap_zones,
    )
    for point in split_points:
        mark_radius(
            split_frames,
            frame_index(point["time_s"], frame_hop_s=frame_hop_s, frame_count=frame_count),
            split_radius_frames,
        )
    for zone in drop_gap_zones:
        start = frame_index(float(zone["start"]), frame_hop_s=frame_hop_s, frame_count=frame_count)
        end = frame_index(float(zone["end"]), frame_hop_s=frame_hop_s, frame_count=frame_count)
        if end <= start:
            end = min(frame_count - 1, start + 1)
        for index in range(start, min(frame_count, end + 1)):
            drop_gap_frames[index] = 1
    return {
        "audio_id": audio_id,
        "duration_s": duration_s,
        "frame_hop_s": frame_hop_s,
        "frame_count": frame_count,
        "baseline_segments": baseline_segments,
        "planned_segments": planned_segments,
        "actual_segments": list(detail.get("actual_segments") or []),
        "actual_gap_segments": list(detail.get("actual_gap_segments") or []),
        "candidate_count": int(detail.get("candidate_count") or 0),
        "split_points": split_points,
        "drop_gap_zones": drop_gap_zones,
        "split_frames": split_frames,
        "drop_gap_frames": drop_gap_frames,
        "action_counts": {
            "split_point": len(split_points),
            "drop_gap": len(drop_gap_zones),
            "baseline_segment": len(baseline_segments),
            "planned_segment": len(planned_segments),
        },
    }


def export_boundary_imitation_targets(
    *,
    plan_details: Path,
    output_dir: Path,
    frame_hop_s: float,
    split_radius_frames: int,
    min_drop_gap_s: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = load_jsonl(plan_details)
    rows = [
        build_target_row(
            detail,
            frame_hop_s=frame_hop_s,
            split_radius_frames=split_radius_frames,
            min_drop_gap_s=min_drop_gap_s,
        )
        for detail in details
    ]
    target_path = output_dir / "imitation_targets.jsonl"
    write_jsonl(target_path, rows)
    action_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    for row in rows:
        action_counts.update(row["action_counts"])
        frame_counts["split_positive_frames"] += int(sum(row["split_frames"]))
        frame_counts["drop_gap_positive_frames"] += int(sum(row["drop_gap_frames"]))
        frame_counts["frames"] += int(row["frame_count"])
    summary = {
        "plan_details": str(plan_details),
        "target_path": str(target_path),
        "rows": len(rows),
        "frame_hop_s": frame_hop_s,
        "split_radius_frames": split_radius_frames,
        "min_drop_gap_s": min_drop_gap_s,
        "action_counts": dict(action_counts),
        "frame_counts": dict(frame_counts),
        "positive_frame_ratios": {
            "split": frame_counts["split_positive_frames"] / max(1, frame_counts["frames"]),
            "drop_gap": frame_counts["drop_gap_positive_frames"] / max(1, frame_counts["frames"]),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export v1.21 keep/split/drop-gap imitation targets from reward planner details."
    )
    parser.add_argument("--plan-details", required=True)
    parser.add_argument("--frame-hop-s", type=float, default=1.0 / 29.97)
    parser.add_argument("--split-radius-frames", type=int, default=1)
    parser.add_argument("--min-drop-gap-s", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "v1-21-imitation-targets"),
    )
    args = parser.parse_args(argv)
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.split_radius_frames < 0:
        parser.error("--split-radius-frames must be non-negative")
    if args.min_drop_gap_s < 0.0:
        parser.error("--min-drop-gap-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_boundary_imitation_targets(
        plan_details=Path(args.plan_details),
        output_dir=Path(args.output_dir),
        frame_hop_s=float(args.frame_hop_s),
        split_radius_frames=int(args.split_radius_frames),
        min_drop_gap_s=float(args.min_drop_gap_s),
    )
    print(f"targets={summary['target_path']}")
    print(
        "rows={rows} split_points={split_points} drop_gaps={drop_gaps} "
        "split_ratio={split_ratio:.6f} drop_ratio={drop_ratio:.6f}".format(
            rows=summary["rows"],
            split_points=summary["action_counts"].get("split_point", 0),
            drop_gaps=summary["action_counts"].get("drop_gap", 0),
            split_ratio=summary["positive_frame_ratios"]["split"],
            drop_ratio=summary["positive_frame_ratios"]["drop_gap"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
