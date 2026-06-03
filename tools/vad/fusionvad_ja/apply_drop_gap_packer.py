#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools" / "vad" / "fusionvad_ja"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.vad.fusionvad_ja.benchmark_boundary_predictions import (  # noqa: E402
    BoundaryRow,
    frames_to_segments,
    load_boundary_rows,
    load_prediction_rows,
    merge_segments,
    summarize_values,
)
from vad.base import SpeechSegment  # noqa: E402


@dataclass(frozen=True)
class DropGapRun:
    start_frame: int
    end_frame: int
    score: float

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def start_s(self, frame_hop_s: float) -> float:
        return self.start_frame * frame_hop_s

    def end_s(self, frame_hop_s: float) -> float:
        return self.end_frame * frame_hop_s


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def normalize_binary_frames(values: Sequence[Any], *, target_len: int) -> list[int]:
    frames = [1 if int(value) else 0 for value in values]
    if len(frames) == target_len:
        return frames
    if target_len <= 0:
        return []
    if not frames:
        return [0] * target_len
    source_positions = np.linspace(0.0, 1.0, num=len(frames), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    resized = np.interp(target_positions, source_positions, np.asarray(frames, dtype=np.float32))
    return [1 if value >= 0.5 else 0 for value in resized]


def normalize_float_frames(values: Sequence[Any], *, target_len: int) -> list[float]:
    floats = [float(value) for value in values]
    if len(floats) == target_len:
        return floats
    if target_len <= 0:
        return []
    if not floats:
        return [0.0] * target_len
    source_positions = np.linspace(0.0, 1.0, num=len(floats), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    resized = np.interp(target_positions, source_positions, np.asarray(floats, dtype=np.float32))
    return [float(value) for value in resized]


def probability_values(row: Mapping[str, Any], name: str) -> list[float]:
    probabilities = row.get("probabilities")
    if isinstance(probabilities, Mapping):
        values = probabilities.get(name)
        if isinstance(values, list):
            return [float(value) for value in values]
    values = row.get(f"{name}_frames")
    if isinstance(values, list):
        return [float(value) for value in values]
    return []


def high_probability_runs(
    values: Sequence[float],
    *,
    threshold: float,
    min_frames: int,
) -> list[DropGapRun]:
    runs: list[DropGapRun] = []
    run_start: int | None = None
    for index, value in enumerate([*values, -1.0]):
        if float(value) >= threshold:
            if run_start is None:
                run_start = index
            continue
        if run_start is not None and index - run_start >= min_frames:
            run = [float(item) for item in values[run_start:index]]
            runs.append(
                DropGapRun(
                    start_frame=run_start,
                    end_frame=index,
                    score=max(run),
                )
            )
        run_start = None
    return runs


def segment_frame_bounds(segment: SpeechSegment, *, frame_hop_s: float, frame_count: int) -> tuple[int, int]:
    start = max(0, min(frame_count, int(round(segment.start / frame_hop_s))))
    end = max(0, min(frame_count, int(round(segment.end / frame_hop_s))))
    return start, end


def frames_from_segments(
    segments: Iterable[SpeechSegment],
    *,
    frame_hop_s: float,
    frame_count: int,
) -> list[int]:
    frames = [0] * frame_count
    for segment in segments:
        start, end = segment_frame_bounds(segment, frame_hop_s=frame_hop_s, frame_count=frame_count)
        for index in range(start, end):
            frames[index] = 1
    return frames


def apply_runs_to_segment(
    segment: SpeechSegment,
    *,
    runs: Sequence[DropGapRun],
    frame_hop_s: float,
    min_child_s: float,
    drop_gap_frames: list[int],
) -> tuple[list[SpeechSegment], list[dict[str, Any]], dict[str, int]]:
    parts = [segment]
    applied: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "outside_parent": 0,
        "near_edge": 0,
        "short_child": 0,
    }
    for run in sorted(runs, key=lambda item: (item.start_frame, item.end_frame)):
        run_start_s = run.start_s(frame_hop_s)
        run_end_s = run.end_s(frame_hop_s)
        next_parts: list[SpeechSegment] = []
        changed = False
        for part in parts:
            if changed:
                next_parts.append(part)
                continue
            if run_end_s <= part.start or run_start_s >= part.end:
                next_parts.append(part)
                continue
            zone_start = max(part.start, run_start_s)
            zone_end = min(part.end, run_end_s)
            if zone_start <= part.start or zone_end >= part.end:
                skipped["near_edge"] += 1
                next_parts.append(part)
                continue
            left = SpeechSegment(start=part.start, end=zone_start)
            right = SpeechSegment(start=zone_end, end=part.end)
            if left.end - left.start < min_child_s or right.end - right.start < min_child_s:
                skipped["short_child"] += 1
                next_parts.append(part)
                continue
            for index in range(max(0, run.start_frame), min(len(drop_gap_frames), run.end_frame)):
                drop_gap_frames[index] = 1
            next_parts.extend([left, right])
            applied.append(
                {
                    "start": zone_start,
                    "end": zone_end,
                    "duration_s": zone_end - zone_start,
                    "score": run.score,
                    "start_frame": run.start_frame,
                    "end_frame": run.end_frame,
                }
            )
            changed = True
        if not changed:
            skipped["outside_parent"] += 1
        parts = sorted(next_parts, key=lambda item: (item.start, item.end))
    return parts, applied, skipped


def apply_drop_gap_packer(
    *,
    boundary_manifest: Path,
    baseline_predictions: Path,
    drop_gap_predictions: Path,
    output_dir: Path,
    drop_gap_threshold: float,
    min_drop_gap_frames: int,
    min_drop_gap_s: float,
    min_child_s: float,
    min_parent_s: float,
    baseline_merge_gap_s: float,
    min_segment_s: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = load_boundary_rows(boundary_manifest)
    baseline_rows = load_prediction_rows(baseline_predictions)
    drop_rows = load_prediction_rows(drop_gap_predictions)
    output_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    skipped_counts: dict[str, int] = {}
    applied_zone_durations: list[float] = []
    before_durations: list[float] = []
    after_durations: list[float] = []
    rows_touched = 0
    parent_segments_touched = 0
    parent_segments_considered = 0
    segments_before = 0
    segments_after = 0
    frames_removed = 0
    applied_zone_count = 0
    run_skip_counts = {
        "outside_parent": 0,
        "near_edge": 0,
        "short_child": 0,
    }

    def mark_skip(audio_id: str, reason: str) -> None:
        skipped.append({"audio_id": audio_id, "reason": reason})
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    for audio_id, boundary in boundary_rows.items():
        baseline = baseline_rows.get(audio_id)
        drop = drop_rows.get(audio_id)
        if baseline is None:
            mark_skip(audio_id, "missing_baseline_prediction")
            continue
        if drop is None:
            mark_skip(audio_id, "missing_drop_gap_prediction")
            continue
        raw_frames = baseline.get("speech_frames") or baseline.get("predictions")
        if not isinstance(raw_frames, list):
            mark_skip(audio_id, "missing_speech_frames")
            continue
        frame_count = len(raw_frames)
        if frame_count <= 0:
            frame_count = max(1, int(round(boundary.duration_s / boundary.frame_hop_s)))
        speech_frames = normalize_binary_frames(raw_frames, target_len=frame_count)
        drop_probs_raw = probability_values(drop, "drop_gap")
        if not drop_probs_raw:
            mark_skip(audio_id, "missing_drop_gap_probabilities")
            continue
        drop_probs = normalize_float_frames(drop_probs_raw, target_len=frame_count)
        min_run_frames = max(min_drop_gap_frames, int(round(min_drop_gap_s / boundary.frame_hop_s)))
        runs = high_probability_runs(drop_probs, threshold=drop_gap_threshold, min_frames=min_run_frames)
        baseline_segments = merge_segments(
            frames_to_segments(speech_frames, frame_hop_s=boundary.frame_hop_s, duration_s=boundary.duration_s),
            duration_s=boundary.duration_s,
            merge_gap_s=baseline_merge_gap_s,
            min_segment_s=min_segment_s,
        )
        segments_before += len(baseline_segments)
        before_durations.extend(max(0.0, segment.end - segment.start) for segment in baseline_segments)
        drop_gap_frames = [0] * frame_count
        packed_segments: list[SpeechSegment] = []
        row_applied: list[dict[str, Any]] = []
        row_parent_touched = 0
        for segment_index, segment in enumerate(baseline_segments):
            duration_s = max(0.0, segment.end - segment.start)
            if duration_s < min_parent_s:
                packed_segments.append(segment)
                continue
            parent_segments_considered += 1
            segment_runs = [
                run
                for run in runs
                if segment.start < run.start_s(boundary.frame_hop_s)
                and run.end_s(boundary.frame_hop_s) < segment.end
            ]
            if not segment_runs:
                packed_segments.append(segment)
                continue
            child_segments, applied, run_skips = apply_runs_to_segment(
                segment,
                runs=segment_runs,
                frame_hop_s=boundary.frame_hop_s,
                min_child_s=min_child_s,
                drop_gap_frames=drop_gap_frames,
            )
            for reason, count in run_skips.items():
                run_skip_counts[reason] += int(count)
            packed_segments.extend(child_segments)
            if applied:
                row_parent_touched += 1
                for zone in applied:
                    zone["parent_segment_index"] = segment_index
                row_applied.extend(applied)
        output_speech_frames = list(speech_frames)
        for index, value in enumerate(drop_gap_frames):
            if value:
                output_speech_frames[index] = 0
        packed_segments = merge_segments(
            frames_to_segments(output_speech_frames, frame_hop_s=boundary.frame_hop_s, duration_s=boundary.duration_s),
            duration_s=boundary.duration_s,
            merge_gap_s=0.0,
            min_segment_s=min_segment_s,
        )
        row_frames_removed = sum(1 for before, after in zip(speech_frames, output_speech_frames) if before and not after)
        frames_removed += row_frames_removed
        segments_after += len(packed_segments)
        after_durations.extend(max(0.0, segment.end - segment.start) for segment in packed_segments)
        applied_zone_count += len(row_applied)
        parent_segments_touched += row_parent_touched
        if row_applied:
            rows_touched += 1
            applied_zone_durations.extend(float(zone["duration_s"]) for zone in row_applied)
        payload = {
            "audio_id": audio_id,
            "source": "v1-21-drop-gap-offline-packer",
            "duration_s": boundary.duration_s,
            "frame_hop_s": boundary.frame_hop_s,
            "frame_count": frame_count,
            "speech_frames": output_speech_frames,
            "cut_frames": drop_gap_frames,
            "drop_gap_frames": drop_gap_frames,
            "drop_gap_packer": {
                "drop_gap_threshold": drop_gap_threshold,
                "min_drop_gap_frames": min_run_frames,
                "min_drop_gap_s": min_run_frames * boundary.frame_hop_s,
                "min_child_s": min_child_s,
                "min_parent_s": min_parent_s,
                "segments_before": len(baseline_segments),
                "segments_after": len(packed_segments),
                "frames_removed": row_frames_removed,
                "applied_zone_count": len(row_applied),
                "parent_segments_touched": row_parent_touched,
                "baseline_audio_id": str(baseline.get("audio_id") or audio_id),
            },
            "applied_drop_gap_zones": row_applied,
        }
        output_rows.append(payload)

    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, output_rows)
    summary = {
        "boundary_manifest": str(boundary_manifest),
        "baseline_predictions": str(baseline_predictions),
        "drop_gap_predictions": str(drop_gap_predictions),
        "predictions": str(prediction_path),
        "rows": len(output_rows),
        "skipped": len(skipped),
        "skipped_counts": skipped_counts,
        "drop_gap_threshold": drop_gap_threshold,
        "min_drop_gap_frames_requested": min_drop_gap_frames,
        "min_drop_gap_s_requested": min_drop_gap_s,
        "min_child_s": min_child_s,
        "min_parent_s": min_parent_s,
        "baseline_merge_gap_s": baseline_merge_gap_s,
        "min_segment_s": min_segment_s,
        "rows_touched": rows_touched,
        "parent_segments_considered": parent_segments_considered,
        "parent_segments_touched": parent_segments_touched,
        "applied_zone_count": applied_zone_count,
        "segments_before": segments_before,
        "segments_after": segments_after,
        "segment_count_delta": segments_after - segments_before,
        "segment_count_delta_ratio": (segments_after - segments_before) / max(1, segments_before),
        "frames_removed": frames_removed,
        "seconds_removed": frames_removed * mean([row.frame_hop_s for row in boundary_rows.values()])
        if boundary_rows and frames_removed
        else 0.0,
        "run_skip_counts": run_skip_counts,
        "segment_duration_before": summarize_values(before_durations),
        "segment_duration_after": summarize_values(after_durations),
        "applied_zone_duration": summarize_values(applied_zone_durations),
        "skipped_rows": skipped,
    }
    write_json(output_dir / "drop_gap_packer_summary.json", summary)
    print(f"predictions={prediction_path}")
    print(f"summary={output_dir / 'drop_gap_packer_summary.json'}")
    print(
        f"rows={summary['rows']} touched={rows_touched} applied_zones={applied_zone_count} "
        f"segments={segments_before}->{segments_after} removed_s={summary['seconds_removed']:.2f}",
        flush=True,
    )
    return summary


def run(args: argparse.Namespace) -> None:
    apply_drop_gap_packer(
        boundary_manifest=Path(args.boundary_manifest),
        baseline_predictions=Path(args.baseline_predictions),
        drop_gap_predictions=Path(args.drop_gap_predictions),
        output_dir=Path(args.output_dir),
        drop_gap_threshold=float(args.drop_gap_threshold),
        min_drop_gap_frames=int(args.min_drop_gap_frames),
        min_drop_gap_s=float(args.min_drop_gap_s),
        min_child_s=float(args.min_child_s),
        min_parent_s=float(args.min_parent_s),
        baseline_merge_gap_s=float(args.baseline_merge_gap_s),
        min_segment_s=float(args.min_segment_s),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply v1.21 drop-gap scorer as an offline speech-island packer."
    )
    parser.add_argument("--boundary-manifest", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--drop-gap-predictions", required=True)
    parser.add_argument("--drop-gap-threshold", type=float, default=0.90)
    parser.add_argument("--min-drop-gap-frames", type=int, default=3)
    parser.add_argument("--min-drop-gap-s", type=float, default=0.60)
    parser.add_argument("--min-child-s", type=float, default=0.50)
    parser.add_argument("--min-parent-s", type=float, default=8.0)
    parser.add_argument("--baseline-merge-gap-s", type=float, default=0.0)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "drop-gap-packer"),
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.drop_gap_threshold <= 1.0:
        parser.error("--drop-gap-threshold must be in [0, 1]")
    if args.min_drop_gap_frames <= 0:
        parser.error("--min-drop-gap-frames must be positive")
    if args.min_drop_gap_s < 0.0:
        parser.error("--min-drop-gap-s must be non-negative")
    if args.min_child_s < 0.0:
        parser.error("--min-child-s must be non-negative")
    if args.min_parent_s < 0.0:
        parser.error("--min-parent-s must be non-negative")
    if args.baseline_merge_gap_s < 0.0:
        parser.error("--baseline-merge-gap-s must be non-negative")
    if args.min_segment_s < 0.0:
        parser.error("--min-segment-s must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
