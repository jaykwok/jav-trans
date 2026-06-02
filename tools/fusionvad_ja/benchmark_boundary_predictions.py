#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.base import SpeechSegment  # noqa: E402


class BoundaryRow:
    def __init__(
        self,
        *,
        audio_id: str,
        duration_s: float,
        frame_hop_s: float,
        actual_speech_segments: list[SpeechSegment],
        transition_regions: list[SpeechSegment],
        overlap_segments: list[SpeechSegment],
    ) -> None:
        self.audio_id = audio_id
        self.duration_s = duration_s
        self.frame_hop_s = frame_hop_s
        self.actual_speech_segments = actual_speech_segments
        self.transition_regions = transition_regions
        self.overlap_segments = overlap_segments


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_segments(items: Iterable[Mapping[str, Any]], *, duration_s: float) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    for item in items:
        try:
            start = max(0.0, min(float(item.get("start", item.get("start_s", 0.0))), duration_s))
            end = max(0.0, min(float(item.get("end", item.get("end_s", 0.0))), duration_s))
        except (TypeError, ValueError):
            continue
        if end > start:
            segments.append(SpeechSegment(start=start, end=end))
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def load_boundary_rows(path: Path) -> dict[str, BoundaryRow]:
    rows: dict[str, BoundaryRow] = {}
    for payload in load_jsonl(path):
        audio_id = str(payload.get("audio_id") or "")
        if not audio_id:
            continue
        duration_s = float(payload.get("duration_s") or 0.0)
        rows[audio_id] = BoundaryRow(
            audio_id=audio_id,
            duration_s=duration_s,
            frame_hop_s=float(payload.get("frame_hop_s") or 0.02),
            actual_speech_segments=parse_segments(
                payload.get("actual_speech_segments") or [],
                duration_s=duration_s,
            ),
            transition_regions=parse_segments(
                payload.get("transition_regions") or [],
                duration_s=duration_s,
            ),
            overlap_segments=parse_segments(
                payload.get("overlap_segments") or [],
                duration_s=duration_s,
            ),
        )
    return rows


def load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for payload in load_jsonl(path):
        audio_id = str(payload.get("audio_id") or "")
        if audio_id:
            rows[audio_id] = payload
    return rows


def pad_frames(values: list[int], *, pad_frames: int) -> list[int]:
    if pad_frames <= 0:
        return [1 if int(value) else 0 for value in values]
    out = [0] * len(values)
    for index, value in enumerate(values):
        if not int(value):
            continue
        start = max(0, index - pad_frames)
        end = min(len(out), index + pad_frames + 1)
        for offset in range(start, end):
            out[offset] = 1
    return out


def frames_to_segments(values: Iterable[int], *, frame_hop_s: float, duration_s: float) -> list[SpeechSegment]:
    frames = [1 if int(value) else 0 for value in values]
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, value in enumerate(frames + [0]):
        if value and start_index is None:
            start_index = index
        if not value and start_index is not None:
            start = max(0.0, min(start_index * frame_hop_s, duration_s))
            end = max(0.0, min(index * frame_hop_s, duration_s))
            if end > start:
                segments.append(SpeechSegment(start=start, end=end))
            start_index = None
    return segments


def merge_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    merge_gap_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    merged: list[SpeechSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = max(0.0, min(segment.start, duration_s))
        end = max(0.0, min(segment.end, duration_s))
        if end - start < min_segment_s:
            continue
        if not merged or start - merged[-1].end > merge_gap_s:
            merged.append(SpeechSegment(start=start, end=end))
        else:
            merged[-1].end = max(merged[-1].end, end)
    return merged


def split_segments_at_cuts(
    segments: Iterable[SpeechSegment],
    cut_segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    cuts = sorted(
        max(0.0, min((cut.start + cut.end) * 0.5, duration_s))
        for cut in cut_segments
        if cut.end > cut.start
    )
    if not cuts:
        return list(segments)
    output: list[SpeechSegment] = []
    minimum = max(0.0, float(min_segment_s))
    for segment in segments:
        start = max(0.0, min(segment.start, duration_s))
        end = max(0.0, min(segment.end, duration_s))
        if end - start < minimum:
            continue
        split_points = [cut for cut in cuts if start + minimum <= cut <= end - minimum]
        current = start
        for cut in split_points:
            if cut - current >= minimum:
                output.append(SpeechSegment(start=current, end=cut))
                current = cut
        if end - current >= minimum:
            output.append(SpeechSegment(start=current, end=end))
    return output


def split_segments_at_cuts_constrained(
    segments: Iterable[SpeechSegment],
    cut_segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    target_child_s: float,
    min_child_s: float,
    max_splits_per_segment: int,
) -> list[SpeechSegment]:
    cut_points = sorted(
        max(0.0, min((cut.start + cut.end) * 0.5, duration_s))
        for cut in cut_segments
        if cut.end > cut.start
    )
    if not cut_points:
        return list(segments)
    output: list[SpeechSegment] = []
    target = max(0.001, float(target_child_s))
    minimum = max(0.0, float(min_child_s))
    max_splits = max(0, int(max_splits_per_segment))
    for segment in segments:
        start = max(0.0, min(segment.start, duration_s))
        end = max(0.0, min(segment.end, duration_s))
        if end <= start:
            continue
        if end - start <= target:
            output.append(SpeechSegment(start=start, end=end))
            continue
        candidates = [
            point
            for point in cut_points
            if start + minimum <= point <= end - minimum
        ]
        if not candidates:
            output.append(SpeechSegment(start=start, end=end))
            continue

        selected: list[float] = []
        current = start
        remaining = list(candidates)
        while end - current > target and remaining and (max_splits <= 0 or len(selected) < max_splits):
            low = current + minimum
            high = end - minimum
            viable = [point for point in remaining if low <= point <= high]
            if not viable:
                break
            ideal = min(current + target, high)
            before = [point for point in viable if point <= ideal]
            cut = max(before) if before else min(viable, key=lambda point: abs(point - ideal))
            if cut <= current or end - cut < minimum:
                break
            selected.append(cut)
            current = cut
            remaining = [point for point in remaining if point > cut]

        if not selected:
            output.append(SpeechSegment(start=start, end=end))
            continue
        current = start
        for cut in selected:
            if cut - current > 0.0:
                output.append(SpeechSegment(start=current, end=cut))
            current = cut
        if end - current > 0.0:
            output.append(SpeechSegment(start=current, end=end))
    return output


def union_segments(segments: Iterable[SpeechSegment], *, duration_s: float) -> list[SpeechSegment]:
    return merge_segments(
        segments,
        duration_s=duration_s,
        merge_gap_s=0.0,
        min_segment_s=0.0,
    )


def overlap_s(left: SpeechSegment, right: SpeechSegment) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def total_duration(segments: Iterable[SpeechSegment]) -> float:
    return sum(max(0.0, segment.end - segment.start) for segment in segments)


def gap_segments(segments: Iterable[SpeechSegment], *, min_gap_s: float) -> list[SpeechSegment]:
    gaps: list[SpeechSegment] = []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        gap_start = previous.end
        gap_end = current.start
        if gap_end - gap_start >= min_gap_s:
            gaps.append(SpeechSegment(start=gap_start, end=gap_end))
    return gaps


def intersection_duration(left: Iterable[SpeechSegment], right: Iterable[SpeechSegment]) -> float:
    right_values = list(right)
    return sum(overlap_s(left_segment, right_segment) for left_segment in left for right_segment in right_values)


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for raw_start, raw_end in sorted(intervals):
        start = float(raw_start)
        end = float(raw_end)
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _start_weight_integral(start_ratio: float, end_ratio: float, *, exponent: float) -> float:
    start = max(0.0, min(1.0, start_ratio))
    end = max(0.0, min(1.0, end_ratio))
    if end <= start:
        return 0.0
    if exponent == 0.0:
        return end - start
    power = exponent + 1.0
    return ((1.0 - start) ** power - (1.0 - end) ** power) / power


def start_weighted_overlap_score(
    target: SpeechSegment,
    predictions: Iterable[SpeechSegment],
    *,
    exponent: float,
) -> tuple[float, float]:
    """Return start-biased overlap and denominator for one speech island.

    The target island is normalized to x=[0, 1]. Weight is highest near the
    start: w(x)=(1-x)^exponent. This makes late-only overlaps score much lower
    than linear duration recall, matching subtitle fallback risk where missing
    the beginning of a line is worse than trimming the end.
    """

    duration = segment_duration(target)
    if duration <= 0.0:
        return 0.0, 0.0
    intervals: list[tuple[float, float]] = []
    for prediction in predictions:
        start = max(target.start, prediction.start)
        end = min(target.end, prediction.end)
        if end > start:
            intervals.append(((start - target.start) / duration, (end - target.start) / duration))
    weighted_overlap = sum(
        _start_weight_integral(start, end, exponent=exponent) * duration
        for start, end in _merge_intervals(intervals)
    )
    weighted_total = _start_weight_integral(0.0, 1.0, exponent=exponent) * duration
    return weighted_overlap, weighted_total


def best_segment_match(
    target: SpeechSegment,
    predictions: list[SpeechSegment],
    *,
    min_overlap_ratio: float,
) -> tuple[SpeechSegment | None, float]:
    best: SpeechSegment | None = None
    best_overlap = 0.0
    target_duration = max(1e-9, target.end - target.start)
    for prediction in predictions:
        overlap = overlap_s(target, prediction)
        if overlap > best_overlap:
            best = prediction
            best_overlap = overlap
    if best is None or best_overlap / target_duration < min_overlap_ratio:
        return None, best_overlap
    return best, best_overlap


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if upper == lower:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def summarize_errors(values: list[float]) -> dict[str, float]:
    absolute = [abs(value) for value in values]
    return {
        "count": float(len(values)),
        "mean_abs_s": mean(absolute) if absolute else 0.0,
        "p50_abs_s": percentile(absolute, 0.50),
        "p90_abs_s": percentile(absolute, 0.90),
        "p95_abs_s": percentile(absolute, 0.95),
        "signed_mean_s": mean(values) if values else 0.0,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min_s": 0.0,
            "mean_s": 0.0,
            "p50_s": 0.0,
            "p90_s": 0.0,
            "p95_s": 0.0,
            "max_s": 0.0,
        }
    return {
        "count": float(len(values)),
        "min_s": min(values),
        "mean_s": mean(values),
        "p50_s": percentile(values, 0.50),
        "p90_s": percentile(values, 0.90),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values),
    }


def segment_duration(segment: SpeechSegment) -> float:
    return max(0.0, segment.end - segment.start)


def max_gap_overlap_s(segment: SpeechSegment, gaps: Iterable[SpeechSegment]) -> float:
    return max((overlap_s(segment, gap) for gap in gaps), default=0.0)


def benchmark_boundary_predictions(
    *,
    boundary_manifest: Path,
    predictions: Path,
    output_dir: Path,
    pad_s: float,
    merge_gap_s: float,
    min_segment_s: float,
    min_overlap_ratio: float,
    cut_min_gap_s: float = 0.5,
    fallback_target_duration_s: float = 8.0,
    fallback_gap_overlap_s: float = 0.5,
    start_weighted_recall_exponent: float = 2.0,
    cut_split_mode: str = "off",
    cut_split_target_s: float | None = None,
    cut_split_min_child_s: float = 1.5,
    cut_split_max_splits_per_segment: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = load_boundary_rows(boundary_manifest)
    prediction_rows = load_prediction_rows(predictions)
    pad_frames_by_hop: dict[float, int] = {}
    details = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    matched_segments = 0
    missed_segments = 0
    predicted_segments = 0
    unsupported_predictions = 0
    speech_duration = 0.0
    predicted_duration = 0.0
    overlap_duration = 0.0
    transition_predicted_duration = 0.0
    transition_duration = 0.0
    overlap_speech_duration = 0.0
    overlap_speech_predicted_duration = 0.0
    cut_gap_count = 0
    cut_gap_covered_count = 0
    cut_predicted_segments = 0
    cut_supported_segments = 0
    predicted_segment_durations: list[float] = []
    long_predicted_segments = 0
    predicted_gap_crossing_segments = 0
    predicted_gap_overlap_values: list[float] = []
    start_weighted_overlap = 0.0
    start_weighted_total = 0.0
    skipped = []

    for audio_id, boundary in boundary_rows.items():
        prediction = prediction_rows.get(audio_id)
        if prediction is None:
            skipped.append({"audio_id": audio_id, "reason": "missing_prediction"})
            continue
        raw_frames = prediction.get("speech_frames") or prediction.get("predictions")
        if not isinstance(raw_frames, list):
            skipped.append({"audio_id": audio_id, "reason": "missing_speech_frames"})
            continue
        if boundary.frame_hop_s not in pad_frames_by_hop:
            pad_frames_by_hop[boundary.frame_hop_s] = max(0, int(round(pad_s / boundary.frame_hop_s)))
        frames = pad_frames([int(value) for value in raw_frames], pad_frames=pad_frames_by_hop[boundary.frame_hop_s])
        pred_segments = merge_segments(
            frames_to_segments(frames, frame_hop_s=boundary.frame_hop_s, duration_s=boundary.duration_s),
            duration_s=boundary.duration_s,
            merge_gap_s=merge_gap_s,
            min_segment_s=min_segment_s,
        )
        actual_segments = boundary.actual_speech_segments
        actual_union_segments = union_segments(actual_segments, duration_s=boundary.duration_s)
        actual_gap_segments = gap_segments(actual_segments, min_gap_s=cut_min_gap_s)
        audio_speech_duration = total_duration(actual_union_segments)
        audio_predicted_duration = total_duration(pred_segments)
        audio_overlap_duration = intersection_duration(pred_segments, actual_union_segments)
        audio_transition_duration = total_duration(boundary.transition_regions)
        audio_transition_predicted = intersection_duration(pred_segments, boundary.transition_regions)
        audio_overlap_speech_duration = total_duration(boundary.overlap_segments)
        audio_overlap_speech_predicted = intersection_duration(pred_segments, boundary.overlap_segments)
        cut_frames_raw = prediction.get("cut_frames")
        if isinstance(cut_frames_raw, list):
            cut_segments = merge_segments(
                frames_to_segments(
                    [int(value) for value in cut_frames_raw],
                    frame_hop_s=boundary.frame_hop_s,
                    duration_s=boundary.duration_s,
                ),
                duration_s=boundary.duration_s,
                merge_gap_s=boundary.frame_hop_s,
                min_segment_s=0.0,
            )
        else:
            cut_segments = []
        raw_pred_segments = list(pred_segments)
        if cut_split_mode == "split":
            pred_segments = split_segments_at_cuts(
                pred_segments,
                cut_segments,
                duration_s=boundary.duration_s,
                min_segment_s=min_segment_s,
            )
        elif cut_split_mode == "constrained":
            pred_segments = split_segments_at_cuts_constrained(
                pred_segments,
                cut_segments,
                duration_s=boundary.duration_s,
                target_child_s=cut_split_target_s or fallback_target_duration_s,
                min_child_s=cut_split_min_child_s,
                max_splits_per_segment=cut_split_max_splits_per_segment,
            )
        cut_gap_count += len(actual_gap_segments)
        audio_cut_gap_covered = sum(
            1
            for gap in actual_gap_segments
            if any(overlap_s(gap, cut_segment) > 0.0 for cut_segment in cut_segments)
        )
        cut_gap_covered_count += audio_cut_gap_covered
        cut_predicted_segments += len(cut_segments)
        audio_cut_supported = sum(
            1
            for cut_segment in cut_segments
            if any(overlap_s(cut_segment, gap) > 0.0 for gap in actual_gap_segments)
        )
        cut_supported_segments += audio_cut_supported

        speech_duration += audio_speech_duration
        predicted_duration += audio_predicted_duration
        overlap_duration += audio_overlap_duration
        transition_duration += audio_transition_duration
        transition_predicted_duration += audio_transition_predicted
        overlap_speech_duration += audio_overlap_speech_duration
        overlap_speech_predicted_duration += audio_overlap_speech_predicted
        predicted_segments += len(pred_segments)
        audio_predicted_details = []
        for prediction_segment in pred_segments:
            duration = segment_duration(prediction_segment)
            max_gap_overlap = max_gap_overlap_s(prediction_segment, actual_gap_segments)
            crosses_truth_gap = max_gap_overlap >= fallback_gap_overlap_s
            predicted_segment_durations.append(duration)
            predicted_gap_overlap_values.append(max_gap_overlap)
            if duration > fallback_target_duration_s:
                long_predicted_segments += 1
            if crosses_truth_gap:
                predicted_gap_crossing_segments += 1
            audio_predicted_details.append(
                {
                    "start": prediction_segment.start,
                    "end": prediction_segment.end,
                    "duration_s": duration,
                    "max_truth_gap_overlap_s": max_gap_overlap,
                    "crosses_truth_gap": crosses_truth_gap,
                    "fallback_safe": duration <= fallback_target_duration_s and not crosses_truth_gap,
                }
            )

        matched_for_audio = 0
        missed_for_audio = 0
        audio_start_weighted_overlap = 0.0
        audio_start_weighted_total = 0.0
        for target in actual_segments:
            weighted_overlap, weighted_total = start_weighted_overlap_score(
                target,
                pred_segments,
                exponent=start_weighted_recall_exponent,
            )
            start_weighted_overlap += weighted_overlap
            start_weighted_total += weighted_total
            audio_start_weighted_overlap += weighted_overlap
            audio_start_weighted_total += weighted_total
            best, overlap = best_segment_match(target, pred_segments, min_overlap_ratio=min_overlap_ratio)
            if best is None:
                missed_segments += 1
                missed_for_audio += 1
                continue
            matched_segments += 1
            matched_for_audio += 1
            start_errors.append(best.start - target.start)
            end_errors.append(best.end - target.end)
        for prediction_segment in pred_segments:
            pred_overlap = sum(overlap_s(prediction_segment, target) for target in actual_segments)
            if pred_overlap <= 0.0:
                unsupported_predictions += 1

        details.append(
            {
                "audio_id": audio_id,
                "duration_s": boundary.duration_s,
                "actual_segments": [{"start": item.start, "end": item.end} for item in actual_segments],
                "actual_union_segments": [{"start": item.start, "end": item.end} for item in actual_union_segments],
                "actual_gap_segments": [{"start": item.start, "end": item.end} for item in actual_gap_segments],
                "predicted_segments": audio_predicted_details,
                "cut_segments": [{"start": item.start, "end": item.end} for item in cut_segments],
                "actual_segment_count": len(actual_segments),
                "actual_union_segment_count": len(actual_union_segments),
                "actual_gap_count": len(actual_gap_segments),
                "predicted_segment_count": len(pred_segments),
                "raw_predicted_segment_count": len(raw_pred_segments),
                "long_predicted_segment_count": sum(
                    1 for item in audio_predicted_details if item["duration_s"] > fallback_target_duration_s
                ),
                "predicted_gap_crossing_segment_count": sum(
                    1 for item in audio_predicted_details if item["crosses_truth_gap"]
                ),
                "cut_segment_count": len(cut_segments),
                "cut_gap_covered_count": audio_cut_gap_covered,
                "cut_supported_segment_count": audio_cut_supported,
                "matched_segment_count": matched_for_audio,
                "missed_segment_count": missed_for_audio,
                "speech_duration_s": audio_speech_duration,
                "predicted_duration_s": audio_predicted_duration,
                "overlap_duration_s": audio_overlap_duration,
                "missed_speech_s": max(0.0, audio_speech_duration - audio_overlap_duration),
                "start_weighted_overlap_s": audio_start_weighted_overlap,
                "start_weighted_speech_s": audio_start_weighted_total,
                "start_weighted_speech_recall": (
                    audio_start_weighted_overlap / audio_start_weighted_total
                    if audio_start_weighted_total > 0.0
                    else 0.0
                ),
                "extra_audio_s": max(0.0, audio_predicted_duration - audio_overlap_duration),
                "transition_duration_s": audio_transition_duration,
                "transition_predicted_s": audio_transition_predicted,
                "overlap_speech_duration_s": audio_overlap_speech_duration,
                "overlap_speech_predicted_s": audio_overlap_speech_predicted,
            }
        )

    missed_speech_s = max(0.0, speech_duration - overlap_duration)
    extra_audio_s = max(0.0, predicted_duration - overlap_duration)
    start_weighted_missed = max(0.0, start_weighted_total - start_weighted_overlap)
    summary = {
        "boundary_manifest": str(boundary_manifest),
        "predictions": str(predictions),
        "evaluated": len(details),
        "skipped": len(skipped),
        "pad_s": pad_s,
        "merge_gap_s": merge_gap_s,
        "min_segment_s": min_segment_s,
        "min_overlap_ratio": min_overlap_ratio,
        "cut_min_gap_s": cut_min_gap_s,
        "fallback_target_duration_s": fallback_target_duration_s,
        "fallback_gap_overlap_s": fallback_gap_overlap_s,
        "start_weighted_recall_exponent": start_weighted_recall_exponent,
        "cut_split_mode": cut_split_mode,
        "cut_split_target_s": cut_split_target_s or fallback_target_duration_s,
        "cut_split_min_child_s": cut_split_min_child_s,
        "cut_split_max_splits_per_segment": cut_split_max_splits_per_segment,
        "actual_segment_count": sum(len(row.actual_speech_segments) for row in boundary_rows.values()),
        "predicted_segment_count": predicted_segments,
        "matched_segment_count": matched_segments,
        "missed_segment_count": missed_segments,
        "unsupported_prediction_count": unsupported_predictions,
        "long_predicted_segment_count": long_predicted_segments,
        "long_predicted_segment_ratio": long_predicted_segments / max(1, predicted_segments),
        "predicted_gap_crossing_segment_count": predicted_gap_crossing_segments,
        "predicted_gap_crossing_segment_ratio": predicted_gap_crossing_segments / max(1, predicted_segments),
        "speech_duration_s": speech_duration,
        "predicted_duration_s": predicted_duration,
        "overlap_duration_s": overlap_duration,
        "missed_speech_s": missed_speech_s,
        "start_weighted_overlap_s": start_weighted_overlap,
        "start_weighted_speech_s": start_weighted_total,
        "start_weighted_missed_s": start_weighted_missed,
        "start_weighted_speech_recall": (
            start_weighted_overlap / start_weighted_total if start_weighted_total > 0.0 else 0.0
        ),
        "extra_audio_s": extra_audio_s,
        "speech_duration_recall": overlap_duration / speech_duration if speech_duration > 0.0 else 0.0,
        "extra_audio_ratio": predicted_duration / speech_duration if speech_duration > 0.0 else 0.0,
        "unsupported_prediction_ratio": unsupported_predictions / max(1, predicted_segments),
        "transition_duration_s": transition_duration,
        "transition_predicted_s": transition_predicted_duration,
        "transition_predicted_ratio": (
            transition_predicted_duration / transition_duration if transition_duration > 0.0 else 0.0
        ),
        "overlap_speech_duration_s": overlap_speech_duration,
        "overlap_speech_predicted_s": overlap_speech_predicted_duration,
        "overlap_speech_recall": (
            overlap_speech_predicted_duration / overlap_speech_duration if overlap_speech_duration > 0.0 else 0.0
        ),
        "cut_gap_count": cut_gap_count,
        "cut_gap_covered_count": cut_gap_covered_count,
        "cut_gap_coverage_ratio": cut_gap_covered_count / cut_gap_count if cut_gap_count > 0 else 0.0,
        "cut_predicted_segment_count": cut_predicted_segments,
        "cut_supported_segment_count": cut_supported_segments,
        "cut_supported_ratio": (
            cut_supported_segments / cut_predicted_segments if cut_predicted_segments > 0 else 0.0
        ),
        "predicted_segment_duration": summarize_values(predicted_segment_durations),
        "predicted_gap_overlap": summarize_values(predicted_gap_overlap_values),
        "start_error": summarize_errors(start_errors),
        "end_error": summarize_errors(end_errors),
        "skipped_rows": skipped,
    }
    summary_path = output_dir / "boundary_benchmark_summary.json"
    details_path = output_dir / "boundary_benchmark_details.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(f"details={details_path}")
    print(
        f"evaluated={summary['evaluated']} recall={summary['speech_duration_recall']:.4f} "
        f"start_weighted_recall={summary['start_weighted_speech_recall']:.4f} "
        f"missed_speech_s={summary['missed_speech_s']:.2f} extra_audio_ratio={summary['extra_audio_ratio']:.4f} "
        f"start_p50={summary['start_error']['p50_abs_s']:.3f} end_p50={summary['end_error']['p50_abs_s']:.3f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    benchmark_boundary_predictions(
        boundary_manifest=Path(args.boundary_manifest),
        predictions=Path(args.predictions),
        output_dir=Path(args.output_dir),
        pad_s=args.pad_s,
        merge_gap_s=args.merge_gap_s,
        min_segment_s=args.min_segment_s,
        min_overlap_ratio=args.min_overlap_ratio,
        cut_min_gap_s=args.cut_min_gap_s,
        fallback_target_duration_s=args.fallback_target_duration_s,
        fallback_gap_overlap_s=args.fallback_gap_overlap_s,
        start_weighted_recall_exponent=args.start_weighted_recall_exponent,
        cut_split_mode=args.cut_split_mode,
        cut_split_target_s=args.cut_split_target_s,
        cut_split_min_child_s=args.cut_split_min_child_s,
        cut_split_max_splits_per_segment=args.cut_split_max_splits_per_segment,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VAD frame predictions against synthetic boundary truth.")
    parser.add_argument("--boundary-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--pad-s", type=float, default=0.2)
    parser.add_argument("--merge-gap-s", type=float, default=0.15)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--cut-min-gap-s", type=float, default=0.5)
    parser.add_argument(
        "--fallback-target-duration-s",
        type=float,
        default=8.0,
        help="Target maximum segment duration if forced alignment falls back to coarse VAD timing.",
    )
    parser.add_argument(
        "--fallback-gap-overlap-s",
        type=float,
        default=0.5,
        help="Minimum overlap with a truth non-speech gap to flag a predicted segment as gap-crossing.",
    )
    parser.add_argument(
        "--start-weighted-recall-exponent",
        type=float,
        default=2.0,
        help=(
            "Exponent gamma for start-biased recall weight w(x)=(1-x)^gamma over each truth speech island. "
            "0 equals linear duration recall; larger values penalize late-only overlap more strongly."
        ),
    )
    parser.add_argument(
        "--cut-split-mode",
        choices=("off", "split", "constrained"),
        default="off",
        help="Use cut_frames as split boundaries for predicted speech segments without deleting speech.",
    )
    parser.add_argument(
        "--cut-split-target-s",
        type=float,
        default=None,
        help="Target child duration for constrained cut splitting. Defaults to --fallback-target-duration-s.",
    )
    parser.add_argument("--cut-split-min-child-s", type=float, default=1.5)
    parser.add_argument("--cut-split-max-splits-per-segment", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "boundary-benchmark"),
    )
    args = parser.parse_args(argv)
    if args.pad_s < 0.0:
        parser.error("--pad-s must be non-negative")
    if args.merge_gap_s < 0.0:
        parser.error("--merge-gap-s must be non-negative")
    if args.min_segment_s < 0.0:
        parser.error("--min-segment-s must be non-negative")
    if not 0.0 <= args.min_overlap_ratio <= 1.0:
        parser.error("--min-overlap-ratio must be in [0, 1]")
    if args.cut_min_gap_s < 0.0:
        parser.error("--cut-min-gap-s must be non-negative")
    if args.fallback_target_duration_s <= 0.0:
        parser.error("--fallback-target-duration-s must be positive")
    if args.fallback_gap_overlap_s < 0.0:
        parser.error("--fallback-gap-overlap-s must be non-negative")
    if args.start_weighted_recall_exponent < 0.0:
        parser.error("--start-weighted-recall-exponent must be non-negative")
    if args.cut_split_target_s is not None and args.cut_split_target_s <= 0.0:
        parser.error("--cut-split-target-s must be positive")
    if args.cut_split_min_child_s < 0.0:
        parser.error("--cut-split-min-child-s must be non-negative")
    if args.cut_split_max_splits_per_segment < 0:
        parser.error("--cut-split-max-splits-per-segment must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
