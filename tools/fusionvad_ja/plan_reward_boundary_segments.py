#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools" / "fusionvad_ja"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_boundary_predictions import (  # noqa: E402
    BoundaryRow,
    best_segment_match,
    frames_to_segments,
    gap_segments,
    intersection_duration,
    load_boundary_rows,
    load_prediction_rows,
    max_gap_overlap_s,
    merge_segments,
    overlap_s,
    summarize_errors,
    summarize_values,
    total_duration,
    union_segments,
)
from vad.base import SpeechSegment  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    time_s: float
    kind: str
    score: float
    start_s: float | None = None
    end_s: float | None = None

    @property
    def zone_start_s(self) -> float:
        return self.time_s if self.start_s is None else self.start_s

    @property
    def zone_end_s(self) -> float:
        return self.time_s if self.end_s is None else self.end_s

    @property
    def can_drop_zone(self) -> bool:
        return self.kind in {"cut", "cut_drop", "valley", "oracle_gap"}


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


def threshold_frames(values: Sequence[Any], *, threshold: float) -> list[int]:
    return [1 if float(value) >= threshold else 0 for value in values]


def pad_frames(values: Sequence[int], *, pad_frames: int) -> list[int]:
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


def frame_probabilities(row: Mapping[str, Any], name: str) -> list[float]:
    probabilities = row.get("probabilities")
    if isinstance(probabilities, Mapping):
        values = probabilities.get(name)
        if isinstance(values, list):
            return [float(value) for value in values]
    values = row.get(f"{name}_frames")
    if isinstance(values, list):
        return [float(value) for value in values]
    return []


def baseline_segments_from_prediction(
    *,
    boundary: BoundaryRow,
    prediction: Mapping[str, Any],
    speech_threshold: float,
    pad_s: float,
    merge_gap_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    speech_probabilities = frame_probabilities(prediction, "speech")
    if not speech_probabilities:
        raw = prediction.get("speech_frames") or prediction.get("predictions") or []
        speech_probabilities = [float(value) for value in raw]
    pad = max(0, int(round(pad_s / boundary.frame_hop_s)))
    frames = pad_frames(
        threshold_frames(speech_probabilities, threshold=speech_threshold),
        pad_frames=pad,
    )
    return merge_segments(
        frames_to_segments(frames, frame_hop_s=boundary.frame_hop_s, duration_s=boundary.duration_s),
        duration_s=boundary.duration_s,
        merge_gap_s=merge_gap_s,
        min_segment_s=min_segment_s,
    )


def run_candidates(
    values: Sequence[float],
    *,
    frame_hop_s: float,
    threshold: float,
    mode: str,
    min_frames: int,
    kind: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    run_start: int | None = None
    comparator = (lambda value: value >= threshold) if mode == "high" else (lambda value: value <= threshold)
    for index, raw in enumerate([*values, math.inf if mode == "low" else -math.inf]):
        value = float(raw)
        if comparator(value):
            if run_start is None:
                run_start = index
            continue
        if run_start is not None and index - run_start >= min_frames:
            run = [float(item) for item in values[run_start:index]]
            if mode == "high":
                score = max(run)
            else:
                score = max(0.0, min(1.0, 1.0 - (sum(run) / len(run))))
            candidates.append(
                Candidate(
                    time_s=((run_start + index) / 2.0) * frame_hop_s,
                    kind=kind,
                    score=score,
                    start_s=run_start * frame_hop_s,
                    end_s=index * frame_hop_s,
                )
            )
        run_start = None
    return candidates


def oracle_gap_candidates(
    *,
    actual_segments: Sequence[SpeechSegment],
    min_gap_s: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for previous, current in zip(actual_segments, actual_segments[1:]):
        gap = current.start - previous.end
        if gap < min_gap_s:
            continue
        candidates.append(
            Candidate(
                time_s=(previous.end + current.start) / 2.0,
                kind="oracle_gap",
                score=min(1.0, gap / max(min_gap_s, 1e-9)),
                start_s=previous.end,
                end_s=current.start,
            )
        )
    return candidates


def probability_candidates(
    *,
    prediction: Mapping[str, Any],
    frame_hop_s: float,
    cut_threshold: float,
    endpoint_threshold: float,
    valley_threshold: float,
    min_candidate_frames: int,
) -> list[Candidate]:
    cut_drop = frame_probabilities(prediction, "cut_drop")
    cut_point = frame_probabilities(prediction, "cut_point")
    cut = frame_probabilities(prediction, "cut")
    speech = frame_probabilities(prediction, "speech")
    start = frame_probabilities(prediction, "start")
    end = frame_probabilities(prediction, "end")
    endpoint = [max(left, right) for left, right in zip(start, end)]
    candidates: list[Candidate] = []
    if cut_drop:
        candidates.extend(
            run_candidates(
                cut_drop,
                frame_hop_s=frame_hop_s,
                threshold=cut_threshold,
                mode="high",
                min_frames=min_candidate_frames,
                kind="cut_drop",
            )
        )
    elif cut:
        candidates.extend(
            run_candidates(
                cut,
                frame_hop_s=frame_hop_s,
                threshold=cut_threshold,
                mode="high",
                min_frames=min_candidate_frames,
                kind="cut",
            )
        )
    if cut_point:
        candidates.extend(
            run_candidates(
                cut_point,
                frame_hop_s=frame_hop_s,
                threshold=cut_threshold,
                mode="high",
                min_frames=min_candidate_frames,
                kind="cut_point",
            )
        )
    if endpoint:
        candidates.extend(
            run_candidates(
                endpoint,
                frame_hop_s=frame_hop_s,
                threshold=endpoint_threshold,
                mode="high",
                min_frames=min_candidate_frames,
                kind="endpoint",
            )
        )
    if speech:
        candidates.extend(
            run_candidates(
                speech,
                frame_hop_s=frame_hop_s,
                threshold=valley_threshold,
                mode="low",
                min_frames=min_candidate_frames,
                kind="valley",
            )
        )
    return candidates


def merge_candidates(
    candidates: Iterable[Candidate],
    *,
    frame_hop_s: float,
    lower: float,
    upper: float,
) -> list[Candidate]:
    by_frame: dict[int, Candidate] = {}
    for candidate in candidates:
        if not lower <= candidate.time_s <= upper:
            continue
        frame = int(round(candidate.time_s / frame_hop_s))
        previous = by_frame.get(frame)
        if previous is None or candidate.score > previous.score:
            by_frame[frame] = Candidate(
                time_s=frame * frame_hop_s,
                kind=candidate.kind,
                score=float(candidate.score),
                start_s=candidate.start_s,
                end_s=candidate.end_s,
            )
    return [by_frame[key] for key in sorted(by_frame)]


def segment_cost(
    segment: SpeechSegment,
    *,
    truth_gaps: Sequence[SpeechSegment],
    target_duration_s: float,
    gap_overlap_s: float,
    duration_weight: float,
    gap_crossing_weight: float,
    gap_overlap_weight: float,
    use_truth_cost: bool,
) -> float:
    duration = max(0.0, segment.end - segment.start)
    duration_over = max(0.0, duration - target_duration_s)
    cost = duration_weight * duration_over * duration_over
    if use_truth_cost:
        gap_overlap = max_gap_overlap_s(segment, truth_gaps)
        if gap_overlap >= gap_overlap_s:
            cost += gap_crossing_weight + gap_overlap_weight * gap_overlap
    return cost


def plan_segment(
    base: SpeechSegment,
    *,
    candidates: Sequence[Candidate],
    truth_gaps: Sequence[SpeechSegment],
    target_duration_s: float,
    min_child_s: float,
    gap_overlap_s: float,
    max_children: int,
    duration_weight: float,
    gap_crossing_weight: float,
    gap_overlap_weight: float,
    split_penalty: float,
    candidate_reward_weight: float,
    use_truth_cost: bool,
) -> list[SpeechSegment]:
    if max_children <= 1 or base.end - base.start <= min_child_s * 2.0:
        return [base]
    zone_plan = plan_segment_by_cut_zones(
        base,
        candidates=candidates,
        truth_gaps=truth_gaps,
        target_duration_s=target_duration_s,
        min_child_s=min_child_s,
        gap_overlap_s=gap_overlap_s,
        max_children=max_children,
        duration_weight=duration_weight,
        gap_crossing_weight=gap_crossing_weight,
        gap_overlap_weight=gap_overlap_weight,
        split_penalty=split_penalty,
        candidate_reward_weight=candidate_reward_weight,
        use_truth_cost=use_truth_cost,
    )
    if zone_plan is not None:
        return zone_plan
    values = [base.start, *[candidate.time_s for candidate in candidates], base.end]
    end_index = len(values) - 1
    if end_index <= 0:
        return [base]
    candidate_by_index = {index + 1: candidate for index, candidate in enumerate(candidates)}
    inf = float("inf")
    dp = [[inf] * len(values) for _ in range(max_children + 1)]
    parent: list[list[int | None]] = [[None] * len(values) for _ in range(max_children + 1)]
    dp[0][0] = 0.0
    for parts in range(1, max_children + 1):
        for current in range(1, len(values)):
            for previous in range(0, current):
                if dp[parts - 1][previous] == inf:
                    continue
                start = values[previous]
                end = values[current]
                if end - start < min_child_s:
                    continue
                cost = segment_cost(
                    SpeechSegment(start=start, end=end),
                    truth_gaps=truth_gaps,
                    target_duration_s=target_duration_s,
                    gap_overlap_s=gap_overlap_s,
                    duration_weight=duration_weight,
                    gap_crossing_weight=gap_crossing_weight,
                    gap_overlap_weight=gap_overlap_weight,
                    use_truth_cost=use_truth_cost,
                )
                if current != end_index:
                    candidate = candidate_by_index.get(current)
                    score = candidate.score if candidate is not None else 0.0
                    cost += split_penalty - candidate_reward_weight * score
                proposed = dp[parts - 1][previous] + cost
                if proposed < dp[parts][current]:
                    dp[parts][current] = proposed
                    parent[parts][current] = previous

    best_parts = min(range(1, max_children + 1), key=lambda parts: dp[parts][end_index])
    if dp[best_parts][end_index] == inf:
        return [base]
    indices = [end_index]
    current = end_index
    parts = best_parts
    while parts > 0:
        previous = parent[parts][current]
        if previous is None:
            return [base]
        indices.append(previous)
        current = previous
        parts -= 1
    indices = sorted(indices)
    planned = [
        SpeechSegment(start=values[left], end=values[right])
        for left, right in zip(indices, indices[1:])
        if values[right] > values[left]
    ]
    return planned or [base]


def plan_segment_by_cut_zones(
    base: SpeechSegment,
    *,
    candidates: Sequence[Candidate],
    truth_gaps: Sequence[SpeechSegment],
    target_duration_s: float,
    min_child_s: float,
    gap_overlap_s: float,
    max_children: int,
    duration_weight: float,
    gap_crossing_weight: float,
    gap_overlap_weight: float,
    split_penalty: float,
    candidate_reward_weight: float,
    use_truth_cost: bool,
) -> list[SpeechSegment] | None:
    zones = [
        candidate
        for candidate in candidates
        if candidate.can_drop_zone
        if candidate.zone_end_s - candidate.zone_start_s > 1e-6
        and base.start < candidate.zone_start_s < candidate.zone_end_s < base.end
    ]
    if not zones:
        return None

    def total_cost(segments: Sequence[SpeechSegment]) -> float:
        return sum(
            segment_cost(
                segment,
                truth_gaps=truth_gaps,
                target_duration_s=target_duration_s,
                gap_overlap_s=gap_overlap_s,
                duration_weight=duration_weight,
                gap_crossing_weight=gap_crossing_weight,
                gap_overlap_weight=gap_overlap_weight,
                use_truth_cost=use_truth_cost,
            )
            for segment in segments
        )

    segments = [base]
    selected = 0
    ordered_zones = sorted(
        zones,
        key=lambda candidate: (
            -candidate.score,
            candidate.zone_start_s,
            candidate.zone_end_s,
        ),
    )
    for candidate in ordered_zones:
        if selected >= max(0, max_children - 1):
            break
        next_segments: list[SpeechSegment] = []
        changed = False
        for segment in segments:
            zone_start = max(segment.start, candidate.zone_start_s)
            zone_end = min(segment.end, candidate.zone_end_s)
            if zone_end <= zone_start or zone_start <= segment.start or zone_end >= segment.end:
                next_segments.append(segment)
                continue
            left = SpeechSegment(start=segment.start, end=zone_start)
            right = SpeechSegment(start=zone_end, end=segment.end)
            if left.end - left.start < min_child_s or right.end - right.start < min_child_s:
                next_segments.append(segment)
                continue
            before = total_cost([segment])
            after = total_cost([left, right]) + split_penalty - candidate_reward_weight * candidate.score
            if after < before:
                next_segments.extend([left, right])
                selected += 1
                changed = True
            else:
                next_segments.append(segment)
        segments = sorted(next_segments, key=lambda item: (item.start, item.end))
        if len(segments) >= max_children:
            break
        if not changed:
            continue
    return segments if selected > 0 else None


def summarize_plan(
    *,
    boundary_rows: Mapping[str, BoundaryRow],
    segments_by_audio: Mapping[str, Sequence[SpeechSegment]],
    cut_min_gap_s: float,
    fallback_target_duration_s: float,
    fallback_gap_overlap_s: float,
    min_overlap_ratio: float,
) -> dict[str, Any]:
    start_errors: list[float] = []
    end_errors: list[float] = []
    durations: list[float] = []
    gap_overlaps: list[float] = []
    predicted_segments = 0
    matched_segments = 0
    missed_segments = 0
    unsupported_predictions = 0
    long_segments = 0
    gap_crossing_segments = 0
    speech_duration = 0.0
    predicted_duration = 0.0
    overlap_duration = 0.0

    for audio_id, boundary in boundary_rows.items():
        planned = list(segments_by_audio.get(audio_id) or [])
        actual = boundary.actual_speech_segments
        actual_union = union_segments(actual, duration_s=boundary.duration_s)
        truth_gaps = gap_segments(actual, min_gap_s=cut_min_gap_s)
        speech_duration += total_duration(actual_union)
        predicted_duration += total_duration(planned)
        overlap_duration += intersection_duration(planned, actual_union)
        predicted_segments += len(planned)
        for segment in planned:
            duration = max(0.0, segment.end - segment.start)
            gap_overlap = max_gap_overlap_s(segment, truth_gaps)
            durations.append(duration)
            gap_overlaps.append(gap_overlap)
            if duration > fallback_target_duration_s:
                long_segments += 1
            if gap_overlap >= fallback_gap_overlap_s:
                gap_crossing_segments += 1
            if sum(overlap_s(segment, target) for target in actual) <= 0.0:
                unsupported_predictions += 1
        for target in actual:
            best, _overlap = best_segment_match(target, planned, min_overlap_ratio=min_overlap_ratio)
            if best is None:
                missed_segments += 1
                continue
            matched_segments += 1
            start_errors.append(best.start - target.start)
            end_errors.append(best.end - target.end)

    missed_speech = max(0.0, speech_duration - overlap_duration)
    extra_audio = max(0.0, predicted_duration - overlap_duration)
    return {
        "actual_segment_count": sum(len(row.actual_speech_segments) for row in boundary_rows.values()),
        "predicted_segment_count": predicted_segments,
        "matched_segment_count": matched_segments,
        "missed_segment_count": missed_segments,
        "unsupported_prediction_count": unsupported_predictions,
        "long_predicted_segment_count": long_segments,
        "long_predicted_segment_ratio": long_segments / max(1, predicted_segments),
        "predicted_gap_crossing_segment_count": gap_crossing_segments,
        "predicted_gap_crossing_segment_ratio": gap_crossing_segments / max(1, predicted_segments),
        "speech_duration_s": speech_duration,
        "predicted_duration_s": predicted_duration,
        "overlap_duration_s": overlap_duration,
        "missed_speech_s": missed_speech,
        "extra_audio_s": extra_audio,
        "speech_duration_recall": overlap_duration / speech_duration if speech_duration > 0.0 else 0.0,
        "extra_audio_ratio": predicted_duration / speech_duration if speech_duration > 0.0 else 0.0,
        "unsupported_prediction_ratio": unsupported_predictions / max(1, predicted_segments),
        "predicted_segment_duration": summarize_values(durations),
        "predicted_gap_overlap": summarize_values(gap_overlaps),
        "start_error": summarize_errors(start_errors),
        "end_error": summarize_errors(end_errors),
    }


def segment_payload(segment: SpeechSegment) -> dict[str, float]:
    return {
        "start": round(segment.start, 6),
        "end": round(segment.end, 6),
        "duration_s": round(max(0.0, segment.end - segment.start), 6),
    }


def keep_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    min_segment_s: float,
) -> list[SpeechSegment]:
    kept: list[SpeechSegment] = []
    for segment in segments:
        start = max(0.0, min(segment.start, duration_s))
        end = max(0.0, min(segment.end, duration_s))
        if end - start >= min_segment_s:
            kept.append(SpeechSegment(start=start, end=end, score=segment.score))
    return sorted(kept, key=lambda item: (item.start, item.end))


def plan_reward_boundary_segments(
    *,
    boundary_manifest: Path,
    predictions: Path,
    output_dir: Path,
    candidate_source: str,
    use_truth_cost: bool,
    speech_threshold: float,
    cut_threshold: float,
    endpoint_threshold: float,
    valley_threshold: float,
    pad_s: float,
    merge_gap_s: float,
    min_segment_s: float,
    min_overlap_ratio: float,
    cut_min_gap_s: float,
    fallback_target_duration_s: float,
    fallback_gap_overlap_s: float,
    min_child_s: float,
    max_children: int,
    min_candidate_frames: int,
    duration_weight: float,
    gap_crossing_weight: float,
    gap_overlap_weight: float,
    split_penalty: float,
    candidate_reward_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = load_boundary_rows(boundary_manifest)
    prediction_rows = load_prediction_rows(predictions)
    baseline_by_audio: dict[str, list[SpeechSegment]] = {}
    planned_by_audio: dict[str, list[SpeechSegment]] = {}
    details: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for audio_id, boundary in boundary_rows.items():
        prediction = prediction_rows.get(audio_id)
        if prediction is None:
            skipped.append({"audio_id": audio_id, "reason": "missing_prediction"})
            continue
        baseline = baseline_segments_from_prediction(
            boundary=boundary,
            prediction=prediction,
            speech_threshold=speech_threshold,
            pad_s=pad_s,
            merge_gap_s=merge_gap_s,
            min_segment_s=min_segment_s,
        )
        baseline_by_audio[audio_id] = baseline
        truth_gaps = gap_segments(boundary.actual_speech_segments, min_gap_s=cut_min_gap_s)
        probability = probability_candidates(
            prediction=prediction,
            frame_hop_s=boundary.frame_hop_s,
            cut_threshold=cut_threshold,
            endpoint_threshold=endpoint_threshold,
            valley_threshold=valley_threshold,
            min_candidate_frames=min_candidate_frames,
        )
        oracle = oracle_gap_candidates(
            actual_segments=boundary.actual_speech_segments,
            min_gap_s=cut_min_gap_s,
        )
        if candidate_source == "probability":
            raw_candidates = probability
        elif candidate_source == "oracle":
            raw_candidates = oracle
        else:
            raw_candidates = [*probability, *oracle]

        planned: list[SpeechSegment] = []
        audio_candidate_count = 0
        for segment in baseline:
            candidates = merge_candidates(
                raw_candidates,
                frame_hop_s=boundary.frame_hop_s,
                lower=segment.start + min_child_s,
                upper=segment.end - min_child_s,
            )
            audio_candidate_count += len(candidates)
            planned.extend(
                plan_segment(
                    segment,
                    candidates=candidates,
                    truth_gaps=truth_gaps,
                    target_duration_s=fallback_target_duration_s,
                    min_child_s=min_child_s,
                    gap_overlap_s=fallback_gap_overlap_s,
                    max_children=max_children,
                    duration_weight=duration_weight,
                    gap_crossing_weight=gap_crossing_weight,
                    gap_overlap_weight=gap_overlap_weight,
                    split_penalty=split_penalty,
                    candidate_reward_weight=candidate_reward_weight,
                    use_truth_cost=use_truth_cost,
                )
            )
        planned = keep_segments(
            segments=planned,
            duration_s=boundary.duration_s,
            min_segment_s=min_segment_s,
        )
        planned_by_audio[audio_id] = planned
        details.append(
            {
                "audio_id": audio_id,
                "duration_s": boundary.duration_s,
                "candidate_count": audio_candidate_count,
                "baseline_segments": [segment_payload(segment) for segment in baseline],
                "planned_segments": [segment_payload(segment) for segment in planned],
                "actual_segments": [segment_payload(segment) for segment in boundary.actual_speech_segments],
                "actual_gap_segments": [segment_payload(segment) for segment in truth_gaps],
            }
        )

    baseline_summary = summarize_plan(
        boundary_rows=boundary_rows,
        segments_by_audio=baseline_by_audio,
        cut_min_gap_s=cut_min_gap_s,
        fallback_target_duration_s=fallback_target_duration_s,
        fallback_gap_overlap_s=fallback_gap_overlap_s,
        min_overlap_ratio=min_overlap_ratio,
    )
    planned_summary = summarize_plan(
        boundary_rows=boundary_rows,
        segments_by_audio=planned_by_audio,
        cut_min_gap_s=cut_min_gap_s,
        fallback_target_duration_s=fallback_target_duration_s,
        fallback_gap_overlap_s=fallback_gap_overlap_s,
        min_overlap_ratio=min_overlap_ratio,
    )
    summary = {
        "boundary_manifest": str(boundary_manifest),
        "predictions": str(predictions),
        "candidate_source": candidate_source,
        "use_truth_cost": use_truth_cost,
        "evaluated": len(details),
        "skipped": len(skipped),
        "skipped_rows": skipped,
        "parameters": {
            "speech_threshold": speech_threshold,
            "cut_threshold": cut_threshold,
            "endpoint_threshold": endpoint_threshold,
            "valley_threshold": valley_threshold,
            "pad_s": pad_s,
            "merge_gap_s": merge_gap_s,
            "min_segment_s": min_segment_s,
            "min_overlap_ratio": min_overlap_ratio,
            "cut_min_gap_s": cut_min_gap_s,
            "fallback_target_duration_s": fallback_target_duration_s,
            "fallback_gap_overlap_s": fallback_gap_overlap_s,
            "min_child_s": min_child_s,
            "max_children": max_children,
            "min_candidate_frames": min_candidate_frames,
            "duration_weight": duration_weight,
            "gap_crossing_weight": gap_crossing_weight,
            "gap_overlap_weight": gap_overlap_weight,
            "split_penalty": split_penalty,
            "candidate_reward_weight": candidate_reward_weight,
        },
        "baseline": baseline_summary,
        "planned": planned_summary,
        "delta": {
            "predicted_segment_count": (
                planned_summary["predicted_segment_count"] - baseline_summary["predicted_segment_count"]
            ),
            "long_predicted_segment_count": (
                planned_summary["long_predicted_segment_count"] - baseline_summary["long_predicted_segment_count"]
            ),
            "predicted_gap_crossing_segment_count": (
                planned_summary["predicted_gap_crossing_segment_count"]
                - baseline_summary["predicted_gap_crossing_segment_count"]
            ),
            "missed_speech_s": planned_summary["missed_speech_s"] - baseline_summary["missed_speech_s"],
            "extra_audio_ratio": planned_summary["extra_audio_ratio"] - baseline_summary["extra_audio_ratio"],
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "plan_details.jsonl", details)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R19 offline reward-shaped boundary planner. It does not change the "
            "runtime pipeline; it scores whether candidate cuts can reduce "
            "fallback-unsafe long speech-island chunks."
        )
    )
    parser.add_argument("--boundary-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--candidate-source",
        choices=("probability", "oracle", "hybrid"),
        default="probability",
    )
    parser.add_argument(
        "--use-truth-cost",
        action="store_true",
        help="Use synthetic truth gaps in the planner cost. This is for training-target/oracle analysis only.",
    )
    parser.add_argument("--speech-threshold", type=float, default=0.02)
    parser.add_argument("--cut-threshold", type=float, default=0.96)
    parser.add_argument("--endpoint-threshold", type=float, default=0.50)
    parser.add_argument("--valley-threshold", type=float, default=0.20)
    parser.add_argument("--pad-s", type=float, default=0.2)
    parser.add_argument("--merge-gap-s", type=float, default=0.15)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--cut-min-gap-s", type=float, default=0.5)
    parser.add_argument("--fallback-target-duration-s", type=float, default=8.0)
    parser.add_argument("--fallback-gap-overlap-s", type=float, default=0.5)
    parser.add_argument("--min-child-s", type=float, default=1.5)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--min-candidate-frames", type=int, default=2)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--gap-crossing-weight", type=float, default=64.0)
    parser.add_argument("--gap-overlap-weight", type=float, default=8.0)
    parser.add_argument("--split-penalty", type=float, default=0.75)
    parser.add_argument("--candidate-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "r19-reward-boundary-plan"),
    )
    args = parser.parse_args(argv)
    for name in (
        "speech_threshold",
        "cut_threshold",
        "endpoint_threshold",
        "valley_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in (
        "pad_s",
        "merge_gap_s",
        "min_segment_s",
        "min_overlap_ratio",
        "cut_min_gap_s",
        "fallback_gap_overlap_s",
        "min_child_s",
        "duration_weight",
        "gap_crossing_weight",
        "gap_overlap_weight",
        "split_penalty",
        "candidate_reward_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.fallback_target_duration_s <= 0.0:
        parser.error("--fallback-target-duration-s must be positive")
    if args.max_children <= 0:
        parser.error("--max-children must be positive")
    if args.min_candidate_frames <= 0:
        parser.error("--min-candidate-frames must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = plan_reward_boundary_segments(
        boundary_manifest=Path(args.boundary_manifest),
        predictions=Path(args.predictions),
        output_dir=Path(args.output_dir),
        candidate_source=args.candidate_source,
        use_truth_cost=bool(args.use_truth_cost),
        speech_threshold=args.speech_threshold,
        cut_threshold=args.cut_threshold,
        endpoint_threshold=args.endpoint_threshold,
        valley_threshold=args.valley_threshold,
        pad_s=args.pad_s,
        merge_gap_s=args.merge_gap_s,
        min_segment_s=args.min_segment_s,
        min_overlap_ratio=args.min_overlap_ratio,
        cut_min_gap_s=args.cut_min_gap_s,
        fallback_target_duration_s=args.fallback_target_duration_s,
        fallback_gap_overlap_s=args.fallback_gap_overlap_s,
        min_child_s=args.min_child_s,
        max_children=args.max_children,
        min_candidate_frames=args.min_candidate_frames,
        duration_weight=args.duration_weight,
        gap_crossing_weight=args.gap_crossing_weight,
        gap_overlap_weight=args.gap_overlap_weight,
        split_penalty=args.split_penalty,
        candidate_reward_weight=args.candidate_reward_weight,
    )
    print(f"summary={Path(args.output_dir) / 'summary.json'}")
    print(
        "segments={base}->{planned} long={base_long}->{planned_long} gap_cross={base_gap}->{planned_gap} "
        "recall={recall:.6f} extra={extra:.4f}".format(
            base=summary["baseline"]["predicted_segment_count"],
            planned=summary["planned"]["predicted_segment_count"],
            base_long=summary["baseline"]["long_predicted_segment_count"],
            planned_long=summary["planned"]["long_predicted_segment_count"],
            base_gap=summary["baseline"]["predicted_gap_crossing_segment_count"],
            planned_gap=summary["planned"]["predicted_gap_crossing_segment_count"],
            recall=summary["planned"]["speech_duration_recall"],
            extra=summary["planned"]["extra_audio_ratio"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
