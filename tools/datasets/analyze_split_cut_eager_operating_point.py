#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICIES = ("current_threshold", "gate_cut", "probability_argmax")
SAME_SENTENCE_FLAGS = {"short_pause", "breath", "same_sentence"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        round(float(candidate["time_s"]), 6),
        round(float(candidate.get("p_cut") or 0.0), 8),
        round(float(candidate.get("p_continue") or 0.0), 8),
        round(float(candidate.get("p_unsure") or 0.0), 8),
    )


def segment_candidates(
    segment: dict[str, Any],
    *,
    include_primary: bool = True,
) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    fields = (
        ("primary_cut_candidates", "weak_cut_candidates")
        if include_primary
        else ("weak_cut_candidates",)
    )
    for field in fields:
        for raw in segment.get(field) or []:
            if not isinstance(raw, dict) or "time_s" not in raw:
                continue
            candidate = dict(raw)
            candidate["runtime_accepted"] = field == "primary_cut_candidates"
            unique[_candidate_key(candidate)] = candidate
    return sorted(unique.values(), key=lambda item: float(item["time_s"]))


def selected(candidate: dict[str, Any], *, policy: str, threshold: float) -> bool:
    p_cut = float(candidate.get("p_cut") or 0.0)
    p_continue = float(candidate.get("p_continue") or 0.0)
    p_unsure = float(candidate.get("p_unsure") or 0.0)
    if policy == "current_threshold":
        return bool(candidate.get("runtime_accepted"))
    if policy == "gate_cut":
        return p_cut >= 0.5
    if policy == "probability_argmax":
        return p_cut > p_continue and p_cut > p_unsure
    raise ValueError(f"unknown policy: {policy}")


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))])


def duration_metrics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_s": sum(values) / len(values) if values else 0.0,
        "p50_s": _quantile(values, 0.50),
        "p90_s": _quantile(values, 0.90),
        "p95_s": _quantile(values, 0.95),
        "p99_s": _quantile(values, 0.99),
        "max_s": max(values, default=0.0),
        "over_5s": sum(value > 5.0 for value in values),
        "over_8s": sum(value > 8.0 for value in values),
    }


def replay_durations(
    segments: list[dict[str, Any]],
    *,
    policy: str,
    short_core_max_s: float,
    short_threshold: float,
    normal_threshold: float,
) -> tuple[list[float], int]:
    durations: list[float] = []
    accepted = 0
    for segment in segments:
        start = float(segment.get("chunk_acoustic_start", segment["start"]))
        end = float(segment.get("chunk_acoustic_end", segment["end"]))
        threshold = short_threshold if end - start <= short_core_max_s else normal_threshold
        cuts = sorted(
            {
                float(candidate["time_s"])
                for candidate in segment_candidates(segment, include_primary=False)
                if start < float(candidate["time_s"]) < end
                and selected(candidate, policy=policy, threshold=threshold)
            }
        )
        boundaries = [start, *cuts, end]
        durations.extend(
            right - left
            for left, right in zip(boundaries, boundaries[1:])
            if right > left
        )
        accepted += len(cuts)
    return durations, accepted


def _nearest_candidate(
    candidates: list[dict[str, Any]],
    time_s: float,
    tolerance_s: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(float(item["time_s"]) - time_s))
    return nearest if abs(float(nearest["time_s"]) - time_s) <= tolerance_s else None


def truth_metrics(
    candidates: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    policy: str,
    threshold: float,
    tolerance_s: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    matched = 0
    same_sentence_total = 0
    same_sentence_false_cut = 0
    for truth in labels:
        candidate = _nearest_candidate(candidates, float(truth["time_s"]), tolerance_s)
        if candidate is None:
            continue
        matched += 1
        predicted = selected(candidate, policy=policy, threshold=threshold)
        label = str(truth.get("label") or "unsure")
        flags = {str(value) for value in truth.get("flags") or []}
        if label == "cut":
            tp += int(predicted)
            fn += int(not predicted)
        elif label == "continue":
            fp += int(predicted)
            if flags & SAME_SENTENCE_FLAGS:
                same_sentence_total += 1
                same_sentence_false_cut += int(predicted)
    return {
        "matched_truth_count": matched,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "cut_precision": tp / (tp + fp) if tp + fp else 0.0,
        "cut_recall": tp / (tp + fn) if tp + fn else 0.0,
        "same_sentence_continue_count": same_sentence_total,
        "same_sentence_false_cut_count": same_sentence_false_cut,
        "same_sentence_false_cut_rate": (
            same_sentence_false_cut / same_sentence_total
            if same_sentence_total
            else 0.0
        ),
    }


def analyze(
    *,
    aligned_segments_path: Path,
    timings_path: Path | None = None,
    omni_labels_path: Path | None = None,
    match_tolerance_s: float = 0.08,
) -> dict[str, Any]:
    aligned = _read_json(aligned_segments_path)
    segments = [dict(item) for item in aligned.get("segments") or []]
    candidates = [
        candidate
        for segment in segments
        for candidate in segment_candidates(segment)
    ]
    signature_source = aligned
    if timings_path is not None:
        signature_source = _read_json(timings_path).get("asr_details") or {}
    split_signature = (
        signature_source.get("boundary_signature", {})
        .get("boundary_pipeline", {})
        .get("semantic_split_model", {})
    )
    decision = dict(split_signature.get("decision_config") or {})
    short_core_max_s = float(decision.get("short_core_max_s", 6.0))
    short_threshold = float(decision.get("short_core_cut_threshold", 0.90))
    normal_threshold = float(decision.get("normal_cut_threshold", 0.75))
    labels = _read_jsonl(omni_labels_path) if omni_labels_path else []
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        durations, accepted = replay_durations(
            segments,
            policy=policy,
            short_core_max_s=short_core_max_s,
            short_threshold=short_threshold,
            normal_threshold=normal_threshold,
        )
        summary: dict[str, Any] = {
            "accepted_additional_cut_count": accepted,
            "duration": duration_metrics(durations),
        }
        if labels:
            summary["omni_truth"] = truth_metrics(
                candidates,
                labels,
                policy=policy,
                threshold=normal_threshold,
                tolerance_s=match_tolerance_s,
            )
        policies[policy] = summary
    argmax_truth = policies["probability_argmax"].get("omni_truth") or {}
    precision = float(argmax_truth.get("cut_precision") or 0.0)
    same_sentence_rate = float(
        argmax_truth.get("same_sentence_false_cut_rate") or 0.0
    )
    return {
        "schema": "split_cut_eager_operating_point_summary_v1",
        "source": str(aligned_segments_path),
        "segment_count": len(segments),
        "candidate_count": len(candidates),
        "decision_config": decision,
        "policies": policies,
        "gate": {
            "minimum_cut_precision": 0.90,
            "maximum_same_sentence_false_cut_rate": 0.02,
            "argmax_precision_pass": precision >= 0.90,
            "same_sentence_pause_pass": same_sentence_rate <= 0.02,
            "v2_runtime_ab_allowed": precision >= 0.90 and same_sentence_rate <= 0.02,
        },
        "notes": {
            "duration_is_evaluation_only": True,
            "proposal_coverage_requires_free_boundary_labels": True,
            "word_timing_available": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-segments", required=True)
    parser.add_argument("--timings", default="")
    parser.add_argument("--omni-labels", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-tolerance-s", type=float, default=0.08)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(
        aligned_segments_path=Path(args.aligned_segments),
        timings_path=Path(args.timings) if args.timings else None,
        omni_labels_path=Path(args.omni_labels) if args.omni_labels else None,
        match_tolerance_s=max(0.0, args.match_tolerance_s),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
