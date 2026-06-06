#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.chunk_packer import PackedChunk, pack_speech_segments  # noqa: E402
from boundary.ja.backend import SpeechBoundaryJaBackend, SpeechBoundaryJaConfig  # noqa: E402
from boundary.refiner import load_frame_sequence_refiner_checkpoint  # noqa: E402
from boundary.sequence_features import (  # noqa: E402
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)


@dataclass(frozen=True)
class PlannerProfile:
    name: str
    target_chunk_s: float = 3.0
    max_core_chunk_s: float = 5.0
    max_padded_chunk_s: float = 9.0
    min_chunk_s: float = 0.4
    start_weight: float = 1.5
    target_padding_s: float = 2.0
    max_splits_per_segment: int = 16
    sequence_batch_size: int = 256
    dp_chunk_base_cost: float = 0.04
    dp_over_target_weight: float = 0.30
    dp_far_over_target_weight: float = 1.50
    dp_under_min_weight: float = 0.20
    dp_long_gap_weight: float = 0.35
    dp_split_merge_weight: float = 0.35
    risk_target_s: float = 5.0

    def to_config(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "target_chunk_s": self.target_chunk_s,
            "max_core_chunk_s": self.max_core_chunk_s,
            "max_padded_chunk_s": self.max_padded_chunk_s,
            "min_chunk_s": self.min_chunk_s,
            "start_weight": self.start_weight,
            "target_padding_s": self.target_padding_s,
            "max_splits_per_segment": self.max_splits_per_segment,
            "sequence_batch_size": self.sequence_batch_size,
            "dp_chunk_base_cost": self.dp_chunk_base_cost,
            "dp_over_target_weight": self.dp_over_target_weight,
            "dp_far_over_target_weight": self.dp_far_over_target_weight,
            "dp_under_min_weight": self.dp_under_min_weight,
            "dp_long_gap_weight": self.dp_long_gap_weight,
            "dp_split_merge_weight": self.dp_split_merge_weight,
            "risk_target_s": self.risk_target_s,
        }


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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
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


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(data),
        "min": round(min(data), 6),
        "p50": round(quantile(data, 0.50), 6),
        "p90": round(quantile(data, 0.90), 6),
        "p95": round(quantile(data, 0.95), 6),
        "max": round(max(data), 6),
        "mean": round(statistics.fmean(data), 6),
    }


def row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def chunk_to_row(index: int, chunk: PackedChunk) -> dict[str, Any]:
    return {
        "chunk_index": index,
        "start": round(float(chunk.start), 6),
        "end": round(float(chunk.end), 6),
        "duration_s": round(float(chunk.duration), 6),
        "core_start": None if chunk.core_start is None else round(float(chunk.core_start), 6),
        "core_end": None if chunk.core_end is None else round(float(chunk.core_end), 6),
        "core_duration_s": round(
            max(0.0, float(chunk.core_end or 0.0) - float(chunk.core_start or 0.0)),
            6,
        ),
        "left_padding_s": round(float(chunk.left_padding_s), 6),
        "right_padding_s": round(float(chunk.right_padding_s), 6),
        "speech_island_count": len(chunk.speech_segments),
        "internal_gap_count": int(chunk.internal_gap_count),
        "internal_gap_max_s": round(float(chunk.internal_gap_max_s), 6),
        "split_reason": chunk.split_reason,
        "boundary_score": None if chunk.boundary_score is None else round(float(chunk.boundary_score), 6),
        "boundary_reason": chunk.boundary_reason,
        "boundary_source": chunk.boundary_source,
        "boundary_decision_merge": chunk.boundary_decision_merge,
        "boundary_merge_prob": (
            None if chunk.boundary_merge_prob is None else round(float(chunk.boundary_merge_prob), 6)
        ),
        "boundary_split_prob": (
            None if chunk.boundary_split_prob is None else round(float(chunk.boundary_split_prob), 6)
        ),
        "boundary_refine_delta_s": (
            None if chunk.boundary_refine_delta_s is None else round(float(chunk.boundary_refine_delta_s), 6)
        ),
        "boundary_decision_source": chunk.boundary_decision_source,
        "speech_segments": [
            {
                "start": round(float(segment.start), 6),
                "end": round(float(segment.end), 6),
                "score": None if segment.score is None else round(float(segment.score), 6),
            }
            for segment in chunk.speech_segments
        ],
    }


def diagnostics_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = project_path(value)
    if path.is_dir():
        path = path / "diagnostics.jsonl"
    return path if path.exists() else None


def fallback_risk_intervals(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return [], []
    fallback: list[dict[str, Any]] = []
    sentinel: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        start = row_float(row, "start")
        end = row_float(row, "end", start)
        if end <= start:
            continue
        fallback_type = str(row.get("fallback_type") or "").strip()
        is_fallback = fallback_type not in {"", "none"}
        has_sentinel = bool(row.get("sentinel_lines") or [])
        interval = {
            "start": start,
            "end": end,
            "duration_s": max(0.0, end - start),
            "fallback_type": fallback_type,
            "fallback_subtype": str(row.get("fallback_subtype") or "").strip(),
            "chunk_index": row.get("chunk_index"),
        }
        if is_fallback:
            fallback.append(interval)
        if has_sentinel:
            sentinel.append(interval)
    return fallback, sentinel


def mapped_risk_rows(
    rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    *,
    risk_target_s: float,
) -> list[dict[str, Any]]:
    if not intervals:
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        overlaps = [
            overlap_s(start, end, float(interval["start"]), float(interval["end"]))
            for interval in intervals
        ]
        total_overlap = sum(value for value in overlaps if value > 0.0)
        if total_overlap <= 0.0:
            continue
        copy = dict(row)
        copy["mapped_previous_risk_overlap_s"] = round(total_overlap, 6)
        copy["mapped_previous_risk_unsafe"] = float(row["duration_s"]) > risk_target_s
        result.append(copy)
    return result


def summarize_chunks(
    rows: list[dict[str, Any]],
    *,
    profile: PlannerProfile,
    previous_fallback_intervals: list[dict[str, Any]],
    previous_sentinel_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback_risk = mapped_risk_rows(
        rows,
        previous_fallback_intervals,
        risk_target_s=profile.risk_target_s,
    )
    sentinel_risk = mapped_risk_rows(
        rows,
        previous_sentinel_intervals,
        risk_target_s=profile.risk_target_s,
    )
    unsafe_risk = [
        row
        for row in fallback_risk
        if bool(row.get("mapped_previous_risk_unsafe"))
    ]
    split_reasons = Counter(str(row["split_reason"]) for row in rows)
    decision_sources = Counter(str(row.get("boundary_decision_source") or "none") for row in rows)
    return {
        "profile": profile.to_config(),
        "chunk_count": len(rows),
        "duration_s": stats(float(row["duration_s"]) for row in rows),
        "core_duration_s": stats(float(row["core_duration_s"]) for row in rows),
        "speech_island_count": stats(float(row["speech_island_count"]) for row in rows),
        "internal_gap_max_s": stats(float(row["internal_gap_max_s"]) for row in rows),
        "chunk_over_target_count": sum(1 for row in rows if float(row["duration_s"]) > profile.target_chunk_s),
        "chunk_over_20s_count": sum(1 for row in rows if float(row["duration_s"]) > 20.0),
        "chunk_over_30s_count": sum(1 for row in rows if float(row["duration_s"]) > 30.0),
        "mapped_previous_fallback_chunk_count": len(fallback_risk),
        "mapped_previous_fallback_unsafe_count": len(unsafe_risk),
        "mapped_previous_fallback_safe_ratio": round(
            (len(fallback_risk) - len(unsafe_risk)) / len(fallback_risk),
            6,
        )
        if fallback_risk
        else 1.0,
        "mapped_previous_fallback_duration_s": stats(float(row["duration_s"]) for row in fallback_risk),
        "mapped_previous_sentinel_chunk_count": len(sentinel_risk),
        "mapped_previous_sentinel_duration_s": stats(float(row["duration_s"]) for row in sentinel_risk),
        "split_reason_counts": dict(split_reasons.most_common()),
        "decision_source_counts": dict(decision_sources.most_common()),
        "planner_dp_count": split_reasons.get("planner_dp", 0),
        "learned_split_count": sum(
            count
            for reason, count in split_reasons.items()
            if reason.startswith("boundary_refiner:learned_sequence_split")
        ),
    }


def default_profiles(args: argparse.Namespace) -> list[PlannerProfile]:
    base = PlannerProfile(
        name="baseline_jav_short",
        target_chunk_s=args.boundary_planner_target_chunk_s,
        max_core_chunk_s=args.boundary_planner_max_core_chunk_s,
        max_padded_chunk_s=args.boundary_planner_max_padded_chunk_s,
        min_chunk_s=args.boundary_planner_min_chunk_s,
        start_weight=args.boundary_planner_start_weight,
        target_padding_s=args.boundary_planner_target_padding_s,
        max_splits_per_segment=args.boundary_planner_max_splits_per_segment,
        sequence_batch_size=args.boundary_planner_sequence_batch_size,
        risk_target_s=args.risk_target_s,
    )
    return [
        base,
        replace(
            base,
            name="duration_tight_3s",
            target_chunk_s=3.0,
            dp_over_target_weight=0.60,
            dp_far_over_target_weight=3.00,
            dp_split_merge_weight=0.25,
        ),
        replace(
            base,
            name="gap_strict_3s",
            target_chunk_s=3.0,
            dp_over_target_weight=0.60,
            dp_far_over_target_weight=3.00,
            dp_long_gap_weight=0.90,
            dp_split_merge_weight=0.20,
        ),
        replace(
            base,
            name="fallback_safe_4s",
            target_chunk_s=4.0,
            max_core_chunk_s=5.0,
            max_padded_chunk_s=9.0,
            dp_over_target_weight=1.00,
            dp_far_over_target_weight=4.00,
            dp_long_gap_weight=0.90,
            dp_split_merge_weight=0.15,
        ),
        replace(
            base,
            name="start_priority_3s",
            target_chunk_s=3.0,
            start_weight=2.00,
            dp_over_target_weight=0.80,
            dp_far_over_target_weight=3.00,
            dp_long_gap_weight=0.70,
            dp_split_merge_weight=0.20,
        ),
    ]


def build_feature_provider(
    payload: Any,
    *,
    duration_s: float,
    target_chunk_s: float,
    args: argparse.Namespace,
) -> FrameSequenceFeatureProvider:
    if not isinstance(payload, dict) or payload.get("schema") != FRAME_SEQUENCE_FRAMES_SCHEMA:
        raise ValueError(f"SpeechBoundary-JA did not return {FRAME_SEQUENCE_FRAMES_SCHEMA}")
    ptm = payload.get("ptm")
    mfcc = payload.get("mfcc")
    if not isinstance(ptm, list) or not isinstance(mfcc, list):
        raise ValueError("sequence feature frames must contain ptm and mfcc arrays")
    return FrameSequenceFeatureProvider(
        duration_s=duration_s,
        frame_hop_s=float(payload["frame_hop_s"]),
        ptm=ptm,
        mfcc=mfcc,
        config=FrameSequenceFeatureConfig(
            left_context_s=args.boundary_frame_sequence_left_context_s,
            right_context_s=args.boundary_frame_sequence_right_context_s,
            max_ptm_dims=args.boundary_frame_sequence_max_ptm_dims,
            include_mfcc=bool(args.boundary_frame_sequence_include_mfcc),
        ),
        target_chunk_s=target_chunk_s,
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    audio = project_path(args.audio)
    output_dir = project_path(args.output_dir)
    diagnostics = diagnostics_path(args.diagnostics)
    previous_fallback, previous_sentinel = fallback_risk_intervals(diagnostics)

    os.environ["BOUNDARY_REFINER_ENABLED"] = "1"
    boundary_config = SpeechBoundaryJaConfig(
        threshold=args.speech_boundary_threshold,
        pad_s=args.speech_boundary_pad_s,
        ptm=args.speech_boundary_ptm,
        model_path=str(project_path(args.speech_boundary_model_path)),
        device=args.speech_boundary_device,
        dtype=args.speech_boundary_dtype,
        attention=args.speech_boundary_attention,
        window_s=args.speech_boundary_window_s,
        overlap_s=args.speech_boundary_overlap_s,
        min_segment_s=args.speech_boundary_min_segment_s,
        merge_gap_s=args.speech_boundary_merge_gap_s,
        cut_threshold=args.speech_boundary_cut_threshold,
        apply_cut_to_speech=args.speech_boundary_apply_cut_to_speech,
        export_sequence_features=True,
        sequence_feature_max_ptm_dims=args.boundary_frame_sequence_max_ptm_dims,
    )
    started = time.perf_counter()
    backend = SpeechBoundaryJaBackend(boundary_config)
    segmentation = backend.segment(str(audio))
    segment_time_s = time.perf_counter() - started

    frame_scores = segmentation.parameters.get("frame_scores")
    cut_frame_scores = segmentation.parameters.get("cut_frame_scores")
    score_frame_hop_s = float(segmentation.parameters.get("frame_hop_s") or boundary_config.frame_hop_s)
    sequence_frames = segmentation.parameters.get("sequence_feature_frames")
    refiner = load_frame_sequence_refiner_checkpoint(
        project_path(args.boundary_refiner_model_path),
        threshold=args.boundary_refiner_threshold,
        backbone_override=args.boundary_refiner_backbone,
        device=args.boundary_refiner_device,
    )

    profile_summaries: list[dict[str, Any]] = []
    for profile in default_profiles(args):
        profile_started = time.perf_counter()
        provider = build_feature_provider(
            sequence_frames,
            duration_s=segmentation.audio_duration_sec,
            target_chunk_s=profile.target_chunk_s,
            args=args,
        )
        provider.validate_for_checkpoint(refiner.feature_names, refiner.feature_schema_hash)
        chunks = pack_speech_segments(
            segmentation.segments,
            frame_hop_s=args.boundary_feature_frame_hop_s,
            max_core_chunk_s=profile.max_core_chunk_s,
            max_padded_chunk_s=profile.max_padded_chunk_s,
            target_chunk_s=profile.target_chunk_s,
            min_chunk_s=profile.min_chunk_s,
            target_padding_s=profile.target_padding_s,
            start_weight=profile.start_weight,
            frame_scores=frame_scores,
            score_frame_hop_s=score_frame_hop_s,
            cut_frame_scores=cut_frame_scores,
            boundary_refiner=None,
            sequence_boundary_refiner=refiner,
            sequence_feature_provider=provider,
            max_splits_per_segment=profile.max_splits_per_segment,
            sequence_batch_size=profile.sequence_batch_size,
            dp_chunk_base_cost=profile.dp_chunk_base_cost,
            dp_over_target_weight=profile.dp_over_target_weight,
            dp_far_over_target_weight=profile.dp_far_over_target_weight,
            dp_under_min_weight=profile.dp_under_min_weight,
            dp_long_gap_weight=profile.dp_long_gap_weight,
            dp_split_merge_weight=profile.dp_split_merge_weight,
        )
        rows = [chunk_to_row(index, chunk) for index, chunk in enumerate(chunks)]
        summary = summarize_chunks(
            rows,
            profile=profile,
            previous_fallback_intervals=previous_fallback,
            previous_sentinel_intervals=previous_sentinel,
        )
        summary["runtime_s"] = round(time.perf_counter() - profile_started, 6)
        profile_summaries.append(summary)
        write_jsonl(output_dir / f"{profile.name}.chunks.jsonl", rows)

    payload = {
        "schema": "dp_cost_real_boundary_sweep_v1",
        "audio": project_rel(audio),
        "diagnostics": project_rel(diagnostics),
        "output_dir": project_rel(output_dir),
        "segment_time_s": round(segment_time_s, 6),
        "speech_segment_count": len(segmentation.segments),
        "speech_group_count": len(segmentation.groups),
        "audio_duration_s": round(float(segmentation.audio_duration_sec), 6),
        "previous_fallback_interval_count": len(previous_fallback),
        "previous_sentinel_interval_count": len(previous_sentinel),
        "boundary_signature": backend.signature(),
        "refiner_signature": refiner.signature(),
        "profiles": profile_summaries,
        "interpretation": {
            "sweep_kind": "real_boundary_rerun_no_asr",
            "fallback_count_gate": "observe_only",
            "previous_fallback_mapping": "time-overlap risk estimate only; ASR and forced alignment are not rerun",
            "primary_goal": "shorter fallback-risk chunks, fewer 20-30s coarse chunks, readable start-focused boundaries",
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_jsonl(output_dir / "profiles.jsonl", profile_summaries)
    (output_dir / "summary.md").write_text(build_markdown(payload), encoding="utf-8")
    return payload


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Real DP Cost Sweep",
        "",
        f"- audio: `{summary['audio']}`",
        f"- diagnostics: `{summary['diagnostics']}`",
        f"- sweep: `SpeechBoundary-JA + frame-sequence refiner + DP planner`, no ASR/aligner rerun",
        f"- segment time: `{summary['segment_time_s']:.2f}s`",
        f"- speech segments/groups: `{summary['speech_segment_count']}` / `{summary['speech_group_count']}`",
        f"- previous fallback/sentinel intervals mapped by time overlap: `{summary['previous_fallback_interval_count']}` / `{summary['previous_sentinel_interval_count']}`",
        "",
        "| profile | chunks | >target | >20s | risk chunks | unsafe risk | risk safe | duration p50/p90/max | risk p50/p90/max | split reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["profiles"]:
        duration = row["duration_s"]
        risk = row["mapped_previous_fallback_duration_s"]
        split_reasons = ", ".join(
            f"{key}:{value}"
            for key, value in list(row["split_reason_counts"].items())[:3]
        )
        lines.append(
            f"| `{row['profile']['name']}` | {row['chunk_count']} | "
            f"{row['chunk_over_target_count']} | {row['chunk_over_20s_count']} | "
            f"{row['mapped_previous_fallback_chunk_count']} | "
            f"{row['mapped_previous_fallback_unsafe_count']} | "
            f"{row['mapped_previous_fallback_safe_ratio']:.3f} | "
            f"{duration['p50']:.2f}/{duration['p90']:.2f}/{duration['max']:.2f}s | "
            f"{risk['p50']:.2f}/{risk['p90']:.2f}/{risk['max']:.2f}s | "
            f"{split_reasons} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- This reruns boundary extraction and planner decisions, but does not rerun Qwen ASR or forced alignment.",
            "- Fallback count is an observation, not a hard failure gate.",
            "- Mapped fallback-risk metrics reuse the previous diagnostics by time overlap; they are a cheap ranking signal before GPU ASR closure.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun SpeechBoundary-JA once and sweep DP planner costs without ASR."
    )
    parser.add_argument("--audio", required=True, help="Prepared 16 kHz mono wav.")
    parser.add_argument("--diagnostics", help="Previous diagnostics.jsonl or diagnostics directory for risk mapping.")
    parser.add_argument("--output-dir", default="agents/temp/speech-boundary-ja/dp-cost-real-sweep")
    parser.add_argument(
        "--boundary-refiner-model-path",
        default="datasets/train/fusionvad-ja/v1-23-boundary-refiner/qwen3-asr-0.6b-full29239/boundary-refiner-frame-sequence-mamba2-v1/boundary_refiner.pt",
    )
    parser.add_argument("--boundary-refiner-backbone", default="transformers.Mamba2Model")
    parser.add_argument("--boundary-refiner-device", default="auto")
    parser.add_argument("--boundary-refiner-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-feature-frame-hop-s", type=float, default=0.02)
    parser.add_argument("--boundary-frame-sequence-left-context-s", type=float, default=0.60)
    parser.add_argument("--boundary-frame-sequence-right-context-s", type=float, default=0.60)
    parser.add_argument("--boundary-frame-sequence-max-ptm-dims", type=int, default=64)
    parser.add_argument("--boundary-frame-sequence-include-mfcc", type=int, default=1)
    parser.add_argument("--boundary-planner-max-core-chunk-s", type=float, default=5.0)
    parser.add_argument("--boundary-planner-max-padded-chunk-s", type=float, default=9.0)
    parser.add_argument("--boundary-planner-target-chunk-s", type=float, default=3.0)
    parser.add_argument("--boundary-planner-min-chunk-s", type=float, default=0.4)
    parser.add_argument("--boundary-planner-start-weight", type=float, default=1.5)
    parser.add_argument("--boundary-planner-target-padding-s", type=float, default=2.0)
    parser.add_argument("--boundary-planner-max-splits-per-segment", type=int, default=16)
    parser.add_argument("--boundary-planner-sequence-batch-size", type=int, default=256)
    parser.add_argument("--risk-target-s", type=float, default=5.0)
    parser.add_argument("--speech-boundary-threshold", type=float, default=0.200)
    parser.add_argument("--speech-boundary-pad-s", type=float, default=0.2)
    parser.add_argument("--speech-boundary-ptm", default="jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame")
    parser.add_argument("--speech-boundary-model-path", default="models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame")
    parser.add_argument("--speech-boundary-device", default="auto")
    parser.add_argument("--speech-boundary-dtype", default="bfloat16")
    parser.add_argument("--speech-boundary-attention", default="sdpa")
    parser.add_argument("--speech-boundary-window-s", type=float, default=30.0)
    parser.add_argument("--speech-boundary-overlap-s", type=float, default=1.0)
    parser.add_argument("--speech-boundary-min-segment-s", type=float, default=0.05)
    parser.add_argument("--speech-boundary-merge-gap-s", type=float, default=0.0)
    parser.add_argument("--speech-boundary-cut-threshold", type=float, default=0.500)
    parser.add_argument("--speech-boundary-apply-cut-to-speech", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run_sweep(parse_args(argv))
    print(f"summary={project_rel(Path(summary['output_dir']) / 'summary.json')}")
    print(f"markdown={project_rel(Path(summary['output_dir']) / 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
