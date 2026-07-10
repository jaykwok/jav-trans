#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.backend import (  # noqa: E402
    SpeechBoundaryJaConfig,
    decode_speech_island_segments,
)
from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.dual_head import (  # noqa: E402
    load_dual_head_checkpoint,
    score_dual_head_probabilities_batch,
)
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import load_speech_island_scorer_checkpoint  # noqa: E402
from boundary.ja.proposal import load_boundary_proposal_checkpoint  # noqa: E402
from tools.boundary.ja.build_runtime_semantic_split_dataset import (  # noqa: E402
    _batch_items,
    _score_windowed_batches,
    label_runtime_proposals,
    semantic_split_truth_boundaries,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_dual_windowed_batches(
    bundle,
    feature_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    window_frames: int,
    overlap_frames: int,
    max_batch_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if window_frames <= overlap_frames:
        raise ValueError("window_frames must exceed overlap_frames")
    stride = window_frames - overlap_frames
    tasks: list[tuple[int, int, int]] = []
    totals: list[int] = []
    for pair_index, (ptm, mfcc) in enumerate(feature_pairs):
        total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        totals.append(total)
        for start in range(0, max(1, total), stride):
            end = min(total, start + window_frames)
            if start < end:
                tasks.append((pair_index, start, end))
    speech_sums = [np.zeros(total, dtype=np.float64) for total in totals]
    proposal_sums = [np.zeros(total, dtype=np.float64) for total in totals]
    score_counts = [np.zeros(total, dtype=np.float64) for total in totals]
    group: list[tuple[int, int, int]] = []
    group_max = 0

    def flush() -> None:
        nonlocal group, group_max
        if not group:
            return
        scores = score_dual_head_probabilities_batch(
            bundle,
            feature_pairs=[
                (feature_pairs[index][0][start:end], feature_pairs[index][1][start:end])
                for index, start, end in group
            ],
        )
        for (index, start, end), (speech, proposal) in zip(
            group, scores, strict=True
        ):
            speech_sums[index][start:end] += speech
            proposal_sums[index][start:end] += proposal
            score_counts[index][start:end] += 1.0
        group = []
        group_max = 0

    for task in tasks:
        frames = task[2] - task[1]
        next_max = max(group_max, frames)
        if group and next_max * (len(group) + 1) > max_batch_frames:
            flush()
        group.append(task)
        group_max = next_max
    flush()
    speech = [
        np.asarray(total / np.maximum(count, 1.0), dtype=np.float32)
        for total, count in zip(speech_sums, score_counts, strict=True)
    ]
    proposal = [
        np.asarray(total / np.maximum(count, 1.0), dtype=np.float32)
        for total, count in zip(proposal_sums, score_counts, strict=True)
    ]
    return speech, proposal


def match_times(
    left: list[float], right: list[float], *, tolerance_s: float
) -> int:
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs = sorted(
        (abs(a - b), ai, bi)
        for ai, a in enumerate(left)
        for bi, b in enumerate(right)
    )
    for distance, left_index, right_index in pairs:
        if distance > tolerance_s:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
    return len(used_left)


def _decoded_stats(
    *,
    speech: np.ndarray,
    proposal: np.ndarray,
    record: Any,
    config: SpeechBoundaryJaConfig,
    core_margin_s: float,
    cut_match_s: float,
) -> dict[str, Any]:
    duration_s = min(float(record.duration_s), len(speech) * record.frame_hop_s)
    decoded = decode_speech_island_segments(
        speech_probabilities=speech,
        candidate_probabilities=proposal,
        duration_s=duration_s,
        config=config,
    )
    truth = [
        dict(row)
        for row in semantic_split_truth_boundaries(
            dict(record.boundary_metadata or {})
        )
        if 0.0 < float(row["time_s"]) < duration_s
    ]
    eligible = 0
    matched = 0
    candidate_times: list[float] = []
    for segment in decoded.segments:
        core_truth = [
            float(row["time_s"])
            for row in truth
            if float(segment.start) + core_margin_s
            <= float(row["time_s"])
            <= float(segment.end) - core_margin_s
        ]
        proposals = list(segment.weak_cut_candidates)
        candidate_times.extend(float(row["time_s"]) for row in proposals)
        eligible += len(core_truth)
        if proposals and core_truth:
            _labels, _cuts, matched_truth, _proposal_truth = label_runtime_proposals(
                proposals,
                truth_times_s=core_truth,
                cut_match_s=cut_match_s,
                unsure_radius_s=cut_match_s,
            )
            matched += len(matched_truth)
    return {
        "eligible": eligible,
        "matched": matched,
        "candidate_times": candidate_times,
        "speech_frames": np.asarray(decoded.dilated_frames, dtype=bool),
        "island_count": len(decoded.segments),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    vram_safety_ratio = apply_vram_safety_cap()
    records = read_jsonl(Path(args.labels))
    manifest = _rows(Path(args.feature_manifest))
    selected = []
    for row in manifest:
        record = records[int(row["label_index"])]
        partition = str((record.boundary_metadata or {}).get("source_partition") or "train")
        if partition == args.partition and record.boundary_metadata:
            selected.append((row, record))
        if args.max_rows and len(selected) >= args.max_rows:
            break
    if not selected:
        raise ValueError("no gate rows selected")

    dual = load_dual_head_checkpoint(args.dual_checkpoint, device=args.device)
    speech_teacher = load_speech_island_scorer_checkpoint(
        args.speech_checkpoint, device=args.device
    )
    proposal_teacher = load_boundary_proposal_checkpoint(
        args.proposal_checkpoint, device=args.device
    )
    config = SpeechBoundaryJaConfig(
        speech_on_threshold=args.speech_on_threshold,
        speech_off_threshold=args.speech_off_threshold,
        frame_hop_s=args.frame_hop_s,
        frame_dilation_s=args.dilation_s,
        split_smooth_s=args.split_smooth_s,
        split_nms_s=args.split_nms_s,
        split_snap_s=args.split_snap_s,
        min_split_segment_s=args.min_split_segment_s,
        split_score_quantile=args.split_score_quantile,
        split_prominence_quantile=args.split_prominence_quantile,
    )
    totals = {
        "dual_eligible": 0,
        "dual_matched": 0,
        "legacy_eligible": 0,
        "legacy_matched": 0,
        "dual_candidates": 0,
        "legacy_candidates": 0,
        "position_matches": 0,
        "speech_intersection": 0,
        "speech_union": 0,
        "dual_islands": 0,
        "legacy_islands": 0,
    }
    window_frames = int(round(args.score_window_s / args.frame_hop_s))
    overlap_frames = int(round(args.score_overlap_s / args.frame_hop_s))
    processed = 0
    for batch in _batch_items(
        selected,
        batch_size=args.batch_size,
        max_batch_frames=args.max_batch_frames,
    ):
        loaded = []
        for row, record in batch:
            ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            count = min(len(ptm), len(mfcc), len(record.speech_frames))
            loaded.append((record, ptm[:count], mfcc[:count]))
        pairs = [(ptm, mfcc) for _record, ptm, mfcc in loaded]
        dual_speech, dual_proposal = _score_dual_windowed_batches(
            dual,
            pairs,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
            max_batch_frames=args.max_batch_frames,
        )
        legacy_speech = _score_windowed_batches(
            speech_teacher,
            pairs,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
            max_batch_frames=args.max_batch_frames,
        )
        legacy_proposal = _score_windowed_batches(
            proposal_teacher,
            pairs,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
            max_batch_frames=args.max_batch_frames,
        )
        for index, (record, _ptm, _mfcc) in enumerate(loaded):
            dual_stats = _decoded_stats(
                speech=dual_speech[index],
                proposal=dual_proposal[index],
                record=record,
                config=config,
                core_margin_s=args.core_margin_s,
                cut_match_s=args.cut_match_s,
            )
            legacy_stats = _decoded_stats(
                speech=legacy_speech[index],
                proposal=legacy_proposal[index],
                record=record,
                config=config,
                core_margin_s=args.core_margin_s,
                cut_match_s=args.cut_match_s,
            )
            for prefix, stats in (("dual", dual_stats), ("legacy", legacy_stats)):
                totals[f"{prefix}_eligible"] += stats["eligible"]
                totals[f"{prefix}_matched"] += stats["matched"]
                totals[f"{prefix}_candidates"] += len(stats["candidate_times"])
                totals[f"{prefix}_islands"] += stats["island_count"]
            totals["position_matches"] += match_times(
                dual_stats["candidate_times"],
                legacy_stats["candidate_times"],
                tolerance_s=args.position_match_s,
            )
            dual_mask = dual_stats["speech_frames"]
            legacy_mask = legacy_stats["speech_frames"]
            count = min(len(dual_mask), len(legacy_mask))
            totals["speech_intersection"] += int(
                np.logical_and(dual_mask[:count], legacy_mask[:count]).sum()
            )
            totals["speech_union"] += int(
                np.logical_or(dual_mask[:count], legacy_mask[:count]).sum()
            )
        processed += len(loaded)
        if args.log_every and (
            processed == len(selected) or processed % args.log_every < len(loaded)
        ):
            print(f"dual_head_gate={processed}/{len(selected)}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    dual_recall = totals["dual_matched"] / max(1, totals["dual_eligible"])
    legacy_recall = totals["legacy_matched"] / max(1, totals["legacy_eligible"])
    position_recall = totals["position_matches"] / max(
        1, totals["legacy_candidates"]
    )
    position_precision = totals["position_matches"] / max(
        1, totals["dual_candidates"]
    )
    candidate_jaccard = totals["position_matches"] / max(
        1,
        totals["dual_candidates"]
        + totals["legacy_candidates"]
        - totals["position_matches"],
    )
    speech_jaccard = totals["speech_intersection"] / max(
        1, totals["speech_union"]
    )
    coverage_pass = dual_recall > args.min_proposal_coverage
    position_pass = position_recall >= args.min_position_match_recall
    summary = {
        "schema": "speech_proposal_dual_head_gate_v1",
        "rows": len(selected),
        "partition": args.partition,
        "dual_eligible_proposal_recall": dual_recall,
        "legacy_eligible_proposal_recall": legacy_recall,
        "proposal_coverage_gate_pass": coverage_pass,
        "candidate_position_match_recall": position_recall,
        "candidate_position_match_precision": position_precision,
        "candidate_position_jaccard": candidate_jaccard,
        "candidate_position_gate_pass": position_pass,
        "candidate_count_churn": (
            abs(totals["dual_candidates"] - totals["legacy_candidates"])
            / max(1, totals["legacy_candidates"])
        ),
        "speech_mask_jaccard": speech_jaccard,
        "island_count_churn": (
            abs(totals["dual_islands"] - totals["legacy_islands"])
            / max(1, totals["legacy_islands"])
        ),
        "promote_allowed": coverage_pass and position_pass,
        "stop_reason": (
            ""
            if coverage_pass and position_pass
            else (
                "proposal_coverage_below_gate"
                if not coverage_pass
                else "candidate_position_inconsistency"
            )
        ),
        "counts": totals,
        "vram_safety_ratio": vram_safety_ratio,
        "shared_vram_budget": False,
        "config": {
            "position_match_s": args.position_match_s,
            "min_proposal_coverage": args.min_proposal_coverage,
            "min_position_match_recall": args.min_position_match_recall,
            "score_window_s": args.score_window_s,
            "score_overlap_s": args.score_overlap_s,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate the Stage F dual head.")
    parser.add_argument("--dual-checkpoint", required=True)
    parser.add_argument("--speech-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-frames", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--speech-on-threshold", type=float, default=0.15)
    parser.add_argument("--speech-off-threshold", type=float, default=0.15)
    parser.add_argument("--dilation-s", type=float, default=0.2)
    parser.add_argument("--split-smooth-s", type=float, default=0.08)
    parser.add_argument("--split-nms-s", type=float, default=0.12)
    parser.add_argument("--split-snap-s", type=float, default=0.10)
    parser.add_argument("--min-split-segment-s", type=float, default=0.08)
    parser.add_argument("--split-score-quantile", type=float, default=0.10)
    parser.add_argument("--split-prominence-quantile", type=float, default=0.10)
    parser.add_argument("--core-margin-s", type=float, default=0.08)
    parser.add_argument("--cut-match-s", type=float, default=0.30)
    parser.add_argument("--position-match-s", type=float, default=0.02)
    parser.add_argument("--score-window-s", type=float, default=20.0)
    parser.add_argument("--score-overlap-s", type=float, default=4.0)
    parser.add_argument("--min-proposal-coverage", type=float, default=0.98)
    parser.add_argument("--min-position-match-recall", type=float, default=0.98)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
