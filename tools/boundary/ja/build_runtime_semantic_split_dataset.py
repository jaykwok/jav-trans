#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.ja.backend import (  # noqa: E402
    SpeechBoundaryJaConfig,
    _bootstrap_frame_scores,
    decode_speech_island_segments,
)
from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.features import FeatureConfig, load_cached_feature  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.sequence_store import (  # noqa: E402
    StreamingFrameWriter,
    save_sequence_dataset,
)
from boundary.ja.model import (  # noqa: E402
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities_batch,
)
from boundary.ja.proposal import load_boundary_proposal_checkpoint  # noqa: E402
from boundary.sequence_features import (  # noqa: E402
    SPLIT_CANDIDATE_SCALAR_NAMES,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    load_ptm_projection,
    parse_extra_context_scales,
)
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA  # noqa: E402


LABELS = {"cut": 0, "continue": 1, "unsure": 2}
STRUCTURAL_ROLE_IDS = {
    "none": 0,
    "speech_to_speech": 1,
    "speech_to_noise": 2,
    "noise_to_speech": 3,
}
IGNORE_ID = -100


def boundary_noise_bucket(boundary: dict[str, Any]) -> str:
    count = int(boundary.get("inter_noise_unit_count") or 0)
    if count <= 0:
        return "touching_11"
    if count == 1:
        return "single_noise_101"
    return "multi_noise_1001_plus"


def semantic_split_truth_boundaries(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit = metadata.get("semantic_split_boundaries") or []
    if explicit:
        return [dict(item) for item in explicit]
    result: list[dict[str, Any]] = []
    for boundary in metadata.get("utterance_boundaries") or []:
        row = dict(boundary)
        noise_count = int(row.get("inter_noise_unit_count") or 0)
        if noise_count <= 0:
            row["structural_role"] = "speech_to_speech"
            result.append(row)
            continue
        previous_end = float(row["previous_speech_end_s"])
        next_start = float(row["next_speech_start_s"])
        result.extend(
            (
                {
                    **row,
                    "time_s": previous_end,
                    "structural_role": "speech_to_noise",
                },
                {
                    **row,
                    "time_s": next_start,
                    "structural_role": "noise_to_speech",
                },
            )
        )
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_runtime_proposals(
    proposals: Iterable[dict[str, Any]],
    *,
    truth_times_s: Iterable[float],
    truth_regions_s: Iterable[tuple[float, float]] | None = None,
    cut_match_s: float,
    unsure_radius_s: float,
) -> tuple[list[str], set[int], set[int], dict[int, int]]:
    """Match at most one real runtime proposal to each true semantic boundary."""

    proposal_rows = list(proposals)
    truth = [float(value) for value in truth_times_s]
    regions = (
        [(float(left), float(right)) for left, right in truth_regions_s]
        if truth_regions_s is not None
        else [(value, value) for value in truth]
    )
    if len(regions) != len(truth):
        raise ValueError("truth_regions_s length must match truth_times_s")

    def region_distance(proposal_time: float, truth_index: int) -> float:
        left, right = regions[truth_index]
        lower, upper = min(left, right), max(left, right)
        if lower <= proposal_time <= upper:
            return 0.0
        return min(abs(proposal_time - lower), abs(proposal_time - upper))

    cut_indexes: set[int] = set()
    matched_truth_indexes: set[int] = set()
    proposal_truth: dict[int, int] = {}
    pairs = sorted(
        (
            (
                region_distance(float(proposal["time_s"]), truth_index),
                abs(float(proposal["time_s"]) - truth_time),
                proposal_index,
                truth_index,
            )
            for proposal_index, proposal in enumerate(proposal_rows)
            for truth_index, truth_time in enumerate(truth)
        ),
        key=lambda item: (item[0], item[1]),
    )
    for distance, _midpoint_distance, proposal_index, truth_index in pairs:
        if distance > cut_match_s:
            break
        if proposal_index in cut_indexes or truth_index in matched_truth_indexes:
            continue
        cut_indexes.add(proposal_index)
        matched_truth_indexes.add(truth_index)
        proposal_truth[proposal_index] = truth_index

    labels: list[str] = []
    for proposal_index, proposal in enumerate(proposal_rows):
        if proposal_index in cut_indexes:
            labels.append("cut")
            continue
        proposal_time = float(proposal["time_s"])
        nearest = min(
            (
                region_distance(proposal_time, truth_index)
                for truth_index in range(len(truth))
            ),
            default=float("inf"),
        )
        labels.append("unsure" if nearest <= unsure_radius_s else "continue")
    return labels, cut_indexes, matched_truth_indexes, proposal_truth


def _batch_items(
    items: list[tuple[dict[str, Any], Any]],
    *,
    batch_size: int,
    max_batch_frames: int,
) -> list[list[tuple[dict[str, Any], Any]]]:
    ordered = sorted(items, key=lambda item: int(item[0].get("frame_count") or 0))
    result: list[list[tuple[dict[str, Any], Any]]] = []
    current: list[tuple[dict[str, Any], Any]] = []
    current_max = 0
    for item in ordered:
        frames = int(item[0].get("frame_count") or 0)
        next_max = max(current_max, frames)
        if current and (
            len(current) >= batch_size
            or next_max * (len(current) + 1) > max_batch_frames
        ):
            result.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, frames)
    if current:
        result.append(current)
    return result


def _score_windowed_batches(
    bundle,
    feature_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    window_frames: int,
    overlap_frames: int,
    max_batch_frames: int,
) -> list[np.ndarray]:
    """Score long sequences with the runtime's fixed scoring windows.

    Mirrors the runtime backend, which scores 20s windows with 4s overlap and
    averages probabilities in overlap regions; whole-sequence forwards OOM on
    the Mamba2 chunked scan (transient memory grows with the square of the
    per-item chunk count).
    """
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
            if start >= end:
                continue
            tasks.append((pair_index, start, end))
    score_sums = [np.zeros(total, dtype=np.float64) for total in totals]
    score_counts = [np.zeros(total, dtype=np.float64) for total in totals]
    group: list[tuple[int, int, int]] = []
    group_max = 0

    def _flush() -> None:
        nonlocal group, group_max
        if not group:
            return
        scored = score_speech_island_probabilities_batch(
            bundle,
            feature_pairs=[
                (feature_pairs[index][0][start:end], feature_pairs[index][1][start:end])
                for index, start, end in group
            ],
        )
        for (index, start, end), scores in zip(group, scored):
            score_sums[index][start:end] += scores
            score_counts[index][start:end] += 1.0
        group = []
        group_max = 0

    for task in tasks:
        frames = task[2] - task[1]
        next_max = max(group_max, frames)
        if group and next_max * (len(group) + 1) > max_batch_frames:
            _flush()
        group.append(task)
        group_max = max(group_max, frames)
    _flush()
    return [
        np.asarray(total_sum / np.maximum(count, 1.0), dtype=np.float32)
        for total_sum, count in zip(score_sums, score_counts)
    ]


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap()
    records = read_jsonl(Path(args.labels))
    manifest_rows = _rows(Path(args.feature_manifest))
    selected: list[tuple[dict[str, Any], Any]] = []
    for row in manifest_rows:
        record = records[int(row["label_index"])]
        if record.boundary_metadata:
            selected.append((row, record))
        if args.max_rows and len(selected) >= args.max_rows:
            break
    if not selected:
        raise ValueError("no feature rows with boundary metadata")

    scorer = load_speech_island_scorer_checkpoint(
        args.scorer_checkpoint,
        device=args.device,
    )
    proposal_bundle = (
        load_boundary_proposal_checkpoint(args.proposal_checkpoint, device=args.device)
        if args.proposal_checkpoint
        else None
    )
    feature_config = FeatureConfig(frame_hop_s=args.frame_hop_s)
    decode_config = SpeechBoundaryJaConfig(
        threshold=args.speech_threshold,
        speech_on_threshold=args.speech_on_threshold,
        speech_off_threshold=args.speech_off_threshold,
        frame_dilation_s=args.dilation_s,
        frame_hop_s=args.frame_hop_s,
        split_smooth_s=args.split_smooth_s,
        split_nms_s=args.split_nms_s,
        split_snap_s=args.split_snap_s,
        min_split_segment_s=args.min_split_segment_s,
        split_score_quantile=args.split_score_quantile,
        split_prominence_quantile=args.split_prominence_quantile,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_writer = StreamingFrameWriter(output)
    extra_context_scales = parse_extra_context_scales(args.extra_context_scales)
    ptm_projection = load_ptm_projection(args.ptm_projection)
    scalar_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    partitions: list[str] = []
    group_rows: list[str] = []
    time_rows: list[float] = []
    role_rows: list[int] = []
    offset_rows: list[float] = []
    pair_keys: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    total_truth = 0
    eligible_truth = 0
    proposed_truth = 0
    decoded_island_separated_truth = 0
    truth_by_noise_bucket: Counter[str] = Counter()
    eligible_by_noise_bucket: Counter[str] = Counter()
    matched_by_noise_bucket: Counter[str] = Counter()
    separated_by_noise_bucket: Counter[str] = Counter()
    processed = 0

    batches = _batch_items(
        selected,
        batch_size=args.batch_size,
        max_batch_frames=args.max_batch_frames,
    )
    for batch in batches:
        loaded: list[tuple[dict[str, Any], Any, np.ndarray, np.ndarray]] = []
        for row, record in batch:
            ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames))
            if total > 0:
                loaded.append((row, record, ptm[:total], mfcc[:total]))
        score_window_frames = max(1, int(round(args.score_window_s / args.frame_hop_s)))
        score_overlap_frames = max(
            0, int(round(args.score_overlap_s / args.frame_hop_s))
        )
        speech_batch = _score_windowed_batches(
            scorer,
            [(item[2], item[3]) for item in loaded],
            window_frames=score_window_frames,
            overlap_frames=score_overlap_frames,
            max_batch_frames=args.max_batch_frames,
        )
        proposal_batch = (
            _score_windowed_batches(
                proposal_bundle,
                [(item[2], item[3]) for item in loaded],
                window_frames=score_window_frames,
                overlap_frames=score_overlap_frames,
                max_batch_frames=args.max_batch_frames,
            )
            if proposal_bundle is not None
            else [None] * len(loaded)
        )
        for (row, record, ptm, mfcc), speech, proposal_scores in zip(
            loaded, speech_batch, proposal_batch
        ):
            total = min(ptm.shape[0], mfcc.shape[0], speech.size)
            ptm = ptm[:total]
            mfcc = mfcc[:total]
            speech = np.asarray(speech[:total], dtype=np.float32)
            if proposal_scores is not None:
                candidate_probabilities = np.asarray(
                    proposal_scores[:total], dtype=np.float32
                )
            else:
                audio_path = str(row.get("audio_path") or row.get("audio") or "")
                audio, sample_rate = load_audio_16k_mono(audio_path)
                if sample_rate != 16000:
                    raise ValueError(
                        f"expected 16kHz audio, got {sample_rate}: {audio_path}"
                    )
                _, candidate_probabilities = _bootstrap_frame_scores(
                    audio=audio,
                    sample_rate=sample_rate,
                    ptm=ptm,
                    mfcc=mfcc,
                    config=feature_config,
                )
            duration_s = min(float(record.duration_s), total * record.frame_hop_s)
            decoded = decode_speech_island_segments(
                speech_probabilities=speech,
                candidate_probabilities=candidate_probabilities,
                duration_s=duration_s,
                config=decode_config,
            )
            metadata = dict(record.boundary_metadata or {})
            truth_boundaries = [
                dict(item)
                for item in semantic_split_truth_boundaries(metadata)
                if 0.0 < float(item["time_s"]) < duration_s
            ]
            truth_times = [float(item["time_s"]) for item in truth_boundaries]
            truth_buckets = [
                boundary_noise_bucket(item) for item in truth_boundaries
            ]
            total_truth += len(truth_times)
            truth_by_noise_bucket.update(truth_buckets)
            truth_covered_by_island: set[int] = set()
            provider = FrameSequenceFeatureProvider(
                duration_s=duration_s,
                frame_hop_s=record.frame_hop_s,
                ptm=ptm,
                mfcc=mfcc,
                config=FrameSequenceFeatureConfig(max_ptm_dims=args.ptm_dim),
            )
            for segment_index, segment in enumerate(decoded.segments):
                core_start = float(segment.start)
                core_end = float(segment.end)
                core_truth_pairs = [
                    (truth_index, truth_boundaries[truth_index])
                    for truth_index, truth_time in enumerate(truth_times)
                    if core_start + args.core_margin_s
                    <= truth_time
                    <= core_end - args.core_margin_s
                ]
                truth_covered_by_island.update(index for index, _ in core_truth_pairs)
                eligible_truth += len(core_truth_pairs)
                eligible_by_noise_bucket.update(
                    truth_buckets[index] for index, _ in core_truth_pairs
                )
                proposals = list(segment.weak_cut_candidates)
                if not proposals:
                    continue
                labels, cut_indexes, matched_local_truth, proposal_truth = label_runtime_proposals(
                    proposals,
                    truth_times_s=[
                        float(boundary["time_s"]) for _, boundary in core_truth_pairs
                    ],
                    truth_regions_s=[
                        (
                            float(boundary["time_s"]),
                            float(boundary["time_s"]),
                        )
                        for _, boundary in core_truth_pairs
                    ],
                    cut_match_s=args.cut_match_s,
                    unsure_radius_s=args.unsure_radius_s,
                )
                proposed_truth += len(matched_local_truth)
                matched_global_truth = {
                    core_truth_pairs[local_index][0]
                    for local_index in matched_local_truth
                }
                matched_by_noise_bucket.update(
                    truth_buckets[index] for index in matched_global_truth
                )
                keep_indexes = set(cut_indexes)
                keep_indexes.update(
                    index for index, label in enumerate(labels) if label == "unsure"
                )
                continue_indexes = [
                    index for index, label in enumerate(labels) if label == "continue"
                ]
                continue_indexes.sort(
                    key=lambda index: float(proposals[index].get("strength") or 0.0),
                    reverse=True,
                )
                if args.max_continue_per_core > 0:
                    keep_indexes.update(
                        continue_indexes[: args.max_continue_per_core]
                    )
                else:
                    keep_indexes.update(continue_indexes)
                partition = str(metadata.get("source_partition") or "train")
                group_id = f"{record.audio_id}|island{segment_index:04d}"
                for proposal_index, (proposal, label) in enumerate(zip(proposals, labels)):
                    if proposal_index not in keep_indexes:
                        continue
                    offset_target = float("nan")
                    if label == "cut":
                        truth_local = proposal_truth[proposal_index]
                        boundary_row = core_truth_pairs[truth_local][1]
                        offset_target = float(boundary_row["time_s"]) - float(
                            proposal["time_s"]
                        )
                        role_name = str(
                            boundary_row.get("structural_role") or "speech_to_speech"
                        )
                        role_id = STRUCTURAL_ROLE_IDS.get(role_name, IGNORE_ID)
                        if role_name in {"speech_to_noise", "noise_to_speech"}:
                            pair_key = (
                                f"{record.audio_id}|b{int(boundary_row.get('index') or 0)}"
                            )
                        else:
                            pair_key = ""
                    elif label == "continue":
                        role_name = "none"
                        role_id = STRUCTURAL_ROLE_IDS["none"]
                        pair_key = ""
                    else:
                        role_name = ""
                        role_id = IGNORE_ID
                        pair_key = ""
                    frames, scalars = provider.features_for_split_candidate(
                        core_start_s=core_start,
                        core_end_s=core_end,
                        candidate=proposal,
                        speech_probabilities=speech,
                        left_context_s=1.6,
                        right_context_s=1.6,
                        gap_context_s=0.3,
                        left_bins=8,
                        gap_bins=4,
                        right_bins=8,
                        ptm_dim=args.ptm_dim,
                        extra_context_scales=extra_context_scales,
                        ptm_projection_mean=(
                            ptm_projection["mean"] if ptm_projection else None
                        ),
                        ptm_projection_components=(
                            ptm_projection["components"] if ptm_projection else None
                        ),
                    )
                    frame_writer.append(frames)
                    scalar_rows.append(scalars)
                    label_rows.append(LABELS[label])
                    partitions.append(partition)
                    group_rows.append(group_id)
                    time_rows.append(float(proposal["time_s"]))
                    role_rows.append(role_id)
                    offset_rows.append(offset_target)
                    pair_keys.append(pair_key)
                    label_counts[label] += 1
                    partition_counts[partition] += 1
                    if role_id != IGNORE_ID:
                        role_counts[role_name] += 1
                    nearest_truth = min(
                        (
                            abs(
                                float(proposal["time_s"])
                                - float(boundary["time_s"])
                            )
                            for _, boundary in core_truth_pairs
                        ),
                        default=None,
                    )
                    metadata_rows.append(
                        {
                            "audio_id": record.audio_id,
                            "partition": partition,
                            "segment_index": segment_index,
                            "nearest_truth_role": (
                                min(
                                    core_truth_pairs,
                                    key=lambda item: abs(
                                        float(proposal["time_s"])
                                        - float(item[1]["time_s"])
                                    ),
                                )[1].get("structural_role")
                                if core_truth_pairs
                                else ""
                            ),
                            "core_start": core_start,
                            "core_end": core_end,
                            "label": label,
                            "nearest_truth_distance_s": nearest_truth,
                            **dict(proposal),
                        }
                    )
            decoded_island_separated_truth += len(truth_times) - len(
                truth_covered_by_island
            )
            separated_by_noise_bucket.update(
                truth_buckets[index]
                for index in range(len(truth_times))
                if index not in truth_covered_by_island
            )
        processed += len(batch)
        if args.log_every and (
            processed == len(selected) or processed % args.log_every < len(batch)
        ):
            print(
                f"runtime_split_dataset_rows={processed}/{len(selected)} "
                f"candidates={len(label_rows)} eligible_truth={eligible_truth} "
                f"proposed_truth={proposed_truth}",
                flush=True,
            )

    if frame_writer.rows == 0:
        raise ValueError("runtime Scorer produced no semantic split candidates")
    _sidecar_path, frame_shape = frame_writer.finalize()
    scalar_array = np.stack(scalar_rows)
    pair_id_by_key: dict[str, int] = {}
    pair_ids: list[int] = []
    for key in pair_keys:
        if not key:
            pair_ids.append(-1)
            continue
        pair_ids.append(pair_id_by_key.setdefault(key, len(pair_id_by_key)))
    complete_pair_count = sum(
        1 for count in Counter(v for v in pair_ids if v >= 0).values() if count >= 2
    )
    save_sequence_dataset(
        output,
        frames_finalized=True,
        scalar_features=scalar_array,
        labels=np.asarray(label_rows, dtype=np.int64),
        partitions=np.asarray(partitions),
        group_ids=np.asarray(group_rows),
        times_s=np.asarray(time_rows, dtype=np.float32),
        structural_roles=np.asarray(role_rows, dtype=np.int64),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        omni_aux=np.full((len(label_rows), 3), -1.0, dtype=np.float32),
        offset_targets_s=np.asarray(offset_rows, dtype=np.float32),
    )
    metadata_path = output.with_suffix(".jsonl")
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    proposal_recall_by_noise_bucket = {
        bucket: {
            "truth": int(truth_by_noise_bucket[bucket]),
            "island_separated": int(separated_by_noise_bucket[bucket]),
            "eligible": int(eligible_by_noise_bucket[bucket]),
            "matched": int(matched_by_noise_bucket[bucket]),
            "eligible_proposal_recall": (
                matched_by_noise_bucket[bucket] / eligible_by_noise_bucket[bucket]
                if eligible_by_noise_bucket[bucket]
                else 1.0
            ),
        }
        for bucket in sorted(truth_by_noise_bucket)
    }
    summary = {
        "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "candidate_source": (
            "learned_boundary_proposal_v1"
            if proposal_bundle is not None
            else "runtime_scorer_weak_cut_candidates"
        ),
        "count": len(label_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "group_count": len(set(group_rows)),
        "structural_role_counts": dict(sorted(role_counts.items())),
        "pair_count": len(pair_id_by_key),
        "complete_pair_count": complete_pair_count,
        "frame_shape": list(frame_shape),
        "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
        "truth_boundary_count": total_truth,
        "eligible_truth_boundary_count": eligible_truth,
        "island_separated_truth_boundary_count": decoded_island_separated_truth,
        "matched_runtime_proposal_count": proposed_truth,
        "eligible_candidate_proposal_recall": (
            proposed_truth / eligible_truth if eligible_truth else 1.0
        ),
        "proposal_recall_by_noise_bucket": proposal_recall_by_noise_bucket,
        "config": {
            "cut_match_s": args.cut_match_s,
            "unsure_radius_s": args.unsure_radius_s,
            "max_continue_per_core": args.max_continue_per_core,
            "speech_on_threshold": args.speech_on_threshold,
            "speech_off_threshold": args.speech_off_threshold,
            "split_score_quantile": args.split_score_quantile,
            "split_prominence_quantile": args.split_prominence_quantile,
            "proposal_checkpoint": str(args.proposal_checkpoint or ""),
            "extra_context_scales": extra_context_scales,
            "ptm_projection": str(args.ptm_projection or ""),
            "score_window_s": args.score_window_s,
            "score_overlap_s": args.score_overlap_s,
        },
        "output": str(output),
        "metadata": str(metadata_path),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Semantic Split data only from the current runtime Scorer's real "
            "weak-cut proposals and exact runtime candidate features."
        )
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument(
        "--proposal-checkpoint",
        default="",
        help=(
            "Optional BoundaryProposalScorer checkpoint; replaces the bootstrap "
            "energy heuristic as the split candidate source."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--speech-threshold", type=float, default=0.15)
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
    parser.add_argument("--unsure-radius-s", type=float, default=0.60)
    parser.add_argument(
        "--max-continue-per-core",
        type=int,
        default=8,
        help="Strongest continue proposals kept per core; 0 keeps all.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--score-window-s",
        type=float,
        default=20.0,
        help="Scoring window length; matches the runtime backend window_s.",
    )
    parser.add_argument(
        "--score-overlap-s",
        type=float,
        default=4.0,
        help="Scoring window overlap; matches the runtime backend overlap_s.",
    )
    parser.add_argument(
        "--extra-context-scales",
        default="3.2:4,6.4:4",
        help=(
            "Coarser candidate context scales as '<seconds>:<bins_per_side>,...' "
            "appended after the base 1.6s bins; empty string disables."
        ),
    )
    parser.add_argument(
        "--ptm-projection",
        default="",
        help=(
            "Optional variance-preserving PTM projection npz "
            "(compute_ptm_projection.py); replaces the ptm[:, :ptm_dim] slice."
        ),
    )
    parser.add_argument("--max-batch-frames", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()
    if args.cut_match_s <= 0.0:
        parser.error("--cut-match-s must be positive")
    if args.unsure_radius_s < args.cut_match_s:
        parser.error("--unsure-radius-s must be >= --cut-match-s")
    if args.max_continue_per_core < 0:
        parser.error("--max-continue-per-core must be non-negative (0 keeps all)")
    return args


if __name__ == "__main__":
    run(parse_args())
