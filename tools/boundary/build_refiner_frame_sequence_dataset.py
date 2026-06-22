#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import validate_checkpoint_repo_id
from boundary.ja import LabelRecord, TeacherSegment, load_cached_feature
from boundary.ja.backend import SpeechBoundaryJaConfig, decode_frame_boundary_segments
from boundary.ja.model import (
    load_feature_frame_scorer_checkpoint,
    score_feature_frame_boundary_probabilities_batch,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FrameSequenceFeatureConfig,
    boundary_window_sequence_features,
    feature_extraction_hash,
    feature_extraction_signature,
    frame_sequence_feature_names,
    get_feature_dim,
    validate_sequence_features,
)

DATASET_SCHEMA = "boundary_edge_refiner_dataset_v6"
DATASET_SOURCE = "scorer_v4_predicted_island_edges"


@dataclass(frozen=True)
class FrameSequenceConfig:
    left_context_s: float = 0.60
    right_context_s: float = 0.60
    max_ptm_dims: int = 64
    include_mfcc: bool = True
    frame_hop_s: float = 0.02
    threshold: float = 0.200
    speech_on_threshold: float | None = None
    speech_off_threshold: float | None = None
    frame_dilation_s: float = 0.2
    min_segment_s: float = 0.05
    drop_gap_threshold: float = 0.75
    split_target_s: float = 5.0
    split_smooth_s: float = 0.08
    split_nms_s: float = 0.20
    split_snap_s: float = 0.10
    min_split_segment_s: float = 0.08
    split_score_quantile: float = 0.50
    split_prominence_quantile: float = 0.50
    boundary_delta_max_s: float = 0.5
    max_edge_alignment_distance_s: float = 0.5
    min_edge_overlap_s: float = 0.0

    def feature_config(self) -> FrameSequenceFeatureConfig:
        return FrameSequenceFeatureConfig(
            left_context_s=self.left_context_s,
            right_context_s=self.right_context_s,
            max_ptm_dims=self.max_ptm_dims,
            include_mfcc=self.include_mfcc,
        )

    def decoder_config(self, *, frame_hop_s: float, ptm_repo_id: str) -> SpeechBoundaryJaConfig:
        return SpeechBoundaryJaConfig(
            threshold=self.threshold,
            speech_on_threshold=self.speech_on_threshold,
            speech_off_threshold=self.speech_off_threshold,
            frame_dilation_s=self.frame_dilation_s,
            frame_hop_s=frame_hop_s,
            ptm=ptm_repo_id,
            model_path="",
            min_segment_s=self.min_segment_s,
            drop_gap_threshold=self.drop_gap_threshold,
            split_target_s=self.split_target_s,
            split_smooth_s=self.split_smooth_s,
            split_nms_s=self.split_nms_s,
            split_snap_s=self.split_snap_s,
            min_split_segment_s=self.min_split_segment_s,
            split_score_quantile=self.split_score_quantile,
            split_prominence_quantile=self.split_prominence_quantile,
            no_download=True,
        )


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


def build_frame_sequence_dataset(
    *,
    labels_paths: Sequence[Path],
    feature_manifest_paths: Sequence[Path],
    scorer_checkpoint_path: Path,
    output_jsonl: Path,
    summary_json: Path | None = None,
    config: FrameSequenceConfig = FrameSequenceConfig(),
    limit: int | None = None,
    scorer_device: str = "cpu",
    scorer_batch_size: int = 1,
    scorer_max_padded_frames: int = 4096,
    log_every: int = 0,
) -> dict[str, Any]:
    if len(labels_paths) != len(feature_manifest_paths):
        raise ValueError("labels_paths and feature_manifest_paths must have the same length")
    if summary_json is None:
        summary_json = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    scorer = load_feature_frame_scorer_checkpoint(scorer_checkpoint_path, device=scorer_device)
    scorer_signature = _scorer_signature(scorer)
    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    feature_names: list[str] | None = None
    feature_schema_hash = ""

    with output_jsonl.open("w", encoding="utf-8") as output_handle:
        for labels_path, feature_manifest_path in zip(labels_paths, feature_manifest_paths, strict=True):
            records = _load_light_label_records(labels_path)
            batch: list[tuple[LabelRecord, dict[str, Any], np.ndarray, np.ndarray]] = []
            for manifest_row in _iter_feature_manifest(feature_manifest_path):
                stop = limit is not None and counters["manifest_rows_selected"] >= limit
                if stop:
                    break
                before_len = len(batch)
                _append_manifest_row_to_batch(
                    batch=batch,
                    records=records,
                    manifest_row=manifest_row,
                    feature_manifest_path=feature_manifest_path,
                    scorer=scorer,
                    counters=counters,
                )
                if (
                    len(batch) > before_len
                    and len(batch) > 1
                    and _batch_padded_frames(batch) > int(scorer_max_padded_frames)
                ):
                    overflow_item = batch.pop()
                    feature_names, feature_schema_hash = _flush_batch(
                        batch=batch,
                        output_handle=output_handle,
                        labels_path=labels_path,
                        feature_manifest_path=feature_manifest_path,
                        scorer=scorer,
                        scorer_signature=scorer_signature,
                        config=config,
                        counters=counters,
                        reason_counts=reason_counts,
                        feature_names=feature_names,
                        feature_schema_hash=feature_schema_hash,
                        limit=limit,
                    )
                    batch.append(overflow_item)
                if len(batch) >= max(1, int(scorer_batch_size)):
                    feature_names, feature_schema_hash = _flush_batch(
                        batch=batch,
                        output_handle=output_handle,
                        labels_path=labels_path,
                        feature_manifest_path=feature_manifest_path,
                        scorer=scorer,
                        scorer_signature=scorer_signature,
                        config=config,
                        counters=counters,
                        reason_counts=reason_counts,
                        feature_names=feature_names,
                        feature_schema_hash=feature_schema_hash,
                        limit=limit,
                    )
                    if log_every > 0 and counters["manifest_rows_seen"] % int(log_every) == 0:
                        _print_progress(counters)
                    if limit is not None and counters["manifest_rows_selected"] >= limit:
                        break
            if batch:
                feature_names, feature_schema_hash = _flush_batch(
                    batch=batch,
                    output_handle=output_handle,
                    labels_path=labels_path,
                    feature_manifest_path=feature_manifest_path,
                    scorer=scorer,
                    scorer_signature=scorer_signature,
                    config=config,
                    counters=counters,
                    reason_counts=reason_counts,
                    feature_names=feature_names,
                    feature_schema_hash=feature_schema_hash,
                    limit=limit,
                )
            if limit is not None and counters["manifest_rows_selected"] >= limit:
                break
    summary = {
        "schema": DATASET_SCHEMA,
        "source": DATASET_SOURCE,
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "labels": [str(path) for path in labels_paths],
        "feature_manifests": [str(path) for path in feature_manifest_paths],
        "scorer_checkpoint": scorer_signature,
        "scorer_device": str(scorer_device),
        "scorer_batch_size": int(scorer_batch_size),
        "scorer_max_padded_frames": int(scorer_max_padded_frames),
        "config": asdict(config),
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": feature_schema_hash,
        "feature_names": feature_names or [],
        "feature_dim": len(feature_names or []),
        "counts": dict(counters),
        "label_reasons": dict(reason_counts),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _append_manifest_row_to_batch(
    *,
    batch: list[tuple[LabelRecord, dict[str, Any], np.ndarray, np.ndarray]],
    records: Mapping[int, LabelRecord],
    manifest_row: Mapping[str, Any],
    feature_manifest_path: Path,
    scorer: Any,
    counters: Counter[str],
) -> None:
    counters["manifest_rows_seen"] += 1
    row = dict(manifest_row)
    label_index = int(row.get("label_index") or 0)
    record = records.get(label_index)
    if record is None:
        counters["skipped_bad_label_index"] += 1
        return
    ptm_repo_id = _manifest_ptm_repo_id(row)
    validate_checkpoint_repo_id(
        scorer.metadata.get("ptm_repo_id"),
        ptm_repo_id,
        checkpoint_kind="SpeechBoundary-JA scorer",
        metadata_key="metadata.ptm_repo_id",
    )
    ptm, mfcc = load_cached_feature(_resolve_feature_path(row, feature_manifest_path))
    batch.append((record, row, ptm, mfcc))


def _batch_padded_frames(
    batch: Sequence[tuple[LabelRecord, dict[str, Any], np.ndarray, np.ndarray]],
) -> int:
    if not batch:
        return 0
    max_len = max(min(int(ptm.shape[0]), int(mfcc.shape[0])) for _record, _row, ptm, mfcc in batch)
    return int(max_len) * len(batch)


def _flush_batch(
    *,
    batch: list[tuple[LabelRecord, dict[str, Any], np.ndarray, np.ndarray]],
    output_handle: Any,
    labels_path: Path,
    feature_manifest_path: Path,
    scorer: Any,
    scorer_signature: Mapping[str, Any],
    config: FrameSequenceConfig,
    counters: Counter[str],
    reason_counts: Counter[str],
    feature_names: list[str] | None,
    feature_schema_hash: str,
    limit: int | None = None,
) -> tuple[list[str] | None, str]:
    if not batch:
        return feature_names, feature_schema_hash
    probabilities = score_feature_frame_boundary_probabilities_batch(
        scorer,
        feature_pairs=[(ptm, mfcc) for _record, _row, ptm, mfcc in batch],
    )
    for (record, manifest_row, ptm, mfcc), row_probabilities in zip(batch, probabilities, strict=True):
        if limit is not None and counters["manifest_rows_selected"] >= limit:
            break
        row = _sequence_row(
            record,
            manifest_row=manifest_row,
            labels_path=labels_path,
            feature_manifest_path=feature_manifest_path,
            ptm=ptm,
            mfcc=mfcc,
            scorer_signature=scorer_signature,
            probabilities=row_probabilities,
            config=config,
        )
        if row is None:
            counters["skipped_no_sequence_items"] += 1
            continue
        if feature_names is None:
            feature_names = list(row["feature_names"])
            feature_schema_hash = str(row["feature_schema_hash"])
        elif feature_names != list(row["feature_names"]):
            raise ValueError("feature_names changed across frame sequence rows")
        elif feature_schema_hash != str(row["feature_schema_hash"]):
            raise ValueError("feature_schema_hash changed across frame sequence rows")
        output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        counters["manifest_rows_selected"] += 1
        counters["sequences"] += 1
        counters["sequence_items"] += len(row["sequence_features"])
        counters["start_supervised"] += sum(
            1 for item in row["sequence_boundary_delta_weights"] if float(item[0]) > 0.0
        )
        counters["end_supervised"] += sum(
            1 for item in row["sequence_boundary_delta_weights"] if float(item[1]) > 0.0
        )
        counters["start_target_clipped"] += sum(
            1 for item in row.get("sequence_target_clipped", []) if bool(item[0])
        )
        counters["end_target_clipped"] += sum(
            1 for item in row.get("sequence_target_clipped", []) if bool(item[1])
        )
        for reason in row["sequence_reasons"]:
            reason_counts[str(reason)] += 1
    batch.clear()
    return feature_names, feature_schema_hash


def _print_progress(counters: Counter[str]) -> None:
    print(
        "refiner_dataset_progress="
        f"{counters['manifest_rows_seen']} "
        f"selected={counters['manifest_rows_selected']} "
        f"items={counters['sequence_items']}",
        flush=True,
    )


def _sequence_row(
    record: LabelRecord,
    *,
    manifest_row: Mapping[str, Any],
    labels_path: Path,
    feature_manifest_path: Path,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    scorer_signature: Mapping[str, Any],
    probabilities: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: FrameSequenceConfig,
) -> dict[str, Any] | None:
    true_segments = _record_segments(record)
    if not true_segments:
        return None

    ptm_repo_id = _manifest_ptm_repo_id(manifest_row)
    frame_hop_s = float(manifest_row.get("frame_hop_s") or record.frame_hop_s or config.frame_hop_s)
    speech_probs, split_probs, drop_gap_probs = probabilities
    decoded = decode_frame_boundary_segments(
        speech_probabilities=speech_probs,
        split_probabilities=split_probs,
        drop_gap_probabilities=drop_gap_probs,
        duration_s=record.duration_s,
        config=config.decoder_config(frame_hop_s=frame_hop_s, ptm_repo_id=ptm_repo_id),
    )
    predicted_segments = [
        Segment(start=float(segment.start), end=float(segment.end))
        for segment in decoded.segments
        if float(segment.end) > float(segment.start)
    ]
    if len(predicted_segments) < 2:
        return None

    sequence_features: list[list[float]] = []
    sequence_boundary_delta_targets: list[list[float]] = []
    sequence_boundary_delta_weights: list[list[float]] = []
    sequence_target_clipped: list[list[bool]] = []
    sequence_reasons: list[str] = []
    edge_alignments: list[dict[str, Any]] = []
    gap_indexes: list[int] = []

    for index, (left, right) in enumerate(zip(predicted_segments, predicted_segments[1:])):
        left_true = _align_true_segment(left, true_segments, edge="end", config=config)
        right_true = _align_true_segment(right, true_segments, edge="start", config=config)
        start_weight = 1.0 if right_true is not None else 0.0
        end_weight = 0.6 if left_true is not None else 0.0
        if start_weight <= 0.0 and end_weight <= 0.0:
            continue
        start_delta, start_clipped = _target_delta(
            actual=(right_true.start if right_true is not None else right.start),
            predicted=right.start,
            limit_s=config.boundary_delta_max_s,
        )
        end_delta, end_clipped = _target_delta(
            actual=(left_true.end if left_true is not None else left.end),
            predicted=left.end,
            limit_s=config.boundary_delta_max_s,
        )
        sequence_features.append(
            _candidate_features(
                left=left,
                right=right,
                record=record,
                frame_hop_s=frame_hop_s,
                ptm=ptm,
                mfcc=mfcc,
                config=config,
            )
        )
        sequence_boundary_delta_targets.append([start_delta, end_delta])
        sequence_boundary_delta_weights.append([start_weight, end_weight])
        sequence_target_clipped.append([start_clipped, end_clipped])
        sequence_reasons.append(DATASET_SOURCE)
        gap_indexes.append(index)
        edge_alignments.append(
            {
                "index": index,
                "predicted_left": _segment_payload(left),
                "predicted_right": _segment_payload(right),
                "true_left": None if left_true is None else _segment_payload(left_true),
                "true_right": None if right_true is None else _segment_payload(right_true),
                "target_start_delta_s": start_delta,
                "target_end_delta_s": end_delta,
                "start_weight": start_weight,
                "end_weight": end_weight,
                "start_target_clipped": start_clipped,
                "end_target_clipped": end_clipped,
            }
        )

    if not sequence_features:
        return None
    feature_config = config.feature_config()
    feature_names = frame_sequence_feature_names(
        config=feature_config,
        ptm_dim=min(int(ptm.shape[1]), config.max_ptm_dims),
        mfcc_dim=int(mfcc.shape[1]),
    )
    validate_sequence_features(
        sequence_features,
        feature_names=feature_names,
    )
    schema_hash = feature_extraction_hash(
        config=feature_config,
        feature_names=feature_names,
    )
    return {
        "schema": DATASET_SCHEMA,
        "source": record.source,
        "dataset_source": DATASET_SOURCE,
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": schema_hash,
        "feature_signature": feature_extraction_signature(
            config=feature_config,
            feature_names=feature_names,
        ),
        "audio_id": record.audio_id,
        "label_index": int(manifest_row.get("label_index") or 0),
        "feature_names": feature_names,
        "feature_dim": get_feature_dim(
            config=feature_config,
            ptm_dim=min(int(ptm.shape[1]), config.max_ptm_dims),
            mfcc_dim=int(mfcc.shape[1]),
        ),
        "sequence_features": sequence_features,
        "sequence_boundary_delta_targets": sequence_boundary_delta_targets,
        "sequence_boundary_delta_weights": sequence_boundary_delta_weights,
        "sequence_target_clipped": sequence_target_clipped,
        "sequence_reasons": sequence_reasons,
        "gap_indexes": gap_indexes,
        "predicted_segments": [_segment_payload(segment) for segment in predicted_segments],
        "true_segments": [_segment_payload(segment) for segment in true_segments],
        "edge_alignments": edge_alignments,
        "metadata": {
            "dataset_source": DATASET_SOURCE,
            "labels_path": str(labels_path),
            "feature_manifest_path": str(feature_manifest_path),
            "feature_path": str(manifest_row.get("feature_path") or ""),
            "ptm_repo_id": ptm_repo_id,
            "frame_hop_s": frame_hop_s,
            "frame_count": int(manifest_row.get("frame_count") or ptm.shape[0]),
            "ptm_dim": int(ptm.shape[1]),
            "mfcc_dim": int(mfcc.shape[1]),
            "ptm_used_dim": min(int(ptm.shape[1]), config.max_ptm_dims),
            "scorer_checkpoint": dict(scorer_signature),
            "decoder": {
                "speech_on_threshold": float(decoded.speech_on_threshold),
                "speech_off_threshold": float(decoded.speech_off_threshold),
                "frame_dilation_s": float(config.frame_dilation_s),
                "drop_gap_threshold": float(config.drop_gap_threshold),
                "split_strategy": "adaptive_topk_peak",
                "split_target_s": float(config.split_target_s),
                "split_score_quantile": float(config.split_score_quantile),
                "split_prominence_quantile": float(config.split_prominence_quantile),
                "split_smooth_s": float(config.split_smooth_s),
                "split_nms_s": float(config.split_nms_s),
                "split_snap_s": float(config.split_snap_s),
                "min_segment_s": float(config.min_segment_s),
                "min_split_segment_s": float(config.min_split_segment_s),
                "boundary_delta_max_s": float(config.boundary_delta_max_s),
            },
        },
    }


def _candidate_features(
    *,
    left: Segment,
    right: Segment,
    record: LabelRecord,
    frame_hop_s: float,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    config: FrameSequenceConfig,
) -> list[float]:
    return boundary_window_sequence_features(
        left_start_s=left.start,
        left_end_s=left.end,
        right_start_s=right.start,
        right_end_s=right.end,
        duration_s=record.duration_s,
        frame_hop_s=frame_hop_s,
        ptm=ptm,
        mfcc=mfcc,
        config=config.feature_config(),
    )


def _align_true_segment(
    predicted: Segment,
    true_segments: Sequence[Segment],
    *,
    edge: str,
    config: FrameSequenceConfig,
) -> Segment | None:
    if edge not in {"start", "end"}:
        raise ValueError("edge must be 'start' or 'end'")
    candidates: list[tuple[int, float, float, Segment]] = []
    min_overlap = max(0.0, float(config.min_edge_overlap_s))
    max_distance = max(0.0, float(config.max_edge_alignment_distance_s))
    for true in true_segments:
        overlap = _overlap_s(predicted, true)
        edge_distance = (
            abs(float(true.start) - float(predicted.start))
            if edge == "start"
            else abs(float(true.end) - float(predicted.end))
        )
        center_distance = abs(_center_s(true) - _center_s(predicted))
        has_overlap = overlap >= max(1e-6, min_overlap)
        if not has_overlap and edge_distance > max_distance:
            continue
        candidates.append((0 if has_overlap else 1, edge_distance, center_distance, true))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


def _target_delta(*, actual: float, predicted: float, limit_s: float) -> tuple[float, bool]:
    limit = max(0.0, float(limit_s))
    raw = float(actual) - float(predicted)
    clipped = max(-limit, min(limit, raw))
    return round(clipped, 6), abs(clipped - raw) > 1e-9


def _overlap_s(left: Segment, right: Segment) -> float:
    return max(0.0, min(float(left.end), float(right.end)) - max(float(left.start), float(right.start)))


def _center_s(segment: Segment) -> float:
    return (float(segment.start) + float(segment.end)) / 2.0


def _segment_payload(segment: Segment) -> dict[str, float]:
    return {"start": round(float(segment.start), 6), "end": round(float(segment.end), 6)}


def _record_segments(record: LabelRecord) -> list[Segment]:
    metadata = record.boundary_metadata or {}
    actual = metadata.get("actual_speech_segments")
    if isinstance(actual, list) and actual:
        return _normalize_segments(actual, duration_s=record.duration_s)
    supervised = record.teacher_segments.get("supervised")
    if supervised:
        return _normalize_segments(supervised, duration_s=record.duration_s)
    all_segments: list[TeacherSegment] = []
    for segments in record.teacher_segments.values():
        all_segments.extend(segments)
    return _normalize_segments(all_segments, duration_s=record.duration_s)


def _normalize_segments(
    values: Iterable[Mapping[str, Any] | TeacherSegment],
    *,
    duration_s: float,
) -> list[Segment]:
    segments: list[Segment] = []
    for item in values:
        if isinstance(item, TeacherSegment):
            raw_start, raw_end = item.start, item.end
        else:
            raw_start, raw_end = item.get("start", 0.0), item.get("end", 0.0)
        start = max(0.0, min(float(raw_start), duration_s))
        end = max(0.0, min(float(raw_end), duration_s))
        if end > start:
            segments.append(Segment(start=start, end=end))
    return sorted(segments, key=lambda item: (item.start, item.end))


def _load_light_label_records(path: Path) -> dict[int, LabelRecord]:
    records: dict[int, LabelRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for label_index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            records[label_index] = LabelRecord(
                audio_id=str(payload.get("audio_id") or ""),
                source=str(payload.get("source") or ""),
                duration_s=float(payload.get("duration_s") or 0.0),
                text=str(payload.get("text") or ""),
                teacher_segments=_light_teacher_segments(payload.get("teacher_segments")),
                frame_hop_s=float(payload.get("frame_hop_s") or 0.02),
                speech_frames=[],
                label_quality=str(payload.get("label_quality") or ""),
                frame_weights=None,
                boundary_metadata=(
                    None
                    if payload.get("boundary_metadata") is None
                    else dict(payload.get("boundary_metadata") or {})
                ),
            )
    return records


def _light_teacher_segments(raw: Any) -> dict[str, list[TeacherSegment]]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, list[TeacherSegment]] = {}
    for name, values in raw.items():
        segments = []
        for item in list(values or []):
            if not isinstance(item, Mapping):
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or 0.0)
            if end <= start:
                continue
            segments.append(
                TeacherSegment(
                    start=start,
                    end=end,
                    score=None if item.get("score") is None else float(item.get("score")),
                )
            )
        result[str(name)] = segments
    return result


def _manifest_ptm_repo_id(row: Mapping[str, Any]) -> str:
    ptm = str(row.get("ptm") or "").strip()
    if not ptm:
        raise ValueError("feature manifest row missing ptm repo id")
    return ptm


def _scorer_signature(scorer: Any) -> dict[str, Any]:
    signature = scorer.signature() if callable(getattr(scorer, "signature", None)) else {}
    if not isinstance(signature, Mapping):
        signature = {}
    return dict(signature)


def _iter_feature_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f"feature manifest must be a JSON list or JSONL: {path}")
            for row in rows:
                if isinstance(row, Mapping):
                    yield dict(row)
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"feature manifest JSONL row must be an object: {path}:{line_number}")
            yield dict(row)


def _resolve_feature_path(row: Mapping[str, Any], manifest_path: Path) -> Path:
    raw_path = str(row.get("feature_path") or "")
    if not raw_path:
        raise ValueError(f"feature manifest row missing feature_path: {manifest_path}")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = (manifest_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scorer-driven edge-only sequence rows for Boundary Refiner v6."
    )
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--feature-manifest", action="append", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-batch-size", type=int, default=1)
    parser.add_argument("--scorer-max-padded-frames", type=int, default=4096)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--left-context-s", type=float, default=0.60)
    parser.add_argument("--right-context-s", type=float, default=0.60)
    parser.add_argument("--max-ptm-dims", type=int, default=64)
    parser.add_argument("--no-mfcc", action="store_true")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=0.200)
    parser.add_argument("--speech-on-threshold", type=float, default=None)
    parser.add_argument("--speech-off-threshold", type=float, default=None)
    parser.add_argument("--frame-dilation-s", type=float, default=0.2)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument("--drop-gap-threshold", type=float, default=0.75)
    parser.add_argument("--split-target-s", type=float, default=5.0)
    parser.add_argument("--split-score-quantile", type=float, default=0.50)
    parser.add_argument("--split-prominence-quantile", type=float, default=0.50)
    parser.add_argument("--split-smooth-s", type=float, default=0.08)
    parser.add_argument("--split-nms-s", type=float, default=0.20)
    parser.add_argument("--split-snap-s", type=float, default=0.10)
    parser.add_argument("--min-split-segment-s", type=float, default=0.08)
    parser.add_argument("--boundary-delta-max-s", type=float, default=0.5)
    parser.add_argument("--max-edge-alignment-distance-s", type=float, default=0.5)
    parser.add_argument("--min-edge-overlap-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    if len(args.labels) != len(args.feature_manifest):
        parser.error("--labels and --feature-manifest must be provided the same number of times")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.scorer_batch_size <= 0:
        parser.error("--scorer-batch-size must be positive")
    if args.scorer_max_padded_frames <= 0:
        parser.error("--scorer-max-padded-frames must be positive")
    if args.left_context_s <= 0.0 or args.right_context_s <= 0.0:
        parser.error("context seconds must be positive")
    if args.max_ptm_dims <= 0:
        parser.error("--max-ptm-dims must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.frame_dilation_s < 0.0:
        parser.error("--frame-dilation-s must be non-negative")
    if args.min_segment_s < 0.0 or args.min_split_segment_s < 0.0:
        parser.error("minimum segment durations must be non-negative")
    if args.split_target_s < 0.0:
        parser.error("--split-target-s must be non-negative")
    for name in ("split_score_quantile", "split_prominence_quantile"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.split_smooth_s < 0.0 or args.split_nms_s < 0.0 or args.split_snap_s < 0.0:
        parser.error("split smooth/nms/snap seconds must be non-negative")
    if args.boundary_delta_max_s <= 0.0:
        parser.error("--boundary-delta-max-s must be positive")
    if args.max_edge_alignment_distance_s < 0.0:
        parser.error("--max-edge-alignment-distance-s must be non-negative")
    if args.min_edge_overlap_s < 0.0:
        parser.error("--min-edge-overlap-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frame_sequence_dataset(
        labels_paths=[Path(path) for path in args.labels],
        feature_manifest_paths=[Path(path) for path in args.feature_manifest],
        scorer_checkpoint_path=Path(args.scorer_checkpoint),
        scorer_device=args.scorer_device,
        scorer_batch_size=args.scorer_batch_size,
        scorer_max_padded_frames=args.scorer_max_padded_frames,
        output_jsonl=Path(args.output_jsonl),
        summary_json=Path(args.summary_json) if args.summary_json else None,
        limit=args.limit,
        log_every=args.log_every,
        config=FrameSequenceConfig(
            left_context_s=args.left_context_s,
            right_context_s=args.right_context_s,
            max_ptm_dims=args.max_ptm_dims,
            include_mfcc=not args.no_mfcc,
            frame_hop_s=args.frame_hop_s,
            threshold=args.threshold,
            speech_on_threshold=args.speech_on_threshold,
            speech_off_threshold=args.speech_off_threshold,
            frame_dilation_s=args.frame_dilation_s,
            min_segment_s=args.min_segment_s,
            drop_gap_threshold=args.drop_gap_threshold,
            split_target_s=args.split_target_s,
            split_score_quantile=args.split_score_quantile,
            split_prominence_quantile=args.split_prominence_quantile,
            split_smooth_s=args.split_smooth_s,
            split_nms_s=args.split_nms_s,
            split_snap_s=args.split_snap_s,
            min_split_segment_s=args.min_split_segment_s,
            boundary_delta_max_s=args.boundary_delta_max_s,
            max_edge_alignment_distance_s=args.max_edge_alignment_distance_s,
            min_edge_overlap_s=args.min_edge_overlap_s,
        ),
    )
    print(f"dataset={summary['output_jsonl']}")
    print(f"summary={summary['summary_json']}")
    print(f"counts={json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
