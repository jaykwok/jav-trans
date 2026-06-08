#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    FrameSequenceFeatureConfig,
    feature_extraction_hash,
    feature_extraction_signature,
    frame_sequence_feature_names,
    gap_window_sequence_features,
    get_feature_dim,
    validate_sequence_features,
)
from boundary.ja import LabelRecord, TeacherSegment, load_cached_feature

DATASET_SCHEMA = "boundary_refiner_frame_sequence_dataset_v2"


@dataclass(frozen=True)
class FrameSequenceConfig:
    left_context_s: float = 0.60
    right_context_s: float = 0.60
    max_ptm_dims: int = 64
    include_mfcc: bool = True
    target_chunk_s: float = 3.0
    safe_merge_gap_s: float = 0.12
    long_gap_split_s: float = 0.60
    synthetic_merge_positives_per_record: int = 1
    synthetic_merge_gap_min_s: float = 0.04
    synthetic_merge_gap_s: float = 0.04
    synthetic_merge_min_segment_s: float = 1.20
    context_max_padding_s: float = 1.5
    context_max_speech_overlap_s: float = 0.25

    def feature_config(self) -> FrameSequenceFeatureConfig:
        return FrameSequenceFeatureConfig(
            left_context_s=self.left_context_s,
            right_context_s=self.right_context_s,
            max_ptm_dims=self.max_ptm_dims,
            include_mfcc=self.include_mfcc,
            target_chunk_s=self.target_chunk_s,
        )


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


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


def build_frame_sequence_dataset(
    *,
    labels_paths: Sequence[Path],
    feature_manifest_paths: Sequence[Path],
    output_jsonl: Path,
    summary_json: Path | None = None,
    config: FrameSequenceConfig = FrameSequenceConfig(),
    limit: int | None = None,
) -> dict[str, Any]:
    if len(labels_paths) != len(feature_manifest_paths):
        raise ValueError("labels_paths and feature_manifest_paths must have the same length")
    if summary_json is None:
        summary_json = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    feature_names: list[str] | None = None
    feature_schema_hash = ""

    with output_jsonl.open("w", encoding="utf-8") as output_handle:
        for labels_path, feature_manifest_path in zip(labels_paths, feature_manifest_paths, strict=True):
            records = _load_light_label_records(labels_path)
            for manifest_row in _iter_feature_manifest(feature_manifest_path):
                if limit is not None and counters["manifest_rows_selected"] >= limit:
                    break
                label_index = int(manifest_row.get("label_index") or 0)
                record = records.get(label_index)
                if record is None:
                    counters["skipped_bad_label_index"] += 1
                    continue
                ptm, mfcc = load_cached_feature(_resolve_feature_path(manifest_row, feature_manifest_path))
                row = _sequence_row(
                    record,
                    manifest_row=manifest_row,
                    labels_path=labels_path,
                    feature_manifest_path=feature_manifest_path,
                    ptm=ptm,
                    mfcc=mfcc,
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
                labels = [int(value) for value in row["sequence_labels"]]
                counters["sequence_items"] += len(labels)
                counters["merge_positive"] += sum(labels)
                counters["split_negative"] += len(labels) - sum(labels)
                for reason in row["sequence_reasons"]:
                    reason_counts[str(reason)] += 1
            if limit is not None and counters["manifest_rows_selected"] >= limit:
                break
    summary = {
        "schema": DATASET_SCHEMA,
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "labels": [str(path) for path in labels_paths],
        "feature_manifests": [str(path) for path in feature_manifest_paths],
        "config": asdict(config),
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": feature_schema_hash,
        "feature_names": feature_names or [],
        "feature_dim": len(feature_names or []),
        "counts": dict(counters),
        "label_reasons": dict(reason_counts),
        "class_balance": {
            "merge_positive": int(counters["merge_positive"]),
            "split_negative": int(counters["split_negative"]),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _sequence_row(
    record: LabelRecord,
    *,
    manifest_row: Mapping[str, Any],
    labels_path: Path,
    feature_manifest_path: Path,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    config: FrameSequenceConfig,
) -> dict[str, Any] | None:
    segments = _record_segments(record)
    if not segments and config.synthetic_merge_positives_per_record <= 0:
        return None
    sequence_features: list[list[float]] = []
    sequence_labels: list[int] = []
    sequence_context_targets: list[list[float]] = []
    sequence_reasons: list[str] = []
    gap_indexes: list[int] = []

    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        label = _label_gap(index=index, left=left, right=right, record=record, config=config)
        if label is None:
            continue
        merge_target, reason = label
        sequence_features.append(
            _candidate_features(
                left=left,
                right=right,
                record=record,
                ptm=ptm,
                mfcc=mfcc,
                config=config,
            )
        )
        sequence_labels.append(1 if merge_target else 0)
        sequence_context_targets.append(
            _context_targets(
                left=left,
                right=right,
                merge_target=merge_target,
                reason=reason,
                config=config,
            )
        )
        sequence_reasons.append(reason)
        gap_indexes.append(index)

    synthetic_count = 0
    for index, segment in enumerate(segments):
        if synthetic_count >= config.synthetic_merge_positives_per_record:
            break
        gap_s = _synthetic_merge_gap_s(
            audio_id=record.audio_id,
            segment_index=index,
            positive_index=synthetic_count,
            config=config,
        )
        if segment.duration_s < config.synthetic_merge_min_segment_s + gap_s:
            continue
        midpoint = (segment.start + segment.end) / 2.0
        left = Segment(segment.start, midpoint - gap_s / 2.0)
        right = Segment(midpoint + gap_s / 2.0, segment.end)
        if left.end <= left.start or right.end <= right.start:
            continue
        sequence_features.append(
            _candidate_features(
                left=left,
                right=right,
                record=record,
                ptm=ptm,
                mfcc=mfcc,
                config=config,
            )
        )
        sequence_labels.append(1)
        sequence_context_targets.append([0.0, 0.0])
        sequence_reasons.append("merge_synthetic_intra_island")
        gap_indexes.append(index)
        synthetic_count += 1

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
        labels=sequence_labels,
        feature_names=feature_names,
    )
    schema_hash = feature_extraction_hash(
        config=feature_config,
        feature_names=feature_names,
    )
    return {
        "schema": DATASET_SCHEMA,
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": schema_hash,
        "feature_signature": feature_extraction_signature(
            config=feature_config,
            feature_names=feature_names,
        ),
        "audio_id": record.audio_id,
        "source": record.source,
        "label_index": int(manifest_row.get("label_index") or 0),
        "feature_names": feature_names,
        "feature_dim": get_feature_dim(
            config=feature_config,
            ptm_dim=min(int(ptm.shape[1]), config.max_ptm_dims),
            mfcc_dim=int(mfcc.shape[1]),
        ),
        "sequence_features": sequence_features,
        "sequence_labels": sequence_labels,
        "sequence_context_targets": sequence_context_targets,
        "sequence_reasons": sequence_reasons,
        "gap_indexes": gap_indexes,
        "metadata": {
            "labels_path": str(labels_path),
            "feature_manifest_path": str(feature_manifest_path),
            "feature_path": str(manifest_row.get("feature_path") or ""),
            "frame_hop_s": float(manifest_row.get("frame_hop_s") or record.frame_hop_s),
            "frame_count": int(manifest_row.get("frame_count") or ptm.shape[0]),
            "ptm_dim": int(ptm.shape[1]),
            "mfcc_dim": int(mfcc.shape[1]),
            "ptm_used_dim": min(int(ptm.shape[1]), config.max_ptm_dims),
        },
    }


def _candidate_features(
    *,
    left: Segment,
    right: Segment,
    record: LabelRecord,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    config: FrameSequenceConfig,
) -> list[float]:
    return gap_window_sequence_features(
        left_start_s=left.start,
        left_end_s=left.end,
        right_start_s=right.start,
        right_end_s=right.end,
        duration_s=record.duration_s,
        frame_hop_s=record.frame_hop_s,
        ptm=ptm,
        mfcc=mfcc,
        config=config.feature_config(),
    )


def _label_gap(
    *,
    index: int,
    left: Segment,
    right: Segment,
    record: LabelRecord,
    config: FrameSequenceConfig,
) -> tuple[bool, str] | None:
    gap_s = right.start - left.end
    if gap_s < 0.0:
        return False, "split_overlap"
    metadata = record.boundary_metadata or {}
    boundaries = {
        int(item.get("index")): dict(item)
        for item in list(metadata.get("utterance_boundaries") or [])
        if str(item.get("index", "")).lstrip("-").isdigit()
    }
    boundary = boundaries.get(index)
    if boundary:
        boundary_type = str(boundary.get("boundary_type") or "")
        if boundary_type == "gap_zone":
            return False, "split_gap_zone"
        if boundary_type == "cut_point":
            return False, "split_cut_point"
    if gap_s >= config.long_gap_split_s:
        return False, "split_long_gap"

    source_ids = [str(value) for value in list(metadata.get("source_audio_ids") or [])]
    previous_source = source_ids[index] if index < len(source_ids) else ""
    next_source = source_ids[index + 1] if index + 1 < len(source_ids) else ""
    same_source = bool(previous_source and previous_source == next_source)
    if gap_s <= config.safe_merge_gap_s and same_source:
        return True, "merge_same_source_short_gap"
    return None


def _context_targets(
    *,
    left: Segment,
    right: Segment,
    merge_target: bool,
    reason: str,
    config: FrameSequenceConfig,
) -> list[float]:
    if merge_target:
        return [0.0, 0.0]
    gap_s = max(0.0, right.start - left.end)
    max_padding = max(0.0, float(config.context_max_padding_s))
    overlap_cap = max(0.0, min(float(config.context_max_speech_overlap_s), max_padding))
    if reason in {"split_overlap", "split_cut_point"}:
        budget = min(overlap_cap, gap_s * 0.45 if gap_s > 0.0 else overlap_cap)
    elif reason == "split_gap_zone":
        budget = min(max_padding, gap_s * 0.35)
    elif reason == "split_long_gap":
        budget = min(max_padding, gap_s * 0.45)
    else:
        budget = min(max_padding, gap_s * 0.40)
    budget = max(0.0, budget)
    return [budget, budget]


def _synthetic_merge_gap_s(
    *,
    audio_id: str,
    segment_index: int,
    positive_index: int,
    config: FrameSequenceConfig,
) -> float:
    lower = max(0.0, min(config.synthetic_merge_gap_min_s, config.synthetic_merge_gap_s))
    upper = max(0.0, max(config.synthetic_merge_gap_min_s, config.synthetic_merge_gap_s))
    if upper <= lower:
        return upper
    digest = hashlib.sha1(f"{audio_id}:{segment_index}:{positive_index}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return lower + (upper - lower) * fraction


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature-window sequence rows for the Boundary Refiner."
    )
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--feature-manifest", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--left-context-s", type=float, default=0.60)
    parser.add_argument("--right-context-s", type=float, default=0.60)
    parser.add_argument("--max-ptm-dims", type=int, default=64)
    parser.add_argument("--no-mfcc", action="store_true")
    parser.add_argument("--target-chunk-s", type=float, default=3.0)
    parser.add_argument("--safe-merge-gap-s", type=float, default=0.12)
    parser.add_argument("--long-gap-split-s", type=float, default=0.60)
    parser.add_argument("--synthetic-merge-positives-per-record", type=int, default=1)
    parser.add_argument("--synthetic-merge-gap-min-s", type=float, default=0.04)
    parser.add_argument("--synthetic-merge-gap-s", type=float, default=0.04)
    parser.add_argument("--synthetic-merge-min-segment-s", type=float, default=1.20)
    parser.add_argument("--context-max-padding-s", type=float, default=1.5)
    parser.add_argument("--context-max-speech-overlap-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    if len(args.labels) != len(args.feature_manifest):
        parser.error("--labels and --feature-manifest must be provided the same number of times")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.left_context_s <= 0.0 or args.right_context_s <= 0.0:
        parser.error("context seconds must be positive")
    if args.max_ptm_dims <= 0:
        parser.error("--max-ptm-dims must be positive")
    if args.target_chunk_s <= 0.0:
        parser.error("--target-chunk-s must be positive")
    if args.safe_merge_gap_s < 0.0 or args.long_gap_split_s < 0.0:
        parser.error("gap thresholds must be non-negative")
    if args.synthetic_merge_positives_per_record < 0:
        parser.error("--synthetic-merge-positives-per-record must be non-negative")
    if args.synthetic_merge_gap_min_s < 0.0:
        parser.error("--synthetic-merge-gap-min-s must be non-negative")
    if args.synthetic_merge_gap_s < 0.0:
        parser.error("--synthetic-merge-gap-s must be non-negative")
    if args.context_max_padding_s <= 0.0:
        parser.error("--context-max-padding-s must be positive")
    if args.context_max_speech_overlap_s < 0.0:
        parser.error("--context-max-speech-overlap-s must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frame_sequence_dataset(
        labels_paths=[Path(path) for path in args.labels],
        feature_manifest_paths=[Path(path) for path in args.feature_manifest],
        output_jsonl=Path(args.output_jsonl),
        summary_json=Path(args.summary_json) if args.summary_json else None,
        limit=args.limit,
        config=FrameSequenceConfig(
            left_context_s=args.left_context_s,
            right_context_s=args.right_context_s,
            max_ptm_dims=args.max_ptm_dims,
            include_mfcc=not args.no_mfcc,
            target_chunk_s=args.target_chunk_s,
            safe_merge_gap_s=args.safe_merge_gap_s,
            long_gap_split_s=args.long_gap_split_s,
            synthetic_merge_positives_per_record=args.synthetic_merge_positives_per_record,
            synthetic_merge_gap_min_s=args.synthetic_merge_gap_min_s,
            synthetic_merge_gap_s=args.synthetic_merge_gap_s,
            synthetic_merge_min_segment_s=args.synthetic_merge_min_segment_s,
            context_max_padding_s=args.context_max_padding_s,
            context_max_speech_overlap_s=args.context_max_speech_overlap_s,
        ),
    )
    print(f"dataset={summary['output_jsonl']}")
    print(f"summary={summary['summary_json']}")
    print(f"counts={json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"class_balance={json.dumps(summary['class_balance'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
