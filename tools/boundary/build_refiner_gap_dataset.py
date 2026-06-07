#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.refiner import DEFAULT_REFINER_FEATURES, RefinerInput, refiner_input_to_features
from boundary.ja import LabelRecord, TeacherSegment, load_label_records

DATASET_SCHEMA = "boundary_refiner_gap_dataset_v1"


@dataclass(frozen=True)
class GapDatasetConfig:
    target_chunk_s: float = 3.0
    safe_merge_gap_s: float = 0.12
    long_gap_split_s: float = 0.60
    synthetic_merge_positives_per_record: int = 0
    synthetic_merge_gap_s: float = 0.04
    synthetic_merge_min_segment_s: float = 1.20
    merge_unspecified_short_gaps: bool = False

    @property
    def gap_merge_s(self) -> float:
        return max(0.2, min(1.5, self.target_chunk_s / 6.0))


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    score: float | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


def build_gap_dataset(
    *,
    labels_paths: Sequence[Path],
    feature_manifest_paths: Sequence[Path],
    output_jsonl: Path,
    summary_json: Path | None = None,
    output_sequence_jsonl: Path | None = None,
    config: GapDatasetConfig = GapDatasetConfig(),
    limit: int | None = None,
) -> dict[str, Any]:
    if len(labels_paths) != len(feature_manifest_paths):
        raise ValueError("labels_paths and feature_manifest_paths must have the same length")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if summary_json is None:
        summary_json = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    feature_dim_counts: Counter[str] = Counter()

    for labels_path, feature_manifest_path in zip(labels_paths, feature_manifest_paths, strict=True):
        records = load_label_records(labels_path)
        manifest_rows = _load_feature_manifest(feature_manifest_path)
        for manifest_row in manifest_rows:
            if limit is not None and counters["manifest_rows_selected"] >= limit:
                break
            label_index = int(manifest_row.get("label_index") or 0)
            if label_index < 0 or label_index >= len(records):
                counters["skipped_bad_label_index"] += 1
                continue
            record = records[label_index]
            counters["manifest_rows_selected"] += 1
            ptm_dim = _required_positive_int(manifest_row, "ptm_dim", feature_manifest_path)
            mfcc_dim = int(manifest_row.get("mfcc_dim") or 0)
            if ptm_dim > 0:
                feature_dim_counts[f"ptm_dim:{ptm_dim}"] += 1
            if mfcc_dim > 0:
                feature_dim_counts[f"mfcc_dim:{mfcc_dim}"] += 1
            for row in _rows_for_record(
                record,
                manifest_row=manifest_row,
                labels_path=labels_path,
                feature_manifest_path=feature_manifest_path,
                config=config,
            ):
                rows.append(row)
                counters["samples"] += 1
                counters["merge_positive" if row["merge_target"] else "split_negative"] += 1
                reason_counts[str(row["label_reason"])] += 1
        if limit is not None and counters["manifest_rows_selected"] >= limit:
            break

    _write_jsonl(output_jsonl, rows)
    sequence_rows = _sequence_rows(rows)
    if output_sequence_jsonl is not None:
        _write_jsonl(output_sequence_jsonl, sequence_rows)
    summary = {
        "schema": DATASET_SCHEMA,
        "output_jsonl": str(output_jsonl),
        "output_sequence_jsonl": str(output_sequence_jsonl) if output_sequence_jsonl else "",
        "summary_json": str(summary_json),
        "labels": [str(path) for path in labels_paths],
        "feature_manifests": [str(path) for path in feature_manifest_paths],
        "config": asdict(config),
        "feature_names": list(DEFAULT_REFINER_FEATURES),
        "counts": dict(counters),
        "label_reasons": dict(reason_counts),
        "feature_dims": dict(feature_dim_counts),
        "class_balance": {
            "merge_positive": int(counters["merge_positive"]),
            "split_negative": int(counters["split_negative"]),
        },
        "sequence_counts": {
            "sequences": len(sequence_rows),
            "sequence_items": sum(len(row["sequence_labels"]) for row in sequence_rows),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _rows_for_record(
    record: LabelRecord,
    *,
    manifest_row: Mapping[str, Any],
    labels_path: Path,
    feature_manifest_path: Path,
    config: GapDatasetConfig,
) -> list[dict[str, Any]]:
    segments = _record_segments(record)
    if len(segments) < 2 and config.synthetic_merge_positives_per_record <= 0:
        return []

    rows: list[dict[str, Any]] = []
    metadata = record.boundary_metadata or {}
    source_ids = [str(value) for value in list(metadata.get("source_audio_ids") or [])]
    boundaries = {
        int(item.get("index")): dict(item)
        for item in list(metadata.get("utterance_boundaries") or [])
        if str(item.get("index", "")).lstrip("-").isdigit()
    }

    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        gap_s = right.start - left.end
        label = _label_gap(
            index=index,
            gap_s=gap_s,
            boundary=boundaries.get(index),
            source_ids=source_ids,
            config=config,
        )
        if label is None:
            continue
        merge_target, reason = label
        rows.append(
            _sample_row(
                record,
                manifest_row=manifest_row,
                labels_path=labels_path,
                feature_manifest_path=feature_manifest_path,
                gap_index=index,
                left=left,
                right=right,
                merge_target=merge_target,
                label_reason=reason,
                config=config,
                synthetic=False,
            )
        )

    rows.extend(
        _synthetic_merge_rows(
            record,
            manifest_row=manifest_row,
            labels_path=labels_path,
            feature_manifest_path=feature_manifest_path,
            segments=segments,
            config=config,
        )
    )
    return rows


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
            raw_start = item.start
            raw_end = item.end
            score = item.score
        else:
            raw_start = item.get("start", 0.0)
            raw_end = item.get("end", 0.0)
            raw_score = item.get("score")
            score = None if raw_score is None else float(raw_score)
        start = max(0.0, min(float(raw_start), duration_s))
        end = max(0.0, min(float(raw_end), duration_s))
        if end > start:
            segments.append(Segment(start=start, end=end, score=score))
    return sorted(segments, key=lambda item: (item.start, item.end))


def _label_gap(
    *,
    index: int,
    gap_s: float,
    boundary: Mapping[str, Any] | None,
    source_ids: Sequence[str],
    config: GapDatasetConfig,
) -> tuple[bool, str] | None:
    if gap_s < 0.0:
        return (False, "split_overlap")
    if boundary:
        boundary_type = str(boundary.get("boundary_type") or "")
        if boundary_type == "gap_zone":
            return (False, "split_gap_zone")
        if boundary_type == "cut_point":
            return (False, "split_cut_point")
    if gap_s >= config.long_gap_split_s:
        return (False, "split_long_gap")

    previous_source = source_ids[index] if index < len(source_ids) else ""
    next_source = source_ids[index + 1] if index + 1 < len(source_ids) else ""
    same_source = bool(previous_source and previous_source == next_source)
    if gap_s <= config.safe_merge_gap_s and same_source:
        return (True, "merge_same_source_short_gap")
    if config.merge_unspecified_short_gaps and gap_s <= config.safe_merge_gap_s:
        return (True, "merge_unspecified_short_gap")
    return None


def _synthetic_merge_rows(
    record: LabelRecord,
    *,
    manifest_row: Mapping[str, Any],
    labels_path: Path,
    feature_manifest_path: Path,
    segments: Sequence[Segment],
    config: GapDatasetConfig,
) -> list[dict[str, Any]]:
    count = max(0, int(config.synthetic_merge_positives_per_record))
    if count <= 0:
        return []
    rows: list[dict[str, Any]] = []
    gap_s = max(0.0, float(config.synthetic_merge_gap_s))
    for index, segment in enumerate(segments):
        if len(rows) >= count:
            break
        if segment.duration_s < config.synthetic_merge_min_segment_s + gap_s:
            continue
        midpoint = (segment.start + segment.end) / 2.0
        left = Segment(start=segment.start, end=midpoint - gap_s / 2.0, score=segment.score)
        right = Segment(start=midpoint + gap_s / 2.0, end=segment.end, score=segment.score)
        if left.end <= left.start or right.end <= right.start:
            continue
        rows.append(
            _sample_row(
                record,
                manifest_row=manifest_row,
                labels_path=labels_path,
                feature_manifest_path=feature_manifest_path,
                gap_index=index,
                left=left,
                right=right,
                merge_target=True,
                label_reason="merge_synthetic_intra_island",
                config=config,
                synthetic=True,
            )
        )
    return rows


def _sample_row(
    record: LabelRecord,
    *,
    manifest_row: Mapping[str, Any],
    labels_path: Path,
    feature_manifest_path: Path,
    gap_index: int,
    left: Segment,
    right: Segment,
    merge_target: bool,
    label_reason: str,
    config: GapDatasetConfig,
    synthetic: bool,
) -> dict[str, Any]:
    gap_s = right.start - left.end
    gap_boundary_score = max(0.0, min(1.0, gap_s / max(config.gap_merge_s, 1e-6)))
    item = RefinerInput(
        gap_s=gap_s,
        left_start=left.start,
        left_end=left.end,
        right_start=right.start,
        right_end=right.end,
        current_core_s=left.duration_s,
        proposed_core_s=max(0.0, right.end - left.start),
        gap_merge_s=config.gap_merge_s,
        left_score=1.0 if left.score is None else left.score,
        right_score=1.0 if right.score is None else right.score,
        gap_boundary_score=gap_boundary_score,
    )
    feature_names = DEFAULT_REFINER_FEATURES
    return {
        "schema": DATASET_SCHEMA,
        "audio_id": record.audio_id,
        "source": record.source,
        "label_index": int(manifest_row.get("label_index") or 0),
        "gap_index": int(gap_index),
        "merge_target": bool(merge_target),
        "label": 1 if merge_target else 0,
        "label_reason": label_reason,
        "synthetic": bool(synthetic),
        "feature_names": list(feature_names),
        "features": refiner_input_to_features(item, feature_names),
        "refiner_input": asdict(item),
        "metadata": {
            "labels_path": str(labels_path),
            "feature_manifest_path": str(feature_manifest_path),
            "feature_path": str(manifest_row.get("feature_path") or ""),
            "ptm": str(manifest_row.get("ptm") or ""),
            "ptm_dim": _required_positive_int(manifest_row, "ptm_dim", feature_manifest_path),
            "mfcc_dim": int(manifest_row.get("mfcc_dim") or 0),
            "frame_count": int(manifest_row.get("frame_count") or 0),
            "frame_hop_s": float(manifest_row.get("frame_hop_s") or record.frame_hop_s),
        },
    }


def _load_feature_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"feature manifest must be a JSON list: {path}")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        _required_positive_int(item, "ptm_dim", path)
        normalized.append(item)
    return normalized


def _required_positive_int(row: Mapping[str, Any], key: str, path: Path) -> int:
    try:
        value = int(row.get(key) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature manifest row has invalid {key}: {path}") from exc
    if value <= 0:
        audio_id = str(row.get("audio_id") or row.get("label_index") or "")
        suffix = f" row={audio_id}" if audio_id else ""
        raise ValueError(f"feature manifest row is missing positive {key}: {path}{suffix}")
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sequence_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("audio_id") or ""),
            int(row.get("label_index") or 0),
            str(row.get("metadata", {}).get("feature_path") or ""),
        )
        grouped.setdefault(key, []).append(row)

    sequence_rows: list[dict[str, Any]] = []
    for (audio_id, label_index, feature_path), group in sorted(grouped.items()):
        sorted_group = sorted(group, key=lambda item: int(item.get("gap_index") or 0))
        feature_names = list(sorted_group[0].get("feature_names") or DEFAULT_REFINER_FEATURES)
        sequence_rows.append(
            {
                "schema": "boundary_refiner_sequence_dataset_v1",
                "audio_id": audio_id,
                "source": str(sorted_group[0].get("source") or ""),
                "label_index": label_index,
                "feature_names": feature_names,
                "sequence_features": [list(item["features"]) for item in sorted_group],
                "sequence_labels": [int(item.get("label", item.get("merge_target", 0))) for item in sorted_group],
                "sequence_reasons": [str(item.get("label_reason") or "") for item in sorted_group],
                "gap_indexes": [int(item.get("gap_index") or 0) for item in sorted_group],
                "metadata": {
                    "feature_path": feature_path,
                    "item_count": len(sorted_group),
                    "source_schema": DATASET_SCHEMA,
                },
            }
        )
    return sequence_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build supervised gap samples for the Boundary Refiner."
    )
    parser.add_argument("--labels", action="append", required=True, help="SpeechBoundary label JSONL. Repeatable.")
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        help="feature_manifest.json. Repeatable; order must match --labels.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-sequence-jsonl", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected feature rows for smoke runs.")
    parser.add_argument("--target-chunk-s", type=float, default=9.0)
    parser.add_argument("--safe-merge-gap-s", type=float, default=0.12)
    parser.add_argument("--long-gap-split-s", type=float, default=0.60)
    parser.add_argument("--synthetic-merge-positives-per-record", type=int, default=0)
    parser.add_argument("--synthetic-merge-gap-s", type=float, default=0.04)
    parser.add_argument("--synthetic-merge-min-segment-s", type=float, default=1.20)
    parser.add_argument(
        "--merge-unspecified-short-gaps",
        action="store_true",
        help="Treat unlabeled very short gaps as merge positives. Off by default.",
    )
    args = parser.parse_args(argv)
    if len(args.labels) != len(args.feature_manifest):
        parser.error("--labels and --feature-manifest must be provided the same number of times")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.target_chunk_s <= 0.0:
        parser.error("--target-chunk-s must be positive")
    if args.safe_merge_gap_s < 0.0:
        parser.error("--safe-merge-gap-s must be non-negative")
    if args.long_gap_split_s < 0.0:
        parser.error("--long-gap-split-s must be non-negative")
    if args.synthetic_merge_positives_per_record < 0:
        parser.error("--synthetic-merge-positives-per-record must be non-negative")
    if args.synthetic_merge_gap_s < 0.0:
        parser.error("--synthetic-merge-gap-s must be non-negative")
    if args.synthetic_merge_min_segment_s <= 0.0:
        parser.error("--synthetic-merge-min-segment-s must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_gap_dataset(
        labels_paths=[Path(path) for path in args.labels],
        feature_manifest_paths=[Path(path) for path in args.feature_manifest],
        output_jsonl=Path(args.output_jsonl),
        summary_json=Path(args.summary_json) if args.summary_json else None,
        output_sequence_jsonl=Path(args.output_sequence_jsonl) if args.output_sequence_jsonl else None,
        limit=args.limit,
        config=GapDatasetConfig(
            target_chunk_s=args.target_chunk_s,
            safe_merge_gap_s=args.safe_merge_gap_s,
            long_gap_split_s=args.long_gap_split_s,
            synthetic_merge_positives_per_record=args.synthetic_merge_positives_per_record,
            synthetic_merge_gap_s=args.synthetic_merge_gap_s,
            synthetic_merge_min_segment_s=args.synthetic_merge_min_segment_s,
            merge_unspecified_short_gaps=args.merge_unspecified_short_gaps,
        ),
    )
    print(f"dataset={summary['output_jsonl']}")
    print(f"summary={summary['summary_json']}")
    print(f"counts={json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"class_balance={json.dumps(summary['class_balance'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
