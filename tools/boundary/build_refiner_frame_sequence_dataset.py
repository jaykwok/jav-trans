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
    boundary_window_sequence_features,
    feature_extraction_hash,
    feature_extraction_signature,
    frame_sequence_feature_names,
    get_feature_dim,
    validate_sequence_features,
)
from boundary.ja import LabelRecord, TeacherSegment, load_cached_feature

DATASET_SCHEMA = "boundary_refiner_frame_sequence_dataset_v5"


@dataclass(frozen=True)
class FrameSequenceConfig:
    left_context_s: float = 0.60
    right_context_s: float = 0.60
    max_ptm_dims: int = 64
    include_mfcc: bool = True
    target_chunk_s: float = 3.0
    long_gap_split_s: float = 0.60
    synthetic_boundary_delta_jitter_s: float = 0.0
    synthetic_boundary_delta_seed: int = 240609

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
                counters["sequence_items"] += len(row["sequence_features"])
                counters["start_supervised"] += sum(
                    1 for item in row["sequence_boundary_delta_weights"] if float(item[0]) > 0.0
                )
                counters["end_supervised"] += sum(
                    1 for item in row["sequence_boundary_delta_weights"] if float(item[1]) > 0.0
                )
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
    if not segments:
        return None
    sequence_features: list[list[float]] = []
    sequence_boundary_delta_targets: list[list[float]] = []
    sequence_boundary_delta_weights: list[list[float]] = []
    sequence_reasons: list[str] = []
    gap_indexes: list[int] = []

    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        reason = _boundary_reason(index=index, left=left, right=right, record=record, config=config)
        if reason is None:
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
        sequence_boundary_delta_targets.append(
            _boundary_delta_targets(
                record=record,
                gap_index=index,
                reason=reason,
                config=config,
            )
        )
        sequence_boundary_delta_weights.append([1.0, 0.6])
        sequence_reasons.append(reason)
        gap_indexes.append(index)

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
        "sequence_boundary_delta_targets": sequence_boundary_delta_targets,
        "sequence_boundary_delta_weights": sequence_boundary_delta_weights,
        "sequence_reasons": sequence_reasons,
        "gap_indexes": gap_indexes,
        "metadata": {
            "labels_path": str(labels_path),
            "feature_manifest_path": str(feature_manifest_path),
            "feature_path": str(manifest_row.get("feature_path") or ""),
            "ptm_repo_id": str(manifest_row.get("ptm") or ""),
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
    return boundary_window_sequence_features(
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


def _boundary_reason(
    *,
    index: int,
    left: Segment,
    right: Segment,
    record: LabelRecord,
    config: FrameSequenceConfig,
) -> str | None:
    gap_s = right.start - left.end
    if gap_s < 0.0:
        return "boundary_overlap"
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
            return "boundary_gap_zone"
        if boundary_type == "cut_point":
            return "boundary_cut_point"
    if gap_s >= config.long_gap_split_s:
        return "boundary_long_gap"
    return None


def _boundary_delta_targets(
    *,
    record: LabelRecord,
    gap_index: int,
    reason: str,
    config: FrameSequenceConfig,
) -> list[float]:
    if config.synthetic_boundary_delta_jitter_s <= 0.0:
        return [0.0, 0.0]
    if reason not in {"boundary_cut_point", "boundary_gap_zone", "boundary_long_gap", "boundary_overlap"}:
        return [0.0, 0.0]
    limit = max(0.0, float(config.synthetic_boundary_delta_jitter_s))
    digest = hashlib.sha1(
        f"{config.synthetic_boundary_delta_seed}:{record.audio_id}:{gap_index}:{reason}".encode("utf-8")
    ).digest()
    start_fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    end_fraction = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    start_delta = (start_fraction * 2.0 - 1.0) * limit
    end_delta = (end_fraction * 2.0 - 1.0) * limit
    return [round(start_delta, 6), round(end_delta, 6)]


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
    parser.add_argument("--long-gap-split-s", type=float, default=0.60)
    parser.add_argument(
        "--synthetic-boundary-delta-jitter-s",
        type=float,
        default=0.0,
        help="Optional deterministic start/end delta supervision for split boundaries.",
    )
    parser.add_argument("--synthetic-boundary-delta-seed", type=int, default=240609)
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
    if args.long_gap_split_s < 0.0:
        parser.error("--long-gap-split-s must be non-negative")
    if args.synthetic_boundary_delta_jitter_s < 0.0:
        parser.error("--synthetic-boundary-delta-jitter-s must be non-negative")
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
            long_gap_split_s=args.long_gap_split_s,
            synthetic_boundary_delta_jitter_s=args.synthetic_boundary_delta_jitter_s,
            synthetic_boundary_delta_seed=args.synthetic_boundary_delta_seed,
        ),
    )
    print(f"dataset={summary['output_jsonl']}")
    print(f"summary={summary['summary_json']}")
    print(f"counts={json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
