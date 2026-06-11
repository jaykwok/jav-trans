#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
    FRAME_SEQUENCE_FRAMES_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    feature_extraction_hash,
    feature_extraction_signature,
    validate_sequence_features,
)

DATASET_SCHEMA = "boundary_refiner_frame_sequence_dataset_v5"
SILVER_SOURCE_SCHEMA = "speech_boundary_silver_refiner_dataset_v1"


@dataclass(frozen=True)
class SilverRefinerDatasetConfig:
    left_context_s: float = 0.60
    right_context_s: float = 0.60
    max_ptm_dims: int = 64
    include_mfcc: bool = True
    target_chunk_s: float = 3.0
    max_delta_s: float = 0.75
    max_sequence_items: int = 512
    start_delta_weight_floor: float = 1.0
    end_delta_weight_floor: float = 0.6

    def feature_config(self) -> FrameSequenceFeatureConfig:
        return FrameSequenceFeatureConfig(
            left_context_s=self.left_context_s,
            right_context_s=self.right_context_s,
            max_ptm_dims=self.max_ptm_dims,
            include_mfcc=self.include_mfcc,
            target_chunk_s=self.target_chunk_s,
        )


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path) -> str:
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def build_silver_refiner_dataset(
    *,
    silver_labels_path: Path,
    sequence_feature_manifest_path: Path,
    output_jsonl: Path,
    summary_json: Path | None = None,
    config: SilverRefinerDatasetConfig = SilverRefinerDatasetConfig(),
    limit_cases: int | None = None,
) -> dict[str, Any]:
    if summary_json is None:
        summary_json = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    silver_by_aligned = _load_silver_by_aligned(silver_labels_path)
    feature_lookup = _load_sequence_feature_manifest(sequence_feature_manifest_path)
    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    feature_names: list[str] | None = None
    feature_schema_hash = ""

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for case_index, (aligned_path_key, silver_rows) in enumerate(
            sorted(silver_by_aligned.items())
        ):
            if limit_cases is not None and case_index >= limit_cases:
                break
            aligned_path = project_path(aligned_path_key)
            if not aligned_path.exists():
                counters["skipped_missing_aligned_path"] += 1
                continue
            chunks = _load_transcript_chunks(aligned_path)
            if len(chunks) < 2:
                counters["skipped_too_few_chunks"] += 1
                continue
            feature_row = _feature_row_for_case(silver_rows, feature_lookup)
            if feature_row is None:
                counters["skipped_missing_sequence_features"] += 1
                continue
            provider = _load_feature_provider(feature_row, config=config)
            case_feature_names = provider.feature_names()
            case_schema_hash = provider.feature_schema_hash()
            if feature_names is None:
                feature_names = case_feature_names
                feature_schema_hash = case_schema_hash
            elif feature_names != case_feature_names:
                raise ValueError(f"feature_names changed for {aligned_path}")
            elif feature_schema_hash != case_schema_hash:
                raise ValueError(f"feature_schema_hash changed for {aligned_path}")

            silver_by_chunk = {
                int(row["chunk_index"]): row
                for row in silver_rows
                if str(row.get("schema") or "") == "speech_boundary_silver_display_v1"
            }
            sequence_items = _case_sequence_items(
                chunks=chunks,
                silver_by_chunk=silver_by_chunk,
                provider=provider,
                feature_names=case_feature_names,
                config=config,
                counters=counters,
                reason_counts=reason_counts,
            )
            if not sequence_items:
                counters["skipped_no_supervised_gaps"] += 1
                continue
            for block_index, block in enumerate(
                _chunks(sequence_items, max(1, int(config.max_sequence_items)))
            ):
                row = _dataset_row(
                    block,
                    aligned_path=aligned_path,
                    feature_row=feature_row,
                    block_index=block_index,
                    feature_names=case_feature_names,
                    feature_schema_hash=case_schema_hash,
                    provider=provider,
                    config=config,
                )
                validate_sequence_features(
                    row["sequence_features"],
                    feature_names=row["feature_names"],
                )
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counters["rows"] += 1
                counters["sequence_items"] += len(block)
                counters["start_supervised"] += sum(
                    1 for item in block if item["delta_weights"][0] > 0.0
                )
                counters["end_supervised"] += sum(
                    1 for item in block if item["delta_weights"][1] > 0.0
                )

    summary = {
        "schema": SILVER_SOURCE_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "silver_labels": str(silver_labels_path),
        "sequence_feature_manifest": str(sequence_feature_manifest_path),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "config": asdict(config),
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": feature_schema_hash,
        "feature_names": feature_names or [],
        "feature_dim": len(feature_names or []),
        "counts": dict(counters),
        "reasons": dict(reason_counts),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _load_silver_by_aligned(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"silver label row must be an object: {path}:{line_number}")
            aligned_path = str(row.get("aligned_path") or "")
            if not aligned_path:
                raise ValueError(f"silver label row missing aligned_path: {path}:{line_number}")
            grouped[aligned_path].append(dict(row))
    return dict(grouped)


def _load_sequence_feature_manifest(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"sequence feature manifest row must be an object: {path}:{line_number}")
            feature_path = str(row.get("feature_path") or "")
            if not feature_path:
                raise ValueError(f"sequence feature manifest row missing feature_path: {path}:{line_number}")
            for key in _feature_lookup_keys(row):
                lookup[key] = dict(row)
    return lookup


def _feature_lookup_keys(row: Mapping[str, Any]) -> list[str]:
    keys = []
    video = str(row.get("video") or "")
    source_audio_path = str(row.get("source_audio_path") or "")
    if video:
        keys.append(f"video:{video}")
    if source_audio_path:
        keys.append(f"source:{project_rel(project_path(source_audio_path))}")
    return keys


def _feature_row_for_case(
    silver_rows: Sequence[Mapping[str, Any]],
    feature_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not silver_rows:
        return None
    video = str(silver_rows[0].get("video") or "")
    if video and f"video:{video}" in feature_lookup:
        return dict(feature_lookup[f"video:{video}"])
    source_audio_path = str(silver_rows[0].get("source_audio_path") or "")
    source_key = f"source:{project_rel(project_path(source_audio_path))}"
    if source_audio_path and source_key in feature_lookup:
        return dict(feature_lookup[source_key])
    return None


def _load_feature_provider(
    row: Mapping[str, Any],
    *,
    config: SilverRefinerDatasetConfig,
) -> FrameSequenceFeatureProvider:
    feature_path = project_path(str(row["feature_path"]))
    with np.load(feature_path) as data:
        ptm = np.asarray(data["ptm"], dtype=np.float32)
        mfcc = np.asarray(data["mfcc"], dtype=np.float32)
        duration_s = float(np.asarray(data["duration_s"]).reshape(-1)[0])
        frame_hop_s = float(np.asarray(data["frame_hop_s"]).reshape(-1)[0])
    return FrameSequenceFeatureProvider(
        duration_s=duration_s,
        frame_hop_s=frame_hop_s,
        ptm=ptm,
        mfcc=mfcc,
        config=config.feature_config(),
    )


def _load_transcript_chunks(aligned_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(aligned_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return []
    raw_chunks = ((payload.get("asr_details") or {}).get("transcript_chunks") or [])
    chunks: list[dict[str, Any]] = []
    for raw in raw_chunks:
        if not isinstance(raw, Mapping):
            continue
        try:
            index = int(raw.get("index"))
            start = float(raw.get("start"))
            end = float(raw.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        chunks.append({"index": index, "start": start, "end": end})
    return sorted(chunks, key=lambda item: int(item["index"]))


def _case_sequence_items(
    *,
    chunks: Sequence[Mapping[str, Any]],
    silver_by_chunk: Mapping[int, Mapping[str, Any]],
    provider: FrameSequenceFeatureProvider,
    feature_names: Sequence[str],
    config: SilverRefinerDatasetConfig,
    counters: Counter[str],
    reason_counts: Counter[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for left, right in zip(chunks, chunks[1:]):
        left_index = int(left["index"])
        right_index = int(right["index"])
        if right_index != left_index + 1:
            counters["skipped_non_consecutive_chunk_index"] += 1
            continue
        left_silver = silver_by_chunk.get(left_index)
        right_silver = silver_by_chunk.get(right_index)
        if left_silver is None and right_silver is None:
            continue
        left_start, left_end = float(left["start"]), float(left["end"])
        right_start, right_end = float(right["start"]), float(right["end"])
        if right_start < left_end:
            counters["overlapping_chunk_gap"] += 1
        features = provider.features_for_boundary(
            left_start_s=left_start,
            left_end_s=left_end,
            right_start_s=right_start,
            right_end_s=right_end,
        )
        start_delta, start_weight, start_clipped = _boundary_delta(
            silver=right_silver,
            label_key="display_start_label",
            core_value=right_start,
            weight_key="start_weight",
            weight_floor=config.start_delta_weight_floor,
            config=config,
        )
        end_delta, end_weight, end_clipped = _boundary_delta(
            silver=left_silver,
            label_key="display_end_label",
            core_value=left_end,
            weight_key="end_weight",
            weight_floor=config.end_delta_weight_floor,
            config=config,
        )
        if start_clipped:
            counters["start_delta_clipped"] += 1
        if end_clipped:
            counters["end_delta_clipped"] += 1
        reasons = []
        if right_silver is not None:
            reasons.append("start_from_forced_first_word")
        if left_silver is not None:
            reasons.append("end_from_forced_last_word")
        reason = "+".join(reasons) if reasons else "unsupervised"
        reason_counts[reason] += 1
        items.append(
            {
                "features": features,
                "delta_targets": [start_delta, end_delta],
                "delta_weights": [start_weight, end_weight],
                "gap_index": left_index,
                "left_chunk_index": left_index,
                "right_chunk_index": right_index,
                "reason": reason,
            }
        )
    validate_sequence_features(
        [item["features"] for item in items],
        feature_names=feature_names,
    ) if items else None
    return items


def _boundary_delta(
    *,
    silver: Mapping[str, Any] | None,
    label_key: str,
    core_value: float,
    weight_key: str,
    weight_floor: float,
    config: SilverRefinerDatasetConfig,
) -> tuple[float, float, bool]:
    if silver is None:
        return 0.0, 0.0, False
    label_policy = silver.get("label_policy")
    policy = label_policy if isinstance(label_policy, Mapping) else {}
    try:
        label = float(silver[label_key])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0, False
    raw_delta = label - float(core_value)
    clipped = max(-config.max_delta_s, min(config.max_delta_s, raw_delta))
    weight = max(float(weight_floor), float(policy.get(weight_key, weight_floor)))
    if not math.isfinite(weight) or weight < 0.0:
        weight = 0.0
    return round(float(clipped), 6), round(weight, 6), abs(clipped - raw_delta) > 1e-9


def _dataset_row(
    block: Sequence[Mapping[str, Any]],
    *,
    aligned_path: Path,
    feature_row: Mapping[str, Any],
    block_index: int,
    feature_names: Sequence[str],
    feature_schema_hash: str,
    provider: FrameSequenceFeatureProvider,
    config: SilverRefinerDatasetConfig,
) -> dict[str, Any]:
    feature_config = config.feature_config()
    return {
        "schema": DATASET_SCHEMA,
        "source_schema": SILVER_SOURCE_SCHEMA,
        "feature_schema": FRAME_SEQUENCE_FEATURE_SCHEMA,
        "feature_schema_hash": feature_schema_hash,
        "feature_signature": feature_extraction_signature(
            config=feature_config,
            feature_names=feature_names,
        ),
        "audio_id": str(feature_row.get("video") or aligned_path.stem),
        "source": "real_forced_aligner_silver",
        "label_index": block_index,
        "feature_names": list(feature_names),
        "feature_dim": len(feature_names),
        "sequence_features": [list(item["features"]) for item in block],
        "sequence_boundary_delta_targets": [list(item["delta_targets"]) for item in block],
        "sequence_boundary_delta_weights": [list(item["delta_weights"]) for item in block],
        "sequence_reasons": [str(item["reason"]) for item in block],
        "gap_indexes": [int(item["gap_index"]) for item in block],
        "metadata": {
            "aligned_path": project_rel(aligned_path),
            "feature_path": str(feature_row.get("feature_path") or ""),
            "source_audio_path": str(feature_row.get("source_audio_path") or ""),
            "block_index": block_index,
            "left_chunk_index": int(block[0]["left_chunk_index"]),
            "right_chunk_index": int(block[-1]["right_chunk_index"]),
            "frame_hop_s": float(provider.frame_hop_s),
            "feature_frame_schema": FRAME_SEQUENCE_FRAMES_SCHEMA,
            "ptm_used_dim": provider.frame_dims()[1],
            "mfcc_dim": provider.frame_dims()[2],
            "config": asdict(config),
        },
    }


def _chunks(values: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert forced-aligner silver labels into Boundary Refiner v5 sequence rows."
    )
    parser.add_argument("--silver-labels", required=True)
    parser.add_argument("--sequence-feature-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--left-context-s", type=float, default=0.60)
    parser.add_argument("--right-context-s", type=float, default=0.60)
    parser.add_argument("--max-ptm-dims", type=int, default=64)
    parser.add_argument("--no-mfcc", action="store_true")
    parser.add_argument("--target-chunk-s", type=float, default=3.0)
    parser.add_argument("--max-delta-s", type=float, default=0.75)
    parser.add_argument("--max-sequence-items", type=int, default=512)
    parser.add_argument("--start-delta-weight-floor", type=float, default=1.0)
    parser.add_argument("--end-delta-weight-floor", type=float, default=0.6)
    args = parser.parse_args(argv)
    if args.limit_cases is not None and args.limit_cases <= 0:
        parser.error("--limit-cases must be positive")
    if args.left_context_s <= 0.0 or args.right_context_s <= 0.0:
        parser.error("context seconds must be positive")
    if args.max_ptm_dims <= 0:
        parser.error("--max-ptm-dims must be positive")
    if args.target_chunk_s <= 0.0:
        parser.error("--target-chunk-s must be positive")
    if args.max_delta_s <= 0.0:
        parser.error("--max-delta-s must be positive")
    if args.max_sequence_items <= 0:
        parser.error("--max-sequence-items must be positive")
    if args.start_delta_weight_floor < 0.0 or args.end_delta_weight_floor < 0.0:
        parser.error("delta weight floors must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_silver_refiner_dataset(
        silver_labels_path=Path(args.silver_labels),
        sequence_feature_manifest_path=Path(args.sequence_feature_manifest),
        output_jsonl=Path(args.output_jsonl),
        summary_json=Path(args.summary_json) if args.summary_json else None,
        limit_cases=args.limit_cases,
        config=SilverRefinerDatasetConfig(
            left_context_s=args.left_context_s,
            right_context_s=args.right_context_s,
            max_ptm_dims=args.max_ptm_dims,
            include_mfcc=not args.no_mfcc,
            target_chunk_s=args.target_chunk_s,
            max_delta_s=args.max_delta_s,
            max_sequence_items=args.max_sequence_items,
            start_delta_weight_floor=args.start_delta_weight_floor,
            end_delta_weight_floor=args.end_delta_weight_floor,
        ),
    )
    print(f"dataset={summary['output_jsonl']}")
    print(f"summary={summary['summary_json']}")
    print(f"counts={json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
