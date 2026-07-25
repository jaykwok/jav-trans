#!/usr/bin/env python3
"""Freeze every audio-valid source in the Scorer v12 partition inventory."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import wave
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
)
from tools.boundary.ja.build_vocal_envelope_scorer_v12_pilot_manifest import (  # noqa: E402
    CONTRACT_ID,
    FRAME_HOP_S,
    PARTITION_SCHEMA,
    _index,
    _jsonl,
    _sha256,
    _source_id,
    _validated_audio_identity,
)

SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_full_manifest_summary_v1"


def build_full_manifest(
    *, source_windows: Path, partition_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    source_windows = source_windows.resolve()
    partition_manifest = partition_manifest.resolve()
    sources = _index(
        ({**row, "source_id": _source_id(row)} for row in _jsonl(source_windows)),
        "source_id",
        name="full-source windows",
    )
    partition_rows = _jsonl(partition_manifest)
    partitions = _index(partition_rows, "source_id", name="partition manifest")
    output_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    video_partitions: dict[str, str] = {}
    partition_counts: Counter[str] = Counter()
    for source_id in sorted(partitions):
        partition_row = partitions[source_id]
        if partition_row.get("schema") != PARTITION_SCHEMA:
            raise ValueError("unexpected frozen partition schema")
        if partition_row.get("boundary_serialization_contract_id") != CONTRACT_ID:
            raise ValueError("frozen partition central contract mismatch")
        partition = str(partition_row.get("partition") or "")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"invalid v12 full partition: {source_id}")
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"partition identity is absent from source windows: {source_id}")
        video_id = str(partition_row.get("video_id") or "")
        if not video_id or video_id != str(source.get("video_id") or ""):
            raise ValueError(f"v12 full video identity mismatch: {source_id}")
        try:
            audio, geometry, actual_sha, duration_delta = _validated_audio_identity(
                source_id, sources=sources, source_windows=source_windows
            )
        except (FileNotFoundError, ValueError, wave.Error) as error:
            rejected.append(
                {"source_id": source_id, "partition": partition, "reason": str(error)}
            )
            continue
        previous = video_partitions.setdefault(video_id, partition)
        if previous != partition:
            raise ValueError(f"v12 full video crosses partitions: {video_id}")
        partition_counts[partition] += 1
        output_rows.append(
            {
                "schema": VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "source_id": source_id,
                "video_id": video_id,
                "partition": partition,
                "core_ids": [source_id],
                "audio": str(audio),
                "audio_sha256": actual_sha,
                "duration_s": float(geometry["duration_s"]),
                "source_manifest_duration_s": float(source.get("duration_s") or 0.0),
                "audio_duration_delta_s": round(duration_delta, 9),
                "frame_count": int(geometry["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "sample_rate": int(geometry["sample_rate"]),
                "sample_count": int(geometry["sample_count"]),
                "channels": int(geometry["channels"]),
                "sample_width_bytes": int(geometry["sample_width_bytes"]),
                "source_kind": "real_full_source",
                "synthetic_composite": False,
                "pilot_role": "train" if partition == "train" else "heldout",
                "identity_source": "complete_frozen_full_source_and_partition_manifests",
                "identity_reused_from_v11_only": False,
                "v11_truth_inherited": False,
                "v11_span_inherited": False,
                "asr_text_used_as_truth": False,
                "training_manifest_allowed": False,
            }
        )
    if set(partition_counts) != {"train", "val", "test"}:
        raise ValueError("v12 full manifest requires non-empty train, val and test")
    if len({row["source_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("v12 full source IDs must be unique")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "source_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "source_count": len(output_rows),
        "partition_counts": dict(sorted(partition_counts.items())),
        "video_count": len(video_partitions),
        "core_count": len(output_rows),
        "rejected_identities": rejected,
        "source_windows": str(source_windows),
        "source_windows_sha256": _sha256(source_windows),
        "partition_manifest": str(partition_manifest),
        "partition_manifest_sha256": _sha256(partition_manifest),
        "selection_policy": "all_audio_valid_frozen_partition_sources_v1",
        "identity_selection_uses_labels": False,
        "v11_truth_inherited": False,
        "v11_span_inherited": False,
        "asr_text_used_as_truth": False,
        "training_manifest_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_full_manifest(
                source_windows=Path(args.source_windows),
                partition_manifest=Path(args.partition_manifest),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
