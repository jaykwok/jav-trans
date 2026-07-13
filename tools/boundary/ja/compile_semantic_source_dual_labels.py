#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.ja.dataset import LabelRecord, frame_count, write_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_MEMBERSHIP_LABELS,
    SPEECH_ISLAND_SCORER_LABELS,
)


SCHEMA = "semantic_source_dual_supervision_v1"
MEMBERSHIP_BY_SOURCE_GATE = {
    "discardable": "outside",
    "contains_semantic": "inside",
    "unsure": "unsure",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_count(labels: list[str], *, inactive: str) -> int:
    runs = 0
    active = False
    for label in labels:
        retained = label != inactive
        if retained and not active:
            runs += 1
        active = retained
    return runs


def _candidate_for_time(candidates: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    for index, candidate in enumerate(candidates):
        start = float(candidate["context_start_s"])
        end = float(candidate["context_end_s"])
        if start <= time_s < end or (index + 1 == len(candidates) and time_s <= end):
            return candidate
    raise ValueError(f"no semantic candidate cell covers frame time {time_s:.6f}s")


def compile_record(
    row: dict[str, Any],
    *,
    frame_hop_s: float = 0.02,
    partition: str = "train",
) -> tuple[LabelRecord, dict[str, Any]]:
    candidates = sorted(
        [dict(candidate) for candidate in row["candidates"]],
        key=lambda candidate: float(candidate["context_start_s"]),
    )
    duration_s = float(row["duration_s"])
    total = frame_count(duration_s, frame_hop_s)
    source_gate = dict(row["source_gate"])
    membership_label = MEMBERSHIP_BY_SOURCE_GATE[str(source_gate["label"])]
    content_labels: list[str] = []
    content_weights: list[float] = []
    for frame_index in range(total):
        center_s = min(duration_s, (frame_index + 0.5) * frame_hop_s)
        candidate = _candidate_for_time(candidates, center_s)
        label = str(candidate["label"])
        if label not in SPEECH_ISLAND_SCORER_LABELS:
            raise ValueError(f"invalid semantic content label: {label!r}")
        content_labels.append(label)
        content_weights.append(float(candidate["confidence"]))
    membership_labels = [membership_label] * total
    membership_weights = [float(source_gate["confidence"])] * total
    speech_frames = [0 if membership_label == "outside" else 1] * total
    record = LabelRecord(
        audio_id=str(row["sample_id"]),
        source=str(row.get("source") or ""),
        duration_s=duration_s,
        text=str(row.get("reference_text") or ""),
        teacher_segments={},
        frame_hop_s=float(frame_hop_s),
        speech_frames=speech_frames,
        label_quality="supervised",
        frame_weights=[1.0] * total,
        boundary_metadata={
            "schema": SCHEMA,
            "source_partition": partition,
            "source_gate": source_gate,
            "semantic_class_frames": content_labels,
            "semantic_class_weights": content_weights,
            "semantic_membership_frames": membership_labels,
            "semantic_membership_weights": membership_weights,
            "content_label_source": "candidate_cells",
            "membership_label_source": "source_gate",
        },
    )
    summary = {
        "sample_id": str(row["sample_id"]),
        "frame_count": total,
        "source_gate": str(source_gate["label"]),
        "membership_label": membership_label,
        "content_label_counts": dict(sorted(Counter(content_labels).items())),
        "content_retained_run_count": _run_count(
            content_labels, inactive="discardable"
        ),
        "membership_island_count": _run_count(
            membership_labels, inactive="outside"
        ),
        "partition": partition,
    }
    return record, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("semantic source dual supervision is 1.7B-only")
    labels = _rows(Path(args.labels))
    features = {
        str(row["audio_id"]): dict(row) for row in _rows(Path(args.feature_manifest))
    }
    validation_ids = {str(value) for value in args.validation_sample_id}
    records: list[LabelRecord] = []
    summaries: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for label_index, row in enumerate(labels):
        sample_id = str(row["sample_id"])
        record, summary = compile_record(
            row,
            frame_hop_s=float(args.frame_hop_s),
            partition="validation" if sample_id in validation_ids else "train",
        )
        feature_row = features[sample_id]
        ptm, mfcc = load_cached_feature(Path(str(feature_row["feature_path"])))
        records.append(record)
        summaries.append(summary)
        manifest_rows.append(
            {
                **feature_row,
                "audio_id": sample_id,
                "label_index": label_index,
                "frame_count": int(min(ptm.shape[0], mfcc.shape[0])),
                "frame_hop_s": float(args.frame_hop_s),
                "ptm": args.ptm_repo_id,
                "ptm_dim": int(ptm.shape[1]),
                "mfcc_dim": int(mfcc.shape[1]),
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.jsonl"
    manifest_path = output / "feature_manifest.jsonl"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    result = {
        "schema": SCHEMA,
        "sample_count": len(records),
        "content_labels": list(SPEECH_ISLAND_SCORER_LABELS),
        "membership_labels": list(SPEECH_ISLAND_MEMBERSHIP_LABELS),
        "labels": str(labels_path),
        "feature_manifest": str(manifest_path),
        "samples": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile semantic content and source-membership supervision."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--validation-sample-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
