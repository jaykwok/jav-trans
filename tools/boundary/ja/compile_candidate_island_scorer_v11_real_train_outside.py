#!/usr/bin/env python3
"""Compile ASR-confirmed real train outside frames with unsure complements."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


INPUT_SCHEMA = "candidate_island_scorer_v11_real_outside_asr_selection_v1"
SCHEMA = "candidate_island_scorer_v11_real_train_outside_source_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_real_train_outside_summary_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spans(labels: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(labels, dtype=np.int64)
    edges = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(values) != 0) + 1,
            np.asarray([len(values)], dtype=np.int64),
        )
    )
    result: list[dict[str, Any]] = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        label = "outside_candidate" if int(values[start]) == 0 else "unsure"
        result.append(
            {
                "label": label,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_s": round(int(start) * FRAME_HOP_S, 6),
                "end_s": round(int(end) * FRAME_HOP_S, 6),
            }
        )
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    enriched_path = Path(args.asr_enriched_selection).resolve()
    if not enriched_path.is_file():
        raise FileNotFoundError(enriched_path)
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()
    skipped_no_outside_source_count = 0
    asr_text_spans = asr_error_spans = 0
    for row in _rows(enriched_path):
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("real train outside input has missing or duplicate source_id")
        seen.add(source_id)
        if row.get("schema") != INPUT_SCHEMA:
            raise ValueError(f"wrong real train outside ASR selection schema: {source_id}")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError(f"wrong central Boundary contract: {source_id}")
        if row.get("partition") != "train":
            raise ValueError(f"real outside source is not train-only: {source_id}")
        frame_count = int(row.get("frame_count") or 0)
        if frame_count <= 0:
            raise ValueError(f"real outside source has invalid geometry: {source_id}")
        labels = np.full(frame_count, -100, dtype=np.int64)
        for span in row.get("prediction_spans") or ():
            if span.get("label") != "asr_probe_candidate":
                raise ValueError(f"ASR probe span label changed: {source_id}")
            start = int(span.get("start_frame", -1))
            end = int(span.get("end_frame", -1))
            if start < 0 or end <= start or end > frame_count or np.any(labels[start:end] == 0):
                raise ValueError(f"real outside ASR spans overlap or escape source: {source_id}")
            evidence = dict(span.get("asr_probe") or {})
            if not evidence:
                raise ValueError(f"real outside span lacks ASR evidence: {source_id}")
            has_error = bool(evidence.get("error_kind"))
            has_text = bool(evidence.get("nonempty_text"))
            asr_error_spans += int(has_error)
            asr_text_spans += int(has_text)
            if not has_error and not has_text:
                labels[start:end] = 0
        outside_frames = int(np.sum(labels == 0))
        unsure_frames = int(np.sum(labels == -100))
        input_counts["outside_candidate"] += outside_frames
        input_counts["unsure"] += unsure_frames
        if not outside_frames:
            skipped_no_outside_source_count += 1
            continue
        counts["outside_candidate"] += outside_frames
        counts["unsure"] += unsure_frames
        output_rows.append(
            {
                "schema": SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": str(row.get("video_id") or ""),
                "partition": "train",
                "input_distribution": "real_workflow_source_window_gemini_asr_masked_v1",
                "synthetic_composite": False,
                "audio": str(row["audio"]),
                "audio_sha256": str(row["audio_sha256"]),
                "duration_s": float(row["duration_s"]),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "core_ids": [f"real-train-outside-source::{source_id}"],
                "canonical_spans": _spans(labels),
                "annotation_provenance": "gemini_outside_complement_plus_1p7b_asr_empty_v1",
                "gemini_output_used_as_inside_truth": False,
                "asr_text_used_as_inside_truth": False,
                "asr_empty_used_without_gemini_outside": False,
                "unsure_training_label": -100,
                "training_manifest_allowed": True,
            }
        )
    if not output_rows:
        raise ValueError("ASR-confirmed real train outside compile produced no usable sources")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "real_train_outside_sources.jsonl"
    _write_jsonl(manifest, output_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "asr_enriched_selection": str(enriched_path),
        "asr_enriched_selection_sha256": _sha256(enriched_path),
        "source_count": len(output_rows),
        "canonical_frame_counts": dict(sorted(counts.items())),
        "input_frame_counts": dict(sorted(input_counts.items())),
        "skipped_no_outside_source_count": skipped_no_outside_source_count,
        "asr_text_span_count": asr_text_spans,
        "asr_error_span_count": asr_error_spans,
        "real_train_outside_sources": str(manifest),
        "real_train_outside_sources_sha256": _sha256(manifest),
        "gemini_output_used_as_inside_truth": False,
        "asr_text_used_as_inside_truth": False,
        "asr_empty_used_without_gemini_outside": False,
        "unsure_training_label": -100,
        "training_manifest_allowed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-enriched-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
