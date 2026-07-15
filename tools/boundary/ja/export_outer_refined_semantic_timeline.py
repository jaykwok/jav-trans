#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.outer_refiner_v2 import (  # noqa: E402
    OuterEdgeRefinerV2,
    PairedOuterEdgePrediction,
    load_outer_edge_refiner_v2,
)


SCHEMA = "outer_refined_semantic_timeline_v1"
AUDIO_CONTRACT = "learned_outer_refined_island_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shift_span(
    row: dict[str, Any], *, start_s: float, end_s: float, offset_s: float
) -> dict[str, Any]:
    shifted = dict(row)
    shifted["start_s"] = float(start_s) - offset_s
    shifted["end_s"] = float(end_s) - offset_s
    shifted["source_start_s"] = float(start_s)
    shifted["source_end_s"] = float(end_s)
    return shifted


def rebase_timeline_row(
    row: dict[str, Any],
    *,
    output_audio: str,
    prediction: PairedOuterEdgePrediction,
    outer_checkpoint_sha256: str,
) -> dict[str, Any]:
    source_start = float(prediction.start_s)
    source_end = float(prediction.end_s)
    duration = source_end - source_start
    alignments = []
    violations: list[str] = []
    for item in row.get("semantic_alignments") or []:
        if item.get("status") != "matched":
            alignments.append(dict(item))
            continue
        original_start = float(item["start_s"])
        original_end = float(item["end_s"])
        if original_start < source_start or original_end > source_end:
            violations.append(str(item["unit_id"]))
        alignments.append(
            _shift_span(
                item,
                start_s=original_start,
                end_s=original_end,
                offset_s=source_start,
            )
        )
    events = []
    for event in row.get("semantic_events") or []:
        shifted = dict(event)
        if event.get("status") == "matched":
            shifted["source_interval_start_s"] = float(event["interval_start_s"])
            shifted["source_interval_end_s"] = float(event["interval_end_s"])
            shifted["interval_start_s"] = (
                float(event["interval_start_s"]) - source_start
            )
            shifted["interval_end_s"] = float(event["interval_end_s"]) - source_start
        events.append(shifted)
    result = {
        **row,
        "schema": SCHEMA,
        "audio": output_audio,
        "duration_s": duration,
        "source_audio": str(row["audio"]),
        "source_duration_s": float(row["duration_s"]),
        "source_span": {"start_s": source_start, "end_s": source_end},
        "audio_contract": AUDIO_CONTRACT,
        "pipeline_entry_stage": "semantic_split_v3",
        "semantic_alignments": alignments,
        "semantic_timeline": [
            {
                "unit_id": str(item["unit_id"]),
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
            }
            for item in alignments
            if item.get("status") == "matched"
        ],
        "semantic_events": events,
        "outer_prediction": {
            **asdict(prediction),
            "class_probabilities": None,
            "checkpoint_sha256": outer_checkpoint_sha256,
        },
        "outer_alignment_violations": violations,
        "training_ready": not violations and not prediction.abstain_reason,
    }
    return result


def _features(path: Path, *, total: int) -> np.ndarray:
    with np.load(path) as data:
        ptm_key = "ptm2048" if "ptm2048" in data.files else "ptm"
        ptm = np.asarray(data[ptm_key], dtype=np.float32)[:total]
        mfcc = np.asarray(data["mfcc"], dtype=np.float32)[:total]
    total = min(total, ptm.shape[0], mfcc.shape[0])
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    return np.concatenate((ptm[:total], mfcc[:total], position), axis=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    apply_vram_safety_cap(0.95)
    labels = _rows(Path(args.timeline_labels))
    features = {
        str(row.get("sample_id") or row["audio_id"]): row
        for row in _rows(Path(args.feature_manifest))
    }
    model: OuterEdgeRefinerV2 = load_outer_edge_refiner_v2(
        args.outer_checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for row in labels:
        sample_id = str(row["sample_id"])
        feature_row = features[sample_id]
        total = int(feature_row["frame_count"])
        frame_features = _features(Path(feature_row["feature_path"]), total=total)
        prediction = model.predict_islands(
            frame_feature_groups=[frame_features],
            raw_spans=[(0.0, float(row["duration_s"]))],
            frame_hop_s=float(feature_row.get("frame_hop_s") or 0.02),
        )[0]
        audio, sample_rate = load_audio_16k_mono(str(row["audio"]))
        start_sample = max(0, min(len(audio), int(round(prediction.start_s * sample_rate))))
        end_sample = max(start_sample, min(len(audio), int(round(prediction.end_s * sample_rate))))
        output_audio = audio_dir / f"{sample_id}.wav"
        sf.write(
            str(output_audio),
            np.asarray(audio[start_sample:end_sample], dtype=np.float32),
            sample_rate,
            subtype="PCM_16",
        )
        output_rows.append(
            rebase_timeline_row(
                row,
                output_audio=str(output_audio),
                prediction=prediction,
                outer_checkpoint_sha256=model.sha256,
            )
        )
    labels_path = output_dir / "outer_refined_timeline.jsonl"
    _write(labels_path, output_rows)
    summary = {
        "schema": SCHEMA,
        "sample_count": len(output_rows),
        "training_ready_count": sum(bool(row["training_ready"]) for row in output_rows),
        "alignment_violation_count": sum(
            bool(row["outer_alignment_violations"]) for row in output_rows
        ),
        "abstain_count": sum(
            bool(row["outer_prediction"]["abstain_reason"]) for row in output_rows
        ),
        "audio_contract": AUDIO_CONTRACT,
        "outer_checkpoint_sha256": model.sha256,
        "labels": str(labels_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run learned Outer v2 on confirmed semantic sources and rebase timelines."
    )
    parser.add_argument("--timeline-labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--outer-checkpoint", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
