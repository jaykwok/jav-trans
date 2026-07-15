#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS  # noqa: E402
from boundary.outer_refiner_v2 import load_outer_edge_refiner_v2  # noqa: E402


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _features_and_labels(row: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(row["source_feature_path"]) as source:
        ptm = source["ptm"].astype(np.float32)
        mfcc = source["mfcc"].astype(np.float32)
    with np.load(row["feature_path"]) as labels:
        targets = labels["labels"].astype(np.int64)
    total = min(ptm.shape[0], mfcc.shape[0], targets.shape[0])
    position = (
        np.arange(total, dtype=np.float32) / max(1, total - 1)
    ).reshape(-1, 1)
    return (
        np.ascontiguousarray(
            np.concatenate((ptm[:total], mfcc[:total], position), axis=1)
        ),
        targets[:total],
    )


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(np.mean(array)),
        "p50_s": float(np.percentile(array, 50)),
        "p95_s": float(np.percentile(array, 95)),
        "max_s": float(np.max(array)),
    }


def _worst(rows: list[dict], key: str) -> dict:
    row = max(rows, key=lambda item: float(item[key]))
    return {
        "audio_id": row["audio_id"],
        "error_s": float(row[key]),
        "truth_start_s": float(row["truth_start_s"]),
        "truth_end_s": float(row["truth_end_s"]),
        "predicted_start_s": float(row["predicted_start_s"]),
        "predicted_end_s": float(row["predicted_end_s"]),
        "start_action": row["start_action"],
        "end_action": row["end_action"],
        "abstain_reason": row["abstain_reason"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap(0.95)
    model = load_outer_edge_refiner_v2(
        args.checkpoint,
        device=args.device,
        expected_ptm_repo_id=QWEN_ASR_17B_REPO_ID,
    )
    val_rows = [
        row
        for row in _rows(Path(args.dataset_manifest))
        if row.get("partition") != "train"
    ]
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    evaluated: list[dict] = []
    for row in val_rows:
        features, labels = _features_and_labels(row)
        truth_target = np.flatnonzero(labels == target_index)
        if truth_target.size == 0:
            continue
        frame_hop_s = float(row.get("frame_hop_s", args.frame_hop_s))
        prediction = model.predict_islands(
            frame_feature_groups=[features],
            raw_spans=[(0.0, len(labels) * frame_hop_s)],
            frame_hop_s=frame_hop_s,
        )[0]
        truth_start_s = float(truth_target[0]) * frame_hop_s
        truth_end_s = float(truth_target[-1] + 1) * frame_hop_s
        start_signed_s = float(prediction.start_s) - truth_start_s
        end_signed_s = float(prediction.end_s) - truth_end_s
        evaluated.append(
            {
                "audio_id": row.get("audio_id"),
                "source_audio": str(
                    Path(args.dataset_manifest).resolve().parent.parent
                    / "audio"
                    / f"{row.get('audio_id')}.wav"
                ),
                "frame_hop_s": frame_hop_s,
                "raw_end_s": len(labels) * frame_hop_s,
                "truth_start_s": truth_start_s,
                "truth_end_s": truth_end_s,
                "predicted_start_s": float(prediction.start_s),
                "predicted_end_s": float(prediction.end_s),
                "start_signed_s": start_signed_s,
                "end_signed_s": end_signed_s,
                "start_absolute_s": abs(start_signed_s),
                "end_absolute_s": abs(end_signed_s),
                "start_inward_s": max(start_signed_s, 0.0),
                "end_inward_s": max(-end_signed_s, 0.0),
                "start_outward_s": max(-start_signed_s, 0.0),
                "end_outward_s": max(end_signed_s, 0.0),
                "start_action": prediction.start_action,
                "end_action": prediction.end_action,
                "abstain_reason": prediction.abstain_reason,
            }
        )
    if not evaluated:
        raise ValueError("validation partition has no semantic-target rows")
    metric_keys = (
        "start_absolute_s",
        "end_absolute_s",
        "start_inward_s",
        "end_inward_s",
        "start_outward_s",
        "end_outward_s",
    )
    metrics = {
        "schema": "outer_edge_refiner_v2_directional_metrics_v1",
        "checkpoint": str(Path(args.checkpoint)),
        "validation_count": len(evaluated),
        **{
            key: _distribution([float(row[key]) for row in evaluated])
            for key in metric_keys
        },
        "within_tolerance": {
            f"{int(tolerance * 1000)}ms": {
                "start": sum(
                    float(row["start_absolute_s"]) <= tolerance
                    for row in evaluated
                )
                / len(evaluated),
                "end": sum(
                    float(row["end_absolute_s"]) <= tolerance
                    for row in evaluated
                )
                / len(evaluated),
            }
            for tolerance in (0.1, 0.2, 0.3)
        },
        "worst_cases": {key: _worst(evaluated, key) for key in metric_keys},
    }
    output = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.details_output:
        _write_jsonl(Path(args.details_output), evaluated)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate directional Outer Edge Refiner v2 boundary errors."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--details-output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
