#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (  # noqa: E402
    AdditionFusionBiLSTM,
    effective_frame_weights,
    frame_classification_counts,
    load_cached_feature,
    load_label_records,
    metrics_from_frame_counts,
)


def probability_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    quantiles = np.quantile(array, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": float(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p90": float(quantiles[6]),
        "p95": float(quantiles[7]),
        "p99": float(quantiles[8]),
    }


def load_addition_model(*, checkpoint_path: Path, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config") or {})
    model = AdditionFusionBiLSTM(
        whisper_dim=int(checkpoint["whisper_dim"]),
        mfcc_dim=int(checkpoint["mfcc_dim"]),
        fusion_dim=int(config.get("fusion_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_predictions(
    *,
    labels_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
    threshold: float = 0.5,
    include_probabilities: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(labels_path)
    feature_rows = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("feature manifest must be a JSON list")

    model = load_addition_model(checkpoint_path=checkpoint_path, device=device)
    prediction_path = output_dir / "predictions.jsonl"
    skipped = []
    aggregate: Counter[str] = Counter()
    all_probabilities: list[float] = []
    speech_probabilities: list[float] = []
    nonspeech_probabilities: list[float] = []
    rows = 0

    with prediction_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for row_index, row in enumerate(feature_rows):
            label_index = int(row["label_index"])
            if label_index < 0 or label_index >= len(records):
                skipped.append({"row_index": row_index, "reason": "label_index_out_of_range"})
                continue
            record = records[label_index]
            try:
                whisper, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            except Exception as exc:
                skipped.append(
                    {
                        "row_index": row_index,
                        "audio_id": str(row.get("audio_id") or record.audio_id),
                        "reason": "feature_load_error",
                        "error": str(exc),
                    }
                )
                continue

            weights = effective_frame_weights(record)
            frame_count = min(whisper.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
            if frame_count <= 0:
                skipped.append(
                    {
                        "row_index": row_index,
                        "audio_id": str(row.get("audio_id") or record.audio_id),
                        "reason": "empty_frames",
                    }
                )
                continue

            whisper_tensor = torch.from_numpy(np.ascontiguousarray(whisper[:frame_count], dtype=np.float32)).to(
                device
            ).unsqueeze(0)
            mfcc_tensor = torch.from_numpy(np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32)).to(
                device
            ).unsqueeze(0)
            logits = model(whisper_tensor, mfcc_tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[:frame_count]
            predictions = (probabilities >= threshold).astype(np.int32)
            labels = np.asarray(record.speech_frames[:frame_count], dtype=np.int32)
            active_weights = np.asarray(weights[:frame_count], dtype=np.float32)
            active = active_weights > 0.0
            counts = frame_classification_counts(
                labels=labels,
                predictions=predictions,
                weights=active_weights,
            )
            aggregate.update(counts)
            active_probabilities = probabilities[active]
            all_probabilities.extend(float(value) for value in active_probabilities)
            speech_probabilities.extend(float(value) for value in active_probabilities[labels[active] > 0])
            nonspeech_probabilities.extend(float(value) for value in active_probabilities[labels[active] <= 0])
            payload: dict[str, Any] = {
                "audio_id": record.audio_id,
                "source": record.source,
                "label_quality": record.label_quality,
                "duration_s": record.duration_s,
                "frame_hop_s": record.frame_hop_s,
                "threshold": float(threshold),
                "frame_count": int(frame_count),
                "speech_frames": predictions.astype(int).tolist(),
                "probability_summary": probability_summary(active_probabilities.tolist()),
                "speech_label_probability_summary": probability_summary(
                    active_probabilities[labels[active] > 0].tolist()
                ),
                "nonspeech_label_probability_summary": probability_summary(
                    active_probabilities[labels[active] <= 0].tolist()
                ),
                "counts": counts,
            }
            if include_probabilities:
                payload["probabilities"] = [float(value) for value in probabilities]
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1

    metrics_path = output_dir / "prediction_metrics.json"
    metrics = metrics_from_frame_counts(
        counts=dict(aggregate),
        windows=rows,
        metrics_path=metrics_path,
        threshold=threshold,
    )
    summary = {
        "labels": str(labels_path),
        "feature_manifest": str(feature_manifest_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "threshold": float(threshold),
        "predictions": str(prediction_path),
        "rows": rows,
        "skipped": len(skipped),
        "metrics": {
            "loss": metrics.loss,
            "frame_accuracy": metrics.frame_accuracy,
            "positive_ratio": metrics.positive_ratio,
            "predicted_positive_ratio": metrics.predicted_positive_ratio,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "frames": metrics.frames,
            "windows": metrics.windows,
            "threshold": metrics.threshold,
        },
        "probability_summary": probability_summary(all_probabilities),
        "speech_label_probability_summary": probability_summary(speech_probabilities),
        "nonspeech_label_probability_summary": probability_summary(nonspeech_probabilities),
        "counts": dict(aggregate),
        "skipped_rows": skipped,
    }
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"predictions={prediction_path}")
    print(f"metrics={metrics_path}")
    print(
        f"rows={rows} f1={metrics.f1:.4f} precision={metrics.precision:.4f} "
        f"recall={metrics.recall:.4f} predicted_positive_ratio={metrics.predicted_positive_ratio:.4f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    export_predictions(
        labels_path=Path(args.labels),
        feature_manifest_path=Path(args.feature_manifest),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
        threshold=args.threshold,
        include_probabilities=args.include_probabilities,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frame predictions from a FusionVAD-JA addition-fusion BiLSTM checkpoint."
    )
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json from build_feature_cache.py.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA addition BiLSTM checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "addition-bilstm-predictions"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
