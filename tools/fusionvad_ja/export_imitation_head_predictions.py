#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (  # noqa: E402
    AdditionFusionImitationBiLSTM,
    frame_classification_counts,
    load_cached_feature,
    metrics_from_frame_counts,
)
from vad.fusionvad_ja.train import resize_binary_frames  # noqa: E402

OUTPUT_NAMES = ("split", "drop_gap")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def load_imitation_model(*, checkpoint_path: Path, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = str(checkpoint.get("model_type") or "")
    if model_type and model_type != "addition_imitation_bilstm":
        raise ValueError(f"unsupported imitation checkpoint model_type={model_type!r}")
    config = dict(checkpoint.get("config") or {})
    model = AdditionFusionImitationBiLSTM(
        whisper_dim=int(checkpoint["whisper_dim"]),
        mfcc_dim=int(checkpoint["mfcc_dim"]),
        fusion_dim=int(config.get("fusion_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_thresholds(values: list[str]) -> list[float]:
    thresholds: list[float] = []
    for raw in values:
        for part in raw.split(","):
            if not part.strip():
                continue
            value = float(part)
            if not 0.0 <= value <= 1.0:
                raise ValueError("thresholds must be in [0, 1]")
            thresholds.append(value)
    return sorted(set(thresholds))


def metrics_payload(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    rows: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int32)
        counts = frame_classification_counts(
            labels=labels.astype(np.int32),
            predictions=predictions,
            weights=np.ones(labels.shape[0], dtype=np.float32),
        )
        payload[f"{threshold:.4f}"] = asdict(
            metrics_from_frame_counts(counts=counts, windows=rows, threshold=threshold)
        )
    return payload


def export_predictions(
    *,
    feature_manifest_path: Path,
    imitation_targets_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    thresholds: list[float],
    include_probabilities: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("feature manifest must be a JSON list")
    target_rows = load_jsonl(imitation_targets_path)
    targets_by_audio = {str(row.get("audio_id") or ""): row for row in target_rows}
    model, checkpoint = load_imitation_model(checkpoint_path=checkpoint_path, device=device)
    torch_device = torch.device(device)

    skipped: list[dict[str, Any]] = []
    rows = 0
    probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    target_probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    non_target_probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    target_values: dict[str, list[int]] = {name: [] for name in OUTPUT_NAMES}
    aggregate_at_default: dict[str, Counter[str]] = {name: Counter() for name in OUTPUT_NAMES}
    prediction_path = output_dir / "predictions.jsonl"

    with prediction_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for row_index, row in enumerate(feature_rows):
            audio_id = str(row.get("audio_id") or "")
            target = targets_by_audio.get(audio_id)
            if not target:
                skipped.append({"row_index": row_index, "audio_id": audio_id, "reason": "missing_target"})
                continue
            try:
                whisper, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            except Exception as exc:
                skipped.append(
                    {
                        "row_index": row_index,
                        "audio_id": audio_id,
                        "reason": "feature_load_error",
                        "error": str(exc),
                    }
                )
                continue
            split_targets = np.asarray(target.get("split_frames") or [], dtype=np.int32)
            drop_gap_targets = np.asarray(target.get("drop_gap_frames") or [], dtype=np.int32)
            frame_count = min(whisper.shape[0], mfcc.shape[0])
            if frame_count <= 0:
                skipped.append({"row_index": row_index, "audio_id": audio_id, "reason": "empty_frames"})
                continue
            split_targets = resize_binary_frames(split_targets, frame_count).astype(np.int32)
            drop_gap_targets = resize_binary_frames(drop_gap_targets, frame_count).astype(np.int32)

            whisper_tensor = torch.from_numpy(np.ascontiguousarray(whisper[:frame_count], dtype=np.float32)).to(
                torch_device
            ).unsqueeze(0)
            mfcc_tensor = torch.from_numpy(np.ascontiguousarray(mfcc[:frame_count], dtype=np.float32)).to(
                torch_device
            ).unsqueeze(0)
            logits = model(whisper_tensor, mfcc_tensor)
            probabilities = {
                name: torch.sigmoid(logits[name]).detach().cpu().numpy().reshape(-1)[:frame_count]
                for name in OUTPUT_NAMES
            }
            targets = {
                "split": split_targets,
                "drop_gap": drop_gap_targets,
            }
            payload: dict[str, Any] = {
                "audio_id": audio_id,
                "frame_count": int(frame_count),
                "probability_summary": {
                    name: probability_summary(probabilities[name].tolist()) for name in OUTPUT_NAMES
                },
                "target_probability_summary": {
                    name: probability_summary(probabilities[name][targets[name] > 0].tolist())
                    for name in OUTPUT_NAMES
                },
                "non_target_probability_summary": {
                    name: probability_summary(probabilities[name][targets[name] <= 0].tolist())
                    for name in OUTPUT_NAMES
                },
                "positive_frames": {name: int(targets[name].sum()) for name in OUTPUT_NAMES},
            }
            if include_probabilities:
                payload["probabilities"] = {
                    name: [float(value) for value in probabilities[name]] for name in OUTPUT_NAMES
                }
                payload["targets"] = {name: targets[name].astype(int).tolist() for name in OUTPUT_NAMES}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

            for name in OUTPUT_NAMES:
                probability_values[name].extend(float(value) for value in probabilities[name])
                target_probability_values[name].extend(float(value) for value in probabilities[name][targets[name] > 0])
                non_target_probability_values[name].extend(
                    float(value) for value in probabilities[name][targets[name] <= 0]
                )
                target_values[name].extend(int(value) for value in targets[name])
                default_predictions = (probabilities[name] >= 0.5).astype(np.int32)
                aggregate_at_default[name].update(
                    frame_classification_counts(
                        labels=targets[name],
                        predictions=default_predictions,
                        weights=np.ones(frame_count, dtype=np.float32),
                    )
                )
            rows += 1

    probability_arrays = {name: np.asarray(probability_values[name], dtype=np.float32) for name in OUTPUT_NAMES}
    target_arrays = {name: np.asarray(target_values[name], dtype=np.int32) for name in OUTPUT_NAMES}
    output_metrics = {
        name: metrics_payload(
            labels=target_arrays[name],
            probabilities=probability_arrays[name],
            thresholds=thresholds,
            rows=rows,
        )
        for name in OUTPUT_NAMES
    }
    summary = {
        "feature_manifest": str(feature_manifest_path),
        "imitation_targets": str(imitation_targets_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "checkpoint_model_type": str(checkpoint.get("model_type") or ""),
        "thresholds": thresholds,
        "predictions": str(prediction_path),
        "rows": rows,
        "skipped": len(skipped),
        "probability_summary": {name: probability_summary(probability_values[name]) for name in OUTPUT_NAMES},
        "target_probability_summary": {
            name: probability_summary(target_probability_values[name]) for name in OUTPUT_NAMES
        },
        "non_target_probability_summary": {
            name: probability_summary(non_target_probability_values[name]) for name in OUTPUT_NAMES
        },
        "output_metrics": output_metrics,
        "default_threshold_counts": {name: dict(aggregate_at_default[name]) for name in OUTPUT_NAMES},
        "skipped_rows": skipped,
    }
    metrics_path = output_dir / "prediction_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"predictions={prediction_path}")
    print(f"metrics={metrics_path}")
    for name in OUTPUT_NAMES:
        best_threshold, best_metrics = max(
            output_metrics[name].items(),
            key=lambda item: (float(item[1]["f1"]), float(item[1]["recall"])),
        )
        target_p50 = summary["target_probability_summary"][name]["p50"]
        non_target_p50 = summary["non_target_probability_summary"][name]["p50"]
        print(
            f"{name}: best_threshold={best_threshold} f1={best_metrics['f1']:.4f} "
            f"precision={best_metrics['precision']:.4f} recall={best_metrics['recall']:.4f} "
            f"target_p50={target_p50:.4f} non_target_p50={non_target_p50:.4f}",
            flush=True,
        )
    return summary


def run(args: argparse.Namespace) -> None:
    export_predictions(
        feature_manifest_path=Path(args.feature_manifest),
        imitation_targets_path=Path(args.imitation_targets),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
        thresholds=parse_thresholds(args.threshold),
        include_probabilities=args.include_probabilities,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export/sweep FusionVAD-JA v1.21 imitation split/drop-gap head predictions."
    )
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--imitation-targets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threshold",
        action="append",
        default=["0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.40,0.50"],
        help="Threshold or comma-separated thresholds; can be repeated.",
    )
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "imitation-head-predictions"),
    )
    args = parser.parse_args(argv)
    try:
        parse_thresholds(args.threshold)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    run(parse_args())
