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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (  # noqa: E402
    AdditionFusionEndpointBiLSTM,
    effective_frame_weights,
    endpoint_targets_from_record,
    frame_classification_counts,
    load_cached_feature,
    load_label_records,
    metrics_from_frame_counts,
)

OUTPUT_NAMES = ("speech", "start", "end", "cut_drop", "cut_point")


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


def load_endpoint_model(*, checkpoint_path: Path, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config") or {})
    model_type = str(checkpoint.get("model_type") or "")
    if model_type and model_type != "addition_endpoint_bilstm":
        raise ValueError(f"unsupported endpoint checkpoint model_type={model_type!r}")
    model = AdditionFusionEndpointBiLSTM(
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


def thresholded(values: np.ndarray, *, threshold: float) -> np.ndarray:
    return (values >= float(threshold)).astype(np.int32)


def output_metrics_payload(
    *,
    counts: Mapping[str, int],
    rows: int,
    threshold: float,
) -> dict[str, Any]:
    metrics = metrics_from_frame_counts(counts=counts, windows=rows, threshold=threshold)
    return asdict(metrics)


def export_predictions(
    *,
    labels_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
    speech_threshold: float = 0.5,
    start_threshold: float = 0.5,
    end_threshold: float = 0.5,
    cut_threshold: float = 0.5,
    apply_cut_to_speech: bool = False,
    include_probabilities: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(labels_path)
    feature_rows = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("feature manifest must be a JSON list")

    thresholds = {
        "speech": float(speech_threshold),
        "start": float(start_threshold),
        "end": float(end_threshold),
        "cut_drop": float(cut_threshold),
        "cut_point": float(cut_threshold),
    }
    model, checkpoint = load_endpoint_model(checkpoint_path=checkpoint_path, device=device)
    checkpoint_config = dict(checkpoint.get("config") or {})
    boundary_radius_frames = int(checkpoint_config.get("boundary_radius_frames", 1))
    cut_min_gap_s = float(checkpoint_config.get("cut_min_gap_s", 0.5))
    cut_boundary_radius_frames = int(checkpoint_config.get("cut_boundary_radius_frames", 0))

    prediction_path = output_dir / "predictions.jsonl"
    skipped = []
    aggregate: dict[str, Counter[str]] = {name: Counter() for name in OUTPUT_NAMES}
    probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    target_probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    non_target_probability_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    rows = 0
    torch_device = torch.device(device)

    with prediction_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for row_index, row in enumerate(feature_rows):
            label_index = int(row["label_index"])
            if label_index < 0 or label_index >= len(records):
                skipped.append({"row_index": row_index, "reason": "label_index_out_of_range"})
                continue
            record = records[label_index]
            audio_id = str(row.get("audio_id") or record.audio_id)
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

            weights = effective_frame_weights(record)
            frame_count = min(whisper.shape[0], mfcc.shape[0], len(record.speech_frames), len(weights))
            if frame_count <= 0:
                skipped.append({"row_index": row_index, "audio_id": audio_id, "reason": "empty_frames"})
                continue

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
            predictions = {name: thresholded(probabilities[name], threshold=thresholds[name]) for name in OUTPUT_NAMES}
            raw_speech_predictions = predictions["speech"].copy()
            if apply_cut_to_speech:
                cut_union = np.maximum(predictions["cut_drop"], predictions["cut_point"])
                predictions["speech"] = np.logical_and(
                    predictions["speech"] > 0,
                    cut_union <= 0,
                ).astype(np.int32)

            labels = np.asarray(record.speech_frames[:frame_count], dtype=np.int32)
            start_targets, end_targets, cut_drop_targets, cut_point_targets = endpoint_targets_from_record(
                record,
                frame_count=frame_count,
                boundary_radius_frames=boundary_radius_frames,
                cut_min_gap_s=cut_min_gap_s,
                cut_boundary_radius_frames=cut_boundary_radius_frames,
            )
            targets = {
                "speech": labels,
                "start": start_targets.astype(np.int32),
                "end": end_targets.astype(np.int32),
                "cut_drop": cut_drop_targets.astype(np.int32),
                "cut_point": cut_point_targets.astype(np.int32),
            }
            active_weights = np.asarray(weights[:frame_count], dtype=np.float32)
            active = active_weights > 0.0
            output_counts: dict[str, dict[str, int]] = {}
            for name in OUTPUT_NAMES:
                counts = frame_classification_counts(
                    labels=targets[name],
                    predictions=predictions[name],
                    weights=active_weights,
                )
                output_counts[name] = counts
                aggregate[name].update(counts)
                active_probabilities = probabilities[name][active]
                active_targets = targets[name][active] > 0
                probability_values[name].extend(float(value) for value in active_probabilities)
                target_probability_values[name].extend(float(value) for value in active_probabilities[active_targets])
                non_target_probability_values[name].extend(
                    float(value) for value in active_probabilities[np.logical_not(active_targets)]
                )

            payload: dict[str, Any] = {
                "audio_id": record.audio_id,
                "source": record.source,
                "label_quality": record.label_quality,
                "duration_s": record.duration_s,
                "frame_hop_s": record.frame_hop_s,
                "thresholds": thresholds,
                "apply_cut_to_speech": bool(apply_cut_to_speech),
                "frame_count": int(frame_count),
                "speech_frames": predictions["speech"].astype(int).tolist(),
                "start_frames": predictions["start"].astype(int).tolist(),
                "end_frames": predictions["end"].astype(int).tolist(),
                "cut_drop_frames": predictions["cut_drop"].astype(int).tolist(),
                "cut_point_frames": predictions["cut_point"].astype(int).tolist(),
                "cut_frames": np.maximum(predictions["cut_drop"], predictions["cut_point"]).astype(int).tolist(),
                "probability_summary": {
                    name: probability_summary(probabilities[name][active].tolist()) for name in OUTPUT_NAMES
                },
                "target_probability_summary": {
                    name: probability_summary(probabilities[name][active][targets[name][active] > 0].tolist())
                    for name in OUTPUT_NAMES
                },
                "non_target_probability_summary": {
                    name: probability_summary(probabilities[name][active][targets[name][active] <= 0].tolist())
                    for name in OUTPUT_NAMES
                },
                "counts": output_counts["speech"],
                "output_counts": output_counts,
            }
            if apply_cut_to_speech:
                payload["speech_frames_raw"] = raw_speech_predictions.astype(int).tolist()
            if include_probabilities:
                payload["probabilities"] = {
                    name: [float(value) for value in probabilities[name]] for name in OUTPUT_NAMES
                }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1

    metrics_path = output_dir / "prediction_metrics.json"
    output_metrics = {
        name: output_metrics_payload(
            counts=dict(aggregate[name]),
            rows=rows,
            threshold=thresholds[name],
        )
        for name in OUTPUT_NAMES
    }
    summary = {
        "labels": str(labels_path),
        "feature_manifest": str(feature_manifest_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "thresholds": thresholds,
        "apply_cut_to_speech": bool(apply_cut_to_speech),
        "boundary_radius_frames": boundary_radius_frames,
        "cut_min_gap_s": cut_min_gap_s,
        "predictions": str(prediction_path),
        "rows": rows,
        "skipped": len(skipped),
        "metrics": output_metrics["speech"],
        "output_metrics": output_metrics,
        "probability_summary": {name: probability_summary(probability_values[name]) for name in OUTPUT_NAMES},
        "target_probability_summary": {
            name: probability_summary(target_probability_values[name]) for name in OUTPUT_NAMES
        },
        "non_target_probability_summary": {
            name: probability_summary(non_target_probability_values[name]) for name in OUTPUT_NAMES
        },
        "counts": dict(aggregate["speech"]),
        "output_counts": {name: dict(aggregate[name]) for name in OUTPUT_NAMES},
        "skipped_rows": skipped,
    }
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    speech_metrics = output_metrics["speech"]
    cut_drop_metrics = output_metrics["cut_drop"]
    cut_point_metrics = output_metrics["cut_point"]
    print(f"predictions={prediction_path}")
    print(f"metrics={metrics_path}")
    print(
        f"rows={rows} speech_f1={speech_metrics['f1']:.4f} "
        f"speech_precision={speech_metrics['precision']:.4f} speech_recall={speech_metrics['recall']:.4f} "
        f"cut_drop_f1={cut_drop_metrics['f1']:.4f} cut_drop_recall={cut_drop_metrics['recall']:.4f} "
        f"cut_point_f1={cut_point_metrics['f1']:.4f} cut_point_recall={cut_point_metrics['recall']:.4f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    export_predictions(
        labels_path=Path(args.labels),
        feature_manifest_path=Path(args.feature_manifest),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
        speech_threshold=args.speech_threshold,
        start_threshold=args.start_threshold,
        end_threshold=args.end_threshold,
        cut_threshold=args.cut_threshold,
        apply_cut_to_speech=args.apply_cut_to_speech,
        include_probabilities=args.include_probabilities,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frame predictions from a FusionVAD-JA endpoint/boundary refiner checkpoint."
    )
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json from build_feature_cache.py.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA endpoint refiner checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--start-threshold", type=float, default=0.5)
    parser.add_argument("--end-threshold", type=float, default=0.5)
    parser.add_argument("--cut-threshold", type=float, default=0.5)
    parser.add_argument(
        "--apply-cut-to-speech",
        action="store_true",
        help="Set speech frames to 0 where cut_frames are active; research-only gate approximation.",
    )
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "endpoint-refiner-predictions"),
    )
    args = parser.parse_args(argv)
    for name in ("speech_threshold", "start_threshold", "end_threshold", "cut_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
