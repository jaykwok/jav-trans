#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools" / "fusionvad_ja"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_addition_predictions import export_predictions as export_addition_predictions  # noqa: E402
from export_endpoint_refiner_predictions import export_predictions as export_endpoint_predictions  # noqa: E402
from vad_recall_metrics import compute_recall_metrics  # noqa: E402


DEFAULT_OPERATING_POINT = "fusionvad-ja-v1.17-endpoint-refiner-boundary32768"
DEFAULT_THRESHOLD = 0.020
DEFAULT_CUT_THRESHOLD = 0.960
DEFAULT_PAD_S = 0.2
DEFAULT_FRAME_HOP_S = 0.02


def checkpoint_model_type(checkpoint_path: Path) -> str:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return str(checkpoint.get("model_type") or "addition_bilstm")


def export_operating_point(
    *,
    labels_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    threshold: float,
    cut_threshold: float,
    apply_cut_to_speech: bool,
    pad_s: float,
    frame_hop_s: float,
    operating_point: str,
    include_probabilities: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "frame-predictions"
    model_type = checkpoint_model_type(checkpoint_path)
    if model_type == "addition_endpoint_bilstm":
        prediction_summary = export_endpoint_predictions(
            labels_path=labels_path,
            feature_manifest_path=feature_manifest_path,
            checkpoint_path=checkpoint_path,
            output_dir=predictions_dir,
            device=device,
            speech_threshold=threshold,
            start_threshold=0.5,
            end_threshold=0.5,
            cut_threshold=cut_threshold,
            apply_cut_to_speech=apply_cut_to_speech,
            include_probabilities=include_probabilities,
        )
    else:
        prediction_summary = export_addition_predictions(
            labels_path=labels_path,
            feature_manifest_path=feature_manifest_path,
            checkpoint_path=checkpoint_path,
            output_dir=predictions_dir,
            device=device,
            threshold=threshold,
            include_probabilities=include_probabilities,
        )
    recall_metrics_path = output_dir / "high_recall_metrics.json"
    recall_summary = compute_recall_metrics(
        labels_path=labels_path,
        predictions_path=Path(prediction_summary["predictions"]),
        output_path=recall_metrics_path,
        pad_s=pad_s,
        frame_hop_s=frame_hop_s,
        prediction_threshold=threshold,
    )
    summary = {
        "operating_point": operating_point,
        "labels": str(labels_path),
        "feature_manifest": str(feature_manifest_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "threshold": float(threshold),
        "cut_threshold": float(cut_threshold),
        "apply_cut_to_speech": bool(apply_cut_to_speech),
        "checkpoint_model_type": model_type,
        "pad_s": float(pad_s),
        "frame_hop_s": float(frame_hop_s),
        "predictions": prediction_summary["predictions"],
        "raw_prediction_metrics": str(predictions_dir / "prediction_metrics.json"),
        "high_recall_metrics": str(recall_metrics_path),
        "raw": prediction_summary["metrics"],
        "padded": {
            "precision": recall_summary["precision"],
            "recall": recall_summary["recall"],
            "f1": recall_summary["f1"],
            "positive_ratio": recall_summary["positive_ratio"],
            "predicted_positive_ratio": recall_summary["predicted_positive_ratio"],
            "missed_speech_seconds": recall_summary["missed_speech_seconds"],
            "missed_speech_segments": recall_summary["missed_speech_segments"],
            "extra_audio_seconds": recall_summary["extra_audio_seconds"],
            "extra_audio_ratio": recall_summary["extra_audio_ratio"],
            "evaluated": recall_summary["evaluated"],
            "skipped": recall_summary["skipped"],
        },
    }
    summary_path = output_dir / "operating_point_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"operating_point_summary={summary_path}")
    print(
        f"operating_point={operating_point} threshold={threshold:g} pad_s={pad_s:g} "
        f"recall={summary['padded']['recall']:.4f} "
        f"missed_speech_seconds={summary['padded']['missed_speech_seconds']:.2f} "
        f"extra_audio_ratio={summary['padded']['extra_audio_ratio']:.4f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    export_operating_point(
        labels_path=Path(args.labels),
        feature_manifest_path=Path(args.feature_manifest),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device=args.device,
        threshold=args.threshold,
        cut_threshold=args.cut_threshold,
        apply_cut_to_speech=args.apply_cut_to_speech,
        pad_s=args.pad_s,
        frame_hop_s=args.frame_hop_s,
        operating_point=args.operating_point,
        include_probabilities=args.include_probabilities,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fixed FusionVAD-JA high-recall operating point for reproducible evaluation."
    )
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json from build_feature_cache.py.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA addition BiLSTM checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cut-threshold", type=float, default=DEFAULT_CUT_THRESHOLD)
    parser.add_argument("--apply-cut-to-speech", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad-s", type=float, default=DEFAULT_PAD_S)
    parser.add_argument("--frame-hop-s", type=float, default=DEFAULT_FRAME_HOP_S)
    parser.add_argument("--operating-point", default=DEFAULT_OPERATING_POINT)
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "operating-point"),
    )
    args = parser.parse_args(argv)
    if args.threshold < 0.0:
        parser.error("--threshold must be non-negative")
    if args.cut_threshold < 0.0:
        parser.error("--cut-threshold must be non-negative")
    if args.pad_s < 0.0:
        parser.error("--pad-s must be non-negative")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
