#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import (
    FeatureTrainConfig,
    load_label_records,
    train_addition_fusion_classifier,
)


def load_training_inputs(
    *,
    labels_paths: list[str],
    feature_manifest_paths: list[str],
) -> tuple[list, list[dict]]:
    if len(labels_paths) != len(feature_manifest_paths):
        raise ValueError("--labels and --feature-manifest must be provided the same number of times")
    records = []
    feature_rows = []
    for labels_path, feature_manifest_path in zip(labels_paths, feature_manifest_paths, strict=True):
        label_offset = len(records)
        batch_records = load_label_records(Path(labels_path))
        records.extend(batch_records)
        batch_rows = json.loads(Path(feature_manifest_path).read_text(encoding="utf-8"))
        if not isinstance(batch_rows, list):
            raise ValueError(f"feature manifest must be a JSON list: {feature_manifest_path}")
        for row in batch_rows:
            adjusted = dict(row)
            adjusted["label_index"] = int(adjusted["label_index"]) + label_offset
            feature_rows.append(adjusted)
    return records, feature_rows


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, feature_rows = load_training_inputs(
        labels_paths=args.labels,
        feature_manifest_paths=args.feature_manifest,
    )
    metrics = train_addition_fusion_classifier(
        records=records,
        feature_manifest_rows=feature_rows,
        output_dir=output_dir,
        config=FeatureTrainConfig(
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            fusion_dim=args.fusion_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            max_trainable_parameters=args.max_trainable_parameters,
            log_interval_steps=args.log_interval_steps,
            batch_size=args.batch_size,
            init_checkpoint=args.init_checkpoint,
            positive_loss_weight=args.positive_loss_weight,
            boundary_loss_weight=args.boundary_loss_weight,
            gap_loss_weight=args.gap_loss_weight,
        ),
    )
    print(f"checkpoint={metrics.checkpoint}")
    print(f"metrics={metrics.metrics_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FusionVAD-JA addition-fusion BiLSTM from cached features.")
    parser.add_argument("--labels", action="append", required=True, help="FusionVAD-JA label JSONL. Repeatable.")
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        help="feature_manifest.json from build_feature_cache.py. Repeatable; order must match --labels.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-trainable-parameters", type=int, default=2_000_000)
    parser.add_argument("--log-interval-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-checkpoint", help="Optional checkpoint used to initialize model weights.")
    parser.add_argument(
        "--positive-loss-weight",
        type=float,
        default=1.0,
        help="BCE positive-class loss weight. Values >1 bias training toward recall.",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=0.0,
        help="Opt-in boundary transition loss weight for speech-island endpoint training.",
    )
    parser.add_argument(
        "--gap-loss-weight",
        type=float,
        default=0.0,
        help="Opt-in loss weight that suppresses speech probability inside internal non-speech gaps.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "addition-bilstm-train"))
    args = parser.parse_args(argv)
    if args.positive_loss_weight <= 0.0:
        parser.error("--positive-loss-weight must be positive")
    if args.boundary_loss_weight < 0.0:
        parser.error("--boundary-loss-weight must be non-negative")
    if args.gap_loss_weight < 0.0:
        parser.error("--gap-loss-weight must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
