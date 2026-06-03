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

from vad.fusionvad_ja import ImitationTrainConfig, train_imitation_classifier  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> None:
    feature_rows = json.loads(Path(args.feature_manifest).read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("--feature-manifest must point to a JSON list")
    imitation_rows = load_jsonl(Path(args.imitation_targets))
    metrics = train_imitation_classifier(
        feature_manifest_rows=feature_rows,
        imitation_rows=imitation_rows,
        output_dir=Path(args.output_dir),
        config=ImitationTrainConfig(
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
            split_loss_weight=args.split_loss_weight,
            drop_gap_loss_weight=args.drop_gap_loss_weight,
            split_positive_loss_weight=args.split_positive_loss_weight,
            drop_gap_positive_loss_weight=args.drop_gap_positive_loss_weight,
            save_interval_steps=args.save_interval_steps,
            window_frames=args.window_frames,
            positive_window_ratio=args.positive_window_ratio,
            positive_jitter_frames=args.positive_jitter_frames,
            balanced_frame_loss=not args.no_balanced_frame_loss,
        ),
    )
    print(f"checkpoint={metrics.checkpoint}")
    print(f"metrics={metrics.metrics_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FusionVAD-JA v1.21 imitation split/drop-gap head.")
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--imitation-targets", required=True)
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
    parser.add_argument("--split-loss-weight", type=float, default=1.0)
    parser.add_argument("--drop-gap-loss-weight", type=float, default=1.0)
    parser.add_argument("--split-positive-loss-weight", type=float, default=32.0)
    parser.add_argument("--drop-gap-positive-loss-weight", type=float, default=16.0)
    parser.add_argument("--save-interval-steps", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=128)
    parser.add_argument("--positive-window-ratio", type=float, default=0.9)
    parser.add_argument("--positive-jitter-frames", type=int, default=16)
    parser.add_argument("--no-balanced-frame-loss", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "imitation-head-train"),
    )
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    for name in (
        "split_loss_weight",
        "drop_gap_loss_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "split_positive_loss_weight",
        "drop_gap_positive_loss_weight",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.save_interval_steps < 0:
        parser.error("--save-interval-steps must be non-negative")
    if args.window_frames < 0:
        parser.error("--window-frames must be non-negative")
    if not 0.0 <= args.positive_window_ratio <= 1.0:
        parser.error("--positive-window-ratio must be in [0, 1]")
    if args.positive_jitter_frames < 0:
        parser.error("--positive-jitter-frames must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
