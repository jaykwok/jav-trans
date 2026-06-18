#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.ja import (  # noqa: E402
    FeatureScorerTrainConfig,
    load_label_records,
    train_feature_frame_scorer,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"feature manifest row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def _checkpoint_name(rows: list[dict[str, Any]], explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    ptm = str(rows[0].get("ptm") or "").strip() if rows else ""
    if not ptm:
        return "speech_boundary_ja_feature_scorer.pt"
    return f"speech_boundary_ja_feature_scorer.{qwen_asr_repo_tag(ptm)}.pt"


def run(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    feature_manifest_path = Path(args.feature_manifest)
    output_dir = Path(args.output_dir)
    records = load_label_records(labels_path)
    rows = _read_jsonl(feature_manifest_path)
    metrics = train_feature_frame_scorer(
        records=records,
        feature_manifest_rows=rows,
        output_dir=output_dir,
        config=FeatureScorerTrainConfig(
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            state_size=args.state_size,
            num_heads=args.num_heads,
            n_groups=args.n_groups,
            chunk_size=args.chunk_size,
            bidirectional=not args.unidirectional,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            cut_positive_weight=args.cut_positive_weight,
            cut_negative_weight=args.cut_negative_weight,
            cut_loss_weight=args.cut_loss_weight,
            cut_min_gap_s=args.cut_min_gap_s,
            cut_boundary_radius_frames=args.cut_boundary_radius_frames,
            focal_gamma=args.focal_gamma,
            eval_ratio=args.eval_ratio,
            threshold=args.threshold,
            cut_threshold=args.cut_threshold,
            max_eval_windows=args.max_eval_windows,
        ),
        labels_path=str(labels_path),
        feature_manifest_path=str(feature_manifest_path),
        checkpoint_name=_checkpoint_name(rows, args.checkpoint_name),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "speech_boundary_ja_mamba2_frame_boundary_scorer_train_summary_v3",
                "labels": str(labels_path),
                "feature_manifest": str(feature_manifest_path),
                "metrics": asdict(metrics),
                "runtime_status": {
                    "default_replaced": False,
                    "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
                    "note": "Candidate scorer is runtime-loadable only through an explicit repo-id map.",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint={metrics.checkpoint}")
    print(f"metrics={metrics.metrics_path}")
    print(f"summary={summary_path}")
    print("default_replaced=false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a runtime-loadable SpeechBoundary-JA Mamba2 speech+cut frame boundary scorer v3."
    )
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="Feature cache manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint and metrics.")
    parser.add_argument(
        "--checkpoint-name",
        default="",
        help="Checkpoint file name. Default appends the feature manifest PTM repo id tag.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=15.0)
    parser.add_argument("--cut-positive-weight", type=float, default=4.0)
    parser.add_argument("--cut-negative-weight", type=float, default=1.0)
    parser.add_argument("--cut-loss-weight", type=float, default=1.0)
    parser.add_argument("--cut-min-gap-s", type=float, default=0.5)
    parser.add_argument("--cut-boundary-radius-frames", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cut-threshold", type=float, default=0.5)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.num_layers <= 0:
        parser.error("--num-layers must be positive")
    if args.state_size <= 0:
        parser.error("--state-size must be positive")
    if args.num_heads <= 0:
        parser.error("--num-heads must be positive")
    if args.n_groups <= 0:
        parser.error("--n-groups must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.positive_weight <= 0.0:
        parser.error("--positive-weight must be positive")
    if args.negative_weight <= 0.0:
        parser.error("--negative-weight must be positive")
    if args.cut_positive_weight <= 0.0:
        parser.error("--cut-positive-weight must be positive")
    if args.cut_negative_weight <= 0.0:
        parser.error("--cut-negative-weight must be positive")
    if args.cut_loss_weight <= 0.0:
        parser.error("--cut-loss-weight must be positive")
    if args.cut_min_gap_s < 0.0:
        parser.error("--cut-min-gap-s must be non-negative")
    if args.cut_boundary_radius_frames < 0:
        parser.error("--cut-boundary-radius-frames must be non-negative")
    if args.focal_gamma < 0.0:
        parser.error("--focal-gamma must be non-negative")
    if not 0.0 <= args.eval_ratio < 1.0:
        parser.error("--eval-ratio must be in [0, 1)")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.cut_threshold <= 1.0:
        parser.error("--cut-threshold must be in [0, 1]")
    if args.max_eval_windows <= 0:
        parser.error("--max-eval-windows must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
