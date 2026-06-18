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
            dropout=args.dropout,
            eval_ratio=args.eval_ratio,
            threshold=args.threshold,
            max_eval_windows=args.max_eval_windows,
        ),
        labels_path=str(labels_path),
        feature_manifest_path=str(feature_manifest_path),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "speech_boundary_ja_feature_scorer_train_summary_v1",
                "labels": str(labels_path),
                "feature_manifest": str(feature_manifest_path),
                "metrics": asdict(metrics),
                "runtime_status": {
                    "default_replaced": False,
                    "opt_in_env": "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT",
                    "note": "Candidate scorer is runtime-loadable only when explicitly enabled.",
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
        description="Train a runtime-loadable SpeechBoundary-JA feature frame scorer."
    )
    parser.add_argument("--labels", required=True, help="SpeechBoundary-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="Feature cache manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint and metrics.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.dropout < 0.0:
        parser.error("--dropout must be non-negative")
    if not 0.0 <= args.eval_ratio < 1.0:
        parser.error("--eval-ratio must be in [0, 1)")
    if args.max_eval_windows <= 0:
        parser.error("--max-eval-windows must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
