#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import evaluate_addition_fusion_classifier, load_label_records


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    feature_rows = json.loads(Path(args.feature_manifest).read_text(encoding="utf-8"))
    if not isinstance(feature_rows, list):
        raise ValueError("feature manifest must be a JSON list")

    thresholds = [
        round(args.min_threshold + (index * args.step), 6)
        for index in range(int(round((args.max_threshold - args.min_threshold) / args.step)) + 1)
    ]
    results = []
    for threshold in thresholds:
        metrics = evaluate_addition_fusion_classifier(
            records=records,
            feature_manifest_rows=feature_rows,
            checkpoint_path=Path(args.checkpoint),
            output_dir=output_dir / f"threshold-{threshold:.3f}",
            device=args.device,
            threshold=threshold,
        )
        payload = asdict(metrics)
        results.append(payload)
        print(
            f"threshold={threshold:.3f} f1={metrics.f1:.4f} "
            f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
            f"predicted_positive_ratio={metrics.predicted_positive_ratio:.4f}",
            flush=True,
        )

    eligible = [
        row
        for row in results
        if float(row["precision"]) >= args.min_precision and float(row["recall"]) >= args.min_recall
    ]
    candidates = eligible if eligible else results
    best = max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            -abs(float(row["predicted_positive_ratio"]) - float(row["positive_ratio"])),
        ),
    )
    summary = {
        "labels": args.labels,
        "feature_manifest": args.feature_manifest,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "selection": "constraints" if eligible else "best_f1",
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "best_threshold": float(best["threshold"]),
        "best": best,
        "results": results,
    }
    summary_path = output_dir / "threshold_calibration.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"best_threshold={summary['best_threshold']:.3f}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FusionVAD-JA addition-BiLSTM thresholds on a validation split.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA validation label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json for the validation split.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA addition BiLSTM checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "threshold-calibration"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
