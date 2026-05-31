#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    metrics = evaluate_addition_fusion_classifier(
        records=records,
        feature_manifest_rows=feature_rows,
        checkpoint_path=Path(args.checkpoint),
        output_dir=output_dir,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"metrics={metrics.metrics_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FusionVAD-JA addition-fusion BiLSTM on cached features.")
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--feature-manifest", required=True, help="feature_manifest.json from build_feature_cache.py.")
    parser.add_argument("--checkpoint", required=True, help="FusionVAD-JA addition BiLSTM checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "addition-bilstm-eval"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
