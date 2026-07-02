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

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.speech_train import (  # noqa: E402
    SpeechIslandTrainConfig,
    train_speech_island_scorer,
)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> None:
    labels = Path(args.labels)
    manifest = Path(args.feature_manifest)
    rows = _rows(manifest)
    repo_tag = qwen_asr_repo_tag(str(rows[0].get("ptm") or ""))
    metrics = train_speech_island_scorer(
        records=read_jsonl(labels),
        feature_manifest_rows=rows,
        output_dir=Path(args.output_dir),
        config=SpeechIslandTrainConfig(
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            focal_gamma=args.focal_gamma,
            max_train_frames=args.max_train_frames,
            max_eval_frames=args.max_eval_frames,
            max_eval_windows=args.max_eval_windows,
            log_every=args.log_every,
            ptm_dim=args.ptm_dim,
        ),
        labels_path=str(labels),
        feature_manifest_path=str(manifest),
        normalization_checkpoint=args.normalization_checkpoint,
        checkpoint_name=f"speech_island_scorer_v8.{repo_tag}.pt",
    )
    summary = Path(args.output_dir) / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "speech_island_scorer_train_summary_v8",
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint={metrics.checkpoint}")
    print(f"metrics={metrics.metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train speech-only SpeechIslandScorer v8.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=15.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-frames", type=int, default=1024)
    parser.add_argument("--max-eval-frames", type=int, default=1024)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ptm-dim", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
