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

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.semantic_speech_train import (  # noqa: E402
    SemanticSpeechTrainConfig,
    train_semantic_speech_scorer,
)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> None:
    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Semantic Speech Scorer v9 promotion is 1.7B-only")
    apply_vram_safety_cap(0.95)
    labels = Path(args.labels)
    manifest = Path(args.feature_manifest)
    metrics = train_semantic_speech_scorer(
        records=read_jsonl(labels),
        feature_manifest_rows=_rows(manifest),
        output_dir=Path(args.output_dir),
        config=SemanticSpeechTrainConfig(
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            discardable_weight=args.discardable_weight,
            semantic_target_weight=args.semantic_target_weight,
            unsure_weight=args.unsure_weight,
            membership_outside_weight=args.membership_outside_weight,
            membership_inside_weight=args.membership_inside_weight,
            membership_unsure_weight=args.membership_unsure_weight,
            membership_loss_weight=args.membership_loss_weight,
            focal_gamma=args.focal_gamma,
            max_train_frames=args.max_train_frames,
            max_eval_frames=args.max_eval_frames,
            max_eval_windows=args.max_eval_windows,
            log_every=args.log_every,
            raw_ptm_dim=args.raw_ptm_dim,
            projected_ptm_dim=args.projected_ptm_dim,
        ),
        labels_path=str(labels),
        feature_manifest_path=str(manifest),
        checkpoint_name=(
            f"semantic_speech_scorer_v9.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
        ),
        warm_start_checkpoint=args.warm_start_checkpoint,
    )
    summary = Path(args.output_dir) / "summary.json"
    summary.write_text(
        json.dumps(
            {"schema": "semantic_speech_scorer_train_summary_v9", "metrics": asdict(metrics)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(asdict(metrics), ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1.7B Semantic Speech Scorer v9.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--discardable-weight", type=float, default=1.0)
    parser.add_argument("--semantic-target-weight", type=float, default=3.0)
    parser.add_argument("--unsure-weight", type=float, default=1.5)
    parser.add_argument("--membership-outside-weight", type=float, default=1.0)
    parser.add_argument("--membership-inside-weight", type=float, default=2.0)
    parser.add_argument("--membership-unsure-weight", type=float, default=1.5)
    parser.add_argument("--membership-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-frames", type=int, default=1024)
    parser.add_argument("--max-eval-frames", type=int, default=1024)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--raw-ptm-dim", type=int, default=2048)
    parser.add_argument("--projected-ptm-dim", type=int, default=128)
    parser.add_argument("--warm-start-checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
