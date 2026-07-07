#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.split_model import load_semantic_split_verifier  # noqa: E402


LABELS = ("cut", "continue", "unsure")


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    cut_threshold: float,
) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    gated = predicted.copy()
    gated[(predicted == 0) & (probabilities[:, 0] < cut_threshold)] = 2
    confusion = np.zeros((3, 3), dtype=np.int64)
    gated_confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, actual, gated_actual in zip(truth, predicted, gated):
        confusion[int(expected), int(actual)] += 1
        gated_confusion[int(expected), int(gated_actual)] += 1

    def cut_metrics(matrix: np.ndarray) -> tuple[float, float]:
        tp = int(matrix[0, 0])
        recall = tp / max(1, int(matrix[0].sum()))
        precision = tp / max(1, int(matrix[:, 0].sum()))
        return precision, recall

    precision, recall = cut_metrics(confusion)
    gated_precision, gated_recall = cut_metrics(gated_confusion)
    return {
        "accuracy": float((predicted == truth).mean()),
        "cut_precision": precision,
        "cut_recall": recall,
        "gated_cut_threshold": cut_threshold,
        "gated_cut_precision": gated_precision,
        "gated_cut_recall": gated_recall,
        "confusion": confusion.tolist(),
    }


def fused_probabilities(
    main: np.ndarray,
    expert: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    return (1.0 - alpha) * main + alpha * expert


def _probabilities(model: Any, frames: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for start in range(0, len(frames), 256):
        rows.extend(
            [
                [decision.p_cut, decision.p_continue, decision.p_unsure]
                for decision in model.decide(
                    frame_features=frames[start : start + 256],
                    scalar_features=scalars[start : start + 256],
                )
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _load_eval_rows(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = np.load(args.dataset)
    frames = bundle["frame_features"].astype(np.float32)
    scalars = bundle["scalar_features"].astype(np.float32)
    if args.index_labels_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.index_labels_jsonl).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if str(row.get("label") or "") in LABELS]
        indexes = np.asarray([int(row["index"]) for row in rows], dtype=np.int64)
        truth = np.asarray(
            [LABELS.index(str(row["label"])) for row in rows],
            dtype=np.int64,
        )
        return frames[indexes], scalars[indexes], truth
    truth = bundle["labels"].astype(np.int64)
    if args.partition:
        mask = bundle["partitions"].astype(str) == args.partition
        frames = frames[mask]
        scalars = scalars[mask]
        truth = truth[mask]
    return frames, scalars, truth


def run(args: argparse.Namespace) -> None:
    frames, scalars, truth = _load_eval_rows(args)
    main = load_semantic_split_verifier(
        args.main_checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    expert = load_semantic_split_verifier(
        args.expert_checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    main_probabilities = _probabilities(main, frames, scalars)
    expert_probabilities = _probabilities(expert, frames, scalars)
    result = {
        "schema": "semantic_split_expert_fusion_evaluation_v1",
        "dataset": args.dataset,
        "partition": args.partition,
        "count": int(truth.size),
        "main_checkpoint": args.main_checkpoint,
        "expert_checkpoint": args.expert_checkpoint,
        "alphas": {
            f"{alpha:g}": classification_metrics(
                truth,
                fused_probabilities(
                    main_probabilities,
                    expert_probabilities,
                    alpha=alpha,
                ),
                cut_threshold=args.cut_threshold,
            )
            for alpha in args.alpha
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index-labels-jsonl", default="")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--main-checkpoint", required=True)
    parser.add_argument("--expert-checkpoint", required=True)
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--cut-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.alpha:
        args.alpha = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    if any(not 0.0 <= value <= 1.0 for value in args.alpha):
        parser.error("--alpha must be in [0, 1]")
    if args.index_labels_jsonl:
        args.partition = ""
    return args


if __name__ == "__main__":
    run(parse_args())
