#!/usr/bin/env python3
"""Evaluate a repo-bound Acoustic Split v4 binary checkpoint with event-run gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.split_model import load_acoustic_split_v4_planner  # noqa: E402
from tools.boundary.ja.train_acoustic_split_v4_model import (  # noqa: E402
    evaluate,
    gate_passes,
    partition_group_names,
)
from tools.boundary.ja.acoustic_split_v4_dataset import (  # noqa: E402
    load_island_dataset,
)


def run(args: argparse.Namespace) -> None:
    import torch

    planner = load_acoustic_split_v4_planner(
        args.checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    data = load_island_dataset(Path(args.dataset))
    partitions = partition_group_names(data)
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in planner.normalization.items()
    }
    metrics = {}
    for partition in ("val", "test"):
        row = evaluate(
            planner.model,
            data,
            partitions[partition],
            normalization=normalization,
            device=torch.device(planner.device),
            batch_islands=args.batch_islands,
            max_batch_candidates=args.max_batch_candidates,
        )
        row["gate_passed"] = gate_passes(row)
        metrics[partition] = row
    proposal_summary = {}
    if args.dataset_summary:
        proposal_summary = json.loads(Path(args.dataset_summary).read_text("utf-8"))
    payload = {
        "schema": "semantic_split_model_v4_event_gate_metrics_v1",
        "checkpoint": planner.signature(),
        "dataset": str(Path(args.dataset)),
        "decision_mode": "binary_argmax_cut",
        "event_contract": "consecutive_argmax_cut_run",
        "event_match": "ordered_candidate_truth_basin_no_time_threshold",
        "proposal_coverage": proposal_summary.get(
            "eligible_candidate_proposal_recall"
        ),
        "proposal_coverage_gate_passed": (
            proposal_summary.get("eligible_candidate_proposal_recall", 0.0) >= 0.95
            if proposal_summary
            else None
        ),
        "partitions": metrics,
        "promotion_ready": bool(
            metrics["val"]["gate_passed"]
            and metrics["test"]["gate_passed"]
            and (
                not proposal_summary
                or proposal_summary.get("eligible_candidate_proposal_recall", 0.0)
                >= 0.95
            )
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-summary", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-islands", type=int, default=8)
    parser.add_argument("--max-batch-candidates", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
