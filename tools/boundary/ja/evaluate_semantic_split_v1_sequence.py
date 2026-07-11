#!/usr/bin/env python3
"""Replay a legacy Semantic Split v1 checkpoint on grouped v2 datasets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.split_model import load_semantic_split_verifier  # noqa: E402
from tools.boundary.ja.train_semantic_split_island_model import (  # noqa: E402
    _bootstrap_f1_ci,
    load_island_dataset,
    split_group_names,
)


def _dataset_names(data: dict[str, Any], partition: str) -> list[str]:
    if partition == "all":
        return sorted(data["groups"])
    train, val = split_group_names(data)
    return val if partition == "val" else train


def adapt_legacy_frames(
    frames: np.ndarray,
    *,
    raw_ptm_dim: int,
    expected_frame_dim: int,
) -> np.ndarray:
    raw_frame_dim = int(frames.shape[-1])
    non_ptm_dim = raw_frame_dim - raw_ptm_dim
    legacy_ptm_dim = expected_frame_dim - non_ptm_dim
    if raw_ptm_dim <= 0 or non_ptm_dim < 0 or legacy_ptm_dim <= 0:
        raise ValueError(
            f"cannot adapt raw frame dim {raw_frame_dim} with PTM dim {raw_ptm_dim} "
            f"to legacy frame dim {expected_frame_dim}"
        )
    if legacy_ptm_dim > raw_ptm_dim:
        raise ValueError("legacy PTM dimension exceeds raw PTM dimension")
    return np.concatenate(
        (frames[..., :legacy_ptm_dim], frames[..., raw_ptm_dim:]), axis=-1
    )


def summarize(
    data: dict[str, Any], names: list[str], accepted: dict[int, bool]
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "continue_total": 0, "continue_cut": 0}
    )
    pair_hits: dict[int, list[bool]] = defaultdict(list)
    islands: list[tuple[str, int, int, int]] = []
    for name in names:
        indexes = data["groups"][name]
        domain = str(data["dataset_roles"][int(indexes[0])])
        island_tp = island_fp = island_fn = 0
        for index in indexes.tolist():
            label = int(data["labels"][index])
            decision = bool(accepted[index])
            row = counts[str(data["dataset_roles"][index])]
            if label == 0:
                if decision:
                    row["tp"] += 1
                    island_tp += 1
                else:
                    row["fn"] += 1
                    island_fn += 1
            elif label == 1:
                row["continue_total"] += 1
                if decision:
                    row["fp"] += 1
                    row["continue_cut"] += 1
                    island_fp += 1
            pair_id = int(data["pair_ids"][index])
            if pair_id >= 0 and label == 0:
                pair_hits[pair_id].append(decision)
        islands.append((domain, island_tp, island_fp, island_fn))

    domains: dict[str, dict[str, float | int]] = {}
    for domain, row in sorted(counts.items()):
        precision = row["tp"] / max(1, row["tp"] + row["fp"])
        recall = row["tp"] / max(1, row["tp"] + row["fn"])
        domains[domain] = {
            "cut_precision": precision,
            "cut_recall": recall,
            "cut_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "continue_false_cut": row["continue_cut"] / max(1, row["continue_total"]),
            "cut_truth": row["tp"] + row["fn"],
        }
    real_domains = sorted(domain for domain in counts if domain.startswith("real_"))
    pool_domains = real_domains or sorted(counts)
    pooled = {
        key: sum(counts[domain][key] for domain in pool_domains)
        for key in ("tp", "fp", "fn", "continue_total", "continue_cut")
    }
    precision = pooled["tp"] / max(1, pooled["tp"] + pooled["fp"])
    recall = pooled["tp"] / max(1, pooled["tp"] + pooled["fn"])
    complete_pairs = [hits for hits in pair_hits.values() if len(hits) >= 2]
    return {
        "domains": domains,
        "macro_cut_f1": float(np.mean([row["cut_f1"] for row in domains.values()])),
        "pooled_real": {
            "domains": pool_domains,
            "cut_precision": precision,
            "cut_recall": recall,
            "cut_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "continue_false_cut": pooled["continue_cut"] / max(1, pooled["continue_total"]),
            "cut_truth": pooled["tp"] + pooled["fn"],
            **_bootstrap_f1_ci(
                [
                    (tp, fp, fn)
                    for domain, tp, fp, fn in islands
                    if domain in set(pool_domains)
                ]
            ),
        },
        "pair_isolation_rate": sum(all(hits[:2]) for hits in complete_pairs)
        / max(1, len(complete_pairs)),
        "complete_pair_count": len(complete_pairs),
    }


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap()
    verifier = load_semantic_split_verifier(
        args.checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    data = load_island_dataset(Path(args.dataset))
    names = _dataset_names(data, args.partition)
    indexes = np.concatenate([data["groups"][name] for name in names])
    expected_dim = int(np.asarray(verifier.normalization["frame_mean"]).shape[-1])
    accepted: dict[int, bool] = {}
    for start in range(0, indexes.size, args.batch_size):
        batch_indexes = indexes[start : start + args.batch_size]
        frames = adapt_legacy_frames(
            data["frames"][batch_indexes],
            raw_ptm_dim=args.raw_ptm_dim,
            expected_frame_dim=expected_dim,
        )
        decisions = verifier.decide(
            frame_features=frames,
            scalar_features=data["scalars"][batch_indexes],
        )
        accepted.update(
            (int(index), decision.label == "cut")
            for index, decision in zip(batch_indexes.tolist(), decisions)
        )
    payload = {
        "schema": "semantic_split_v1_sequence_replay_v1",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "partition": args.partition,
        "raw_ptm_dim": args.raw_ptm_dim,
        "metrics": summarize(data, names, accepted),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit legacy v1 replay on grouped Semantic Split datasets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--raw-ptm-dim", type=int, required=True)
    parser.add_argument("--partition", choices=("val", "train", "all"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
