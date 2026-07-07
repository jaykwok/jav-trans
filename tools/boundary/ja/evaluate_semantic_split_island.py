#!/usr/bin/env python3
"""Evaluate a Semantic Split v2 island checkpoint on sequence NPZ datasets.

Reports, per dataset and per dataset_role domain, the deployment-aligned cut
precision/recall/F1 at the checkpoint-calibrated (or overridden) thresholds,
the continue false-cut rate, and the paired-cut isolation rate, plus an
optional threshold sweep table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.split_model import (  # noqa: E402
    SEMANTIC_SPLIT_V2_DEFAULT_DECISION,
    SEMANTIC_SPLIT_V2_SCHEMA,
    IslandCandidateSequenceNetwork,
)
from tools.boundary.ja.train_semantic_split_island_model import (  # noqa: E402
    evaluate_island_model,
    load_island_dataset,
    split_group_names,
)


def _load_model(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEMANTIC_SPLIT_V2_SCHEMA:
        raise ValueError(
            f"expected {SEMANTIC_SPLIT_V2_SCHEMA!r} checkpoint, got {payload.get('schema')!r}"
        )
    model = IslandCandidateSequenceNetwork(**dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    normalization = {
        key: np.asarray(payload["normalization"][key], dtype=np.float32)
        for key in ("frame_mean", "frame_std", "scalar_mean", "scalar_std")
    }
    decision = {
        **SEMANTIC_SPLIT_V2_DEFAULT_DECISION,
        **dict(payload.get("decision_config") or {}),
    }
    return model, normalization, decision


def _dataset_names(data, partition: str) -> list[str]:
    if partition == "all":
        return sorted(data["groups"])
    if partition == "val":
        _train, val = split_group_names(data)
        return val
    if partition == "train":
        train, _val = split_group_names(data)
        return train
    raise ValueError(f"unsupported partition: {partition}")


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap()
    model, normalization, decision = _load_model(Path(args.checkpoint), args.device)
    normal = (
        args.normal_cut_threshold
        if args.normal_cut_threshold is not None
        else float(decision["normal_cut_threshold"])
    )
    short = (
        args.short_core_cut_threshold
        if args.short_core_cut_threshold is not None
        else float(decision["short_core_cut_threshold"])
    )
    report: dict[str, dict] = {}
    for entry in args.dataset:
        name, _, raw_path = entry.rpartition("=")
        path = Path(raw_path)
        name = name or path.stem
        data = load_island_dataset(path)
        names = _dataset_names(data, args.partition)
        metrics = evaluate_island_model(
            model,
            data,
            names,
            normalization=normalization,
            device=args.device,
            batch_islands=args.batch_islands,
            max_batch_candidates=args.max_batch_candidates,
            normal_cut_threshold=normal,
            short_core_cut_threshold=short,
            short_core_max_s=float(decision["short_core_max_s"]),
        )
        metrics.pop("gate_by_row", None)
        sweep = []
        if args.sweep:
            for threshold in np.arange(0.30, 0.96, 0.05):
                swept = evaluate_island_model(
                    model,
                    data,
                    names,
                    normalization=normalization,
                    device=args.device,
                    batch_islands=args.batch_islands,
                    max_batch_candidates=args.max_batch_candidates,
                    normal_cut_threshold=float(threshold),
                    short_core_cut_threshold=float(threshold),
                    short_core_max_s=float(decision["short_core_max_s"]),
                )
                swept.pop("gate_by_row", None)
                sweep.append(
                    {
                        "threshold": round(float(threshold), 2),
                        "macro_cut_f1": swept["macro_cut_f1"],
                        "pair_isolation_rate": swept["pair_isolation_rate"],
                        "domains": {
                            key: {
                                "cut_precision": value["cut_precision"],
                                "cut_recall": value["cut_recall"],
                            }
                            for key, value in swept["domains"].items()
                        },
                    }
                )
        report[name] = {
            "dataset": str(path),
            "partition": args.partition,
            "group_count": len(names),
            "thresholds": {"normal": normal, "short_core": short},
            "metrics": metrics,
            **({"threshold_sweep": sweep} if sweep else {}),
        }
    payload = {
        "schema": "semantic_split_island_eval_v1",
        "checkpoint": str(Path(args.checkpoint)),
        "decision_config": decision,
        "datasets": report,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic Split v2 island checkpoints per domain."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Sequence NPZ path, optionally prefixed as name=path. Repeatable.",
    )
    parser.add_argument("--partition", choices=("val", "train", "all"), default="val")
    parser.add_argument("--normal-cut-threshold", type=float)
    parser.add_argument("--short-core-cut-threshold", type=float)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--batch-islands", type=int, default=8)
    parser.add_argument("--max-batch-candidates", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run(parse_args())
