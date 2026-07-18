#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.binary_edge_refiner import (  # noqa: E402
    BINARY_EDGE_IGNORE_INDEX,
    BinaryFrameEdgeNetwork,
    canonical_to_binary_labels,
    decode_binary_edge_logits,
)
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.inner_refiner_v2 import build_inner_edge_refiner_v2_checkpoint  # noqa: E402
from tools.boundary.ja.edge_frame_dataset import (  # noqa: E402
    load_edge_row,
    normalize_edge_features,
    read_edge_rows,
)


@lru_cache(maxsize=64)
def load_source_features(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        return source["ptm"].astype(np.float32), source["mfcc"].astype(np.float32)


def load_binary(row: dict):
    if row.get("label_path"):
        ptm, mfcc = load_source_features(str(row["source_feature_path"]))
        with np.load(row["label_path"]) as payload:
            labels = payload["labels"].astype(np.int64)
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        ptm = ptm[start:end]
        mfcc = mfcc[start:end]
        total = min(len(ptm), len(mfcc), len(labels))
        if total <= 0 or total != len(labels):
            raise ValueError(f"Inner v2 feature/label slice mismatch: {row.get('row_id')}")
        position = (
            np.arange(total, dtype=np.float32) / max(1, total - 1)
        ).reshape(-1, 1)
        auxiliary = [position]
        if row.get("acoustic_start_frame") is not None:
            acoustic_start = int(row["acoustic_start_frame"]) - start
            acoustic_end = int(row["acoustic_end_frame"]) - start
            acoustic_length = max(1, acoustic_end - acoustic_start)
            acoustic_position = (
                (np.arange(total, dtype=np.float32) - acoustic_start)
                / max(1, acoustic_length - 1)
            )
            auxiliary.append(np.clip(acoustic_position, -1.0, 2.0).reshape(-1, 1))
        features = np.concatenate((ptm[:total], mfcc[:total], *auxiliary), axis=1)
        return features, canonical_to_binary_labels(labels[:total]), np.ones(total, dtype=np.float32)
    features, canonical, weights = load_edge_row(row)
    return features, canonical_to_binary_labels(canonical), weights


def compute_normalization(rows: list[dict]) -> dict[str, list[float]]:
    first, _labels, _weights = load_binary(rows[0])
    feature_sum = np.zeros(first.shape[1], dtype=np.float64)
    square_sum = np.zeros(first.shape[1], dtype=np.float64)
    frame_count = 0
    for row in rows:
        features, labels, _weights = load_binary(row)
        valid = labels != BINARY_EDGE_IGNORE_INDEX
        values = features[valid].astype(np.float64)
        feature_sum += values.sum(axis=0)
        square_sum += np.square(values).sum(axis=0)
        frame_count += len(values)
    mean = feature_sum / max(frame_count, 1)
    variance = square_sum / max(frame_count, 1) - np.square(mean)
    return {
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32).tolist(),
    }


def evaluate(model, rows, normalization, device, tolerance_frames: int) -> dict:
    import torch

    start_hits = end_hits = count = all_background = 0
    start_errors, end_errors = [], []
    tp = fp = fn = 0
    model.eval()
    with torch.inference_mode():
        for row in rows:
            features, labels, _weights = load_binary(row)
            truth = np.flatnonzero(labels == 1)
            if not truth.size:
                continue
            logits = model(torch.from_numpy(normalize_edge_features(features, normalization)).unsqueeze(0).to(device))[0].cpu().numpy()
            predicted = np.argmax(logits, axis=1)
            valid = labels != BINARY_EDGE_IGNORE_INDEX
            tp += int(np.sum((predicted[valid] == 1) & (labels[valid] == 1)))
            fp += int(np.sum((predicted[valid] == 1) & (labels[valid] == 0)))
            fn += int(np.sum((predicted[valid] == 0) & (labels[valid] == 1)))
            try:
                start, end = decode_binary_edge_logits(logits, raw_start_s=0.0, raw_end_s=float(len(labels)), frame_hop_s=1.0)
                ps, pe = int(round(start)), int(round(end))
            except ValueError:
                all_background += 1
                count += 1
                continue
            start_error, end_error = abs(ps - int(truth[0])), abs(pe - int(truth[-1] + 1))
            start_errors.append(start_error); end_errors.append(end_error)
            start_hits += int(start_error <= tolerance_frames); end_hits += int(end_error <= tolerance_frames); count += 1
    return {
        "count": count,
        "start_coverage": start_hits / max(count, 1), "end_coverage": end_hits / max(count, 1),
        "start_mae_frames": float(np.mean(start_errors)) if start_errors else float("inf"),
        "end_mae_frames": float(np.mean(end_errors)) if end_errors else float("inf"),
        "semantic_precision": tp / max(tp + fp, 1), "semantic_recall": tp / max(tp + fn, 1),
        "all_background_count": all_background,
    }


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap(0.95)
    rows = read_edge_rows(Path(args.dataset_manifest))
    train_rows = [row for row in rows if str(row.get("partition")) == "train"]
    val_rows = [row for row in rows if str(row.get("partition")) == "val"]
    test_rows = [row for row in rows if str(row.get("partition")) == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Inner v2 requires fixed train/val/test partitions")
    normalization = compute_normalization(train_rows)
    first, _labels, _weights = load_binary(train_rows[0])
    position_dim = int(first.shape[1]) - args.raw_ptm_dim - args.mfcc_dim
    model_config = {
        "ptm_input_dim": args.raw_ptm_dim, "ptm_projected_dim": args.projected_ptm_dim,
        "mfcc_dim": args.mfcc_dim, "position_dim": position_dim, "hidden_size": args.hidden_size,
        "num_layers": args.num_layers, "state_size": 32, "num_heads": 4, "head_dim": 64,
        "n_groups": 2, "conv_kernel": 4, "chunk_size": 8, "bidirectional": True, "output_dim": 2,
    }
    if position_dim <= 0:
        raise ValueError("Inner v2 frame feature dimension mismatch")
    device = torch.device(args.device)
    model = BinaryFrameEdgeNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    train_order = np.empty(0, dtype=np.int64)
    train_position = 0
    counts = Counter()
    for row in rows:
        if row.get("label_path"):
            with np.load(row["label_path"]) as payload:
                canonical = payload["labels"].astype(np.int64)
        else:
            _f, canonical, _w = load_edge_row(row)
        counts.update(
            background=int(np.sum(canonical == 0)),
            semantic_target=int(np.sum(canonical == 1)),
            unsure=int(np.sum(canonical == 2)),
        )
    best_score = (-1.0, -1.0, -1.0); best_step = 0; best_state = None; started = time.monotonic()
    for step in range(1, args.max_steps + 1):
        if train_position >= len(train_order):
            train_order = rng.permutation(len(train_rows))
            train_position = 0
        row = train_rows[int(train_order[train_position])]
        train_position += 1
        features, labels, source_weights = load_binary(row)
        valid = labels != BINARY_EDGE_IGNORE_INDEX
        if not np.any(valid):
            continue
        model.train()
        logits = model(torch.from_numpy(normalize_edge_features(features, normalization)).unsqueeze(0).to(device))[0]
        target = torch.from_numpy(labels).to(device)
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=BINARY_EDGE_IGNORE_INDEX)
        class_weights = np.where(
            labels == 0, float(args.background_weight), float(args.semantic_weight)
        ).astype(np.float32)
        weights = torch.from_numpy(
            source_weights.astype(np.float32) * class_weights
        ).to(device)
        valid_t = target != BINARY_EDGE_IGNORE_INDEX
        loss = (ce[valid_t] * weights[valid_t]).sum() / weights[valid_t].sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % args.eval_interval == 0 or step == args.max_steps:
            val = evaluate(model, val_rows, normalization, device, int(round(args.tolerance_s / args.frame_hop_s)))
            score = (float(val["all_background_count"] == 0), min(val["start_coverage"], val["end_coverage"]), val["semantic_recall"])
            print(json.dumps({"step": step, "loss": float(loss.detach()), "val": val}), flush=True)
            if score > best_score:
                best_score, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Inner v2 produced no checkpoint")
    model.load_state_dict(best_state)
    train = evaluate(model, train_rows, normalization, device, 15)
    val = evaluate(model, val_rows, normalization, device, 15); test = evaluate(model, test_rows, normalization, device, 15)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"inner_edge_refiner_v2.{qwen_asr_repo_tag(QWEN_ASR_17B_REPO_ID)}.pt"
    torch.save(build_inner_edge_refiner_v2_checkpoint(
        model=model, model_config=model_config,
        feature_config={"raw_ptm_dim": args.raw_ptm_dim, "learned_ptm_projected_dim": args.projected_ptm_dim, "mfcc_dim": args.mfcc_dim, "relative_position_dim": position_dim, "frame_hop_s": args.frame_hop_s, "acoustic_refinement": True},
        normalization=normalization,
        metadata={"ptm_repo_id": QWEN_ASR_17B_REPO_ID, "dataset_manifest": args.dataset_manifest, "trained_steps": args.max_steps, "best_step": best_step, "canonical_label_counts": dict(counts), "excluded_training_count": int(counts["unsure"]), "training_initialization": "random", "checkpoint_selection": "val_inner_acoustic_edge_300ms_coverage_v1", "class_weights": {"background": float(args.background_weight), "semantic_core": float(args.semantic_weight)}, "acoustic_refinement": True, "feeds_asr": True},
    ), checkpoint)
    summary = {"schema": "inner_edge_refiner_v2_binary_training_summary_v1", "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "best_step": best_step, "train": train, "val": val, "test": test, "canonical_label_counts": dict(counts), "excluded_training_count": int(counts["unsure"]), "gate_pass": min(val["start_coverage"], val["end_coverage"], test["start_coverage"], test["end_coverage"]) >= 0.95 and not train["all_background_count"] and not val["all_background_count"] and not test["all_background_count"], "acoustic_refinement": True, "feeds_asr": True, "elapsed_s": time.monotonic() - started}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1.7B binary acoustic Inner Edge Refiner v2.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--raw-ptm-dim", type=int, default=2048); parser.add_argument("--projected-ptm-dim", type=int, default=128); parser.add_argument("--mfcc-dim", type=int, default=40); parser.add_argument("--frame-hop-s", type=float, default=0.02); parser.add_argument("--max-steps", type=int, default=1500); parser.add_argument("--eval-interval", type=int, default=100); parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--background-weight", type=float, default=5.1); parser.add_argument("--semantic-weight", type=float, default=1.0); parser.add_argument("--hidden-size", type=int, default=128); parser.add_argument("--num-layers", type=int, default=2); parser.add_argument("--tolerance-s", type=float, default=0.3); parser.add_argument("--seed", type=int, default=17); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
