#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

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
)
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.outer_refiner_v3 import (  # noqa: E402
    OUTER_EDGE_REFINER_V3_DATASET_CONTRACT,
    OUTER_EDGE_REFINER_V3_UPSTREAM_SCORER_SCHEMA,
    build_outer_edge_refiner_v3_checkpoint,
)
from pipeline.memory_safety import runtime_memory_snapshot  # noqa: E402
from tools.boundary.ja.edge_frame_dataset import (  # noqa: E402
    load_edge_row,
    read_edge_rows,
)


BINARY_SCORER_V10_SCHEMA = OUTER_EDGE_REFINER_V3_UPSTREAM_SCORER_SCHEMA
PARTITIONS = ("train", "val", "test")


def validate_dataset_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Outer v3 dataset is empty")
    source_partitions: dict[str, set[str]] = defaultdict(set)
    core_partitions: dict[str, set[str]] = defaultdict(set)
    core_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    for row in rows:
        source_id = str(row.get("source_id") or "")
        core_id = str(row.get("core_id") or "")
        partition = str(row.get("partition") or "")
        if not source_id or not core_id:
            raise ValueError("Outer v3 rows require independent source_id/core_id")
        if partition not in PARTITIONS:
            raise ValueError(f"Outer v3 row has invalid partition: {partition!r}")
        if row.get("input_distribution") != OUTER_EDGE_REFINER_V3_DATASET_CONTRACT[
            "input_distribution"
        ]:
            raise ValueError("Outer v3 rows must come from post-Scorer v10 islands")
        if row.get("scorer_schema") != BINARY_SCORER_V10_SCHEMA:
            raise ValueError("Outer v3 rows require the binary Scorer v10 schema")
        if int(row.get("frame_count") or 0) <= 0:
            raise ValueError("Outer v3 rows require a positive frame_count")
        source_partitions[source_id].add(partition)
        core_partitions[core_id].add(partition)
        core_counts[core_id] += 1
        partition_counts[partition] += 1
    if any(len(values) != 1 for values in source_partitions.values()):
        raise ValueError("Outer v3 source identity crosses dataset partitions")
    if any(len(values) != 1 for values in core_partitions.values()):
        raise ValueError("Outer v3 core identity crosses dataset partitions")
    if max(core_counts.values(), default=0) > 1:
        raise ValueError("Outer v3 requires max core use count <= 1")
    if any(partition_counts[name] <= 0 for name in PARTITIONS):
        raise ValueError("Outer v3 requires fixed train/val/test partitions")
    return {
        "source_count": len(source_partitions),
        "core_count": len(core_partitions),
        "max_core_use_count": max(core_counts.values(), default=0),
        "partition_counts": dict(partition_counts),
    }


def load_binary(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, canonical, weights = load_edge_row(row)
    if int(features.shape[0]) != int(row["frame_count"]):
        raise ValueError(f"Outer v3 frame_count mismatch: {row.get('core_id')}")
    return features, canonical_to_binary_labels(canonical), weights


def compute_normalization(rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    first, _labels, _weights = load_binary(rows[0])
    total = 0
    feature_sum = np.zeros(first.shape[1], dtype=np.float64)
    square_sum = np.zeros(first.shape[1], dtype=np.float64)
    for row in rows:
        features, labels, weights = load_binary(row)
        valid = (labels != BINARY_EDGE_IGNORE_INDEX) & (weights > 0.0)
        values = features[valid].astype(np.float64)
        feature_sum += values.sum(axis=0)
        square_sum += np.square(values).sum(axis=0)
        total += int(values.shape[0])
    if total <= 0:
        raise ValueError("Outer v3 train partition has no definite frames")
    mean = feature_sum / total
    variance = square_sum / total - np.square(mean)
    return {
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32).tolist(),
    }


def summarize_partition_label_presence(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], Counter[str]]:
    canonical_counts: Counter[str] = Counter()
    presence = {
        name: {"semantic_rows": 0, "all_background_rows": 0}
        for name in PARTITIONS
    }
    for row in rows:
        _features, canonical, _weights = load_edge_row(row)
        canonical_counts.update(
            background=int(np.sum(canonical == 0)),
            semantic_core=int(np.sum(canonical == 1)),
            unsure=int(np.sum(canonical == 2)),
        )
        row_presence = presence[str(row["partition"])]
        if np.any(canonical == 1):
            row_presence["semantic_rows"] += 1
        elif np.any(canonical == 0):
            row_presence["all_background_rows"] += 1
    if any(
        not values["semantic_rows"] or not values["all_background_rows"]
        for values in presence.values()
    ):
        raise ValueError(
            "Outer v3 requires semantic and all-background rows in every partition"
        )
    return presence, canonical_counts


def frame_budget_batches(
    rows: Sequence[dict[str, Any]], *, max_padded_frames: int
) -> list[list[dict[str, Any]]]:
    if max_padded_frames <= 0:
        raise ValueError("max_padded_frames must be positive")
    ordered = sorted(rows, key=lambda row: int(row["frame_count"]))
    result: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    maximum = 0
    for row in ordered:
        proposed = max(maximum, int(row["frame_count"]))
        if batch and proposed * (len(batch) + 1) > max_padded_frames:
            result.append(batch)
            batch = []
            maximum = 0
        batch.append(row)
        maximum = max(maximum, int(row["frame_count"]))
    if batch:
        result.append(batch)
    return result


def pad_batch(rows, normalization):
    import torch

    loaded = [load_binary(row) for row in rows]
    maximum = max(features.shape[0] for features, _labels, _weights in loaded)
    feature_dim = int(loaded[0][0].shape[1])
    features = np.zeros((len(rows), maximum, feature_dim), dtype=np.float32)
    labels = np.full((len(rows), maximum), BINARY_EDGE_IGNORE_INDEX, dtype=np.int64)
    weights = np.zeros((len(rows), maximum), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=np.int64)
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(normalization["feature_std"], dtype=np.float32), 1e-6)
    for index, (row_features, row_labels, row_weights) in enumerate(loaded):
        count = int(row_features.shape[0])
        features[index, :count] = (row_features - mean) / std
        labels[index, :count] = row_labels
        weights[index, :count] = row_weights
        mask[index, :count] = 1
    return tuple(torch.from_numpy(value) for value in (features, labels, weights, mask))


def evaluate(model, rows, normalization, device, *, tolerance_frames, max_padded_frames):
    import torch

    count = start_hits = end_hits = true_speech_deletions = 0
    background_rows = background_drops = tp = fp = fn = 0
    start_errors: list[int] = []
    end_errors: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in frame_budget_batches(rows, max_padded_frames=max_padded_frames):
            features, labels, _weights, mask = pad_batch(batch, normalization)
            probabilities = torch.softmax(
                model(
                    features.to(device),
                    attention_mask=mask.to(device),
                ),
                dim=-1,
            ).cpu().numpy()
            for index, row in enumerate(batch):
                length = int(row["frame_count"])
                truth = labels[index, :length].numpy()
                valid = truth != BINARY_EDGE_IGNORE_INDEX
                predicted = np.argmax(probabilities[index, :length], axis=1)
                tp += int(np.sum((predicted[valid] == 1) & (truth[valid] == 1)))
                fp += int(np.sum((predicted[valid] == 1) & (truth[valid] == 0)))
                fn += int(np.sum((predicted[valid] == 0) & (truth[valid] == 1)))
                truth_core = np.flatnonzero(truth == 1)
                prediction_core = np.flatnonzero((predicted == 1) & valid)
                if not truth_core.size:
                    background_rows += 1
                    background_drops += int(not prediction_core.size)
                    continue
                count += 1
                if not prediction_core.size:
                    true_speech_deletions += 1
                    continue
                start_error = abs(int(prediction_core[0]) - int(truth_core[0]))
                end_error = abs(int(prediction_core[-1]) - int(truth_core[-1]))
                start_errors.append(start_error)
                end_errors.append(end_error)
                start_hits += int(start_error <= tolerance_frames)
                end_hits += int(end_error <= tolerance_frames)
    return {
        "semantic_row_count": count,
        "background_row_count": background_rows,
        "start_coverage": start_hits / max(count, 1),
        "end_coverage": end_hits / max(count, 1),
        "start_mae_frames": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_frames": float(np.mean(end_errors)) if end_errors else None,
        "semantic_precision": tp / max(tp + fp, 1),
        "semantic_recall": tp / max(tp + fn, 1),
        "background_drop_recall": background_drops / max(background_rows, 1),
        "true_speech_deletion_count": true_speech_deletions,
    }


def _memory_snapshot(device) -> dict[str, Any]:
    import torch

    snapshot = runtime_memory_snapshot(require_shared_vram=device.type == "cuda")
    if device.type != "cuda":
        snapshot = {
            key: value
            for key, value in snapshot.items()
            if not key.startswith("shared_vram")
        }
        snapshot.update(
            shared_vram_mb=0.0,
            shared_vram_monitor="not_applicable_cpu_stage",
        )
    if device.type == "cuda":
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated(device) / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved(device) / 2**20, 3),
        )
    if snapshot["physical_ram_used_mb"] > snapshot["physical_ram_budget_mb"]:
        raise MemoryError("Outer v3 exceeded the 0.95 physical RAM budget")
    shared = snapshot.get("shared_vram_mb")
    if device.type == "cuda" and isinstance(shared, (int, float)) and shared > 0.0:
        raise MemoryError("Outer v3 shared VRAM spill is a soft OOM")
    return snapshot


def release_gate_fields(numeric_gate_pass: bool) -> dict[str, Any]:
    return {
        "numeric_gate_pass": bool(numeric_gate_pass),
        "gate_pass": False,
        "promotion_ready": False,
        "manual_zero_clipping_gate": "required_before_promotion",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap(0.95)
    rows = read_edge_rows(Path(args.dataset_manifest))
    dataset_summary = validate_dataset_rows(rows)
    by_partition = {
        name: [row for row in rows if row["partition"] == name]
        for name in PARTITIONS
    }
    normalization = compute_normalization(by_partition["train"])
    first, _labels, _weights = load_binary(by_partition["train"][0])
    position_dim = int(first.shape[1]) - args.raw_ptm_dim - args.mfcc_dim
    if position_dim <= 0:
        raise ValueError("Outer v3 feature width must include relative position")
    model_config = {
        "ptm_input_dim": args.raw_ptm_dim,
        "ptm_projected_dim": args.projected_ptm_dim,
        "mfcc_dim": args.mfcc_dim,
        "position_dim": position_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "state_size": 32,
        "num_heads": 4,
        "head_dim": 64,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
        "valid_prefix_bidirectional": True,
        "output_dim": 2,
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Outer v3 requested CUDA but CUDA is unavailable")
    _memory_snapshot(device)
    model = BinaryFrameEdgeNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(args.seed)
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_state = None
    losses: list[float] = []
    memory_snapshots: list[dict[str, Any]] = []
    partition_label_presence, canonical_counts = summarize_partition_label_presence(rows)
    dataset_summary["partition_label_presence"] = partition_label_presence
    started = time.monotonic()
    train_batches = frame_budget_batches(
        by_partition["train"], max_padded_frames=args.max_batch_frames
    )
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(train_batches) == 0:
            rng.shuffle(train_batches)
        batch = train_batches[(step - 1) % len(train_batches)]
        features, labels, source_weights, mask = pad_batch(batch, normalization)
        model.train()
        logits = model(
            features.to(device), attention_mask=mask.to(device)
        )
        target = labels.to(device)
        valid = target != BINARY_EDGE_IGNORE_INDEX
        loss_rows = F.cross_entropy(
            logits.transpose(1, 2), target,
            reduction="none", ignore_index=BINARY_EDGE_IGNORE_INDEX,
        )
        class_weights = torch.where(
            target == 0,
            torch.as_tensor(args.background_weight, device=device),
            torch.as_tensor(args.semantic_weight, device=device),
        )
        weights = source_weights.to(device) * class_weights
        loss = (loss_rows[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % args.eval_interval == 0 or step == args.max_steps:
            val = evaluate(
                model, by_partition["val"], normalization, device,
                tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
                max_padded_frames=args.max_batch_frames,
            )
            score = (
                float(val["true_speech_deletion_count"] == 0),
                min(
                    val["start_coverage"],
                    val["end_coverage"],
                    val["background_drop_recall"],
                ),
                val["semantic_recall"],
            )
            memory_snapshots.append(_memory_snapshot(device))
            print(json.dumps({"step": step, "loss": losses[-1], "val": val}), flush=True)
            if score > best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Outer v3 training produced no evaluated checkpoint")
    model.load_state_dict(best_state)
    val = evaluate(
        model, by_partition["val"], normalization, device,
        tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
        max_padded_frames=args.max_batch_frames,
    )
    test = evaluate(
        model, by_partition["test"], normalization, device,
        tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
        max_padded_frames=args.max_batch_frames,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / (
        f"outer_edge_refiner_v3.{qwen_asr_repo_tag(QWEN_ASR_17B_REPO_ID)}.pt"
    )
    torch.save(
        build_outer_edge_refiner_v3_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "raw_ptm_dim": args.raw_ptm_dim,
                "learned_ptm_projected_dim": args.projected_ptm_dim,
                "mfcc_dim": args.mfcc_dim,
                "relative_position_dim": position_dim,
                "frame_hop_s": args.frame_hop_s,
            },
            normalization=normalization,
            metadata={
                "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                "dataset_manifest": args.dataset_manifest,
                "dataset_summary": dataset_summary,
                "trained_steps": args.max_steps,
                "best_step": best_step,
                "canonical_label_counts": dict(canonical_counts),
                "excluded_training_count": int(canonical_counts["unsure"]),
                "training_initialization": "random",
                "checkpoint_selection": "val_outer_acoustic_edge_300ms_coverage_v1",
                "class_weights": {
                    "background": args.background_weight,
                    "semantic_core": args.semantic_weight,
                },
            },
        ),
        checkpoint,
    )
    numeric_gate_pass = (
        min(
            val["start_coverage"], val["end_coverage"],
            test["start_coverage"], test["end_coverage"],
            val["background_drop_recall"], test["background_drop_recall"],
        ) >= 0.95
        and val["true_speech_deletion_count"] == 0
        and test["true_speech_deletion_count"] == 0
    )
    allocator = {}
    if device.type == "cuda":
        allocator = {
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
            "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
        }
    summary = {
        "schema": "outer_edge_refiner_v3_binary_training_summary_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "best_step": best_step,
        "mean_train_loss": float(np.mean(losses)),
        "val": val,
        "test": test,
        "dataset": dataset_summary,
        "canonical_label_counts": dict(canonical_counts),
        "excluded_training_count": int(canonical_counts["unsure"]),
        **release_gate_fields(numeric_gate_pass),
        "training_initialization": "random",
        "loss": "weighted_cross_entropy",
        "memory_snapshots": memory_snapshots,
        **allocator,
        "elapsed_s": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del (
        optimizer,
        model,
        best_state,
        logits,
        target,
        valid,
        loss_rows,
        class_weights,
        weights,
        loss,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    summary["memory_after_release"] = _memory_snapshot(device)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1.7B binary Outer v3 from post-Scorer v10 islands."
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-ptm-dim", type=int, default=2048)
    parser.add_argument("--projected-ptm-dim", type=int, default=128)
    parser.add_argument("--mfcc-dim", type=int, default=40)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--max-batch-frames", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--tolerance-s", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
