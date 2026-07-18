#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT,
    SPEECH_ISLAND_SCORER_V10_MODEL_ARCH,
    SPEECH_ISLAND_SCORER_V10_MFCC_DIM,
    SPEECH_ISLAND_SCORER_V10_PROJECTED_PTM_DIM,
    SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM,
    SPEECH_ISLAND_SCORER_V10_SCHEMA,
    BinarySpeechIslandScorerNetwork,
    build_speech_island_scorer_checkpoint,
)
from pipeline.memory_safety import runtime_memory_snapshot  # noqa: E402


PARTITIONS = ("train", "val", "test")
IGNORE_INDEX = -100


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_dataset_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Scorer v10 dataset is empty")
    source_partitions: dict[str, set[str]] = defaultdict(set)
    core_partitions: dict[str, set[str]] = defaultdict(set)
    core_counts: Counter[str] = Counter()
    background_ids: set[str] = set()
    partition_counts: Counter[str] = Counter()
    for row in rows:
        source_id = str(row.get("source_id") or "")
        partition = str(row.get("partition") or "")
        core_ids = [str(value) for value in row.get("core_ids") or ()]
        row_role = str(row.get("row_role") or "")
        background_id = str(row.get("background_id") or "")
        if not source_id:
            raise ValueError("Scorer v10 rows require source_id")
        if row_role == "speech" and not core_ids:
            raise ValueError("Scorer v10 speech rows require core_ids")
        if row_role == "all_background":
            if core_ids or not background_id:
                raise ValueError(
                    "Scorer v10 all-background rows require background_id and no core_ids"
                )
            if background_id in background_ids:
                raise ValueError("Scorer v10 background identity is duplicated")
            background_ids.add(background_id)
        elif row_role != "speech":
            raise ValueError("Scorer v10 row_role must be speech/all_background")
        if partition not in PARTITIONS:
            raise ValueError(f"Scorer v10 row has invalid partition: {partition!r}")
        if row.get("boundary_serialization_contract_id") != (
            SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT[
                "boundary_serialization_contract_id"
            ]
        ):
            raise ValueError("Scorer v10 rows require the central Boundary contract")
        if row.get("input_distribution") != SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT[
            "input_distribution"
        ]:
            raise ValueError("Scorer v10 rows require full source windows")
        if row.get("canonical_label_schema") != SPEECH_ISLAND_SCORER_V10_DATASET_CONTRACT[
            "canonical_label_schema"
        ]:
            raise ValueError("Scorer v10 rows require canonical frame labels")
        if row.get("ptm_repo_id") != QWEN_ASR_17B_REPO_ID:
            raise ValueError("Scorer v10 dataset is bound to the 1.7B PTM repo")
        if not str(row.get("feature_path") or "") or not str(row.get("label_path") or ""):
            raise ValueError("Scorer v10 rows require feature_path and label_path")
        if int(row.get("frame_count") or 0) <= 0:
            raise ValueError("Scorer v10 rows require a positive frame_count")
        if source_id in source_partitions:
            raise ValueError(f"Scorer v10 source is duplicated: {source_id!r}")
        source_partitions[source_id].add(partition)
        for core_id in core_ids:
            core_partitions[core_id].add(partition)
            core_counts[core_id] += 1
        partition_counts[partition] += 1
    if any(len(values) != 1 for values in core_partitions.values()):
        raise ValueError("Scorer v10 core identity crosses dataset partitions")
    if max(core_counts.values(), default=0) > 1:
        raise ValueError("Scorer v10 requires max core use count <= 1")
    if any(partition_counts[name] <= 0 for name in PARTITIONS):
        raise ValueError("Scorer v10 requires fixed train/val/test partitions")
    return {
        "source_count": len(source_partitions),
        "core_count": len(core_partitions),
        "background_identity_count": len(background_ids),
        "max_core_use_count": max(core_counts.values(), default=0),
        "partition_counts": dict(partition_counts),
    }


def load_binary_row(
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_path = Path(str(row["feature_path"]))
    label_path = Path(str(row["label_path"]))
    with np.load(feature_path) as features:
        ptm = np.asarray(features["ptm"], dtype=np.float32)
        mfcc = np.asarray(features["mfcc"], dtype=np.float32)
    with np.load(label_path) as labels:
        canonical = np.asarray(labels["canonical_labels"], dtype=np.int64)
        weights = np.asarray(
            labels["frame_weights"] if "frame_weights" in labels else np.ones(canonical.shape[0]),
            dtype=np.float32,
        )
    if ptm.ndim != 2 or mfcc.ndim != 2 or ptm.shape[0] != mfcc.shape[0]:
        raise ValueError(f"Scorer v10 feature shape mismatch: {row.get('source_id')}")
    if canonical.ndim != 1 or canonical.shape[0] != ptm.shape[0]:
        raise ValueError(f"Scorer v10 label shape mismatch: {row.get('source_id')}")
    if weights.shape != canonical.shape:
        raise ValueError(f"Scorer v10 frame weight shape mismatch: {row.get('source_id')}")
    if np.any(~np.isin(canonical, (0, 1, 2))):
        raise ValueError("Scorer v10 canonical labels must be background/speech/unsure")
    if int(row["frame_count"]) != int(ptm.shape[0]):
        raise ValueError(f"Scorer v10 frame_count mismatch: {row.get('source_id')}")
    labels_binary = np.where(canonical == 2, IGNORE_INDEX, canonical).astype(np.int64)
    return ptm, mfcc, labels_binary, weights


def summarize_partition_labels(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], Counter[str]]:
    presence = {
        partition: {"speech_rows": 0, "all_background_rows": 0}
        for partition in PARTITIONS
    }
    counts: Counter[str] = Counter()
    for row in rows:
        _ptm, _mfcc, labels, _weights = load_binary_row(row)
        counts.update(
            background=int(np.sum(labels == 0)),
            speech=int(np.sum(labels == 1)),
            unsure=int(np.sum(labels == IGNORE_INDEX)),
        )
        current = presence[str(row["partition"])]
        if str(row["row_role"]) == "speech":
            if not np.any(labels == 1):
                raise ValueError("Scorer v10 speech row has no canonical speech frame")
            current["speech_rows"] += 1
        elif str(row["row_role"]) == "all_background":
            if np.any(labels == 1) or not np.any(labels == 0):
                raise ValueError("Scorer v10 all-background row contradicts canonical labels")
            current["all_background_rows"] += 1
    if any(
        not values["speech_rows"] or not values["all_background_rows"]
        for values in presence.values()
    ):
        raise ValueError(
            "Scorer v10 requires speech and all-background rows in every partition"
        )
    return presence, counts


def compute_mfcc_normalization(rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    total = 0
    feature_sum: np.ndarray | None = None
    square_sum: np.ndarray | None = None
    for row in rows:
        _ptm, mfcc, labels, weights = load_binary_row(row)
        valid = (labels != IGNORE_INDEX) & (weights > 0.0)
        values = mfcc[valid].astype(np.float64)
        if feature_sum is None:
            feature_sum = np.zeros(values.shape[1], dtype=np.float64)
            square_sum = np.zeros(values.shape[1], dtype=np.float64)
        feature_sum += values.sum(axis=0)
        square_sum += np.square(values).sum(axis=0)
        total += int(values.shape[0])
    if not total or feature_sum is None or square_sum is None:
        raise ValueError("Scorer v10 train partition has no definite MFCC frames")
    mean = feature_sum / total
    variance = square_sum / total - np.square(mean)
    return {
        "mfcc_mean": mean.astype(np.float32).tolist(),
        "mfcc_std": np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32).tolist(),
    }


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


def pad_batch(rows: Sequence[dict[str, Any]]):
    import torch

    loaded = [load_binary_row(row) for row in rows]
    maximum = max(item[0].shape[0] for item in loaded)
    ptm_dim = int(loaded[0][0].shape[1])
    mfcc_dim = int(loaded[0][1].shape[1])
    ptm = np.zeros((len(rows), maximum, ptm_dim), dtype=np.float32)
    mfcc = np.zeros((len(rows), maximum, mfcc_dim), dtype=np.float32)
    labels = np.full((len(rows), maximum), IGNORE_INDEX, dtype=np.int64)
    weights = np.zeros((len(rows), maximum), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=np.int64)
    for index, (row_ptm, row_mfcc, row_labels, row_weights) in enumerate(loaded):
        length = int(row_ptm.shape[0])
        ptm[index, :length] = row_ptm
        mfcc[index, :length] = row_mfcc
        labels[index, :length] = row_labels
        weights[index, :length] = row_weights
        mask[index, :length] = 1
    return tuple(torch.from_numpy(value) for value in (ptm, mfcc, labels, weights, mask))


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[values.astype(bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def evaluate(model, rows, device, *, max_padded_frames: int, tolerance_frames: int):
    import torch

    speech_rows = background_rows = 0
    speech_runs = start_hits = end_hits = true_speech_deletions = 0
    tp = fp = fn = 0
    background_drops = 0
    start_errors: list[int] = []
    end_errors: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in frame_budget_batches(rows, max_padded_frames=max_padded_frames):
            ptm, mfcc, labels, _weights, mask = pad_batch(batch)
            probabilities = torch.softmax(
                model(ptm.to(device), mfcc.to(device), attention_mask=mask.to(device)),
                dim=-1,
            ).cpu().numpy()
            for index, row in enumerate(batch):
                length = int(row["frame_count"])
                truth = labels[index, :length].numpy()
                valid = truth != IGNORE_INDEX
                predicted = np.argmax(probabilities[index, :length], axis=1)
                tp += int(np.sum((predicted == 1) & (truth == 1) & valid))
                fp += int(np.sum((predicted == 1) & (truth == 0) & valid))
                fn += int(np.sum((predicted == 0) & (truth == 1) & valid))
                truth_runs = _runs(truth == 1)
                if not truth_runs:
                    background_rows += 1
                    background_drops += int(not np.any((predicted == 1) & valid))
                    continue
                speech_rows += 1
                for start, end in truth_runs:
                    speech_runs += 1
                    predicted_run = np.flatnonzero(
                        (predicted[start:end] == 1) & valid[start:end]
                    )
                    if not predicted_run.size:
                        true_speech_deletions += 1
                        continue
                    predicted_start = start + int(predicted_run[0])
                    predicted_end = start + int(predicted_run[-1])
                    start_error = abs(predicted_start - start)
                    end_error = abs(predicted_end - (end - 1))
                    start_errors.append(start_error)
                    end_errors.append(end_error)
                    start_hits += int(start_error <= tolerance_frames)
                    end_hits += int(end_error <= tolerance_frames)
    return {
        "speech_row_count": speech_rows,
        "background_row_count": background_rows,
        "speech_run_count": speech_runs,
        "start_coverage": start_hits / max(speech_runs, 1),
        "end_coverage": end_hits / max(speech_runs, 1),
        "start_mae_frames": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_frames": float(np.mean(end_errors)) if end_errors else None,
        "speech_precision": tp / max(tp + fp, 1),
        "speech_recall": tp / max(tp + fn, 1),
        "background_drop_recall": background_drops / max(background_rows, 1),
        "true_speech_deletion_count": true_speech_deletions,
    }


def _memory_snapshot(device) -> dict[str, Any]:
    import torch

    snapshot = runtime_memory_snapshot(require_shared_vram=device.type == "cuda")
    if device.type != "cuda":
        snapshot = {
            key: value for key, value in snapshot.items() if not key.startswith("shared_vram")
        }
        snapshot.update(shared_vram_mb=0.0, shared_vram_monitor="not_applicable_cpu_stage")
    if device.type == "cuda":
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated(device) / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved(device) / 2**20, 3),
        )
    if snapshot["physical_ram_used_mb"] > snapshot["physical_ram_budget_mb"]:
        raise MemoryError("Scorer v10 exceeded the 0.95 physical RAM budget")
    if device.type == "cuda" and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError("Scorer v10 shared VRAM spill is a soft OOM")
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

    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Scorer v10 is 1.7B-only")
    if args.max_steps <= 0 or args.eval_interval <= 0:
        raise ValueError("Scorer v10 max_steps/eval_interval must be positive")
    apply_vram_safety_cap(0.95)
    rows = _read_rows(Path(args.dataset_manifest))
    dataset_summary = validate_dataset_rows(rows)
    by_partition = {
        name: [row for row in rows if row["partition"] == name] for name in PARTITIONS
    }
    partition_presence, canonical_counts = summarize_partition_labels(rows)
    dataset_summary["partition_label_presence"] = partition_presence
    normalization = compute_mfcc_normalization(by_partition["train"])
    first_ptm, first_mfcc, _labels, _weights = load_binary_row(by_partition["train"][0])
    if int(first_ptm.shape[1]) != SPEECH_ISLAND_SCORER_V10_RAW_PTM_DIM:
        raise ValueError("Scorer v10 requires raw PTM2048 features")
    if int(first_mfcc.shape[1]) != SPEECH_ISLAND_SCORER_V10_MFCC_DIM:
        raise ValueError("Scorer v10 requires MFCC40 features")
    model_config = {
        "raw_ptm_dim": int(first_ptm.shape[1]),
        "projected_ptm_dim": SPEECH_ISLAND_SCORER_V10_PROJECTED_PTM_DIM,
        "mfcc_dim": int(first_mfcc.shape[1]),
        "position_dim": 2,
        "mfcc_mean": normalization["mfcc_mean"],
        "mfcc_std": normalization["mfcc_std"],
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
        "model_arch": SPEECH_ISLAND_SCORER_V10_MODEL_ARCH,
        "output_dim": 2,
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Scorer v10 requested CUDA but CUDA is unavailable")
    _memory_snapshot(device)
    model = BinarySpeechIslandScorerNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    train_batches = frame_budget_batches(
        by_partition["train"], max_padded_frames=args.max_batch_frames
    )
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_state = None
    losses: list[float] = []
    memory_snapshots: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(train_batches) == 0:
            rng.shuffle(train_batches)
        batch = train_batches[(step - 1) % len(train_batches)]
        ptm, mfcc, labels, source_weights, mask = pad_batch(batch)
        model.train()
        logits = model(ptm.to(device), mfcc.to(device), attention_mask=mask.to(device))
        target = labels.to(device)
        valid = target != IGNORE_INDEX
        loss_rows = F.cross_entropy(
            logits.transpose(1, 2), target, reduction="none", ignore_index=IGNORE_INDEX
        )
        class_weights = torch.where(
            target == 0,
            torch.as_tensor(args.background_weight, device=device),
            torch.as_tensor(args.speech_weight, device=device),
        )
        weights = source_weights.to(device) * class_weights
        loss = (loss_rows[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % args.eval_interval == 0 or step == args.max_steps:
            val = evaluate(
                model,
                by_partition["val"],
                device,
                max_padded_frames=args.max_batch_frames,
                tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
            )
            memory_snapshots.append(_memory_snapshot(device))
            score = (
                float(val["true_speech_deletion_count"] == 0),
                min(val["start_coverage"], val["end_coverage"], val["background_drop_recall"]),
                val["speech_recall"],
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Scorer v10 training produced no evaluated checkpoint")
    model.load_state_dict(best_state)
    val = evaluate(model, by_partition["val"], device, max_padded_frames=args.max_batch_frames,
                   tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)))
    test = evaluate(model, by_partition["test"], device, max_padded_frames=args.max_batch_frames,
                    tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"speech_island_scorer_v10.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    torch.save(
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization=normalization,
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset_manifest": args.dataset_manifest,
                "dataset_summary": dataset_summary,
                "trained_steps": args.max_steps,
                "best_step": best_step,
                "canonical_label_counts": dict(canonical_counts),
                "excluded_training_count": int(canonical_counts["unsure"]),
                "training_initialization": "random",
                "checkpoint_selection": "val_binary_speech_edge_300ms_coverage_v1",
                "class_weights": {
                    "background": args.background_weight,
                    "speech": args.speech_weight,
                },
            },
            schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
        ),
        checkpoint,
    )
    numeric_gate_pass = (
        min(
            val["start_coverage"], val["end_coverage"], test["start_coverage"],
            test["end_coverage"], val["background_drop_recall"],
            test["background_drop_recall"],
        ) >= 0.95
        and val["true_speech_deletion_count"] == 0
        and test["true_speech_deletion_count"] == 0
    )
    summary = {
        "schema": "speech_island_scorer_v10_binary_training_summary_v1",
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
        "decision_mode": "binary_frame_argmax",
        "memory_snapshots": memory_snapshots,
        "elapsed_s": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del optimizer, model, best_state, logits, target, valid, loss_rows, class_weights, weights, loss
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
        description="Train the 1.7B binary Speech Island Scorer v10 from canonical full-source frames."
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--tolerance-s", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--max-batch-frames", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--speech-weight", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
