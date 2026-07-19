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
from pipeline.memory_safety import (  # noqa: E402
    reset_shared_vram_baseline,
    runtime_memory_snapshot,
)


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


def predicted_run_structure(predicted_speech) -> dict[str, int]:
    """Count learned argmax islands inside one canonical speech run."""

    import torch

    values = predicted_speech.to(dtype=torch.bool).reshape(-1)
    if not values.numel():
        return {
            "predicted_run_count": 0,
            "continuous": 0,
            "fragmented": 0,
            "internal_drop_gap_count": 0,
            "internal_drop_frame_count": 0,
        }
    previous = torch.cat(
        (torch.zeros(1, dtype=torch.bool, device=values.device), values[:-1])
    )
    run_count = int(torch.sum(values & ~previous).item())
    present = torch.nonzero(values).flatten()
    internal_drop_frames = 0
    if present.numel():
        first = int(present[0].item())
        last = int(present[-1].item())
        internal_drop_frames = int(torch.sum(~values[first : last + 1]).item())
    return {
        "predicted_run_count": run_count,
        "continuous": int(run_count == 1),
        "fragmented": int(run_count > 1),
        "internal_drop_gap_count": max(0, run_count - 1),
        "internal_drop_frame_count": internal_drop_frames,
    }


def speech_continuity_auxiliary_loss(logits, target):
    """Penalize learned speech-probability jumps only inside true speech runs."""

    import torch

    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("Scorer v10 continuity logits must have shape [batch,frames,2]")
    if target.shape != logits.shape[:2]:
        raise ValueError("Scorer v10 continuity target shape mismatch")
    speech_probability = torch.softmax(logits, dim=-1)[..., 1]
    adjacent_speech = (
        (target[:, :-1] == 1)
        & (target[:, 1:] == 1)
        & (target[:, :-1] != IGNORE_INDEX)
        & (target[:, 1:] != IGNORE_INDEX)
    )
    pair_count = adjacent_speech.sum()
    squared_delta = torch.square(
        speech_probability[:, 1:] - speech_probability[:, :-1]
    )
    loss = (squared_delta * adjacent_speech.to(squared_delta.dtype)).sum()
    loss = loss / pair_count.to(squared_delta.dtype).clamp_min(1.0)
    return loss, pair_count


def evaluate(model, rows, device, *, max_padded_frames: int, tolerance_frames: int):
    import torch

    speech_rows = background_rows = 0
    speech_runs = start_hits = end_hits = true_speech_deletions = 0
    tp = fp = fn = 0
    background_drops = 0
    continuous_speech_runs = fragmented_speech_runs = 0
    predicted_runs_within_truth = 0
    internal_drop_gap_count = internal_drop_frame_count = 0
    start_errors: list[int] = []
    end_errors: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in frame_budget_batches(rows, max_padded_frames=max_padded_frames):
            ptm, mfcc, labels, _weights, mask = pad_batch(batch)
            predictions = torch.argmax(
                model(ptm.to(device), mfcc.to(device), attention_mask=mask.to(device)),
                dim=-1,
            )
            labels_device = labels.to(device)
            for index, row in enumerate(batch):
                length = int(row["frame_count"])
                truth_cpu = labels[index, :length].numpy()
                truth = labels_device[index, :length]
                valid = truth != IGNORE_INDEX
                predicted = predictions[index, :length]
                tp += int(torch.sum((predicted == 1) & (truth == 1) & valid).item())
                fp += int(torch.sum((predicted == 1) & (truth == 0) & valid).item())
                fn += int(torch.sum((predicted == 0) & (truth == 1) & valid).item())
                truth_runs = _runs(truth_cpu == 1)
                if not truth_runs:
                    background_rows += 1
                    background_drops += int(
                        not bool(torch.any((predicted == 1) & valid).item())
                    )
                    continue
                speech_rows += 1
                for start, end in truth_runs:
                    speech_runs += 1
                    predicted_run = torch.nonzero(
                        (predicted[start:end] == 1) & valid[start:end]
                    ).flatten()
                    structure = predicted_run_structure(
                        (predicted[start:end] == 1) & valid[start:end]
                    )
                    continuous_speech_runs += structure["continuous"]
                    fragmented_speech_runs += structure["fragmented"]
                    predicted_runs_within_truth += structure["predicted_run_count"]
                    internal_drop_gap_count += structure["internal_drop_gap_count"]
                    internal_drop_frame_count += structure["internal_drop_frame_count"]
                    if not predicted_run.numel():
                        true_speech_deletions += 1
                        continue
                    predicted_start = start + int(predicted_run[0].item())
                    predicted_end = start + int(predicted_run[-1].item())
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
        "continuous_speech_run_count": continuous_speech_runs,
        "fragmented_speech_run_count": fragmented_speech_runs,
        "speech_run_continuity": continuous_speech_runs / max(speech_runs, 1),
        "predicted_run_count_within_truth": predicted_runs_within_truth,
        "prediction_to_truth_run_ratio": predicted_runs_within_truth / max(speech_runs, 1),
        "internal_drop_gap_count": internal_drop_gap_count,
        "internal_drop_frame_count": internal_drop_frame_count,
    }


def _memory_snapshot(device, *, stage: str) -> dict[str, Any]:
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
            cuda_max_allocated_mb=round(
                torch.cuda.max_memory_allocated(device) / 2**20, 3
            ),
            cuda_max_reserved_mb=round(
                torch.cuda.max_memory_reserved(device) / 2**20, 3
            ),
        )
    if snapshot["physical_ram_used_mb"] > snapshot["physical_ram_budget_mb"]:
        raise MemoryError("Scorer v10 exceeded the 0.95 physical RAM budget")
    if device.type == "cuda" and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError(
            "Scorer v10 shared VRAM spill is a soft OOM: "
            f"shared_vram_mb={float(snapshot.get('shared_vram_mb') or 0.0):.3f} "
            f"raw_mb={snapshot.get('shared_vram_raw_mb')} "
            f"baseline_mb={snapshot.get('shared_vram_baseline_mb')}"
        )
    snapshot["stage"] = stage
    return snapshot


def _update_peak_memory(
    peak: dict[str, float], snapshot: dict[str, Any]
) -> None:
    for key in (
        "physical_ram_used_mb",
        "shared_vram_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_max_allocated_mb",
        "cuda_max_reserved_mb",
    ):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            peak[key] = max(float(value), float(peak.get(key, 0.0)))


def release_gate_fields(numeric_gate_pass: bool) -> dict[str, Any]:
    return {
        "numeric_gate_pass": bool(numeric_gate_pass),
        "gate_pass": False,
        "promotion_ready": False,
        "manual_zero_clipping_gate": "required_before_promotion",
    }


def numeric_gate_pass(val: dict[str, Any], test: dict[str, Any]) -> bool:
    return (
        min(
            val["start_coverage"],
            val["end_coverage"],
            test["start_coverage"],
            test["end_coverage"],
            val["background_drop_recall"],
            test["background_drop_recall"],
            val["speech_run_continuity"],
            test["speech_run_continuity"],
        )
        >= 0.95
        and val["true_speech_deletion_count"] == 0
        and test["true_speech_deletion_count"] == 0
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if args.continuity_weight < 0.0:
        raise ValueError("--continuity-weight must be non-negative")

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
    del first_ptm, first_mfcc, _labels, _weights
    if device.type == "cuda":
        torch.cuda.init()
        context_warmup = torch.ones(1, device=device)
        context_warmup.add_(1.0)
        torch.cuda.synchronize(device)
        del context_warmup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
    memory_snapshots: list[dict[str, Any]] = [
        _memory_snapshot(device, stage="context_baseline")
    ]
    memory_peak: dict[str, float] = {}
    _update_peak_memory(memory_peak, memory_snapshots[-1])
    rng = np.random.default_rng(args.seed)
    train_batches = frame_budget_batches(
        by_partition["train"], max_padded_frames=args.max_batch_frames
    )
    if device.type == "cuda":
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        warmup_model = BinarySpeechIslandScorerNetwork(**model_config).to(device)
        warmup_optimizer = torch.optim.AdamW(
            warmup_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        warmup_batch = max(
            train_batches,
            key=lambda batch: max(int(row["frame_count"]) for row in batch)
            * len(batch),
        )
        warmup_ptm, warmup_mfcc, warmup_labels, warmup_source_weights, warmup_mask = (
            pad_batch(warmup_batch)
        )
        warmup_model.train()
        warmup_logits = warmup_model(
            warmup_ptm.to(device),
            warmup_mfcc.to(device),
            attention_mask=warmup_mask.to(device),
        )
        warmup_target = warmup_labels.to(device)
        warmup_valid = warmup_target != IGNORE_INDEX
        warmup_loss_rows = F.cross_entropy(
            warmup_logits.transpose(1, 2),
            warmup_target,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        warmup_class_weights = torch.where(
            warmup_target == 0,
            torch.as_tensor(args.background_weight, device=device),
            torch.as_tensor(args.speech_weight, device=device),
        )
        warmup_weights = warmup_source_weights.to(device) * warmup_class_weights
        warmup_loss = (
            warmup_loss_rows[warmup_valid] * warmup_weights[warmup_valid]
        ).sum() / warmup_weights[warmup_valid].sum().clamp_min(1e-6)
        warmup_optimizer.zero_grad(set_to_none=True)
        warmup_loss.backward()
        warmup_optimizer.step()
        eval_warmup_val = evaluate(
            warmup_model,
            by_partition["val"],
            device,
            max_padded_frames=args.max_batch_frames,
            tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
        )
        eval_warmup_test = evaluate(
            warmup_model,
            by_partition["test"],
            device,
            max_padded_frames=args.max_batch_frames,
            tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
        )
        torch.cuda.synchronize(device)
        del (
            warmup_optimizer,
            warmup_model,
            warmup_ptm,
            warmup_mfcc,
            warmup_labels,
            warmup_source_weights,
            warmup_mask,
            warmup_logits,
            warmup_target,
            warmup_valid,
            warmup_loss_rows,
            warmup_class_weights,
            warmup_weights,
            warmup_loss,
            eval_warmup_val,
            eval_warmup_test,
        )
        gc.collect()
        torch.cuda.empty_cache()
        execution_baseline = reset_shared_vram_baseline(required=True)
        execution_baseline["stage"] = "execution_baseline"
        memory_snapshots.append(execution_baseline)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = BinarySpeechIslandScorerNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    memory_snapshots.append(_memory_snapshot(device, stage="model_initialized"))
    _update_peak_memory(memory_peak, memory_snapshots[-1])
    best_score = (-1.0, -1.0, -1.0)
    best_step = 0
    best_state = None
    losses: list[float] = []
    primary_losses: list[float] = []
    continuity_losses: list[float] = []
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
        primary_loss = (loss_rows[valid] * weights[valid]).sum() / weights[
            valid
        ].sum().clamp_min(1e-6)
        continuity_loss, continuity_pair_count = speech_continuity_auxiliary_loss(
            logits, target
        )
        loss = primary_loss + float(args.continuity_weight) * continuity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        primary_losses.append(float(primary_loss.detach().cpu()))
        continuity_losses.append(float(continuity_loss.detach().cpu()))
        step_memory = _memory_snapshot(device, stage=f"train_step_{step}")
        _update_peak_memory(memory_peak, step_memory)
        if step % args.eval_interval == 0 or step == args.max_steps:
            val = evaluate(
                model,
                by_partition["val"],
                device,
                max_padded_frames=args.max_batch_frames,
                tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)),
            )
            memory_snapshots.append(
                _memory_snapshot(device, stage=f"validation_step_{step}")
            )
            _update_peak_memory(memory_peak, memory_snapshots[-1])
            score = (
                float(val["true_speech_deletion_count"] == 0),
                min(
                    val["start_coverage"],
                    val["end_coverage"],
                    val["background_drop_recall"],
                    val["speech_run_continuity"],
                ),
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
    memory_snapshots.append(_memory_snapshot(device, stage="final_validation"))
    _update_peak_memory(memory_peak, memory_snapshots[-1])
    test = evaluate(model, by_partition["test"], device, max_padded_frames=args.max_batch_frames,
                    tolerance_frames=int(round(args.tolerance_s / args.frame_hop_s)))
    memory_snapshots.append(_memory_snapshot(device, stage="final_test"))
    _update_peak_memory(memory_peak, memory_snapshots[-1])
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
                "seed": args.seed,
                "canonical_label_counts": dict(canonical_counts),
                "excluded_training_count": int(canonical_counts["unsure"]),
                "training_initialization": "random",
                "checkpoint_selection": "val_binary_speech_edge_continuity_300ms_coverage_v2",
                "class_weights": {
                    "background": args.background_weight,
                    "speech": args.speech_weight,
                },
                "continuity_auxiliary": {
                    "kind": "speech_probability_adjacent_total_variation_v1",
                    "weight": float(args.continuity_weight),
                    "runtime_effect": "none_binary_argmax_unchanged",
                },
            },
            schema=SPEECH_ISLAND_SCORER_V10_SCHEMA,
        ),
        checkpoint,
    )
    numeric_gate_pass_value = numeric_gate_pass(val, test)
    summary = {
        "schema": "speech_island_scorer_v10_binary_training_summary_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "best_step": best_step,
        "mean_train_loss": float(np.mean(losses)),
        "mean_primary_loss": float(np.mean(primary_losses)),
        "mean_continuity_loss": float(np.mean(continuity_losses)),
        "val": val,
        "test": test,
        "dataset": dataset_summary,
        "canonical_label_counts": dict(canonical_counts),
        "excluded_training_count": int(canonical_counts["unsure"]),
        **release_gate_fields(numeric_gate_pass_value),
        "training_initialization": "random",
        "seed": args.seed,
        "loss": (
            "weighted_cross_entropy"
            if args.continuity_weight == 0.0
            else "weighted_cross_entropy+speech_probability_adjacent_total_variation_v1"
        ),
        "continuity_weight": float(args.continuity_weight),
        "decision_mode": "binary_frame_argmax",
        "memory_snapshots": memory_snapshots,
        "memory_peak": memory_peak,
        "elapsed_s": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del optimizer, model, best_state, logits, target, valid, loss_rows
    del class_weights, weights
    del loss, primary_loss, continuity_loss, continuity_pair_count
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    summary["memory_after_release"] = _memory_snapshot(
        device, stage="post_release"
    )
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
    parser.add_argument("--continuity-weight", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
