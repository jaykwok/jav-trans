#!/usr/bin/env python3
"""Train Acoustic Split v3 with direct cut/continue/unsure argmax decisions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.sequence_features import (  # noqa: E402
    SPLIT_CANDIDATE_SCALAR_NAMES,
    parse_extra_context_scales,
)
from boundary.sequence_store import chunked_frame_stats  # noqa: E402
from boundary.split_model import (  # noqa: E402
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    IslandCandidateSequenceNetwork,
    build_acoustic_split_v3_checkpoint,
)
from tools.boundary.ja.train_semantic_split_island_model import (  # noqa: E402
    _pair_loss,
    _pad_batch,
    island_batches,
    load_island_dataset,
)


IGNORE_ID = -100


def partition_group_names(data: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for name, indexes in sorted(data["groups"].items()):
        partitions = set(data["partitions"][indexes].tolist())
        if len(partitions) != 1:
            raise ValueError(f"group {name!r} crosses dataset partitions")
        result[str(next(iter(partitions)))].append(name)
    for required in ("train", "val", "test"):
        if not result.get(required):
            raise ValueError(f"sequence dataset has no {required} groups")
    return dict(result)


def build_lr_scheduler(optimizer, *, warmup_steps: int, max_steps: int):
    import torch

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def initialize_from_v2_checkpoint(
    model,
    path: Path,
    *,
    raw_frame_mean: np.ndarray,
    raw_frame_std: np.ndarray,
    ptm_dim: int,
) -> dict[str, Any]:
    """Warm-start v3 while embedding v2's learned affine PTM projection."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    own = model.state_dict()
    loaded = 0
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            loaded += 1
    old_segments = state.get("segment_embedding")
    if old_segments is not None and "segment_embedding" in own:
        count = min(int(old_segments.shape[0]), int(own["segment_embedding"].shape[0]))
        own["segment_embedding"][:count] = old_segments[:count]
    model.load_state_dict(own)

    feature_config = dict(payload.get("feature_config") or {})
    projection = dict(feature_config.get("ptm_projection") or {})
    components = np.asarray(projection["components"], dtype=np.float64)
    projection_mean = np.asarray(projection["mean"], dtype=np.float64)
    old_normalization = dict(payload["normalization"])
    old_mean = np.asarray(old_normalization["frame_mean"], dtype=np.float64)
    old_std = np.maximum(
        np.asarray(old_normalization["frame_std"], dtype=np.float64), 1e-6
    )
    projected_dim = int(components.shape[0])
    if components.shape != (projected_dim, ptm_dim):
        raise ValueError("v2 PTM projection dimensions do not match v3 input")
    raw_mean = np.asarray(raw_frame_mean, dtype=np.float64)
    raw_std = np.maximum(np.asarray(raw_frame_std, dtype=np.float64), 1e-6)
    projector_weight = (
        components * raw_std[:ptm_dim][None, :] / old_std[:projected_dim, None]
    )
    ptm_offset = (
        components @ (raw_mean[:ptm_dim] - projection_mean)
        - old_mean[:projected_dim]
    ) / old_std[:projected_dim]

    mfcc_dim = int(raw_mean.size - ptm_dim)
    old_mfcc_mean = old_mean[projected_dim : projected_dim + mfcc_dim]
    old_mfcc_std = old_std[projected_dim : projected_dim + mfcc_dim]
    mfcc_scale = raw_std[ptm_dim:] / old_mfcc_std
    mfcc_offset = (raw_mean[ptm_dim:] - old_mfcc_mean) / old_mfcc_std
    with torch.no_grad():
        model.ptm_projector.weight.copy_(
            torch.from_numpy(projector_weight.astype(np.float32)).to(
                model.ptm_projector.weight.device
            )
        )
        frame_weight = model.frame_proj.weight
        old_mfcc_weight = frame_weight[:, projected_dim:].clone()
        frame_weight[:, projected_dim:].mul_(
            torch.from_numpy(mfcc_scale.astype(np.float32)).to(frame_weight.device)[
                None, :
            ]
        )
        model.frame_proj.bias.add_(
            frame_weight[:, :projected_dim]
            @ torch.from_numpy(ptm_offset.astype(np.float32)).to(frame_weight.device)
            + old_mfcc_weight
            @ torch.from_numpy(mfcc_offset.astype(np.float32)).to(frame_weight.device)
        )
    return {
        "path": str(path),
        "loaded_tensors": loaded,
        "embedded_projection": "learned_v2_linear_then_trainable",
        "projected_dim": projected_dim,
    }


def _classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for expected, actual in zip(truth.tolist(), predicted.tolist()):
        if expected in (0, 1, 2):
            matrix[expected, actual] += 1
    rows: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(SEMANTIC_SPLIT_LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        rows[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-9, precision + recall),
            "support": int(matrix[index, :].sum()),
        }
    return {
        "labels": rows,
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "confusion_matrix": matrix.tolist(),
    }


def event_run_counts(truth: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    def runs(values: np.ndarray) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        position = 0
        while position < values.size:
            if int(values[position]) != 0:
                position += 1
                continue
            start = position
            while position + 1 < values.size and int(values[position + 1]) == 0:
                position += 1
            result.append((start, position))
            position += 1
        return result

    truth_events = runs(truth)
    predicted_events = runs(predicted)
    exact = sum(event in set(truth_events) for event in predicted_events)
    truth_basins: list[set[int]] = [set() for _ in truth_events]
    for position, label in enumerate(truth.tolist()):
        if label == 1 or not truth_events:
            continue
        nearest = min(
            range(len(truth_events)),
            key=lambda index: (
                0
                if truth_events[index][0] <= position <= truth_events[index][1]
                else min(
                    abs(position - truth_events[index][0]),
                    abs(position - truth_events[index][1]),
                ),
                index,
            ),
        )
        truth_basins[nearest].add(position)
    basin_matched = 0
    unmatched = set(range(len(truth_basins)))
    for start, end in predicted_events:
        indexes = set(range(start, end + 1))
        candidates = [
            basin_index
            for basin_index in unmatched
            if indexes & truth_basins[basin_index]
        ]
        if not candidates:
            continue
        matched = max(
            candidates,
            key=lambda basin_index: len(indexes & truth_basins[basin_index]),
        )
        unmatched.remove(matched)
        basin_matched += 1
    return {
        "truth": len(truth_events),
        "predicted": len(predicted_events),
        "exact": exact,
        "basin_matched": basin_matched,
    }


def evaluate(
    model,
    data: dict[str, Any],
    names: list[str],
    *,
    normalization: dict[str, np.ndarray],
    device,
    batch_islands: int,
    max_batch_candidates: int,
) -> dict[str, Any]:
    import torch

    truth_rows: list[np.ndarray] = []
    predicted_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    event_truth = event_predicted = event_exact = event_basin_matched = 0
    model.eval()
    with torch.inference_mode():
        for batch in island_batches(
            names,
            data["groups"],
            batch_islands=batch_islands,
            max_batch_candidates=max_batch_candidates,
        ):
            frames, scalars, mask, labels, *_rest = _pad_batch(
                data,
                batch,
                frame_mean=normalization["frame_mean"],
                frame_std=normalization["frame_std"],
                scalar_mean=normalization["scalar_mean"],
                scalar_std=normalization["scalar_std"],
            )
            logits = model(
                frames.to(device), scalars.to(device), mask.to(device)
            )["label"]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for row, name in enumerate(batch):
                count = int(data["groups"][name].size)
                truth = labels[row, :count].numpy()
                predicted = probabilities[row, :count].argmax(axis=-1)
                valid = truth != IGNORE_ID
                truth_rows.append(truth[valid])
                predicted_rows.append(predicted[valid])
                probability_rows.append(probabilities[row, :count][valid])

                event_counts = event_run_counts(truth, predicted)
                event_truth += event_counts["truth"]
                event_predicted += event_counts["predicted"]
                event_exact += event_counts["exact"]
                event_basin_matched += event_counts["basin_matched"]
    truth = np.concatenate(truth_rows)
    predicted = np.concatenate(predicted_rows)
    probabilities = np.concatenate(probability_rows)
    metrics = _classification_metrics(truth, predicted)
    metrics["mean_probability"] = {
        label: float(probabilities[:, index].mean())
        for index, label in enumerate(SEMANTIC_SPLIT_LABELS)
    }
    metrics["event_runs"] = {
        "truth": event_truth,
        "predicted": event_predicted,
        "exact": event_exact,
        "exact_precision": event_exact / max(1, event_predicted),
        "exact_recall": event_exact / max(1, event_truth),
        "basin_matched": event_basin_matched,
        "basin_precision": event_basin_matched / max(1, event_predicted),
        "basin_recall": event_basin_matched / max(1, event_truth),
    }
    return metrics


def gate_passes(metrics: dict[str, Any]) -> bool:
    labels = metrics["labels"]
    events = metrics["event_runs"]
    return bool(
        events["basin_recall"] >= 0.95
        and events["basin_precision"] >= 0.90
        and labels["continue"]["recall"] >= 0.90
    )


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap()
    data = load_island_dataset(Path(args.dataset))
    partitions = partition_group_names(data)
    train_names = partitions["train"]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_rows = np.concatenate([data["groups"][name] for name in train_names])
    frame_mean, frame_std = chunked_frame_stats(data["frames"], train_rows)
    normalization = {
        "frame_mean": frame_mean,
        "frame_std": np.maximum(frame_std, 1e-6),
        "scalar_mean": data["scalars"][train_rows].mean(axis=0),
        "scalar_std": np.maximum(data["scalars"][train_rows].std(axis=0), 1e-6),
    }
    extra_context_scales = parse_extra_context_scales(args.extra_context_scales)
    extra_scale_bins = [
        [int(scale["left_bins"]), int(scale["right_bins"])]
        for scale in extra_context_scales
    ]
    expected_bins = 20 + sum(left + right for left, right in extra_scale_bins)
    if int(data["frames"].shape[1]) != expected_bins:
        raise ValueError(
            f"dataset bins {data['frames'].shape[1]} do not match configured {expected_bins}"
        )
    raw_frame_dim = int(data["frames"].shape[2])
    if raw_frame_dim <= args.ptm_dim:
        raise ValueError("dataset must include non-PTM acoustic frame features")
    model_config = {
        "frame_dim": args.ptm_projector_dim + (raw_frame_dim - args.ptm_dim),
        "scalar_dim": int(data["scalars"].shape[1]),
        "hidden_size": args.hidden_size,
        "candidate_layers": args.candidate_layers,
        "island_layers": args.island_layers,
        "state_size": 32,
        "num_heads": 4,
        "head_dim": (args.hidden_size * 2) // 4,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
        "dropout": args.dropout,
        "extra_scale_bins": extra_scale_bins,
        "ptm_input_dim": args.ptm_dim,
        "ptm_projected_dim": args.ptm_projector_dim,
        "ptm_projector_residual": False,
    }
    device = torch.device(args.device)
    model = IslandCandidateSequenceNetwork(**model_config).to(device)
    init_report = (
        initialize_from_v2_checkpoint(
            model,
            Path(args.init_v2_checkpoint),
            raw_frame_mean=normalization["frame_mean"],
            raw_frame_std=normalization["frame_std"],
            ptm_dim=args.ptm_dim,
        )
        if args.init_v2_checkpoint
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = build_lr_scheduler(
        optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps
    )
    class_weights = torch.tensor(
        [args.cut_weight, args.continue_weight, args.unsure_weight],
        dtype=torch.float32,
        device=device,
    )
    losses: list[float] = []
    best_score = -1.0
    best_step = 0
    best_state: dict[str, Any] | None = None
    best_val: dict[str, Any] | None = None

    block_size = max(args.batch_islands, args.shuffle_block_groups)
    train_blocks = [
        train_names[start : start + block_size]
        for start in range(0, len(train_names), block_size)
    ]
    pending_batches: list[list[str]] = []

    def sample_batch() -> list[str]:
        nonlocal pending_batches
        if not pending_batches:
            order = rng.permutation(len(train_blocks)).tolist()
            epoch_names = [
                name for block_index in order for name in train_blocks[block_index]
            ]
            pending_batches = island_batches(
                epoch_names,
                data["groups"],
                batch_islands=args.batch_islands,
                max_batch_candidates=args.max_batch_candidates,
            )
            pending_batches.reverse()
        return pending_batches.pop()

    for step in range(1, args.max_steps + 1):
        model.train()
        frames, scalars, mask, labels, roles, pairs, *_rest = _pad_batch(
            data,
            sample_batch(),
            frame_mean=normalization["frame_mean"],
            frame_std=normalization["frame_std"],
            scalar_mean=normalization["scalar_mean"],
            scalar_std=normalization["scalar_std"],
        )
        model_outputs = model(
            frames.to(device), scalars.to(device), mask.to(device)
        )
        label_logits = model_outputs["label"]
        flat_logits = label_logits.reshape(-1, len(SEMANTIC_SPLIT_LABELS))
        flat_labels = labels.to(device).reshape(-1)
        raw = F.cross_entropy(
            flat_logits,
            flat_labels,
            weight=class_weights,
            ignore_index=IGNORE_ID,
            reduction="none",
        )
        valid = flat_labels != IGNORE_ID
        if args.focal_gamma > 0:
            probabilities = torch.softmax(flat_logits[valid], dim=-1)
            selected = probabilities.gather(1, flat_labels[valid, None]).squeeze(1)
            raw = raw[valid] * torch.pow(1.0 - selected, args.focal_gamma)
        else:
            raw = raw[valid]
        loss = raw.mean()
        role_loss = F.cross_entropy(
            model_outputs["role"].reshape(-1, 4),
            roles.to(device).reshape(-1),
            ignore_index=IGNORE_ID,
        )
        pair_term = _pair_loss(
            torch.softmax(label_logits, dim=-1)[..., 0],
            labels.to(device),
            pairs.to(device),
        )
        loss = loss + args.role_aux_weight * torch.nan_to_num(role_loss)
        if pair_term is not None:
            loss = loss + args.pair_loss_weight * pair_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and step % args.log_every == 0:
            print(
                f"acoustic_split_v3_train={step}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg={np.mean(losses[-args.log_every:]):.6f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.max_steps:
            val = evaluate(
                model,
                data,
                partitions["val"],
                normalization=normalization,
                device=device,
                batch_islands=args.eval_batch_islands,
                max_batch_candidates=args.max_batch_candidates,
            )
            cut = val["labels"]["cut"]
            cont = val["labels"]["continue"]
            events = val["event_runs"]
            event_f1 = (
                2.0
                * events["basin_precision"]
                * events["basin_recall"]
                / max(
                    1e-9,
                    events["basin_precision"] + events["basin_recall"],
                )
            )
            score = float(event_f1 + 0.25 * cont["recall"])
            print(
                f"acoustic_split_v3_eval step={step} "
                f"cut_P={cut['precision']:.4f} cut_R={cut['recall']:.4f} "
                f"event_P={events['basin_precision']:.4f} "
                f"event_R={events['basin_recall']:.4f} "
                f"continue_R={cont['recall']:.4f} accuracy={val['accuracy']:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_val = val
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None or best_val is None:
        raise RuntimeError("training produced no evaluated checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test = evaluate(
        model,
        data,
        partitions["test"],
        normalization=normalization,
        device=device,
        batch_islands=args.eval_batch_islands,
        max_batch_candidates=args.max_batch_candidates,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"semantic_split_model_v3.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    torch.save(
        build_acoustic_split_v3_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
                "ptm_dim": args.ptm_dim,
                "mfcc_dim": raw_frame_dim - args.ptm_dim,
                "left_context_s": 1.6,
                "right_context_s": 1.6,
                "gap_context_s": 0.3,
                "left_bins": 8,
                "gap_bins": 4,
                "right_bins": 8,
                "extra_context_scales": extra_context_scales,
                "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
                "ptm_projection": {
                    "kind": "learned_linear_in_checkpoint",
                    "input_dim": args.ptm_dim,
                    "output_dim": args.ptm_projector_dim,
                },
            },
            normalization={key: value.tolist() for key, value in normalization.items()},
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset": str(Path(args.dataset)),
                "trained_steps": args.max_steps,
                "best_step": best_step,
                "training_decision": "direct_three_class_argmax",
                "timing_output": "semantic_event_only",
                "init_report": init_report,
                "loss": {
                    "cut_weight": args.cut_weight,
                    "continue_weight": args.continue_weight,
                    "unsure_weight": args.unsure_weight,
                    "focal_gamma": args.focal_gamma,
                    "role_aux_weight": args.role_aux_weight,
                    "pair_loss_weight": args.pair_loss_weight,
                },
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "schema": "semantic_split_model_v3_training_metrics",
        "decision_mode": "argmax_cut",
        "train_group_count": len(partitions["train"]),
        "val_group_count": len(partitions["val"]),
        "test_group_count": len(partitions["test"]),
        "best_step": best_step,
        "mean_train_loss": float(np.mean(losses)),
        "val": best_val,
        "test": test,
        "val_gate_passed": gate_passes(best_val),
        "test_gate_passed": gate_passes(test),
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--ptm-dim", type=int, default=2048)
    parser.add_argument("--ptm-projector-dim", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-islands", type=int, default=8)
    parser.add_argument("--eval-batch-islands", type=int, default=8)
    parser.add_argument("--max-batch-candidates", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--island-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cut-weight", type=float, default=3.0)
    parser.add_argument("--continue-weight", type=float, default=1.0)
    parser.add_argument("--unsure-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--role-aux-weight", type=float, default=0.3)
    parser.add_argument("--pair-loss-weight", type=float, default=0.3)
    parser.add_argument("--init-v2-checkpoint", default="")
    parser.add_argument(
        "--shuffle-block-groups",
        type=int,
        default=64,
        help="Shuffle contiguous group blocks to preserve memmap read locality.",
    )
    parser.add_argument("--extra-context-scales", default="3.2:4,6.4:4")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.ptm_projector_dim <= 0 or args.ptm_projector_dim >= args.ptm_dim:
        parser.error("--ptm-projector-dim must be between 1 and ptm-dim-1")
    return args


if __name__ == "__main__":
    run(parse_args())
