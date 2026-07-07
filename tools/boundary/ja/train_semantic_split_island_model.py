#!/usr/bin/env python3
"""Train the v2 island-level Semantic Split model on sequence NPZ datasets.

Dataset contract (produced by build_runtime_semantic_split_dataset /
compile_joint_boundary_preasr_dataset / merge_semantic_split_datasets in
sequence mode): flat candidate rows plus ``group_ids`` marking island
membership, ``labels`` in {0 cut, 1 continue, 2 unsure, -100 context-only},
``structural_roles`` (-100 ignore), ``pair_ids`` (-1 none) marking the two
structural cuts that isolate one noise run, and ``omni_aux`` (-1 unknown).
"""
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.sequence_features import (  # noqa: E402
    PTM_PROJECTION_SCHEMA,
    SPLIT_CANDIDATE_SCALAR_NAMES,
    load_ptm_projection,
    parse_extra_context_scales,
    ptm_projection_digest,
)
from boundary.sequence_store import (  # noqa: E402
    chunked_frame_stats,
    load_sequence_arrays,
)
from boundary.split_model import (  # noqa: E402
    SEMANTIC_SPLIT_FEATURE_SCHEMA,
    SEMANTIC_SPLIT_LABELS,
    SEMANTIC_SPLIT_OMNI_AUX_NAMES,
    SEMANTIC_SPLIT_STRUCTURAL_ROLES,
    SEMANTIC_SPLIT_V2_DEFAULT_DECISION,
    IslandCandidateSequenceNetwork,
    build_semantic_split_island_checkpoint,
)


IGNORE_ID = -100
CORE_DURATION_SCALAR_INDEX = SPLIT_CANDIDATE_SCALAR_NAMES.index("core_duration_s")


def load_island_dataset(path: Path) -> dict[str, Any]:
    bundle = load_sequence_arrays(Path(path))
    required = ("frame_features", "scalar_features", "labels", "partitions", "group_ids")
    for key in required:
        if key not in bundle:
            raise ValueError(f"sequence dataset missing {key!r}: {path}")
    count = int(bundle["labels"].shape[0])
    group_ids = bundle["group_ids"].astype(str)
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids.tolist()):
        groups.setdefault(group_id, []).append(index)
    roles = (
        bundle["dataset_roles"].astype(str)
        if "dataset_roles" in bundle
        else np.asarray(["default"] * count)
    )
    frames = bundle["frame_features"]
    if not isinstance(frames, np.memmap):
        # copy=False: stored float32 loads as-is; a copy would double the
        # multi-GB frame array on a 16GB-RAM box.
        frames = frames.astype(np.float32, copy=False)
    return {
        "frames": frames,
        "scalars": bundle["scalar_features"].astype(np.float32, copy=False),
        "labels": bundle["labels"].astype(np.int64),
        "partitions": bundle["partitions"].astype(str),
        "dataset_roles": roles,
        "structural_roles": (
            bundle["structural_roles"].astype(np.int64)
            if "structural_roles" in bundle
            else np.full(count, IGNORE_ID, dtype=np.int64)
        ),
        "pair_ids": (
            bundle["pair_ids"].astype(np.int64)
            if "pair_ids" in bundle
            else np.full(count, -1, dtype=np.int64)
        ),
        "omni_aux": (
            bundle["omni_aux"].astype(np.float32)
            if "omni_aux" in bundle
            else np.full((count, 3), -1.0, dtype=np.float32)
        ),
        "offsets": (
            bundle["offset_targets_s"].astype(np.float32)
            if "offset_targets_s" in bundle
            else np.full(count, np.nan, dtype=np.float32)
        ),
        "groups": {name: np.asarray(indexes, dtype=np.int64) for name, indexes in groups.items()},
    }


def split_group_names(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    train: list[str] = []
    val: list[str] = []
    for name, indexes in sorted(data["groups"].items()):
        partition = str(data["partitions"][indexes[0]])
        (val if partition == "val" else train).append(name)
    if not val:
        raise ValueError("sequence dataset has no val partition groups")
    if not train:
        raise ValueError("sequence dataset has no train partition groups")
    return train, val


def island_batches(
    names: list[str],
    groups: dict[str, np.ndarray],
    *,
    batch_islands: int,
    max_batch_candidates: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    candidates = 0
    for name in names:
        count = int(groups[name].size)
        if current and (
            len(current) >= batch_islands
            or candidates + count > max_batch_candidates
        ):
            batches.append(current)
            current = []
            candidates = 0
        current.append(name)
        candidates += count
    if current:
        batches.append(current)
    return batches


def _pad_batch(
    data: dict[str, Any],
    names: list[str],
    *,
    frame_mean: np.ndarray,
    frame_std: np.ndarray,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
):
    import torch

    groups = data["groups"]
    counts = [int(groups[name].size) for name in names]
    max_count = max(counts)
    bins = int(data["frames"].shape[1])
    frame_dim = int(data["frames"].shape[2])
    scalar_dim = int(data["scalars"].shape[1])
    frames = np.zeros((len(names), max_count, bins, frame_dim), dtype=np.float32)
    scalars = np.zeros((len(names), max_count, scalar_dim), dtype=np.float32)
    mask = np.zeros((len(names), max_count), dtype=np.int64)
    labels = np.full((len(names), max_count), IGNORE_ID, dtype=np.int64)
    roles = np.full((len(names), max_count), IGNORE_ID, dtype=np.int64)
    pairs = np.full((len(names), max_count), -1, dtype=np.int64)
    omni = np.full((len(names), max_count, 3), -1.0, dtype=np.float32)
    offsets = np.full((len(names), max_count), np.nan, dtype=np.float32)
    core_durations = np.zeros(len(names), dtype=np.float32)
    for row, name in enumerate(names):
        indexes = groups[name]
        count = int(indexes.size)
        frames[row, :count] = (data["frames"][indexes] - frame_mean) / frame_std
        scalars[row, :count] = (data["scalars"][indexes] - scalar_mean) / scalar_std
        mask[row, :count] = 1
        labels[row, :count] = data["labels"][indexes]
        roles[row, :count] = data["structural_roles"][indexes]
        pairs[row, :count] = data["pair_ids"][indexes]
        omni[row, :count] = data["omni_aux"][indexes]
        offsets[row, :count] = data["offsets"][indexes]
        core_durations[row] = float(
            data["scalars"][indexes[0], CORE_DURATION_SCALAR_INDEX]
        )
    return (
        torch.from_numpy(frames),
        torch.from_numpy(scalars),
        torch.from_numpy(mask),
        torch.from_numpy(labels),
        torch.from_numpy(roles),
        torch.from_numpy(pairs),
        torch.from_numpy(omni),
        torch.from_numpy(offsets),
        core_durations,
    )


def _pair_loss(gate_probabilities, labels, pairs):
    """Soft-AND objective on the two structural cuts isolating one noise run."""

    import torch

    losses = []
    for row in range(pairs.shape[0]):
        by_pair: dict[int, list[int]] = defaultdict(list)
        for position in range(pairs.shape[1]):
            pair_id = int(pairs[row, position])
            if pair_id >= 0 and int(labels[row, position]) == 0:
                by_pair[pair_id].append(position)
        for positions in by_pair.values():
            if len(positions) < 2:
                continue
            first, second = positions[0], positions[1]
            losses.append(
                1.0 - gate_probabilities[row, first] * gate_probabilities[row, second]
            )
    if not losses:
        return None
    return torch.stack(losses).mean()


def _bootstrap_f1_ci(
    island_counts: list[tuple[int, int, int]],
    *,
    samples: int = 1000,
    seed: int = 20260705,
) -> dict[str, list[float]]:
    """95% bootstrap CI over islands for pooled cut precision/recall/F1."""

    counts = np.asarray(island_counts, dtype=np.int64)
    if counts.shape[0] < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, counts.shape[0], size=(samples, counts.shape[0]))
    tp = counts[draws, 0].sum(axis=1).astype(np.float64)
    fp = counts[draws, 1].sum(axis=1).astype(np.float64)
    fn = counts[draws, 2].sum(axis=1).astype(np.float64)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2 * precision * recall / np.maximum(1e-9, precision + recall)
    def interval(values: np.ndarray) -> list[float]:
        return [
            round(float(np.percentile(values, 2.5)), 4),
            round(float(np.percentile(values, 97.5)), 4),
        ]
    return {
        "cut_precision_ci95": interval(precision),
        "cut_recall_ci95": interval(recall),
        "cut_f1_ci95": interval(f1),
    }


def evaluate_island_model(
    model,
    data: dict[str, Any],
    names: list[str],
    *,
    normalization: dict[str, np.ndarray],
    device,
    batch_islands: int,
    max_batch_candidates: int,
    normal_cut_threshold: float,
    short_core_cut_threshold: float,
    short_core_max_s: float,
) -> dict[str, Any]:
    """Deployment-aligned metrics, reported per dataset_role domain."""

    import torch

    gate_rows: list[np.ndarray] = []
    offset_pred_rows: list[np.ndarray] = []
    row_indexes: list[np.ndarray] = []
    thresholds: list[float] = []
    with torch.inference_mode():
        for batch in island_batches(
            names,
            data["groups"],
            batch_islands=batch_islands,
            max_batch_candidates=max_batch_candidates,
        ):
            frames, scalars, mask, _labels, _roles, _pairs, _omni, _offsets, cores = _pad_batch(
                data,
                batch,
                frame_mean=normalization["frame_mean"],
                frame_std=normalization["frame_std"],
                scalar_mean=normalization["scalar_mean"],
                scalar_std=normalization["scalar_std"],
            )
            outputs = model(
                frames.to(device), scalars.to(device), mask.to(device)
            )
            gate = torch.sigmoid(outputs["gate"]).cpu().numpy()
            offset_pred = outputs["offset"].cpu().numpy()
            for row, name in enumerate(batch):
                count = int(data["groups"][name].size)
                gate_rows.append(gate[row, :count])
                offset_pred_rows.append(offset_pred[row, :count])
                row_indexes.append(data["groups"][name])
                thresholds.append(
                    short_core_cut_threshold
                    if cores[row] <= short_core_max_s
                    else normal_cut_threshold
                )
    domain_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "continue_total": 0, "continue_cut": 0}
    )
    pair_hits: dict[int, list[bool]] = defaultdict(list)
    gate_by_row: dict[int, float] = {}
    accept_by_row: dict[int, bool] = {}
    island_counts: list[tuple[str, int, int, int]] = []
    offset_errors: list[float] = []
    for gate, offset_pred, indexes, threshold in zip(
        gate_rows, offset_pred_rows, row_indexes, thresholds
    ):
        island_domain = str(data["dataset_roles"][int(indexes[0])])
        island_tp = island_fp = island_fn = 0
        for position, index in enumerate(indexes.tolist()):
            probability = float(gate[position])
            accepted = probability >= threshold
            gate_by_row[index] = probability
            accept_by_row[index] = accepted
            label = int(data["labels"][index])
            domain = str(data["dataset_roles"][index])
            counts = domain_counts[domain]
            if label == 0:
                target_offset = float(data["offsets"][index])
                if np.isfinite(target_offset):
                    offset_errors.append(
                        abs(float(offset_pred[position]) - target_offset)
                    )
                if accepted:
                    counts["tp"] += 1
                    island_tp += 1
                else:
                    counts["fn"] += 1
                    island_fn += 1
            elif label == 1:
                counts["continue_total"] += 1
                if accepted:
                    counts["fp"] += 1
                    counts["continue_cut"] += 1
                    island_fp += 1
            pair_id = int(data["pair_ids"][index])
            if pair_id >= 0 and label == 0:
                pair_hits[pair_id].append(accepted)
        island_counts.append((island_domain, island_tp, island_fp, island_fn))
    domains: dict[str, dict[str, float]] = {}
    for domain, counts in sorted(domain_counts.items()):
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        domains[domain] = {
            "cut_precision": precision,
            "cut_recall": recall,
            "cut_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "continue_false_cut": (
                counts["continue_cut"] / max(1, counts["continue_total"])
            ),
            "cut_truth": counts["tp"] + counts["fn"],
        }
    complete_pairs = [hits for hits in pair_hits.values() if len(hits) >= 2]
    isolated = sum(1 for hits in complete_pairs if all(hits[:2]))
    macro_f1 = (
        float(np.mean([row["cut_f1"] for row in domains.values()]))
        if domains
        else 0.0
    )
    # Deployment-domain pooled metrics: real anchors merged so a 7-truth
    # domain cannot swing the gate; hardmix-only datasets fall back to all.
    real_domains = sorted(
        domain for domain in domain_counts if domain.startswith("real_")
    )
    pool_domains = real_domains or sorted(domain_counts)
    pooled = {
        key: sum(domain_counts[domain][key] for domain in pool_domains)
        for key in ("tp", "fp", "fn", "continue_total", "continue_cut")
    }
    pooled_precision = pooled["tp"] / max(1, pooled["tp"] + pooled["fp"])
    pooled_recall = pooled["tp"] / max(1, pooled["tp"] + pooled["fn"])
    pooled_real = {
        "domains": pool_domains,
        "cut_precision": pooled_precision,
        "cut_recall": pooled_recall,
        "cut_f1": 2 * pooled_precision * pooled_recall
        / max(1e-9, pooled_precision + pooled_recall),
        "continue_false_cut": (
            pooled["continue_cut"] / max(1, pooled["continue_total"])
        ),
        "cut_truth": pooled["tp"] + pooled["fn"],
        **_bootstrap_f1_ci(
            [
                (tp, fp, fn)
                for domain, tp, fp, fn in island_counts
                if domain in set(pool_domains)
            ]
        ),
    }
    return {
        "domains": domains,
        "macro_cut_f1": macro_f1,
        "pooled_real": pooled_real,
        "pair_isolation_rate": isolated / max(1, len(complete_pairs)),
        "complete_pair_count": len(complete_pairs),
        "offset_mae_s": (
            float(np.mean(offset_errors)) if offset_errors else None
        ),
        "offset_supervised_count": len(offset_errors),
        "gate_by_row": gate_by_row,
    }


def calibrate_thresholds(
    data: dict[str, Any],
    gate_by_row: dict[int, float],
    *,
    short_core_max_s: float,
    min_precision: float,
    domain_prefix: str = "",
) -> dict[str, float]:
    """Pick per-regime gate thresholds maximizing F1 at acceptable precision.

    ``domain_prefix`` restricts calibration rows to matching ``dataset_roles``
    (e.g. ``real_`` so hardmix rows cannot dominate the deployment operating
    point); when no row matches, all rows are used.
    """

    roles = data["dataset_roles"]
    matches_prefix = (
        bool(domain_prefix)
        and any(str(role).startswith(domain_prefix) for role in roles.tolist())
    )
    short_rows: list[tuple[float, int]] = []
    normal_rows: list[tuple[float, int]] = []
    for name, indexes in data["groups"].items():
        core = float(data["scalars"][indexes[0], CORE_DURATION_SCALAR_INDEX])
        bucket = short_rows if core <= short_core_max_s else normal_rows
        for index in indexes.tolist():
            if matches_prefix and not str(roles[index]).startswith(domain_prefix):
                continue
            label = int(data["labels"][index])
            if index in gate_by_row and label in (0, 1):
                bucket.append((gate_by_row[index], label))

    def best(rows: list[tuple[float, int]], fallback: float) -> float:
        if not rows:
            return fallback
        best_threshold = fallback
        best_f1 = -1.0
        for threshold in np.arange(0.30, 0.96, 0.05):
            tp = sum(1 for p, label in rows if label == 0 and p >= threshold)
            fp = sum(1 for p, label in rows if label == 1 and p >= threshold)
            fn = sum(1 for p, label in rows if label == 0 and p < threshold)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            if precision >= min_precision and f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(threshold), 2)
        return best_threshold

    return {
        "normal_cut_threshold": best(
            normal_rows, SEMANTIC_SPLIT_V2_DEFAULT_DECISION["normal_cut_threshold"]
        ),
        "short_core_cut_threshold": best(
            short_rows, SEMANTIC_SPLIT_V2_DEFAULT_DECISION["short_core_cut_threshold"]
        ),
        "short_core_max_s": short_core_max_s,
        "min_chunk_after_split_s": SEMANTIC_SPLIT_V2_DEFAULT_DECISION[
            "min_chunk_after_split_s"
        ],
    }


def _init_from_checkpoint(model, path: str) -> dict[str, int]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or {}
    own = model.state_dict()
    loaded = 0
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            loaded += 1
    model.load_state_dict(own)
    return {"loaded_tensors": loaded, "total_tensors": len(own)}


def run(args: argparse.Namespace) -> None:
    import math

    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap()
    data = load_island_dataset(Path(args.dataset))
    train_names, val_names = split_group_names(data)
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
            f"dataset has {int(data['frames'].shape[1])} bins but "
            f"--extra-context-scales implies {expected_bins}; rebuild the dataset "
            "or pass matching scales"
        )
    ptm_projection = load_ptm_projection(args.ptm_projection)
    raw_frame_dim = int(data["frames"].shape[2])
    if args.ptm_projector_dim:
        if args.ptm_projection:
            raise ValueError(
                "--ptm-projector-dim learns its own projection; drop --ptm-projection"
            )
        if raw_frame_dim <= args.ptm_dim:
            raise ValueError(
                f"dataset frame dim {raw_frame_dim} leaves no non-PTM features "
                f"beyond --ptm-dim {args.ptm_dim}"
            )
        model_frame_dim = args.ptm_projector_dim + (raw_frame_dim - args.ptm_dim)
    else:
        model_frame_dim = raw_frame_dim
    model_config = {
        "frame_dim": model_frame_dim,
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
        "ptm_input_dim": args.ptm_dim if args.ptm_projector_dim else 0,
        "ptm_projected_dim": args.ptm_projector_dim,
        "ptm_projector_residual": bool(args.ptm_projector_residual),
    }
    device = torch.device(args.device)
    model = IslandCandidateSequenceNetwork(**model_config).to(device)
    init_report = (
        _init_from_checkpoint(model, args.init_checkpoint)
        if args.init_checkpoint
        else None
    )
    if args.freeze_backbone:
        # Pin every weight inherited from the init checkpoint so the only
        # trainable parameters are the zero-init residual projector; this
        # isolates whether the non-linear residual can lift a fixed P.
        frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith("ptm_residual"):
                param.requires_grad_(False)
                frozen += 1
        init_report = init_report or {}
        init_report["frozen_params"] = frozen
    if args.freeze_ptm_projector:
        # Pin the in-model PTM projector at the init checkpoint's weights so it
        # stays the fixed P projection (approach a: normalize->project->frame_proj
        # pipeline preserved, no double-normalization) while frame_proj / encoder
        # / heads train on the resulting projected-basis input.
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("ptm_projector"):
                param.requires_grad_(False)
                frozen += 1
        init_report = init_report or {}
        init_report["frozen_ptm_projector"] = frozen
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        import torch

        def _lr_at(step: int) -> float:
            if step < args.warmup_steps:
                return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(
                1, args.max_steps - args.warmup_steps
            )
            return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_at)
    cut_groups = [
        name
        for name in train_names
        if bool((data["labels"][data["groups"][name]] == 0).any())
    ]
    nocut_groups = [name for name in train_names if name not in set(cut_groups)]
    if not cut_groups:
        raise ValueError("training set has no islands with cut labels")
    losses: list[float] = []
    best_score = -1.0
    best_state: dict | None = None
    best_metrics: dict[str, Any] | None = None

    def sample_batch() -> list[str]:
        names: list[str] = []
        for _slot in range(args.batch_islands):
            pool = cut_groups
            if nocut_groups and float(rng.random()) >= args.cut_island_ratio:
                pool = nocut_groups
            names.append(pool[int(rng.integers(0, len(pool)))])
        return names

    for step in range(1, args.max_steps + 1):
        model.train()
        frames, scalars, mask, labels, roles, pairs, omni, offsets, _cores = _pad_batch(
            data,
            sample_batch(),
            frame_mean=normalization["frame_mean"],
            frame_std=normalization["frame_std"],
            scalar_mean=normalization["scalar_mean"],
            scalar_std=normalization["scalar_std"],
        )
        frames = frames.to(device)
        scalars = scalars.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        roles = roles.to(device)
        omni = omni.to(device)
        offsets = offsets.to(device)
        outputs = model(frames, scalars, mask)
        gate_logits = outputs["gate"]
        gate_targets = (labels == 0).to(dtype=gate_logits.dtype)
        gate_weights = torch.zeros_like(gate_logits)
        gate_weights = torch.where(
            labels == 0, torch.full_like(gate_logits, args.cut_weight), gate_weights
        )
        gate_weights = torch.where(
            labels == 1,
            torch.full_like(gate_logits, args.continue_weight),
            gate_weights,
        )
        gate_weights = torch.where(
            labels == 2,
            torch.full_like(gate_logits, args.unsure_gate_weight),
            gate_weights,
        )
        raw_gate = F.binary_cross_entropy_with_logits(
            gate_logits, gate_targets, reduction="none"
        )
        probabilities = torch.sigmoid(gate_logits)
        pt = torch.where(gate_targets > 0.5, probabilities, 1.0 - probabilities)
        focal = torch.pow(1.0 - pt, args.focal_gamma) if args.focal_gamma > 0 else 1.0
        gate_loss = (raw_gate * focal * gate_weights).sum() / gate_weights.sum().clamp_min(
            1e-6
        )
        label_loss = F.cross_entropy(
            outputs["label"].reshape(-1, len(SEMANTIC_SPLIT_LABELS)),
            labels.reshape(-1),
            ignore_index=IGNORE_ID,
        )
        role_loss = F.cross_entropy(
            outputs["role"].reshape(-1, len(SEMANTIC_SPLIT_STRUCTURAL_ROLES)),
            roles.reshape(-1),
            ignore_index=IGNORE_ID,
        )
        omni_mask = omni >= 0.0
        if omni_mask.any():
            omni_loss = (
                F.binary_cross_entropy_with_logits(
                    outputs["omni"], omni.clamp_min(0.0), reduction="none"
                )
                * omni_mask.to(dtype=gate_logits.dtype)
            ).sum() / omni_mask.sum().clamp_min(1)
        else:
            omni_loss = torch.zeros((), device=device)
        pair_term = _pair_loss(probabilities, labels, pairs)
        offset_mask = torch.isfinite(offsets) & (labels == 0)
        offset_term = (
            F.huber_loss(
                outputs["offset"][offset_mask],
                offsets[offset_mask],
                delta=0.1,
            )
            if bool(offset_mask.any())
            else None
        )
        loss = (
            gate_loss
            + args.label_aux_weight * torch.nan_to_num(label_loss)
            + args.role_aux_weight * torch.nan_to_num(role_loss)
            + args.omni_aux_weight * omni_loss
        )
        if pair_term is not None:
            loss = loss + args.pair_loss_weight * pair_term
        if offset_term is not None:
            loss = loss + args.offset_weight * offset_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and step % args.log_every == 0:
            print(
                f"island_split_train={step}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg_loss={np.mean(losses[-args.log_every:]):.6f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            metrics = evaluate_island_model(
                model,
                data,
                val_names,
                normalization=normalization,
                device=device,
                batch_islands=args.batch_islands,
                max_batch_candidates=args.max_batch_candidates,
                normal_cut_threshold=SEMANTIC_SPLIT_V2_DEFAULT_DECISION[
                    "normal_cut_threshold"
                ],
                short_core_cut_threshold=SEMANTIC_SPLIT_V2_DEFAULT_DECISION[
                    "short_core_cut_threshold"
                ],
                short_core_max_s=SEMANTIC_SPLIT_V2_DEFAULT_DECISION["short_core_max_s"],
            )
            score = (
                metrics["pooled_real"]["cut_f1"]
                + 0.1 * metrics["pair_isolation_rate"]
            )
            pooled = metrics["pooled_real"]
            print(
                f"island_split_eval step={step} "
                f"pooled_real_f1={pooled['cut_f1']:.4f} "
                f"(P={pooled['cut_precision']:.4f} R={pooled['cut_recall']:.4f} "
                f"ci95={pooled.get('cut_f1_ci95')}) "
                f"macro_cut_f1={metrics['macro_cut_f1']:.4f} "
                f"pair_isolation={metrics['pair_isolation_rate']:.4f} "
                f"domains={json.dumps({k: round(v['cut_f1'], 4) for k, v in metrics['domains'].items()})}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_metrics = {
                    key: value for key, value in metrics.items() if key != "gate_by_row"
                }
                best_metrics["step"] = step
                best_metrics["gate_by_row"] = metrics["gate_by_row"]

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    model.eval()
    decision_config = calibrate_thresholds(
        data,
        best_metrics.pop("gate_by_row"),
        short_core_max_s=SEMANTIC_SPLIT_V2_DEFAULT_DECISION["short_core_max_s"],
        min_precision=args.calibration_min_precision,
        domain_prefix=args.calibration_domain_prefix,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"semantic_split_model_v2.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    torch.save(
        build_semantic_split_island_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
                "ptm_dim": args.ptm_dim,
                "mfcc_dim": int(data["frames"].shape[2]) - args.ptm_dim,
                "left_context_s": 1.6,
                "right_context_s": 1.6,
                "gap_context_s": 0.3,
                "left_bins": 8,
                "gap_bins": 4,
                "right_bins": 8,
                "extra_context_scales": extra_context_scales,
                "ptm_projection": ptm_projection,
                "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
            },
            normalization={
                "frame_mean": normalization["frame_mean"].tolist(),
                "frame_std": normalization["frame_std"].tolist(),
                "scalar_mean": normalization["scalar_mean"].tolist(),
                "scalar_std": normalization["scalar_std"].tolist(),
            },
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset": str(Path(args.dataset)),
                "trained_steps": args.max_steps,
                "best_step": best_metrics["step"],
                "sampling": {
                    "batch_islands": args.batch_islands,
                    "cut_island_ratio": args.cut_island_ratio,
                },
                "loss": {
                    "cut_weight": args.cut_weight,
                    "continue_weight": args.continue_weight,
                    "unsure_gate_weight": args.unsure_gate_weight,
                    "focal_gamma": args.focal_gamma,
                    "label_aux_weight": args.label_aux_weight,
                    "role_aux_weight": args.role_aux_weight,
                    "omni_aux_weight": args.omni_aux_weight,
                    "pair_loss_weight": args.pair_loss_weight,
                    "offset_weight": args.offset_weight,
                },
                "init_checkpoint": str(args.init_checkpoint or ""),
                "init_report": init_report,
            },
            decision_config=decision_config,
        ),
        checkpoint_path,
    )
    learned_projection: dict[str, Any] | None = None
    if args.ptm_projector_dim and not args.ptm_projector_residual:
        # Fold the trained projector with the frame normalization so the
        # exported npz reproduces the model's PTM path on RAW ptm frames:
        # W((x - mu) / sigma) == (x - mu) @ (W / sigma).T
        weight = (
            model.ptm_projector.weight.detach().cpu().numpy().astype(np.float64)
        )
        mean = np.asarray(
            normalization["frame_mean"], dtype=np.float64
        )[: args.ptm_dim]
        std = np.asarray(
            normalization["frame_std"], dtype=np.float64
        )[: args.ptm_dim]
        components = (weight / std[None, :]).astype(np.float32)
        projection_path = output_dir / "learned_ptm_projection.npz"
        np.savez(
            projection_path,
            schema=np.asarray(PTM_PROJECTION_SCHEMA),
            mean=mean.astype(np.float32),
            components=components,
        )
        learned_projection = {
            "path": str(projection_path),
            "input_dim": args.ptm_dim,
            "dim": args.ptm_projector_dim,
            "digest": ptm_projection_digest(mean.astype(np.float32), components),
        }
    final_metrics = evaluate_island_model(
        model,
        data,
        val_names,
        normalization=normalization,
        device=device,
        batch_islands=args.batch_islands,
        max_batch_candidates=args.max_batch_candidates,
        normal_cut_threshold=decision_config["normal_cut_threshold"],
        short_core_cut_threshold=decision_config["short_core_cut_threshold"],
        short_core_max_s=decision_config["short_core_max_s"],
    )
    final_metrics.pop("gate_by_row", None)
    metrics_payload = {
        "labels": list(SEMANTIC_SPLIT_LABELS),
        "structural_roles": list(SEMANTIC_SPLIT_STRUCTURAL_ROLES),
        "omni_aux_names": list(SEMANTIC_SPLIT_OMNI_AUX_NAMES),
        "train_group_count": len(train_names),
        "val_group_count": len(val_names),
        "loss": float(np.mean(losses)),
        "best_step": best_metrics["step"],
        "best_val_at_default_thresholds": best_metrics,
        "decision_config": decision_config,
        "val_at_calibrated_thresholds": final_metrics,
        "checkpoint": str(checkpoint_path),
        **(
            {"learned_ptm_projection": learned_projection}
            if learned_projection
            else {}
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics_payload, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train island-level Semantic Split v2 on sequence NPZ datasets."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ptm-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument(
        "--ptm-projector-dim",
        type=int,
        default=0,
        help=(
            "Learn a Linear(ptm_dim -> this) PTM input projector inside the "
            "model (identity-slice init) instead of training on pre-truncated "
            "features; exports the normalization-folded affine as "
            "learned_ptm_projection.npz. 0 disables."
        ),
    )
    parser.add_argument(
        "--ptm-projector-residual",
        action="store_true",
        help=(
            "Add a zero-init LN+MLP residual on top of the linear projector "
            "(non-foldable). step-0 output equals the linear projector, so a "
            "B-continue vs A-residual run isolates the residual's value. No "
            "affine is exported in this mode."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze every parameter except the ptm_residual projector (requires "
            "--ptm-projector-residual). Used to test whether the non-linear "
            "residual can lift a fixed P without the backbone-drift confound of "
            "continued constant-LR training."
        ),
    )
    parser.add_argument(
        "--freeze-ptm-projector",
        action="store_true",
        help=(
            "Freeze the in-model ptm_projector at the init checkpoint's weights "
            "(requires --ptm-projector-dim). Approach a: keeps the fixed P "
            "projection and its normalize->project->frame_proj pipeline, avoiding "
            "the double-normalization of pre-projected data; frame_proj / encoder "
            "/ heads still train."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-islands", type=int, default=8)
    parser.add_argument("--max-batch-candidates", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine"),
        default="none",
        help="none = constant LR; cosine decays to 0 over max_steps after warmup "
        "(avoids the late-step overfitting oscillation seen under constant LR).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup steps before cosine decay (or before constant LR).",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--island-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cut-weight", type=float, default=2.0)
    parser.add_argument("--continue-weight", type=float, default=1.0)
    parser.add_argument("--unsure-gate-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-aux-weight", type=float, default=0.3)
    parser.add_argument("--role-aux-weight", type=float, default=0.3)
    parser.add_argument("--omni-aux-weight", type=float, default=0.2)
    parser.add_argument("--pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--offset-weight", type=float, default=0.3)
    parser.add_argument(
        "--extra-context-scales",
        default="3.2:4,6.4:4",
        help=(
            "Must match the dataset builder's scales "
            "('<seconds>:<bins_per_side>,...'; empty string for base-only bins)."
        ),
    )
    parser.add_argument(
        "--ptm-projection",
        default="",
        help=(
            "PTM projection npz used at dataset build time; embedded into the "
            "checkpoint feature_config so runtime applies the same transform."
        ),
    )
    parser.add_argument("--cut-island-ratio", type=float, default=0.7)
    parser.add_argument("--calibration-min-precision", type=float, default=0.85)
    parser.add_argument(
        "--calibration-domain-prefix",
        default="real_",
        help=(
            "Restrict threshold calibration to dataset_roles with this prefix "
            "(deployment domain); falls back to all rows when nothing matches."
        ),
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_islands <= 0 or args.max_batch_candidates <= 0:
        parser.error("batch sizes must be positive")
    if not 0.0 < args.cut_island_ratio <= 1.0:
        parser.error("--cut-island-ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
