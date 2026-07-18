from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from boundary.sequence_store import load_sequence_arrays


IGNORE_ID = -100


def load_island_dataset(path: Path) -> dict[str, Any]:
    bundle = load_sequence_arrays(path)
    required = (
        "frame_features",
        "scalar_features",
        "labels",
        "partitions",
        "group_ids",
    )
    for key in required:
        if key not in bundle:
            raise ValueError(f"sequence dataset missing {key!r}: {path}")
    count = int(bundle["labels"].shape[0])
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(bundle["group_ids"].astype(str).tolist()):
        groups[group_id].append(index)
    frames = bundle["frame_features"]
    if not isinstance(frames, np.memmap):
        frames = frames.astype(np.float32, copy=False)
    return {
        "frames": frames,
        "scalars": bundle["scalar_features"].astype(np.float32, copy=False),
        "labels": bundle["labels"].astype(np.int64),
        "partitions": bundle["partitions"].astype(str),
        "dataset_roles": (
            bundle["dataset_roles"].astype(str)
            if "dataset_roles" in bundle
            else np.asarray(["default"] * count)
        ),
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
        "groups": {
            name: np.asarray(indexes, dtype=np.int64)
            for name, indexes in groups.items()
        },
    }


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


def pad_batch(
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
    for row, name in enumerate(names):
        indexes = groups[name]
        count = int(indexes.size)
        frames[row, :count] = (data["frames"][indexes] - frame_mean) / frame_std
        scalars[row, :count] = (data["scalars"][indexes] - scalar_mean) / scalar_std
        mask[row, :count] = 1
        labels[row, :count] = data["labels"][indexes]
        roles[row, :count] = data["structural_roles"][indexes]
        pairs[row, :count] = data["pair_ids"][indexes]
    return tuple(
        torch.from_numpy(value)
        for value in (frames, scalars, mask, labels, roles, pairs)
    )


def pair_loss(gate_probabilities, labels, pairs):
    """Soft-AND objective on the two cuts isolating one background run."""

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
            first, second = positions[:2]
            losses.append(
                1.0
                - gate_probabilities[row, first]
                * gate_probabilities[row, second]
            )
    return torch.stack(losses).mean() if losses else None
