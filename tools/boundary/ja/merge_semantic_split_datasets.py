#!/usr/bin/env python3
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

from boundary.sequence_store import (  # noqa: E402
    load_sequence_arrays,
    open_frames_memmap_for_write,
    save_sequence_dataset,
)


_FRAME_COPY_CHUNK = 16384


def group_row_indexes(bundle: dict) -> dict[str, np.ndarray]:
    """Row indexes per island group, preserving the stored row order."""

    group_ids = bundle["group_ids"].astype(str)
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids.tolist()):
        groups.setdefault(group_id, []).append(index)
    return {
        group_id: np.asarray(indexes, dtype=np.int64)
        for group_id, indexes in groups.items()
    }


def stratified_sample_groups(
    bundle: dict,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    """Sample whole island groups, stratified by (partition, has-cut)."""

    groups = group_row_indexes(bundle)
    names = sorted(groups)
    if fraction >= 1.0:
        return names
    labels = bundle["labels"].astype(np.int64)
    partitions = bundle["partitions"].astype(str)
    strata: dict[tuple[str, str], list[str]] = {}
    for name in names:
        indexes = groups[name]
        partition = str(partitions[indexes[0]])
        has_cut = "cut" if bool((labels[indexes] == 0).any()) else "nocut"
        strata.setdefault((partition, has_cut), []).append(name)
    selected: list[str] = []
    for key in sorted(strata):
        members = strata[key]
        count = max(1, int(round(len(members) * fraction)))
        chosen = rng.choice(len(members), size=min(count, len(members)), replace=False)
        selected.extend(members[int(position)] for position in np.sort(chosen))
    return selected


def _bundle_mode(bundles: list[dict], paths: list[str]) -> str:
    grouped = ["group_ids" in bundle for bundle in bundles]
    if all(grouped):
        return "sequence"
    if not any(grouped):
        return "rows"
    missing = [path for path, has in zip(paths, grouped) if not has]
    raise ValueError(
        "cannot mix grouped and ungrouped Semantic Split datasets; rebuild these "
        f"without group_ids or recompile them with island groups: {missing}"
    )


def _validate_current_sequence_bundle(bundle: dict, path: str) -> None:
    required = {
        "frame_features",
        "scalar_features",
        "labels",
        "partitions",
        "group_ids",
        "source_ids",
        "core_ids",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(
            f"{path}: Split v4 merge requires current sequence identities; "
            f"missing {missing}"
        )
    count = int(bundle["labels"].shape[0])
    for key in sorted(required.difference({"frame_features"})):
        if int(bundle[key].shape[0]) != count:
            raise ValueError(f"{path}: {key} row count does not match labels")
    partitions = bundle["partitions"].astype(str)
    source_ids = bundle["source_ids"].astype(str)
    core_ids = bundle["core_ids"].astype(str)
    groups = group_row_indexes(bundle)
    source_partition: dict[str, str] = {}
    core_group: dict[str, str] = {}
    for group_id, indexes in groups.items():
        group_partitions = set(partitions[indexes].tolist())
        group_sources = set(source_ids[indexes].tolist())
        group_cores = set(core_ids[indexes].tolist())
        if len(group_partitions) != 1:
            raise ValueError(f"{path}: group {group_id!r} crosses partitions")
        if len(group_sources) != 1 or "" in group_sources:
            raise ValueError(f"{path}: group {group_id!r} has invalid source identity")
        if len(group_cores) != 1 or "" in group_cores:
            raise ValueError(f"{path}: group {group_id!r} has invalid core identity")
        partition = next(iter(group_partitions))
        source_id = next(iter(group_sources))
        core_id = next(iter(group_cores))
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"{path}: invalid frozen partition {partition!r}")
        previous_partition = source_partition.setdefault(source_id, partition)
        if previous_partition != partition:
            raise ValueError(
                f"{path}: source {source_id!r} crosses frozen partitions"
            )
        previous_group = core_group.setdefault(core_id, group_id)
        if previous_group != group_id:
            raise ValueError(
                f"{path}: core {core_id!r} is reused by multiple Split groups"
            )


def _sequence_defaults(bundle: dict, indexes: np.ndarray) -> dict[str, np.ndarray]:
    count = int(indexes.size)
    values: dict[str, np.ndarray] = {}
    values["times_s"] = (
        bundle["times_s"][indexes].astype(np.float32)
        if "times_s" in bundle
        else np.zeros(count, dtype=np.float32)
    )
    values["structural_roles"] = (
        bundle["structural_roles"][indexes].astype(np.int64)
        if "structural_roles" in bundle
        else np.full(count, -100, dtype=np.int64)
    )
    values["pair_ids"] = (
        bundle["pair_ids"][indexes].astype(np.int64)
        if "pair_ids" in bundle
        else np.full(count, -1, dtype=np.int64)
    )
    values["omni_aux"] = (
        bundle["omni_aux"][indexes].astype(np.float32)
        if "omni_aux" in bundle
        else np.full((count, 3), -1.0, dtype=np.float32)
    )
    values["offset_targets_s"] = (
        bundle["offset_targets_s"][indexes].astype(np.float32)
        if "offset_targets_s" in bundle
        else np.full(count, np.nan, dtype=np.float32)
    )
    return values


def run(args: argparse.Namespace) -> None:
    bundles = [load_sequence_arrays(Path(path)) for path in args.dataset]
    frame_shape = tuple(bundles[0]["frame_features"].shape[1:])
    scalar_shape = tuple(bundles[0]["scalar_features"].shape[1:])
    for path, bundle in zip(args.dataset, bundles):
        if tuple(bundle["frame_features"].shape[1:]) != frame_shape:
            raise ValueError(f"frame feature shape mismatch: {path}")
        if tuple(bundle["scalar_features"].shape[1:]) != scalar_shape:
            raise ValueError(f"scalar feature shape mismatch: {path}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fractions = [float(value) for value in args.fraction]
    rng = np.random.default_rng(args.seed)
    mode = _bundle_mode(bundles, args.dataset)
    if mode != "sequence":
        raise ValueError(
            "row-wise Semantic Split datasets are retired; Split v4 merge "
            "requires whole candidate-island sequences with source_ids/core_ids"
        )
    for path, bundle in zip(args.dataset, bundles):
        _validate_current_sequence_bundle(bundle, path)
    _run_sequence_mode(
        args,
        bundles=bundles,
        fractions=fractions,
        rng=rng,
        output=output,
    )


def _run_sequence_mode(
    args: argparse.Namespace,
    *,
    bundles: list[dict],
    fractions: list[float],
    rng: np.random.Generator,
    output: Path,
) -> None:
    """Merge current Split v4 island groups whole without core reuse.

    Frames are never fully materialized: source frames arrive memmap-backed
    (or in-memory for legacy npz) and stream chunk-by-chunk into a pre-sized
    output sidecar, so merged multi-GB datasets fit a 16GB-RAM box.
    """

    plans: list[dict] = []
    source_partition: dict[str, str] = {}
    core_owner: dict[str, str] = {}
    for path, role, bundle, fraction in zip(
        args.dataset, args.role, bundles, fractions
    ):
        groups = group_row_indexes(bundle)
        selected = stratified_sample_groups(bundle, fraction=fraction, rng=rng)
        flat = (
            np.concatenate([groups[name] for name in selected])
            if selected
            else np.zeros(0, dtype=np.int64)
        )
        lengths = np.asarray(
            [int(groups[name].size) for name in selected], dtype=np.int64
        )
        selected_sources: list[str] = []
        selected_cores: list[str] = []
        for name in selected:
            indexes = groups[name]
            partition = str(bundle["partitions"][indexes[0]])
            source_id = str(bundle["source_ids"][indexes[0]])
            core_id = str(bundle["core_ids"][indexes[0]])
            previous_partition = source_partition.setdefault(source_id, partition)
            if previous_partition != partition:
                raise ValueError(
                    f"source {source_id!r} crosses input dataset partitions"
                )
            owner = f"{path}:{name}"
            previous_owner = core_owner.setdefault(core_id, owner)
            if previous_owner != owner:
                raise ValueError(
                    f"core {core_id!r} is reused across Split merge inputs"
                )
            selected_sources.append(source_id)
            selected_cores.append(core_id)
        plans.append(
            {
                "path": path,
                "role": role,
                "bundle": bundle,
                "fraction": fraction,
                "group_count": len(groups),
                "selected": selected,
                "flat": flat,
                "lengths": lengths,
                "selected_sources": selected_sources,
                "selected_cores": selected_cores,
            }
        )
    total_rows = sum(int(plan["flat"].size) for plan in plans)
    if total_rows == 0:
        raise ValueError("sequence merge selected no rows")
    frame_shape = tuple(bundles[0]["frame_features"].shape[1:])
    out_frames = open_frames_memmap_for_write(
        output, rows=total_rows, row_shape=frame_shape
    )
    scalar_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    partition_parts: list[np.ndarray] = []
    source_id_parts: list[np.ndarray] = []
    core_id_parts: list[np.ndarray] = []
    role_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    structural_parts: list[np.ndarray] = []
    pair_parts: list[np.ndarray] = []
    omni_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = []
    pair_offset = 0
    row_offset = 0
    source_summaries: list[dict] = []
    for plan in plans:
        bundle = plan["bundle"]
        role = plan["role"]
        flat = plan["flat"]
        lengths = plan["lengths"]
        selected = plan["selected"]
        frames_all = bundle["frame_features"]
        scalars_all = np.asarray(bundle["scalar_features"], dtype=np.float32)
        labels_all = np.asarray(bundle["labels"], dtype=np.int64)
        partitions_all = bundle["partitions"].astype(str)
        total = int(labels_all.shape[0])
        defaults_all = _sequence_defaults(bundle, np.arange(total, dtype=np.int64))
        for start in range(0, int(flat.size), _FRAME_COPY_CHUNK):
            chunk = flat[start : start + _FRAME_COPY_CHUNK]
            out_frames[row_offset : row_offset + chunk.size] = frames_all[chunk]
            row_offset += int(chunk.size)
        scalar_parts.append(scalars_all[flat])
        label_parts.append(labels_all[flat])
        partition_parts.append(partitions_all[flat])
        source_id_parts.append(bundle["source_ids"].astype(str)[flat])
        core_id_parts.append(bundle["core_ids"].astype(str)[flat])
        role_parts.append(np.full(int(flat.size), role))
        group_parts.append(
            np.repeat(
                np.asarray([f"{role}::{name}" for name in selected]),
                lengths,
            )
        )
        time_parts.append(defaults_all["times_s"][flat])
        structural_parts.append(defaults_all["structural_roles"][flat])
        pairs = defaults_all["pair_ids"][flat].copy()
        pairs[pairs >= 0] += pair_offset
        pair_parts.append(pairs)
        omni_parts.append(defaults_all["omni_aux"][flat])
        offset_parts.append(defaults_all["offset_targets_s"][flat])
        if pairs.size and int(pairs.max()) >= 0:
            pair_offset = int(pairs.max()) + 1
        source_summaries.append(
            {
                "path": str(plan["path"]),
                "role": role,
                "group_count": plan["group_count"],
                "sampled_group_count": len(selected),
                "fraction": plan["fraction"],
                "effective_group_count": len(selected),
                "effective_count": int(flat.size),
            }
        )
    out_frames.flush()
    del out_frames
    labels = np.concatenate(label_parts)
    group_ids = np.concatenate(group_parts)
    save_sequence_dataset(
        output,
        frames_finalized=True,
        compress=bool(getattr(args, "compress", False)),
        scalar_features=np.concatenate(scalar_parts),
        labels=labels,
        partitions=np.concatenate(partition_parts),
        source_ids=np.concatenate(source_id_parts),
        core_ids=np.concatenate(core_id_parts),
        dataset_roles=np.concatenate(role_parts),
        group_ids=group_ids,
        times_s=np.concatenate(time_parts),
        structural_roles=np.concatenate(structural_parts),
        pair_ids=np.concatenate(pair_parts),
        omni_aux=np.concatenate(omni_parts),
        offset_targets_s=np.concatenate(offset_parts),
    )
    summary = {
        "schema": "semantic_split_merged_dataset_v3",
        "mode": "sequence",
        "output": str(output),
        "count": int(labels.shape[0]),
        "group_count": int(np.unique(group_ids).size),
        "sources": source_summaries,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--role", action="append", required=True)
    parser.add_argument("--fraction", action="append", type=float)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Write a compressed npz (single-threaded deflate, slow on multi-GB "
            "outputs). Default writes uncompressed for speed."
        ),
    )
    args = parser.parse_args()
    if len(args.dataset) != len(args.role):
        parser.error("--dataset and --role counts must match")
    if args.fraction is None:
        args.fraction = [1.0] * len(args.dataset)
    if len(args.fraction) != len(args.dataset):
        parser.error("--fraction and --dataset counts must match")
    if any(not 0.0 < value <= 1.0 for value in args.fraction):
        parser.error("--fraction values must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
