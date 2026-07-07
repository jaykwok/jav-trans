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


SEQUENCE_KEYS = ("group_ids", "times_s", "structural_roles", "pair_ids", "omni_aux")
_FRAME_COPY_CHUNK = 16384


def stratified_sample_indexes(
    bundle: dict,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    total = int(bundle["labels"].shape[0])
    if fraction >= 1.0:
        return np.arange(total, dtype=np.int64)
    labels = bundle["labels"].astype(np.int64)
    partitions = bundle["partitions"].astype(str)
    selected: list[np.ndarray] = []
    for partition in sorted(set(partitions.tolist())):
        for label in sorted(set(labels[partitions == partition].tolist())):
            indexes = np.flatnonzero((partitions == partition) & (labels == label))
            count = max(1, int(round(indexes.size * fraction)))
            selected.append(
                np.sort(rng.choice(indexes, size=min(count, indexes.size), replace=False))
            )
    return np.sort(np.concatenate(selected))


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
    repeats = [max(1, int(value)) for value in args.repeat]
    fractions = [float(value) for value in args.fraction]
    rng = np.random.default_rng(args.seed)
    mode = _bundle_mode(bundles, args.dataset)
    if mode == "sequence":
        _run_sequence_mode(
            args,
            bundles=bundles,
            repeats=repeats,
            fractions=fractions,
            rng=rng,
            output=output,
        )
        return
    sampled_indexes = [
        stratified_sample_indexes(bundle, fraction=fraction, rng=rng)
        for bundle, fraction in zip(bundles, fractions)
    ]
    roles = np.concatenate(
        [
            np.full(int(indexes.size) * repeat, role)
            for role, indexes, repeat in zip(args.role, sampled_indexes, repeats)
        ]
    )
    def repeated(
        bundle: dict,
        key: str,
        indexes: np.ndarray,
        repeat: int,
    ) -> np.ndarray:
        values = bundle[key][indexes]
        return (
            values
            if repeat == 1
            else np.concatenate([values] * repeat, axis=0)
        )

    save = np.savez_compressed if getattr(args, "compress", False) else np.savez
    save(
        output,
        frame_features=np.concatenate(
            [
                repeated(bundle, "frame_features", indexes, repeat).astype(np.float32)
                for bundle, indexes, repeat in zip(
                    bundles, sampled_indexes, repeats
                )
            ]
        ),
        scalar_features=np.concatenate(
            [
                repeated(bundle, "scalar_features", indexes, repeat).astype(np.float32)
                for bundle, indexes, repeat in zip(
                    bundles, sampled_indexes, repeats
                )
            ]
        ),
        labels=np.concatenate(
            [
                repeated(bundle, "labels", indexes, repeat).astype(np.int64)
                for bundle, indexes, repeat in zip(
                    bundles, sampled_indexes, repeats
                )
            ]
        ),
        partitions=np.concatenate(
            [
                repeated(bundle, "partitions", indexes, repeat).astype(str)
                for bundle, indexes, repeat in zip(
                    bundles, sampled_indexes, repeats
                )
            ]
        ),
        dataset_roles=roles,
    )
    summary = {
        "schema": "semantic_split_merged_dataset_v1",
        "mode": mode,
        "output": str(output),
        "count": int(
            sum(
                indexes.size * repeat
                for indexes, repeat in zip(sampled_indexes, repeats)
            )
        ),
        "sources": [
            {
                "path": str(path),
                "role": role,
                "count": int(bundle["labels"].shape[0]),
                "fraction": fraction,
                "sampled_count": int(indexes.size),
                "repeat": repeat,
                "effective_count": int(indexes.size * repeat),
            }
            for path, role, bundle, fraction, indexes, repeat in zip(
                args.dataset,
                args.role,
                bundles,
                fractions,
                sampled_indexes,
                repeats,
            )
        ],
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def _run_sequence_mode(
    args: argparse.Namespace,
    *,
    bundles: list[dict],
    repeats: list[int],
    fractions: list[float],
    rng: np.random.Generator,
    output: Path,
) -> None:
    """Merge island groups whole; repeats duplicate groups under new ids.

    Frames are never fully materialized: source frames arrive memmap-backed
    (or in-memory for legacy npz) and stream chunk-by-chunk into a pre-sized
    output sidecar, so merged multi-GB datasets fit a 16GB-RAM box.
    """

    plans: list[dict] = []
    for path, role, bundle, fraction, repeat in zip(
        args.dataset, args.role, bundles, fractions, repeats
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
        plans.append(
            {
                "path": path,
                "role": role,
                "bundle": bundle,
                "repeat": repeat,
                "fraction": fraction,
                "group_count": len(groups),
                "selected": selected,
                "flat": flat,
                "lengths": lengths,
            }
        )
    total_rows = sum(int(plan["flat"].size) * int(plan["repeat"]) for plan in plans)
    if total_rows == 0:
        raise ValueError("sequence merge selected no rows")
    frame_shape = tuple(bundles[0]["frame_features"].shape[1:])
    out_frames = open_frames_memmap_for_write(
        output, rows=total_rows, row_shape=frame_shape
    )
    scalar_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    partition_parts: list[np.ndarray] = []
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
        repeat = int(plan["repeat"])
        frames_all = bundle["frame_features"]
        scalars_all = np.asarray(bundle["scalar_features"], dtype=np.float32)
        labels_all = np.asarray(bundle["labels"], dtype=np.int64)
        partitions_all = bundle["partitions"].astype(str)
        total = int(labels_all.shape[0])
        defaults_all = _sequence_defaults(bundle, np.arange(total, dtype=np.int64))
        sampled_rows = 0
        for repeat_index in range(repeat):
            suffix = "" if repeat_index == 0 else f"#r{repeat_index}"
            sampled_rows += int(flat.size)
            for start in range(0, int(flat.size), _FRAME_COPY_CHUNK):
                chunk = flat[start : start + _FRAME_COPY_CHUNK]
                out_frames[row_offset : row_offset + chunk.size] = frames_all[chunk]
                row_offset += int(chunk.size)
            scalar_parts.append(scalars_all[flat])
            label_parts.append(labels_all[flat])
            partition_parts.append(partitions_all[flat])
            role_parts.append(np.full(int(flat.size), role))
            group_parts.append(
                np.repeat(
                    np.asarray([f"{role}::{name}{suffix}" for name in selected]),
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
                "repeat": repeat,
                "effective_group_count": len(selected) * repeat,
                "effective_count": sampled_rows,
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
        dataset_roles=np.concatenate(role_parts),
        group_ids=group_ids,
        times_s=np.concatenate(time_parts),
        structural_roles=np.concatenate(structural_parts),
        pair_ids=np.concatenate(pair_parts),
        omni_aux=np.concatenate(omni_parts),
        offset_targets_s=np.concatenate(offset_parts),
    )
    summary = {
        "schema": "semantic_split_merged_dataset_v2",
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
    parser.add_argument("--repeat", action="append", type=int)
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
    if args.repeat is None:
        args.repeat = [1] * len(args.dataset)
    if len(args.repeat) != len(args.dataset):
        parser.error("--repeat and --dataset counts must match")
    if any(value <= 0 for value in args.repeat):
        parser.error("--repeat values must be positive")
    if args.fraction is None:
        args.fraction = [1.0] * len(args.dataset)
    if len(args.fraction) != len(args.dataset):
        parser.error("--fraction and --dataset counts must match")
    if any(not 0.0 < value <= 1.0 for value in args.fraction):
        parser.error("--fraction values must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
