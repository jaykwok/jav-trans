from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools.boundary.ja.merge_semantic_split_datasets import (
    run,
    stratified_sample_groups,
    stratified_sample_indexes,
)


def test_stratified_fraction_preserves_each_partition_label(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    labels = np.asarray([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
    partitions = np.asarray(["train"] * 20 + ["val"] * 20)
    np.savez(
        path,
        labels=labels,
        partitions=partitions,
        frame_features=np.zeros((40, 2, 3), dtype=np.float32),
        scalar_features=np.zeros((40, 2), dtype=np.float32),
    )
    bundle = np.load(path)

    indexes = stratified_sample_indexes(
        bundle,
        fraction=0.25,
        rng=np.random.default_rng(7),
    )

    assert indexes.size == 8
    strata = {
        (str(partitions[index]), int(labels[index]))
        for index in indexes
    }
    assert strata == {("train", 0), ("train", 1), ("val", 0), ("val", 1)}


def _write_sequence_bundle(
    path: Path,
    *,
    groups: list[tuple[str, list[int], str]],
    pair_ids: list[int] | None = None,
) -> None:
    labels: list[int] = []
    group_ids: list[str] = []
    partitions: list[str] = []
    for name, group_labels, partition in groups:
        labels.extend(group_labels)
        group_ids.extend([name] * len(group_labels))
        partitions.extend([partition] * len(group_labels))
    count = len(labels)
    np.savez(
        path,
        frame_features=np.arange(count * 2 * 3, dtype=np.float32).reshape(count, 2, 3),
        scalar_features=np.zeros((count, 2), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        partitions=np.asarray(partitions),
        group_ids=np.asarray(group_ids),
        times_s=np.arange(count, dtype=np.float32),
        structural_roles=np.full(count, -100, dtype=np.int64),
        pair_ids=np.asarray(
            pair_ids if pair_ids is not None else [-1] * count, dtype=np.int64
        ),
        omni_aux=np.full((count, 3), -1.0, dtype=np.float32),
    )


def test_sequence_mode_keeps_islands_whole_and_remaps_pairs(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_sequence_bundle(
        first,
        groups=[
            ("a|island0", [0, 1, 0], "train"),
            ("a|island1", [1, 1], "val"),
        ],
        pair_ids=[0, -1, 0, -1, -1],
    )
    _write_sequence_bundle(
        second,
        groups=[("b|island0", [0, 0], "train")],
        pair_ids=[0, 0],
    )
    output = tmp_path / "merged.npz"
    run(
        argparse.Namespace(
            dataset=[str(first), str(second)],
            role=["real", "hardmix"],
            repeat=[1, 2],
            fraction=[1.0, 1.0],
            seed=3,
            output=str(output),
        )
    )
    merged = np.load(output)
    group_ids = merged["group_ids"].astype(str).tolist()
    # Islands stay whole and repeats create distinct island ids.
    assert group_ids.count("real::a|island0") == 3
    assert group_ids.count("real::a|island1") == 2
    assert group_ids.count("hardmix::b|island0") == 2
    assert group_ids.count("hardmix::b|island0#r1") == 2
    # Frames live in the memmap sidecar and match the source rows.
    from boundary.sequence_store import load_sequence_arrays

    arrays = load_sequence_arrays(output)
    assert "frame_features" not in merged
    assert arrays["frame_features"].shape == (9, 2, 3)
    first_frames = np.load(first)["frame_features"]
    np.testing.assert_array_equal(np.asarray(arrays["frame_features"][:3]), first_frames[:3])
    # Pair ids from different bundles / repeats never collide.
    pairs = merged["pair_ids"]
    real_pair = {int(v) for v in pairs[:3] if v >= 0}
    hardmix_pairs = {int(v) for v in pairs[5:] if v >= 0}
    assert len(real_pair) == 1
    assert len(hardmix_pairs) == 2
    assert not (real_pair & hardmix_pairs)
    summary = json.loads(output.with_suffix(".summary.json").read_text("utf-8"))
    assert summary["mode"] == "sequence"
    assert summary["group_count"] == 4


def test_sequence_group_sampling_is_stratified(tmp_path: Path) -> None:
    path = tmp_path / "bundle.npz"
    groups = []
    for index in range(10):
        groups.append((f"cut{index}", [0, 1], "train"))
        groups.append((f"nocut{index}", [1, 1], "train"))
    _write_sequence_bundle(path, groups=groups)
    bundle = np.load(path)
    selected = stratified_sample_groups(
        bundle, fraction=0.3, rng=np.random.default_rng(5)
    )
    has_cut = [name for name in selected if name.startswith("cut")]
    no_cut = [name for name in selected if name.startswith("nocut")]
    assert len(has_cut) == 3
    assert len(no_cut) == 3


def test_mixed_grouped_and_ungrouped_inputs_error(tmp_path: Path) -> None:
    grouped = tmp_path / "grouped.npz"
    _write_sequence_bundle(grouped, groups=[("a|island0", [0, 1], "train")])
    flat = tmp_path / "flat.npz"
    np.savez(
        flat,
        frame_features=np.zeros((2, 2, 3), dtype=np.float32),
        scalar_features=np.zeros((2, 2), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int64),
        partitions=np.asarray(["train", "train"]),
    )
    with pytest.raises(ValueError, match="cannot mix"):
        run(
            argparse.Namespace(
                dataset=[str(grouped), str(flat)],
                role=["real", "legacy"],
                repeat=[1, 1],
                fraction=[1.0, 1.0],
                seed=3,
                output=str(tmp_path / "merged.npz"),
            )
        )
