from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES
from boundary.sequence_store import frames_sidecar_path, save_sequence_dataset
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA
from tools.boundary.ja.acoustic_split_v4_dataset import (
    SPLIT_V4_DATASET_SUMMARY_SCHEMA,
    SPLIT_V4_INPUT_DISTRIBUTION,
    SPLIT_V4_MFCC_DIM,
    SPLIT_V4_PTM_DIM,
    SPLIT_V4_UPSTREAM_SHA_FIELDS,
    file_sha256,
)
from tools.boundary.ja.merge_semantic_split_datasets import (
    run,
    stratified_sample_groups,
)


UPSTREAM_SHA256 = {
    field: character * 64
    for field, character in zip(SPLIT_V4_UPSTREAM_SHA_FIELDS, "abc")
}


def _write_sequence_bundle(
    path: Path,
    *,
    groups: list[tuple[str, list[int], str]],
    pair_ids: list[int] | None = None,
) -> None:
    labels: list[int] = []
    group_ids: list[str] = []
    partitions: list[str] = []
    source_ids: list[str] = []
    core_ids: list[str] = []
    for name, group_labels, partition in groups:
        source_id = name.split("|island", 1)[0]
        core_id = f"{name}:core"
        labels.extend(group_labels)
        group_ids.extend([name] * len(group_labels))
        partitions.extend([partition] * len(group_labels))
        source_ids.extend([source_id] * len(group_labels))
        core_ids.extend([core_id] * len(group_labels))
    count = len(labels)
    frame_dim = SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM
    scalar_dim = len(SPLIT_CANDIDATE_SCALAR_NAMES)
    frames = np.arange(count * 2 * frame_dim, dtype=np.float32).reshape(
        count, 2, frame_dim
    )
    save_sequence_dataset(
        path,
        frames=frames,
        scalar_features=np.zeros((count, scalar_dim), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        partitions=np.asarray(partitions),
        source_ids=np.asarray(source_ids),
        core_ids=np.asarray(core_ids),
        group_ids=np.asarray(group_ids),
        times_s=np.arange(count, dtype=np.float32),
        structural_roles=np.full(count, -100, dtype=np.int64),
        pair_ids=np.asarray(
            pair_ids if pair_ids is not None else [-1] * count, dtype=np.int64
        ),
        omni_aux=np.full((count, 3), -1.0, dtype=np.float32),
    )
    summary = {
        "schema": SPLIT_V4_DATASET_SUMMARY_SCHEMA,
        "training_manifest_allowed": True,
        "output": str(path.resolve()),
        "dataset_sha256": file_sha256(path),
        "frame_sidecar_sha256": file_sha256(frames_sidecar_path(path)),
        "input_distribution": SPLIT_V4_INPUT_DISTRIBUTION,
        "feature_schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": SPLIT_V4_PTM_DIM,
        "mfcc_dim": SPLIT_V4_MFCC_DIM,
        "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
        **UPSTREAM_SHA256,
        "count": count,
        "frame_bins": 2,
        "frame_dim": frame_dim,
        "scalar_dim": scalar_dim,
        "group_count": len(groups),
        "source_count": len(set(source_ids)),
        "core_count": len(set(core_ids)),
        "source_feature_audio_sha256": {
            path.stem: {
                "feature_path": str(path.resolve()),
                "feature_sha256": "d" * 64,
                "audio_wav": str((path.parent / f"{path.stem}.wav").resolve()),
                "audio_wav_sha256": "e" * 64,
            }
        },
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_sequence_mode_keeps_islands_whole_and_remaps_pairs(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_sequence_bundle(
        first,
        groups=[
            ("a|island0", [0, 1, 0], "train"),
            ("c|island1", [1, 1], "val"),
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
            fraction=[1.0, 1.0],
            seed=3,
            output=str(output),
        )
    )
    merged = np.load(output)
    group_ids = merged["group_ids"].astype(str).tolist()
    # Islands stay whole; current training data never duplicates a core.
    assert group_ids.count("real::a|island0") == 3
    assert group_ids.count("real::c|island1") == 2
    assert group_ids.count("hardmix::b|island0") == 2
    # Frames live in the memmap sidecar and match the source rows.
    from boundary.sequence_store import load_sequence_arrays

    arrays = load_sequence_arrays(output)
    assert "frame_features" not in merged
    assert arrays["frame_features"].shape == (
        7,
        2,
        SPLIT_V4_PTM_DIM + SPLIT_V4_MFCC_DIM,
    )
    assert arrays["source_ids"].astype(str).tolist() == ["a"] * 3 + ["c"] * 2 + ["b"] * 2
    first_frames = np.load(frames_sidecar_path(first))
    np.testing.assert_array_equal(np.asarray(arrays["frame_features"][:3]), first_frames[:3])
    # Pair ids from different bundles / repeats never collide.
    pairs = merged["pair_ids"]
    real_pair = {int(v) for v in pairs[:3] if v >= 0}
    hardmix_pairs = {int(v) for v in pairs[5:] if v >= 0}
    assert len(real_pair) == 1
    assert len(hardmix_pairs) == 1
    assert not (real_pair & hardmix_pairs)
    summary = json.loads(output.with_suffix(".summary.json").read_text("utf-8"))
    assert summary["mode"] == "sequence"
    assert summary["schema"] == SPLIT_V4_DATASET_SUMMARY_SCHEMA
    assert summary["training_manifest_allowed"] is True
    assert summary["group_count"] == 3
    assert summary["dataset_sha256"] == file_sha256(output)


def test_sequence_group_sampling_is_stratified(tmp_path: Path) -> None:
    path = tmp_path / "bundle.npz"
    groups = []
    for index in range(10):
        groups.append((f"cut{index}", [0, 1], "train"))
        groups.append((f"nocut{index}", [1, 1], "train"))
    _write_sequence_bundle(path, groups=groups)
    from boundary.sequence_store import load_sequence_arrays

    bundle = load_sequence_arrays(path)
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
                fraction=[1.0, 1.0],
                seed=3,
                output=str(tmp_path / "merged.npz"),
            )
        )


def test_row_wise_dataset_is_retired(tmp_path: Path) -> None:
    flat = tmp_path / "flat.npz"
    np.savez(
        flat,
        frame_features=np.zeros((2, 2, 3), dtype=np.float32),
        scalar_features=np.zeros((2, 2), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int64),
        partitions=np.asarray(["train", "train"]),
    )
    with pytest.raises(ValueError, match="row-wise Semantic Split datasets are retired"):
        run(
            argparse.Namespace(
                dataset=[str(flat)],
                role=["legacy"],
                fraction=[1.0],
                seed=3,
                output=str(tmp_path / "merged.npz"),
            )
        )


def test_merge_rejects_core_reuse_across_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_sequence_bundle(first, groups=[("a|island0", [0, 1], "train")])
    _write_sequence_bundle(second, groups=[("a|island0", [1, 1], "train")])
    with pytest.raises(ValueError, match="reused across Split merge inputs"):
        run(
            argparse.Namespace(
                dataset=[str(first), str(second)],
                role=["real", "hardmix"],
                fraction=[1.0, 1.0],
                seed=3,
                output=str(tmp_path / "merged.npz"),
            )
        )
