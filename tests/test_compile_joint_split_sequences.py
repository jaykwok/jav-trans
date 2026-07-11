from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.datasets.compile_joint_boundary_preasr_dataset import _compile_split


def _write_window_bundle(path: Path) -> None:
    np.savez(
        path,
        frame_features=np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3),
        scalar_features=np.arange(4 * 2, dtype=np.float32).reshape(4, 2),
        proposal_times_s=np.asarray([0.5, 1.0, 2.2, 2.6], dtype=np.float32),
        core_starts_s=np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float32),
        core_ends_s=np.asarray([1.5, 1.5, 3.0, 3.0], dtype=np.float32),
        accepted=np.asarray([False, True, False, False]),
        p_cut=np.zeros(4, dtype=np.float32),
        p_continue=np.zeros(4, dtype=np.float32),
        p_unsure=np.zeros(4, dtype=np.float32),
    )


def test_compile_split_emits_whole_islands_with_ignore_context(tmp_path: Path) -> None:
    bundle_path = tmp_path / "w0" / "semantic_split_features.npz"
    bundle_path.parent.mkdir(parents=True)
    _write_window_bundle(bundle_path)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "semantic_split_features": str(bundle_path),
        }
    ]
    labels = [
        {
            "window_id": "w0",
            "feature_index": 1,
            "label": "cut",
            "time_s": 1.0,
            "left_complete": True,
            "right_complete": True,
            "merged_better": False,
        }
    ]

    summary = _compile_split(
        dataset=tmp_path,
        windows=windows,
        labels=labels,
        val_percent=20,
    )

    output = np.load(tmp_path / "semantic_split" / "features.npz")
    # Only the labeled island is emitted, but with ALL of its candidates.
    assert output["labels"].tolist() == [-100, 0]
    assert output["feature_indexes"].tolist() == [0, 1]
    group_ids = output["group_ids"].astype(str).tolist()
    assert len(set(group_ids)) == 1
    assert group_ids[0].startswith("w0|core0.000-")
    omni = output["omni_aux"].tolist()
    assert omni[0] == [-1.0, -1.0, -1.0]
    assert omni[1] == [1.0, 1.0, 0.0]
    assert output["structural_roles"].tolist() == [-100, -100]
    assert output["pair_ids"].tolist() == [-1, -1]
    assert summary["labeled_count"] == 1
    assert summary["context_only_count"] == 1
    assert summary["group_count"] == 1


def test_compile_split_reads_and_writes_named_variant(tmp_path: Path) -> None:
    canonical = tmp_path / "w0" / "semantic_split_features.npz"
    canonical.parent.mkdir(parents=True)
    _write_window_bundle(canonical)
    variant = canonical.with_name("semantic_split_features.06b.npz")
    _write_window_bundle(variant)
    windows = [
        {
            "window_id": "w0",
            "video_id": "vid0",
            "semantic_split_features": str(canonical),
        }
    ]
    labels = [{"window_id": "w0", "feature_index": 1, "label": "cut"}]

    summary = _compile_split(
        dataset=tmp_path,
        windows=windows,
        labels=labels,
        val_percent=20,
        feature_variant="06b",
        output_variant="06b",
    )

    assert summary["output"].endswith("features.06b.npz")
    assert (tmp_path / "semantic_split" / "features.06b.npz").exists()
    assert (tmp_path / "semantic_split" / "summary.06b.json").exists()
