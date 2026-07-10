from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from boundary.sequence_features import PTM_PROJECTION_SCHEMA, ptm_projection_digest
from boundary.sequence_store import load_sequence_arrays, save_sequence_dataset
from tools.boundary.ja.project_semantic_split_deploy_features import project_dataset


def _write_checkpoint(path: Path, *, digest: str = "") -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    components = np.asarray(
        [[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]], dtype=np.float32
    )
    actual_digest = ptm_projection_digest(mean, components)
    torch.save(
        {
            "feature_config": {
                "mfcc_dim": 2,
                "ptm_projection": {
                    "schema": PTM_PROJECTION_SCHEMA,
                    "mean": mean,
                    "components": components,
                    "input_dim": 3,
                    "dim": 2,
                    "digest": digest or actual_digest,
                },
            }
        },
        path,
    )
    return mean, components


def test_project_dataset_streams_folded_ptm_and_preserves_mfcc(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "projected.npz"
    checkpoint = tmp_path / "split.pt"
    mean, components = _write_checkpoint(checkpoint)
    frames = np.asarray(
        [
            [[2.0, 4.0, 8.0, 10.0, 11.0], [1.0, 2.0, 3.0, 12.0, 13.0]],
            [[0.0, 1.0, 2.0, 14.0, 15.0], [4.0, 6.0, 9.0, 16.0, 17.0]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int64)
    save_sequence_dataset(source, frames=frames, labels=labels)

    summary = project_dataset(
        dataset_path=source,
        checkpoint_path=checkpoint,
        output_path=output,
        device="cpu",
        batch_rows=1,
    )

    result = load_sequence_arrays(output)
    expected_ptm = (frames[..., :3] - mean) @ components.T
    np.testing.assert_allclose(result["frame_features"][..., :2], expected_ptm)
    np.testing.assert_array_equal(result["frame_features"][..., 2:], frames[..., 3:])
    np.testing.assert_array_equal(result["labels"], labels)
    assert summary["output_frame_dim"] == 4
    assert summary["vram_safety_ratio"] is None
    assert output.with_suffix(".summary.json").exists()


def test_project_dataset_rejects_projection_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    checkpoint = tmp_path / "split.pt"
    _write_checkpoint(checkpoint, digest="wrong")
    save_sequence_dataset(
        source,
        frames=np.zeros((1, 1, 5), dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        project_dataset(
            dataset_path=source,
            checkpoint_path=checkpoint,
            output_path=tmp_path / "out.npz",
            device="cpu",
            batch_rows=1,
        )
