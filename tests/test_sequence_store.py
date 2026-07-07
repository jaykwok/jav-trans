from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary.sequence_store import (
    StreamingFrameWriter,
    chunked_frame_stats,
    frames_sidecar_path,
    load_sequence_arrays,
    save_sequence_dataset,
    transform_window_npz,
)


def test_streaming_writer_roundtrips_memmap_frames(tmp_path: Path) -> None:
    dataset = tmp_path / "seq.npz"
    writer = StreamingFrameWriter(dataset)
    first = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    second = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) + 100.0
    writer.append(first)
    writer.append(second)  # single row (bins, dim) form
    sidecar, shape = writer.finalize()

    assert sidecar == frames_sidecar_path(dataset)
    assert shape == (3, 4, 3)
    save_sequence_dataset(
        dataset,
        frames_finalized=True,
        labels=np.asarray([0, 1, 0], dtype=np.int64),
    )
    arrays = load_sequence_arrays(dataset)
    assert isinstance(arrays["frame_features"], np.memmap)
    np.testing.assert_array_equal(np.asarray(arrays["frame_features"][:2]), first)
    np.testing.assert_array_equal(np.asarray(arrays["frame_features"][2]), second)
    assert not sidecar.with_suffix(".npy.tmp").exists()


def test_streaming_writer_rejects_shape_change_and_empty(tmp_path: Path) -> None:
    writer = StreamingFrameWriter(tmp_path / "seq.npz")
    writer.append(np.zeros((1, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape changed"):
        writer.append(np.zeros((1, 5, 3), dtype=np.float32))

    empty = StreamingFrameWriter(tmp_path / "empty.npz")
    with pytest.raises(ValueError, match="no frame rows"):
        empty.finalize()


def test_legacy_in_npz_datasets_load_unchanged(tmp_path: Path) -> None:
    dataset = tmp_path / "legacy.npz"
    frames = np.ones((2, 4, 3), dtype=np.float32)
    np.savez(dataset, frame_features=frames, labels=np.asarray([0, 1]))
    arrays = load_sequence_arrays(dataset)
    np.testing.assert_array_equal(arrays["frame_features"], frames)


def test_chunked_frame_stats_matches_dense_computation(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    frames = rng.normal(2.0, 3.0, size=(50, 4, 6)).astype(np.float32)
    rows = np.asarray([0, 3, 7, 20, 49], dtype=np.int64)

    mean, std = chunked_frame_stats(frames, rows, chunk_rows=2)

    dense = frames[rows]
    np.testing.assert_allclose(mean, dense.mean(axis=(0, 1)), rtol=1e-5)
    np.testing.assert_allclose(std, dense.std(axis=(0, 1)), rtol=1e-4)


def test_transform_window_npz_applies_concurrently_and_preserves_arrays(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(3)
    paths = []
    originals = []
    for index in range(8):
        path = tmp_path / f"w{index}.npz"
        frames = rng.normal(size=(4, 3, 10)).astype(np.float32)
        originals.append(frames)
        np.savez_compressed(
            path,
            frame_features=frames,
            labels=np.asarray([0, 1, 0, 1]),
        )
        paths.append(path)

    written = transform_window_npz(
        paths,
        lambda arrays: {
            **arrays,
            "frame_features": arrays["frame_features"] * 2.0,
        },
        workers=4,
    )

    assert written == 8
    for path, frames in zip(paths, originals):
        with np.load(path) as handle:
            np.testing.assert_allclose(handle["frame_features"], frames * 2.0)
            assert handle["labels"].tolist() == [0, 1, 0, 1]


def test_transform_window_npz_skips_when_transform_returns_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "w.npz"
    original = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np.savez_compressed(path, frame_features=original)

    written = transform_window_npz([path], lambda arrays: None)

    assert written == 0
    with np.load(path) as handle:
        np.testing.assert_array_equal(handle["frame_features"], original)
