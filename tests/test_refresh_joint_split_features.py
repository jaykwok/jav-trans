from __future__ import annotations

import numpy as np
import pytest

from boundary.sequence_features import parse_extra_context_scales
from tools.datasets.refresh_joint_split_features import (
    expected_bins,
    rebuild_npz_arrays,
)


def test_expected_bins_matches_builder_layout() -> None:
    assert expected_bins([]) == 20
    scales = parse_extra_context_scales("3.2:4,6.4:4")
    assert expected_bins(scales) == 36


def _original(count: int) -> dict[str, np.ndarray]:
    return {
        "frame_features": np.zeros((count, 20, 168), dtype=np.float32),
        "scalar_features": np.zeros((count, 13), dtype=np.float32),
        "proposal_times_s": np.arange(count, dtype=np.float32),
        "accepted": np.ones(count, dtype=np.int64),
        "p_cut": np.full(count, 0.25, dtype=np.float32),
    }


def test_rebuild_preserves_aux_arrays_and_swaps_features() -> None:
    original = _original(3)
    frames = [np.full((36, 168), i, dtype=np.float32) for i in range(3)]
    scalars = [np.full(13, i, dtype=np.float32) for i in range(3)]
    arrays = rebuild_npz_arrays(original, frames, scalars)
    assert arrays["frame_features"].shape == (3, 36, 168)
    assert float(arrays["frame_features"][2, 0, 0]) == 2.0
    assert float(arrays["scalar_features"][1, 0]) == 1.0
    np.testing.assert_array_equal(
        arrays["proposal_times_s"], original["proposal_times_s"]
    )
    np.testing.assert_array_equal(arrays["p_cut"], original["p_cut"])
    np.testing.assert_array_equal(arrays["accepted"], original["accepted"])


def test_rebuild_rejects_row_count_mismatch() -> None:
    original = _original(3)
    frames = [np.zeros((36, 168), dtype=np.float32)] * 2
    scalars = [np.zeros(13, dtype=np.float32)] * 2
    with pytest.raises(ValueError, match="row count mismatch"):
        rebuild_npz_arrays(original, frames, scalars)
