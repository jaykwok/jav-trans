from __future__ import annotations

import numpy as np
import pytest

from tools.boundary.ja.evaluate_semantic_split_v1_sequence import adapt_legacy_frames


def test_adapt_legacy_frames_keeps_ptm_prefix_and_non_ptm_tail() -> None:
    frames = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    adapted = adapt_legacy_frames(frames, raw_ptm_dim=7, expected_frame_dim=5)
    expected = np.concatenate((frames[..., :2], frames[..., 7:]), axis=-1)
    np.testing.assert_array_equal(adapted, expected)


def test_adapt_legacy_frames_rejects_impossible_dimensions() -> None:
    frames = np.zeros((2, 3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="legacy PTM dimension exceeds"):
        adapt_legacy_frames(frames, raw_ptm_dim=4, expected_frame_dim=12)
