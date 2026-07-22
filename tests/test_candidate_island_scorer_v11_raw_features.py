from __future__ import annotations

import numpy as np
import pytest

from tools.boundary.ja.extract_candidate_island_scorer_v11_raw_features import (
    _safe_id,
    align_audio_to_canonical_frames,
    align_raw_features,
    parse_args,
)


def test_v11_raw_feature_alignment_keeps_full_ptm_width() -> None:
    ptm = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
    mfcc = np.arange(6 * 40, dtype=np.float32).reshape(6, 40)
    aligned_ptm, aligned_mfcc = align_raw_features(
        ptm=ptm, mfcc=mfcc, expected_frames=5
    )
    assert aligned_ptm.shape == (5, 2048)
    assert aligned_ptm.dtype == np.float32
    assert aligned_mfcc.shape == (5, 40)
    np.testing.assert_array_equal(aligned_mfcc, mfcc[:5])


def test_v11_raw_feature_alignment_rejects_projected_or_short_features() -> None:
    with pytest.raises(ValueError, match="raw PTM2048"):
        align_raw_features(
            ptm=np.zeros((3, 128), dtype=np.float32),
            mfcc=np.zeros((5, 40), dtype=np.float32),
            expected_frames=5,
        )
    with pytest.raises(ValueError, match="does not cover canonical"):
        align_raw_features(
            ptm=np.zeros((3, 2048), dtype=np.float32),
            mfcc=np.zeros((4, 40), dtype=np.float32),
            expected_frames=5,
        )


@pytest.mark.parametrize(
    ("option", "value"),
    (("--limit", "0"), ("--memory-log-every", "0"), ("--summary-every", "-1")),
)
def test_v11_raw_feature_extractor_rejects_nonpositive_counts(
    option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--canonical-sources",
                "canonical.jsonl",
                "--model-path",
                "model",
                "--output-dir",
                "output",
                option,
                value,
            ]
        )


def test_v11_raw_feature_filename_cannot_collide_after_prefix_truncation() -> None:
    prefix = "source-" + "x" * 220
    first = _safe_id(prefix + "-first")
    second = _safe_id(prefix + "-second")
    assert first != second
    assert len(first) <= 172
    assert len(second) <= 172


def test_v11_audio_geometry_keeps_partial_tail_and_trims_subframe_overrun() -> None:
    partial = np.arange(641, dtype=np.float32)
    np.testing.assert_array_equal(
        align_audio_to_canonical_frames(
            partial, expected_frames=3, declared_sample_count=641
        ),
        partial,
    )
    overrun = np.arange(965, dtype=np.float32)
    aligned = align_audio_to_canonical_frames(
        overrun, expected_frames=3, declared_sample_count=965
    )
    assert aligned.shape == (960,)
    with pytest.raises(ValueError, match="geometry mismatch"):
        align_audio_to_canonical_frames(np.zeros(640), expected_frames=3)
