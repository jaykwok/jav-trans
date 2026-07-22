from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asr.backends import qwen
from boundary.ja.model import (
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities_batch,
)
from boundary.ja.proposal import load_boundary_proposal_checkpoint


@pytest.mark.parametrize(
    "loader,checkpoint",
    [
        (
            load_speech_island_scorer_checkpoint,
            Path(
                qwen.repo_checkpoint_path(
                    qwen.QWEN_ASR_17B_REPO_ID,
                    "speech_island_scorer",
                    "v8",
                )
            ),
        ),
        (
            load_boundary_proposal_checkpoint,
            Path(
                qwen.DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO[
                    qwen.QWEN_ASR_17B_REPO_ID
                ]
            ),
        ),
    ],
)
def test_scorer_and_proposal_batching_preserves_order_and_probabilities(
    loader, checkpoint
) -> None:
    rng = np.random.default_rng(41)
    pairs = [
        (
            rng.normal(size=(frames, 128)).astype(np.float32),
            rng.normal(size=(frames, 40)).astype(np.float32),
        )
        for frames in (5, 11, 7)
    ]
    bundle = loader(checkpoint, device="cpu")
    batched = score_speech_island_probabilities_batch(bundle, feature_pairs=pairs)
    single = [
        score_speech_island_probabilities_batch(bundle, feature_pairs=[pair])[0]
        for pair in pairs
    ]

    assert [len(row) for row in batched] == [5, 11, 7]
    for batched_row, single_row in zip(batched, single, strict=True):
        np.testing.assert_allclose(batched_row, single_row, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(batched_row >= 0.5, single_row >= 0.5)
