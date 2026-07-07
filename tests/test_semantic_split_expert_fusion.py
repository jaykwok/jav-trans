from __future__ import annotations

import numpy as np

from tools.boundary.ja.evaluate_semantic_split_expert_fusion import (
    classification_metrics,
    fused_probabilities,
)


def test_expert_fusion_is_arithmetic_probability_mixture() -> None:
    main = np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32)
    expert = np.asarray([[0.2, 0.7, 0.1]], dtype=np.float32)

    fused = fused_probabilities(main, expert, alpha=0.25)

    np.testing.assert_allclose(fused, [[0.65, 0.25, 0.1]])


def test_runtime_gate_reports_cut_precision_and_recall() -> None:
    truth = np.asarray([0, 0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.7, 0.2, 0.1]],
        dtype=np.float32,
    )

    metrics = classification_metrics(truth, probabilities, cut_threshold=0.75)

    assert metrics["cut_recall"] == 1.0
    assert metrics["cut_precision"] == 2 / 3
    assert metrics["gated_cut_recall"] == 0.5
    assert metrics["gated_cut_precision"] == 1.0
