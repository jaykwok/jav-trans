from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.speech_train import _crop, _normalize


def test_speech_training_crop_keeps_features_labels_and_weights_aligned() -> None:
    features = np.arange(60, dtype=np.float32).reshape(10, 6)
    labels = np.arange(10, dtype=np.float32)
    weights = np.arange(10, dtype=np.float32) + 1.0
    cropped_features, cropped_labels, cropped_weights = _crop(
        features,
        labels,
        weights,
        max_frames=4,
        rng=np.random.default_rng(7),
        random=False,
    )

    assert cropped_features.shape == (4, 6)
    assert cropped_labels.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert cropped_weights.tolist() == [4.0, 5.0, 6.0, 7.0]


def test_speech_training_normalization_uses_checkpoint_statistics() -> None:
    features = np.asarray([[2.0, 5.0]], dtype=np.float32)
    normalized = _normalize(
        features,
        {"feature_mean": [1.0, 1.0], "feature_std": [1.0, 2.0]},
    )

    assert normalized[0].tolist() == pytest.approx([1.0, 2.0])
