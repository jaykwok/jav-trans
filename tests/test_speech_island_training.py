from __future__ import annotations

import numpy as np
import pytest

from boundary.ja.speech_train import _crop, _normalize
from boundary.ja.dataset import LabelRecord
from boundary.ja.semantic_speech_train import _class_indexes
from boundary.ja.model import SemanticSpeechScorerNetwork


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


def _record(*, semantic_frames=None) -> LabelRecord:
    return LabelRecord(
        audio_id="sample",
        source="test",
        duration_s=0.06,
        text="",
        teacher_segments={},
        frame_hop_s=0.02,
        speech_frames=[0, 1, 1],
        label_quality="supervised",
        boundary_metadata=(
            {} if semantic_frames is None else {"semantic_class_frames": semantic_frames}
        ),
    )


def test_semantic_speech_v9_requires_explicit_three_class_frames() -> None:
    with pytest.raises(ValueError, match="must not train from legacy binary"):
        _class_indexes(_record(), total=3)

    indexes = _class_indexes(
        _record(
            semantic_frames=["discardable", "semantic_target", "unsure"]
        ),
        total=3,
    )
    assert indexes.tolist() == [0, 1, 2]


def test_semantic_speech_v9_uses_trainable_full_ptm_projection() -> None:
    torch = pytest.importorskip("torch")
    model = SemanticSpeechScorerNetwork(
        raw_ptm_dim=8,
        projected_ptm_dim=2,
        mfcc_dim=2,
        mfcc_mean=[0.0, 0.0],
        mfcc_std=[1.0, 1.0],
        hidden_size=8,
        num_layers=1,
        state_size=4,
        num_heads=2,
        head_dim=8,
        n_groups=1,
        conv_kernel=2,
        chunk_size=2,
    )
    assert tuple(model.ptm_projector.weight.shape) == (2, 8)
    assert model.ptm_projector.weight.requires_grad
    with torch.no_grad():
        model.ptm_projector.weight.zero_()
        model.ptm_projector.bias.zero_()
        model.ptm_projector.weight[0, 7] = 1.0
    ptm = torch.zeros((1, 1, 8))
    ptm[0, 0, 7] = 3.0
    projected = model.project_ptm(ptm)
    assert projected[0, 0, 0].item() == pytest.approx(3.0)
