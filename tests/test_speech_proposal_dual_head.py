from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from boundary.ja.dual_head import (
    SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH,
    SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA,
    build_dual_head_checkpoint,
    build_dual_head_model,
    load_dual_head_checkpoint,
    score_dual_head_probabilities_batch,
)
from boundary.backbones import SpeechIslandSequenceClassifier
from tools.boundary.ja.train_speech_proposal_dual_head import (
    _pack_features,
    initialize_dual_head_from_teachers,
)


def _config() -> dict:
    return {
        "raw_ptm_dim": 4,
        "projected_ptm_dim": 2,
        "mfcc_dim": 2,
        "input_dim": 6,
        "encoder_input_dim": 4,
        "hidden_size": 16,
        "num_layers": 1,
        "state_size": 8,
        "num_heads": 4,
        "n_groups": 1,
        "chunk_size": 8,
        "conv_kernel": 3,
        "bidirectional": True,
        "model_arch": SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH,
        "output_dim": 2,
    }


def _normalization() -> dict:
    return {
        "schema": SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA,
        "mfcc_mean": [0.25, -0.5],
        "mfcc_std": [2.0, 0.5],
    }


def test_dual_head_checkpoint_roundtrip_scores_both_heads(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    model = build_dual_head_model(_config(), _normalization())
    path = tmp_path / "dual.pt"
    torch.save(
        build_dual_head_checkpoint(
            model=model,
            model_config=_config(),
            normalization=_normalization(),
            metadata={"ptm_repo_id": "test/repo", "trained_steps": 1},
        ),
        path,
    )
    bundle = load_dual_head_checkpoint(path, device="cpu")
    scored = score_dual_head_probabilities_batch(
        bundle,
        feature_pairs=[
            (np.zeros((5, 4), dtype=np.float32), np.zeros((5, 2), dtype=np.float32))
        ],
    )
    assert scored[0][0].shape == (5,)
    assert scored[0][1].shape == (5,)
    assert bundle.signature()["metadata"]["output_heads"] == [
        "speech_prob",
        "boundary_prob",
    ]


def test_dual_head_rejects_single_output_config() -> None:
    config = _config()
    config["output_dim"] = 1
    with pytest.raises(ValueError, match="output_dim=2"):
        build_dual_head_model(config, _normalization())


def test_dual_head_initialization_copies_both_teacher_heads() -> None:
    pytest.importorskip("transformers")
    dual = build_dual_head_model(_config(), _normalization())
    speech = SpeechIslandSequenceClassifier(
        input_dim=4, hidden_size=16, num_layers=1, output_dim=1,
        state_size=8, num_heads=4, n_groups=1, chunk_size=8,
        conv_kernel=3, bidirectional=True,
    )
    proposal = SpeechIslandSequenceClassifier(
        input_dim=4, hidden_size=16, num_layers=1, output_dim=1,
        state_size=8, num_heads=4, n_groups=1, chunk_size=8,
        conv_kernel=3, bidirectional=True,
    )
    with torch.no_grad():
        speech.head.weight.fill_(1.0)
        speech.head.bias.fill_(2.0)
        proposal.head.weight.fill_(3.0)
        proposal.head.bias.fill_(4.0)
    teacher_normalization = {
        "feature_mean": [1.0, -2.0, 0.25, -0.5],
        "feature_std": [2.0, 4.0, 2.0, 0.5],
    }
    initialize_dual_head_from_teachers(
        dual, speech, proposal, teacher_normalization
    )
    torch.testing.assert_close(dual.encoder.head.weight[0], speech.head.weight[0])
    torch.testing.assert_close(dual.encoder.head.bias[0], speech.head.bias[0])
    torch.testing.assert_close(dual.encoder.head.weight[1], proposal.head.weight[0])
    torch.testing.assert_close(dual.encoder.head.bias[1], proposal.head.bias[0])


def test_dual_head_full_ptm_projection_warm_start_matches_speech_teacher() -> None:
    pytest.importorskip("transformers")
    torch.manual_seed(7)
    dual = build_dual_head_model(_config(), _normalization())
    speech = SpeechIslandSequenceClassifier(
        input_dim=4, hidden_size=16, num_layers=1, output_dim=1,
        state_size=8, num_heads=4, n_groups=1, chunk_size=8,
        conv_kernel=3, bidirectional=True,
    )
    proposal = SpeechIslandSequenceClassifier(
        input_dim=4, hidden_size=16, num_layers=1, output_dim=1,
        state_size=8, num_heads=4, n_groups=1, chunk_size=8,
        conv_kernel=3, bidirectional=True,
    )
    teacher_normalization = {
        "feature_mean": [1.0, -2.0, 0.25, -0.5],
        "feature_std": [2.0, 4.0, 2.0, 0.5],
    }
    initialize_dual_head_from_teachers(
        dual, speech, proposal, teacher_normalization
    )
    dual.eval()
    speech.eval()
    ptm = torch.randn(1, 6, 4)
    mfcc = torch.randn(1, 6, 2)
    teacher_features = torch.cat((ptm[..., :2], mfcc), dim=-1)
    mean = torch.tensor(teacher_normalization["feature_mean"])
    std = torch.tensor(teacher_normalization["feature_std"])
    with torch.inference_mode():
        actual = dual(ptm, mfcc)[..., 0]
        expected = speech((teacher_features - mean) / std)[..., 0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(dual.ptm_projector.weight[:, 2:]) == 0


def test_dual_head_projection_can_learn_from_ptm_dimensions_beyond_prefix() -> None:
    pytest.importorskip("transformers")
    torch.manual_seed(11)
    dual = build_dual_head_model(_config(), _normalization())
    ptm = torch.randn(1, 5, 4)
    mfcc = torch.randn(1, 5, 2)
    dual(ptm, mfcc).sum().backward()
    assert torch.count_nonzero(dual.ptm_projector.weight.grad[:, 2:]) > 0


def test_packed_feature_cache_reads_full_ptm_by_file_offset(tmp_path: Path) -> None:
    rows = []
    expected = {}
    for index, frames in enumerate((3, 5)):
        path = tmp_path / f"source-{index}.npz"
        ptm = np.arange(frames * 4, dtype=np.float32).reshape(frames, 4) + index
        mfcc = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) - index
        np.savez(path, ptm=ptm, mfcc=mfcc)
        rows.append({"feature_path": str(path), "frame_count": frames})
        expected[str(path)] = (ptm, mfcc)
    cache_dir = tmp_path / "packed"
    cache = _pack_features(
        rows,
        cache_dir=cache_dir,
        raw_ptm_dim=4,
        mfcc_dim=2,
        workers=2,
    )
    reused = _pack_features(
        rows,
        cache_dir=cache_dir,
        raw_ptm_dim=4,
        mfcc_dim=2,
        workers=2,
    )
    for path, (ptm, mfcc) in expected.items():
        np.testing.assert_array_equal(cache[path][0], ptm)
        np.testing.assert_array_equal(cache[path][1], mfcc)
        np.testing.assert_array_equal(reused[path][0], ptm)
