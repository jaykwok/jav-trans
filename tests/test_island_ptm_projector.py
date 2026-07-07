from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boundary.split_model import IslandCandidateSequenceNetwork

PTM_FULL = 32
PTM_PROJECTED = 8
MFCC = 3
FRAME_DIM = PTM_PROJECTED + MFCC
BINS = 20
SCALARS = 13


def _network(**overrides):
    config = {
        "frame_dim": FRAME_DIM,
        "scalar_dim": SCALARS,
        "hidden_size": 32,
        "candidate_layers": 1,
        "island_layers": 1,
        "head_dim": 16,
        "ptm_input_dim": PTM_FULL,
        "ptm_projected_dim": PTM_PROJECTED,
    }
    config.update(overrides)
    return IslandCandidateSequenceNetwork(**config)


def test_projector_identity_init_matches_truncation() -> None:
    model = _network().eval()
    ptm = torch.randn(4, BINS, PTM_FULL)
    projected = model.ptm_projector(ptm)
    torch.testing.assert_close(projected, ptm[..., :PTM_PROJECTED])


def test_forward_accepts_full_dim_frames() -> None:
    model = _network().eval()
    frames = torch.randn(2, 3, BINS, PTM_FULL + MFCC)
    scalars = torch.randn(2, 3, SCALARS)
    mask = torch.ones(2, 3)
    outputs = model(frames, scalars, mask)
    assert outputs["gate"].shape == (2, 3)
    assert outputs["offset"].shape == (2, 3)


def test_forward_rejects_wrong_full_dim() -> None:
    model = _network().eval()
    frames = torch.randn(1, 2, BINS, FRAME_DIM)
    scalars = torch.randn(1, 2, SCALARS)
    mask = torch.ones(1, 2)
    with pytest.raises(ValueError, match="PTM"):
        model(frames, scalars, mask)


def test_projector_config_validation() -> None:
    with pytest.raises(ValueError, match="together"):
        _network(ptm_projected_dim=0)
    with pytest.raises(ValueError, match="room"):
        _network(frame_dim=PTM_PROJECTED, ptm_projected_dim=PTM_PROJECTED)


def test_residual_projector_step0_equals_linear() -> None:
    """A-residual with zero-init final layer: z_A == z_B at step 0."""

    linear = _network(ptm_projector_residual=False).eval()
    residual = _network(ptm_projector_residual=True).eval()
    # Share the linear projector weights so the only difference is the residual.
    residual.ptm_projector.load_state_dict(linear.ptm_projector.state_dict())
    ptm = torch.randn(4, BINS, PTM_FULL)
    with torch.no_grad():
        z_b = linear.ptm_projector(ptm)
        z_a = residual.ptm_projector(ptm) + residual.ptm_residual(ptm)
    torch.testing.assert_close(z_a, z_b)
    # Residual final layer must be exactly zero at init.
    torch.testing.assert_close(residual.ptm_residual[-1].weight, torch.zeros_like(residual.ptm_residual[-1].weight))


def test_residual_forward_matches_linear_at_init() -> None:
    linear = _network(ptm_projector_residual=False).eval()
    residual = _network(ptm_projector_residual=True).eval()
    residual.ptm_projector.load_state_dict(linear.ptm_projector.state_dict())
    # Match all other submodules so step-0 forward is identical.
    for name, param in linear.state_dict().items():
        if name.startswith("ptm_projector") or name.startswith("ptm_residual"):
            continue
        residual.state_dict()[name].copy_(param)
    frames = torch.randn(2, 3, BINS, PTM_FULL + MFCC)
    scalars = torch.randn(2, 3, SCALARS)
    mask = torch.ones(2, 3)
    with torch.no_grad():
        out_b = linear(frames, scalars, mask)
        out_a = residual(frames, scalars, mask)
    for key in ("gate", "offset", "label"):
        torch.testing.assert_close(out_a[key], out_b[key], atol=1e-6, rtol=1e-5)


def test_checkpoint_roundtrip_keeps_projector() -> None:
    model = _network()
    with torch.no_grad():
        model.ptm_projector.weight.normal_()
    clone = _network()
    clone.load_state_dict(model.state_dict())
    ptm = torch.randn(2, BINS, PTM_FULL)
    torch.testing.assert_close(
        clone.ptm_projector(ptm), model.ptm_projector(ptm)
    )


def test_normalization_fold_reproduces_model_ptm_path() -> None:
    """(x-mu)/sigma @ W.T == (x-mu) @ (W/sigma).T — the trainer export."""

    rng = np.random.default_rng(7)
    weight = rng.normal(size=(PTM_PROJECTED, PTM_FULL))
    mean = rng.normal(size=PTM_FULL)
    std = rng.uniform(0.5, 2.0, size=PTM_FULL)
    raw = rng.normal(size=(50, PTM_FULL))
    model_path = ((raw - mean) / std) @ weight.T
    components = weight / std[None, :]
    folded = (raw - mean) @ components.T
    np.testing.assert_allclose(folded, model_path, atol=1e-10)
