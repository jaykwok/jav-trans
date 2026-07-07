from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from boundary.sequence_features import (
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
    ptm_projection_digest,
)

FULL_DIM = 12
PROJECTED_DIM = 5
CAPPED_DIM = 6
MFCC_DIM = 3
FRAMES = 200
HOP_S = 0.02


def _projection(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mean = rng.normal(size=FULL_DIM).astype(np.float32)
    components = rng.normal(size=(PROJECTED_DIM, FULL_DIM)).astype(np.float32)
    return mean, components


def _full_frames(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ptm = rng.normal(size=(FRAMES, FULL_DIM)).astype(np.float32)
    mfcc = rng.normal(size=(FRAMES, MFCC_DIM)).astype(np.float32)
    return ptm, mfcc


def _pre_projected(
    ptm: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    centered = ptm.astype(np.float64) - mean.astype(np.float64)
    return (centered @ components.T.astype(np.float64)).astype(np.float32)


def _split_kwargs(
    speech: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> dict:
    return {
        "core_start_s": 0.0,
        "core_end_s": FRAMES * HOP_S,
        "candidate": {"time_s": 2.0, "score": 0.5, "prominence": 0.2},
        "speech_probabilities": speech,
        "left_context_s": 1.6,
        "right_context_s": 1.6,
        "gap_context_s": 0.3,
        "left_bins": 8,
        "gap_bins": 4,
        "right_bins": 8,
        "ptm_dim": 128,
        "extra_context_scales": (),
        "ptm_projection_mean": mean,
        "ptm_projection_components": components,
    }


def test_provider_pre_projected_matches_full_projection() -> None:
    rng = np.random.default_rng(7)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    speech = rng.uniform(size=FRAMES).astype(np.float32)
    training_provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm,
        mfcc=mfcc,
    )
    runtime_provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm[:, :CAPPED_DIM],
        mfcc=mfcc,
        ptm_projected=_pre_projected(ptm, mean, components),
        ptm_projected_digest=ptm_projection_digest(mean, components),
    )
    assert not training_provider.has_pre_projected_ptm
    assert runtime_provider.has_pre_projected_ptm
    kwargs = _split_kwargs(speech, mean, components)
    train_frames, train_scalars = training_provider.features_for_split_candidate(
        **kwargs
    )
    runtime_frames, runtime_scalars = runtime_provider.features_for_split_candidate(
        **kwargs
    )
    assert train_frames.shape == (20, PROJECTED_DIM + MFCC_DIM)
    np.testing.assert_allclose(runtime_frames, train_frames, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(runtime_scalars, train_scalars, rtol=1e-6, atol=1e-6)


def test_provider_pre_projected_dim_mismatch_raises() -> None:
    rng = np.random.default_rng(11)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm[:, :CAPPED_DIM],
        mfcc=mfcc,
        ptm_projected=np.zeros((FRAMES, PROJECTED_DIM + 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="pre-projected ptm dim mismatch"):
        provider.features_for_split_candidate(
            **_split_kwargs(np.ones(FRAMES, dtype=np.float32), mean, components)
        )


def test_provider_rejects_short_pre_projected_frames() -> None:
    rng = np.random.default_rng(13)
    ptm, mfcc = _full_frames(rng)
    with pytest.raises(ValueError, match="fewer frames"):
        FrameSequenceFeatureProvider(
            duration_s=FRAMES * HOP_S,
            frame_hop_s=HOP_S,
            ptm=ptm,
            mfcc=mfcc,
            ptm_projected=np.zeros((FRAMES - 1, PROJECTED_DIM), dtype=np.float32),
        )


def test_runtime_without_projection_ignores_pre_projected_payload() -> None:
    rng = np.random.default_rng(17)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm[:, :CAPPED_DIM],
        mfcc=mfcc,
        ptm_projected=_pre_projected(ptm, mean, components),
        config=FrameSequenceFeatureConfig(max_ptm_dims=CAPPED_DIM),
    )
    kwargs = _split_kwargs(np.ones(FRAMES, dtype=np.float32), None, None)
    kwargs["ptm_dim"] = CAPPED_DIM
    frames, _ = provider.features_for_split_candidate(**kwargs)
    assert frames.shape == (20, CAPPED_DIM + MFCC_DIM)


def test_backend_window_overlap_projection_identity() -> None:
    """Averaging per-window projections equals projecting averaged frames.

    Mirrors the accumulation in SpeechBoundaryJaBackend.segment(): float64
    sums per overlapped window divided by coverage counts.
    """

    rng = np.random.default_rng(19)
    mean, components = _projection(rng)
    total = 50
    window, stride = 20, 15
    full = rng.normal(size=(total, FULL_DIM)).astype(np.float32)
    ptm_sum = np.zeros((total, FULL_DIM), dtype=np.float64)
    projected_sum = np.zeros((total, PROJECTED_DIM), dtype=np.float64)
    counts = np.zeros((total, 1), dtype=np.float64)
    for start in range(0, total, stride):
        end = min(total, start + window)
        chunk = full[start:end]
        ptm_sum[start:end] += chunk
        projected_sum[start:end] += (
            chunk.astype(np.float64) - mean
        ) @ components.T
        counts[start:end] += 1.0
    averaged_full = ptm_sum / counts
    expected = (averaged_full - mean) @ components.T.astype(np.float64)
    np.testing.assert_allclose(projected_sum / counts, expected, atol=1e-10)


def test_backend_signature_and_config_carry_projection_digest(
    tmp_path, monkeypatch
) -> None:
    from boundary.ja.backend import SpeechBoundaryJaBackend, SpeechBoundaryJaConfig

    rng = np.random.default_rng(23)
    mean, components = _projection(rng)
    projection_path = tmp_path / "ptm_projection.npz"
    np.savez(
        projection_path,
        schema=np.asarray("speech_boundary_ja_ptm_projection_v1"),
        mean=mean,
        components=components,
    )
    backend = SpeechBoundaryJaBackend(
        config=SpeechBoundaryJaConfig(
            sequence_ptm_projection=str(projection_path)
        )
    )
    assert backend.signature()["sequence_ptm_projection_digest"] == (
        ptm_projection_digest(mean, components)
    )

    bare = SpeechBoundaryJaBackend(config=SpeechBoundaryJaConfig())
    assert bare.signature()["sequence_ptm_projection_digest"] == ""

    monkeypatch.setenv(
        "SPEECH_BOUNDARY_JA_SEQUENCE_PTM_PROJECTION", str(projection_path)
    )
    assert SpeechBoundaryJaConfig.from_env().sequence_ptm_projection == str(
        projection_path
    )


def test_pipeline_provider_accepts_ndarray_and_projected_payload() -> None:
    from asr.pipeline import _sequence_feature_provider_from_result

    rng = np.random.default_rng(29)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    digest = ptm_projection_digest(mean, components)
    payload = {
        "schema": "speech_boundary_ja_sequence_feature_frames_v1",
        "frame_hop_s": HOP_S,
        "ptm": ptm[:, :CAPPED_DIM],
        "mfcc": mfcc,
        "ptm_dim": CAPPED_DIM,
        "mfcc_dim": MFCC_DIM,
        "ptm_projected": _pre_projected(ptm, mean, components),
        "ptm_projected_dim": PROJECTED_DIM,
        "ptm_projection_digest": digest,
    }
    provider = _sequence_feature_provider_from_result(
        payload, duration_s=FRAMES * HOP_S
    )
    assert provider is not None
    assert provider.has_pre_projected_ptm
    assert provider.ptm_projected_digest == digest

    legacy = _sequence_feature_provider_from_result(
        {
            "schema": "speech_boundary_ja_sequence_feature_frames_v1",
            "frame_hop_s": HOP_S,
            "ptm": ptm[:, :CAPPED_DIM].tolist(),
            "mfcc": mfcc.tolist(),
        },
        duration_s=FRAMES * HOP_S,
    )
    assert legacy is not None
    assert not legacy.has_pre_projected_ptm


def _verifier_with_projection(
    mean: np.ndarray, components: np.ndarray, digest: str
) -> SimpleNamespace:
    return SimpleNamespace(
        feature_config={
            "left_context_s": 1.6,
            "right_context_s": 1.6,
            "gap_context_s": 0.3,
            "left_bins": 8,
            "gap_bins": 4,
            "right_bins": 8,
            "ptm_dim": 128,
            "extra_context_scales": [],
            "ptm_projection": {
                "mean": mean,
                "components": components,
                "digest": digest,
            },
        }
    )


def test_split_features_validates_projection_digest() -> None:
    from boundary.runtime_pipeline import _split_features

    rng = np.random.default_rng(31)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    digest = ptm_projection_digest(mean, components)
    speech = np.ones(FRAMES, dtype=np.float32)
    provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm[:, :CAPPED_DIM],
        mfcc=mfcc,
        ptm_projected=_pre_projected(ptm, mean, components),
        ptm_projected_digest=digest,
    )
    proposals = [{"time_s": 2.0, "score": 0.5}]

    frames, scalars = _split_features(
        proposals,
        core_start=0.0,
        core_end=FRAMES * HOP_S,
        speech=speech,
        provider=provider,
        verifier=_verifier_with_projection(mean, components, digest),
    )
    assert frames.shape == (1, 20, PROJECTED_DIM + MFCC_DIM)
    assert scalars.shape[0] == 1

    with pytest.raises(ValueError, match="projection mismatch"):
        _split_features(
            proposals,
            core_start=0.0,
            core_end=FRAMES * HOP_S,
            speech=speech,
            provider=provider,
            verifier=_verifier_with_projection(mean, components, "deadbeef"),
        )


def test_split_checkpoint_projection_npz_materializes_and_caches(tmp_path) -> None:
    import torch

    from asr.pipeline import _split_checkpoint_projection_npz
    from boundary.sequence_features import load_ptm_projection

    rng = np.random.default_rng(41)
    mean, components = _projection(rng)
    digest = ptm_projection_digest(mean, components)
    checkpoint = tmp_path / "split.pt"
    torch.save(
        {
            "feature_config": {
                "schema": "semantic_split_candidate_features_v1",
                "ptm_projection": {
                    "mean": mean,
                    "components": components,
                    "digest": digest,
                },
            }
        },
        checkpoint,
    )
    npz_path = _split_checkpoint_projection_npz(checkpoint)
    assert npz_path
    assert npz_path.startswith(str(tmp_path))
    loaded = load_ptm_projection(npz_path)
    assert loaded is not None
    assert loaded["digest"] == digest
    assert _split_checkpoint_projection_npz(checkpoint) == npz_path

    bare = tmp_path / "bare.pt"
    torch.save({"feature_config": {"schema": "semantic_split_candidate_features_v1"}}, bare)
    assert _split_checkpoint_projection_npz(bare) == ""

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"checkpoint")
    assert _split_checkpoint_projection_npz(corrupt) == ""

    assert _split_checkpoint_projection_npz(tmp_path / "missing.pt") == ""


def test_split_features_digest_fallback_computed_from_basis() -> None:
    from boundary.runtime_pipeline import _split_features

    rng = np.random.default_rng(37)
    mean, components = _projection(rng)
    ptm, mfcc = _full_frames(rng)
    provider = FrameSequenceFeatureProvider(
        duration_s=FRAMES * HOP_S,
        frame_hop_s=HOP_S,
        ptm=ptm[:, :CAPPED_DIM],
        mfcc=mfcc,
        ptm_projected=_pre_projected(ptm, mean, components),
        ptm_projected_digest=ptm_projection_digest(mean, components),
    )
    verifier = _verifier_with_projection(mean, components, "")
    frames, _ = _split_features(
        [{"time_s": 2.0}],
        core_start=0.0,
        core_end=FRAMES * HOP_S,
        speech=np.ones(FRAMES, dtype=np.float32),
        provider=provider,
        verifier=verifier,
    )
    assert frames.shape == (1, 20, PROJECTED_DIM + MFCC_DIM)
