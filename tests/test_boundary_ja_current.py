from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_06B_REPO_ID, QWEN_ASR_17B_REPO_ID, QWEN_ASR_REPO_ID, qwen_asr_repo_tag
from boundary.ja import (
    FeatureConfig,
    FeatureScorerTrainConfig,
    MAMBA2_FRAME_SCORER_MODEL_ARCH,
    MAMBA2_FRAME_SCORER_SCHEMA,
    TeacherSegment,
    TrainConfig,
    align_feature_frames,
    build_feature_frame_scorer_model,
    build_feature_frame_scorer_checkpoint,
    build_supervised_record,
    build_training_windows,
    count_trainable_parameters,
    endpoint_targets_from_record,
    frame_classification_counts,
    frame_count,
    is_low_frame_rate_ptm,
    is_qwen3_asr_ptm,
    metrics_from_frame_counts,
    load_cached_feature,
    qwen3_asr_audio_output_lengths,
    qwen3_asr_repo_id,
    resize_binary_frames,
    score_feature_frame_boundary_probabilities,
    sample_hf_audio_16k_mono,
    segments_to_frame_labels,
    stable_hf_audio_id,
    train_feature_frame_scorer,
    train_tiny_frame_classifier,
    write_feature_cache,
    write_jsonl,
)
from boundary.ja.backend import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OPERATING_POINT,
    SpeechBoundaryJaBackend,
    SpeechBoundaryJaConfig,
    decode_frame_boundary_segments,
    _hysteresis_frames,
    _validate_scorer_checkpoint_repo,
)
from boundary.ja.manifest import TrainingExample
from boundary.ja.model import TinyFrameClassifier, load_feature_frame_scorer_checkpoint
from boundary.ja.train import _feature_training_arrays
from tools.boundary.ja.build_feature_cache import (
    _combine_workflow_window_features,
    _workflow_window_starts,
)

def _require_mamba2():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mamba2Model"):
        pytest.skip("transformers.Mamba2Model is unavailable")


def _mamba2_scorer_config(*, ptm_dim: int, mfcc_dim: int) -> dict:
    return {
        "ptm_dim": ptm_dim,
        "mfcc_dim": mfcc_dim,
        "input_dim": ptm_dim + mfcc_dim,
        "hidden_size": 4,
        "num_layers": 1,
        "state_size": 8,
        "num_heads": 2,
        "n_groups": 1,
        "chunk_size": 4,
        "bidirectional": True,
        "output_dim": 2,
        "model_arch": MAMBA2_FRAME_SCORER_MODEL_ARCH,
        "split_adapter_kernel_size": 3,
    }


def _build_mamba2_scorer(*, ptm_dim: int, mfcc_dim: int):
    _require_mamba2()
    model_config = _mamba2_scorer_config(ptm_dim=ptm_dim, mfcc_dim=mfcc_dim)
    return (
        build_feature_frame_scorer_model(
            schema=MAMBA2_FRAME_SCORER_SCHEMA,
            model_config=model_config,
        ),
        model_config,
    )



def _registered_placeholder(tmp_path: Path, repo_id: str) -> Path:
    path = tmp_path / f"speech_boundary_ja_frame_boundary_scorer_v7.{qwen_asr_repo_tag(repo_id)}.pt"
    path.write_bytes(b"placeholder")
    return path


def test_hf_audio_sample_normalizes_to_16k_mono():
    stereo_8k = np.column_stack(
        [
            np.ones(8000, dtype=np.float32) * 0.2,
            np.ones(8000, dtype=np.float32) * 0.4,
        ]
    )

    audio, sample_rate = sample_hf_audio_16k_mono(
        {
            "ogg": {"array": stereo_8k, "sampling_rate": 8000},
        }
    )

    assert sample_rate == 16000
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert np.isclose(float(np.mean(audio[100:-100])), 0.3, atol=1e-3)


def test_hf_audio_sample_decodes_audio_bytes_mapping():
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8000, dtype=np.float32), 8000, format="WAV")

    audio, sample_rate = sample_hf_audio_16k_mono(
        {
            "audio": {"bytes": buffer.getvalue(), "path": "sample.wav"},
        }
    )

    assert sample_rate == 16000
    assert audio.shape == (16000,)


def test_label_record_and_endpoint_targets_keep_boundary_metadata():
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="unit",
            duration_s=1.0,
            speech_segments=[TeacherSegment(0.0, 0.3), TeacherSegment(0.7, 1.0)],
            frame_hop_s=0.1,
        ),
        boundary_metadata={
            "cut_point_segments": [{"time_s": 0.5}],
        },
    )

    starts, ends, split_points = endpoint_targets_from_record(
        record,
        frame_count=10,
        boundary_radius_frames=0,
        split_boundary_radius_frames=0,
    )

    assert starts.tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert ends.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert split_points.tolist()[5] == 1


def test_endpoint_targets_mark_supervised_gap_boundaries_as_split_points():
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="unit",
            duration_s=1.0,
            speech_segments=[TeacherSegment(0.0, 0.2), TeacherSegment(0.8, 1.0)],
            frame_hop_s=0.1,
        ),
        boundary_metadata={
            "cut_point_segments": [{"time_s": 0.5}],
            "disable_implicit_gap_drop": True,
        },
    )

    _starts, _ends, split_points = endpoint_targets_from_record(
        record,
        frame_count=10,
        boundary_radius_frames=0,
        split_boundary_radius_frames=0,
    )

    assert split_points.tolist()[5] == 1


def test_feature_training_arrays_support_head_specific_frame_weights(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    cached = write_feature_cache(
        output_dir=tmp_path / "features",
        audio_id="clip",
        source="unit",
        audio_path=audio_path,
        config=FeatureConfig(feature_dim=2),
        bundle={
            "ptm": np.ones((10, 4), dtype=np.float32),
            "mfcc": np.ones((10, 2), dtype=np.float32),
            "duration_s": 1.0,
            "sample_rate": 16000,
        },
        compressed=False,
    )
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="unit",
            duration_s=1.0,
            speech_segments=[TeacherSegment(0.0, 0.3), TeacherSegment(0.7, 1.0)],
            frame_hop_s=0.1,
        ),
        frame_weights=[1.0] * 10,
        boundary_metadata={
            "head_frame_weights": {
                "speech": [1.0] * 10,
                "split_boundary": [0.5] * 10,
            },
        },
    )

    _features, labels, weights = _feature_training_arrays(
        row={"label_index": 0, "feature_path": cached.feature_path},
        records=[record],
    )

    assert labels.shape == (10, 2)
    assert weights[:, 0].tolist() == [1.0] * 10
    assert weights[:, 1].tolist() == [0.5] * 10


def test_feature_training_arrays_ignore_old_head_weight_aliases(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    cached = write_feature_cache(
        output_dir=tmp_path / "features",
        audio_id="clip",
        source="unit",
        audio_path=audio_path,
        config=FeatureConfig(feature_dim=2),
        bundle={
            "ptm": np.ones((5, 4), dtype=np.float32),
            "mfcc": np.ones((5, 2), dtype=np.float32),
            "duration_s": 0.5,
            "sample_rate": 16000,
        },
        compressed=False,
    )
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="unit",
            duration_s=0.5,
            speech_segments=[TeacherSegment(0.0, 0.5)],
            frame_hop_s=0.1,
        ),
        frame_weights=[1.0] * 5,
        boundary_metadata={
            "head_frame_weights": {
                "speech_prob": [0.25] * 5,
                "split": [0.25] * 5,
                "split_boundary_prob": [0.25] * 5,
            },
        },
    )

    _features, _labels, weights = _feature_training_arrays(
        row={"label_index": 0, "feature_path": cached.feature_path},
        records=[record],
    )

    assert weights[:, 0].tolist() == [1.0] * 5
    assert weights[:, 1].tolist() == [1.0] * 5


def test_feature_training_arrays_reject_bad_head_weight_length(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    cached = write_feature_cache(
        output_dir=tmp_path / "features",
        audio_id="clip",
        source="unit",
        audio_path=audio_path,
        config=FeatureConfig(feature_dim=2),
        bundle={
            "ptm": np.ones((5, 4), dtype=np.float32),
            "mfcc": np.ones((5, 2), dtype=np.float32),
            "duration_s": 0.5,
            "sample_rate": 16000,
        },
        compressed=False,
    )
    record = replace(
        build_supervised_record(
            audio_id="clip",
            source="unit",
            duration_s=0.5,
            speech_segments=[TeacherSegment(0.0, 0.5)],
            frame_hop_s=0.1,
        ),
        frame_weights=[1.0] * 5,
        boundary_metadata={"head_frame_weights": {"split_boundary": [1.0, 1.0]}},
    )

    with pytest.raises(ValueError, match="head_frame_weights.split_boundary length"):
        _feature_training_arrays(
            row={"label_index": 0, "feature_path": cached.feature_path},
            records=[record],
        )


def test_frame_helpers_and_metrics():
    labels = segments_to_frame_labels(
        [TeacherSegment(0.2, 0.5)],
        duration_s=1.0,
        frame_hop_s=0.1,
    )
    counts = frame_classification_counts(labels=labels, predictions=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    metrics = metrics_from_frame_counts(counts=counts, windows=1)

    assert frame_count(1.0, 0.1) == 10
    assert labels == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert counts["true_positive"] == 2
    assert counts["false_negative"] == 1
    assert metrics.recall == pytest.approx(2 / 3)


def test_feature_schema_uses_ptm_names():
    assert is_qwen3_asr_ptm(QWEN_ASR_06B_REPO_ID)
    assert is_qwen3_asr_ptm(QWEN_ASR_17B_REPO_ID)
    assert is_low_frame_rate_ptm(QWEN_ASR_06B_REPO_ID)
    assert qwen3_asr_repo_id(QWEN_ASR_06B_REPO_ID) == QWEN_ASR_06B_REPO_ID
    assert stable_hf_audio_id(dataset_name="owner/repo", split="train", index=7) == "owner_repo-train-000007"

    ptm = np.ones((3, 4), dtype=np.float32)
    mfcc = np.ones((6, 2), dtype=np.float32)
    aligned_ptm, aligned_mfcc = align_feature_frames(ptm, mfcc, resize_ptm=True)

    assert FeatureConfig().ptm == QWEN_ASR_REPO_ID
    assert aligned_ptm.shape == (6, 4)
    assert aligned_mfcc.shape == (6, 2)
    assert resize_binary_frames(np.asarray([0, 1], dtype=np.float32), 4).tolist() == [0, 0, 1, 1]


def test_feature_cache_can_store_truncated_ptm_dim(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    cached = write_feature_cache(
        output_dir=tmp_path / "features",
        audio_id="clip",
        source="unit",
        audio_path=audio_path,
        config=FeatureConfig(feature_dim=2),
        bundle={
            "ptm": np.arange(12, dtype=np.float32).reshape(3, 4),
            "mfcc": np.arange(6, dtype=np.float32).reshape(3, 2),
            "duration_s": 0.3,
            "sample_rate": 16000,
        },
        compressed=False,
    )

    ptm, mfcc = load_cached_feature(Path(cached.feature_path))

    assert cached.ptm_dim == 2
    assert ptm.shape == (3, 2)
    assert mfcc.shape == (3, 2)
    assert ptm.tolist() == [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]


def test_feature_cache_workflow_windows_average_overlap_frames():
    config = FeatureConfig(frame_hop_s=0.02, window_s=0.06, overlap_s=0.02, ptm="unit")
    starts = _workflow_window_starts(
        sample_count=10,
        sample_rate=100,
        window_s=0.06,
        overlap_s=0.02,
    )

    bundle = _combine_workflow_window_features(
        windows=[
            {"start_sample": 0, "mfcc": np.ones((3, 1), dtype=np.float32)},
            {"start_sample": 4, "mfcc": np.ones((3, 1), dtype=np.float32) * 3.0},
            {"start_sample": 8, "mfcc": np.ones((1, 1), dtype=np.float32) * 5.0},
        ],
        ptm_features=[
            np.ones((3, 1), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32) * 3.0,
            np.ones((1, 1), dtype=np.float32) * 5.0,
        ],
        duration_s=0.10,
        sample_rate=100,
        config=config,
    )

    assert starts == [0, 4, 8]
    assert bundle["window_count"] == 3
    assert bundle["feature_coverage_ratio"] == 1.0
    assert bundle["ptm"].reshape(-1).tolist() == [1.0, 1.0, 2.0, 3.0, 4.0]
    assert bundle["mfcc"].reshape(-1).tolist() == [1.0, 1.0, 2.0, 3.0, 4.0]


def test_qwen_audio_output_lengths_formula():
    torch = pytest.importorskip("torch")
    lengths = qwen3_asr_audio_output_lengths(torch.tensor([100, 200, 250]))
    assert lengths.tolist() == [13, 26, 33]


def test_tiny_training_smoke(tmp_path):
    import soundfile as sf

    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
    record = build_supervised_record(
        audio_id="clip",
        source="unit",
        duration_s=1.0,
        speech_segments=[TeacherSegment(0.2, 0.5)],
        frame_hop_s=0.1,
    )
    example = TrainingExample(
        audio_id="clip",
        source="unit",
        label_quality="supervised",
        duration_s=record.duration_s,
        frame_hop_s=record.frame_hop_s,
        audio_path=str(audio_path),
        label_index=0,
        speech_frame_count=sum(record.speech_frames),
        frame_count=len(record.speech_frames),
    )

    windows = build_training_windows(records=[record], examples=[example], window_s=1.0)
    metrics = train_tiny_frame_classifier(
        records=[record],
        examples=[example],
        output_dir=tmp_path / "train",
        config=TrainConfig(window_s=1.0, max_steps=1, device="cpu"),
    )

    assert windows[0][0].shape == (16000,)
    assert windows[0][1].shape == (10,)
    assert Path(metrics.checkpoint).exists()
    assert count_trainable_parameters(TinyFrameClassifier()) > 0


def test_write_jsonl_and_bootstrap_backend_signature(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"text": "こんにちは"}])

    cfg = SpeechBoundaryJaConfig()

    assert path.read_text(encoding="utf-8").strip()
    assert DEFAULT_MODEL_PATH == "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame"
    assert DEFAULT_OPERATING_POINT == "qwen-mamba2-frame-boundary-scorer-v7"
    assert cfg.ptm == QWEN_ASR_REPO_ID
    assert cfg.threshold == 0.5
    assert cfg.scorer_checkpoint == ""
    assert cfg.scorer_checkpoint_repo_id == ""
    assert not hasattr(cfg, "imitation_checkpoint")


def test_backend_scorer_defaults_to_registered_06b_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = _registered_placeholder(tmp_path, QWEN_ASR_06B_REPO_ID)
    monkeypatch.setattr(
        "boundary.ja.backend.DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO",
        {QWEN_ASR_06B_REPO_ID: str(checkpoint_path)},
    )
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_PTM", QWEN_ASR_06B_REPO_ID)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", raising=False)

    cfg = SpeechBoundaryJaConfig.from_env()

    assert cfg.scorer_checkpoint == str(checkpoint_path.resolve())
    assert cfg.scorer_checkpoint_repo_id == QWEN_ASR_06B_REPO_ID


def test_backend_scorer_defaults_to_registered_17b_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = _registered_placeholder(tmp_path, QWEN_ASR_17B_REPO_ID)
    monkeypatch.setattr(
        "boundary.ja.backend.DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO",
        {QWEN_ASR_17B_REPO_ID: str(checkpoint_path)},
    )
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_PTM", QWEN_ASR_17B_REPO_ID)
    monkeypatch.delenv("SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO", raising=False)

    cfg = SpeechBoundaryJaConfig.from_env()

    assert cfg.scorer_checkpoint == str(checkpoint_path.resolve())
    assert cfg.scorer_checkpoint_repo_id == QWEN_ASR_17B_REPO_ID

def test_backend_scorer_checkpoint_env_resolves_by_ptm_repo_id(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "speech_boundary_ja_frame_boundary_scorer_v7.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_PTM", QWEN_ASR_17B_REPO_ID)
    monkeypatch.setenv(
        "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        f"{QWEN_ASR_06B_REPO_ID}=missing.pt,{QWEN_ASR_17B_REPO_ID}={checkpoint_path}",
    )

    cfg = SpeechBoundaryJaConfig.from_env()

    assert cfg.scorer_checkpoint == str(checkpoint_path.resolve())
    assert cfg.scorer_checkpoint_repo_id == QWEN_ASR_17B_REPO_ID
    sig = SpeechBoundaryJaBackend(cfg).signature()
    assert sig["scorer_checkpoint"] == str(checkpoint_path.resolve())
    assert sig["scorer_checkpoint_repo_id"] == QWEN_ASR_17B_REPO_ID

def test_backend_scorer_checkpoint_rejects_repo_metadata_mismatch():
    scorer = SimpleNamespace(metadata={"ptm_repo_id": QWEN_ASR_06B_REPO_ID})

    with pytest.raises(ValueError, match="does not match selected repo"):
        _validate_scorer_checkpoint_repo(scorer, QWEN_ASR_17B_REPO_ID)


def test_hysteresis_frames_uses_activation_and_deactivation_thresholds():
    frames = _hysteresis_frames(
        np.asarray([0.10, 0.80, 0.60, 0.45, 0.55, 0.49, 0.71], dtype=np.float32),
        on_threshold=0.70,
        off_threshold=0.50,
    )

    assert frames.tolist() == [0, 1, 1, 0, 0, 0, 1]


def test_hysteresis_frames_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="greater than or equal"):
        _hysteresis_frames(
            np.asarray([0.5], dtype=np.float32),
            on_threshold=0.40,
            off_threshold=0.50,
        )


def test_scorer_split_uses_adaptive_peaks_below_old_absolute_threshold(monkeypatch):
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_THRESHOLD", "0.99")
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_PROMINENCE", "0.99")

    frame_hop_s = 0.1
    cfg = replace(
        SpeechBoundaryJaConfig(),
        frame_hop_s=frame_hop_s,
        threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
    )

    assert cfg.split_score_quantile == pytest.approx(0.50)
    assert not hasattr(cfg, "split_threshold")
    assert not hasattr(cfg, "split_prominence")

    frame_count = 600
    split_probs = np.full(frame_count, 0.05, dtype=np.float32)
    for peak in (50, 110, 170, 230, 290, 350, 410, 470, 530):
        split_probs[peak - 1 : peak + 2] = [0.16, 0.36, 0.16]
    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(frame_count, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=60.0,
        config=cfg,
    )

    assert len(result.segments) == 10
    assert max(segment.end - segment.start for segment in result.segments) < 8.0




def test_scorer_split_peak_selection_keeps_all_effective_peaks_after_nms():
    frame_hop_s = 0.1
    cfg = SpeechBoundaryJaConfig(
        frame_hop_s=frame_hop_s,
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.2,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.0,
        split_prominence_quantile=0.0,
        video_fps=240.0,
    )
    frame_total = 428
    split_probs = np.full(frame_total, 0.02, dtype=np.float32)
    for frame, value in (
        (35, 0.45),
        (42, 0.44),
        (61, 0.21),
        (65, 0.32),
        (109, 0.39),
        (146, 0.29),
        (326, 0.38),
        (340, 0.40),
        (163, 0.28),
        (191, 0.29),
        (277, 0.19),
    ):
        split_probs[frame - 1 : frame + 2] = [0.08, value, 0.08]

    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(frame_total, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=42.8,
        config=cfg,
    )

    assert len(result.segments) == 12
    assert max(segment.end - segment.start for segment in result.segments) < 9.0
    assert any(15.0 <= segment.start <= 20.0 for segment in result.segments)


def test_scorer_decoder_exports_primary_and_weak_cut_candidates():
    cfg = SpeechBoundaryJaConfig(
        frame_hop_s=0.1,
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.1,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.75,
        split_prominence_quantile=0.0,
        video_fps=24.0,
    )
    split_probs = np.full(100, 0.01, dtype=np.float32)
    split_probs[20] = 0.90
    split_probs[50] = 0.30

    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(100, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=10.0,
        config=cfg,
    )

    assert [(segment.start, segment.end) for segment in result.segments] == [
        (0.0, pytest.approx(2.0)),
        (pytest.approx(2.0), 10.0),
    ]
    assert result.segments[1].primary_cut_candidates[0]["time_s"] == pytest.approx(2.0)
    assert result.segments[1].weak_cut_candidates[0]["time_s"] == pytest.approx(5.0)
    assert result.segments[1].weak_cut_candidates[0]["kind"] == "weak"


def test_micro_chunk_resolver_merges_middle_segment_into_left_when_left_split_weaker():
    frame_hop_s = 0.1
    cfg = SpeechBoundaryJaConfig(
        frame_hop_s=frame_hop_s,
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.1,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.0,
        split_prominence_quantile=0.0,
        video_fps=24.0,
    )
    split_probs = np.full(30, 0.01, dtype=np.float32)
    split_probs[10] = 0.30
    split_probs[15] = 0.80

    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(30, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=3.0,
        config=cfg,
    )

    assert [(segment.start, segment.end) for segment in result.segments] == [
        (0.0, pytest.approx(1.5)),
        (pytest.approx(1.5), 3.0),
    ]
    assert result.segments[0].micro_resolve_action == "merge_micro_into_left"
    assert result.segments[0].below_subtitle_min_duration is False
    assert result.segments[0].weak_cut_candidates[0]["time_s"] == pytest.approx(1.0)
    assert result.segments[0].weak_cut_candidates[0]["downgraded_from"] == "primary"


def test_micro_chunk_resolver_merges_middle_segment_into_right_when_right_split_weaker():
    frame_hop_s = 0.1
    cfg = SpeechBoundaryJaConfig(
        frame_hop_s=frame_hop_s,
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.1,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.0,
        split_prominence_quantile=0.0,
        video_fps=24.0,
    )
    split_probs = np.full(30, 0.01, dtype=np.float32)
    split_probs[10] = 0.80
    split_probs[15] = 0.30

    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(30, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=3.0,
        config=cfg,
    )

    assert [(segment.start, segment.end) for segment in result.segments] == [
        (0.0, pytest.approx(1.0)),
        (pytest.approx(1.0), 3.0),
    ]
    assert result.segments[1].micro_resolve_action == "merge_micro_into_right"
    assert result.segments[1].below_subtitle_min_duration is False


def test_micro_chunk_resolver_preserves_balanced_short_middle_segment_for_model_route():
    frame_hop_s = 0.1
    cfg = SpeechBoundaryJaConfig(
        frame_hop_s=frame_hop_s,
        threshold=0.5,
        speech_on_threshold=0.5,
        speech_off_threshold=0.5,
        frame_dilation_s=0.0,
        min_segment_s=0.0,
        split_smooth_s=0.0,
        split_nms_s=0.1,
        split_snap_s=0.0,
        min_split_segment_s=0.1,
        split_score_quantile=0.0,
        split_prominence_quantile=0.0,
        video_fps=24.0,
    )
    split_probs = np.full(30, 0.01, dtype=np.float32)
    split_probs[10] = 0.80
    split_probs[15] = 0.82

    result = decode_frame_boundary_segments(
        speech_probabilities=np.full(30, 0.9, dtype=np.float32),
        split_probabilities=split_probs,
        duration_s=3.0,
        config=cfg,
    )

    assert [(segment.start, segment.end) for segment in result.segments] == [
        (0.0, pytest.approx(1.0)),
        (pytest.approx(1.0), pytest.approx(1.5)),
        (pytest.approx(1.5), 3.0),
    ]
    middle = result.segments[1]
    assert middle.subtitle_min_duration_s == pytest.approx(20.0 / 24.0)
    assert middle.below_subtitle_min_duration is True
    assert middle.micro_chunk_candidate is True
    assert middle.micro_resolve_action == "preserve_micro_candidate"
    assert middle.left_split_score == pytest.approx(0.80)
    assert middle.right_split_score == pytest.approx(0.82)


def test_feature_frame_scorer_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    model, model_config = _build_mamba2_scorer(ptm_dim=4, mfcc_dim=2)
    checkpoint = build_feature_frame_scorer_checkpoint(
        model=model,
        model_config=model_config,
        normalization={
            "feature_mean": [0.0] * 6,
            "feature_std": [1.0] * 6,
        },
        metadata={"operating_point": "unit"},
    )
    checkpoint_path = tmp_path / "feature_scorer.pt"
    torch.save(checkpoint, checkpoint_path)

    bundle = load_feature_frame_scorer_checkpoint(checkpoint_path, device="cpu")
    speech_probs, split_probs = score_feature_frame_boundary_probabilities(
        bundle,
        ptm=np.ones((3, 4), dtype=np.float32),
        mfcc=np.ones((3, 2), dtype=np.float32),
    )

    assert bundle.signature()["schema"] == MAMBA2_FRAME_SCORER_SCHEMA
    assert bundle.signature()["model_type"] == "mamba2_frame_boundary_scorer"
    assert bundle.signature()["metadata"]["decoder"] == "topographic_split_micro_resolver_v5"
    assert bundle.input_dim == 6
    assert speech_probs.shape == (3,)
    assert split_probs.shape == (3,)
    assert np.all((0.0 <= speech_probs) & (speech_probs <= 1.0))
    assert np.all((0.0 <= split_probs) & (split_probs <= 1.0))




def test_feature_frame_scorer_rejects_removed_v1_schema(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "feature_scorer_v1.pt"
    torch.save(
        {
            "schema": "speech_boundary_ja_feature_scorer_v1",
            "model_type": "feature_frame_scorer",
            "model_config": {
                "ptm_dim": 4,
                "mfcc_dim": 2,
                "input_dim": 6,
                "hidden_size": 16,
                "dropout": 0.0,
            },
            "normalization": {"feature_mean": [0.0] * 6, "feature_std": [1.0] * 6},
            "metadata": {},
            "model_state_dict": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="speech_boundary_ja_mamba2_frame_boundary_scorer_v7"):
        load_feature_frame_scorer_checkpoint(checkpoint_path, device="cpu")


def test_feature_frame_scorer_rejects_old_decoder_contract(tmp_path):
    torch = pytest.importorskip("torch")
    model, model_config = _build_mamba2_scorer(ptm_dim=4, mfcc_dim=2)
    checkpoint = build_feature_frame_scorer_checkpoint(
        model=model,
        model_config=model_config,
        normalization={
            "feature_mean": [0.0] * 6,
            "feature_std": [1.0] * 6,
        },
        metadata={"operating_point": "unit"},
    )
    checkpoint["metadata"]["decoder"] = "topographic_split_v2"
    checkpoint_path = tmp_path / "feature_scorer_old_decoder.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="topographic_split_micro_resolver_v5"):
        load_feature_frame_scorer_checkpoint(checkpoint_path, device="cpu")


def test_feature_frame_scorer_training_from_cached_features(tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    _require_mamba2()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels_path = tmp_path / "labels.jsonl"
    records = [
        replace(
            build_supervised_record(
                audio_id="pos",
                source="unit",
                duration_s=0.4,
                speech_segments=[{"start": 0.1, "end": 0.3}],
                frame_hop_s=0.1,
            ),
            boundary_metadata={
                "dataset_schema": "unit_scorer_dataset",
                "native_example_type": "positive_speech_timeline",
                "asr_repo_id": QWEN_ASR_17B_REPO_ID,
                "source_mix": {"speech": [{"source_group": "unit"}]},
                "speech_label_dilation_s": 0.06,
                "split_boundary_radius_frames": 1,
                "negative_policy": "unit",
                "seed": 7,
            },
        ),
        replace(
            build_supervised_record(
                audio_id="neg",
                source="unit",
                duration_s=0.4,
                speech_segments=[],
                frame_hop_s=0.1,
            ),
            boundary_metadata={
                "dataset_schema": "unit_scorer_dataset",
                "native_example_type": "pure_hard_negative",
                "negative_source": {"negative_source": "synthetic"},
                "source_partition": "train",
            },
        ),
    ]
    write_jsonl(labels_path, records)
    rows = []
    for index, record in enumerate(records):
        feature_path = feature_dir / f"{record.audio_id}.npz"
        np.savez(
            feature_path,
            ptm=np.ones((4, 3), dtype=np.float32) * (index + 1),
            mfcc=np.ones((4, 2), dtype=np.float32) * (index + 2),
        )
        rows.append(
            {
                "label_index": index,
                "feature_path": str(feature_path),
                "frame_count": 4,
                "ptm_dim": 3,
                "mfcc_dim": 2,
                "ptm": QWEN_ASR_17B_REPO_ID,
            }
        )

    metrics = train_feature_frame_scorer(
        records=records,
        feature_manifest_rows=rows,
        output_dir=tmp_path / "train",
        config=FeatureScorerTrainConfig(
            max_steps=2,
            hidden_size=4,
            num_layers=1,
            state_size=8,
            num_heads=2,
            n_groups=1,
            chunk_size=4,
            device="cpu",
        ),
        labels_path=str(labels_path),
        feature_manifest_path=str(tmp_path / "feature_manifest.jsonl"),
    )

    assert Path(metrics.checkpoint).exists()
    assert metrics.schema == MAMBA2_FRAME_SCORER_SCHEMA
    assert metrics.input_dim == 5
    assert 0.0 <= metrics.speech_f1 <= 1.0
    assert 0.0 <= metrics.split_boundary_f1 <= 1.0
    bundle = load_feature_frame_scorer_checkpoint(metrics.checkpoint, device="cpu")
    assert bundle.signature()["metadata"]["trained_steps"] == 2
    assert bundle.signature()["metadata"]["ptm_repo_id"] == QWEN_ASR_17B_REPO_ID
    assert bundle.metadata["dataset_schema"] == "unit_scorer_dataset"
    assert bundle.metadata["feature_hash"]
    assert bundle.metadata["dataset"]["native_example_type_counts"] == {
        "positive_speech_timeline": 1,
        "pure_hard_negative": 1,
    }


def test_feature_frame_scorer_training_rejects_missing_ptm_repo_id(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    del torch
    records = [
        build_supervised_record(
            audio_id="missing-ptm",
            source="unit",
            duration_s=0.4,
            speech_segments=[{"start": 0.1, "end": 0.3}],
            frame_hop_s=0.1,
        )
    ]
    rows = [
        {
            "label_index": 0,
            "feature_path": str(tmp_path / "missing.npz"),
            "frame_count": 4,
            "ptm_dim": 3,
            "mfcc_dim": 2,
        }
    ]

    with pytest.raises(ValueError, match="PTM repo id"):
        train_feature_frame_scorer(
            records=records,
            feature_manifest_rows=rows,
            output_dir=tmp_path / "train",
            config=FeatureScorerTrainConfig(max_steps=1, device="cpu"),
        )





def test_backend_scorer_is_opt_in_and_keeps_segment_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model, model_config = _build_mamba2_scorer(ptm_dim=4, mfcc_dim=2)
    checkpoint_path = tmp_path / "feature_scorer.pt"
    torch.save(
        build_feature_frame_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization={"feature_mean": [0.0] * 6, "feature_std": [1.0] * 6},
            metadata={
                "operating_point": "unit",
                "trained_steps": 1,
                "ptm_repo_id": QWEN_ASR_REPO_ID,
            },
        ),
        checkpoint_path,
    )

    class FakeExtractor:
        model = None

        def extract(self, audio, *, sample_rate):
            return np.ones((5, 4), dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr("boundary.ja.backend.build_ptm_feature_extractor", lambda _config: FakeExtractor())
    monkeypatch.setattr(
        "boundary.ja.backend.extract_mfcc",
        lambda _audio, sample_rate, config: np.ones((5, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "boundary.ja.backend.load_audio_16k_mono",
        lambda _path: (np.ones(1600, dtype=np.float32), 16000),
    )

    cfg = SpeechBoundaryJaConfig(
        threshold=0.0,
        frame_dilation_s=0.0,
        frame_hop_s=0.02,
        window_s=1.0,
        overlap_s=0.0,
        min_segment_s=0.0,
        scorer_checkpoint=str(checkpoint_path),
        scorer_device="cpu",
    )
    result = SpeechBoundaryJaBackend(cfg).segment(str(tmp_path / "audio.wav"))

    assert result.segments
    assert result.segments[0].start == 0.0
    assert result.segments[0].end > result.segments[0].start
    assert result.parameters["runtime_device"]["score_model"] == "mamba2_frame_boundary_scorer_v7"
    assert result.parameters["scorer_checkpoint"]["schema"] == MAMBA2_FRAME_SCORER_SCHEMA
    assert SpeechBoundaryJaConfig().scorer_checkpoint == ""
    assert SpeechBoundaryJaBackend(SpeechBoundaryJaConfig()).signature()["scorer_checkpoint"] == ""
