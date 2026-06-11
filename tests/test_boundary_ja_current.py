from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_06B_REPO_ID, QWEN_ASR_17B_REPO_ID
from boundary.ja import (
    FeatureConfig,
    TeacherSegment,
    TrainConfig,
    align_feature_frames,
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
    sample_hf_audio_16k_mono,
    segments_to_frame_labels,
    stable_hf_audio_id,
    train_tiny_frame_classifier,
    write_feature_cache,
    write_jsonl,
)
from boundary.ja.backend import DEFAULT_MODEL_PATH, DEFAULT_OPERATING_POINT, SpeechBoundaryJaConfig
from boundary.ja.manifest import TrainingExample
from boundary.ja.model import TinyFrameClassifier


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
            "cut_drop_zones": [{"start": 0.3, "end": 0.7}],
            "cut_point_segments": [{"time_s": 0.5}],
        },
    )

    starts, ends, cut_drops, cut_points = endpoint_targets_from_record(
        record,
        frame_count=10,
        boundary_radius_frames=0,
        cut_min_gap_s=0.5,
        cut_boundary_radius_frames=0,
    )

    assert record.boundary_metadata["cut_drop_zones"] == [{"start": 0.3, "end": 0.7}]
    assert starts.tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert ends.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert cut_drops.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert cut_points.tolist()[5] == 1


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

    assert FeatureConfig().ptm == QWEN_ASR_06B_REPO_ID
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
    assert DEFAULT_MODEL_PATH == "models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame"
    assert DEFAULT_OPERATING_POINT == "qwen-feature-energy-bootstrap-v1"
    assert cfg.ptm == QWEN_ASR_06B_REPO_ID
    assert not hasattr(cfg, "checkpoint")
    assert not hasattr(cfg, "imitation_checkpoint")
