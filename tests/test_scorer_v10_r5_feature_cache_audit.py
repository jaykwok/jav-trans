from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.features import FeatureConfig, cache_key_for_audio
from boundary.ja.model import SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA
from tools.audits.audit_scorer_v10_r5_feature_cache import audit_feature_cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    source_id = "source"
    canonical_path = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical_path,
        [
            {
                "schema": "speech_scorer_v10_canonical_source_v1",
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_label_schema": "speech_scorer_canonical_frames_v1",
                "source_id": source_id,
                "audio": str(audio_path),
                "row_role": "all_background",
                "partition": "train",
                "core_ids": [],
                "background_id": "background-source",
                "background_source_ids": [],
                "background_source_video_ids": [],
                "sample_rate": 16000,
                "sample_count": 1600,
                "duration_s": 0.1,
                "input_distribution": "full_source_windows",
                "canonical_spans": [
                    {
                        "start_sample": 0,
                        "end_sample": 1600,
                        "label": "background",
                    }
                ],
            }
        ],
    )
    audio_manifest = tmp_path / "audio-manifest.json"
    audio_manifest.write_text(
        json.dumps(
            [
                {
                    "audio_id": source_id,
                    "audio": str(audio_path),
                    "partition": "train",
                    "row_role": "all_background",
                }
            ]
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [{"audio_id": source_id, "speech_frames": [0] * 5}])
    r5_summary = tmp_path / "r5-summary.json"
    r5_summary.write_text(
        json.dumps(
            {
                "schema": "speech_scorer_v10_corrected_canonical_r5_summary_v1",
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "canonical_sources": str(canonical_path),
                "canonical_sources_sha256": _sha256(canonical_path),
                "audio_manifest": str(audio_manifest),
                "audio_manifest_sha256": _sha256(audio_manifest),
                "feature_cache_labels": str(labels),
                "feature_cache_labels_sha256": _sha256(labels),
                "audio_bytes_changed": False,
                "source_identity_changed": False,
                "partition_identity_changed": False,
                "replacement_manual_review_complete": True,
                "replacement_resolution_pass": True,
                "feature_cache_labels_ready": True,
                "training_manifest_ready": False,
                "checkpoint_promotion_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    config = FeatureConfig(
        ptm=QWEN_ASR_17B_REPO_ID,
        frame_hop_s=0.02,
        window_s=30.0,
        overlap_s=5.0,
        n_mfcc=40,
        n_fft=400,
        feature_dim=None,
        device="cuda",
        dtype="bfloat16",
        revision=None,
        model_path="models/test-1.7b",
        download=False,
        attention="sdpa",
        language="Japanese",
    )
    feature_path = tmp_path / "source.npz"
    np.savez(
        feature_path,
        ptm=np.zeros((5, 2048), dtype=np.float32),
        mfcc=np.zeros((5, 40), dtype=np.float32),
        duration_s=np.asarray([0.1], dtype=np.float32),
        sample_rate=np.asarray([16000], dtype=np.int32),
    )
    feature_manifest = tmp_path / "feature-manifest.jsonl"
    _write_jsonl(
        feature_manifest,
        [
            {
                "audio_id": source_id,
                "audio_path": str(audio_path),
                "cache_key": cache_key_for_audio(
                    audio_path=audio_path, config=config
                ),
                "duration_s": 0.1,
                "feature_coverage_ratio": 1.0,
                "feature_overlap_s": 5.0,
                "feature_path": str(feature_path),
                "feature_window_count": 1,
                "feature_window_s": 30.0,
                "frame_count": 5,
                "frame_hop_s": 0.02,
                "label_index": 0,
                "label_quality": "negative",
                "mfcc_dim": 40,
                "ptm": QWEN_ASR_17B_REPO_ID,
                "ptm_dim": 2048,
                "source": "stale-r2",
                "speech_frame_count": 1,
            }
        ],
    )
    old_audio_manifest = tmp_path / "old-audio-manifest.json"
    old_audio_manifest.write_text(audio_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    old_labels = tmp_path / "old-labels.jsonl"
    old_labels.write_text(labels.read_text(encoding="utf-8"), encoding="utf-8")
    feature_summary = tmp_path / "feature-summary.json"
    feature_summary.write_text(
        json.dumps(
            {
                "records": 1,
                "examples": 1,
                "cached": 1,
                "errors": 0,
                "skipped": 0,
                "compressed": False,
                "ptm_window_batch_size": 1,
                "feature_window_s": 30.0,
                "feature_overlap_s": 5.0,
                "config": asdict(config),
                "feature_manifest": str(feature_manifest),
                "source_manifest": str(old_audio_manifest),
                "labels": str(old_labels),
            }
        ),
        encoding="utf-8",
    )
    return r5_summary, feature_summary, feature_manifest, canonical_path, audio_path


def test_r5_cache_audit_removes_stale_labels_and_binds_content(tmp_path: Path) -> None:
    r5_summary, feature_summary, feature_manifest, _canonical_path, audio_path = (
        _fixture(tmp_path)
    )
    output = tmp_path / "audit"
    summary = audit_feature_cache(
        r5_summary_path=r5_summary,
        feature_summary_path=feature_summary,
        feature_manifest_path=feature_manifest,
        output_dir=output,
    )

    signed = json.loads(
        (output / "signed_feature_manifest.jsonl").read_text(encoding="utf-8")
    )
    assert signed["schema"] == SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA
    assert "label_quality" not in signed
    assert "speech_frame_count" not in signed
    assert summary["training_manifest_allowed"] is True
    assert summary["extraction_time_content_sha256_available"] is False

    sf.write(audio_path, np.ones(1600, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="cache key no longer matches"):
        audit_feature_cache(
            r5_summary_path=r5_summary,
            feature_summary_path=feature_summary,
            feature_manifest_path=feature_manifest,
            output_dir=tmp_path / "rejected",
        )
