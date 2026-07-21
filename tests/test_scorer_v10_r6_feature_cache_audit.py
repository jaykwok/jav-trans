from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.features import FeatureConfig, cache_key_for_audio
from boundary.ja.model import (
    SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
    SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
)
from tools.audits.audit_scorer_v10_r6_incremental_feature_cache import audit
from tools.boundary.ja.build_speech_island_scorer_v10_sparse_train_layout import (
    SUMMARY_SCHEMA as R6_SUMMARY_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _signed_row(
    source_id: str,
    audio: Path,
    feature: Path,
    *,
    frame_count: int,
    feature_config_sha256: str,
) -> dict:
    return {
        "schema": SPEECH_ISLAND_SCORER_V10_RAW_CACHE_ROW_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "feature_extractor_schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "feature_config_sha256": feature_config_sha256,
        "source_id": source_id,
        "audio_path": str(audio),
        "audio_sha256": _sha256(audio),
        "audio_size_bytes": audio.stat().st_size,
        "audio_sample_rate": 16000,
        "audio_sample_count": int(sf.info(audio).frames),
        "cache_key": "base-key",
        "feature_path": str(feature),
        "feature_sha256": _sha256(feature),
        "feature_size_bytes": feature.stat().st_size,
        "frame_count": frame_count,
        "frame_hop_s": 0.02,
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "ptm_dim": 2048,
        "mfcc_dim": 40,
        "feature_window_s": 30.0,
        "feature_overlap_s": 5.0,
        "feature_window_count": 1,
        "feature_coverage_ratio": 1.0,
        "cache_binding_sha256": "0" * 64,
    }


def test_incremental_feature_audit_replaces_only_changed_rows(tmp_path: Path) -> None:
    changed_audio = tmp_path / "changed.wav"
    unchanged_audio = tmp_path / "unchanged.wav"
    sf.write(changed_audio, np.linspace(-0.2, 0.2, 4000, dtype=np.float32), 16000)
    sf.write(unchanged_audio, np.linspace(0.2, -0.2, 4000, dtype=np.float32), 16000)
    frame_count = 13
    changed_feature = tmp_path / "changed.npz"
    unchanged_feature = tmp_path / "unchanged.npz"
    np.savez(
        changed_feature,
        ptm=np.zeros((frame_count, 2048), dtype=np.float32),
        mfcc=np.zeros((frame_count, 40), dtype=np.float32),
        duration_s=np.asarray([0.25], dtype=np.float32),
        sample_rate=np.asarray([16000], dtype=np.int32),
    )
    np.savez(
        unchanged_feature,
        ptm=np.ones((frame_count, 2048), dtype=np.float32),
        mfcc=np.ones((frame_count, 40), dtype=np.float32),
        duration_s=np.asarray([0.25], dtype=np.float32),
        sample_rate=np.asarray([16000], dtype=np.int32),
    )

    config_dict = {
        "ptm": QWEN_ASR_17B_REPO_ID,
        "frame_hop_s": 0.02,
        "window_s": 30.0,
        "overlap_s": 5.0,
        "n_mfcc": 40,
        "n_fft": 400,
        "feature_dim": None,
        "device": "cuda",
        "dtype": "bfloat16",
        "model_path": "models\\fixture",
        "download": False,
        "revision": None,
        "attention": "sdpa",
        "language": "Japanese",
    }
    config_payload = {
        "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_EXTRACTOR_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "config": config_dict,
        "feature_window_s": 30.0,
        "feature_overlap_s": 5.0,
        "ptm_window_batch_size": 1,
        "compressed": False,
    }
    config_sha = _canonical_digest(config_payload)
    base_rows = [
        _signed_row(
            "changed",
            changed_audio,
            unchanged_feature,
            frame_count=frame_count,
            feature_config_sha256=config_sha,
        ),
        _signed_row(
            "unchanged",
            unchanged_audio,
            unchanged_feature,
            frame_count=frame_count,
            feature_config_sha256=config_sha,
        ),
    ]
    base_manifest = tmp_path / "base.jsonl"
    base_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
    )
    base_gate = tmp_path / "base-gate.json"
    _write_json(
        base_gate,
        {
            "schema": SPEECH_ISLAND_SCORER_V10_FEATURE_CACHE_GATE_SCHEMA,
            "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
            "signed_feature_manifest": str(base_manifest),
            "signed_feature_manifest_sha256": _sha256(base_manifest),
            "feature_config": config_payload,
            "feature_config_sha256": config_sha,
        },
    )

    sources = [
        {
            "schema": "speech_scorer_v10_canonical_source_v1",
            "source_id": "changed",
            "audio": str(changed_audio),
            "partition": "train",
            "sample_rate": 16000,
            "sample_count": 4000,
        },
        {
            "schema": "speech_scorer_v10_canonical_source_v1",
            "source_id": "unchanged",
            "audio": str(unchanged_audio),
            "partition": "val",
            "sample_rate": 16000,
            "sample_count": 4000,
        },
    ]
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(
        "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
    )
    audio_manifest = tmp_path / "audio.json"
    labels = tmp_path / "labels.jsonl"
    _write_json(audio_manifest, [{"audio_id": row["source_id"]} for row in sources])
    labels.write_text("{}\n{}\n", encoding="utf-8")
    r6_summary = tmp_path / "r6.json"
    _write_json(
        r6_summary,
        {
            "schema": R6_SUMMARY_SCHEMA,
            "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
            "audio_bytes_changed": True,
            "changed_partition": "train",
            "heldout_audio_identity_changed": False,
            "canonical_sources": str(canonical),
            "canonical_sources_sha256": _sha256(canonical),
            "audio_manifest": str(audio_manifest),
            "audio_manifest_sha256": _sha256(audio_manifest),
            "feature_cache_labels": str(labels),
            "feature_cache_labels_sha256": _sha256(labels),
            "selected_source_ids": ["changed"],
            "selected_source_count": 1,
        },
    )

    config = FeatureConfig(**config_dict)
    raw_manifest = tmp_path / "changed-raw.jsonl"
    raw_manifest.write_text(
        json.dumps(
            {
                "audio_id": "changed",
                "audio_path": str(changed_audio),
                "cache_key": cache_key_for_audio(audio_path=changed_audio, config=config),
                "feature_path": str(changed_feature),
                "frame_count": frame_count,
                "ptm_dim": 2048,
                "mfcc_dim": 40,
                "ptm": QWEN_ASR_17B_REPO_ID,
                "frame_hop_s": 0.02,
                "feature_window_s": 30.0,
                "feature_overlap_s": 5.0,
                "feature_window_count": 1,
                "feature_coverage_ratio": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    changed_summary = tmp_path / "changed-summary.json"
    _write_json(
        changed_summary,
        {
            "records": 1,
            "examples": 1,
            "cached": 1,
            "errors": 0,
            "skipped": 0,
            "feature_manifest": str(raw_manifest),
            "config": config_dict,
            "feature_window_s": 30.0,
            "feature_overlap_s": 5.0,
            "ptm_window_batch_size": 1,
            "compressed": False,
        },
    )

    output = tmp_path / "output"
    result = audit(
        r6_summary_path=r6_summary,
        base_feature_gate_path=base_gate,
        changed_feature_summary_path=changed_summary,
        changed_feature_manifest_path=raw_manifest,
        output_dir=output,
    )
    assert result["changed_source_count"] == 1
    assert result["reused_signed_source_count"] == 1
    assert result["training_manifest_allowed"] is True
    merged = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "signed_feature_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert merged["unchanged"] == base_rows[1]
    assert merged["changed"]["feature_sha256"] == _sha256(changed_feature)
    assert Path(merged["changed"]["feature_path"]).resolve() == changed_feature.resolve()
