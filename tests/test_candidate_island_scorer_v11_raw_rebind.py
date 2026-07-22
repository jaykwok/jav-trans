from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
)
from tools.boundary.ja.rebind_candidate_island_scorer_v11_raw_features import (
    rebind_raw_features,
)


CONTRACT = ACOUSTIC_BINARY_V12_CONTRACT.contract_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _feature(path: Path, value: float) -> None:
    np.savez(
        path,
        ptm=np.full((4, 2048), value, dtype=np.float32),
        mfcc=np.full((4, 40), value, dtype=np.float32),
        frame_hop_s=np.asarray([0.02], dtype=np.float32),
    )


def _canonical(source_id: str, partition: str, audio_sha: str) -> dict:
    return {
        "schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_SOURCE_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT,
        "canonical_label_schema": CANDIDATE_ISLAND_SCORER_V11_CANONICAL_LABEL_SCHEMA,
        "source_id": source_id,
        "partition": partition,
        "frame_count": 4,
        "frame_hop_s": 0.02,
        "audio_sha256": audio_sha,
        "training_manifest_allowed": True,
    }


def _raw(source_id: str, partition: str, audio_sha: str, feature: Path) -> dict:
    return {
        "schema": CANDIDATE_ISLAND_SCORER_V11_RAW_CACHE_ROW_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT,
        "source_id": source_id,
        "partition": partition,
        "feature_extractor_schema": CANDIDATE_ISLAND_SCORER_V11_FEATURE_EXTRACTOR_SCHEMA,
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "feature_path": str(feature),
        "feature_sha256": _sha256(feature),
        "frame_count": 4,
        "frame_hop_s": 0.02,
        "ptm_dim": 2048,
        "mfcc_dim": 40,
        "audio_sha256": audio_sha,
        "canonical_sources_sha256": "old-or-subset-canonical",
    }


def test_raw_rebind_reuses_exact_sources_and_finalizes_only_missing(tmp_path: Path) -> None:
    first_feature = tmp_path / "first.npz"
    second_feature = tmp_path / "second.npz"
    _feature(first_feature, 1.0)
    _feature(second_feature, 2.0)
    canonical_path = tmp_path / "canonical.jsonl"
    _write(
        canonical_path,
        [
            _canonical("first", "train", "a" * 64),
            _canonical("second", "train", "b" * 64),
        ],
    )
    prior = tmp_path / "prior.jsonl"
    _write(prior, [_raw("first", "train", "a" * 64, first_feature)])
    output = tmp_path / "rebind"

    prepared = rebind_raw_features(
        canonical_sources=canonical_path,
        prior_raw_feature_manifest=prior,
        output_dir=output,
    )
    assert prepared["reused_source_count"] == 1
    assert prepared["missing_source_ids"] == ["second"]
    assert prepared["complete"] is False
    assert prepared["training_manifest_allowed"] is False
    missing = [
        json.loads(line)
        for line in (output / "missing_canonical_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in missing] == ["second"]

    newly_extracted = tmp_path / "new.jsonl"
    _write(newly_extracted, [_raw("second", "train", "b" * 64, second_feature)])
    final = rebind_raw_features(
        canonical_sources=canonical_path,
        prior_raw_feature_manifest=prior,
        new_raw_feature_manifest=newly_extracted,
        output_dir=output,
    )
    assert final["complete"] is True
    assert final["training_manifest_allowed"] is True
    rows = [
        json.loads(line)
        for line in (output / "raw_feature_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_id"] for row in rows} == {"first", "second"}
    assert all(row["canonical_sources_sha256"] == _sha256(canonical_path) for row in rows)
    first = next(row for row in rows if row["source_id"] == "first")
    assert Path(first["feature_path"]).resolve() == first_feature.resolve()
    assert first["reused_feature_bytes_unchanged"] is True


def test_raw_rebind_rejects_feature_from_different_audio(tmp_path: Path) -> None:
    feature = tmp_path / "feature.npz"
    _feature(feature, 1.0)
    canonical_path = tmp_path / "canonical.jsonl"
    prior = tmp_path / "prior.jsonl"
    _write(canonical_path, [_canonical("source", "train", "a" * 64)])
    _write(prior, [_raw("source", "train", "b" * 64, feature)])

    with pytest.raises(ValueError, match="audio identity mismatch"):
        rebind_raw_features(
            canonical_sources=canonical_path,
            prior_raw_feature_manifest=prior,
            output_dir=tmp_path / "rebind",
        )
