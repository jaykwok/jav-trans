from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from boundary.ja.model import (
    CANDIDATE_ISLAND_SCORER_V11_RAW_PREEXTRACT_SOURCE_SCHEMA,
)
from tools.boundary.ja.prepare_candidate_island_scorer_v11_raw_preextract import (
    AUDIT_ITEM_SCHEMA,
    AUDIT_SUMMARY_SCHEMA,
    TEACHER_SOURCE_SCHEMA,
    prepare_raw_preextract_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> Path:
    source_audio = tmp_path / "source.wav"
    audit_audio = tmp_path / "audit" / "audio" / "source-000.wav"
    audit_audio.parent.mkdir(parents=True)
    payload = b"RIFF-safe-identical-audio"
    source_audio.write_bytes(payload)
    audit_audio.write_bytes(payload)
    audio_sha = _sha256(source_audio)
    source_id = "source-a-w00"
    source_manifest = tmp_path / "teacher.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "schema": TEACHER_SOURCE_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": "source-a",
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "duration_s": 0.2,
                "sample_rate": 16000,
                "sample_count": 3200,
                "audio": str(source_audio),
                "audio_sha256": audio_sha,
            }
        ],
    )
    audit_manifest = tmp_path / "audit" / "audit_manifest.jsonl"
    _write_jsonl(
        audit_manifest,
        [
            {
                "schema": AUDIT_ITEM_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "video_id": "source-a",
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "duration_s": 0.2,
                "audio": "audio/source-000.wav",
                "audio_sha256": audio_sha,
            }
        ],
    )
    summary_path = tmp_path / "audit" / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": AUDIT_SUMMARY_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "training_manifest_allowed": False,
                "human_full_source_confirmation_required": True,
                "manual_gate_status": "pending",
                "audit_manifest": str(audit_manifest),
                "audit_manifest_sha256": _sha256(audit_manifest),
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
                "selected_source_ids": [source_id],
                "source_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_v11_raw_preextract_manifest_is_label_free_and_not_training_ready(
    tmp_path: Path,
) -> None:
    summary_path = _fixture(tmp_path)
    result = prepare_raw_preextract_manifest(
        audit_summary_path=summary_path, output_dir=tmp_path / "output"
    )
    rows = [
        json.loads(line)
        for line in Path(result["raw_preextract_sources"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["schema"] == CANDIDATE_ISLAND_SCORER_V11_RAW_PREEXTRACT_SOURCE_SCHEMA
    assert rows[0]["labels_available"] is False
    assert rows[0]["training_manifest_allowed"] is False
    assert result["feature_extraction_allowed"] is True
    assert result["training_manifest_allowed"] is False


def test_v11_raw_preextract_rejects_a_summary_that_claims_training_ready(
    tmp_path: Path,
) -> None:
    summary_path = _fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_manifest_allowed"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="non-training audit summary"):
        prepare_raw_preextract_manifest(
            audit_summary_path=summary_path, output_dir=tmp_path / "output"
        )
