from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boundary.ja.vocal_envelope_v12 import (
    VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
)
from tools.audits.record_vocal_envelope_scorer_v12_approval import record_approval
from tools.boundary.ja.vocal_envelope_scorer_v12_calibration import (
    CALIBRATION_TEACHER_CONTRACT,
    evidence_span_signature,
)


CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_teacher_audit_summary_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build_artifacts(tmp_path: Path) -> dict[str, Path | dict]:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"fixed-audio-identity")
    audio_sha = _sha256(source_audio)
    audit_dir = tmp_path / "audit"
    audit_audio = audit_dir / "audio" / "source-000.wav"
    audit_audio.parent.mkdir(parents=True)
    audit_audio.write_bytes(source_audio.read_bytes())

    source = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_SOURCE_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "source_id": "source-1",
        "video_id": "video-1",
        "partition": "train",
        "core_ids": ["core-1"],
        "audio": str(source_audio),
        "audio_sha256": audio_sha,
        "duration_s": 1.0,
        "frame_count": 50,
        "frame_hop_s": 0.02,
        "sample_rate": 16_000,
        "sample_count": 16_000,
    }
    source_manifest = tmp_path / "sources.jsonl"
    _write(source_manifest, [source])
    source_sha = _sha256(source_manifest)

    vocal_span = {
        "label": "vocal_candidate",
        "start_frame": 0,
        "end_frame": 50,
        "start_s": 0.0,
        "end_s": 1.0,
        "reason": "voiced speech",
    }
    preaudit_row = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        **CALIBRATION_TEACHER_CONTRACT,
        "source_id": source["source_id"],
        "video_id": source["video_id"],
        "partition": source["partition"],
        "core_ids": source["core_ids"],
        "audio": str(source_audio),
        "audio_sha256": audio_sha,
        "duration_s": source["duration_s"],
        "frame_count": source["frame_count"],
        "frame_hop_s": source["frame_hop_s"],
        "sample_rate": source["sample_rate"],
        "sample_count": source["sample_count"],
        "source_manifest_sha256": source_sha,
        "vocal_spans": [vocal_span],
        "non_vocal_spans": [],
        "unsure_spans": [],
        "conflict_spans": [],
    }
    preaudit = tmp_path / "preaudit.jsonl"
    _write(preaudit, [preaudit_row])
    preaudit_sha = _sha256(preaudit)
    signature = evidence_span_signature(
        preaudit_row, frame_count=50, source_id="source-1"
    )

    audit_row = {
        "schema": VOCAL_ENVELOPE_SCORER_V12_AUDIT_ITEM_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        "source_id": source["source_id"],
        "video_id": source["video_id"],
        "partition": source["partition"],
        "audio": "audio/source-000.wav",
        "audio_sha256": audio_sha,
        "duration_s": source["duration_s"],
        "frame_count": source["frame_count"],
        "source_manifest_sha256": source_sha,
        "preaudit_sha256": preaudit_sha,
        "evidence_span_signature": [list(item) for item in signature],
        "vocal_spans": [vocal_span],
        "non_vocal_spans": [],
        "unsure_spans": [],
    }
    audit_manifest = audit_dir / "audit_manifest.jsonl"
    _write(audit_manifest, [audit_row])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "task_semantics": VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha,
        "preaudit": str(preaudit),
        "preaudit_sha256": preaudit_sha,
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "source_count": 1,
        "selected_partitions": [],
        "skipped_calibration_source_count": 0,
        "skipped_calibration_source_ids": [],
        "manual_verdict_schema": VOCAL_ENVELOPE_SCORER_V12_MANUAL_VERDICT_SCHEMA,
        "manual_gate_status": "pending",
        "training_manifest_allowed": False,
    }
    summary_path = audit_dir / "summary.json"
    _write_summary(summary_path, summary)
    return {
        "source_manifest": source_manifest,
        "preaudit": preaudit,
        "audit_manifest": audit_manifest,
        "audit_summary": summary_path,
        "audit_audio": audit_audio,
        "output": audit_dir / "manual_verdicts.jsonl",
        "source": source,
        "preaudit_row": preaudit_row,
        "audit_row": audit_row,
        "summary": summary,
    }


def _record(artifacts: dict[str, Path | dict]) -> dict:
    return record_approval(
        audit_manifest=artifacts["audit_manifest"],  # type: ignore[arg-type]
        source_manifest=artifacts["source_manifest"],  # type: ignore[arg-type]
        preaudit=artifacts["preaudit"],  # type: ignore[arg-type]
        output=artifacts["output"],  # type: ignore[arg-type]
        note="reviewed in app",
        approved_by="user",
    )


def test_records_exact_artifact_chain_bound_blanket_approval(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)

    summary = _record(artifacts)

    output = artifacts["output"]
    assert isinstance(output, Path)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["approved_count"] == 1
    assert row["approved"] is True
    assert row["training_manifest_allowed"] is True
    assert row["task_semantics"] == VOCAL_ENVELOPE_SCORER_V12_TASK_SEMANTICS
    assert row["vocal_purity"] == "definite_vocal_excludes_separable_nonvoice"
    assert row["approval_provenance"] == "explicit_user_blanket_approval"
    assert row["source_manifest_sha256"] == _sha256(
        artifacts["source_manifest"]  # type: ignore[arg-type]
    )
    assert row["preaudit_sha256"] == _sha256(
        artifacts["preaudit"]  # type: ignore[arg-type]
    )
    assert row["audit_manifest_sha256"] == _sha256(
        artifacts["audit_manifest"]  # type: ignore[arg-type]
    )
    assert row["audit_summary_sha256"] == _sha256(
        artifacts["audit_summary"]  # type: ignore[arg-type]
    )
    assert row["evidence_span_signature"] == [["vocal_candidate", 0, 50]]


def test_refuses_audit_manifest_changed_after_summary(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    audit_row = dict(artifacts["audit_row"])  # type: ignore[arg-type]
    audit_row["unexpected"] = True
    _write(artifacts["audit_manifest"], [audit_row])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="audit summary audit_manifest_sha256 mismatch"):
        _record(artifacts)


def test_refuses_resigned_audit_with_preaudit_span_content_drift(
    tmp_path: Path,
) -> None:
    artifacts = _build_artifacts(tmp_path)
    audit_row = dict(artifacts["audit_row"])  # type: ignore[arg-type]
    audit_row["vocal_spans"] = [dict(audit_row["vocal_spans"][0])]
    audit_row["vocal_spans"][0]["reason"] = "manually replaced"
    audit_manifest = artifacts["audit_manifest"]
    assert isinstance(audit_manifest, Path)
    _write(audit_manifest, [audit_row])
    summary = dict(artifacts["summary"])  # type: ignore[arg-type]
    summary["audit_manifest_sha256"] = _sha256(audit_manifest)
    _write_summary(artifacts["audit_summary"], summary)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="audit/preaudit vocal_spans mismatch"):
        _record(artifacts)


def test_refuses_non_exact_preaudit_source_ids_even_if_hashes_are_rebound(
    tmp_path: Path,
) -> None:
    artifacts = _build_artifacts(tmp_path)
    preaudit_row = dict(artifacts["preaudit_row"])  # type: ignore[arg-type]
    rogue = dict(preaudit_row)
    rogue["source_id"] = "source-2"
    preaudit = artifacts["preaudit"]
    assert isinstance(preaudit, Path)
    _write(preaudit, [preaudit_row, rogue])

    with pytest.raises(ValueError, match="exact same source IDs"):
        _record(artifacts)


def test_refuses_old_preaudit_schema_after_all_hashes_are_rebound(
    tmp_path: Path,
) -> None:
    artifacts = _build_artifacts(tmp_path)
    preaudit_row = dict(artifacts["preaudit_row"])  # type: ignore[arg-type]
    preaudit_row["schema"] = "vocal_envelope_scorer_v12_single_pass_tristate_preaudit_v2"
    preaudit = artifacts["preaudit"]
    audit_manifest = artifacts["audit_manifest"]
    assert isinstance(preaudit, Path)
    assert isinstance(audit_manifest, Path)
    _write(preaudit, [preaudit_row])
    preaudit_sha = _sha256(preaudit)
    audit_row = dict(artifacts["audit_row"])  # type: ignore[arg-type]
    audit_row["preaudit_sha256"] = preaudit_sha
    _write(audit_manifest, [audit_row])
    summary = dict(artifacts["summary"])  # type: ignore[arg-type]
    summary["preaudit_sha256"] = preaudit_sha
    summary["audit_manifest_sha256"] = _sha256(audit_manifest)
    _write_summary(artifacts["audit_summary"], summary)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="wrong preaudit schema"):
        _record(artifacts)


def test_refuses_audit_audio_replacement(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    audit_audio = artifacts["audit_audio"]
    assert isinstance(audit_audio, Path)
    audit_audio.write_bytes(b"different audio")

    with pytest.raises(ValueError, match="audit source-1 audio SHA mismatch"):
        _record(artifacts)


def test_requires_sibling_or_explicit_audit_summary(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    summary_path = artifacts["audit_summary"]
    assert isinstance(summary_path, Path)
    summary_path.unlink()

    with pytest.raises(FileNotFoundError, match="audit summary is required"):
        _record(artifacts)
