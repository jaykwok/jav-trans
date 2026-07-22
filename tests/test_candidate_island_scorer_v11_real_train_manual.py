from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import wave

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.compile_candidate_island_scorer_v11_real_train_manual import (
    compile_real_train_manual,
)


CONTRACT = ACOUSTIC_BINARY_V12_CONTRACT.contract_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _wav(path: Path, frame_count: int = 8) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frame_count * 320)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_rows: list[dict] = []
    audit_rows: list[dict] = []
    verdict_rows: list[dict] = []
    audit_dir = tmp_path / "audit"
    audio_dir = audit_dir / "audio"
    audio_dir.mkdir(parents=True)
    for index in range(2):
        source_id = f"video-{index}-w00"
        video_id = f"video-{index}"
        audio = tmp_path / f"{source_id}.wav"
        _wav(audio)
        digest = _sha256(audio)
        copied = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(audio, copied)
        source_rows.append(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_source_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "audio": str(audio),
                "audio_sha256": digest,
                "sample_rate": 16000,
                "sample_count": 8 * 320,
                "duration_s": 0.16,
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "teacher_only": True,
                "training_manifest_allowed": False,
            }
        )
        audit_rows.append(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_review_item_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "duration_s": 0.16,
                "audio": f"audio/source-{index:03d}.wav",
                "audio_sha256": digest,
            }
        )
        spans = [
            {"label": "outside_candidate", "start_frame": 0, "end_frame": 2},
            {"label": "inside_candidate", "start_frame": 2, "end_frame": 6},
            {"label": "unsure", "start_frame": 6, "end_frame": 7},
            {"label": "outside_candidate", "start_frame": 7, "end_frame": 8},
        ]
        verdict_rows.append(
            {
                "schema": "candidate_island_scorer_v11_train_manual_verdict_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "partition": "train",
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "reviewed_full_source": True,
                "verdict": "complete_with_target_inside_candidate",
                "spans": spans,
            }
        )
    source_manifest = tmp_path / "train_teacher_sources.jsonl"
    audit_manifest = audit_dir / "audit_manifest.jsonl"
    verdicts = audit_dir / "manual_verdicts.jsonl"
    preaudit = tmp_path / "preaudit.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    _write(source_manifest, source_rows)
    _write(audit_manifest, audit_rows)
    _write(verdicts, verdict_rows)
    _write(preaudit, [{"source_id": row["source_id"]} for row in source_rows])
    _write(excluded, [{"source_id": "not-selected"}])
    summary = {
        "schema": "candidate_island_scorer_v11_train_teacher_review_summary_v1",
        "boundary_serialization_contract_id": CONTRACT,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "preaudit": str(preaudit),
        "preaudit_sha256": _sha256(preaudit),
        "exclude_sources": str(excluded),
        "exclude_sources_sha256": _sha256(excluded),
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "source_count": 2,
        "video_count": 2,
        "selected_source_ids": [row["source_id"] for row in source_rows],
        "teacher_output_used_as_annotation_seed": True,
        "teacher_output_used_as_truth": False,
        "human_full_source_confirmation_required": True,
        "unselected_source_label_inheritance": False,
        "training_manifest_allowed": False,
    }
    summary_path = audit_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, audit_manifest, verdicts


def test_compile_real_train_manual_binds_human_full_source_truth(tmp_path: Path) -> None:
    summary_path, audit_manifest, verdicts = _fixture(tmp_path)
    result = compile_real_train_manual(
        audit_summary=summary_path,
        audit_manifest=audit_manifest,
        manual_verdicts=verdicts,
        output_dir=tmp_path / "compiled",
    )

    assert result["source_count"] == result["video_count"] == 2
    assert result["canonical_frame_counts"] == {
        "inside_candidate": 8,
        "outside_candidate": 6,
        "unsure": 2,
    }
    assert result["teacher_output_used_as_truth"] is False
    assert result["unselected_source_label_inheritance"] is False
    assert result["training_manifest_allowed"] is True
    output = Path(result["real_train_manual_sources"])
    if not output.is_absolute():
        output = Path.cwd() / output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(row["reviewed_full_source"] is True for row in rows)
    assert all(row["unsure_training_label"] == -100 for row in rows)
    assert all(row["teacher_output_used_as_truth"] is False for row in rows)


def test_compile_real_train_manual_rejects_incomplete_or_teacher_truth(tmp_path: Path) -> None:
    summary_path, audit_manifest, verdicts = _fixture(tmp_path)
    rows = [json.loads(line) for line in verdicts.read_text(encoding="utf-8").splitlines()]
    rows[0]["reviewed_full_source"] = False
    _write(verdicts, rows)
    with pytest.raises(ValueError, match="not fully reviewed"):
        compile_real_train_manual(
            audit_summary=summary_path,
            audit_manifest=audit_manifest,
            manual_verdicts=verdicts,
            output_dir=tmp_path / "incomplete",
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["teacher_output_used_as_truth"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="human-truth contract"):
        compile_real_train_manual(
            audit_summary=summary_path,
            audit_manifest=audit_manifest,
            manual_verdicts=verdicts,
            output_dir=tmp_path / "teacher-truth",
        )
