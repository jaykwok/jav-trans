from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import wave

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.boundary.ja.compile_candidate_island_scorer_v11_real_train_dual_evidence import (
    compile_real_train_dual_evidence,
)


CONTRACT = ACOUSTIC_BINARY_V12_CONTRACT.contract_id
MODEL = "google/gemini-3.5-flash-lite"
PROFILE = "dual-evidence-protect-remove-v1"
PROTECT = "candidate_island_scorer_v11_protect_evidence_v2_high_recall"
REMOVE = "candidate_island_scorer_v11_remove_evidence_v1"
PROMPT = f"{PROTECT}__{REMOVE}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _runs(states: list[str], state: str, label: str) -> list[dict]:
    result: list[dict] = []
    start: int | None = None
    for index, value in enumerate([*states, "end"]):
        if value == state and start is None:
            start = index
        elif value != state and start is not None:
            result.append(
                {
                    "label": label,
                    "start_frame": start,
                    "end_frame": index,
                    "start_s": start * 0.02,
                    "end_s": index * 0.02,
                }
            )
            start = None
    return result


def _merge(protect: tuple[int, int], remove: tuple[int, int]) -> dict[str, list[dict]]:
    states: list[str] = []
    for frame in range(8):
        has_protect = protect[0] <= frame < protect[1]
        has_remove = remove[0] <= frame < remove[1]
        if has_protect and not has_remove:
            states.append("inside")
        elif has_remove and not has_protect:
            states.append("outside")
        elif has_protect and has_remove:
            states.append("conflict")
        else:
            states.append("unresolved")
    conflict = _runs(states, "conflict", "unsure")
    unresolved = _runs(states, "unresolved", "unsure")
    return {
        "islands": _runs(states, "inside", "inside_candidate"),
        "safe_outside_spans": _runs(states, "outside", "outside_candidate"),
        "conflict_spans": conflict,
        "unresolved_spans": unresolved,
        "unsure_spans": sorted(
            [*conflict, *unresolved], key=lambda span: span["start_frame"]
        ),
    }


def _teacher_summary(manifest: Path, labels: Path, *, sources: int, frames: int) -> dict:
    return {
        "schema": "candidate_island_scorer_v11_dual_evidence_summary_v1",
        "boundary_serialization_contract_id": CONTRACT,
        "prompt_profile": PROFILE,
        "prompt_version": PROMPT,
        "protect_prompt_version": PROTECT,
        "remove_prompt_version": REMOVE,
        "model": MODEL,
        "env_file_name": "gemini",
        "audio_content_mode": "input_audio_raw",
        "base_url_host": "openrouter.ai",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "labels": str(labels),
        "source_count": sources,
        "frame_count": frames,
        "failed_closed_count": 0,
        "manual_review_required": True,
        "training_manifest_allowed": False,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    audit_dir = tmp_path / "audit"
    audio_dir = audit_dir / "audio"
    audio_dir.mkdir(parents=True)
    source_rows: list[dict] = []
    audit_rows: list[dict] = []
    evidence_rows: list[dict] = []
    counts = {"inside_candidate": 0, "outside_candidate": 0, "unsure": 0}
    conflicts = 0
    for index, (protect, remove) in enumerate((((0, 4), (4, 6)), ((0, 3), (2, 5)))):
        source_id = f"video-{index}-w00"
        video_id = f"video-{index}"
        source_audio = tmp_path / f"{source_id}.wav"
        _wav(source_audio)
        audio_sha = _sha256(source_audio)
        audit_audio = audio_dir / f"source-{index:03d}.wav"
        shutil.copy2(source_audio, audit_audio)
        source_rows.append(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_source_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "video_id": video_id,
                "partition": "train",
                "audio": str(source_audio),
                "audio_sha256": audio_sha,
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
                "audio_sha256": audio_sha,
            }
        )
        merged = _merge(protect, remove)
        for field in ("islands", "safe_outside_spans", "unsure_spans"):
            for span in merged[field]:
                counts[span["label"]] += span["end_frame"] - span["start_frame"]
        conflicts += sum(
            span["end_frame"] - span["start_frame"]
            for span in merged["conflict_spans"]
        )
        evidence_rows.append(
            {
                "schema": "candidate_island_scorer_v11_dual_evidence_preaudit_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": source_id,
                "partition": "train",
                "audio": str(audit_audio),
                "audio_sha256": audio_sha,
                "frame_count": 8,
                "frame_hop_s": 0.02,
                "model": MODEL,
                "env_file_name": "gemini",
                "base_url_host": "openrouter.ai",
                "prompt_profile": PROFILE,
                "prompt_version": PROMPT,
                "protect_prompt_version": PROTECT,
                "remove_prompt_version": REMOVE,
                "merge_contract": (
                    "protect_only=inside; remove_only=outside; "
                    "overlap_or_neither=unsure"
                ),
                "unmarked_semantics": "unsure_ignore_minus_100",
                "teacher_failed_closed": False,
                "reviewed_full_source": False,
                "human_review_required": True,
                "training_manifest_allowed": False,
                "protected_evidence_spans": [
                    {
                        "label": "inside_candidate",
                        "start_frame": protect[0],
                        "end_frame": protect[1],
                        "start_s": protect[0] * 0.02,
                        "end_s": protect[1] * 0.02,
                    }
                ],
                "remove_evidence_spans": [
                    {
                        "label": "outside_candidate",
                        "start_frame": remove[0],
                        "end_frame": remove[1],
                        "start_s": remove[0] * 0.02,
                        "end_s": remove[1] * 0.02,
                    }
                ],
                **merged,
            }
        )
    source_manifest = tmp_path / "sources.jsonl"
    audit_manifest = audit_dir / "audit_manifest.jsonl"
    evidence = tmp_path / "teacher_preaudit.jsonl"
    seed = tmp_path / "seed.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    _write(source_manifest, source_rows)
    _write(audit_manifest, audit_rows)
    _write(evidence, evidence_rows)
    _write(seed, [{"source_id": row["source_id"]} for row in source_rows])
    _write(excluded, [{"source_id": "not-selected"}])
    audit_summary = audit_dir / "summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "schema": "candidate_island_scorer_v11_train_teacher_review_summary_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "audit_manifest": str(audit_manifest),
                "audit_manifest_sha256": _sha256(audit_manifest),
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
                "preaudit": str(seed),
                "preaudit_sha256": _sha256(seed),
                "exclude_sources": str(excluded),
                "exclude_sources_sha256": _sha256(excluded),
                "selected_source_ids": [row["source_id"] for row in source_rows],
                "source_count": 2,
                "video_count": 2,
                "teacher_output_used_as_truth": False,
                "unselected_source_label_inheritance": False,
                "training_manifest_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    teacher_summary = tmp_path / "teacher_summary.json"
    teacher_payload = _teacher_summary(audit_manifest, evidence, sources=2, frames=16)
    teacher_payload.update(
        {
            "source_ids": [row["source_id"] for row in source_rows],
            "inside_frames": counts["inside_candidate"],
            "outside_frames": counts["outside_candidate"],
            "unsure_frames": counts["unsure"],
            "conflict_frames": conflicts,
        }
    )
    teacher_summary.write_text(json.dumps(teacher_payload), encoding="utf-8")

    calibration_dir = tmp_path / "calibration"
    calibration_candidate = calibration_dir / "candidate.jsonl"
    calibration_manifest = calibration_dir / "manifest.jsonl"
    human_verdicts = calibration_dir / "human.jsonl"
    per_source = calibration_dir / "per_source.jsonl"
    gap_verdicts = calibration_dir / "manual_verdicts.jsonl"
    _write(calibration_candidate, [{"source_id": "heldout"}])
    _write(calibration_manifest, [{"source_id": "heldout"}])
    _write(human_verdicts, [{"source_id": "heldout"}])
    gaps = [
        {
            "gap_id": f"heldout::bridge-gap::{index:04d}-{index + 1:04d}",
            "start_frame": index,
            "end_frame": index + 1,
        }
        for index in range(23)
    ]
    _write(
        per_source,
        [
            {
                "schema": "candidate_island_dual_evidence_review_item_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": "heldout",
                "partition": "test",
                "failed_closed_count": 0,
                "bridged_background_gaps": gaps,
            }
        ],
    )
    _write(
        gap_verdicts,
        [
            {
                "schema": "candidate_island_scorer_v11_bridge_gap_manual_verdict_v3",
                "boundary_serialization_contract_id": CONTRACT,
                "gap_id": gap["gap_id"],
                "source_id": "heldout",
                "partition": "test",
                "start_frame": gap["start_frame"],
                "end_frame": gap["end_frame"],
                "content_verdict": "no_semantic_dialogue",
                "semantic_coverage_verdict": "not_applicable_no_semantic",
                "combined_verdict": "acceptable_nonsemantic_bridge",
                "verdict": "acceptable_nonsemantic_bridge",
                "complete": True,
            }
            for gap in gaps
        ],
    )
    calibration_teacher_summary = calibration_dir / "teacher_summary.json"
    calibration_teacher_summary.write_text(
        json.dumps(
            _teacher_summary(
                calibration_manifest, calibration_candidate, sources=1, frames=23
            )
        ),
        encoding="utf-8",
    )
    calibration_summary = calibration_dir / "summary.json"
    calibration_summary.write_text(
        json.dumps(
            {
                "schema": "candidate_island_dual_evidence_review_summary_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "candidate": str(calibration_candidate),
                "candidate_sha256": _sha256(calibration_candidate),
                "manifest": str(calibration_manifest),
                "manifest_sha256": _sha256(calibration_manifest),
                "human_verdicts": str(human_verdicts),
                "human_verdicts_sha256": _sha256(human_verdicts),
                "per_source": str(per_source),
                "manual_verdicts": str(gap_verdicts),
                "manual_verdict_schema": (
                    "candidate_island_scorer_v11_bridge_gap_manual_verdict_v3"
                ),
                "source_count": 1,
                "frame_count": 23,
                "protect_recall": 0.95,
                "final_outside_precision": 1.0,
                "zero_true_speech_outside": True,
                "unsafe_outside_frames": 0,
                "failed_closed_count": 0,
                "bridged_gap_count": 23,
            }
        ),
        encoding="utf-8",
    )
    return {
        "audit_summary": audit_summary,
        "audit_manifest": audit_manifest,
        "teacher_summary": teacher_summary,
        "teacher_preaudit": evidence,
        "calibration_summary": calibration_summary,
        "calibration_teacher_summary": calibration_teacher_summary,
        "calibration_gap_verdicts": gap_verdicts,
    }


def _compile(paths: dict[str, Path], output: Path) -> dict:
    return compile_real_train_dual_evidence(**paths, output_dir=output)


def test_compile_real_train_dual_evidence_binds_calibrated_three_state(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _compile(paths, tmp_path / "compiled")

    assert result["source_count"] == result["video_count"] == 2
    assert result["canonical_frame_counts"] == {
        "inside_candidate": 6,
        "outside_candidate": 4,
        "unsure": 6,
    }
    assert result["conflict_frames_mapped_to_unsure"] == 1
    assert result["calibration_protect_recall"] == 0.95
    assert result["calibration_gap_verdict_count"] == 23
    assert result["human_full_source_confirmed"] is False
    assert result["teacher_evidence_used_as_training_supervision"] is True
    rows = [
        json.loads(line)
        for line in Path(result["real_train_dual_evidence_sources"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["unsure_training_label"] == -100 for row in rows)
    assert all(row["calibration_gate_passed"] is True for row in rows)
    assert all(row["human_full_source_confirmed"] is False for row in rows)


def test_compile_real_train_dual_evidence_rejects_merge_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["teacher_preaudit"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["safe_outside_spans"] = []
    _write(paths["teacher_preaudit"], rows)
    with pytest.raises(ValueError, match="three-state merge mismatch"):
        _compile(paths, tmp_path / "merge-drift")


def test_compile_real_train_dual_evidence_rejects_weak_calibration(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    summary = json.loads(paths["calibration_summary"].read_text(encoding="utf-8"))
    summary["protect_recall"] = 0.949
    paths["calibration_summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration gate failed"):
        _compile(paths, tmp_path / "weak-calibration")
