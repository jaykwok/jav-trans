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
from tools.boundary.ja.label_candidate_island_scorer_v11_dual_evidence_with_omni import (
    PROMPT_PROFILE,
    PROMPT_VERSION,
    PROTECT_PROMPT_VERSION,
    REMOVE_PROMPT_VERSION,
    TEACHER_EXECUTION_CONTRACT_ID,
)
from tools.omni.timestamp_contract import TIMESTAMP_CONTRACT_ID


CONTRACT = ACOUSTIC_BINARY_V12_CONTRACT.contract_id
MODEL = "google/gemini-3.6-flash"
PROFILE = PROMPT_PROFILE
PROTECT = PROTECT_PROMPT_VERSION
REMOVE = REMOVE_PROMPT_VERSION
PROMPT = PROMPT_VERSION


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
        "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "teacher_execution_contract_id": TEACHER_EXECUTION_CONTRACT_ID,
        "model": MODEL,
        "env_file_name": "gemini",
        "provider_profile": "gemini",
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "max_tokens": 8192,
        "exclude_reasoning": False,
        "require_provider_parameters": True,
        "response_format": {"type": "json_object"},
        "audio_content_mode": "input_audio_raw",
        "base_url_host": "openrouter.ai",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "labels": str(labels),
        "source_count": sources,
        "frame_count": frames,
        "failed_closed_count": 0,
        "reasoning_contract_satisfied": True,
        "protect_reasoning_evidence_count": sources,
        "remove_reasoning_evidence_count": sources,
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
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "teacher_execution_contract_id": TEACHER_EXECUTION_CONTRACT_ID,
                "provider_profile": "gemini",
                "reasoning_effort": "medium",
                "max_tokens": 8192,
                "exclude_reasoning": False,
                "require_provider_parameters": True,
                "protect_reasoning": {
                    "reasoning_tokens": 64,
                    "reasoning_evidence_present": True,
                },
                "remove_reasoning": {
                    "reasoning_tokens": 64,
                    "reasoning_evidence_present": True,
                },
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
    excluded = tmp_path / "remaining_outside.jsonl"
    original_outside = tmp_path / "original_outside.jsonl"
    parent_teacher_summary = tmp_path / "parent_teacher_summary.json"
    parent_teacher_preaudit = tmp_path / "parent_teacher_preaudit.jsonl"
    _write(source_manifest, source_rows)
    _write(audit_manifest, audit_rows)
    _write(evidence, evidence_rows)
    _write(excluded, [{"source_id": "not-selected"}])
    _write(
        original_outside,
        [{"source_id": row["source_id"]} for row in source_rows]
        + [{"source_id": "not-selected"}],
    )
    _write(parent_teacher_preaudit, [{"source_id": "parent"}])
    parent_teacher_summary.write_text(json.dumps({"source_id": "parent"}), encoding="utf-8")
    teacher_summary = tmp_path / "teacher_summary.json"
    teacher_payload = _teacher_summary(audit_manifest, evidence, sources=2, frames=16)
    teacher_payload.update(
        {
            "source_ids": [row["source_id"] for row in source_rows],
            "inside_frames": counts["inside_candidate"],
            "outside_frames": counts["outside_candidate"],
            "unsure_frames": counts["unsure"],
            "conflict_frames": conflicts,
            "selection_derived": True,
            "selection_policy": (
                "one_per_video_mixed_then_balance_coverage_conflict_source_id_v1"
            ),
            "selection_parent_teacher_summary": str(parent_teacher_summary),
            "selection_parent_teacher_summary_sha256": _sha256(parent_teacher_summary),
            "selection_parent_teacher_preaudit": str(parent_teacher_preaudit),
            "selection_parent_teacher_preaudit_sha256": _sha256(parent_teacher_preaudit),
        }
    )
    teacher_summary.write_text(json.dumps(teacher_payload), encoding="utf-8")
    selected_ids = [row["source_id"] for row in source_rows]
    audit_summary = audit_dir / "summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "schema": (
                    "candidate_island_scorer_v11_dual_evidence_train_selection_summary_v1"
                ),
                "boundary_serialization_contract_id": CONTRACT,
                "selection_policy": (
                    "one_per_video_mixed_then_balance_coverage_conflict_source_id_v1"
                ),
                "audit_manifest": str(audit_manifest),
                "audit_manifest_sha256": _sha256(audit_manifest),
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": _sha256(source_manifest),
                "preaudit": str(evidence),
                "preaudit_sha256": _sha256(evidence),
                "exclude_sources": str(excluded),
                "exclude_sources_sha256": _sha256(excluded),
                "remaining_outside_sources": str(excluded),
                "remaining_outside_sources_sha256": _sha256(excluded),
                "original_outside_sources": str(original_outside),
                "original_outside_sources_sha256": _sha256(original_outside),
                "parent_teacher_summary": str(parent_teacher_summary),
                "parent_teacher_summary_sha256": _sha256(parent_teacher_summary),
                "parent_teacher_preaudit": str(parent_teacher_preaudit),
                "parent_teacher_preaudit_sha256": _sha256(parent_teacher_preaudit),
                "selected_teacher_summary": str(teacher_summary),
                "selected_teacher_summary_sha256": _sha256(teacher_summary),
                "selected_source_ids": selected_ids,
                "replaced_outside_source_ids": selected_ids,
                "remaining_outside_source_ids": ["not-selected"],
                "remaining_outside_source_count": 1,
                "source_count": 2,
                "video_count": 2,
                "teacher_output_used_as_truth": False,
                "teacher_output_used_as_calibrated_evidence": True,
                "unselected_source_label_inheritance": False,
                "training_manifest_allowed": False,
            }
        ),
        encoding="utf-8",
    )

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
        for index in range(3)
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
                "protect_recall": 0.10,
                "final_outside_precision": 0.95,
                "human_inside_frames": 20,
                "true_speech_retention": 0.95,
                "unsafe_outside_frames": 1,
                "failed_closed_count": 0,
                "bridged_gap_count": len(gaps),
            }
        ),
        encoding="utf-8",
    )
    ab_dir = tmp_path / "ab"
    ab_per_source = ab_dir / "per_source.jsonl"
    ab_verdicts = ab_dir / "manual_verdicts.jsonl"
    base_review = ab_dir / "base.json"
    candidate_review = ab_dir / "candidate.json"
    _write(ab_per_source, [{"source_id": "heldout"}])
    _write(
        ab_verdicts,
        [
            {
                "schema": "candidate_island_dual_evidence_ab_manual_verdict_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "source_id": "heldout",
                "base_name": "Medium",
                "candidate_name": "High",
                "comparison_verdict": "base_better",
            }
        ],
    )
    base_review.write_text(json.dumps({"arm": "Medium"}), encoding="utf-8")
    candidate_review.write_text(json.dumps({"arm": "High"}), encoding="utf-8")
    calibration_ab_summary = ab_dir / "summary.json"
    calibration_ab_summary.write_text(
        json.dumps(
            {
                "schema": "candidate_island_dual_evidence_ab_review_summary_v1",
                "boundary_serialization_contract_id": CONTRACT,
                "base_name": "Medium",
                "candidate_name": "High",
                "source_count": 1,
                "base_metrics": {
                    "true_speech_retention": 0.95,
                    "final_outside_precision": 0.95,
                    "protect_recall": 0.10,
                },
                "base_review": str(base_review),
                "base_review_sha256": _sha256(base_review),
                "candidate_review": str(candidate_review),
                "candidate_review_sha256": _sha256(candidate_review),
                "per_source": str(ab_per_source),
                "manual_verdicts": str(ab_verdicts),
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
        "calibration_ab_summary": calibration_ab_summary,
        "calibration_ab_verdicts": ab_verdicts,
    }


def _compile(paths: dict[str, Path], output: Path) -> dict:
    return compile_real_train_dual_evidence(**paths, output_dir=output)


def _rebind_selection(paths: dict[str, Path]) -> None:
    summary = json.loads(paths["audit_summary"].read_text(encoding="utf-8"))
    summary["selected_teacher_summary_sha256"] = _sha256(paths["teacher_summary"])
    summary["preaudit_sha256"] = _sha256(paths["teacher_preaudit"])
    paths["audit_summary"].write_text(json.dumps(summary), encoding="utf-8")


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
    assert result["calibration_protect_recall"] == 0.10
    assert result["calibration_protect_recall_is_diagnostic_only"] is True
    assert result["calibration_true_speech_retention"] == 0.95
    assert result["calibration_true_speech_retention_gate"] == 0.95
    assert result["calibration_final_outside_precision_gate"] == 0.95
    assert result["calibration_bridged_gap_count"] == 3
    assert result["calibration_bridge_gap_manual_gate"] is False
    assert result["calibration_ab_verdict_counts"] == {"base_better": 1}
    assert result["calibration_selected_arm"] == "Medium"
    assert result["human_full_source_confirmed"] is False
    assert result["teacher_evidence_used_as_training_supervision"] is True
    assert result["teacher_timestamp_contract_id"] == TIMESTAMP_CONTRACT_ID
    assert result["teacher_execution_contract_id"] == TEACHER_EXECUTION_CONTRACT_ID
    rows = [
        json.loads(line)
        for line in Path(result["real_train_dual_evidence_sources"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["unsure_training_label"] == -100 for row in rows)
    assert all(row["calibration_gate_passed"] is True for row in rows)
    assert all(row["human_full_source_confirmed"] is False for row in rows)
    assert all(
        row["teacher_timestamp_contract_id"] == TIMESTAMP_CONTRACT_ID for row in rows
    )
    assert all(
        row["teacher_execution_contract_id"] == TEACHER_EXECUTION_CONTRACT_ID
        for row in rows
    )


def test_compile_real_train_dual_evidence_rejects_merge_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["teacher_preaudit"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["safe_outside_spans"] = []
    _write(paths["teacher_preaudit"], rows)
    _rebind_selection(paths)
    with pytest.raises(ValueError, match="three-state merge mismatch"):
        _compile(paths, tmp_path / "merge-drift")


def test_compile_real_train_dual_evidence_rejects_weak_calibration(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    summary = json.loads(paths["calibration_summary"].read_text(encoding="utf-8"))
    summary["unsafe_outside_frames"] = 2
    summary["true_speech_retention"] = 0.90
    paths["calibration_summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration gate failed"):
        _compile(paths, tmp_path / "weak-calibration")


def test_compile_real_train_dual_evidence_rejects_old_timestamp_contract(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    summary = json.loads(paths["teacher_summary"].read_text(encoding="utf-8"))
    summary.pop("teacher_timestamp_contract_id")
    paths["teacher_summary"].write_text(json.dumps(summary), encoding="utf-8")
    _rebind_selection(paths)
    with pytest.raises(ValueError, match="teacher_timestamp_contract_id"):
        _compile(paths, tmp_path / "old-timestamp-contract")


def test_compile_real_train_dual_evidence_rejects_signature_only_reasoning(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["teacher_preaudit"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["protect_reasoning"] = {
        "reasoning_tokens": 0,
        "reasoning_signature_count": 1,
        "reasoning_signature_formats": ["google-gemini-v1"],
        "reasoning_transport_evidence_present": True,
        "reasoning_evidence_present": False,
    }
    _write(paths["teacher_preaudit"], rows)
    _rebind_selection(paths)
    with pytest.raises(ValueError, match="training-grade reasoning"):
        _compile(paths, tmp_path / "signature-only")
