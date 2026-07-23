from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
import tools.boundary.ja.compile_candidate_island_scorer_v11_downstream_isolation as downstream_tools
from tools.boundary.ja.compile_candidate_island_scorer_v11_downstream_isolation import (
    AUDIT_SCHEMA,
    BRIDGE_VERDICT_SCHEMA,
    DUAL_REVIEW_SUMMARY_SCHEMA,
    EVIDENCE_SCHEMA,
    HELDOUT_AUDIT_ITEM_SCHEMA,
    HELDOUT_VERDICT_SCHEMA,
    REQUIREMENT_SCHEMA,
    RESPONSIBILITY_VERDICT_SCHEMA,
    SELECTION_SCHEMA,
    compile_downstream_isolation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_wav(path: Path, *, frames: int = 10) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames * 320)


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    contract = ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    audit_dir = tmp_path / "source-audit"
    audio_dir = audit_dir / "audio"
    audio_dir.mkdir(parents=True)
    heldout_rows = []
    audit_rows = []
    bridge_rows = []
    for source_id, partition in (("source-a", "val"), ("source-b", "test")):
        audio = audio_dir / f"{source_id}.wav"
        _write_wav(audio)
        heldout_rows.append(
            {
                "schema": HELDOUT_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "source_id": source_id,
                "partition": partition,
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "reviewed_full_source": True,
                "spans": [
                    {"label": "outside_candidate", "start_frame": 0, "end_frame": 1},
                    {"label": "inside_candidate", "start_frame": 1, "end_frame": 3},
                    {"label": "outside_candidate", "start_frame": 3, "end_frame": 6},
                    {"label": "inside_candidate", "start_frame": 6, "end_frame": 9},
                    {"label": "outside_candidate", "start_frame": 9, "end_frame": 10},
                ],
            }
        )
        audit_rows.append(
            {
                "schema": HELDOUT_AUDIT_ITEM_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "source_id": source_id,
                "partition": partition,
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "duration_s": 0.2,
                "audio": f"audio/{source_id}.wav",
                "audio_sha256": _sha256(audio),
            }
        )
        gap_id = f"{source_id}::bridge-gap::000003-000006"
        bridge_rows.append(
            {
                "schema": BRIDGE_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "gap_id": gap_id,
                "source_id": source_id,
                "partition": partition,
                "start_frame": 3,
                "end_frame": 6,
                "start_s": 0.06,
                "end_s": 0.12,
                "duration_s": 0.06,
                "content_verdict": "no_semantic_dialogue",
                "semantic_coverage_verdict": "not_applicable_no_semantic",
                "envelope_verdict": "overmerged_independent_background",
                "combined_verdict": "teacher_overmerged_independent_background",
                "verdict": "teacher_overmerged_independent_background",
                "complete": True,
            }
        )
    heldout = tmp_path / "heldout.jsonl"
    manifest = audit_dir / "audit_manifest.jsonl"
    bridges = tmp_path / "bridge_verdicts.jsonl"
    selection = tmp_path / "selection.jsonl"
    _write_jsonl(heldout, heldout_rows)
    _write_jsonl(manifest, audit_rows)
    _write_jsonl(bridges, bridge_rows)
    selected = bridge_rows[0]
    _write_jsonl(
        selection,
        [
            {
                "schema": SELECTION_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "gap_id": selected["gap_id"],
                "source_id": selected["source_id"],
                "partition": selected["partition"],
                "start_frame": selected["start_frame"],
                "end_frame": selected["end_frame"],
                "decision": "independent_background_needs_downstream_isolation",
            }
        ],
    )
    review_summary = tmp_path / "review_summary.json"
    review_summary.write_text(
        json.dumps(
            {
                "schema": DUAL_REVIEW_SUMMARY_SCHEMA,
                "boundary_serialization_contract_id": contract,
                "human_verdicts": str(heldout),
                "human_verdicts_sha256": _sha256(heldout),
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
                "manual_verdicts": str(bridges),
            }
        ),
        encoding="utf-8",
    )
    return {
        "heldout": heldout,
        "manifest": manifest,
        "bridges": bridges,
        "selection": selection,
        "review_summary": review_summary,
        "selected_gap_id": selected["gap_id"],
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_downstream_isolation_missing_evidence_stays_unsure_and_preserves_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downstream_tools, "PROJECT_ROOT", tmp_path)
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"
    summary = compile_downstream_isolation(
        heldout_verdicts=fixture["heldout"],
        review_summary=fixture["review_summary"],
        bridge_verdicts=fixture["bridges"],
        selection=fixture["selection"],
        output_dir=output,
    )

    assert summary["requirement_count"] == 1
    assert summary["source_count"] == 1
    assert summary["evidence_status_counts"] == {"evidence_missing": 1}
    assert summary["all_requirements_evidence_missing"] is True
    assert summary["raw_manual_verdicts_modified"] is False
    requirement = _rows(output / "downstream_isolation_requirements.jsonl")[0]
    assert requirement["schema"] == REQUIREMENT_SCHEMA
    assert requirement["duty_label"] == (
        "independent_background_needs_downstream_isolation"
    )
    assert requirement["scorer_canonical_label"] == "unsure"
    assert requirement["scorer_training_label"] == -100
    assert requirement["missing_stages"] == [
        "scorer_candidate_islands",
        "proposal_candidates",
        "split_events",
        "provisional_sub_islands",
        "cueqc_decisions",
    ]
    audit = _rows(output / "downstream_isolation_audit.jsonl")[0]
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["evidence_status"] == "evidence_missing"
    assert audit["source_audio_sha256"]
    responsibility = {
        row["source_id"]: row
        for row in _rows(output / "responsibility_verdicts.jsonl")
    }
    assert responsibility["source-a"]["schema"] == RESPONSIBILITY_VERDICT_SCHEMA
    assert responsibility["source-a"]["spans"][2] == {
        "label": "unsure",
        "start_frame": 3,
        "end_frame": 6,
        "start_s": 0.06,
        "end_s": 0.12,
    }
    assert responsibility["source-a"]["unsure_training_label"] == -100
    assert responsibility["source-b"]["spans"][2]["label"] == "outside_candidate"
    assert responsibility["source-b"]["downstream_isolation_requirement_ids"] == []
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "evidence_missing" in page
    assert "播放器只播放精确区间" in page
    assert "/audio/source-a.wav" in page
    assert Path(fixture["heldout"]).read_text(encoding="utf-8").count("unsure") == 0


def test_downstream_isolation_requires_bound_argmax_workflow_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downstream_tools, "PROJECT_ROOT", tmp_path)
    fixture = _fixture(tmp_path)
    checkpoint_shas = {
        "scorer": "1" * 64,
        "proposal": "2" * 64,
        "split": "3" * 64,
        "cueqc": "4" * 64,
    }
    selected_gap_id = str(fixture["selected_gap_id"])
    audio_sha = _rows(Path(fixture["manifest"]))[0]["audio_sha256"]
    evidence = tmp_path / "evidence.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "schema": EVIDENCE_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "requirement_id": selected_gap_id,
                "source_id": "source-a",
                "partition": "val",
                "start_frame": 3,
                "end_frame": 6,
                "source_audio_sha256": audio_sha,
                "scorer_checkpoint_sha256": checkpoint_shas["scorer"],
                "proposal_checkpoint_sha256": checkpoint_shas["proposal"],
                "split_checkpoint_sha256": checkpoint_shas["split"],
                "cueqc_checkpoint_sha256": checkpoint_shas["cueqc"],
                "scorer_candidate_islands": [
                    {"start_frame": 1, "end_frame": 9}
                ],
                "proposal_candidates": [{"frame": 3}, {"frame": 6}],
                "split_events": [
                    {"frame": 3, "argmax_label": "cut"},
                    {"frame": 6, "argmax_label": "cut"},
                ],
                "provisional_sub_islands": [
                    {"island_id": "background", "start_frame": 3, "end_frame": 6}
                ],
                "cueqc_decisions": [
                    {"island_id": "background", "argmax_label": "drop"}
                ],
            }
        ],
    )
    output = tmp_path / "bound-output"
    summary = compile_downstream_isolation(
        heldout_verdicts=fixture["heldout"],
        review_summary=fixture["review_summary"],
        bridge_verdicts=fixture["bridges"],
        selection=fixture["selection"],
        downstream_evidence=evidence,
        expected_checkpoint_shas=checkpoint_shas,
        output_dir=output,
    )
    assert summary["evidence_status_counts"] == {
        "downstream_isolation_demonstrated": 1
    }
    requirement = _rows(output / "downstream_isolation_requirements.jsonl")[0]
    assert requirement["scorer_canonical_label"] == "inside_candidate"
    assert requirement["scorer_training_label"] == 1
    audit = _rows(output / "downstream_isolation_audit.jsonl")[0]
    assert all(audit["checks"].values())
    responsibility = _rows(output / "responsibility_verdicts.jsonl")[0]
    assert responsibility["spans"][1] == {
        "label": "inside_candidate",
        "start_frame": 1,
        "end_frame": 9,
        "start_s": 0.02,
        "end_s": 0.18,
    }

    bad_rows = _rows(evidence)
    bad_rows[0]["split_checkpoint_sha256"] = "5" * 64
    bad_evidence = tmp_path / "bad-evidence.jsonl"
    _write_jsonl(bad_evidence, bad_rows)
    with pytest.raises(ValueError, match="split checkpoint SHA mismatch"):
        compile_downstream_isolation(
            heldout_verdicts=fixture["heldout"],
            review_summary=fixture["review_summary"],
            bridge_verdicts=fixture["bridges"],
            selection=fixture["selection"],
            downstream_evidence=bad_evidence,
            expected_checkpoint_shas=checkpoint_shas,
            output_dir=tmp_path / "bad-output",
        )


def test_downstream_isolation_rejects_unselected_or_relabelled_gap(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selected = _rows(Path(fixture["selection"]))
    selected[0]["source_id"] = "source-b"
    bad_selection = tmp_path / "bad-selection.jsonl"
    _write_jsonl(bad_selection, selected)
    with pytest.raises(ValueError, match="selection/bridge identity mismatch"):
        compile_downstream_isolation(
            heldout_verdicts=fixture["heldout"],
            review_summary=fixture["review_summary"],
            bridge_verdicts=fixture["bridges"],
            selection=bad_selection,
            output_dir=tmp_path / "bad-selection-output",
        )


def test_downstream_isolation_audio_root_fallback_is_sha_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downstream_tools, "PROJECT_ROOT", tmp_path)
    fixture = _fixture(tmp_path)
    fallback = tmp_path / "verified-audio-root"
    fallback.mkdir()
    source_audio = tmp_path / "source-audit" / "audio" / "source-a.wav"
    source_audio.replace(fallback / source_audio.name)
    summary = compile_downstream_isolation(
        heldout_verdicts=fixture["heldout"],
        review_summary=fixture["review_summary"],
        bridge_verdicts=fixture["bridges"],
        selection=fixture["selection"],
        audio_root=fallback,
        output_dir=tmp_path / "fallback-output",
    )
    assert summary["audio_root"] == "verified-audio-root"

    (fallback / source_audio.name).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="audio SHA mismatch"):
        compile_downstream_isolation(
            heldout_verdicts=fixture["heldout"],
            review_summary=fixture["review_summary"],
            bridge_verdicts=fixture["bridges"],
            selection=fixture["selection"],
            audio_root=fallback,
            output_dir=tmp_path / "bad-fallback-output",
        )
