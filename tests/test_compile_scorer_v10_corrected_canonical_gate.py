from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.audits.compile_scorer_v10_corrected_canonical_gate import compile_gate


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path, source_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"source_id": source_id}) + "\n" for source_id in source_ids),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_gate_binds_complete_repair_and_replacement_chain(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.jsonl"
    r1_canonical = tmp_path / "r1.jsonl"
    r2_canonical = tmp_path / "r2.jsonl"
    original.write_text("original\n", encoding="utf-8")
    r1_canonical.write_text("r1\n", encoding="utf-8")
    r2_canonical.write_text("r2\n", encoding="utf-8")

    initial = tmp_path / "initial.json"
    span = tmp_path / "span.json"
    replacement7 = tmp_path / "replacement7.json"
    replacement2 = tmp_path / "replacement2.json"
    r1_summary = tmp_path / "r1-summary.json"
    r2_summary = tmp_path / "r2-summary.json"
    replacement7_manifest = tmp_path / "replacement7.jsonl"
    replacement2_manifest = tmp_path / "replacement2.jsonl"
    _write_manifest(replacement7_manifest, ["repair", "replacement-a"])
    _write_manifest(replacement2_manifest, ["replacement-b"])
    _write(
        initial,
        {
            "schema": "speech_scorer_v10_canonical_manual_gate_v1",
            "canonical_sources_sha256": _sha(original),
            "complete": True,
            "manual_gate_pass": False,
            "training_manifest_allowed": False,
            "risk_count": 2,
            "verdict_counts": {"correct": 18},
        },
    )
    _write(
        span,
        {
            "schema": "speech_scorer_v10_canonical_span_repair_gate_v1",
            "canonical_sources_sha256": _sha(original),
            "complete": True,
            "canonical_recompile_ready": True,
            "target_count": 10,
        },
    )
    _write(
        r1_summary,
        {
            "schema": "speech_scorer_v10_corrected_canonical_summary_v1",
            "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
            "input_canonical_sources_sha256": _sha(original),
            "repair_gate_sha256": _sha(span),
            "canonical_sources": str(r1_canonical),
            "replacement_audit_source_ids": ["repair", "replacement-a"],
            "quarantined_background_ids": ["bg-a"],
            "ignored_core_ids": ["core-a"],
        },
    )
    _write(
        replacement7,
        {
            "schema": "speech_scorer_v10_canonical_manual_gate_v1",
            "canonical_sources_sha256": _sha(r1_canonical),
            "complete": True,
            "manual_gate_pass": False,
            "training_manifest_allowed": False,
            "canonical_recompile_ready": True,
            "quarantined_background_ids": ["bg-b"],
            "risk_count": 1,
            "target_count": 2,
            "verdict_count": 2,
            "verdict_counts": {"correct": 1, "contains_target_speech": 1},
            "audit_manifest": str(replacement7_manifest),
        },
    )
    _write(
        r2_summary,
        {
            "schema": "speech_scorer_v10_corrected_canonical_summary_v1",
            "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
            "input_canonical_sources_sha256": _sha(r1_canonical),
            "repair_gate_sha256": _sha(replacement7),
            "previous_summary": str(r1_summary),
            "canonical_sources": str(r2_canonical),
            "replacement_audit_source_ids": ["replacement-b"],
            "quarantined_background_ids": ["bg-a", "bg-b"],
            "ignored_core_ids": ["core-a", "core-b"],
            "dataset": {"source_count": 10},
            "canonical_frame_counts": {"speech": 20, "background": 10, "unsure": 1},
        },
    )
    _write(
        replacement2,
        {
            "schema": "speech_scorer_v10_canonical_manual_gate_v1",
            "canonical_sources_sha256": _sha(r2_canonical),
            "complete": True,
            "manual_gate_pass": True,
            "training_manifest_allowed": True,
            "risk_count": 0,
            "target_count": 1,
            "verdict_count": 1,
            "verdict_counts": {"correct": 1},
            "audit_manifest": str(replacement2_manifest),
        },
    )

    result = compile_gate(
        initial_gate=initial,
        span_repair_gate=span,
        corrected_r1_summary=r1_summary,
        replacement7_gate=replacement7,
        corrected_r2_summary=r2_summary,
        replacement2_gate=replacement2,
        output=tmp_path / "combined.json",
    )
    assert result["manual_gate_pass"] is True
    assert result["feature_cache_allowed"] is True
    assert result["boundary_serialization_contract_id"] == "boundary_acoustic_binary_v12"
    assert result["canonical_sources_sha256"] == _sha(r2_canonical)
    assert result["quarantined_background_ids"] == ["bg-a", "bg-b"]

    replacement2.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        compile_gate(
            initial_gate=initial,
            span_repair_gate=span,
            corrected_r1_summary=r1_summary,
            replacement7_gate=replacement7,
            corrected_r2_summary=r2_summary,
            replacement2_gate=replacement2,
            output=tmp_path / "rejected.json",
        )
