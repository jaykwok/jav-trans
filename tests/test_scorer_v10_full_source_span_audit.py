from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.evaluate_scorer_v10_full_source_span_audit import evaluate
from tools.audits.generate_scorer_v10_full_source_span_audit_html import (
    ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    build_audit,
)


def _write_selection(tmp_path: Path) -> tuple[Path, str]:
    source_id = "scorer-bg-source"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "source.wav").write_bytes(b"RIFF-test")
    selection = tmp_path / "prediction_manifest.jsonl"
    selection.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "partition": "train",
                "row_role": "all_background",
                "category": "background_false_keep",
                "frame_count": 10,
                "duration_s": 0.2,
                "audio": "audio/source.wav",
                "prediction_spans": [
                    {
                        "label": "model_speech",
                        "start_frame": 8,
                        "end_frame": 9,
                    }
                ],
                "asr_probe_summary": {"texts_in_workflow_order": ["漏检"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection, source_id


def test_full_source_span_page_has_no_model_seed_and_saves_complete_truth(
    tmp_path: Path,
) -> None:
    selection, source_id = _write_selection(tmp_path)
    index = build_audit(
        prediction_audit_manifest=selection,
        output_dir=tmp_path / "audit",
        source_ids={source_id},
    )
    page = index.read_text(encoding="utf-8")
    assert "本页不显示模型输出" in page
    assert "未标出的差集只有在勾选" in page
    assert "reviewed_full_source" in page
    assert "complete_with_target_speech" in page
    assert "model_speech" not in page
    assert "prediction_spans" not in page
    assert "漏检" not in page
    manifest = [
        json.loads(line)
        for line in (index.parent / "audit_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest == [
            {
                "schema": ITEM_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
            "partition": "train",
            "frame_count": 10,
            "frame_hop_s": 0.02,
            "duration_s": 0.2,
            "audio": "audio/source-000.wav",
        }
    ]
    summary = json.loads((index.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_output_used_as_annotation_seed"] is False
    assert (
        summary["boundary_serialization_contract_id"]
        == ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert summary["asr_output_used_as_annotation_seed"] is False
    assert (
        summary[
            "unmarked_complement_becomes_background_only_after_full_source_confirmation"
        ]
        is True
    )
    assert summary["unsure_training_label"] == -100
    assert summary["training_manifest_allowed"] is False
    assert (index.parent / "audio" / "source-000.wav").is_file()
    script = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script is not None
    node = shutil.which("node")
    if node is not None:
        parsed = subprocess.run(
            [node, "--check", "-"],
            input=script.group(1),
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_full_source_span_gate_requires_gap_free_coverage(tmp_path: Path) -> None:
    selection, source_id = _write_selection(tmp_path)
    audit_dir = tmp_path / "audit"
    build_audit(
        prediction_audit_manifest=selection,
        output_dir=audit_dir,
        source_ids={source_id},
    )
    verdicts = tmp_path / "manual_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "reviewed_full_source": True,
                "verdict": "complete_with_target_speech",
                "spans": [
                    {
                        "label": "background",
                        "start_frame": 0,
                        "end_frame": 2,
                        "start_s": 0.0,
                        "end_s": 0.04,
                    },
                    {
                        "label": "speech",
                        "start_frame": 2,
                        "end_frame": 7,
                        "start_s": 0.04,
                        "end_s": 0.14,
                    },
                    {
                        "label": "unsure",
                        "start_frame": 7,
                        "end_frame": 8,
                        "start_s": 0.14,
                        "end_s": 0.16,
                    },
                    {
                        "label": "background",
                        "start_frame": 8,
                        "end_frame": 10,
                        "start_s": 0.16,
                        "end_s": 0.2,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gate_path = evaluate(
        audit_manifest=audit_dir / "audit_manifest.jsonl",
        manual_verdicts=verdicts,
        output_dir=tmp_path / "gate",
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["manual_gate_passed"] is True
    assert (
        gate["boundary_serialization_contract_id"]
        == ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    )
    assert gate["canonical_recompile_allowed"] is True
    assert gate["training_manifest_allowed"] is False
    assert gate["label_frame_counts"] == {
        "background": 4,
        "speech": 5,
        "unsure": 1,
    }
    decisions = [
        json.loads(line)
        for line in (gate_path.parent / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert decisions[0]["model_output_used_as_truth"] is False
    assert decisions[0]["asr_output_used_as_truth"] is False

    broken = json.loads(verdicts.read_text(encoding="utf-8"))
    broken["spans"][1]["start_frame"] = 3
    broken_verdicts = tmp_path / "broken.jsonl"
    broken_verdicts.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gap-free"):
        evaluate(
            audit_manifest=audit_dir / "audit_manifest.jsonl",
            manual_verdicts=broken_verdicts,
            output_dir=tmp_path / "broken-gate",
        )


def test_full_source_span_gate_keeps_unreviewed_rows_pending(tmp_path: Path) -> None:
    selection, source_id = _write_selection(tmp_path)
    audit_dir = tmp_path / "audit"
    build_audit(
        prediction_audit_manifest=selection,
        output_dir=audit_dir,
        source_ids={source_id},
    )
    verdicts = tmp_path / "manual_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "schema": MANUAL_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "partition": "train",
                "frame_count": 10,
                "frame_hop_s": 0.02,
                "reviewed_full_source": False,
                "verdict": "unreviewed",
                "spans": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gate_path = evaluate(
        audit_manifest=audit_dir / "audit_manifest.jsonl",
        manual_verdicts=verdicts,
        output_dir=tmp_path / "gate",
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["manual_gate_passed"] is False
    assert gate["canonical_recompile_allowed"] is False
    assert gate["unreviewed_source_ids"] == [source_id]
