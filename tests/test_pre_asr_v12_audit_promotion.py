from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.asr.cueqc.promote_pre_asr_v12_after_audit import (
    V12_SCHEMA,
    promote_after_audit,
    validate_audit_gate,
    validate_model_gate,
)


REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


def _model_gate(*, drop_pass: bool = True, keep_pass: bool = True) -> dict:
    return {
        "schema": "pre_asr_cueqc_v12_gate_summary_v1",
        "v12_threshold": 0.5,
        "gate": {
            "drop_recall_pass": drop_pass,
            "keep_recall_pass": keep_pass,
            "min_drop_recall": 0.98,
            "long_false_drop_count": 22,
            "long_false_drop_min_s": 0.0,
        },
        "v12": {
            "drop_recall": 0.983,
            "semantic_keep_recall": 0.991,
            "false_drop_count": 39,
            "false_keep_count": 155,
        },
    }


def _audit_gate(*, complete: bool = True, promote_allowed: bool = True, keep_deletions: int = 0) -> dict:
    return {
        "schema": "pre_asr_v12_false_drop_audit_gate_summary_v1",
        "complete": complete,
        "promote_allowed": promote_allowed,
        "target_manifest_count": 22,
        "reviewed_target_count": 22 if complete else 21,
        "manual_verdict_counts": {"drop": 21, "unsure": 1},
        "true_semantic_keep_deletion_count": keep_deletions,
        "uncertain_count": 1,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_checkpoint(path: Path, *, repo_id: str = REPO_ID, schema: str = V12_SCHEMA) -> Path:
    torch = pytest.importorskip("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": schema,
            "arch": "cueqc_pre_asr_semantic_chunk_v12",
            "feature_schema": "pre_asr_cueqc_features_v9",
            "runtime_adapter": "pre_asr_semantic_chunk_sequence_v4",
            "model_config": {
                "valid_prefix_temporal": True,
                "ptm_encoder_mode": "token_attention",
                "semantic_auxiliary": True,
                "late_fusion": True,
            },
            "metadata": {
                "asr_repo_id": repo_id,
                "semantic_split_weights_sha256": "a" * 64,
                "artifact": {"name": "pre_asr_cueqc"},
                "ptm_pooling_schemas": ["pre_asr_chunk_pooled_ptm_v1"],
            },
            "semantic_split_weights_sha256": "a" * 64,
            "decision_config": {
                "model_only": True,
                "hard_keep_veto": False,
                "hard_drop_rule": False,
                "keep_veto": False,
            },
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        path,
    )
    return path


def test_validate_model_and_audit_gate_reject_failures() -> None:
    with pytest.raises(ValueError, match="drop_recall_pass=false"):
        validate_model_gate(_model_gate(drop_pass=False))
    with pytest.raises(ValueError, match="audit gate failed"):
        validate_audit_gate(_audit_gate(promote_allowed=False, keep_deletions=1))


def test_promote_after_audit_dry_run_reports_registry_update(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "candidate.pt")
    model_gate = _write_json(tmp_path / "model_gate.json", _model_gate())
    audit_gate = _write_json(tmp_path / "audit_gate.json", _audit_gate())

    summary = promote_after_audit(
        candidate_checkpoint=checkpoint,
        output_checkpoint=tmp_path / "pre_asr_cueqc_v12.pt",
        model_gate_path=model_gate,
        audit_gate_path=audit_gate,
        asr_repo_id=REPO_ID,
        source_training_run="agents/temp/train",
        dry_run=True,
    )

    assert summary["promoted"] is False
    assert summary["candidate_contract"]["schema"] == V12_SCHEMA
    assert summary["selected_validation"]["manual_false_drop_audit"]["uncertain_count"] == 1
    assert summary["registry_update_required"]["mapping"] == "DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO"


def test_promote_after_audit_writes_promoted_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = _write_checkpoint(tmp_path / "candidate.pt")
    output = tmp_path / "production.pt"

    summary = promote_after_audit(
        candidate_checkpoint=checkpoint,
        output_checkpoint=output,
        model_gate_path=_write_json(tmp_path / "model_gate.json", _model_gate()),
        audit_gate_path=_write_json(tmp_path / "audit_gate.json", _audit_gate()),
        asr_repo_id=REPO_ID,
        source_training_run="agents/temp/train",
        dry_run=False,
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert summary["promoted"] is True
    assert payload["metadata"]["artifact"] == {
        "name": "pre_asr_cueqc",
        "display_name": "Pre-ASR CueQC",
        "version": "v12",
        "pipeline_stage": 5,
        "pipeline_role": "final_chunk_keep_drop_routing",
        "checkpoint_format_version": 1,
        "production_filename": "production.pt",
        "promoted": True,
        "promoted_at": payload["metadata"]["artifact"]["promoted_at"],
        "self_contained": True,
        "source_training_run": "agents/temp/train",
    }
    assert payload["metadata"]["selected_validation"]["model_gate"]["drop_recall"] == pytest.approx(0.983)
    assert payload["decision_config"]["drop_threshold"] == 0.5


def test_promote_after_audit_rejects_wrong_repo_checkpoint(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "candidate.pt", repo_id="other/repo")
    with pytest.raises(ValueError, match="does not match"):
        promote_after_audit(
            candidate_checkpoint=checkpoint,
            output_checkpoint=tmp_path / "production.pt",
            model_gate_path=_write_json(tmp_path / "model_gate.json", _model_gate()),
            audit_gate_path=_write_json(tmp_path / "audit_gate.json", _audit_gate()),
            asr_repo_id=REPO_ID,
            source_training_run="agents/temp/train",
            dry_run=True,
        )
