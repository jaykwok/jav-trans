from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID
from asr.pre_asr_cueqc import (
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_MODEL_ARCH,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCHEMA,
)
from tools.asr.cueqc.promote_pre_asr_v13_after_audit import promote


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(tmp_path: Path, *, num_classes: int = 2) -> tuple[Path, Path, Path]:
    torch = pytest.importorskip("torch")
    split = tmp_path / "split.pt"
    inner = tmp_path / "inner.pt"
    split.write_bytes(b"split-v4")
    inner.write_bytes(b"inner-v1")
    path = tmp_path / "candidate.pt"
    torch.save(
        {
            "schema": PRE_ASR_CUEQC_SCHEMA,
            "arch": PRE_ASR_CUEQC_MODEL_ARCH,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
            "model_config": {
                "num_classes": num_classes,
                "valid_prefix_temporal": True,
                "ptm_encoder_mode": "token_attention",
                "semantic_auxiliary": True,
                "late_fusion": True,
            },
            "decision_config": {
                "decision_mode": "argmax",
                "model_only": True,
                "hard_keep_veto": False,
                "hard_drop_rule": False,
                "keep_veto": False,
            },
            "semantic_split_weights_sha256": _sha(split),
            "inner_edge_refiner_weights_sha256": _sha(inner),
            "metadata": {
                "asr_repo_id": QWEN_ASR_17B_REPO_ID,
                "training_labels": ["drop", "keep"],
                "excluded_training_labels": ["unsure"],
                "excluded_training_label_count": 7,
                "ptm_pooling_schemas": ["pre_asr_chunk_learned_projected_ptm_v3"],
                "ptm_projection_digest": "projection-sha",
                "semantic_split_weights_sha256": _sha(split),
                "inner_edge_refiner_weights_sha256": _sha(inner),
            },
        },
        path,
    )
    return path, split, inner


def test_v13_promoter_accepts_only_binary_argmax_with_closed_false_drop_gate(
    tmp_path: Path,
) -> None:
    candidate, split, inner = _candidate(tmp_path)
    model_gate = tmp_path / "model-gate.json"
    audit_gate = tmp_path / "audit-gate.json"
    _write_json(
        model_gate,
        {
            "gate": {
                "passed": True,
                "min_keep_recall": 0.95,
                "min_drop_recall": 0.95,
            },
            "all_false_drop_count": 0,
        },
    )
    _write_json(
        audit_gate,
        {
            "complete": True,
            "promote_allowed": True,
            "true_semantic_keep_deletion_count": 0,
            "target_manifest_count": 0,
        },
    )

    summary = promote(
        candidate_checkpoint=candidate,
        output_checkpoint=tmp_path / "production.pt",
        model_gate_path=model_gate,
        audit_gate_path=audit_gate,
        split_checkpoint=split,
        inner_checkpoint=inner,
        asr_repo_id=QWEN_ASR_17B_REPO_ID,
        source_training_run="agents/temp/train",
        dry_run=True,
    )

    assert summary["candidate_contract"]["num_classes"] == 2
    assert summary["candidate_contract"]["decision_mode"] == "argmax"
    assert summary["promoted"] is False


def test_v13_promoter_rejects_legacy_three_class_candidate(tmp_path: Path) -> None:
    candidate, split, inner = _candidate(tmp_path, num_classes=3)
    model_gate = tmp_path / "model-gate.json"
    audit_gate = tmp_path / "audit-gate.json"
    _write_json(model_gate, {})
    _write_json(audit_gate, {})

    with pytest.raises(ValueError, match="binary CueQC v13 head"):
        promote(
            candidate_checkpoint=candidate,
            output_checkpoint=tmp_path / "production.pt",
            model_gate_path=model_gate,
            audit_gate_path=audit_gate,
            split_checkpoint=split,
            inner_checkpoint=inner,
            asr_repo_id=QWEN_ASR_17B_REPO_ID,
            source_training_run="agents/temp/train",
            dry_run=True,
        )
