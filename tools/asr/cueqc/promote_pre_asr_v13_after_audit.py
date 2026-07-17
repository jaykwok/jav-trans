#!/usr/bin/env python3
"""Promote CueQC v13 only after binary role-holdout and false-drop gates pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import (  # noqa: E402
    QWEN_ASR_17B_REPO_ID,
    qwen_asr_repo_tag,
)
from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_ARTIFACT,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_MODEL_ARCH,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCHEMA,
)
from tools.workflows.promote_torch_checkpoint import promote_checkpoint  # noqa: E402


ACTIVE_POOLING_SCHEMA = "pre_asr_chunk_learned_projected_ptm_v3"
SUMMARY_SCHEMA = "pre_asr_cueqc_v13_audit_gated_promotion_summary_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_output_checkpoint(asr_repo_id: str) -> Path:
    tag = qwen_asr_repo_tag(asr_repo_id)
    return PROJECT_ROOT / "src" / "checkpoints" / tag / f"pre_asr_cueqc_v13.{tag}.pt"


def validate_candidate(
    path: Path,
    *,
    asr_repo_id: str,
    split_checkpoint: Path,
    inner_checkpoint: Path,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("candidate checkpoint must be a mapping")
    expected = {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "arch": PRE_ASR_CUEQC_MODEL_ARCH,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"candidate checkpoint {key} must be {value!r}")
    config = dict(payload.get("model_config") or {})
    if int(config.get("num_classes") or 0) != 2:
        raise ValueError("candidate checkpoint must use the binary CueQC v13 head")
    for key, value in {
        "valid_prefix_temporal": True,
        "ptm_encoder_mode": "token_attention",
        "semantic_auxiliary": True,
        "late_fusion": True,
    }.items():
        if config.get(key) != value:
            raise ValueError(f"candidate model_config.{key} must be {value!r}")
    decision = dict(payload.get("decision_config") or {})
    if decision.get("decision_mode") != "argmax":
        raise ValueError("candidate decision_mode must be argmax")
    if "drop_threshold" in decision:
        raise ValueError("CueQC v13 candidate must not contain drop_threshold")
    if not bool(decision.get("model_only")):
        raise ValueError("CueQC v13 candidate must be model_only")
    for key in ("hard_keep_veto", "hard_drop_rule", "keep_veto"):
        if bool(decision.get(key)):
            raise ValueError(f"candidate decision_config.{key} must be false")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("asr_repo_id") != asr_repo_id:
        raise ValueError("candidate asr_repo_id mismatch")
    if list(metadata.get("training_labels") or ()) != ["drop", "keep"]:
        raise ValueError("candidate training_labels must be drop/keep")
    if list(metadata.get("excluded_training_labels") or ()) != ["unsure"]:
        raise ValueError("candidate must exclude unsure from training")
    if int(metadata.get("excluded_training_label_count", -1)) < 0:
        raise ValueError("candidate excluded_training_label_count is required")
    if list(metadata.get("ptm_pooling_schemas") or ()) != [ACTIVE_POOLING_SCHEMA]:
        raise ValueError("candidate PTM pooling schema mismatch")
    projection_digest = str(metadata.get("ptm_projection_digest") or "")
    if not projection_digest:
        raise ValueError("candidate ptm_projection_digest is required")
    split_sha = _sha256(split_checkpoint)
    inner_sha = _sha256(inner_checkpoint)
    if str(payload.get("semantic_split_weights_sha256") or "") != split_sha:
        raise ValueError("candidate Split checkpoint SHA mismatch")
    if str(payload.get("inner_edge_refiner_weights_sha256") or "") != inner_sha:
        raise ValueError("candidate Inner checkpoint SHA mismatch")
    if str(metadata.get("semantic_split_weights_sha256") or "") != split_sha:
        raise ValueError("candidate metadata Split checkpoint SHA mismatch")
    if str(metadata.get("inner_edge_refiner_weights_sha256") or "") != inner_sha:
        raise ValueError("candidate metadata Inner checkpoint SHA mismatch")
    return {
        **expected,
        "num_classes": 2,
        "decision_mode": "argmax",
        "training_labels": ["drop", "keep"],
        "excluded_training_labels": ["unsure"],
        "excluded_training_label_count": int(metadata["excluded_training_label_count"]),
        "ptm_projection_digest": projection_digest,
        "semantic_split_weights_sha256": split_sha,
        "inner_edge_refiner_weights_sha256": inner_sha,
    }


def validate_gates(model_gate: Mapping[str, Any], audit_gate: Mapping[str, Any]) -> dict[str, Any]:
    gate = dict(model_gate.get("gate") or {})
    if not bool(gate.get("passed")):
        raise ValueError("CueQC v13 binary model gate did not pass")
    if float(gate.get("min_keep_recall") or 0.0) != 0.95:
        raise ValueError("CueQC v13 keep recall gate must be exactly 0.95")
    if float(gate.get("min_drop_recall") or 0.0) != 0.95:
        raise ValueError("CueQC v13 drop recall gate must be exactly 0.95")
    if not bool(audit_gate.get("complete")) or not bool(audit_gate.get("promote_allowed")):
        raise ValueError("CueQC v13 false-drop manual gate is incomplete or failed")
    if int(audit_gate.get("true_semantic_keep_deletion_count", -1)) != 0:
        raise ValueError("CueQC v13 true semantic keep deletion count must be zero")
    if int(model_gate.get("all_false_drop_count", -1)) != int(
        audit_gate.get("target_manifest_count", -2)
    ):
        raise ValueError("model false-drop count and manual audit target count differ")
    return {
        "model_gate": dict(model_gate),
        "manual_false_drop_gate": dict(audit_gate),
    }


def promote(
    *,
    candidate_checkpoint: Path,
    output_checkpoint: Path,
    model_gate_path: Path,
    audit_gate_path: Path,
    split_checkpoint: Path,
    inner_checkpoint: Path,
    asr_repo_id: str,
    source_training_run: str,
    dry_run: bool,
) -> dict[str, Any]:
    contract = validate_candidate(
        candidate_checkpoint,
        asr_repo_id=asr_repo_id,
        split_checkpoint=split_checkpoint,
        inner_checkpoint=inner_checkpoint,
    )
    selected_validation = validate_gates(_json(model_gate_path), _json(audit_gate_path))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "dry_run": bool(dry_run),
        "candidate_checkpoint": str(candidate_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "candidate_contract": contract,
        "selected_validation": selected_validation,
    }
    if dry_run:
        summary["promoted"] = False
        return summary
    payload = promote_checkpoint(
        input_path=candidate_checkpoint,
        output_path=output_checkpoint,
        artifact_name=str(PRE_ASR_CUEQC_ARTIFACT["name"]),
        display_name=str(PRE_ASR_CUEQC_ARTIFACT["display_name"]),
        version=str(PRE_ASR_CUEQC_ARTIFACT["version"]),
        pipeline_stage=int(PRE_ASR_CUEQC_ARTIFACT["pipeline_stage"]),
        pipeline_role=str(PRE_ASR_CUEQC_ARTIFACT["pipeline_role"]),
        source_training_run=source_training_run,
        selected_validation=selected_validation,
        promotion_reason=(
            "CueQC v13 binary argmax val/test gates passed and every false-drop "
            "target was manually closed with zero true-speech deletion."
        ),
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )
    summary["promoted"] = True
    summary["artifact"] = dict(payload["metadata"]["artifact"])
    summary["decision_config"] = dict(payload.get("decision_config") or {})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--model-gate", required=True)
    parser.add_argument("--audit-gate", required=True)
    parser.add_argument("--split-checkpoint", required=True)
    parser.add_argument("--inner-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default="")
    parser.add_argument("--asr-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--source-training-run", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = (
        Path(args.output_checkpoint)
        if args.output_checkpoint
        else default_output_checkpoint(args.asr_repo_id)
    )
    summary = promote(
        candidate_checkpoint=Path(args.candidate_checkpoint),
        output_checkpoint=output,
        model_gate_path=Path(args.model_gate),
        audit_gate_path=Path(args.audit_gate),
        split_checkpoint=Path(args.split_checkpoint),
        inner_checkpoint=Path(args.inner_checkpoint),
        asr_repo_id=args.asr_repo_id,
        source_training_run=args.source_training_run,
        dry_run=args.dry_run,
    )
    if args.summary_output:
        Path(args.summary_output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
