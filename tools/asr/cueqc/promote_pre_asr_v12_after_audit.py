#!/usr/bin/env python3
"""Promote a Pre-ASR CueQC v12 checkpoint only after D7 gates pass."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from asr.pre_asr_cueqc import PRE_ASR_CUEQC_ARTIFACT  # noqa: E402
from tools.workflows.promote_torch_checkpoint import promote_checkpoint  # noqa: E402

SUMMARY_SCHEMA = "pre_asr_cueqc_v12_audit_gated_promotion_summary_v1"
DEFAULT_CANDIDATE = (
    "agents/temp/20260709_121935_pre-asr-v12-train-v3-labels/"
    "pre_asr_cueqc_v12.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
)
DEFAULT_MODEL_GATE = "agents/temp/20260709_122753_pre-asr-v12-train-v3-labels-gate/summary.json"
DEFAULT_AUDIT_GATE = (
    "agents/audits/20260709_122906_pre-asr-v12-v3-train-long-false-drop-audit/"
    "gate_summary.json"
)
V12_SCHEMA = "cueqc_pre_asr_semantic_chunk_v12_binary"
V12_ARCH = "cueqc_pre_asr_semantic_chunk_v12"
V12_FEATURE_SCHEMA = "pre_asr_cueqc_features_v9"
V12_RUNTIME_ADAPTER = "pre_asr_semantic_chunk_sequence_v4"
ACTIVE_POOLING_SCHEMA = "pre_asr_chunk_pooled_ptm_v1"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_output_checkpoint(asr_repo_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "src"
        / "asr"
        / "checkpoints"
        / f"pre_asr_cueqc_v12.{qwen_asr_repo_tag(asr_repo_id)}.pt"
    )


def validate_model_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    gate = summary.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("model gate summary missing gate object")
    v12 = summary.get("v12")
    if not isinstance(v12, Mapping):
        raise ValueError("model gate summary missing v12 metrics")
    failures: list[str] = []
    if not bool(gate.get("drop_recall_pass")):
        failures.append("drop_recall_pass=false")
    if not bool(gate.get("keep_recall_pass")):
        failures.append("keep_recall_pass=false")
    if float(v12.get("drop_recall", 0.0)) < float(gate.get("min_drop_recall", 0.98)):
        failures.append("v12.drop_recall below min_drop_recall")
    if failures:
        raise ValueError(f"model gate failed: {', '.join(failures)}")
    return {
        "v12_threshold": float(summary.get("v12_threshold", 0.5)),
        "drop_recall": float(v12.get("drop_recall", 0.0)),
        "keep_recall": float(v12.get("semantic_keep_recall", 0.0)),
        "false_drop_count": int(v12.get("false_drop_count", 0)),
        "false_keep_count": int(v12.get("false_keep_count", 0)),
        "long_false_drop_count": int(gate.get("long_false_drop_count", 0)),
        "long_false_drop_min_s": float(gate.get("long_false_drop_min_s", 0.8)),
    }


def validate_audit_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not bool(summary.get("complete")):
        failures.append("complete=false")
    if not bool(summary.get("promote_allowed")):
        failures.append("promote_allowed=false")
    if int(summary.get("true_semantic_keep_deletion_count", 0)) != 0:
        failures.append("true_semantic_keep_deletion_count>0")
    if failures:
        raise ValueError(f"audit gate failed: {', '.join(failures)}")
    return {
        "target_manifest_count": int(summary.get("target_manifest_count", 0)),
        "reviewed_target_count": int(summary.get("reviewed_target_count", 0)),
        "manual_verdict_counts": dict(summary.get("manual_verdict_counts") or {}),
        "uncertain_count": int(summary.get("uncertain_count", 0)),
    }


def validate_candidate_checkpoint(path: Path, *, asr_repo_id: str) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate checkpoint must be a mapping: {path}")
    schema = str(payload.get("schema") or "").strip()
    if schema != V12_SCHEMA:
        raise ValueError(f"candidate checkpoint schema must be {V12_SCHEMA!r}, got {schema!r}")
    arch = str(payload.get("arch") or "").strip()
    if arch != V12_ARCH:
        raise ValueError(f"candidate checkpoint arch must be {V12_ARCH!r}, got {arch!r}")
    if str(payload.get("feature_schema") or "") != V12_FEATURE_SCHEMA:
        raise ValueError("candidate checkpoint feature_schema is not active v9")
    if str(payload.get("runtime_adapter") or "") != V12_RUNTIME_ADAPTER:
        raise ValueError("candidate checkpoint runtime_adapter is not active v4")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate checkpoint missing metadata")
    actual_repo = str(metadata.get("asr_repo_id") or "").strip()
    if actual_repo != asr_repo_id:
        raise ValueError(
            f"candidate checkpoint asr_repo_id={actual_repo!r} does not match {asr_repo_id!r}"
        )
    split_sha = str(payload.get("semantic_split_weights_sha256") or metadata.get("semantic_split_weights_sha256") or "")
    if not split_sha:
        raise ValueError("candidate checkpoint missing semantic_split_weights_sha256")
    model_config = dict(payload.get("model_config") or {})
    required_model_config = {
        "valid_prefix_temporal": True,
        "ptm_encoder_mode": "token_attention",
        "semantic_auxiliary": True,
        "late_fusion": True,
    }
    for key, expected in required_model_config.items():
        if model_config.get(key) != expected:
            raise ValueError(f"candidate checkpoint model_config.{key} must be {expected!r}")
    pooling_schemas = [str(item) for item in metadata.get("ptm_pooling_schemas") or ()]
    if pooling_schemas != [ACTIVE_POOLING_SCHEMA]:
        raise ValueError(
            f"candidate checkpoint ptm_pooling_schemas must be {[ACTIVE_POOLING_SCHEMA]!r}"
        )
    decision = dict(payload.get("decision_config") or {})
    if not bool(decision.get("model_only")):
        raise ValueError("candidate checkpoint must be model_only")
    for key in ("hard_keep_veto", "hard_drop_rule", "keep_veto"):
        if bool(decision.get(key)):
            raise ValueError(f"candidate checkpoint decision_config.{key} must be false")
    return {
        "schema": schema,
        "arch": arch,
        "asr_repo_id": actual_repo,
        "semantic_split_weights_sha256": split_sha,
        "model_config": required_model_config,
        "ptm_pooling_schemas": pooling_schemas,
    }


def build_selected_validation(
    *,
    model_gate_path: Path,
    model_gate: Mapping[str, Any],
    audit_gate_path: Path,
    audit_gate: Mapping[str, Any],
) -> dict[str, Any]:
    model_validation = validate_model_gate(model_gate)
    audit_validation = validate_audit_gate(audit_gate)
    if float(model_validation["long_false_drop_min_s"]) != 0.0:
        raise ValueError("promotion requires an all-false-drop audit with min duration 0.0")
    if int(model_validation["long_false_drop_count"]) != int(
        audit_validation["target_manifest_count"]
    ):
        raise ValueError("model and manual audit false-drop target counts do not match")
    return {
        "schema": "pre_asr_cueqc_v12_selected_validation_v1",
        "model_gate": {
            "path": repo_rel(model_gate_path),
            **model_validation,
        },
        "manual_false_drop_audit": {
            "path": repo_rel(audit_gate_path),
            **audit_validation,
        },
    }


def promote_after_audit(
    *,
    candidate_checkpoint: Path,
    output_checkpoint: Path,
    model_gate_path: Path,
    audit_gate_path: Path,
    asr_repo_id: str,
    source_training_run: str,
    dry_run: bool,
) -> dict[str, Any]:
    candidate_contract = validate_candidate_checkpoint(candidate_checkpoint, asr_repo_id=asr_repo_id)
    model_gate = read_json(model_gate_path)
    audit_gate = read_json(audit_gate_path)
    selected_validation = build_selected_validation(
        model_gate_path=model_gate_path,
        model_gate=model_gate,
        audit_gate_path=audit_gate_path,
        audit_gate=audit_gate,
    )
    threshold = float(selected_validation["model_gate"]["v12_threshold"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "dry_run": bool(dry_run),
        "candidate_checkpoint": repo_rel(candidate_checkpoint),
        "output_checkpoint": repo_rel(output_checkpoint),
        "asr_repo_id": asr_repo_id,
        "source_training_run": source_training_run,
        "candidate_contract": candidate_contract,
        "selected_validation": selected_validation,
        "drop_threshold": threshold,
        "registry_update_required": {
            "file": "src/asr/backends/qwen.py",
            "mapping": "DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO",
            "repo_id": asr_repo_id,
            "value": repo_rel(output_checkpoint),
        },
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
        drop_threshold=threshold,
        promotion_reason=(
            "D7 v12 model gate passed and manual all-false-drop audit found no "
            "confirmed semantic keep deletion."
        ),
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )
    summary["promoted"] = True
    summary["artifact"] = dict(payload["metadata"]["artifact"])
    summary["decision_config"] = dict(payload.get("decision_config") or {})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-checkpoint", default="")
    parser.add_argument("--model-gate", default=DEFAULT_MODEL_GATE)
    parser.add_argument("--audit-gate", default=DEFAULT_AUDIT_GATE)
    parser.add_argument("--asr-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--source-training-run", default="agents/temp/20260709_121935_pre-asr-v12-train-v3-labels")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_checkpoint = (
        project_path(args.output_checkpoint)
        if args.output_checkpoint
        else default_output_checkpoint(args.asr_repo_id)
    )
    summary = promote_after_audit(
        candidate_checkpoint=project_path(args.candidate_checkpoint),
        output_checkpoint=output_checkpoint,
        model_gate_path=project_path(args.model_gate),
        audit_gate_path=project_path(args.audit_gate),
        asr_repo_id=str(args.asr_repo_id),
        source_training_run=str(args.source_training_run),
        dry_run=bool(args.dry_run),
    )
    if args.summary_output:
        write_json(project_path(args.summary_output), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
