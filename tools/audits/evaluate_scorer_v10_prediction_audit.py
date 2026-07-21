#!/usr/bin/env python3
"""Compile Scorer v10 residual verdicts into a strict manual gate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_scorer_v10_prediction_audit_html import (  # noqa: E402
    SUMMARY_SCHEMA,
    VERDICT_SCHEMA,
)


RESULT_SCHEMA = "speech_scorer_v10_prediction_manual_gate_v3"
CATEGORY_VERDICTS = {
    "speech_deletion": {
        "true_speech_deleted",
        "canonical_should_be_background",
        "unsure",
        "unreviewed",
    },
    "speech_edge_or_partial": {
        "true_speech_clipped",
        "same_asr_unit_fragmented",
        "canonical_should_be_background",
        "unsure",
        "unreviewed",
    },
    "long_residual": {
        "acceptable_long_residual",
        "missed_background_or_gap",
        "true_speech_edge_clipped",
        "unsure",
        "unreviewed",
    },
    "background_false_keep": {
        "model_false_keep",
        "canonical_contains_target_speech",
        "unsure",
        "unreviewed",
    },
}
ZERO_CLIPPING_VERDICTS = {
    "true_speech_deleted",
    "true_speech_clipped",
    "true_speech_edge_clipped",
}
BACKGROUND_BEHAVIOR_VERDICTS = {
    "model_false_keep",
    "missed_background_or_gap",
}
WORKFLOW_CONTINUITY_VERDICTS = {"same_asr_unit_fragmented"}
CANONICAL_REPAIR_VERDICTS = {
    "canonical_should_be_background",
    "canonical_contains_target_speech",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    *,
    audit_summary: Path,
    audit_manifest: Path,
    manual_verdicts: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer v10 prediction audit summary schema")
    summary_manifest = Path(str(summary.get("audit_manifest") or ""))
    if summary_manifest.resolve() != audit_manifest.resolve():
        raise ValueError("prediction audit manifest does not match its summary")

    targets: dict[str, dict[str, Any]] = {}
    for row in _rows(audit_manifest):
        audit_id = str(row.get("audit_id") or "")
        category = str(row.get("category") or "")
        if not audit_id or audit_id in targets:
            raise ValueError("prediction audit manifest requires unique audit_id values")
        if category not in CATEGORY_VERDICTS:
            raise ValueError(f"invalid prediction audit category: {category}")
        targets[audit_id] = row
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("prediction audit manifest count does not match its summary")
    actual_categories = Counter(str(row["category"]) for row in targets.values())
    expected_categories = {
        str(key): int(value)
        for key, value in dict(summary.get("category_counts") or {}).items()
    }
    if dict(actual_categories) != expected_categories:
        raise ValueError("prediction audit category counts do not match its summary")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != VERDICT_SCHEMA:
            raise ValueError("invalid Scorer v10 prediction manual verdict schema")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in targets or audit_id in verdicts:
            raise ValueError(f"invalid or duplicate prediction verdict: {audit_id}")
        target = targets[audit_id]
        for field in ("source_id", "partition", "row_role", "category"):
            if str(row.get(field) or "") != str(target.get(field) or ""):
                raise ValueError(f"prediction verdict {field} does not match target: {audit_id}")
        verdict = str(row.get("verdict") or "unreviewed")
        category = str(target["category"])
        if verdict not in CATEGORY_VERDICTS[category]:
            raise ValueError(f"invalid {category} verdict: {verdict}")
        verdicts[audit_id] = row

    missing_ids = sorted(set(targets) - set(verdicts))
    unreviewed_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "unreviewed") == "unreviewed"
    )
    unsure_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "") == "unsure"
    )
    zero_clipping_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "") in ZERO_CLIPPING_VERDICTS
    )
    background_behavior_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "") in BACKGROUND_BEHAVIOR_VERDICTS
    )
    workflow_continuity_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "") in WORKFLOW_CONTINUITY_VERDICTS
    )
    canonical_repair_ids = sorted(
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "") in CANONICAL_REPAIR_VERDICTS
    )
    verdict_counts = Counter(
        str(row.get("verdict") or "unreviewed") for row in verdicts.values()
    )
    category_verdict_counts: dict[str, dict[str, int]] = {}
    for audit_id, target in targets.items():
        category = str(target["category"])
        verdict = str(verdicts.get(audit_id, {}).get("verdict") or "missing")
        category_verdict_counts.setdefault(category, {})[verdict] = (
            category_verdict_counts.setdefault(category, {}).get(verdict, 0) + 1
        )

    review_complete = not missing_ids and not unreviewed_ids
    zero_clipping_pass = review_complete and not zero_clipping_ids and not unsure_ids
    background_behavior_pass = (
        review_complete and not background_behavior_ids and not unsure_ids
    )
    workflow_continuity_pass = (
        review_complete and not workflow_continuity_ids and not unsure_ids
    )
    canonical_consistency_pass = (
        review_complete and not canonical_repair_ids and not unsure_ids
    )
    residual_gate_pass = (
        zero_clipping_pass
        and background_behavior_pass
        and workflow_continuity_pass
        and canonical_consistency_pass
    )
    result = {
        "schema": RESULT_SCHEMA,
        "audit_summary": str(audit_summary),
        "audit_summary_sha256": _sha256(audit_summary),
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "manual_verdicts_sha256": _sha256(manual_verdicts),
        "audit_item_count": len(targets),
        "manual_verdict_count": len(verdicts),
        "missing_count": len(missing_ids),
        "missing_audit_ids": missing_ids,
        "unreviewed_count": len(unreviewed_ids),
        "unreviewed_audit_ids": unreviewed_ids,
        "unsure_count": len(unsure_ids),
        "unsure_audit_ids": unsure_ids,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "category_verdict_counts": category_verdict_counts,
        "manual_review_complete": review_complete,
        "zero_clipping_violation_count": len(zero_clipping_ids),
        "zero_clipping_violation_ids": zero_clipping_ids,
        "zero_clipping_pass": zero_clipping_pass,
        "background_behavior_issue_count": len(background_behavior_ids),
        "background_behavior_issue_ids": background_behavior_ids,
        "background_behavior_pass": background_behavior_pass,
        "workflow_continuity_issue_count": len(workflow_continuity_ids),
        "workflow_continuity_issue_ids": workflow_continuity_ids,
        "workflow_continuity_pass": workflow_continuity_pass,
        "canonical_repair_required": bool(canonical_repair_ids),
        "canonical_repair_count": len(canonical_repair_ids),
        "canonical_repair_ids": canonical_repair_ids,
        "canonical_consistency_pass": canonical_consistency_pass,
        "residual_gate_pass": residual_gate_pass,
        "checkpoint_promotion_authorized": False,
        "checkpoint_promotion_blockers": [
            *([] if review_complete else ["prediction_manual_review_incomplete"]),
            *([] if not unsure_ids else ["prediction_manual_verdict_unsure"]),
            *([] if not zero_clipping_ids else ["manual_zero_clipping_violation"]),
            *(
                []
                if not background_behavior_ids
                else ["manual_background_behavior_issue"]
            ),
            *(
                []
                if not workflow_continuity_ids
                else ["manual_workflow_continuity_issue"]
            ),
            *([] if not canonical_repair_ids else ["canonical_repair_and_rescore_required"]),
            "separate_fragmentation_gate_required",
            "workflow_binding_gate_required",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_summary=Path(args.audit_summary),
                audit_manifest=Path(args.audit_manifest),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
