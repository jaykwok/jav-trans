#!/usr/bin/env python3
"""Compile manual labels for unresolved Scorer fragmentation atomic units."""
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

from tools.audits.generate_scorer_v10_fragment_atomic_repair_audit_html import (  # noqa: E402
    ATOMIC_ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    RELATION_SCHEMA,
    SUMMARY_SCHEMA,
)


RESULT_SCHEMA = "speech_scorer_v10_fragment_atomic_repair_gate_v1"
DECISION_SCHEMA = "speech_scorer_v10_fragment_atomic_repair_decision_v1"
LABELS = {"speech", "background", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *,
    audit_summary: Path,
    atomic_manifest: Path,
    relation_manifest: Path,
    manual_verdicts: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer fragment atomic audit summary schema")
    if str(summary.get("atomic_manifest") or "") != str(atomic_manifest):
        raise ValueError("atomic manifest does not match its audit summary")
    if str(summary.get("relation_manifest") or "") != str(relation_manifest):
        raise ValueError("relation manifest does not match its audit summary")

    targets: dict[str, dict[str, Any]] = {}
    for row in _rows(atomic_manifest):
        if row.get("schema") != ATOMIC_ITEM_SCHEMA:
            raise ValueError("invalid Scorer fragment atomic item schema")
        atomic_id = str(row.get("atomic_id") or "")
        if not atomic_id or atomic_id in targets:
            raise ValueError("atomic manifest requires unique atomic_id values")
        inferred = str(row.get("inferred_label") or "")
        if inferred and inferred not in LABELS:
            raise ValueError(f"invalid inferred atomic label: {inferred}")
        if bool(row.get("review_required")) == bool(inferred):
            raise ValueError("atomic review_required does not match inferred label")
        targets[atomic_id] = row
    if len(targets) != int(summary.get("atomic_unit_count") or -1):
        raise ValueError("atomic manifest count does not match its summary")

    relations: list[dict[str, Any]] = []
    for row in _rows(relation_manifest):
        if row.get("schema") != RELATION_SCHEMA:
            raise ValueError("invalid Scorer fragment atomic relation schema")
        if row.get("left_atomic_id") not in targets or row.get("right_atomic_id") not in targets:
            raise ValueError("atomic relation references an unknown unit")
        relations.append(row)
    if len(relations) != int(summary.get("relation_count") or -1):
        raise ValueError("atomic relation count does not match its summary")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != MANUAL_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer fragment atomic manual verdict schema")
        atomic_id = str(row.get("atomic_id") or "")
        if atomic_id not in targets or atomic_id in verdicts:
            raise ValueError(f"invalid or duplicate atomic verdict: {atomic_id}")
        if not targets[atomic_id].get("review_required"):
            raise ValueError(f"manual verdict targets an auto-resolved unit: {atomic_id}")
        verdict = str(row.get("verdict") or "unreviewed")
        if verdict not in LABELS | {"unreviewed"}:
            raise ValueError(f"invalid atomic verdict: {verdict}")
        verdicts[atomic_id] = row

    review_ids = {
        atomic_id
        for atomic_id, row in targets.items()
        if bool(row.get("review_required"))
    }
    missing_ids = sorted(review_ids - set(verdicts))
    unreviewed_ids = sorted(
        atomic_id
        for atomic_id, row in verdicts.items()
        if str(row.get("verdict") or "unreviewed") == "unreviewed"
    )
    complete = not missing_ids and not unreviewed_ids and set(verdicts) == review_ids

    decisions: dict[str, str] = {}
    decision_rows: list[dict[str, Any]] = []
    for atomic_id, target in targets.items():
        inferred = str(target.get("inferred_label") or "")
        manual = str(verdicts.get(atomic_id, {}).get("verdict") or "")
        label = inferred or ("" if manual == "unreviewed" else manual)
        if label:
            decisions[atomic_id] = label
        decision_rows.append(
            {
                **target,
                "schema": DECISION_SCHEMA,
                "label": label or "unreviewed",
                "label_source": (
                    "fragment_topology_constraint_v1"
                    if inferred
                    else "manual_fragment_atomic_repair_v1"
                ),
                "manual_updated_at": str(
                    verdicts.get(atomic_id, {}).get("updated_at") or ""
                ),
            }
        )

    violations: list[str] = []
    for relation in relations:
        left = decisions.get(str(relation["left_atomic_id"]), "")
        right = decisions.get(str(relation["right_atomic_id"]), "")
        if left == "speech" and right == "speech":
            violations.append(str(relation["relation_id"]))

    decisions_path = output.with_suffix(".decisions.jsonl")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decision_rows),
        encoding="utf-8",
    )
    label_counts = Counter(decisions.values())
    canonical_recompile_ready = complete and not violations
    result = {
        "schema": RESULT_SCHEMA,
        "audit_summary": str(audit_summary),
        "audit_summary_sha256": hashlib.sha256(audit_summary.read_bytes()).hexdigest(),
        "fragmentation_audit_manifest": str(
            summary.get("fragmentation_audit_manifest") or ""
        ),
        "fragmentation_audit_manifest_sha256": str(
            summary.get("fragmentation_audit_manifest_sha256") or ""
        ),
        "fragmentation_manual_verdicts": str(
            summary.get("fragmentation_manual_verdicts") or ""
        ),
        "fragmentation_manual_verdicts_sha256": str(
            summary.get("fragmentation_manual_verdicts_sha256") or ""
        ),
        "canonical_sources": str(summary.get("canonical_sources") or ""),
        "canonical_sources_sha256": str(
            summary.get("canonical_sources_sha256") or ""
        ),
        "atomic_unit_count": len(targets),
        "auto_resolved_count": len(targets) - len(review_ids),
        "manual_target_count": len(review_ids),
        "manual_verdict_count": len(verdicts),
        "missing_count": len(missing_ids),
        "missing_atomic_ids": missing_ids,
        "unreviewed_count": len(unreviewed_ids),
        "unreviewed_atomic_ids": unreviewed_ids,
        "complete": complete,
        "atomic_label_counts": dict(sorted(label_counts.items())),
        "unsure_count": int(label_counts["unsure"]),
        "relation_count": len(relations),
        "relation_violation_count": len(violations),
        "relation_violation_ids": violations,
        "decisions": str(decisions_path),
        "canonical_recompile_ready": canonical_recompile_ready,
        "training_manifest_allowed": False,
        "checkpoint_promotion_authorized": False,
        "promotion_blockers": [
            *([] if complete else ["fragment_atomic_manual_review_incomplete"]),
            *([] if not violations else ["fragment_atomic_relation_violation"]),
            "canonical_recompile_and_reaudit_required",
            "separate_residual_zero_clipping_gate_required",
        ],
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--atomic-manifest", required=True)
    parser.add_argument("--relation-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_summary=Path(args.audit_summary),
                atomic_manifest=Path(args.atomic_manifest),
                relation_manifest=Path(args.relation_manifest),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
