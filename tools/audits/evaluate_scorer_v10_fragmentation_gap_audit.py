#!/usr/bin/env python3
"""Compile manual Scorer v10 fragmentation-gap verdicts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_scorer_v10_fragmentation_gap_audit_html import (
    VERDICT_SCHEMA,
)


ALLOWED_VERDICTS = {
    "same_asr_unit_keep_continuous",
    "separate_drop_nonsemantic",
    "separate_keep_both_speech",
    "cluster_not_speech_core",
    "unsure",
    "unreviewed",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *, audit_manifest: Path, manual_verdicts: Path, output: Path
) -> dict[str, Any]:
    audit_rows = _rows(audit_manifest)
    if not audit_rows:
        raise ValueError("Scorer v10 fragmentation audit manifest is empty")
    targets = {str(row["audit_id"]): row for row in audit_rows}
    if len(targets) != len(audit_rows):
        raise ValueError("duplicate audit_id in fragmentation audit manifest")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != VERDICT_SCHEMA:
            raise ValueError("invalid Scorer v10 fragmentation manual verdict schema")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in targets:
            raise ValueError(f"manual verdict has no fragmentation target: {audit_id}")
        if audit_id in verdicts:
            raise ValueError(f"duplicate fragmentation manual verdict: {audit_id}")
        verdict = str(row.get("verdict") or "unreviewed")
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"invalid fragmentation verdict: {verdict}")
        verdicts[audit_id] = row

    missing_ids = sorted(set(targets) - set(verdicts))
    verdict_counts = Counter(
        str(verdicts[audit_id].get("verdict") or "unreviewed")
        for audit_id in targets
        if audit_id in verdicts
    )
    unreviewed_count = int(verdict_counts["unreviewed"])
    same_asr_unit_count = int(verdict_counts["same_asr_unit_keep_continuous"])
    separate_drop_nonsemantic_count = int(
        verdict_counts["separate_drop_nonsemantic"]
    )
    separate_keep_both_speech_count = int(
        verdict_counts["separate_keep_both_speech"]
    )
    cluster_not_speech_core_count = int(
        verdict_counts["cluster_not_speech_core"]
    )
    unsure_count = int(verdict_counts["unsure"])
    review_complete = not missing_ids and unreviewed_count == 0
    model_behavior_pass = (
        review_complete and same_asr_unit_count == 0 and unsure_count == 0
    )
    canonical_repair_required = (
        separate_drop_nonsemantic_count > 0
        or separate_keep_both_speech_count > 0
        or cluster_not_speech_core_count > 0
    )
    canonical_consistency_pass = (
        review_complete and not canonical_repair_required and unsure_count == 0
    )
    fragmentation_gate_pass = model_behavior_pass and canonical_consistency_pass

    partition_verdict_counts: dict[str, dict[str, int]] = {}
    for audit_id, target in targets.items():
        verdict = str(
            verdicts.get(audit_id, {}).get("verdict") or "missing"
        )
        partition = str(target["partition"])
        partition_verdict_counts.setdefault(partition, {})[verdict] = (
            partition_verdict_counts.setdefault(partition, {}).get(verdict, 0) + 1
        )

    result = {
        "schema": "speech_scorer_v10_fragmentation_gap_manual_gate_v3",
        "audit_manifest": str(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "audit_item_count": len(targets),
        "submitted_verdict_count": len(verdicts),
        "missing_count": len(missing_ids),
        "missing_audit_ids": missing_ids,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "partition_verdict_counts": partition_verdict_counts,
        "manual_review_complete": review_complete,
        "model_wrong_fragmentation_count": same_asr_unit_count,
        "model_behavior_pass": model_behavior_pass,
        "canonical_repair_required": canonical_repair_required,
        "canonical_repair_reason_count": (
            separate_drop_nonsemantic_count
            + separate_keep_both_speech_count
            + cluster_not_speech_core_count
        ),
        "separate_drop_nonsemantic_count": separate_drop_nonsemantic_count,
        "separate_keep_both_speech_count": separate_keep_both_speech_count,
        "cluster_not_speech_core_count": cluster_not_speech_core_count,
        "canonical_consistency_pass": canonical_consistency_pass,
        "fragmentation_gate_pass": fragmentation_gate_pass,
        "checkpoint_promotion_authorized": False,
        "checkpoint_promotion_blockers": [
            *([] if review_complete else ["fragmentation_manual_review_incomplete"]),
            *(
                []
                if same_asr_unit_count == 0
                else ["same_asr_unit_wrongly_fragmented"]
            ),
            *([] if unsure_count == 0 else ["fragmentation_verdict_unsure"]),
            *(
                []
                if not canonical_repair_required
                else ["canonical_truth_run_repair_required"]
            ),
            *(
                []
                if cluster_not_speech_core_count == 0
                else ["canonical_fragment_cluster_not_speech_core_repair_required"]
            ),
            "separate_residual_zero_clipping_gate_required",
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
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_manifest=Path(args.audit_manifest),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
