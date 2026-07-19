#!/usr/bin/env python3
"""Evaluate the saved Scorer v10 canonical-data listening gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "speech_scorer_v10_canonical_data_audit_summary_v1"
VERDICT_SCHEMA = "speech_scorer_v10_canonical_manual_verdict_v1"
GATE_SCHEMA = "speech_scorer_v10_canonical_manual_gate_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *, audit_manifest: Path, audit_summary: Path, manual_verdicts: Path, output: Path
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer v10 canonical audit summary schema")
    targets = {str(row["source_id"]): row for row in _rows(audit_manifest)}
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("Scorer v10 canonical audit manifest does not match its summary")
    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != VERDICT_SCHEMA:
            raise ValueError("invalid Scorer v10 canonical manual verdict schema")
        source_id = str(row.get("source_id") or "")
        if source_id not in targets:
            raise ValueError(f"manual verdict has no canonical audit target: {source_id}")
        if source_id in verdicts:
            raise ValueError(f"duplicate canonical manual verdict: {source_id}")
        verdicts[source_id] = row
    counts = Counter(
        str(verdicts.get(source_id, {}).get("verdict") or "unreviewed")
        for source_id in targets
    )
    complete = set(verdicts) == set(targets) and counts["unreviewed"] == 0
    manual_gate_pass = complete and counts["correct"] == len(targets)
    canonical_path = Path(str(summary.get("canonical_sources") or ""))
    canonical = {
        str(row["source_id"]): row for row in _rows(canonical_path)
    }
    quarantined_background_ids: set[str] = set()
    unsupported_risk_source_ids: list[str] = []
    for source_id in targets:
        verdict = str(verdicts.get(source_id, {}).get("verdict") or "unreviewed")
        if verdict in {"correct", "unreviewed"}:
            continue
        source = canonical.get(source_id)
        if (
            source is not None
            and source.get("row_role") == "all_background"
            and verdict == "contains_target_speech"
        ):
            background_id = str(source.get("background_id") or "")
            if background_id:
                quarantined_background_ids.add(background_id)
            else:
                unsupported_risk_source_ids.append(source_id)
        else:
            unsupported_risk_source_ids.append(source_id)
    canonical_recompile_ready = (
        complete
        and bool(quarantined_background_ids)
        and not unsupported_risk_source_ids
    )
    result = {
        "schema": GATE_SCHEMA,
        "canonical_sources": str(summary.get("canonical_sources") or ""),
        "canonical_sources_sha256": str(summary.get("canonical_sources_sha256") or ""),
        "audit_manifest": str(audit_manifest),
        "target_count": len(targets),
        "verdict_count": len(verdicts),
        "verdict_counts": dict(sorted(counts.items())),
        "complete": complete,
        "risk_count": len(targets) - counts["correct"],
        "quarantined_background_ids": sorted(quarantined_background_ids),
        "unsupported_risk_source_ids": sorted(unsupported_risk_source_ids),
        "canonical_recompile_ready": canonical_recompile_ready,
        "manual_gate_pass": manual_gate_pass,
        "training_manifest_allowed": manual_gate_pass,
        "promotion_ready": False,
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
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_manifest=Path(args.audit_manifest),
                audit_summary=Path(args.audit_summary),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
