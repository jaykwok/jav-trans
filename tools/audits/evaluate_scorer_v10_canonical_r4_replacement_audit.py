#!/usr/bin/env python3
"""Evaluate the rendered-placement audit for Scorer v10 canonical r4."""
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

from tools.audits.generate_scorer_v10_canonical_r4_replacement_audit_html import (
    ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA,
)


RESULT_SCHEMA = "speech_scorer_v10_canonical_r4_replacement_manual_gate_v3"
VERDICTS = {
    "repair_speech_correct",
    "source_event_not_target",
    "not_target_after_render",
    "boundary_incomplete",
    "unsure",
    "unreviewed",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    *, audit_summary: Path, audit_manifest: Path, manual_verdicts: Path, output: Path
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 replacement audit summary")
    if Path(str(summary.get("audit_manifest") or "")).resolve() != audit_manifest.resolve():
        raise ValueError("Scorer canonical r4 replacement manifest mismatch")
    if _sha256(audit_manifest) != str(summary.get("audit_manifest_sha256") or ""):
        raise ValueError("Scorer canonical r4 replacement manifest changed")
    for path_key, sha_key in (
        ("canonical_summary", "canonical_summary_sha256"),
        ("canonical_sources", "canonical_sources_sha256"),
        ("repair_placements", "repair_placements_sha256"),
        ("repair_events", "repair_events_sha256"),
    ):
        path = Path(str(summary.get(path_key) or ""))
        if not path.is_file() or _sha256(path) != str(summary.get(sha_key) or ""):
            raise ValueError(f"Scorer canonical r4 replacement evidence changed: {path_key}")

    targets: dict[str, dict[str, Any]] = {}
    for row in _rows(audit_manifest):
        item_id = str(row.get("item_id") or "")
        if row.get("schema") != ITEM_SCHEMA or not item_id or item_id in targets:
            raise ValueError("invalid or duplicate Scorer canonical r4 replacement item")
        targets[item_id] = row
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("Scorer canonical r4 replacement item count mismatch")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != MANUAL_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer canonical r4 replacement verdict schema")
        item_id = str(row.get("item_id") or "")
        if item_id not in targets or item_id in verdicts:
            raise ValueError("invalid or duplicate Scorer canonical r4 replacement verdict")
        target = targets[item_id]
        for field in (
            "placement_id",
            "event_id",
            "source_id",
            "target_source_id",
            "partition",
            "role",
        ):
            if str(row.get(field) or "") != str(target.get(field) or ""):
                raise ValueError(f"Scorer canonical r4 replacement verdict {field} mismatch")
        if bool(row.get("core_registered")) != bool(target.get("core_registered")):
            raise ValueError("Scorer canonical r4 replacement core registration mismatch")
        verdict = str(row.get("verdict") or "unreviewed")
        if verdict not in VERDICTS:
            raise ValueError(f"invalid Scorer canonical r4 replacement verdict: {verdict}")
        verdicts[item_id] = row

    missing_ids = sorted(set(targets) - set(verdicts))
    unreviewed_ids = sorted(
        item_id
        for item_id, row in verdicts.items()
        if str(row.get("verdict") or "unreviewed") == "unreviewed"
    )
    unsure_ids = sorted(
        item_id for item_id, row in verdicts.items() if row.get("verdict") == "unsure"
    )
    rejected_ids = sorted(
        item_id
        for item_id, row in verdicts.items()
        if row.get("verdict") in {"not_target_after_render", "boundary_incomplete"}
    )
    items_by_event: dict[str, list[str]] = {}
    for item_id, target in targets.items():
        items_by_event.setdefault(str(target["event_id"]), []).append(item_id)
    source_event_repair_ids: list[str] = []
    source_event_repair_item_ids: list[str] = []
    for event_id, item_ids in items_by_event.items():
        values = {
            str(verdicts.get(item_id, {}).get("verdict") or "missing")
            for item_id in item_ids
        }
        if "source_event_not_target" not in values:
            continue
        if values != {"source_event_not_target"}:
            raise ValueError(
                f"source-event rejection must cover its complete placement group: {event_id}"
            )
        source_event_repair_ids.append(event_id)
        source_event_repair_item_ids.extend(item_ids)
    counts = Counter(str(row.get("verdict") or "unreviewed") for row in verdicts.values())
    complete = not missing_ids and not unreviewed_ids
    passed = (
        complete
        and not unsure_ids
        and not rejected_ids
        and not source_event_repair_ids
        and all(
            row.get("verdict") == "repair_speech_correct"
            for row in verdicts.values()
        )
    )
    result = {
        "schema": RESULT_SCHEMA,
        "audit_summary": str(audit_summary),
        "audit_summary_sha256": _sha256(audit_summary),
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "manual_verdicts_sha256": _sha256(manual_verdicts),
        "canonical_summary": str(summary["canonical_summary"]),
        "canonical_summary_sha256": str(summary["canonical_summary_sha256"]),
        "canonical_sources": str(summary["canonical_sources"]),
        "canonical_sources_sha256": str(summary["canonical_sources_sha256"]),
        "target_count": len(targets),
        "verdict_count": len(verdicts),
        "verdict_counts": dict(sorted(counts.items())),
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "unreviewed_count": len(unreviewed_ids),
        "unreviewed_ids": unreviewed_ids,
        "unsure_count": len(unsure_ids),
        "unsure_ids": unsure_ids,
        "repair_followup_count": len(rejected_ids),
        "repair_followup_ids": rejected_ids,
        "source_event_repair_count": len(source_event_repair_ids),
        "source_event_repair_ids": sorted(source_event_repair_ids),
        "source_event_repair_item_count": len(source_event_repair_item_ids),
        "source_event_repair_item_ids": sorted(source_event_repair_item_ids),
        "manual_review_complete": complete,
        "canonical_repair_pass": passed,
        "feature_cache_relabel_allowed": passed,
        "training_manifest_allowed": False,
        "checkpoint_promotion_authorized": False,
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
