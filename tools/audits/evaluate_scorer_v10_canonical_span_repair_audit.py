#!/usr/bin/env python3
"""Evaluate exact-span repairs after a failed Scorer v10 canonical audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "speech_scorer_v10_canonical_span_repair_audit_summary_v1"
ITEM_SCHEMA = "speech_scorer_v10_canonical_span_repair_item_v1"
VERDICT_SCHEMA = "speech_scorer_v10_canonical_span_manual_verdict_v1"
RESULT_SCHEMA = "speech_scorer_v10_canonical_span_repair_gate_v1"
LABELS = {"speech", "background", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *, audit_manifest: Path, audit_summary: Path, manual_verdicts: Path, output: Path
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer v10 span-repair summary schema")
    targets: dict[str, dict[str, Any]] = {}
    for row in _rows(audit_manifest):
        if row.get("schema") != ITEM_SCHEMA:
            raise ValueError("invalid Scorer v10 span-repair item schema")
        span_id = str(row.get("span_id") or "")
        if not span_id or span_id in targets:
            raise ValueError("span-repair manifest requires unique span_id values")
        targets[span_id] = row
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("span-repair manifest does not match its summary")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != VERDICT_SCHEMA:
            raise ValueError("invalid Scorer v10 span-repair verdict schema")
        span_id = str(row.get("span_id") or "")
        if span_id not in targets or span_id in verdicts:
            raise ValueError(f"invalid or duplicate span verdict: {span_id}")
        verdict = str(row.get("verdict") or "")
        if verdict not in LABELS:
            raise ValueError(f"invalid span-repair label: {verdict!r}")
        verdicts[span_id] = row

    complete = set(verdicts) == set(targets)
    counts = Counter(str(row["verdict"]) for row in verdicts.values())
    changed = 0
    quarantined_background_ids = set(summary.get("quarantined_background_ids") or ())
    background_to_speech: set[str] = set()
    speech_to_background_core_ids: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for span_id, target in targets.items():
        verdict = str(verdicts.get(span_id, {}).get("verdict") or "unreviewed")
        changed += int(verdict != "unreviewed" and verdict != target["original_label"])
        if target["original_label"] == "background" and verdict != "background":
            background_id = str(target.get("background_id") or "")
            if background_id:
                quarantined_background_ids.add(background_id)
            if verdict == "speech" and background_id:
                background_to_speech.add(background_id)
        if target["original_label"] == "speech" and verdict == "background":
            core_id = str(target.get("core_id") or "")
            if core_id:
                speech_to_background_core_ids.add(core_id)
        decisions.append(
            {
                **target,
                "verdict": verdict,
                "note": str(verdicts.get(span_id, {}).get("note") or ""),
            }
        )

    decisions_path = output.with_suffix(".decisions.jsonl")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    result = {
        "schema": RESULT_SCHEMA,
        "canonical_sources": str(summary.get("canonical_sources") or ""),
        "canonical_sources_sha256": str(summary.get("canonical_sources_sha256") or ""),
        "target_count": len(targets),
        "verdict_count": len(verdicts),
        "verdict_counts": dict(sorted(counts.items())),
        "complete": complete,
        "changed_span_count": changed,
        "quarantined_background_ids": sorted(quarantined_background_ids),
        "background_assets_relabelled_speech": sorted(background_to_speech),
        "cores_relabelled_background": sorted(speech_to_background_core_ids),
        "decisions": str(decisions_path),
        "canonical_recompile_ready": complete,
        "training_manifest_allowed": False,
        "promotion_ready": False,
    }
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
