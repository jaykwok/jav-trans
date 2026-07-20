#!/usr/bin/env python3
"""Audit unused identities available for an independent Scorer v10 evaluation suite."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _negative_type(row: dict[str, Any]) -> str:
    text = " ".join(map(str, row.get("omni_flags") or [])).lower()
    if "breath" in text:
        return "breathing"
    if "kiss" in text:
        return "kissing"
    if "moan" in text or "groan" in text:
        return "moaning"
    if "music" in text:
        return "music"
    if "noise" in text or "impact" in text:
        return "noise"
    return "non_speech"


def audit(*, canonical: Path, cores: Path, joint_labels: Path, output_dir: Path) -> dict[str, Any]:
    canonical_rows = _rows(canonical)
    used_cores = {str(core) for row in canonical_rows for core in row.get("core_ids") or []}
    used_negatives = {
        str(value)
        for row in canonical_rows
        for value in [*(row.get("background_source_ids") or []), row.get("background_id")]
        if value
    }
    unused_cores = [row for row in _rows(cores) if str(row["audio_id"]) not in used_cores]
    negatives: list[dict[str, Any]] = []
    for path in sorted(joint_labels.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in payload.get("pre_asr_labels") or []:
            if row.get("label") != "definite_drop":
                continue
            if str(row.get("candidate_id") or "") in used_negatives:
                continue
            if not Path(str(row.get("audio") or "")).is_file():
                continue
            negatives.append(row)
    type_counts = Counter(_negative_type(row) for row in negatives)
    required_types = ("breathing", "kissing", "moaning", "music", "noise", "non_speech")
    missing = [name for name in required_types if type_counts[name] < 3]
    output_dir.mkdir(parents=True, exist_ok=True)
    core_path = output_dir / "unused_cores.jsonl"
    negative_path = output_dir / "unused_negatives.jsonl"
    core_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unused_cores), encoding="utf-8")
    negative_path.write_text("".join(json.dumps({**row, "eval_type": _negative_type(row)}, ensure_ascii=False) + "\n" for row in negatives), encoding="utf-8")
    summary = {
        "schema": "speech_scorer_v10_independent_eval_inventory_v1",
        "diagnostic_only": True,
        "canonical_source_count": len(canonical_rows),
        "canonical_used_core_count": len(used_cores),
        "canonical_used_negative_count": len(used_negatives),
        "unused_core_count": len(unused_cores),
        "unused_negative_count": len(negatives),
        "unused_negative_type_counts": dict(type_counts),
        "minimum_required_per_type": 3,
        "missing_or_underfilled_types": missing,
        "complete_stratified_eval_ready": not missing,
        "partial_smoke_allowed_types": [name for name in required_types if type_counts[name] >= 1],
        "identity_overlap_allowed": False,
        "unused_cores": str(core_path),
        "unused_negatives": str(negative_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--joint-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(audit(canonical=Path(args.canonical), cores=Path(args.cores), joint_labels=Path(args.joint_labels), output_dir=Path(args.output_dir)), ensure_ascii=False))
