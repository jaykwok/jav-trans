#!/usr/bin/env python3
"""Compile a strict source-level override for a Scorer all-background recheck."""
from __future__ import annotations

import argparse
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
    audit_truth_drop_spans,
)


RESULT_SCHEMA = "speech_scorer_v10_background_source_recheck_manual_gate_v1"
OVERRIDE_SCHEMA = "speech_scorer_v10_background_source_recheck_override_v1"
EXPECTED_CATEGORY = "background_false_keep"
EXPECTED_OLD_VERDICT = "canonical_contains_target_speech"
EXPECTED_NEW_VERDICT = "model_false_keep"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _prediction_identity(
    row: dict[str, Any], *, generated: bool
) -> dict[str, Any]:
    """Return the deterministic page row, excluding only copied audio location."""

    normalized = dict(row)
    if not generated:
        category = str(normalized.get("category") or "normal")
        if (
            category == "normal"
            and float(normalized.get("max_predicted_speech_run_s") or 0.0) > 8.0
        ):
            category = "long_residual"
        normalized["category"] = category
        normalized["audit_id"] = f"{category}:{normalized['source_id']}"
        normalized["truth_drop_spans"] = audit_truth_drop_spans(normalized)
    normalized.pop("audio", None)
    return normalized


def _unique_row(
    rows: list[dict[str, Any]], *, key: str, value: str, description: str
) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get(key) or "") == value]
    if len(matches) != 1:
        raise ValueError(f"{description} must contain exactly one matching row")
    return matches[0]


def evaluate(
    *,
    original_audit_manifest: Path,
    original_manual_verdicts: Path,
    recheck_summary: Path,
    recheck_audit_manifest: Path,
    recheck_manual_verdicts: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(recheck_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer background source recheck summary schema")
    if Path(str(summary.get("selection") or "")).resolve() != (
        original_audit_manifest.resolve()
    ):
        raise ValueError("source recheck is not bound to the original residual manifest")
    if Path(str(summary.get("audit_manifest") or "")).resolve() != (
        recheck_audit_manifest.resolve()
    ):
        raise ValueError("source recheck manifest does not match its summary")
    if int(summary.get("review_item_count") or -1) != 1:
        raise ValueError("source recheck must contain exactly one review item")
    source_filter = [str(value) for value in summary.get("source_filter") or ()]
    if len(source_filter) != 1:
        raise ValueError("source recheck requires exactly one source filter")
    source_id = source_filter[0]

    recheck_rows = _rows(recheck_audit_manifest)
    if len(recheck_rows) != 1:
        raise ValueError("source recheck manifest must contain exactly one row")
    recheck_row = recheck_rows[0]
    if str(recheck_row.get("source_id") or "") != source_id:
        raise ValueError("source recheck row does not match its source filter")
    audit_id = str(recheck_row.get("audit_id") or "")
    if not audit_id:
        raise ValueError("source recheck row is missing audit_id")

    original_row = _unique_row(
        _rows(original_audit_manifest),
        key="audit_id",
        value=audit_id,
        description="original residual manifest",
    )
    if _prediction_identity(original_row, generated=False) != _prediction_identity(
        recheck_row, generated=True
    ):
        raise ValueError("source recheck prediction evidence differs from the original row")
    for field, expected in (
        ("source_id", source_id),
        ("category", EXPECTED_CATEGORY),
        ("row_role", "all_background"),
    ):
        if str(original_row.get(field) or "") != expected:
            raise ValueError(f"source recheck has invalid {field}")

    original_verdict = _unique_row(
        _rows(original_manual_verdicts),
        key="audit_id",
        value=audit_id,
        description="original residual verdicts",
    )
    recheck_verdict_rows = _rows(recheck_manual_verdicts)
    if len(recheck_verdict_rows) != 1:
        raise ValueError("source recheck verdicts must contain exactly one row")
    recheck_verdict = recheck_verdict_rows[0]
    for verdict, description in (
        (original_verdict, "original residual"),
        (recheck_verdict, "source recheck"),
    ):
        if verdict.get("schema") != VERDICT_SCHEMA:
            raise ValueError(f"invalid {description} verdict schema")
        for field in ("audit_id", "source_id", "partition", "row_role", "category"):
            if str(verdict.get(field) or "") != str(original_row.get(field) or ""):
                raise ValueError(f"{description} verdict {field} mismatch")
    if str(original_verdict.get("verdict") or "") != EXPECTED_OLD_VERDICT:
        raise ValueError("source recheck can only withdraw canonical_contains_target_speech")
    if str(recheck_verdict.get("verdict") or "") != EXPECTED_NEW_VERDICT:
        raise ValueError("source recheck must resolve to model_false_keep")

    override = {
        "schema": OVERRIDE_SCHEMA,
        "source_id": source_id,
        "audit_id": audit_id,
        "partition": str(original_row["partition"]),
        "row_role": "all_background",
        "category": EXPECTED_CATEGORY,
        "original_verdict": EXPECTED_OLD_VERDICT,
        "replacement_verdict": EXPECTED_NEW_VERDICT,
        "override_action": "withdraw_canonical_contains_target_speech",
        "canonical_action": "retain_all_background",
        "exclude_from_background_speech_repair": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    overrides_path = output.with_suffix(".overrides.jsonl")
    overrides_path.write_text(
        json.dumps(override, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evidence = {
        "original_prediction_audit_manifest": _evidence(original_audit_manifest),
        "original_prediction_manual_verdicts": _evidence(original_manual_verdicts),
        "recheck_summary": _evidence(recheck_summary),
        "recheck_audit_manifest": _evidence(recheck_audit_manifest),
        "recheck_manual_verdicts": _evidence(recheck_manual_verdicts),
        "overrides": _evidence(overrides_path),
    }
    result = {
        "schema": RESULT_SCHEMA,
        "source_id": source_id,
        "audit_id": audit_id,
        "original_verdict": EXPECTED_OLD_VERDICT,
        "replacement_verdict": EXPECTED_NEW_VERDICT,
        "override_count": 1,
        "overridden_source_ids": [source_id],
        "overrides": str(overrides_path),
        "evidence": evidence,
        "manual_review_complete": True,
        "canonical_override_ready": True,
        "background_speech_repair_exclusion_allowed": True,
        "training_manifest_allowed": False,
        "checkpoint_promotion_authorized": False,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-audit-manifest", required=True)
    parser.add_argument("--original-manual-verdicts", required=True)
    parser.add_argument("--recheck-summary", required=True)
    parser.add_argument("--recheck-audit-manifest", required=True)
    parser.add_argument("--recheck-manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                original_audit_manifest=Path(args.original_audit_manifest),
                original_manual_verdicts=Path(args.original_manual_verdicts),
                recheck_summary=Path(args.recheck_summary),
                recheck_audit_manifest=Path(args.recheck_audit_manifest),
                recheck_manual_verdicts=Path(args.recheck_manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
