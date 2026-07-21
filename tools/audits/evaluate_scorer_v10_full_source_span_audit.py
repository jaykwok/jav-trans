#!/usr/bin/env python3
"""Validate complete full-source frame truth from the Scorer v10 repair page."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT
from tools.audits.generate_scorer_v10_full_source_span_audit_html import (
    FRAME_HOP_S,
    ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
)


GATE_SCHEMA = "speech_scorer_v10_full_source_span_manual_gate_v1"
DECISION_SCHEMA = "speech_scorer_v10_full_source_span_decision_v1"
LABELS = {"speech", "background", "unsure"}
COMPLETE_VERDICTS = {
    "complete_with_target_speech",
    "complete_with_unsure_only",
    "complete_all_background",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_spans(
    *, source_id: str, frame_count: int, spans: Any
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"reviewed full-source row has no spans: {source_id}")
    normalized: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    cursor = 0
    previous_label = ""
    for index, raw in enumerate(spans):
        if not isinstance(raw, dict):
            raise ValueError(f"full-source span is not an object: {source_id}")
        label = str(raw.get("label") or "")
        start = int(raw.get("start_frame") if raw.get("start_frame") is not None else -1)
        end = int(raw.get("end_frame") if raw.get("end_frame") is not None else -1)
        if label not in LABELS:
            raise ValueError(f"invalid full-source label: {source_id}:{label}")
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(
                f"full-source spans must be ordered and gap-free: {source_id}:{index}"
            )
        if label == previous_label:
            raise ValueError(
                f"adjacent identical full-source labels must be merged: "
                f"{source_id}:{index}"
            )
        expected_start_s = round(start * FRAME_HOP_S, 6)
        expected_end_s = round(end * FRAME_HOP_S, 6)
        if abs(float(raw.get("start_s") or 0.0) - expected_start_s) > 1e-6:
            raise ValueError(f"full-source start_s mismatch: {source_id}:{index}")
        if abs(float(raw.get("end_s") or 0.0) - expected_end_s) > 1e-6:
            raise ValueError(f"full-source end_s mismatch: {source_id}:{index}")
        normalized.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_s": expected_start_s,
                "end_s": expected_end_s,
            }
        )
        counts[label] += end - start
        cursor = end
        previous_label = label
    if cursor != frame_count:
        raise ValueError(f"full-source spans do not reach final frame: {source_id}")
    return normalized, counts


def evaluate(
    *, audit_manifest: Path, manual_verdicts: Path, output_dir: Path
) -> Path:
    manifest_rows = _rows(audit_manifest)
    manifest: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if row.get("schema") != ITEM_SCHEMA:
            raise ValueError("invalid full-source audit item schema")
        if (
            row.get("boundary_serialization_contract_id")
            != ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("full-source audit item has the wrong central contract")
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in manifest:
            raise ValueError("full-source audit manifest requires unique source ids")
        manifest[source_id] = row
    if not manifest:
        raise ValueError("full-source audit manifest is empty")

    verdict_rows = _rows(manual_verdicts)
    verdicts: dict[str, dict[str, Any]] = {}
    for row in verdict_rows:
        if row.get("schema") != MANUAL_VERDICT_SCHEMA:
            raise ValueError("invalid full-source manual verdict schema")
        source_id = str(row.get("source_id") or "")
        if source_id not in manifest or source_id in verdicts:
            raise ValueError(f"unknown or duplicate full-source verdict: {source_id}")
        verdicts[source_id] = row
    if set(verdicts) != set(manifest):
        raise ValueError("full-source manual verdict coverage is incomplete")

    decisions: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    unreviewed: list[str] = []
    for source_id, item in manifest.items():
        verdict = verdicts[source_id]
        for field in (
            "boundary_serialization_contract_id",
            "source_id",
            "partition",
            "frame_count",
        ):
            if str(verdict.get(field)) != str(item.get(field)):
                raise ValueError(f"full-source verdict {field} mismatch: {source_id}")
        if abs(float(verdict.get("frame_hop_s") or 0.0) - FRAME_HOP_S) > 1e-9:
            raise ValueError(f"full-source frame hop mismatch: {source_id}")
        verdict_name = str(verdict.get("verdict") or "")
        reviewed = verdict.get("reviewed_full_source") is True
        if not reviewed or verdict_name == "unreviewed":
            if reviewed or verdict_name != "unreviewed" or verdict.get("spans"):
                raise ValueError(f"inconsistent unreviewed full-source row: {source_id}")
            unreviewed.append(source_id)
            verdict_counts["unreviewed"] += 1
            continue
        if verdict_name not in COMPLETE_VERDICTS:
            raise ValueError(f"invalid complete full-source verdict: {source_id}")
        spans, counts = _validate_spans(
            source_id=source_id,
            frame_count=int(item["frame_count"]),
            spans=verdict.get("spans"),
        )
        expected_verdict = (
            "complete_with_target_speech"
            if counts["speech"]
            else (
                "complete_with_unsure_only"
                if counts["unsure"]
                else "complete_all_background"
            )
        )
        if verdict_name != expected_verdict:
            raise ValueError(f"full-source verdict/label mismatch: {source_id}")
        verdict_counts[verdict_name] += 1
        total_counts.update(counts)
        decisions.append(
            {
                "schema": DECISION_SCHEMA,
                "boundary_serialization_contract_id": (
                    ACOUSTIC_BINARY_V12_CONTRACT.contract_id
                ),
                "source_id": source_id,
                "partition": str(item["partition"]),
                "frame_count": int(item["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "verdict": verdict_name,
                "spans": spans,
                "model_output_used_as_truth": False,
                "asr_output_used_as_truth": False,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "decisions.jsonl"
    decision_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    gate = {
        "schema": GATE_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "manual_verdicts_sha256": _sha256(manual_verdicts),
        "source_count": len(manifest),
        "reviewed_source_count": len(decisions),
        "unreviewed_source_ids": sorted(unreviewed),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "label_frame_counts": dict(sorted(total_counts.items())),
        "all_reviewed_sources_have_gap_free_full_coverage": not unreviewed,
        "model_output_used_as_truth": False,
        "asr_output_used_as_truth": False,
        "unsure_training_label": -100,
        "manual_gate_passed": not unreviewed,
        "canonical_recompile_allowed": not unreviewed,
        "training_manifest_allowed": False,
        "decisions": str(decision_path),
        "decisions_sha256": _sha256(decision_path),
    }
    gate_path = output_dir / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        evaluate(
            audit_manifest=Path(args.audit_manifest),
            manual_verdicts=Path(args.manual_verdicts),
            output_dir=Path(args.output_dir),
        )
    )
