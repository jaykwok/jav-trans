#!/usr/bin/env python3
"""Select only candidate-vs-baseline extra Scorer v10 speech-drop spans."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_scorer_v10_prediction_audit_html import (  # noqa: E402
    CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
    CHECKPOINT_AB_REMAINING_DROP_CONTRACT,
    FRAME_HOP_S,
    VERDICT_SCHEMA,
    audit_truth_drop_spans,
    truth_drop_spans,
)


SELECTION_SCHEMA = "scorer_v10_checkpoint_ab_extra_drop_selection_v2"
BACKGROUND_VERDICT = "canonical_should_be_background"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_mask(row: dict[str, Any]) -> np.ndarray:
    values = np.zeros(int(row["frame_count"]), dtype=bool)
    for span in truth_drop_spans(row):
        values[int(span["start_frame"]): int(span["end_frame"])] = True
    return values


def _spans(values: np.ndarray) -> list[dict[str, Any]]:
    padded = np.r_[False, np.asarray(values, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return [
        {
            "label": "truth_speech_model_background",
            "start_frame": int(start),
            "end_frame": int(end),
            "start_s": int(start) * FRAME_HOP_S,
            "end_s": int(end) * FRAME_HOP_S,
        }
        for start, end in changes
    ]


def _load_audited_background_masks(
    *, audit_manifest: Path, manual_verdicts: Path
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_rows: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    for row in _rows(audit_manifest):
        audit_id = str(row.get("audit_id") or "")
        source_id = str(row.get("source_id") or "")
        if not audit_id or audit_id in manifest_rows:
            raise ValueError("prior Scorer audit manifest requires unique audit_id values")
        if not source_id or source_id in source_ids:
            raise ValueError("prior Scorer audit manifest requires unique source_id values")
        manifest_rows[audit_id] = row
        source_ids.add(source_id)

    verdict_rows: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != VERDICT_SCHEMA:
            raise ValueError("invalid prior Scorer manual verdict schema")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in manifest_rows or audit_id in verdict_rows:
            raise ValueError(f"invalid or duplicate prior Scorer verdict: {audit_id}")
        target = manifest_rows[audit_id]
        for field in ("source_id", "partition", "row_role", "category"):
            if str(row.get(field) or "") != str(target.get(field) or ""):
                raise ValueError(
                    f"prior Scorer verdict {field} does not match manifest: {audit_id}"
                )
        verdict_rows[audit_id] = row

    masks: dict[str, np.ndarray] = {}
    accepted_rows: dict[str, dict[str, Any]] = {}
    accepted_audit_ids: list[str] = []
    accepted_frame_count = 0
    for audit_id, verdict in verdict_rows.items():
        if str(verdict.get("verdict") or "") != BACKGROUND_VERDICT:
            continue
        target = manifest_rows[audit_id]
        if target.get("audit_truth_drop_contract") != CHECKPOINT_AB_EXTRA_DROP_CONTRACT:
            raise ValueError(
                "prior background carryover requires an exact checkpoint A/B audit"
            )
        frame_count = int(target.get("frame_count") or 0)
        if frame_count <= 0:
            raise ValueError("prior Scorer audit row requires frame_count")
        mask = np.zeros(frame_count, dtype=bool)
        spans = audit_truth_drop_spans(target)
        for span in spans:
            start = int(span["start_frame"])
            end = int(span["end_frame"])
            if start < 0 or end <= start or end > frame_count:
                raise ValueError("prior Scorer audited background span is invalid")
            mask[start:end] = True
        source_id = str(target["source_id"])
        masks[source_id] = mask
        accepted_rows[source_id] = target
        accepted_audit_ids.append(audit_id)
        accepted_frame_count += int(np.sum(mask))

    provenance = {
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "manual_verdicts_sha256": _sha256(manual_verdicts),
        "accepted_verdict": BACKGROUND_VERDICT,
        "accepted_audit_ids": sorted(accepted_audit_ids),
        "accepted_source_count": len(masks),
        "accepted_frame_count": accepted_frame_count,
    }
    return masks, accepted_rows, provenance


def build_selection(
    *,
    baseline: Path,
    candidate: Path,
    output: Path,
    prior_audit_manifest: Path | None = None,
    prior_manual_verdicts: Path | None = None,
) -> dict[str, Any]:
    if (prior_audit_manifest is None) != (prior_manual_verdicts is None):
        raise ValueError(
            "prior audit manifest and manual verdicts must be provided together"
        )
    audited_background_masks: dict[str, np.ndarray] = {}
    audited_background_rows: dict[str, dict[str, Any]] = {}
    carryover_provenance: dict[str, Any] | None = None
    if prior_audit_manifest is not None and prior_manual_verdicts is not None:
        (
            audited_background_masks,
            audited_background_rows,
            carryover_provenance,
        ) = _load_audited_background_masks(
            audit_manifest=prior_audit_manifest,
            manual_verdicts=prior_manual_verdicts,
        )

    baseline_rows = {row["source_id"]: row for row in _rows(baseline)}
    candidate_rows = {row["source_id"]: row for row in _rows(candidate)}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Scorer v10 checkpoint A/B source identity mismatch")
    selected: list[dict[str, Any]] = []
    total_extra_frames = 0
    carried_background_frames = 0
    fully_carried_source_count = 0
    for source_id in sorted(candidate_rows):
        baseline_row = baseline_rows[source_id]
        candidate_row = candidate_rows[source_id]
        for key in ("frame_count", "audio", "truth_spans", "partition", "row_role"):
            if baseline_row.get(key) != candidate_row.get(key):
                raise ValueError(
                    f"Scorer v10 checkpoint A/B canonical field mismatch: {source_id} {key}"
                )
        baseline_drop = _drop_mask(baseline_row)
        candidate_drop = _drop_mask(candidate_row)
        extra = candidate_drop & ~baseline_drop
        if not np.any(extra):
            continue
        total_extra_frames += int(np.sum(extra))
        known_background = audited_background_masks.get(str(source_id))
        if known_background is not None:
            if len(known_background) != len(extra):
                raise ValueError(
                    f"prior Scorer audit frame_count mismatch: {source_id}"
                )
            prior_row = audited_background_rows[str(source_id)]
            for key in ("frame_count", "truth_spans", "partition", "row_role"):
                if prior_row.get(key) != candidate_row.get(key):
                    raise ValueError(
                        f"prior Scorer audit canonical field mismatch: {source_id} {key}"
                    )
            carried = extra & known_background
        else:
            carried = np.zeros_like(extra)
        remaining = extra & ~carried
        carried_count = int(np.sum(carried))
        carried_background_frames += carried_count
        if not np.any(remaining):
            fully_carried_source_count += 1
            continue
        category = str(candidate_row.get("category") or "speech_edge_or_partial")
        if category not in {"speech_deletion", "speech_edge_or_partial"}:
            category = "speech_edge_or_partial"
        selected.append(
            {
                **candidate_row,
                "category": category,
                "audit_truth_drop_contract": (
                    CHECKPOINT_AB_REMAINING_DROP_CONTRACT
                    if carryover_provenance is not None
                    else CHECKPOINT_AB_EXTRA_DROP_CONTRACT
                ),
                "audit_truth_drop_spans": _spans(remaining),
                "baseline_false_negative_frames": int(np.sum(baseline_drop)),
                "candidate_false_negative_frames": int(np.sum(candidate_drop)),
                "candidate_extra_false_negative_frames": int(np.sum(extra)),
                "candidate_extra_false_negative_frames_carried_as_background": carried_count,
                "candidate_extra_false_negative_frames_requiring_review": int(
                    np.sum(remaining)
                ),
                "carried_audited_background_spans": _spans(carried),
            }
        )
    if not selected:
        raise ValueError("Scorer v10 checkpoint A/B has no extra speech-drop spans")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "schema": SELECTION_SCHEMA,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "output": str(output),
        "selection_contract": (
            CHECKPOINT_AB_REMAINING_DROP_CONTRACT
            if carryover_provenance is not None
            else CHECKPOINT_AB_EXTRA_DROP_CONTRACT
        ),
        "source_count": len(selected),
        "extra_false_negative_frame_count": total_extra_frames,
        "total_extra_false_negative_frame_count_before_carryover": total_extra_frames,
        "carried_audited_background_frame_count": carried_background_frames,
        "remaining_false_negative_frame_count_requiring_review": sum(
            int(row["candidate_extra_false_negative_frames_requiring_review"])
            for row in selected
        ),
        "fully_carried_source_count": fully_carried_source_count,
        "partition_counts": dict(Counter(str(row["partition"]) for row in selected)),
        "category_counts": dict(Counter(str(row["category"]) for row in selected)),
        "carryover_provenance": carryover_provenance,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-audit-manifest")
    parser.add_argument("--prior-manual-verdicts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_selection(
                baseline=Path(args.baseline),
                candidate=Path(args.candidate),
                output=Path(args.output),
                prior_audit_manifest=(
                    Path(args.prior_audit_manifest)
                    if args.prior_audit_manifest
                    else None
                ),
                prior_manual_verdicts=(
                    Path(args.prior_manual_verdicts)
                    if args.prior_manual_verdicts
                    else None
                ),
            ),
            ensure_ascii=False,
        )
    )
