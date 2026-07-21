#!/usr/bin/env python3
"""Apply audited Scorer v10 r4 replacement verdicts to canonical data."""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.evaluate_scorer_v10_canonical_r4_replacement_audit import (  # noqa: E402
    RESULT_SCHEMA as REPLACEMENT_GATE_SCHEMA,
)
from tools.audits.generate_scorer_v10_canonical_r4_replacement_audit_html import (  # noqa: E402
    ITEM_SCHEMA as REPLACEMENT_ITEM_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA as REPLACEMENT_SUMMARY_SCHEMA,
)
from tools.boundary.ja.apply_speech_island_scorer_v10_canonical_r4_repairs import (  # noqa: E402
    CHANGED_SPAN_SCHEMA,
    PLACEMENT_SCHEMA,
    SUMMARY_SCHEMA as R4_SUMMARY_SCHEMA,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_r5_summary_v1"
REJECTED_PLACEMENT_SCHEMA = (
    "speech_scorer_v10_canonical_r5_rejected_repair_placement_v1"
)
ROLLBACK_SPAN_SCHEMA = "speech_scorer_v10_canonical_r5_rollback_span_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_path(left: object, right: Path) -> bool:
    return Path(str(left or "")).resolve() == right.resolve()


def _require_sha(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected or ""):
        raise ValueError(f"Scorer canonical r5 {label} changed: {path}")


def _frame_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        labels = canonical_frame_labels(row, frame_hop_s=FRAME_HOP_S)
        counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
    return counts


def _covering_span(
    spans: list[dict[str, Any]], *, start: int, end: int
) -> dict[str, Any]:
    matches = [
        span
        for span in spans
        if start >= int(span["start_sample"]) and end <= int(span["end_sample"])
    ]
    if len(matches) != 1:
        raise ValueError("Scorer canonical r5 span topology is invalid")
    return matches[0]


def _merge_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for span in spans:
        if merged:
            previous = merged[-1]
            previous_payload = {
                key: value
                for key, value in previous.items()
                if key not in {"start_sample", "end_sample"}
            }
            current_payload = {
                key: value
                for key, value in span.items()
                if key not in {"start_sample", "end_sample"}
            }
            if (
                int(previous["end_sample"]) == int(span["start_sample"])
                and previous_payload == current_payload
            ):
                previous["end_sample"] = int(span["end_sample"])
                continue
        merged.append(copy.deepcopy(span))
    return merged


def _restore_rejected_ranges(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    operations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not operations:
        return copy.deepcopy(current), []
    if (
        str(current["source_id"]) != str(baseline["source_id"])
        or int(current["sample_count"]) != int(baseline["sample_count"])
        or str(current["partition"]) != str(baseline["partition"])
        or str(current["audio"]) != str(baseline["audio"])
    ):
        raise ValueError("Scorer canonical r5 baseline identity changed")

    ordered = sorted(
        operations, key=lambda item: (int(item["start_sample"]), int(item["end_sample"]))
    )
    for left, right in zip(ordered, ordered[1:]):
        if int(right["start_sample"]) < int(left["end_sample"]):
            raise ValueError("Scorer canonical r5 rollback ranges overlap")

    boundaries = {
        int(span[side])
        for row in (current, baseline)
        for span in row["canonical_spans"]
        for side in ("start_sample", "end_sample")
    }
    rollback_rows: list[dict[str, Any]] = []
    for operation in ordered:
        start = int(operation["start_sample"])
        end = int(operation["end_sample"])
        if start < 0 or end <= start or end > int(current["sample_count"]):
            raise ValueError("Scorer canonical r5 rollback range is invalid")
        boundaries.update((start, end))
        cursor = start
        while cursor < end:
            current_span = _covering_span(
                current["canonical_spans"], start=cursor, end=cursor + 1
            )
            piece_end = min(end, int(current_span["end_sample"]))
            if (
                str(current_span.get("label") or "") != "speech"
                or str(current_span.get("repair_placement_id") or "")
                != str(operation["placement_id"])
            ):
                raise ValueError(
                    "Scorer canonical r5 rejected range is not its proposed speech repair"
                )
            baseline_span = _covering_span(
                baseline["canonical_spans"], start=cursor, end=piece_end
            )
            if str(baseline_span.get("label") or "") != "background":
                raise ValueError(
                    "Scorer canonical r5 refuses rollback over baseline target speech"
                )
            cursor = piece_end
        rollback_rows.append(
            {
                "schema": ROLLBACK_SPAN_SCHEMA,
                "source_id": str(current["source_id"]),
                "partition": str(current["partition"]),
                "start_sample": start,
                "end_sample": end,
                "placement_id": str(operation["placement_id"]),
                "event_id": str(operation["event_id"]),
                "verdict": str(operation["verdict"]),
            }
        )

    pieces: list[dict[str, Any]] = []
    points = sorted(boundaries)
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        covering = [
            operation
            for operation in ordered
            if start >= int(operation["start_sample"])
            and end <= int(operation["end_sample"])
        ]
        if len(covering) > 1:
            raise ValueError("Scorer canonical r5 rollback coverage is ambiguous")
        source = baseline if covering else current
        piece = copy.deepcopy(
            _covering_span(source["canonical_spans"], start=start, end=end)
        )
        piece["start_sample"] = start
        piece["end_sample"] = end
        pieces.append(piece)

    corrected = copy.deepcopy(current)
    corrected["canonical_spans"] = _merge_spans(pieces)
    present_core_ids = {
        str(span.get("core_id") or "")
        for span in corrected["canonical_spans"]
        if str(span.get("label") or "") == "speech" and span.get("core_id")
    }
    ordered_core_ids = [
        *[str(value) for value in current.get("core_ids") or ()],
        *[str(value) for value in baseline.get("core_ids") or ()],
    ]
    corrected["core_ids"] = list(
        dict.fromkeys(value for value in ordered_core_ids if value in present_core_ids)
    )
    if (
        str(baseline.get("row_role") or "") == "all_background"
        and not present_core_ids
        and all(
            str(span.get("label") or "") == "background"
            for span in corrected["canonical_spans"]
        )
    ):
        corrected = copy.deepcopy(baseline)
    else:
        corrected["canonical_repair_contract"] = (
            "manual_prediction_and_replacement_resolved_v1"
        )
    return corrected, rollback_rows


def _validate_evidence(
    *, r4_summary_path: Path, replacement_gate_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    r4 = json.loads(r4_summary_path.read_text(encoding="utf-8-sig"))
    if r4.get("schema") != R4_SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 summary schema")
    gate = json.loads(replacement_gate_path.read_text(encoding="utf-8-sig"))
    if gate.get("schema") != REPLACEMENT_GATE_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 replacement gate schema")
    if (
        not gate.get("manual_review_complete")
        or int(gate.get("missing_count") or 0)
        or int(gate.get("unreviewed_count") or 0)
        or int(gate.get("unsure_count") or 0)
    ):
        raise ValueError("Scorer canonical r5 replacement review is incomplete")
    if not _same_path(gate.get("canonical_summary"), r4_summary_path) or str(
        gate.get("canonical_summary_sha256") or ""
    ) != _sha256(r4_summary_path):
        raise ValueError("Scorer canonical r5 gate does not bind its r4 summary")

    audit_summary_path = Path(str(gate.get("audit_summary") or ""))
    audit_manifest_path = Path(str(gate.get("audit_manifest") or ""))
    manual_verdicts_path = Path(str(gate.get("manual_verdicts") or ""))
    for path, key, label in (
        (audit_summary_path, "audit_summary_sha256", "audit summary"),
        (audit_manifest_path, "audit_manifest_sha256", "audit manifest"),
        (manual_verdicts_path, "manual_verdicts_sha256", "manual verdicts"),
    ):
        _require_sha(path, gate.get(key), label=label)

    audit_summary = json.loads(
        audit_summary_path.read_text(encoding="utf-8-sig")
    )
    if audit_summary.get("schema") != REPLACEMENT_SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer canonical r4 replacement summary schema")
    if (
        not _same_path(audit_summary.get("canonical_summary"), r4_summary_path)
        or str(audit_summary.get("canonical_summary_sha256") or "")
        != _sha256(r4_summary_path)
        or not _same_path(audit_summary.get("audit_manifest"), audit_manifest_path)
        or str(audit_summary.get("audit_manifest_sha256") or "")
        != _sha256(audit_manifest_path)
    ):
        raise ValueError("Scorer canonical r5 audit evidence chain changed")

    manifest = _rows(audit_manifest_path)
    verdict_rows = _rows(manual_verdicts_path)
    if any(row.get("schema") != REPLACEMENT_ITEM_SCHEMA for row in manifest):
        raise ValueError("invalid Scorer canonical r4 replacement item schema")
    if any(row.get("schema") != MANUAL_VERDICT_SCHEMA for row in verdict_rows):
        raise ValueError("invalid Scorer canonical r4 replacement verdict schema")
    verdicts = {str(row["item_id"]): row for row in verdict_rows}
    if len(verdicts) != len(verdict_rows):
        raise ValueError("duplicate Scorer canonical r4 replacement verdict")
    if set(verdicts) != {str(row["item_id"]) for row in manifest}:
        raise ValueError("Scorer canonical r4 replacement verdict coverage changed")
    return r4, gate, manifest, verdicts


def apply_verdicts(
    *, r4_summary_path: Path, replacement_gate_path: Path, output_dir: Path
) -> dict[str, Any]:
    r4, gate, audit_manifest, verdicts = _validate_evidence(
        r4_summary_path=r4_summary_path,
        replacement_gate_path=replacement_gate_path,
    )

    canonical_path = Path(str(r4.get("canonical_sources") or ""))
    baseline_path = Path(str(r4.get("input_canonical_sources") or ""))
    placements_path = Path(str(r4.get("repair_placements") or ""))
    changed_spans_path = Path(str(r4.get("changed_spans") or ""))
    for path, key, label in (
        (canonical_path, "canonical_sources_sha256", "r4 canonical sources"),
        (baseline_path, "input_canonical_sources_sha256", "r4 input canonical"),
        (placements_path, "repair_placements_sha256", "r4 repair placements"),
        (changed_spans_path, "changed_spans_sha256", "r4 changed spans"),
    ):
        _require_sha(path, r4.get(key), label=label)
    if (
        not _same_path(gate.get("canonical_sources"), canonical_path)
        or str(gate.get("canonical_sources_sha256") or "") != _sha256(canonical_path)
    ):
        raise ValueError("Scorer canonical r5 gate does not bind r4 canonical sources")

    current_rows = _rows(canonical_path)
    baseline_rows = _rows(baseline_path)
    _validate_sources(current_rows)
    _validate_sources(baseline_rows)
    current_by_id = {str(row["source_id"]): row for row in current_rows}
    baseline_by_id = {str(row["source_id"]): row for row in baseline_rows}
    if list(current_by_id) != list(baseline_by_id):
        raise ValueError("Scorer canonical r5 source identity or order changed")

    placements = _rows(placements_path)
    placement_by_id = {str(row["placement_id"]): row for row in placements}
    if (
        len(placement_by_id) != len(placements)
        or any(row.get("schema") != PLACEMENT_SCHEMA for row in placements)
    ):
        raise ValueError("invalid Scorer canonical r4 repair placements")
    audit_by_id = {str(row["item_id"]): row for row in audit_manifest}
    if set(audit_by_id) != set(placement_by_id):
        raise ValueError("Scorer canonical r5 audit placement coverage changed")
    for placement_id, placement in placement_by_id.items():
        audit = audit_by_id[placement_id]
        for key in (
            "event_id",
            "source_id",
            "target_source_id",
            "partition",
            "role",
            "tile_index",
            "occurrence_index",
            "mapped_start_sample",
            "mapped_end_sample",
            "placement_core_id",
            "background_label_change_sample_count",
            "core_registered",
        ):
            if audit.get(key) != placement.get(key):
                raise ValueError(
                    f"Scorer canonical r5 audited placement changed: {placement_id} {key}"
                )
        audit_ranges = [
            (int(span["start_sample"]), int(span["end_sample"]))
            for span in audit["background_label_change_ranges"]
        ]
        placement_ranges = [
            (int(span["start_sample"]), int(span["end_sample"]))
            for span in placement["background_label_change_ranges"]
        ]
        if audit_ranges != placement_ranges:
            raise ValueError(
                "Scorer canonical r5 audited placement change ranges changed: "
                f"{placement_id}"
            )

    items_by_event: dict[str, list[str]] = defaultdict(list)
    for item_id, row in audit_by_id.items():
        items_by_event[str(row["event_id"])].append(item_id)
    source_rejected_events: set[str] = set()
    rejected_ids: set[str] = set()
    for event_id, item_ids in items_by_event.items():
        values = {str(verdicts[item_id]["verdict"]) for item_id in item_ids}
        if "source_event_not_target" in values:
            if values != {"source_event_not_target"}:
                raise ValueError("Scorer canonical r5 source rejection group is incomplete")
            source_rejected_events.add(event_id)
            rejected_ids.update(item_ids)
    followup_ids = {
        item_id
        for item_id, row in verdicts.items()
        if str(row["verdict"]) in {"boundary_incomplete", "not_target_after_render"}
    }
    rejected_ids.update(followup_ids)
    allowed = {
        "repair_speech_correct",
        "source_event_not_target",
        "boundary_incomplete",
        "not_target_after_render",
    }
    if any(str(row["verdict"]) not in allowed for row in verdicts.values()):
        raise ValueError("Scorer canonical r5 contains an unresolved verdict")
    if (
        sorted(source_rejected_events)
        != sorted(str(value) for value in gate.get("source_event_repair_ids") or ())
        or sorted(followup_ids)
        != sorted(str(value) for value in gate.get("repair_followup_ids") or ())
    ):
        raise ValueError("Scorer canonical r5 gate verdict policy changed")

    rollback_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_rows: list[dict[str, Any]] = []
    accepted_placements: list[dict[str, Any]] = []
    rejected_core_ids: set[str] = set()
    for placement in placements:
        placement_id = str(placement["placement_id"])
        if placement_id not in rejected_ids:
            if str(verdicts[placement_id]["verdict"]) != "repair_speech_correct":
                raise ValueError("Scorer canonical r5 accepted placement lacks approval")
            accepted_placements.append(placement)
            continue
        verdict = str(verdicts[placement_id]["verdict"])
        rollback_sample_count = 0
        for span in placement["background_label_change_ranges"]:
            start = int(span["start_sample"])
            end = int(span["end_sample"])
            rollback_by_source[str(placement["target_source_id"])].append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "placement_id": placement_id,
                    "event_id": str(placement["event_id"]),
                    "verdict": verdict,
                }
            )
            rollback_sample_count += end - start
        if placement.get("core_registered"):
            rejected_core_ids.add(str(placement["placement_core_id"]))
        rejected_rows.append(
            {
                "schema": REJECTED_PLACEMENT_SCHEMA,
                "placement_id": placement_id,
                "event_id": str(placement["event_id"]),
                "source_id": str(placement["source_id"]),
                "target_source_id": str(placement["target_source_id"]),
                "partition": str(placement["partition"]),
                "role": str(placement["role"]),
                "verdict": verdict,
                "proposed_core_registered": bool(placement["core_registered"]),
                "rollback_sample_count": rollback_sample_count,
            }
        )

    corrected_by_id = {key: copy.deepcopy(value) for key, value in current_by_id.items()}
    rollback_rows: list[dict[str, Any]] = []
    for source_id, operations in rollback_by_source.items():
        corrected, source_rollbacks = _restore_rejected_ranges(
            current=current_by_id[source_id],
            baseline=baseline_by_id[source_id],
            operations=operations,
        )
        corrected_by_id[source_id] = corrected
        rollback_rows.extend(source_rollbacks)
    corrected_rows = [corrected_by_id[str(row["source_id"])] for row in current_rows]
    dataset = _validate_sources(corrected_rows)
    identity_fields = (
        "source_id",
        "audio",
        "partition",
        "sample_rate",
        "sample_count",
        "duration_s",
    )
    for current, corrected in zip(current_rows, corrected_rows, strict=True):
        if any(current.get(key) != corrected.get(key) for key in identity_fields):
            raise ValueError("Scorer canonical r5 changed source/audio/partition identity")
    if r4.get("audio_bytes_changed") is not False:
        raise ValueError("Scorer canonical r5 input does not prove immutable audio bytes")
    present_core_ids = {
        str(core_id)
        for row in corrected_rows
        for core_id in row.get("core_ids") or ()
    }
    leaked = sorted(rejected_core_ids & present_core_ids)
    if leaked:
        raise ValueError(f"Scorer canonical r5 rejected repair cores remain: {leaked}")

    changed_spans = _rows(changed_spans_path)
    if any(row.get("schema") != CHANGED_SPAN_SCHEMA for row in changed_spans):
        raise ValueError("invalid Scorer canonical r4 changed span schema")
    retained_changed_spans = [
        row
        for row in changed_spans
        if str(row.get("repair_placement_id") or "") not in rejected_ids
    ]

    feature_labels: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    for source in corrected_rows:
        labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
        feature_labels.append(
            {
                "audio_id": source["source_id"],
                "source": "scorer_v10_corrected_canonical_r5",
                "duration_s": source["duration_s"],
                "text": "",
                "teacher_segments": {},
                "frame_hop_s": FRAME_HOP_S,
                "speech_frames": (
                    labels == CANONICAL_LABELS["speech"]
                ).astype(int).tolist(),
                "label_quality": (
                    "negative" if source["row_role"] == "all_background" else "supervised"
                ),
                "frame_weights": weights.tolist(),
                "boundary_metadata": {
                    "schema": SOURCE_SCHEMA,
                    "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
                    "row_role": source["row_role"],
                    "partition": source["partition"],
                    "unsure_frame_count": int(
                        np.sum(labels == CANONICAL_LABELS["unsure"])
                    ),
                    "canonical_repair_contract": source.get(
                        "canonical_repair_contract", ""
                    ),
                },
            }
        )
        audio_manifest.append(
            {
                "audio_id": source["source_id"],
                "audio": source["audio"],
                "partition": source["partition"],
                "row_role": source["row_role"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sources_output = output_dir / "canonical_sources.jsonl"
    labels_output = output_dir / "feature_cache_labels.jsonl"
    audio_output = output_dir / "audio_manifest.json"
    accepted_output = output_dir / "accepted_repair_placements.jsonl"
    rejected_output = output_dir / "rejected_repair_placements.jsonl"
    rollback_output = output_dir / "rollback_spans.jsonl"
    changed_output = output_dir / "changed_spans.jsonl"
    _write_jsonl(sources_output, corrected_rows)
    _write_jsonl(labels_output, feature_labels)
    _write_jsonl(accepted_output, accepted_placements)
    _write_jsonl(rejected_output, rejected_rows)
    _write_jsonl(rollback_output, rollback_rows)
    _write_jsonl(changed_output, retained_changed_spans)
    audio_output.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    before_counts = _frame_counts(current_rows)
    after_counts = _frame_counts(corrected_rows)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ),
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_r4_summary": str(r4_summary_path),
        "input_r4_summary_sha256": _sha256(r4_summary_path),
        "input_r4_canonical_sources": str(canonical_path),
        "input_r4_canonical_sources_sha256": _sha256(canonical_path),
        "baseline_canonical_sources": str(baseline_path),
        "baseline_canonical_sources_sha256": _sha256(baseline_path),
        "replacement_gate": str(replacement_gate_path),
        "replacement_gate_sha256": _sha256(replacement_gate_path),
        "replacement_manual_verdicts": str(gate["manual_verdicts"]),
        "replacement_manual_verdicts_sha256": str(
            gate["manual_verdicts_sha256"]
        ),
        "replacement_manual_review_complete": True,
        "replacement_verdicts_applied": True,
        "source_event_rejection_count": len(source_rejected_events),
        "source_event_rejection_ids": sorted(source_rejected_events),
        "placement_followup_count": len(followup_ids),
        "placement_followup_ids": sorted(followup_ids),
        "proposed_repair_placement_count": len(placements),
        "accepted_repair_placement_count": len(accepted_placements),
        "rejected_repair_placement_count": len(rejected_rows),
        "rejected_registered_core_count": len(rejected_core_ids),
        "rejected_registered_core_ids": sorted(rejected_core_ids),
        "rollback_span_count": len(rollback_rows),
        "rollback_sample_count": sum(
            int(row["end_sample"]) - int(row["start_sample"])
            for row in rollback_rows
        ),
        "canonical_frame_counts_before": dict(before_counts),
        "canonical_frame_counts_after": dict(after_counts),
        "canonical_frame_count_delta": {
            label: int(after_counts[label] - before_counts[label])
            for label in ("speech", "background", "unsure")
        },
        "dataset": dataset,
        "canonical_sources": str(sources_output),
        "canonical_sources_sha256": _sha256(sources_output),
        "feature_cache_labels": str(labels_output),
        "feature_cache_labels_sha256": _sha256(labels_output),
        "audio_manifest": str(audio_output),
        "audio_manifest_sha256": _sha256(audio_output),
        "accepted_repair_placements": str(accepted_output),
        "accepted_repair_placements_sha256": _sha256(accepted_output),
        "rejected_repair_placements": str(rejected_output),
        "rejected_repair_placements_sha256": _sha256(rejected_output),
        "rollback_spans": str(rollback_output),
        "rollback_spans_sha256": _sha256(rollback_output),
        "changed_spans": str(changed_output),
        "changed_spans_sha256": _sha256(changed_output),
        "audio_bytes_changed": False,
        "source_identity_changed": False,
        "partition_identity_changed": False,
        "unsure_training_mapping": -100,
        "replacement_audit_required": False,
        "replacement_resolution_pass": True,
        "feature_cache_labels_ready": True,
        "feature_cache_reuse_pending_signature_audit": True,
        "training_manifest_ready": False,
        "checkpoint_promotion_authorized": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-summary", required=True)
    parser.add_argument("--replacement-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_verdicts(
                r4_summary_path=Path(args.r4_summary),
                replacement_gate_path=Path(args.replacement_gate),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
