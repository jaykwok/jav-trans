#!/usr/bin/env python3
"""Apply all audited Scorer v10 r4 label repairs and dependency propagation."""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
import math
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
from tools.audits.evaluate_scorer_v10_background_speech_repair_audit import (  # noqa: E402
    EVENT_SCHEMA as SPEECH_REPAIR_EVENT_SCHEMA,
    RESULT_SCHEMA as SPEECH_REPAIR_GATE_SCHEMA,
)
from tools.audits.evaluate_scorer_v10_prediction_audit import (  # noqa: E402
    RESULT_SCHEMA as PREDICTION_GATE_SCHEMA,
)
from tools.audits.generate_scorer_v10_prediction_audit_html import (  # noqa: E402
    SUMMARY_SCHEMA as PREDICTION_SUMMARY_SCHEMA,
    VERDICT_SCHEMA as PREDICTION_VERDICT_SCHEMA,
    audit_truth_drop_spans,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    _write_jsonl,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_r4_summary_v1"
BACKGROUND_REPAIR_SCHEMA = "speech_scorer_v10_prediction_background_repair_v1"
DEPENDENCY_MAPPING_SCHEMA = "speech_scorer_v10_background_dependency_mapping_v1"
PLACEMENT_SCHEMA = "speech_scorer_v10_background_speech_repair_placement_v1"
CHANGED_SPAN_SCHEMA = "speech_scorer_v10_canonical_r4_changed_span_v1"
FRAME_HOP_S = 0.02
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _evidence(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _round_sample(seconds: float) -> int:
    return int(round(float(seconds) * SAMPLE_RATE))


def _placement_core_id(
    *,
    event_id: str,
    target_source_id: str,
    role: str,
    occurrence_index: int,
    mapped_start_sample: int,
    mapped_end_sample: int,
) -> str:
    payload = (
        "scorer-v10-canonical-r4-placement-v1\0"
        f"{event_id}\0{target_source_id}\0{role}\0{occurrence_index}\0"
        f"{mapped_start_sample}\0{mapped_end_sample}"
    ).encode("utf-8")
    return "scorer-v10-repair-core-" + hashlib.sha256(payload).hexdigest()


def _validate_prediction_audit(
    *,
    audit_dir: Path,
    canonical_by_source: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = audit_dir / "summary.json"
    manifest_path = audit_dir / "audit_manifest.jsonl"
    verdicts_path = audit_dir / "manual_verdicts.jsonl"
    gate_path = audit_dir / "manual_gate.json"
    for path in (summary_path, manifest_path, verdicts_path, gate_path):
        if not path.is_file():
            raise ValueError(f"Scorer prediction repair evidence is missing: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != PREDICTION_SUMMARY_SCHEMA:
        raise ValueError("invalid Scorer prediction repair summary schema")
    if Path(str(summary.get("audit_manifest") or "")).resolve() != manifest_path.resolve():
        raise ValueError("Scorer prediction repair manifest does not match its summary")
    if gate.get("schema") not in {
        PREDICTION_GATE_SCHEMA,
        "speech_scorer_v10_prediction_manual_gate_v2",
    }:
        raise ValueError("invalid Scorer prediction repair gate schema")
    for key, path in (
        ("audit_summary", summary_path),
        ("audit_manifest", manifest_path),
        ("manual_verdicts", verdicts_path),
    ):
        if Path(str(gate.get(key) or "")).resolve() != path.resolve():
            raise ValueError(f"Scorer prediction repair gate has another {key}")
    if gate.get("manual_review_complete") is not True or int(gate.get("unsure_count") or 0):
        raise ValueError("Scorer prediction repair audit is incomplete or unsure")

    targets: dict[str, dict[str, Any]] = {}
    for row in _rows(manifest_path):
        audit_id = str(row.get("audit_id") or "")
        if not audit_id or audit_id in targets:
            raise ValueError("Scorer prediction repair targets require unique audit_id")
        targets[audit_id] = row
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("Scorer prediction repair target count mismatch")
    category_counts = Counter(str(row.get("category") or "") for row in targets.values())
    if dict(category_counts) != {
        str(key): int(value)
        for key, value in dict(summary.get("category_counts") or {}).items()
    }:
        raise ValueError("Scorer prediction repair category counts mismatch")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(verdicts_path):
        if row.get("schema") != PREDICTION_VERDICT_SCHEMA:
            raise ValueError("invalid Scorer prediction repair verdict schema")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in targets or audit_id in verdicts:
            raise ValueError("invalid or duplicate Scorer prediction repair verdict")
        target = targets[audit_id]
        for field in ("source_id", "partition", "row_role", "category"):
            if str(row.get(field) or "") != str(target.get(field) or ""):
                raise ValueError(f"Scorer prediction repair verdict {field} mismatch")
        verdicts[audit_id] = row
    if set(verdicts) != set(targets):
        raise ValueError("Scorer prediction repair verdict set is incomplete")
    canonical_repair_ids = {
        audit_id
        for audit_id, row in verdicts.items()
        if str(row.get("verdict") or "")
        in {"canonical_should_be_background", "canonical_contains_target_speech"}
    }
    if canonical_repair_ids != set(gate.get("canonical_repair_ids") or ()):
        raise ValueError("Scorer prediction repair gate canonical ids mismatch")

    repairs: list[dict[str, Any]] = []
    for audit_id, verdict in verdicts.items():
        if str(verdict.get("verdict") or "") != "canonical_should_be_background":
            continue
        target = targets[audit_id]
        source_id = str(target["source_id"])
        canonical = canonical_by_source.get(source_id)
        if canonical is None:
            raise ValueError(f"Scorer prediction repair source is missing: {source_id}")
        if (
            str(target.get("partition") or "") != str(canonical["partition"])
            or str(target.get("row_role") or "") != "speech"
            or str(canonical.get("row_role") or "") != "speech"
        ):
            raise ValueError("Scorer prediction repair source identity mismatch")
        labels = canonical_frame_labels(canonical, frame_hop_s=FRAME_HOP_S)
        if int(target.get("frame_count") or -1) != len(labels):
            raise ValueError("Scorer prediction repair frame count mismatch")
        stored_spans = list(target.get("truth_drop_spans") or ())
        if stored_spans != audit_truth_drop_spans(target):
            raise ValueError("Scorer prediction repair truth-drop spans changed")
        if not stored_spans:
            raise ValueError("canonical background verdict has no exact truth-drop span")
        for span_index, span in enumerate(stored_spans):
            start_frame = int(span["start_frame"])
            end_frame = int(span["end_frame"])
            if (
                start_frame < 0
                or end_frame <= start_frame
                or end_frame > len(labels)
                or not np.all(labels[start_frame:end_frame] == CANONICAL_LABELS["speech"])
            ):
                raise ValueError("prediction background repair is not current canonical speech")
            repairs.append(
                {
                    "schema": BACKGROUND_REPAIR_SCHEMA,
                    "source_id": source_id,
                    "partition": str(canonical["partition"]),
                    "audit_dir": str(audit_dir),
                    "audit_id": audit_id,
                    "category": str(target["category"]),
                    "span_index": span_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_sample": min(int(canonical["sample_count"]), start_frame * 320),
                    "end_sample": min(int(canonical["sample_count"]), end_frame * 320),
                    "verdict": "canonical_should_be_background",
                }
            )
    expected_count = int((gate.get("verdict_counts") or {}).get("canonical_should_be_background") or 0)
    selected_ids = {
        str(row["audit_id"])
        for row in repairs
    }
    if len(selected_ids) != expected_count:
        raise ValueError("Scorer prediction background verdict count mismatch")
    return repairs, {
        "audit_dir": str(audit_dir),
        "canonical_background_verdict_count": expected_count,
        "exact_span_count": len(repairs),
        "evidence": {
            "summary": _evidence(summary_path),
            "manifest": _evidence(manifest_path),
            "manual_verdicts": _evidence(verdicts_path),
            "manual_gate": _evidence(gate_path),
        },
    }


def _background_repair_ops(
    repairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    evidence_by_frame: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for repair in repairs:
        evidence_id = (
            f"{Path(str(repair['audit_dir'])).name}:{repair['audit_id']}:"
            f"{repair['start_frame']}-{repair['end_frame']}"
        )
        for frame in range(int(repair["start_frame"]), int(repair["end_frame"])):
            evidence_by_frame[str(repair["source_id"])][frame].add(evidence_id)

    result: dict[str, list[dict[str, Any]]] = {}
    for source_id, frames in evidence_by_frame.items():
        ordered = sorted(frames)
        runs: list[dict[str, Any]] = []
        start = previous = ordered[0]
        evidence_ids = set(frames[start])
        for frame in ordered[1:]:
            if frame != previous + 1:
                runs.append(
                    {
                        "start_sample": start * 320,
                        "end_sample": (previous + 1) * 320,
                        "evidence_ids": sorted(evidence_ids),
                    }
                )
                start = frame
                evidence_ids = set()
            evidence_ids.update(frames[frame])
            previous = frame
        runs.append(
            {
                "start_sample": start * 320,
                "end_sample": (previous + 1) * 320,
                "evidence_ids": sorted(evidence_ids),
            }
        )
        result[source_id] = runs
    return result


def _relabel(
    *,
    row: dict[str, Any],
    operations: list[dict[str, Any]],
    expected_label: str,
    new_label: str,
    label_source: str,
    changed_spans: list[dict[str, Any]],
) -> None:
    if not operations:
        return
    operations = sorted(
        operations, key=lambda item: (int(item["start_sample"]), int(item["end_sample"]))
    )
    for left, right in zip(operations, operations[1:]):
        if int(right["start_sample"]) < int(left["end_sample"]):
            raise ValueError(f"canonical r4 relabel operations overlap: {row['source_id']}")
    boundaries = {
        int(span[side])
        for span in row["canonical_spans"]
        for side in ("start_sample", "end_sample")
    }
    for operation in operations:
        start = max(0, int(operation["start_sample"]))
        end = min(int(row["sample_count"]), int(operation["end_sample"]))
        if end <= start:
            raise ValueError("canonical r4 relabel operation is empty")
        operation["start_sample"] = start
        operation["end_sample"] = end
        boundaries.update((start, end))

    original_spans = list(row["canonical_spans"])
    result: list[dict[str, Any]] = []
    for start, end in zip(sorted(boundaries), sorted(boundaries)[1:]):
        if end <= start:
            continue
        source_spans = [
            span
            for span in original_spans
            if start >= int(span["start_sample"]) and end <= int(span["end_sample"])
        ]
        if len(source_spans) != 1:
            raise ValueError("canonical r4 relabel span topology is invalid")
        operation_rows = [
            operation
            for operation in operations
            if start >= int(operation["start_sample"])
            and end <= int(operation["end_sample"])
        ]
        if len(operation_rows) > 1:
            raise ValueError("canonical r4 relabel coverage is ambiguous")
        original = source_spans[0]
        piece = copy.deepcopy(original)
        piece["start_sample"] = start
        piece["end_sample"] = end
        if operation_rows:
            operation = operation_rows[0]
            if str(original.get("label") or "") != expected_label:
                raise ValueError(
                    f"canonical r4 expected {expected_label}, got {original.get('label')}: "
                    f"{row['source_id']} {start}-{end}"
                )
            piece["label"] = new_label
            piece["label_source"] = label_source
            piece["manual_original_label"] = expected_label
            if new_label == "background":
                origin_core_id = str(piece.pop("core_id", "") or "")
                if origin_core_id:
                    piece["origin_core_id"] = origin_core_id
                piece["prediction_repair_evidence_ids"] = list(
                    operation.get("evidence_ids") or ()
                )
            elif new_label == "speech":
                origin_background_id = str(piece.pop("background_id", "") or "")
                if origin_background_id:
                    piece["origin_background_id"] = origin_background_id
                piece["core_id"] = str(operation["core_id"])
                piece["repair_event_id"] = str(operation["event_id"])
                piece["repair_placement_id"] = str(operation["placement_id"])
                piece["repair_role"] = str(operation["role"])
            changed_spans.append(
                {
                    "schema": CHANGED_SPAN_SCHEMA,
                    "source_id": str(row["source_id"]),
                    "partition": str(row["partition"]),
                    "start_sample": start,
                    "end_sample": end,
                    "original_label": expected_label,
                    "label": new_label,
                    "label_source": label_source,
                    "core_id": str(piece.get("core_id") or ""),
                    "origin_core_id": str(piece.get("origin_core_id") or ""),
                    "repair_event_id": str(piece.get("repair_event_id") or ""),
                    "repair_placement_id": str(piece.get("repair_placement_id") or ""),
                }
            )
        result.append(piece)
    row["canonical_spans"] = result
    present_core_ids = {
        str(span.get("core_id") or "")
        for span in result
        if str(span.get("label") or "") == "speech" and span.get("core_id")
    }
    existing_core_ids = [str(value) for value in row.get("core_ids") or ()]
    operation_core_ids = [
        str(operation.get("core_id") or "")
        for operation in operations
        if operation.get("core_id")
    ]
    row["core_ids"] = [
        core_id
        for core_id in [*existing_core_ids, *operation_core_ids]
        if core_id in present_core_ids
    ]
    row["core_ids"] = list(dict.fromkeys(row["core_ids"]))


def map_event_to_rendered_audio(
    *,
    event_start_sample: int,
    event_end_sample: int,
    source_sample_count: int,
    source_offset_sample: int,
    rendered_start_sample: int,
    rendered_end_sample: int,
) -> list[dict[str, int]]:
    """Map one exact source event through the builder's crop-or-tile contract."""

    event_start_sample = int(event_start_sample)
    event_end_sample = int(event_end_sample)
    source_sample_count = int(source_sample_count)
    source_offset_sample = int(source_offset_sample)
    rendered_start_sample = int(rendered_start_sample)
    rendered_end_sample = int(rendered_end_sample)
    rendered_length = rendered_end_sample - rendered_start_sample
    if (
        source_sample_count <= 0
        or event_start_sample < 0
        or event_end_sample <= event_start_sample
        or event_end_sample > source_sample_count
        or rendered_length <= 0
    ):
        raise ValueError("invalid source-event mapping geometry")

    blocks: list[tuple[int, int, int]] = []
    if source_sample_count >= rendered_length:
        if source_offset_sample < 0 or source_offset_sample + rendered_length > source_sample_count:
            raise ValueError("crop offset is outside the source asset")
        blocks.append((source_offset_sample, source_offset_sample + rendered_length, 0))
    else:
        if source_offset_sample != 0:
            raise ValueError("tiled source assets must use offset zero")
        output_cursor = 0
        while output_cursor < rendered_length:
            block_length = min(source_sample_count, rendered_length - output_cursor)
            blocks.append((0, block_length, output_cursor))
            output_cursor += block_length

    mapped: list[dict[str, int]] = []
    for tile_index, (source_start, source_end, output_start) in enumerate(blocks):
        overlap_start = max(source_start, event_start_sample)
        overlap_end = min(source_end, event_end_sample)
        if overlap_end <= overlap_start:
            continue
        start = rendered_start_sample + output_start + (overlap_start - source_start)
        end = rendered_start_sample + output_start + (overlap_end - source_start)
        mapped.append(
            {
                "tile_index": tile_index,
                "source_start_sample": overlap_start,
                "source_end_sample": overlap_end,
                "mapped_start_sample": start,
                "mapped_end_sample": end,
            }
        )
    return mapped


def _manifest_uses(
    *,
    composite_manifest: Path,
    affected_background_ids: set[str],
    active_source_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    uses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inactive: set[str] = set()
    for row in _rows(composite_manifest):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("composite source manifest row is missing sample_id")

        def add_use(detail: dict[str, Any], *, role: str, start: int, end: int) -> None:
            background_id = str(detail.get("audio_id") or "")
            if background_id not in affected_background_ids:
                return
            if sample_id not in active_source_ids:
                inactive.add(sample_id)
                return
            uses[background_id].append(
                {
                    "sample_id": sample_id,
                    "partition": str(row.get("source_partition") or ""),
                    "role": role,
                    "rendered_start_sample": int(start),
                    "rendered_end_sample": int(end),
                    "source_offset_sample": _round_sample(
                        float(detail.get("source_offset_s") or 0.0)
                    ),
                    "source_duration_sample": _round_sample(
                        float(detail.get("duration_s") or 0.0)
                    ),
                    "source_audio": str(detail.get("audio") or ""),
                }
            )

        unit = dict(row.get("negative_unit_span") or {})
        if unit:
            add_use(
                dict(unit.get("source") or {}),
                role="negative_unit",
                start=int(unit["start_sample"]),
                end=int(unit["end_sample"]),
            )
        gaps = dict(row.get("inter_unit_gaps") or {})
        gap_sources = list(gaps.get("sources") or ())
        if gaps:
            if len(gap_sources) != 2:
                raise ValueError("composite inter-unit gaps require two source rows")
            add_use(
                dict(gap_sources[0]),
                role="left_gap",
                start=int(gaps["left_start_sample"]),
                end=int(gaps["left_end_sample"]),
            )
            add_use(
                dict(gap_sources[1]),
                role="right_gap",
                start=int(gaps["right_start_sample"]),
                end=int(gaps["right_end_sample"]),
            )
        overlay = dict(row.get("additive_overlay") or {})
        if overlay:
            add_use(
                dict(overlay.get("source") or {}),
                role="additive_overlay",
                start=0,
                end=int(row["sample_count"]),
            )
    return dict(uses), sorted(inactive)


def _speech_repair_gate(
    *, gate_path: Path, canonical_sources: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if gate.get("schema") != SPEECH_REPAIR_GATE_SCHEMA:
        raise ValueError("invalid Scorer background-speech repair gate schema")
    if (
        gate.get("manual_review_complete") is not True
        or gate.get("canonical_repair_ready") is not True
        or int(gate.get("unsure_count") or 0)
        or int(gate.get("boundary_followup_count") or 0)
        or int(gate.get("source_without_target_count") or 0)
    ):
        raise ValueError("Scorer background-speech repair gate is not ready")
    if str(gate.get("canonical_sources_sha256") or "") != _sha256(canonical_sources):
        raise ValueError("background-speech repair gate is bound to another canonical manifest")
    events_path = Path(str(gate.get("repair_events") or ""))
    if (
        not events_path.is_file()
        or _sha256(events_path) != str(gate.get("repair_events_sha256") or "")
    ):
        raise ValueError("background-speech repair events changed after gate compilation")
    decisions_path = Path(str(gate.get("decisions") or ""))
    if (
        not decisions_path.is_file()
        or _sha256(decisions_path) != str(gate.get("decisions_sha256") or "")
    ):
        raise ValueError("background-speech repair decisions changed after gate compilation")
    source_recheck = str(gate.get("source_recheck_gate") or "")
    if source_recheck:
        source_recheck_path = Path(source_recheck)
        if (
            not source_recheck_path.is_file()
            or _sha256(source_recheck_path)
            != str(gate.get("source_recheck_gate_sha256") or "")
        ):
            raise ValueError("background source recheck gate changed")
    events = _rows(events_path)
    if len(events) != int(gate.get("repair_event_count") or -1):
        raise ValueError("background-speech repair event count mismatch")
    seen: set[str] = set()
    for event in events:
        event_id = str(event.get("event_id") or "")
        if event.get("schema") != SPEECH_REPAIR_EVENT_SCHEMA or not event_id or event_id in seen:
            raise ValueError("invalid or duplicate background-speech repair event")
        seen.add(event_id)
    return gate, events


def _intersections_by_label(
    row: dict[str, Any], *, start: int, end: int
) -> tuple[list[tuple[int, int]], int]:
    background: list[tuple[int, int]] = []
    speech_samples = 0
    for span in row["canonical_spans"]:
        overlap_start = max(start, int(span["start_sample"]))
        overlap_end = min(end, int(span["end_sample"]))
        if overlap_end <= overlap_start:
            continue
        label = str(span.get("label") or "")
        if label == "unsure":
            raise ValueError(
                f"speech repair overlaps canonical unsure: {row['source_id']} "
                f"{overlap_start}-{overlap_end}"
            )
        if label == "background":
            if str(span.get("label_source") or "") == "manual_prediction_background_repair_v1":
                raise ValueError(
                    f"direct background verdict conflicts with propagated speech: {row['source_id']}"
                )
            background.append((overlap_start, overlap_end))
        elif label == "speech":
            speech_samples += overlap_end - overlap_start
        else:
            raise ValueError("canonical source has an invalid label")
    return background, speech_samples


def apply_repairs(
    *,
    canonical_sources: Path,
    prediction_audit_dirs: list[Path],
    background_speech_repair_gate: Path,
    composite_source_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not prediction_audit_dirs:
        raise ValueError("canonical r4 requires prediction background repair audits")
    original_rows = _rows(canonical_sources)
    _validate_sources(original_rows)
    canonical_by_source = {str(row["source_id"]): row for row in original_rows}
    if len(canonical_by_source) != len(original_rows):
        raise ValueError("canonical r4 input has duplicate source_id values")
    before_counts = _frame_counts(original_rows)

    prediction_repairs: list[dict[str, Any]] = []
    prediction_evidence: list[dict[str, Any]] = []
    for audit_dir in prediction_audit_dirs:
        repairs, evidence = _validate_prediction_audit(
            audit_dir=audit_dir,
            canonical_by_source=canonical_by_source,
        )
        prediction_repairs.extend(repairs)
        prediction_evidence.append(evidence)
    background_ops = _background_repair_ops(prediction_repairs)
    unique_background_frames = {
        (source_id, frame)
        for source_id, operations in background_ops.items()
        for operation in operations
        for frame in range(
            int(operation["start_sample"]) // 320,
            math.ceil(int(operation["end_sample"]) / 320),
        )
    }

    corrected_by_source = {
        source_id: copy.deepcopy(row) for source_id, row in canonical_by_source.items()
    }
    changed_spans: list[dict[str, Any]] = []
    removed_core_ids: set[str] = set()
    for source_id, operations in background_ops.items():
        row = corrected_by_source[source_id]
        for operation in operations:
            operation["end_sample"] = min(
                int(row["sample_count"]), int(operation["end_sample"])
            )
        previous_core_ids = set(str(value) for value in row.get("core_ids") or ())
        _relabel(
            row=row,
            operations=operations,
            expected_label="speech",
            new_label="background",
            label_source="manual_prediction_background_repair_v1",
            changed_spans=changed_spans,
        )
        remaining = set(str(value) for value in row.get("core_ids") or ())
        removed_core_ids.update(previous_core_ids - remaining)
        if row.get("row_role") != "speech" or not row.get("core_ids"):
            raise ValueError("prediction background repairs removed all target speech from a source")
        row["canonical_repair_contract"] = "manual_prediction_background_repair_v1"

    speech_gate, events = _speech_repair_gate(
        gate_path=background_speech_repair_gate,
        canonical_sources=canonical_sources,
    )
    events_by_background: dict[str, list[dict[str, Any]]] = defaultdict(list)
    controls_by_background: dict[str, str] = {}
    for event in events:
        source_id = str(event["source_id"])
        row = corrected_by_source.get(source_id)
        if row is None:
            raise ValueError(f"background-speech repair control is missing: {source_id}")
        background_id = str(event["background_id"])
        if (
            str(row.get("row_role") or "") != "all_background"
            or str(row.get("background_id") or "") != background_id
            or str(row.get("partition") or "") != str(event["partition"])
            or int(row["sample_rate"]) != SAMPLE_RATE
            or int(event["start_sample"]) < 0
            or int(event["end_sample"]) > int(row["sample_count"])
        ):
            raise ValueError("background-speech repair event does not match its control")
        events_by_background[background_id].append(event)
        controls_by_background[background_id] = source_id
    affected_background_ids = set(events_by_background)
    if len(controls_by_background) != int(speech_gate.get("repair_source_count") or -1):
        raise ValueError("background-speech repair source count mismatch")

    active_speech_ids = {
        source_id
        for source_id, row in canonical_by_source.items()
        if str(row.get("row_role") or "") == "speech"
    }
    uses_by_background, inactive_dependency_ids = _manifest_uses(
        composite_manifest=composite_source_manifest,
        affected_background_ids=affected_background_ids,
        active_source_ids=active_speech_ids,
    )
    for background_id in affected_background_ids:
        canonical_dependents = {
            str(row["source_id"])
            for row in original_rows
            if str(row.get("row_role") or "") == "speech"
            and background_id in {str(value) for value in row.get("background_source_ids") or ()}
        }
        manifest_dependents = {
            str(use["sample_id"]) for use in uses_by_background.get(background_id, ())
        }
        if canonical_dependents != manifest_dependents:
            raise ValueError(
                f"background dependency mapping mismatch for {background_id}: "
                f"canonical={sorted(canonical_dependents)} manifest={sorted(manifest_dependents)}"
            )

    dependency_mappings: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    placement_operations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_core_ids = {
        str(core_id)
        for row in original_rows
        for core_id in row.get("core_ids") or ()
    }

    def register_placement(
        *,
        event: dict[str, Any],
        target_source_id: str,
        role: str,
        occurrence_index: int,
        mapped_start: int,
        mapped_end: int,
        mapping_id: str,
        tile_index: int,
    ) -> None:
        row = corrected_by_source[target_source_id]
        background_ranges, already_speech = _intersections_by_label(
            row, start=mapped_start, end=mapped_end
        )
        core_id = _placement_core_id(
            event_id=str(event["event_id"]),
            target_source_id=target_source_id,
            role=role,
            occurrence_index=occurrence_index,
            mapped_start_sample=mapped_start,
            mapped_end_sample=mapped_end,
        )
        if core_id in used_core_ids:
            raise ValueError("canonical r4 generated a duplicate repair core id")
        placement_id = (
            f"{target_source_id}::{role}::{event['event_id']}::occ{occurrence_index:02d}"
        )
        changed_sample_count = sum(end - start for start, end in background_ranges)
        placement = {
            "schema": PLACEMENT_SCHEMA,
            "placement_id": placement_id,
            "mapping_id": mapping_id,
            "event_id": str(event["event_id"]),
            "event_core_id": str(event["core_id"]),
            "source_id": str(event["source_id"]),
            "background_id": str(event["background_id"]),
            "target_source_id": target_source_id,
            "partition": str(row["partition"]),
            "role": role,
            "tile_index": tile_index,
            "occurrence_index": occurrence_index,
            "mapped_start_sample": mapped_start,
            "mapped_end_sample": mapped_end,
            "mapped_start_s": mapped_start / SAMPLE_RATE,
            "mapped_end_s": mapped_end / SAMPLE_RATE,
            "placement_core_id": core_id,
            "background_label_change_ranges": [
                {"start_sample": start, "end_sample": end}
                for start, end in background_ranges
            ],
            "background_label_change_sample_count": changed_sample_count,
            "already_speech_sample_count": already_speech,
            "core_registered": bool(background_ranges),
        }
        placements.append(placement)
        if background_ranges:
            used_core_ids.add(core_id)
            for start, end in background_ranges:
                placement_operations[target_source_id].append(
                    {
                        "start_sample": start,
                        "end_sample": end,
                        "core_id": core_id,
                        "event_id": str(event["event_id"]),
                        "placement_id": placement_id,
                        "role": role,
                    }
                )

    for background_id in sorted(affected_background_ids):
        control_source_id = controls_by_background[background_id]
        control = corrected_by_source[control_source_id]
        original_background_id = str(control["background_id"])
        control["row_role"] = "speech"
        control["background_id"] = ""
        control["repaired_background_id"] = original_background_id
        for occurrence_index, event in enumerate(
            sorted(events_by_background[background_id], key=lambda item: int(item["event_index"]))
        ):
            register_placement(
                event=event,
                target_source_id=control_source_id,
                role="control",
                occurrence_index=occurrence_index,
                mapped_start=int(event["start_sample"]),
                mapped_end=int(event["end_sample"]),
                mapping_id=f"{control_source_id}::control::{event['event_id']}",
                tile_index=0,
            )

        source_sample_count = int(control["sample_count"])
        for use_index, use in enumerate(
            sorted(
                uses_by_background.get(background_id, ()),
                key=lambda item: (
                    str(item["sample_id"]),
                    str(item["role"]),
                    int(item["rendered_start_sample"]),
                ),
            )
        ):
            target = corrected_by_source[str(use["sample_id"])]
            if str(target["partition"]) != str(control["partition"]):
                raise ValueError("background repair dependency crosses a partition")
            rendered_length = int(use["rendered_end_sample"]) - int(
                use["rendered_start_sample"]
            )
            if rendered_length != int(use["source_duration_sample"]):
                raise ValueError("background dependency duration does not match its rendered span")
            if Path(str(use["source_audio"])).resolve() != Path(str(control["audio"])).resolve():
                raise ValueError("background dependency source audio does not match its control")
            for event in sorted(
                events_by_background[background_id], key=lambda item: int(item["event_index"])
            ):
                mapped = map_event_to_rendered_audio(
                    event_start_sample=int(event["start_sample"]),
                    event_end_sample=int(event["end_sample"]),
                    source_sample_count=source_sample_count,
                    source_offset_sample=int(use["source_offset_sample"]),
                    rendered_start_sample=int(use["rendered_start_sample"]),
                    rendered_end_sample=int(use["rendered_end_sample"]),
                )
                mapping_id = (
                    f"{use['sample_id']}::{use['role']}::{background_id}::"
                    f"use{use_index:02d}::{event['event_id']}"
                )
                dependency_mappings.append(
                    {
                        "schema": DEPENDENCY_MAPPING_SCHEMA,
                        "mapping_id": mapping_id,
                        "background_id": background_id,
                        "source_id": control_source_id,
                        "target_source_id": str(use["sample_id"]),
                        "partition": str(use["partition"]),
                        "role": str(use["role"]),
                        "event_id": str(event["event_id"]),
                        "source_sample_count": source_sample_count,
                        "source_offset_sample": int(use["source_offset_sample"]),
                        "rendered_start_sample": int(use["rendered_start_sample"]),
                        "rendered_end_sample": int(use["rendered_end_sample"]),
                        "mapped_intervals": mapped,
                    }
                )
                for occurrence_index, interval in enumerate(mapped):
                    register_placement(
                        event=event,
                        target_source_id=str(use["sample_id"]),
                        role=str(use["role"]),
                        occurrence_index=occurrence_index,
                        mapped_start=int(interval["mapped_start_sample"]),
                        mapped_end=int(interval["mapped_end_sample"]),
                        mapping_id=mapping_id,
                        tile_index=int(interval["tile_index"]),
                    )

    for source_id, operations in placement_operations.items():
        row = corrected_by_source[source_id]
        _relabel(
            row=row,
            operations=operations,
            expected_label="background",
            new_label="speech",
            label_source="manual_background_speech_repair_propagated_v1",
            changed_spans=changed_spans,
        )
        row["canonical_repair_gate"] = str(background_speech_repair_gate)
        row["canonical_repair_contract"] = (
            "manual_prediction_background_and_contaminated_asset_repair_v1"
        )

    corrected = [corrected_by_source[str(row["source_id"])] for row in original_rows]
    dataset = _validate_sources(corrected)
    after_counts = _frame_counts(corrected)
    if dataset["max_core_use_count"] != 1:
        raise ValueError("canonical r4 repair core identity was reused")

    feature_labels: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    for source in corrected:
        labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
        feature_labels.append(
            {
                "audio_id": source["source_id"],
                "source": "scorer_v10_corrected_canonical_r4",
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
    sources_path = output_dir / "canonical_sources.jsonl"
    labels_path = output_dir / "feature_cache_labels.jsonl"
    audio_manifest_path = output_dir / "audio_manifest.json"
    prediction_repairs_path = output_dir / "prediction_background_repairs.jsonl"
    dependency_mappings_path = output_dir / "dependency_mappings.jsonl"
    placements_path = output_dir / "repair_placements.jsonl"
    changed_spans_path = output_dir / "changed_spans.jsonl"
    _write_jsonl(sources_path, corrected)
    _write_jsonl(labels_path, feature_labels)
    _write_jsonl(prediction_repairs_path, prediction_repairs)
    _write_jsonl(dependency_mappings_path, dependency_mappings)
    _write_jsonl(placements_path, placements)
    _write_jsonl(changed_spans_path, changed_spans)
    audio_manifest_path.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    control_source_ids = sorted(controls_by_background.values())
    dependent_source_ids = sorted(
        {str(row["target_source_id"]) for row in dependency_mappings if row["mapped_intervals"]}
    )
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_canonical_sources": str(canonical_sources),
        "input_canonical_sources_sha256": _sha256(canonical_sources),
        "prediction_audits": prediction_evidence,
        "prediction_background_verdict_count": sum(
            int(row["canonical_background_verdict_count"])
            for row in prediction_evidence
        ),
        "prediction_background_exact_span_count": len(prediction_repairs),
        "prediction_background_unique_source_count": len(background_ops),
        "prediction_background_unique_frame_count": len(unique_background_frames),
        "background_speech_repair_gate": str(background_speech_repair_gate),
        "background_speech_repair_gate_sha256": _sha256(
            background_speech_repair_gate
        ),
        "background_speech_repair_event_count": len(events),
        "background_speech_repair_asset_count": len(affected_background_ids),
        "background_speech_repair_asset_ids": sorted(affected_background_ids),
        "background_speech_control_source_ids": control_source_ids,
        "composite_source_manifest": str(composite_source_manifest),
        "composite_source_manifest_sha256": _sha256(composite_source_manifest),
        "active_dependency_use_count": sum(
            len(values) for values in uses_by_background.values()
        ),
        "active_dependency_source_ids": dependent_source_ids,
        "inactive_dependency_source_ids": inactive_dependency_ids,
        "dependency_mapping_count": len(dependency_mappings),
        "repair_placement_count": len(placements),
        "repair_registered_core_count": sum(
            bool(row["core_registered"]) for row in placements
        ),
        "repair_no_label_change_placement_count": sum(
            not bool(row["core_registered"]) for row in placements
        ),
        "repair_background_to_speech_sample_count": sum(
            int(row["background_label_change_sample_count"]) for row in placements
        ),
        "removed_core_count": len(removed_core_ids),
        "removed_core_ids": sorted(removed_core_ids),
        "changed_span_count": len(changed_spans),
        "canonical_frame_counts_before": dict(before_counts),
        "canonical_frame_counts_after": dict(after_counts),
        "canonical_frame_count_delta": {
            label: int(after_counts[label] - before_counts[label])
            for label in ("speech", "background", "unsure")
        },
        "dataset": dataset,
        "canonical_sources": str(sources_path),
        "canonical_sources_sha256": _sha256(sources_path),
        "feature_cache_labels": str(labels_path),
        "feature_cache_labels_sha256": _sha256(labels_path),
        "audio_manifest": str(audio_manifest_path),
        "audio_manifest_sha256": _sha256(audio_manifest_path),
        "prediction_background_repairs": str(prediction_repairs_path),
        "prediction_background_repairs_sha256": _sha256(prediction_repairs_path),
        "dependency_mappings": str(dependency_mappings_path),
        "dependency_mappings_sha256": _sha256(dependency_mappings_path),
        "repair_placements": str(placements_path),
        "repair_placements_sha256": _sha256(placements_path),
        "changed_spans": str(changed_spans_path),
        "changed_spans_sha256": _sha256(changed_spans_path),
        "replacement_audit_source_ids": [
            *control_source_ids,
            *[value for value in dependent_source_ids if value not in control_source_ids],
        ],
        "audio_bytes_changed": False,
        "source_identity_changed": False,
        "partition_identity_changed": False,
        "unsure_training_mapping": -100,
        "replacement_audit_required": True,
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
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--prediction-audit-dir", action="append", required=True)
    parser.add_argument("--background-speech-repair-gate", required=True)
    parser.add_argument("--composite-source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_repairs(
                canonical_sources=Path(args.canonical_sources),
                prediction_audit_dirs=[
                    Path(value) for value in args.prediction_audit_dir
                ],
                background_speech_repair_gate=Path(
                    args.background_speech_repair_gate
                ),
                composite_source_manifest=Path(args.composite_source_manifest),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
