#!/usr/bin/env python3
"""Apply audited Scorer v10 fragment atoms to corrected canonical labels."""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.audits.evaluate_scorer_v10_fragment_atomic_repair_audit import (  # noqa: E402
    DECISION_SCHEMA,
    RESULT_SCHEMA as GATE_SCHEMA,
)
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    _write_jsonl,
    canonical_frame_labels,
)


SUMMARY_SCHEMA = "speech_scorer_v10_fragment_atomic_corrected_canonical_summary_v1"
FRAME_HOP_S = 0.02
LABELS = {"speech", "background", "unsure"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_gate(
    path: Path, *, canonical_sources: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = json.loads(path.read_text(encoding="utf-8-sig"))
    if gate.get("schema") != GATE_SCHEMA:
        raise ValueError("invalid Scorer fragment atomic repair gate schema")
    if gate.get("complete") is not True or gate.get("canonical_recompile_ready") is not True:
        raise ValueError("fragment atomic repair gate is incomplete")
    if int(gate.get("relation_violation_count") or 0):
        raise ValueError("fragment atomic repair gate has relation violations")
    if str(gate.get("canonical_sources_sha256") or "") != _sha256(canonical_sources):
        raise ValueError("fragment atomic repair gate is bound to another canonical manifest")

    for path_key, sha_key in (
        ("audit_summary", "audit_summary_sha256"),
        ("fragmentation_audit_manifest", "fragmentation_audit_manifest_sha256"),
        ("fragmentation_manual_verdicts", "fragmentation_manual_verdicts_sha256"),
    ):
        bound_path = Path(str(gate.get(path_key) or ""))
        if not bound_path.is_file() or _sha256(bound_path) != str(gate.get(sha_key) or ""):
            raise ValueError(f"fragment atomic repair evidence changed: {path_key}")

    decisions_path = Path(str(gate.get("decisions") or ""))
    if not decisions_path.is_file():
        raise ValueError("fragment atomic repair decisions are missing")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _rows(decisions_path):
        if row.get("schema") != DECISION_SCHEMA:
            raise ValueError("invalid fragment atomic repair decision schema")
        atomic_id = str(row.get("atomic_id") or "")
        if not atomic_id or atomic_id in seen:
            raise ValueError("fragment atomic decisions require unique atomic_id values")
        seen.add(atomic_id)
        if str(row.get("label") or "") not in LABELS:
            raise ValueError("fragment atomic decision has an invalid label")
        decisions.append(row)
    if len(decisions) != int(gate.get("atomic_unit_count") or -1):
        raise ValueError("fragment atomic decision count does not match its gate")
    return gate, decisions


def _frame_label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source in rows:
        labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
        counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
    return counts


def apply_repairs(
    *, canonical_sources: Path, atomic_repair_gate: Path, output_dir: Path
) -> dict[str, Any]:
    gate, decision_rows = _load_gate(
        atomic_repair_gate, canonical_sources=canonical_sources
    )
    original_rows = _rows(canonical_sources)
    _validate_sources(original_rows)
    original_by_source = {str(row["source_id"]): row for row in original_rows}

    decisions_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    affected_core_ids: set[str] = set()
    for decision in decision_rows:
        source_id = str(decision["source_id"])
        if source_id not in original_by_source:
            raise ValueError(f"fragment decision source is missing: {source_id}")
        decisions_by_source[source_id].append(decision)
        affected_core_ids.add(str(decision.get("core_id") or ""))

    corrected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    changed_atoms: list[dict[str, Any]] = []
    verified_atoms: list[dict[str, Any]] = []
    removed_core_ids: set[str] = set()
    for original in original_rows:
        source_id = str(original["source_id"])
        source_decisions = sorted(
            decisions_by_source.get(source_id, ()),
            key=lambda row: (int(row["start_sample"]), int(row["end_sample"])),
        )
        if not source_decisions:
            corrected.append(copy.deepcopy(original))
            continue

        for left, right in zip(source_decisions, source_decisions[1:]):
            if int(right["start_sample"]) < int(left["end_sample"]):
                raise ValueError(f"fragment atomic decisions overlap: {source_id}")

        decisions_by_span: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for decision in source_decisions:
            span_index = int(decision["canonical_span_index"])
            spans = list(original.get("canonical_spans") or ())
            if span_index < 0 or span_index >= len(spans):
                raise ValueError("fragment atomic decision has an invalid span index")
            span = spans[span_index]
            if (
                str(span.get("label") or "") != "speech"
                or str(span.get("core_id") or "") != str(decision.get("core_id") or "")
                or int(decision["start_sample"]) < int(span["start_sample"])
                or int(decision["end_sample"]) > int(span["end_sample"])
                or int(decision["sample_rate"]) != int(original["sample_rate"])
                or int(decision["sample_count"]) != int(original["sample_count"])
                or str(decision.get("original_canonical_label") or "") != "speech"
            ):
                raise ValueError(
                    f"fragment atomic decision does not match canonical: {decision['atomic_id']}"
                )
            decisions_by_span[span_index].append(decision)

        row = copy.deepcopy(original)
        new_spans: list[dict[str, Any]] = []
        for span_index, raw_span in enumerate(original["canonical_spans"]):
            span_decisions = decisions_by_span.get(span_index, ())
            if not span_decisions:
                new_spans.append(copy.deepcopy(raw_span))
                continue
            boundaries = {
                int(raw_span["start_sample"]), int(raw_span["end_sample"])
            }
            for decision in span_decisions:
                boundaries.add(int(decision["start_sample"]))
                boundaries.add(int(decision["end_sample"]))
            ordered = sorted(boundaries)
            for start_sample, end_sample in zip(ordered, ordered[1:]):
                if end_sample <= start_sample:
                    raise ValueError("fragment atomic repair created an empty span")
                covering = [
                    decision
                    for decision in span_decisions
                    if start_sample >= int(decision["start_sample"])
                    and end_sample <= int(decision["end_sample"])
                ]
                if len(covering) > 1:
                    raise ValueError("fragment atomic repair has overlapping coverage")
                piece = copy.deepcopy(raw_span)
                piece["start_sample"] = start_sample
                piece["end_sample"] = end_sample
                if covering:
                    decision = covering[0]
                    label = str(decision["label"])
                    original_core_id = str(piece.get("core_id") or "")
                    piece["label"] = label
                    piece["label_source"] = str(decision["label_source"])
                    piece["atomic_repair_id"] = str(decision["atomic_id"])
                    piece["manual_original_label"] = "speech"
                    if label != "speech":
                        piece.pop("core_id", None)
                        piece["origin_core_id"] = original_core_id
                        changed_atoms.append(
                            {
                                "schema": "speech_scorer_v10_fragment_atomic_changed_span_v1",
                                "atomic_id": str(decision["atomic_id"]),
                                "source_id": source_id,
                                "partition": str(original["partition"]),
                                "core_id": original_core_id,
                                "start_sample": start_sample,
                                "end_sample": end_sample,
                                "start_frame": int(decision["start_frame"]),
                                "end_frame": int(decision["end_frame"]),
                                "label": label,
                                "label_source": str(decision["label_source"]),
                            }
                        )
                    else:
                        verified_atoms.append(
                            {
                                "schema": "speech_scorer_v10_fragment_atomic_verified_span_v1",
                                "atomic_id": str(decision["atomic_id"]),
                                "source_id": source_id,
                                "partition": str(original["partition"]),
                                "core_id": original_core_id,
                                "start_sample": start_sample,
                                "end_sample": end_sample,
                                "start_frame": int(decision["start_frame"]),
                                "end_frame": int(decision["end_frame"]),
                                "label": "speech",
                                "label_source": str(decision["label_source"]),
                            }
                        )
                new_spans.append(piece)

        row["canonical_spans"] = new_spans
        remaining_core_ids = {
            str(span.get("core_id") or "")
            for span in new_spans
            if str(span.get("label") or "") == "speech" and span.get("core_id")
        }
        original_core_ids = [str(value) for value in original.get("core_ids") or ()]
        row["core_ids"] = [
            core_id for core_id in original_core_ids if core_id in remaining_core_ids
        ]
        removed = [
            core_id for core_id in original_core_ids if core_id not in remaining_core_ids
        ]
        removed_core_ids.update(removed)
        if removed:
            row.setdefault("ignored_core_ids", []).extend(
                {
                    "core_id": core_id,
                    "manual_label": "background_or_unsure",
                    "reason": "no_definite_speech_after_fragment_atomic_repair",
                }
                for core_id in removed
            )
        row["canonical_repair_gate"] = str(atomic_repair_gate)
        row["canonical_repair_contract"] = "manual_fragment_atomic_repair_v1"
        if not row["core_ids"] or not any(
            str(span.get("label") or "") == "speech" for span in new_spans
        ):
            dropped.append(
                {
                    "source_id": source_id,
                    "partition": str(original["partition"]),
                    "row_role": str(original["row_role"]),
                    "reason": "no_definite_speech_after_fragment_atomic_repair",
                    "core_ids": original_core_ids,
                    "duration_s": float(original["duration_s"]),
                }
            )
            continue
        corrected.append(row)

    dataset_summary = _validate_sources(corrected)
    before_counts = _frame_label_counts(original_rows)
    after_counts = _frame_label_counts(corrected)

    feature_labels: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    for source in corrected:
        labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
        feature_labels.append(
            {
                "audio_id": source["source_id"],
                "source": "scorer_v10_fragment_atomic_corrected_canonical",
                "duration_s": source["duration_s"],
                "text": "",
                "teacher_segments": {},
                "frame_hop_s": FRAME_HOP_S,
                "speech_frames": (
                    labels == CANONICAL_LABELS["speech"]
                ).astype(int).tolist(),
                "label_quality": (
                    "negative"
                    if source["row_role"] == "all_background"
                    else "supervised"
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
    changed_path = output_dir / "changed_atomic_spans.jsonl"
    verified_path = output_dir / "verified_speech_atomic_spans.jsonl"
    dropped_path = output_dir / "dropped_sources.jsonl"
    _write_jsonl(sources_path, corrected)
    _write_jsonl(labels_path, feature_labels)
    _write_jsonl(changed_path, changed_atoms)
    _write_jsonl(verified_path, verified_atoms)
    _write_jsonl(dropped_path, dropped)
    audio_manifest_path.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    affected_source_ids = sorted(decisions_by_source)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_canonical_sources": str(canonical_sources),
        "input_canonical_sources_sha256": _sha256(canonical_sources),
        "atomic_repair_gate": str(atomic_repair_gate),
        "atomic_repair_gate_sha256": _sha256(atomic_repair_gate),
        "fragmentation_audit_manifest": str(
            gate.get("fragmentation_audit_manifest") or ""
        ),
        "fragmentation_audit_manifest_sha256": str(
            gate.get("fragmentation_audit_manifest_sha256") or ""
        ),
        "fragmentation_manual_verdicts": str(
            gate.get("fragmentation_manual_verdicts") or ""
        ),
        "fragmentation_manual_verdicts_sha256": str(
            gate.get("fragmentation_manual_verdicts_sha256") or ""
        ),
        "affected_source_count": len(affected_source_ids),
        "affected_source_ids": affected_source_ids,
        "affected_core_count": len(affected_core_ids - {""}),
        "changed_atomic_span_count": len(changed_atoms),
        "verified_speech_atomic_span_count": len(verified_atoms),
        "removed_core_count": len(removed_core_ids),
        "removed_core_ids": sorted(removed_core_ids),
        "dropped_source_count": len(dropped),
        "canonical_frame_counts_before": dict(before_counts),
        "canonical_frame_counts_after": dict(after_counts),
        "canonical_frame_count_delta": {
            label: int(after_counts[label] - before_counts[label])
            for label in ("speech", "background", "unsure")
        },
        "dataset": dataset_summary,
        "canonical_sources": str(sources_path),
        "canonical_sources_sha256": _sha256(sources_path),
        "feature_cache_labels": str(labels_path),
        "audio_manifest": str(audio_manifest_path),
        "changed_atomic_spans": str(changed_path),
        "verified_speech_atomic_spans": str(verified_path),
        "dropped_sources": str(dropped_path),
        "replacement_audit_source_ids": sorted(
            set(affected_source_ids) - {str(row["source_id"]) for row in dropped}
        ),
        "audio_bytes_changed": False,
        "existing_feature_cache_authorized": False,
        "feature_cache_reuse_pending_signature_audit": True,
        "replacement_audit_required": True,
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
    parser.add_argument("--atomic-repair-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_repairs(
                canonical_sources=Path(args.canonical_sources),
                atomic_repair_gate=Path(args.atomic_repair_gate),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
