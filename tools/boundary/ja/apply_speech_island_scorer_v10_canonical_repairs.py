#!/usr/bin/env python3
"""Apply audited span repairs to Scorer v10 canonical sources."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from tools.boundary.ja.compile_speech_island_scorer_v10_canonical import (  # noqa: E402
    CANONICAL_LABELS,
    CANONICAL_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    _validate_sources,
    _write_jsonl,
    canonical_frame_labels,
)


GATE_SCHEMA = "speech_scorer_v10_canonical_span_repair_gate_v1"
DECISION_SCHEMA = "speech_scorer_v10_canonical_span_repair_item_v1"
SUMMARY_SCHEMA = "speech_scorer_v10_corrected_canonical_summary_v1"
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_gate(path: Path, *, canonical_sources: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8-sig"))
    if gate.get("schema") != GATE_SCHEMA:
        raise ValueError("invalid Scorer v10 canonical span-repair gate schema")
    canonical_sha256 = hashlib.sha256(canonical_sources.read_bytes()).hexdigest()
    if gate.get("canonical_sources_sha256") != canonical_sha256:
        raise ValueError("span-repair gate is bound to another canonical manifest")
    if gate.get("complete") is not True or gate.get("canonical_recompile_ready") is not True:
        raise ValueError("span-repair gate is incomplete")
    return gate


def apply_repairs(
    *,
    canonical_sources: Path,
    repair_gate: Path,
    output_dir: Path,
    previous_audit_manifest: Path | None = None,
) -> dict[str, Any]:
    gate = _load_gate(repair_gate, canonical_sources=canonical_sources)
    decisions_path = Path(str(gate.get("decisions") or ""))
    decisions: dict[str, dict[str, Any]] = {}
    for row in _rows(decisions_path):
        if row.get("schema") != DECISION_SCHEMA:
            raise ValueError("invalid Scorer v10 canonical span-repair decision schema")
        span_id = str(row.get("span_id") or "")
        if not span_id or span_id in decisions:
            raise ValueError("span-repair decisions require unique span_id values")
        if str(row.get("verdict") or "") not in CANONICAL_LABELS:
            raise ValueError("span-repair decision has an invalid canonical label")
        decisions[span_id] = row

    quarantined_background_ids = {
        str(value) for value in gate.get("quarantined_background_ids") or ()
    }
    corrected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    changed_spans: list[dict[str, Any]] = []
    ignored_core_ids: set[str] = set()
    for original in _rows(canonical_sources):
        row = copy.deepcopy(original)
        source_id = str(row["source_id"])
        row_role = str(row["row_role"])
        row_background_ids = {str(value) for value in row.get("background_source_ids") or ()}
        matched_quarantine = sorted(row_background_ids & quarantined_background_ids)
        if matched_quarantine:
            dropped.append(
                {
                    "source_id": source_id,
                    "partition": row["partition"],
                    "row_role": row_role,
                    "reason": (
                        "quarantined_all_background_control"
                        if row_role == "all_background"
                        else "quarantined_background_augmentation"
                    ),
                    "background_ids": matched_quarantine,
                    "core_ids": list(row.get("core_ids") or ()),
                    "duration_s": float(row["duration_s"]),
                    "background_type": str(row.get("background_type") or ""),
                    "additive_overlay": row.get("additive_overlay") is not None,
                }
            )
            ignored_core_ids.update(str(value) for value in row.get("core_ids") or ())
            continue

        definite_core_ids = [str(value) for value in row.get("core_ids") or ()]
        row_ignored_cores: list[dict[str, str]] = []
        for span_index, span in enumerate(row["canonical_spans"]):
            span_id = f"{source_id}::span{span_index:02d}"
            decision = decisions.get(span_id)
            if decision is None:
                continue
            if (
                decision.get("source_id") != source_id
                or int(decision["start_sample"]) != int(span["start_sample"])
                or int(decision["end_sample"]) != int(span["end_sample"])
                or decision.get("original_label") != span.get("label")
            ):
                raise ValueError(f"span-repair decision does not match canonical span: {span_id}")
            verdict = str(decision["verdict"])
            original_label = str(span["label"])
            if verdict == original_label:
                continue
            if original_label == "background":
                background_id = str(span.get("background_id") or "")
                if background_id not in quarantined_background_ids:
                    raise ValueError(
                        "changed background spans must quarantine their source asset"
                    )
            core_id = str(span.get("core_id") or "")
            if original_label == "speech" and verdict != "speech" and core_id:
                definite_core_ids = [value for value in definite_core_ids if value != core_id]
                ignored_core_ids.add(core_id)
                row_ignored_cores.append(
                    {"core_id": core_id, "manual_label": verdict, "span_id": span_id}
                )
            span["label"] = verdict
            span["label_source"] = "manual_canonical_span_repair_v1"
            span["manual_original_label"] = original_label
            span["manual_note"] = str(decision.get("note") or "")
            changed_spans.append(
                {
                    "span_id": span_id,
                    "source_id": source_id,
                    "original_label": original_label,
                    "label": verdict,
                    "core_id": core_id,
                    "background_id": str(span.get("background_id") or ""),
                }
            )
        row["core_ids"] = definite_core_ids
        if row_ignored_cores:
            row["ignored_core_ids"] = row_ignored_cores
        row["canonical_repair_gate"] = str(repair_gate)
        row["canonical_repair_contract"] = "manual_span_repair_and_asset_quarantine_v1"
        if row_role == "speech" and (
            not definite_core_ids
            or not any(span["label"] == "speech" for span in row["canonical_spans"])
        ):
            dropped.append(
                {
                    "source_id": source_id,
                    "partition": row["partition"],
                    "row_role": row_role,
                    "reason": "no_definite_speech_after_manual_repair",
                    "background_ids": [],
                    "core_ids": list(original.get("core_ids") or ()),
                    "duration_s": float(row["duration_s"]),
                    "background_type": str(row.get("background_type") or ""),
                    "additive_overlay": row.get("additive_overlay") is not None,
                }
            )
            ignored_core_ids.update(str(value) for value in original.get("core_ids") or ())
            continue
        corrected.append(row)

    dataset_summary = _validate_sources(corrected)
    repair_source_ids = sorted(
        {
            str(decision["source_id"])
            for decision in decisions.values()
            if any(row["source_id"] == decision["source_id"] for row in corrected)
        }
    )
    replacement_audit_source_ids: list[str] = list(repair_source_ids)
    if previous_audit_manifest is not None:
        previously_audited = {
            str(row["source_id"]) for row in _rows(previous_audit_manifest)
        }
        excluded = previously_audited | set(replacement_audit_source_ids)

        def background_family(value: str) -> str:
            text = value.lower()
            if "speech_fragment" in text:
                return "semantic_leakage_risk"
            if any(
                token in text
                for token in ("music", "impact", "mechan", "vehicle", "noise")
            ):
                return "music_impact_noise"
            if "kiss" in text:
                return "kiss"
            if any(
                token in text for token in ("moan", "groan", "cry", "sob", "vocal")
            ):
                return "moan_cry_vocal"
            if "breath" in text:
                return "breathing"
            if any(token in text for token in ("silence", "pause", "empty")):
                return "silence"
            return "other"

        for removed in dropped:
            pool = [
                row
                for row in corrected
                if row["partition"] == removed["partition"]
                and row["row_role"] == removed["row_role"]
                and row["source_id"] not in excluded
            ]
            if not pool:
                raise ValueError("corrected canonical data has no replacement audit source")
            if removed["row_role"] == "all_background":
                family = background_family(str(removed.get("background_type") or ""))
                selected = min(
                    pool,
                    key=lambda row: (
                        background_family(str(row.get("background_type") or "")) != family,
                        abs(
                            float(row["duration_s"])
                            - float(removed.get("duration_s") or 0.0)
                        ),
                        str(row["source_id"]),
                    ),
                )
            else:
                selected = min(
                    pool,
                    key=lambda row: (
                        (row.get("additive_overlay") is not None)
                        != bool(removed.get("additive_overlay")),
                        abs(
                            float(row["duration_s"])
                            - float(removed.get("duration_s") or 0.0)
                        ),
                        str(row["source_id"]),
                    ),
                )
            replacement_audit_source_ids.append(str(selected["source_id"]))
            excluded.add(str(selected["source_id"]))
    label_counts: Counter[str] = Counter()
    feature_labels: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    for source in corrected:
        labels = canonical_frame_labels(source, frame_hop_s=FRAME_HOP_S)
        label_counts.update(
            background=int(np.sum(labels == CANONICAL_LABELS["background"])),
            speech=int(np.sum(labels == CANONICAL_LABELS["speech"])),
            unsure=int(np.sum(labels == CANONICAL_LABELS["unsure"])),
        )
        weights = np.where(labels == CANONICAL_LABELS["unsure"], 0.0, 1.0)
        feature_labels.append(
            {
                "audio_id": source["source_id"],
                "source": "scorer_v10_corrected_canonical_full_source",
                "duration_s": source["duration_s"],
                "text": "",
                "teacher_segments": {},
                "frame_hop_s": FRAME_HOP_S,
                "speech_frames": (labels == CANONICAL_LABELS["speech"]).astype(int).tolist(),
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
    dropped_path = output_dir / "dropped_sources.jsonl"
    changed_path = output_dir / "changed_spans.jsonl"
    _write_jsonl(sources_path, corrected)
    _write_jsonl(labels_path, feature_labels)
    _write_jsonl(dropped_path, dropped)
    _write_jsonl(changed_path, changed_spans)
    audio_manifest_path.write_text(
        json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "canonical_label_schema": CANONICAL_LABEL_SCHEMA,
        "input_canonical_sources": str(canonical_sources),
        "input_canonical_sources_sha256": hashlib.sha256(
            canonical_sources.read_bytes()
        ).hexdigest(),
        "repair_gate": str(repair_gate),
        "repair_gate_sha256": hashlib.sha256(repair_gate.read_bytes()).hexdigest(),
        "quarantined_background_ids": sorted(quarantined_background_ids),
        "ignored_core_ids": sorted(ignored_core_ids),
        "dropped_source_count": len(dropped),
        "dropped_source_reason_counts": dict(
            sorted(Counter(row["reason"] for row in dropped).items())
        ),
        "changed_span_count": len(changed_spans),
        "repair_source_ids": repair_source_ids,
        "replacement_audit_source_ids": replacement_audit_source_ids,
        "previous_audit_manifest": str(previous_audit_manifest or ""),
        "canonical_frame_counts": dict(label_counts),
        "dataset": dataset_summary,
        "canonical_sources": str(sources_path),
        "feature_cache_labels": str(labels_path),
        "audio_manifest": str(audio_manifest_path),
        "dropped_sources": str(dropped_path),
        "changed_spans": str(changed_path),
        "replacement_audit_required": True,
        "feature_cache_ready": False,
        "training_ready": False,
        "promotion_ready": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--repair-gate", required=True)
    parser.add_argument("--previous-audit-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            apply_repairs(
                canonical_sources=Path(args.canonical_sources),
                repair_gate=Path(args.repair_gate),
                output_dir=Path(args.output_dir),
                previous_audit_manifest=(
                    Path(args.previous_audit_manifest)
                    if args.previous_audit_manifest
                    else None
                ),
            ),
            ensure_ascii=False,
        )
    )
