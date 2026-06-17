#!/usr/bin/env python3
"""Compile CueQC v3-Fusion Stage 2a training features.

Stage 2a deliberately does not trust unaudited high-confidence drop pseudo
labels. It combines:

* the original cold-start labeled feature bundle;
* manual false-drop audit labels for sampled drop predictions;
* high-confidence keep pseudo labels.

Manual audit labels take priority over every other source. Duplicate sample IDs
are emitted once.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FEATURE_SCHEMA = "cueqc_mamba_v3_fusion_features"
FALSE_DROP_AUDIT_SCHEMA = "cueqc_false_drop_audit_label_v1"
PSEUDO_LABEL_SCHEMA = "cueqc_pseudo_label_v3_fusion_v1"


@dataclass(frozen=True)
class LabelSource:
    label: int
    display_decision: str
    label_source: str
    priority: int
    source_row: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _load_feature_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != FEATURE_SCHEMA:
        raise ValueError(f"unexpected feature schema in {path}: {bundle.get('schema')!r}")
    samples = list(bundle.get("samples") or [])
    labels = bundle.get("labels")
    meta = list(bundle.get("meta") or [])
    if labels is None:
        raise ValueError(f"missing labels tensor in {path}")
    if len(samples) != len(labels.tolist()) or len(samples) != len(meta):
        raise ValueError(f"sample/label/meta length mismatch in {path}")
    return bundle


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or "").strip()


def _confidence(row: Mapping[str, Any], key: str, fallback: str = "confidence") -> float:
    try:
        return float(row.get(key) if row.get(key) is not None else row.get(fallback) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _compatible_config(base: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    keys = (
        "asr_dim",
        "token_dim",
        "decoder_dim",
        "structured_dim",
        "feature_source",
        "uses_bge",
        "text_embedding",
        "token_feature_names",
        "decoder_feature_names",
        "structured_feature_names",
    )
    return all(base.get(key) == other.get(key) for key in keys)


def _manual_label(row: dict[str, Any]) -> LabelSource | None:
    decision = str(row.get("manual_decision") or "").strip()
    if decision == "drop_ok":
        return LabelSource(
            label=0,
            display_decision="drop",
            label_source="manual_false_drop_audit_drop_ok",
            priority=100,
            source_row=row,
        )
    if decision == "false_drop_keep" or row.get("is_false_drop") is True:
        return LabelSource(
            label=1,
            display_decision="keep",
            label_source="manual_false_drop_audit_false_drop_keep",
            priority=100,
            source_row=row,
        )
    return None


def _pseudo_keep_label(row: dict[str, Any], *, min_keep_confidence: float) -> LabelSource | None:
    display = str(row.get("display_hint") or "").strip()
    targets = row.get("targets") if isinstance(row.get("targets"), Mapping) else {}
    target_display = str(targets.get("display_decision") or "").strip()
    if display != "keep" and target_display != "keep":
        return None
    if _confidence(row, "display_prob_keep") < min_keep_confidence:
        return None
    return LabelSource(
        label=1,
        display_decision="keep",
        label_source="cueqc_v3_high_conf_keep_pseudo",
        priority=50,
        source_row=row,
    )


def _put_label(labels: dict[str, LabelSource], sample_id: str, label: LabelSource) -> None:
    current = labels.get(sample_id)
    if current is None or label.priority >= current.priority:
        labels[sample_id] = label


def _copy_sample(sample: Mapping[str, Any], label: int) -> dict[str, Any]:
    copied = dict(sample)
    copied["__label__"] = int(label)
    return copied


def _meta_with_label(meta: Mapping[str, Any], label: LabelSource | None, fallback_label: int) -> dict[str, Any]:
    copied = dict(meta)
    copied["display_hint"] = label.display_decision if label else ("drop" if int(fallback_label) == 0 else "keep")
    copied["cueqc_stage2a_label"] = int(label.label if label else fallback_label)
    copied["cueqc_stage2a_label_source"] = label.label_source if label else "cold_start_seed"
    if label:
        row = label.source_row
        copied["cueqc_stage2a_manual_decision"] = row.get("manual_decision", "")
        copied["cueqc_stage2a_reason_tags"] = row.get("reason_tags", [])
        copied["cueqc_stage2a_notes"] = row.get("notes", "")
        copied["cueqc_stage2a_confidence"] = row.get("confidence", row.get("display_prob_keep", row.get("display_prob_drop")))
    return copied


def compile_stage2a(
    *,
    cold_start_features: Path,
    full_features: Path,
    pseudo_labels: Path,
    false_drop_audit_labels: Path | list[Path],
    output: Path,
    summary_path: Path,
    min_keep_confidence: float,
) -> dict[str, Any]:
    cold_bundle = _load_feature_bundle(cold_start_features)
    full_bundle = _load_feature_bundle(full_features)
    cold_config = dict(cold_bundle.get("feature_config") or {})
    full_config = dict(full_bundle.get("feature_config") or {})
    if not _compatible_config(cold_config, full_config):
        raise ValueError("cold-start and full feature bundles have incompatible feature_config")

    audit_paths = (
        [false_drop_audit_labels]
        if isinstance(false_drop_audit_labels, Path)
        else list(false_drop_audit_labels)
    )
    audit_rows: list[dict[str, Any]] = []
    audit_path_summaries: list[dict[str, Any]] = []
    for audit_path in audit_paths:
        rows = read_jsonl(audit_path)
        audit_rows.extend(rows)
        decisions = Counter(str(row.get("manual_decision") or "") for row in rows)
        false_drop_count = sum(
            1
            for row in rows
            if row.get("manual_decision") == "false_drop_keep" or row.get("is_false_drop") is True
        )
        audit_path_summaries.append(
            {
                "path": str(audit_path),
                "records": len(rows),
                "manual_decisions": dict(decisions.most_common()),
                "false_drop_count": int(false_drop_count),
                "false_drop_rate": round(false_drop_count / len(rows), 4) if rows else 0.0,
            }
        )
    pseudo_rows = read_jsonl(pseudo_labels)
    selected_full_labels: dict[str, LabelSource] = {}
    counters: Counter[str] = Counter()

    for row in audit_rows:
        sid = _sample_id(row)
        if not sid:
            counters["audit_missing_sample_id"] += 1
            continue
        if row.get("schema") not in {FALSE_DROP_AUDIT_SCHEMA, None, ""}:
            counters["audit_schema_other"] += 1
        label = _manual_label(row)
        if label is None:
            counters[f"audit_skipped:{row.get('manual_decision') or 'missing'}"] += 1
            continue
        _put_label(selected_full_labels, sid, label)
        counters[f"audit_selected:{label.display_decision}"] += 1

    for row in pseudo_rows:
        sid = _sample_id(row)
        if not sid:
            counters["pseudo_missing_sample_id"] += 1
            continue
        display = str(row.get("display_hint") or "").strip()
        if row.get("schema") not in {PSEUDO_LABEL_SCHEMA, None, ""}:
            counters["pseudo_schema_other"] += 1
        if display == "drop":
            counters["pseudo_skipped_unaudited_drop"] += 1
            continue
        label = _pseudo_keep_label(row, min_keep_confidence=min_keep_confidence)
        if label is None:
            counters[f"pseudo_skipped:{display or 'missing'}"] += 1
            continue
        if sid not in selected_full_labels:
            _put_label(selected_full_labels, sid, label)
            counters["pseudo_selected:keep"] += 1
        else:
            counters["pseudo_overridden_by_manual"] += 1

    full_meta = list(full_bundle.get("meta") or [])
    full_samples = list(full_bundle.get("samples") or [])
    full_index_by_sample = {
        _sample_id(meta): index
        for index, meta in enumerate(full_meta)
        if _sample_id(meta)
    }

    output_samples: list[dict[str, Any]] = []
    output_labels: list[int] = []
    output_meta: list[dict[str, Any]] = []
    emitted: set[str] = set()

    missing_full_rows: list[dict[str, Any]] = []
    for sid, label in selected_full_labels.items():
        index = full_index_by_sample.get(sid)
        if index is None:
            missing_full_rows.append({"sample_id": sid, "label_source": label.label_source})
            counters["selected_missing_from_full_features"] += 1
            continue
        output_samples.append(_copy_sample(full_samples[index], label.label))
        output_labels.append(label.label)
        output_meta.append(_meta_with_label(full_meta[index], label, label.label))
        emitted.add(sid)
        counters[f"emitted_full:{label.label_source}"] += 1

    cold_labels = cold_bundle["labels"].tolist()
    for sample, raw_label, meta in zip(cold_bundle["samples"], cold_labels, cold_bundle["meta"]):
        sid = _sample_id(meta)
        if sid and sid in emitted:
            counters["cold_duplicate_skipped_manual_or_pseudo"] += 1
            continue
        selected = selected_full_labels.get(sid)
        label_value = int(selected.label if selected else raw_label)
        if label_value not in {0, 1}:
            counters["cold_skipped_unlabeled"] += 1
            continue
        output_samples.append(_copy_sample(sample, label_value))
        output_labels.append(label_value)
        output_meta.append(_meta_with_label(meta, selected, label_value))
        if sid:
            emitted.add(sid)
        counters[f"emitted_cold:{'manual_override' if selected else 'seed'}"] += 1

    label_counts = Counter(output_labels)
    if any(label not in {0, 1} for label in output_labels):
        raise RuntimeError("compiled Stage 2a bundle contains unlabeled samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_config = dict(cold_config)
    feature_config.update(
        {
            "source_mode": "cueqc_v3_stage2a_self_training",
            "cold_start_features": str(cold_start_features),
            "full_features": str(full_features),
            "pseudo_labels": str(pseudo_labels),
            "false_drop_audit_labels": [str(path) for path in audit_paths],
            "min_keep_confidence": float(min_keep_confidence),
        }
    )
    payload = {
        "schema": FEATURE_SCHEMA,
        "version": 3,
        "samples": output_samples,
        "labels": torch.tensor(output_labels, dtype=torch.long),
        "meta": output_meta,
        "feature_config": feature_config,
        "label_config": {"drop": 0, "keep": 1},
    }
    torch.save(payload, output)

    manual_decisions = Counter(str(row.get("manual_decision") or "") for row in audit_rows)
    false_drops = sum(1 for row in audit_rows if row.get("manual_decision") == "false_drop_keep" or row.get("is_false_drop") is True)
    summary = {
        "schema": "cueqc_stage2a_feature_compile_summary_v1",
        "output": str(output),
        "records": len(output_samples),
        "labels": {"drop": int(label_counts.get(0, 0)), "keep": int(label_counts.get(1, 0))},
        "source_counts": dict(counters.most_common()),
        "audit": {
            "paths": [str(path) for path in audit_paths],
            "path_summaries": audit_path_summaries,
            "records": len(audit_rows),
            "manual_decisions": dict(manual_decisions.most_common()),
            "false_drop_count": int(false_drops),
            "false_drop_rate": round(false_drops / len(audit_rows), 4) if audit_rows else 0.0,
        },
        "policy": {
            "manual_labels_override_all_sources": True,
            "unaudited_drop_pseudo_labels": "skipped",
            "high_conf_keep_pseudo_labels": "included",
            "min_keep_confidence": float(min_keep_confidence),
            "uncertain_manual_labels": "skipped",
        },
        "stage3_boundary_feedback": {
            "planned": True,
            "not_compiled_here": True,
            "candidate_source": "manual_false_drop_audit_drop_ok",
            "candidate_count": int(manual_decisions.get("drop_ok", 0)),
        },
        "missing_full_rows": missing_full_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile CueQC v3-Fusion Stage 2a training feature bundle.")
    parser.add_argument("--cold-start-features", required=True, help="Original labeled cold-start .pt feature bundle.")
    parser.add_argument("--full-features", required=True, help="Full 10-film unlabeled .pt feature bundle.")
    parser.add_argument("--pseudo-labels", required=True, help="cueqc_pseudo_labels.high_conf.jsonl from predict_v3_fusion.py.")
    parser.add_argument(
        "--false-drop-audit-labels",
        action="append",
        required=True,
        help="cueqc_false_drop_audit_labels.jsonl exported by an audit page. Repeat for multiple rounds.",
    )
    parser.add_argument("--output", required=True, help="Output labeled .pt feature bundle.")
    parser.add_argument("--summary", default="", help="Output summary JSON. Defaults to <output-dir>/summary.json.")
    parser.add_argument("--min-keep-confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    if not 0.5 <= args.min_keep_confidence <= 1.0:
        parser.error("--min-keep-confidence must be in [0.5, 1.0]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.parent / "summary.json"
    summary = compile_stage2a(
        cold_start_features=Path(args.cold_start_features),
        full_features=Path(args.full_features),
        pseudo_labels=Path(args.pseudo_labels),
        false_drop_audit_labels=[Path(path) for path in args.false_drop_audit_labels],
        output=output,
        summary_path=summary_path,
        min_keep_confidence=float(args.min_keep_confidence),
    )
    print(f"output={output}")
    print(f"summary={summary_path}")
    print(f"records={summary['records']} labels={summary['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
