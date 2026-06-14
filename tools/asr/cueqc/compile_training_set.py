#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CONTENT_LABELS = {"dialogue", "non_dialogue", "mixed", "uncertain"}
DISPLAY_DECISIONS = {"keep", "drop", "compact", "review"}
ALIGNMENT_POLICIES = {"align", "skip_align_fallback"}
QC_DECISIONS = {"keep", "review", "reject"}


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _label_by_sample(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        out[sample_id] = dict(row)
    return out


def _valid_label(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized in allowed else fallback


def _cluster_consistency(labels: list[dict[str, Any]]) -> dict[str, Any]:
    if not labels:
        return {"count": 0, "agreement": 0.0, "low_consistency": True}
    display_counts = Counter(str(row.get("display_decision") or "") for row in labels)
    content_counts = Counter(str(row.get("content_label") or "") for row in labels)
    qc_counts = Counter(str(row.get("qc_decision") or "") for row in labels)
    agreement = max(display_counts.values() or [0]) / max(1, sum(display_counts.values()))
    return {
        "count": len(labels),
        "agreement": round(agreement, 4),
        "low_consistency": agreement < 0.67,
        "display_counts": dict(display_counts.most_common()),
        "content_counts": dict(content_counts.most_common()),
        "qc_counts": dict(qc_counts.most_common()),
    }


def compile_records(
    *,
    clusters: list[dict[str, Any]],
    manual_labels: list[dict[str, Any]],
    min_cluster_agreement: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels_by_sample = _label_by_sample(manual_labels)
    cluster_label_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manual_labels:
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id:
            cluster_label_rows[cluster_id].append(dict(row))
    consistency = {
        cluster_id: _cluster_consistency(rows)
        for cluster_id, rows in cluster_label_rows.items()
    }
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for row in clusters:
        sample_id = str(row.get("sample_id") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        label = labels_by_sample.get(sample_id)
        if not sample_id:
            skipped.append({"reason": "missing_sample_id", "cluster_id": cluster_id})
            counters["missing_sample_id"] += 1
            continue
        if not label:
            skipped.append({"sample_id": sample_id, "cluster_id": cluster_id, "reason": "missing_manual_label"})
            counters["missing_manual_label"] += 1
            continue
        content_label = _valid_label(label.get("content_label"), CONTENT_LABELS, "uncertain")
        display_decision = _valid_label(label.get("display_decision"), DISPLAY_DECISIONS, "review")
        alignment_policy = _valid_label(label.get("alignment_policy"), ALIGNMENT_POLICIES, "align")
        qc_decision = _valid_label(label.get("qc_decision"), QC_DECISIONS, "review")
        uncertain = content_label == "uncertain" or display_decision == "review" or qc_decision == "review"
        cluster_info = consistency.get(cluster_id, {"agreement": 1.0, "low_consistency": False})
        low_consistency = float(cluster_info.get("agreement") or 0.0) < min_cluster_agreement
        if low_consistency:
            if qc_decision == "reject":
                qc_decision = "review"
            if display_decision == "drop":
                display_decision = "review"
            if content_label != "dialogue":
                content_label = "uncertain"
        hard_reject_target = (
            qc_decision == "reject"
            and not uncertain
            and not low_consistency
            and content_label != "uncertain"
        )
        record = {
            "schema": "cueqc_training_record_v1",
            "sample_id": sample_id,
            "cluster_id": cluster_id,
            "features": {
                "text_features": row.get("text_features", {}),
                "qc": row.get("qc", {}),
                "boundary": row.get("boundary", {}),
                "adjacency": row.get("adjacency", {}),
                "asr_signals": row.get("asr_signals", {}),
                "alignment_diagnostics": row.get("alignment_diagnostics", {}),
            },
            "text": row.get("text", ""),
            "raw_text": row.get("raw_text", ""),
            "audio": row.get("audio", {}),
            "targets": {
                "content_type": content_label,
                "display_hint": display_decision,
                "alignment_policy": alignment_policy if not low_consistency else "align",
                "qc_decision": qc_decision,
                "hard_reject_target": hard_reject_target,
            },
            "label_meta": {
                "notes": str(label.get("notes") or ""),
                "low_cluster_consistency": bool(low_consistency),
                "cluster_agreement": float(cluster_info.get("agreement") or 0.0),
                "uncertain": bool(uncertain or low_consistency),
            },
        }
        records.append(record)
        counters[f"content:{content_label}"] += 1
        counters[f"display:{display_decision}"] += 1
        counters[f"align:{record['targets']['alignment_policy']}"] += 1
        counters[f"qc:{qc_decision}"] += 1
        if hard_reject_target:
            counters["hard_reject_target"] += 1
    summary = {
        "schema": "cueqc_training_compile_summary_v1",
        "clusters": len(clusters),
        "manual_labels": len(manual_labels),
        "records": len(records),
        "skipped": len(skipped),
        "min_cluster_agreement": min_cluster_agreement,
        "counts": dict(counters.most_common()),
        "cluster_consistency": consistency,
    }
    return records, skipped, summary


def run(args: argparse.Namespace) -> int:
    records, skipped, summary = compile_records(
        clusters=read_jsonl(Path(args.clusters)),
        manual_labels=read_jsonl(Path(args.labels)),
        min_cluster_agreement=args.min_cluster_agreement,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "cueqc_train.jsonl"
    skipped_path = output_dir / "cueqc_train_skipped.json"
    summary_path = output_dir / "summary.json"
    write_jsonl(train_path, records)
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary.update(
        {
            "clusters_path": args.clusters,
            "labels_path": args.labels,
            "train_path": str(train_path),
            "skipped_path": str(skipped_path),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"train={train_path}")
    print(f"records={len(records)} skipped={len(skipped)} hard_reject={summary['counts'].get('hard_reject_target', 0)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile manually labeled CueQC clusters into training JSONL.")
    parser.add_argument("--clusters", required=True, help="cueqc_clusters.jsonl")
    parser.add_argument("--labels", required=True, help="cueqc_manual_labels.jsonl exported from audit HTML")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-cluster-agreement", type=float, default=0.67)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_cluster_agreement <= 1.0:
        parser.error("--min-cluster-agreement must be in [0, 1]")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
