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


DISPLAY_DECISIONS = {"keep", "drop"}
TARGET_DECISION_VERSION = "cueqc_display_binary_v1"


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


def _broadcast_cluster_labels(
    clusters: list[Mapping[str, Any]],
    cluster_labels: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand cluster-level display_decision into per-sample pseudo-labels.

    Reads the ``cueqc_cluster_labels.jsonl`` exported from the audit page (one
    row per cluster with ``display_decision``) and broadcasts only explicit
    binary keep/drop seed labels. Audit rows marked mixed/skip abstain by leaving
    ``display_decision`` empty; they remain review metadata and do not enter
    cold-start training.
    """
    decision_by_cluster: dict[str, str] = {}
    reason_by_cluster: dict[str, str] = {}
    confirmed_merge_target_by_cluster: dict[str, str] = {}
    for row in cluster_labels:
        schema = str(row.get("schema") or "").strip()
        label_source = str(row.get("label_source") or row.get("source") or "").strip().lower()
        if schema.startswith("cueqc_ai_suggestion") or label_source in {
            "ai_suggestion",
            "grok_suggestion",
            "gemini_suggestion",
            "llm_suggestion",
        }:
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        seed_action = str(row.get("seed_action") or "").strip()
        decision = str(row.get("display_decision") or "").strip()
        merge_action = str(row.get("merge_action") or "").strip()
        merge_target = str(
            row.get("merge_target_cluster_id")
            or row.get("tail_merge_target_cluster_id")
            or row.get("tail_merge_suggested_target_cluster_id")
            or ""
        ).strip()
        merge_confirmed = row.get("merge_confirmed") is True or merge_action in {
            "confirm_merge",
            "confirmed_merge",
        }
        if cluster_id and merge_target and merge_confirmed and row.get("training_label_included") is not False:
            confirmed_merge_target_by_cluster[cluster_id] = merge_target
        if seed_action != "use_seed" or row.get("training_label_included") is False:
            continue
        if cluster_id and decision:
            decision_by_cluster[cluster_id] = decision
            reason_by_cluster[cluster_id] = str(row.get("classification_reason") or "").strip()

    def resolve_decision(cluster_id: str) -> tuple[str, str, str]:
        decision = decision_by_cluster.get(cluster_id, "")
        if decision:
            return decision, cluster_id, "cluster_broadcast"
        seen = {cluster_id}
        target = confirmed_merge_target_by_cluster.get(cluster_id, "")
        while target and target not in seen:
            seen.add(target)
            decision = decision_by_cluster.get(target, "")
            if decision:
                return decision, target, "cluster_confirmed_merge_broadcast"
            target = confirmed_merge_target_by_cluster.get(target, "")
        return "", "", ""

    expanded: list[dict[str, Any]] = []
    for row in clusters:
        sample_id = str(row.get("sample_id") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not sample_id or not cluster_id:
            continue
        cluster_decision, decision_cluster_id, label_source = resolve_decision(cluster_id)
        if not cluster_decision:
            continue
        final_decision = _valid_label(cluster_decision, DISPLAY_DECISIONS, "")
        if not final_decision:
            continue

        expanded.append({
            "sample_id": sample_id,
            "cluster_id": cluster_id,
            "display_decision": final_decision,
            "notes": reason_by_cluster.get(decision_cluster_id, ""),
            "label_source": label_source,
            "cluster_display_decision": cluster_decision,
            "decision_cluster_id": decision_cluster_id,
            "merged_into_cluster_id": decision_cluster_id if decision_cluster_id != cluster_id else "",
        })
    return expanded


def _valid_label(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized in allowed else fallback


def _cluster_consistency(labels: list[dict[str, Any]]) -> dict[str, Any]:
    if not labels:
        return {"count": 0, "agreement": 0.0, "low_consistency": True}
    display_counts = Counter(str(row.get("display_decision") or "") for row in labels)
    agreement = max(display_counts.values() or [0]) / max(1, sum(display_counts.values()))
    return {
        "count": len(labels),
        "agreement": round(agreement, 4),
        "low_consistency": agreement < 0.67,
        "display_counts": dict(display_counts.most_common()),
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
        display_decision = _valid_label(label.get("display_decision"), DISPLAY_DECISIONS, "")
        if not display_decision:
            skipped.append({"sample_id": sample_id, "cluster_id": cluster_id, "reason": "missing_display_decision"})
            counters["missing_display_decision"] += 1
            continue
        cluster_info = consistency.get(cluster_id, {"agreement": 1.0, "low_consistency": False})
        low_consistency = float(cluster_info.get("agreement") or 0.0) < min_cluster_agreement
        record = {
            "schema": "cueqc_training_record_v1",
            "sample_id": sample_id,
            "cluster_id": cluster_id,
            "features": {
                "text_features": row.get("text_features", {}),
                "cue_features": row.get("cue_features", {}),
                "boundary": row.get("boundary", {}),
                "adjacency": row.get("adjacency", {}),
                "asr_signals": row.get("asr_signals", {}),
                "subtitle_timing": row.get("subtitle_timing", {}),
            },
            "text": row.get("text", ""),
            "raw_text": row.get("raw_text", ""),
            "audio": row.get("audio", {}),
            "targets": {
                "display_decision": display_decision,
                "display_label": 0 if display_decision == "drop" else 1,
            },
            "label_meta": {
                "notes": str(label.get("notes") or ""),
                "label_source": str(label.get("label_source") or "manual"),
                "cluster_display_decision": str(label.get("cluster_display_decision") or ""),
                "decision_cluster_id": str(label.get("decision_cluster_id") or cluster_id),
                "merged_into_cluster_id": str(label.get("merged_into_cluster_id") or ""),
                "low_cluster_consistency": bool(low_consistency),
                "cluster_agreement": float(cluster_info.get("agreement") or 0.0),
                "coarse_cluster_seed": str(label.get("label_source") or "manual")
                in {"cluster_broadcast", "cluster_confirmed_merge_broadcast"},
            },
        }
        records.append(record)
        counters[f"display:{display_decision}"] += 1
    summary = {
        "schema": "cueqc_training_compile_summary_v1",
        "target_decision_version": TARGET_DECISION_VERSION,
        "target_labels": {
            "display_decision": sorted(DISPLAY_DECISIONS),
        },
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
    clusters = read_jsonl(Path(args.clusters))
    cluster_label_rows = read_jsonl(Path(args.cluster_labels))
    manual_labels = _broadcast_cluster_labels(clusters, cluster_label_rows)
    records, skipped, summary = compile_records(
        clusters=clusters,
        manual_labels=manual_labels,
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
            "cluster_labels_path": args.cluster_labels,
            "train_path": str(train_path),
            "skipped_path": str(skipped_path),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"train={train_path}")
    print(f"records={len(records)} skipped={len(skipped)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile cluster-level CueQC keep/drop labels into training JSONL.")
    parser.add_argument("--clusters", required=True, help="cueqc_clusters.jsonl")
    parser.add_argument(
        "--cluster-labels",
        required=True,
        help="cueqc_cluster_labels.jsonl; broadcasts only explicit keep/drop seed labels, while mixed/skip rows abstain",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-cluster-agreement", type=float, default=0.67)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_cluster_agreement <= 1.0:
        parser.error("--min-cluster-agreement must be in [0, 1]")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
