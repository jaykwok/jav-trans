#!/usr/bin/env python3
"""Audit Scorer v11 supervision topology before another GPU training run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ID = "boundary_acoustic_binary_v12"
SOURCE_SCHEMA = "candidate_island_scorer_v11_canonical_source_v1"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_supervision_distribution_audit_v1"
SOURCE_STATS_SCHEMA = "candidate_island_scorer_v11_source_supervision_stats_v1"
LABEL_ORDER = ("outside_candidate", "inside_candidate", "unsure")
LABEL_SET = set(LABEL_ORDER)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
        "max": max(values) if values else None,
    }


def _source_kind(row: dict[str, Any]) -> str:
    value = str(row.get("source_kind") or "")
    if value:
        return value
    if str(row.get("partition") or "") in {"val", "test"}:
        return "heldout_full_source"
    raise ValueError(f"train source has no source_kind: {row.get('source_id')!r}")


def _validate_source(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "")
    if not source_id:
        raise ValueError("canonical source has no source_id")
    if row.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"wrong canonical source schema: {source_id}")
    if row.get("boundary_serialization_contract_id") != CONTRACT_ID:
        raise ValueError(f"wrong Boundary contract: {source_id}")
    if row.get("training_manifest_allowed") is not True:
        raise ValueError(f"canonical source is not training-ready: {source_id}")
    partition = str(row.get("partition") or "")
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"invalid partition: {source_id}")
    frame_count = int(row.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError(f"invalid frame_count: {source_id}")
    label_frames: Counter[str] = Counter()
    cursor = 0
    spans = row.get("canonical_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"canonical source has no spans: {source_id}")
    for span in spans:
        label = str(span.get("label") or "")
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        if label not in LABEL_SET:
            raise ValueError(f"invalid label: {source_id}:{label!r}")
        if start != cursor or end <= start or end > frame_count:
            raise ValueError(f"non-contiguous canonical spans: {source_id}")
        label_frames[label] += end - start
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"canonical spans do not cover source tail: {source_id}")
    definite = label_frames["inside_candidate"] + label_frames["outside_candidate"]
    presence = [label for label in LABEL_ORDER if label_frames[label] > 0]
    return {
        "schema": SOURCE_STATS_SCHEMA,
        "source_id": source_id,
        "partition": partition,
        "source_kind": _source_kind(row),
        "annotation_provenance": str(row.get("annotation_provenance") or ""),
        "synthetic_composite": bool(row.get("synthetic_composite", False)),
        "frame_count": frame_count,
        "label_frame_counts": {
            label: int(label_frames[label]) for label in LABEL_ORDER
        },
        "label_presence": "+".join(presence),
        "has_both_definite_labels": (
            label_frames["inside_candidate"] > 0
            and label_frames["outside_candidate"] > 0
        ),
        "definite_frame_count": definite,
        "outside_definite_fraction": (
            label_frames["outside_candidate"] / definite if definite else None
        ),
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    outside_fractions: list[float] = []
    mixed = 0
    zero_inside = 0
    zero_outside = 0
    for row in rows:
        counts = row["label_frame_counts"]
        labels.update(counts)
        presence[str(row["label_presence"])] += 1
        mixed += int(bool(row["has_both_definite_labels"]))
        zero_inside += int(int(counts["inside_candidate"]) == 0)
        zero_outside += int(int(counts["outside_candidate"]) == 0)
        fraction = row["outside_definite_fraction"]
        if fraction is not None:
            outside_fractions.append(float(fraction))
    count = len(rows)
    return {
        "source_count": count,
        "label_frame_counts": {
            label: int(labels[label]) for label in LABEL_ORDER
        },
        "label_presence_source_counts": dict(sorted(presence.items())),
        "mixed_inside_outside_source_count": mixed,
        "mixed_inside_outside_source_fraction": mixed / count if count else None,
        "zero_inside_source_count": zero_inside,
        "zero_outside_source_count": zero_outside,
        "outside_definite_fraction_per_source": _distribution(outside_fractions),
    }


def audit_distribution(
    *, canonical_sources: Path, output_dir: Path
) -> dict[str, Any]:
    source_rows = _read_jsonl(canonical_sources)
    if not source_rows:
        raise ValueError("canonical source manifest is empty")
    source_stats: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        stats = _validate_source(row)
        source_id = str(stats["source_id"])
        if source_id in seen:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        source_stats.append(stats)

    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kinds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_real: list[dict[str, Any]] = []
    heldout_real: list[dict[str, Any]] = []
    for row in source_stats:
        partitions[str(row["partition"])].append(row)
        kinds[str(row["source_kind"])].append(row)
        if row["partition"] == "train" and not row["synthetic_composite"]:
            train_real.append(row)
        if row["partition"] in {"val", "test"}:
            heldout_real.append(row)

    train_real_summary = _summarize(train_real)
    heldout_summary = _summarize(heldout_real)
    calibrated = _summarize(
        kinds.get("real_train_full_source_calibrated_dual_evidence", [])
    )
    train_mixed_fraction = train_real_summary["mixed_inside_outside_source_fraction"]
    heldout_mixed_fraction = heldout_summary["mixed_inside_outside_source_fraction"]
    mismatch = bool(
        train_real
        and heldout_real
        and train_mixed_fraction is not None
        and heldout_mixed_fraction is not None
        and float(train_mixed_fraction) < float(heldout_mixed_fraction)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_stats_path = output_dir / "source_stats.jsonl"
    _write_jsonl(source_stats_path, source_stats)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": CONTRACT_ID,
        "canonical_sources": _display(canonical_sources),
        "canonical_sources_sha256": _sha256(canonical_sources),
        "source_stats": _display(source_stats_path),
        "source_stats_sha256": _sha256(source_stats_path),
        "source_count": len(source_stats),
        "partition_summaries": {
            key: _summarize(value) for key, value in sorted(partitions.items())
        },
        "source_kind_summaries": {
            key: _summarize(value) for key, value in sorted(kinds.items())
        },
        "real_train_full_source_summary": train_real_summary,
        "heldout_real_full_source_summary": heldout_summary,
        "calibrated_dual_evidence_summary": calibrated,
        "source_level_mixed_supervision_mismatch": mismatch,
        "decision": {
            "gpu_retrain_recommended": False if mismatch else None,
            "status": (
                "rebuild_real_mixed_supervision_before_gpu_retrain"
                if mismatch
                else "distribution_review_still_required"
            ),
            "reason": (
                "real train full-source supervision contains fewer same-source "
                "inside+outside examples than held-out full sources"
                if mismatch
                else "this audit reports topology and does not replace held-out gates"
            ),
            "no_runtime_threshold_or_loss_reweighting_substitute": True,
        },
        "training_manifest_allowed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return audit_distribution(
        canonical_sources=Path(args.canonical_sources).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
