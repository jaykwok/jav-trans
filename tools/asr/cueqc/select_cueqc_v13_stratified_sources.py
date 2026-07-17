#!/usr/bin/env python3
"""Select a deterministic 85/10/5 CueQC v13 composite source subset."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PARTITION_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}
SUMMARY_SCHEMA = "cueqc_v13_stratified_source_selection_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _partition_counts(count: int) -> dict[str, int]:
    if count <= 0:
        raise ValueError("count must be positive")
    exact = {name: count * ratio for name, ratio in PARTITION_RATIOS.items()}
    result = {name: int(value) for name, value in exact.items()}
    remainder = count - sum(result.values())
    order = sorted(
        PARTITION_RATIOS,
        key=lambda name: (-(exact[name] - result[name]), list(PARTITION_RATIOS).index(name)),
    )
    for name in order[:remainder]:
        result[name] += 1
    return result


def _validate_sources(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    partitions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PARTITION_RATIOS
    }
    sample_ids: set[str] = set()
    core_owners: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"source row {row_number} is missing sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id in source manifest: {sample_id}")
        sample_ids.add(sample_id)

        partition = str(row.get("source_partition") or "").strip()
        if partition not in partitions:
            raise ValueError(
                f"source row {sample_id} has unsupported source_partition: {partition!r}"
            )
        partitions[partition].append(row)

        core_spans = row.get("core_spans")
        if not isinstance(core_spans, list) or not core_spans:
            raise ValueError(f"source row {sample_id} has no core_spans")
        for core_number, core in enumerate(core_spans, start=1):
            if not isinstance(core, dict):
                raise ValueError(
                    f"source row {sample_id} core {core_number} is not an object"
                )
            core_id = str(core.get("core_id") or "").strip()
            if not core_id:
                raise ValueError(
                    f"source row {sample_id} core {core_number} is missing core_id"
                )
            previous = core_owners.get(core_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate core_id in source manifest: {core_id} "
                    f"(samples {previous} and {sample_id})"
                )
            core_owners[core_id] = sample_id
    return partitions


def _reuse_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        sample_id
        for row in _rows(path)
        if (sample_id := str(row.get("sample_id") or "").strip())
    }


def select(
    *,
    source_manifest: Path,
    output: Path,
    count: int = 2048,
    seed: int = 20260716,
    reuse_jsonl: Path | None = None,
) -> dict[str, Any]:
    targets = _partition_counts(count)
    partitions = _validate_sources(_rows(source_manifest))
    reused_ids = _reuse_ids(reuse_jsonl)
    selected: list[dict[str, Any]] = []
    selected_partition_counts: dict[str, int] = {}
    reused_partition_counts: dict[str, int] = {}

    for partition, target in targets.items():
        inventory = list(partitions[partition])
        if len(inventory) < target:
            raise ValueError(
                f"insufficient {partition} inventory: need {target}, found {len(inventory)}"
            )
        random.Random(f"{seed}:{partition}").shuffle(inventory)
        inventory.sort(
            key=lambda row: str(row["sample_id"]) not in reused_ids
        )
        chosen = inventory[:target]
        selected.extend(chosen)
        selected_partition_counts[partition] = len(chosen)
        reused_partition_counts[partition] = sum(
            str(row["sample_id"]) in reused_ids for row in chosen
        )

    selected_core_counts = Counter(
        str(core["core_id"])
        for row in selected
        for core in row["core_spans"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_manifest": str(source_manifest),
        "reuse_jsonl": str(reuse_jsonl) if reuse_jsonl is not None else None,
        "output": str(output),
        "seed": seed,
        "requested_count": count,
        "selected_count": len(selected),
        "partition_counts": selected_partition_counts,
        "core_count": len(selected_core_counts),
        "max_core_use": max(selected_core_counts.values(), default=0),
        "reused_count": sum(reused_partition_counts.values()),
        "reused_partition_counts": reused_partition_counts,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--reuse-jsonl",
        help="existing runtime or selection JSONL whose sample_ids should be reused first",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            select(
                source_manifest=Path(args.source_manifest),
                output=Path(args.output),
                count=args.count,
                seed=args.seed,
                reuse_jsonl=Path(args.reuse_jsonl) if args.reuse_jsonl else None,
            ),
            ensure_ascii=False,
        )
    )
