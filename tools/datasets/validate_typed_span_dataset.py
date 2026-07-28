#!/usr/bin/env python3
"""Validate a typed-span dataset and propose a leakage-safe stratified split.

The v11 post-mortem (docs/HISTORY.md:51) is the thing this guards against: train
inside-truth was 100% synthetic while held-out was 100% real full-source, giving
the model a provenance shortcut and collapsing real-domain metrics.  So the split
this proposes must put BOTH provenances in every partition, keep whole videos on
one side, and report class balance per provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TYPES = ("speech", "non_semantic_vocal", "non_vocal", "unsure")
SPLIT_SCHEMA = "typed_span_split_v1"


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_bucket(key: str, buckets: int) -> int:
    """Deterministic group assignment; no RNG so the split is reproducible."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def validate(dataset: Path, *, split_output: Path | None) -> dict:
    examples: list[dict] = list(_rows(dataset))
    problems: list[str] = []

    by_provenance: Counter = Counter()
    type_time: dict[str, Counter] = defaultdict(Counter)
    type_count: dict[str, Counter] = defaultdict(Counter)
    covered_time: Counter = Counter()
    total_time: Counter = Counter()
    groups_by_provenance: dict[str, set[str]] = defaultdict(set)
    boundary_points: Counter = Counter()

    for example in examples:
        provenance = str(example.get("provenance"))
        by_provenance[provenance] += 1
        groups_by_provenance[provenance].add(str(example.get("group_id")))
        duration = float(example.get("duration_s") or 0.0)
        total_time[provenance] += duration

        spans = example.get("spans") or []
        previous_end = -1.0
        covered = 0.0
        for span in spans:
            start = float(span["start_s"])
            end = float(span["end_s"])
            span_type = str(span["type"])
            if span_type not in TYPES:
                problems.append(f"{example['example_id']}: unknown type {span_type}")
            if end <= start:
                problems.append(f"{example['example_id']}: empty span {start}-{end}")
            if start < previous_end - 1e-6:
                problems.append(
                    f"{example['example_id']}: overlapping spans at {start:.3f}s"
                )
            if end > duration + 1e-3:
                problems.append(
                    f"{example['example_id']}: span ends {end:.3f}s past duration "
                    f"{duration:.3f}s"
                )
            previous_end = end
            covered += end - start
            type_time[provenance][span_type] += end - start
            type_count[provenance][span_type] += 1
        covered_time[provenance] += covered

        coverage = str(example.get("coverage"))
        if coverage == "complete" and abs(covered - duration) > 0.05:
            problems.append(
                f"{example['example_id']}: declared complete but covers "
                f"{covered:.2f}/{duration:.2f}s"
            )
        if coverage == "sparse" and covered > duration + 1e-3:
            problems.append(f"{example['example_id']}: sparse coverage exceeds duration")
        boundary_points[provenance] += len(example.get("boundary_points") or [])

    print(f"dataset          : {dataset}")
    print(f"examples         : {len(examples)}")
    print()
    print("=== per provenance ===")
    for provenance, count in by_provenance.most_common():
        hours = total_time[provenance] / 3600.0
        labelled = covered_time[provenance] / 3600.0
        print(f"\n{provenance}: {count} examples, {hours:.2f} h audio, "
              f"{labelled:.2f} h labelled ({100.0 * labelled / hours:.1f}%)")
        print(f"  groups (leakage units): {len(groups_by_provenance[provenance])}")
        print(f"  boundary points       : {boundary_points[provenance]}")
        span_total = sum(type_time[provenance].values()) or 1.0
        for span_type in TYPES:
            minutes = type_time[provenance][span_type] / 60.0
            share = 100.0 * type_time[provenance][span_type] / span_total
            print(
                f"    {span_type:<20} {minutes:>8.1f} min  {share:>5.1f}%  "
                f"n={type_count[provenance][span_type]}"
            )

    # The headline number: how much SUPERVISED (non-unsure) signal exists, and is
    # it balanced enough to train and select on - the thing v12's target lacked.
    print()
    print("=== supervised signal (unsure excluded) ===")
    for provenance in by_provenance:
        supervised = sum(
            type_time[provenance][t] for t in ("speech", "non_semantic_vocal", "non_vocal")
        )
        if not supervised:
            continue
        print(f"{provenance}:")
        for span_type in ("speech", "non_semantic_vocal", "non_vocal"):
            share = 100.0 * type_time[provenance][span_type] / supervised
            print(f"    {span_type:<20} {share:>5.1f}%")

    summary: dict = {
        "schema": "typed_span_validation_v1",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "example_count": len(examples),
        "examples_per_provenance": dict(by_provenance),
        "groups_per_provenance": {k: len(v) for k, v in groups_by_provenance.items()},
        "type_seconds_per_provenance": {
            k: {t: round(v[t], 3) for t in TYPES} for k, v in type_time.items()
        },
        "problem_count": len(problems),
        "problems": problems[:40],
    }

    if split_output is not None:
        summary["split"] = _propose_split(examples, split_output)

    print()
    if problems:
        print(f"=== {len(problems)} structural problem(s) ===")
        for line in problems[:20]:
            print(f"  - {line}")
    else:
        print("=== no structural problems ===")
    return summary


def _connected_components(examples: list[dict]) -> dict[str, str]:
    """Union-find over shared leakage units -> one component key per example.

    A synthetic composite reuses galgame speech units and a real definite-drop
    background bed; two composites sharing any unit must not land on opposite
    sides of a split, or the model sees the same audio in train and eval.  The
    bed link also ties synthetic examples to the real window they borrowed from.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for example in examples:
        members = [str(m) for m in (example.get("group_members") or [])]
        if not members:
            members = [f"group:{example.get('group_id')}"]
        anchor = f"example:{example['example_id']}"
        find(anchor)
        for member in members:
            union(anchor, member)

    return {
        str(example["example_id"]): find(f"example:{example['example_id']}")
        for example in examples
    }


def _propose_split(examples: list[dict], output: Path) -> dict:
    """Group-disjoint, provenance-stratified split.

    Components come from union-find over shared audio units, so no source unit,
    video, or background bed appears on both sides.  Bucketing is applied within
    each provenance so all three partitions contain both.
    """
    component_of = _connected_components(examples)
    components: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        components[component_of[str(example["example_id"])]].append(example)

    # A component can span provenances (a synthetic composite that borrowed a bed
    # from a real window).  Assign per component, then record which provenances
    # each partition actually received.
    assignment: dict[str, str] = {}
    for key in sorted(components):
        bucket = _stable_bucket(key, 10)
        if bucket == 8:
            assignment[key] = "val"
        elif bucket == 9:
            assignment[key] = "test"
        else:
            assignment[key] = "train"

    per_partition: dict[str, Counter] = defaultdict(Counter)
    supervised_seconds: dict[str, Counter] = defaultdict(Counter)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(components):
            partition = assignment[key]
            for example in components[key]:
                provenance = str(example.get("provenance"))
                per_partition[partition][provenance] += 1
                for span in example.get("spans") or []:
                    span_type = str(span["type"])
                    if span_type == "unsure":
                        continue
                    supervised_seconds[partition][span_type] += (
                        float(span["end_s"]) - float(span["start_s"])
                    )
                handle.write(
                    json.dumps(
                        {
                            "schema": SPLIT_SCHEMA,
                            "example_id": example["example_id"],
                            "provenance": provenance,
                            "component_id": key,
                            "partition": partition,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print()
    print("=== proposed split (union-find group-disjoint, provenance-stratified) ===")
    print(f"  leakage components: {len(components)}")
    for partition in ("train", "val", "test"):
        counts = dict(per_partition[partition])
        seconds = supervised_seconds[partition]
        detail = "  ".join(
            f"{t}={seconds[t] / 60.0:.0f}min" for t in ("speech", "non_semantic_vocal", "non_vocal")
        )
        print(f"  {partition:<6} examples={counts}")
        print(f"         supervised: {detail}")

    both = [
        partition
        for partition in ("train", "val", "test")
        if len(per_partition[partition]) < 2
    ]
    if both:
        print(f"  WARNING: partitions missing a provenance: {both}")

    return {
        "output": str(output),
        "leakage_components": len(components),
        "examples_per_partition": {
            k: dict(v) for k, v in per_partition.items()
        },
        "supervised_seconds_per_partition": {
            k: {t: round(v[t], 1) for t in ("speech", "non_semantic_vocal", "non_vocal")}
            for k, v in supervised_seconds.items()
        },
        "partitions_missing_a_provenance": both,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-output", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = validate(
        Path(args.dataset).expanduser().resolve(),
        split_output=(
            Path(args.split_output).expanduser().resolve()
            if args.split_output
            else None
        ),
    )
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "problems"},
                     ensure_ascii=False, indent=2, sort_keys=True))
