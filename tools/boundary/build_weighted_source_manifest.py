#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping) or row.get("error") or not row.get("audio"):
            continue
        audio_path = Path(str(row["audio"]))
        if not audio_path.exists():
            continue
        rows.append(dict(row))
    return rows


def build_weighted_manifest(
    *,
    specs: list[str],
    output_manifest: Path,
    total_rows: int,
    seed: int,
) -> dict[str, Any]:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    rng = random.Random(seed)
    groups: list[tuple[str, float, list[dict[str, Any]], Path]] = []
    for spec in specs:
        name, weight, path = parse_spec(spec)
        rows = load_manifest(path)
        if not rows:
            raise ValueError(f"no valid rows for source group {name}: {path}")
        groups.append((name, weight, rows, path))
    total_weight = sum(weight for _, weight, _, _ in groups)
    if total_weight <= 0.0:
        raise ValueError("sum of source weights must be positive")

    output_rows: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    target_counts = []
    remaining = total_rows
    for index, (name, weight, rows, path) in enumerate(groups):
        if index == len(groups) - 1:
            count = remaining
        else:
            count = int(round(total_rows * weight / total_weight))
            count = max(0, min(remaining, count))
        remaining -= count
        target_counts.append((name, count, rows, path))

    for name, count, rows, path in target_counts:
        for sample_index in range(count):
            row = dict(rng.choice(rows))
            row["source_mix_group"] = name
            row["source_mix_manifest"] = str(path)
            row["source_mix_sample_index"] = sample_index
            output_rows.append(row)
            group_counts[name] += 1
            source_counts[str(row.get("source") or "")] += 1
    rng.shuffle(output_rows)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "output_manifest": str(output_manifest),
        "total_rows": len(output_rows),
        "seed": seed,
        "source_specs": specs,
        "group_counts": dict(group_counts),
        "source_counts": dict(source_counts),
        "duration_s_total": sum(float(row.get("duration_s") or 0.0) for row in output_rows),
    }
    summary_path = output_manifest.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_spec(spec: str) -> tuple[str, float, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError("source spec must be name=weight=manifest_path")
    name = parts[0].strip()
    if not name:
        raise ValueError("source spec name must not be empty")
    weight = float(parts[1])
    if weight < 0.0:
        raise ValueError("source spec weight must be non-negative")
    path = Path(parts[2])
    if not path.exists():
        raise FileNotFoundError(path)
    return name, weight, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weighted source manifest from materialized speech-island manifests.")
    parser.add_argument("--source", action="append", required=True, help="name=weight=manifest.json. Repeatable.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--total-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=240604)
    args = parser.parse_args(argv)
    if args.total_rows <= 0:
        parser.error("--total-rows must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_weighted_manifest(
        specs=args.source,
        output_manifest=Path(args.output_manifest),
        total_rows=args.total_rows,
        seed=args.seed,
    )
    print(f"manifest={summary['output_manifest']}")
    print(f"summary={summary['output_manifest'].rsplit('.', 1)[0]}.summary.json")
    print(f"group_counts={json.dumps(summary['group_counts'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
