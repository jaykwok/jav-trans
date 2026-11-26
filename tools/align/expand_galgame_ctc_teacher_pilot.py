#!/usr/bin/env python3
"""Expand a Galgame teacher pilot while retaining every paid base row."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.select_galgame_ctc_teacher_pilot import (  # noqa: E402
    DEFAULT_BINS,
    read_jsonl,
    select_pilot,
    write_jsonl,
)


SUMMARY_SCHEMA = "galgame_ctc_teacher_expansion_summary_v1"


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("audio_id") or row.get("source_id") or "")


def expand_pilot(
    source_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    *,
    multiplier: int,
    minimum_acoustic_chars: int = 4,
    group_block: int = 200,
    val_fraction: float = 0.10,
    seed: int = 20260808,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")
    base_ids = [_row_id(row) for row in base_rows]
    if any(not value for value in base_ids):
        raise ValueError("base manifest row lacks audio/source ID")
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("base manifest contains duplicate IDs")

    def bin_index(duration_s: float) -> int | None:
        return next(
            (
                index
                for index, (start_s, end_s, _count) in enumerate(DEFAULT_BINS)
                if start_s <= duration_s < end_s
            ),
            None,
        )

    base_counts: Counter[int] = Counter()
    for row in base_rows:
        index = bin_index(float(row.get("duration_s") or 0.0))
        if index is None:
            raise ValueError(f"base row outside duration bins: {_row_id(row)}")
        base_counts[index] += 1

    additional_bins: list[tuple[float, float, int]] = []
    for index, (start_s, end_s, original_count) in enumerate(DEFAULT_BINS):
        target = original_count * multiplier
        additional = target - base_counts[index]
        if additional < 0:
            raise ValueError(
                f"base bin [{start_s}, {end_s}) already exceeds target {target}"
            )
        additional_bins.append((start_s, end_s, additional))

    base_id_set = set(base_ids)
    available = [row for row in source_rows if _row_id(row) not in base_id_set]
    additions, addition_summary = select_pilot(
        available,
        bins=additional_bins,
        minimum_acoustic_chars=minimum_acoustic_chars,
        group_block=group_block,
        val_fraction=val_fraction,
        seed=seed,
    )
    combined = [dict(row) for row in base_rows] + additions
    combined.sort(
        key=lambda row: int(
            row.get("source_index")
            if row.get("source_index") is not None
            else row.get("index") or 0
        )
    )
    combined_ids = [_row_id(row) for row in combined]
    combined_id_set = set(combined_ids)
    if len(combined_id_set) != len(combined_ids):
        raise AssertionError("expanded manifest contains duplicate IDs")
    group_partitions: dict[str, str] = {}
    for row in combined:
        group = str(row["source_group"])
        partition = str(row["partition"])
        previous = group_partitions.setdefault(group, partition)
        if previous != partition:
            raise AssertionError(f"group {group} crosses partitions")

    total_s = sum(float(row["duration_s"]) for row in combined)
    added_s = sum(float(row["duration_s"]) for row in additions)
    return combined, {
        "schema": SUMMARY_SCHEMA,
        "multiplier": multiplier,
        "base_rows": len(base_rows),
        "added_rows": len(additions),
        "selected_rows": len(combined),
        "base_rows_retained": sum(value in combined_id_set for value in base_ids),
        "audio_hours": round(total_s / 3600.0, 6),
        "incremental_audio_hours": round(added_s / 3600.0, 6),
        "partitions": dict(Counter(str(row["partition"]) for row in combined)),
        "source_groups": len(group_partitions),
        "additional_selection": addition_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--minimum-acoustic-chars", type=int, default=4)
    parser.add_argument("--group-block", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--price-per-hour-usd", type=float, default=0.10)
    parser.add_argument("--budget-usd", type=float, default=3.50)
    args = parser.parse_args()

    output = Path(args.output)
    selected, summary = expand_pilot(
        read_jsonl(Path(args.manifest)),
        read_jsonl(Path(args.base_manifest)),
        multiplier=args.multiplier,
        minimum_acoustic_chars=args.minimum_acoustic_chars,
        group_block=args.group_block,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    cost = float(summary["audio_hours"]) * args.price_per_hour_usd
    if cost > args.budget_usd + 1e-12:
        raise SystemExit(
            f"expanded pilot would cost ${cost:.6f}, above ${args.budget_usd:.6f}"
        )
    write_jsonl(output, selected)
    summary.update(
        {
            "manifest": str(Path(args.manifest)),
            "base_manifest": str(Path(args.base_manifest)),
            "output": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "price_per_hour_usd": args.price_per_hour_usd,
            "estimated_cost_usd": round(cost, 9),
            "budget_usd": args.budget_usd,
        }
    )
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
